"""Offline synthetic pipeline/Forge tests. No real AVE optimization or network."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from fusion.baseline_tracking import TrackingUnavailable
from fusion.contracts import ContractError
from fusion.embedding_data import EmbeddingDataset
from fusion.mlp_model import LateFusionMLP, MLPConfig
from fusion.mlp_tracking import EPOCH_KEYS, ForgeMLPTracker
from fusion.mlp_training import (
    EarlyStopping, Predictions, evaluate_frozen, fit_operating_point,
    model_fingerprint, predict, primary_metrics, train_epoch,
)
from fusion.train_mlp import (
    MLPExperimentConfig, VerifiedData, load_selected_checkpoint, run_experiment,
    save_checkpoint,
)
from fusion.training_contracts import (
    CheckpointMetadata, TrainingConfig, build_optimizer, make_loader, seed_everything,
)
from test_fusion_pair_generator import ROWS, bundles, record
from test_fusion_mlp import manifest


def predictions(scores, targets, *, split="val", partition="primary", model="b"*64):
    s = np.asarray(scores, dtype=np.float64)
    z = np.log(s/(1-s))
    return Predictions(split, partition, "a"*64, model,
                       tuple((split, str(i), str(i)) for i in range(len(s))),
                       np.array(targets, dtype=np.int64), z, .5)


class FakeTracker:
    instances = []
    def __init__(self, output):
        self.output = output
        self.metadata = {"status": "NOT_STARTED"}
        self.rows = []
        self.aborted = False
        self.__class__.instances.append(self)
    def start(self, config):
        self.config = config
        self.metadata = {"status": "ONLINE_WRITE_VERIFIED", "run_url": "synthetic-only"}
        return self.metadata
    def log_epoch(self, row):
        self.rows.append(dict(row))
    def finish(self, metrics, files, history, manifest_sha256, *, before_finalize=None):
        assert self.rows == history
        assert all(p.is_file() for p in files)
        if before_finalize is not None:
            before_finalize()
        self.metadata.update(status="COMPLETE", remote_summary_verified=True,
                             remote_history_epochs_verified=len(history))
        return self.metadata
    def abort(self):
        self.aborted = True


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        audio, vision = bundles({"train": ROWS, "val": (record("v1",4),record("v2",4),
                                                               record("v3",5),record("v4",5))})
        self.train = EmbeddingDataset(manifest(audio,vision),audio,vision,split="train")
        value = manifest(audio,vision,"val")
        self.val = EmbeddingDataset(value,audio,vision,split="val")
        self.challenge = EmbeddingDataset(value,audio,vision,split="val",partition="challenge")
        self.training = TrainingConfig(batch_size=3,epochs=3,selection_metric="roc_auc")
        self.model = LateFusionMLP(MLPConfig(hidden_dims=(8,),dropout=.2))
        self.optimizer = build_optimizer(self.model,self.training.optimizer)

    def config(self, name="run", **kwargs):
        # Only synthetic tempfile output policy is mocked; production path
        # rejection is exercised separately and never weakened.
        with patch("fusion.train_mlp._output_path"):
            return MLPExperimentConfig(*([self.root]*7),self.root/name,
                                       model=self.model.config, training=self.training, **kwargs)

    def verified(self):
        baseline = {"method":"jensen_shannon","val_roc_auc":.9696418405485012,
                    "val_average_precision":.9728053582663733,"val_f1":.9172932330827067,
                    "val_challenge_rejection":.15671641791044777,"train_roc_auc":.9999484255023423,
                    "train_average_precision":.999947967824911,"train_f1":.9913895486935868}
        return VerifiedData(self.train,self.val,self.challenge,{},[],{"fixture":"synthetic only"},baseline)

    def execute(self, config=None, tracker=FakeTracker):
        with patch("fusion.train_mlp._repository",return_value={"commit":"c"*40}), \
             patch("fusion.train_mlp.prepare_inputs",return_value=self.verified()):
            return run_experiment(config or self.config(),tracker_factory=tracker,progress=lambda _:None)

    def test_optimization_covers_irregular_final_batch_and_changes_only_train(self):
        before = model_fingerprint(self.model)
        result = train_epoch(self.model,make_loader(self.train,self.training,role="train"),self.optimizer,self.training)
        self.assertEqual(result["count"],len(self.train))
        self.assertEqual(result["optimizer_steps"],6)  # 16 items, final singleton.
        self.assertNotEqual(before,model_fingerprint(self.model))
        self.assertTrue(self.model.training)
        for data, role in ((self.val,"val"),(self.challenge,"train")):
            with self.assertRaises(ContractError):
                train_epoch(self.model,torch.utils.data.DataLoader(data,batch_size=3),self.optimizer,self.training)

    def test_validation_is_eval_no_grad_and_never_updates_parameters(self):
        before = model_fingerprint(self.model)
        modes = []
        def hook(module,args):
            modes.append((module.training,torch.is_grad_enabled()))
        handle = self.model.register_forward_pre_hook(hook)
        try:
            first = predict(self.model,self.val,self.training)
            second = predict(self.model,self.val,self.training)
        finally:
            handle.remove()
        self.assertTrue(all(not train and not grad for train,grad in modes))
        self.assertEqual(before,model_fingerprint(self.model))
        np.testing.assert_array_equal(first.logits,second.logits)

    def test_index_side_channel_preserves_seeded_shuffle_across_epochs(self):
        original = EmbeddingDataset.__getitem__
        visits = []
        def observe(dataset, index):
            visits.append(index)
            return original(dataset, index)
        expected_loader = make_loader(self.train,self.training,role="train")
        actual_loader = make_loader(self.train,self.training,role="train")
        with patch.object(EmbeddingDataset,"__getitem__",observe):
            expected = []
            for _ in range(2):
                visits.clear()
                list(expected_loader)
                expected.append(list(visits))
            actual = []
            for _ in range(2):
                visits.clear()
                train_epoch(self.model,actual_loader,self.optimizer,self.training)
                actual.append(list(visits))
        self.assertEqual(actual,expected)
        self.assertNotEqual(actual[0],actual[1])
        for order in actual:
            self.assertEqual(sorted(order),list(range(len(self.train))))

    def test_duplicate_indices_with_full_count_are_rejected_before_bad_batch_update(self):
        indices = list(range(len(self.train)))
        indices[-1] = indices[0]
        loader = torch.utils.data.DataLoader(self.train,batch_size=len(indices),sampler=indices)
        before = model_fingerprint(self.model)
        with self.assertRaisesRegex(ContractError,"duplicate original"), patch.object(self.optimizer,"step") as step:
            train_epoch(self.model,loader,self.optimizer,self.training)
        step.assert_not_called()
        self.assertEqual(before,model_fingerprint(self.model))

    def test_repeated_single_index_across_batches_is_rejected(self):
        loader = torch.utils.data.DataLoader(self.train,batch_size=1,sampler=[0]*len(self.train))
        with patch.object(self.optimizer,"step") as step, self.assertRaisesRegex(ContractError,"duplicate original"):
            train_epoch(self.model,loader,self.optimizer,self.training)
        self.assertEqual(step.call_count,1)  # No second update on the repeated item.

    def test_missing_index_epoch_is_rejected(self):
        loader = torch.utils.data.DataLoader(self.train,batch_size=3,sampler=list(range(len(self.train)-1)))
        with self.assertRaisesRegex(ContractError,"complete TRAIN"):
            train_epoch(self.model,loader,self.optimizer,self.training)

    def test_empty_and_invalid_training_loaders_fail_closed(self):
        for loader in (None, torch.utils.data.DataLoader(self.train,batch_size=3,sampler=[]),
                       torch.utils.data.DataLoader(self.train,batch_size=None),
                       torch.utils.data.DataLoader(self.train,batch_size=3,drop_last=True),
                       torch.utils.data.DataLoader(self.train,batch_size=3,collate_fn=lambda x:x)):
            with self.subTest(loader=type(loader).__name__), self.assertRaises(ContractError), \
                 patch.object(self.optimizer,"step") as step:
                train_epoch(self.model,loader,self.optimizer,self.training)
            step.assert_not_called()
        for index in (-1,len(self.train),True,1.5):
            loader = torch.utils.data.DataLoader(self.train,batch_size=1,sampler=[index])
            with self.subTest(index=index), self.assertRaisesRegex(ContractError,"original integer indices"):
                train_epoch(self.model,loader,self.optimizer,self.training)

    def test_training_indices_and_targets_never_enter_predictive_features(self):
        shapes = []
        def check(module,args,kwargs):
            self.assertEqual(set(kwargs),{"audio","visual","validity"})
            batch = len(kwargs["audio"])
            self.assertEqual(kwargs["audio"].shape,(batch,10,256))
            self.assertEqual(kwargs["visual"].shape,(batch,10,512))
            self.assertEqual(kwargs["validity"].shape,(batch,10))
            self.assertEqual(kwargs["validity"].dtype,torch.bool)
            shapes.append(batch)
        handle = self.model.register_forward_pre_hook(check,with_kwargs=True)
        self.addCleanup(handle.remove)
        # Make all features identical. Distinct index coverage must still work.
        original = EmbeddingDataset.__getitem__
        def identical(dataset,index):
            features,_ = original(dataset,0)
            _,target = original(dataset,index)
            return features,target
        with patch.object(EmbeddingDataset,"__getitem__",identical):
            result = train_epoch(self.model,make_loader(self.train,self.training,role="train"),self.optimizer,self.training)
        self.assertEqual(result["count"],len(self.train))
        self.assertEqual(shapes,[3,3,3,3,3,1])

    def test_validation_never_uses_training_index_path(self):
        with patch("fusion.mlp_training._indexed_training_loader",side_effect=AssertionError("train only")):
            result = predict(self.model,self.val,self.training)
        self.assertEqual(len(result.logits),len(self.val))

    def test_original_indices_remain_auditable_with_worker_prefetch(self):
        config=replace(self.training,num_workers=1)
        result=train_epoch(self.model,make_loader(self.train,config,role="train"),self.optimizer,config)
        self.assertEqual(result["count"],len(self.train))
        repeated=torch.utils.data.DataLoader(self.train,batch_size=3,num_workers=1,sampler=[0]*len(self.train))
        with self.assertRaisesRegex(ContractError,"duplicate original"):
            train_epoch(self.model,repeated,self.optimizer,config)

    def test_early_stopping_exact_ties_keep_earliest_epoch(self):
        rule = EarlyStopping(2)
        self.assertTrue(rule.observe(1,.8))
        self.assertFalse(rule.observe(2,.8))
        self.assertFalse(rule.observe(3,.7))
        self.assertEqual(rule.best_epoch,1)
        self.assertTrue(rule.should_stop)
        with self.assertRaises(ContractError): rule.observe(4,.9)
        with self.assertRaises(ContractError): EarlyStopping(0)
        with self.assertRaises(ContractError): EarlyStopping().observe(2,.7)
        with self.assertRaises(ContractError): EarlyStopping().observe(1,float("nan"))

    def test_threshold_manual_tie_and_metrics(self):
        # Independent hand calculation: threshold .2 yields TP=2, FP=1, FN=0.
        pred = predictions([.1,.2,.3,.4],[0,1,0,1])
        policy = fit_operating_point(pred,selected_epoch=2)
        values = primary_metrics(pred,threshold=policy.threshold)
        self.assertAlmostEqual(policy.threshold,.2)  # .2 gives F1=.8; .4 gives 2/3.
        self.assertAlmostEqual(values["roc_auc"],.75)
        self.assertAlmostEqual(values["average_precision"],5/6)
        self.assertEqual(values["counts"],{"tp":2,"tn":1,"fp":1,"fn":0})
        tied = predictions([.1,.2,.3,.4],[1,0,0,1])
        # Both .1 and .4 yield exactly F1=2/3; largest threshold wins.
        self.assertAlmostEqual(fit_operating_point(tied,selected_epoch=1).threshold,.4)

    def test_challenge_and_test_cannot_select_threshold_or_update(self):
        with self.assertRaises(ContractError): predict(self.model,self.challenge,self.training)
        with self.assertRaises(ContractError): fit_operating_point(predictions([.2,.8],[0,1],split="train"),selected_epoch=1)
        challenge = predictions([.2,.8],[0,0],partition="challenge")
        with self.assertRaises(ContractError): fit_operating_point(challenge,selected_epoch=1)
        with self.assertRaises(ContractError): predictions([.2,.8],[0,1],split="test")
        val = predict(self.model,self.val,self.training)
        policy = fit_operating_point(val,selected_epoch=1)
        result = predict(self.model,self.challenge,self.training,frozen_policy=policy)
        report = evaluate_frozen(result,policy)
        self.assertEqual(report["count"],len(self.challenge))
        self.assertAlmostEqual(report["false_acceptance_rate"]+report["rejection_rate"],1)
        with torch.no_grad(): next(self.model.parameters()).add_(1)
        with self.assertRaises(ContractError): predict(self.model,self.challenge,self.training,frozen_policy=policy)

    def test_complete_synthetic_pipeline_reload_freeze_and_artifacts(self):
        config = self.config()
        result = self.execute(config)
        self.assertEqual(result["status"],"COMPLETE")
        self.assertTrue((config.output_dir/"completion.json").is_file())
        reload = json.loads((config.output_dir/"checkpoint_reload.json").read_text())
        self.assertTrue(reload["identical_all_validation_logits"])
        history = json.loads((config.output_dir/"history.json").read_text())
        best = max(history,key=lambda r:r["val_roc_auc"])
        self.assertEqual(result["final"]["selected_epoch"],best["epoch"])
        self.assertEqual(result["final"]["frozen_threshold"],result["final"]["val_challenge"]["threshold"])
        self.assertFalse(result["final"]["final_test_scored"])
        names=[p.name for p in config.output_dir.iterdir()]
        self.assertFalse(any(n.startswith("test_") for n in names))

    def test_repeated_seed_has_identical_weights_predictions_and_selection(self):
        first = self.config("one")
        second = self.config("two")
        a,b = self.execute(first),self.execute(second)
        self.assertEqual(a["final"],b["final"])
        self.assertEqual((first.output_dir/"val_primary_scores.csv").read_bytes(),
                         (second.output_dir/"val_primary_scores.csv").read_bytes())

    def test_safe_checkpoint_load_rejects_changed_bytes(self):
        config=self.config()
        self.execute(config)
        path=config.output_dir/"best.pt"
        path.write_bytes(path.read_bytes()+b"changed")
        with self.assertRaisesRegex(ContractError,"hash/size"):
            load_selected_checkpoint(config.output_dir)

    def test_tracking_preflight_failure_prevents_all_optimization(self):
        class Failure(FakeTracker):
            def start(self,config): raise RuntimeError("FAKE_CREDENTIAL_NEVER_PERSIST")
        config=self.config()
        with patch("fusion.train_mlp.train_epoch",side_effect=AssertionError("must not optimize")), \
             self.assertRaises(ContractError):
            self.execute(config,Failure)
        state=json.loads((config.output_dir/"state.json").read_text())
        self.assertEqual(state["status"],"FAILED")
        self.assertFalse(state["optimization_started"])
        self.assertNotIn("FAKE_CREDENTIAL",json.dumps(state))
        self.assertFalse((config.output_dir/"completion.json").exists())

    def test_upload_failure_never_publishes_completion(self):
        class Failure(FakeTracker):
            def finish(self,*args,**kwargs): raise ConnectionError("FAKE_CREDENTIAL_NEVER_PERSIST")
        config=self.config()
        with self.assertRaises(ContractError): self.execute(config,Failure)
        state=json.loads((config.output_dir/"state.json").read_text())
        self.assertEqual(state["status"],"FAILED")
        self.assertTrue((config.output_dir/"best.pt").is_file())
        self.assertFalse((config.output_dir/"completion.json").exists())
        self.assertNotIn("FAKE_CREDENTIAL",(config.output_dir/"tracking.json").read_text())

    def test_upload_time_manifest_tampering_is_caught_before_tracker_success(self):
        class Tamper(FakeTracker):
            def finish(self,*args,**kwargs):
                path=self.output/"output_manifest.json"
                path.write_text(path.read_text()+"\n")
                return super().finish(*args,**kwargs)
        config=self.config()
        with self.assertRaises(ContractError): self.execute(config,Tamper)
        self.assertNotEqual(Tamper.instances[-1].metadata["status"],"COMPLETE")
        self.assertTrue(Tamper.instances[-1].aborted)
        self.assertFalse((config.output_dir/"completion.json").exists())

    def test_interrupt_preserves_explicit_interrupted_state(self):
        config=self.config()
        with patch("fusion.train_mlp.train_epoch",side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.execute(config)
        state=json.loads((config.output_dir/"state.json").read_text())
        self.assertEqual(state["status"],"INTERRUPTED")
        self.assertFalse((config.output_dir/"completion.json").exists())

    def test_production_output_must_be_fresh_ignored_and_local(self):
        with self.assertRaises(ContractError):
            MLPExperimentConfig(*([self.root]*7),self.root/"outside")
        with patch("fusion.train_mlp._output_path"), self.assertRaises(ContractError):
            MLPExperimentConfig(*([self.root]*7),self.root/"invalid",patience=0)


class ForgeTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_wrong_runtime_destinations_and_offline_fail_without_sdk(self):
        for env in ({"WANDB_MODE":"offline"},{"WANDB_BASE_URL":"https://api.wandb.ai"},
                    {"WANDB_ENTITY":"other"},{"WANDB_PROJECT":"other"}):
            with patch.dict(os.environ,env,clear=True),self.assertRaises(TrackingUnavailable):
                ForgeMLPTracker(self.root).preflight_access()

    def tracker(self):
        tracker=ForgeMLPTracker(self.root)
        tracker.sdk=MagicMock()
        tracker.api=MagicMock()
        run=tracker.sdk.init.return_value
        run.settings.mode="online"; run.settings.base_url="https://api.forge.coreweave.com"
        run.entity="GC2-AValidate"; run.project="avalidate"; run.id="mock"
        run.url="https://forge.coreweave.com/GC2-AValidate/avalidate/runs/mock"
        uploaded=run.log_artifact.return_value
        uploaded.qualified_name="GC2-AValidate/avalidate/mock:v0"
        uploaded.digest="mockdigest"
        tracker.api.artifact.return_value.digest="mockdigest"
        return tracker,run

    def test_online_readiness_requires_uploaded_artifact_readback(self):
        tracker,run=self.tracker()
        with patch.object(tracker,"preflight_access"):
            result=tracker.start({"training":{"seed":42}})
        self.assertEqual(result["status"],"ONLINE_WRITE_VERIFIED")
        run.log_artifact.return_value.wait.assert_called_once_with(timeout=180)
        tracker.api.artifact.assert_called_once()
        self.assertEqual(run.log.call_args.kwargs["step"],0)

    def test_readiness_failure_is_nonsecret_and_not_ready(self):
        tracker,run=self.tracker()
        run.log_artifact.side_effect=RuntimeError("FAKE_CREDENTIAL_NEVER_PERSIST")
        with patch.object(tracker,"preflight_access"),self.assertRaises(TrackingUnavailable) as caught:
            tracker.start({"training":{"seed":42}})
        self.assertNotIn("FAKE_CREDENTIAL",str(caught.exception))
        self.assertNotEqual(tracker.metadata["status"],"ONLINE_WRITE_VERIFIED")

    def test_epoch_curves_use_actual_steps_and_fixed_threshold_tags(self):
        tracker,run=self.tracker(); tracker.run=run
        row={k:.5 for k in EPOCH_KEYS}; row["epoch"]=3
        tracker.log_epoch(row)
        self.assertEqual(run.log.call_args.kwargs["step"],3)
        self.assertIn("val/f1_at_0_5",run.log.call_args.args[0])

    def finalization_fixture(self):
        tracker,run=self.tracker(); tracker.run=run
        row={k:.5 for k in EPOCH_KEYS}; row["epoch"]=1
        metrics={"final/val_roc_auc":.75}
        remote=tracker.api.run.return_value
        remote.summary={**metrics,"reproducibility_manifest_sha256":"a"*64}
        remote.scan_history.return_value=[{"epoch":1,**{tag:row[k] for k,tag in EPOCH_KEYS.items()}}]
        files=[]
        for name in ("best.pt","checkpoint_metadata.json","frozen_operating_point.json","history.json"):
            path=self.root/name; path.write_bytes(b"synthetic"); files.append(path)
        return tracker,run,remote,metrics,files,[row]

    def test_final_upload_remote_history_and_summary_are_verified_before_success(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        events=MagicMock()
        for name,method in (("upload",run.log_artifact),("wait",run.log_artifact.return_value.wait),
                            ("reload",remote.load),("history",remote.scan_history),
                            ("artifact",tracker.api.artifact),("finish",run.finish)):
            events.attach_mock(method,name)
        before_finalize=MagicMock()
        events.attach_mock(before_finalize,"local_integrity")
        result=tracker.finish(metrics,files,history,"a"*64,before_finalize=before_finalize)
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual(len(result["artifacts"]),2)
        self.assertEqual(result["remote_history_epochs_verified"],1)
        names=[call[0] for call in events.mock_calls]
        self.assertEqual(names[-2:],["local_integrity","finish"])
        self.assertEqual(names.count("artifact"),2)
        self.assertLess(names.index("history"),names.index("finish"))
        run.finish.assert_called_once_with(exit_code=0)
        remote.load.assert_called_once_with(force=True)
        self.assertFalse(remote.scan_history.call_args.kwargs["use_cache"])
        tracker.abort(); tracker.abort()
        run.finish.assert_called_once_with(exit_code=0)
        with self.assertRaises(TrackingUnavailable): tracker.finish(metrics,files,history,"a"*64)

    def assert_failed_finalization(self,tracker,run,metrics,files,history):
        with patch("fusion.mlp_tracking.time.sleep"), self.assertRaises(TrackingUnavailable) as caught:
            tracker.finish(metrics,files,history,"a"*64)
        self.assertNotIn("FAKE_CREDENTIAL",str(caught.exception))
        self.assertEqual(tracker.metadata["status"],"INCOMPLETE")
        self.assertNotIn("remote_summary_verified",tracker.metadata)
        self.assertTrue(all(c.kwargs["exit_code"]==1 for c in run.finish.call_args_list))
        run.finish.assert_called_once_with(exit_code=1)
        tracker.abort(); tracker.abort()
        run.finish.assert_called_once_with(exit_code=1)
        with self.assertRaises(TrackingUnavailable): tracker.finish(metrics,files,history,"a"*64)
        return caught

    def test_summary_mismatch_aborts_without_success(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        remote.summary={"final/val_roc_auc":.1}
        self.assert_failed_finalization(tracker,run,metrics,files,history)

    def test_artifact_mismatch_aborts_without_success(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        tracker.api.artifact.return_value.digest="different"
        self.assert_failed_finalization(tracker,run,metrics,files,history)

    def test_manifest_summary_mismatch_aborts_without_success(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        remote.summary["reproducibility_manifest_sha256"]="b"*64
        self.assert_failed_finalization(tracker,run,metrics,files,history)

    def test_readback_timeout_aborts_without_success_and_redacts(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        tracker.api.run.side_effect=TimeoutError("FAKE_CREDENTIAL_NEVER_PERSIST")
        self.assert_failed_finalization(tracker,run,metrics,files,history)
        self.assertEqual(tracker.api.run.call_count,3)

    def test_history_failure_aborts_without_success(self):
        for value in ([],[{"epoch":2,**{tag:.5 for tag in EPOCH_KEYS.values()}}]):
            with self.subTest(value=value):
                tracker,run,remote,metrics,files,history=self.finalization_fixture()
                remote.scan_history.return_value=value
                self.assert_failed_finalization(tracker,run,metrics,files,history)

    def test_upload_and_wait_failures_abort_without_success_and_redact(self):
        for stage in ("upload","wait"):
            with self.subTest(stage=stage):
                tracker,run,remote,metrics,files,history=self.finalization_fixture()
                target=run.log_artifact if stage=="upload" else run.log_artifact.return_value.wait
                target.side_effect=ConnectionError("FAKE_CREDENTIAL_NEVER_PERSIST")
                self.assert_failed_finalization(tracker,run,metrics,files,history)
                tracker.api.run.assert_not_called()

    def test_transient_active_run_readback_is_refreshed_before_success(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        tracker.api.run.side_effect=[TimeoutError("not yet visible"),remote]
        with patch("fusion.mlp_tracking.time.sleep") as sleep:
            result=tracker.finish(metrics,files,history,"a"*64)
        self.assertEqual(result["status"],"COMPLETE")
        sleep.assert_called_once_with(1)
        self.assertEqual(tracker.api.flush.call_count,2)
        run.finish.assert_called_once_with(exit_code=0)

    def test_local_integrity_failure_aborts_before_success(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        with self.assertRaises(TrackingUnavailable):
            tracker.finish(metrics,files,history,"a"*64,
                           before_finalize=MagicMock(side_effect=ValueError("FAKE_CREDENTIAL_NEVER_PERSIST")))
        run.finish.assert_called_once_with(exit_code=1)
        self.assertEqual(tracker.metadata["status"],"INCOMPLETE")

    def test_success_finalization_error_is_incomplete_with_unknown_remote_state(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        terminal=[]
        def sdk_finish(*,exit_code):
            if terminal: return  # Real SDK ignores finish calls once finished.
            terminal.append(exit_code)
            raise TimeoutError("FAKE_CREDENTIAL_NEVER_PERSIST")
        run.finish.side_effect=sdk_finish
        with self.assertRaises(TrackingUnavailable) as caught:
            tracker.finish(metrics,files,history,"a"*64)
        self.assertNotIn("FAKE_CREDENTIAL",str(caught.exception))
        self.assertEqual(terminal,[0])  # Server state cannot be safely inferred/reversed.
        self.assertEqual([c.kwargs["exit_code"] for c in run.finish.call_args_list],[0,1])
        self.assertEqual(tracker.metadata["status"],"INCOMPLETE")
        self.assertEqual(tracker.metadata["finalization_state"],"UNKNOWN_AFTER_SUCCESS_FINALIZATION_ERROR")
        tracker.abort(); tracker.abort()
        self.assertEqual(run.finish.call_count,2)

    def test_abort_error_is_redacted_and_cleanup_is_idempotent(self):
        tracker,run,remote,metrics,files,history=self.finalization_fixture()
        run.finish.side_effect=ConnectionError("FAKE_CREDENTIAL_NEVER_PERSIST")
        tracker.abort(); tracker.abort()
        run.finish.assert_called_once_with(exit_code=1)
        self.assertEqual(tracker.metadata["status"],"INCOMPLETE")
        self.assertEqual(tracker.metadata["finalization_state"],"UNKNOWN_AFTER_ABORT_ERROR")
        self.assertNotIn("FAKE_CREDENTIAL",json.dumps(tracker.metadata))


if __name__=="__main__":
    unittest.main()
