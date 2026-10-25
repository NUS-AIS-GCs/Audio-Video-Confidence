"""Offline study configuration/selection tests; synthetic data and mocked Forge only."""
from dataclasses import asdict, replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fusion.artifact_integration import _file_entry, _inventory, _json
from fusion.baseline_tracking import TrackingUnavailable
from fusion.contracts import ContractError
from fusion.mlp_model import LateFusionMLP, MLPConfig
from fusion.mlp_study import (
    ORDER, SCHEMA, SELECTION_NOTICE, chosen_config, experiment_configs, finish_frozen_study, rank_candidates, run_study, selected_challenge,
)
from fusion.mlp_tracking import STUDY_GROUP
from fusion.train_mlp import MLPExperimentConfig
from fusion.training_contracts import TrainingConfig
from test_fusion_mlp_training import FakeTracker
import test_fusion_mlp_training as fixtures


def candidate(label, auc=.9, ap=.9, f1=.9):
    return {"label":label,"val_roc_auc":auc,"val_average_precision":ap,"val_f1":f1,
            "artifact_inventory_fingerprint":"a"*64,"manifest_sha256":{"train":"b"*64,"val":"c"*64},
            "row_reference_sha256":{"train_primary":"d"*64,"val_primary":"e"*64}}


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.policy=patch("fusion.train_mlp._output_path"); self.policy.start(); self.addCleanup(self.policy.stop)

    def base(self,**kwargs):
        return MLPExperimentConfig(*([self.root]*7),self.root/"study",**kwargs)

    def test_exact_three_configs_and_parameter_counts(self):
        configs=experiment_configs(self.base())
        self.assertEqual([label for label,_ in configs],["E1","E2","E3"])
        expected=[((256,128,64),.2,.0001,238081),((512,256,128),.5,.0001,558081),((512,256,128),.2,.01,558081)]
        for (label,config),(widths,dropout,decay,count) in zip(configs,expected):
            self.assertEqual(config.model,replace(MLPConfig(),hidden_dims=widths,dropout=dropout))
            fixed=TrainingConfig(batch_size=64,epochs=30,selection_metric="roc_auc")
            self.assertEqual(config.training,replace(fixed,optimizer=replace(fixed.optimizer,weight_decay=decay)))
            self.assertEqual(sum(p.numel() for p in LateFusionMLP(config.model).parameters()),count)
            self.assertFalse(config.evaluate_challenge)
            self.assertEqual((config.patience,config.cpu_threads,config.pair_seed),(5,4,42))
            self.assertEqual(config.tracking_group,STUDY_GROUP)
            self.assertTrue(config.run_name.startswith(label))
        self.assertEqual(len({c.run_name for _,c in configs}),3)

    def test_protocol_drift_is_rejected(self):
        for base in (self.base(model=MLPConfig(dropout=.1)),self.base(patience=6),
                     self.base(training=TrainingConfig(seed=43,batch_size=64,epochs=30,selection_metric="roc_auc")),
                     self.base(training=TrainingConfig(batch_size=64,epochs=29,selection_metric="roc_auc"))):
            with self.assertRaises(ContractError): experiment_configs(base)

    def test_chosen_config_reuses_existing_objects_without_fresh_output_revalidation(self):
        base=self.base(); configs=experiment_configs(base)
        base.output_dir.mkdir()
        for _,config in configs: config.output_dir.mkdir()
        with patch("fusion.mlp_study.replace",side_effect=AssertionError("must reuse frozen configuration")):
            self.assertIs(chosen_config(base,configs,"original"),base)
            for label,config in configs: self.assertIs(chosen_config(base,configs,label),config)
        with self.assertRaises(ContractError): chosen_config(base,configs,"E4")

    def test_postprocessing_recovery_never_starts_training_and_rejects_missing_frozen_evidence(self):
        with patch("fusion.mlp_study.run_experiment",side_effect=AssertionError("no training")), \
             patch("fusion.mlp_study.ForgeMLPTracker",side_effect=AssertionError("no new W&B run")), \
             self.assertRaises(FileNotFoundError):
            finish_frozen_study(self.root)

    def test_postprocessing_recovery_completes_existing_decision_without_new_runs(self):
        case=self.recovery_fixture()
        study=case["study"]
        selection_before=(study/"selection.json").read_bytes()
        with patch("fusion.mlp_study.run_experiment",side_effect=AssertionError("no training")) as train, \
             patch("fusion.mlp_study.ForgeMLPTracker",side_effect=AssertionError("no W&B run")) as tracker, \
             patch("fusion.mlp_study.primary_evidence",side_effect=lambda label,output:case["rows"][label]), \
             patch("fusion.mlp_study.load_selected_checkpoint",side_effect=AssertionError("no real checkpoints")), \
             patch("fusion.mlp_study._source_inventory",return_value=({},[])), \
             patch("fusion.mlp_study._source_archive"), \
             patch("fusion.mlp_study.selected_challenge",return_value={"count":402}) as diagnostic:
            result=finish_frozen_study(study)
        train.assert_not_called(); tracker.assert_not_called(); diagnostic.assert_called_once()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertTrue(result["postprocessing_recovered"])
        self.assertEqual(result["new_training_run_attempts"],3)
        self.assertEqual((study/"selection.json").read_bytes(),selection_before)
        self.assertTrue((study/"postprocessing/recovery.json").is_file())

    def recovery_fixture(self, *, name="recover", integrity=True):
        """Complete metadata with opaque dummy checkpoint bytes; never torch.load."""
        study=self.root/name; study.mkdir()
        base=replace(self.base(),output_dir=study)
        configs=dict(experiment_configs(base))
        original=self.root/(name+"-original"); original.mkdir()
        rows={}; experiments=[]
        for label in ORDER:
            config=base if label=="original" else configs[label]
            output=original if label=="original" else config.output_dir
            if label!="original": output.mkdir()
            row=candidate(label,auc=.91 if label=="E1" else .9)
            row.update(output_dir=str(output),selected_epoch=1,frozen_threshold=.5,
                       run_url=f"https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/mock-{label}",artifacts=[])
            (output/"best.pt").write_bytes(("opaque synthetic checkpoint "+label).encode())
            (output/"frozen_operating_point.json").write_bytes(_json({"threshold":.5}).encode())
            row["checkpoint_sha256"]=_file_entry("c",output/"best.pt")["sha256"]
            row["threshold_sha256"]=_file_entry("t",output/"frozen_operating_point.json")["sha256"]
            config_record={"model":asdict(config.model),"training":asdict(config.training),"code_sha256":"a"*64,
                           "patience":5,"cpu_threads":4,"pair_seed":42,
                           "parameter_count":sum(p.numel() for p in LateFusionMLP(config.model).parameters()),
                           "wandb_group":config.tracking_group,"wandb_name":config.run_name}
            (output/"config.json").write_bytes(_json(config_record).encode())
            (output/"checkpoint_metadata.json").write_bytes(_json({"model":config_record["model"],"training":config_record["training"]}).encode())
            (output/"tracking.json").write_bytes(_json({"run_url":row["run_url"],"run_id":f"mock-{label}",
                "group":config.tracking_group,"artifacts":[]}).encode())
            rows[label]=row
            if label!="original":
                experiments.append({"label":label,"output_dir":str(output),"model":asdict(config.model),
                    "training":asdict(config.training),"patience":5,"cpu_threads":4,"pair_seed":42,
                    "parameter_count":config_record["parameter_count"],"tracking_group":config.tracking_group,"run_name":config.run_name})
        baseline={"val_roc_auc":.95,"val_average_precision":.94,"val_f1":.93}
        plan={"schema":SCHEMA,"experiments":experiments,"original":deepcopy(rows["original"]),"baseline":baseline,
              "fixed_order":list(ORDER),"ranking":["val_roc_auc","val_average_precision","val_f1","fixed_order"],
              "selection_notice":SELECTION_NOTICE}
        attempts=[{"label":label,"status":"COMPLETE","output_dir":str(configs[label].output_dir)} for label in ORDER[1:]]
        comparison={"candidates":[deepcopy(rows[label]) for label in ORDER],"baseline":deepcopy(baseline),
                    "delta_vs_jsd":{label:{key:rows[label][key]-baseline[key] for key in baseline} for label in ORDER}}
        ranked=rank_candidates(deepcopy(comparison["candidates"]))
        selection={"schema":SCHEMA,"ranked_candidates":ranked,"winner":deepcopy(ranked[0]),
                   "configuration_selection_frozen":True,"challenge_used_for_selection":False,
                   "attempts":deepcopy(attempts),"selection_notice":SELECTION_NOTICE}
        case={"study":study,"rows":rows,"plan":plan,"attempts":attempts,"comparison":comparison,
              "selection":selection,"integrity":integrity}
        preservation=_inventory({f"original/{p.name}":p for p in original.iterdir()})
        (study/"preservation_before.json").write_bytes(_json(preservation).encode())
        self.save_recovery_fixture(case)
        return case

    def save_recovery_fixture(self,case):
        study=case["study"]
        for name,value in (("plan.json",case["plan"]),("attempts.json",case["attempts"]),
                           ("primary_comparison.json",case["comparison"])):
            (study/name).write_bytes(_json(value).encode())
        case["selection"]["plan_sha256"]=_file_entry("p",study/"plan.json")["sha256"]
        (study/"selection.json").write_bytes(_json(case["selection"]).encode())
        if case["integrity"]:
            rank_bytes=json.dumps(case["selection"]["ranked_candidates"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()
            evidence={"schema":SCHEMA,"ranking_sha256":hashlib.sha256(rank_bytes).hexdigest()}
            for key,name in (("plan","plan.json"),("selection","selection.json"),("comparison","primary_comparison.json")):
                evidence[key+"_sha256"]=_file_entry(key,study/name)["sha256"]
            (study/"selection_integrity.json").write_bytes(_json(evidence).encode())

    def assert_recovery_rejected(self,case):
        study=case["study"]
        before={str(p.relative_to(study)):p.read_bytes() for p in study.rglob("*") if p.is_file()}
        dirs_before={str(p.relative_to(study)) for p in study.rglob("*") if p.is_dir()}
        with patch("fusion.mlp_study._write") as write,patch("fusion.mlp_study._source_archive") as archive, \
             patch("fusion.mlp_study.selected_challenge") as diagnostic, \
             patch("fusion.mlp_study.primary_evidence",side_effect=lambda label,output:case["rows"][label]) as primary, \
             patch("fusion.mlp_study.load_selected_checkpoint",side_effect=AssertionError("no real checkpoints")), \
             patch("fusion.mlp_study.run_experiment",side_effect=AssertionError("no training")), \
             patch("fusion.mlp_study.ForgeMLPTracker",side_effect=AssertionError("no W&B")),self.assertRaises(ContractError):
            finish_frozen_study(study)
        write.assert_not_called(); archive.assert_not_called(); diagnostic.assert_not_called()
        self.assertEqual(before,{str(p.relative_to(study)):p.read_bytes() for p in study.rglob("*") if p.is_file()})
        self.assertEqual(dirs_before,{str(p.relative_to(study)) for p in study.rglob("*") if p.is_dir()})
        return primary

    def test_recovery_original_only_ranking_rejected_before_checkpoint_access(self):
        case=self.recovery_fixture()
        row=deepcopy(case["rows"]["original"])
        case["selection"].update(ranked_candidates=[row],winner=row)
        self.save_recovery_fixture(case)
        self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_three_candidate_ranking_rejected(self):
        case=self.recovery_fixture(); case["selection"]["ranked_candidates"].pop()
        self.save_recovery_fixture(case); self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_duplicate_candidate_ids_rejected(self):
        for label in ("E1","E2","E3"):
            with self.subTest(label=label):
                case=self.recovery_fixture(name="duplicate-"+label)
                # Duplicate the specific experiment, omitting original instead.
                case["selection"]["ranked_candidates"]=[deepcopy(case["rows"][n]) for n in ("E1","E2","E3",label)]
                self.save_recovery_fixture(case); self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_unexpected_candidate_id_rejected(self):
        case=self.recovery_fixture(); case["selection"]["ranked_candidates"][-1]["label"]="E4"
        self.save_recovery_fixture(case); self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_missing_ledger_entry_rejected(self):
        case=self.recovery_fixture(); case["attempts"].pop()
        case["selection"]["attempts"]=deepcopy(case["attempts"])
        self.save_recovery_fixture(case); self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_candidate_ledger_directory_and_run_mismatches_rejected(self):
        for key,value in (("output_dir",str(self.root/"wrong-run")),("run_id","wrong-id")):
            with self.subTest(key=key):
                case=self.recovery_fixture(name="ledger-"+key)
                case["attempts"][0][key]=value
                case["selection"]["attempts"]=deepcopy(case["attempts"])
                self.save_recovery_fixture(case); self.assert_recovery_rejected(case)

    def test_recovery_checkpoint_and_threshold_byte_mismatches_rejected(self):
        for name in ("best.pt","frozen_operating_point.json"):
            with self.subTest(name=name):
                case=self.recovery_fixture(name="bytes-"+name)
                path=case["study"]/"E1"/name; path.write_bytes(path.read_bytes()+b"changed")
                self.assert_recovery_rejected(case)

    def test_recovery_candidate_config_and_checkpoint_metadata_mismatches_rejected(self):
        for name in ("config.json","checkpoint_metadata.json"):
            with self.subTest(name=name):
                case=self.recovery_fixture(name="config-"+name)
                path=case["study"]/"E1"/name; value=json.loads(path.read_text())
                value["training"]["optimizer"]["weight_decay"] = .5
                path.write_bytes(_json(value).encode()); self.assert_recovery_rejected(case)

    def test_recovery_tracking_run_mismatch_rejected(self):
        case=self.recovery_fixture(); path=case["study"]/"E1/tracking.json"
        value=json.loads(path.read_text()); value["run_id"]="different-id"
        path.write_bytes(_json(value).encode()); self.assert_recovery_rejected(case)

    def test_recovery_immutable_original_reference_mismatch_rejected(self):
        case=self.recovery_fixture(); case["plan"]["original"]["output_dir"]=str(self.root/"impostor")
        self.save_recovery_fixture(case); self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_frozen_winner_inconsistency_rejected(self):
        case=self.recovery_fixture(); case["selection"]["winner"]=deepcopy(case["rows"]["original"])
        self.save_recovery_fixture(case); self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_unsorted_ranking_rejected(self):
        case=self.recovery_fixture(); rows=case["selection"]["ranked_candidates"]
        rows[1],rows[2]=rows[2],rows[1]
        self.save_recovery_fixture(case); self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_tampered_stored_ranking_selection_and_comparison_hashes_rejected(self):
        for key in ("ranking_sha256","selection_sha256","comparison_sha256"):
            with self.subTest(key=key):
                case=self.recovery_fixture(name="hash-"+key)
                path=case["study"]/"selection_integrity.json"; value=json.loads(path.read_text()); value[key]="0"*64
                path.write_bytes(_json(value).encode()); self.assert_recovery_rejected(case).assert_not_called()

    def test_recovery_legacy_ranking_and_recorded_selection_integrity_are_checked(self):
        case=self.recovery_fixture(integrity=False)
        case["selection"]["ranked_candidates"][0]["val_roc_auc"] += .001
        self.save_recovery_fixture(case); self.assert_recovery_rejected(case).assert_not_called()
        recorded=self.recovery_fixture(name="recorded",integrity=False)
        audit=recorded["study"]/"postprocessing"; audit.mkdir()
        (audit/"recovery.json").write_bytes(_json({"frozen_selection_sha256":"0"*64}).encode())
        self.assert_recovery_rejected(recorded).assert_not_called()

    def test_recovery_legacy_complete_ranking_remains_supported_without_rewriting_selection(self):
        case=self.recovery_fixture(integrity=False); study=case["study"]
        selection_before=(study/"selection.json").read_bytes()
        with patch("fusion.mlp_study.primary_evidence",side_effect=lambda label,output:case["rows"][label]), \
             patch("fusion.mlp_study._source_inventory",return_value=({},[])),patch("fusion.mlp_study._source_archive"), \
             patch("fusion.mlp_study.load_selected_checkpoint",side_effect=AssertionError("no checkpoints")), \
             patch("fusion.mlp_study.run_experiment",side_effect=AssertionError("no training")), \
             patch("fusion.mlp_study.ForgeMLPTracker",side_effect=AssertionError("no W&B")), \
             patch("fusion.mlp_study.selected_challenge",return_value={"count":402}):
            result=finish_frozen_study(study)
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual((study/"selection.json").read_bytes(),selection_before)
        self.assertFalse((study/"selection_integrity.json").exists())

    def test_recovery_completed_reentry_rejected_without_any_output(self):
        case=self.recovery_fixture(); (case["study"]/"result.json").write_text('{"status":"COMPLETE"}')
        self.assert_recovery_rejected(case).assert_not_called()

    def test_ranking_uses_full_precision_then_ap_f1_fixed_order(self):
        rows=[candidate(label) for label in reversed(ORDER)]
        self.assertEqual([r["label"] for r in rank_candidates(rows)],list(ORDER))
        rows[0]["val_f1"]+=.01
        self.assertEqual(rank_candidates(rows)[0]["label"],"E3")
        rows[1]["val_average_precision"]+=.01
        self.assertEqual(rank_candidates(rows)[0]["label"],"E2")
        rows[2]["val_roc_auc"]+=1e-12
        self.assertEqual(rank_candidates(rows)[0]["label"],"E1")
        rows[3]["val_roc_auc"]+=2e-12
        self.assertEqual(rank_candidates(rows)[0]["label"],"original")

    def test_ranking_rejects_invalid_inventory_and_ignores_challenge(self):
        class Forbidden:
            def __float__(self): raise AssertionError("challenge must not select")
        row=candidate("original"); row["challenge_rejection"]=Forbidden()
        self.assertEqual(rank_candidates([row])[0]["label"],"original")
        for rows in ([],[candidate("E1")],[candidate("original"),candidate("original")],
                     [candidate("original"),candidate("jensen_shannon")],[candidate("original",auc=float("nan"))]):
            with self.assertRaises(ContractError): rank_candidates(rows)

    def orchestrate(self,failed=None):
        base=self.base(); configs=experiment_configs(base)
        reference=self.root/"original"; reference.mkdir()
        baseline={"val_roc_auc":.95,"val_average_precision":.94,"val_f1":.93}
        data=SimpleNamespace(paths={},baseline=baseline)
        observed=[]
        def run(config,progress):
            label=config.output_dir.name
            observed.append(label)
            self.assertFalse(config.evaluate_challenge)
            self.assertTrue((base.output_dir/"plan.json").is_file())
            if label==failed: raise ConnectionError("FAKE_CREDENTIAL_NEVER_PERSIST")
        def evidence(label,output): return candidate(label,auc=.91 if label=="E1" else .9)
        def challenge(study,winner,config):
            self.assertEqual(observed,["E1","E2","E3"])
            frozen=json.loads((study/"selection.json").read_text())
            self.assertEqual(frozen["winner"],winner)
            self.assertTrue(frozen["configuration_selection_frozen"])
            self.assertFalse(frozen["challenge_used_for_selection"])
            return {"count":402,"exploratory":True}
        with patch("fusion.mlp_study.preflight",return_value=(configs,{},candidate("original"),data,{})), \
             patch("fusion.mlp_study._source_inventory",return_value=({},[])), \
             patch("fusion.mlp_study._source_archive"), \
             patch("fusion.mlp_study.primary_evidence",side_effect=evidence), \
             patch("fusion.mlp_study.selected_challenge",side_effect=challenge) as diagnostic:
            result=run_study(base,reference,experiment_runner=run,progress=lambda _:None)
        diagnostic.assert_called_once()
        return base,result

    def test_study_runs_fixed_inventory_freezes_then_challenges(self):
        base,result=self.orchestrate()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual(result["new_training_run_attempts"],3)
        self.assertEqual(result["selected_label"],"E1")
        comparison=json.loads((base.output_dir/"primary_comparison.json").read_text())
        self.assertAlmostEqual(comparison["delta_vs_jsd"]["E1"]["val_roc_auc"],-.04)

    def test_failed_attempt_is_recorded_once_redacted_and_not_replaced(self):
        base,result=self.orchestrate(failed="E2")
        self.assertEqual(result["status"],"PARTIAL")
        self.assertEqual([a["label"] for a in result["attempts"]],["E1","E2","E3"])
        self.assertEqual([a["status"] for a in result["attempts"]],["COMPLETE","FAILED","COMPLETE"])
        self.assertNotIn("FAKE_CREDENTIAL",json.dumps(result))
        selection=json.loads((base.output_dir/"selection.json").read_text())
        self.assertNotIn("E2",[r["label"] for r in selection["ranked_candidates"]])

    def test_primary_only_runner_never_predicts_challenge_or_test(self):
        fixture=fixtures.TrainingPipelineTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        config=fixture.config(evaluate_challenge=False,tracking_group=STUDY_GROUP,run_name="E1-offline-fixture")
        from fusion.train_mlp import predict
        seen=[]
        def guarded(model,data,*args,**kwargs):
            self.assertEqual(data.partition,"primary")
            self.assertIn(data.split,("train","val"))
            seen.append((data.split,data.partition))
            return predict(model,data,*args,**kwargs)
        with patch("fusion.train_mlp.predict",side_effect=guarded): result=fixture.execute(config)
        self.assertEqual(result["status"],"COMPLETE")
        self.assertNotIn("val_challenge",result["final"])
        self.assertFalse((config.output_dir/"val_challenge_scores.csv").exists())
        self.assertTrue(seen)
        self.assertEqual(FakeTracker.instances[-1].config["wandb_group"],STUDY_GROUP)

    def test_tracking_study_group_name_and_invalid_overrides(self):
        fixture=fixtures.ForgeTrackingTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        tracker,run=fixture.tracker()
        with patch.object(tracker,"preflight_access"):
            result=tracker.start({"training":{"seed":42},"wandb_group":STUDY_GROUP,"wandb_name":"E2-dropout-0p5-seed42"})
        self.assertEqual(result["group"],STUDY_GROUP)
        self.assertEqual(tracker.sdk.init.call_args.kwargs["name"],"E2-dropout-0p5-seed42")
        self.assertEqual(tracker.sdk.init.call_args.kwargs["group"],STUDY_GROUP)
        for override in ({"wandb_group":"unapproved"},{"wandb_name":"invalid name with spaces"}):
            tracker,run=fixture.tracker()
            with patch.object(tracker,"preflight_access") as access,self.assertRaises(TrackingUnavailable):
                tracker.start({"training":{"seed":42},**override})
            access.assert_not_called(); tracker.sdk.init.assert_not_called()

    def test_challenge_requires_persisted_decision_and_bound_checkpoint(self):
        fixture=fixtures.TrainingPipelineTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        config=fixture.config(evaluate_challenge=False)
        fixture.execute(config)
        winner={"label":"E1","output_dir":str(config.output_dir),
                "checkpoint_sha256":_file_entry("c",config.output_dir/"best.pt")["sha256"],
                "threshold_sha256":_file_entry("t",config.output_dir/"frozen_operating_point.json")["sha256"]}
        with patch("fusion.mlp_study.predict") as predict,self.assertRaises(FileNotFoundError):
            selected_challenge(self.root,winner,config,data_factory=lambda _:fixture.verified())
        predict.assert_not_called()
        selection={"winner":winner,"configuration_selection_frozen":True}
        (self.root/"selection.json").write_text(json.dumps(selection))
        result=selected_challenge(self.root,winner,config,data_factory=lambda _:fixture.verified())
        self.assertEqual(result["count"],len(fixture.challenge))
        self.assertTrue(result["exploratory"])
        self.assertFalse(result["selection_changed"])
        winner["checkpoint_sha256"]="0"*64
        (self.root/"selection.json").write_text(json.dumps({**selection,"winner":winner}))
        with self.assertRaises(ContractError): selected_challenge(self.root,winner,config,data_factory=lambda _:fixture.verified())


if __name__=="__main__":
    unittest.main()
