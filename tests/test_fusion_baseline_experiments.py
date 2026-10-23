"""Offline file fixtures and mocked Forge only; never use real export roots."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from fusion import ContractError
from fusion.artifact_integration import _json, run_integration
from fusion.baseline_tracking import BASE_URL, ENTITY, PROJECT, ForgeTracker, TrackingUnavailable
from fusion.baselines import METHODS
from fusion.run_baseline_experiments import (
    ExperimentConfig, ROOT, _expected_files, rank_methods, run_experiments, upload_saved,
)
from test_fusion_artifact_integration import write_fixture


class MockTracker:
    instances = []

    def __init__(self, output):
        self.output = output
        self.metadata = {"status": "RUN_STARTED", "base_url": BASE_URL, "entity": ENTITY, "project": PROJECT,
                         "run_url": "https://forge.coreweave.com/GC2-AValidate/avalidate/runs/mock"}
        self.calls = []
        self.__class__.instances.append(self)

    def start(self, config):
        self.calls.append(("start", config))

    def upload(self, metrics, rows, files, digest):
        self.calls.append(("upload", metrics, rows, files, digest))
        assert all(path.is_file() for path in files)
        return dict(self.metadata, status="COMPLETE", artifact_name="GC2-AValidate/avalidate/mock:v0")

    def abort(self):
        self.calls.append(("abort",))


def fresh_output(path, inputs):
    path = Path(path).resolve()
    if path.exists() or any(path.is_relative_to(p) or p.is_relative_to(path) for p in inputs):
        raise ContractError("fresh separate output required")
    return path


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        a, v, official, counts = write_fixture(self.root)
        (a / "SOURCE.txt").write_text("synthetic fixture: beats_lora_r8_v1 beats_iter3", encoding="utf-8")
        (v / "SOURCE.txt").write_text("synthetic fixture: resnet18_v1", encoding="utf-8")
        self.integration = self.root / "phase2"
        self.report = run_integration(a, v, official, expected_counts=counts, output_dir=self.integration)
        reference = self.root / "reference"
        reference.mkdir()
        for name in ("Annotations.txt", "train_order.h5", "val_order.h5", "test_order.h5"):
            (reference / name).write_bytes(b"synthetic reference; verifier mocked in runner tests")
        evidence = self.root / "verified" / "verification.json"
        evidence.parent.mkdir()
        self.reference_report = {"schema": "gc2-ave-reference-v1", "status": "PASS", "fixture": "synthetic-only"}
        evidence.write_text(_json(self.reference_report), encoding="utf-8")
        for target, value in (("fusion.run_baseline_experiments.OFFICIAL_COUNTS", counts),
                              ("fusion.run_baseline_experiments._output_path", fresh_output)):
            p = patch(target, value)
            p.start(); self.addCleanup(p.stop)
        p = patch("fusion.run_baseline_experiments.verify_reference", return_value=self.reference_report)
        self.reference_mock = p.start(); self.addCleanup(p.stop)
        self.config = ExperimentConfig(a, v, official, reference, evidence, self.integration, self.root / "result")
        MockTracker.instances.clear()

    def run_fixture(self):
        return run_experiments(self.config, tracker_factory=MockTracker)

    def test_full_run_counts_lifecycle_exports_and_mock_upload(self):
        result = self.run_fixture()
        self.assertEqual((result["local_experiment_status"], result["tracking_status"]), ("COMPLETE", "COMPLETE"))
        summary = json.loads((self.config.output_dir / "dataset_summary.json").read_text())
        for split in ("train", "val"):
            n = summary["annotation_counts"][split]
            self.assertEqual(summary["pair_counts"][split], {"primary": 2 * n, "challenge": n})
        self.assertTrue(summary["all_scores_finite"])
        config = json.loads((self.config.output_dir / "config.json").read_text())
        self.assertEqual(config["exports"]["audio"]["model"], "BEATs+LoRA")
        self.assertEqual(config["exports"]["vision"]["model"], "ResNet-18")
        self.assertEqual(config["epochs"], "not applicable")
        calls = MockTracker.instances[-1].calls
        self.assertEqual([c[0] for c in calls], ["start", "upload"])
        self.assertEqual(len(calls[-1][3]), len(_expected_files()) + 1)
        for method in METHODS:
            report = json.loads((self.config.output_dir / f"reports/{method}_val_primary.json").read_text())
            self.assertEqual(report["context"]["fitting_pair_reuse_count"], report["classification"]["count"])
            self.assertIn("no independent generalization", report["context"]["validation_reuse_note"])
            policy = json.loads((self.config.output_dir / f"policies/{method}.json").read_text())
            self.assertEqual(policy["policy"]["source"]["split"], "val")
            self.assertEqual(policy["policy"]["source"]["partition"], "primary")

    def test_no_test_manifest_deserialization_or_test_scoring(self):
        import fusion.run_baseline_experiments as runner
        reader, extractor = runner.manifest_from_json, runner.extract_pair_probabilities
        seen = []
        def read(text, *args, **kwargs):
            split = json.loads(text)["config"]["split"]
            self.assertIn(split, ("train", "val")); seen.append(split)
            return reader(text, *args, **kwargs)
        def extract(*args, **kwargs):
            self.assertIn(kwargs["split"], ("train", "val"))
            self.assertNotIn("limit_per_partition", kwargs)
            return extractor(*args, **kwargs)
        with patch.object(runner, "manifest_from_json", side_effect=read), patch.object(runner, "extract_pair_probabilities", side_effect=extract):
            self.run_fixture()
        self.assertEqual(seen, ["train", "val"])
        self.assertFalse(any("test" in p.name for p in self.config.output_dir.rglob("*.csv")))

    def test_all_policies_frozen_before_any_operating_report(self):
        import fusion.run_baseline_experiments as runner
        freeze, evaluate = runner.freeze_threshold, runner.evaluate_primary
        frozen_methods = []
        def freezing(policy):
            result = freeze(policy)
            frozen_methods.append(policy.method)
            return result
        def evaluating(*args):
            self.assertEqual(tuple(frozen_methods), METHODS)
            return evaluate(*args)
        with patch.object(runner, "freeze_threshold", side_effect=freezing), patch.object(runner, "evaluate_primary", side_effect=evaluating):
            self.run_fixture()

    def test_missing_and_corrupted_inputs_fail_before_tracking(self):
        path = self.config.audio_root / "train_probs.npy"
        original = path.read_bytes()
        path.write_bytes(b"corrupt")
        with self.assertRaises(ContractError):
            self.run_fixture()
        self.assertEqual(MockTracker.instances, [])
        path.write_bytes(original)
        path.unlink()
        with self.assertRaises(ContractError):
            self.run_fixture()
        self.assertFalse(self.config.output_dir.exists())

    def test_manifest_hash_corruption_and_missing_completion_fail(self):
        manifest = self.integration / "train_pairs.json"
        manifest.write_text(manifest.read_text() + " ")
        with self.assertRaises(ContractError):
            self.run_fixture()
        (self.integration / "validation.json").unlink()
        with self.assertRaises(OSError):
            self.run_fixture()
        self.assertEqual(MockTracker.instances, [])

    def test_fresh_reference_mismatch_blocks_experiment(self):
        self.reference_mock.return_value = {"status": "FAIL"}
        with self.assertRaises(ContractError):
            self.run_fixture()
        self.assertEqual(MockTracker.instances, [])

    def test_seed_mismatch_and_invalid_seed_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ContractError):
            run_experiments(replace(self.config, seed=1), tracker_factory=MockTracker)
        for seed in (True, -1, 2**64):
            with self.assertRaises(ContractError):
                replace(self.config, seed=seed)

    def test_truncated_manifests_and_wrong_split_rejected(self):
        import fusion.run_baseline_experiments as runner
        original = runner.manifest_from_json
        from dataclasses import replace
        def truncated(*args, **kwargs):
            manifest = original(*args, **kwargs)
            return replace(manifest, primary_pairs=manifest.primary_pairs[:-1])
        with patch.object(runner, "manifest_from_json", side_effect=truncated), self.assertRaises(ContractError):
            self.run_fixture()
        self.assertFalse((self.config.output_dir / "completion.json").exists())
        with self.assertRaises(ContractError):
            runner._full_manifest('{"config":{"split":"test"}}', "train", None, None, "0" * 64, 42)

    def test_tracking_failure_preserves_complete_local_evidence_without_secret(self):
        class Failure(MockTracker):
            def start(self, config):
                raise RuntimeError("FAKE_CREDENTIAL_DO_NOT_PERSIST")
        result = run_experiments(self.config, tracker_factory=Failure)
        self.assertEqual(result["local_experiment_status"], "COMPLETE")
        self.assertEqual(result["tracking_status"], "BLOCKED_WANDB_UPLOAD")
        text = (self.config.output_dir / "tracking.json").read_text()
        self.assertIn("RuntimeError", text)
        self.assertNotIn("FAKE_CREDENTIAL", text)
        self.assertTrue((self.config.output_dir / "output_manifest.json").is_file())

    def test_upload_failure_preserves_statistical_evidence(self):
        class Failure(MockTracker):
            def upload(self, *args):
                raise ConnectionError("FAKE_CREDENTIAL_DO_NOT_PERSIST")
        result = run_experiments(self.config, tracker_factory=Failure)
        self.assertEqual(result["tracking_status"], "BLOCKED_WANDB_UPLOAD")
        text = (self.config.output_dir / "tracking.json").read_text()
        self.assertNotIn("FAKE_CREDENTIAL", text)
        self.assertIn("ConnectionError", text)

    def test_stable_serialization_and_inputs_unchanged(self):
        import fusion.run_baseline_experiments as runner
        input_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for root in
            (self.config.audio_root, self.config.vision_root, self.integration) for p in root.iterdir() if p.is_file()}
        self.run_fixture()
        before = {name: (self.config.output_dir / name).read_bytes() for name in _expected_files()}
        from dataclasses import replace
        with patch.object(runner, "_repository", return_value=json.loads(before["config.json"])["git"]):
            second = replace(self.config, output_dir=self.root / "second")
            run_experiments(second, tracker_factory=MockTracker)
        for name, data in before.items():
            self.assertEqual(data, (second.output_dir / name).read_bytes(), name)
        self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in input_hashes.items()))

    def test_challenges_separate_and_never_headline_ranked(self):
        self.run_fixture()
        for method in METHODS:
            challenge = json.loads((self.config.output_dir / f"reports/{method}_val_challenge.json").read_text())
            self.assertEqual(challenge["positive_count"], 0)
            self.assertNotIn("roc_auc", challenge)
            self.assertNotIn("f1", challenge)
            self.assertEqual(challenge["classified_as_original"] + challenge["rejected"], challenge["count"])

    def test_uploaded_evidence_corruption_is_rejected_before_new_run(self):
        self.run_fixture()
        (self.config.output_dir / "comparison.csv").write_text("changed")
        # Temporary fixtures are outside ROOT; bypass only the upload location guard.
        with patch("fusion.run_baseline_experiments.ROOT", self.root), patch("fusion.run_baseline_experiments.subprocess.run") as command:
            command.return_value.returncode = 0
            previous = len(MockTracker.instances)
            with self.assertRaises(ContractError):
                upload_saved(self.config.output_dir, tracker_factory=MockTracker)
            self.assertEqual(len(MockTracker.instances), previous)

    def test_saved_upload_retries_without_recomputing_predictions(self):
        self.run_fixture()
        before = (self.config.output_dir / "output_manifest.json").read_bytes()
        with patch("fusion.run_baseline_experiments.ROOT", self.root), patch("fusion.run_baseline_experiments.subprocess.run") as command, patch("fusion.run_baseline_experiments._compute", side_effect=AssertionError("must not rescore")):
            command.return_value.returncode = 0
            result = upload_saved(self.config.output_dir, tracker_factory=MockTracker)
        self.assertEqual(result["tracking_status"], "COMPLETE")
        self.assertEqual(before, (self.config.output_dir / "output_manifest.json").read_bytes())

    def test_artifact_path_escape_rejected_before_new_tracking_run(self):
        self.run_fixture()
        manifest = self.config.output_dir / "output_manifest.json"
        payload = json.loads(manifest.read_text())
        payload["files"][0]["role"] = "../FAKE_SECRET_FILE"
        manifest.write_text(_json(payload))
        completion_path = self.config.output_dir / "completion.json"
        completion = json.loads(completion_path.read_text())
        completion["output_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        completion_path.write_text(_json(completion))
        with patch("fusion.run_baseline_experiments.ROOT", self.root), patch("fusion.run_baseline_experiments.subprocess.run") as command:
            command.return_value.returncode = 0
            previous = len(MockTracker.instances)
            with self.assertRaises(ContractError):
                upload_saved(self.config.output_dir, tracker_factory=MockTracker)
            self.assertEqual(len(MockTracker.instances), previous)


class RankingAndConfigurationTests(unittest.TestCase):
    def rows(self):
        return [dict(method=m, val_roc_auc=.5, val_average_precision=.5, val_f1=.5,
                     val_challenge_rejection=1. if i else 0.) for i, m in enumerate(METHODS)]

    def test_ranking_ties_use_fixed_order_and_ignore_challenge(self):
        rows = self.rows()
        self.assertEqual([r["method"] for r in rank_methods(rows[::-1])], list(METHODS))
        for row in rows:
            row["val_challenge_rejection"] = 1 - row["val_challenge_rejection"]
        self.assertEqual(rank_methods(rows)[0]["method"], METHODS[0])

    def test_ranking_metric_precedence_and_invalid_values(self):
        for field in ("val_f1", "val_average_precision", "val_roc_auc"):
            rows = self.rows()
            rows[-1][field] = .6
            self.assertEqual(rank_methods(rows)[0]["method"], METHODS[-1])
        rows = self.rows(); rows[-1]["val_roc_auc"] = .51; rows[0]["val_f1"] = 1.
        self.assertEqual(rank_methods(rows)[0]["method"], METHODS[-1])
        for value in (None, np.nan, np.inf, -.1, 1.1, True):
            rows = self.rows(); rows[0]["val_roc_auc"] = value
            with self.assertRaises(ContractError):
                rank_methods(rows)

    def test_output_must_be_fresh_ignored_inside_repo(self):
        paths = [ROOT / "_local/datasets/AVE"] * 4
        for output in (ROOT / "docs/not_ignored", ROOT / "_local/datasets/AVE/subdir", ROOT.parent / "outside"):
            with self.assertRaises(ContractError):
                ExperimentConfig(*paths, ROOT / "_local/fusion/verified/verification.json",
                                 ROOT / "_local/fusion/task28_real", output)


class ForgeAdapterTests(unittest.TestCase):
    def test_offline_and_wrong_destination_are_explicit_blockers(self):
        for env in ({"WANDB_MODE": "offline"}, {"WANDB_BASE_URL": "https://api.wandb.ai"},
                    {"WANDB_ENTITY": "wrong"}, {"WANDB_PROJECT": "wrong"}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(TrackingUnavailable):
                ForgeTracker(Path("unused")).start({})

    def test_mocked_sdk_auth_failure_does_not_leak_exception_credentials(self):
        wb, sdk, lib = types.ModuleType("wandb"), types.ModuleType("wandb.sdk"), types.ModuleType("wandb.sdk.lib")
        lib.wbauth = types.SimpleNamespace(authenticate_session=MagicMock(side_effect=RuntimeError("FAKE_CREDENTIAL_DO_NOT_PERSIST")))
        modules = {"wandb": wb, "wandb.sdk": sdk, "wandb.sdk.lib": lib}
        with patch.dict("sys.modules", modules), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(TrackingUnavailable) as caught:
                ForgeTracker(Path("unused")).start({})
        self.assertNotIn("FAKE_CREDENTIAL", str(caught.exception))
        self.assertIn("RuntimeError", str(caught.exception))

    def test_missing_sdk_credentials_never_prompts_or_initializes_run(self):
        wb, sdk, lib = types.ModuleType("wandb"), types.ModuleType("wandb.sdk"), types.ModuleType("wandb.sdk.lib")
        wb.init = MagicMock()
        lib.wbauth = types.SimpleNamespace(authenticate_session=MagicMock(return_value=None))
        with patch.dict("sys.modules", {"wandb": wb, "wandb.sdk": sdk, "wandb.sdk.lib": lib}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(TrackingUnavailable, "No configured credentials"):
                ForgeTracker(Path("unused")).start({})
        wb.init.assert_not_called()
        self.assertFalse(lib.wbauth.authenticate_session.call_args.kwargs["prompt"])

    def test_mocked_sdk_success_uses_forge_and_waits_for_artifact(self):
        wb, sdk, lib = types.ModuleType("wandb"), types.ModuleType("wandb.sdk"), types.ModuleType("wandb.sdk.lib")
        auth = types.SimpleNamespace(host=types.SimpleNamespace(url=BASE_URL))
        lib.wbauth = types.SimpleNamespace(authenticate_session=MagicMock(return_value=auth))
        run = types.SimpleNamespace(settings=types.SimpleNamespace(mode="online", base_url=BASE_URL),
            entity=ENTITY, project=PROJECT, id="mock", url="https://forge.coreweave.com/GC2-AValidate/avalidate/runs/mock",
            log=MagicMock(), finish=MagicMock(), summary={})
        artifact = types.SimpleNamespace(add_file=MagicMock(), wait=MagicMock(),
            qualified_name="GC2-AValidate/avalidate/mock:v0", digest="mock-digest", url="https://forge.coreweave.com/mock")
        run.log_artifact = MagicMock(return_value=artifact)
        api = types.SimpleNamespace(viewer=object(), project=MagicMock(return_value=object()))
        wb.Api = MagicMock(return_value=api)
        wb.Settings = lambda **kwargs: kwargs
        wb.init = MagicMock(return_value=run)
        wb.Artifact = MagicMock(return_value=artifact)
        wb.Table = MagicMock(return_value="mock-table")
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"wandb": wb, "wandb.sdk": sdk, "wandb.sdk.lib": lib}), patch.dict(os.environ, {}, clear=True), patch("fusion.baseline_tracking.importlib.metadata.version", return_value="mock-sdk"):
            output = Path(temp)
            file = output / "report.json"
            file.write_text("{}")
            tracker = ForgeTracker(output)
            tracker.start({"method": "comparison"})
            result = tracker.upload({"metric": .5}, [{"method": "top1"}], [file], "f" * 64)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(wb.init.call_args.kwargs["settings"]["base_url"], BASE_URL)
        self.assertEqual(wb.init.call_args.kwargs["entity"], ENTITY)
        self.assertNotIn("api_key", wb.init.call_args.kwargs)
        artifact.wait.assert_called_once_with(timeout=180)
        run.finish.assert_called_once_with(exit_code=0)


if __name__ == "__main__":
    unittest.main()
