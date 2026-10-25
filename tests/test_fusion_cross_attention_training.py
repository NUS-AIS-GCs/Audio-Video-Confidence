"""Offline Phase 5 lifecycle tests; all feature tensors and tracking are synthetic."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fusion.artifact_integration import manifest_to_json
from fusion.contracts import ContractError
from fusion.cross_attention_model import CrossAttentionConfig
from fusion.embedding_data import EmbeddingDataset
from fusion.mlp_model import MLPConfig
from fusion.mlp_training import fit_operating_point, model_fingerprint, predict
from fusion.pair_contracts import PairType, SamplingPolicy
from fusion.pair_generator import generate_pairs, GenerationConfig
from fusion.train_cross_attention import CrossAttentionExperimentConfig, run_experiment, initial_training
from fusion.train_mlp import VerifiedData, load_selected_checkpoint, _verify_output_manifest
from fusion.training_contracts import TrainingConfig
from test_fusion_mlp_training import FakeTracker
from test_fusion_pair_generator import ROWS, bundles, record


class AttentionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio, self.vision = bundles({"train": ROWS, "val": (record("v1", 4), record("v2", 4),
                                                                 record("v3", 5), record("v4", 5))})
        self.context = "a" * 64
        self.documents, self.data = {}, {}
        for split in ("train", "val"):
            manifest = generate_pairs(self.audio, self.vision, policy=SamplingPolicy(42),
                config=GenerationConfig(split, easy_negative_budget=7, challenge_budget=3))
            text = manifest_to_json(manifest, self.audio, self.vision, artifact_fingerprint=self.context)
            self.documents[split] = text
            self.data[split] = self.scoped(text, split)
        self.model = CrossAttentionConfig(d_model=8, num_heads=2, feedforward_dim=16, dropout=.1, attention_dropout=.1)

    def scoped(self, text, split="train", digest=None):
        return EmbeddingDataset.from_scoped_json(text, self.audio, self.vision, split=split,
            expected_manifest_sha256=digest or hashlib.sha256(text.encode()).hexdigest(),
            artifact_fingerprint=self.context)

    def config(self, name="run", **changes):
        with patch("fusion.train_cross_attention._output_path"):
            return CrossAttentionExperimentConfig(*([self.root] * 8), self.root / name,
                model=self.model, training=TrainingConfig(seed=42, batch_size=3, epochs=3,
                    selection_metric="roc_auc", optimizer=initial_training().optimizer),
                patience=2, cpu_threads=2, **changes)

    def verified(self):
        return VerifiedData(self.data["train"], self.data["val"], None, {}, [],
            {"fixture": "synthetic TRAIN/val only"},
            {name: {"val_roc_auc": .5, "val_average_precision": .5, "val_f1": .5}
             for name in ("e3", "jsd")})

    def execute(self, config=None, tracker=FakeTracker):
        with patch("fusion.train_cross_attention.repository", return_value={"commit": "b"*40}):
            return run_experiment(config or self.config(), tracker_factory=tracker,
                data_factory=lambda _: self.verified(), progress=lambda _: None)

    def test_scoped_reader_preserves_original_hash_and_every_primary_row(self):
        for split, dataset in self.data.items():
            self.assertEqual(dataset.provenance.manifest_sha256,
                             hashlib.sha256(self.documents[split].encode()).hexdigest())
            self.assertTrue(all(p.pair_type is not PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE for p in dataset.pairs))
            for i, pair in enumerate(dataset.pairs):
                features, target = dataset[i]
                self.assertEqual(set(features), {"audio", "visual", "validity"})
                torch.testing.assert_close(features["audio"], torch.tensor(
                    self.audio.splits[split].embeddings[pair.audio.row_index], dtype=torch.float32))
                torch.testing.assert_close(features["visual"], torch.tensor(
                    self.vision.splits[split].embeddings[pair.visual.row_index], dtype=torch.float32))
                self.assertEqual(target.item(), pair.original_pair_target)

    def test_only_opaque_all_inventory_context_can_differ(self):
        raw = json.loads(self.documents["train"])
        raw["report"]["source_validation"]["missing_audio_metadata_splits"] = []
        raw["report"]["source_validation"]["missing_visual_metadata_splits"] = []
        raw["report"]["source_validation"]["source_overlaps"] = []
        text = json.dumps(raw)
        dataset = self.scoped(text)
        self.assertEqual(len(dataset), len(self.data["train"]))
        self.assertEqual(dataset.provenance.manifest_sha256, hashlib.sha256(text.encode()).hexdigest())
        with self.assertRaises(ContractError):
            self.scoped(text, digest="0"*64)
        for mutate in (
            lambda value: value["primary_pairs"][0].__setitem__("original_pair_target", 0),
            lambda value: value["primary_pairs"][0]["audio"].__setitem__("row_index", 999),
            lambda value: value["report"]["source_validation"].__setitem__("pair_count", 0),
            lambda value: value["config"].__setitem__("split", "test"),
        ):
            value = json.loads(text); mutate(value)
            with self.assertRaises(ContractError):
                self.scoped(json.dumps(value))
        with self.assertRaises(ContractError):
            self.scoped(text, split="test")

    def test_complete_attention_pipeline_freeze_reload_coverage_and_comparison(self):
        config = self.config()
        modes = []
        def observe(model, data, training, **kwargs):
            modes.append(tuple(p.requires_grad for p in model.parameters()))
            return predict(model, data, training, **kwargs)
        with patch("fusion.train_mlp.predict", side_effect=observe) as prediction, \
             patch("fusion.train_mlp.fit_operating_point", wraps=fit_operating_point) as threshold:
            result = self.execute(config)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(threshold.call_count, 1)
        self.assertTrue(modes and all(not any(flags) for flags in modes))
        self.assertTrue(all(call.args[1].partition == "primary" and call.args[1].split in ("train", "val")
                            for call in prediction.call_args_list))
        history = json.loads((config.output_dir / "history.json").read_text())
        self.assertTrue(all(row["train_count"] == 16 and row["distinct_index_coverage_verified"] for row in history))
        self.assertTrue(all(row["optimizer_steps"] == 6 * row["epoch"] for row in history))
        selected = max(history, key=lambda row: row["val_roc_auc"])
        self.assertEqual(result["final"]["selected_epoch"], selected["epoch"])
        self.assertEqual(result["comparison"]["challenge_evaluated"], False)
        self.assertEqual(set(result["comparison"]["delta_cross_attention_minus_reference"]), {"e3", "jsd"})
        model, metadata = load_selected_checkpoint(config.output_dir)
        self.assertEqual(metadata.schema, "gc2-fusion-cross-attention-checkpoint-v1")
        self.assertEqual(model.config, config.model)
        check = json.loads((config.output_dir / "checkpoint_reload.json").read_text())
        self.assertTrue(check["identical_all_validation_logits"])
        self.assertEqual(check["count"], len(self.data["val"]))
        self.assertTrue((config.output_dir / "preregistration.json").is_file())
        self.assertFalse(any("challenge_scores" in path.name or path.name.startswith("test_")
                             for path in config.output_dir.iterdir()))

    def test_repeat_seed_reproduces_selected_state_and_primary_predictions(self):
        a, b = self.config("a"), self.config("b")
        first, second = self.execute(a), self.execute(b)
        self.assertEqual(first["final"], second["final"])
        self.assertEqual((a.output_dir / "val_primary_scores.csv").read_bytes(),
                         (b.output_dir / "val_primary_scores.csv").read_bytes())

    def test_config_and_architecture_guards(self):
        for kwargs in ({"evaluate_challenge": True}, {"tracking_group": "phase4_learned_fusion_mlp"}):
            with self.assertRaises(ContractError):
                self.config(**kwargs)
        with self.assertRaises(ContractError):
            run_experiment(object())
        config = self.config()
        object.__setattr__(config, "model", MLPConfig())
        with self.assertRaises(ContractError):
            config.model.__post_init__() if type(config.model) is CrossAttentionConfig else run_experiment(config)

    def test_preflight_failure_prevents_optimizer_updates(self):
        class Failure(FakeTracker):
            def start(self, config):
                raise RuntimeError("FAKE_CREDENTIAL_NEVER_PERSIST")
        config = self.config()
        with patch("fusion.train_mlp.train_epoch", side_effect=AssertionError("no optimization")), \
             self.assertRaises(ContractError):
            self.execute(config, Failure)
        state = json.loads((config.output_dir / "state.json").read_text())
        self.assertFalse(state["optimization_started"])
        self.assertFalse((config.output_dir / "completion.json").exists())
        self.assertNotIn("FAKE_CREDENTIAL", json.dumps(state))

    def test_changed_frozen_configuration_stops_before_first_optimizer_step(self):
        class Change(FakeTracker):
            def start(self, config):
                result = super().start(config)
                config["training"]["optimizer"]["learning_rate"] = .1
                return result
        config = self.config()
        with patch("fusion.train_mlp.train_epoch", side_effect=AssertionError("no optimization")), \
             self.assertRaises(ContractError):
            self.execute(config, Change)
        self.assertFalse(json.loads((config.output_dir / "state.json").read_text())["optimization_started"])
        self.assertFalse((config.output_dir / "completion.json").exists())

    def test_upload_failure_or_local_plan_tamper_never_publishes_completion(self):
        class UploadFailure(FakeTracker):
            def finish(self, *args, **kwargs):
                raise RuntimeError("synthetic upload failure")
        class Tamper(FakeTracker):
            def finish(self, *args, **kwargs):
                path = self.output / "preregistration.json"
                path.write_text(path.read_text() + " ")
                return super().finish(*args, **kwargs)
        for tracker in (UploadFailure, Tamper):
            config = self.config(tracker.__name__)
            with self.assertRaises(ContractError):
                self.execute(config, tracker)
            self.assertFalse((config.output_dir / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
