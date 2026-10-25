"""Synthetic topology, checkpoint and frozen-inventory tests; no real training."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fusion.contracts import ContractError
from fusion.cross_attention_model import (
    BidirectionalCrossAttention, CrossAttentionConfig,
    UnidirectionalAttentionConfig, UnidirectionalCrossAttention,
)
from fusion.cross_attention_ablations import (
    AblationExperimentConfig, ForgeAblationTracker, GROUP, LABELS, NAMES,
    configurations, rank_attention, variant_model,
)
from fusion.model_interface import build_model, model_config_from_dict, validate_live_model
from fusion.train_mlp import load_selected_checkpoint, save_checkpoint
from fusion.training_contracts import build_optimizer
from test_fusion_checkpoint_binding import metadata
from test_fusion_mlp_training import FakeTracker
import test_fusion_cross_attention_training as training_fixture
from fusion.train_cross_attention import run_experiment as run_initial, CrossAttentionExperimentConfig
from fusion.train_mlp import VerifiedData
from fusion.training_contracts import TrainingConfig
import fusion.cross_attention_ablations as study_module


def small(label):
    return replace(variant_model(label), d_model=8, num_heads=2, feedforward_dim=16,
                   dropout=0., attention_dropout=0.)


class AblationTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def features(self, batch=3):
        generator = torch.Generator().manual_seed(91)
        a = torch.randn(batch, 10, 256, generator=generator)
        v = torch.randn(batch, 10, 512, generator=generator)
        mask = torch.ones(batch, 10, dtype=torch.bool); mask[:, [1, 6, 9]] = False
        return a, v, mask

    def test_exact_inventory_parameter_counts_and_absent_branches(self):
        self.assertEqual(LABELS, ("A1", "A2", "A3"))
        for label, expected in (("A1", 724481), ("A2", 724481), ("A3", 1251841)):
            model = build_model(variant_model(label)).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertEqual(model.correspondence_head.in_features, 512 if label == "A3" else 256)
            self.assertEqual(hasattr(model, "audio_to_video"), label != "A2")
            self.assertEqual(hasattr(model, "video_to_audio"), label != "A1")
            self.assertEqual(model.temporal_positions.count_nonzero().item() == 0, label == "A3")
            for size in (1, 3, 7):
                with torch.no_grad():
                    output = model(*self.features(size), return_bin_representations=True, return_attention_weights=True)
                self.assertEqual(output.logits.shape, (size, 1))
                self.assertEqual(output.bin_representations.shape, (size, 10, 512 if label == "A3" else 256))
                self.assertEqual(output.audio_to_video_attention is None, label == "A2")
                self.assertEqual(output.video_to_audio_attention is None, label == "A1")

    def test_qkv_directions_and_no_passive_stream_or_zero_branch(self):
        for label, branch, q_side, c_side in (("A1", "audio_to_video", "audio", "visual"),
                                               ("A2", "video_to_audio", "visual", "audio")):
            model = build_model(small(label)).eval()
            a, v, mask = self.features()
            invalid = ~mask.unsqueeze(-1)
            streams = {"audio": (model.audio_projection(a.masked_fill(invalid, 0)) + model.temporal_positions).masked_fill(invalid, 0),
                       "visual": (model.visual_projection(v.masked_fill(invalid, 0)) + model.temporal_positions).masked_fill(invalid, 0)}
            seen = []
            handle = getattr(model, branch).attention.register_forward_pre_hook(lambda _, args: seen.append(args))
            try:
                output = model(a, v, mask, return_bin_representations=True)
            finally:
                handle.remove()
            torch.testing.assert_close(seen[0][0], streams[q_side])
            torch.testing.assert_close(seen[0][1], streams[c_side])
            self.assertIs(seen[0][1], seen[0][2])
            expected, _ = getattr(model, branch)(streams[q_side], streams[c_side], mask)
            torch.testing.assert_close(output.bin_representations, expected)
            torch.testing.assert_close(output.logits, model.correspondence_head(expected.sum(1)/mask.sum(1, keepdim=True)))

    def test_both_modalities_input_dependent_finite_gradients_and_mask_invariance(self):
        for label in LABELS:
            model = build_model(small(label)).eval()
            a, v, mask = self.features()
            a.requires_grad_(); v.requires_grad_()
            output = model(a, v, mask, return_attention_weights=True)
            self.assertGreater((output.logits-model(-a, v, mask).logits).abs().max(), 0)
            self.assertGreater((output.logits-model(a, -v, mask).logits).abs().max(), 0)
            changed_a, changed_v = a.detach().clone(), v.detach().clone()
            changed_a[~mask] = torch.finfo(a.dtype).max
            changed_v[~mask] = -torch.finfo(v.dtype).max
            self.assertTrue(torch.equal(output.logits, model(changed_a, changed_v, mask).logits))
            output.logits.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            for gradient in (a.grad, v.grad):
                self.assertGreater(gradient[mask].abs().sum(), 0)
                self.assertEqual(gradient[~mask].abs().sum(), 0)
            for weights in (output.audio_to_video_attention, output.video_to_audio_attention):
                if weights is not None:
                    torch.testing.assert_close(weights.sum(-1), mask[:, None, :].float().expand(3, 2, 10))
                    self.assertEqual(weights.masked_select(~mask[:, None, None, :].expand_as(weights)).abs().sum(), 0)

    def test_deterministic_initialization_and_legacy_serialized_configuration(self):
        original = CrossAttentionConfig()
        self.assertNotIn("attention_direction", asdict(original))
        self.assertEqual(original.attention_direction, "bidirectional")
        for label in LABELS:
            before = torch.get_rng_state().clone()
            first, second = build_model(small(label)), build_model(small(label))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            for key, value in first.state_dict().items():
                self.assertTrue(torch.equal(value, second.state_dict()[key]))

    def test_malformed_topology_and_checkpoint_direction_rejection(self):
        for direction in (None, True, "bidirectional", "none", "video_to_video"):
            with self.assertRaises(ContractError):
                UnidirectionalAttentionConfig(attention_direction=direction)
        with self.assertRaises(ContractError):
            CrossAttentionConfig(architecture="unidirectional_cross_attention")
        with self.assertRaises(ContractError):
            UnidirectionalAttentionConfig(architecture="bidirectional_cross_attention")
        with self.assertRaises(ContractError):
            BidirectionalCrossAttention(small("A1"))
        with self.assertRaises(ContractError):
            UnidirectionalCrossAttention(CrossAttentionConfig())
        fields = asdict(small("A1")); fields.pop("attention_direction")
        with self.assertRaises(ContractError):
            model_config_from_dict(fields, "gc2-fusion-cross-attention-checkpoint-v1")
        with self.assertRaises(ContractError):
            validate_live_model(build_model(small("A1")), small("A2"))

    def test_all_variant_checkpoint_roundtrips_and_state_swaps(self):
        for label in LABELS:
            model = build_model(small(label)).eval()
            meta = metadata(model.config)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                save_checkpoint(root, model, build_optimizer(model, meta.training.optimizer), meta)
                loaded, restored = load_selected_checkpoint(root)
                self.assertEqual(restored, meta)
                self.assertEqual(type(loaded), type(model))
                self.assertTrue(torch.equal(model(*self.features()).logits, loaded(*self.features()).logits))
        a1, a2 = build_model(small("A1")), build_model(small("A2"))
        with self.assertRaises(RuntimeError):
            a2.load_state_dict(a1.state_dict(), strict=True)

    def test_ranking_uses_auc_ap_f1_then_fixed_order(self):
        from fusion.cross_attention_ablations import ORDER
        rows = [{"label": label, "val_roc_auc": .9, "val_average_precision": .8, "val_f1": .7} for label in reversed(ORDER)]
        self.assertEqual([row["label"] for row in rank_attention(rows)], list(ORDER))
        rows[0]["val_roc_auc"] = .91
        self.assertEqual(rank_attention(rows)[0]["label"], "A3")
        with self.assertRaises(ContractError):
            rank_attention(rows[:-1])

    def test_configuration_inventory_and_tracker_destinations(self):
        from fusion.train_cross_attention import CrossAttentionExperimentConfig
        with tempfile.TemporaryDirectory() as directory, patch("fusion.train_cross_attention._output_path"):
            root = Path(directory)
            base = CrossAttentionExperimentConfig(*([root] * 8), root / "unused")
            variants = configurations(base, root / "study")
            self.assertEqual(tuple(variants), LABELS)
            for label, config in variants.items():
                self.assertEqual(config.model, variant_model(label))
                self.assertEqual(config.training, base.training)
                self.assertEqual(config.run_name, NAMES[label])
                tracker = ForgeAblationTracker(root, label)
                self.assertEqual(tracker.metadata["group"], GROUP)
                self.assertIn(label.lower(), tracker.artifact_prefix)
                self.assertFalse(config.evaluate_challenge)
            with self.assertRaises(ContractError):
                replace(variants["A1"], model=variant_model("A2"))


class StudyTracker(FakeTracker):
    labels = []
    fail_label = None
    tamper = False

    def __init__(self, output, label="initial"):
        super().__init__(output)
        self.label = label
        type(self).labels.append(label)

    def start(self, config):
        if self.label == type(self).fail_label:
            raise RuntimeError("synthetic failed experiment")
        result = super().start(config)
        if type(self).tamper and self.label == "A1":
            path = self.output.parent / "protocol.json"
            path.write_text(path.read_text() + " ")
        return result

    def finish(self, *args, **kwargs):
        result = super().finish(*args, **kwargs)
        self.metadata["artifacts"] = [{"name": "synthetic-"+self.label, "url": "synthetic-only"}]
        return dict(self.metadata)


class AblationStudyLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = training_fixture.AttentionTrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.training = TrainingConfig(batch_size=3, epochs=2, selection_metric="roc_auc")
        self.data = self.fixture.verified()
        self.data.summary.update(artifact_inventory_fingerprint=self.fixture.context,
            manifest_sha256={s: d.provenance.manifest_sha256 for s, d in self.fixture.data.items()})
        with patch("fusion.train_cross_attention._output_path"):
            self.base = CrossAttentionExperimentConfig(*([self.root] * 8), self.root / "initial",
                model=CrossAttentionConfig(d_model=8, num_heads=2, feedforward_dim=16, dropout=0., attention_dropout=0.),
                training=self.training, cpu_threads=4)
        with patch("fusion.train_cross_attention.repository", return_value={"commit": "b"*40}):
            run_initial(self.base, tracker_factory=StudyTracker, data_factory=lambda _: self.data, progress=lambda _: None)
        self.initial = study_module.primary_evidence("initial", self.base.output_dir,
            schema=study_module.INITIAL_SCHEMA, include_details=True)
        self.e3 = {**self.initial, "label": "E3"}
        self.data.baseline = {"e3": self.e3, "jsd": {key: self.initial[key] for key in (
            "val_roc_auc", "val_average_precision", "val_f1", "train_roc_auc", "train_average_precision", "train_f1")}}
        for split in ("train", "val"):
            final = json.loads((self.base.output_dir / "final_metrics.json").read_text())[split+"_primary"]
            path = self.root / f"reports/jensen_shannon_{split}_primary.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"classification": final}))
        path = self.root / "policies/jensen_shannon.json"; path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"policy": {"threshold": .5}}))
        StudyTracker.labels = []; StudyTracker.fail_label = None; StudyTracker.tamper = False

    def execute(self):
        original_model = study_module.variant_model
        original_evidence = study_module.primary_evidence
        def model(label):
            return replace(original_model(label), d_model=8, num_heads=2, feedforward_dim=16,
                           dropout=0., attention_dropout=0.)
        def evidence(label, output, **kwargs):
            return self.e3 if label == "E3" else original_evidence(label, output, **kwargs)
        with patch("fusion.cross_attention_ablations._output_path"), \
             patch("fusion.train_cross_attention._output_path"), \
             patch("fusion.cross_attention_ablations.variant_model", side_effect=model), \
             patch("fusion.cross_attention_ablations.initial_training", return_value=self.training), \
             patch("fusion.cross_attention_ablations.primary_evidence", side_effect=evidence), \
             patch("fusion.cross_attention_ablations.repository", return_value={"commit": "b"*40}):
            return study_module.run_study(self.base, self.base.output_dir, self.root / "study",
                data_factory=lambda _: self.data, tracker_factory=StudyTracker, progress=lambda _: None)

    def test_exactly_three_runs_with_frozen_plan_and_uploaded_full_comparison(self):
        result = self.execute()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(StudyTracker.labels, ["A1", "A2", "A3"])
        root = self.root / "study"
        plan = json.loads((root / "protocol.json").read_text())
        self.assertEqual([v["label"] for v in plan["experiments"]], ["A1", "A2", "A3"])
        self.assertFalse(plan["challenge_evaluation"])
        final = json.loads((root / "primary_comparison.json").read_text())
        uploaded = json.loads((root / "A3/baseline_comparison.json").read_text())["study_primary_comparison"]
        self.assertEqual(final, uploaded)
        self.assertEqual(len(final["models"]), 5)
        for label in ("A1", "A2", "A3"):
            self.assertTrue((root / label / "checkpoint_reload.json").is_file())
            self.assertFalse(any("challenge_scores" in p.name or p.name.startswith("test_") for p in (root / label).iterdir()))

    def test_failed_attempt_is_preserved_without_restart_or_later_runs(self):
        StudyTracker.fail_label = "A2"
        with self.assertRaises(ContractError):
            self.execute()
        self.assertEqual(StudyTracker.labels, ["A1", "A2"])
        root = self.root / "study"
        ledger = json.loads((root / "attempts.json").read_text())
        self.assertEqual([r["status"] for r in ledger], ["COMPLETE", "FAILED"])
        self.assertFalse((root / "completion.json").exists())
        self.assertFalse((root / "A3").exists())

    def test_changed_study_protocol_fails_before_any_ablation_optimizer_step(self):
        StudyTracker.tamper = True
        with patch("fusion.train_mlp.train_epoch", side_effect=AssertionError("must not optimize")), \
             self.assertRaises(ContractError):
            self.execute()
        self.assertEqual(StudyTracker.labels, ["A1"])
        root = self.root / "study"
        state = json.loads((root / "A1/state.json").read_text())
        self.assertFalse(state["optimization_started"])
        self.assertFalse((root / "completion.json").exists())
