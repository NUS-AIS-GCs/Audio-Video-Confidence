"""Offline synthetic numerical checks; no model downloads or online runs."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fusion.artifact_integration import run_integration
from fusion.baseline_tracking import TrackingUnavailable
from fusion.contracts import ContractError
from fusion.cross_attention_model import (
    BidirectionalCrossAttention, CrossAttentionBranch, CrossAttentionConfig,
)
from fusion.cross_attention_tracking import ForgeCrossAttentionTracker, GROUP
from fusion.embedding_data import EmbeddingDataset
from fusion.mlp_model import LateFusionMLP, MLPConfig
from fusion.mlp_tracking import EPOCH_KEYS, ForgeMLPTracker
from fusion.mlp_training import model_fingerprint, predict, train_epoch
from fusion.model_interface import build_model
from fusion.smoke_cross_attention import run_smoke
from fusion.train_mlp import load_selected_checkpoint, save_checkpoint, MLPExperimentConfig
from fusion.training_contracts import (
    CheckpointMetadata, TrainingConfig, backward_checked, build_optimizer,
    correspondence_loss, make_loader, move_batch, validate_split_isolation,
)
from test_fusion_artifact_integration import write_fixture
from test_fusion_mlp import manifest
from test_fusion_pair_generator import ROWS, bundles, record


def small_config(**kwargs):
    return CrossAttentionConfig(**{"d_model": 8, "num_heads": 2, "feedforward_dim": 16,
                                   "dropout": 0., "attention_dropout": 0., **kwargs})


def inputs(size=3, *, dtype=torch.float32, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(73)
    return (torch.randn(size, 10, 256, generator=generator, dtype=dtype, device=device),
            torch.randn(size, 10, 512, generator=generator, dtype=dtype, device=device),
            torch.ones(size, 10, dtype=torch.bool, device=device))


class CPUFriendlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)


class CrossAttentionArchitectureTests(CPUFriendlyTests):
    def test_default_projections_shapes_and_parameter_count(self):
        model = BidirectionalCrossAttention().eval()
        self.assertEqual((model.audio_projection.in_features, model.audio_projection.out_features), (256, 256))
        self.assertEqual((model.visual_projection.in_features, model.visual_projection.out_features), (512, 256))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1251841)
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()))
        for size in (1, 3, 7):
            out = model(*inputs(size), return_probabilities=True, return_bin_representations=True,
                        return_attention_weights=True)
            self.assertEqual(tuple(out.logits.shape), (size, 1))
            self.assertEqual(tuple(out.bin_representations.shape), (size, 10, 512))
            for weights in (out.audio_to_video_attention, out.video_to_audio_attention):
                self.assertEqual(tuple(weights.shape), (size, 4, 10, 10))
                self.assertTrue(torch.isfinite(weights).all())
            torch.testing.assert_close(out.probabilities, out.logits.sigmoid())
        out = model(*inputs(1))
        self.assertIsNone(out.probabilities)
        self.assertIsNone(out.bin_representations)
        self.assertIsNone(out.audio_to_video_attention)
        self.assertIsNone(out.video_to_audio_attention)

    def test_both_qkv_directions_use_original_projected_modalities(self):
        model = BidirectionalCrossAttention(small_config()).eval()
        audio, visual, mask = inputs()
        mask[:, [1, 7]] = False
        invalid = ~mask.unsqueeze(-1)
        expected_a = (model.audio_projection(audio.masked_fill(invalid, 0))
                      + model.temporal_positions).masked_fill(invalid, 0)
        expected_v = (model.visual_projection(visual.masked_fill(invalid, 0))
                      + model.temporal_positions).masked_fill(invalid, 0)
        seen = []
        def inspect(_module, args, kwargs):
            seen.append((args, kwargs))
        handles = [b.attention.register_forward_pre_hook(inspect, with_kwargs=True)
                   for b in (model.audio_to_video, model.video_to_audio)]
        try:
            model(audio, visual, mask)
        finally:
            for handle in handles:
                handle.remove()
        for (args, kwargs), q, kv in zip(seen, (expected_a, expected_v), (expected_v, expected_a)):
            for actual, expected in zip(args, (q, kv, kv)):
                torch.testing.assert_close(actual, expected)
            self.assertIs(args[1], args[2])
            self.assertTrue(torch.equal(kwargs["key_padding_mask"], ~mask))
            self.assertFalse(kwargs["average_attn_weights"])

    def test_valid_configurations_and_optional_feedforward(self):
        for dim, heads in ((4, 1), (8, 2), (12, 3), (16, 4), (7, 1)):
            for ff in (0, 11):
                model = BidirectionalCrossAttention(small_config(d_model=dim, num_heads=heads,
                                                                 feedforward_dim=ff, activation="relu"))
                self.assertEqual(model(*inputs(1)).logits.shape, (1, 1))
                self.assertEqual(model.audio_to_video.feedforward is None, ff == 0)

    def test_invalid_configs_rejected(self):
        for kwargs in ({"d_model": 0}, {"d_model": True}, {"d_model": 3.}, {"num_heads": 0},
                       {"num_heads": True}, {"d_model": 7, "num_heads": 2}, {"feedforward_dim": -1},
                       {"feedforward_dim": True}, {"dropout": True}, {"dropout": 1.},
                       {"attention_dropout": float("nan")}, {"attention_dropout": float("inf")},
                       {"attention_dropout": -.1}, {"position_encoding": "automatic"},
                       {"activation": "bad"}, {"initialization_seed": True},
                       {"initialization_seed": -1}, {"initialization_seed": 2**32},
                       {"audio_dim": 128}, {"visual_dim": 256}, {"temporal_bins": True},
                       {"temporal_bins": 9}, {"aggregation": "event_mean"}, {"architecture": "mlp"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                CrossAttentionConfig(**kwargs)
        with self.assertRaises(ContractError):
            BidirectionalCrossAttention(MLPConfig())
        with self.assertRaises(ContractError):
            build_model({})

    def test_initialization_preserves_rng_and_is_seeded(self):
        before = torch.get_rng_state().clone()
        first = BidirectionalCrossAttention(small_config())
        second = BidirectionalCrossAttention(small_config())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(model_fingerprint(first), model_fingerprint(second))
        self.assertNotEqual(model_fingerprint(first), model_fingerprint(
            BidirectionalCrossAttention(small_config(initialization_seed=43))))

    def test_train_eval_dropout_and_diagnostic_semantics(self):
        model = BidirectionalCrossAttention(small_config(dropout=.5, attention_dropout=.5))
        batch = inputs()
        first = model(*batch, return_attention_weights=True)
        second = model(*batch)
        self.assertFalse(torch.equal(first.logits, second.logits))
        self.assertFalse(torch.allclose(first.audio_to_video_attention.sum(-1), torch.ones(3, 2, 10)))
        model.eval()
        first = model(*batch, return_attention_weights=True)
        self.assertTrue(torch.equal(first.logits, model(*batch).logits))
        torch.testing.assert_close(first.audio_to_video_attention.sum(-1), torch.ones(3, 2, 10))

    def test_gradients_through_modalities_and_all_parameters(self):
        model = BidirectionalCrossAttention(small_config())
        audio, visual, mask = inputs()
        audio.requires_grad_(); visual.requires_grad_()
        out = model(audio, visual, mask)
        loss = correspondence_loss(out.logits, torch.tensor([[1.], [0.], [1.]]), TrainingConfig())
        self.assertTrue(math.isfinite(backward_checked(loss, model, max_gradient_norm=5)))
        for tensor in (audio.grad, visual.grad):
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertGreater(tensor.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_explicit_mask_shape_dtype_and_output_switch_guards(self):
        model = BidirectionalCrossAttention(small_config())
        a, v, mask = inputs()
        for audio, visual, validity in ((a[:, :9], v, mask), (a[:, :, :128], v, mask),
                                       (a, v[:1], mask), (a[:0], v[:0], mask[:0]),
                                       (a.double(), v, mask), (a.long(), v, mask),
                                       (a, v, None), (a, v, mask.float()), (a, v, mask[:, :9])):
            with self.subTest(shape=getattr(audio, "shape", None)), self.assertRaises(ContractError):
                model(audio, visual, validity)
        for switch in ("return_probabilities", "return_bin_representations", "return_attention_weights"):
            with self.assertRaises(ContractError):
                model(a, v, mask, **{switch: 1})

    def test_float64_and_noncontiguous_inputs(self):
        model = BidirectionalCrossAttention(small_config()).double().eval()
        a, v, mask = inputs(dtype=torch.float64)
        a = a.transpose(0, 1).contiguous().transpose(0, 1)
        self.assertFalse(a.is_contiguous())
        out = model(a, v, mask)
        self.assertEqual(out.logits.dtype, torch.float64)
        self.assertEqual(model.temporal_positions.dtype, torch.float64)
        self.assertTrue(torch.isfinite(out.logits).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_backward_and_device_guard(self):
        model = BidirectionalCrossAttention(small_config()).cuda()
        a, v, mask = inputs(device="cuda")
        backward_checked(model(a, v, mask).logits.sum(), model, max_gradient_norm=5)
        with self.assertRaises(ContractError):
            model(a, v, mask.cpu())


class AttentionNumericalTests(CPUFriendlyTests):
    def test_independent_scalar_attention_softmax_value_sum_and_residual_norm(self):
        branch = CrossAttentionBranch(small_config(d_model=2, num_heads=1, feedforward_dim=0)).double().eval()
        with torch.no_grad():
            branch.attention.in_proj_weight.copy_(torch.cat([torch.eye(2)] * 3))
            branch.attention.in_proj_bias.zero_()
            branch.attention.out_proj.weight.copy_(torch.eye(2))
            branch.attention.out_proj.bias.zero_()
        q = torch.zeros(1, 10, 2, dtype=torch.float64)
        kv = torch.zeros_like(q)
        q[0, 0] = torch.tensor([1., 0.]); q[0, 3] = torch.tensor([0., 2.])
        kv[0, 0] = torch.tensor([2., 0.]); kv[0, 3] = torch.tensor([0., 1.])
        mask = torch.zeros(1, 10, dtype=torch.bool); mask[:, [0, 3]] = True
        actual, weights = branch(q, kv, mask)
        for i in (0, 3):
            # Independent Python scalar dot/exp/sum, without torch attention or softmax.
            scores = [sum(q[0, i, k].item() * kv[0, j, k].item() for k in range(2)) / math.sqrt(2)
                      for j in (0, 3)]
            exp = [math.exp(s - max(scores)) for s in scores]
            expected_w = [e / sum(exp) for e in exp]
            expected_u = [q[0, i, k].item() + sum(w * kv[0, j, k].item()
                          for w, j in zip(expected_w, (0, 3))) for k in range(2)]
            mean = sum(expected_u) / 2
            variance = sum((x - mean)**2 for x in expected_u) / 2
            expected = [(x - mean) / math.sqrt(variance + branch.attention_norm.eps) for x in expected_u]
            torch.testing.assert_close(weights[0, 0, i, [0, 3]], torch.tensor(expected_w, dtype=torch.float64),
                                       rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(actual[0, i], torch.tensor(expected, dtype=torch.float64),
                                       rtol=1e-12, atol=1e-12)

    def test_mixed_masks_key_query_normalization_and_masked_mean(self):
        model = BidirectionalCrossAttention(small_config()).eval()
        a, v, mask = inputs()
        mask[0, 1:] = False
        mask[1, [0, 2, 5, 8]] = False
        out = model(a, v, mask, return_attention_weights=True, return_bin_representations=True)
        for w in (out.audio_to_video_attention, out.video_to_audio_attention):
            self.assertTrue(torch.isfinite(w).all())
            self.assertEqual(w.masked_select(~mask[:, None, :, None].expand_as(w)).abs().sum(), 0)
            self.assertEqual(w.masked_select(~mask[:, None, None, :].expand_as(w)).abs().sum(), 0)
            torch.testing.assert_close(w.sum(-1), mask[:, None, :].float().expand(3, 2, 10))
            torch.testing.assert_close(w[0, :, 0, 0], torch.ones(2))
        self.assertEqual(out.bin_representations[~mask].abs().sum(), 0)
        expected = torch.stack([out.bin_representations[i, mask[i]].mean(0) for i in range(3)])
        torch.testing.assert_close(out.logits, model.correspondence_head(expected))

    def test_invalid_bin_invariance_and_zero_input_gradients_in_eval_and_train(self):
        model = BidirectionalCrossAttention(small_config(dropout=.2, attention_dropout=.2))
        a, v, mask = inputs()
        mask[:, [1, 4, 9]] = False
        changed_a, changed_v = a.clone(), v.clone()
        changed_a[~mask] = torch.finfo(a.dtype).max
        changed_v[~mask] = -torch.finfo(v.dtype).max
        for train in (False, True):
            model.train(train)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(19)
                first = model(a, v, mask, return_attention_weights=True)
                torch.manual_seed(19)
                second = model(changed_a, changed_v, mask, return_attention_weights=True)
            self.assertTrue(torch.equal(first.logits, second.logits))
            self.assertTrue(torch.equal(first.audio_to_video_attention, second.audio_to_video_attention))
            self.assertTrue(torch.equal(first.video_to_audio_attention, second.video_to_audio_attention))
        a.requires_grad_(); v.requires_grad_()
        model.eval()(a, v, mask).logits.sum().backward()
        for grad in (a.grad, v.grad):
            self.assertEqual(grad[~mask].abs().sum(), 0)
            self.assertGreater(grad[mask].abs().sum(), 0)

    def test_nonfinite_rejected_even_in_invalid_bins_and_intermediates(self):
        a, v, mask = inputs()
        mask[:, 9] = False
        for side in (0, 1):
            for bad in (float("nan"), float("inf"), -float("inf")):
                batch = [a.clone(), v.clone(), mask]
                batch[side][0, 9, 0] = bad
                with self.assertRaises(ContractError):
                    BidirectionalCrossAttention(small_config())(*batch)
        model = BidirectionalCrossAttention(small_config())
        with torch.no_grad():
            model.audio_projection.weight.fill_(torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(ContractError, "nonfinite"):
            model(torch.full_like(a, 2), v, mask)
        model = BidirectionalCrossAttention(small_config())
        with torch.no_grad():
            model.audio_to_video.attention.in_proj_weight.fill_(float("inf"))
        with self.assertRaisesRegex(ContractError, "nonfinite"):
            model(a, v, mask)

    def test_one_all_masked_example_rejects_whole_batch(self):
        a, v, mask = inputs()
        mask[1] = False
        with self.assertRaisesRegex(ContractError, "all-masked"):
            BidirectionalCrossAttention(small_config())(a, v, mask)

    def test_sinusoidal_formula_original_indices_and_both_streams(self):
        model = BidirectionalCrossAttention(small_config(d_model=7, num_heads=1)).eval()
        expected = np.zeros((10, 7))
        for t in range(10):
            for k in range(7):
                angle = t / 10000**(2 * (k // 2) / 7)
                expected[t, k] = math.sin(angle) if k % 2 == 0 else math.cos(angle)
        np.testing.assert_allclose(model.temporal_positions[0].numpy(), expected, atol=3e-7, rtol=1e-6)
        with torch.no_grad():
            for layer in (model.audio_projection, model.visual_projection):
                layer.weight.zero_(); layer.bias.zero_()
        a, v, mask = inputs(1); mask[:, [1, 2, 5]] = False
        seen = []
        handle = model.audio_to_video.attention.register_forward_pre_hook(lambda _, args: seen.append(args))
        try:
            model(a, v, mask)
        finally:
            handle.remove()
        for stream in seen[0]:
            torch.testing.assert_close(stream[0, mask[0]], model.temporal_positions[0, mask[0]])
            self.assertEqual(stream[0, ~mask[0]].abs().sum(), 0)
        self.assertFalse(torch.equal(model.temporal_positions[:, 3], model.temporal_positions[:, 1]))

    def test_position_ablation_and_joint_permutation_behavior(self):
        a, v, mask = inputs()
        permutation = torch.tensor([9, 6, 1, 8, 4, 2, 5, 0, 3, 7])
        no_position = BidirectionalCrossAttention(small_config(position_encoding="none")).eval()
        self.assertEqual(no_position.temporal_positions.abs().sum(), 0)
        torch.testing.assert_close(no_position(a, v, mask).logits,
                                   no_position(a[:, permutation], v[:, permutation], mask[:, permutation]).logits)
        positioned = BidirectionalCrossAttention(small_config()).eval()
        self.assertGreater((positioned(a, v, mask).logits - positioned(
            a[:, permutation], v[:, permutation], mask[:, permutation]).logits).abs().max(), 1e-5)

    def test_no_leakage_across_batch_examples_or_input_mutation(self):
        model = BidirectionalCrossAttention(small_config()).eval()
        a, v, mask = inputs()
        before = (a.clone(), v.clone(), mask.clone())
        together = model(a, v, mask, return_attention_weights=True)
        for i in range(3):
            alone = model(a[i:i+1], v[i:i+1], mask[i:i+1], return_attention_weights=True)
            torch.testing.assert_close(together.logits[i:i+1], alone.logits)
            torch.testing.assert_close(together.audio_to_video_attention[i:i+1], alone.audio_to_video_attention)
        changed_a = a.clone(); changed_a[1:] *= 17
        changed_v = v.clone(); changed_v[1:] *= -11
        torch.testing.assert_close(together.logits[:1], model(changed_a, changed_v, mask).logits[:1])
        for actual, expected in zip((a, v, mask), before):
            self.assertTrue(torch.equal(actual, expected))


class CrossAttentionTrainingTests(CPUFriendlyTests):
    def setUp(self):
        self.audio, self.vision = bundles({"train": ROWS, "val": (record("v1", 4), record("v2", 4),
                                                                 record("v3", 5), record("v4", 5)),
                                           "test": (record("t1", 6), record("t2", 6),
                                                    record("t3", 7), record("t4", 7))})
        self.train = EmbeddingDataset(manifest(self.audio, self.vision), self.audio, self.vision, split="train")
        val = manifest(self.audio, self.vision, "val")
        self.val = EmbeddingDataset(val, self.audio, self.vision, split="val")
        self.challenge = EmbeddingDataset(val, self.audio, self.vision, split="val", partition="challenge")
        self.config = TrainingConfig(batch_size=3, epochs=2, selection_metric="roc_auc")
        self.model = BidirectionalCrossAttention(small_config())

    def metadata(self, model=None, **kwargs):
        model = self.model if model is None else model
        return CheckpointMetadata(model.config, self.config, "a"*40, "b"*64,
                                  (self.train.provenance, self.val.provenance),
                                  (("audio", "synthetic-a"), ("vision", "synthetic-v")),
                                  (("python", "3"), ("numpy", "2"), ("torch", str(torch.__version__))),
                                  validate_split_isolation(self.train, self.val),
                                  schema=("gc2-fusion-checkpoint-v1" if type(model) is LateFusionMLP
                                          else "gc2-fusion-cross-attention-checkpoint-v1"), **kwargs)

    def test_one_optimizer_step_bce_and_feature_metadata_boundary(self):
        features, targets = move_batch(next(iter(make_loader(self.train, self.config, role="train"))), "cpu")
        self.assertEqual(set(features), {"audio", "visual", "validity"})
        optimizer = build_optimizer(self.model, self.config.optimizer)
        before = model_fingerprint(self.model)
        logits = self.model(**features).logits
        loss = correspondence_loss(logits, targets, self.config)
        torch.testing.assert_close(loss, nn.BCEWithLogitsLoss()(logits, targets))
        backward_checked(loss, self.model, max_gradient_norm=5)
        optimizer.step()
        self.assertNotEqual(before, model_fingerprint(self.model))
        for key in ("targets", "pair_id", "split", "category", "challenge"):
            with self.assertRaises(TypeError):
                self.model(**features, **{key: targets})
            with self.assertRaises(ContractError):
                move_batch(({**features, key: targets}, targets), "cpu")

    def test_shared_epoch_distinct_coverage_validation_and_split_guards(self):
        optimizer = build_optimizer(self.model, self.config.optimizer)
        seen = []
        handle = self.model.register_forward_pre_hook(lambda _, args, kw: seen.append(set(kw)), with_kwargs=True)
        try:
            row = train_epoch(self.model, make_loader(self.train, self.config, role="train"), optimizer, self.config)
        finally:
            handle.remove()
        self.assertEqual(row["count"], 16)
        self.assertEqual(row["optimizer_steps"], 6)  # includes final singleton
        self.assertTrue(all(keys == {"audio", "visual", "validity"} for keys in seen))
        before = model_fingerprint(self.model)
        result = predict(self.model, self.val, self.config)
        self.assertEqual(result.split, "val")
        self.assertEqual(before, model_fingerprint(self.model))
        self.assertFalse(self.model.training)
        for dataset, role in ((self.val, "train"), (self.challenge, "train"), (self.train, "test")):
            with self.assertRaises(ContractError):
                make_loader(dataset, self.config, role=role)
        final = EmbeddingDataset(manifest(self.audio, self.vision, "test"), self.audio, self.vision, split="test")
        with self.assertRaises(ContractError):
            make_loader(final, self.config, role="train")
        with self.assertRaises(ContractError):
            predict(self.model, final, self.config)
        with self.assertRaises(ContractError):
            predict(self.model, self.challenge, self.config)
        with self.assertRaises(ContractError):
            train_epoch(self.model, DataLoader(self.train, batch_sampler=[[0, 0]]), optimizer, self.config)

    def test_checkpoint_roundtrip_safe_reload_and_architecture_binding(self):
        metadata = self.metadata(epoch=1, selected_epoch=1, validation_metrics=(("roc_auc", .5),))
        decoded = CheckpointMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))
        self.assertEqual(decoded, metadata)
        model = build_model(decoded.model).eval()
        self.assertIs(type(model), BidirectionalCrossAttention)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            optimizer = build_optimizer(model, self.config.optimizer)
            save_checkpoint(output, model, optimizer, metadata)
            loaded, restored = load_selected_checkpoint(output, expected_metadata=metadata)
            self.assertEqual(restored, metadata)
            self.assertEqual(model_fingerprint(model), model_fingerprint(loaded))
            self.assertTrue(torch.equal(model(*inputs(1)).logits, loaded(*inputs(1)).logits))
        for changed in ({"schema": "gc2-fusion-checkpoint-v1"}, {"schema": "unknown"},
                        {"model": {**asdict(metadata.model), "num_heads": 3}},
                        {"model": {**asdict(metadata.model), "extra": 1}}):
            with self.assertRaises(ContractError):
                CheckpointMetadata.from_dict({**metadata.to_dict(), **changed})
        with self.assertRaises(ContractError):
            replace(metadata, data=(self.train.provenance, replace(self.val.provenance, split="test")))

    def test_historical_mlp_schema_factory_and_outputs_unchanged(self):
        direct = LateFusionMLP(MLPConfig(hidden_dims=(8,), dropout=0)).eval()
        built = build_model(direct.config).eval()
        self.assertEqual(model_fingerprint(direct), model_fingerprint(built))
        a, v, mask = inputs()
        self.assertTrue(torch.equal(direct(a, v, mask).logits, built(a, v, mask).logits))
        metadata = self.metadata(direct)
        self.assertEqual(metadata.schema, "gc2-fusion-checkpoint-v1")
        self.assertNotIn("architecture", metadata.to_dict()["model"])
        self.assertEqual(CheckpointMetadata.from_dict(json.loads(json.dumps(metadata.to_dict()))), metadata)
        with patch("fusion.train_mlp._output_path"), self.assertRaises(ContractError):
            MLPExperimentConfig(*([Path("synthetic")] * 8), model=self.model.config)

    def test_phase5_tracker_inherits_remote_verification_before_finalize(self):
        # No SDK or network: verify the inherited lifecycle itself, not a copied implementation.
        self.assertIs(ForgeCrossAttentionTracker.finish, ForgeMLPTracker.finish)
        self.assertIs(ForgeCrossAttentionTracker._verify_remote, ForgeMLPTracker._verify_remote)
        self.assertIs(ForgeCrossAttentionTracker.abort, ForgeMLPTracker.abort)
        tracker = ForgeCrossAttentionTracker(Path("synthetic"))
        self.assertEqual(tracker.metadata["base_url"], "https://api.forge.coreweave.com")
        self.assertEqual(tracker.metadata["entity"], "GC2-AValidate")
        self.assertEqual(tracker.metadata["project"], "avalidate")
        self.assertEqual(tracker.metadata["group"], GROUP)
        self.assertEqual(tracker.allowed_groups, (GROUP,))
        self.assertEqual(tracker.artifact_prefix, "phase5-cross-attention")
        with patch.object(ForgeMLPTracker, "start", return_value={"synthetic": True}) as start:
            self.assertEqual(tracker.start({"model": asdict(self.model.config)}), {"synthetic": True})
            start.assert_called_once()
        with self.assertRaises(TrackingUnavailable):
            tracker.start({"model": asdict(MLPConfig())})
        with patch.object(tracker, "preflight_access", side_effect=AssertionError("no network")), \
             self.assertRaises(TrackingUnavailable):
            tracker.start({"model": asdict(self.model.config), "training": {"seed": 42}, "wandb_group": "other"})

    def test_independent_producer_masks_keep_original_rows_and_adapter_intersection(self):
        n = len(self.audio.splits["train"].records)
        audio_valid, visual_valid = np.ones((n, 10), dtype=bool), np.ones((n, 10), dtype=bool)
        for i in range(n):
            audio_valid[i, i % 10] = False
            visual_valid[i, (i + 3) % 10] = False
        audio = replace(self.audio, splits={**self.audio.splits,
            "train": replace(self.audio.splits["train"], input_validity=audio_valid)})
        vision = replace(self.vision, splits={**self.vision.splits,
            "train": replace(self.vision.splits["train"], input_validity=visual_valid)})
        data = EmbeddingDataset(manifest(audio, vision), audio, vision, split="train")
        for i, pair in enumerate(data.pairs):
            features, target = data[i]
            ai, vi = pair.audio.row_index, pair.visual.row_index
            self.assertEqual(pair.audio.record, audio.splits["train"].records[ai])
            self.assertEqual(pair.visual.record, vision.splits["train"].records[vi])
            np.testing.assert_array_equal(features["validity"], audio_valid[ai] & visual_valid[vi])
            self.assertEqual(target.item(), pair.original_pair_target)
        features, _ = next(iter(make_loader(data, self.config, role="train")))
        out = self.model.eval()(**features, return_attention_weights=True)
        torch.testing.assert_close(out.audio_to_video_attention.sum(-1),
                                   features["validity"][:, None, :].float().expand(3, 2, 10))

    def test_mock_phase5_tracking_names_and_verified_finalization_order(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = ForgeCrossAttentionTracker(directory)
            tracker.sdk, tracker.api = MagicMock(), MagicMock()
            run = tracker.sdk.init.return_value
            run.settings.mode = "online"; run.settings.base_url = "https://api.forge.coreweave.com"
            run.entity = "GC2-AValidate"; run.project = "avalidate"; run.id = "mock"
            run.url = "https://forge.coreweave.com/GC2-AValidate/avalidate/runs/mock"
            uploaded = run.log_artifact.return_value
            uploaded.qualified_name = "GC2-AValidate/avalidate/mock:v0"; uploaded.digest = "mockdigest"
            tracker.api.artifact.return_value.digest = "mockdigest"
            with patch.object(tracker, "preflight_access"), \
                 patch("fusion.mlp_tracking.importlib.metadata.version", return_value="synthetic"):
                tracker.start({"model": asdict(self.model.config), "training": {"seed": 42}})
            self.assertEqual(tracker.sdk.init.call_args.kwargs["group"], GROUP)
            self.assertEqual(tracker.sdk.init.call_args.kwargs["name"], "phase5-cross-attention-seed42-initial")
            self.assertEqual(tracker.sdk.Artifact.call_args.args[0], "phase5-cross-attention-online-readiness")
            self.assertEqual(tracker.metadata["status"], "ONLINE_WRITE_VERIFIED")
            row = {key: .5 for key in EPOCH_KEYS}; row["epoch"] = 1
            metrics = {"final/val_roc_auc": .75}
            remote = tracker.api.run.return_value
            remote.summary = {**metrics, "reproducibility_manifest_sha256": "a"*64}
            events = []
            remote.scan_history.side_effect = lambda **kw: events.append("remote") or [
                {"epoch": 1, **{tag: row[key] for key, tag in EPOCH_KEYS.items()}}]
            run.finish.side_effect = lambda **kw: events.append("finish")
            files = []
            for name in ("best.pt", "checkpoint_metadata.json", "frozen_operating_point.json", "history.json"):
                path = Path(directory) / name
                path.write_bytes(b"synthetic")
                files.append(path)
            tracker.finish(metrics, files, [row], "a"*64,
                           before_finalize=lambda: events.append("local_integrity"))
            self.assertEqual(events, ["remote", "local_integrity", "finish"])
            self.assertEqual([call.args[0] for call in tracker.sdk.Artifact.call_args_list], [
                "phase5-cross-attention-online-readiness", "phase5-cross-attention-selected-checkpoint",
                "phase5-cross-attention-reproducibility-reports"])
            run.finish.assert_called_once_with(exit_code=0)


class CrossAttentionSmokeTests(CPUFriendlyTests):
    def test_shared_source_integrity_train_only_rows_diagnostics_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, vision, official, counts = write_fixture(root)
            integration = root / "integration"
            report = run_integration(audio, vision, official, expected_counts=counts, output_dir=integration)
            self.assertEqual(report["status"], "COMPLETE")
            splits = []
            original = EmbeddingDataset.from_json.__func__
            def inspect(cls, *args, **kwargs):
                splits.append(kwargs["split"])
                return original(cls, *args, **kwargs)
            with patch.object(EmbeddingDataset, "from_json", classmethod(inspect)), \
                 patch.object(ForgeMLPTracker, "start", side_effect=AssertionError("no W&B")):
                result = run_smoke(audio, vision, official, integration, expected_counts=counts,
                                   config=small_config())
            self.assertEqual(splits, ["train"])
            self.assertEqual(result["split"], "train")
            self.assertEqual(result["row_reference_mismatches"], 0)
            self.assertEqual(result["audio_to_video_attention_shape"], [8, 2, 10, 10])
            self.assertTrue(result["finite_attention_and_correct_masks"])
            self.assertTrue(result["producer_inputs_and_manifests_unchanged"])
            for key in ("trained", "metrics_computed", "wandb_run_created"):
                self.assertFalse(result[key])
            path = integration / "train_pairs.json"
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "manifest hash"):
                run_smoke(audio, vision, official, integration, expected_counts=counts, config=small_config())


if __name__ == "__main__":
    unittest.main()
