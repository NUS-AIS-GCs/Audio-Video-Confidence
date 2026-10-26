"""Offline checkpoint/configuration regressions; synthetic provenance only."""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from decimal import Decimal
from unittest.mock import patch

import torch
from torch import nn

from fusion.artifact_integration import _file_entry
from fusion.contracts import ContractError
from fusion.cross_attention_model import BidirectionalCrossAttention, CrossAttentionConfig
from fusion.embedding_data import EmbeddingProvenance
from fusion.mlp_model import LateFusionMLP, MLPConfig
from fusion.model_interface import build_model
from fusion.train_mlp import load_selected_checkpoint, save_checkpoint
from fusion.training_contracts import (
    CheckpointMetadata, TrainingConfig, backward_checked, build_optimizer, correspondence_loss,
)


def attention_config(**changes):
    return CrossAttentionConfig(**{"d_model": 8, "num_heads": 2, "feedforward_dim": 16,
                                   "dropout": 0., "attention_dropout": 0., **changes})


def provenance(split):
    return EmbeddingProvenance(split, "a"*64, "b"*64, None,
        tuple((f"{modality}/{split}/{kind}", "c"*64)
              for modality in ("audio", "vision") for kind in ("embeddings", "validity")), ())


def metadata(config):
    return CheckpointMetadata(
        config, TrainingConfig(selection_metric="roc_auc"), "d"*40, "e"*64,
        (provenance("train"), provenance("val")),
        (("audio", "synthetic"), ("vision", "synthetic")),
        (("python", "3"), ("numpy", "2"), ("torch", str(torch.__version__))), (),
        epoch=1, selected_epoch=1, validation_metrics=(("roc_auc", .5),),
        schema=("gc2-fusion-checkpoint-v1" if type(config) is MLPConfig
                else "gc2-fusion-cross-attention-checkpoint-v1"))


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


class CheckpointBindingTests(unittest.TestCase):
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
        self.model = BidirectionalCrossAttention(attention_config()).eval()

    def save(self, model=None, meta=None):
        model = self.model if model is None else model
        meta = metadata(model.config) if meta is None else meta
        save_checkpoint(self.root, model, build_optimizer(model, meta.training.optimizer), meta)
        return meta

    def rewrite(self, change, *, update_model_hash=False):
        path = self.root / "best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        change(payload)
        if update_model_hash:
            payload["model_sha256"] = state_fingerprint(payload["model_state"])
        torch.save(payload, path)
        (self.root / "checkpoint_integrity.json").write_text(
            json.dumps(_file_entry("best.pt", path)), encoding="utf-8")

    def assert_save_rejected(self, model, config, pattern):
        with self.assertRaisesRegex(ContractError, pattern):
            self.save(model, metadata(config))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_original_sinusoidal_none_metadata_mismatch_rejected_before_writes(self):
        self.assert_save_rejected(self.model, replace(self.model.config, position_encoding="none"),
                                  "architecture/configuration")

    def test_equivalent_integer_and_float_dropout_checkpoint_binding(self):
        for integer_live in (True, False):
            value, declaration = (0, 0.) if integer_live else (0., 0)
            model = BidirectionalCrossAttention(attention_config(dropout=value, attention_dropout=value)).eval()
            config = replace(model.config, dropout=declaration, attention_dropout=declaration)
            meta = self.save(model, metadata(config))
            loaded, restored = load_selected_checkpoint(self.root, expected_metadata=meta)
            self.assertEqual(restored, meta)
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]))

    def test_numeric_dropout_comparison_rejects_bad_types_and_exact_mismatches(self):
        for invalid in (False, True, float("nan"), float("inf"), Decimal("0"), .000000000000001):
            for branch, field in (("attention", "dropout"), ("residual_dropout", "p")):
                with self.subTest(invalid=invalid, branch=branch):
                    model = BidirectionalCrossAttention(attention_config())
                    setattr(getattr(model.audio_to_video, branch), field, invalid)
                    self.assert_save_rejected(model, model.config, "live model setting")

    def test_wrong_heads_dimension_feedforward_dropout_activation_and_architecture(self):
        for change in ({"num_heads": 4}, {"d_model": 16}, {"feedforward_dim": 0},
                       {"feedforward_dim": 32}, {"dropout": .2}, {"attention_dropout": .2},
                       {"activation": "relu"}):
            with self.subTest(change=change):
                self.assert_save_rejected(self.model, replace(self.model.config, **change),
                                          "architecture/configuration")
        self.assert_save_rejected(self.model, MLPConfig(hidden_dims=(8,)), "architecture/configuration")
        mlp = LateFusionMLP(MLPConfig(hidden_dims=(8,)))
        self.assert_save_rejected(mlp, self.model.config, "architecture/configuration")

    def test_live_head_mutation_cannot_hide_behind_matching_or_forged_config(self):
        self.model.audio_to_video.attention.num_heads = 4
        self.assert_save_rejected(self.model, self.model.config, "live model setting")
        self.model.config = replace(self.model.config, num_heads=4)
        self.assert_save_rejected(self.model, self.model.config, "live model setting")  # other branch still 2

    def test_live_module_mutations_are_rejected(self):
        changes = (
            lambda model: setattr(model.visual_projection, "out_features", 99),
            lambda model: setattr(model.video_to_audio.residual_dropout, "p", .7),
            lambda model: setattr(model.audio_to_video.attention, "dropout", .3),
            lambda model: setattr(model.audio_to_video.attention_norm, "eps", .2),
            lambda model: setattr(model.audio_to_video.feedforward, "0", nn.Linear(8, 23)),
            lambda model: setattr(model.audio_to_video.feedforward, "1", nn.ReLU()),
            lambda model: setattr(model.audio_to_video.feedforward, "_modules",
                                  dict(reversed(list(model.audio_to_video.feedforward._modules.items())))),
        )
        for change in changes:
            with self.subTest(change=change):
                model = BidirectionalCrossAttention(attention_config())
                change(model)
                self.assert_save_rejected(model, model.config, "live model|checkpoint state")

    def test_live_buffer_and_nonfinite_weights_rejected_before_writes(self):
        for position in ("sinusoidal", "none"):
            model = BidirectionalCrossAttention(attention_config(position_encoding=position))
            model.temporal_positions[0, 0, 0] += 1
            self.assert_save_rejected(model, model.config, "buffer disagrees")
        model = BidirectionalCrossAttention(attention_config())
        positions = model._buffers.pop("temporal_positions")
        model.register_parameter("temporal_positions", nn.Parameter(positions))
        self.assert_save_rejected(model, model.config, "parameter/buffer coverage")
        model = BidirectionalCrossAttention(attention_config())
        model.register_buffer("unexpected_nonpersistent", torch.zeros(1), persistent=False)
        self.assert_save_rejected(model, model.config, "parameter/buffer coverage")
        with torch.no_grad():
            self.model.correspondence_head.weight.fill_(float("nan"))
        self.assert_save_rejected(self.model, self.model.config, "nonfinite")

    def test_valid_sinusoidal_and_none_roundtrip_with_safe_loading_and_dtypes(self):
        for position in ("sinusoidal", "none"):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(position=position, dtype=dtype):
                    model = BidirectionalCrossAttention(attention_config(position_encoding=position)).to(dtype).eval()
                    meta = self.save(model)
                    with patch("fusion.train_mlp.torch.load", wraps=torch.load) as safe_load:
                        loaded, restored = load_selected_checkpoint(self.root, expected_metadata=meta)
                    self.assertTrue(safe_load.call_args.kwargs["weights_only"])
                    self.assertEqual(restored, meta)
                    for name, tensor in model.state_dict().items():
                        self.assertEqual(loaded.state_dict()[name].dtype, dtype)
                        self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]))
                    generator = torch.Generator().manual_seed(213)
                    a = torch.randn(3, 10, 256, generator=generator, dtype=dtype)
                    v = torch.randn(3, 10, 512, generator=generator, dtype=dtype)
                    mask = torch.ones(3, 10, dtype=torch.bool)
                    self.assertTrue(torch.equal(model(a, v, mask).logits, loaded(a, v, mask).logits))

    def test_load_rejects_missing_positional_buffer_even_with_rehashed_file(self):
        self.save()
        self.rewrite(lambda payload: payload["model_state"].pop("temporal_positions"), update_model_hash=True)
        with self.assertRaisesRegex(ContractError, "state dictionary coverage"):
            load_selected_checkpoint(self.root)

    def test_load_rejects_altered_buffer_values_even_with_matching_fingerprints(self):
        self.save()
        def change(payload):
            payload["model_state"]["temporal_positions"][0, 0, 0] += .25
        self.rewrite(change, update_model_hash=True)
        with self.assertRaisesRegex(ContractError, "buffer disagrees"):
            load_selected_checkpoint(self.root)

    def test_none_checkpoint_rejects_unexpected_sinusoidal_buffer(self):
        model = BidirectionalCrossAttention(attention_config(position_encoding="none"))
        self.save(model)
        def change(payload):
            payload["model_state"]["temporal_positions"] = self.model.temporal_positions.clone()
        self.rewrite(change, update_model_hash=True)
        with self.assertRaisesRegex(ContractError, "buffer disagrees"):
            load_selected_checkpoint(self.root)

    def test_wrong_buffer_shape_nonfinite_value_and_unexpected_buffer_rejected(self):
        for change in (
            lambda state: state.__setitem__("temporal_positions", torch.zeros(1, 9, 8)),
            lambda state: state["temporal_positions"].fill_(float("inf")),
            lambda state: state.__setitem__("unexpected_position", torch.zeros(1, 10, 8)),
        ):
            with self.subTest(change=change):
                self.save()
                self.rewrite(lambda payload: change(payload["model_state"]))
                with self.assertRaisesRegex(ContractError, "state dictionary|state tensor"):
                    load_selected_checkpoint(self.root)

    def test_state_shape_type_dtype_and_coverage_rejections(self):
        for change in (
            lambda state: state.__setitem__("audio_projection.weight", torch.zeros(8, 255)),
            lambda state: state.__setitem__("audio_projection.bias", [0.] * 8),
            lambda state: state.__setitem__("audio_projection.bias", torch.zeros(8, dtype=torch.int32)),
            lambda state: state.__setitem__("audio_projection.bias", torch.zeros(8, dtype=torch.float64)),
            lambda state: state.pop("correspondence_head.weight"),
        ):
            with self.subTest(change=change):
                self.save()
                self.rewrite(lambda payload: change(payload["model_state"]))
                with self.assertRaisesRegex(ContractError, "state tensor|state dictionary|share one dtype"):
                    load_selected_checkpoint(self.root)
        self.save()
        self.rewrite(lambda payload: payload.__setitem__("model_state", []))
        with self.assertRaisesRegex(ContractError, "state dictionary"):
            load_selected_checkpoint(self.root)

    def test_persisted_binding_detects_same_shape_head_dropout_and_feedforward_claims(self):
        for change in ({"num_heads": 4}, {"dropout": .2}, {"attention_dropout": .3},
                       {"feedforward_dim": 32}, {"d_model": 16}, {"position_encoding": "none"}):
            with self.subTest(change=change):
                original = self.save()
                changed = replace(original, model=replace(original.model, **change))
                (self.root / "checkpoint_metadata.json").write_text(json.dumps(changed.to_dict()), encoding="utf-8")
                # Reconcile the old experiment hash too: this must fail the explicit binding.
                self.rewrite(lambda payload: payload.__setitem__("experiment_fingerprint", changed.experiment_fingerprint))
                with self.assertRaisesRegex(ContractError, "binding disagrees"):
                    load_selected_checkpoint(self.root)

    def test_positional_validation_is_independent_of_both_configuration_records(self):
        original = self.save()
        changed = replace(original, model=replace(original.model, position_encoding="none"))
        (self.root / "checkpoint_metadata.json").write_text(json.dumps(changed.to_dict()), encoding="utf-8")
        def change(payload):
            payload["experiment_fingerprint"] = changed.experiment_fingerprint
            payload["model_binding"]["config"]["position_encoding"] = "none"
        self.rewrite(change)
        with self.assertRaisesRegex(ContractError, "buffer disagrees"):
            load_selected_checkpoint(self.root)

    def test_missing_malformed_and_inconsistent_bindings_and_payloads_rejected(self):
        for change in (
            lambda payload: payload.pop("model_binding"),
            lambda payload: payload.__setitem__("model_binding", []),
            lambda payload: payload["model_binding"].__setitem__("architecture", "late_fusion_mlp"),
            lambda payload: payload["model_binding"]["config"].__setitem__("num_heads", True),
            lambda payload: payload["model_binding"]["config"].__setitem__("extra", 1),
        ):
            with self.subTest(change=change):
                self.save()
                self.rewrite(change)
                with self.assertRaisesRegex(ContractError, "binding"):
                    load_selected_checkpoint(self.root)
        for payload in ([], {}, {"model_state": {}}):
            self.save()
            self.rewrite(lambda original: None)  # keep valid file inventory setup
            path = self.root / "best.pt"
            torch.save(payload, path)
            (self.root / "checkpoint_integrity.json").write_text(json.dumps(_file_entry("best.pt", path)), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "malformed checkpoint payload"):
                load_selected_checkpoint(self.root)

    def test_mlp_legacy_without_binding_and_new_mlp_binding_are_compatible(self):
        model = LateFusionMLP(MLPConfig(hidden_dims=(8,), dropout=0.)).eval()
        self.save(model)
        fresh, _ = load_selected_checkpoint(self.root)
        self.rewrite(lambda payload: payload.pop("model_binding"))
        legacy, _ = load_selected_checkpoint(self.root)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, fresh.state_dict()[name]))
            self.assertTrue(torch.equal(value, legacy.state_dict()[name]))

    def test_weight_architecture_swap_and_file_corruption_rejected(self):
        self.save()
        mlp_state = LateFusionMLP(MLPConfig(hidden_dims=(8,))).state_dict()
        self.rewrite(lambda payload: payload.__setitem__("model_state", mlp_state), update_model_hash=True)
        with self.assertRaisesRegex(ContractError, "state dictionary coverage"):
            load_selected_checkpoint(self.root)
        model = LateFusionMLP(MLPConfig(hidden_dims=(8,)))
        self.save(model)
        self.rewrite(lambda payload: payload.__setitem__("model_state", self.model.state_dict()), update_model_hash=True)
        with self.assertRaisesRegex(ContractError, "state dictionary coverage"):
            load_selected_checkpoint(self.root)
        self.save()
        path = self.root / "best.pt"
        path.write_bytes(path.read_bytes() + b"changed")
        with patch("fusion.train_mlp.torch.load", side_effect=AssertionError("must check hashes before loading")), \
             self.assertRaisesRegex(ContractError, "hash/size"):
            load_selected_checkpoint(self.root)

    def test_fingerprint_and_expected_metadata_checks_still_fail_closed(self):
        meta = self.save()
        with self.assertRaisesRegex(ContractError, "metadata differs"):
            load_selected_checkpoint(self.root, expected_metadata=replace(meta, git_commit="f"*40))
        for field in ("model_sha256", "experiment_fingerprint"):
            self.save()
            self.rewrite(lambda payload: payload.__setitem__(field, "f"*64))
            with self.assertRaisesRegex(ContractError, "fingerprint"):
                load_selected_checkpoint(self.root)

    def test_dimension_and_fabricated_configuration_guards_before_construction(self):
        with patch("fusion.cross_attention_model.nn.Linear", side_effect=AssertionError("must not construct")):
            for change in ({"d_model": 1, "num_heads": 1}, {"d_model": True}, {"d_model": 7},
                           {"num_heads": 0}, {"num_heads": True}, {"feedforward_dim": -1},
                           {"dropout": 1.}, {"attention_dropout": float("nan")}):
                with self.subTest(change=change), self.assertRaises(ContractError):
                    BidirectionalCrossAttention(attention_config(**change))
            for config in ({"d_model": 8}, MLPConfig(), object()):
                with self.assertRaises(ContractError):
                    BidirectionalCrossAttention(config)
            forged = attention_config()
            object.__setattr__(forged, "d_model", 1)
            with self.assertRaises(ContractError):
                build_model(forged)
        class UnsupportedConfig(CrossAttentionConfig):
            pass
        with self.assertRaises(ContractError):
            build_model(UnsupportedConfig())

    def test_supported_minimum_and_default_dimensions_have_finite_gradients(self):
        self.assertEqual((CrossAttentionConfig().d_model, CrossAttentionConfig().num_heads), (256, 4))
        for dimension, heads in ((2, 1), (2, 2), (3, 1), (8, 4)):
            model = BidirectionalCrossAttention(attention_config(d_model=dimension, num_heads=heads)).double()
            a = torch.randn(3, 10, 256, dtype=torch.float64, requires_grad=True)
            v = torch.randn(3, 10, 512, dtype=torch.float64, requires_grad=True)
            logits = model(a, v, torch.ones(3, 10, dtype=torch.bool)).logits
            loss = correspondence_loss(logits, torch.tensor([[1.], [0.], [1.]], dtype=torch.float64), TrainingConfig())
            backward_checked(loss, model, max_gradient_norm=5)
            self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(v.grad).all())


if __name__ == "__main__":
    unittest.main()
