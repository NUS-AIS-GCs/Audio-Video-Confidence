"""Offline synthetic Phase 6 tests. No real weights, media, fitting or network."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fusion.contracts import ContractError
from fusion.cross_attention_model import UnidirectionalAttentionConfig
from fusion.inference import (CANDIDATES, FrozenCandidate, FrozenFusionInference,
                              _sha, expected_provenance, load_verified_train_exports)
from fusion.inference_contracts import (BIN_STARTS, EncoderProvenance, FeatureBatch,
                                       RawMediaInput, encode_raw_media)
from fusion.mlp_model import MLPConfig
from fusion.mlp_training import Predictions, model_fingerprint
from fusion.model_interface import build_model
from fusion.train_mlp import save_checkpoint
from fusion.training_contracts import build_optimizer
from test_fusion_checkpoint_binding import metadata


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = MLPConfig(hidden_dims=(8,), dropout=.2)
        self.make_checkpoint(self.config)

    def make_checkpoint(self, config, candidate="E3"):
        self.model = build_model(config)
        meta = metadata(config)
        exports = tuple((f"{modality}/{name}", "f"*64) for modality in ("audio", "vision")
                        for name in ("SOURCE.txt", "README.md", "class_names.json",
                                     "train_emb.npy", "train_ids.csv", "train_probs.npy"))
        self.meta = replace(meta, data=tuple(replace(d, input_export_hashes=exports) for d in meta.data))
        save_checkpoint(self.root, self.model,
                        build_optimizer(self.model, self.meta.training.optimizer), self.meta)
        self.threshold = CANDIDATES[candidate].threshold
        self.point = {"threshold": self.threshold, "model_sha256": model_fingerprint(self.model),
                      "selected_epoch": 1, "validation_manifest_sha256": self.meta.data[1].manifest_sha256,
                      "decision": "probability >= threshold"}
        (self.root / "frozen_operating_point.json").write_text(json.dumps(self.point))
        self.binding = self.pin(candidate)

    def pin(self, candidate="E3"):
        return FrozenCandidate(candidate, _sha(self.root / "best.pt"),
            _sha(self.root / "checkpoint_metadata.json"), _sha(self.root / "frozen_operating_point.json"),
            model_fingerprint(self.model), self.threshold)

    def adapter(self, binding=None):
        binding = binding or self.binding
        with patch("fusion.inference.CANDIDATES", {binding.name: binding}):
            return FrozenFusionInference(self.root, candidate=binding.name)

    def features(self, batch=2):
        gen = torch.Generator().manual_seed(7)
        return FeatureBatch(torch.randn(batch, 10, 256, generator=gen),
            torch.randn(batch, 10, 512, generator=gen), torch.ones(batch, 10, dtype=torch.bool),
            torch.ones(batch, 10, dtype=torch.bool), expected_provenance(self.meta))

    def test_singleton_and_multiple_batches(self):
        adapter = self.adapter()
        for n in (1, 2, 5):
            result = adapter.predict(self.features(n))
            self.assertEqual(len(result.predictions), n)
            self.assertTrue(all(np.isfinite(p.sigmoid_score) for p in result.predictions))
            self.assertEqual(len(result.predictions[0].valid_bins), 10)

    def test_output_shapes_rejected(self):
        adapter, f = self.adapter(), self.features()
        for changes in ({"audio": torch.zeros(2, 9, 256)}, {"visual": torch.zeros(2, 10, 511)},
                        {"audio": torch.zeros(0, 10, 256)}, {"visual": torch.zeros(1, 10, 512)},
                        {"audio": torch.zeros(2, 10, 256, dtype=torch.int64)}):
            with self.subTest(changes=list(changes)), self.assertRaises(ContractError):
                adapter.predict(replace(f, **changes))

    def test_missing_modality(self):
        for name in ("audio", "visual"):
            with self.assertRaises(ContractError):
                self.adapter().predict(replace(self.features(), **{name: None}))

    def test_mask_intersection(self):
        f = self.features()
        f.audio_validity[:, 2] = False
        f.visual_validity[:, 4] = False
        mask = self.adapter().predict(f).predictions[0].valid_bins
        self.assertFalse(mask[2]); self.assertFalse(mask[4]); self.assertEqual(sum(mask), 8)

    def test_masks_are_boolean_explicit_and_nonempty(self):
        adapter, f = self.adapter(), self.features()
        for value in (None, torch.ones(2, 10), torch.ones(2, 9, dtype=torch.bool),
                      torch.zeros(2, 10, dtype=torch.bool)):
            with self.subTest(value=type(value)), self.assertRaises(ContractError):
                adapter.predict(replace(f, audio_validity=value))

    def test_invalid_content_cannot_affect_prediction(self):
        adapter, f = self.adapter(), self.features()
        f.audio_validity[:, 3] = False
        f.visual_validity[:, 3] = False
        first = adapter.predict(f)
        f.audio[:, 3] = 1e30
        f.visual[:, 3] = -1e30
        self.assertEqual(first, adapter.predict(f))

    def test_nonfinite_including_masked_bins(self):
        for modality in ("audio", "visual"):
            for value in (float('nan'), float('inf'), -float('inf')):
                f = self.features()
                getattr(f, modality)[:, 0] = value
                f.audio_validity[:, 0] = False
                with self.subTest(modality=modality, value=value), self.assertRaises(ContractError):
                    self.adapter().predict(f)

    def test_float32_overflow_is_rejected(self):
        f = self.features()
        with self.assertRaises(ContractError):
            self.adapter().predict(replace(f, audio=torch.full((2, 10, 256), 1e100, dtype=torch.float64)))

    def test_provenance_mismatches(self):
        adapter, f = self.adapter(), self.features()
        for p in (replace(f.provenance[0], run_identity="other"),
                  replace(f.provenance[0], preprocessing_profile="different-normalization"),
                  replace(f.provenance[0], encoder_checkpoint_sha256="a"*64),
                  replace(f.provenance[0], origin="raw_media"),
                  replace(f.provenance[0], export_hashes=())):
            with self.subTest(provenance=p), self.assertRaises(ContractError):
                adapter.predict(replace(f, provenance=(p, f.provenance[1])))

    def test_missing_or_swapped_provenance(self):
        f = self.features()
        for value in ((), f.provenance[::-1], (None, None)):
            with self.assertRaises(ContractError):
                self.adapter().predict(replace(f, provenance=value))

    def test_temporal_index_mismatch(self):
        f = self.features()
        for changes in ({"visual_bin_starts": tuple(reversed(BIN_STARTS))},
                        {"audio_bin_starts": tuple(t+.5 for t in BIN_STARTS)},
                        {"duration_seconds": 9.9}, {"timing_basis": "event_annotation"},
                        {"audio_bin_starts": (False,)+BIN_STARTS[1:]}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.adapter().predict(replace(f, **changes))

    def test_missing_checkpoint(self):
        (self.root / "best.pt").unlink()
        with self.assertRaisesRegex(ContractError, "cannot load"):
            self.adapter()

    def test_corrupt_checkpoint(self):
        (self.root / "best.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ContractError, "pinned"):
            self.adapter()

    def test_safe_loader_is_reused(self):
        original = torch.load
        with patch("torch.load", wraps=original) as load:
            self.adapter()
        self.assertTrue(load.call_args.kwargs["weights_only"])
        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")

    def test_corrupt_payload_rejected_by_safe_loader_even_with_recomputed_hashes(self):
        (self.root / "best.pt").write_bytes(b"corrupt")
        integrity = json.loads((self.root / "checkpoint_integrity.json").read_text())
        integrity.update(sha256=_sha(self.root / "best.pt"), size_bytes=7)
        (self.root / "checkpoint_integrity.json").write_text(json.dumps(integrity))
        with self.assertRaises(ContractError):
            self.adapter(self.pin())

    def test_wrong_checkpoint_architecture(self):
        value = json.loads((self.root / "checkpoint_metadata.json").read_text())
        value["model"]["hidden_dims"] = [16]
        (self.root / "checkpoint_metadata.json").write_text(json.dumps(value))
        with self.assertRaises(ContractError):
            self.adapter(self.pin())

    def test_model_threshold_mismatch(self):
        self.point["model_sha256"] = "0"*64
        (self.root / "frozen_operating_point.json").write_text(json.dumps(self.point))
        with self.assertRaisesRegex(ContractError, "binding"):
            self.adapter(self.pin())

    def test_threshold_changes_rejected(self):
        self.point["threshold"] = .5
        (self.root / "frozen_operating_point.json").write_text(json.dumps(self.point))
        with self.assertRaises(ContractError):
            self.adapter()
        with self.assertRaisesRegex(ContractError, "binding"):
            self.adapter(self.pin())

    def test_no_silent_candidate_or_device_fallback(self):
        for changes in ({"candidate": "unknown"}, {"device": "cuda"}, {"device": "mps"}):
            with self.assertRaises(ContractError):
                FrozenFusionInference(self.root, **changes)

    def test_eval_no_grad_frozen_and_cpu(self):
        adapter = self.adapter()
        observed = []
        handle = adapter._model.register_forward_pre_hook(
            lambda m, args: observed.append((m.training, torch.is_grad_enabled(), args[0].device.type)))
        self.addCleanup(handle.remove)
        f = self.features()
        f.audio.requires_grad_(True)
        adapter._model.train()  # adapter restores eval on every call
        adapter.predict(f)
        self.assertEqual(observed, [(False, False, "cpu")])
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in adapter._model.parameters()))
        self.assertIsNone(f.audio.grad)

    def test_deterministic_and_no_rng_changes(self):
        adapter, f = self.adapter(), self.features()
        rng = torch.get_rng_state().clone()
        self.assertEqual(adapter.predict(f), adapter.predict(f))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_model_mutation_rejected(self):
        adapter = self.adapter()
        with torch.no_grad():
            next(adapter._model.parameters()).add_(1)
        with self.assertRaisesRegex(ContractError, "changed"):
            adapter.predict(self.features())

    def test_metadata_does_not_change_predictive_inputs(self):
        adapter, f = self.adapter(), self.features()
        observed = []
        handle = adapter._model.register_forward_pre_hook(lambda m, args: observed.append(len(args)))
        self.addCleanup(handle.remove)
        first = adapter.predict(f)
        second = adapter.predict(replace(f, diagnostics={"ground_truth": [0, 1], "category": "ignored"}))
        self.assertEqual(first.predictions, second.predictions)
        self.assertEqual(observed, [3, 3])

    def test_original_substituted_semantics(self):
        # Deliberately fixed synthetic outputs; no real model substitution.
        adapter = self.adapter()
        from fusion.mlp_model import FusionOutput
        with patch.object(adapter._model, "forward", return_value=FusionOutput(torch.tensor([[-20.], [20.]]))):
            result = adapter.predict(self.features())
        self.assertEqual([(p.predicted_target, p.pairing_decision) for p in result.predictions],
                         [(0, "substituted"), (1, "original")])
        self.assertFalse(result.causal_source_identity_claim)
        self.assertIn("uncalibrated", result.score_semantics)

    def test_sigmoid_matches_phase45_float64_convention(self):
        adapter, f = self.adapter(), self.features()
        result = adapter.predict(f)
        legacy = Predictions("train", "primary", "a"*64, "b"*64,
                             (("train", "a", "a"), ("train", "b", "b")),
                             np.array([1, 0]), np.array([p.logit for p in result.predictions]), 0.)
        np.testing.assert_array_equal(legacy.probabilities, [p.sigmoid_score for p in result.predictions])

    def test_nonfinite_scores_rejected(self):
        from fusion.mlp_model import FusionOutput
        adapter = self.adapter()
        with patch.object(adapter._model, "forward", return_value=FusionOutput(torch.tensor([[float('nan')], [0.]]))):
            with self.assertRaisesRegex(ContractError, "finite"):
                adapter.predict(self.features())

    def test_output_logit_shape_is_strict(self):
        from fusion.mlp_model import FusionOutput
        adapter = self.adapter()
        for value in (torch.zeros(2), torch.zeros(1, 2), torch.zeros(2, 1, 1)):
            with patch.object(adapter._model, "forward", return_value=FusionOutput(value)):
                with self.assertRaisesRegex(ContractError, "B,1"):
                    adapter.predict(self.features())

    def test_a2_loader_and_forward_compatibility(self):
        self.make_checkpoint(UnidirectionalAttentionConfig(d_model=8, num_heads=2, feedforward_dim=16,
                             attention_direction="video_to_audio"), "A2")
        adapter, f = self.adapter(), self.features()
        actual = adapter.predict(f)
        with torch.no_grad():
            expected = adapter._model(f.audio, f.visual, f.audio_validity).logits[:, 0].numpy()
        np.testing.assert_array_equal(expected, [p.logit for p in actual.predictions])
        self.assertEqual(actual.threshold, 0.0026640388978268056)

    def test_e3_and_a2_have_distinct_immutable_bindings(self):
        self.assertNotEqual(CANDIDATES['E3'].checkpoint_sha256, CANDIDATES['A2'].checkpoint_sha256)
        self.assertEqual(CANDIDATES['E3'].threshold, 0.009579099009671126)
        with self.assertRaises(TypeError):
            CANDIDATES['other'] = self.binding

    def test_export_reader_checks_bytes_before_loading(self):
        with self.assertRaises((ContractError, FileNotFoundError)):
            load_verified_train_exports(self.root, self.root, self.meta)

    def test_unbound_sidecars_rejected(self):
        (self.root / "train_valid.npy").write_bytes(b"unbound")
        with self.assertRaisesRegex(ContractError, "sidecars"):
            load_verified_train_exports(self.root, self.root, self.meta)

    def test_structured_serializable_result(self):
        value = self.adapter().predict(self.features()).to_dict()
        self.assertEqual(len(value['predictions']), 2)
        self.assertEqual(value['provenance'][0]['encoder_checkpoint_sha256'], None)
        json.dumps(value, allow_nan=False)

    def precision_adapters(self):
        for candidate, config in (
            ("E3", self.config),
            ("A2", UnidirectionalAttentionConfig(d_model=8, num_heads=2, feedforward_dim=16,
                                                attention_direction="video_to_audio")),
        ):
            self.make_checkpoint(config, candidate)
            yield candidate, self.adapter()

    @staticmethod
    def caller_precision_state():
        return (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"),
                torch.is_autocast_cache_enabled(), torch.is_grad_enabled())

    def test_both_models_preserve_float32_results_under_cpu_autocast(self):
        caller = self.caller_precision_state()
        for candidate, adapter in self.precision_adapters():
            before = model_fingerprint(adapter._model)
            for n in (1, 3):
                f = self.features(n)
                if n > 1:
                    f.audio_validity[0, 2] = False
                    f.visual_validity[1, 4:7] = False
                    f.audio_validity[2, :9] = False
                reference = adapter.predict(f)
                observed = []

                def inspect(module, args, output):
                    observed.append((module.training, torch.is_grad_enabled(),
                                     torch.is_autocast_enabled("cpu"),
                                     args[0].dtype, args[1].dtype, output.logits.dtype))

                handle = adapter._model.register_forward_hook(inspect)
                try:
                    rng = torch.get_rng_state().clone()
                    for dtype in (torch.bfloat16, torch.float16):
                        with self.subTest(candidate=candidate, batch=n, dtype=dtype):
                            with torch.autocast("cpu", dtype=dtype, cache_enabled=False):
                                active = self.caller_precision_state()
                                self.assertTrue(active[0])
                                adapter._model.train()  # predict must restore eval even under autocast
                                self.assertEqual(reference, adapter.predict(f))
                                self.assertEqual(reference, adapter.predict(f))
                                self.assertEqual(active, self.caller_precision_state())
                            self.assertEqual(caller, self.caller_precision_state())
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                finally:
                    handle.remove()
                self.assertEqual(observed, [(False, False, False, torch.float32,
                                             torch.float32, torch.float32)] * 4)
            self.assertEqual(before, model_fingerprint(adapter._model))
            self.assertTrue(all(not p.requires_grad and p.grad is None
                                for p in adapter._model.parameters()))
            self.assertFalse(adapter._model.training)

    def test_autocast_state_restored_after_validation_and_forward_errors(self):
        caller = self.caller_precision_state()
        for candidate, adapter in self.precision_adapters():
            f = self.features()
            invalid = replace(f, audio=torch.full_like(f.audio, float('nan')))
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(candidate=candidate, dtype=dtype):
                    with torch.autocast("cpu", dtype=dtype):
                        active = self.caller_precision_state()
                        with self.assertRaises(ContractError):
                            adapter.predict(invalid)
                        self.assertEqual(active, self.caller_precision_state())
                        with patch.object(adapter._model, "forward", side_effect=RuntimeError("synthetic failure")):
                            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                                adapter.predict(f)
                        self.assertEqual(active, self.caller_precision_state())
                    self.assertEqual(caller, self.caller_precision_state())

    def test_nested_disabled_autocast_state_preserved_for_both_models(self):
        caller = self.caller_precision_state()
        for _, adapter in self.precision_adapters():
            f = self.features(1)
            reference = adapter.predict(f)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                active = self.caller_precision_state()
                with torch.autocast("cpu", enabled=False):
                    disabled = self.caller_precision_state()
                    self.assertFalse(disabled[0])
                    self.assertEqual(reference, adapter.predict(f))
                    self.assertEqual(disabled, self.caller_precision_state())
                self.assertEqual(active, self.caller_precision_state())
            self.assertEqual(caller, self.caller_precision_state())

    def test_exact_frozen_threshold_equality_with_and_without_autocast(self):
        for candidate, adapter in self.precision_adapters():
            f = self.features(3)
            threshold = CANDIDATES[candidate].threshold
            scores = np.array([np.nextafter(threshold, -np.inf), threshold,
                               np.nextafter(threshold, np.inf)], dtype=np.float64)
            for dtype in (None, torch.bfloat16, torch.float16):
                with self.subTest(candidate=candidate, dtype=dtype):
                    with torch.autocast("cpu", enabled=dtype is not None, dtype=dtype):
                        # Inject exact boundary scores; float32 logits cannot in
                        # general invert to an arbitrary float64 cutoff exactly.
                        with patch("fusion.inference.np.exp", return_value=scores):
                            result = adapter.predict(f)
                    self.assertEqual(result.threshold, threshold)
                    self.assertEqual([p.sigmoid_score for p in result.predictions], scores.tolist())
                    self.assertEqual([p.predicted_target for p in result.predictions], [0, 1, 1])
                    self.assertEqual([p.pairing_decision for p in result.predictions],
                                     ["substituted", "original", "original"])

    def test_non_float32_model_output_is_rejected_without_silent_recasting(self):
        from fusion.mlp_model import FusionOutput
        for candidate, adapter in self.precision_adapters():
            for dtype in (torch.float16, torch.bfloat16, torch.float64):
                with self.subTest(candidate=candidate, dtype=dtype):
                    output = FusionOutput(torch.zeros(2, 1, dtype=dtype))
                    with patch.object(adapter._model, "forward", return_value=output):
                        with self.assertRaisesRegex(ContractError, "float32"):
                            adapter.predict(self.features())

    def test_unsupported_live_model_device_is_rejected_before_inference(self):
        for _, adapter in self.precision_adapters():
            f = self.features()
            adapter._model.to("meta")
            with self.assertRaisesRegex(ContractError, "CPU"):
                adapter.predict(f)


class RawContractTests(unittest.TestCase):
    def media(self):
        return RawMediaInput(torch.zeros(1, 160000), torch.zeros(1, 10, 224, 224, 3, dtype=torch.uint8),
                             16000, 10., 10., 0., 0., BIN_STARTS)

    def test_canonical_contract_but_execution_blocked(self):
        value = self.media()
        value.validate()
        with self.assertRaisesRegex(ContractError, "BLOCKED"):
            encode_raw_media(value)

    def test_arbitrary_duration_and_offset_rejected(self):
        for changes in ({"audio_duration": 9.}, {"video_duration": 11.},
                        {"audio_start_seconds": .1}, {"video_start_seconds": -.1}):
            with self.assertRaises(ContractError):
                replace(self.media(), **changes).validate()

    def test_resampling_channels_and_numeric_errors_require_explicit_upstream_work(self):
        for changes in ({"sample_rate": 44100}, {"waveform": torch.zeros(1, 2, 160000)},
                        {"waveform": torch.zeros(1, 159999)},
                        {"waveform": torch.full((1, 160000), float('nan'))},
                        {"waveform": torch.full((1, 160000), 1.1)}):
            with self.assertRaises(ContractError):
                replace(self.media(), **changes).validate()

    def test_frame_cadence_shape_and_timestamps(self):
        for changes in ({"frames": torch.zeros(1, 9, 224, 224, 3, dtype=torch.uint8)},
                        {"frames": torch.zeros(1, 10, 224, 224, 3)},
                        {"frame_timestamps": tuple(t+.5 for t in BIN_STARTS)}):
            with self.assertRaises(ContractError):
                replace(self.media(), **changes).validate()

    def test_missing_raw_content_rejected_before_global_context_encoding(self):
        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[0, 3] = False
        for name in ("audio_validity", "visual_validity"):
            with self.assertRaises(ContractError):
                replace(self.media(), **{name: mask}).validate()

    def test_unsupported_raw_file_rejected(self):
        with self.assertRaisesRegex(ContractError, "decoding is unsupported"):
            encode_raw_media("unseen.mp4")
