"""Offline synthetic integration only; no real encoder or dataset dependency."""
from dataclasses import FrozenInstanceError, asdict, dataclass, replace
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from fusion.encoder_interfaces import (Artifact, EncoderOutput, EncoderSpec, EncoderStamp, FrozenEncoder,
    IntegrationError, AUDIO_PREPROCESSING, VIDEO_PREPROCESSING, DEPENDENCIES, digest, load_encoder, readiness)
from fusion.end_to_end import (DecodedMedia, DecodedTensorDecoder, DecodedTensorSource, EncodedBundle,
                               MediaFusionPipeline, TrainParityFusion, _bind_reference)
from fusion.contracts import load_bundle
from fusion.inference_contracts import BIN_STARTS, RawMediaInput
from fusion.inference import FrozenFusionInference


def signature(modality):
    return EncoderStamp(modality, "beats_lora_r8_v1 / beats_iter3" if modality == "audio" else "resnet18_v1",
                        (("trained", "a"*64),), "b"*64, "c"*64, (("torch", str(torch.__version__)),))


@dataclass(frozen=True)
class StubPrediction:
    candidate: str
    threshold: float

    def to_dict(self):
        return {"candidate": self.candidate, "threshold": self.threshold, "synthetic": True}


class StubDecoder:
    is_synthetic = True
    signature = "d"*64

    def __init__(self, media, events):
        self.media, self.events = media, events

    def decode(self, source):
        self.events.append("decode")
        return self.media


class StubEncoder:
    is_synthetic = True

    def __init__(self, modality, events):
        self.provenance, self.events = signature(modality), events
        self.modality = modality
        self.output_change = lambda x: x
        self.blocked = False
        self.parameter = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)

    def require_ready(self):
        if self.blocked:
            raise IntegrationError("MISSING_ARTIFACT", "readiness", "synthetic missing checkpoint")

    def encode(self, media):
        self.events.append(self.modality)
        assert not torch.is_grad_enabled() and not torch.is_autocast_enabled("cpu")
        dim = 256 if self.modality == "audio" else 512
        b = media.raw.waveform.shape[0]
        x = torch.arange(10, dtype=torch.float32)[None, :, None].expand(b, 10, dim).clone()
        return self.output_change(EncoderOutput(x, torch.ones(b, 10, dtype=torch.bool), BIN_STARTS, self.provenance))


class StubFusion:
    is_synthetic = True

    def __init__(self, events):
        self.events, self.bundles = events, []

    def predict(self, bundle):
        self.events.append("fusion")
        self.bundles.append(bundle)
        return {"E3": StubPrediction("E3", .009579099009671126),
                "A2": StubPrediction("A2", .0026640388978268056)}


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = RawMediaInput(torch.zeros(2, 160000), torch.zeros(2, 10, 224, 224, 3, dtype=torch.uint8),
                            16000, 10., 10., 0., 0., BIN_STARTS,
                            torch.ones(2, 10, dtype=torch.bool), torch.ones(2, 10, dtype=torch.bool))
        self.media = DecodedMedia(raw, ("a", "b"), ("a", "b"), "d"*64, (), "shared_clock_pts_v1")
        self.events = []
        self.decoder = StubDecoder(self.media, self.events)
        self.audio, self.visual = (StubEncoder(m, self.events) for m in ("audio", "vision"))
        self.fusion = StubFusion(self.events)
        self.pipeline = MediaFusionPipeline(self.decoder, self.audio, self.visual, self.fusion, allow_synthetic=True)

    def artifact(self, name, content):
        p = self.root / name
        p.write_bytes(content)
        return Artifact(str(p), hashlib.sha256(content).hexdigest())

    def source(self, change=None):
        np.save(self.root / "wave.npy", self.media.raw.waveform.numpy())
        np.save(self.root / "frames.npy", self.media.raw.frames.numpy())
        a, v = (self.artifact(n, (self.root/n).read_bytes()) for n in ("wave.npy", "frames.npy"))
        meta = {"schema": "gc2-decoded-ten-bin-v1", "decoder_signature": "d"*64,
                "timing_verification": "shared_clock_pts_v1", "waveform_sha256": a.sha256, "frames_sha256": v.sha256,
                "sample_rate": 16000, "audio_duration": 10., "video_duration": 10.,
                "audio_start_seconds": 0., "video_start_seconds": 0., "frame_timestamps": list(BIN_STARTS),
                "audio_sample_ids": list(self.media.audio_sample_ids), "visual_sample_ids": list(self.media.visual_sample_ids),
                "audio_validity": [[True]*10]*2, "visual_validity": [[True]*10]*2}
        if change:
            meta.update(change)
        return DecodedTensorSource(a, v, self.artifact("manifest.json", json.dumps(meta).encode()))

    def spec(self, modality="audio", artifacts=None):
        roles = ("base", "trained", "normalization") if modality == "audio" else ("trained",)
        return EncoderSpec(modality, signature(modality).run_identity,
                           artifacts or tuple((r, Artifact(str(self.root/r), "a"*64)) for r in roles),
                           "{}", AUDIO_PREPROCESSING if modality == "audio" else VIDEO_PREPROCESSING,
                           tuple((n, "not-installed-fixture") for n in DEPENDENCIES[modality]))

    def test_execution_order_shapes_and_both_prediction_interfaces(self):
        result = self.pipeline.run(None)
        self.assertEqual(self.events, ["decode", "audio", "vision", "fusion"])
        self.assertEqual(result["status"], "SYNTHETIC")
        self.assertEqual(result["feature_schema"]["audio"], [2, 10, 256])
        self.assertEqual(result["feature_schema"]["visual"], [2, 10, 512])
        self.assertEqual(result["predictions"]["E3"]["threshold"], .009579099009671126)
        self.assertEqual(result["predictions"]["A2"]["threshold"], .0026640388978268056)

    def test_original_temporal_indices_preserved(self):
        self.pipeline.run(None)
        b = self.fusion.bundles[0]
        self.assertEqual(b.audio.bin_starts, BIN_STARTS)
        self.assertEqual(b.audio.embeddings[0, :, 0].tolist(), list(range(10)))

    def test_original_synthetic_flag_upgrade_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            self.pipeline.synthetic = False
        result = self.pipeline.run(None)
        self.assertEqual(result["status"], "SYNTHETIC")
        self.assertTrue(self.fusion.bundles[-1].synthetic)
        self.assertEqual(self.events, ["decode", "audio", "vision", "fusion"])

    def test_every_component_binding_and_optin_are_frozen(self):
        for name in ("decoder", "audio_encoder", "visual_encoder", "fusion", "allow_synthetic"):
            with self.subTest(name=name), self.assertRaises(FrozenInstanceError):
                setattr(self.pipeline, name, None)
        self.assertEqual(self.pipeline.run(None)["status"], "SYNTHETIC")

    def test_each_component_mode_mutation_rejected_before_execution(self):
        for component in (self.decoder, self.audio, self.visual, self.fusion):
            with self.subTest(component=type(component).__name__):
                component.is_synthetic = False
                result = self.pipeline.try_run(None)
                self.assertEqual(result["status"], "REJECTED")
                self.assertEqual(result["error"]["code"], "COMPONENT_MUTATED")
                self.assertEqual(self.events, [])
                component.is_synthetic = True

    def test_non_boolean_mode_declarations_rejected(self):
        self.audio.is_synthetic = 1
        self.assertEqual(self.pipeline.try_run(None)["status"], "REJECTED")
        self.assertEqual(self.events, [])
        with self.assertRaisesRegex(IntegrationError, "booleans"):
            MediaFusionPipeline(self.decoder, self.audio, self.visual, self.fusion, allow_synthetic=True)

    def test_changed_decoder_and_producer_bindings_rejected(self):
        self.decoder.signature = "e"*64
        self.assertEqual(self.pipeline.try_run(None)["status"], "REJECTED")
        self.decoder.signature = "d"*64
        self.audio.provenance = replace(self.audio.provenance, configuration_sha256="f"*64)
        self.assertEqual(self.pipeline.try_run(None)["status"], "REJECTED")
        self.assertEqual(self.events, [])

    def test_mode_mutation_during_execution_prevents_parity_call(self):
        def mutate(output):
            self.visual.is_synthetic = False
            return output
        self.audio.output_change = mutate
        result = self.pipeline.try_run(None)
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(self.events, ["decode", "audio"])
        self.assertEqual(self.fusion.bundles, [])

    def test_matching_exports_from_synthetic_path_keep_synthetic_status(self):
        bridge, _, calls = self.bridge_fixture()
        # Even exact reference values cannot turn a stub parity callback into
        # authenticated encoder execution. Real gate itself rejects this bundle.
        self.events.clear()
        self.fusion.predict = lambda b: {"E3": StubPrediction("E3", .5), "A2": StubPrediction("A2", .5)}
        self.assertEqual(self.pipeline.run(None)["status"], "SYNTHETIC")
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(IntegrationError, "synthetic"):
            bridge.predict(self.fusion.bundles[-1])

    def test_forged_parity_success_objects_cannot_upgrade_status(self):
        forged = SimpleNamespace(to_dict=lambda: {"status": "TRAIN_RAW_PARITY_VERIFIED", "verified": True})
        self.fusion.predict = lambda b: {"E3": forged, "A2": forged}
        result = self.pipeline.run(None)
        self.assertEqual(result["status"], "SYNTHETIC")
        self.assertTrue(result["predictions"]["E3"]["verified"])

    def test_partial_visual_failure_returns_no_predictions_or_stale_success(self):
        self.assertEqual(self.pipeline.run(None)["status"], "SYNTHETIC")
        self.events.clear()
        with patch.object(self.visual, "encode", side_effect=RuntimeError("visual failure")):
            result = self.pipeline.try_run(None)
        self.assertEqual(result["status"], "REJECTED")
        self.assertNotIn("predictions", result)
        self.assertEqual(self.events, ["decode", "audio"])
        self.assertEqual(len(self.fusion.bundles), 1)
        self.assertEqual(self.pipeline.run(None)["status"], "SYNTHETIC")

    def test_parity_failure_and_missing_result_never_reuse_previous_success(self):
        self.pipeline.run(None)
        for output in (RuntimeError("parity incomplete"), {"E3": StubPrediction("E3", .5)}):
            with patch.object(self.fusion, "predict", side_effect=output if isinstance(output, Exception) else None,
                              return_value=output):
                result = self.pipeline.try_run(None)
            self.assertEqual(result["status"], "REJECTED")
            self.assertNotIn("predictions", result)
        self.assertEqual(self.pipeline.run(None)["status"], "SYNTHETIC")

    def test_mode_mutation_during_result_serialization_rejected(self):
        def serialize():
            self.fusion.is_synthetic = False
            return {"verified": True}
        self.fusion.predict = lambda b: {k: SimpleNamespace(to_dict=serialize) for k in ("E3", "A2")}
        result = self.pipeline.try_run(None)
        self.assertEqual(result["status"], "REJECTED")
        self.assertNotIn("predictions", result)

    def test_validity_intersection_propagates_without_annotation_labels(self):
        def audio_change(x):
            x.validity[0, 2] = False
            return x
        self.audio.output_change = audio_change
        result = self.pipeline.run(None)
        self.assertFalse(result["feature_schema"]["validity"][0][2])
        self.assertEqual(sum(result["feature_schema"]["validity"][0]), 9)

    def test_missing_checkpoint_prevents_any_encoder_execution(self):
        self.visual.blocked = True
        result = self.pipeline.try_run(None)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(self.events, [])

    def test_missing_real_artifacts_block_before_loader_or_model_construction(self):
        with patch("torch.load") as load:
            with self.assertRaises(IntegrationError):
                load_encoder(self.spec())
        load.assert_not_called()
        self.assertEqual(readiness(self.spec())["status"], "BLOCKED")

    def test_missing_normalization_is_identified(self):
        report = readiness(self.spec())
        self.assertIn("normalization", [i.get("role") for i in report["issues"]])

    def test_checkpoint_hash_mismatch_rejected(self):
        artifact = self.artifact("weight", b"synthetic checkpoint bytes")
        with self.assertRaisesRegex(IntegrationError, "bytes differ"):
            replace(artifact, sha256="0"*64).verify()

    def test_no_untrained_public_encoder_construction(self):
        with self.assertRaisesRegex(IntegrationError, "load_encoder"):
            FrozenEncoder(self.spec(), torch.nn.Linear(1, 1))

    def test_no_implicit_synthetic_or_unknown_production_component(self):
        with self.assertRaises(IntegrationError):
            MediaFusionPipeline(self.decoder, self.audio, self.visual, self.fusion)
        for x in (self.decoder, self.audio, self.visual, self.fusion):
            x.is_synthetic = False
        with self.assertRaisesRegex(IntegrationError, "concrete"):
            MediaFusionPipeline(self.decoder, self.audio, self.visual, self.fusion)

    def test_decoded_tensor_source_verifies_hashes_and_metadata(self):
        decoded = DecodedTensorDecoder("d"*64).decode(self.source())
        self.assertEqual(decoded.raw.waveform.dtype, torch.float32)
        self.assertEqual(decoded.raw.frames.dtype, torch.uint8)
        self.assertEqual(decoded.audio_sample_ids, ("a", "b"))

    def test_compressed_media_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(IntegrationError, "compressed"):
            DecodedTensorDecoder("d"*64).decode("clip.mp4")

    def test_source_corruption_rejected(self):
        source = self.source()
        Path(source.waveform.path).write_bytes(b"corrupted")
        with self.assertRaises(IntegrationError):
            DecodedTensorDecoder("d"*64).decode(source)

    def test_duration_rate_offsets_and_timing_fail_before_encoders(self):
        for changes in ({"audio_duration": 9}, {"sample_rate": 44100}, {"video_start_seconds": .1}):
            self.decoder.media = replace(self.media, raw=replace(self.media.raw, **changes))
            self.events.clear()
            self.assertEqual(self.pipeline.try_run(None)["status"], "REJECTED")
            self.assertEqual(self.events, ["decode"])

    def test_unverified_clock_and_wrong_decoder_rejected(self):
        for changes in ({"timing_verification": "unverified"}, {"decoder_signature": "e"*64}):
            self.decoder.media = replace(self.media, **changes)
            self.assertEqual(self.pipeline.try_run(None)["status"], "REJECTED")

    def test_duplicate_reordered_or_bad_timestamps_rejected(self):
        for value in ((0.,)*10, tuple(reversed(BIN_STARTS)), tuple(t+.5 for t in BIN_STARTS)):
            with self.assertRaises(IntegrationError):
                DecodedTensorDecoder("d"*64).decode(self.source({"frame_timestamps": value}))

    def test_invalid_raw_masks_and_missing_modality_rejected(self):
        for change in ({"audio_validity": None}, {"audio_validity": torch.zeros(2, 10, dtype=torch.bool)},
                       {"waveform": None}, {"frames": None}):
            self.decoder.media = replace(self.media, raw=replace(self.media.raw, **change))
            self.assertEqual(self.pipeline.try_run(None)["status"], "REJECTED")

    def test_numeric_masks_and_extra_label_metadata_rejected(self):
        for change in ({"audio_validity": [[1]*10]*2}, {"event_labels": [True]*10}):
            with self.assertRaises(IntegrationError):
                DecodedTensorDecoder("d"*64).decode(self.source(change))

    def test_encoder_output_shape_finiteness_mask_and_provenance_rejected(self):
        changes = ({"embeddings": torch.zeros(2, 10, 255)},
                   {"embeddings": torch.full((2, 10, 256), float('nan'))},
                   {"validity": torch.zeros(2, 10, dtype=torch.bool)},
                   {"validity": torch.ones(2, 10)}, {"bin_starts": tuple(reversed(BIN_STARTS))},
                   {"provenance": replace(self.audio.provenance, run_identity="wrong")})
        for change in changes:
            self.audio.output_change = lambda x, c=change: replace(x, **c)
            self.assertEqual(self.pipeline.try_run(None)["status"], "REJECTED")

    def test_no_parameter_updates_and_deterministic_orchestration(self):
        before = self.audio.parameter.detach().clone()
        self.assertEqual(self.pipeline.run(None), self.pipeline.run(None))
        self.assertTrue(torch.equal(before, self.audio.parameter))
        self.assertIsNone(self.audio.parameter.grad)

    def test_caller_autocast_state_restored_on_success_and_failure(self):
        initial = (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"), torch.is_grad_enabled())
        for dtype in (torch.bfloat16, torch.float16):
            reference = self.pipeline.run(None)
            with torch.autocast("cpu", dtype=dtype, cache_enabled=False):
                self.assertEqual(reference, self.pipeline.run(None))
                self.assertTrue(torch.is_autocast_enabled("cpu"))
                self.assertEqual(torch.get_autocast_dtype("cpu"), dtype)
                self.assertFalse(torch.is_autocast_cache_enabled())
                self.visual.blocked = True
                self.pipeline.try_run(None)
                self.visual.blocked = False
                self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertEqual(initial, (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"), torch.is_grad_enabled()))

    def test_preprocessing_and_device_mismatch_rejected(self):
        with self.assertRaises(IntegrationError):
            replace(self.spec(), preprocessing="wrong normalization").validate()
        with self.assertRaises(IntegrationError):
            load_encoder(self.spec(), device="cuda")

    def test_structured_encoder_failure_has_stage_and_no_fallback(self):
        with patch.object(self.audio, "encode", side_effect=RuntimeError("decoder/encoder failure")):
            result = self.pipeline.try_run(None)
        self.assertEqual(result["error"]["stage"], "audio")
        self.assertEqual(self.events, ["decode"])

    def test_foundation_checksum_documents_actual_bytes(self):
        root = Path(__file__).resolve().parents[1]
        if not (root/"requirements-fusion-training.txt").exists():
            self.skipTest("local pre-existing untracked requirements file unavailable")
        actual = hashlib.sha256((root/"requirements-fusion-training.txt").read_bytes()).hexdigest()
        text = (root/"docs/fusion_end_to_end_inference_foundation.md").read_text(encoding="utf-8")
        self.assertIn("byte-level SHA-256", text)
        self.assertIn(actual, text)

    def bridge_fixture(self):
        # Synthetic source files with independently pinned byte hashes. Bypass
        # model construction only for offline gate tests, never readiness evidence.
        self.pipeline.run(None)
        bundle = replace(self.fusion.bundles[-1], synthetic=False,
            decoded=replace(self.media, audio_sample_ids=("a__00", "b__00"), visual_sample_ids=("a__00", "b__00")))
        # Distinct rows make a row-swap attack observable without changing shapes.
        bundle.audio.embeddings[1].add_(100)
        bundle.visual.embeddings[1].add_(100)
        bridge = object.__new__(TrainParityFusion)
        from fusion.inference_contracts import EncoderProvenance
        exports, provenance = [], []
        for key, output in (("audio", bundle.audio), ("visual", bundle.visual)):
            root = Path(tempfile.mkdtemp(prefix="reference-" + key, dir=self.root))
            names = [f"event{i:02d}" for i in range(28)] + ["background"]
            (root/"class_names.json").write_text(json.dumps(names), encoding="utf-8")
            (root/"SOURCE.txt").write_text("synthetic source-file integrity fixture", encoding="utf-8")
            (root/"README.md").write_text("synthetic fixture; not trained encoder evidence", encoding="utf-8")
            with (root/"train_ids.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("sample_id", "clip_id", "category", "start", "end"))
                writer.writerows((("a__00", "a", names[0], 1, 2), ("b__00", "b", names[0], 1, 2)))
            np.save(root/"train_emb.npy", output.embeddings.numpy().astype(np.float16))
            np.save(root/"train_probs.npy", np.full((2, 10, 29), 1/29, dtype=np.float16))
            modality = "audio" if key == "audio" else "vision"
            exports.append(load_bundle(root, modality, splits=("train",), expected_counts=None))
            for value in (exports[-1].splits["train"].embeddings, exports[-1].splits["train"].probabilities):
                if getattr(value, "_mmap", None) is not None:
                    self.addCleanup(value._mmap.close)
            pins = tuple((p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.iterdir()))
            provenance.append(EncoderProvenance(modality, "synthetic-file-fixture", pins))
        calls = []
        def predict(features):
            calls.append(features)
            return StubPrediction("fixture", .5)
        for name, value in (("is_synthetic", False), ("expected_audio", self.audio.provenance),
                ("expected_visual", self.visual.provenance), ("audio", exports[0]), ("visual", exports[1]),
                ("provenance", tuple(provenance)), ("adapters", {"E3": SimpleNamespace(predict=predict),
                                                                           "A2": SimpleNamespace(predict=predict)}),
                ("_references", tuple(_bind_reference(b, p) for b, p in zip(exports, provenance))),
                ("_provenance_sha256", digest([asdict(p) for p in provenance]))):
            object.__setattr__(bridge, name, value)
        return bridge, bundle, calls

    def test_train_parity_bridge_uses_produced_roundtrip_features_identically(self):
        bridge, bundle, calls = self.bridge_fixture()
        result = bridge.predict(bundle)
        self.assertEqual(set(result), {"E3", "A2"})
        self.assertIs(calls[0], calls[1])
        self.assertTrue(torch.equal(calls[0].audio, bundle.audio.embeddings))

    def test_parity_mismatch_rejects_before_any_fusion_call(self):
        bridge, bundle, calls = self.bridge_fixture()
        changed = bundle.audio.embeddings.clone()
        changed[0, 0, 0] = 1
        with self.assertRaisesRegex(IntegrationError, "exactly reproduce"):
            bridge.predict(replace(bundle, audio=replace(bundle.audio, embeddings=changed)))
        self.assertEqual(calls, [])

    def test_parity_bridge_rejects_synthetic_unknown_identity_and_partial_validity(self):
        bridge, bundle, calls = self.bridge_fixture()
        for change in ({"synthetic": True}, {"decoded": replace(bundle.decoded, audio_sample_ids=("unknown", "b"))}):
            with self.assertRaises(IntegrationError):
                bridge.predict(replace(bundle, **change))
        mask = bundle.audio.validity.clone()
        mask[0, 0] = False
        with self.assertRaises(IntegrationError):
            bridge.predict(replace(bundle, audio=replace(bundle.audio, validity=mask)))
        self.assertEqual(calls, [])

    def test_feature_only_bridge_cannot_claim_raw_readiness(self):
        bridge, bundle, calls = self.bridge_fixture()
        object.__setattr__(bridge, "expected_audio", None)
        object.__setattr__(bridge, "expected_visual", None)
        with self.assertRaisesRegex(IntegrationError, "feature replay only"):
            bridge.predict(bundle)
        self.assertEqual(calls, [])

    @contextmanager
    def production_admission_fixture(self):
        """Control-flow fixture only: patched methods do NOT prove encoder execution.

        Use the concrete decoder and actual parity comparison, but no trained
        encoder or fusion forward. Success must therefore remain conservative.
        """
        bridge, bundle, _ = self.bridge_fixture()
        self.media = bundle.decoded
        calls, events = [], []
        audio, visual = object.__new__(FrozenEncoder), object.__new__(FrozenEncoder)
        audio.provenance, visual.provenance = bundle.audio.provenance, bundle.visual.provenance
        bridge.adapters.update({k: object.__new__(FrozenFusionInference) for k in ("E3", "A2")})
        def ready(encoder):
            events.append("ready-" + encoder.provenance.modality)
        def encode(encoder, decoded):
            events.append(encoder.provenance.modality)
            return bundle.audio if encoder.provenance.modality == "audio" else bundle.visual
        def predict(adapter, features):
            calls.append(features)
            return StubPrediction("control-flow-fixture", .5)
        with patch.object(FrozenEncoder, "require_ready", new=ready), \
             patch.object(FrozenEncoder, "encode", new=encode), \
             patch.object(FrozenFusionInference, "predict", new=predict):
            pipeline = MediaFusionPipeline(DecodedTensorDecoder("d"*64), audio, visual, bridge)
            yield pipeline, bridge, calls, events

    def test_concrete_admission_path_uses_real_gate_and_conservative_status(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            result = pipeline.run(self.source())
            self.assertEqual(result["status"], "TRAIN_EXPORT_PARITY_MATCHED")
            self.assertNotEqual(result["status"], "TRAIN_RAW_PARITY_VERIFIED")
            self.assertEqual(events, ["ready-audio", "ready-vision", "audio", "vision"])
            self.assertEqual(len(calls), 2)
            self.assertIs(calls[0], calls[1])
            self.assertTrue(torch.equal(calls[0].audio, self.fusion.bundles[-1].audio.embeddings))

    def test_synthetic_component_inserted_in_real_mode_rejected(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            for component in pipeline._components():
                object.__setattr__(component, "is_synthetic", True)
                self.assertEqual(pipeline.try_run(None)["status"], "REJECTED")
                object.__setattr__(component, "is_synthetic", False)
            self.assertEqual(events, [])
            self.assertEqual(calls, [])
            with self.assertRaisesRegex(IntegrationError, "synthetic"):
                MediaFusionPipeline(self.decoder, pipeline.audio_encoder, pipeline.visual_encoder, bridge)

    def test_declared_real_parity_stub_cannot_replace_concrete_gate(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            forged = SimpleNamespace(is_synthetic=False, predict=lambda b: {"verified": True})
            with self.assertRaisesRegex(IntegrationError, "concrete"):
                MediaFusionPipeline(pipeline.decoder, pipeline.audio_encoder, pipeline.visual_encoder, forged)
            self.assertEqual(calls, [])
            self.assertEqual(events, [])

    def test_production_method_and_fusion_adapter_replacements_rejected(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            for component, method in ((pipeline.decoder, "decode"), (pipeline.audio_encoder, "encode"),
                    (pipeline.visual_encoder, "require_ready"), (bridge, "predict"), (bridge, "predict_features"),
                    (bridge.adapters["E3"], "predict")):
                if component is bridge:
                    # Frozen/slotted bridge forbids method shadowing as well.
                    with self.assertRaises((FrozenInstanceError, TypeError)):
                        setattr(component, method, lambda *args: {"verified": True})
                    continue
                setattr(component, method, lambda *args: {"verified": True})
                result = pipeline.try_run(None)
                self.assertEqual(result["status"], "REJECTED")
                self.assertEqual(result["error"]["code"], "COMPONENT_MUTATED")
                delattr(component, method)
            bridge.adapters["A2"] = SimpleNamespace(predict=lambda *a: {"verified": True})
            self.assertEqual(pipeline.try_run(None)["status"], "REJECTED")
            self.assertEqual(calls, [])
            self.assertEqual(events, [])

    def test_production_gate_rechecks_producer_binding_each_run(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            with self.assertRaises(FrozenInstanceError):
                bridge.expected_audio = replace(bridge.expected_audio, configuration_sha256="e"*64)
            object.__setattr__(bridge, "expected_audio", replace(bridge.expected_audio, configuration_sha256="e"*64))
            result = pipeline.try_run(None)
            self.assertEqual(result["status"], "REJECTED")
            self.assertEqual(result["error"]["code"], "PROVENANCE_MISMATCH")
            self.assertEqual(calls, [])
            self.assertEqual(events, [])

    def test_concrete_gate_failure_after_success_has_no_stale_predictions(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            self.assertEqual(pipeline.run(self.source())["status"], "TRAIN_EXPORT_PARITY_MATCHED")
            result = pipeline.try_run(self.source({"audio_sample_ids": ["unknown", "b"]}))
            self.assertEqual(result["status"], "REJECTED")
            self.assertNotIn("predictions", result)
            self.assertEqual(len(calls), 2)
            self.assertEqual(pipeline.run(self.source())["status"], "TRAIN_EXPORT_PARITY_MATCHED")
            self.assertEqual(len(calls), 4)

    def substitution_case(self, key):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            output = getattr(self.fusion.bundles[-1], key)
            output.embeddings[0, 0, 0] += 1
            source = getattr(bridge, key).splits["train"]
            forged = replace(getattr(bridge, key), splits={"train": replace(source,
                embeddings=output.embeddings.numpy().astype(np.float16))})
            self.assertEqual(pipeline.try_run(self.source())["status"], "REJECTED")
            self.assertEqual(calls, [])
            with self.assertRaises(FrozenInstanceError):
                setattr(bridge, key, forged)
            result = pipeline.try_run(self.source())
            self.assertEqual(result["status"], "REJECTED")
            self.assertEqual(result["error"]["code"], "PARITY_UNVERIFIED")
            self.assertNotIn("predictions", result)
            self.assertEqual(calls, [])
            # Even reflection replacing only the reference cannot bypass admission.
            object.__setattr__(bridge, key, forged)
            result = pipeline.try_run(self.source())
            self.assertEqual(result["status"], "REJECTED")
            self.assertEqual(result["error"]["code"], "REFERENCE_INTEGRITY")
            self.assertEqual(calls, [])

    def test_audio_reference_substitution_rejected_before_both_models(self):
        self.substitution_case("audio")

    def test_visual_reference_substitution_rejected_before_both_models(self):
        self.substitution_case("visual")

    def inplace_reference_case(self, key):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            source = getattr(bridge, key).splits["train"]
            with self.assertRaises(ValueError):
                source.embeddings[0, 0, 0] = 123
            with self.assertRaises(ValueError):
                source.embeddings.setflags(write=True)
            # Independently test content validation if a writable nested array is
            # introduced using reflection, without rewriting the source file.
            writable = source.embeddings.copy()
            object.__setattr__(source, "embeddings", writable)
            self.assertEqual(pipeline.run(self.source())["status"], "TRAIN_EXPORT_PARITY_MATCHED")
            calls.clear()
            writable[0, 0, 0] += 1
            result = pipeline.try_run(self.source())
            self.assertEqual(result["status"], "REJECTED")
            self.assertEqual(result["error"]["code"], "REFERENCE_INTEGRITY")
            self.assertEqual(calls, [])

    def test_inplace_audio_reference_mutation_detected(self):
        self.inplace_reference_case("audio")

    def test_inplace_visual_reference_mutation_detected(self):
        self.inplace_reference_case("visual")

    def source_hash_spoof_case(self, key):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            cached = getattr(bridge, key)
            path = cached.root / "train_emb.npy"
            admitted = dict(bridge.provenance[0 if key == "audio" else 1].export_hashes)[path.name]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), admitted)
            payload = bytearray(path.read_bytes())
            payload[-2:] = np.float16(123).tobytes()
            # Only temporary synthetic fixture bytes change; preserve mmap size.
            with path.open("r+b") as stream:
                stream.write(payload)
            claimed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertNotEqual(claimed, admitted)
            cached.provenance["files"][path.name]["sha256"] = claimed
            spoof = tuple(replace(p, export_hashes=tuple((n, claimed if n == path.name else sha)
                for n, sha in p.export_hashes)) for p in bridge.provenance)
            with self.assertRaises(FrozenInstanceError):
                bridge.provenance = spoof
            with self.assertRaises(FrozenInstanceError):
                bridge._references = tuple(replace(r, content_sha256=claimed) for r in bridge._references)
            result = pipeline.try_run(self.source())
            self.assertEqual(result["status"], "REJECTED")
            self.assertEqual(result["error"]["code"], "HASH_MISMATCH")
            self.assertEqual(calls, [])

    def test_recomputed_audio_hash_cannot_replace_pinned_file_hash(self):
        self.source_hash_spoof_case("audio")

    def test_recomputed_visual_hash_cannot_replace_pinned_file_hash(self):
        self.source_hash_spoof_case("visual")

    def test_swapped_reference_rows_with_valid_shapes_rejected(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            source = bridge.audio.splits["train"]
            swapped = source.embeddings[::-1].copy()
            self.assertEqual(swapped.shape, source.embeddings.shape)
            object.__setattr__(source, "embeddings", swapped)
            self.assertEqual(pipeline.try_run(self.source())["status"], "REJECTED")
            self.assertEqual(calls, [])

    def test_changed_reference_source_identity_rejected(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            source = bridge.visual.splits["train"]
            changed = (replace(source.records[0], sample_id="z__00", clip_id="z"), source.records[1])
            object.__setattr__(source, "records", changed)
            object.__setattr__(bridge.visual, "metadata", {"train": changed})
            self.assertEqual(pipeline.try_run(self.source())["status"], "REJECTED")
            self.assertEqual(calls, [])

    def test_changed_reference_producer_version_rejected(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            spoof = (replace(bridge.provenance[0], run_identity="different-producer"), bridge.provenance[1])
            with self.assertRaises(FrozenInstanceError):
                bridge.provenance = spoof
            object.__setattr__(bridge, "provenance", spoof)
            result = pipeline.try_run(self.source())
            self.assertEqual(result["status"], "REJECTED")
            self.assertEqual(result["error"]["code"], "REFERENCE_INTEGRITY")
            self.assertEqual(calls, [])

    def test_reference_metadata_immutability_and_nested_content_check(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            with self.assertRaises(FrozenInstanceError):
                bridge.audio.splits["train"].records[0].sample_id = "changed"
            with self.assertRaises(TypeError):
                bridge.audio.splits["train"] = bridge.visual.splits["train"]
            with self.assertRaises(FrozenInstanceError):
                bridge._references[0].content_sha256 = "a"*64
            bridge.visual.provenance["source"] = "changed nested declaration"
            self.assertEqual(pipeline.try_run(self.source())["status"], "REJECTED")
            self.assertEqual(calls, [])

    def test_valid_archived_fixture_checks_independent_source_hashes(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            for reference, producer in zip((bridge.audio, bridge.visual), bridge.provenance):
                for name, expected in producer.export_hashes:
                    self.assertEqual(hashlib.sha256((reference.root/name).read_bytes()).hexdigest(), expected)
            self.assertEqual(pipeline.run(self.source())["status"], "TRAIN_EXPORT_PARITY_MATCHED")
            self.assertEqual(len(calls), 2)
            self.assertIs(calls[0], calls[1])

    def test_reference_failure_then_valid_admission_does_not_cache_failure(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            output = self.fusion.bundles[-1].audio.embeddings
            original = output.clone()
            output[0, 0, 0] += 1
            self.assertEqual(pipeline.try_run(self.source())["status"], "REJECTED")
            self.assertEqual(calls, [])
            output.copy_(original)
            self.assertEqual(pipeline.run(self.source())["status"], "TRAIN_EXPORT_PARITY_MATCHED")
            self.assertEqual(len(calls), 2)

    def test_reference_success_then_nested_mutation_cannot_reuse_admission(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            self.assertEqual(pipeline.run(self.source())["status"], "TRAIN_EXPORT_PARITY_MATCHED")
            calls.clear()
            original = bridge.audio.provenance["source"]
            bridge.audio.provenance["source"] = "altered after success"
            result = pipeline.try_run(self.source())
            self.assertEqual(result["status"], "REJECTED")
            self.assertNotIn("predictions", result)
            self.assertEqual(calls, [])
            bridge.audio.provenance["source"] = original
            self.assertEqual(pipeline.run(self.source())["status"], "TRAIN_EXPORT_PARITY_MATCHED")
            self.assertEqual(len(calls), 2)

    def test_synthetic_replay_of_file_backed_fixture_remains_synthetic(self):
        bridge, bundle, calls = self.bridge_fixture()
        self.audio.output_change = lambda x: replace(x, embeddings=torch.from_numpy(
            bridge.audio.splits["train"].embeddings.astype(np.float32)))
        self.visual.output_change = lambda x: replace(x, embeddings=torch.from_numpy(
            bridge.visual.splits["train"].embeddings.astype(np.float32)))
        self.assertEqual(self.pipeline.run(None)["status"], "SYNTHETIC")
        self.assertEqual(calls, [])

    def test_missing_authenticated_reference_binding_blocks_before_models(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            object.__setattr__(bridge, "_references", ())
            result = pipeline.try_run(self.source())
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(calls, [])

    def test_both_reference_integrities_rechecked_after_input_validation(self):
        with self.production_admission_fixture() as (pipeline, bridge, calls, events):
            original = EncodedBundle.validate
            validations = []
            def mutate_after_validation(bundle, *args):
                result = original(bundle, *args)
                validations.append(1)
                if len(validations) == 2:  # inside raw parity, after the first integrity check
                    bridge.visual.provenance["source"] = "changed during comparison"
                return result
            with patch.object(EncodedBundle, "validate", new=mutate_after_validation):
                result = pipeline.try_run(self.source())
            self.assertEqual(result["status"], "REJECTED")
            self.assertEqual(len(validations), 2)
            self.assertEqual(calls, [])

    def test_safe_checkpoint_reader_is_weights_only_and_rejects_corruption(self):
        from fusion.encoder_interfaces import _safe_payload
        path = self.root/"synthetic.pt"
        torch.save({"config": {}, "model": {"weight": torch.ones(1)}}, path)
        artifact = Artifact(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
        original = torch.load
        with patch("torch.load", wraps=original) as load:
            _safe_payload(artifact)
        self.assertTrue(load.call_args.kwargs["weights_only"])
        corrupt = self.artifact("corrupt.pt", b"not a checkpoint")
        with self.assertRaises(IntegrationError):
            _safe_payload(corrupt)

    def test_strict_encoder_state_rejects_missing_shape_dtype_and_nonfinite(self):
        from fusion.encoder_interfaces import _strict_state
        model = torch.nn.Linear(2, 1)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        for change in ({}, {**original, "weight": torch.zeros(1, 3)},
                       {**original, "weight": original["weight"].double()},
                       {**original, "weight": torch.full((1, 2), float('nan'))}):
            with self.assertRaises(IntegrationError):
                _strict_state(model, change)
        self.assertTrue(all(torch.equal(original[k], v) for k, v in model.state_dict().items()))

    def test_specs_require_known_hashes_complete_versions_and_canonical_config(self):
        for value in (replace(self.spec(), configuration_json='{ "bad": true }'),
                      replace(self.spec(), runtime_versions=()), replace(self.spec(), artifacts=(("base", Artifact("a", "unknown")),))):
            with self.assertRaises(IntegrationError):
                value.validate()

    def test_malformed_numpy_source_does_not_fabricate_content(self):
        source = self.source()
        corrupt = self.artifact("malformed.npy", b"not a numpy file")
        with self.assertRaises(IntegrationError):
            DecodedTensorDecoder("d"*64).decode(replace(source, waveform=corrupt))
