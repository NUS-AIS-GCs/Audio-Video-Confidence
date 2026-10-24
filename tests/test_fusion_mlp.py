"""Offline lightweight Phase 4 checks; synthetic tensors are not experiments."""
from dataclasses import FrozenInstanceError, fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fusion.contracts import ContractError
from fusion.artifact_integration import manifest_to_json
from fusion.embedding_data import EmbeddingDataset, EmbeddingFeatures
from fusion.mlp_model import LateFusionMLP, MLPConfig
from fusion.pair_contracts import PairType, SamplingPolicy
from fusion.pair_generator import GenerationConfig, generate_pairs
from fusion.training_contracts import (
    CheckpointMetadata, OptimizerConfig, TrainingConfig, backward_checked,
    build_optimizer, correspondence_loss, make_loader, move_batch,
    require_primary_dataset, resolve_device, seed_everything,
    validate_split_isolation, validate_targets,
)
from fusion.smoke_mlp import run_smoke
from fusion.artifact_integration import run_integration
from test_fusion_artifact_integration import write_fixture
from test_fusion_pair_generator import ROWS, bundles, hard_policy, record


def manifest(audio, vision, split="train", **kwargs):
    return generate_pairs(audio, vision, policy=SamplingPolicy(42),
                          config=GenerationConfig(split, easy_negative_budget=7,
                                                  challenge_budget=3, **kwargs))


def small_model(**kwargs):
    return LateFusionMLP(MLPConfig(hidden_dims=(16, 8), dropout=0, **kwargs))


class EmbeddingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.audio, self.vision = bundles()
        self.manifest = manifest(self.audio, self.vision)

    def dataset(self, **kwargs):
        return EmbeddingDataset(self.manifest, self.audio, self.vision, split="train", **kwargs)

    def test_original_row_references_shuffled_modalities_and_binary_targets(self):
        data = self.dataset()
        self.assertTrue(any(p.audio.row_index != p.visual.row_index for p in data.pairs))
        self.assertEqual(data.features.audio.shape, (16, 10, 256))
        self.assertEqual(data.features.visual.shape, (16, 10, 512))
        self.assertEqual(data.targets.shape, (16, 1))
        for i, pair in enumerate(data.pairs):
            features, target = data[i]
            np.testing.assert_array_equal(features["audio"], self.audio.splits["train"].embeddings[pair.audio.row_index])
            np.testing.assert_array_equal(features["visual"], self.vision.splits["train"].embeddings[pair.visual.row_index])
            self.assertEqual(target.item(), pair.original_pair_target)
        self.assertEqual(set(data.targets[:, 0]), {0, 1})

    def test_complete_primary_and_challenge_separation(self):
        primary, challenge = self.dataset(), self.dataset(partition="challenge")
        self.assertEqual(primary.pairs, self.manifest.primary_pairs)
        self.assertEqual(challenge.pairs, self.manifest.challenge_pairs)
        self.assertTrue(all(p.pair_type is PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE for p in challenge.pairs))
        self.assertFalse(set(primary.pairs) & set(challenge.pairs))
        self.assertTrue((challenge.targets == 0).all())
        with self.assertRaises(ContractError):
            require_primary_dataset(challenge, "train")

    def test_empty_annotation_keeps_ten_bins_and_positive_target(self):
        data = self.dataset()
        i = next(i for i, p in enumerate(data.pairs) if p.audio.record.end == 0
                 and p.pair_type is PairType.ORIGINAL_POSITIVE)
        features, target = data[i]
        self.assertEqual(target.item(), 1)
        self.assertTrue(features["validity"].all())
        self.assertEqual(features["audio"].shape, (10, 256))
        self.assertEqual(data.pairs[i].semantic_compatibility.value, "UNKNOWN")

    def test_feature_target_metadata_isolation(self):
        data = self.dataset()
        self.assertEqual([f.name for f in fields(EmbeddingFeatures)], ["audio", "visual"])
        features, target = data[0]
        self.assertEqual(set(features), {"audio", "visual", "validity"})
        self.assertEqual(target.shape, (1,))
        with self.assertRaises(FrozenInstanceError):
            data.pairs[0].pair_type = PairType.CROSS_CLASS_EASY_NEGATIVE
        with self.assertRaises(AttributeError):
            data.pairs = ()
        with self.assertRaises(FrozenInstanceError):
            data.provenance.split = "test"

    def test_no_mutation_and_snapshot_independence(self):
        before = self.audio.splits["train"].embeddings.copy()
        data = self.dataset()
        features, target = data[0]
        features["audio"].zero_()
        target.zero_()
        np.testing.assert_array_equal(before, self.audio.splits["train"].embeddings)
        self.assertFalse(data.features.audio.flags.writeable)
        with self.assertRaises(ValueError):
            data.targets.flags.writeable = True
        self.audio.splits["train"].embeddings[:] = 0
        self.assertTrue(data[0][0]["audio"].any())

    def test_mask_uses_explicit_validity_without_annotations(self):
        valid = np.ones((9, 10), dtype=bool)
        valid[:, 0] = False
        self.audio = replace(self.audio, splits={"train": replace(self.audio.splits["train"], input_validity=valid)})
        data = self.dataset()
        self.assertFalse(data.validity[:, 0].any())
        self.assertTrue(data.validity[:, 1:].all())
        valid[:] = False
        with self.assertRaises(ContractError):
            self.dataset()

    def test_dimensions_nan_infinity_and_bad_dtype_fail(self):
        for shape, dtype, bad in (((9, 9, 512), np.float16, None),
                                  ((9, 10, 128), np.float16, None),
                                  ((9, 10, 512), np.float16, np.nan),
                                  ((9, 10, 512), np.float16, np.inf),
                                  ((9, 10, 512), np.int32, None)):
            with self.subTest(shape=shape, bad=bad), self.assertRaises(ContractError):
                emb = np.zeros(shape, dtype=dtype)
                if bad is not None:
                    emb[0, 0, 0] = bad
                vision = replace(self.vision, splits={"train": replace(self.vision.splits["train"], embeddings=emb)})
                EmbeddingDataset(self.manifest, self.audio, vision, split="train")

    def test_audio_dimensions_and_nonfinite_values_fail(self):
        for shape, bad in (((9, 10, 255), None), ((9, 10, 256), np.nan),
                           ((9, 10, 256), -np.inf)):
            value = np.zeros(shape, dtype=np.float16)
            if bad is not None:
                value[0, 0, 0] = bad
            audio = replace(self.audio, splits={"train": replace(self.audio.splits["train"], embeddings=value)})
            with self.assertRaises(ContractError):
                EmbeddingDataset(self.manifest, audio, self.vision, split="train")

    def test_finite_float64_visual_values_cannot_overflow_float32_minibatches(self):
        original = self.vision.splits["train"].embeddings.astype(np.float64)
        for bad in (np.finfo(np.float64).max, -np.finfo(np.float64).max):
            with self.subTest(value=bad):
                values = original.copy()
                values[0, 0, 0] = bad
                vision = replace(self.vision, splits={"train": replace(
                    self.vision.splits["train"], embeddings=values)})
                self.assertTrue(np.isfinite(values).all())
                before = values.copy()
                with self.assertRaisesRegex(ContractError, "float32 minibatch range"):
                    EmbeddingDataset(self.manifest, self.audio, vision, split="train")
                np.testing.assert_array_equal(values, before)
        vision = replace(self.vision, splits={"train": replace(
            self.vision.splits["train"], embeddings=original)})
        data = EmbeddingDataset(self.manifest, self.audio, vision, split="train")
        features, _ = data[0]
        self.assertEqual(features["visual"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(features["visual"]).all())
        np.testing.assert_array_equal(features["visual"], original[data.pairs[0].visual.row_index])

    def test_bad_split_modality_partition_and_limits_fail(self):
        for kwargs in ({"partition": "mixed"}, {"limit": True}, {"limit": -1}, {"limit": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                self.dataset(**kwargs)
        with self.assertRaises(ContractError):
            EmbeddingDataset(self.manifest, self.audio, self.vision, split="val")
        with self.assertRaises(ContractError):
            EmbeddingDataset(self.manifest, self.vision, self.audio, split="train")

    def test_full_coverage_and_partial_prefix_guards(self):
        truncated = manifest(self.audio, self.vision, positive_budget=8)
        with self.assertRaises(ContractError):
            EmbeddingDataset(truncated, self.audio, self.vision, split="train")
        partial = self.dataset(limit=3)
        self.assertEqual(partial.pairs, self.manifest.primary_pairs[:3])
        with self.assertRaises(ContractError):
            require_primary_dataset(partial, "train")
        self.assertEqual(self.dataset(limit=0).features.audio.shape, (0, 10, 256))

    def test_manifest_record_row_and_partition_tampering_fail(self):
        pair = self.manifest.primary_pairs[0]
        altered = replace(pair, audio=replace(pair.audio, row_index=8))
        variants = (replace(self.manifest, primary_pairs=(altered,) + self.manifest.primary_pairs[1:]),
                    replace(self.manifest, primary_pairs=self.manifest.primary_pairs[:-1]),
                    replace(self.manifest, primary_pairs=self.manifest.primary_pairs + self.manifest.challenge_pairs))
        for value in variants:
            with self.assertRaises(ContractError):
                EmbeddingDataset(value, self.audio, self.vision, split="train")

    def test_safe_serialization_artifact_context_and_fingerprints(self):
        context = "a" * 64
        serialized = manifest_to_json(self.manifest, self.audio, self.vision, artifact_fingerprint=context)
        data = EmbeddingDataset.from_json(serialized, self.audio, self.vision,
                                          split="train", artifact_fingerprint=context)
        self.assertEqual(data.provenance.artifact_fingerprint, context)
        self.assertEqual(data.provenance.source_array_hashes, self.dataset().provenance.source_array_hashes)
        with self.assertRaises(ContractError):
            EmbeddingDataset.from_json(serialized, self.audio, self.vision, split="train")
        broken = json.loads(serialized)
        broken["primary_pairs"][0]["original_pair_target"] = 0
        with self.assertRaises(ContractError):
            EmbeddingDataset.from_json(json.dumps(broken), self.audio, self.vision,
                                       split="train", artifact_fingerprint=context)

    def test_hard_policy_excluded_from_phase4(self):
        value = generate_pairs(self.audio, self.vision, policy=SamplingPolicy(42, hard_policy()),
                               config=GenerationConfig("train", enable_hard_negatives=True, hard_negative_budget=2))
        with self.assertRaises(ContractError):
            EmbeddingDataset(value, self.audio, self.vision, split="train")


class ModelTests(unittest.TestCase):
    def test_default_architecture_parameter_count(self):
        model = LateFusionMLP()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 558081)
        self.assertFalse(any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()))

    def test_variable_batch_shapes_probabilities_and_bin_representations(self):
        model = small_model().eval()
        for size in (1, 3, 7):
            output = model(torch.randn(size, 10, 256), torch.randn(size, 10, 512),
                           return_probabilities=True, return_bin_representations=True)
            self.assertEqual(output.logits.shape, (size, 1))
            self.assertEqual(output.bin_representations.shape, (size, 10, 8))
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertTrue(((output.probabilities >= 0) & (output.probabilities <= 1)).all())
            torch.testing.assert_close(output.probabilities, output.logits.sigmoid())
        output = model(torch.zeros(1, 10, 256), torch.zeros(1, 10, 512))
        self.assertIsNone(output.probabilities)
        self.assertIsNone(output.bin_representations)

    def test_deterministic_initialization_preserves_rng(self):
        before = torch.get_rng_state().clone()
        first, second = small_model(), small_model()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))
        other = small_model(initialization_seed=43)
        self.assertFalse(torch.equal(next(first.parameters()), next(other.parameters())))

    def test_model_does_not_mutate_feature_tensors_or_masks(self):
        audio, visual = torch.randn(2, 10, 256), torch.randn(2, 10, 512)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[:, 8:] = False
        before = (audio.clone(), visual.clone(), mask.clone())
        small_model()(audio, visual, mask)
        for actual, expected in zip((audio, visual, mask), before):
            self.assertTrue(torch.equal(actual, expected))

    def test_masked_mean_invariance_and_zero_invalid_gradients(self):
        model = small_model().eval()
        audio = torch.randn(2, 10, 256, requires_grad=True)
        visual = torch.randn(2, 10, 512, requires_grad=True)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[:, 5:] = False
        output = model(audio, visual, mask, return_bin_representations=True)
        changed = audio.detach().clone()
        changed[:, 5:] = torch.finfo(torch.float32).max
        torch.testing.assert_close(output.logits, model(changed, visual, mask).logits)
        self.assertTrue((output.bin_representations[:, 5:] == 0).all())
        pooled = output.bin_representations[:, :5].mean(dim=1)
        torch.testing.assert_close(output.logits, model.correspondence_head(pooled))
        output.logits.sum().backward()
        self.assertTrue((audio.grad[:, 5:] == 0).all())
        self.assertTrue((visual.grad[:, 5:] == 0).all())
        self.assertTrue((audio.grad[:, :5] != 0).any())

    def test_shape_dtype_nonfinite_and_mask_guards(self):
        model = small_model()
        audio, visual = torch.zeros(2, 10, 256), torch.zeros(2, 10, 512)
        for a, v, mask in ((audio[:, :9], visual, None), (audio, visual[:1], None),
                           (audio.double(), visual, None), (audio[:0], visual[:0], None),
                           (audio, visual, torch.zeros(2, 10, dtype=torch.bool)),
                           (audio, visual, torch.ones(2, 10)),
                           (audio, visual, torch.ones(2, 9, dtype=torch.bool))):
            with self.assertRaises(ContractError):
                model(a, v, mask)
        for bad in (float("nan"), float("inf")):
            broken = audio.clone()
            broken[0, 0, 0] = bad
            with self.assertRaises(ContractError):
                model(broken, visual)

    def test_invalid_architecture_configs_fail(self):
        for kwargs in ({"hidden_dims": []}, {"hidden_dims": (True,)}, {"hidden_dims": (0,)},
                       {"dropout": float("nan")}, {"dropout": 1}, {"dropout": True},
                       {"activation": "unknown"}, {"initialization_seed": -1},
                       {"audio_dim": 512}, {"temporal_bins": True}, {"aggregation": "event_mean"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                MLPConfig(**kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_backward(self):
        model = small_model().cuda()
        output = model(torch.randn(3, 10, 256, device="cuda"), torch.randn(3, 10, 512, device="cuda"))
        loss = correspondence_loss(output.logits, torch.tensor([[1.], [0.], [1.]], device="cuda"), TrainingConfig())
        backward_checked(loss, model, max_gradient_norm=5)


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        # The shared clip has different annotations across splits: documented
        # source overlap remains visible while annotation identities are disjoint.
        metadata = {"train": ROWS, "val": (record("a", 4), record("val2", 5), record("val3", 5))}
        self.audio, self.vision = bundles(metadata)
        self.train = EmbeddingDataset(manifest(self.audio, self.vision), self.audio, self.vision, split="train")
        val = generate_pairs(self.audio, self.vision, policy=SamplingPolicy(42),
                             config=GenerationConfig("val", easy_negative_budget=3, challenge_budget=2))
        self.val = EmbeddingDataset(val, self.audio, self.vision, split="val")

    def test_seed_numpy_torch_and_loader_order_reproducibility(self):
        config = TrainingConfig(batch_size=3)
        seed_everything(config)
        first = (np.random.rand(3), torch.rand(3))
        seed_everything(config)
        second = (np.random.rand(3), torch.rand(3))
        np.testing.assert_array_equal(first[0], second[0])
        self.assertTrue(torch.equal(first[1], second[1]))
        a = list(make_loader(self.train, config, role="train"))
        b = list(make_loader(self.train, config, role="train"))
        self.assertEqual(a[-1][1].shape, (1, 1))
        for x, y in zip(a, b):
            torch.testing.assert_close(x[0]["audio"], y[0]["audio"])
            torch.testing.assert_close(x[1], y[1])

    def test_cpu_minibatch_forward_backward_optimizer_integration(self):
        config, model = TrainingConfig(batch_size=3), small_model()
        optimizer = build_optimizer(model, config.optimizer)
        before = next(model.parameters()).detach().clone()
        for batch in make_loader(self.train, config, role="train"):
            features, targets = move_batch(batch, resolve_device(config))
            optimizer.zero_grad(set_to_none=True)
            loss = correspondence_loss(model(**features).logits, targets, config)
            norm = backward_checked(loss, model, max_gradient_norm=config.max_gradient_norm)
            self.assertTrue(np.isfinite(norm))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            break  # One synthetic minibatch, no training experiment.
        self.assertFalse(torch.equal(before, next(model.parameters())))

    def test_bce_loss_stability_weighting_and_single_class_batches(self):
        logits = torch.tensor([[1000.], [-1000.], [0.]], requires_grad=True)
        targets = torch.tensor([[0.], [1.], [1.]])
        for config in (TrainingConfig(), TrainingConfig(class_balance="explicit_positive_weight", positive_weight=2)):
            expected = torch.nn.BCEWithLogitsLoss(pos_weight=None if config.positive_weight is None else torch.tensor([2.]))
            torch.testing.assert_close(correspondence_loss(logits, targets, config), expected(logits, targets))
        self.assertTrue(torch.isfinite(correspondence_loss(logits[:1], targets[:1], TrainingConfig())))

    def test_invalid_targets_and_logits_fail(self):
        for values in (torch.tensor([0., 1.]), torch.tensor([[0], [1]]), torch.tensor([[0.], [.5]]),
                       torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
                       torch.tensor([[True]]), torch.zeros(0, 1), torch.ones(1, 1, requires_grad=True)):
            with self.assertRaises(ContractError):
                validate_targets(values)
        with self.assertRaises(ContractError):
            correspondence_loss(torch.zeros(2, 1), torch.zeros(1, 1), TrainingConfig())
        with self.assertRaises(ContractError):
            correspondence_loss(torch.full((1, 1), float("inf")), torch.zeros(1, 1), TrainingConfig())

    def test_invalid_gradient_and_loss_clear_gradients(self):
        model = small_model()
        with self.assertRaises(ContractError):
            backward_checked(torch.tensor(float("nan")), model, max_gradient_norm=5)
        hook = next(model.parameters()).register_hook(lambda grad: grad * float("nan"))
        loss = model(torch.randn(1, 10, 256), torch.randn(1, 10, 512)).logits.sum()
        with self.assertRaises(ContractError):
            backward_checked(loss, model, max_gradient_norm=5)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        hook.remove()

    def test_split_isolation_and_official_source_overlap_reporting(self):
        self.assertEqual(validate_split_isolation(self.train, self.val), ("a",))
        with self.assertRaises(ContractError):
            validate_split_isolation(self.train, self.train)
        for role in ("val", "test"):
            with self.assertRaises(ContractError):
                make_loader(self.train, TrainingConfig(), role=role)
        # A mocked malformed object demonstrates the identity guard independently
        # of Phase 2, which itself rejects cross-split annotation duplication.
        with patch.object(type(self.val), "pairs", new_callable=unittest.mock.PropertyMock,
                          return_value=self.train.pairs):
            with self.assertRaises(ContractError):
                validate_split_isolation(self.train, self.val)

    def test_positive_only_full_dataset_rejected(self):
        positive = generate_pairs(self.audio, self.vision, policy=SamplingPolicy(42), config=GenerationConfig("train"))
        data = EmbeddingDataset(positive, self.audio, self.vision, split="train")
        with self.assertRaises(ContractError):
            make_loader(data, TrainingConfig(), role="train")

    def test_device_unavailability_and_feature_schema_guard(self):
        with patch("torch.cuda.is_available", return_value=False), self.assertRaises(ContractError):
            resolve_device(TrainingConfig(device="cuda"))
        features, targets = next(iter(make_loader(self.train, TrainingConfig(), role="train")))
        features["category"] = torch.zeros(len(targets))
        with self.assertRaises(ContractError):
            move_batch((features, targets), "cpu")

    def test_strict_training_and_optimizer_configuration(self):
        for kwargs in ({"seed": True}, {"seed": 2**32}, {"batch_size": 0}, {"epochs": -1},
                       {"num_workers": .5}, {"device": "auto"}, {"deterministic": 1},
                       {"positive_weight": 2}, {"class_balance": "explicit_positive_weight"},
                       {"max_gradient_norm": float("inf")}, {"selection_split": "test"},
                       {"selection_partition": "challenge"}, {"training_split": "val"},
                       {"wandb_base_url": "https://api.wandb.ai"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                TrainingConfig(**kwargs)
        for kwargs in ({"name": "sgd"}, {"learning_rate": 0}, {"weight_decay": -1},
                       {"betas": (.9, 1)}, {"epsilon": float("nan")}):
            with self.assertRaises(ContractError):
                OptimizerConfig(**kwargs)
        with self.assertRaises(TypeError):
            TrainingConfig(unknown=True)

    def checkpoint(self, **kwargs):
        return CheckpointMetadata(model=MLPConfig(), training=TrainingConfig(), git_commit="a" * 40,
                                  code_sha256="b" * 64, data=(self.train.provenance, self.val.provenance),
                                  export_identities=(("audio", "synthetic-audio"), ("vision", "synthetic-vision")),
                                  software_versions=(("python", "synthetic"), ("numpy", np.__version__),
                                                     ("torch", torch.__version__)),
                                  source_overlap=("a",), **kwargs)

    def test_checkpoint_metadata_json_roundtrip_and_fingerprint(self):
        metadata = self.checkpoint()
        value = json.loads(json.dumps(metadata.to_dict()))
        result = CheckpointMetadata.from_dict(value)
        self.assertEqual(result, metadata)
        later = self.checkpoint(epoch=1, optimizer_steps=1, optimizer_state_present=True,
                                selected_epoch=1, validation_metrics=(("loss", .7), ("roc_auc", .6)))
        self.assertEqual(metadata.experiment_fingerprint, later.experiment_fingerprint)
        changed = replace(metadata, training=TrainingConfig(seed=43))
        self.assertNotEqual(metadata.experiment_fingerprint, changed.experiment_fingerprint)
        value["training"]["seed"] = 99
        with self.assertRaises(ContractError):
            CheckpointMetadata.from_dict(value)

    def test_checkpoint_invalid_configuration_and_evidence_fail(self):
        valid = self.checkpoint()
        for changes in ({"git_commit": "HEAD"}, {"code_sha256": "bad"}, {"epoch": -1},
                        {"epoch": 21}, {"selected_epoch": 1}, {"optimizer_steps": 1},
                        {"epoch": 1, "selected_epoch": 1},
                        {"validation_metrics": (("loss", float("nan")),)},
                        {"data": (self.val.provenance, self.train.provenance)},
                        {"data": (replace(self.train.provenance, source_array_hashes=()), self.val.provenance)},
                        {"data": (replace(self.train.provenance, source_array_hashes=("bad",)), self.val.provenance)},
                        {"validation_metrics": ("bad",)},
                        {"export_identities": ()}, {"software_versions": ()}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(valid, **changes)


class SmokeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio, self.vision, self.official, self.counts = write_fixture(self.root)
        self.integration = self.root / "integration"
        run_integration(self.audio, self.vision, self.official,
                        output_dir=self.integration, expected_counts=self.counts)

    def run_smoke(self):
        return run_smoke(self.audio, self.vision, self.official, self.integration,
                         limit=4, expected_counts=self.counts)

    def test_only_train_manifest_and_forward_no_metrics_or_mutation(self):
        original_reader = EmbeddingDataset.from_json
        calls = []
        def reader(text, *args, **kwargs):
            calls.append(json.loads(text)["config"]["split"])
            return original_reader(text, *args, **kwargs)
        with patch.object(EmbeddingDataset, "from_json", side_effect=reader):
            result = self.run_smoke()
        self.assertEqual(calls, ["train"])
        self.assertEqual(result["audio_shape"], [4, 10, 256])
        self.assertEqual(result["visual_shape"], [4, 10, 512])
        self.assertEqual(result["logit_shape"], [4, 1])
        self.assertEqual(result["positive_count"], 2)
        self.assertEqual(result["negative_count"], 2)
        self.assertTrue(result["finite_logits"])
        self.assertTrue(result["producer_inputs_and_manifests_unchanged"])
        self.assertFalse(result["metrics_computed"])
        self.assertFalse(result["trained"])
        self.assertFalse(result["wandb_run_created"])

    def test_source_hash_mismatch_blocks_before_forward(self):
        path = self.audio / "SOURCE.txt"
        path.write_text("changed", encoding="utf-8")
        with patch.object(LateFusionMLP, "forward", side_effect=AssertionError("must not forward")):
            with self.assertRaises(ContractError):
                self.run_smoke()

    def test_opaque_test_manifest_hash_mismatch_blocks_before_train_reader(self):
        path = self.integration / "test_pairs.json"
        path.write_bytes(path.read_bytes() + b" ")
        with patch.object(EmbeddingDataset, "from_json", side_effect=AssertionError("must not read")):
            with self.assertRaises(ContractError):
                self.run_smoke()


if __name__ == "__main__":
    unittest.main()
