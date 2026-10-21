"""Phase 3 synthetic verification only; no real exports or experiment runs."""
from dataclasses import fields, replace
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import ContractError
from fusion.artifact_integration import _load_exports, manifest_from_json, manifest_to_json
from fusion.baseline_data import ProbabilityFeatures, extract_pair_probabilities
from fusion.baseline_metrics import (
    PairScoreRecord, evaluate_records, make_score_records, ranking_metrics, summarize_scores,
)
from fusion.baselines import (
    METHODS, cosine_similarity, dot_product_agreement, jensen_shannon_similarity,
    score_probabilities, top1_agreement,
)
from fusion.pair_contracts import PairType, SamplingPolicy
from fusion.pair_generator import GenerationConfig, generate_pairs
from test_fusion_artifact_integration import write_fixture
from test_fusion_pair_generator import ROWS, bundles, hard_policy, record


def onehot(index, n=1):
    result = np.zeros((n, 10, 29), dtype=np.float64)
    result[:, :, index] = 1
    return result


def predictions(bundle):
    """Synthetic predictions depend on storage row/bin, never annotation labels."""
    splits = {}
    for split, data in bundle.splits.items():
        values = np.empty((len(data.records), 10, 29), dtype=np.float16)
        for i in range(len(values)):
            for t in range(10):
                values[i, t] = np.roll(np.arange(1, 30, dtype=float), (i * 7 + t) % 29) / 435
        splits[split] = replace(data, probabilities=values)
    return replace(bundle, splits=splits)


class BaselineAdapterTests(unittest.TestCase):
    def setUp(self):
        self.audio, self.vision = map(predictions, bundles())
        self.manifest = generate_pairs(
            self.audio, self.vision, policy=SamplingPolicy(42),
            config=GenerationConfig("train", easy_negative_budget=7, challenge_budget=5))

    def extract(self, **kwargs):
        return extract_pair_probabilities(self.manifest, self.audio, self.vision,
                                          split="train", **kwargs)

    def test_original_and_substituted_rows_with_shuffled_visual_order(self):
        result = self.extract()
        self.assertTrue(any(p.audio.row_index != p.visual.row_index for p in result.primary.pairs))
        for part in (result.primary, result.challenge):
            for i, pair in enumerate(part.pairs):
                np.testing.assert_array_equal(part.features.audio[i],
                    self.audio.splits["train"].probabilities[pair.audio.row_index])
                np.testing.assert_array_equal(part.features.visual[i],
                    self.vision.splits["train"].probabilities[pair.visual.row_index])
            self.assertEqual(part.features.audio.shape, (len(part.pairs), 10, 29))
        self.assertTrue(any(p.original_pair_target == 0 for p in result.primary.pairs))

    def test_features_exclude_metadata_and_targets(self):
        result = self.extract()
        self.assertEqual([f.name for f in fields(ProbabilityFeatures)], ["audio", "visual"])
        part = result.primary
        direct = score_probabilities(part.features.audio, part.features.visual)
        rows = make_score_records(part)
        for i, row in enumerate(rows):
            self.assertEqual(row.target, row.pair.original_pair_target)
            self.assertEqual(row.scores, tuple((m, direct.pair_scores[m][i]) for m in METHODS))
        # Changing legitimate provenance labels while keeping predictions fixed
        # cannot change scores: scorer has no metadata/target argument.
        original = next(p for p in part.pairs if p.original_pair_target == 1)
        swap = next(p for p in part.pairs if p.original_pair_target == 0)
        same_scores = rows[0].scores
        self.assertEqual(PairScoreRecord(original, "primary", same_scores).scores,
                         PairScoreRecord(swap, "primary", same_scores).scores)

    def test_empty_annotation_keeps_predictions_and_unknown_semantics(self):
        part = self.extract().primary
        i = next(i for i, p in enumerate(part.pairs)
                 if p.pair_type is PairType.ORIGINAL_POSITIVE and p.audio.record.end == 0)
        pair = part.pairs[i]
        self.assertEqual(pair.semantic_compatibility.value, "UNKNOWN")
        self.assertFalse(pair.event_supervision.semantic_eligible)
        self.assertEqual(make_score_records(part)[i].target, 1)  # Provenance only.
        np.testing.assert_array_equal(part.features.audio[i],
            self.audio.splits["train"].probabilities[pair.audio.row_index])

    def test_deterministic_prefix_readonly_and_source_immutability(self):
        before = self.audio.splits["train"].probabilities.copy()
        first, second = self.extract(limit_per_partition=3), self.extract(limit_per_partition=3)
        self.assertEqual(first.primary.pairs, self.manifest.primary_pairs[:3])
        np.testing.assert_array_equal(first.primary.features.audio, second.primary.features.audio)
        self.assertFalse(first.primary.features.audio.flags.writeable)
        self.assertFalse(np.shares_memory(first.primary.features.audio, before))
        with self.assertRaises(ValueError):
            first.primary.features.audio[0, 0, 0] = 0
        np.testing.assert_array_equal(before, self.audio.splits["train"].probabilities)
        self.assertEqual(self.extract(limit_per_partition=0).primary.features.audio.shape, (0, 10, 29))

    def test_challenges_are_separate_and_one_class_metrics_undefined(self):
        data = self.extract()
        self.assertTrue(all(p.pair_type is PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE
                            for p in data.challenge.pairs))
        self.assertFalse(any(p.pair_type is PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE
                             for p in data.primary.pairs))
        result = evaluate_records(make_score_records(data.challenge), split="train", partition="challenge")
        self.assertEqual((result.count, result.positive_count, result.negative_count), (5, 0, 5))
        for method in result.methods:
            self.assertIsNone(method.roc_auc.value)
            self.assertEqual(method.average_precision.undefined_reason, "only_one_target_class")

    def test_wrong_split_missing_probabilities_and_modality_fail(self):
        with self.assertRaises(ContractError):
            extract_pair_probabilities(self.manifest, self.audio, self.vision, split="val")
        missing = replace(self.vision, splits={"train": replace(self.vision.splits["train"], probabilities=None)})
        with self.assertRaises(ContractError):
            extract_pair_probabilities(self.manifest, self.audio, missing, split="train")
        with self.assertRaises(ContractError):
            extract_pair_probabilities(self.manifest, self.vision, self.audio, split="train")
        for limit in (-1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ContractError):
                self.extract(limit_per_partition=limit)

    def test_corrupted_row_references_rejected_before_selection(self):
        first = self.manifest.primary_pairs[0]
        for index in (-1, len(ROWS), (first.audio.row_index + 1) % len(ROWS)):
            # Simulate a reconstructed input bypassing frozen constructors.
            source = object.__new__(type(first.audio))
            source.__dict__.update(vars(first.audio) | {"row_index": index})
            pair = object.__new__(type(first))
            pair.__dict__.update(vars(first) | {"audio": source})
            bad = object.__new__(type(self.manifest))
            bad.__dict__.update(vars(self.manifest) | {"primary_pairs": (pair,) + self.manifest.primary_pairs[1:]})
            with self.subTest(index=index), self.assertRaises(ContractError):
                extract_pair_probabilities(bad, self.audio, self.vision, split="train", limit_per_partition=0)

    def test_nan_and_infinity_artifacts_rejected(self):
        for value in (np.nan, np.inf):
            values = self.audio.splits["train"].probabilities.copy()
            values[0, 0, 0] = value
            bad = replace(self.audio, splits={"train": replace(self.audio.splits["train"], probabilities=values)})
            with self.subTest(value=value), self.assertRaises(ContractError):
                extract_pair_probabilities(self.manifest, bad, self.vision, split="train")

    def test_explicit_invalid_input_bins_rejected_without_annotation_masking(self):
        valid = np.ones((len(ROWS), 10), dtype=bool)
        valid[0, 0] = False
        bad = replace(self.audio, splits={"train": replace(self.audio.splits["train"], input_validity=valid)})
        with self.assertRaisesRegex(ContractError, "explicitly invalid"):
            extract_pair_probabilities(self.manifest, bad, self.vision, split="train")
        good = replace(self.audio, splits={"train": replace(self.audio.splits["train"], input_validity=np.ones_like(valid))})
        np.testing.assert_array_equal(
            extract_pair_probabilities(self.manifest, good, self.vision, split="train").primary.features.audio,
            self.extract().primary.features.audio)

    def test_other_synthetic_splits_do_not_enter_train_features(self):
        a, v = map(predictions, bundles({"train": ROWS,
            "val": (record("val_a", 0), record("val_b", 1)),
            "test": (record("test_a", 0), record("test_b", 1))}))
        manifest = generate_pairs(a, v, policy=SamplingPolicy(42), config=self.manifest.config)
        result = extract_pair_probabilities(manifest, a, v, split="train")
        np.testing.assert_array_equal(result.primary.features.audio, self.extract().primary.features.audio)
        self.assertTrue(all(p.split == "train" for p in result.primary.pairs))

    def test_hard_pairs_and_empty_partition_supported(self):
        manifest = generate_pairs(self.audio, self.vision, policy=SamplingPolicy(42, hard_policy()),
            config=GenerationConfig("train", hard_negative_budget=3, enable_hard_negatives=True))
        result = extract_pair_probabilities(manifest, self.audio, self.vision, split="train")
        self.assertEqual(sum(p.pair_type is PairType.CROSS_CLASS_HARD_NEGATIVE for p in result.primary.pairs), 3)
        self.assertEqual(result.challenge.features.audio.shape, (0, 10, 29))
        self.assertEqual(make_score_records(result.challenge), ())

    def test_phase2_serialized_file_contract_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            aroot, vroot, _, counts = write_fixture(Path(temp))
            with _load_exports({"audio": aroot, "vision": vroot}, counts) as (a, v):
                manifest = generate_pairs(a, v, policy=SamplingPolicy(42),
                    config=GenerationConfig("train", easy_negative_budget=3, challenge_budget=2))
                text = manifest_to_json(manifest, a, v, artifact_fingerprint="e" * 64)
                restored = manifest_from_json(text, a, v, expected_artifact_fingerprint="e" * 64)
                result = extract_pair_probabilities(restored, a, v, split="train")
                self.assertEqual(result.primary.pairs, manifest.primary_pairs)
                self.assertTrue(all(np.isfinite(x).all() for x in
                    score_probabilities(result.primary.features.audio, result.primary.features.visual).pair_scores.values()))


class ScoringTests(unittest.TestCase):
    functions = (top1_agreement, dot_product_agreement, cosine_similarity, jensen_shannon_similarity)

    def test_identical_and_disjoint_onehot(self):
        for function in self.functions:
            with self.subTest(function=function.__name__):
                np.testing.assert_array_equal(function(onehot(0), onehot(0)), np.ones((1, 10)))
                np.testing.assert_array_equal(function(onehot(0), onehot(1)), np.zeros((1, 10)))

    def test_uniform_top1_ties_dot_and_js_definitions(self):
        uniform = np.full((1, 10, 29), 1 / 29)
        self.assertAlmostEqual(dot_product_agreement(uniform, uniform)[0, 0], 1 / 29)
        for function in (top1_agreement, cosine_similarity, jensen_shannon_similarity):
            self.assertAlmostEqual(function(uniform, uniform)[0, 0], 1)
        self.assertEqual(top1_agreement(uniform, onehot(0))[0, 0], 1)
        self.assertEqual(top1_agreement(uniform, onehot(1))[0, 0], 0)
        mixture = (onehot(0) + onehot(1)) / 2
        # JSD([1,0], [.5,.5]) = .5 log2(4/3) + .25 log2(2/3) + .25.
        expected = 1 - (.5 * np.log2(4 / 3) + .25 * np.log2(2 / 3) + .25)
        self.assertAlmostEqual(jensen_shannon_similarity(onehot(0), mixture)[0, 0], expected)

    def test_background_dominance_is_retained_not_annotation_masked(self):
        a, v = .99 * onehot(28) + .01 * onehot(0), .99 * onehot(28) + .01 * onehot(1)
        scores = score_probabilities(a, v).pair_scores
        self.assertEqual(scores["top1"][0], 1)
        self.assertAlmostEqual(scores["dot_product"][0], .9801)
        self.assertGreater(scores["cosine"][0], .99)
        self.assertAlmostEqual(scores["jensen_shannon"][0], .99)

    def test_zero_vectors_abstain_for_one_or_both_sides(self):
        zero = np.zeros((1, 10, 29))
        for function in self.functions:
            for a, v in ((zero, zero), (zero, onehot(2)), (onehot(2), zero)):
                with self.subTest(function=function.__name__):
                    np.testing.assert_array_equal(function(a, v), np.zeros((1, 10)))

    def test_nonfinite_negative_unnormalized_and_malformed_inputs_fail(self):
        bad_values = []
        for value in (np.nan, np.inf, -np.inf, -1., 1.01):
            bad = onehot(0)
            bad[0, 0, 0] = value
            bad_values.append(bad)
        bad_values.extend([onehot(0) * .5, onehot(0).astype(int), np.zeros((1, 9, 29)), np.zeros((1, 10, 28))])
        for function in self.functions:
            for bad in bad_values:
                with self.subTest(function=function.__name__, shape=bad.shape), self.assertRaises(ContractError):
                    function(bad, onehot(0))
            with self.assertRaises(ContractError):
                function(onehot(0, 2), onehot(0))

    def test_float16_renormalization_stability_symmetry_and_input_immutability(self):
        a = np.full((2, 10, 29), 1 / 29, dtype=np.float16)
        b = onehot(4, 2)
        b[:, :, 4] = 1 - 1e-14
        b[:, :, 5] = 1e-14
        before = a.copy()
        for function in self.functions:
            first = function(a, b)
            np.testing.assert_array_equal(first, function(a, b))
            np.testing.assert_allclose(first, function(b, a), rtol=0, atol=1e-15)
            self.assertTrue(np.isfinite(first).all())
            self.assertTrue(((first >= 0) & (first <= 1)).all())
        np.testing.assert_array_equal(a, before)

    def test_fixed_mean_over_all_ten_bins_and_score_ordering(self):
        a, v = onehot(0, 3), onehot(1, 3)
        v[0] = a[0]
        v[1, :4] = a[1, :4]
        result = score_probabilities(a, v)
        for name in METHODS:
            np.testing.assert_allclose(result.pair_scores[name], [1, .4, 0], rtol=0, atol=1e-15)
            np.testing.assert_allclose(result.pair_scores[name], result.per_bin[name].mean(axis=1))
            self.assertFalse(result.pair_scores[name].flags.writeable)

    def test_subnormal_probabilities_are_finite_without_floating_errors(self):
        a, v = onehot(0), onehot(0)
        a[:, :, 1] = np.nextafter(0., 1.)
        v[:, :, 2] = np.nextafter(0., 1.)
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            for function in self.functions:
                values = function(a, v)
                self.assertTrue(np.isfinite(values).all())
                np.testing.assert_allclose(values, np.ones((1, 10)), rtol=0, atol=1e-14)


class MetricTests(unittest.TestCase):
    def test_known_ranking_metrics_and_ties(self):
        auc, ap = ranking_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
        self.assertAlmostEqual(auc.value, .75)
        self.assertAlmostEqual(ap.value, 5 / 6)
        for order in itertools.permutations(range(4)):
            y = np.array([0, 1, 0, 1])[list(order)]
            auc, ap = ranking_metrics(y, np.ones(4))
            self.assertEqual(auc.value, .5)
            self.assertEqual(ap.value, .5)

    def test_ranking_metrics_match_independent_pairwise_and_threshold_oracles(self):
        y = np.array([0, 1, 0, 1], dtype=int)
        for scores in itertools.product((0., .5, 1.), repeat=4):
            auc, ap = ranking_metrics(y, scores)
            positive = [s for s, target in zip(scores, y) if target]
            negative = [s for s, target in zip(scores, y) if not target]
            expected_auc = sum((p > n) + .5 * (p == n) for p in positive for n in negative) / 4
            expected_ap = 0
            for threshold in sorted(set(scores), reverse=True):
                selected = [target for target, s in zip(y, scores) if s >= threshold]
                new_positive = sum(target for target, s in zip(y, scores) if s == threshold)
                expected_ap += new_positive / 2 * sum(selected) / len(selected)
            self.assertAlmostEqual(auc.value, expected_auc)
            self.assertAlmostEqual(ap.value, expected_ap)

    def test_empty_and_each_one_class_explicitly_undefined(self):
        for y, s in (([], []), ([0, 0], [.1, .2]), ([1, 1], [.1, .2])):
            for metric in ranking_metrics(y, s):
                self.assertIsNone(metric.value)
                self.assertIsNotNone(metric.undefined_reason)
        result = evaluate_records((), split="val", partition="primary")
        self.assertEqual(result.count, 0)
        self.assertEqual(result.methods[0].roc_auc.undefined_reason, "empty_partition")

    def test_ranking_invalid_targets_and_scores_fail(self):
        for y, s in (([0, 2], [.1, .2]), ([0., 1.], [.1, .2]), ([True, False], [.1, .2]),
                     ([0, 1], [np.nan, .2]), ([0, 1], [np.inf, .2]), ([0], [.1, .2]), ([[0, 1]], [[.1, .2]])):
            with self.subTest(y=y), self.assertRaises(ContractError):
                ranking_metrics(y, s)

    def test_summary_definition_and_empty_distribution(self):
        result = summarize_scores([0, 1, 2, 3])
        self.assertEqual((result.count, result.minimum, result.q25, result.median, result.q75, result.maximum),
                         (4, 0, .75, 1.5, 2.25, 3))
        self.assertEqual(result.mean, 1.5)
        self.assertAlmostEqual(result.std, np.sqrt(1.25))
        self.assertIsNone(summarize_scores([]).mean)
        with self.assertRaises(ContractError):
            summarize_scores([np.inf])

    def test_record_evaluation_preserves_counts_distributions_and_rejects_mixing(self):
        a, v = map(predictions, bundles())
        manifest = generate_pairs(a, v, policy=SamplingPolicy(42),
                                 config=GenerationConfig("train", easy_negative_budget=3, challenge_budget=2))
        data = extract_pair_probabilities(manifest, a, v, split="train")
        primary, challenge = make_score_records(data.primary), make_score_records(data.challenge)
        result = evaluate_records(primary, split="train", partition="primary")
        self.assertEqual((result.count, result.positive_count, result.negative_count), (12, 9, 3))
        self.assertEqual(result, evaluate_records(primary, split="train", partition="primary"))
        self.assertEqual(result.methods[0].positive_scores.count, 9)
        self.assertEqual(result.methods[0].negative_scores.count, 3)
        for records, split, partition in ((primary, "test", "primary"), (primary, "val", "primary"),
                                          (primary + challenge, "train", "primary"),
                                          (primary[::-1], "train", "primary"),
                                          (primary + primary, "train", "primary")):
            with self.subTest(split=split, partition=partition), self.assertRaises(ContractError):
                evaluate_records(records, split=split, partition=partition)

    def test_malformed_result_scores_and_partition_cannot_enter_evaluation(self):
        a, v = bundles()
        manifest = generate_pairs(a, v, policy=SamplingPolicy(42), config=GenerationConfig("train"))
        part = extract_pair_probabilities(manifest, a, v, split="train").primary
        first = make_score_records(part)[0]
        for value in (np.nan, np.inf, -1., 1.1, True):
            with self.subTest(value=value), self.assertRaises(ContractError):
                replace(first, scores=((METHODS[0], value),) + first.scores[1:])
        with self.assertRaises(ContractError):
            replace(first, partition="challenge")


if __name__ == "__main__":
    unittest.main()
