"""Offline synthetic operating-point regression and independent math oracles."""
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import itertools
import json
import unittest

import numpy as np

from fusion.baseline_data import extract_pair_probabilities
from fusion.baseline_evaluation import (
    SCORE_BIN_EDGES, SCORE_MEANING, evaluate_primary, evaluate_challenge,
    evaluation_report_to_json, frozen_policy_to_json, score_reliability, threshold_sensitivity,
)
from fusion.baseline_metrics import PairScoreRecord, make_score_records
from fusion.baseline_protocol import ScoreDataset, ScoreProvenance, validate_score_dataset
from fusion.baseline_thresholds import (
    ConfusionCounts, FrozenThresholdPolicy, ThresholdPolicy, classification_metrics,
    freeze_threshold, select_threshold, select_thresholds, threshold_candidates,
)
from fusion.baselines import METHODS, score_probabilities
from fusion.contracts import ContractError
from fusion.pair_contracts import SamplingPolicy
from fusion.pair_generator import GenerationConfig, generate_pairs
from test_fusion_pair_generator import ROWS, bundles, record


def fixture(*, split="val", partition="primary", positives=2, negatives=2, challenges=2,
            scores=None, metadata=None):
    a, v = bundles({split: ROWS} if metadata is None else metadata)
    manifest = generate_pairs(a, v, policy=SamplingPolicy(42), config=GenerationConfig(
        split, positive_budget=positives, easy_negative_budget=negatives, challenge_budget=challenges))
    data = extract_pair_probabilities(manifest, a, v, split=split)
    records = make_score_records(getattr(data, partition))
    if scores is not None:
        records = rescore(records, scores)
    dataset = validate_score_dataset(records, manifest, a, v, split=split, partition=partition,
                                     source_reference="synthetic-unit-fixture")
    return dataset, manifest, a, v


def rescore(records, scores):
    assert len(records) == len(scores)
    return tuple(replace(r, scores=tuple((method, float(value)) for method in METHODS))
                 for r, value in zip(records, scores))


def tie_dataset():
    dataset, _, _, _ = fixture()
    pos, neg = iter((.1, .4)), iter((.2, .3))
    values = [next(pos if r.target else neg) for r in dataset.records]
    return replace(dataset, records=rescore(dataset.records, values))


def raw_like(value, **changes):
    out = object.__new__(type(value))
    out.__dict__.update(vars(value) | changes)
    return out


class ThresholdSelectionTests(unittest.TestCase):
    def test_exhaustive_threshold_selection_matches_independent_prediction_oracle(self):
        base, _, _, _ = fixture()
        y = [r.target for r in base.records]
        for scores in itertools.product((0., .5, 1.), repeat=4):
            dataset = replace(base, records=rescore(base.records, scores))
            candidates = threshold_candidates(dataset, method="top1")
            observed = sorted(set(scores))
            # Probe scores, gaps and external cutoffs independently: every
            # possible >= prediction vector must occur exactly once in the sweep.
            probes = [-1., 2.] + observed + [(a + b) / 2 for a, b in zip(observed, observed[1:])]
            realizable = {tuple(s >= t for s in scores) for t in probes}
            actual = {tuple(s >= c.threshold for s in scores) for c in candidates}
            self.assertEqual(actual, realizable)
            self.assertEqual(len(candidates), len(realizable))
            best = []
            for candidate in candidates:
                pred = [s >= candidate.threshold for s in scores]
                tp = sum(p and target == 1 for p, target in zip(pred, y))
                fp = sum(p and target == 0 for p, target in zip(pred, y))
                fn = sum(not p and target == 1 for p, target in zip(pred, y))
                value = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
                best.append((value, candidate.threshold))
            policy = select_threshold(dataset, method="top1")
            self.assertEqual(policy.threshold, max(best)[1])
            self.assertAlmostEqual(policy.selection.metrics.f1, float(max(best)[0]))

    def test_exact_f1_ties_choose_largest_threshold(self):
        dataset = tie_dataset()
        policy = select_threshold(dataset, method="cosine")
        self.assertEqual(policy.threshold, .4)
        self.assertEqual(threshold_sensitivity(policy).equally_optimal_thresholds, (.1, .4))
        self.assertEqual(policy, select_threshold(dataset, method="cosine"))

    def test_all_four_methods_and_validation_provenance(self):
        dataset = tie_dataset()
        policies = select_thresholds(dataset)
        self.assertEqual(tuple(p.method for p in policies), METHODS)
        for policy in policies:
            self.assertEqual((policy.validation_count, policy.validation_positive_count,
                              policy.validation_negative_count), (4, 2, 2))
            self.assertEqual(policy.source_fingerprint, dataset.fingerprint)
            self.assertEqual(policy.provenance, dataset.provenance)
            self.assertEqual(policy.candidate_count, 5)

    def test_method_specific_scores_select_and_apply_the_correct_policy(self):
        dataset, _, _, _ = fixture()
        records = tuple(replace(r, scores=tuple((name, float(.6 + i / 10) if r.target else .1)
                                                for i, name in enumerate(METHODS))) for r in dataset.records)
        dataset = replace(dataset, records=records)
        challenge, _, _, _ = fixture(partition="challenge", scores=[.75, .75])
        for i, policy in enumerate(select_thresholds(dataset)):
            self.assertEqual(policy.threshold, float(.6 + i / 10))
            frozen = freeze_threshold(policy)
            self.assertEqual(evaluate_primary(dataset, frozen).classification.f1, 1.)
            self.assertEqual(evaluate_challenge(challenge, frozen).false_acceptance_rate, 1. if i < 2 else 0.)

    def test_constant_scores_and_extreme_operating_points(self):
        for value in (0., .6, 1.):
            dataset, _, _, _ = fixture(scores=[value] * 4)
            points = threshold_candidates(dataset, method="top1")
            self.assertEqual(len(points), 2)
            self.assertEqual(points[0].metrics.positive_prediction_rate, 1.)
            self.assertEqual(points[-1].metrics.positive_prediction_rate, 0.)
            self.assertGreater(points[-1].threshold, value)
            self.assertTrue(np.isfinite(points[-1].threshold))
            self.assertEqual(select_threshold(dataset, method="top1").threshold, value)

    def test_exact_threshold_ties_are_positive(self):
        result = classification_metrics([1, 0, 1, 0], [.4, .4, .39, .39], threshold=.4)
        self.assertEqual(result.counts, ConfusionCounts(1, 1, 1, 1))

    def test_train_test_challenge_and_empty_fit_rejection(self):
        train, _, _, _ = fixture(split="train")
        challenge, _, _, _ = fixture(partition="challenge")
        empty, _, _, _ = fixture(positives=0, negatives=0)
        test = raw_like(train, split="test")
        for data in (train, challenge, empty, test):
            with self.subTest(split=data.split, partition=data.partition), self.assertRaises(ContractError):
                select_threshold(data, method="top1")

    def test_one_class_is_explicitly_unfit_but_candidates_are_defined(self):
        for positives, negatives in ((0, 2), (2, 0)):
            dataset, _, _, _ = fixture(positives=positives, negatives=negatives)
            with self.assertRaisesRegex(ContractError, "both correspondence target classes"):
                select_threshold(dataset, method="top1")
            self.assertEqual(len(threshold_candidates(dataset, method="top1")), 2)

    def test_unknown_method_rejected(self):
        dataset, _, _, _ = fixture()
        with self.assertRaises(ContractError):
            select_threshold(dataset, method="probability")

    def test_immutable_policy_and_validation_evidence(self):
        policy = select_threshold(tie_dataset(), method="top1")
        frozen = freeze_threshold(policy)
        for obj, field, value in ((policy, "threshold", .2), (frozen, "policy_fingerprint", "0" * 64),
                                  (policy.source, "split", "train"), (policy.source.records[0], "scores", ())):
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, field, value)
        self.assertIsInstance(policy.source.records, tuple)
        self.assertEqual(frozen, freeze_threshold(policy))

    def test_reconstructed_policy_fields_and_shadowed_hooks_rejected(self):
        policy = select_threshold(tie_dataset(), method="top1")
        changes = ({"threshold": .2}, {"threshold": True}, {"objective": "test_f1"},
                   {"tie_break": "first"}, {"validation_count": 99}, {"validation_positive_count": 9},
                   {"validation_negative_count": 9}, {"candidate_count": 1},
                   {"source_fingerprint": "0" * 64}, {"method": "unknown"},
                   {"protocol_version": "future-test-v1"},
                   {"source": raw_like(policy.source, split="test")})
        for change in changes:
            bad = raw_like(policy, **change, __post_init__=lambda: None)
            with self.subTest(change=change), self.assertRaises(ContractError):
                freeze_threshold(bad)
        missing = object.__new__(ThresholdPolicy)
        with self.assertRaises(ContractError):
            freeze_threshold(missing)

    def test_reconstructed_selection_metrics_cannot_forge_optimum(self):
        policy = select_threshold(tie_dataset(), method="top1")
        fake_metrics = raw_like(policy.selection.metrics, f1=1.)
        fake_point = raw_like(policy.selection, metrics=fake_metrics)
        with self.assertRaises(ContractError):
            freeze_threshold(raw_like(policy, selection=fake_point))
        other = threshold_candidates(policy.source, method="top1")[1]
        with self.assertRaisesRegex(ContractError, "optimum"):
            freeze_threshold(raw_like(policy, selection=other, threshold=other.threshold))

    def test_malformed_frozen_policy_and_subclasses_rejected(self):
        policy = select_threshold(tie_dataset(), method="top1")
        frozen = freeze_threshold(policy)
        for change in ({"policy_fingerprint": "0" * 64}, {"protocol_version": "test"},
                       {"policy": raw_like(policy, threshold=.2)}):
            bad = raw_like(frozen, **change, __post_init__=lambda: None)
            with self.assertRaises(ContractError):
                evaluate_primary(policy.source, bad)
        class SubPolicy(ThresholdPolicy):
            pass
        subclass = object.__new__(SubPolicy)
        subclass.__dict__.update(vars(policy))
        with self.assertRaises(ContractError):
            freeze_threshold(subclass)


class ClassificationMetricTests(unittest.TestCase):
    def test_confusion_and_all_metrics_against_independent_math_oracle(self):
        for y in itertools.product((0, 1), repeat=4):
            for pred in itertools.product((0, 1), repeat=4):
                result = classification_metrics(y, pred, threshold=.5)
                tp = sum(t == 1 and p == 1 for t, p in zip(y, pred))
                tn = sum(t == 0 and p == 0 for t, p in zip(y, pred))
                fp = sum(t == 0 and p == 1 for t, p in zip(y, pred))
                fn = sum(t == 1 and p == 0 for t, p in zip(y, pred))
                self.assertEqual(result.counts, ConfusionCounts(tp, tn, fp, fn))
                ratio = lambda a, b: a / b if b else 0.
                self.assertEqual(result.accuracy, (tp + tn) / 4)
                self.assertEqual(result.precision, ratio(tp, tp + fp))
                self.assertEqual(result.recall, ratio(tp, tp + fn))
                self.assertEqual(result.specificity, ratio(tn, tn + fp))
                self.assertEqual(result.f1, ratio(2 * tp, 2 * tp + fp + fn))
                self.assertEqual(result.macro_f1, (ratio(2 * tp, 2 * tp + fp + fn) + ratio(2 * tn, 2 * tn + fp + fn)) / 2)
                self.assertEqual(result.positive_prediction_rate, sum(pred) / 4)
                expected_ba = (tp / sum(y) + tn / (4 - sum(y))) / 2 if 0 < sum(y) < 4 else None
                self.assertEqual(result.balanced_accuracy, expected_ba)

    def test_zero_division_and_degenerate_class_conventions(self):
        empty = classification_metrics([], [], threshold=.5)
        self.assertIsNone(empty.accuracy)
        self.assertIsNone(empty.balanced_accuracy)
        self.assertIsNone(empty.positive_prediction_rate)
        self.assertEqual((empty.precision, empty.recall, empty.f1, empty.macro_f1, empty.specificity), (0.,) * 5)
        only_negative = classification_metrics([0, 0], [0., 0.], threshold=.5)
        self.assertEqual((only_negative.accuracy, only_negative.precision, only_negative.recall,
                          only_negative.f1, only_negative.macro_f1, only_negative.specificity), (1., 0., 0., 0., .5, 1.))
        self.assertIsNone(only_negative.balanced_accuracy)
        self.assertEqual(classification_metrics([1, 1], [1., 1.], threshold=.5).macro_f1, .5)

    def test_all_positive_negative_and_above_one_threshold(self):
        self.assertEqual(classification_metrics([0, 1], [0., 1.], threshold=0).positive_prediction_rate, 1.)
        self.assertEqual(classification_metrics([0, 1], [0., 1.], threshold=np.nextafter(1., np.inf)).positive_prediction_rate, 0.)

    def test_float16_above_one_sentinel_reproduction(self):
        result = classification_metrics([0, 1], np.ones(2, dtype=np.float16),
                                        threshold=np.nextafter(1.0, np.inf))
        self.assertEqual(result.counts, ConfusionCounts(tp=0, tn=1, fp=0, fn=1))

    def test_float32_above_one_sentinel_reproduction(self):
        result = classification_metrics([0, 1], np.ones(2, dtype=np.float32),
                                        threshold=np.nextafter(1.0, np.inf))
        self.assertEqual(result.counts, ConfusionCounts(tp=0, tn=1, fp=0, fn=1))

    def test_equal_and_neighbor_thresholds_at_exact_float_scores(self):
        for dtype in (np.float16, np.float32, np.float64):
            for value in (0., .5, 1.):
                cases = ((np.nextafter(value, -np.inf), ConfusionCounts(1, 0, 1, 0)),
                         (value, ConfusionCounts(1, 0, 1, 0)),
                         (np.nextafter(value, np.inf), ConfusionCounts(0, 1, 0, 1)))
                for threshold, expected in cases:
                    with self.subTest(dtype=dtype, value=value, threshold=threshold):
                        result = classification_metrics([0, 1], np.full(2, value, dtype=dtype),
                                                        threshold=threshold)
                        self.assertEqual(result.counts, expected)

    def test_exact_score_values_consistent_across_float_dtypes(self):
        # All scores are exactly representable in every tested dtype. Expected
        # counts are specified directly, independently of classification_metrics.
        cases = ((0., ConfusionCounts(2, 0, 2, 0)),
                 (np.nextafter(0., np.inf), ConfusionCounts(1, 0, 2, 1)),
                 (np.nextafter(.25, -np.inf), ConfusionCounts(1, 0, 2, 1)),
                 (.25, ConfusionCounts(1, 0, 2, 1)),
                 (np.nextafter(.25, np.inf), ConfusionCounts(1, 1, 1, 1)),
                 (.5, ConfusionCounts(1, 1, 1, 1)),
                 (np.nextafter(.5, np.inf), ConfusionCounts(0, 1, 1, 2)),
                 (1., ConfusionCounts(0, 1, 1, 2)),
                 (np.nextafter(1., np.inf), ConfusionCounts(0, 2, 0, 2)))
        for dtype in (np.float16, np.float32, np.float64):
            scores = np.array([0., .25, .5, 1.], dtype=dtype)
            for threshold, expected in cases:
                with self.subTest(dtype=dtype, threshold=threshold):
                    self.assertEqual(classification_metrics([1, 0, 1, 0], scores,
                                     threshold=threshold).counts, expected)

    def test_readonly_source_scores_unchanged_after_comparison(self):
        for dtype in (np.float16, np.float32, np.float64):
            scores = np.array([0., .5, 1.], dtype=dtype)
            before = scores.copy()
            scores.flags.writeable = False
            result = classification_metrics([0, 1, 0], scores, threshold=np.nextafter(.5, np.inf))
            with self.subTest(dtype=dtype):
                self.assertEqual(result.counts, ConfusionCounts(0, 1, 1, 1))
                np.testing.assert_array_equal(scores, before)
                self.assertEqual(scores.dtype, before.dtype)
                self.assertEqual(scores.tobytes(), before.tobytes())
                self.assertFalse(scores.flags.writeable)

    def test_validation_preserved_across_float_score_dtypes(self):
        for dtype in (np.float16, np.float32, np.float64):
            for value in (np.nan, np.inf, -np.inf, -.25, 1.25):
                with self.subTest(dtype=dtype, value=value), self.assertRaises(ContractError):
                    classification_metrics([0, 1], np.array([value, .5], dtype=dtype), threshold=.5)
            for targets in ([0, 2], [False, True], [0., 1.], [0], [[0, 1]]):
                with self.subTest(dtype=dtype, targets=targets), self.assertRaises(ContractError):
                    classification_metrics(targets, np.array([0., 1.], dtype=dtype), threshold=.5)

    def test_integer_score_dtypes_remain_supported(self):
        for dtype in (np.int8, np.uint8, np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                result = classification_metrics([0, 1], np.array([0, 1], dtype=dtype), threshold=.5)
                self.assertEqual(result.counts, ConfusionCounts(1, 1, 0, 0))

    def test_invalid_targets_scores_and_thresholds_rejected(self):
        for y, s in (([0., 1.], [.2, .8]), ([False, True], [.2, .8]), ([0, 2], [.2, .8]),
                     ([0, 1], [-.1, .8]), ([0, 1], [.2, 1.1]), ([0, 1], [np.nan, .8]),
                     ([0, 1], [.2, np.inf]), ([0], [.2, .8]), ([[0, 1]], [[.2, .8]])):
            with self.subTest(y=y, s=s), self.assertRaises(ContractError):
                classification_metrics(y, s, threshold=.5)
        for threshold in (np.nan, np.inf, -np.inf, True, "0.5"):
            with self.subTest(threshold=threshold), self.assertRaises(ContractError):
                classification_metrics([0, 1], [.2, .8], threshold=threshold)


class ScoreDatasetTests(unittest.TestCase):
    def test_phase2_manifest_binding_and_full_partition_coverage(self):
        dataset, manifest, a, v = fixture()
        with self.assertRaisesRegex(ContractError, "exactly"):
            validate_score_dataset(dataset.records[:1], manifest, a, v, split="val", partition="primary", source_reference="fixture")
        self.assertEqual(dataset.provenance.generation, manifest.provenance)
        self.assertEqual(len(dataset.fingerprint), 64)
        self.assertEqual(dataset.fingerprint, replace(dataset).fingerprint)
        changed = replace(dataset, records=rescore(dataset.records, [.1, .2, .3, .4]))
        self.assertNotEqual(dataset.fingerprint, changed.fingerprint)

    def test_mixed_partition_duplicates_and_missing_scores_rejected(self):
        dataset, manifest, a, v = fixture()
        challenge, _, _, _ = fixture(partition="challenge")
        missing = raw_like(dataset.records[0])
        del missing.__dict__["scores"]
        for records in (dataset.records + dataset.records, dataset.records + challenge.records,
                        (missing,) + dataset.records[1:], dataset.records[::-1]):
            with self.assertRaises(ContractError):
                validate_score_dataset(records, manifest, a, v, split="val", partition="primary", source_reference="fixture")

    def test_reconstructed_targets_and_bounds_revalidated_at_fit_and_report(self):
        dataset = tie_dataset()
        frozen = freeze_threshold(select_threshold(dataset, method="top1"))
        record0 = dataset.records[0]
        for value in (-.1, 1.1, np.nan, np.inf, True):
            bad_record = raw_like(record0, scores=((METHODS[0], value),) + record0.scores[1:])
            bad = raw_like(dataset, records=(bad_record,) + dataset.records[1:])
            with self.assertRaises(ContractError):
                select_threshold(bad, method="top1")
            with self.assertRaises(ContractError):
                evaluate_primary(bad, frozen)
        bad_pair = raw_like(record0.pair, pair_type=99)
        with self.assertRaises(ContractError):
            select_threshold(raw_like(dataset, records=(raw_like(record0, pair=bad_pair),) + dataset.records[1:]), method="top1")

    def test_mutable_input_list_is_snapshotted_and_subclass_rejected(self):
        dataset, manifest, a, v = fixture()
        records = list(dataset.records)
        snapshot = validate_score_dataset(records, manifest, a, v, split="val", partition="primary", source_reference="fixture")
        records.clear()
        self.assertEqual(len(snapshot.records), 4)
        class MaliciousList(list):
            pass
        with self.assertRaises(ContractError):
            validate_score_dataset(MaliciousList(dataset.records), manifest, a, v, split="val", partition="primary", source_reference="fixture")

    def test_no_target_leakage_or_scorer_changes_from_threshold_selection(self):
        dataset, manifest, a, v = fixture(positives=None)
        data = extract_pair_probabilities(manifest, a, v, split="val")
        before = score_probabilities(data.primary.features.audio, data.primary.features.visual)
        select_threshold(dataset, method="top1")
        after = score_probabilities(data.primary.features.audio, data.primary.features.visual)
        for method in METHODS:
            np.testing.assert_array_equal(before.pair_scores[method], after.pair_scores[method])
        empty = next(r for r in dataset.records if r.pair.audio.record.end == 0)
        self.assertEqual(empty.pair.semantic_compatibility.value, "UNKNOWN")
        self.assertFalse(empty.pair.event_supervision.semantic_eligible)

    def test_source_overlap_and_missing_metadata_warnings_preserved(self):
        rows = (record("a", 1), record("va", 1), record("vb", 0), record("vc", 0))
        dataset, _, _, _ = fixture(metadata={"train": ROWS, "val": rows})
        report = evaluate_primary(dataset, freeze_threshold(select_threshold(dataset, method="top1")))
        self.assertTrue(any("1 shared source videos across train&val" in w for w in report.context.warnings))
        self.assertTrue(any("unavailable for test" in w for w in report.context.warnings))
        self.assertTrue(any("encoder exposure" in w for w in report.context.warnings))


class ReliabilityAndReportTests(unittest.TestCase):
    def test_fixed_bins_boundaries_composition_and_empty_bins(self):
        dataset, _, _, _ = fixture(positives=None, negatives=2, scores=[i / 10 for i in range(11)])
        bins = score_reliability(dataset, method="top1")
        self.assertEqual(SCORE_BIN_EDGES, tuple(i / 10 for i in range(11)))
        self.assertEqual(tuple(b.count for b in bins), (1,) * 9 + (2,))
        self.assertEqual(sum(b.positive_count for b in bins), sum(r.target for r in dataset.records))
        self.assertEqual(sum(b.negative_count for b in bins), 2)
        self.assertFalse(any(b.upper_inclusive for b in bins[:-1]))
        self.assertTrue(bins[-1].upper_inclusive)
        for b in bins:
            self.assertEqual(b.observed_original_pair_frequency, b.positive_count / b.count)
        constant, _, _, _ = fixture(scores=[.1] * 4)
        bins = score_reliability(constant, method="top1")
        self.assertEqual(bins[1].count, 4)
        self.assertIsNone(bins[0].mean_score)
        self.assertIsNone(bins[0].observed_original_pair_frequency)

    def test_reliability_and_serialized_reports_have_no_raw_score_ece_or_brier(self):
        dataset = tie_dataset()
        policy = freeze_threshold(select_threshold(dataset, method="top1"))
        report = evaluate_primary(dataset, policy)
        self.assertEqual(report.context.score_meaning, SCORE_MEANING)
        payload = json.loads(evaluation_report_to_json(report))
        self.assertNotIn("ece", payload)
        self.assertNotIn("brier", payload)
        self.assertEqual([f.name for f in fields(report.score_bins[0])][-1], "observed_original_pair_frequency")

    def test_frozen_primary_report_counts_distributions_and_validation_reuse(self):
        dataset = tie_dataset()
        frozen = freeze_threshold(select_threshold(dataset, method="top1"))
        report = evaluate_primary(dataset, frozen)
        self.assertEqual(report.classification.counts, ConfusionCounts(1, 2, 0, 1))
        self.assertEqual(report.context.fitting_pair_reuse_count, 4)
        self.assertGreater(report.context.fitting_source_video_reuse_count, 0)
        self.assertIn("no independent generalization", report.context.validation_reuse_note)
        self.assertEqual(report.threshold_independent.positive_scores.count, 2)
        self.assertEqual(report.threshold_independent.negative_scores.count, 2)
        self.assertEqual(report.context.frozen_policy_fingerprint, frozen.policy_fingerprint)

    def test_primary_and_challenge_reports_cannot_be_pooled(self):
        dataset = tie_dataset()
        frozen = freeze_threshold(select_threshold(dataset, method="top1"))
        challenge, _, _, _ = fixture(partition="challenge")
        with self.assertRaises(ContractError):
            evaluate_primary(challenge, frozen)
        with self.assertRaises(ContractError):
            evaluate_challenge(dataset, frozen)

    def test_reconstructed_positive_challenge_label_and_split_spoof_rejected(self):
        dataset = tie_dataset()
        frozen = freeze_threshold(select_threshold(dataset, method="top1"))
        challenge, _, _, _ = fixture(partition="challenge")
        positive = next(r for r in dataset.records if r.target == 1)
        bad_record = raw_like(positive, partition="challenge", __post_init__=lambda: None)
        bad = raw_like(challenge, records=(bad_record,) + challenge.records[1:])
        with self.assertRaises(ContractError):
            evaluate_challenge(bad, frozen)
        for split in ("train", "test"):
            with self.assertRaises(ContractError):
                evaluate_challenge(raw_like(challenge, split=split), frozen)

    def test_challenge_false_acceptance_rejection_and_threshold_equality(self):
        frozen = freeze_threshold(select_threshold(tie_dataset(), method="top1"))
        challenge, _, _, _ = fixture(partition="challenge", challenges=4, scores=[.4, .39, .9, .1])
        report = evaluate_challenge(challenge, frozen)
        self.assertEqual((report.count, report.positive_count, report.negative_count), (4, 0, 4))
        self.assertEqual((report.classified_as_original, report.rejected), (2, 2))
        self.assertEqual((report.false_acceptance_rate, report.rejection_rate), (.5, .5))
        self.assertEqual(report.context.threshold, .4)
        self.assertEqual(report.context.method, "top1")
        self.assertEqual(report.score_distribution.count, 4)
        self.assertFalse(hasattr(report, "roc_auc"))
        self.assertFalse(hasattr(report, "f1"))

    def test_empty_challenges_have_explicit_undefined_rates(self):
        frozen = freeze_threshold(select_threshold(tie_dataset(), method="top1"))
        challenge, _, _, _ = fixture(partition="challenge", challenges=0)
        report = evaluate_challenge(challenge, frozen)
        self.assertEqual((report.count, report.classified_as_original, report.rejected), (0, 0, 0))
        self.assertIsNone(report.false_acceptance_rate)
        self.assertIsNone(report.rejection_rate)
        self.assertIsNone(report.score_distribution.mean)
        self.assertTrue(all(b.count == 0 for b in report.score_bins))

    def test_freeze_required_and_test_split_unreachable(self):
        dataset = tie_dataset()
        policy = select_threshold(dataset, method="top1")
        with self.assertRaises(ContractError):
            evaluate_primary(dataset, policy)
        frozen = freeze_threshold(policy)
        test = raw_like(dataset, split="test", __post_init__=lambda: None)
        for operation in (lambda: evaluate_primary(test, frozen), lambda: evaluate_challenge(test, frozen),
                          lambda: score_reliability(test, method="top1")):
            with self.assertRaises(ContractError):
                operation()

    def test_train_diagnostics_do_not_fit_threshold(self):
        frozen = freeze_threshold(select_threshold(tie_dataset(), method="top1"))
        train, _, _, _ = fixture(split="train")
        report = evaluate_primary(train, frozen)
        self.assertEqual(report.context.split, "train")
        self.assertEqual(report.context.threshold, frozen.policy.threshold)
        self.assertIn("not held-out", report.context.validation_reuse_note)
        self.assertEqual(report.context.fitting_pair_reuse_count, 0)

    def test_threshold_sensitivity_is_complete_validation_only_sweep(self):
        policy = select_threshold(tie_dataset(), method="top1")
        sensitivity = threshold_sensitivity(policy)
        points = sensitivity.validation_operating_points
        self.assertEqual(len(points), policy.candidate_count)
        self.assertEqual(points[0].metrics.positive_prediction_rate, 1.)
        self.assertEqual(points[-1].metrics.positive_prediction_rate, 0.)
        self.assertIn("not resampling stability", sensitivity.interpretation)
        self.assertTrue(all(p.metrics.count == 4 for p in points))

    def test_reproducible_json_reports_and_frozen_evidence(self):
        dataset = tie_dataset()
        frozen = freeze_threshold(select_threshold(dataset, method="top1"))
        first = evaluation_report_to_json(evaluate_primary(dataset, frozen))
        second = evaluation_report_to_json(evaluate_primary(dataset, frozen))
        self.assertEqual(first.encode(), second.encode())
        self.assertTrue(first.endswith("\n"))
        self.assertEqual(json.loads(first)["context"]["split"], "val")
        encoded = frozen_policy_to_json(frozen)
        self.assertEqual(encoded, frozen_policy_to_json(freeze_threshold(frozen.policy)))
        self.assertEqual(json.loads(encoded)["policy"]["source_fingerprint"], dataset.fingerprint)


if __name__ == "__main__":
    unittest.main()
