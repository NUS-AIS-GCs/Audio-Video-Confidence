"""Phase 7 deterministic synthetic TRAIN fixtures; no real-data readers or runs."""
from dataclasses import replace
import json
from unittest.mock import patch
import unittest

import numpy as np

from fusion.contracts import ContractError, Record
from fusion.baseline_metrics import ranking_metrics
from fusion.baseline_thresholds import classification_metrics
from fusion.evaluation_contracts import (
    CorrespondencePrediction, EvaluationProvenance, FROZEN_MODELS, FrozenOperatingPoint,
    PredictionSet, align_prediction_sets, canonical_json,
)
from fusion.evaluation_metrics import evaluate_predictions
from fusion.pair_contracts import (
    PairRecord, PairSource, PairType, SamplingPolicy, SamplingProvenance, split_seed,
)


def pair(index, target, split="train"):
    policy = SamplingPolicy(42)
    provenance = SamplingProvenance(42, split_seed(42, split), policy.fingerprint, "a" * 64, "b" * 64)
    audio = Record(f"audio{index}__00", f"audio{index}", "Event 00", 0, 10, 0)
    visual = audio if target else Record(f"visual{index}__01", f"visual{index}", "Event 01", 0, 10, 1)
    kind = PairType.ORIGINAL_POSITIVE if target else PairType.CROSS_CLASS_EASY_NEGATIVE
    return PairRecord(PairSource(split, audio, index), PairSource(split, visual, index),
                      kind, policy, provenance)


def fixture(targets=(0, 0, 1, 1), scores=(0.1, 0.4, 0.35, 0.8), *, kind="uncalibrated_sigmoid"):
    provenance = EvaluationProvenance("train", "primary", "c" * 64, "d" * 64,
                                      "synthetic TRAIN scores", "e" * 64, "synthetic overlap evidence")
    data = PredictionSet("synthetic", "f" * 64, kind, provenance,
                         tuple(CorrespondencePrediction(pair(i, y), y, float(s))
                               for i, (y, s) in enumerate(zip(targets, scores))))
    point = FrozenOperatingPoint("synthetic", "f" * 64, 0.4, kind,
                                 "synthetic frozen validation policy", "1" * 64, "2" * 64)
    return data, point


def evaluate(targets, scores, **kwargs):
    return evaluate_predictions(*fixture(targets, scores, **kwargs))


class EvaluationMetricTests(unittest.TestCase):
    def test_perfect_classification(self):
        result = evaluate((0, 0, 1, 1), (0.0, 0.1, 0.8, 1.0))
        for value in result["metrics"].values():
            self.assertEqual(value, {"value": 1.0, "undefined_reason": None})
        self.assertAlmostEqual(result["reliability"]["brier_score"]["value"], 0.0125)

    def test_completely_incorrect_classification(self):
        result = evaluate((0, 0, 1, 1), (0.8, 1.0, 0.0, 0.1))
        for metric in ("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"):
            self.assertEqual(result["metrics"][metric]["value"], 0.0)
        self.assertAlmostEqual(result["metrics"]["average_precision"]["value"], 5 / 12)

    def test_mixed_predictions(self):
        result = evaluate((0, 0, 1, 1), (0.1, 0.4, 0.35, 0.8))
        for name in ("accuracy", "precision", "recall", "f1", "macro_f1"):
            self.assertEqual(result["metrics"][name]["value"], 0.5)
        self.assertEqual(result["support"], {"positive": 2, "negative": 2})

    def test_class_imbalance_and_fixed_two_class_macro(self):
        result = evaluate((0,) * 9 + (1,), (0.0,) * 10)
        self.assertEqual(result["metrics"]["accuracy"]["value"], 0.9)
        self.assertEqual(result["metrics"]["f1"]["value"], 0.0)
        self.assertAlmostEqual(result["metrics"]["macro_f1"]["value"], 9 / 19)
        self.assertEqual(result["support"], {"positive": 1, "negative": 9})

    def test_single_negative_class(self):
        result = evaluate((0, 0), (0.0, 0.1))
        self.assertEqual(result["metrics"]["accuracy"]["value"], 1)
        self.assertEqual(result["metrics"]["macro_f1"],
                         {"value": 0.5, "undefined_reason": "undefined_class_f1_zero_filled"})
        self.assertEqual(result["metrics"]["recall"]["undefined_reason"], "no_positive_targets")
        self.assertEqual(result["metrics"]["f1"]["undefined_reason"], "no_positive_targets_or_predictions")

    def test_single_positive_class(self):
        result = evaluate((1, 1), (0.8, 1.0))
        self.assertEqual(result["metrics"]["f1"]["value"], 1)
        self.assertEqual(result["metrics"]["macro_f1"]["value"], 0.5)

    def test_one_class_ranking_and_curves_explicitly_unavailable(self):
        for targets in ((0, 0), (1, 1)):
            result = evaluate(targets, (0.2, 0.8))
            for name in ("roc_auc", "average_precision"):
                self.assertEqual(result["metrics"][name],
                                 {"value": None, "undefined_reason": "only_one_target_class"})
            for curve in result["curves"].values():
                self.assertEqual(curve, {"points": [], "undefined_reason": "only_one_target_class"})
            canonical_json(result)

    def test_precision_undefined_is_zero_with_reason(self):
        result = evaluate((0, 1), (0.0, 0.1))
        self.assertEqual(result["metrics"]["precision"],
                         {"value": 0.0, "undefined_reason": "no_positive_predictions"})
        self.assertIsNone(result["metrics"]["f1"]["undefined_reason"])

    def test_confusion_orientation_asymmetric(self):
        result = evaluate((0, 0, 0, 1, 1), (0.8, 0.9, 0.1, 0.8, 0.1))
        self.assertEqual(result["confusion_matrix"], {
            "labels": [0, 1], "rows": "true", "columns": "predicted",
            "values": [[1, 2], [1, 1]], "counts": {"tp": 1, "tn": 1, "fp": 2, "fn": 1}})

    def test_threshold_equality_is_positive_and_next_float_below_is_negative(self):
        data, point = fixture((1, 1, 0), (0.4, np.nextafter(0.4, 0), 0.4))
        result = evaluate_predictions(data, point)
        self.assertEqual([s["predicted_target"] for s in result["samples"]], [1, 0, 1])
        self.assertEqual(result["operating_point"]["threshold"], 0.4)

    def test_roc_and_pr_hand_calculations(self):
        result = evaluate((0, 0, 1, 1), (0.1, 0.4, 0.35, 0.8))
        self.assertEqual(result["metrics"]["roc_auc"]["value"], 0.75)
        self.assertAlmostEqual(result["metrics"]["average_precision"]["value"], 5 / 6)
        roc = result["curves"]["roc"]["points"]
        pr = result["curves"]["precision_recall"]["points"]
        self.assertEqual([(p["fpr"], p["tpr"]) for p in roc],
                         [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 1), (1, 1)])
        self.assertEqual([(p["recall"], p["precision"]) for p in pr],
                         [(0, 1), (0.5, 1), (0.5, 0.5), (1, 2 / 3), (1, 0.5)])
        self.assertIsNone(roc[0]["threshold"])
        self.assertTrue(roc[0]["above_maximum"])
        self.assertEqual([p["threshold"] for p in roc[1:]], [0.8, 0.4, 0.35, 0.1])

    def test_tied_scores_grouped_without_order_dependence(self):
        result = evaluate((0, 1, 0, 1), (0.5,) * 4)
        self.assertEqual(result["metrics"]["roc_auc"]["value"], 0.5)
        self.assertEqual(result["metrics"]["average_precision"]["value"], 0.5)
        self.assertEqual(len(result["curves"]["roc"]["points"]), 2)
        self.assertEqual(result["curves"]["roc"]["points"][-1]["fpr"], 1)

    def test_legacy_metric_functions_are_reused_exactly(self):
        data, point = fixture()
        with patch("fusion.evaluation_metrics.classification_metrics", wraps=classification_metrics) as c:
            with patch("fusion.evaluation_metrics.ranking_metrics", wraps=ranking_metrics) as r:
                result = evaluate_predictions(data, point)
        c.assert_called_once()
        r.assert_called_once()
        legacy = classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.4)
        for name in ("accuracy", "precision", "recall", "f1", "macro_f1"):
            self.assertEqual(result["metrics"][name]["value"], getattr(legacy, name))

    def test_uncalibrated_sigmoid_is_explicit(self):
        result = evaluate((0, 1), (0.2, 0.8))
        self.assertEqual(result["reliability"]["score_kind"], "uncalibrated_sigmoid")
        self.assertIn("not calibration proof", result["reliability"]["interpretation"])
        self.assertAlmostEqual(result["reliability"]["brier_score"]["value"], 0.04)

    def test_semantic_similarity_has_no_brier_probability_claim(self):
        result = evaluate((0, 1), (0.2, 0.8), kind="semantic_similarity")
        self.assertEqual(result["reliability"]["brier_score"],
                         {"value": None, "undefined_reason": "score_not_a_probability"})

    def test_reliability_boundaries_and_empty_bins(self):
        result = evaluate((0, 1, 1, 0), (0.0, 0.1, 0.9, 1.0))
        bins = result["reliability"]["bins"]
        self.assertEqual([b["count"] for b in bins], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertIsNone(bins[2]["mean_score"])
        self.assertIsNone(bins[2]["observed_positive_frequency"])
        self.assertEqual(bins[9]["observed_positive_frequency"], 0.5)
        self.assertTrue(bins[9]["upper_inclusive"])
        self.assertFalse(bins[0]["upper_inclusive"])

    def test_deterministic_json_and_input_permutation(self):
        data, point = fixture()
        expected = canonical_json(evaluate_predictions(data, point))
        reversed_data = replace(data, records=data.records[::-1])
        self.assertEqual(data.fingerprint, reversed_data.fingerprint)
        self.assertEqual(canonical_json(evaluate_predictions(reversed_data, point)), expected)
        self.assertEqual(json.dumps(json.loads(expected), sort_keys=True, separators=(",", ":")), expected)
        self.assertNotIn("NaN", expected)
        self.assertNotIn("Infinity", expected)

    def test_numpy_source_metadata_remains_json_compatible(self):
        data, point = fixture((1,), (1.0,))
        source = data.records[0].pair.audio
        source = replace(source, row_index=np.int64(0), record=replace(source.record, label=np.int64(0)))
        record = replace(data.records[0], pair=replace(data.records[0].pair, audio=source, visual=source))
        result = evaluate_predictions(replace(data, records=(record,)), point)
        json.dumps(result, allow_nan=False)
        self.assertIs(type(result["samples"][0]["pair"]["audio"]["row_index"]), int)


class EvaluationContractTests(unittest.TestCase):
    def test_empty_input_rejected(self):
        data, _ = fixture()
        with self.assertRaisesRegex(ContractError, "nonempty"):
            replace(data, records=())

    def test_duplicate_identity_rejected(self):
        data, _ = fixture()
        with self.assertRaisesRegex(ContractError, "duplicate"):
            replace(data, records=(data.records[0], replace(data.records[0], score=0.9)))

    def test_nonfinite_scores_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ContractError):
                CorrespondencePrediction(pair(0, 1), 1, value)

    def test_invalid_score_types_and_bounds_rejected(self):
        for value in (True, 1, "0.5", -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ContractError):
                CorrespondencePrediction(pair(0, 1), 1, value)

    def test_wrong_label_encoding_and_reversed_target_rejected(self):
        for value in (True, 1.0, "1", -1, 2, 0, np.int64(1)):
            with self.subTest(value=value), self.assertRaises(ContractError):
                CorrespondencePrediction(pair(0, 1), value, 0.5)

    def test_test_and_challenge_provenance_rejected(self):
        data, _ = fixture()
        for changes in ({"split": "test"}, {"partition": "challenge"}, {"split": "TRAIN"}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(data.provenance, **changes)

    def test_mixed_record_split_rejected(self):
        data, _ = fixture()
        record = CorrespondencePrediction(pair(0, 1, "val"), 1, 0.5)
        with self.assertRaisesRegex(ContractError, "mixed splits"):
            replace(data, records=(record,))

    def test_challenge_record_cannot_be_disguised_as_primary(self):
        data, _ = fixture()
        original = pair(0, 0)
        visual = replace(original.visual, record=Record("visual0__00", "visual0", "Event 00", 0, 10, 0))
        challenge = replace(original, visual=visual, pair_type=PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE)
        with self.assertRaisesRegex(ContractError, "challenge"):
            replace(data, records=(CorrespondencePrediction(challenge, 0, 0.5),))

    def test_model_artifact_and_interpretation_must_match_threshold(self):
        data, point = fixture()
        for changes in ({"model_id": "other"}, {"artifact_sha256": "0" * 64},
                        {"score_kind": "semantic_similarity"}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                evaluate_predictions(replace(data, **changes), point)

    def test_threshold_provenance_and_positive_direction_required(self):
        _, point = fixture()
        for changes in ({"threshold": float("nan")}, {"threshold": True}, {"positive_target": 0},
                        {"positive_target": True}, {"decision_rule": "score > threshold"},
                        {"fitting_split": "test"}, {"fitting_partition": "challenge"},
                        {"evidence_reference": ""}, {"evidence_sha256": "bad"}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(point, **changes)

    def test_historical_pins_and_no_threshold_transfer(self):
        expected = {"JSD": 0.32543530977887525, "E3": 0.009579099009671126,
                    "A2": 0.0026640388978268056}
        data, _ = fixture()
        for name, threshold in expected.items():
            point = FROZEN_MODELS[name]
            self.assertEqual(point.threshold, threshold)
            candidate = replace(data, model_id=name, artifact_sha256=point.artifact_sha256,
                                score_kind=point.score_kind)
            # Public synthetic construction can no longer claim frozen-model attribution.
            with self.assertRaisesRegex(ContractError, "authenticated"):
                evaluate_predictions(candidate, point)
            with self.assertRaises(ContractError):
                evaluate_predictions(candidate, replace(point, threshold=0.5))
        with self.assertRaises(TypeError):
            FROZEN_MODELS["E3"] = FROZEN_MODELS["A2"]

    def test_calibrated_probability_requires_explicit_evidence(self):
        data, point = fixture()
        with self.assertRaises(ContractError):
            replace(point, score_kind="calibrated_probability")
        calibrated = replace(point, score_kind="calibrated_probability",
                             calibration_reference="synthetic independent calibration evidence", calibration_sha256="3" * 64)
        result = evaluate_predictions(replace(data, score_kind="calibrated_probability"), calibrated)
        self.assertEqual(result["reliability"]["score_kind"], "calibrated_probability")
        with self.assertRaises(ContractError):
            replace(point, calibration_reference="invented")

    def test_no_model_execution_io_fitting_or_tracker_needed(self):
        data, point = fixture()
        with patch("builtins.open", side_effect=AssertionError("no file reads")):
            with patch("fusion.baseline_thresholds.select_threshold", side_effect=AssertionError("no fitting")):
                result = evaluate_predictions(data, point)
        self.assertEqual(result["operating_point"]["threshold"], point.threshold)

    def test_revalidate_reconstructed_inputs(self):
        data, point = fixture()
        forged = object.__new__(EvaluationProvenance)
        forged.__dict__.update(vars(data.provenance) | {"split": "test"})
        forged_data = object.__new__(PredictionSet)
        forged_data.__dict__.update(vars(data) | {"provenance": forged})
        with self.assertRaises(ContractError):
            evaluate_predictions(forged_data, point)

    def test_alignment_uses_identities_not_input_order(self):
        data, _ = fixture()
        other = replace(data, model_id="other", records=tuple(replace(r, score=0.3) for r in data.records[::-1]))
        aligned = align_prediction_sets((data, other))
        self.assertEqual(aligned["targets"], [r.target for r in data.ordered_records()])
        self.assertEqual(aligned["models"]["other"]["scores"], [0.3] * 4)
        self.assertEqual(canonical_json(aligned), canonical_json(align_prediction_sets((other, data))))

    def test_alignment_rejects_missing_or_duplicate_models(self):
        data, _ = fixture()
        for datasets in ((data,), (data, data), (data, replace(data, model_id="other", records=data.records[:-1]))):
            with self.subTest(datasets=len(datasets)), self.assertRaises(ContractError):
                align_prediction_sets(datasets)

    def test_alignment_rejects_manifest_split_or_overlap_mismatch(self):
        data, _ = fixture()
        for field in ("pair_manifest_sha256", "source_overlap_sha256"):
            other = replace(data, model_id="other", provenance=replace(data.provenance, **{field: "0" * 64}))
            with self.subTest(field=field), self.assertRaises(ContractError):
                align_prediction_sets((data, other))
        other = replace(data, model_id="other", provenance=replace(data.provenance, split="val"),
                        records=tuple(CorrespondencePrediction(pair(i, r.target, "val"), r.target, r.score)
                                      for i, r in enumerate(data.records)))
        with self.assertRaises(ContractError):
            align_prediction_sets((data, other))

    def test_alignment_rejects_changed_source_rows_under_same_identity(self):
        data, _ = fixture()
        first = data.records[0]
        changed = replace(first, pair=replace(first.pair, audio=replace(first.pair.audio, row_index=99)))
        other = replace(data, model_id="other", records=(changed,) + data.records[1:])
        with self.assertRaisesRegex(ContractError, "source/split"):
            align_prediction_sets((data, other))


if __name__ == "__main__":
    unittest.main()
