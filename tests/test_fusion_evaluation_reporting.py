"""Deterministic plot mapping and synthetic authenticated TRAIN report tests.

Small pinned fixtures reuse the audited admission test producer; no real TEST,
challenge or validation predictions, model training or tracking are involved.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from fusion.contracts import ContractError
from fusion import evaluation_plots as plots
from fusion import evaluation_reporting as reporting
from fusion.evaluation_admission import AuthenticatedPredictionSet, admit_archived_train_predictions
from fusion.evaluation_contracts import canonical_json
from test_fusion_evaluation import fixture
from test_fusion_evaluation_admission import evidence


def handles(archive):
    return tuple(admit_archived_train_predictions(name, repository_root=archive.root)
                 for name in reporting.MODEL_ORDER)


def forged(handle, dataset):
    result = object.__new__(AuthenticatedPredictionSet)
    for field, value in (("_dataset", dataset), ("_root", handle._root),
                         ("_fingerprint", dataset.fingerprint), ("_files", handle._files)):
        object.__setattr__(result, field, value)
    return result


class PlotMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            cls.report = reporting.build_train_comparison(handles(archive))
        cls.mpl, cls.Figure, cls.Canvas = plots._load_backend()

    def axes(self):
        fig = self.Figure(figsize=(7, 4))
        self.Canvas(fig)
        return fig.subplots()

    def test_roc_coordinates_and_axis_orientation(self):
        ax = self.axes()
        plots._draw_curve(ax, self.report["models"], curve="roc")
        for line, model in zip(ax.lines, self.report["models"]):
            points = model["curves"]["roc"]["points"]
            self.assertEqual(list(line.get_xdata()), [p["fpr"] for p in points])
            self.assertEqual(list(line.get_ydata()), [p["tpr"] for p in points])
            self.assertEqual(line.get_linestyle(), "None")
        self.assertEqual(ax.get_xlabel(), "False positive rate")
        self.assertEqual(ax.get_ylabel(), "True positive rate (recall)")

    def test_pr_coordinates_and_axis_orientation(self):
        ax = self.axes()
        plots._draw_curve(ax, self.report["models"], curve="precision_recall")
        for line, model in zip(ax.lines, self.report["models"]):
            points = model["curves"]["precision_recall"]["points"]
            self.assertEqual(list(line.get_xdata()), [p["recall"] for p in points])
            self.assertEqual(list(line.get_ydata()), [p["precision"] for p in points])
        self.assertIn("Recall", ax.get_xlabel())
        self.assertIn("Precision", ax.get_ylabel())

    def test_confusion_values_and_label_order(self):
        ax = self.axes()
        model = self.report["models"][0]
        plots._draw_confusion(ax, model, self.report["count"])
        self.assertEqual(ax.images[0].get_array().tolist(), model["confusion_matrix"]["values"])
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["0 substituted", "1 original"])
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["0 substituted", "1 original"])
        self.assertEqual(ax.get_xlabel(), "Predicted label")
        self.assertEqual(ax.get_ylabel(), "True label")

    def test_confusion_wrong_orientation_rejected(self):
        model = deepcopy(self.report["models"][0])
        model["confusion_matrix"]["rows"] = "predicted"
        with self.assertRaises(ContractError):
            plots._draw_confusion(self.axes(), model, 4)

    def test_reliability_maps_only_nonempty_bins(self):
        model = self.report["models"][1]
        ax = self.axes()
        plots._draw_reliability(ax, model)
        nonempty = [b for b in model["reliability"]["bins"] if b["count"]]
        self.assertEqual(list(ax.lines[0].get_xdata()), [b["mean_score"] for b in nonempty])
        self.assertEqual(list(ax.lines[0].get_ydata()), [b["observed_positive_frequency"] for b in nonempty])
        self.assertIn("uncalibrated", ax.get_xlabel())
        self.assertIn("diagnostic", ax.get_title())

    def test_jsd_reliability_has_no_probability_diagonal(self):
        ax = self.axes()
        plots._draw_reliability(ax, self.report["models"][0])
        self.assertEqual(len(ax.lines), 1)
        self.assertIn("not probability", ax.get_xlabel())

    def test_undefined_curves_are_annotated_without_fake_points(self):
        model = deepcopy(self.report["models"][0])
        model["curves"]["roc"] = {"points": [], "undefined_reason": "only_one_target_class"}
        model["metrics"]["roc_auc"] = {"value": None, "undefined_reason": "only_one_target_class"}
        ax = self.axes()
        plots._draw_curve(ax, [model], curve="roc")
        self.assertEqual(list(ax.lines[0].get_xdata()), [])
        self.assertTrue(any("only_one_target_class" in t.get_text() for t in ax.texts))
        self.assertIn("undefined", ax.lines[0].get_label())

    def test_metrics_table_preserves_undefined_reasons_and_thresholds(self):
        ax = self.axes()
        plots._draw_table(ax, self.report)
        texts = [cell.get_text().get_text() for cell in ax.tables[0].get_celld().values()]
        self.assertIn("undefined: score_not_a_probability", texts)
        for model in self.report["models"]:
            self.assertIn(repr(model["threshold_provenance"]["threshold"]), texts)

    def test_zero_filled_undefined_metric_is_marked(self):
        self.assertEqual(plots._format_metric({"value": 0.0, "undefined_reason": "no_positive_predictions"}), "0.000000 *")

    def test_missing_plot_dependency_has_explicit_error(self):
        with patch("fusion.evaluation_plots.importlib.import_module", side_effect=ModuleNotFoundError("matplotlib")):
            with self.assertRaisesRegex(ContractError, "Matplotlib is required"):
                plots._load_backend()

    def test_unreadable_plot_dependency_has_explicit_error(self):
        with patch("fusion.evaluation_plots.importlib.import_module", side_effect=PermissionError("package unreadable")):
            with self.assertRaisesRegex(ContractError, "runtime access"):
                plots._load_backend()


class ReportingIntegrityTests(unittest.TestCase):
    def test_valid_authenticated_train_comparison_and_exact_thresholds(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            report = reporting.build_train_comparison(handles(archive))
            self.assertEqual(report["evidence_authenticity"], "authenticated_archived_train")
            self.assertEqual(report["split"], "train")
            self.assertEqual(report["count"], 4)
            self.assertEqual(report["support"], {"positive": 2, "negative": 2})
            for model in report["models"]:
                self.assertEqual(model["threshold_provenance"]["threshold"], archive.pins[model["model_id"]].threshold)

    def test_metric_results_are_copied_from_existing_evaluator(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            data = handles(archive)
            actual = reporting.build_train_comparison(data)
            from fusion.evaluation_metrics import evaluate_predictions
            for handle, model in zip(data, actual["models"]):
                expected = evaluate_predictions(handle, archive.pins[handle.dataset.model_id])
                for field in ("metrics", "curves", "confusion_matrix", "reliability"):
                    self.assertEqual(model[field], expected[field])

    def test_report_json_is_portable_and_strict(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            report = reporting.build_train_comparison(handles(archive))
            encoded = canonical_json(report)
            self.assertEqual(json.loads(encoded), report)
            self.assertNotIn(str(archive.root), encoded)
            self.assertNotIn("evidence_reference", encoded)
            self.assertNotIn("score_source_reference", encoded)
            self.assertNotIn("NaN", encoded)
            self.assertNotIn("Infinity", encoded)

    def test_fixed_model_order_and_no_data_dependent_selection(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            data = handles(archive)
            first = reporting.build_train_comparison(data)
            second = reporting.build_train_comparison(data[::-1])
            self.assertEqual(canonical_json(first), canonical_json(second))
            self.assertEqual(first["model_order"], ["JSD", "E3", "A2"])
            self.assertFalse(first["model_selection_performed"])
            self.assertNotIn("winner", first)

    def test_wrong_and_duplicate_model_names_fail_before_metrics(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            data = handles(archive)
            with patch.object(reporting, "evaluate_predictions") as metrics:
                for bad in ((data[0], data[0], data[2]),
                            (forged(data[0], replace(data[0].dataset, model_id="unknown")),) + data[1:]):
                    with self.assertRaises(ContractError):
                        reporting.build_train_comparison(bad)
                metrics.assert_not_called()

    def test_incomplete_admission_fails_closed(self):
        with self.assertRaises(ContractError):
            reporting.build_train_comparison(tuple(object.__new__(AuthenticatedPredictionSet) for _ in range(3)))

    def test_wrong_checkpoint_fails_before_any_metrics(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            data = handles(archive)
            bad = forged(data[1], replace(data[1].dataset, artifact_sha256="0" * 64))
            with patch.object(reporting, "evaluate_predictions") as metrics:
                with self.assertRaises(ContractError):
                    reporting.build_train_comparison((data[0], bad, data[2]))
                metrics.assert_not_called()

    def test_modified_archive_fails_before_metrics(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            data = handles(archive)
            path = archive.root / data[1].dataset.provenance.score_source_reference
            path.write_bytes(path.read_bytes() + b"changed")
            with patch.object(reporting, "evaluate_predictions") as metrics:
                with self.assertRaises(ContractError):
                    reporting.build_train_comparison(data)
                metrics.assert_not_called()

    def test_missing_evidence_production_entry_fails_without_output(self):
        with evidence() as archive:
            (archive.root / "docs/fusion_cross_attention_phase5_freeze.json").unlink()
            with self.assertRaises(ContractError):
                reporting.generate_train_report(repository_root=archive.root,
                                                output_directory="_local/phase7_reporting/missing")
            self.assertFalse((archive.root / "_local/phase7_reporting").exists())

    def test_mismatched_pair_identities_reject(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            data = handles(archive)
            bad = forged(data[1], replace(data[1].dataset, records=data[1].dataset.records[:-1]))
            with self.assertRaises(ContractError):
                reporting.build_train_comparison((data[0], bad, data[2]))

    def test_mixed_and_exploratory_inputs_cannot_claim_comparison(self):
        with evidence() as archive:
            data = handles(archive)
            with self.assertRaisesRegex(ContractError, "exploratory/mixed"):
                reporting.build_train_comparison((data[0], data[1], fixture()[0]))

    def test_validation_test_and_challenge_are_rejected_before_admission(self):
        with patch.object(reporting, "admit_archived_train_predictions") as admit:
            for split, partition in (("val", "primary"), ("test", "primary"), ("train", "challenge")):
                with self.assertRaises(ContractError):
                    reporting.generate_train_report(repository_root="unused", output_directory="unused",
                                                    split=split, partition=partition)
            admit.assert_not_called()

    def test_output_must_be_fresh_and_inside_ignored_reporting_directory(self):
        with evidence() as archive:
            for output in ("report.json", "_local/fusion/overwrite", "../outside", archive.root / "absolute",
                           "_local/phase7_reporting"):
                with self.subTest(output=output), self.assertRaises(ContractError):
                    reporting._output_path(archive.root, output)
            existing = archive.root / "_local/phase7_reporting/existing"
            existing.mkdir(parents=True)
            with self.assertRaises(ContractError):
                reporting._output_path(archive.root, "_local/phase7_reporting/existing")

    def test_repeated_generation_has_identical_png_json_and_manifest_bytes(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            data = handles(archive)
            manifests = []
            for name in ("first", "second"):
                manifests.append(reporting.write_train_comparison(
                    data, repository_root=archive.root, output_directory="_local/phase7_reporting/" + name))
            self.assertEqual(manifests[0], manifests[1])
            first = archive.root / "_local/phase7_reporting/first"
            second = archive.root / "_local/phase7_reporting/second"
            self.assertEqual({p.name for p in first.iterdir()}, set(plots.PNG_NAMES) | {"report.json", "manifest.json"})
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            for entry in manifests[0]["files"]:
                self.assertEqual(hashlib.sha256((first / entry["name"]).read_bytes()).hexdigest(), entry["sha256"])

    def test_missing_dependency_does_not_publish_partial_report(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            with patch("fusion.evaluation_plots.importlib.import_module", side_effect=ModuleNotFoundError("matplotlib")):
                with self.assertRaisesRegex(ContractError, "Matplotlib is required"):
                    reporting.write_train_comparison(handles(archive), repository_root=archive.root,
                                                     output_directory="_local/phase7_reporting/failure")
            base = archive.root / "_local/phase7_reporting"
            self.assertFalse((base / "failure").exists())
            self.assertEqual(list(base.glob(".staging-*")), [])

    def test_plotting_failure_does_not_publish_partial_report(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            with patch.object(reporting, "_render_pngs", side_effect=RuntimeError("plot failure")):
                with self.assertRaisesRegex(ContractError, "rendering or publication failed"):
                    reporting.write_train_comparison(handles(archive), repository_root=archive.root,
                                                     output_directory="_local/phase7_reporting/failure")
            self.assertFalse((archive.root / "_local/phase7_reporting/failure").exists())
            self.assertEqual(list((archive.root / "_local/phase7_reporting").glob(".staging-*")), [])

    def test_source_change_during_rendering_prevents_publication(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            data = handles(archive)
            original = reporting._render_pngs
            def mutate(report, directory):
                result = original(report, directory)
                path = archive.root / data[1].dataset.provenance.score_source_reference
                path.write_bytes(path.read_bytes() + b"changed after plot")
                return result
            with patch.object(reporting, "_render_pngs", side_effect=mutate):
                with self.assertRaises(ContractError):
                    reporting.write_train_comparison(data, repository_root=archive.root,
                                                     output_directory="_local/phase7_reporting/failure")
            self.assertFalse((archive.root / "_local/phase7_reporting/failure").exists())

    def test_report_generation_does_not_write_historical_inputs(self):
        with evidence() as archive, patch.object(reporting, "FROZEN_MODELS", archive.pins):
            before = dict(archive.files)
            reporting.write_train_comparison(handles(archive), repository_root=archive.root,
                                             output_directory="_local/phase7_reporting/preserved")
            self.assertTrue(all((archive.root / path).read_bytes() == content for path, content in before.items()))


if __name__ == "__main__":
    unittest.main()
