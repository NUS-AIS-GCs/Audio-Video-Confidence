"""Offline synthetic archive fixtures; the production trust anchor is never an API argument.

Tests replace module-owned pins only inside mocks. The separate real TRAIN
admission check uses the unchanged committed freeze, without mocks or scoring.
"""
from contextlib import contextmanager
import csv
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from fusion.contracts import ContractError
from fusion import evaluation_admission as admission
from fusion.evaluation_contracts import FROZEN_MODELS, align_prediction_sets, canonical_json
from fusion.evaluation_metrics import evaluate_predictions
from test_fusion_evaluation import fixture


def digest(content):
    return hashlib.sha256(content).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class ArchiveFixture:
    """Small independent producer with a pinned transitive manifest chain."""
    def __init__(self, root, changes=None):
        self.root = root
        self.changes = changes or {}
        self.files = {}
        self.pins = {}
        self.data, _ = fixture()
        self.build()

    def put(self, path, content):
        content = self.changes.get(path, lambda b: b)(content)
        self.files[path] = content
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return digest(content)

    def csv(self, rows, fields):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode()

    def build(self):
        records = self.data.records
        export_hashes = []
        for modality, role in (("audio", "audio"), ("vision", "visual")):
            rows = []
            for record in records:
                source = getattr(record.pair, role).record
                rows.append({k: getattr(source, k) for k in ("sample_id", "clip_id", "category", "start", "end")})
            root = admission._EXPORTS[modality]
            for name, content in {
                "train_ids.csv": self.csv(rows, list(rows[0])), "SOURCE.txt": b"synthetic frozen producer",
                "README.md": b"synthetic metadata only", "class_names.json": b'["Event 00","Event 01"]',
            }.items():
                export_hashes.append([modality + "/" + name, self.put(root + "/" + name, content)])
        artifact = "c" * 64
        document = {"schema": "gc2-pair-manifest-v1", "artifact_fingerprint": artifact,
                    "config": {"split": "train"}, "policy": asdict(records[0].pair.policy),
                    "provenance": {"source_provenance": asdict(records[0].pair.provenance)},
                    "primary_pairs": [], "challenge_pairs": []}
        for record in self.data.ordered_records():
            pair = record.pair
            document["primary_pairs"].append({"audio": asdict(pair.audio), "visual": asdict(pair.visual),
                                             "pair_type": pair.pair_type.value, "identity": list(pair.identity),
                                             "original_pair_target": pair.original_pair_target})
        train_sha = self.put(admission._PAIRS, encoded(document))
        model_specs = {"E3": {"synthetic": "MLP"}, "A2": {"synthetic": "attention"}}
        manifests = {}
        frozen_models = []
        for name in ("E3", "JSD", "A2"):
            root = {"E3": admission._E3, "A2": admission._A2, "JSD": admission._P3}[name]
            original = FROZEN_MODELS[name]
            score_rows = []
            for i, record in enumerate(self.data.ordered_records()):
                pair = record.pair
                row = {"split": "train", "partition": "primary",
                       "audio_sample_id": pair.audio.record.sample_id, "visual_sample_id": pair.visual.record.sample_id,
                       "audio_row": pair.audio.row_index, "visual_row": pair.visual.row_index, "target": record.target}
                if name == "JSD":
                    row.update(pair_type=pair.pair_type.value, jensen_shannon=record.score)
                else:
                    logit = (-2., -1., 0., 2.)[i]
                    row.update(logit=logit, probability=float(np.exp(-np.logaddexp(0, -logit))))
                score_rows.append(row)
            score_name = "scores/train_primary.csv" if name == "JSD" else "train_primary_scores.csv"
            file_hashes = {score_name: self.put(root + "/" + score_name, self.csv(score_rows, list(score_rows[0])))}
            if name == "JSD":
                operating_name = "policies/jensen_shannon.json"
                operating = {"policy": {"threshold": original.threshold,
                                         "source_fingerprint": original.validation_score_fingerprint}}
                file_hashes["config.json"] = self.put(root + "/config.json", encoded({"exports": {
                    "audio": {"features": "beats_iter3", "model": "BEATs+LoRA", "run": "beats_lora_r8_v1"},
                    "vision": {"model": "ResNet-18", "run": "resnet18_v1"}}}))
            else:
                file_hashes["best.pt"] = self.put(root + "/best.pt", b"synthetic checkpoint " + name.encode())
                metadata = {"export_identities": admission._PRODUCERS, "model": model_specs[name],
                            "selected_epoch": 17 if name == "E3" else 18,
                            "data": [{"split": "train", "manifest_sha256": train_sha}]}
                file_hashes["checkpoint_metadata.json"] = self.put(root + "/checkpoint_metadata.json", encoded(metadata))
                operating_name = "frozen_operating_point.json"
                operating = {"threshold": original.threshold, "selected_epoch": metadata["selected_epoch"],
                             "validation_prediction_fingerprint": original.validation_score_fingerprint}
            file_hashes[operating_name] = self.put(root + "/" + operating_name, encoded(operating))
            if name == "A2":
                prereg = {"configuration": {"input_hashes": [
                    {"role": "phase4_e3/" + role, "sha256": sha} for role, sha in manifests["E3"].items()]}}
                file_hashes["preregistration.json"] = self.put(root + "/preregistration.json", encoded(prereg))
            manifest = {"files": [{"role": role, "sha256": sha} for role, sha in sorted(file_hashes.items())]}
            manifest_sha = self.put(root + "/output_manifest.json", encoded(manifest))
            file_hashes["output_manifest.json"] = manifest_sha
            manifests[name] = file_hashes
            self.pins[name] = replace(original,
                                      artifact_sha256=manifest_sha if name == "JSD" else file_hashes["best.pt"],
                                      evidence_sha256=file_hashes[operating_name])
            model = {"label": name, "frozen_threshold": original.threshold}
            if name == "JSD":
                model["output_manifest_sha256"] = manifest_sha
            else:
                model.update(model=model_specs[name], selected_epoch=metadata["selected_epoch"])
            frozen_models.append(model)
        freeze = {"models": frozen_models, "phase5_runs": [{"label": "A2", "evidence_sha256": manifests["A2"]}],
                  "data": {"artifact_inventory_fingerprint": artifact, "manifest_sha256": {"train": train_sha},
                           "train_primary": {"count": 4, "positives": 2}},
                  "data_provenance": [{"split": "train", "input_export_hashes": export_hashes}]}
        self.freeze_sha = self.put(admission._FREEZE_PATH, encoded(freeze))


@contextmanager
def evidence(changes=None):
    with tempfile.TemporaryDirectory() as temp:
        archive = ArchiveFixture(Path(temp), changes)
        pins = MappingProxyType(archive.pins)
        with patch.object(admission, "_FREEZE_SHA256", archive.freeze_sha), \
                patch.object(admission, "FROZEN_MODELS", pins), \
                patch("fusion.evaluation_contracts.FROZEN_MODELS", pins):
            yield archive


def change_csv(transform):
    def change(content):
        reader = csv.DictReader(io.StringIO(content.decode()))
        rows = list(reader)
        transform(rows)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode()
    return change


class AdmissionTests(unittest.TestCase):
    def test_identical_synthetic_scores_cannot_claim_e3(self):
        self.assert_spoof_rejected("E3")

    def test_identical_synthetic_scores_cannot_claim_a2(self):
        self.assert_spoof_rejected("A2")

    def test_identical_synthetic_scores_cannot_claim_jsd(self):
        self.assert_spoof_rejected("JSD")

    def assert_spoof_rejected(self, name):
        data, _ = fixture()
        pin = FROZEN_MODELS[name]
        claimed = replace(data, model_id=name, artifact_sha256=pin.artifact_sha256, score_kind=pin.score_kind)
        with self.assertRaisesRegex(ContractError, "authenticated"):
            evaluate_predictions(claimed, pin)
        with self.assertRaises(ContractError):
            align_prediction_sets((claimed, replace(data, model_id="other")))

    def test_recomputed_caller_hash_does_not_authenticate_scores(self):
        data, _ = fixture()
        pin = FROZEN_MODELS["E3"]
        changed = replace(data, model_id="E3", artifact_sha256=pin.artifact_sha256,
                          records=tuple(replace(r, score=1-r.score) for r in data.records),
                          provenance=replace(data.provenance, score_source_sha256=digest(b"changed scores")))
        with self.assertRaises(ContractError):
            evaluate_predictions(changed, pin)

    def test_unverified_e3_with_wrong_checkpoint_rejected_by_alignment(self):
        data, _ = fixture()
        for name in ("E3", "A2", "JSD"):
            with self.subTest(model=name), self.assertRaisesRegex(ContractError, "checkpoint"):
                align_prediction_sets((replace(data, model_id=name, artifact_sha256="0"*64),
                                       replace(data, model_id="other")))

    def test_synthetic_metrics_and_alignment_remain_explicitly_unverified(self):
        data, point = fixture()
        result = evaluate_predictions(data, point)
        self.assertEqual(result["trust"]["level"], "unverified_exploratory")
        self.assertIsNone(result["trust"]["authenticated_model"])
        aligned = align_prediction_sets((data, replace(data, model_id="other")))
        self.assertEqual(aligned["trust_level"], "unverified_exploratory")
        self.assertFalse(aligned["authenticated_model_comparison"])

    def test_valid_pinned_archives_admit_all_models_and_compare(self):
        with evidence() as archive:
            handles = tuple(admission.admit_archived_train_predictions(name, repository_root=archive.root)
                            for name in ("E3", "A2", "JSD"))
            for handle in handles:
                name = handle.dataset.model_id
                result = evaluate_predictions(handle, archive.pins[name])
                self.assertEqual(result["trust"]["authenticated_model"], name)
                self.assertEqual(result["trust"]["level"], "authenticated_archived_train")
                self.assertEqual(result["count"], 4)
                self.assertEqual(result["provenance"]["split"], "train")
                canonical_json(result)
            comparison = align_prediction_sets(handles)
            self.assertTrue(comparison["authenticated_model_comparison"])
            self.assertEqual(len(comparison["identities"]), 4)

    def test_authenticated_fixture_metrics_match_scalar_expectations(self):
        with evidence() as archive:
            handle = admission.admit_archived_train_predictions("E3", repository_root=archive.root)
            result = evaluate_predictions(handle, archive.pins["E3"])
            # Tiny frozen E3 cutoff predicts all four synthetic samples positive.
            self.assertEqual(result["confusion_matrix"]["values"], [[0, 2], [0, 2]])
            self.assertEqual(result["metrics"]["accuracy"]["value"], 0.5)
            self.assertAlmostEqual(result["metrics"]["f1"]["value"], 2/3)
            self.assertAlmostEqual(result["metrics"]["macro_f1"]["value"], 1/3)
            scores = [1/(1+math.exp(-z)) for z in (-2., -1., 0., 2.)]
            # Canonical fixture labels are positive, positive, negative, negative.
            expected = sum((s-y)**2 for s,y in zip(scores, (1,1,0,0)))/4
            self.assertAlmostEqual(result["reliability"]["brier_score"]["value"], expected)

    def test_direct_admission_construction_and_replace_cannot_grant_trust(self):
        with self.assertRaises(ContractError):
            admission.AuthenticatedPredictionSet()
        with evidence() as archive:
            handle = admission.admit_archived_train_predictions("E3", repository_root=archive.root)
            with self.assertRaises(ContractError):
                replace(handle, _dataset=replace(handle.dataset, model_id="A2"))

    def test_inspection_copy_does_not_retain_authority(self):
        with evidence() as archive:
            handle = admission.admit_archived_train_predictions("E3", repository_root=archive.root)
            with self.assertRaisesRegex(ContractError, "authenticated"):
                evaluate_predictions(replace(handle.dataset), archive.pins["E3"])

    def test_admitted_model_cannot_use_another_models_threshold(self):
        with evidence() as archive:
            handle = admission.admit_archived_train_predictions("E3", repository_root=archive.root)
            for point in (archive.pins["A2"], replace(archive.pins["E3"], threshold=0.5)):
                with self.assertRaises(ContractError):
                    evaluate_predictions(handle, point)

    def test_admitted_and_exploratory_comparison_cannot_mix(self):
        with evidence() as archive:
            handle = admission.admit_archived_train_predictions("E3", repository_root=archive.root)
            data, _ = fixture()
            with self.assertRaisesRegex(ContractError, "mix"):
                align_prediction_sets((handle, data))

    def test_missing_archive_fails_closed(self):
        with evidence() as archive:
            (archive.root / admission._E3 / "train_primary_scores.csv").unlink()
            with patch("fusion.evaluation_metrics.classification_metrics") as numeric:
                with self.assertRaisesRegex(ContractError, "missing authenticated"):
                    admission.admit_archived_train_predictions("E3", repository_root=archive.root)
                with self.assertRaises(ContractError):
                    evaluate_predictions(None, archive.pins["E3"])
                with self.assertRaises(ContractError):
                    align_prediction_sets((None, fixture()[0]))
                numeric.assert_not_called()

    def test_changed_scores_after_successful_admission_fail_evaluation_and_alignment(self):
        with evidence() as archive:
            e3 = admission.admit_archived_train_predictions("E3", repository_root=archive.root)
            a2 = admission.admit_archived_train_predictions("A2", repository_root=archive.root)
            p = archive.root / admission._E3 / "train_primary_scores.csv"
            p.write_bytes(p.read_bytes().replace(b"primary", b"changed", 1))
            with self.assertRaisesRegex(ContractError, "hash mismatch"):
                evaluate_predictions(e3, archive.pins["E3"])
            with self.assertRaises(ContractError):
                align_prediction_sets((e3, a2))

    def test_nested_admitted_score_change_invalidates_previous_admission(self):
        with evidence() as archive:
            handle = admission.admit_archived_train_predictions("E3", repository_root=archive.root)
            # Also defend reflective nested mutation without accepting recomputed fingerprints.
            handle.dataset.records[0].__dict__["score"] = 0.123
            with self.assertRaisesRegex(ContractError, "changed after admission"):
                evaluate_predictions(handle, archive.pins["E3"])

    def test_recomputed_handle_fingerprint_cannot_authorize_changed_scores(self):
        with evidence() as archive:
            handle = admission.admit_archived_train_predictions("E3", repository_root=archive.root)
            handle.dataset.records[0].__dict__["score"] = 0.123
            object.__setattr__(handle, "_fingerprint", handle.dataset.fingerprint)
            with self.assertRaisesRegex(ContractError, "independently authenticated"):
                evaluate_predictions(handle, archive.pins["E3"])

    def test_reconstructed_handle_with_invented_hashes_cannot_grant_trust(self):
        with evidence() as archive:
            data, _ = fixture()
            pin = archive.pins["E3"]
            data = replace(data, model_id="E3", artifact_sha256=pin.artifact_sha256)
            forged = object.__new__(admission.AuthenticatedPredictionSet)
            for field, value in (("_dataset", data), ("_root", archive.root),
                                 ("_fingerprint", data.fingerprint), ("_files", ())):
                object.__setattr__(forged, field, value)
            with self.assertRaisesRegex(ContractError, "independently authenticated"):
                evaluate_predictions(forged, pin)

    def test_incomplete_reconstructed_handle_fails_closed(self):
        with self.assertRaisesRegex(ContractError, "incomplete authenticated"):
            evaluate_predictions(object.__new__(admission.AuthenticatedPredictionSet), FROZEN_MODELS["E3"])

    def test_self_consistent_archive_rewrite_cannot_replace_trust_anchor(self):
        with evidence() as archive:
            archive.changes[admission._E3 + "/train_primary_scores.csv"] = change_csv(
                lambda rows: rows[0].update(probability="0.123"))
            archive.build()  # Recompute archive, preregistration and freeze; code pin stays old.
            with self.assertRaisesRegex(ContractError, "hash mismatch"):
                admission.admit_archived_train_predictions("E3", repository_root=archive.root)

    def assert_bad_csv(self, change, message, model="E3"):
        root = admission._E3 if model == "E3" else admission._A2
        with evidence({root + "/train_primary_scores.csv": change_csv(change)}) as archive:
            with self.assertRaisesRegex(ContractError, message):
                admission.admit_archived_train_predictions(model, repository_root=archive.root)

    def test_source_row_900_rejected_even_inside_a_synthetic_pinned_archive(self):
        self.assert_bad_csv(lambda rows: rows[0].update(audio_row="900"), "source row")

    def test_correct_scores_attached_to_wrong_identities_rejected(self):
        self.assert_bad_csv(lambda rows: rows[0].update(audio_sample_id=rows[1]["audio_sample_id"]), "identity")
        self.assert_bad_csv(lambda rows: rows[0].update(audio_sample_id=""), "identity")

    def test_reordered_scores_without_identity_remapping_rejected(self):
        def change(rows):
            rows[0]["probability"], rows[1]["probability"] = rows[1]["probability"], rows[0]["probability"]
        self.assert_bad_csv(change, "sigmoid")

    def test_reordered_complete_csv_rows_rejected(self):
        self.assert_bad_csv(lambda rows: rows.reverse(), "score ordering")

    def test_incorrect_split_rejected_before_authoritative_metrics(self):
        self.assert_bad_csv(lambda rows: rows[0].update(split="val"), "score ordering")

    def test_challenge_declaration_rejected_before_authoritative_metrics(self):
        self.assert_bad_csv(lambda rows: rows[0].update(partition="challenge"), "score ordering")

    def test_original_substituted_target_flip_rejected(self):
        self.assert_bad_csv(lambda rows: rows[0].update(target=str(1-int(rows[0]["target"]))), "target")

    def test_duplicate_identity_rejected(self):
        self.assert_bad_csv(lambda rows: rows.__setitem__(1, dict(rows[0])), "identity")

    def test_missing_sample_rejected(self):
        self.assert_bad_csv(lambda rows: rows.pop(), "missing or extra")

    def test_nonfinite_archived_logit_rejected(self):
        self.assert_bad_csv(lambda rows: rows[0].update(logit="nan"), "sigmoid")

    def test_wrong_producer_version_rejected(self):
        def change(content):
            value = json.loads(content)
            value["export_identities"][0][1] = "wrong_audio_version"
            return encoded(value)
        with evidence({admission._E3 + "/checkpoint_metadata.json": change}) as archive:
            with self.assertRaisesRegex(ContractError, "producer version"):
                admission.admit_archived_train_predictions("E3", repository_root=archive.root)

    def test_wrong_checkpoint_bytes_rejected(self):
        with evidence() as archive:
            (archive.root / admission._E3 / "best.pt").write_bytes(b"wrong checkpoint")
            with self.assertRaisesRegex(ContractError, "hash mismatch"):
                admission.admit_archived_train_predictions("E3", repository_root=archive.root)

    def test_manifest_row_itself_must_match_pinned_exporter_metadata(self):
        def change(content):
            value = json.loads(content)
            value["primary_pairs"][0]["audio"]["row_index"] = 900
            return encoded(value)
        with evidence({admission._PAIRS: change}) as archive:
            with self.assertRaisesRegex(ContractError, "source identity/row"):
                admission.admit_archived_train_predictions("E3", repository_root=archive.root)

    def test_missing_root_evidence_and_unsupported_model_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ContractError, "missing authenticated"):
                admission.admit_archived_train_predictions("E3", repository_root=root)
            with self.assertRaisesRegex(ContractError, "unknown archived model"):
                admission.admit_archived_train_predictions("unknown", repository_root=root)

    def test_loader_has_no_caller_supplied_hash_or_split_override(self):
        with tempfile.TemporaryDirectory() as root:
            for changes in ({"expected_sha256": "0"*64}, {"split": "test"}, {"split": "val"}):
                with self.assertRaises(TypeError):
                    admission.admit_archived_train_predictions("E3", repository_root=root, **changes)


if __name__ == "__main__":
    unittest.main()
