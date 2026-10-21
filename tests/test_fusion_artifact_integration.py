"""Task 2.8 synthetic file tests only; never use the local real exports."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fusion import ContractError, load_bundle
from fusion.artifact_integration import (
    OFFICIAL_FILES, manifest_from_json, manifest_to_json, run_integration,
)
from test_fusion_pair_integration import NAMES, TRAIN, row


def write_fixture(root):
    audio, vision, official = root / "audio", root / "vision", root / "official"
    missing_labels = [i for i in range(28) if i not in {r.label for r in TRAIN}]
    metadata = {
        "train": TRAIN,
        "val": (row("a", 1), row("v", 2), row("vx", 1), row("vy", 2)) + tuple(
            row(f"val{i}_{j}", i) for i in missing_labels[:12] for j in range(2)),
        "test": (row("f", 0), row("t", 1), row("tx", 0), row("ty", 1)) + tuple(
            row(f"test{i}_{j}", i) for i in missing_labels[12:] for j in range(2)),
    }
    ids = sorted({r.sample_id for rows in metadata.values() for r in rows})
    for modality, path in (("audio", audio), ("vision", vision)):
        path.mkdir()
        (path / "class_names.json").write_text(json.dumps(NAMES), encoding="utf-8")
        (path / "SOURCE.txt").write_text("synthetic file fixture only", encoding="utf-8")
        (path / "README.md").write_text("synthetic file fixture only", encoding="utf-8")
        for split, records in metadata.items():
            ordered = sorted(records, key=lambda r: r.sample_id, reverse=modality == "audio")
            if modality == "vision":
                ordered = ordered[1:] + ordered[:1]
            with (path / f"{split}_ids.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("sample_id", "clip_id", "category", "start", "end"))
                writer.writerows((r.sample_id, r.clip_id, r.category, r.start, r.end) for r in ordered)
            dim = 256 if modality == "audio" else 512
            np.save(path / f"{split}_emb.npy", np.stack([
                np.full((10, dim), ids.index(r.sample_id), dtype=np.float16) for r in ordered]))
            np.save(path / f"{split}_probs.npy", np.full((len(ordered), 10, 29), 1 / 29, dtype=np.float16))
    official.mkdir()

    def lines(records):
        return "".join(f"{r.category}&{r.clip_id}&good&{r.start}&{r.end}\n" for r in records)

    (official / "Annotations.txt").write_text(lines([r for rows in metadata.values() for r in rows]), encoding="utf-8")
    for split, filename in zip(("train", "val", "test"), OFFICIAL_FILES[1:]):
        (official / filename).write_text(lines(metadata[split]), encoding="utf-8")
    return audio, vision, official, {s: len(rows) for s, rows in metadata.items()}


class ArtifactIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio, self.vision, self.official, self.counts = write_fixture(self.root)

    def run_files(self, **kwargs):
        return run_integration(self.audio, self.vision, self.official,
                               expected_counts=self.counts, **kwargs)

    def test_happy_path_hashes_all_files_preserves_rows_and_round_trips_all_splits(self):
        output = self.root / "results"
        result = self.run_files(output_dir=output)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["binary_validation"], "VALIDATED_LOCAL_FILES")
        self.assertEqual(len(result["artifacts"]), 28)
        self.assertEqual(json.loads((output / "validation.json").read_text()), result)
        bundles = {m: load_bundle(path, m, expected_counts=self.counts)
                   for m, path in (("audio", self.audio), ("vision", self.vision))}
        for entry in result["artifacts"]:
            data = Path(entry["path"]).read_bytes()
            self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(entry["size_bytes"], len(data))
        for split, n in self.counts.items():
            self.assertEqual(result["alignment"][split]["matched"], n)
            self.assertGreater(result["alignment"][split]["different_row_indices"], 0)
            report = result["manifests"][split]
            self.assertEqual([c["selected"] for c in report["pair_type_counts"]], [n, n, 0, n])
            path = output / f"{split}_pairs.json"
            text = path.read_text(encoding="utf-8")
            manifest = manifest_from_json(text, bundles["audio"], bundles["vision"],
                                          expected_artifact_fingerprint=result["artifact_inventory_fingerprint"])
            self.assertEqual(text, manifest_to_json(manifest, bundles["audio"], bundles["vision"],
                                                  artifact_fingerprint=result["artifact_inventory_fingerprint"]))
            self.assertEqual(result["outputs"][path.name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            for pair in manifest.primary_pairs + manifest.challenge_pairs:
                for source, bundle in ((pair.audio, bundles["audio"]), (pair.visual, bundles["vision"])):
                    self.assertEqual(bundle.splits[split].records[source.row_index], source.record)
            self.assertIsNone(manifest.policy.hard_negative_policy)
        empty = next(p for p in json.loads((output / "train_pairs.json").read_text())["primary_pairs"]
                     if p["identity"][1:] == ["VWi2ENBuTbw__08"] * 2)
        self.assertEqual(empty["audio_annotation_activity"], [False] * 10)
        self.assertEqual(empty["semantic_compatibility"], "UNKNOWN")
        self.assertFalse(any(empty["event_supervision"].values()))
        self.assertEqual(result["leakage"]["audio"]["source_overlap"]["train&val"], ["a"])

    def test_deterministic_regeneration_has_identical_report_and_manifest_bytes(self):
        first, second = self.root / "one", self.root / "two"
        self.assertEqual(self.run_files(output_dir=first), self.run_files(output_dir=second))
        for path in first.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())

    def test_missing_official_metadata_blocks_pairs_but_validates_real_input_files(self):
        for name in OFFICIAL_FILES:
            (self.official / name).unlink()
        output = self.root / "blocked"
        result = self.run_files(output_dir=output)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["binary_validation"], "VALIDATED_LOCAL_FILES")
        self.assertEqual(result["official_metadata"], "NOT_VERIFIED")
        self.assertEqual({Path(p).name for p in result["missing_official_files"]}, set(OFFICIAL_FILES))
        self.assertEqual(result["manifests"], {})
        self.assertFalse(list(output.glob("*_pairs.json")))

    def test_verified_mirror_header_preserves_every_annotation_and_source_file(self):
        original = self.run_files()
        path = self.official / "Annotations.txt"
        content = "Category&VideoID&Quality&StartTime&EndTime\n" + path.read_text(encoding="utf-8")
        path.write_text(content, encoding="utf-8")
        result = self.run_files()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["alignment"], original["alignment"])
        self.assertEqual(result["manifests"], original["manifests"])
        self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_mirror_header_does_not_hide_malformed_annotation_rows(self):
        path = self.official / "Annotations.txt"
        content = "Category&VideoID&Quality&StartTime&EndTime\n" + path.read_text(encoding="utf-8")
        path.write_text(content.replace("&1&4", "&1.5&4", 1), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "integer_start"):
            self.run_files()

    def test_missing_binary_csv_mapping_or_producer_note_is_explicit_and_blocks(self):
        for filename in ("train_emb.npy", "val_probs.npy", "test_ids.csv", "class_names.json", "SOURCE.txt"):
            path = self.audio / filename
            contents = path.read_bytes()
            path.unlink()
            result = self.run_files()
            self.assertIn(str(path.resolve()), result["missing_export_files"])
            self.assertEqual(result["binary_validation"], "NOT_VALIDATED")
            self.assertEqual(result["manifests"], {})
            path.write_bytes(contents)

    def test_corrupt_truncated_and_pickle_arrays_fail(self):
        path = self.audio / "train_emb.npy"
        original = path.read_bytes()
        for data in (b"not numpy", original[:512]):
            path.write_bytes(data)
            with self.assertRaises(ContractError):
                self.run_files()
        np.save(path, np.array([{"unsafe": "object"}], dtype=object))
        with self.assertRaises(ContractError):
            self.run_files()

    def test_wrong_shapes_temporal_dimensions_row_counts_and_dtypes_fail(self):
        path = self.vision / "train_emb.npy"
        original = path.read_bytes()
        for shape, dtype in (((9, 10, 511), "float16"), ((9, 9, 512), "float16"),
                             ((8, 10, 512), "float16"), ((9, 512), "float16"),
                             ((9, 10, 512), "float32"), ((9, 10, 512), "int16")):
            with self.subTest(shape=shape, dtype=dtype):
                np.save(path, np.zeros(shape, dtype=dtype))
                with self.assertRaises(ContractError):
                    self.run_files()
        path.write_bytes(original)

    def test_nan_infinity_and_invalid_probability_distributions_fail(self):
        for name in ("train_emb.npy", "train_probs.npy"):
            path = self.audio / name
            array = np.load(path, allow_pickle=False)
            for value in (np.nan, np.inf, -np.inf):
                changed = array.copy()
                changed.flat[0] = value
                np.save(path, changed)
                with self.assertRaises(ContractError):
                    self.run_files()
            np.save(path, array)
        path = self.vision / "train_probs.npy"
        for value in (-0.1, 1.1, 0.1):
            np.save(path, np.full((9, 10, 29), value, dtype=np.float16))
            with self.assertRaises(ContractError):
                self.run_files()

    def test_duplicate_missing_and_conflicting_annotation_ids_fail(self):
        path = self.vision / "train_ids.csv"
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines()
        for text in ("\n".join(lines + [lines[1]]) + "\n", "\n".join(lines[:-1]) + "\n",
                     original.replace("a__00,a,", "changed__00,changed,")):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ContractError):
                self.run_files()

    def test_wrong_class_mapping_fails_even_when_arrays_have_correct_width(self):
        names = list(NAMES)
        names[0], names[1] = names[1], names[0]
        (self.vision / "class_names.json").write_text(json.dumps(names), encoding="utf-8")
        with self.assertRaises(ContractError):
            self.run_files()

    def test_official_interval_membership_and_vocabulary_mismatches_fail(self):
        path = self.official / "trainSet.txt"
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace("&1&4", "&2&4", 1), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "union"):
            self.run_files()
        # Change both references consistently; exporter metadata must still fail.
        annotation = self.official / "Annotations.txt"
        annotation.write_text(annotation.read_text(encoding="utf-8").replace("&1&4", "&2&4", 1), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "official annotation mismatch"):
            self.run_files()

    def test_incorrect_expected_hash_fails_and_matching_subset_is_explicit(self):
        with self.assertRaisesRegex(ContractError, "hash mismatch"):
            self.run_files(expected_hashes={"audio/train_emb.npy": "0" * 64})
        digest = hashlib.sha256((self.audio / "train_emb.npy").read_bytes()).hexdigest()
        result = self.run_files(expected_hashes={"audio/train_emb.npy": digest})
        self.assertEqual(result["trusted_checksums"], "SUPPLIED_ENTRIES_MATCH")
        self.assertEqual(result["expected_hash_roles"], ["audio/train_emb.npy"])

    def test_official_split_membership_cannot_be_inferred_from_matching_export_csvs(self):
        val, test = self.official / "valSet.txt", self.official / "testSet.txt"
        val_lines, test_lines = val.read_text().splitlines(), test.read_text().splitlines()
        val_lines[0], test_lines[0] = test_lines[0], val_lines[0]
        val.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
        test.write_text("\n".join(test_lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "official annotation mismatch"):
            self.run_files()

    def test_consistent_export_vocabulary_is_compared_with_official_vocabulary(self):
        for name in OFFICIAL_FILES:
            path = self.official / name
            path.write_text(path.read_text(encoding="utf-8").replace(NAMES[0] + "&", NAMES[0] + " wrong&"), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "mapping conflicts with official"):
            self.run_files()

    def test_input_changed_during_validation_aborts_before_publishing(self):
        from fusion import artifact_integration as module
        original = module._verify_official

        def changed(*args):
            original(*args)
            path = self.audio / "README.md"
            path.write_text("changed while validation was running", encoding="utf-8")

        output = self.root / "never_published"
        with patch.object(module, "_verify_official", side_effect=changed):
            with self.assertRaisesRegex(ContractError, "changed during integration"):
                self.run_files(output_dir=output)
        self.assertFalse(output.exists())

    def test_safe_manifest_reader_rejects_types_masks_order_and_artifact_context_tampering(self):
        output = self.root / "roundtrip"
        result = self.run_files(output_dir=output)
        a = load_bundle(self.audio, "audio", expected_counts=self.counts)
        v = load_bundle(self.vision, "vision", expected_counts=self.counts)
        text = (output / "train_pairs.json").read_text(encoding="utf-8")
        for change in ("row_type", "mask", "order", "context", "unknown", "missing"):
            data = json.loads(text)
            if change == "row_type":
                data["primary_pairs"][0]["audio"]["row_index"] = float(data["primary_pairs"][0]["audio"]["row_index"])
            elif change == "mask":
                data["primary_pairs"][0]["audio_annotation_activity"][0] = "false"
            elif change == "order":
                data["primary_pairs"].reverse()
            elif change == "context":
                data["artifact_fingerprint"] = "0" * 64
            elif change == "unknown":
                data["extra"] = "unexpected"
            else:
                del data["provenance"]
            with self.subTest(change=change), self.assertRaises(ContractError):
                manifest_from_json(json.dumps(data), a, v, expected_artifact_fingerprint=result["artifact_inventory_fingerprint"])
        for invalid in ('{"schema":1,"schema":2}', '{"seed":NaN}'):
            with self.assertRaises(ContractError):
                manifest_from_json(invalid, a, v)

    def test_outputs_cannot_overwrite_inputs_or_existing_results(self):
        for output in (self.audio, self.audio / "nested", self.official, self.root):
            with self.assertRaises(ContractError):
                self.run_files(output_dir=output)
        occupied = self.root / "occupied"
        occupied.mkdir()
        with self.assertRaisesRegex(ContractError, "already exists"):
            self.run_files(output_dir=occupied)

    def test_no_source_bytes_are_modified_by_successful_integration(self):
        paths = list(self.audio.iterdir()) + list(self.vision.iterdir()) + list(self.official.iterdir())
        before = {str(p): p.read_bytes() for p in paths}
        self.run_files()
        self.assertEqual({str(p): p.read_bytes() for p in paths}, before)


if __name__ == "__main__":
    unittest.main()
