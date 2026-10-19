"""Synthetic contract tests: no videos, checkpoints, network, or model packages."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from fusion.contracts import (
    Alignment, Bundle, ContractError, Record, SplitArtifacts,
    align_on_sample_id, annotation_activity, audit_leakage,
    load_bundle, validate_bundle,
)

NAMES = [f"Event {i:02d}" for i in range(28)] + ["background"]
ROWS = [
    ("alpha__00", "alpha", NAMES[0], 6, 8),
    ("beta__01", "beta", NAMES[1], 0, 10),
    ("alpha__02", "alpha", NAMES[2], 2, 5),
]
FIELDS = ["sample_id", "clip_id", "category", "start", "end"]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_split(self, root, split="train", rows=None, dim=512, audio=False, probs=False):
        rows = ROWS if rows is None else rows
        with (root / f"{split}_ids.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        embeddings = np.stack([
            np.tile(np.resize(np.frombuffer(hashlib.sha256(row[0].encode()).digest(), dtype=np.uint8),
                              256 if audio else dim).astype(np.float16), (10, 1))
            for row in rows
        ])
        np.save(root / f"{split}_emb.npy", embeddings, allow_pickle=False)
        if audio or probs:
            probabilities = np.full((len(rows), 10, 29), 1 / 29, dtype=np.float16)
            np.save(root / f"{split}_probs.npy", probabilities, allow_pickle=False)

    def fixture(self, modality="audio", *, rows=None, probs=False, dim=512, splits=("train",)):
        root = self.root / f"{modality}_{len(list(self.root.iterdir()))}"
        root.mkdir()
        if modality == "audio" or probs:
            (root / "class_names.json").write_text(json.dumps(NAMES), encoding="utf-8")
        (root / "SOURCE.txt").write_text("synthetic run: contract_fixture_v1\n", encoding="utf-8")
        if modality == "audio":
            (root / "README.md").write_text("Synthetic fixture, not AVE data.\n", encoding="utf-8")
        for split in splits:
            self.write_split(root, split, rows, dim, modality == "audio", probs)
        return root

    def load(self, root, modality="audio", splits=("train",), **kwargs):
        return load_bundle(root, modality, splits=splits, expected_counts=None, **kwargs)

    def replace_array(self, root, suffix, value, split="train"):
        np.save(root / f"{split}_{suffix}.npy", value, allow_pickle=False)

    def assert_bad(self, root, pattern, modality="audio", **kwargs):
        with self.assertRaisesRegex(ContractError, pattern):
            self.load(root, modality, **kwargs)

    def test_audio_only_bundle_is_valid_but_alignment_pending(self):
        audio = self.load(self.fixture())
        self.assertTrue(validate_bundle(audio).complete)
        alignment = align_on_sample_id(audio, split="train")
        self.assertFalse(alignment.complete)
        self.assertEqual(alignment.unmatched_audio, tuple(row[0] for row in ROWS))
        self.assertIn("vision/train", alignment.missing_inputs[-1])
        with self.assertRaisesRegex(ContractError, "pending"):
            alignment.require_complete()

    def test_aligned_bundles(self):
        audio, vision = self.load(self.fixture()), self.load(self.fixture("vision"), "vision")
        result = align_on_sample_id(audio, vision, split="train")
        result.require_complete()
        self.assertEqual(result.sample_ids, tuple(row[0] for row in ROWS))
        np.testing.assert_array_equal(result.audio_indices, [0, 1, 2])
        np.testing.assert_array_equal(result.visual_indices, [0, 1, 2])

    def test_shuffled_visual_rows_preserve_original_array_indices(self):
        audio = self.load(self.fixture())
        vision = self.load(self.fixture("vision", rows=[ROWS[2], ROWS[0], ROWS[1]]), "vision")
        result = align_on_sample_id(audio, vision, split="train")
        self.assertTrue(result.complete)
        np.testing.assert_array_equal(result.visual_indices, [1, 2, 0])
        np.testing.assert_array_equal(
            audio.splits["train"].embeddings[result.audio_indices, :, :32],
            vision.splits["train"].embeddings[result.visual_indices, :, :32],
        )

    def test_duplicate_sample_ids_rejected(self):
        self.assert_bad(self.fixture(rows=[ROWS[0], ROWS[0]]), "duplicate sample_id")

    def test_two_annotations_share_clip_without_collapsing(self):
        audio = self.load(self.fixture())
        result = align_on_sample_id(audio, self.load(self.fixture("vision"), "vision"), split="train")
        self.assertTrue(result.complete)
        result.require_complete()
        self.assertEqual(len(result.sample_ids), 3)
        self.assertEqual(audit_leakage(audio).multiple_annotations["train"]["alpha"], ("alpha__00", "alpha__02"))

    def test_missing_rows_are_reported_in_both_directions(self):
        for side in ("audio", "vision"):
            with self.subTest(side=side):
                audio = self.load(self.fixture(rows=ROWS[:2] if side == "audio" else ROWS))
                vision = self.load(self.fixture("vision", rows=ROWS[:2] if side == "vision" else ROWS), "vision")
                result = align_on_sample_id(audio, vision, split="train")
                self.assertFalse(result.complete)
                unmatched = result.unmatched_visual if side == "audio" else result.unmatched_audio
                self.assertEqual(unmatched, (ROWS[2][0],))
                self.assertEqual(len(result.audio_indices), 2)

    def test_incorrect_audio_embedding_shapes(self):
        for shape in ((3, 10, 128), (3, 9, 256), (3, 10), (3, 10, 0)):
            with self.subTest(shape=shape):
                root = self.fixture()
                self.replace_array(root, "emb", np.zeros(shape, dtype=np.float16))
                self.assert_bad(root, "shape")

    def test_dynamic_visual_dimension_and_consistency(self):
        vision = self.load(self.fixture("vision", dim=37), "vision")
        self.assertEqual(vision.provenance["splits"]["train"]["embedding_dimension"], 37)
        root = self.fixture("vision", splits=("train", "val"))
        self.write_split(root, "val", dim=31)
        self.assert_bad(root, "dimension differs", "vision", splits=("train", "val"))

    def test_csv_array_row_count_mismatch(self):
        root = self.fixture()
        self.replace_array(root, "emb", np.zeros((2, 10, 256), dtype=np.float16))
        self.assert_bad(root, "expected shape")

    def test_nonfinite_embeddings_and_probabilities(self):
        for suffix, shape in (("emb", (3, 10, 256)), ("probs", (3, 10, 29))):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(suffix=suffix, bad=bad):
                    root = self.fixture()
                    value = np.zeros(shape, dtype=np.float16)
                    value[0, 0, 0] = bad
                    self.replace_array(root, suffix, value)
                    self.assert_bad(root, "NaN or infinity")

    def test_probability_normalization_and_range(self):
        for bad, pattern in ((0, "sums"), (-0.1, r"\[0,1\]"), (1.1, r"\[0,1\]")):
            with self.subTest(bad=bad):
                root = self.fixture()
                self.replace_array(root, "probs", np.full((3, 10, 29), bad, dtype=np.float16))
                self.assert_bad(root, pattern)

    def test_float16_probability_tolerance(self):
        root = self.fixture()
        probabilities = np.zeros((3, 10, 29), dtype=np.float16)
        probabilities[..., 0] = np.float16(0.9995)
        self.replace_array(root, "probs", probabilities)
        self.assertTrue(validate_bundle(self.load(root)).complete)
        probabilities[..., 0] = np.float16(0.998)
        self.replace_array(root, "probs", probabilities)
        self.assert_bad(root, "sums")

    def test_different_visual_class_order_rejected(self):
        root = self.fixture("vision", probs=True)
        names = NAMES.copy()
        names[0], names[1] = names[1], names[0]
        (root / "class_names.json").write_text(json.dumps(names), encoding="utf-8")
        self.assert_bad(root, "class order", "vision")

    def test_semantically_different_mapping_rejected(self):
        audio = self.load(self.fixture())
        root = self.fixture("vision", probs=True)
        names = NAMES.copy()
        names[27] = "Other event"
        (root / "class_names.json").write_text(json.dumps(names), encoding="utf-8")
        vision = self.load(root, "vision")
        with self.assertRaisesRegex(ContractError, "semantic class mappings differ"):
            align_on_sample_id(audio, vision, split="train")

    def test_missing_optional_visual_probabilities(self):
        vision = self.load(self.fixture("vision"), "vision")
        self.assertIsNone(vision.splits["train"].probabilities)
        self.assertIsNone(vision.class_names)
        self.assertTrue(validate_bundle(vision).complete)
        self.assertTrue(any("probabilities unavailable" in n for n in validate_bundle(vision).notes))

    def test_visual_probabilities_require_explicit_mapping(self):
        root = self.fixture("vision", probs=True)
        (root / "class_names.json").unlink()
        bundle = self.load(root, "vision")
        self.assertFalse(validate_bundle(bundle).complete)
        self.assertIn(str(root / "class_names.json"), bundle.missing_inputs)

    def test_conflicting_category_and_intervals(self):
        for change in ("category", "interval"):
            with self.subTest(change=change):
                rows = [list(row) for row in ROWS]
                if change == "category":
                    rows[0][2] = "Unknown event"
                else:
                    rows[0][3] = 5
                vision = self.load(self.fixture("vision", rows=rows), "vision")
                with self.assertRaisesRegex(ContractError, "conflict"):
                    align_on_sample_id(self.load(self.fixture()), vision, split="train")

    def test_half_open_boundaries(self):
        activity = annotation_activity(self.load(self.fixture()).metadata["train"])
        np.testing.assert_array_equal(activity[0], [False] * 6 + [True, True, False, False])
        self.assertTrue(activity[1].all())
        self.assertEqual(activity.dtype, np.dtype(bool))

    def test_invalid_intervals_rejected(self):
        for start, end in ((-1, 3), (-1, -1), (1, 1), (5, 5), (10, 10),
                           (9, 2), (8, 6), (0, -1), (0, 11),
                           ("1.5", 3), ("NaN", 4), ("0.0", 0)):
            with self.subTest(start=start, end=end):
                rows = [list(ROWS[0])]
                rows[0][3:5] = [start, end]
                self.assert_bad(self.fixture(rows=rows), "interval|integer")

    def test_official_empty_interval_has_no_annotated_activity(self):
        record = Record("VWi2ENBuTbw__08", "VWi2ENBuTbw", "Church bell", 0, 0, 8)
        for zero in (0, np.int64(0)):
            with self.subTest(zero_type=type(zero)):
                activity = annotation_activity([replace(record, start=zero, end=zero)])
                np.testing.assert_array_equal(activity, np.zeros((1, 10), dtype=bool))
                self.assertEqual(activity.dtype, np.dtype(bool))
                self.assertFalse(activity.flags.writeable)

    def test_official_empty_interval_retained_in_complete_shuffled_join(self):
        empty = ("VWi2ENBuTbw__08", "VWi2ENBuTbw", "Church bell", 0, 0)
        rows = [("before__08", "before", "Church bell", 1, 2), empty,
                ("after__08", "after", "Church bell", 3, 4)]
        # Synthetic vocabulary preserves the real sample's category at index 8.
        names = [f"A fixture {i:02d}" for i in range(8)] + ["Church bell"] + NAMES[9:]
        bundles = []
        for modality, ordered in (("audio", rows), ("vision", [rows[2], rows[0], empty])):
            root = self.fixture(modality, rows=ordered, probs=True)
            (root / "class_names.json").write_text(json.dumps(names), encoding="utf-8")
            bundle = load_bundle(root, modality, splits=("train",), expected_counts={"train": 3})
            self.assertTrue(validate_bundle(bundle).complete)
            self.assertEqual(bundle.requested_splits, ("train",))
            self.assertEqual(tuple(bundle.metadata), ("train",))
            self.assertEqual(len(bundle.splits["train"].embeddings), 3)
            index = ordered.index(empty)
            self.assertEqual(bundle.metadata["train"][index], Record(*empty, 8))
            np.testing.assert_array_equal(
                bundle.splits["train"].embeddings[index],
                np.load(root / "train_emb.npy", allow_pickle=False)[index],
            )
            self.assertFalse(annotation_activity(bundle.metadata["train"])[index].any())
            self.assertIsNone(bundle.splits["train"].input_validity)
            self.assertIsNone(bundle.splits["train"].timestamps)
            bundles.append(bundle)
        audio, vision = bundles
        alignment = align_on_sample_id(audio, vision, split="train")
        alignment.require_complete()
        self.assertEqual(alignment.sample_ids, tuple(row[0] for row in rows))
        self.assertEqual((alignment.audio_row_count, alignment.visual_row_count), (3, 3))
        np.testing.assert_array_equal(alignment.audio_indices, [0, 1, 2])
        np.testing.assert_array_equal(alignment.visual_indices, [1, 2, 0])
        np.testing.assert_array_equal(
            audio.splits["train"].embeddings[alignment.audio_indices, :, :32],
            vision.splits["train"].embeddings[alignment.visual_indices, :, :32],
        )
        # An empty annotation is still a required source row for full coverage.
        omitted = replace(alignment, sample_ids=(rows[0][0], rows[2][0]),
                          audio_indices=np.array([0, 2]), visual_indices=np.array([1, 0]))
        self.assert_alignment_rejected(omitted, "full source-row coverage")

    def test_empty_annotation_does_not_override_explicit_input_validity(self):
        rows = [("alpha__00", "alpha", NAMES[0], 0, 0)]
        for modality in ("audio", "vision"):
            with self.subTest(modality=modality):
                root = self.fixture(modality, rows=rows)
                validity = np.array([[True, False] * 5], dtype=bool)
                self.replace_array(root, "valid", validity)
                bundle = self.load(root, modality)
                self.assertTrue(validate_bundle(bundle).complete)
                self.assertFalse(annotation_activity(bundle.metadata["train"]).any())
                np.testing.assert_array_equal(bundle.splits["train"].input_validity, validity)

    def test_empty_annotation_remains_in_leakage_audit(self):
        rows = [("alpha__00", "alpha", NAMES[0], 0, 0)]
        for modality in ("audio", "vision"):
            with self.subTest(modality=modality):
                bundle = self.load(self.fixture(modality, rows=rows, splits=("train", "val")),
                                   modality, splits=("train", "val"))
                report = audit_leakage(bundle)
                self.assertEqual(report.sample_overlap["train&val"], ("alpha__00",))
                self.assertEqual(report.source_overlap["train&val"], ("alpha",))
                self.assertEqual(report.duplicate_sample_ids, {"train": (), "val": ()})

    def test_cross_split_annotation_and_source_overlap_are_distinct(self):
        root = self.fixture(splits=("train", "val", "test"))
        self.write_split(root, "val", rows=[ROWS[0]], audio=True)
        self.write_split(root, "test", rows=[("alpha__03", "alpha", NAMES[3], 1, 2)], audio=True)
        audit = audit_leakage(self.load(root, splits=("train", "val", "test")))
        self.assertEqual(audit.sample_overlap["train&val"], ("alpha__00",))
        self.assertEqual(audit.sample_overlap["train&test"], ())
        self.assertEqual(audit.source_overlap["train&test"], ("alpha",))
        self.assertEqual(audit.missing_splits, ())

    def test_missing_real_timestamps_and_validity_remain_unknown(self):
        bundle = self.load(self.fixture())
        data = bundle.splits["train"]
        self.assertIsNone(data.timestamps)
        self.assertIsNone(data.input_validity)
        self.assertFalse(annotation_activity(data.records)[0, 0])
        self.assertTrue(any("timestamps unavailable" in n for n in validate_bundle(bundle).notes))

    def test_optional_validity_is_independent_of_activity(self):
        root = self.fixture("vision")
        validity = np.ones((3, 10), dtype=bool)
        validity[0, 6] = False
        self.replace_array(root, "valid", validity)
        bundle = self.load(root, "vision")
        self.assertTrue(annotation_activity(bundle.metadata["train"])[0, 6])
        self.assertFalse(bundle.splits["train"].input_validity[0, 6])
        self.assertTrue(bundle.splits["train"].input_validity[0, 0])

    def test_optional_timestamp_contract(self):
        root = self.fixture("vision")
        times = np.tile(np.arange(10, dtype=np.float64), (3, 1))
        self.replace_array(root, "timestamps", times)
        np.testing.assert_array_equal(self.load(root, "vision").splits["train"].timestamps, times)
        for bad in (np.nan, np.inf, -1, 10, 2.0):
            with self.subTest(bad=bad):
                broken = times.copy()
                broken[0, 0] = bad
                self.replace_array(root, "timestamps", broken)
                self.assert_bad(root, "timestamp|NaN", "vision")

    def test_repeated_timestamp_requires_invalid_flag(self):
        root = self.fixture("vision")
        times = np.tile(np.arange(10, dtype=np.float32), (3, 1))
        times[0, 9] = 8
        self.replace_array(root, "timestamps", times)
        self.assert_bad(root, "outside.*bin", "vision")
        valid = np.ones((3, 10), dtype=bool)
        valid[0, 9] = False
        self.replace_array(root, "valid", valid)
        self.assertTrue(validate_bundle(self.load(root, "vision")).complete)

    def test_invalid_or_missing_paths(self):
        for path in (self.root / "absent", self.root / "file.txt"):
            if path.name == "file.txt":
                path.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "directory"):
                self.load(path)
        root = self.fixture()
        (root / "train_emb.npy").unlink()
        (root / "train_emb.npy").mkdir()
        self.assert_bad(root, "artifact file")

    def test_missing_required_artifacts_are_partial(self):
        for name in ("train_emb.npy", "train_ids.csv", "train_probs.npy", "class_names.json", "SOURCE.txt", "README.md"):
            with self.subTest(name=name):
                root = self.fixture()
                (root / name).unlink()
                bundle = self.load(root)
                self.assertFalse(validate_bundle(bundle).complete)
                self.assertIn(str(root / name), bundle.missing_inputs)
                self.assertFalse(align_on_sample_id(bundle, self.load(self.fixture("vision"), "vision"), split="train").complete)

    def test_partial_metadata_still_audited(self):
        root = self.fixture(splits=("train", "val"))
        (root / "val_emb.npy").unlink()
        audit = audit_leakage(self.load(root, splits=("train", "val")))
        self.assertEqual(audit.source_overlap["train&val"], ("alpha", "beta"))
        self.assertEqual(audit.missing_splits, ("test",))

    def test_missing_inputs_do_not_hide_invalid_existing_arrays(self):
        for missing, corrupted in (("ids.csv", "emb"), ("emb.npy", "probs")):
            with self.subTest(missing=missing):
                root = self.fixture()
                (root / f"train_{missing}").unlink()
                shape = (3, 10, 256 if corrupted == "emb" else 29)
                array = np.zeros(shape, dtype=np.float16)
                array[0, 0, 0] = np.nan
                self.replace_array(root, corrupted, array)
                self.assert_bad(root, "NaN")

    def test_probability_and_optional_metadata_shapes(self):
        for suffix, array in (
            ("probs", np.ones((3, 10, 28), dtype=np.float16)),
            ("probs", np.ones((2, 10, 29), dtype=np.float16)),
            ("valid", np.ones((3, 9), dtype=bool)),
            ("timestamps", np.ones((3, 9), dtype=np.float32)),
        ):
            with self.subTest(suffix=suffix, shape=array.shape):
                root = self.fixture()
                self.replace_array(root, suffix, array)
                self.assert_bad(root, "shape|validity")

    def test_csv_headers_and_malformed_rows(self):
        for contents in (
            "sample_id,clip_id\nalpha__00,alpha\n",
            "sample_id,clip_id,category,start,end,end\nalpha__00,alpha,Event 00,0,1,1\n",
            "sample_id,clip_id,category,start,end\nalpha__00,alpha,Event 00,0\n",
            "sample_id,clip_id,category,start,end\nalpha__00,alpha,Event 00,0,1,extra\n",
            "sample_id,clip_id,category,start,end\n",
        ):
            with self.subTest(contents=contents):
                root = self.fixture()
                (root / "train_ids.csv").write_text(contents, encoding="utf-8")
                self.assert_bad(root, "headers|malformed|no samples")

    def test_corrupt_numpy_and_mapping_files(self):
        root = self.fixture()
        (root / "class_names.json").write_text("{not JSON", encoding="utf-8")
        self.assert_bad(root, "invalid class mapping JSON")
        (root / "class_names.json").write_text(json.dumps(NAMES), encoding="utf-8")
        (root / "train_emb.npy").write_bytes(b"not a NumPy file")
        self.assert_bad(root, "cannot load")

    def test_duplicate_ids_auditable_even_though_loader_rejects_them(self):
        bundle = self.load(self.fixture())
        duplicated = replace(bundle, metadata={"train": (bundle.metadata["train"][0],) * 2})
        self.assertEqual(audit_leakage(duplicated).duplicate_sample_ids["train"], ("alpha__00",))

    def test_public_activity_rejects_noninteger_intervals(self):
        for start, end in ((1.5, 3), (0.0, 0), (0, 0.0), ("0", 0),
                           (False, 0), (0, False), (np.bool_(False), 0), (0, np.bool_(False))):
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ContractError, "integer interval"):
                    annotation_activity([Record("alpha__00", "alpha", NAMES[0], start, end, 0)])
        self.assertEqual(annotation_activity([]).shape, (0, 10))

    def test_unrequested_split_cannot_claim_alignment(self):
        audio, vision = self.load(self.fixture()), self.load(self.fixture("vision"), "vision")
        alignment = align_on_sample_id(audio, vision, split="test")
        self.assertFalse(alignment.complete)
        self.assertEqual(len(alignment.missing_inputs), 2)

    def test_empty_bundle_path_rejected(self):
        with self.assertRaisesRegex(ContractError, "nonempty"):
            self.load("")

    def test_deterministic_repeated_alignment(self):
        audio = self.load(self.fixture())
        vision = self.load(self.fixture("vision", rows=list(reversed(ROWS))), "vision")
        first = align_on_sample_id(audio, vision, split="train")
        second = align_on_sample_id(audio, vision, split="train")
        self.assertEqual(first.sample_ids, second.sample_ids)
        self.assertEqual(first.notes, second.notes)
        np.testing.assert_array_equal(first.visual_indices, second.visual_indices)

    def test_invalid_sample_identity_and_csv(self):
        for column, value in ((0, "other__00"), (0, "alpha__28"), (0, "alpha__0"),
                              (1, "../alpha"), (2, ""), (2, " Event 00")):
            with self.subTest(column=column, value=value):
                rows = [list(ROWS[0])]
                rows[0][column] = value
                self.assert_bad(self.fixture(rows=rows), "sample_id|clip_id|category|event label")

    def test_category_index_consistency_without_visual_mapping(self):
        rows = [ROWS[0], ("beta__01", "beta", NAMES[0], 0, 1)]
        self.assert_bad(self.fixture("vision", rows=rows), "conflicting label indices", "vision")

    def test_class_mapping_structure(self):
        for names in (NAMES[:-1], NAMES[:-1] + ["other"], [NAMES[0]] * 29, {}, NAMES[:-1] + [None]):
            with self.subTest(names=names):
                root = self.fixture()
                (root / "class_names.json").write_text(json.dumps(names), encoding="utf-8")
                self.assert_bad(root, "class|names|background")

    def test_unsupported_dtypes_and_validity_coercion_rejected(self):
        for dtype in (np.int16, np.complex64, np.float32):
            with self.subTest(dtype=dtype):
                root = self.fixture()
                self.replace_array(root, "emb", np.zeros((3, 10, 256), dtype=dtype))
                self.assert_bad(root, "float16")
        root = self.fixture("vision")
        self.replace_array(root, "valid", np.ones((3, 10), dtype=np.int8))
        self.assert_bad(root, "validity", "vision")

    def test_pickle_and_archive_artifacts_rejected(self):
        root = self.fixture()
        np.save(root / "train_emb.npy", np.array([{"unsafe": True}], dtype=object))
        self.assert_bad(root, "cannot load")
        with (root / "train_emb.npy").open("wb") as stream:
            np.savez(stream, arr=np.zeros((3, 10, 256)))
        self.assert_bad(root, "archive")

    def test_hashes_and_readonly_sources(self):
        root = self.fixture()
        path = root / "train_emb.npy"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        plain = self.load(root)
        self.assertNotIn("sha256", plain.provenance["files"][path.name])
        hashed = self.load(root, hash_files=True)
        self.assertEqual(hashed.provenance["files"][path.name]["sha256"], digest)
        json.dumps(hashed.provenance)
        with self.assertRaises(ValueError):
            hashed.splits["train"].embeddings[0, 0, 0] = 999
        align_on_sample_id(hashed, self.load(self.fixture("vision"), "vision"), split="train")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_official_count_enforcement_is_explicitly_overridable(self):
        root = self.fixture()
        with self.assertRaisesRegex(ContractError, "expected 3339"):
            load_bundle(root, "audio", splits=("train",))
        bundle = load_bundle(root, "audio", splits=("train",), expected_counts={"train": 3})
        self.assertTrue(validate_bundle(bundle).complete)

    def test_invalid_split_and_modality_arguments(self):
        root = self.fixture()
        for splits in ((), ("validation",), ("train", "train")):
            with self.subTest(splits=splits), self.assertRaisesRegex(ContractError, "splits"):
                self.load(root, splits=splits)
        with self.assertRaisesRegex(ContractError, "modality"):
            self.load(root, "video")
        with self.assertRaisesRegex(ContractError, "Unknown split"):
            align_on_sample_id(self.load(root), split="validation")

    def test_metadata_and_split_mappings_are_immutable(self):
        bundle = self.load(self.fixture())
        with self.assertRaises(TypeError):
            bundle.metadata["train"] = tuple(reversed(bundle.metadata["train"]))
        with self.assertRaises(TypeError):
            bundle.splits["train"] = replace(bundle.splits["train"], probabilities=None)

    def test_constructor_snapshots_caller_owned_record_containers(self):
        original = self.load(self.fixture())
        rows = list(original.metadata["train"])
        metadata = {"train": rows}
        splits = dict(original.splits)
        data = replace(original.splits["train"], records=rows)
        splits["train"] = data
        copied = replace(original, metadata=metadata, splits=splits)
        rows.reverse()
        metadata.clear()
        splits.clear()
        self.assertEqual(copied.metadata["train"], original.metadata["train"])
        self.assertEqual(data.records, original.metadata["train"])
        self.assertTrue(validate_bundle(copied).complete)

    def test_reversed_metadata_is_rejected_before_alignment(self):
        audio, vision = self.load(self.fixture()), self.load(self.fixture("vision"), "vision")
        broken = replace(vision, metadata={"train": tuple(reversed(vision.metadata["train"]))})
        with self.assertRaisesRegex(ContractError, "membership or order"):
            validate_bundle(broken)
        with self.assertRaisesRegex(ContractError, "membership or order"):
            align_on_sample_id(audio, broken, split="train")

    def test_metadata_membership_mismatch_rejected(self):
        bundle = self.load(self.fixture())
        broken = replace(bundle, metadata={"train": bundle.metadata["train"][:-1]})
        with self.assertRaisesRegex(ContractError, "membership or order"):
            validate_bundle(broken)

    def test_empty_or_missing_metadata_with_array_rows_rejected(self):
        bundle = self.load(self.fixture())
        for metadata in ({"train": ()}, {}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ContractError, "metadata"):
                validate_bundle(replace(bundle, metadata=metadata))

    def test_missing_audio_probabilities_after_loading_is_incomplete(self):
        audio, vision = self.load(self.fixture()), self.load(self.fixture("vision"), "vision")
        changed = replace(audio, splits={"train": replace(audio.splits["train"], probabilities=None)})
        self.assertEqual(changed.missing_inputs, ())
        report = validate_bundle(changed)
        self.assertFalse(report.complete)
        self.assertIn(str(audio.root / "train_probs.npy"), report.missing_inputs)
        alignment = align_on_sample_id(changed, vision, split="train")
        self.assertFalse(alignment.complete)
        with self.assertRaisesRegex(ContractError, "pending"):
            alignment.require_complete()

    def test_empty_requested_split_structures_are_incomplete(self):
        audio = self.load(self.fixture())
        changed = replace(audio, splits={}, metadata={})
        report = validate_bundle(changed)
        self.assertFalse(report.complete)
        for suffix in ("ids.csv", "emb.npy", "probs.npy"):
            self.assertIn(str(audio.root / f"train_{suffix}"), report.missing_inputs)

    def test_constructed_bundle_without_ids_cannot_align_completely(self):
        original = self.load(self.fixture())
        data = original.splits["train"]
        constructed = Bundle(
            original.root, "audio", ("train",),
            {"train": SplitArtifacts(None, data.embeddings, data.probabilities)},
            {}, original.class_names, (), original.provenance,
        )
        report = validate_bundle(constructed)
        self.assertFalse(report.complete)
        self.assertIn(str(original.root / "train_ids.csv"), report.missing_inputs)
        alignment = align_on_sample_id(constructed, self.load(self.fixture("vision"), "vision"), split="train")
        with self.assertRaisesRegex(ContractError, "pending"):
            alignment.require_complete()

    def test_invalid_interval_after_loading_rejected_by_validation_and_alignment(self):
        audio, vision = self.load(self.fixture()), self.load(self.fixture("vision"), "vision")
        bad = (replace(audio.metadata["train"][0], start=9, end=2), *audio.metadata["train"][1:])
        changed = replace(audio, metadata={"train": bad}, splits={
            "train": replace(audio.splits["train"], records=bad)
        })
        with self.assertRaisesRegex(ContractError, "integer interval"):
            validate_bundle(changed)
        with self.assertRaisesRegex(ContractError, "integer interval"):
            align_on_sample_id(changed, vision, split="train")

    def test_public_record_identity_invariants_revalidated(self):
        original = self.load(self.fixture())
        for changes in ({"sample_id": "wrong__00"}, {"clip_id": "../wrong"},
                        {"category": "background"}, {"category": ""}, {"label": -1},
                        {"label": 28}, {"label": True}, {"start": False}, {"end": 11}):
            with self.subTest(changes=changes):
                records = (replace(original.metadata["train"][0], **changes), *original.metadata["train"][1:])
                changed = replace(original, metadata={"train": records}, splits={
                    "train": replace(original.splits["train"], records=records)
                })
                with self.assertRaises(ContractError):
                    validate_bundle(changed)

    def test_missing_embeddings_after_loading_is_incomplete(self):
        for modality in ("audio", "vision"):
            with self.subTest(modality=modality):
                original = self.load(self.fixture(modality), modality)
                changed = replace(original, splits={"train": replace(original.splits["train"], embeddings=None)})
                report = validate_bundle(changed)
                self.assertFalse(report.complete)
                self.assertIn(str(original.root / "train_emb.npy"), report.missing_inputs)

    def test_public_mapping_and_probability_semantics_revalidated(self):
        audio = self.load(self.fixture())
        for names in (NAMES[:-1], [NAMES[0]] * 29, list(reversed(NAMES))):
            with self.subTest(names=names), self.assertRaises(ContractError):
                validate_bundle(replace(audio, class_names=names))
        changed = replace(audio, class_names=None)
        self.assertFalse(validate_bundle(changed).complete)

    def test_same_category_different_clips_select_correct_rows(self):
        rows = [*ROWS, ("gamma__00", "gamma", NAMES[0], 6, 8)]
        audio = self.load(self.fixture(rows=rows))
        vision = self.load(self.fixture("vision", rows=[rows[3], rows[1], rows[0], rows[2]]), "vision")
        alignment = align_on_sample_id(audio, vision, split="train")
        alignment.require_complete()
        np.testing.assert_array_equal(alignment.visual_indices, [2, 1, 3, 0])
        self.assertEqual(tuple(vision.splits["train"].records[i].sample_id for i in alignment.visual_indices),
                         alignment.sample_ids)
        self.assertFalse(np.array_equal(audio.splits["train"].embeddings[0], audio.splits["train"].embeddings[3]))
        np.testing.assert_array_equal(audio.splits["train"].embeddings[alignment.audio_indices, :, :32],
                                      vision.splits["train"].embeddings[alignment.visual_indices, :, :32])

    def test_invalid_csv_utf8_raises_chained_contract_error(self):
        root = self.fixture()
        (root / "train_ids.csv").write_bytes(b"sample_id,clip_id,category,start,end\n\xff")
        with self.assertRaisesRegex(ContractError, "IDs CSV must be valid UTF-8") as caught:
            self.load(root)
        self.assertIsInstance(caught.exception.__cause__, UnicodeDecodeError)
        self.assertIn("train_ids.csv", str(caught.exception))

    def test_count_contract_persists_when_bundle_reconstructed(self):
        original = load_bundle(self.fixture(), "audio", splits=("train",), expected_counts={"train": 3})
        records = original.metadata["train"][:2]
        data = original.splits["train"]
        changed = replace(original, metadata={"train": records}, splits={"train": replace(
            data, records=records, embeddings=data.embeddings[:2], probabilities=data.probabilities[:2]
        )})
        with self.assertRaisesRegex(ContractError, "expected 3 samples"):
            validate_bundle(changed)

    def test_arrays_modified_after_loading_are_revalidated(self):
        original = self.load(self.fixture())
        for field, array in (("embeddings", np.zeros((2, 10, 256), dtype=np.float16)),
                             ("probabilities", np.full((3, 10, 29), np.nan, dtype=np.float16)),
                             ("input_validity", np.ones((3, 10), dtype=np.int8))):
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_bundle(replace(original, splits={"train": replace(original.splits["train"], **{field: array})}))

    def test_empty_alignment_cannot_pass_guard(self):
        empty = Alignment("train", (), np.array([], dtype=np.int64), np.array([], dtype=np.int64), (), (), (), ())
        self.assertFalse(empty.complete)
        with self.assertRaisesRegex(ContractError, "pending"):
            empty.require_complete()

    def valid_alignment(self):
        audio = self.load(self.fixture())
        vision = self.load(self.fixture("vision", rows=[ROWS[2], ROWS[0], ROWS[1]]), "vision")
        result = align_on_sample_id(audio, vision, split="train")
        result.require_complete()
        return result

    def assert_alignment_rejected(self, alignment, message):
        self.assertFalse(alignment.complete)
        with self.assertRaisesRegex(ContractError, message):
            alignment.require_complete()

    def test_alignment_index_lengths_rejected(self):
        valid = self.valid_alignment()
        for name in ("audio_indices", "visual_indices"):
            for indices in (np.array([0, 1]), np.array([0, 1, 2, 3])):
                with self.subTest(name=name, length=len(indices)):
                    self.assert_alignment_rejected(replace(valid, **{name: indices}), "length")

    def test_alignment_negative_indices_rejected(self):
        valid = self.valid_alignment()
        for name in ("audio_indices", "visual_indices"):
            indices = getattr(valid, name).copy()
            indices[0] = -1
            with self.subTest(name=name):
                self.assert_alignment_rejected(replace(valid, **{name: indices}), "negative")

    def test_alignment_source_bounds_rejected(self):
        valid = self.valid_alignment()
        for name, count in (("audio_indices", valid.audio_row_count),
                            ("visual_indices", valid.visual_row_count)):
            for out_of_bounds in (count, 999):
                indices = getattr(valid, name).copy()
                indices[0] = out_of_bounds
                with self.subTest(name=name, index=out_of_bounds):
                    self.assert_alignment_rejected(replace(valid, **{name: indices}), "outside source row count")

    def test_alignment_duplicate_indices_rejected(self):
        valid = self.valid_alignment()
        for name in ("audio_indices", "visual_indices"):
            indices = getattr(valid, name).copy()
            indices[0] = indices[1]
            with self.subTest(name=name):
                self.assert_alignment_rejected(replace(valid, **{name: indices}), "duplicate indices")

    def test_alignment_noninteger_indices_rejected(self):
        valid = self.valid_alignment()
        for name in ("audio_indices", "visual_indices"):
            for dtype in (np.float32, np.float64, np.bool_, np.complex64, object, "U2"):
                with self.subTest(name=name, dtype=dtype):
                    self.assert_alignment_rejected(
                        replace(valid, **{name: getattr(valid, name).astype(dtype)}), "integers"
                    )

    def test_alignment_multidimensional_indices_rejected(self):
        valid = self.valid_alignment()
        for name in ("audio_indices", "visual_indices"):
            for indices in (np.arange(3).reshape(1, 3), np.arange(3).reshape(3, 1), np.array(0)):
                with self.subTest(name=name, shape=indices.shape):
                    self.assert_alignment_rejected(replace(valid, **{name: indices}), "one-dimensional")

    def test_alignment_duplicate_sample_ids_rejected(self):
        valid = self.valid_alignment()
        self.assert_alignment_rejected(
            replace(valid, sample_ids=(valid.sample_ids[0], valid.sample_ids[0], valid.sample_ids[2])),
            "duplicate aligned sample IDs",
        )

    def test_alignment_row_counts_required_and_validated(self):
        valid = self.valid_alignment()
        for name in ("audio_row_count", "visual_row_count"):
            for count in (None, True, np.bool_(True), -1, 0, 3.0, "3", 2 ** 100):
                with self.subTest(name=name, count=count):
                    self.assert_alignment_rejected(replace(valid, **{name: count}), "source row count")

    def test_alignment_counts_are_actual_source_lengths(self):
        valid = self.valid_alignment()
        self.assertEqual((valid.audio_row_count, valid.visual_row_count), (3, 3))
        audio = self.load(self.fixture())
        vision = self.load(self.fixture("vision", rows=ROWS[:2]), "vision")
        partial = align_on_sample_id(audio, vision, split="train")
        self.assertEqual(len(partial.sample_ids), 2)
        self.assertEqual((partial.audio_row_count, partial.visual_row_count), (3, 2))
        self.assert_alignment_rejected(partial, "audio-only IDs")

    def test_alignment_indices_are_readonly_snapshots(self):
        valid = self.valid_alignment()
        source = valid.visual_indices.copy()
        copied = replace(valid, visual_indices=source)
        source[0] = 999
        np.testing.assert_array_equal(copied.visual_indices, valid.visual_indices)
        copied.require_complete()
        with self.assertRaises(ValueError):
            copied.visual_indices[0] = 999

    def test_alignment_diagnostics_prevent_completion(self):
        valid = self.valid_alignment()
        for name in ("missing_inputs", "unmatched_audio", "unmatched_visual"):
            with self.subTest(name=name):
                self.assert_alignment_rejected(replace(valid, **{name: ("unavailable__00",)}), "unavailable__00")

    def test_alignment_malformed_reconstructed_containers_rejected(self):
        valid = self.valid_alignment()
        for changes in ({"sample_ids": None}, {"sample_ids": ("", "b", "c")},
                        {"audio_indices": [0, 1, 2]}, {"visual_indices": None},
                        {"missing_inputs": None}, {"notes": "not a sequence"}, {"split": "unknown"},
                        {"missing_inputs": np.array(["one", "two"])},
                        {"unmatched_audio": np.array(["one", "two"])},
                        {"unmatched_visual": np.array(["one", "two"])}):
            with self.subTest(changes=changes):
                self.assert_alignment_rejected(replace(valid, **changes), "alignment pending or invalid")

    def test_alignment_valid_unsigned_indices_and_snapshot_ids(self):
        valid = self.valid_alignment()
        ids = list(valid.sample_ids)
        copied = replace(valid, sample_ids=ids, audio_indices=valid.audio_indices.astype(np.uint64))
        ids.reverse()
        self.assertEqual(copied.sample_ids, valid.sample_ids)
        self.assertTrue(copied.complete)
        copied.require_complete()

    def test_alignment_simultaneous_truncation_rejected(self):
        valid = self.valid_alignment()
        truncated = replace(
            valid, sample_ids=valid.sample_ids[:2],
            audio_indices=valid.audio_indices[:2], visual_indices=valid.visual_indices[:2],
        )
        self.assertEqual((truncated.audio_row_count, truncated.visual_row_count), (3, 3))
        self.assert_alignment_rejected(truncated, "full source-row coverage")
        self.assertEqual(truncated.sample_ids, valid.sample_ids[:2])
        np.testing.assert_array_equal(truncated.audio_indices, valid.audio_indices[:2])
        np.testing.assert_array_equal(truncated.visual_indices, valid.visual_indices[:2])

    def test_alignment_unequal_source_coverage_rejected_without_diagnostics(self):
        for shorter in ("audio", "vision"):
            with self.subTest(shorter=shorter):
                audio = self.load(self.fixture(rows=ROWS[:2] if shorter == "audio" else ROWS))
                vision = self.load(self.fixture("vision", rows=ROWS[:2] if shorter == "vision" else ROWS), "vision")
                partial = align_on_sample_id(audio, vision, split="train")
                expected_counts = (2, 3) if shorter == "audio" else (3, 2)
                self.assertEqual((partial.audio_row_count, partial.visual_row_count), expected_counts)
                self.assertEqual(len(partial.sample_ids), 2)
                self.assertFalse(partial.complete)
                self.assertTrue(partial.unmatched_audio or partial.unmatched_visual)
                reconstructed = replace(partial, unmatched_audio=(), unmatched_visual=())
                self.assert_alignment_rejected(reconstructed, "full source-row coverage")

    def test_alignment_interior_source_row_omission_rejected(self):
        valid = self.valid_alignment()
        for name in ("audio_indices", "visual_indices"):
            with self.subTest(name=name):
                keep = np.flatnonzero(getattr(valid, name) != 1)
                omitted = replace(
                    valid, sample_ids=tuple(valid.sample_ids[i] for i in keep),
                    audio_indices=valid.audio_indices[keep], visual_indices=valid.visual_indices[keep],
                )
                np.testing.assert_array_equal(np.sort(getattr(omitted, name)), [0, 2])
                self.assert_alignment_rejected(omitted, "full source-row coverage")

    def test_alignment_full_source_permutations_accepted(self):
        valid = self.valid_alignment()
        order = np.array([1, 2, 0])
        reordered = replace(
            valid, sample_ids=tuple(valid.sample_ids[i] for i in order),
            audio_indices=valid.audio_indices[order], visual_indices=valid.visual_indices[order],
        )
        self.assertTrue(reordered.complete)
        reordered.require_complete()
        for name in ("audio_indices", "visual_indices"):
            np.testing.assert_array_equal(np.sort(getattr(reordered, name)), np.arange(3))
        self.assertIn("alpha__00", reordered.sample_ids)
        self.assertIn("alpha__02", reordered.sample_ids)

    def test_alignment_empty_sources_cannot_pass_guard(self):
        empty = Alignment(
            "train", (), np.array([], dtype=np.int64), np.array([], dtype=np.int64),
            (), (), (), (), audio_row_count=0, visual_row_count=0,
        )
        self.assert_alignment_rejected(empty, "no aligned sample IDs")


if __name__ == "__main__":
    unittest.main()
