"""Offline fixtures for the pinned-reference verifier; no real export loading."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fusion import ContractError
from fusion import verify_ave_reference as module


class AveReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True, capture_output=True)
        (self.root / ".gitignore").write_text("_local/\n", encoding="utf-8")
        self.metadata, self.reference = self.root / "supplied", self.root / "authors"
        self.metadata.mkdir()
        self.reference.mkdir()
        names = ([f"A event {i:02d}" for i in range(8)] + ["Church bell"]
                 + [f"D event {i:02d}" for i in range(9, 28)])
        self.rows = [f"{name}&{'shared' if i < 2 else 'clip' + str(i)}&good&1&4\n".encode()
                     for i, name in enumerate(names)]
        self.rows[8] = b"Church bell&VWi2ENBuTbw&good&0&0\n"
        self.orders = {"train": np.arange(24, dtype=np.int64), "val": np.array([24, 25]),
                       "test": np.array([26, 27])}
        self.counts = {"train": 24, "val": 2, "test": 2}
        author = b"".join(self.rows)
        (self.reference / "Annotations.txt").write_bytes(author)
        (self.metadata / "Annotations.txt").write_bytes(author)
        for split, name in zip(module.SPLITS, module.METADATA_FILES[1:]):
            (self.metadata / name).write_bytes(b"".join(self.rows[i] for i in self.orders[split]))
            (self.reference / f"{split}_order.h5").write_bytes(split.encode())
        # Authenticate this small fixture only within tests. Production constants
        # and CLI counts cannot be replaced by user-supplied manifests/reports.
        self.hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.reference.iterdir()}
        for name, value in (("_REPO_ROOT", self.root), ("REFERENCE_HASHES", self.hashes),
                            ("OFFICIAL_COUNTS", self.counts)):
            guard = patch.object(module, name, value)
            guard.start()
            self.addCleanup(guard.stop)
        self.reader = patch.object(module, "_read_order", side_effect=lambda data, _: self.orders[data.decode()])
        self.reader.start()
        self.addCleanup(self.reader.stop)

    def verify(self, **kwargs):
        return module.verify_reference(self.metadata, self.reference, **kwargs)

    def change(self, name, transform):
        path = self.metadata / name
        path.write_bytes(transform(path.read_bytes()))

    def test_correct_comparison_retains_shared_sources_and_empty_annotation(self):
        result = self.verify()
        self.assertEqual((result["status"], result["annotation_rows"], result["mismatch_count"]), ("PASS", 28, 0))
        self.assertEqual(result["unique_sources"], 27)
        self.assertEqual(result["multiple_annotation_sources"], 1)
        self.assertEqual(result["empty_annotations"], [{"sample_id": "VWi2ENBuTbw__08", "annotation_index": 8,
                         "split": "train", "annotation_activity": [False] * 10, "event_compatibility": "UNKNOWN"}])
        self.assertEqual([result["split_checks"][s]["records_verified"] for s in module.SPLITS], [24, 2, 2])

    def test_exact_known_header_is_optional_and_source_bytes_are_unchanged(self):
        for header in (b"", module.MIRROR_HEADER + b"\n", module.MIRROR_HEADER + b"\r\n"):
            with self.subTest(header=header):
                content = header + b"".join(self.rows)
                (self.metadata / "Annotations.txt").write_bytes(content)
                self.assertEqual(self.verify()["annotation_rows"], 28)
                self.assertEqual((self.metadata / "Annotations.txt").read_bytes(), content)

    def test_unknown_repeated_or_malformed_header_cannot_hide_data(self):
        for prefix in (b"arbitrary header\n", module.MIRROR_HEADER + b"\n" + module.MIRROR_HEADER + b"\n",
                       module.MIRROR_HEADER + b"\r\r\n"):
            (self.metadata / "Annotations.txt").write_bytes(prefix + b"".join(self.rows))
            with self.subTest(prefix=prefix), self.assertRaises(ContractError):
                self.verify()

    def test_modified_category_is_rejected(self):
        self.change("Annotations.txt", lambda data: data.replace(b"A event 00&", b"Wrong event&", 1))
        with self.assertRaisesRegex(ContractError, "annotation text mismatch"):
            self.verify()

    def test_modified_integer_and_fractional_intervals_are_rejected(self):
        for interval in (b"&2&4", b"&1.5&4", b"&9&2", b"&1&1", b"&-1&4", b"&1&11"):
            (self.metadata / "Annotations.txt").write_bytes(b"".join(self.rows).replace(b"&1&4", interval, 1))
            with self.subTest(interval=interval), self.assertRaises(ContractError):
                self.verify()

    def test_duplicate_annotation_ids_are_rejected(self):
        self.change("Annotations.txt", lambda data: data + self.rows[0])
        with self.assertRaisesRegex(ContractError, "duplicate annotation identity"):
            self.verify()

    def test_split_ordering_is_not_treated_as_set_equivalence(self):
        self.change("valSet.txt", lambda data: b"".join(data.splitlines(keepends=True)[::-1]))
        with self.assertRaisesRegex(ContractError, "split ordering mismatch"):
            self.verify()

    def test_wrong_split_membership_is_rejected(self):
        self.change("valSet.txt", lambda data: data.replace(self.rows[24], self.rows[26]))
        with self.assertRaisesRegex(ContractError, "split membership/annotation mismatch"):
            self.verify()

    def test_duplicate_or_altered_split_rows_are_rejected(self):
        for content in (self.rows[24] * 2, self.rows[24] + self.rows[25].replace(b"good", b"bad")):
            (self.metadata / "valSet.txt").write_bytes(content)
            with self.assertRaises(ContractError):
                self.verify()

    def test_reference_hash_pinning_cannot_be_overridden_by_local_sources_report(self):
        (self.reference / "Annotations.txt").write_bytes(b"tampered")
        (self.reference / "sources.json").write_text('{"sha256":"fabricated"}', encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "pinned author reference hash mismatch"):
            self.verify()

    def test_missing_author_annotation_or_any_hdf5_fails_without_outputs(self):
        for name in self.hashes:
            path = self.reference / name
            data = path.read_bytes()
            path.unlink()
            output = self.root / "_local" / "missing"
            with self.subTest(name=name), self.assertRaisesRegex(ContractError, "missing required reference"):
                self.verify(output_dir=output)
            self.assertFalse(output.exists())
            path.write_bytes(data)

    def test_missing_mirror_split_never_falls_back_to_reference_or_exporter(self):
        (self.metadata / "testSet.txt").unlink()
        with self.assertRaisesRegex(ContractError, "metadata/testSet.txt"):
            self.verify()

    def test_corrupt_hdf5_is_rejected_by_hash_before_reader(self):
        (self.reference / "train_order.h5").write_bytes(b"not hdf5")
        with self.assertRaisesRegex(ContractError, "pinned author reference hash mismatch: train_order"):
            self.verify()
        module._read_order.assert_not_called()

    def test_hdf5_adapter_reports_corrupt_missing_dataset_and_optional_dependency(self):
        self.reader.stop()
        for failure in (OSError("corrupt"), KeyError("order"), ValueError("bad dataset")):
            adapter = types.SimpleNamespace(File=unittest.mock.Mock(side_effect=failure))
            with patch.dict(sys.modules, {"h5py": adapter}), self.assertRaisesRegex(ContractError, "unreadable HDF5"):
                module._read_order(b"invalid", "fixture")
        with patch.dict(sys.modules, {"h5py": None}), self.assertRaisesRegex(ContractError, "optional h5py"):
            module._read_order(b"unused", "fixture")

    def test_official_indices_reject_invalid_shape_dtype_bounds_and_duplicates(self):
        for order in (np.array([[24, 25]]), np.array([24.0, 25.0]), np.array([True, False]),
                      np.array([-1, 25]), np.array([24, 28]), np.array([24, 24]), np.array([24])):
            self.orders["val"] = order
            with self.subTest(order=order), self.assertRaises(ContractError):
                self.verify()

    def test_official_indices_must_be_disjoint_and_cover_every_annotation(self):
        self.orders["test"] = np.array([24, 25])
        (self.metadata / "testSet.txt").write_bytes(self.rows[24] + self.rows[25])
        with self.assertRaisesRegex(ContractError, "disjoint complete"):
            self.verify()

    def test_evidence_is_deterministic_portable_and_independent_of_historical_reports(self):
        first, second = self.root / "_local" / "first", self.root / "_local" / "second"
        inputs = list(self.reference.iterdir()) + list(self.metadata.iterdir())
        before = {p: p.read_bytes() for p in inputs}
        result = self.verify(output_dir=first)
        self.assertFalse((self.root / "_local/fusion/task28/validation.json").exists())
        self.assertEqual(result, self.verify(output_dir=second))
        for path in first.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            self.assertNotIn(str(self.root), path.read_text(encoding="utf-8"))
        self.assertEqual(before, {p: p.read_bytes() for p in inputs})
        expected = json.loads((first / "verified_metadata_hashes.json").read_text(encoding="utf-8"))
        self.assertEqual(expected, {f"official/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in self.metadata.iterdir()})

    def test_explicit_cli_input_output_paths(self):
        output = self.root / "_local" / "cli"
        with contextlib.redirect_stdout(io.StringIO()) as log:
            code = module.main(["--metadata-root", str(self.metadata), "--reference-root", str(self.reference),
                                "--output-dir", str(output)])
        self.assertEqual(code, 0)
        self.assertIn("mismatches=0", log.getvalue())
        self.assertEqual(json.loads((output / "verification.json").read_text())["author_commit"], module.AUTHOR_COMMIT)

    def test_output_rejects_unignored_external_input_and_existing_directories(self):
        existing = self.root / "_local" / "existing"
        existing.mkdir(parents=True)
        for output in (self.root / "unignored", self.root.parent / "outside", self.metadata / "nested", existing):
            with self.subTest(output=output), self.assertRaises(ContractError):
                self.verify(output_dir=output)

    def test_cli_fails_closed_without_reference_inputs(self):
        (self.reference / "val_order.h5").unlink()
        output = self.root / "_local" / "failed"
        with contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertEqual(module.main(["--metadata-root", str(self.metadata), "--reference-root", str(self.reference),
                                          "--output-dir", str(output)]), 1)
        self.assertIn("missing required reference", log.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
