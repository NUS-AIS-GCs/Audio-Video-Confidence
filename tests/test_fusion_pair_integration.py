"""Tasks 2.6--2.7 synthetic integration; no artifacts, fusion tensors or models."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np

from fusion import Bundle, ContractError, Record, SplitArtifacts, annotation_activity, validate_bundle
from fusion.pair_contracts import (
    HardNegativePolicy, HardPolicyOrigin, PairType, SamplingPolicy,
    SemanticCompatibility, TemporalStatus, validate_pair_manifest,
)
from fusion.pair_generator import GenerationConfig, SourceReuse, generate_pairs, validate_generated_manifest
from fusion.pair_integrity import audit_generated_manifest


NAMES = (tuple(f"A event {i:02d}" for i in range(8)) + ("Church bell",)
         + tuple(f"D event {i:02d}" for i in range(9, 28)) + ("background",))


def row(clip, label, start=1, end=4):
    return Record(f"{clip}__{label:02d}", clip, NAMES[label], start, end, label)


TRAIN = (row("a", 0), row("b", 1), row("c", 0), row("d", 1), row("e", 2),
         row("f", 2), row("a", 2), row("VWi2ENBuTbw", 8, 0, 0), row("g", 8))
METADATA = {"train": TRAIN, "val": (row("a", 1), row("v", 2)),
            "test": (row("f", 0), row("t", 1))}


def fixture(metadata=None):
    metadata = METADATA if metadata is None else metadata
    markers = {sid: i + 100 for i, sid in enumerate(sorted({r.sample_id for rows in metadata.values() for r in rows}))}
    result = []
    for modality in ("audio", "vision"):
        records, arrays = {}, {}
        for split, rows in metadata.items():
            order = sorted(rows, key=lambda r: r.sample_id, reverse=modality == "audio")
            # Both modalities differ from the fixture's declared input row order.
            order = tuple(order[1:] + order[:1]) if modality == "vision" else tuple(order)
            records[split] = order
            dim = 256 if modality == "audio" else 512
            embeddings = np.stack([np.full((10, dim), markers[r.sample_id], dtype=np.float16) for r in order])
            arrays[split] = SplitArtifacts(order, embeddings,
                                          np.full((len(order), 10, 29), 1 / 29, dtype=np.float16))
        result.append(Bundle(Path("SYNTHETIC_INTEGRATION") / modality, modality, tuple(metadata),
                             arrays, records, NAMES, (),
                             {"source": "synthetic only", "readme_available": True},
                             {s: len(rows) for s, rows in records.items()}))
    return *result, markers


def policy(origin=HardPolicyOrigin.MANUAL_APPROVED):
    hard = HardNegativePolicy("integration-fixture-v1", origin, NAMES,
                              ((NAMES[0], NAMES[1]), (NAMES[1], NAMES[0])),
                              "synthetic reviewer", "in-memory test declaration", "a" * 64,
                              "val" if origin is HardPolicyOrigin.VALIDATION_CONFUSION else None)
    return SamplingPolicy(1729, hard)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest()


def oracle(seed, budgets):
    """Independent eligibility + ranking oracle; never calls generator helpers."""
    pools = {kind: [] for kind in PairType}
    for a, v in itertools.product(TRAIN, repeat=2):
        if a.sample_id == v.sample_id:
            kind = PairType.ORIGINAL_POSITIVE
        elif a.clip_id == v.clip_id:
            continue
        elif a.label == v.label:
            kind = PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE
        elif (a.label, v.label) in ((0, 1), (1, 0)):
            kind = PairType.CROSS_CLASS_HARD_NEGATIVE
        else:
            kind = PairType.CROSS_CLASS_EASY_NEGATIVE
        pools[kind].append((a.sample_id, v.sample_id))
    split_seed = int(sha(["original-pair-v1", seed, "train"])[:16], 16)
    selected = {}
    for kind, pool in pools.items():
        ranked = sorted(pool, key=lambda ids: (
            sha(["sha256-rank-v1", split_seed, *ids, kind.value]), ids[1], ids[0], kind.value))
        selected[kind] = ranked[:budgets[kind]]
    return pools, selected


class PairIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.audio, self.vision, self.markers = fixture()
        self.policy = policy()
        self.config = GenerationConfig("train", easy_negative_budget=5, hard_negative_budget=3,
                                       challenge_budget=12, enable_hard_negatives=True)
        self.manifest = generate_pairs(self.audio, self.vision, config=self.config, policy=self.policy)

    def test_end_to_end_matches_independent_id_label_count_and_array_reference_oracle(self):
        self.assertTrue(validate_bundle(self.audio).complete)
        self.assertTrue(validate_bundle(self.vision).complete)
        self.assertEqual(self.audio.splits["train"].embeddings.shape, (9, 10, 256))
        self.assertEqual(self.vision.splits["train"].embeddings.shape, (9, 10, 512))
        for bundle in (self.audio, self.vision):
            self.assertEqual(bundle.splits["train"].probabilities.shape, (9, 10, 29))
        budgets = dict(zip(PairType, (9, 5, 3, 12)))
        pools, expected = oracle(1729, budgets)
        pairs = self.manifest.primary_pairs + self.manifest.challenge_pairs
        for kind in PairType:
            actual = {(p.audio.record.sample_id, p.visual.record.sample_id) for p in pairs if p.pair_type is kind}
            self.assertEqual(actual, set(expected[kind]))
        self.assertEqual([len(pools[k]) for k in PairType], [9, 50, 8, 12])
        self.assertEqual([c.selected for c in self.manifest.report.counts], [9, 5, 3, 12])
        self.assertEqual(len({p.identity for p in pairs}), 29)
        for partition in (self.manifest.primary_pairs, self.manifest.challenge_pairs):
            keys = [(p.visual.record.sample_id, p.audio.record.sample_id) for p in partition]
            self.assertEqual(keys, sorted(keys))
        for pair in pairs:
            self.assertEqual(pair.original_pair_target, int(pair.pair_type is PairType.ORIGINAL_POSITIVE))
            self.assertIs(pair.temporal_status, TemporalStatus.NOT_ASSESSED)
            a, v = pair.audio.record, pair.visual.record
            expected_semantic = (SemanticCompatibility.UNKNOWN if a.end == 0 or v.end == 0 else
                                 SemanticCompatibility.COMPATIBLE if a.label == v.label else
                                 SemanticCompatibility.INCOMPATIBLE)
            self.assertIs(pair.semantic_compatibility, expected_semantic)
            self.assertEqual(tuple(vars(pair.event_supervision).values()),
                             (a.end > 0, v.end > 0, a.end > 0 and v.end > 0 and a == v,
                              a.end > 0 and v.end > 0))
            for source, bundle in ((pair.audio, self.audio), (pair.visual, self.vision)):
                data = bundle.splits["train"]
                expected_index = [r.sample_id for r in data.records].index(source.record.sample_id)
                self.assertEqual(source.row_index, expected_index)
                self.assertEqual(source.record, data.records[expected_index])
                np.testing.assert_array_equal(data.embeddings[source.row_index],
                                              self.markers[source.record.sample_id])
        ordered = tuple(sorted(pairs, key=lambda p: (p.visual.record.sample_id, p.audio.record.sample_id)))
        self.assertEqual(validate_pair_manifest(ordered, self.audio, self.vision, policy=self.policy).pair_count, 29)
        self.assertEqual(validate_generated_manifest(self.manifest, self.audio, self.vision).pair_count, 29)

    def test_report_distinguishes_shared_annotations_from_sample_overlap(self):
        report = audit_generated_manifest(self.manifest, self.audio, self.vision)
        self.assertEqual((report.pair_count, report.primary_pair_count, report.challenge_pair_count), (29, 17, 12))
        self.assertEqual(report.source_row_references_checked, 58)
        self.assertEqual((report.same_source_substitutions, report.cross_split_pairs), (0, 0))
        self.assertEqual(report.duplicate_pair_identities, ())
        self.assertTrue(report.supplied_split_membership_preserved)
        for modality in report.metadata_leakage:
            self.assertEqual(dict(modality.source_overlap), {"train&val": ("a",), "train&test": ("f",), "val&test": ()})
            self.assertEqual(dict(modality.sample_overlap), {"train&val": (), "train&test": (), "val&test": ()})
            self.assertEqual(modality.multiple_annotations, (("train", "a", ("a__00", "a__02")),))
            self.assertTrue(all(not duplicates for _, duplicates in modality.duplicate_sample_ids))
            self.assertEqual(modality.missing_metadata_splits, ())
            self.assertEqual(modality.split_row_counts, (("train", 9), ("val", 2), ("test", 2)))
        self.assertEqual(self.manifest.report.same_source_exclusions, 11)
        self.assertEqual(self.manifest.report.same_sample_exclusions, 9)
        self.assertEqual(self.manifest.report.candidate_evaluations, 81)

    def test_genuine_cross_split_sample_overlap_is_reported_without_reassignment(self):
        metadata = dict(METADATA, val=METADATA["val"] + (TRAIN[4],))
        audio, vision, _ = fixture(metadata)
        manifest = generate_pairs(audio, vision, config=self.config, policy=self.policy)
        report = audit_generated_manifest(manifest, audio, vision)
        for modality in report.metadata_leakage:
            self.assertEqual(dict(modality.sample_overlap)["train&val"], ("e__02",))
            self.assertEqual(dict(modality.source_overlap)["train&val"], ("a", "e"))
        self.assertEqual(report.generation_provenance, self.manifest.provenance)
        self.assertTrue(report.supplied_split_membership_preserved)
        self.assertIn(TRAIN[4], audio.metadata["train"])
        self.assertIn(TRAIN[4], audio.metadata["val"])

    def test_empty_official_row_and_challenges_keep_unknown_masked_supervision(self):
        empty_id = "VWi2ENBuTbw__08"
        pairs = self.manifest.primary_pairs + self.manifest.challenge_pairs
        affected = [p for p in pairs if empty_id in (p.audio.record.sample_id, p.visual.record.sample_id)]
        self.assertGreaterEqual(len(affected), 3)
        for pair in affected:
            self.assertIs(pair.semantic_compatibility, SemanticCompatibility.UNKNOWN)
            self.assertFalse(pair.event_supervision.semantic_eligible)
            self.assertFalse(pair.event_supervision.joint_event_eligible)
            for source, eligible in ((pair.audio, pair.event_supervision.audio_annotation_eligible),
                                     (pair.visual, pair.event_supervision.visual_annotation_eligible)):
                self.assertEqual(eligible, source.record.end > 0)
                if source.record.sample_id == empty_id:
                    self.assertEqual(source.record.category, "Church bell")
                    self.assertFalse(annotation_activity([source.record]).any())
        original = next(p for p in affected if p.pair_type is PairType.ORIGINAL_POSITIVE)
        self.assertEqual(original.original_pair_target, 1)
        self.assertFalse(any(vars(original.event_supervision).values()))
        report = audit_generated_manifest(self.manifest, self.audio, self.vision)
        self.assertEqual(report.unknown_semantic_pairs, len(affected))
        self.assertEqual(report.empty_audio_source_pairs, sum(p.audio.record.sample_id == empty_id for p in pairs))
        self.assertEqual(report.empty_visual_source_pairs, sum(p.visual.record.sample_id == empty_id for p in pairs))
        self.assertIsNone(self.audio.splits["train"].input_validity)

    def test_audit_is_deterministic_under_dictionary_order_and_reports_are_frozen(self):
        audio, vision, _ = fixture(dict(reversed(tuple(METADATA.items()))))
        manifest = generate_pairs(audio, vision, config=self.config, policy=self.policy)
        first = audit_generated_manifest(self.manifest, self.audio, self.vision)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(audit_generated_manifest(manifest, audio, vision), first)
        with self.assertRaises(FrozenInstanceError):
            first.pair_count = 0
        with self.assertRaises(FrozenInstanceError):
            first.metadata_leakage[0].sample_overlap = ()

    def test_full_metadata_context_changes_without_changing_train_sampling_provenance(self):
        audio, vision, _ = fixture(dict(METADATA, val=(row("new_val", 1), row("v", 2))))
        manifest = generate_pairs(audio, vision, config=self.config, policy=self.policy)
        old = audit_generated_manifest(self.manifest, self.audio, self.vision)
        new = audit_generated_manifest(manifest, audio, vision)
        self.assertEqual(new.generation_provenance, old.generation_provenance)
        for before, after in zip(old.metadata_leakage, new.metadata_leakage):
            self.assertNotEqual(before.metadata_fingerprint, after.metadata_fingerprint)
            self.assertEqual(dict(after.source_overlap)["train&val"], ())

    def test_missing_other_split_metadata_is_explicit_and_not_zero_overlap_claim(self):
        audio, vision, _ = fixture({"train": TRAIN})
        manifest = generate_pairs(audio, vision, config=self.config, policy=self.policy)
        report = audit_generated_manifest(manifest, audio, vision)
        for modality in report.metadata_leakage:
            self.assertEqual(modality.missing_metadata_splits, ("val", "test"))
            self.assertEqual(modality.sample_overlap, ())
            self.assertEqual(modality.source_overlap, ())

    def test_probability_and_embedding_changes_are_not_falsely_binary_authenticated(self):
        old = audit_generated_manifest(self.manifest, self.audio, self.vision)
        data = self.audio.splits["train"]
        probabilities = np.zeros_like(data.probabilities)
        probabilities[..., -1] = 1
        changed = replace(data, embeddings=np.full_like(data.embeddings, 7),
                          probabilities=probabilities)
        audio = replace(self.audio, splits=dict(self.audio.splits, train=changed))
        new = audit_generated_manifest(self.manifest, audio, self.vision)
        self.assertEqual(old, new)
        self.assertEqual(new.metadata_integrity, "VALIDATED_IN_MEMORY")
        self.assertEqual(new.artifact_binary_integrity, "NOT_VERIFIED")
        self.assertFalse(new.evidence_contents_verified)
        bad = changed.embeddings.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ContractError, "NaN"):
            audit_generated_manifest(self.manifest,
                                     replace(audio, splits=dict(audio.splits, train=replace(changed, embeddings=bad))),
                                     self.vision)

    def test_policy_evidence_declarations_are_preserved_but_never_authenticated(self):
        val_policy = policy(HardPolicyOrigin.VALIDATION_CONFUSION)
        manifest = generate_pairs(self.audio, self.vision, config=self.config, policy=val_policy)
        report = audit_generated_manifest(manifest, self.audio, self.vision)
        self.assertEqual(report.declared_hard_policy, val_policy.hard_negative_policy)
        self.assertEqual(report.declared_hard_policy.evidence_split, "val")
        self.assertFalse(report.evidence_contents_verified)
        malformed = object.__new__(HardNegativePolicy)
        malformed.__dict__.update(vars(val_policy.hard_negative_policy) | {"evidence_split": "test"})
        supplied = object.__new__(SamplingPolicy)
        supplied.__dict__.update(vars(val_policy) | {"hard_negative_policy": malformed})
        altered = object.__new__(type(manifest))
        altered.__dict__.update(vars(manifest) | {"policy": supplied})
        with self.assertRaisesRegex(ContractError, "val, never test"):
            audit_generated_manifest(altered, self.audio, self.vision)

    def test_quotas_reuse_and_exhaustion_at_integration_boundary(self):
        pairs = self.manifest.challenge_pairs
        self.assertLess(len({p.audio.record.clip_id for p in pairs}), len(pairs))
        config = replace(self.config, easy_negative_budget=3, hard_negative_budget=2,
                         challenge_budget=3, source_reuse=SourceReuse.UNIQUE_PER_TYPE)
        unique = generate_pairs(self.audio, self.vision, config=config, policy=self.policy)
        report = audit_generated_manifest(unique, self.audio, self.vision)
        self.assertEqual([c.selected for c in report.pair_type_counts], [9, 3, 2, 3])
        for kind in tuple(PairType)[1:]:
            subset = [p for p in unique.primary_pairs + unique.challenge_pairs if p.pair_type is kind]
            for side in ("audio", "visual"):
                self.assertEqual(len({getattr(p, side).record.clip_id for p in subset}), len(subset))
        for bad in (replace(self.config, easy_negative_budget=51),
                    replace(self.config, challenge_budget=9, source_reuse=SourceReuse.UNIQUE_PER_TYPE)):
            with self.assertRaises(ContractError):
                generate_pairs(self.audio, self.vision, config=bad, policy=self.policy)

    def test_invalid_inputs_fail_before_any_successful_audit_report(self):
        missing = replace(self.audio, metadata={s: r for s, r in self.audio.metadata.items() if s != "train"})
        with self.assertRaises(ContractError):
            audit_generated_manifest(self.manifest, missing, self.vision)
        records = self.vision.splits["train"].records
        bad = (replace(records[0], start=9, end=2),) + records[1:]
        vision = replace(self.vision, metadata=dict(self.vision.metadata, train=bad),
                         splits=dict(self.vision.splits, train=replace(self.vision.splits["train"], records=bad)))
        with self.assertRaises(ContractError):
            audit_generated_manifest(self.manifest, self.audio, vision)
        for changes in ({"primary_pairs": self.manifest.primary_pairs + (self.manifest.primary_pairs[0],)},
                        {"primary_pairs": tuple(reversed(self.manifest.primary_pairs))}):
            altered = replace(self.manifest, **changes)
            with self.assertRaises(ContractError):
                audit_generated_manifest(altered, self.audio, self.vision)

    def test_actual_same_split_duplicate_annotation_id_fails_while_shared_clip_rows_succeed(self):
        audio, vision, _ = fixture(dict(METADATA, train=TRAIN + (TRAIN[0],)))
        with self.assertRaisesRegex(ContractError, "duplicate"):
            generate_pairs(audio, vision, config=self.config, policy=self.policy)
        self.assertEqual(len([p for p in self.manifest.primary_pairs if p.pair_type is PairType.ORIGINAL_POSITIVE
                              and p.audio.record.clip_id == "a"]), 2)

    def test_reconstructed_same_source_cross_split_and_wrong_row_pairs_cannot_get_audit_success(self):
        positives = {p.audio.record.sample_id: p for p in self.manifest.primary_pairs
                     if p.pair_type is PairType.ORIGINAL_POSITIVE}
        template = positives["a__00"]
        cases = ({"pair_type": PairType.CROSS_CLASS_EASY_NEGATIVE,
                  "visual": positives["a__02"].visual},
                 {"visual": replace(template.visual, split="val")},
                 {"audio": replace(template.audio, row_index=(template.audio.row_index + 1) % 9)})
        for changes in cases:
            bad = object.__new__(type(template))
            bad.__dict__.update(vars(template) | changes)
            bad.__dict__["__post_init__"] = lambda: None
            manifest = object.__new__(type(self.manifest))
            manifest.__dict__.update(vars(self.manifest) | {"primary_pairs": (bad,)})
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                audit_generated_manifest(manifest, self.audio, self.vision)

    def test_report_fingerprints_canonicalize_numpy_integer_metadata(self):
        def converted(bundle):
            records = {s: tuple(replace(r, label=np.int64(r.label), start=np.int64(r.start), end=np.int64(r.end))
                                for r in rows) for s, rows in bundle.metadata.items()}
            return replace(bundle, metadata=records,
                           splits={s: replace(data, records=records[s]) for s, data in bundle.splits.items()})
        audio, vision = converted(self.audio), converted(self.vision)
        manifest = generate_pairs(audio, vision, config=self.config, policy=self.policy)
        report = audit_generated_manifest(manifest, audio, vision)
        self.assertEqual(report, audit_generated_manifest(self.manifest, self.audio, self.vision))
        self.assertIs(type(report.empty_audio_source_pairs), int)
        self.assertIs(type(report.empty_visual_source_pairs), int)

    def test_complete_report_is_stable_across_process_hash_seeds(self):
        code = ("import sys,json; from dataclasses import asdict; sys.path.insert(0,'tests'); "
                "from test_fusion_pair_integration import PairIntegrationTests; "
                "from fusion.pair_integrity import audit_generated_manifest; "
                "t=PairIntegrationTests(); t.setUp(); "
                "print(json.dumps(asdict(audit_generated_manifest(t.manifest,t.audio,t.vision)),sort_keys=True))")
        outputs = [subprocess.check_output([sys.executable, "-B", "-c", code], text=True,
                                          env=dict(os.environ, PYTHONHASHSEED=seed)) for seed in ("1", "993")]
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
