"""Synthetic Task 2.1 contracts; no real pair generation, videos, or models."""
from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np

from fusion import Bundle, ContractError, Record, SplitArtifacts, annotation_activity, validate_bundle
from fusion.pair_contracts import (
    EventSupervision, HardNegativePolicy, HardPolicyOrigin, PairRecord, PairSource, PairType,
    SamplingPolicy, SemanticCompatibility, TemporalStatus,
    make_sampling_provenance, pair_order_key, split_seed, validate_pair_manifest,
)

NAMES = tuple(f"Event {i:02d}" for i in range(28)) + ("background",)
ROWS = (
    Record("alpha__00", "alpha", NAMES[0], 1, 4, 0),
    Record("beta__01", "beta", NAMES[1], 2, 5, 1),
    Record("gamma__00", "gamma", NAMES[0], 0, 10, 0),
    Record("alpha__02", "alpha", NAMES[2], 1, 3, 2),
    Record("VWi2ENBuTbw__08", "VWi2ENBuTbw", NAMES[8], 0, 0, 8),
)


def bundle(modality, metadata=None):
    metadata = metadata or {"train": ROWS}
    splits = {}
    for split, records in metadata.items():
        records = tuple(records)
        dim = 256 if modality == "audio" else 512
        embeddings = np.stack([
            np.tile(np.resize(np.frombuffer(hashlib.sha256(r.sample_id.encode()).digest(), dtype=np.uint8),
                              dim).astype(np.float16), (10, 1)) for r in records
        ])
        probabilities = np.full((len(records), 10, 29), 1 / 29, dtype=np.float16)
        splits[split] = SplitArtifacts(records, embeddings, probabilities)
    result = Bundle(Path("synthetic") / modality, modality, tuple(metadata), splits,
                    {s: tuple(r) for s, r in metadata.items()}, NAMES, (),
                    {"source": "synthetic fixture", "readme_available": True},
                    {s: len(r) for s, r in metadata.items()})
    assert validate_bundle(result).complete
    return result


class PairContractTests(unittest.TestCase):
    def setUp(self):
        self.audio = bundle("audio")
        self.vision = bundle("vision", {"train": (ROWS[2], ROWS[3], ROWS[4], ROWS[1], ROWS[0])})
        self.policy = SamplingPolicy(42)

    def hard_policy(self, **changes):
        fields = dict(policy_id="synthetic-neighbors-v1", origin=HardPolicyOrigin.MANUAL_APPROVED,
                      class_names=NAMES, category_pairs=((NAMES[0], NAMES[1]),),
                      approved_by="synthetic reviewer", evidence_reference="synthetic fixture only",
                      evidence_sha256="e" * 64)
        return HardNegativePolicy(**(fields | changes))

    def source(self, b, sample_id, split="train"):
        records = b.splits[split].records
        index = next(i for i, r in enumerate(records) if r.sample_id == sample_id)
        return PairSource(split, records[index], index)

    def pair(self, a="alpha__00", v="alpha__00", kind=PairType.ORIGINAL_POSITIVE,
             *, policy=None, split="train", audio=None, vision=None):
        audio = self.audio if audio is None else audio
        vision = self.vision if vision is None else vision
        policy = self.policy if policy is None else policy
        return PairRecord(self.source(audio, a, split), self.source(vision, v, split), kind,
                          policy, make_sampling_provenance(audio, vision, split=split, policy=policy))

    def validate(self, pairs, *, policy=None, audio=None, vision=None):
        return validate_pair_manifest(pairs, self.audio if audio is None else audio,
                                      self.vision if vision is None else vision,
                                      policy=self.policy if policy is None else policy)

    @staticmethod
    def reconstruct(value, **changes):
        """Simulate supplied data whose dataclass constructor was not called."""
        result = object.__new__(type(value))
        result.__dict__.update(vars(value) | changes)
        return result

    def test_valid_original_positive_uses_shuffled_original_rows(self):
        pair = self.pair()
        self.assertEqual(pair.original_pair_target, 1)
        self.assertEqual(pair.semantic_compatibility, SemanticCompatibility.COMPATIBLE)
        self.assertTrue(pair.event_supervision.joint_event_eligible)
        self.assertEqual((pair.audio.row_index, pair.visual.row_index), (0, 4))
        self.assertEqual(pair.temporal_status, TemporalStatus.NOT_ASSESSED)
        report = self.validate([pair])
        self.assertEqual(report.pair_count, 1)
        self.assertEqual(report.missing_audio_metadata_splits, ("val", "test"))
        np.testing.assert_array_equal(self.audio.splits["train"].embeddings[pair.audio.row_index, :, :32],
                                      self.vision.splits["train"].embeddings[pair.visual.row_index, :, :32])

    def test_valid_cross_class_easy_negative(self):
        pair = self.pair(v="beta__01", kind=PairType.CROSS_CLASS_EASY_NEGATIVE)
        self.assertEqual(pair.original_pair_target, 0)
        self.assertEqual(pair.semantic_compatibility, SemanticCompatibility.INCOMPATIBLE)
        self.assertTrue(pair.event_supervision.audio_annotation_eligible)
        self.assertTrue(pair.event_supervision.visual_annotation_eligible)
        self.assertTrue(pair.event_supervision.semantic_eligible)
        self.assertFalse(pair.event_supervision.joint_event_eligible)
        self.validate([pair])

    def test_different_annotation_of_same_source_is_not_a_negative(self):
        with self.assertRaisesRegex(ContractError, "different source clip"):
            self.pair(v="alpha__02", kind=PairType.CROSS_CLASS_EASY_NEGATIVE)

    def test_same_exact_sample_is_not_a_negative(self):
        with self.assertRaisesRegex(ContractError, "different source clip"):
            self.pair(kind=PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE)

    def test_same_category_different_source_challenge_is_semantically_compatible(self):
        pair = self.pair(v="gamma__00", kind=PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE)
        self.assertEqual(pair.original_pair_target, 0)
        self.assertEqual(pair.semantic_compatibility, SemanticCompatibility.COMPATIBLE)
        self.assertFalse(pair.event_supervision.joint_event_eligible)
        self.validate([pair])

    def test_pair_categories_cannot_be_interchanged(self):
        with self.assertRaisesRegex(ContractError, "different categories"):
            self.pair(v="gamma__00", kind=PairType.CROSS_CLASS_EASY_NEGATIVE)
        with self.assertRaisesRegex(ContractError, "matching category"):
            self.pair(v="beta__01", kind=PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE)
        with self.assertRaisesRegex(ContractError, "same exact sample"):
            self.pair(v="gamma__00")

    def test_cross_split_pair_is_rejected(self):
        pair = self.pair()
        with self.assertRaisesRegex(ContractError, "cross-split"):
            replace(pair, visual=replace(pair.visual, split="val"))

    def test_duplicate_pair_identity_is_rejected(self):
        pair = self.pair()
        with self.assertRaisesRegex(ContractError, "duplicate pair identity"):
            self.validate([pair, pair])

    def test_positive_requires_equal_annotation_metadata(self):
        pair = self.pair()
        altered = replace(pair.visual.record, start=2)
        with self.assertRaisesRegex(ContractError, "same exact sample"):
            replace(pair, visual=replace(pair.visual, record=altered))

    def test_missing_and_malformed_source_metadata_rejected(self):
        pair = self.pair()
        for bad in (None, "alpha__00", {}):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                replace(pair, audio=bad)
        for changes in ({"record": None}, {"split": "validation"}, {"row_index": -1},
                        {"row_index": True}, {"row_index": 0.0}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(pair.audio, **changes)
        for changes in ({"sample_id": "wrong__00"}, {"category": "background"},
                        {"start": 9, "end": 2}, {"start": 0.0, "end": 0}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(pair.audio, record=replace(pair.audio.record, **changes))

    def test_manifest_checks_actual_metadata_and_row_bounds(self):
        pair = self.pair(v="beta__01", kind=PairType.CROSS_CLASS_EASY_NEGATIVE)
        missing = Record("absent__00", "absent", NAMES[0], 1, 4, 0)
        for source in (replace(pair.audio, row_index=2), replace(pair.audio, row_index=99),
                       replace(pair.audio, record=missing),
                       replace(pair.audio, record=replace(pair.audio.record, start=2))):
            with self.subTest(source=source), self.assertRaisesRegex(ContractError, "row index or metadata"):
                self.validate([replace(pair, audio=source)])

    def test_manifest_rejects_incomplete_phase1_inputs(self):
        pair = self.pair()
        changed = replace(self.audio, splits={"train": replace(self.audio.splits["train"], embeddings=None)})
        with self.assertRaisesRegex(ContractError, "pending"):
            self.validate([pair], audio=changed)

    def test_conflicting_category_label_metadata_rejected(self):
        pair = self.pair(v="gamma__00", kind=PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE)
        changed = Record("gamma__01", "gamma", NAMES[0], 0, 10, 1)
        with self.assertRaisesRegex(ContractError, "category/index"):
            replace(pair, visual=replace(pair.visual, record=changed))

    def test_empty_original_retains_provenance_target_but_masks_event_supervision(self):
        pair = self.pair(a=ROWS[4].sample_id, v=ROWS[4].sample_id)
        self.assertEqual(pair.original_pair_target, 1)
        self.assertEqual(pair.semantic_compatibility, SemanticCompatibility.UNKNOWN)
        self.assertEqual(pair.audio.record, ROWS[4])
        self.assertFalse(annotation_activity([pair.audio.record]).any())
        self.assertEqual(tuple(vars(pair.event_supervision).values()), (False,) * 4)
        self.validate([pair])

    def test_empty_swapped_source_masks_only_its_annotation_and_pair_semantics(self):
        for a, v in ((ROWS[4].sample_id, "alpha__00"), ("alpha__00", ROWS[4].sample_id)):
            with self.subTest(a=a, v=v):
                pair = self.pair(a=a, v=v, kind=PairType.CROSS_CLASS_EASY_NEGATIVE)
                masks = pair.event_supervision
                self.assertEqual(pair.original_pair_target, 0)
                self.assertEqual(pair.semantic_compatibility, SemanticCompatibility.UNKNOWN)
                self.assertEqual(masks.audio_annotation_eligible, a != ROWS[4].sample_id)
                self.assertEqual(masks.visual_annotation_eligible, v != ROWS[4].sample_id)
                self.assertFalse(masks.semantic_eligible)
                self.assertFalse(masks.joint_event_eligible)
                self.validate([pair])

    def test_supervision_flags_reject_nonboolean_or_conflicting_eligibility(self):
        for flags in ((1, True, True, True), (False, True, True, False),
                      (False, False, False, True), (True, True, False, False)):
            with self.subTest(flags=flags), self.assertRaises(ContractError):
                EventSupervision(*flags)

    def test_numpy_integer_records_preserve_canonical_provenance_and_boolean_masks(self):
        records = tuple(replace(r, label=np.int64(r.label), start=np.int64(r.start), end=np.int64(r.end))
                        for r in ROWS)
        audio = bundle("audio", {"train": records})
        vision = bundle("vision", {"train": records})
        pair = self.pair(audio=audio, vision=vision)
        python_pair = self.pair(audio=bundle("audio"), vision=bundle("vision"))
        self.assertEqual(pair.provenance, python_pair.provenance)
        self.assertTrue(all(type(value) is bool for value in vars(pair.event_supervision).values()))
        self.validate([pair], audio=audio, vision=vision)

    def test_hard_negatives_disabled_without_policy(self):
        with self.assertRaisesRegex(ContractError, "hard negatives disabled"):
            self.pair(v="beta__01", kind=PairType.CROSS_CLASS_HARD_NEGATIVE)

    def test_explicit_hard_policy_is_directed_and_partitions_easy_bucket(self):
        policy = SamplingPolicy(42, self.hard_policy())
        pair = self.pair(v="beta__01", kind=PairType.CROSS_CLASS_HARD_NEGATIVE, policy=policy)
        self.validate([pair], policy=policy)
        with self.assertRaisesRegex(ContractError, "cannot be labeled easy"):
            replace(pair, pair_type=PairType.CROSS_CLASS_EASY_NEGATIVE)
        with self.assertRaisesRegex(ContractError, "hard negatives disabled"):
            self.pair(a="beta__01", kind=PairType.CROSS_CLASS_HARD_NEGATIVE, policy=policy)

    def test_hard_policy_requires_approval_evidence_and_valid_categories(self):
        for changes in ({"approved_by": ""}, {"evidence_reference": ""}, {"evidence_sha256": "bad"},
                        {"origin": "MANUAL_APPROVED"}, {"category_pairs": ()},
                        {"category_pairs": ((NAMES[0], NAMES[0]),)},
                        {"category_pairs": ((NAMES[0], "background"),)},
                        {"category_pairs": ((NAMES[0], "unknown"),)},
                        {"category_pairs": ((NAMES[0], NAMES[1]),) * 2},
                        {"category_pairs": ((NAMES[1], NAMES[0]), (NAMES[0], NAMES[1]))},
                        {"class_names": list(reversed(NAMES))}, {"evidence_split": "test"}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.hard_policy(**changes)

    def test_confusion_policy_requires_validation_evidence_never_test(self):
        hard = self.hard_policy(origin=HardPolicyOrigin.VALIDATION_CONFUSION, evidence_split="val")
        self.assertEqual(hard.evidence_split, "val")
        for split in (None, "test", "train", "validation"):
            with self.subTest(split=split), self.assertRaisesRegex(ContractError, "val, never test"):
                replace(hard, evidence_split=split)

    def test_hard_policy_vocabulary_bound_to_bundle_mapping(self):
        names = tuple(f"Other {i:02d}" for i in range(28)) + ("background",)
        hard = self.hard_policy(class_names=names, category_pairs=((names[0], names[1]),))
        with self.assertRaisesRegex(ContractError, "vocabulary conflicts"):
            make_sampling_provenance(self.audio, self.vision, split="train", policy=SamplingPolicy(42, hard))

    def test_immutable_records_and_policy_snapshots(self):
        pair = self.pair()
        for obj, field, value in ((pair, "pair_type", PairType.CROSS_CLASS_EASY_NEGATIVE),
                                  (pair.audio, "row_index", 9), (pair.audio.record, "start", 2),
                                  (pair.policy, "seed", 9), (pair.provenance, "root_seed", 9),
                                  (pair.event_supervision, "joint_event_eligible", False)):
            with self.subTest(field=field), self.assertRaises(FrozenInstanceError):
                setattr(obj, field, value)
        names, edges = list(NAMES), [[NAMES[0], NAMES[1]]]
        hard = self.hard_policy(class_names=names, category_pairs=edges)
        fingerprint = SamplingPolicy(42, hard).fingerprint
        names[0] = "changed"
        edges[0][0] = "changed"
        self.assertEqual(SamplingPolicy(42, hard).fingerprint, fingerprint)

    def test_seed_validation(self):
        for seed in (True, np.bool_(False), -1, 2**64, 1.5, "42", None):
            with self.subTest(seed=seed), self.assertRaises(ContractError):
                SamplingPolicy(seed)
        self.assertEqual(SamplingPolicy(np.int64(42)).fingerprint, self.policy.fingerprint)
        with self.assertRaises(ContractError):
            split_seed(42, "validation")
        with self.assertRaises(ContractError):
            SamplingPolicy(42, version="future")
        with self.assertRaises(ContractError):
            SamplingPolicy(42, hard_negative_policy={})

    def test_provenance_rejects_nondeterministic_methods_and_malformed_fields(self):
        pair = self.pair()
        for changes in ({"method": "global-random"}, {"root_seed": True}, {"split_seed": -1},
                        {"policy_fingerprint": ""}, {"audio_metadata_fingerprint": "x" * 64},
                        {"visual_metadata_fingerprint": "A" * 64}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(pair.provenance, **changes)
        for changes in ({"root_seed": 43}, {"split_seed": split_seed(42, "test")},
                        {"policy_fingerprint": "f" * 64}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ContractError, "provenance conflicts"):
                replace(pair, provenance=replace(pair.provenance, **changes))
        for provenance in (None, {}, "random"):
            with self.subTest(provenance=provenance), self.assertRaises(ContractError):
                replace(pair, provenance=provenance)

    def test_well_formed_but_wrong_metadata_hash_rejected_by_manifest(self):
        pair = self.pair()
        for field in ("audio_metadata_fingerprint", "visual_metadata_fingerprint"):
            with self.subTest(field=field), self.assertRaisesRegex(ContractError, "metadata fingerprints"):
                self.validate([replace(pair, provenance=replace(pair.provenance, **{field: "f" * 64}))])

    def test_policy_or_method_change_changes_fingerprint(self):
        self.assertNotEqual(self.policy.fingerprint, SamplingPolicy(43).fingerprint)
        self.assertNotEqual(self.policy.fingerprint, SamplingPolicy(42, self.hard_policy()).fingerprint)
        with self.assertRaisesRegex(ContractError, "conflicting sampling policies"):
            self.validate([self.pair()], policy=SamplingPolicy(43))

    def test_candidate_rank_and_split_seed_are_deterministic_and_rng_independent(self):
        before = random.getstate()
        first, again = self.pair(), self.pair()
        self.assertEqual(first.provenance, again.provenance)
        self.assertEqual(first.sampling_priority, again.sampling_priority)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(len({split_seed(42, s) for s in ("train", "val", "test")}), 3)
        self.assertNotEqual(first.sampling_priority, self.pair(policy=SamplingPolicy(43)).sampling_priority)

    def test_split_seed_stable_across_python_hash_seeds(self):
        code = "from fusion.pair_contracts import split_seed; print(split_seed(42, 'train'))"
        outputs = [subprocess.check_output([sys.executable, "-B", "-c", code],
                                          env=dict(os.environ, PYTHONHASHSEED=value), encoding="utf-8")
                   for value in ("1", "99")]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0].strip(), str(split_seed(42, "train")))

    def test_canonical_order_required_without_silently_sorting(self):
        first, second = self.pair(), self.pair(a="beta__01", v="beta__01")
        ordered = sorted([second, first], key=pair_order_key)
        self.validate(ordered)
        with self.assertRaisesRegex(ContractError, "canonical deterministic ordering"):
            self.validate(list(reversed(ordered)))
        for pairs in ([], (), set(ordered), [None]):
            with self.subTest(pairs=pairs), self.assertRaises(ContractError):
                self.validate(pairs)

    def test_invalid_enum_values_and_temporal_shift_claims_rejected(self):
        pair = self.pair()
        for kind in ("ORIGINAL_POSITIVE", "TIME_SHIFTED_NEGATIVE", None, 0):
            with self.subTest(kind=kind), self.assertRaises(ContractError):
                replace(pair, pair_type=kind)
        for status in ("SYNCHRONIZED", "NOT_ASSESSED", True, None):
            with self.subTest(status=status), self.assertRaises(ContractError):
                replace(pair, temporal_status=status)
        with self.assertRaises(ValueError):
            PairType("WAVEFORM_SHIFT_NEGATIVE")
        with self.assertRaises(TypeError):
            replace(pair, shift_seconds=3)

    def test_distinct_annotations_of_one_clip_keep_distinct_positive_identities(self):
        pairs = sorted([self.pair(), self.pair(a="alpha__02", v="alpha__02")], key=pair_order_key)
        self.assertNotEqual(pairs[0].identity, pairs[1].identity)
        self.assertEqual(pairs[0].audio.record.clip_id, pairs[1].audio.record.clip_id)
        self.assertEqual(self.validate(pairs).pair_count, 2)

    def test_known_source_overlap_reported_without_cross_split_borrowing(self):
        val = (Record("alpha__01", "alpha", NAMES[1], 1, 3, 1),)
        audio = bundle("audio", {"train": ROWS, "val": val})
        vision = bundle("vision", {"train": self.vision.splits["train"].records, "val": val})
        pairs = [self.pair(audio=audio, vision=vision),
                 self.pair(a="alpha__01", v="alpha__01", split="val", audio=audio, vision=vision)]
        report = self.validate(pairs, audio=audio, vision=vision)
        self.assertEqual([(r.modality, r.split_pair, r.clip_ids) for r in report.source_overlaps],
                         [("audio", "train&val", ("alpha",)), ("vision", "train&val", ("alpha",))])
        self.assertEqual(report.missing_visual_metadata_splits, ("test",))
        self.assertEqual(self.pair().provenance, pairs[0].provenance)
        self.assertEqual(self.pair().sampling_priority, pairs[0].sampling_priority)

    def test_metadata_row_reordering_changes_provenance_but_not_candidate_rank(self):
        pair = self.pair()
        vision = bundle("vision")
        reordered = self.pair(vision=vision)
        self.assertNotEqual(pair.provenance.visual_metadata_fingerprint,
                            reordered.provenance.visual_metadata_fingerprint)
        self.assertEqual(pair.sampling_priority, reordered.sampling_priority)
        self.validate([reordered], vision=vision)
        with self.assertRaisesRegex(ContractError, "metadata fingerprints"):
            self.validate([pair], vision=vision)

    def test_source_subclass_cannot_supply_negative_original_row_index(self):
        class UncheckedSource(PairSource):
            def __post_init__(self):
                pass

        original = self.pair(a=ROWS[4].sample_id, v=ROWS[4].sample_id)
        # -1 would select the correct last audio row if allowed to reach indexing.
        source = UncheckedSource("train", original.audio.record, -1)
        with self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
            replace(original, audio=source)
        candidate = self.reconstruct(original, audio=source)
        with self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
            self.validate([candidate])

    def test_pair_subclass_cannot_override_canonical_invariants(self):
        class UncheckedPair(PairRecord):
            def __post_init__(self):
                pass

        original = self.pair()
        cases = (
            {"pair_type": PairType.CROSS_CLASS_EASY_NEGATIVE},
            {"temporal_status": "WAVEFORM_VERIFIED"},
            {"visual": replace(original.visual, split="val")},
            {"visual": self.source(self.vision, "gamma__00")},
        )
        for changes in cases:
            candidate = UncheckedPair(**(vars(original) | changes))
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
                    self.validate([candidate])
                with self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
                    pair_order_key(candidate)

    def test_reconstructed_pair_cannot_shadow_its_validation_hook(self):
        original = self.pair()
        for changes in ({"pair_type": PairType.CROSS_CLASS_EASY_NEGATIVE},
                        {"temporal_status": "WAVEFORM_VERIFIED"},
                        {"visual": replace(original.visual, split="val")}):
            candidate = self.reconstruct(original, **changes)
            candidate.__dict__["__post_init__"] = lambda: None
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.validate([candidate])

    def test_reconstructed_source_cannot_shadow_negative_index_validation(self):
        original = self.pair(a=ROWS[4].sample_id, v=ROWS[4].sample_id)
        source = self.reconstruct(original.audio, row_index=-1)
        source.__dict__["__post_init__"] = lambda: None
        with self.assertRaisesRegex(ContractError, "nonnegative integer"):
            self.validate([self.reconstruct(original, audio=source)])

    def test_policy_provenance_and_record_subclasses_are_unsupported(self):
        original = self.pair()
        for value, field in ((original.policy, "policy"), (original.provenance, "provenance")):
            derived = type("Unchecked" + type(value).__name__, (type(value),),
                           {"__post_init__": lambda self: None})
            candidate = self.reconstruct(original, **{field: derived(**vars(value))})
            with self.subTest(field=field), self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
                self.validate([candidate])
            if field == "policy":
                with self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
                    make_sampling_provenance(self.audio, self.vision, split="train", policy=candidate.policy)

        class DerivedRecord(Record):
            pass

        with self.assertRaisesRegex(ContractError, "exact Phase 1 Record"):
            replace(original.audio, record=DerivedRecord(**vars(original.audio.record)))

    def test_reconstructed_hard_policy_fields_revalidated_on_attachment(self):
        hard = self.hard_policy(origin=HardPolicyOrigin.VALIDATION_CONFUSION, evidence_split="val")
        for changes in ({"policy_id": ""}, {"origin": "VALIDATION_CONFUSION"},
                        {"class_names": list(reversed(NAMES))}, {"category_pairs": ()},
                        {"category_pairs": ((NAMES[0], NAMES[0]),)}, {"approved_by": ""},
                        {"evidence_reference": ""}, {"evidence_sha256": "not-a-digest"},
                        {"evidence_split": "test"}, {"evidence_split": "train"},
                        {"evidence_split": None}):
            supplied = self.reconstruct(hard, **changes)
            supplied.__dict__["__post_init__"] = lambda: None
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                SamplingPolicy(42, supplied)

    def test_invalid_nested_hard_policy_rejected_at_every_public_boundary(self):
        hard = self.hard_policy(origin=HardPolicyOrigin.VALIDATION_CONFUSION, evidence_split="val")
        valid_policy = SamplingPolicy(42, hard)
        original = self.pair(v="beta__01", kind=PairType.CROSS_CLASS_HARD_NEGATIVE, policy=valid_policy)
        for changes, pattern in (({"evidence_split": "test"}, "val, never test"),
                                 ({"approved_by": ""}, "approved_by"),
                                 ({"evidence_sha256": "bad"}, "SHA-256")):
            bad_policy = self.reconstruct(valid_policy, hard_negative_policy=self.reconstruct(hard, **changes))
            candidate = self.reconstruct(original, policy=bad_policy)
            checks = (
                lambda: bad_policy.fingerprint,
                lambda: make_sampling_provenance(self.audio, self.vision, split="train", policy=bad_policy),
                lambda: replace(original, policy=bad_policy),
                lambda: self.validate([candidate], policy=bad_policy),
                lambda: self.validate([candidate], policy=valid_policy),
            )
            for index, check in enumerate(checks):
                with self.subTest(changes=changes, boundary=index), self.assertRaisesRegex(ContractError, pattern):
                    check()

    def test_hard_policy_subclass_is_rejected_recursively(self):
        class UncheckedHardPolicy(HardNegativePolicy):
            def __post_init__(self):
                pass

        supplied = UncheckedHardPolicy(**(vars(self.hard_policy()) | {"approved_by": ""}))
        with self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
            SamplingPolicy(42, supplied)
        bad_policy = self.reconstruct(self.policy, hard_negative_policy=supplied)
        with self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
            make_sampling_provenance(self.audio, self.vision, split="train", policy=bad_policy)

    def test_fabricated_enum_objects_are_not_supported_schema_members(self):
        original = self.pair()
        for enum_type, member, field in ((PairType, PairType.ORIGINAL_POSITIVE, "pair_type"),
                                         (TemporalStatus, TemporalStatus.NOT_ASSESSED, "temporal_status")):
            fabricated = str.__new__(enum_type, member.value)
            fabricated._name_, fabricated._value_ = member.name, member.value
            candidate = self.reconstruct(original, **{field: fabricated})
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.validate([candidate])
        origin = str.__new__(HardPolicyOrigin, HardPolicyOrigin.MANUAL_APPROVED.value)
        origin._name_, origin._value_ = "MANUAL_APPROVED", "MANUAL_APPROVED"
        with self.assertRaises(ContractError):
            SamplingPolicy(42, self.reconstruct(self.hard_policy(), origin=origin))

    def test_reconstructed_missing_fields_raise_contract_errors(self):
        original = self.pair()
        for value, missing, field in ((original, "audio", None), (original.audio, "row_index", "audio"),
                                      (original.policy, "seed", "policy"),
                                      (original.provenance, "root_seed", "provenance")):
            supplied = self.reconstruct(value)
            del supplied.__dict__[missing]
            candidate = supplied if field is None else self.reconstruct(original, **{field: supplied})
            with self.subTest(missing=missing), self.assertRaisesRegex(ContractError, "missing required"):
                self.validate([candidate])

    def test_valid_hard_policy_and_negative_source_reuse_remain_supported(self):
        hard = self.hard_policy(category_pairs=((NAMES[0], NAMES[1]), (NAMES[1], NAMES[0])))
        policy = SamplingPolicy(42, self.reconstruct(hard))
        self.assertEqual(policy.fingerprint, SamplingPolicy(42, hard).fingerprint)
        pairs = [self.pair(a=a, v=v, kind=PairType.CROSS_CLASS_HARD_NEGATIVE, policy=policy)
                 for a, v in (("alpha__00", "beta__01"), ("gamma__00", "beta__01"),
                              ("beta__01", "alpha__00"), ("beta__01", "gamma__00"))]
        report = self.validate(sorted(pairs, key=pair_order_key), policy=policy)
        self.assertEqual(report.pair_count, 4)
        self.assertTrue(all(pair.original_pair_target == 0 for pair in pairs))


if __name__ == "__main__":
    unittest.main()
