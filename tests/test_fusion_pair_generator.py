"""Small synthetic fixtures only; no actual AVE pairs, downloads, or models."""
from __future__ import annotations

import hashlib
import itertools
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fusion import Bundle, ContractError, Record, SplitArtifacts, validate_bundle
from fusion.pair_contracts import (
    HardNegativePolicy, HardPolicyOrigin, PairType, SamplingPolicy,
    SemanticCompatibility, make_sampling_provenance, pair_order_key, validate_pair_manifest,
)
from fusion.pair_generator import (
    GeneratedManifest, GenerationConfig, HardGenerationStatus,
    SourceReuse, generate_pairs, validate_generated_manifest,
)

NAMES = tuple(f"Event {i:02d}" for i in range(28)) + ("background",)


def record(clip, label, start=1, end=5):
    return Record(f"{clip}__{label:02d}", clip, NAMES[label], start, end, label)


ROWS = (record("a", 0), record("b", 1), record("c", 0), record("d", 1),
        record("e", 2), record("f", 2), record("a", 2),
        record("empty", 3, 0, 0), record("g", 3))


def bundles(metadata=None, shuffled=True):
    metadata = {"train": ROWS} if metadata is None else metadata
    result = []
    for modality in ("audio", "vision"):
        ordered = {s: tuple(reversed(rows)) if modality == "vision" and shuffled else tuple(rows)
                   for s, rows in metadata.items()}
        data = {}
        for split, rows in ordered.items():
            dim = 256 if modality == "audio" else 512
            emb = np.stack([np.tile(np.resize(np.frombuffer(hashlib.sha256(r.sample_id.encode()).digest(),
                                                           dtype=np.uint8), dim).astype(np.float16), (10, 1))
                            for r in rows])
            data[split] = SplitArtifacts(rows, emb, np.full((len(rows), 10, 29), 1 / 29, dtype=np.float16))
        b = Bundle(Path("GENERATOR_SYNTHETIC") / modality, modality, tuple(ordered), data, ordered,
                   NAMES, (), {"source": "synthetic fixture", "readme_available": True},
                   {s: len(rows) for s, rows in ordered.items()})
        assert validate_bundle(b).complete
        result.append(b)
    return tuple(result)


def hard_policy():
    return HardNegativePolicy("synthetic-only-v1", HardPolicyOrigin.MANUAL_APPROVED, NAMES,
                              ((NAMES[0], NAMES[1]), (NAMES[1], NAMES[0])),
                              "synthetic reviewer", "synthetic fixture", "e" * 64)


def combined(manifest):
    return tuple(sorted(manifest.primary_pairs + manifest.challenge_pairs, key=pair_order_key))


class PairGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.audio, self.vision = bundles()
        self.policy = SamplingPolicy(42)

    def generate(self, *, config=None, policy=None, audio=None, vision=None, **settings):
        return generate_pairs(self.audio if audio is None else audio, self.vision if vision is None else vision,
                              config=GenerationConfig("train", **settings) if config is None else config,
                              policy=self.policy if policy is None else policy)

    def test_default_generates_every_original_including_empty_and_shared_source_annotations(self):
        result = self.generate()
        self.assertEqual(len(result.primary_pairs), len(ROWS))
        self.assertEqual(result.challenge_pairs, ())
        self.assertEqual({p.audio.record for p in result.primary_pairs}, set(ROWS))
        self.assertTrue(all(p.original_pair_target == 1 for p in result.primary_pairs))
        self.assertEqual(result.report.unselected_positive_ids, ())
        self.assertEqual(result.report.candidate_evaluations, 0)
        self.assertEqual(result.report.hard_status, HardGenerationStatus.DISABLED_NO_POLICY)

    def test_shuffled_row_indices_select_the_actual_original_embeddings(self):
        result = self.generate()
        for pair in result.primary_pairs:
            self.assertEqual(self.audio.splits["train"].records[pair.audio.row_index], pair.audio.record)
            self.assertEqual(self.vision.splits["train"].records[pair.visual.row_index], pair.visual.record)
            np.testing.assert_array_equal(self.audio.splits["train"].embeddings[pair.audio.row_index, :, :32],
                                          self.vision.splits["train"].embeddings[pair.visual.row_index, :, :32])
        self.assertTrue(any(p.audio.row_index != p.visual.row_index for p in result.primary_pairs))

    def test_easy_negatives_obey_all_source_and_category_constraints(self):
        result = self.generate(positive_budget=0, easy_negative_budget=20)
        self.assertEqual(len(result.primary_pairs), 20)
        self.assertEqual(len({p.identity for p in result.primary_pairs}), 20)
        for pair in result.primary_pairs:
            self.assertIs(pair.pair_type, PairType.CROSS_CLASS_EASY_NEGATIVE)
            self.assertEqual(pair.original_pair_target, 0)
            self.assertNotEqual(pair.audio.record.clip_id, pair.visual.record.clip_id)
            self.assertNotEqual(pair.audio.record.sample_id, pair.visual.record.sample_id)
            self.assertNotEqual(pair.audio.record.category, pair.visual.record.category)
            self.assertEqual((pair.audio.split, pair.visual.split), ("train", "train"))

    def test_explicit_hard_policy_and_enablement_generate_only_approved_edges(self):
        policy = SamplingPolicy(42, hard_policy())
        result = self.generate(policy=policy, easy_negative_budget=5, hard_negative_budget=6,
                               enable_hard_negatives=True)
        hard = [p for p in result.primary_pairs if p.pair_type is PairType.CROSS_CLASS_HARD_NEGATIVE]
        easy = [p for p in result.primary_pairs if p.pair_type is PairType.CROSS_CLASS_EASY_NEGATIVE]
        self.assertEqual(len(hard), 6)
        self.assertIs(result.report.hard_status, HardGenerationStatus.ENABLED)
        self.assertTrue(all((p.audio.record.category, p.visual.record.category)
                            in policy.hard_negative_policy.category_pairs for p in hard))
        self.assertTrue(all((p.audio.record.category, p.visual.record.category)
                            not in policy.hard_negative_policy.category_pairs for p in easy))

    def test_hard_generation_requires_both_enablement_and_policy(self):
        with self.assertRaisesRegex(ContractError, "explicit enablement"):
            GenerationConfig("train", hard_negative_budget=1)
        with self.assertRaisesRegex(ContractError, "no approved hard policy"):
            self.generate(hard_negative_budget=1, enable_hard_negatives=True)
        with self.assertRaisesRegex(ContractError, "no approved hard policy"):
            self.generate(enable_hard_negatives=True)
        result = self.generate(policy=SamplingPolicy(42, hard_policy()))
        self.assertIs(result.report.hard_status, HardGenerationStatus.DISABLED_BY_CONFIG)
        self.assertFalse(any(p.pair_type is PairType.CROSS_CLASS_HARD_NEGATIVE for p in combined(result)))

    def test_invalid_reconstructed_hard_policy_cannot_enter_generator(self):
        for changes in ({"approved_by": ""}, {"evidence_sha256": "bad"},
                        {"origin": HardPolicyOrigin.VALIDATION_CONFUSION, "evidence_split": "test"}):
            hard = object.__new__(HardNegativePolicy)
            hard.__dict__.update(vars(hard_policy()) | changes)
            policy = object.__new__(SamplingPolicy)
            policy.__dict__.update(vars(self.policy) | {"hard_negative_policy": hard})
            policy.__dict__["__post_init__"] = lambda: None
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.generate(policy=policy, hard_negative_budget=1, enable_hard_negatives=True)

    def test_same_category_challenges_are_isolated_and_keep_semantic_proxy(self):
        result = self.generate(challenge_budget=8)
        self.assertEqual(len(result.primary_pairs), len(ROWS))
        self.assertEqual(len(result.challenge_pairs), 8)
        self.assertFalse(any(p.pair_type is PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE
                             for p in result.primary_pairs))
        for pair in result.challenge_pairs:
            self.assertEqual(pair.original_pair_target, 0)
            self.assertNotEqual(pair.audio.record.clip_id, pair.visual.record.clip_id)
            self.assertEqual(pair.audio.record.label, pair.visual.record.label)
            expected = (SemanticCompatibility.UNKNOWN if not pair.audio.record.end or not pair.visual.record.end
                        else SemanticCompatibility.COMPATIBLE)
            self.assertIs(pair.semantic_compatibility, expected)
            self.assertFalse(pair.event_supervision.joint_event_eligible)

    def test_challenge_only_output_does_not_fall_back_to_primary(self):
        result = self.generate(positive_budget=0, challenge_budget=2)
        self.assertEqual(result.primary_pairs, ())
        self.assertEqual(len(result.challenge_pairs), 2)
        validate_generated_manifest(result, self.audio, self.vision)

    def test_exact_quotas_and_census_match_independent_brute_force(self):
        policy = SamplingPolicy(42, hard_policy())
        result = self.generate(policy=policy, positive_budget=4, easy_negative_budget=7,
                               hard_negative_budget=3, challenge_budget=5, enable_hard_negatives=True)
        brute = {kind: 0 for kind in PairType}
        brute[PairType.ORIGINAL_POSITIVE] = len(ROWS)
        same_source = 0
        for a, v in itertools.product(ROWS, repeat=2):
            if a.clip_id == v.clip_id:
                same_source += 1
            elif a.category == v.category:
                brute[PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE] += 1
            elif (a.category, v.category) in policy.hard_negative_policy.category_pairs:
                brute[PairType.CROSS_CLASS_HARD_NEGATIVE] += 1
            else:
                brute[PairType.CROSS_CLASS_EASY_NEGATIVE] += 1
        self.assertEqual([c.selected for c in result.report.counts], [4, 7, 3, 5])
        self.assertEqual([c.eligible for c in result.report.counts], [brute[k] for k in PairType])
        self.assertEqual(result.report.same_source_exclusions, same_source)
        self.assertEqual(result.report.same_sample_exclusions, len(ROWS))
        self.assertEqual(result.report.candidate_evaluations, len(ROWS) ** 2)
        self.assertEqual(len(result.report.unselected_positive_ids), len(ROWS) - 4)

    def test_positive_subsets_are_explicit_and_list_every_omitted_identity(self):
        result = self.generate(positive_budget=3)
        selected = {p.audio.record.sample_id for p in result.primary_pairs}
        self.assertEqual(set(result.report.unselected_positive_ids), {r.sample_id for r in ROWS} - selected)
        self.assertEqual((result.report.audio_rows, result.report.visual_rows), (len(ROWS), len(ROWS)))

    def test_same_seed_and_settings_reproduce_all_output_metadata(self):
        settings = dict(positive_budget=4, easy_negative_budget=8, challenge_budget=3)
        self.assertEqual(self.generate(**settings), self.generate(**settings))

    def test_different_seeds_change_choices_when_pools_allow(self):
        identities = [tuple(p.identity for p in self.generate(policy=SamplingPolicy(seed), positive_budget=2,
                                                             easy_negative_budget=3).primary_pairs)
                      for seed in range(4)]
        self.assertGreater(len(set(identities)), 1)

    def test_generated_priorities_match_task21_top_rank_specification(self):
        from fusion.pair_contracts import PairRecord, PairSource
        policy = SamplingPolicy(42)
        provenance = make_sampling_provenance(self.audio, self.vision, split="train", policy=policy)
        candidates = []
        for ai, a in enumerate(self.audio.splits["train"].records):
            for vi, v in enumerate(self.vision.splits["train"].records):
                if a.clip_id != v.clip_id and a.category != v.category:
                    candidates.append(PairRecord(PairSource("train", a, ai), PairSource("train", v, vi),
                                                 PairType.CROSS_CLASS_EASY_NEGATIVE, policy, provenance))
        expected = sorted(candidates, key=lambda p: (p.sampling_priority, p.visual.record.sample_id,
                                                     p.audio.record.sample_id, p.pair_type.value))[:7]
        actual = self.generate(positive_budget=0, easy_negative_budget=7)
        self.assertEqual({p.identity for p in actual.primary_pairs}, {p.identity for p in expected})

    def test_row_permutations_change_indices_and_provenance_not_selected_identities(self):
        original = self.generate(easy_negative_budget=7, challenge_budget=3)
        audio, vision = bundles(shuffled=False)
        ordered = self.generate(audio=audio, vision=vision, easy_negative_budget=7, challenge_budget=3)
        self.assertEqual([p.identity for p in combined(original)], [p.identity for p in combined(ordered)])
        self.assertNotEqual(original.provenance, ordered.provenance)
        self.assertTrue(any(a.visual.row_index != b.visual.row_index
                            for a, b in zip(combined(original), combined(ordered))))

    def test_split_selection_and_metadata_order_are_independent(self):
        metadata = {"test": (record("t0", 0), record("t1", 1)),
                    "val": (record("v0", 0), record("v1", 1)), "train": ROWS}
        audio, vision = bundles(metadata)
        for split in ("train", "val", "test"):
            result = self.generate(audio=audio, vision=vision,
                                   config=GenerationConfig(split, easy_negative_budget=1))
            self.assertTrue(all(p.split == split and p.audio.split == p.visual.split for p in combined(result)))
        train = self.generate(easy_negative_budget=2)
        expanded = self.generate(audio=audio, vision=vision, easy_negative_budget=2)
        self.assertEqual([p.identity for p in combined(train)], [p.identity for p in combined(expanded)])
        self.assertEqual(train.provenance, expanded.provenance)
        self.assertEqual(expanded.report.source_validation.missing_audio_metadata_splits, ())

    def test_known_cross_split_source_overlap_is_reported_not_reassigned(self):
        metadata = {"train": ROWS, "val": (record("a", 1), record("val_only", 0))}
        audio, vision = bundles(metadata)
        result = self.generate(audio=audio, vision=vision)
        self.assertEqual([(x.modality, x.clip_ids) for x in result.report.source_validation.source_overlaps],
                         [("audio", ("a",)), ("vision", ("a",))])
        self.assertEqual(result.report.audio_rows, len(ROWS))

    def test_duplicate_source_annotations_cannot_be_swapped_as_negatives(self):
        audio, vision = bundles({"train": (record("shared", 0), record("shared", 1))})
        self.assertEqual(len(self.generate(audio=audio, vision=vision).primary_pairs), 2)
        with self.assertRaisesRegex(ContractError, "only 0 eligible"):
            self.generate(audio=audio, vision=vision, easy_negative_budget=1)

    def test_insufficient_pools_fail_without_relaxation(self):
        for settings in ({"positive_budget": len(ROWS) + 1}, {"easy_negative_budget": 1000},
                         {"challenge_budget": 1000}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ContractError, "eligible pairs"):
                self.generate(**settings)
        with self.assertRaisesRegex(ContractError, "eligible pairs"):
            self.generate(policy=SamplingPolicy(42, hard_policy()), hard_negative_budget=1000,
                          enable_hard_negatives=True)

    def test_empty_annotations_keep_identity_and_masked_supervision(self):
        result = self.generate(challenge_budget=12)
        empty_original = next(p for p in result.primary_pairs if p.audio.record.clip_id == "empty")
        self.assertEqual(empty_original.original_pair_target, 1)
        self.assertIs(empty_original.semantic_compatibility, SemanticCompatibility.UNKNOWN)
        self.assertFalse(any(vars(empty_original.event_supervision).values()))
        empty_challenges = [p for p in result.challenge_pairs if "empty" in
                            (p.audio.record.clip_id, p.visual.record.clip_id)]
        self.assertEqual(len(empty_challenges), 2)
        self.assertTrue(all(p.semantic_compatibility is SemanticCompatibility.UNKNOWN and
                            not p.event_supervision.semantic_eligible for p in empty_challenges))
        self.assertIsNone(self.audio.splits["train"].input_validity)

    def test_allow_reuse_supports_many_to_one_and_one_to_many(self):
        audio, vision = bundles({"train": ROWS[:3]})
        result = self.generate(audio=audio, vision=vision, positive_budget=0, easy_negative_budget=4)
        pairs = result.primary_pairs
        self.assertEqual(len({p.identity for p in pairs}), 4)
        self.assertLess(len({p.audio.record.clip_id for p in pairs}), len(pairs))
        self.assertLess(len({p.visual.record.clip_id for p in pairs}), len(pairs))

    def test_unique_source_mode_checks_capacity_not_just_raw_pool_count(self):
        audio, vision = bundles({"train": ROWS[:3]})
        result = self.generate(audio=audio, vision=vision, positive_budget=0, easy_negative_budget=2,
                               source_reuse=SourceReuse.UNIQUE_PER_TYPE)
        for side in ("audio", "visual"):
            self.assertEqual(len({getattr(p, side).record.clip_id for p in result.primary_pairs}), 2)
        with self.assertRaisesRegex(ContractError, "distinct-source matching.*maximum 2"):
            self.generate(audio=audio, vision=vision, positive_budget=0, easy_negative_budget=3,
                          source_reuse=SourceReuse.UNIQUE_PER_TYPE)

    def test_unique_source_matching_can_reroute_a_greedy_dead_end(self):
        from fusion.pair_contracts import PairSource
        from fusion.pair_generator import _Candidate, _unique_source_selection
        ax, ay = PairSource("train", record("ax", 0), 0), PairSource("train", record("ay", 1), 1)
        va, vb = PairSource("train", record("va", 2), 2), PairSource("train", record("vb", 3), 3)
        candidates = [_Candidate(ax, va, PairType.CROSS_CLASS_HARD_NEGATIVE, "0" * 64),
                      _Candidate(ay, va, PairType.CROSS_CLASS_HARD_NEGATIVE, "1" * 64),
                      _Candidate(ax, vb, PairType.CROSS_CLASS_HARD_NEGATIVE, "2" * 64)]
        chosen = _unique_source_selection(candidates, 2)
        self.assertEqual({(c.audio.record.clip_id, c.visual.record.clip_id) for c in chosen},
                         {("ay", "va"), ("ax", "vb")})

    def test_unique_matching_agrees_with_exhaustive_small_graph_feasibility(self):
        from fusion.pair_contracts import PairSource
        from fusion.pair_generator import _Candidate, _unique_source_selection
        a = [PairSource("train", record(f"a{i}", 0), i) for i in range(3)]
        v = [PairSource("train", record(f"v{i}", 1), i) for i in range(3)]
        possible = [_Candidate(x, y, PairType.CROSS_CLASS_EASY_NEGATIVE, f"{i:064x}")
                    for i, (x, y) in enumerate(itertools.product(a, v))]
        for mask in range(1 << len(possible)):
            edges = [edge for i, edge in enumerate(possible) if mask & (1 << i)]
            maximum = next(size for size in range(3, -1, -1)
                           if any(len({e.audio.record.clip_id for e in subset}) == size and
                                  len({e.visual.record.clip_id for e in subset}) == size
                                  for subset in itertools.combinations(edges, size)))
            with self.subTest(mask=mask):
                chosen = _unique_source_selection(edges, maximum)
                self.assertEqual(len(chosen), maximum)
                self.assertEqual({e.key for e in chosen},
                                 {e.key for e in _unique_source_selection(list(reversed(edges)), maximum)})
                if maximum < 3:
                    with self.assertRaisesRegex(ContractError, "distinct-source matching"):
                        _unique_source_selection(edges, maximum + 1)

    def test_unique_source_mode_does_not_collapse_original_annotations(self):
        result = self.generate(source_reuse=SourceReuse.UNIQUE_PER_TYPE)
        self.assertEqual(len(result.primary_pairs), len(ROWS))
        self.assertEqual(sum(p.audio.record.clip_id == "a" for p in result.primary_pairs), 2)

    def test_invalid_budgets_configs_and_fabricated_enums_rejected(self):
        for name in ("positive_budget", "easy_negative_budget", "hard_negative_budget", "challenge_budget",
                     "max_candidate_evaluations"):
            for value in (-1, True, 1.5, "1"):
                with self.subTest(name=name, value=value), self.assertRaises(ContractError):
                    GenerationConfig("train", **{name: value})
        for kwargs in ({"source_reuse": "ALLOW"}, {"enable_hard_negatives": 1},
                       {"positive_budget": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                GenerationConfig("train", **kwargs)
        with self.assertRaises(ContractError):
            GenerationConfig("validation")
        fabricated = str.__new__(SourceReuse, SourceReuse.ALLOW.value)
        fabricated._name_, fabricated._value_ = "ALLOW", "ALLOW"
        with self.assertRaises(ContractError):
            GenerationConfig("train", source_reuse=fabricated)

    def test_candidate_work_budget_fails_before_enumeration(self):
        with patch("fusion.pair_generator._candidate", side_effect=AssertionError("must not enumerate")):
            with self.assertRaisesRegex(ContractError, "before enumeration"):
                self.generate(easy_negative_budget=1, max_candidate_evaluations=1)
        result = self.generate(max_candidate_evaluations=0)
        self.assertEqual(result.report.candidate_evaluations, 0)

    def test_generated_manifest_is_immutable_and_validates_under_task21(self):
        result = self.generate(easy_negative_budget=3, challenge_budget=2)
        self.assertEqual(validate_pair_manifest(combined(result), self.audio, self.vision,
                                               policy=self.policy).pair_count, len(ROWS) + 5)
        self.assertEqual(validate_generated_manifest(result, self.audio, self.vision),
                         result.report.source_validation)
        for obj, field, value in ((result, "primary_pairs", ()), (result.config, "split", "test"),
                                  (result.report, "audio_rows", 0),
                                  (result.provenance, "settings_fingerprint", "f" * 64)):
            with self.subTest(field=field), self.assertRaises(FrozenInstanceError):
                setattr(obj, field, value)

    def test_changed_quotas_or_source_reuse_change_settings_fingerprint(self):
        first, second = self.generate(positive_budget=2), self.generate(positive_budget=3)
        self.assertNotEqual(first.provenance.settings_fingerprint, second.provenance.settings_fingerprint)
        third = self.generate(positive_budget=2, source_reuse=SourceReuse.UNIQUE_PER_TYPE)
        self.assertNotEqual(first.provenance.settings_fingerprint, third.provenance.settings_fingerprint)

    def test_manifest_validation_rejects_tampered_quotas_provenance_and_reports(self):
        result = self.generate(positive_budget=3, easy_negative_budget=2)
        cases = (
            replace(result, config=replace(result.config, positive_budget=4)),
            replace(result, provenance=replace(result.provenance, settings_fingerprint="f" * 64)),
            replace(result, report=replace(result.report, same_source_exclusions=999)),
            replace(result, primary_pairs=result.primary_pairs[:-1]),
        )
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                validate_generated_manifest(changed, self.audio, self.vision)

    def test_manifest_validation_enforces_challenge_isolation_order_and_identity(self):
        result = self.generate(challenge_budget=2)
        cases = (replace(result, primary_pairs=result.primary_pairs + result.challenge_pairs, challenge_pairs=()),
                 replace(result, primary_pairs=tuple(reversed(result.primary_pairs))),
                 replace(result, primary_pairs=result.primary_pairs + (result.primary_pairs[-1],)))
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                validate_generated_manifest(changed, self.audio, self.vision)

    def test_manifest_validator_checks_no_reuse_even_with_consistent_settings_hash(self):
        from fusion.pair_generator import GenerationProvenance, _settings
        audio, vision = bundles({"train": ROWS[:3]})
        result = self.generate(audio=audio, vision=vision, positive_budget=0, easy_negative_budget=4)
        config = replace(result.config, source_reuse=SourceReuse.UNIQUE_PER_TYPE)
        provenance = GenerationProvenance(_settings(config, result.policy), result.provenance.source_provenance)
        changed = replace(result, config=config, provenance=provenance)
        with self.assertRaisesRegex(ContractError, "source reuse"):
            validate_generated_manifest(changed, audio, vision)

    def test_invalid_reconstructed_and_subclassed_generator_inputs_rejected(self):
        class UncheckedConfig(GenerationConfig):
            def __post_init__(self):
                pass
        with self.assertRaisesRegex(ContractError, "subclasses are unsupported"):
            self.generate(config=UncheckedConfig("train", easy_negative_budget=-1))
        config = object.__new__(GenerationConfig)
        config.__dict__.update(vars(GenerationConfig("train")) | {"easy_negative_budget": -1})
        config.__dict__["__post_init__"] = lambda: None
        with self.assertRaises(ContractError):
            self.generate(config=config)
        result = self.generate()
        malformed = object.__new__(GeneratedManifest)
        malformed.__dict__.update(vars(result) | {"primary_pairs": []})
        malformed.__dict__["__post_init__"] = lambda: None
        with self.assertRaises(ContractError):
            validate_generated_manifest(malformed, self.audio, self.vision)

    def test_missing_or_inconsistent_phase1_inputs_never_produce_partial_pairs(self):
        audio = replace(self.audio, splits={"train": replace(self.audio.splits["train"], embeddings=None)})
        with self.assertRaisesRegex(ContractError, "pending"):
            self.generate(audio=audio)
        with self.assertRaises(ContractError):
            self.generate(config=GenerationConfig("val"))
        changed = replace(ROWS[0], start=2)
        _, vision = bundles({"train": (changed,) + ROWS[1:]})
        with self.assertRaisesRegex(ContractError, "conflicting"):
            self.generate(vision=vision)

    def test_numpy_integer_config_canonicalization(self):
        first = self.generate(positive_budget=3, easy_negative_budget=2)
        second = self.generate(policy=SamplingPolicy(np.int64(42)), positive_budget=np.int64(3),
                               easy_negative_budget=np.int64(2))
        self.assertEqual(first, second)

    def test_generation_does_not_change_source_arrays_or_metadata(self):
        a = self.audio.splits["train"].embeddings.copy()
        v = self.vision.splits["train"].embeddings.copy()
        self.generate(easy_negative_budget=10, challenge_budget=3)
        np.testing.assert_array_equal(self.audio.splits["train"].embeddings, a)
        np.testing.assert_array_equal(self.vision.splits["train"].embeddings, v)
        self.assertEqual(self.audio.splits["train"].records, ROWS)

    def test_generation_is_independent_of_process_hash_randomization(self):
        code = ("from test_fusion_pair_generator import bundles; "
                "from fusion.pair_contracts import SamplingPolicy; "
                "from fusion.pair_generator import GenerationConfig,generate_pairs; "
                "a,v=bundles(); m=generate_pairs(a,v,config=GenerationConfig('train',positive_budget=2,easy_negative_budget=3),policy=SamplingPolicy(42)); "
                "print(m.provenance.settings_fingerprint, [p.identity for p in m.primary_pairs])")
        code = "import sys; sys.path.insert(0,'tests'); " + code
        outputs = [subprocess.check_output([sys.executable, "-B", "-c", code], encoding="utf-8",
                                          env=dict(os.environ, PYTHONHASHSEED=value)) for value in ("1", "777")]
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
