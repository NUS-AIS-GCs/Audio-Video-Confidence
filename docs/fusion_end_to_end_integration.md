# Phase 6 Tasks 6.4–6.6: strict encoder integration

Date: **2026-10-26, Asia/Singapore**. Status: **PARTIAL — interfaces, decoded-input orchestration,
synthetic tests and real frozen-feature replay implemented; genuine raw encoder execution BLOCKED**.
Neither trained raw encoder has been executed here. Their actual checkpoint reconstruction and raw-to-export
parity remain unverified. Synthetic execution is never evidence of trained encoder readiness.

## Scope and preflight

Repository: `NUS-AIS-GCs/Audio-Video-Confidence`; branch `feature/phase6-end-to-end-inference`;
HEAD `b6ea972ae3ac489a2aecee9c7ab84877c3c44be0`. The audited autocast correction and all prior Phase 6
work were present before this task. Phase 5 is included in branch ancestry.
An authenticated read-only GitHub check confirmed [Phase 5 PR #17](https://github.com/NUS-AIS-GCs/Audio-Video-Confidence/pull/17)
remains open/unmerged, with that head and base `main`. Remote main remains
`ced1c3a1abf41e4c2bc05e2eeff96530e185249a`. Phase 5 remains an upstream merge dependency.

Added: `fusion/encoder_interfaces.py`, `fusion/end_to_end.py`, `tests/test_fusion_end_to_end.py`, this document.
The sole edit to an earlier file is the requirements-checksum paragraph in
`docs/fusion_end_to_end_inference_foundation.md`. Existing audio/vision producers, fusion algorithms,
checkpoint loaders, inference contracts, autocast correction and experiment records are unchanged.

The corrected checksum is the **actual byte-level SHA-256** of `requirements-fusion-training.txt`:
`0b6230b82b66ec1d2537681d7c5cdc7e0a91e660bef01ba154fe8a68bdd23e08` (141 bytes, CRLF).
The old digest `a01116f4399965d31d989d92ec8d14a07cd518b55208d42a9081efd91f2848ab`
is now explicitly labeled LF-normalized content (137 bytes). The requirements file and original preservation
manifest were not rewritten. A local-file-aware documentation regression check verifies the actual digest.

## Implemented path and supported inputs

`MediaFusionPipeline` performs readiness checks for both encoders **before either executes**, then:

1. `DecodedTensorDecoder` verifies and loads pinned decoded waveform/frame arrays and a timing manifest.
2. Canonical media/timing/validity validation runs before inference.
3. Audio encoder preprocessing/inference runs, followed by visual preprocessing/inference.
4. `EncoderOutput` and `EncodedBundle` validate shapes, provenance, original temporal indices and masks.
5. `TrainParityFusion` compares encoder-produced float16 export-roundtrip features to the corresponding
   byte-verified TRAIN exports. No export tensors substitute for mismatching encoder output.
6. Both existing `FrozenFusionInference` adapters consume the **same FeatureBatch** and produce structured
   E3/A2 correspondence results. Encoder/decoder provenance and source IDs remain separate diagnostics.

This is a **decoded-input integration path**. Its input is a `DecodedTensorSource` containing three explicit
`Artifact(path, sha256)` bindings: waveform `.npy`, RGB frame `.npy`, and a JSON manifest. No implicit discovery,
network download, pickle-enabled NumPy loading or imputation occurs. All three hashes are checked before and
after loading. The arrays are copied into owned CPU tensors. Corrupt sources fail with structured errors.

**Compressed `.mp4`/audio-file decoding is not implemented or enabled.** Such input is explicitly rejected;
it does not reach an encoder. A future verified decoder must reproduce explicit FFmpeg mono/resampling and
OpenCV RGB/INTER_LINEAR sampling/resizing with recorded versions and real source timestamps. The existing
producer decoders silently trim/pad/substitute, so they are not called as a strict decoder fallback.
No appropriate TRAIN raw clips, decoded packs or trusted decoder manifests are currently supplied.

## Decoded media and timing contract

| Field | Required contract |
|---|---|
| Waveform | CPU float32 `(B,160000)`, mono, finite, within `[-1,1]`; explicit 16000 Hz. |
| Frames | CPU RGB uint8 `(B,10,224,224,3)`, already explicitly sampled/resized upstream. |
| Durations/origins | Both durations exactly ten seconds; both local starts zero on one upstream verified media clock. |
| Frame timestamps | Original `(0,1,...,9)` seconds; duplicate, reordered, offset or fractional-bin declarations rejected. |
| Validity | Explicit boolean `(B,10)` per modality, independent of AVE labels. Invalid raw bins are rejected. |
| Identities | One TRAIN audio sample ID and visual sample ID per batch row, separate from predictive features. |
| Decoder | Explicit expected decoder signature digest and manifest timing method `shared_clock_pts_v1`. |
| Sources | Exact waveform/frame/manifest byte digests; manifest binds both array hashes. |

The JSON manifest has exactly these keys:

```text
schema (= gc2-decoded-ten-bin-v1), decoder_signature, timing_verification,
waveform_sha256, frames_sha256, sample_rate, audio_duration, video_duration,
audio_start_seconds, video_start_seconds, frame_timestamps,
audio_sample_ids, visual_sample_ids, audio_validity, visual_validity
```

Additional label/category/event fields are rejected. Numeric masks are not coerced to booleans.
`RawMediaInput.validate()` from the existing foundation enforces the canonical tensor contract.
The full ten-second waveform is necessary because BEATs contextualizes across bins: masking fusion output
cannot repair missing raw audio. No new resampling, downmix, crop, stretch, black-frame or last-frame policy
is introduced. Variable-frame-rate and arbitrary-duration compressed media remain unsupported.

The decoder signature is supplied by the trusted integration operator and should hash the external decoder's
complete version/configuration description. Hashes authenticate supplied bytes, not the truth of a malicious
decoder's metadata. This implementation checks a trusted decoder's declared clocks/PTS; it does **not** independently
measure original stream synchronization or authenticate arbitrary user assertions. Diagnostics explicitly record
`physical_synchronization_independently_verified = false`. Genuine source-clock verification is a prerequisite
for a future compressed-media decoder, not a completed claim here.

## Executable encoder interfaces and construction gates

`EncoderSpec` requires the selected run identity, immutable named artifact bindings, full **canonical JSON from
the actual saved configurations**, exact preprocessing descriptor, and pinned required runtime versions.
`readiness(spec)` reports all missing/hash-mismatched artifacts and runtime mismatches without building a model.
No production specs with invented hashes or invented BEATs base configuration were created in this task.

`load_encoder(spec, device="cpu")` checks every artifact/dependency first, loads checkpoints with
`torch.load(..., map_location="cpu", weights_only=True)`, verifies hashes again and validates state coverage,
tensor shapes/dtypes/finiteness. Other devices are rejected. A safe-loader incompatibility is an actionable
failure; unsafe pickle fallback is forbidden. Model construction uses a CPU RNG fork and never downloads weights.
No encoder object is returned before all required trained state has loaded. Direct public construction of
`FrozenEncoder` without the internal loaded-state capability is rejected.

### Audio

Required roles, in order: `base`, `trained`, `normalization`. Supported identity:
`beats_lora_r8_v1 / beats_iter3`. Configuration JSON must bind both `base_config` and `trained_config` exactly
to checkpoint payloads; `trained_config` must match the audited selected producer:

```json
{"alpha":16.0,"emb_dim":256,"encoder":"iter3","features":"beats_iter3","kind":"beats_lora","r":8}
```

The base must be self-supervised BEATs with 12 layers, width 768 and patch size 16. Other base settings are read
from the actual pinned checkpoint and matched to the explicit complete configuration; they are not guessed.
The safe builder reuses `BEATs`, `inject_lora`, `EmbeddingHead` and the producer's `BeatsLoRA.tokens`, `embed`
and `train` methods through a constructor-only subclass. It bypasses the old unpinned checkpoint loader and
cache-dependent `norm_stats()` constructor, not the embedding computations.

The base state loads strictly first. Every trained q/v LoRA and head tensor must be present; frozen base
weights are retained. The separate pinned normalization JSON must declare `encoder="BEATs iter3"`, `dim=768`,
finite 768D means and positive finite stored stds. Stored std values are used verbatim, without recomputing or
adding epsilon. Unlike the historical loader, the new path does not need `train_X.npy` just to infer layout.

Inherited preprocessing uses the original float32 Kaldi fbank (128 bins, 25 ms windows, 10 ms hop,
amplitude scale 32768, `(fbank-15.41663)/(2*6.55582)`), frequency/time-centre pooling, frozen mean/std and
768→256 GELU head. Output must be finite CPU float32 `(B,10,256)`.

### Visual

Required role: `trained`; identity `resnet18_v1`. Explicit `trained_config` must match the saved checkpoint
configuration and select `backbone=resnet18`, `emb_dim=512`. The builder reuses `VisionModel(pretrained=False)`
and strictly loads its entire trained state, including the separate classification head. No ImageNet-only
or randomly initialized backbone is accepted as a substitute.

Upstream decoded RGB uint8 frames are converted to NCHW float32, divided by 255 and ImageNet-normalized with
means `(0.485,0.456,0.406)` and stds `(0.229,0.224,0.225)`, reproducing the producer dataset's inference path.
`VisionModel.embed` supplies the 512D ResNet representation with Identity projection. No augmentation or
classification logits enter fusion. Output must be finite CPU float32 `(B,10,512)`.

Both adapters freeze parameters, restore eval mode before embedding, use local disabled CPU autocast and
no-grad, and check state fingerprints/artifact hashes before and after inference. Architecture representation
and LoRA scales are also checked for mutation. These factories are implemented but **not validated against
the absent real encoder artifacts**; actual payload compatibility can only be established when those arrive.

## Fusion compatibility and result semantics

New raw features cannot simply copy historical export provenance. `TrainParityFusion` requires configured
encoder stamps, rejects synthetic bundles, checks TRAIN sample membership, and demands **exact equality after
the historical float16 export roundtrip** against the original TRAIN embedding arrays. A mismatch raises
`PARITY_UNVERIFIED` before either fusion model executes. Only after content equality is established do those
produced values receive the corresponding verified export provenance for the unchanged foundation adapter.
This is a deliberately conservative TRAIN-reference gate; it does not support unseen media or assert that CPU
encoder outputs will match historical GPU exports. Partial masks are mechanically propagated/tested with stubs,
but real raw parity requires every bin valid because the historical exports contain no decode-validity evidence.

`TrainParityFusion` without encoder stamps supports `predict_features(FeatureBatch)` only. It safely loads the
actual E3/A2 checkpoints and verified TRAIN exports, checks common producer evidence, and passes exactly the
same features/masks to both adapters. Calling its raw bridge without actual encoder provenance is BLOCKED.

| Model | Frozen threshold | Interpretation |
|---|---:|---|
| E3 MLP, primary learned candidate | `0.009579099009671126` | Float64 sigmoid of float32 correspondence logit. |
| A2 video-to-audio attention reference | `0.0026640388978268056` | Same numeric convention, its own checkpoint/threshold binding. |

Decision remains `score >= threshold`: 1/original versus 0/substituted pairing provenance. Scores are
uncalibrated and do not establish physical sound-source causality. The existing autocast correction, model
loaders and threshold policies are reused unchanged. No optional JSD path was added to raw orchestration.

`run()` returns prediction dictionaries, shape/mask schema and separate encoder/decoder/source diagnostics.
`try_run()` returns structured `{status, error:{code,stage,message}}` on failure. Unknown exceptions become
stage-specific rejection; missing assets produce BLOCKED. Errors never select a fallback model or stream.
Production construction requires the concrete decoder/encoder/fusion adapters. Explicitly opted-in stubs
produce `status="SYNTHETIC"`. Successful non-synthetic TRAIN comparison produces
`TRAIN_EXPORT_PARITY_MATCHED`, subject to the admission policy below. The previous
`TRAIN_RAW_PARITY_VERIFIED` status is no longer emitted: these interfaces cannot independently authenticate
encoder execution, and feature equality alone cannot justify that stronger claim.

### Execution-mode remediation (2026-10-26)

The independent audit found a medium-severity defect: execution mode was computed once into a public writable
`pipeline.synthetic` boolean, component types were checked only at construction, and the result status trusted
that boolean. Independently reproduced before correction, `pipeline.synthetic = False` upgraded an entirely
synthetic execution to `TRAIN_RAW_PARITY_VERIFIED`. A stub returning E3/A2-shaped objects could reach the same
status assignment without the real parity gate. Component replacements/changed mode declarations could also
invalidate the original construction checks. Direct `predict_features` is feature inference, not raw admission;
direct `TrainParityFusion.predict` establishes value agreement but cannot authenticate a caller's bundle stamp
or `synthetic=False` declaration. Neither direct API assigns an execution-verification status.

`MediaFusionPipeline` now uses a frozen dataclass with slots. Its mode, opt-in and four component references
cannot be reassigned or deleted through normal attribute operations. Construction binds each explicit boolean
component mode, decoder signature, both producer stamps and the production E3/A2 adapter identities. Every run
rechecks these bindings before readiness, after readiness, after decoding and each encoder, after parity, and
after prediction serialization. Any changed component mode is rejected even if the aggregate mode would still
be synthetic. Production requires the exact concrete adapter types, both executing producer stamps bound to
the real gate, and unchanged concrete frozen fusion adapters. Instance method replacements in the production
path are rejected. The non-synthetic path invokes `TrainParityFusion.predict` directly, which invokes its
concrete feature bridge only after both modalities pass TRAIN identity, mask and exact export comparison.
Merely returning a parity-shaped object or a caller-provided success flag is insufficient for admission.

| Top-level status | Required execution evidence / meaning |
|---|---|
| `SYNTHETIC` | Explicit opt-in, at least one synthetic component, unchanged bound modes/producers, successful tensor/alignment checks and E3/A2 result shape. It establishes orchestration behavior only, even with exactly matching TRAIN features or nested stub success claims. |
| `TRAIN_EXPORT_PARITY_MATCHED` | Explicitly non-synthetic path through the concrete adapters, both bound producer outputs validated, source identities admitted by the real TRAIN parity gate, both modalities matching float16 exports, and both frozen fusion calls completed. It establishes TRAIN feature compatibility under trusted in-process components and external producer metadata; it does not independently authenticate encoder execution. |
| `BLOCKED` | Required assets or raw producer provenance are missing; no success payload. |
| `REJECTED` | Invalid input, changed binding, incomplete output or execution/parity failure; no success payload. |

No cached success evidence is stored on the pipeline. Each call must finish all gates anew; modality failures,
parity failures and serialization-time mode changes return no partial predictions through `try_run`.
Diagnostic metadata remains separate from the common E3/A2 model inputs. Existing prediction/feature-schema/
diagnostics keys are preserved; the stronger non-synthetic status string is intentionally replaced.

This protects the supported configuration API, not arbitrary code running in the Python process. Trusted code
can bypass Python immutability using reflection, replace class implementations or alter process memory; these
interfaces provide no execution attestation or concurrency security boundary. External decoder clock/RGB/source
identity assertions and the operator's artifact/configuration/hash bindings still require trusted evidence.
Copied producer stamps and matching exports cannot distinguish a real forward pass from replayed values.
Consequently **no genuine execution-verified status is offered**, and the missing raw assets remain a blocker.

Assembly, once actual verified specs and a decoded source exist:

```python
audio = load_encoder(audio_spec)  # actual artifacts/configurations/hashes; no defaults
visual = load_encoder(visual_spec)
fusion = TrainParityFusion(e3_directory, a2_directory, audio_export_root, visual_export_root,
                           audio.provenance, visual.provenance)
pipeline = MediaFusionPipeline(DecodedTensorDecoder(expected_decoder_signature), audio, visual, fusion)
result = pipeline.try_run(decoded_tensor_source)
```

The names above are caller-supplied verified specifications, not available local fixtures. Missing assets
raise a structured BLOCKED error during `load_encoder`, before assembly or branch execution.

## Verification and current readiness

31 new offline tests exercise execution order, tensor shapes, temporal indices, validity propagation, pinned
decoded-source loading, malformed/corrupt input, duration/rate/origin/clock rejection, missing checkpoints and
normalization, digest mismatch, safe deserialization, strict state coverage/dtype checks, no direct untrained
construction, explicit synthetic labeling, no metadata leakage, encoder-output validation, parameter stability,
deterministic orchestration, autocast restoration, exact TRAIN parity gating and identical E3/A2 feature inputs.
They use deterministic stubs and synthetic compatibility fixtures; none is evidence of actual trained encoder
execution. The checksum test skips cleanly when the pre-existing untracked local requirements file is absent.

The new **real feature-replay** bridge was separately exercised on the same four verified TRAIN pairs as the
foundation smoke. E3/A2 logits and scores reproduced `_local/phase6_verified_smoke.json` bitwise, with decisions
`original, original, substituted, substituted`. Enclosing CPU bfloat16/float16 contexts gave identical outputs
and restored caller state. This loaded frozen fusion weights, not raw encoders, and generated no new embeddings.
Evidence: `_local/phase6_integration_feature_smoke.json`; focused tests:
`_local/phase6_integration_targeted_tests.log`; full suite: `_local/phase6_integration_full_tests.log`.
At the original integration implementation, the full required command
`.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v` passed:
**553 tests, 551 passed, two unavailable-CUDA skips, zero failures/errors, Python exit code 0**,
in 45.333 seconds. This adds 31 tests to the 522-test baseline. All 31 focused tests passed in 0.152 seconds.
The 558-file historical preservation inventory remains unchanged. All earlier Phase 6 files/evidence except
the explicitly corrected checksum paragraph remain unchanged; the Git index is empty and unchanged.
W&B cache directories retain their prior exclusion from byte-preservation claims. Current source hashes,
scope/credential checks and preservation results are recorded separately in
`_local/phase6_integration_scope_verification.json`; no historical verification manifest was overwritten.

| Requirement | Current local status |
|---|---|
| BEATs iter3 base checkpoint, trusted hash/full config | Missing; no non-fusion encoder checkpoint found. |
| Trained `beats_lora_r8_v1` LoRA/head checkpoint/config/hash | Missing. |
| Frozen `beats_iter3` normalization JSON/hash | Missing. |
| Trained `resnet18_v1` state/config/hash | Missing. |
| PyTorch / NumPy | Available (`2.14.1+cpu` / `2.4.6`). |
| torchaudio, torchvision, soundfile, pandas, tqdm | Missing. |
| OpenCV, FFmpeg, ffprobe for a future verified source decoder | Missing. |
| TRAIN raw clips / trusted decoded tensor pack and source-clock manifest | Missing. |
| Actual CPU raw-to-export reconstruction/parity | Not executed; unverified. |
| Frozen E3/A2 verified feature inference | Executed successfully; unchanged. |

No unknown weights or dependencies were downloaded. The real-data stage stopped at this gate. Raw-to-export
parity requires obtaining all exact assets with trusted hashes/configurations, installing matching runtime
versions, producing a verified TRAIN decoded pack with genuine source-clock/PTS evidence, then executing both
real encoders and passing the export-roundtrip gate. If real CPU outputs differ, do not relax the comparison
or claim parity without a separate scientific compatibility investigation. Future unseen-media deployment
requires its own preprocessing/provenance admission contract; no synthetic certificate was invented here.

No training, threshold tuning, challenge/test inference, W&B run, commit, push, merge or PR occurred. The
existing single-seed, development-validation reuse, coarse-bin, calibration, synchronization and causal
limitations remain. **READY FOR INDEPENDENT INTEGRATION AUDIT** applies to this partial decoded-input foundation,
not to a claim of completed raw-media end-to-end inference.

The subsequent provenance remediation changes only `fusion/end_to_end.py`,
`tests/test_fusion_end_to_end.py` and this document. Seventeen additional regression methods cover the original
flag exploit, each bound component reference and mode, producer/decoder changes, mode mutation during stages
and serialization, matching-export synthetic fixtures, forged parity results, production method/adapter
replacement, the concrete gate's producer binding, partial failure and repeated calls without stale success.
All 48 focused methods pass. The concrete admission control-flow fixture patches encoder and fusion methods;
it exercises the actual decoder and TRAIN comparison logic but is explicitly **not proof of real encoder
execution**. Its successful status remains `TRAIN_EXPORT_PARITY_MATCHED`.

Independent before/after reproduction confirms the assignment now raises `FrozenInstanceError`; subsequent
execution remains `SYNTHETIC`. The real four-pair TRAIN feature bridge reproduces the saved smoke's E3/A2
logits, scores, thresholds and decisions bitwise at the original default CPU thread setting. Enclosing CPU
bfloat16/float16 gives identical predictions and preserves caller state, input tensors and frozen model state.
A preliminary probe with an explicitly different two-thread setting showed normal tiny GEMM rounding drift;
the default-thread comparison is bitwise exact. No raw encoder or new embedding generation was run.
The required full suite completed with **570 tests: 568 passed, two CUDA skips, zero failures/errors,
Python exit code 0**, in 99.885 seconds. This adds 17 regression methods to the audited 553-test baseline.
All 558 historical protected files, requirements bytes and Git index remain unchanged; the earlier W&B cache
exclusion still applies. No files are staged, and only the three remediation files changed in this task.
The correction is **READY FOR TARGETED RE-AUDIT**; genuine raw-encoder readiness remains unverified.

## Parity-reference binding remediation (2026-10-26)

The next targeted audit confirmed that the original flag exploit was closed but found a separate gap in the
same admission chain: `TrainParityFusion.audio` and `.visual` were writable. An in-memory bundle substitution
could change what counted as a TRAIN match. Independently reproducing both substitutions before this correction
changed `REJECTED` into `TRAIN_EXPORT_PARITY_MATCHED` and executed both fusion models. Archived files stayed
unchanged, so a comparison against the substituted cache did not prove agreement with the authenticated export.

The corrected chain reuses the existing Phase 1 bundle and Phase 4/5 checkpoint contracts:

1. `FrozenFusionInference` verifies the hard-pinned E3/A2 checkpoint, metadata and operating-point bytes.
2. `load_verified_train_exports` obtains producer identities and the six TRAIN/source file SHA-256 values per
   modality from that admitted metadata, checks actual files before and after loading, and validates TRAIN-only
   bundles with the existing contracts. Both models must bind the same producer evidence. No new archive format,
   historical manifest, external digest parameter or threshold policy is introduced.
3. `TrainParityFusion` is now a frozen dataclass with slots. Its original reference objects, producer evidence,
   expected encoder stamps and integrity bindings cannot be reassigned through normal attribute operations.
   A private frozen binding captures the original bundle identity, immutable file-path/hash pairs, and a runtime
   content fingerprint **only after** authenticated loading. File digests are copied as primitive tuples, not
   subsequently read from a mutable `Bundle.provenance` dictionary.
4. Every raw parity call verifies **both** original bundles before comparing either modality, and repeats the
   integrity check after both comparisons, before either fusion forward. It hashes the actual source files
   against the original admitted SHA-256 values and checks all live reference contents against their construction
   fingerprints. It never refreshes its expected fingerprints from caller-modified state or caches a successful
   admission for later calls. This adds deliberate file I/O and full-reference validation to each raw admission.
5. The runtime fingerprint covers array dtype, shape and actual logical bytes (embeddings, probabilities and
   optional masks/timestamps), ordered records and metadata, roots, modalities, requested splits, class vocabulary,
   expected counts, missing inputs and provenance metadata. Producer evidence has a separate frozen fingerprint.
   It preserves original row identities/order and the existing ten-bin/export-precision policy. Read-only NumPy
   mappings and frozen metadata block ordinary mutations; the content checks also detect changed nested state
   when an array or metadata object is altered without replacing the outer reference.

The fingerprint is a memory-integrity check anchored to previously authenticated file contents, **not an
independent source of authenticity**. Recomputing a caller's claimed digest cannot update the frozen admission
baseline. A caller cannot supply replacement pins through the constructor. Changing a source file and updating
its cached claimed SHA-256 still fails against the original file pin. Replacing only a reference via reflection
also fails its identity check; changing only live array contents fails its content fingerprint.

Missing authenticated bindings produce `BLOCKED`. Changed references or metadata produce `REFERENCE_INTEGRITY`
rejection; changed source bytes produce `HASH_MISMATCH`. Missing source files also fail closed. No predictions or
parity-success status are returned on these failures. Ordinary feature-only `predict_features` remains a distinct
API and makes no raw parity or encoder-execution claim.

After correction, both independent substitution attempts raise `FrozenInstanceError`; the mismatched inputs
remain `REJECTED` with **zero fusion calls**. Unchanged archived TRAIN comparisons still succeed under exact
float16 export roundtrip, without relaxing the numerical policy. These checks use replay control-flow fixtures
and frozen fusion models, not real encoder execution. Matching exported values cannot authenticate a forward
pass or the truth of source-clock/RGB assertions. `TRAIN_EXPORT_PARITY_MATCHED` retains that limited meaning;
`SYNTHETIC` replay remains explicitly synthetic, and no physical encoder-execution status is emitted.

Sixteen added regression methods exercise both substitutions, both in-place mutation paths, independent source
file hashing and recomputed hash claims, reordered rows, source identity/producer changes, nested metadata,
missing evidence, repeated failure/success calls, synthetic file-backed replay, and reference changes during
comparison. All 64 focused methods pass. The file-backed offline fixtures pin real temporary file bytes and
reuse the existing bundle loader, but deliberately bypass checkpoint construction for unit testing; they are
not evidence of genuine producer checkpoint readiness. Temporary mappings are explicitly closed on Windows.

Only `fusion/end_to_end.py`, `tests/test_fusion_end_to_end.py` and this document change in this remediation.
The actual four-pair E3/A2 TRAIN feature bridge remains bitwise identical to the saved smoke in ordinary CPU
float32 and enclosing bfloat16/float16 autocast, with unchanged thresholds, inputs, frozen weights and caller
state. No raw encoders ran, no encoder-derived embeddings were generated, and no historical exports/manifests
were regenerated. The new synthetic file fixtures exist only in test temporary directories.

Trusted in-process implementation code, SHA-256 and the originally admitted checkpoint metadata remain the
trust anchors. Python immutability alone is not cryptographic authentication: arbitrary reflective replacement
of all private anchors, class-code replacement, hostile native code and concurrent mutation across checks are
outside this API's threat model. The supported public reference replacement and mutable-cache paths are now
closed without using caller-provided success tokens. External producer artifacts and decoder assertions still
need trusted admission; missing raw assets remain a blocker to actual encoder reconstruction and execution.

Final verification: the required `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v` completed
with **586 tests: 584 passed, two CUDA skips, zero failures/errors, Python exit code 0**, in 151.439 seconds.
All 64 focused methods passed, including the 16 added reference-integrity methods. Independent checks hashed
the actual admitted TRAIN files, repeated both original substitutions and both nested-array mutation probes,
and confirmed zero fusion calls on rejection and unchanged model fingerprints. All 558 protected historical
files and the pre-existing requirements bytes remain unchanged; the existing W&B cache exclusion still applies.
The Git index remains empty and unchanged. **READY FOR TARGETED RE-AUDIT** applies to this reference-binding
correction; it does not establish genuine raw encoder readiness.
