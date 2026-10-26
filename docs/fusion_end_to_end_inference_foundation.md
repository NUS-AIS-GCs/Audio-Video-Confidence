# Phase 6 Tasks 6.1–6.3: inference foundation

Audit date: 2026-10-25, Asia/Singapore. **Foundation implemented; raw-media execution BLOCKED.**
The encoder inventory, strict interfaces, frozen feature adapter and offline tests are implemented.
Real E3/A2 feature inference succeeded on four existing TRAIN pairs. Neither raw encoder was executed;
raw-to-export parity and unseen raw-media inference remain unverified. No training, threshold fitting,
challenge evaluation, held-out evaluation, live tracking, application work or producer changes occurred.

## Repository preflight and upstream dependency

Repository remote: `https://github.com/NUS-AIS-GCs/Audio-Video-Confidence.git`.
Required branch: `feature/phase6-end-to-end-inference`, created from
`b6ea972ae3ac489a2aecee9c7ab84877c3c44be0`, the current Phase 5 branch head.
The initial worktree had only the pre-existing untracked `requirements-fusion-training.txt`.
No stash, commit, push, merge or PR was created in this task.

Authenticated read-only GitHub verification confirmed [Phase 5 PR #17](https://github.com/NUS-AIS-GCs/Audio-Video-Confidence/pull/17)
is **open, unmerged**, targeting `main`, with that same head. Remote main is
`ced1c3a1abf41e4c2bc05e2eeff96530e185249a`; local `origin/main` matches.
Phase 4 PR #16 is in main; Phase 4 commit `76add86568daed6f7dbb31659f26c020b264962c`
is an ancestor of this branch. Main is zero commits ahead and three behind this branch.
Phase 6 retains the Phase 5 dependency until #17 merges; do not assume Phase 5 is available on main.
Phase 1 contracts, Phase 2 pairs/manifests, Phase 3 baselines, Phase 4 models/loaders and Phase 5
attention/loaders are present and exercised by the regression suite.

## Task 6.1: actual capability inventory

READY means executed successfully here; PARTIAL means code/evidence exists with unverified execution
or parity; BLOCKED means a required local artifact/dependency is absent.

| Component | Status | Factual evidence / boundary |
|---|---|---|
| BEATs architecture | PARTIAL | Vendored `audio/beats/BEATs.py`, `backbone.py`, `modules.py`; executable calls exist, not run here. |
| Self-supervised BEATs iter3 weights | BLOCKED | No encoder `.pt`/`.pth` in repository/local artifacts. Expected `_local/drive/pretrained/BEATs_iter3*.pt`, excluding AS2M/finetuned variants. Iter3 configuration has no pinned SHA-256. |
| Trained LoRA and 256D head | BLOCKED | Export declares `beats_lora_r8_v1 / beats_iter3`. Expected `_local/drive/artifacts/audio_baseline/beats_lora_r8_v1/best.pt` absent. Training saves LoRA/head tensors only, not frozen BEATs or mean/std. |
| Frozen audio normalization | BLOCKED | `_local/drive/cache/beats_iter3/norm_stats.json` absent. Existing `norm_stats()` also opens `train_X.npy` to infer layout; that cache is absent. |
| Audio waveform preprocessing/export implementation | PARTIAL | `extract_audio`, `features.load_wav`, `BeatsLoRA.embed`, `evaluate.predict`, `export_embeddings` form a code path; never reconstructed locally. |
| Audio TRAIN exports | READY | Verified original bytes, finite `(3339,10,256)` export, run declaration and schema. Does not prove producer checkpoint origin or raw parity. |
| ResNet-18 architecture / heads | PARTIAL | `vision/model.py`: torchvision ResNet-18, fc replaced by Identity, Identity projection at 512D, separate 29-class head. Not executed locally. |
| Trained visual checkpoint | BLOCKED | Expected `_local/drive/artifacts/vision_baseline/resnet18_v1/best.pt` absent. `SOURCE.txt` records a teammate's external Windows path, not a locally available file. ImageNet weights alone would not substitute for the trained model. |
| Frame decode/preprocessing/export | PARTIAL | OpenCV sampler and dataset normalization exist, but no local executable decoder stack or raw clips. |
| Visual TRAIN exports | READY | Verified original bytes, finite `(3339,10,512)` export, `resnet18_v1` declaration and schema. |
| Raw AVE clips, decoded WAV/frame caches | BLOCKED | Repository-wide ignored-file search found no `.mp4` or `.wav`; configured raw/cache roots absent. |
| Frozen E3/A2 feature inference | READY | Real selected checkpoints safely loaded and executed, with finite repeated predictions and canonical-forward checks. |
| Phase 3 JSD reference | READY | Existing validated TRAIN probability exports, matching 29-class vocabulary; unchanged scoring implementation. |
| Demo `src/` branches | PARTIAL | Existing annotation lookups/random or heuristic fallbacks are placeholders; not selected encoder inference and never used by this adapter. |

The only local `.pt` artifacts found are historical fusion checkpoints and their downloaded audit copies.
`pre_trained_encoders/audio_feature.h5` is a feature artifact, not the missing BEATs/LoRA weights.
The local Python environment has PyTorch `2.14.1+cpu` and NumPy `2.4.6`.
Module discovery found no `torchaudio`, `torchvision`, `soundfile`, `cv2`, `pandas` or `tqdm`;
`ffmpeg` is not on PATH. No dependencies or weights were downloaded or installed.
Unlike the earlier Phase 5 freeze report, PyTorch imports and real CPU execution succeeded now;
Windows Application Control did **not** block this session. No security controls were bypassed.

## Task 6.2: raw audio contract and audited transformations

`RawMediaInput` describes a **proposed canonical boundary**, not a decoder. Audio is mono float32
`(B,160000)`, finite and within `[-1,1]`, explicitly 16000 Hz, exactly ten seconds, with local origin zero.
No implicit channel conversion, sample-rate conversion, clipping, padding, trimming or stretching is performed.
Noncanonical waveform inputs fail before encoder invocation. Arbitrary filenames and durations are unsupported.
Even canonical inputs fail clearly in `encode_raw_media`: the required encoder stack is unavailable.

The audited historical audio path is:

1. `audio.extract_audio`: FFmpeg `-vn -ac 1 -ar 16000 -t 10`. Resampling/downmix implementation and
   precise decoder versions are not pinned. Direct stereo `features.load_wav` instead averages channels;
   these policies must not be assumed bitwise equivalent. The loader asserts 16 kHz; it does not resample.
2. `features.load_wav`: float32 SoundFile read, then silently pads short audio with zeros or trims long
   audio to 160000 samples. This legacy behavior is **not** allowed at the strict raw boundary.
3. BEATs filterbank operates on the full ten-second waveform in float32 (no float16 filterbank): scale by
   `2**15`, Kaldi fbank with 128 mel bins, 25 ms windows, 10 ms hop, sample frequency 16000;
   normalize by `(fbank - 15.41663) / (2 * 6.55582)`. It is not the CNN's 64-bin log-Mel path.
4. `BeatsLoRA.tokens` runs the BEATs transformer with q/v LoRA. Documented selected run uses rank 8,
   alpha 16 across 12 layers; LoRA dropout is 0.1 in code. Missing saved configuration prevents independent
   checkpoint-level verification. Frozen backbone stays eval; inference must also disable LoRA/head dropout.
5. `pool_to_seconds`: average eight frequency patches, then time patches whose centres
   `(patch_index*16+8)*0.01` fall in each one-second bin (floor, capped at nine). Output is `(B,10,768)`.
6. Subtract frozen TRAIN 768D means and divide by stored stds. The extractor computes these statistics
   from float16 cached TRAIN embeddings cast to float32, storing `std+1e-5`.
7. Head `Dropout(0.1) -> Linear(768,256) -> GELU` produces `(B,10,256)`.
   The separate dropout/Linear(256,29) classifier supplies probabilities; its logits are not fusion features.
8. Historical export calls `evaluate.load_model -> build_model -> BeatsLoRA.embed` through `predict`,
   then writes float16 embeddings and class probabilities. Thus an executable producer path exists in source.
   No local replay establishes that the handed-off bytes were actually produced by this exact code revision.

BEATs contextualizes the whole clip. An output bin is not an isolated one-second acoustic inference.
Missing raw segments can contaminate other bins through attention; masking after encoding cannot repair this.
The raw contract rejects any declared invalid segment. No annotation/event label determines validity.

## Raw video contract and audited transformations

Canonical frames are RGB uint8 `(B,10,224,224,3)` with explicit sample timestamps `(0,1,...,9)` seconds,
video duration ten seconds, and local origin zero shared with audio. Both duration fields must independently
equal ten; mismatches, offsets, wrong cadence, missing frames and decode failures require rejection upstream.
The container/demux decoder must establish a shared media-clock origin rather than independently zeroing streams.

Historical `vision.extract_frames` reads nominal OpenCV FPS (falling back to 30 if false), requests frame
`min(int(t*fps), max(0,total_frames-1))`, converts BGR to RGB, and resizes directly to 224x224 with
`INTER_LINEAR`. It does not preserve aspect ratio or use the ImageNet resize/centre-crop recipe.
`AVEVisual` converts to channel-first float32, divides by 255, and applies means
`(0.485,0.456,0.406)` and stds `(0.229,0.224,0.225)`. Export uses `augment=False`.
Training augmentation (horizontal flips, brightness and frame cutout) is intentionally disabled for inference.

The full trained ResNet-18 checkpoint includes its classification head. `embed` uses encoder/global average
pooling plus the Identity projection at 512D; the 29-class head is used only to derive softmax probabilities.
`export_embeddings` calls the same `VisionModel`/`evaluate.predict` path and writes float16 embeddings.
Again, a shared source path is present; checkpoint configuration and raw/export parity have not been replayed.

The historical sampler substitutes the last decoded frame on individual failure and black frames on an
unopenable video; it clamps late frame indices. `AVEVisual` can supply black frames for missing cache files.
Those behaviors do not record validity and cannot establish ten genuine decoded samples. The new raw
boundary forbids them. For constant frame rate, a future audited decoder can explicitly reproduce the
historical index rule and record achieved timestamps. Variable frame rate must be rejected until a PTS-based
selection rule and parity study exist; nominal-FPS indexing is not a verified VFR policy. The canonical tensor
validator cannot independently prove RGB channel order, true duration or the supplied timing claims.

## Temporal alignment, validity and feature schema

Time is clip-local seconds; bin `t` covers `[t,t+1)`, `t=0..9`. Original indices are preserved after masking.
Video frame `t` represents that bin; audio pooling uses the time-centre convention above.
Raw inputs require explicit zero origins and exact durations; no rounding tolerance is silently applied.
Features require canonical audio/video bin starts and an explicit duration of ten seconds. No temporal
compaction, frame stretching or modality realignment occurs. Substituted pair sources can differ, while their
local ten-bin coordinate systems must match. These bins do not prove common recording provenance or synchrony.

| Feature boundary | Contract |
|---|---|
| Audio | Nonempty finite floating tensor `(B,10,256)`; cast explicitly to model float32 on CPU; overflow rejected. |
| Visual | Same batch size, finite floating `(B,10,512)`; missing modality rejected. |
| Per-modality validity | Explicit boolean `(B,10)` on each feature device; no numeric coercion. |
| Fusion validity | Boolean intersection of audio/video masks, on CPU; at least one jointly valid bin per pair. |
| Invalid bins | Features must remain finite even when masked; finite placeholders are excluded before projection/MLP, not marked valid. |
| Provenance | Ordered audio/vision records: exact run identity, verified export hashes, explicit preprocessing profile, origin, and encoder checkpoint digest (unknown = `None`). |
| Predictive inputs | Exactly audio, visual, validity. IDs, categories, annotation intervals, targets and diagnostic metadata never enter the model. |
| Diagnostics | Separate `diagnostics` field, never concatenated or used to choose masks/scores. |

No raw decode validity is available in the historical exports. The smoke's explicit all-true masks mean
exported-bin availability only. They do not assert successful raw decoding. Newly appearing unbound mask or
timestamp sidecars are rejected by the local reader. Caller-supplied masks must come from observed input
availability/decoder evidence, never AVE event labels or annotation intervals. Partially masked feature batches
are mechanically supported and tested, but frozen thresholds were selected with historical all-bin inputs;
their operating characteristics on partial masks have not been established.

`EncoderProvenance` equality checks exact declared source/preprocessing compatibility against checkpoint
metadata. `load_verified_train_exports` verifies actual bytes, uses the existing `load_bundle` and
`validate_bundle` contracts, and loads TRAIN only. External tensor provenance remains a trusted-producer
assertion: a copied provenance record does not authenticate arbitrary tensor contents. No raw provenance is
accepted. The preprocessing profile explicitly says `raw-export-parity-unverified`; it is an audited code
declaration, not a retrospectively invented training-time signature.

## Task 6.3: frozen fusion inference

New files: `fusion/inference_contracts.py`, `fusion/inference.py`, `fusion/smoke_inference.py`,
`tests/test_fusion_inference.py`, and this document. Existing source files are unchanged.

```python
from fusion.inference import FrozenFusionInference

adapter = FrozenFusionInference(
    "_local/fusion/phase4_improvement/study_20261024_seed42_01/E3",
    candidate="E3", device="cpu",
)
result = adapter.predict(feature_batch)  # explicit validated FeatureBatch
payload = result.to_dict()
```

CPU is the only supported device for this foundation. Input device placement is handled explicitly;
unsupported model devices fail instead of falling back. The adapter checks pinned checkpoint, metadata and
operating-point SHA-256 values, then reuses `train_mlp.load_selected_checkpoint` (`weights_only=True`,
CPU map location), `model_interface.validate_live_model` and model-state fingerprints. Architecture,
config, epoch, fixed buffers, source exports and threshold/model association must agree. Every call restores
`eval()`, runs under `torch.no_grad()`, and rejects modified weights/settings, float32 overflow, nonfinite
features or output, and wrong logit shape. Parameters are frozen. No tracker or training function is called.

| Candidate | Architecture / role | Frozen threshold |
|---|---|---|
| E3 | MLP 768 -> 512 -> 256 -> 128 -> 1, ReLU, dropout 0.2, masked mean; primary learned candidate | `0.009579099009671126` |
| A2 | Video-to-audio attention, d_model 256, 4 heads, FFN 512, GELU, sinusoidal positions, masked mean; selected attention reference | `0.0026640388978268056` |

Both return one correspondence logit per pair. Scores use the exact Phase 4/5 float64 convention
`exp(-logaddexp(0, -logit))`, with decision `score >= threshold`. Target 1 means predicted **original pairing**;
target 0 means predicted **substituted pairing**. These uncalibrated sigmoid scores are not established
probabilities of truth or physical sound-source causation. Results include logits, scores, target/decision,
valid bins, exact checkpoint/state identities, provenance, preprocessing declaration and separate diagnostics.
No silent candidate substitution or threshold transfer is supported.

Optional Phase 3 JSD is computed only from corresponding, byte-verified TRAIN `(N,10,29)` probability exports
with identical vocabulary. The unchanged `score_probabilities` implements mean over all ten bins of
`1 - JSD_base2`; no learned threshold is applied to JSD. It is not computed from embeddings and no new
masked JSD aggregation is introduced. The new adapter does not change Phase 3–5 schemas or loaders.

### Pinned artifact hashes

| Artifact | SHA-256 |
|---|---|
| E3 best.pt | `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925` |
| E3 metadata | `fd6abe3f0e62f09e19c096f82b6516195a2c8dba93cd3d534e7f88fcfec48e58` |
| E3 operating point | `706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad` |
| A2 best.pt | `0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12` |
| A2 metadata | `aadd9c73e7675249f8192912656ca624aa6d77d50b10cea3f29a1968cce8d460` |
| A2 operating point | `16994e04c6a8421ec8bc22c1fac33691a4a0228ba544f6e57b0730f98c95b384` |

Selected state fingerprints and all required TRAIN producer hashes are checked by the adapter. They identify
the historical fusion artifacts and producer export bytes; absent encoder weight hashes remain unknown.

## Verification and TRAIN-only smoke

Commands (repository root):

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_fusion_inference.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m fusion.smoke_inference --output _local/phase6_verified_smoke.json
```

The smoke output must be a fresh path under ignored `_local`; the CLI refuses overwriting evidence.
It hashes the original TRAIN manifest and TRAIN export files against selected-checkpoint metadata,
checks selected source row references, and chooses two original and two easy substituted TRAIN pairs.
Only TRAIN arrays and TRAIN score references are loaded. Other-split artifacts are only opaque metadata
hash declarations or byte-preservation context, never evaluated or used for implementation choices.

| TRAIN pair order | E3 sigmoid score | A2 sigmoid score | Both decisions |
|---|---:|---:|---|
| Original 1 | 0.9999999970365717 | 0.9999909246503587 | original |
| Original 2 | 0.9999999868673054 | 0.999987800613114 | original |
| Easy substitution 1 | 0.00005472729851657072 | 0.00000841434596788025 | substituted |
| Easy substitution 2 | 0.005335997705970872 | 0.000016394159954158703 | substituted |

Repeated inference was exact in this CPU environment. Direct canonical Phase 4/5 forwards matched exactly.
Archived TRAIN logits differ by at most `1.9073486328125e-6` for E3 and `9.5367431640625e-7` for A2,
within the fixed `rtol=atol=1e-5` integration tolerance. Historical batch/thread numerical rounding is not
claimed bitwise reproducible across configurations. JSD scores were
`[0.9975551352260279, 0.9780001594588906, 0.003539471096493485, 0.004088182541276108]`.
These four training examples are an integration smoke, not a performance estimate.

38 new synthetic tests cover shapes, batch sizes, explicit/intersected/all-invalid masks, masked-bin
invariance, nonfinite/overflow inputs, missing modalities, temporal mismatch, source/preprocessing mismatch,
missing/corrupt/wrong-architecture checkpoints, safe loading, threshold/model mismatch, unsupported devices,
eval/no-grad/frozen parameters, determinism/RNG preservation, pairing semantics, metadata exclusion, exact
legacy sigmoid/forward conventions, structured output and strict unsupported raw-media behavior.
The final required full-suite command completed in 80.701 seconds: **516 tests,
514 passed, two unavailable-CUDA skips, zero failures/errors**, compared with the
478-test historical baseline. All 38 new focused tests passed in 1.924 seconds.
The earlier 514-test run also passed; the 516-test run includes the final two rejection cases.
Because PowerShell stderr redirection returned an inconsistent wrapper status, a final subprocess-captured
repeat explicitly verified **Python exit code 0**, again 516 tests with two skips (80.705 seconds).
That repeat ran the same required unittest arguments without changing code or security settings.

Local evidence: `_local/phase6_verified_smoke.json`, `_local/phase6_targeted_tests.log`,
`_local/phase6_final_full_tests.log`, `_local/phase6_preservation_before.json`, `_local/phase6_upstream.json`.
The exit-code-verified repeat is in `_local/phase6_exit_verified_full_tests.log`.
`_local/phase6_scope_verification.json` records SHA-256 and size for each new source/test/document file,
Python syntax parsing, credential-pattern scanning and trailing-whitespace checks. `git diff --check`
passed; no existing tracked source is modified and the index is empty. Exactly the five intended new
files and the pre-existing untracked requirements file appear in Git status; no datasets or weights
were staged. New local audit logs/scripts/results remain ignored under `_local`.
The preservation inventory covers 558 existing source/artifact/manifest/evidence files, including Phase 5.
All remain byte-identical. W&B cache directories are excluded: an existing cache file denied read access;
this is not a claim that those caches were hashed. Canonical experiment reports/checkpoints/source archives
outside those caches are covered. The requirements file remains untracked. Its actual on-disk
byte-level SHA-256 (141 bytes, CRLF) is
`0b6230b82b66ec1d2537681d7c5cdc7e0a91e660bef01ba154fe8a68bdd23e08`.
The earlier `a01116f4399965d31d989d92ec8d14a07cd518b55208d42a9081efd91f2848ab`
identifies LF-normalized contents (137 bytes), not the current file's byte-level integrity.

## Confirmed defects, scientific limits and remaining integration steps

Confirmed historical behavior needing resolution before raw integration: silent audio pad/trim;
video last-frame/black-frame substitutions and FPS fallback without validity; absent timestamp/decode
sidecars; nonportable producer paths; missing encoder/normalization hashes and decoder versions.
Existing BEATs loading explicitly uses `weights_only=False`; legacy encoder loaders lack the pinned safe
artifact verification used by fusion. They were audited in source and left untouched. `norm_stats` also
depends on cached TRAIN array layout. These are integration limitations, not new claims about observed
failures on specific clips. Existing `src/` placeholder logic is not evidence of model inference.

The scientific limits from earlier phases remain: single seed, repeated development-validation use for
selection, ten source-video IDs shared across official TRAIN/validation annotations, coarse ten-bin
resolution, uncalibrated scores and pairing-provenance targets rather than causal sound-source identity.
Feature/export availability does not establish physical synchrony. No claims about held-out or challenge
performance are made here; neither was run. Partial-mask threshold behavior also remains unverified.

To reach executable raw inference, obtain the exact self-supervised iter3 base, saved LoRA/head/config,
frozen normalization statistics/layout evidence and trained `resnet18_v1` checkpoint with authenticated
hashes; supply a few TRAIN-only original raw clips; install compatible audited decoding/encoder dependencies;
implement safe explicit producer reconstruction without downloads or random-weight substitution;
record decode durations, stream offsets, timestamps, sampling/resampling/downmix versions and true input
validity; reconcile strict rejection with the historical producer behavior; then replay both real encoders
on TRAIN and compare export-roundtrip schema/float16 features and preprocessing signatures before E3 scoring.
Define and verify VFR or arbitrary-duration policies separately before enabling those inputs.

Tasks 6.1–6.3 stop here. **READY FOR PHASE 6 FOUNDATION AUDIT** means this bounded feature foundation is
reviewable. Raw-media deployment remains blocked and unverified; Phase 5 PR #17 remains an upstream dependency.

## Focused CPU autocast remediation (2026-10-25)

The independent foundation audit found one medium-severity precision defect: `torch.no_grad()` does not
disable an enclosing CPU autocast context. Before editing, the actual frozen E3/A2 checkpoints and the
same four byte-verified TRAIN pairs independently reproduced both reported failures:

| Enclosing context | E3 before correction | A2 before correction | Both after correction |
|---|---|---|---|
| CPU bfloat16 autocast | `TypeError: Got unsupported ScalarType BFloat16` at NumPy conversion | Same exception | Succeeds; exact float32-reference logits/scores/decisions |
| CPU float16 autocast | Maximum logit drift `0.008089065551757812` | Maximum logit drift `0.0029439926147460938` | Maximum logit drift **0** |
| No autocast | Validated float32 reference | Validated float32 reference | Bitwise identical to the pre-correction reference |

`FrozenFusionInference.predict` now derives the actual model device, rejects unsupported non-CPU model
devices, and runs validation, model-state checks, input conversion, forward execution and score conversion
inside `torch.no_grad()` plus device-specific `torch.autocast(..., enabled=False)`. Inputs and weights remain
float32; logits must explicitly be float32 `(B,1)` before NumPy conversion. An unexpected lower/higher precision
output is rejected rather than silently recast. No global autocast setting is changed. Context exit restores
the caller's autocast enabled flag, dtype, cache flag and gradient state, including validation/forward errors.
The existing float64 sigmoid calculation, exact model-specific thresholds and inclusive `score >= threshold`
rule are unchanged. No weight/configuration/encoder/source-export change or numerical rounding policy was added.

Six new CPU-friendly unit tests cover both synthetic frozen-model architectures under bfloat16 and float16
autocast, ordinary float32 reference equivalence, singleton and mixed-validity batches, repeated exact results,
float32 forward inputs/outputs, eval/no-grad execution, frozen parameter fingerprints, unchanged RNG state,
caller-state restoration on success and errors, nested disabled autocast, exact threshold equality and strict
output dtype/device handling. They were run against the old adapter first: the focused 44-test suite failed
with seven failures and five errors, exposing the original precision problem and missing precision guards.
After correction all **44 focused tests passed** in 1.384 seconds.

Independent real-checkpoint verification also passed for E3 and A2 with full four-pair batches, singletons and
mixed validity, under both enclosing autocast dtypes. Every context matched its own ordinary float32 result
exactly, including returned scores and decisions, and repeat calls were identical. Hooks observed float32
inputs/logits, eval mode, no gradients and disabled CPU autocast during every forward. Caller autocast stayed
active within its original scope after each call and its prior state was restored outside. Both state
fingerprints were unchanged. The ordinary four-pair logits and float64 scores were bitwise identical to the
captured pre-correction values; the earlier archived-TRAIN tolerance comparisons still passed. Partial-mask
numerical equivalence is an implementation check, not new evidence about threshold performance on partial inputs.

The full required unittest suite passed with **522 tests: 520 passed, two unavailable-CUDA skips, zero
failures/errors; Python exit code 0**, in 45.626 seconds. This is six tests above the 516-test baseline.
No further Phase 3–5 changes were made. All 558 protected files remain byte-identical, including historical
checkpoints, exports, experiment evidence and the untracked requirements file. The pre-existing W&B-cache
exclusion remains unchanged; this is not a new claim of cache hash coverage.

Only `fusion/inference.py`, `tests/test_fusion_inference.py` and this document changed in this remediation.
`fusion/inference_contracts.py` and `fusion/smoke_inference.py` retain their earlier bytes. All pre-existing
Phase 6 work is preserved; nothing was staged, committed or pushed. Earlier Phase 6 evidence remains historical
and untouched. New ignored local evidence is `_local/phase6_autocast_remediation.json`, with captured before/after
results, `_local/phase6_autocast_tests_before.log`, `_local/phase6_autocast_targeted_tests.log`,
`_local/phase6_autocast_full_tests.log`, and `_local/phase6_autocast_changes.diff`. The before-source snapshot and
current verification manifest are separate `phase6_autocast_*` files; the original scope manifest was not rewritten.

**FIXED — READY FOR TARGETED RE-AUDIT.** This correction addresses only caller-autocast leakage. Raw-media
encoder execution/parity, trusted-producer provenance, actual synchronization, partial-mask threshold behavior,
calibration and causal interpretation retain the limitations documented above. No raw encoder integration,
training, threshold tuning, challenge/test evaluation or W&B run occurred.
