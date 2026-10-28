# Phase 8 Stage 2E — backend freeze and PR readiness

Audit date: 2026-10-28. **READY FOR PR**, limited to review of the gated internal
backend. Raw prediction release/final validation remains **FIX REQUIRED**.
Independent repository review is pending; this implementation audit does not
constitute independent approval of numerical parity or release readiness.

## A. Integration audit verdict

Repository `NUS-AIS-GCs/Audio-Video-Confidence`, working branch
`feature/phase8-backend-stage1`, HEAD/base/main:
`eff4e2fe6ed33650c07fb8f719fae9a665a7faf1`. The remote main ref was checked.
Tracked worktree and index are unchanged. There were 30 untracked text files at
preflight, including the unrelated historical training-requirements file. No
commit, push, merge, deployment, retraining or held-out TEST scoring occurred.

Review covered all Stage 1–2D modules, imports, pinned runtime dependencies,
tests, environment examples and dated reports. Stage 2E resolves three concrete
integration weaknesses without changing inference math or opening a public path:

- `train_manifest.py` centralizes artifact/identity authentication. The internal
  backend no longer imports the profiling command to obtain manifest helpers.
  `stage2c_probe.py` re-exports the original helper names for existing callers.
  Duplicate JSON keys, malformed schema, escaping paths and missing/wrong hashes
  fail authentication; the 40 artifact roles and five input roles are unchanged.
- The volume seeder is import-safe and uses explicit pre/post-copy checks instead
  of optimization-sensitive assertions. It refuses overlapping roots and existing
  destinations, verifies every copy, and sets copied files read-only. The Stage 2D
  coordinator mounts the shared modules and invokes the seed module accordingly.
- The internal response validator rejects malformed nested objects, unsafe error
  issues and score/logit inconsistencies. A worker records its thread/MKLDNN
  profile at construction and rejects later drift before artifact access.

No source under `audio`, `vision`, `fusion`, existing deployment/runtime/UI,
Phase 1–8 research code or tracked requirements changed. The deployment pins
already supply the new modules' imports. No new runtime package or network model
download is needed; the existing image executed the updated modules successfully.

Earlier diagnostic tools overlap in descriptive metric computation, but are
retained as dated reproducibility interfaces; they neither gate serving nor
justify a tolerance. The bundle adapters handle authentic producer schemas;
`encoders.py` retains the stricter Phase 6 adapter schema. They are intentionally
distinct. The three environment examples represent different dated layouts and
are diagnostic templates, not interchangeable activation configurations. Earlier
reports describe their stage's findings; this document supplies current status.
No experimental code was silently promoted or historical interface deleted.

## B. Files changed and PR hygiene

The proposed patch is **36 new text files**, listed exactly in
[the changed-file inventory](phase8_backend_changed_files.md): 17 deployment
modules/examples, six test files and thirteen documents. Relative to Stage 2E
preflight, four previously untracked source files were deliberately revised;
their originals are retained in the ignored audit snapshot. The remaining
proposed files are preserved Stage 1–2D additions or new Stage 2E files.

The unrelated pre-existing `requirements-fusion-training.txt` remains unchanged,
untracked and excluded from the PR. Index/staged files remain empty. Private
checkpoints, original/derived exports, source media, local environments, audit
logs and generated arrays remain ignored under local storage. The inventory
audit checks all candidate bytes for text/size, personal absolute paths and
credential indicators. Artifact SHA-256 values and portable relative/container
paths in examples are intentional provenance, not private payloads or secrets.
Existing research binaries already in main are outside this patch.

## C. Backend contract compatibility

| Boundary | Stage 2E result |
| --- | --- |
| Phase 6 `EncoderSpec`/`EncoderOutput` | Geometry, finite float32, artifact/model fingerprints and strict load contracts preserved |
| Phase 6 `TrainParityFusion` | Exact float16 comparison unchanged; still unsuccessful |
| Phase 8 public admission | Authenticated valid media returns `MODEL_NOT_READY`, null predictions, `RAW_INFERENCE_GATE_CLOSED` |
| Proposed public prediction contract | Still unimplemented/inactive; no public `PREDICTED` path added |
| Internal Stage 2D ABI | `aval.internal-train-inference.v1`; five SHA-bound TRAIN MP4s only; experimental success cannot change readiness |
| Gradio | Does not import/call `InternalTrainBackend`; private callback, queue eight, concurrency one preserved |
| E3/A2 | Original checkpoint/model/metadata/operating-point bindings unchanged; no fallback models |

E3 threshold is **0.009579099009671126**; A2 threshold is
**0.0026640388978268056**. Scores remain uncalibrated original-pairing provenance,
not causal sound-source attribution or validated event/temporal predictions.
The internal serializer removes audit tensors. Actual achieved video PTS remain
separate from intended bin starts; `phase6_shared_clock_accepted` remains false.
The numerical candidate is separately named and inactive, with runtime budgets
unset. Approximate compatibility is never reported as exact historical parity.

## D. Docker and security status

Fresh Stage 2E execution used the existing immutable image
`sha256:0ee95003d0f32ab19176ffe86e8cdc015644a0d48c03571e81c94b07858dbaab`,
with current source mounted read-only. No new image was built. Docker is installed
in the per-user Docker Desktop program directory, absent from the default PATH;
the restricted Codex sandbox cannot access it without approved host execution.
Direct host execution succeeded, so Docker is not currently unavailable.

The authenticated native volume was seeded successfully under **`python -O`**;
inference used the image's non-root user, a read-only root/model/source mount,
network none, no published ports, all capabilities dropped, no-new-privileges,
two CPUs, 4 GiB RAM, 128 PIDs and 512 MiB temporary storage. Root was used only
for seeding the new temporary volume. Cleanup removed both Stage 2E volumes.

An actual Gradio server was exercised on container loopback only:

| Check | Observed result |
| --- | --- |
| Anonymous `/config` | HTTP 401 |
| Valid ephemeral login | HTTP 200 |
| Authenticated `/config` | HTTP 200 |
| Authenticated model checkpoint file request | HTTP 403 |
| Genuine valid media through the public callback | `MODEL_NOT_READY`, predictions null |
| Callback input cleanup | Uploaded temporary file removed |
| Both encoders and E3/A2 on permitted TRAIN MP4s | Five completed genuine requests; exact prior Linux feature tensors and all Stage 2D prediction fields preserved |

The first attempt failed at container mount setup because a nested file mount
could not create a target inside the read-only input bind. It ran no inference.
The corrected harness mounts the digest at a separate application path; both
attempts and cleanup results are preserved. No credentials were recorded.

Source/test review confirms 50 MiB admission and Gradio limits, supported codec/
container/stream/duration checks, bounded FFprobe/FFmpeg execution/output,
temporary-input cleanup, sanitized errors and fail-closed readiness. The internal
backend rejects arbitrary media before decode using the TRAIN source SHA and
serializes requests with `RESOURCE_BUSY` on overlap. See the
[internal contract](phase8_stage2d_internal_contract.md) for exact decoder bounds.

The earlier streaming image audit verified all 12 layers and 39,897 regular
entries against known private payload hashes and project artifact/media paths;
none matched. It applies to this identical image ID. `.dockerignore` excludes
local artifacts and checkpoint/media extensions. A future fresh image needs a
fresh layer audit; source exclusions alone do not prove its contents.

Remaining unverified checks: browser transport upload/queue/live-route abuse,
oversized multipart and decompression stress, multi-user/session isolation,
real camera/microphone behavior, deployed TLS/reverse-proxy limits, rate limiting,
dependency vulnerability review, and a fresh final image build. The live branch
also has pre-allocation and timestamp/preprocessing findings described in the
frontend checklist. Unit tests and the loopback checks do not establish these
unperformed security properties. No broad production security certification is
claimed.

Stage 2D's measured native-volume warm mean **3.6531695 s** and cgroup peak
**1.774 GiB** are preserved historical performance evidence, not remeasured
Stage 2E latency or a production SLA. The fresh audit verifies behavior and
features after source fixes; it does not replace the controlled benchmark.

## E. Regression results

| Run | Passed | Skipped | Other evidence |
| --- | ---: | ---: | --- |
| Stage 2D full benchmark | 841 | 2 CUDA | 1,346 passing subtests |
| Stage 2E focused encoders/deployment/fusion/freeze | 239 | 0 | Nine passing subtests; 25.62 s |
| Stage 2E full `tests` suite | 863 | 2 CUDA | 1,346 passing subtests; 126.92 s |
| Isolated frontend deployment/live overlay | 72 | 0 | Node timing test file also passed |

All child exits are zero; no failures/errors. One existing Torch weight-norm
deprecation warning remains. The 22 added freeze tests account for the increase
over Stage 2D. The two unavailable-CUDA tests did not pass and were not executed.
Full regression uses the existing authenticated local fixture/development
environment. Tests needing private artifacts explicitly skip without them; the
same pass count is not promised for a clean checkout or deployment-only image.

Reproduction in an already provisioned Python 3.11 development environment:

```powershell
python -B -m pytest tests/test_phase8_freeze_audit.py tests/test_phase8_internal_backend.py tests/test_phase8_stage2c.py tests/test_phase8_video_bundle.py tests/test_phase8_audio_bundle.py tests/test_phase8_encoder_stage1.py tests/test_phase8_deployment.py tests/test_phase8_backend_contract.py tests/test_fusion_end_to_end.py -q
python -B -m pytest tests -q
```

Do not install training extras into the runtime image just to run the research
suite. Existing research development requirements and the historical optional
W&B/HDF5 environment are separate from deployment dependencies.

## F. Historical preservation

The 1,275-file preflight baseline has **zero unexpected changes**. Exactly four
reviewed local source revisions are allowed: `internal_backend.py`,
`stage2c_probe.py`, `seed_artifact_volume.py` and `run_stage2d_docker.py` under
`deployment`. Their preflight originals are snapshotted. All baseline research
sources, authenticated datasets/exports, checkpoints, frozen models, earlier
reports and Stage 2D measurement evidence remain byte-identical. Separately,
**28 protected archived report files** match their saved anchors. No authentic
dataset or model was regenerated, converted in place or embedded in an image.

## G. Frontend integration findings

The pinned live branch has no actual changed-path conflict with this backend
addition set. Its 13-file change was tested in an ignored overlay, without a
checkout/merge. The upload-results branch was not advertised and could not be
audited. Integration is pending; nearest-frame timing and live preprocessing
cannot be treated as authenticated TRAIN equivalence. See the
[owner integration checklist](phase8_frontend_integration_checklist.md), including
allocation bounds, authenticated upload/results, device capture and failure states.

## H. Evaluation protocol readiness

The [frozen evaluation protocol](phase8_final_evaluation_protocol.md) specifies
the future official TEST cohort, original/swapped definitions, leakage caveats,
E3 primary/A2 comparison, immutable hashes/thresholds, metrics, undefined cases
and no evaluation-driven tuning. It distinguishes the historical-feature
benchmark from any later raw end-to-end comparison. It is ready for independent
review, not execution. No held-out TEST scoring was performed; a TEST-capable
reader/reporting boundary and explicit later authorization remain required.

## I. PR scope and description

[PR draft and commit grouping](phase8_backend_pr.md) cover only the 36 allowlisted
text additions. No staging or GitHub write was performed. Review evidence is
available locally through approved access; private fixtures/logs are intentionally
excluded from source control. An independent reviewer must review and reproduce
applicable evidence before merge. This readiness verdict concerns a gated internal
implementation PR and does not waive any prediction release gate.

## J. Remaining blockers

1. Producer float32 intermediates, exact decode/pixel/waveform/runtime evidence
   and reconciliation of historical numeric differences remain pending.
2. Exact historical parity still fails. The numerical policy is inactive and
   independently justified runtime budgets/timing policy are absent.
3. Five TRAIN clips do not cover the required 6,678 primary TRAIN pair identities.
   A general bounded decoder satisfying strict shared-clock provenance is absent.
4. Frontend integration, its security findings, unavailable upload-results branch
   review and real-device/authenticated browser testing remain incomplete.
5. Independent PR/protocol review, final image/security verification and separate
   release/final-evaluation approval are still required.

Current Linux raw mismatch counts remain **1,591/12,800 audio** and
**59/25,600 video** float16 values. Exact reproduction of the previous Linux run
does not reconcile those historical export differences. Stage 2D raw-versus-export
E3/A2 decisions were stable for 5/5 clips; max score differences were respectively
`8.5946856298591e-9` and `2.652042774276e-7`. The new replay preserves feature
inputs and frozen bindings. These are positive TRAIN smoke tests only.

## K. Decision and evidence index

**READY FOR PR — gated internal backend review only.** Final prediction validation
and release remain **FIX REQUIRED**. Production responses remain
`MODEL_NOT_READY`; no tolerance, fallback or fabricated prediction is enabled.
Stop after Stage 2E.

Ignored local audit evidence under `_local/phase8_stage2e/`:

- `preservation_before.json`, `preservation_after.json`, `source_before/`, and
  `historical_report_verification.json`.
- `focused.log/xml`, `regression.log/xml`, their exit records, and the 85-test
  audit subset log/XML.
- `frontend-test-execution.json`, Python/Node logs and the isolated overlay.
- `runtime/` failed mount attempt; `runtime-retry/` successful seed/authentication/
  five-case replay, execution and volume cleanup records; reproduction harnesses.
- `pr_inventory.json` with exact candidate paths, byte counts and SHA-256 values;
  source hygiene and final Git-state records; `docker-final-state.json` verifies
  non-root image user and no remaining task volumes or containers using the image.
- `backend-review.patch` contains only the 36 candidate additions, with portable
  paths. Its reverse `git apply --check` passed against the reviewed worktree;
  no patch was applied. `final-qa.json` records the patch hash, Python syntax/link
  checks, exact five-case E3/A2 replay and the two unexecuted CUDA test names.

Stage 2D's unchanged `profile-final`, fusion request comparison, image-layer audit
and preservation evidence remain under their original ignored audit directory.
