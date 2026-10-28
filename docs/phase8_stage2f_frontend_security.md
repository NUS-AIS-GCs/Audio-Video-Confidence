# Phase 8 Stage 2F — frontend security and upload completion

Audit date: 2026-10-28 (Singapore). **READY FOR PR**, limited to gated frontend/security changes.
Public prediction release and final end-to-end acceptance remain blocked.

## Preflight

Repository: `NUS-AIS-GCs/Audio-Video-Confidence`. Local main, `origin/main` and the live remote ref
were verified at `f309600c422877899eac3fc5d1cf445fe16acd6c`. The tracked worktree and index were
clean. Work is on `feature/phase8-final-frontend-fixes`; HEAD remains the same base because no
commit was created. Existing untracked `docs/phase8_final_audio_trace_verification.md` and
`requirements-fusion-training.txt` remain unchanged and excluded.

The merged Stage 2E internal backend and PR #24 UI were reviewed. Existing tabs, branding,
capture timing, admission validation and diagnostic boundaries were extended rather than replaced.

## A. Files changed

Fourteen text files comprise this change:

```text
deployment/gradio_app.py
deployment/live/security.py                         new
deployment/live/service.py
deployment/ui/live_section.py
deployment/ui/upload_section.py
deployment/web/live_capture.js
deployment/web/live_timing.js
deployment/web/upload_preview.js                    new
docs/phase8_frontend_live.md
docs/phase8_stage2f_frontend_security.md             new
tests/js/test_live_security.js                      new
tests/test_phase8_frontend_security.py              new
tests/test_phase8_live.py
tests/test_phase8_upload_ui.py
```

No source under `audio/`, `vision/` or `fusion/` changed. `deployment/runtime.py`,
`deployment/internal_backend.py`, Dockerfile and dependency requirements are unchanged.

## B. Live-session ownership

The pinned Gradio 6.30 component route injects a Starlette Request. The bridge resolves the login
cookie against Gradio's server-owned token registry, following the same resolution as its `/user`
route. It does not read client usernames, identity headers or `session_hash` as ownership evidence.
The retained owner is a SHA-256 digest binding the authenticated account and login token; raw login
cookies are not retained in capture objects, returned or logged.

Every capture also receives an independent 256-bit server-minted `tab_key`, in addition to its
128-bit session token. Push, check, stop and restart require the matching authenticated login and
both capabilities. A new tab creates a separate capture without evicting another tab. Restart
requires the old capture's capabilities and rotates them. Unknown sessions, wrong tab keys and
cross-user/login misuse receive the same `LIVE_SESSION_NOT_FOUND` response without diagnostics.
Capabilities remain bearer secrets within their authenticated login; client claims do not mint them.

Missing guard context, missing authentication, unsupported auth modes or a Gradio version other
than 6.30.0 disable live endpoints. HTTP authentication failures return 401 with null predictions;
the UI dispatch fallback returns `MODEL_NOT_READY`, null predictions and a fixed error code.
Future Gradio/auth changes require bridge review and the HTTP regression tests.

## C. Resource bounds and lifecycle

`live_app_kwargs(service)` installs the transport guard and cleanup lifespan for launch or mount.
Omitting the guard disables live dispatch. Both production launch wiring and mounted HTTP tests
use this helper.

| Boundary | Limit / behavior |
| --- | --- |
| Authentication | Before any body reading or body-reader slot allocation |
| Live component body | At most 3 MiB, measured while receiving, including chunked requests |
| Body receive deadline | Five seconds total |
| In-flight live work | Four operations across body reading, JSON parsing and dispatch; excess returns 429 |
| Component encoding | JSON only; compressed and multipart component bodies rejected |
| Retained framework metadata | Session-hash claim limited to 128 ASCII characters before Gradio state allocation; component ID/function/data types bounded |
| Capture registry | Four total sessions; original supported rates and per-push/image limits retained |
| Per push | At most two seconds audio, eight audio chunks, four frames, 400 KiB compressed-image budget per frame (bounded base64 envelope) and 1920-pixel declared sides |
| Browser backlog | At most 1.5 seconds audio, eight chunks, four pending frames |
| Browser requests | Three-second logical deadline; at most three actual outstanding calls; restarts blocked until they settle |
| Capture duration in browser | Record mode stops after 60 seconds if no complete window; rolling mode capped at 30 minutes |
| Idle cleanup | Worker checks at intervals no greater than one second for 20-second idle expiry |
| Stop / expiry / push | One registry lock serializes lifecycle and processing; stopped buffers cannot be resurrected |
| Rejected push | Bounded copy-and-commit transaction preserves the original buffer, counters, diagnostics and deadline |
| Shutdown | Lifespan clears all sessions and joins the expiry worker |

Encoded lengths and image headers are still checked before decoding. Live responses use `no-store`.
The ordinary authenticated file-upload route retains its separate 50 MiB limit and original checks.

Gradio's component promises do not expose AbortSignal. A timeout stops local devices and releases
queued media while bounding actual outstanding calls; it does not pretend to cancel that transport.
Another capture remains disabled until those calls settle or the tab reloads. Late permission grants
release devices immediately; late successful server starts receive a capability-bound stop request.
If stop cannot reach the server, idle expiry handles retained media. UI cleanup wording now says so.

## D. Upload preview and results

The authenticated Upload tab retains the original File uploader and private, concurrency-one
admission callback. A browser-local Blob URL provides video preview without another upload,
server preview endpoint, decoder invocation or exposed private file path. Replacement, removal,
pagehide and component teardown revoke old preview URLs. Unsupported browser codecs get a fixed
preview-only message; server validation remains authoritative.

Instructions state the duration, codecs, size limit and disabled predictions. Verify yields a loading
card and Gradio progress. File changes cancel in-flight verification and clear result/technical
output. Verify remains disabled after a check because the secure callback removes its managed
upload; selection/change re-enables it.

Known errors use fixed sanitized text. Unexpected exceptions and unknown technical fields are
sanitized. Technical JSON always has null predictions, and an internal TRAIN success cannot be
displayed as a public result. The E3 result area reserves decision and uncalibrated correspondence
score fields as `Unavailable`; it contains no fabricated score, event head or per-second head.

## E. Backend compatibility

Valid public admissions still return `MODEL_NOT_READY`, `predictions: null` and
`RAW_INFERENCE_GATE_CLOSED`. Invalid uploads retain `REJECTED` with sanitized codes and null
predictions. No readiness gate, decoder acceptance rule, threshold or model binding was changed.

The Gradio surface does not import or expose `InternalTrainBackend` or its
`aval.internal-train-inference.v1` experimental success response. Live checks remain capture-only:
`CONTRACT_VALID`, `inference: NOT_CONNECTED`, `encoder_compatibility: NOT_VERIFIED`, null
predictions. Geometry/bin labels and reported offsets are not encoder or physical-clock approval.

## F. Security verification

Final focused suite: **190 passed**, zero failures/errors/skips, 27.36 seconds.

The tests exercise the actual pinned Gradio routes, login cookies and raw Request injection through
FastAPI TestClient, rather than only supplying owner strings to the service. They cover anonymous
and invalid login, concurrent tabs/users, spoofed body/header identities, identical client session
hashes, separate logins for one account, cross-user/tab token misuse, missing/malformed keys,
repeated creation, authorized restart, protected-file access, oversized/chunked bodies, stalled body
reads, transport capacity recovery, invalid timestamps and transactional rollback.

Threaded tests verify stop/push serialization and absence of resurrection. Expiry is tested without
registry access; application shutdown clears sessions and terminates workers. An authenticated
multipart upload plus the actual public handler verifies null predictions and managed-file deletion
with media-probe metadata controlled as a fixture and `torch.load` forbidden.

Node: **eight timing checks and five controller/preview tests passed**. Synthetic browser tests cover
stalled pushes, backlog ceilings, blocked restarts, late permissions/session creation, URL revocation,
selection/clear/drop/unmount and preview limits. All browser scripts pass syntax checks.
These are simulated devices/DOM, not real hardware or visual browser acceptance.

## G. Full regression

Final `tests` suite: **962 passed, two CUDA-only skips, 1,346 passing subtests**, 117.51 seconds.
The existing Torch weight-norm deprecation warning remains. No skip is counted as passing.
`git diff --check` and Python AST/source-text checks passed. All tests used the existing Python
3.11.16 development environment with deployment-pinned Torch/Gradio packages; no packages were
installed and no project model was retrained or held-out TEST dataset scored.

Final evidence is under ignored `_local/phase8_stage2f/`:

- `focused-accepted.log/xml`, `regression-accepted.log/xml`.
- `node-security.log`, `node-timing.log`.
- `preservation_before.json`, `preservation_after.json`, `historical_report_verification.json`.
- `frozen_bindings.json`, `source_inventory.json`, `verification_summary.json`.

Earlier local test iterations remain distinct; the accepted files identify the final implementation.
Reproduction in the already provisioned audit environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/_local/phase8_stage1/venv/Lib/site-packages;$PWD"
$env:GRADIO_ANALYTICS_ENABLED='False'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
& ./_local/python/cpython-3.11-windows-x86_64-none/python.exe -B -m pytest tests -q -p no:cacheprovider
node tests/js/test_live_timing.js
node --test tests/js/test_live_security.js
```

Use a fresh writable pytest temporary directory if the host's existing pytest temp directory is
restricted. Do not install research extras into a deployment image to reproduce development tests.

## H. Historical preservation

Preflight recorded 1,409 files. All **1,185 protected local artifact/evidence files** remain byte-identical;
the **28 archived evaluation report files** independently match their saved hashes. Original and
diagnostic copies of E3/A2 checkpoints, metadata and operating-point files match all twelve bound
file hashes. Frozen thresholds remain E3 **0.009579099009671126** and A2 **0.0026640388978268056**.

Only the nine existing files listed in section A differ from preflight; five new text files complete
the fourteen-file change. Original untracked files are preserved. The index is empty: no credential,
checkpoint, model payload, generated evidence or unrelated file was staged.

## I. Remaining deployment blockers

1. Independent encoder parity approval, authenticated producer evidence/coverage and reviewed
   decoder/timing acceptance remain pending. The numerical/public gates stay closed.
2. Real webcam/microphone and browser acceptance, including physical latency/drift, mute,
   disconnect, background tabs and responsive visual inspection, remain required.
3. Final container build/layer inspection, dependency review, TLS/proxy/body/rate/storage quotas
   and deployed multi-user acceptance remain release checks. No deployment was performed here.
4. Public E3 integration and any final held-out evaluation require separate approval. Internal TRAIN
   output and capture validity must never substitute for that approval.

## J. Decision

**READY FOR PR — gated Stage 2F frontend/security changes only.** Public predictions remain
disabled. No commit, push, merge, deployment, retraining or held-out scoring occurred. GC1 Savora
was not accessed. Stop after Stage 2F.
