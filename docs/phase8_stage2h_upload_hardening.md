# Stage 2H: upload security and deployment hardening

Review date: 2026-10-29 (Asia/Singapore).

## A. Preflight and changed files

Base/main and GitHub main: `fca6eabb96e0a8571620499caa8b9fdf6471104a`.
Branch created from that main: `feature/phase8-stage2h-upload-hardening`.
HEAD remains at the base; nothing was committed, pushed, merged or deployed.

Preflight was not a clean working tree: the Stage 2G README wording was already
modified, and these five untracked files already existed and are preserved byte-for-byte:
`deployment/hf-space.env.example`, both final audio/video parity reports,
`docs/phase8_stage2g_deployment.md`, and `requirements-fusion-training.txt`.
The Stage 2G report is a historical audit; this report records its upload fixes.

| File | Stage 2H purpose |
| --- | --- |
| `deployment/uploads.py` | Bounded multipart transport, login-owned handles, leases, expiry, restart cleanup and disabled media routes |
| `deployment/auth.py` | Shared pinned-Gradio server-side login identity; no client ownership claims |
| `deployment/gradio_app.py` | Owned Verify entry point, upload-store lifecycle, SIGTERM/SIGINT shutdown |
| `deployment/live/security.py` | Reuse the same authentication bridge and install both transport guards/lifespans |
| `deployment/ui/upload_section.py` | Scalar handle bridge, fixed expired/auth messages and retention instructions |
| `deployment/requirements-runtime.lock` | Complete Linux/amd64 CPython 3.11 wheel versions and SHA-256 hashes |
| `requirements-deployment.txt`, `Dockerfile` | Direct upload dependencies, hash-only binary installation, pinned frontend and immutable application ownership |
| `tests/test_phase8_upload_hardening.py` | HTTP ownership, transport, disk/lifecycle, concurrency, expiry and restart regressions |
| Three existing frontend/deployment test files | Adapt to scalar handles and injected requests; retain earlier assertions |
| `README.md`, this report | Current deployment handoff; retain earlier Stage 2G work |

The existing preview/result layout, upload media validator, FFprobe limits, Stage 2F
live limits and exact prediction gates were retained. The TRAIN-only internal backend,
model adapters, frozen fusion bindings and historical reports were not changed.

## B. Single-file admission and resource bounds

`UploadTransportMiddleware` owns `/gradio_api/upload` instead of delegating storage to
Gradio's shared multipart/cache handler. It authenticates before reading any body and
accepts exactly one `files` part with a supported filename suffix. Extra file parts,
form fields, malformed/truncated/empty forms, oversized headers, unsupported encodings
and unsupported filenames fail with fixed JSON errors and null predictions.

| Bound | Enforced value |
| --- | --- |
| Actual file bytes | Existing configured ceiling, at most 50 MiB (52,428,800 bytes) |
| Entire multipart body | File ceiling plus 16 KiB; default 52,445,184 bytes |
| Receive/parse deadline | Absolute 30 seconds |
| Simultaneous body readers | Two; excess requests receive 429 without waiting |
| Retained/reserved uploads | Four globally, two per authenticated login; receiving and verifying files count |
| New admitted attempts | Sixteen per rolling minute globally; bounded history, no unbounded per-client map |
| Multipart metadata | 8-KiB total header bytes, at most eight headers, boundary at most 200 bytes |
| Verify/control envelope | 16 KiB, five-second deadline, four inflight requests; scalar inputs only |

Declared Content-Length can trigger rejection before receive; actual streamed bytes are
counted independently, including chunked requests. File bytes stream in bounded parser
slices into a private partial file. Rejection, disconnect, cancellation and timeout close
and delete the partial file and return its reservation. Deletion failure retains the
reservation and retries during expiry instead of silently freeing its disk budget.

Verify queue concurrency remains one with backlog eight. `api_open=False` explicitly
blocks direct named/indexed `/run` and `/api` calls from bypassing that queue; private
API visibility alone is not relied upon for serialization or authorization.

The content-type, client-declared size, original filename, owner headers and opaque handle
are not scientific or media-validation evidence. Verify resolves an authorized store entry,
opens the stored bytes, and calls the unchanged `validate_upload`/FFprobe admission path.
Real codec, stream count, container and duration checks remain server-side. Transport
success alone does not mean the bytes are a valid video or an inference input.

The media budget is at most four 50-MiB stored/in-transit payloads (200 MiB), plus one
bounded validation copy during the serialized Verify path and small metadata. The
ordinary Verify copy is still managed by the existing validator. No second Gradio
media cache is created. This is an application storage budget, not a claim that every
OS/proxy/socket buffer consumes zero memory or disk.

Upstream requirements remain explicit: HTTPS/proxy infrastructure must bound total
connections, request headers, incoming body size, slow clients and login/request rates.
For the default upload ceiling, configure an upload body limit no greater than
52,445,184 bytes while allowing multipart overhead. The application cannot reject bytes
before the provider proxy itself accepts/buffers them. Confirm actual HF edge behaviour;
if it cannot meet the required perimeter limits, add a reviewed proxy rather than
assuming README settings provide those protections.

## C. Abandoned upload cleanup and race handling

Files live only in `GRADIO_TEMP_DIR/aval-owned-uploads`, mode 0700, with file mode 0600
on Linux. Handles are independent random 256-bit values; entries retain only a digest
of the authenticated login identity, not its raw cookie or password.

Ready files expire five minutes after reservation. A worker checks every second without
requiring another request. Logout/token invalidation also makes ready files eligible for
deletion. Closing a browser cannot reliably notify a server, so the same five-minute
bound applies to abandoned browser sessions. The UI states that retention policy.

Verification acquires a single-use lease under the same lock as cleanup. Wrong owners,
expired/unknown handles and duplicate verification attempts receive the same generic
`UPLOAD_NOT_FOUND` response. Cleanup does not unlink a file being verified; its finally
block removes the entry/file on success, error or completion after cancellation. Active
leases continue to count against capacity until released. Tests hold verification open
across expiry and shutdown and prove the bytes remain available until its lease ends.

An exclusive cross-thread process file lock prevents two workers from sharing the store.
After an unclean exit, the next process deletes only recognized payload files in this
dedicated directory before admitting uploads. Unexpected files/symlinks cause startup
failure rather than recursive deletion of arbitrary paths. Failed unlinks remain bounded
and require filesystem repair if the operating system continues denying deletion.

Real production-CMD testing found that Gradio's background server did not handle process
SIGTERM. Main now owns its wait loop, handles SIGTERM/SIGINT and closes Gradio, live and
upload stores in finally blocks. It allows outstanding bounded readers to unwind. Allow
at least 40 seconds of container shutdown grace for worst-case receive cleanup; verify
the actual HF shutdown policy. SIGKILL cannot run cleanup: startup orphan removal and
ephemeral storage are the fallback. No promise of cleanup during an OS crash is made.

## D. Cross-user media isolation

Ownership is derived from a real Starlette request, the pinned built-in Gradio auth mode
and its server-side login-token registry. The owner digest includes both username and
the login token. Separate logins for the same account are isolated, as are different
accounts. Client `session_hash`, username/owner fields and headers do not grant ownership.
Unsupported/missing authentication context fails closed.

All built-in media serving is disabled, including authenticated owner downloads:
`/gradio_api/file=...`, deprecated `/file/...`, `/proxy=...`, and `/stream/...` variants,
with GET/HEAD covered. Anonymous requests receive 401; logged-in requests receive 403
and `Cache-Control: no-store`. The unused recording-processing upload path is disabled.
Only package/static application assets remain served; model paths remain blocked too.
The existing preview uses a browser-local Blob URL and needs no server media download.

Gradio 6.30 moves file-shaped inputs to its cache even with `preprocess=False`. Therefore
the File component has no backend event inputs: a frontend-only change handler forwards
the opaque handle into a hidden scalar component, and Verify authenticates it before
opening or deleting anything. Inputless reset callbacks retain stale-result clearing and
cancellation. Bounded control middleware rejects injected FileData/nested objects before
Gradio traverses them, preventing cached-path and remote-URL preprocessing bypasses.

Static/custom-component routes were reviewed separately: they serve installed component
assets through the framework's safe joins, not this private store. Disabling file/proxy
routes is a deliberate restriction; add any future download feature only with explicit
per-owner authorization. Knowing a path or handle is never sufficient authorization.

## E. Dependency and container assessment

All ten existing `runtime.PINNED_VERSIONS`, including Torch 2.11.0+cpu, TorchVision
0.26.0+cpu, TorchAudio 2.11.0+cpu, NumPy 2.2.6, OpenCV 4.12.0.88, Pillow 12.0.0 and
Gradio 6.30.0, remain unchanged. Encoder runtime profiles, FFmpeg 5.1.9 package pin and
Python 3.11.17 base digest remain unchanged. FileLock 4.2.4 and python-multipart 0.0.32
are now explicit direct dependencies, already present in the verified Linux environment.

The complete closure pins 66 distributions and hashes 68 compatible public wheels from
PyPI and the official PyTorch CPU index. Docker enforces `--require-hashes` and binary-only
installation, then `pip check`. Pip is updated from 24.0 to 26.2.2 and wheel from 0.46.3
to 0.48.0. The newer setuptools 84.2.1 was rejected by the clean resolver because verified
Torch requires `<82`; original setuptools 79.0.1 is retained, with its advisories recorded.
No dependency conflict was bypassed with `--no-deps` or an incompatible Torch upgrade.

The Dockerfile frontend is digest pinned. Application/model directories and copied source
are root-owned and not writable by runtime UID 1000; only the private temporary directory
is writable. Model artifacts, credentials, datasets and user uploads are excluded from
build context/source COPY. Runtime secrets are supplied only at launch, never in image
layers. The default command remains the guarded Gradio application, with no extra service.

Advisory review queried version-specific public PyPI metadata for all 66 packages with
no failed lookups; CPU local-version builds were checked against their upstream release
metadata, not treated as a separate binary vulnerability database. The final scan still reports records for retained Pillow, setuptools
and Torch (aliases mean these are not all distinct vulnerabilities). Packaging-tool
updates remove the old pip records; they do not make the entire image vulnerability-free.
The upload path does not pass user media to Pillow, compile user-supplied TorchScript,
load user checkpoints or build/install packages at runtime. These are exposure reductions,
not blanket CVE fixes. Keep the private access wall and obtain dependency/security review
before broader deployment; changing the verified scientific environment is separate work.
[PyPI Pillow metadata](https://pypi.org/pypi/pillow/12.0.0/json),
[setuptools metadata](https://pypi.org/pypi/setuptools/79.0.1/json),
[Torch metadata](https://pypi.org/pypi/torch/2.11.0/json).

The Debian dependency repository is not snapshot locked. Record the built image digest
and OS package inventory for release reproducibility; wheel hashes do not make OS image
builds byte-identical across future dates. No numerical parity investigation was repeated.

## F. Docker and direct HTTP verification

The final application image uses the clean, hash-verified dependency build followed by
an application-layer rebuild for the SIGTERM fix. Image identity, exact source hashes,
installed closure, layer audit and HTTP records are saved under `_local/phase8_stage2h/`.
Final image: `sha256:ff722b5a2f4971d027e0eb0c4399e88e591925e5ab86939d32535636466e1cdc`.
All 13 layers (40,954 regular files) were inspected without extracting payloads to disk;
no private artifact digest, upload payload path or credential pattern was found, and no
authentication secret is present in image configuration. Exact application source and
all 66 installed distribution versions match the reviewed tree and lock.
The test container runs with no network or published ports, read-only root, UID 1000,
2 CPUs, 4 GiB RAM, 128 PIDs, no capabilities, no-new-privileges and 512 MiB tmpfs.
Only synthetic media and ephemeral in-memory credentials are used; no model is mounted.

Direct HTTP checks cover anonymous config/upload/Verify rejection, multiple-file
rejection with no retained partials, known-path download denial, cross-login Verify
misuse without consumption, actual FFprobe validation, loading/status compatibility,
50-MiB-plus-one rejection, file-object injection, two genuinely stalled TCP requests,
reader/capacity ceilings, real five-minute expiry, forced restart and graceful shutdown.
The owner can still preview locally and Verify a supported ten-second H.264/AAC file;
the result remains `MODEL_NOT_READY`, `RAW_INFERENCE_GATE_CLOSED`, `predictions: null`.

Two different named accounts and separate same-account logins are tested through the
real pinned ASGI HTTP contract. The Docker production configuration supports one shared
username/password pair, so its real-socket cross-login check is supplemented by those
named-account tests; no unsupported multi-user configuration was added.

Final measured counts and image identity are recorded in the completion evidence and
summarized in the Stage 2H handoff. No hosted or physical-device result is claimed.

| Real final-image HTTP/lifecycle check | Result |
| --- | --- |
| Owner Verify on actual 10-s H.264/AAC bytes | Loading and MODEL_NOT_READY/null predictions; managed source deleted |
| Multiple files / 50 MiB + 1 byte | Rejected; no persistent partial file |
| Other account/login and known media path | Retrieval denied; unauthorized Verify could not consume the owner's upload |
| FileData injection / direct REST queue bypass | Rejected before cache traversal / before Verify execution |
| Two TCP connections stalled after headers | Both timed out after the production 30-s deadline; third body reader rejected |
| Four retained uploads | Fifth refused; reserved disk capacity stayed bounded |
| Four abandoned uploads | All deleted after 301.076 s without another request; expired handle rejected |
| Forced process restart | Orphan payloads removed before admission; old login invalidated |
| Graceful SIGTERM | Pending upload deleted and process exited successfully |

Machine-readable results: `docker-http-final.json`, `image-metadata.json`,
`image-inspect.json` and `image-layer-audit.json` in the ignored evidence directory.

## G. Regression results

| Verification | Final result |
| --- | --- |
| Full Python suite | 983 passed, 2 CUDA skips, 1 existing weight-norm deprecation warning; 1,346 subtests passed in 81.10 s |
| Focused cases within that full run | 150 deployment/HTTP/security/upload/live cases |
| Same focused suite in final Linux image | 146 passed, 4 Node-dependent skips in 16.86 s |
| Host JavaScript | Eight timing cases and five controller/preview tests passed |
| Dependency resolution/install | Clean apt/wheel build, wheel hash checks and `pip check` passed |
| Preservation and layer scans | 40 artifact roles, 28 protected historical files and all 13 final image layers verified |

Unavailable Node checks in the runtime image are not counted as passing there; the host
run covers them. All synthetic regression fixtures remain separate from held-out data.
Earlier failures are retained locally: an outdated Gradio test stub, a missing test-only
pytest compatibility shim, the incompatible setuptools upgrade, and the real SIGTERM
cleanup defect corrected in this change. Final source is not relabelled from those runs.

## H. Historical preservation

The preflight preservation manifest covers 1,590 existing files. Only the intended code,
test, Docker/dependency and README edits may differ. All five pre-existing untracked
files are unchanged. All 40 externally manifest-bound model/source files and all 28
protected archived historical report files match their saved hashes. The manifest
digest remains `65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e`.

`deployment/runtime.py`, `deployment/internal_backend.py` and `fusion/inference.py`
remain byte-identical to preflight. E3 threshold remains `0.009579099009671126`;
A2 remains `0.0026640388978268056`, with original checkpoint/metadata bindings.
No TEST evaluation, retraining, threshold tuning, parity-policy change or public
prediction activation occurred. The Git index remains empty.

## I. Outstanding Hugging Face and release checks

1. Use a Private Docker Space, port 7860, single process/replica, unique runtime credential
   Secrets and only intended collaborators. Confirm account entitlement and private access.
2. Confirm proxy early body/connection/receive/rate limits, HTTPS cookies and queue SSE.
   Application limits do not control bytes already accepted by an upstream proxy.
3. Test actual browser File-to-handle forwarding, local preview, file replacement/reset,
   expiry messaging and Verify through the hosted HTTPS page. No server download should work.
4. Test real webcam/microphone permissions, background/unplug/Stop and browser variants;
   retain all live prediction gates. These physical/provider checks were not simulated
   as completed by the local HTTP suite.
5. Verify private temp disk permissions/capacity, five-minute cleanup, single-worker lock,
   HF cold start and a stop/rebuild with sufficient shutdown grace. Forced kills must
   purge recognized old payloads before accepting the next session.
6. Review retained dependency advisories and OS package inventory. Any approved upgrade
   needs its own compatibility/scientific review and regression; do not bypass pins.
7. A gated UI demo needs no model weights. If artifact readiness is to be shown, separately
   authorize and verify an immutable private read-only mount and exact schemas/hashes.
   Do not upload TRAIN exports, frame arrays, trace archives or private evidence.

The Stage 2G runbook remains the hosting reference with the security corrections here.
Neither private hosting nor these upload fixes approve audio/video numerical parity or
authorize a genuine public inference path.

## J. Decision

**READY FOR PR** for the Stage 2H upload-hardening change. All three confirmed upload
blockers are fixed and verified in direct HTTP tests and the final Docker runtime.
The production signal-cleanup and direct-REST serialization gaps found during testing
are fixed too. Final regression, preservation and image-content checks pass.

This is not deployment or scientific prediction approval. The HF/provider/browser
acceptance checks and retained dependency advisory review in section I remain open.
No model was retrained, no threshold changed and no TEST data was scored. Public
predictions remain disabled with null outputs. No commit, push, merge or deployment
was performed, and GC2 remained separate from GC1.
