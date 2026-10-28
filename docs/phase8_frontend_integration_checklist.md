# Stage 2E frontend integration preparation

Read-only remote review on 2026-10-28. Backend HEAD/base:
`eff4e2fe6ed33650c07fb8f719fae9a665a7faf1` on
`feature/phase8-backend-stage1`.

`feature/phase8-frontend-live` was advertised and fetched at
`e7aea5a8e5a744f7b8092fed123a8b7682a55d27`. Its merge base is
`cb5da1c9448146293f9a31aabefc110efbb7298e`; it is six main commits behind and
one commit ahead. Its actual 13-file change adds `deployment/live`,
`deployment/ui`, `deployment/web`, modifies `deployment/gradio_app.py` and
`tests/test_phase8_deployment.py`, and adds Python/Node live tests. Apparent
research deletions in a direct HEAD-to-branch comparison are ancestry differences,
not changes introduced by the frontend commit. Review the merge-base diff.

`feature/phase8-upload-results` was not advertised by the exact remote-ref query.
Its work could not be reviewed; obtain its owner-approved revision before
integration. Absence from this query does not establish absence of private/local
work. No teammate branch was checked out, overwritten, merged or edited.

## Compatibility evidence

There are no file-path overlaps between the 36 proposed backend additions and
the live branch's 13 actual changed paths. The live upload builder retains the
current `DeploymentUI.submit` and readiness JSON contract. Its shared UI changes
still need owner review when rebased onto current main.

An ignored overlay copied current backend modules, overlaid only those 13 files
and ran the branch's deployment/live tests against current research packages:
**72 Python tests passed**; the Node timing test file passed (one Node test-file
result, with internal assertions). This is compatibility evidence for those tests,
not a merge, a browser/device test or production inference verification.

Live capture returns `CONTRACT_VALID` with waveform/frame shapes and
`predictions: null`; it does not load models. A future frontend adapter must keep
this capture-only status distinct from `MODEL_NOT_READY`, `REJECTED` and a future
reviewed prediction status. `aval.internal-train-inference.v1` is not a public
endpoint or a drop-in replacement for the proposed `aval.backend-inference.v1`.

## Capture and security findings requiring owner review

The live branch has owner-bound session hashes and random tokens, four sessions,
20-second idle expiry, one session per browser, bounded retained chunks/frames,
and stop/end/track cleanup. These are static-code findings, not multi-user browser
verification. The branch also has unresolved limits:

- Audio base64 decoding happens before the aggregate decoded-sample bound; bound
  encoded string length and chunk count before allocation, including component
  transport limits. Do not rely solely on post-decode duration checks.
- Compressed frames have a 400 KiB cap, but `cv2.imdecode` runs before the decoded
  dimensions are rejected. Review decompression allocation bounds and isolation.
- The buffer chooses nearest camera frames within 100 ms and then exposes bin
  labels `0..9`; those labels do not prove exact achieved timestamps. Retain
  source/audio clocks, actual frame PTS, uncertainty and validity separately.
- Browser JPEG/canvas resizing, torchaudio resampling and waveform clipping differ
  from the authenticated TRAIN MP4/FFmpeg path. JavaScript PCM scaling also differs
  for positive/negative values. Measure these operations against a separately
  approved capture contract before claiming encoder compatibility.

Do not copy these live tensors into Phase 6 provenance, relabel the achieved PTS
as exact shared-clock evidence, or route them into the five-MP4 whitelist. These
findings belong to the unmerged frontend work; Stage 2E does not change it.

## Integration checklist

- [ ] Pin current main and both owner-approved frontend revisions; compare their
  merge-base changes, rerun the overlay and review Gradio callbacks/theme changes.
- [ ] Preserve configured authentication on config, upload, queue, files and live
  component calls; verify unauthorized and cross-session denial in a real browser.
  Never put credentials or live session tokens in URLs, telemetry or screenshots.
- [ ] Keep one bounded upload, 50 MiB server/Gradio limit, existing codec/container,
  stream/duration rules, private API, queue size eight and callback concurrency one.
  Browser extension filtering is advisory; server rejection remains authoritative.
- [ ] Render `MODEL_NOT_READY` as unavailable with null predictions. Render
  `REJECTED` with sanitized actionable error text. Clear stale results when inputs
  change or a request fails; do not substitute demo/fallback scores.
- [ ] Define progress, pending, resource-busy, queue-full, timeout, expired-auth,
  network-loss and retry states, with bounded retries and upload cleanup. Never
  display internal audit tensors, paths, raw errors or model configuration inputs.
- [ ] Require explicit camera/microphone permission over the approved secure
  origin. Exercise denied/revoked permission, unavailable/busy device, mute,
  track-ended, tab backgrounding, page hide, pause/resume and session expiry.
- [ ] Fix the live allocation/timing/preprocessing findings above; exercise forged,
  oversized, truncated, stale and out-of-order chunks and cross-session tokens.
  Stop tracks, AudioWorklet/context and session resources on every exit path.
- [ ] Keep capture-only `CONTRACT_VALID` separate from inference readiness. Show
  achieved timing/validity truthfully; missing media must fail rather than be
  repaired into a model-ready tensor.
- [ ] After parity/release approval only, review a new public adapter without
  caller-selected checkpoint/threshold/ID. E3 is primary and A2 comparison; show
  uncalibrated pairing-provenance scores with no causal attribution. Event and
  temporal outputs remain null until separately trained/validated heads exist.
- [ ] Obtain frontend/backend owner review, device/browser integration evidence,
  authenticated end-to-end security checks and a separate release decision.

The boxes remain open deliberately: preparation and overlay tests do not
complete frontend integration. Evidence is retained under ignored
`_local/phase8_stage2e/frontend-*` and `frontend-overlay`.
