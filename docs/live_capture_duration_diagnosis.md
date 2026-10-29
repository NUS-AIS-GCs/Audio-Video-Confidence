# Live capture duration rejection diagnosis

2026-10-29, Asia/Singapore. Branch `feature/live-inference-integration`, HEAD `745d915d36cab91ca7fefa749a16ef0515ea3e94`. **No physical device was activated in this investigation.** Only generated arrays, synthetic timing and Chromium fake devices were used. The original one-attempt consent marker and physical logs are preserved.

## What the physical evidence establishes

The saved `physical-browser-error.txt` reports `LIVE_RESEARCH_CAPTURE_TOO_LONG` followed by a 240-second harness timeout. The corresponding session in `component-status.log` has one STARTED record, **52 CAPTURING responses, all `ready:false`**, one limit rejection and STOPPED. The physical worker log contains readiness only, no inference job. The failure was therefore in **capture ingestion, before completed-window sealing/ticket creation**, not signed-ticket expiry, model loading or model inference.

At that time the same primary error identified two different bounds: (1) more than **13 ? native sample rate** audio samples proposed, or (2) attempting a **65th clock observation**, with 64 already retained. There was no 10-second uploaded-container duration check in this flow. The browser's 60-second record deadline and the harness's 240-second wait are separate controls with different error messages. The final displayed `0.0 s` came from Stop resetting the UI and is not evidence that zero seconds were recorded.

**The strongest-supported explanation for the limit is the native audio-sample bound:** 52 nominal quarter-second pushes are consistent with 13 seconds, whereas the observation bound needs 64 committed pushes. However, the old best-effort logger silently swallowed recording errors and retained neither actual sample counts, sample rate, bin statuses, per-push timestamps nor frame counters. Its completeness cannot be certified. Thus the exact historical sub-limit and the reason readiness remained false cannot be proved from the retained evidence. No claim is made that the physical microphone was muted, frames were absent, the browser clock drifted, or the defect below necessarily caused that attempt.

## Confirmed defect and controlled reproduction

A real packet-order defect was reproduced independently. `LiveCaptureService._push` decoded the complete request and copied the buffer transaction, but then added all audio before frames. Each `add_audio` immediately finalized bins. At the precise **1.50-second batch** boundary, audio for seconds 5.0?6.5 advanced the watermark to 6.5 and finalized second 5 as dropped-frame. Its valid frame at time 5.0 was already in the same decoded packet, but was added afterward and marked late. A 1.49-second batch did not cross that boundary and succeeded.

One lost middle bin prevents ten consecutive valid bins before the 13-second capture bound. The legacy implementation consequently returned the same generic limit error even though the packet contained valid, synchronized media and was within both the browser's 1.5-second backlog bound and the server's two-second per-push bound. This is a transport-order defect, not a need to increase a timing tolerance or record a longer model window.

The matrix below used actual PCM16/JPEG decoding, an authenticated capture-service owner, strict clocks and signed ticket validation. It did not use models or physical input.

| Synthetic case | Before | After | Underlying data result |
|---|---|---|---|
| `normal` | ready at 10.5 s | ready at 10.5 s | late-frame count 0 ? 0 |
| `batch_1.49` | ready at 10.74 s | ready at 10.74 s | late-frame count 0 ? 0 |
| `batch_1.50` | rejected at 13.0 s | ready at 10.5 s | late-frame count 1 ? 0 |
| `muted_audio` | rejected at 13.0 s | rejected at 13.0 s | late-frame count 0 ? 0 |
| `missing_frames` | rejected at 13.0 s | rejected at 13.0 s | late-frame count 0 ? 0 |
| `off_grid_frames` | rejected at 13.0 s | rejected at 13.0 s | late-frame count 0 ? 0 |

Muted audio, missing frames and off-grid frames each reproduce **52 successful quarter-second pushes and the same legacy sample-limit rejection**. This observational ambiguity is why a specific physical-input cause cannot be reconstructed. In contrast, the batched-delivery test proves the implementation defect without attributing it retrospectively to the physical attempt.

## Fix and preserved limits

- `LiveBuffer.add_push` defers finalization until **all audio and frames in one already-bounded request** have been validated and inserted into the existing transaction copy. Service publication still occurs only after the complete transaction succeeds. Single-item buffer APIs retain their previous immediate-finalization defaults.
- A frame received after an **earlier committed packet** finalized its bin remains late; it cannot resurrect or repair that bin. The new test explicitly verifies this.
- No changes to ten complete model-input seconds, exactly 160,000 resampled samples, frame offset tolerance **100 ms**, finalization lag **500 ms**, native audio budget **13 seconds**, **64** observations, two-second server audio batches, 1.5-second client backlog, body sizes, concurrency, upload ownership, expiry or cancellation.
- Valid batches at 1.49, 1.50, 1.51 and 2.00 seconds were tested at 16 and 48 kHz. The last two exercise the unchanged server allowance; the client still stops above its own 1.5-second backlog bound. Native sample durations were separately tested with receipt wall clocks at 0.5?, 1? and 2? speed. Native completeness, not packet receipt wall time, determines window readiness; ten valid seconds require at least the existing 0.5-second finalization lag.
- Muted/missing/off-grid cases still fail at 208,000 accepted native samples for 16 kHz, with the next proposed packet at 212,000 rejected. No ticket or inference result is produced. A separate fast-push fixture identifies the 64-observation resource bound at only eight seconds of audio; it remains enforced and is now distinguished from the sample-duration bound.

## Sanitized failure observability and harness correction

Research responses now expose bounded `capture_progress` and, on a limit rejection, `capture_diagnostics`: ingest stage, explicit `audio_samples` or `clock_observations` limit kind, declared rate, accepted/proposed sample counts, sample budget, observation count/budget, server monotonic elapsed time and last committed bin/counter diagnostics. Rejected data never update the buffer. The response contains no PCM, JPEG, login owner, cookies, session/tab capabilities, cancellation key or ticket. A server warning records only fixed limit names, numeric counts and valid-bin count.

The browser releases all devices/session resources as before but retains these sanitized final counters in its diagnostic text until Reset. It does not retain media or render a prediction. The test harness records UTC, callback name and the structured progress; diagnostic recording failures fail the test instead of silently dropping evidence. It stops waiting when a terminal capture error appears instead of misreporting a four-minute result timeout as capture duration.

Fixture output can be isolated with `AVAL_RESEARCH_TEST_OUTPUT`; physical mode **always uses the original fixed consent-marker directory**, so this setting cannot bypass the consumed one-attempt guard. No physical-mode command was executed.

## Executed validation

- Focused Python live/clock/security tests: **117 passed**.
- Full Python regression: **1,109 passed + 1,346 subtests passed**, **two CUDA-dependent skips**, one upstream deprecation warning. Existing tests were not weakened.
- JavaScript regression: **18 passed** (the existing 17 plus diagnostic retention/cleanup).
- Exact isolated image: **41 timing/ticket-security tests passed**, including signed input reconstruction, expiry, cancellation, active-owner revocation, heartbeat, watchdog and cleanup checks.
- Actual Chromium fake-device ? signed-ticket ? genuine BEATs+LoRA/ResNet-18/frozen-E3 replay passed. It also passed second-login rejection, Host/Origin rejection, private-artifact download denial, Reset invalidation, raw capture deletion, no stale display and complete sanitized diagnostic recording.
- Final image/source inventory and ordinary closed gates passed; a networked research worker was explicitly refused. No weights, media or private signing keys are present in the image.

The new fake-device research result was raw E3 **0.005586551744552067** (threshold **0.009579099009671126**), **Audio-substituted-like pairing**, profile SHA-256 `f8cd26bd1cd912bf81c636d5b440c66b5a7b133df2c212bfd82144a639b096db`. Its final progress recorded **171392 native samples / 10.712 seconds**, **43 pushes**, **10.75 server elapsed seconds**, ten valid bins and **0 late frames**. This is a new generated fixture, not the same media as earlier fake captures, a benchmark result, a physical recording or evidence of sound-source causation.

E3 checkpoint `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925`, model fingerprint `d978d0dc9fa9986844936ef74686be7833e69d1291287d77f767307e94fc5025` and threshold **0.009579099009671126** remain unchanged. The verified worker checks artifact/model integrity before and after the genuine forward. Authentication, signed-ticket store/replay code, key handling, public readiness gates and numerical acceptance policy were not modified.

Tested image: `avalidate:live-timing-candidate`, ID `sha256:8443216a693dd3efb66fd2eb343b53ca9f4f9105d6ab85d3d94caa0d70e6ee55`. It retains nonroot UID 1000, no network, read-only root, dropped capabilities/no-new-privileges, four-GiB memory/no extra swap, two-CPU quota, 128 PIDs and bounded `/tmp`. The previous working image tag is preserved.

## Changed files for this diagnosis

1. `deployment/live/buffer.py` ? transaction-level finalization.
2. `deployment/live/service.py` ? ingest the decoded request as one batch.
3. `deployment/live/research_capture.py` ? unchanged named limits and sanitized progress/rejection metadata.
4. `deployment/web/live_capture.js` ? retain rejection diagnostics after resource cleanup.
5. `tests/test_live_capture_timing.py` ? 19 controlled regression cases.
6. `tests/js/test_live_security.js` ? cleanup/diagnostic retention case.
7. `tests/integration_live_browser.cjs` ? complete sanitized logging, terminal-error handling, isolated fixture output and image selection.
8. `docs/live_capture_duration_diagnosis.md` ? this report.
9. `docs/live_input_compatibility_draft.md` ? link the follow-up without replacing prior evidence.

All prior live-integration work is retained. The earlier handoff's artifacts stay in `_local/live_integration`; new evidence is in `_local/live_duration_diagnosis`. Unrelated research reports, training requirements, checkpoints and local evidence remain excluded from this PR and unmodified. No stage, commit, push, merge or deployment occurred.

## Reproduction without physical devices

```powershell
& _local/phase8_stage1/venv/Scripts/python.exe -B -m pytest tests/test_live_capture_timing.py tests/test_live_research.py tests/test_phase8_live.py tests/test_phase8_frontend_security.py -q --basetemp=_local/live_duration_diagnosis/review-temp
node --test tests/js/*.js
$env:AVAL_RESEARCH_IMAGE = 'avalidate:live-timing-candidate'
$env:AVAL_RESEARCH_TEST_OUTPUT = Join-Path $PWD '_local/live_duration_diagnosis/review-browser'
node tests/integration_live_browser.cjs
```

The browser command has **no physical flag** and explicitly launches fake media devices. It uses the existing private artifact mounts and no-network offline worker. The test dependencies and original model acquisition are those documented in the preceding dossier; no replacement checkpoints or downloads are introduced.

## Root-cause confidence, next real-device test and PR readiness

**Confirmed:** pre-ticket ingest-limit rejection; a reproducible same-packet finalization defect; insufficient original failure telemetry; an unnecessary later harness timeout and UI diagnostic loss. **Most supported historical limit:** native audio samples, subject to the old logger's completeness gap. **Not determined:** the original device's sample rate, exact rejected count, missing/muted/late/off-grid bin condition, or whether packet scheduling triggered this defect in that physical attempt.

A new real-device test is technically justified after this fix and telemetry improvement, but **is not authorized and was not performed**. It would require fresh explicit consent and should inspect the new counters rather than extend any duration/synchronization limit. Public Upload and Live predictions remain `MODEL_NOT_READY`; live numerical compatibility and independent production authorization remain blocked.

**PR readiness: YES for the gated research/timing correction**, with the historical physical cause and real-device compatibility explicitly unresolved. No production live readiness or scientific acceptance is claimed.

## Evidence identities

| Local evidence under `_local/live_duration_diagnosis/` | SHA-256 |
|---|---|
| `reproduction-before.json` | `7279a7f97e65e56e5502c614e9056e2d5810a1f8e0668319406bf9e48460ec6d` |
| `reproduction-after.json` | `95be0e42af97d89427c71a3d90cf33a8b410b83ff127a11e1ea4df2ee53e94d0` |
| `timing-tests.log` | `74a7d2434067b5c09ba5a77e899da82d4452459f8a30cd184aa7a4a819de839e` |
| `focused.log` | `49cedcad370184d6da7ea946866a70eae19ae6ce5294993a89825b0838e8c8f1` |
| `python-regression.log` | `135a5fd1090d5c875bd81802ae6dff039ada566df01683d65de8eff83a169939` |
| `javascript-tests.log` | `6b9e086d4037e476147e2f8d9039bc45a3301e5d2083ec5b010dbef383158a25` |
| `docker-timing-tests.log` | `15022c7b91c95ca485a5c5add3e077279c1378f52d67fd8ca0cb812e03a9f778` |
| `browser/browser-fixture-result.json` | `3ce1ff2b737815a811f9428677383e7fde72a1df03c60ab04a865635f9bf8941` |
| `browser/component-diagnostics.jsonl` | `f6f93325e4326f136c318a41d57ed0b1ed30b0bd3b6fe309655ec96e140301f6` |
| `browser-run.log` | `f3117e47bda861dbd575f00726ca92c25ecba2301d2ec932cac85cb0f6d2c0f5` |
| `final-image.json` | `1e1ed7c536325ad41c73786e40b5ad78927f60c6f4bc73f9d4a6f312220cf01e` |
| `networked-refusal.log` | `f4adc7374a02813358443b85eb5ab670ea7321334817c0111625d29c49c79a9f` |
| `docker-build.log` | `829515da05cfd4183f364a318311d26f8fe6c60e2905d997788af37e9ac29342` |

Preserved physical evidence from the prior task:

| Existing `_local/live_integration/` file | SHA-256 |
|---|---|
| `physical-attempt.json` | `cf4fed7504e9b88bc04f53c181a7408c90951273f45c29cb831712a69f63399a` |
| `physical-browser-error.txt` | `b02ecceb9edde5fcf2b0e69940ab29d512933ef927c309d656e306506bf40331` |
| `physical-capture-server.log` | `a025310a3dc303c62d48943f0489f6c0f86b26012f4a8ef6a0ff85a8f99bc11b` |
| `physical-replay-worker.log` | `071a218279755317ea1bc44e182ea91d95fa5a3a415079be128a09daddb2f00c` |
| `component-status.log` | `d4de42589f6d5d7142a30a75bf5208a0fe09d4028282cbb415352effdbf31ac9` |
