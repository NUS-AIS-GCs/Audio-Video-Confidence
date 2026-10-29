# Final combined PR inventory — 30 October 2026

Prepared for owner review; no staging, commit, push, PR creation, merge or deployment performed.
Base: `8a5e513dfbf4acd1c3adbf2e4a7c18f7236e840d` (main / merged PR #32).
This supersedes the earlier 17-file UI and 28-file Head 2 staging lists.
The final local manifest binds working SHA-256, normalized staged SHA-256 and Git blob IDs;
any source/base change requires renewed review. It is `_local/professor_demo/staging-allowlist.json`.

## Exact changed-file staging allowlist

41 files.

- `.gitattributes`
- `README.md`
- `deployment/event_class_names.json`
- `deployment/event_evidence.py`
- `deployment/general_inference.py`
- `deployment/gradio_app.py`
- `deployment/live/buffer.py`
- `deployment/live/replay.py`
- `deployment/live/research_capture.py`
- `deployment/live/service.py`
- `deployment/product_seed.py`
- `deployment/temporal_presence.py`
- `deployment/temporal_presence_catalog.json`
- `deployment/ui/general_research_section.py`
- `deployment/ui/live_section.py`
- `deployment/ui/result_sections.py`
- `deployment/ui/theme.py`
- `deployment/web/live_capture.js`
- `deployment/web/live_research_result.js`
- `deployment/web/live_timing.js`
- `docs/deployment_handoff_20261030.md`
- `docs/final_pr_inventory_20261030.md`
- `docs/local_three_head_demo.md`
- `docs/prediction_diagnosis_20261030.md`
- `docs/product_demo.md`
- `docs/production_hosting.md`
- `docs/temporal_presence.md`
- `docs/temporal_presence_protocol.json`
- `fusion/temporal_presence_data.py`
- `fusion/temporal_presence_model.py`
- `fusion/train_temporal_presence.py`
- `requirements-temporal-presence.txt`
- `tests/integration_live_browser.cjs`
- `tests/integration_microphone.cjs`
- `tests/integration_product_demo.cjs`
- `tests/js/test_event_results.js`
- `tests/js/test_live_security.js`
- `tests/test_event_evidence.py`
- `tests/test_live_research.py`
- `tests/test_temporal_presence.py`
- `tests/test_temporal_presence_runtime.py`

## Inherited required source (already in main)

The PR is the delta above **plus the complete base checkout**. Keep the Dockerfile,
`.dockerignore`, deployment runtime locks, `requirements-deployment.txt`, authentic
audio/vision/fusion implementations, broker/supervisor/worker, signed stores,
ownership/expiry/cancellation, decoder and production service/preflight source.
The final running image was checked against 138 complete application source files;
no application file is missing or different. Do not deploy only files in this list.

## Excluded and preserved

`deployment/artifact_catalog.json` has a worktree status marker but is byte-identical
to HEAD; it is excluded. The following pre-existing historical review/training
materials are preserved unchanged outside this PR scope; they are not needed to
launch the final product. Current diagnosis and Head 2 documentation are included
above; earlier document/allowlist status statements do not describe this final PR.

- `docs/inference_compatibility_final_review.md`
- `docs/phase8_final_audio_trace_verification.md`
- `docs/phase8_final_video_frame_parity_verification.md`
- `docs/phase8_stage2g_deployment.md`
- `requirements-fusion-training.txt`
- `docs/scientific_release_review/README.md`
- `docs/scientific_release_review/audio_evidence_request.md`
- `docs/scientific_release_review/blockers.md`
- `docs/scientific_release_review/package_inventory.json`
- `docs/scientific_release_review/release_decision_draft.md`
- `docs/scientific_release_review/reproduction.md`
- `docs/scientific_release_review/reviewer_checklist.md`
- `docs/scientific_release_review/scope_bindings.json`
- `docs/scientific_release_review/train_confirmation_protocol.md`
- `docs/scientific_release_review/video_evidence_request.md`

All `_local/**`, private artifacts, credentials, keys, datasets, recordings, demo
videos, embeddings, caches, logs and screenshots are excluded. No redistribution
rights for demo media are claimed. The reviewed model handoff contains hashes and
paths only. Never use `git add .` or `git add -A`.

## Focused final evidence

- Host: 196 affected Python tests; another 51 live tests including the Node-dependent
  cases skipped in Docker. JavaScript: 43 Node-reported tests, all passing.
- Exact running image: 138 application files byte-identical to the worktree;
  Python syntax and critical product imports passed. Offline tests: 281 passed,
  three Node-unavailable skips (covered on the host); no failed checks remain.
- Authenticated browser: original/substituted upload and fake-device live
  three-head rendering, ownership, cancellation, Reset/Stop, device release,
  private-file/foreign-Host denial and media cleanup passed. No physical devices.
- Eight prepared local ten-second MP4s completed genuine authenticated three-head
  inference. Five original TRAIN controls and two fixed TRAIN swaps are demonstration
  selections, not independent evaluation. Synthetic bell is an OOD limitation case.

No scientific gate/threshold/model changes were introduced during this handoff.
Historical parity, browser JPEG sensitivity, unresolved physical-speaking Case B,
E3's 51/402 validation false accepts and Head 2's TEST limitations remain explicit.
See [deployment handoff](deployment_handoff_20261030.md) for mandatory separate
private Head 2 provisioning, supported Linux architecture and target acceptance.
