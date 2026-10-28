# Stage 2E proposed pull request

No commit, push, pull request creation or merge was performed. Use the exact
[36-file allowlist](phase8_backend_changed_files.md); never use `git add .` for
this worktree. The pre-existing `requirements-fusion-training.txt` is unrelated
and remains untracked, preserved and excluded.

## Title

`feat(deployment): prepare authentic internal AV inference with production gates closed`

## Proposed description

The deployment UI currently admits authenticated uploads but cannot safely offer
raw-media predictions while encoder parity and decoder timing remain unresolved.
This change adds authentic BEATs iter3/LoRA and ResNet-18 bundle loading plus a
separate, SHA-whitelisted five-clip TRAIN backend using frozen E3/A2. Public
uploads continue to return `MODEL_NOT_READY` with null predictions.

The internal path loads models once, verifies original artifacts and loaded state,
serializes requests, records achieved timing and rejects provenance changes.
Offline Docker tools mount artifacts read-only, keep private checkpoints outside
image layers and report feature drift, resource use and latency. The separately
named numerical policy remains inactive. Shared manifest authentication and the
volume seeder now reject invalid provenance even under Python optimization;
response validation binds scores to logits and immutable operating points.

Validation on the audited environment: focused **239 passed**; full regression
**863 passed, 2 CUDA skips**, with 1,346 passing subtests and one existing Torch
deprecation warning. Stage 2D had 841 passes; 22 new freeze tests account for the
increase. An isolated frontend overlay passed 72 Python tests and its Node timing
test file. The complete security/runtime evidence and preservation results are
in the [Stage 2E audit](phase8_backend_freeze.md).

Limitations: historical exact parity fails; runtime budgets are unreviewed and
inactive; producer evidence, complete TRAIN coverage, a strict general decoder
and frontend integration remain outstanding. Five positive TRAIN cases do not
establish accuracy, calibration or causal source attribution. This PR does not
score held-out TEST, train/tune models, activate predictions or deploy anything.

## Proposed commit grouping

1. **Authentic encoder adapters and admission:** `audio_bundle.py`, `video_bundle.py`,
   `encoders.py`, `train_manifest.py`, and the three stage environment examples.
   Keep examples clearly diagnostic; none is a readiness override.
2. **Internal execution and offline evidence tools:** `internal_backend.py`,
   `paired_diagnostics.py`, `encoder_diagnostics.py`, `train_video_diagnostics.py`,
   `parity_metrics.py`, `stage2c_probe.py`, `stage2d_profile.py`, both Docker
   coordinators and `seed_artifact_volume.py`.
3. **Regression coverage:** the six proposed `tests/test_phase8_*.py` additions,
   including the freeze audit. Document the genuine-artifact fixture dependency
   and honest skips outside the authenticated local environment.
4. **Review and handoff documentation:** the thirteen proposed Markdown additions,
   preserving dated Stage 1–2D findings and adding the Stage 2E audit, exact
   inventory, frontend checklist, evaluation protocol and this PR draft.

Groups describe an eventual reviewed history; no staging or commits are made.

## Merge acceptance for this gated internal PR

- Independent reviewer checks the exact allowlist, source diff, trusted artifact
  bindings, seeder authentication and sanitization changes; verify final file
  hashes against the local audit manifest through approved evidence access.
- Reproduce applicable tests, explicitly listing private-fixture/CUDA skips and
  any unavailable Docker/device checks. Preserve the original artifacts/reports.
- Confirm public Gradio/runtime and Phase 6 exact gates remain unchanged and that
  no experimental result enables public inference or misstates score semantics.
- Confirm no local media, checkpoints, credentials, generated logs or unrelated
  requirements file enter the patch. Review source build exclusions separately
  from the existing image's layer audit; a fresh PR image was not built here.
- Resolve any substantive review defect before merge and obtain repository owner
  approval. Frontend and raw prediction activation require separate integration
  and release reviews.

Parity failure is explicitly disclosed scope for this gated internal PR, not
waived inference acceptance. Release still requires producer reconciliation,
complete TRAIN parity, independently approved numerical/timing policy, general
decoder/security evidence and frontend integration. The future final evaluation
also needs approval of the frozen protocol and explicit execution authorization.
