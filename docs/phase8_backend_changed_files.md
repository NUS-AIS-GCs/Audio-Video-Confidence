# Exact Stage 1–2E proposed changed-file inventory

Base: `eff4e2fe6ed33650c07fb8f719fae9a665a7faf1` (main and backend HEAD at audit).
**36 additions; zero tracked modifications/deletions; index empty.** These are
the complete proposed PR paths, not just the files created during Stage 2E.
The final local `pr_inventory.json` records byte sizes and SHA-256 hashes.

## Deployment — 17 files

```text
deployment/audio-bundle-stage1.env.example
deployment/audio_bundle.py
deployment/encoder-stage1.env.example
deployment/encoder_diagnostics.py
deployment/encoders.py
deployment/internal_backend.py
deployment/paired_diagnostics.py
deployment/parity_metrics.py
deployment/run_stage2c_docker.py
deployment/run_stage2d_docker.py
deployment/seed_artifact_volume.py
deployment/stage2b.env.example
deployment/stage2c_probe.py
deployment/stage2d_profile.py
deployment/train_manifest.py
deployment/train_video_diagnostics.py
deployment/video_bundle.py
```

## Tests — 6 files

```text
tests/test_phase8_audio_bundle.py
tests/test_phase8_encoder_stage1.py
tests/test_phase8_freeze_audit.py
tests/test_phase8_internal_backend.py
tests/test_phase8_stage2c.py
tests/test_phase8_video_bundle.py
```

## Documentation — 13 files

```text
docs/phase8_audio_stage1.md
docs/phase8_backend_changed_files.md
docs/phase8_backend_freeze.md
docs/phase8_backend_pr.md
docs/phase8_encoder_stage1.md
docs/phase8_final_evaluation_protocol.md
docs/phase8_frontend_integration_checklist.md
docs/phase8_stage2b_video_fusion.md
docs/phase8_stage2c_parity_docker.md
docs/phase8_stage2c_parity_policy.md
docs/phase8_stage2d_internal_contract.md
docs/phase8_stage2d_runtime_backend.md
docs/video_export_handoff.md
```

## Preserved and excluded

`requirements-fusion-training.txt` was already untracked before Stage 1 and
remains unchanged. It is outside this PR's scope. All ignored local checkpoints,
datasets, TRAIN MP4s, exports, feature arrays, environments, audit artifacts and
logs remain excluded. No teammate frontend file is included. Existing tracked
research artifacts already in main are preserved, not added by this patch.

Stage 2E intentionally revises four of the pending deployment additions:
`internal_backend.py`, `stage2c_probe.py`, `seed_artifact_volume.py` and
`run_stage2d_docker.py`. It adds `train_manifest.py`, `test_phase8_freeze_audit.py`
and the five Stage 2E documents. Original pending-source snapshots remain local.
All other proposed paths are retained Stage 1–2D work/support documentation.
