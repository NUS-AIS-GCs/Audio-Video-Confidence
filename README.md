---
title: AValidate Deployment Foundation
emoji: 🔊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# AValidate

This repository contains a Docker/Gradio readiness interface. Model inference is disabled until
the exact trained encoder artifacts, decoder contract, and raw-to-TRAIN export parity have been
verified. The interface does not return placeholder, annotation-derived, or synthetic predictions.
The app requires `AVAL_ACCESS_MODE=gradio_auth` and non-empty `AVAL_GRADIO_USERNAME` and
`AVAL_GRADIO_PASSWORD` runtime secrets before startup. Use a Private Space during initial testing;
the app authentication protects its UI and direct event/API requests.

The research Streamlit application remains `app.py` and is not the Docker entry point.

## Frontend and live capture

The dark-first Gradio app has Upload verification, Live verification, How It Works and Model Information tabs.
The Upload tab provides a local video preview, Verify progress and admission/result cards.
The Live tab records ten aligned seconds from a webcam and microphone (or keeps a rolling
ten-second buffer) and shows capture and
sync diagnostics only. Predictions stay disabled in both. See
[docs/phase8_frontend_live.md](docs/phase8_frontend_live.md) for usage, architecture, tests and
integration notes.

Stage 2G hosting settings, security blockers, artifact requirements and the deployment procedure
are in [docs/phase8_stage2g_deployment.md](docs/phase8_stage2g_deployment.md), with public settings
in [deployment/hf-space.env.example](deployment/hf-space.env.example). Public predictions remain
disabled. No Hugging Face deployment has been performed.

Stage 2H upload ownership, transport limits, expiry and container changes are documented in
[docs/phase8_stage2h_upload_hardening.md](docs/phase8_stage2h_upload_hardening.md).
Use that report for the current upload security assessment; Stage 2G records the earlier audit.

The current UI, authenticated internal TRAIN demonstration, typed prediction adapter, validation
evidence and manual release procedure are documented in
[docs/prediction_ui_release.md](docs/prediction_ui_release.md). The default Docker entry point remains
the gated application. `python -m deployment.internal_app` is a separate experimental entry point
requiring the existing authorized, hash-bound TRAIN artifacts; it cannot enable public predictions.
No production inference approval, checkpoint substitution or threshold change is introduced.

The subsequent [authentic upload/live integration handoff](docs/final_prediction_live_handoff.md)
records private runtime provisioning, the genuine uploaded TRAIN path, live cleanup fixes and the
remaining scientific and deployment approvals. General upload and live model predictions stay disabled.
