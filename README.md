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

For the working authenticated **private product demonstration**, use
`python -m deployment.run_product_demo --models PATH_TO_VERIFIED_MODELS --port 7862`.
It combines real MP4/MOV verification and signed-ticket live replay through one isolated worker.
See [local launch and acceptance evidence](docs/product_demo.md). Public release gates remain closed.

The final private candidate presents real event evidence, frozen trained Head 2
temporal presence and E3 correspondence for upload and completed live captures.
Use the [30 October deployment handoff](docs/deployment_handoff_20261030.md)
for the complete source review, separate private Head 2 provisioning and SSH-only
Linux launch requirements. The [latest diagnosis](docs/prediction_diagnosis_20261030.md)
defines current scientific limitations; physical-speaking Case B remains unresolved.
Earlier stage reports below describe historical candidates and do not approve this revision.

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

## General-media research candidate

The optional `deployment.general_research_app` runs the authentic frozen models
on a restricted general-media profile inside a no-network, read-only, resource-limited
Linux research container. It requires existing authenticated upload ownership.
The default app still returns `MODEL_NOT_READY`; live capture remains diagnostics-only.
Historical parity and independent release authorization remain blocked.
See the [DRAFT / NOT APPROVED release dossier](docs/general_upload_release_draft.md)
for supported scope, genuine evidence, reproduction and approval requirements.

## Local live-input research

The optional `deployment.live_research_app` captures authenticated, consented windows
on localhost and exchanges short-lived signed tickets with a separate offline model
worker. The worker retains the existing no-network Linux sandbox requirements.
The default Upload and Live prediction gates remain closed. See the
[DRAFT live-input compatibility dossier](docs/live_input_compatibility_draft.md)
for genuine fixture results, the unsuccessful single physical attempt, numerical
differences, security tests and independent approval requirements.
