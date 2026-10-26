---
title: AValidate Deployment Foundation
emoji: 🔊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# AValidate deployment foundation

This repository contains a Docker/Gradio readiness interface. Model inference is disabled until
the exact trained encoder artifacts, decoder contract, and raw-to-TRAIN export parity have been
verified. The interface does not return placeholder, annotation-derived, or synthetic predictions.
The app requires `AVAL_ACCESS_MODE=gradio_auth` and non-empty `AVAL_GRADIO_USERNAME` and
`AVAL_GRADIO_PASSWORD` runtime secrets before startup. Use a Private Space during initial testing;
the app authentication protects its UI and direct event/API requests.

The research Streamlit application remains `app.py` and is not the Docker entry point.
