# syntax=docker/dockerfile:1
# Pin the official Python 3.11 Bookworm amd64 base image by digest.
FROM --platform=linux/amd64 python:3.11.17-slim-bookworm@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_TEMP_DIR=/tmp/gradio \
    AVAL_MODEL_ROOT=/models \
    AVAL_MAX_UPLOAD_BYTES=52428800 \
    PORT=7860

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
       ca-certificates \
       ffmpeg=7:5.1.9-0+deb12u1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
    && install -d -o 1000 -g 1000 /app /models /tmp/gradio

WORKDIR /app
COPY requirements-deployment.txt /app/requirements-deployment.txt
RUN python -m pip install --no-cache-dir --requirement /app/requirements-deployment.txt

COPY --chown=1000:1000 deployment /app/deployment
COPY --chown=1000:1000 fusion /app/fusion
COPY --chown=1000:1000 audio /app/audio
COPY --chown=1000:1000 vision /app/vision

USER 1000:1000
EXPOSE 7860
CMD ["python", "-m", "deployment.gradio_app"]
