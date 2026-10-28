# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# Pin the official Python 3.11 Bookworm amd64 base image by digest.
FROM python:3.11.17-slim-bookworm@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199

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
    && install -d -m 0755 /app /models \
    && install -d -o 1000 -g 1000 -m 0700 /tmp/gradio

WORKDIR /app
COPY requirements-deployment.txt /app/requirements-deployment.txt
COPY deployment/requirements-runtime.lock /app/requirements-runtime.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       --requirement /app/requirements-runtime.lock \
       --requirement /app/requirements-deployment.txt \
    && python -m pip check

COPY --chown=0:0 deployment /app/deployment
COPY --chown=0:0 fusion /app/fusion
COPY --chown=0:0 audio /app/audio
COPY --chown=0:0 vision /app/vision

USER 1000:1000
EXPOSE 7860
CMD ["python", "-m", "deployment.gradio_app"]
