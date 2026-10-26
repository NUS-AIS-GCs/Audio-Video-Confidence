"""Live capture sessions behind the authenticated Gradio app.

The browser calls three functions (wired in deployment/ui/live_section.py):
  live_start({"sample_rate": 48000})                     -> {"session": token, "limits": {...}}
  live_push({"session": token, "audio": [...], "frames": [...], "client": {...}}) -> diagnostics
  live_stop({"session": token})                          -> {"stopped": true}
plus live_check, which validates the current window against the backend RawMediaInput contract.

Every session belongs to one Gradio browser session (its session hash), is capped in number, size and
idle time, and is freed on stop or timeout. Errors are stable codes; media bytes are never logged.
No model runs here and no prediction is produced.
"""
from __future__ import annotations

import base64
import binascii
import logging
import secrets
import time
from dataclasses import dataclass, field
from threading import Lock

import numpy as np

from .buffer import FRAME_SIZE, LiveBuffer, LiveBufferError

LOGGER = logging.getLogger("avalidate.deployment.live")

MAX_SESSIONS = 4
IDLE_TIMEOUT_S = 20.0
MAX_AUDIO_CHUNKS_PER_PUSH = 8
MAX_AUDIO_SECONDS_PER_PUSH = 2.0
MAX_FRAMES_PER_PUSH = 4
MAX_FRAME_BYTES = 400 * 1024
MAX_FRAME_SIDE = 1920


@dataclass
class _Session:
    owner: str
    buffer: LiveBuffer
    lock: Lock = field(default_factory=Lock)
    last_seen: float = field(default_factory=time.monotonic)
    client: dict = field(default_factory=dict)


def decode_pcm16(text) -> np.ndarray:
    """Base64 little-endian int16 PCM -> float32 in [-1, 1)."""
    if not isinstance(text, str):
        raise LiveBufferError("INVALID_AUDIO_PAYLOAD")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise LiveBufferError("INVALID_AUDIO_PAYLOAD") from None
    if len(raw) % 2:
        raise LiveBufferError("INVALID_AUDIO_PAYLOAD")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def decode_frame(text) -> np.ndarray:
    """Base64 JPEG/PNG camera frame -> (224, 224, 3) uint8 RGB, resized like the training frames
    (OpenCV decode, BGR -> RGB, INTER_LINEAR to 224 x 224)."""
    import cv2
    if not isinstance(text, str) or len(text) > MAX_FRAME_BYTES * 4 // 3 + 4:
        raise LiveBufferError("INVALID_FRAME_PAYLOAD")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise LiveBufferError("INVALID_FRAME_PAYLOAD") from None
    if not (raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"\x89PNG\r\n\x1a\n")):
        raise LiveBufferError("INVALID_FRAME_PAYLOAD")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or max(bgr.shape[:2]) > MAX_FRAME_SIDE:
        raise LiveBufferError("INVALID_FRAME_PAYLOAD")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)


class LiveCaptureService:
    def __init__(self, max_sessions: int = MAX_SESSIONS, idle_timeout_s: float = IDLE_TIMEOUT_S,
                 clock=time.monotonic):
        self.max_sessions = max_sessions
        self.idle_timeout_s = idle_timeout_s
        self.clock = clock
        self._sessions: dict[str, _Session] = {}
        self._lock = Lock()

    # ---- session registry ---------------------------------------------------------------------
    def _expire(self) -> None:
        now = self.clock()
        for token in [t for t, s in self._sessions.items() if now - s.last_seen > self.idle_timeout_s]:
            self._sessions.pop(token).buffer.clear()
            LOGGER.info("live_session_expired")

    def _get(self, owner: str, payload) -> _Session:
        token = payload.get("session") if isinstance(payload, dict) else None
        with self._lock:
            self._expire()
            session = self._sessions.get(token) if isinstance(token, str) else None
        if session is None or not secrets.compare_digest(session.owner, owner):
            raise LiveBufferError("LIVE_SESSION_NOT_FOUND")
        session.last_seen = self.clock()
        return session

    @property
    def active_sessions(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)

    # ---- browser-facing calls (each returns a JSON-safe dict, never raises) --------------------
    def start(self, owner: str, payload) -> dict:
        try:
            sr = payload.get("sample_rate") if isinstance(payload, dict) else None
            if type(sr) is not int:
                raise LiveBufferError("UNSUPPORTED_SAMPLE_RATE")
            buffer = LiveBuffer(sample_rate=sr)
            with self._lock:
                self._expire()
                for t in [t for t, s in self._sessions.items() if s.owner == owner]:
                    self._sessions.pop(t).buffer.clear()          # one live session per browser tab
                if len(self._sessions) >= self.max_sessions:
                    raise LiveBufferError("LIVE_CAPACITY_REACHED")
                token = secrets.token_urlsafe(16)
                self._sessions[token] = _Session(owner=owner, buffer=buffer, last_seen=self.clock())
            LOGGER.info("live_session_started sample_rate=%d", sr)
            return {"status": "STARTED", "session": token,
                    "limits": {"max_frames_per_push": MAX_FRAMES_PER_PUSH,
                               "max_audio_seconds_per_push": MAX_AUDIO_SECONDS_PER_PUSH,
                               "idle_timeout_s": self.idle_timeout_s}}
        except LiveBufferError as exc:
            return {"status": "REJECTED", "error_code": exc.code}

    def push(self, owner: str, payload) -> dict:
        try:
            session = self._get(owner, payload)
            audio, frames = payload.get("audio", []), payload.get("frames", [])
            if not isinstance(audio, list) or not isinstance(frames, list):
                raise LiveBufferError("INVALID_PUSH")
            if len(audio) > MAX_AUDIO_CHUNKS_PER_PUSH or len(frames) > MAX_FRAMES_PER_PUSH:
                raise LiveBufferError("PUSH_TOO_LARGE")
            # decode everything before touching the buffer, so a bad push changes nothing
            chunks = []
            for c in audio:
                if not isinstance(c, dict):
                    raise LiveBufferError("INVALID_PUSH")
                chunks.append((c.get("start"), decode_pcm16(c.get("pcm"))))
            if sum(len(x) for _, x in chunks) > MAX_AUDIO_SECONDS_PER_PUSH * session.buffer.sample_rate:
                raise LiveBufferError("PUSH_TOO_LARGE")
            decoded = []
            for f in frames:
                if not isinstance(f, dict):
                    raise LiveBufferError("INVALID_PUSH")
                decoded.append((f.get("t"), decode_frame(f.get("image"))))
            with session.lock:
                for start, x in chunks:
                    session.buffer.add_audio(start, x)
                for t, rgb in decoded:
                    session.buffer.add_frame(t, rgb)
                client = payload.get("client")
                if isinstance(client, dict):
                    session.client = _client_report(client)
                return {"status": "CAPTURING", **session.buffer.diagnostics(), "client": session.client}
        except LiveBufferError as exc:
            return {"status": "REJECTED", "error_code": exc.code}

    def check(self, owner: str, payload) -> dict:
        """Validate the current window against the backend RawMediaInput contract. No inference."""
        try:
            session = self._get(owner, payload)
            with session.lock:
                media = session.buffer.to_raw_media_input()
            return {"status": "CONTRACT_VALID", "contract": "fusion.inference_contracts.RawMediaInput",
                    "waveform_shape": list(media.waveform.shape), "frames_shape": list(media.frames.shape),
                    "predictions": None}
        except LiveBufferError as exc:
            return {"status": "NOT_READY", "error_code": exc.code, "predictions": None}
        except Exception as exc:                  # contract violation from the backend validator
            LOGGER.warning("live_contract_rejected type=%s", type(exc).__name__)
            return {"status": "CONTRACT_REJECTED", "error_code": "RAW_MEDIA_CONTRACT_REJECTED", "predictions": None}

    def stop(self, owner: str, payload) -> dict:
        token = payload.get("session") if isinstance(payload, dict) else None
        with self._lock:
            session = self._sessions.get(token) if isinstance(token, str) else None
            if session is not None and secrets.compare_digest(session.owner, owner):
                self._sessions.pop(token).buffer.clear()
                LOGGER.info("live_session_stopped")
        return {"status": "STOPPED"}


def _client_report(client: dict) -> dict:
    """Only whitelisted numeric/boolean browser diagnostics are kept."""
    out = {}
    for key in ("audio_muted", "video_muted", "track_ended"):
        if isinstance(client.get(key), bool):
            out[key] = client[key]
    for key in ("dropped_video_frames", "audio_latency_ms", "video_fps", "camera_width", "camera_height"):
        v = client.get(key)
        if type(v) in (int, float) and np.isfinite(v) and abs(v) < 1e6:
            out[key] = v
    return out
