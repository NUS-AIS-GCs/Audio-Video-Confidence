"""Live capture sessions behind the authenticated Gradio app.

The browser calls functions wired in deployment/ui/live_section.py:
  live_start({"sample_rate": 48000}) -> {"session": token, "tab_key": capability, "limits": {...}}
  live_push({"session": token, "tab_key": capability, "audio": [...], "frames": [...]}) -> diagnostics
  live_stop({"session": token, "tab_key": capability}) -> {"status": "STOPPED"}
plus live_check, which validates the current window against the backend RawMediaInput contract.

Every session binds a server-authenticated login and a separate server-minted tab capability.
Registry operations are serialized, pushes commit atomically, and an expiry worker clears idle media.
No model runs here and no prediction is produced.
"""
from __future__ import annotations

import base64
import binascii
import copy
import logging
import secrets
import time
from dataclasses import dataclass, field
from threading import Event, RLock, Thread, current_thread

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

# How live media is produced, so nobody mistakes it for the TRAIN MP4/FFmpeg path. The live path has
# not been measured against that path, so encoder compatibility stays NOT_VERIFIED.
CAPTURE_PROFILE = {
    "id": "aval.live-capture.v0",
    "audio": "browser AudioWorklet float32 -> int16 (x32768, clipped, as FFmpeg flt->s16) -> /32768; "
             "the 10 s window is resampled to 16 kHz with torchaudio when the capture rate differs",
    "video": "browser canvas JPEG (quality 0.9, longest side <= 640) -> OpenCV decode, BGR -> RGB, "
             "INTER_LINEAR to 224 x 224",
    "clock": "browser AudioContext sample counter; camera frames mapped onto it with getOutputTimestamp()",
    "encoder_compatibility": "NOT_VERIFIED",
}


@dataclass
class _Session:
    owner: str = field(repr=False)
    tab_key: str = field(repr=False)
    buffer: LiveBuffer = field(repr=False)
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


def _b64_limit(n_bytes: int) -> int:
    """Longest base64 string that can decode to n_bytes."""
    return 4 * -(-n_bytes // 3)


def image_size(raw: bytes) -> tuple[int, int] | None:
    """(width, height) read from a JPEG SOF or PNG IHDR header without decoding pixels; None if absent."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(raw) < 24 or raw[12:16] != b"IHDR":
            return None
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if not raw.startswith(b"\xff\xd8"):
        return None
    i = 2
    while i + 4 <= len(raw):
        if raw[i] != 0xFF:
            return None
        marker = raw[i + 1]
        if marker == 0xFF:                                   # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:         # markers without a length
            i += 2
            continue
        if marker in (0xD9, 0xDA):                           # end of image / scan before any frame header
            return None
        length = int.from_bytes(raw[i + 2:i + 4], "big")
        if length < 2:
            return None
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > len(raw):
                return None
            return int.from_bytes(raw[i + 7:i + 9], "big"), int.from_bytes(raw[i + 5:i + 7], "big")
        i += 2 + length
    return None


def decode_frame(text) -> np.ndarray:
    """Base64 JPEG/PNG camera frame -> (224, 224, 3) uint8 RGB, resized like the training frames
    (OpenCV decode, BGR -> RGB, INTER_LINEAR to 224 x 224). The header's dimensions are checked before
    any pixel is decoded, so a small compressed file cannot expand into a huge image."""
    import cv2
    if not isinstance(text, str) or len(text) > _b64_limit(MAX_FRAME_BYTES):
        raise LiveBufferError("INVALID_FRAME_PAYLOAD")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise LiveBufferError("INVALID_FRAME_PAYLOAD") from None
    if not (raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"\x89PNG\r\n\x1a\n")):
        raise LiveBufferError("INVALID_FRAME_PAYLOAD")
    size = image_size(raw)
    if size is None or not (1 <= size[0] <= MAX_FRAME_SIDE and 1 <= size[1] <= MAX_FRAME_SIDE):
        raise LiveBufferError("INVALID_FRAME_PAYLOAD")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[:2] != (size[1], size[0]):
        raise LiveBufferError("INVALID_FRAME_PAYLOAD")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)


class LiveCaptureService:
    def __init__(self, max_sessions: int = MAX_SESSIONS, idle_timeout_s: float = IDLE_TIMEOUT_S,
                 clock=time.monotonic):
        if not 1 <= max_sessions <= MAX_SESSIONS or not 0 < idle_timeout_s <= IDLE_TIMEOUT_S:
            raise ValueError("live limits cannot exceed deployment limits")
        self.max_sessions = max_sessions
        self.idle_timeout_s = idle_timeout_s
        self.clock = clock
        self._sessions: dict[str, _Session] = {}
        self._lock = RLock()
        self._shutdown = Event()
        self._worker = None
        self._closed = False

    def _reap(self):
        while not self._shutdown.wait(min(1.0, self.idle_timeout_s / 2)):
            with self._lock:
                self._expire()
                if not self._sessions:
                    self._worker = None
                    return

    def close(self):
        """Stop the worker and release media at application shutdown (also used by tests)."""
        with self._lock:
            self._closed = True
            self._shutdown.set()
            for session in self._sessions.values():
                session.buffer.clear()
            self._sessions.clear()
            worker = self._worker
        if worker is not None and worker is not current_thread():
            worker.join(timeout=2)

    # ---- session registry ---------------------------------------------------------------------
    def _expire(self) -> None:
        now = self.clock()
        for token in [t for t, s in self._sessions.items() if now - s.last_seen >= self.idle_timeout_s]:
            self._sessions.pop(token).buffer.clear()
            LOGGER.info("live_session_expired")

    def _get(self, owner: str, payload) -> _Session:
        token = payload.get("session") if isinstance(payload, dict) else None
        self._expire()
        session = self._sessions.get(token) if isinstance(token, str) and len(token) <= 128 else None
        tab_key = payload.get("tab_key") if isinstance(payload, dict) else None
        if (self._closed or session is None or not isinstance(owner, str) or not owner
                or not isinstance(tab_key, str) or len(tab_key) > 128 or not tab_key.isascii()
                or not secrets.compare_digest(session.owner.encode(), owner.encode())
                or not secrets.compare_digest(session.tab_key.encode(), tab_key.encode())):
            raise LiveBufferError("LIVE_SESSION_NOT_FOUND")
        return session

    @property
    def active_sessions(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)

    # ---- browser-facing calls (each returns a JSON-safe dict, never raises) --------------------
    def start(self, owner: str, payload) -> dict:
        try:
            if not isinstance(owner, str) or not owner:
                raise LiveBufferError("LIVE_AUTHENTICATION_REQUIRED")
            sr = payload.get("sample_rate") if isinstance(payload, dict) else None
            if type(sr) is not int:
                raise LiveBufferError("UNSUPPORTED_SAMPLE_RATE")
            buffer = LiveBuffer(sample_rate=sr)
            with self._lock:
                self._expire()
                if self._closed:
                    raise LiveBufferError("LIVE_UNAVAILABLE")
                # Starting another tab never replaces another capture, even for the same account.
                # An explicit restart must prove possession of the old tab capability first.
                if isinstance(payload, dict) and ("session" in payload or "tab_key" in payload):
                    old = self._get(owner, payload)
                    self._sessions.pop(payload["session"])
                    old.buffer.clear()
                if len(self._sessions) >= self.max_sessions:
                    raise LiveBufferError("LIVE_CAPACITY_REACHED")
                token = secrets.token_urlsafe(16)
                tab_key = secrets.token_urlsafe(32)
                self._sessions[token] = _Session(owner=owner, tab_key=tab_key,
                                                buffer=buffer, last_seen=self.clock())
                if self._worker is None:
                    self._worker = Thread(target=self._reap, name="aval-live-expiry", daemon=True)
                    self._worker.start()
            LOGGER.info("live_session_started sample_rate=%d", sr)
            return {"status": "STARTED", "session": token, "tab_key": tab_key,
                    "limits": {"max_frames_per_push": MAX_FRAMES_PER_PUSH,
                               "max_audio_seconds_per_push": MAX_AUDIO_SECONDS_PER_PUSH,
                               "idle_timeout_s": self.idle_timeout_s}}
        except LiveBufferError as exc:
            return {"status": "REJECTED", "error_code": exc.code}

    def push(self, owner: str, payload) -> dict:
        # The same registry lock covers authentication, decoding, commit, stop and expiry.
        # HTTP transport admits at most four in-flight operations. No stopped session can
        # later be mutated or resurrected by a delayed worker.
        with self._lock:
            return self._push(owner, payload)

    def _push(self, owner: str, payload) -> dict:
        try:
            session = self._get(owner, payload)
            audio, frames = payload.get("audio", []), payload.get("frames", [])
            if not isinstance(audio, list) or not isinstance(frames, list):
                raise LiveBufferError("INVALID_PUSH")
            if len(audio) > MAX_AUDIO_CHUNKS_PER_PUSH or len(frames) > MAX_FRAMES_PER_PUSH:
                raise LiveBufferError("PUSH_TOO_LARGE")
            if not all(isinstance(x, dict) for x in audio + frames):
                raise LiveBufferError("INVALID_PUSH")
            # bound the encoded sizes before decoding anything (base64 is 4 chars per 3 bytes)
            max_samples = int(MAX_AUDIO_SECONDS_PER_PUSH * session.buffer.sample_rate)
            pcm = [c.get("pcm") for c in audio]
            images = [f.get("image") for f in frames]
            if not all(isinstance(p, str) for p in pcm):
                raise LiveBufferError("INVALID_AUDIO_PAYLOAD")
            if sum(map(len, pcm)) > _b64_limit(2 * max_samples) + 4 * len(pcm):
                raise LiveBufferError("PUSH_TOO_LARGE")
            if not all(isinstance(i, str) and len(i) <= _b64_limit(MAX_FRAME_BYTES) for i in images):
                raise LiveBufferError("INVALID_FRAME_PAYLOAD")
            # decode everything before touching the buffer, so a bad push changes nothing
            chunks = [(c.get("start"), decode_pcm16(p)) for c, p in zip(audio, pcm)]
            if sum(len(x) for _, x in chunks) > max_samples:
                raise LiveBufferError("PUSH_TOO_LARGE")
            decoded = [(f.get("t"), decode_frame(i)) for f, i in zip(frames, images)]
            # A bounded copy is a transaction: even a late timestamp failure leaves
            # the existing buffer, counters, client diagnostics and idle deadline unchanged.
            candidate = copy.deepcopy(session.buffer)
            for start, x in chunks:
                candidate.add_audio(start, x)
            for t, rgb in decoded:
                candidate.add_frame(t, rgb)
            client = payload.get("client")
            report = _client_report(client) if isinstance(client, dict) else session.client
            result = {"status": "CAPTURING", **candidate.diagnostics(), "client": report}
            session.buffer, session.client = candidate, report
            session.last_seen = self.clock()
            return result
        except LiveBufferError as exc:
            return {"status": "REJECTED", "error_code": exc.code}
        except Exception:
            return {"status": "REJECTED", "error_code": "LIVE_PUSH_FAILED"}

    def check(self, owner: str, payload) -> dict:
        """Validate the current window against the backend RawMediaInput contract. No inference.

        CONTRACT_VALID means only that the shapes, dtypes and validity masks fit the contract. The achieved
        frame times are reported separately: the contract's timestamps 0..9 are bin labels, not proof
        that each frame was captured exactly on its second."""
        with self._lock:
            return self._check(owner, payload)

    def _check(self, owner: str, payload) -> dict:
        try:
            session = self._get(owner, payload)
            snap = session.buffer.snapshot()
            media = session.buffer.to_raw_media_input(snap)
            session.last_seen = self.clock()
            from deployment.release_gate import unapproved_prediction
            return {"status": "CONTRACT_VALID", "contract": "fusion.inference_contracts.RawMediaInput",
                    "waveform_shape": list(media.waveform.shape), "frames_shape": list(media.frames.shape),
                    "timing": snap["timing"], "capture": CAPTURE_PROFILE,
                    "inference": "NOT_CONNECTED", "predictions": None,
                    "prediction_readiness": unapproved_prediction("live")}
        except LiveBufferError as exc:
            return {"status": "NOT_READY", "error_code": exc.code, "predictions": None}
        except Exception as exc:                  # contract violation from the backend validator
            LOGGER.warning("live_contract_rejected type=%s", type(exc).__name__)
            return {"status": "CONTRACT_REJECTED", "error_code": "RAW_MEDIA_CONTRACT_REJECTED", "predictions": None}

    def stop(self, owner: str, payload) -> dict:
        with self._lock:
            try:
                self._get(owner, payload)
                self._sessions.pop(payload["session"]).buffer.clear()
                LOGGER.info("live_session_stopped")
                return {"status": "STOPPED"}
            except LiveBufferError:
                return {"status": "REJECTED", "error_code": "LIVE_SESSION_NOT_FOUND"}


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
