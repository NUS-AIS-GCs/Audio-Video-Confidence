"""Rolling 10-second live buffer: ten aligned one-second audio/video segments on one shared timeline.

The browser stamps both modalities on one clock (see deployment/web/live_capture.js):
  audio  chunks carry `start`, the index of their first sample on the shared timeline at the
         capture sample rate. The audio sample counter is gapless, so it is the timeline itself.
  frames carry `t`, seconds on the same timeline (the camera frame's capture time mapped onto it).
Second k covers audio samples [k*sr, (k+1)*sr) and takes the actual camera frame closest to k s.

A second is finalised once the stream has moved FINALIZE_LAG_S past its end. It is valid only if its
audio is complete and not muted and it has a frame within FRAME_TOLERANCE_S of its start. Nothing is
repaired (no silence padding, no frame reuse), matching the backend raw-media contract: the buffer is
ready only when the latest ten finalised seconds are all valid. This module never runs a model.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

SEGMENTS = 10
TARGET_SR = 16_000
FRAME_SIZE = 224
SUPPORTED_SAMPLE_RATES = (16_000, 22_050, 24_000, 32_000, 44_100, 48_000)
FRAME_TOLERANCE_S = 0.1          # max distance between a frame's capture time and its second's start
FINALIZE_LAG_S = 0.5             # wait this long past a second's end for late audio or frames
CHUNK_JOIN_TOLERANCE_S = 0.005   # sample-index jitter accepted as contiguous audio
MAX_JUMP_S = 5.0                 # input further ahead of the timeline than this is rejected
MAX_SESSION_S = 30 * 60          # one live session is capped at 30 minutes of timeline

OK = "ok"
MISSING_AUDIO = "missing_audio"
MUTED_AUDIO = "muted_audio"
DROPPED_FRAME = "dropped_frame"
MISSING_BOTH = "missing_audio_and_frame"


class LiveBufferError(Exception):
    """Sanitised error with a stable code; never carries media content."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class _Second:
    audio: np.ndarray                     # (sr,) float32, filled where `have` is True
    have: np.ndarray                      # (sr,) bool
    frame: np.ndarray | None = None       # (224, 224, 3) uint8 RGB
    frame_offset_s: float | None = None   # frame capture time minus the second's start
    status: str | None = None             # set when finalised


@dataclass
class LiveBuffer:
    sample_rate: int
    _seconds: "OrderedDict[int, _Second]" = field(default_factory=OrderedDict)
    _next_sample: int | None = None       # where the next contiguous audio chunk should start
    _audio_end_s: float = 0.0
    _last_frame_t: float = -1.0
    _finalised_upto: int = 0              # seconds [0, _finalised_upto) are final
    _statuses: dict = field(default_factory=dict)
    counters: dict = field(default_factory=lambda: {
        "audio_chunks": 0, "audio_gaps": 0, "gap_samples": 0, "overlap_samples": 0, "late_audio_samples": 0,
        "frames": 0, "late_frames": 0, "off_grid_frames": 0, "replaced_frames": 0})

    def __post_init__(self):
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise LiveBufferError("UNSUPPORTED_SAMPLE_RATE")

    # ---- input --------------------------------------------------------------------------------
    def add_audio(self, start: int, samples: np.ndarray, *, _finalize=True) -> None:
        """start: index of samples[0] on the shared timeline; samples: float32 mono in [-1, 1]."""
        if type(start) is not int or start < 0:
            raise LiveBufferError("INVALID_AUDIO_START")
        if start / self.sample_rate > self.watermark_s() + MAX_JUMP_S:
            raise LiveBufferError("AUDIO_TOO_FAR_AHEAD")
        if start / self.sample_rate > MAX_SESSION_S:
            raise LiveBufferError("SESSION_TOO_LONG")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all() or np.abs(samples).max(initial=0) > 1:
            raise LiveBufferError("INVALID_AUDIO_SAMPLES")
        self.counters["audio_chunks"] += 1
        tol = int(CHUNK_JOIN_TOLERANCE_S * self.sample_rate)
        if self._next_sample is not None:
            jump = start - self._next_sample
            if 0 < abs(jump) <= tol:                  # timestamp jitter: the stream is contiguous
                start = self._next_sample
            elif jump > tol:
                self.counters["audio_gaps"] += 1
                self.counters["gap_samples"] += jump
            elif jump < -tol:                         # repeated samples: keep the first copy only
                overlap = min(-jump, len(samples))
                self.counters["overlap_samples"] += overlap
                samples, start = samples[overlap:], start + overlap
        end = start + len(samples)
        self._next_sample = max(end, self._next_sample or 0)
        sr = self.sample_rate
        for k in range(start // sr, -(-end // sr)):
            lo, hi = max(start, k * sr), min(end, (k + 1) * sr)
            if hi <= lo:
                continue
            if k < self._finalised_upto:
                self.counters["late_audio_samples"] += hi - lo
                continue
            sec = self._second(k)
            sec.audio[lo - k * sr: hi - k * sr] = samples[lo - start: hi - start]
            sec.have[lo - k * sr: hi - k * sr] = True
        self._audio_end_s = max(self._audio_end_s, end / sr)
        if _finalize:
            self._finalise()

    def add_frame(self, t: float, rgb224: np.ndarray, *, _finalize=True) -> None:
        """t: capture time in seconds on the shared timeline; rgb224: (224, 224, 3) uint8 RGB."""
        if type(t) not in (int, float) or not np.isfinite(t) or t < -FRAME_TOLERANCE_S:
            raise LiveBufferError("INVALID_FRAME_TIME")
        if t > self.watermark_s() + MAX_JUMP_S:
            raise LiveBufferError("FRAME_TOO_FAR_AHEAD")
        if t > MAX_SESSION_S:
            raise LiveBufferError("SESSION_TOO_LONG")
        if not (isinstance(rgb224, np.ndarray) and rgb224.dtype == np.uint8
                and rgb224.shape == (FRAME_SIZE, FRAME_SIZE, 3)):
            raise LiveBufferError("INVALID_FRAME")
        self.counters["frames"] += 1
        self._last_frame_t = max(self._last_frame_t, float(t))
        k = int(round(t))
        offset = float(t) - k
        if abs(offset) > FRAME_TOLERANCE_S:
            self.counters["off_grid_frames"] += 1
        elif k < self._finalised_upto:
            self.counters["late_frames"] += 1
        else:
            sec = self._second(k)
            if sec.frame is None or abs(offset) < abs(sec.frame_offset_s):
                self.counters["replaced_frames"] += sec.frame is not None
                sec.frame, sec.frame_offset_s = rgb224, offset
        if _finalize:
            self._finalise()

    def add_push(self, chunks, frames) -> None:
        """Ingest one already-bounded transport batch before finalizing bins.

        The service calls this on its transaction copy. Validate every sample
        and frame with the existing checks; only finalization is deferred until
        both modalities in this same packet have been applied. Previously an
        audio watermark could finalize a bin before its co-delivered frame was
        inserted. Frames late relative to an earlier packet remain late.
        """
        for start, samples in chunks:
            self.add_audio(start, samples, _finalize=False)
        for timestamp, rgb in frames:
            self.add_frame(timestamp, rgb, _finalize=False)
        self._finalise()

    # ---- state --------------------------------------------------------------------------------
    def _second(self, k: int) -> _Second:
        if k not in self._seconds:
            sr = self.sample_rate
            self._seconds[k] = _Second(np.zeros(sr, np.float32), np.zeros(sr, bool))
        return self._seconds[k]

    def watermark_s(self) -> float:
        """How far the shared timeline has advanced. Frames keep it moving if audio stops, and audio
        keeps it moving if the camera stops, so a dead modality shows up as invalid seconds."""
        return max(self._audio_end_s, self._last_frame_t)

    def _finalise(self) -> None:
        upto = int(np.floor(self.watermark_s() - FINALIZE_LAG_S))
        while self._finalised_upto < upto:
            k = self._finalised_upto
            sec = self._second(k)
            audio_ok = bool(sec.have.all())
            frame_ok = sec.frame is not None
            if audio_ok and not frame_ok:
                sec.status = DROPPED_FRAME
            elif not audio_ok:
                sec.status = MISSING_AUDIO if frame_ok else MISSING_BOTH
            elif not np.any(sec.audio):                       # exact digital silence = muted track
                sec.status = MUTED_AUDIO
            else:
                sec.status = OK
            self._statuses[k] = sec.status
            self._finalised_upto += 1
        for k in [k for k in self._seconds if k < self._finalised_upto - SEGMENTS]:
            del self._seconds[k]                              # bounded memory: keep the window only
        for k in [k for k in self._statuses if k < self._finalised_upto - 6 * SEGMENTS]:
            del self._statuses[k]

    def window(self) -> list[int]:
        """The ten seconds of the rolling window (latest finalised), oldest first."""
        return list(range(max(0, self._finalised_upto - SEGMENTS), self._finalised_upto))

    @property
    def ready(self) -> bool:
        w = self.window()
        return len(w) == SEGMENTS and all(self._statuses[k] == OK for k in w)

    def diagnostics(self) -> dict:
        w = self.window()
        segments = [{"second": k, "status": self._statuses[k],
                     "frame_offset_ms": None if self._seconds[k].frame_offset_s is None
                     else round(1000 * self._seconds[k].frame_offset_s, 1)} for k in w]
        offsets = [abs(s["frame_offset_ms"]) for s in segments if s["frame_offset_ms"] is not None]
        return {
            "ready": self.ready,
            "segments_valid": sum(s["status"] == OK for s in segments),
            "segments": segments,
            "window_start_s": w[0] if w else None,
            "timeline_s": round(self.watermark_s(), 3),
            "audio_s": round(self._audio_end_s, 3),
            "video_s": round(max(self._last_frame_t, 0.0), 3),
            # how far frame delivery trails the audio; frames are sent once their second has closed
            "frame_lag_ms": round(1000 * (self._audio_end_s - self._last_frame_t), 1) if self._last_frame_t >= 0 else None,
            "max_frame_offset_ms": max(offsets) if offsets else None,
            "capture_sample_rate": self.sample_rate,
            "counters": dict(self.counters),
        }

    # ---- output -------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """The window as canonical arrays: waveform (160000,) float32 at 16 kHz, frames (10, 224, 224, 3)
        uint8, local timestamps 0..9. Raises NOT_READY unless all ten seconds are valid."""
        if not self.ready:
            raise LiveBufferError("LIVE_BUFFER_NOT_READY")
        w = self.window()
        audio = np.concatenate([self._seconds[k].audio for k in w])
        resampled = _resample(audio, self.sample_rate)
        clipped = int(np.count_nonzero(np.abs(resampled) > 1.0))
        waveform = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        frames = np.stack([self._seconds[k].frame for k in w])
        offsets = [self._seconds[k].frame_offset_s for k in w]
        # what was actually achieved, kept apart from the contract's 0..9 bin labels
        timing = {
            "frame_timestamps_are_bin_labels": True,
            "frame_capture_times_s": [round(i + o, 4) for i, o in enumerate(offsets)],
            "frame_offsets_ms": [round(1000 * o, 1) for o in offsets],
            "max_abs_frame_offset_ms": round(1000 * max(abs(o) for o in offsets), 1),
            "frame_tolerance_ms": round(1000 * FRAME_TOLERANCE_S),
            "source_seconds": list(w),
            "capture_sample_rate": self.sample_rate,
            "resampled": self.sample_rate != TARGET_SR,
            "clipped_samples": clipped,
        }
        return {"waveform": waveform, "frames": frames, "sample_rate": TARGET_SR,
                "frame_timestamps": tuple(float(i) for i in range(SEGMENTS)),
                "source_seconds": tuple(w), "capture_sample_rate": self.sample_rate, "timing": timing}

    def to_raw_media_input(self, snapshot: dict | None = None):
        """Wrap the snapshot in the backend's RawMediaInput contract and validate it (no model run)."""
        import torch
        from fusion.inference_contracts import RawMediaInput
        s = snapshot if snapshot is not None else self.snapshot()
        media = RawMediaInput(
            waveform=torch.from_numpy(s["waveform"].copy())[None],
            frames=torch.from_numpy(s["frames"].copy())[None],
            sample_rate=TARGET_SR, audio_duration=10, video_duration=10,
            audio_start_seconds=0, video_start_seconds=0, frame_timestamps=s["frame_timestamps"],
            audio_validity=torch.ones(1, SEGMENTS, dtype=torch.bool),
            visual_validity=torch.ones(1, SEGMENTS, dtype=torch.bool))
        media.validate()
        return media

    def clear(self) -> None:
        self._seconds.clear()
        self._statuses.clear()


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Band-limited resampling of the whole window at once (no per-chunk edge artefacts), clipped to
    [-1, 1] for the contract. snapshot() reports how many samples the clip changed."""
    return np.clip(_resample(audio, sr), -1.0, 1.0).astype(np.float32)


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        out = audio.astype(np.float32, copy=True)
    else:
        import torch
        import torchaudio.functional as AF
        out = AF.resample(torch.from_numpy(audio.astype(np.float32)), sr, TARGET_SR).numpy()
    out = out[: SEGMENTS * TARGET_SR]
    if len(out) != SEGMENTS * TARGET_SR:
        raise LiveBufferError("RESAMPLE_LENGTH_MISMATCH")
    return out
