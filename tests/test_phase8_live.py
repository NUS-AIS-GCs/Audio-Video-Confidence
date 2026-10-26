"""Phase 8 live capture: rolling buffer timing, A/V synchronisation, failure handling and session cleanup.

Streams are simulated the way the browser produces them: audio in small contiguous chunks indexed by
sample on the shared timeline, camera frames at ~30 fps with jitter, stamped in seconds on that timeline.
"""
import base64
import json
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from deployment.live import buffer as B
from deployment.live.buffer import LiveBuffer, LiveBufferError
from deployment.live.service import LiveCaptureService, decode_frame, decode_pcm16

ROOT = Path(__file__).resolve().parents[1]


def second_level(k):
    """Audio amplitude that encodes which timeline second a sample belongs to."""
    return 0.02 + 0.01 * (k % 50)


def frame_for(k):
    """A 224x224 frame whose pixel value encodes its timeline second."""
    return np.full((224, 224, 3), k % 250, np.uint8)


def feed(buf, seconds, sr=16_000, chunk=0.1, fps=30.0, rng=None, jitter_s=0.004, audio=True, frames=True,
         skip_audio=(), skip_frames=(), t0=0.0):
    """Simulated capture from t0 to t0+seconds, interleaving audio chunks and camera frames in time order."""
    rng = rng or np.random.default_rng(0)
    events = []
    if audio:
        step = int(chunk * sr)
        for start in range(int(t0 * sr), int((t0 + seconds) * sr), step):
            if int(start // sr) in skip_audio:
                continue
            idx = np.arange(start, start + step)
            x = np.sin(2 * np.pi * 440 * idx / sr) * np.array([second_level(i // sr) for i in idx])
            events.append(((start + step) / sr, "a", start, x.astype(np.float32)))
    if frames:
        for i in range(int(t0 * fps), int((t0 + seconds) * fps)):
            t = i / fps + rng.uniform(-jitter_s, jitter_s)
            if t < 0 or int(round(t)) in skip_frames:
                continue
            events.append((t, "v", t, frame_for(int(round(t)))))
    for _, kind, a, b in sorted(events, key=lambda e: e[0]):
        buf.add_audio(a, b) if kind == "a" else buf.add_frame(a, b)


# ---- timing and synchronisation ---------------------------------------------------------------
def test_buffer_becomes_ready_with_ten_aligned_seconds():
    buf = LiveBuffer(16_000)
    feed(buf, 9.0)
    assert not buf.ready and buf.diagnostics()["segments_valid"] < 10
    feed(buf, 3.0, t0=9.0)
    d = buf.diagnostics()
    assert buf.ready and d["segments_valid"] == 10 and len(d["segments"]) == 10
    assert all(s["status"] == B.OK for s in d["segments"])
    assert d["max_frame_offset_ms"] <= 1000 / 30 / 2 + 4 + 1e-6       # within half a frame (+jitter)
    snap = buf.snapshot()
    assert snap["waveform"].shape == (160_000,) and snap["waveform"].dtype == np.float32
    assert snap["frames"].shape == (10, 224, 224, 3) and snap["frames"].dtype == np.uint8
    assert snap["frame_timestamps"] == tuple(float(i) for i in range(10))


def test_audio_and_frame_of_each_segment_come_from_the_same_timeline_second():
    buf = LiveBuffer(16_000)
    feed(buf, 14.0)
    snap = buf.snapshot()
    for i, k in enumerate(snap["source_seconds"]):
        seg = snap["waveform"][i * 16_000:(i + 1) * 16_000]
        assert np.isclose(np.abs(seg).max(), second_level(k), atol=2e-3)   # audio of second k
        assert int(snap["frames"][i, 0, 0, 0]) == k % 250                  # frame of second k
    assert snap["source_seconds"] == tuple(range(snap["source_seconds"][0], snap["source_seconds"][0] + 10))


def test_rolling_window_keeps_latest_ten_seconds_and_bounded_memory():
    buf = LiveBuffer(16_000)
    feed(buf, 40.0, chunk=0.25)
    w = buf.window()
    assert w == list(range(w[-1] - 9, w[-1] + 1)) and w[-1] >= 38
    assert len(buf._seconds) <= 12                                  # window + the seconds still filling


def test_48k_capture_is_resampled_to_exactly_160000_samples_at_16k():
    sr = 48_000
    buf = LiveBuffer(sr)
    feed(buf, 12.0, sr=sr)
    wav = buf.snapshot()["waveform"]
    assert wav.shape == (160_000,)
    spectrum = np.abs(np.fft.rfft(wav[16_000:32_000]))
    assert abs(np.argmax(spectrum) - 440) <= 1                       # 440 Hz tone survives at 1 Hz/bin


def test_closest_frame_to_each_second_is_chosen():
    buf = LiveBuffer(16_000)
    buf.add_audio(0, np.full(1_600, 0.1, np.float32))
    buf.add_frame(0.06, frame_for(1))
    buf.add_frame(0.01, frame_for(2))                                 # closer -> replaces
    buf.add_frame(0.05, frame_for(3))                                 # further -> ignored
    buf.add_audio(1_600, np.full(16_000 * 3, 0.1, np.float32))
    seg = buf.diagnostics()["segments"][0]
    assert seg["frame_offset_ms"] == 10.0 and buf._seconds[0].frame[0, 0, 0] == 2
    assert buf.counters["replaced_frames"] == 1


def test_small_timestamp_jitter_is_contiguous_and_overlap_is_dropped():
    buf = LiveBuffer(16_000)
    buf.add_audio(0, np.full(1600, 0.1, np.float32))
    buf.add_audio(1600 + 40, np.full(1600, 0.1, np.float32))          # 2.5 ms jitter -> joined
    buf.add_audio(3200 - 800, np.full(1600, 0.1, np.float32))         # 50 ms overlap -> first copy kept
    assert buf.counters["audio_gaps"] == 0
    assert buf.counters["overlap_samples"] == 800
    assert buf._seconds[0].have[:4000].all()


# ---- failure handling ---------------------------------------------------------------------------
def test_audio_gap_invalidates_its_second_until_it_leaves_the_window():
    buf = LiveBuffer(16_000)
    feed(buf, 12.0, skip_audio={4})
    d = buf.diagnostics()
    assert not buf.ready and buf.counters["audio_gaps"] >= 1
    assert [s["status"] for s in d["segments"] if s["second"] == 4] == [B.MISSING_AUDIO]
    with pytest.raises(LiveBufferError, match="LIVE_BUFFER_NOT_READY"):
        buf.snapshot()
    feed(buf, 12.0, t0=12.0)                                         # second 4 rolls out -> ready again
    assert buf.ready


def test_dropped_camera_second_is_reported_not_repaired():
    buf = LiveBuffer(16_000)
    feed(buf, 12.0, skip_frames={6})
    statuses = {s["second"]: s["status"] for s in buf.diagnostics()["segments"]}
    assert statuses[6] == B.DROPPED_FRAME and not buf.ready


def test_frames_far_from_the_second_boundary_do_not_count():
    buf = LiveBuffer(16_000)
    feed(buf, 12.0, frames=False)
    for k in range(12):
        buf.add_frame(k + 0.3, frame_for(k))                          # 300 ms off the grid
    assert buf.counters["off_grid_frames"] == 12
    assert all(s["status"] == B.DROPPED_FRAME for s in buf.diagnostics()["segments"])


def test_microphone_stopping_while_camera_runs_marks_missing_audio():
    buf = LiveBuffer(16_000)
    feed(buf, 5.0)
    feed(buf, 7.0, audio=False, t0=5.0)                              # frames keep the timeline moving
    statuses = {s["second"]: s["status"] for s in buf.diagnostics()["segments"]}
    assert statuses[3] == B.OK and statuses[6] == B.MISSING_AUDIO


def test_camera_stopping_while_microphone_runs_marks_dropped_frames():
    buf = LiveBuffer(16_000)
    feed(buf, 5.0)
    feed(buf, 7.0, frames=False, t0=5.0)
    statuses = {s["second"]: s["status"] for s in buf.diagnostics()["segments"]}
    assert statuses[7] == B.DROPPED_FRAME


def test_muted_microphone_digital_silence_is_flagged():
    buf = LiveBuffer(16_000)
    feed(buf, 3.0, frames=True)
    buf.add_audio(48_000, np.zeros(16_000 * 3, np.float32))
    feed(buf, 3.0, audio=False, t0=3.0)
    statuses = {s["second"]: s["status"] for s in buf.diagnostics()["segments"]}
    assert statuses[3] == B.MUTED_AUDIO


def test_late_input_after_finalisation_is_counted_and_ignored():
    buf = LiveBuffer(16_000)
    feed(buf, 4.0)
    before = buf.diagnostics()["segments"]
    buf.add_frame(1.0, frame_for(99))
    overlap = buf.counters["overlap_samples"]
    buf.add_audio(16_000, np.full(100, 0.5, np.float32))             # already-covered samples: dropped
    assert buf.counters["late_frames"] == 1 and buf.counters["overlap_samples"] == overlap + 100
    assert buf.diagnostics()["segments"] == before


@pytest.mark.parametrize("call,code", [
    (lambda b: b.add_audio(0, np.array([np.nan], np.float32)), "INVALID_AUDIO_SAMPLES"),
    (lambda b: b.add_audio(0, np.array([1.5], np.float32)), "INVALID_AUDIO_SAMPLES"),
    (lambda b: b.add_audio(-1, np.zeros(10, np.float32)), "INVALID_AUDIO_START"),
    (lambda b: b.add_audio(1.5, np.zeros(10, np.float32)), "INVALID_AUDIO_START"),
    (lambda b: b.add_audio(16_000 * 60, np.zeros(10, np.float32)), "AUDIO_TOO_FAR_AHEAD"),
    (lambda b: b.add_frame(0.0, np.zeros((10, 10, 3), np.uint8)), "INVALID_FRAME"),
    (lambda b: b.add_frame(float("nan"), frame_for(0)), "INVALID_FRAME_TIME"),
    (lambda b: b.add_frame(100.0, frame_for(0)), "FRAME_TOO_FAR_AHEAD"),
])
def test_invalid_input_is_rejected_with_a_code(call, code):
    with pytest.raises(LiveBufferError, match=code):
        call(LiveBuffer(16_000))


def test_unsupported_sample_rate_is_rejected():
    with pytest.raises(LiveBufferError, match="UNSUPPORTED_SAMPLE_RATE"):
        LiveBuffer(8_000)


def test_ready_window_passes_the_backend_raw_media_contract():
    buf = LiveBuffer(48_000)
    feed(buf, 12.0, sr=48_000)
    media = buf.to_raw_media_input()
    assert tuple(media.waveform.shape) == (1, 160_000) and tuple(media.frames.shape) == (1, 10, 224, 224, 3)
    assert media.audio_validity.all() and media.visual_validity.all()


# ---- service: payloads, ownership, limits and cleanup ----------------------------------------------
def pcm_b64(x):
    return base64.b64encode((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()).decode()


def jpeg_b64(value, size=(480, 640)):
    img = np.full((*size, 3), value, np.uint8)
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def run_session(svc, owner, seconds, sr=16_000, push_s=0.25, fps=30.0):
    token = svc.start(owner, {"sample_rate": sr})["session"]
    last, step, frames_sent = None, int(push_s * sr), set()
    for p in range(int(seconds / push_s)):
        start = p * step
        x = 0.1 * np.sin(2 * np.pi * 440 * np.arange(start, start + step) / sr)
        t_end = (start + step) / sr
        frames = []
        for k in range(int(start / sr), int(t_end) + 1):          # browser sends ~one frame per second
            if k not in frames_sent and k + 1 / fps <= t_end:
                frames.append({"t": k + 0.01, "image": jpeg_b64(k % 250)})
                frames_sent.add(k)
        last = svc.push(owner, {"session": token, "audio": [{"start": start, "pcm": pcm_b64(x)}],
                                "frames": frames, "client": {"video_fps": 30.0, "audio_muted": False}})
    return token, last


def test_service_end_to_end_reaches_ready_and_contract_check_without_predictions():
    svc = LiveCaptureService()
    token, d = run_session(svc, "alice|s1", 12.0)
    assert d["status"] == "CAPTURING" and d["ready"] and d["segments_valid"] == 10
    assert d["client"] == {"video_fps": 30.0, "audio_muted": False}
    r = svc.check("alice|s1", {"session": token})
    assert r["status"] == "CONTRACT_VALID" and r["predictions"] is None
    assert r["waveform_shape"] == [1, 160_000] and r["frames_shape"] == [1, 10, 224, 224, 3]
    assert "prediction" not in json.dumps(d).lower()
    assert svc.stop("alice|s1", {"session": token}) == {"status": "STOPPED"}
    assert svc.active_sessions == 0


def test_check_before_ready_is_not_ready():
    svc = LiveCaptureService()
    token, _ = run_session(svc, "o", 3.0)
    r = svc.check("o", {"session": token})
    assert r["status"] == "NOT_READY" and r["error_code"] == "LIVE_BUFFER_NOT_READY" and r["predictions"] is None


def test_session_belongs_to_its_owner():
    svc = LiveCaptureService()
    token = svc.start("alice|s1", {"sample_rate": 16_000})["session"]
    assert svc.push("mallory|s2", {"session": token, "audio": [], "frames": []})["error_code"] == "LIVE_SESSION_NOT_FOUND"
    svc.stop("mallory|s2", {"session": token})
    assert svc.active_sessions == 1                                  # another owner cannot stop it


def test_capacity_limit_and_restart_replaces_own_session():
    svc = LiveCaptureService(max_sessions=2)
    a1 = svc.start("a", {"sample_rate": 16_000})["session"]
    svc.start("a", {"sample_rate": 16_000})                         # restart: old one is freed
    assert svc.push("a", {"session": a1})["error_code"] == "LIVE_SESSION_NOT_FOUND"
    svc.start("b", {"sample_rate": 16_000})
    assert svc.start("c", {"sample_rate": 16_000}) == {"status": "REJECTED", "error_code": "LIVE_CAPACITY_REACHED"}


def test_idle_sessions_expire_and_free_their_buffer():
    now = [0.0]
    svc = LiveCaptureService(idle_timeout_s=20, clock=lambda: now[0])
    token, _ = run_session(svc, "o", 2.0)
    buf = svc._sessions[token].buffer
    now[0] = 25.0
    assert svc.active_sessions == 0 and not buf._seconds
    assert svc.push("o", {"session": token})["error_code"] == "LIVE_SESSION_NOT_FOUND"


@pytest.mark.parametrize("payload,code", [
    ({"audio": "x"}, "INVALID_PUSH"),
    ({"audio": [{"start": 0, "pcm": "***"}]}, "INVALID_AUDIO_PAYLOAD"),
    ({"audio": [{"start": 0, "pcm": "AAAA"}]}, "INVALID_AUDIO_PAYLOAD"),   # 3 bytes: odd byte count
    ({"frames": [{"t": 0.0, "image": base64.b64encode(b"not an image").decode()}]}, "INVALID_FRAME_PAYLOAD"),
    ({"frames": [{"t": 0.0, "image": jpeg_b64(0)}] * 5}, "PUSH_TOO_LARGE"),
    ({"audio": [{"start": 0, "pcm": pcm_b64(np.zeros(16_000 * 3))}]}, "PUSH_TOO_LARGE"),
    ({"frames": [{"t": 0.0, "image": jpeg_b64(0, size=(2000, 2000))}]}, "INVALID_FRAME_PAYLOAD"),
])
def test_malformed_or_oversized_pushes_are_rejected_and_change_nothing(payload, code):
    svc = LiveCaptureService()
    token = svc.start("o", {"sample_rate": 16_000})["session"]
    r = svc.push("o", {"session": token, **payload})
    assert r == {"status": "REJECTED", "error_code": code}
    assert svc._sessions[token].buffer.counters["audio_chunks"] == 0
    assert svc._sessions[token].buffer.counters["frames"] == 0


def test_bad_start_requests_are_rejected():
    svc = LiveCaptureService()
    assert svc.start("o", {"sample_rate": "48000"})["error_code"] == "UNSUPPORTED_SAMPLE_RATE"
    assert svc.start("o", None)["error_code"] == "UNSUPPORTED_SAMPLE_RATE"
    assert svc.active_sessions == 0


def test_client_report_keeps_only_whitelisted_fields():
    svc = LiveCaptureService()
    token = svc.start("o", {"sample_rate": 16_000})["session"]
    d = svc.push("o", {"session": token, "client": {"video_fps": 29.97, "audio_muted": True,
                                                     "user_agent": "x", "dropped_video_frames": float("inf")}})
    assert d["client"] == {"video_fps": 29.97, "audio_muted": True}


def test_logs_never_include_media_payloads(caplog):
    caplog.set_level(logging.DEBUG, logger="avalidate.deployment.live")
    svc = LiveCaptureService()
    pcm = pcm_b64(0.1 * np.ones(1600))
    token = svc.start("o", {"sample_rate": 16_000})["session"]
    svc.push("o", {"session": token, "audio": [{"start": 0, "pcm": pcm}]})
    svc.stop("o", {"session": token})
    assert pcm[:40] not in caplog.text and token not in caplog.text


def test_frame_decode_matches_training_resize():
    img = (np.arange(480 * 640 * 3) % 251).astype(np.uint8).reshape(480, 640, 3)
    png = base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()
    expected = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(decode_frame(png), expected)


# ---- browser side ----------------------------------------------------------------------------------
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_browser_timing_helpers():
    r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "test_live_timing.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_browser_pcm_encoding_decodes_identically_in_python():
    js = ("const T=require(process.argv[1]);const x=new Float32Array([0,0.5,-0.5,1,-1,0.25,-0.999]);"
          "process.stdout.write(T.bytesToBase64(T.floatToPcm16(x)));")
    out = subprocess.run([NODE, "-e", js, str(ROOT / "deployment" / "web" / "live_timing.js")],
                         capture_output=True, text=True, check=True).stdout
    got = decode_pcm16(out)
    assert np.allclose(got, [0, 0.5, -0.5, 1, -1, 0.25, -0.999], atol=1 / 32767)


def test_frontend_widget_wiring():
    gr = pytest.importorskip("gradio")
    from deployment.ui.live_section import build_live_section, live_js
    js = live_js()
    for fn in ("live_start", "live_push", "live_check", "live_stop"):
        assert f"server.{fn}(" in js
    assert "getUserMedia" in js and "requestVideoFrameCallback" in js and "getTracks()" in js
    with gr.Blocks() as demo:
        build_live_section(gr, LiveCaptureService())
    html = [b for b in demo.blocks.values() if isinstance(b, gr.HTML)]
    assert len(html) == 1
    assert sorted(f.__name__ for f in html[0].server_fns) == ["live_check", "live_push", "live_start", "live_stop"]
