// AValidate live capture: pure timing helpers (no DOM), shared by live_capture.js and the Node tests.
//
// One shared timeline for both modalities. Its origin is the first captured audio sample, and its unit
// is that audio clock: audio sample n sits at (n - originFrame) / sampleRate seconds. Camera frames are
// stamped in performance.now() milliseconds and mapped onto the audio clock with the AudioContext's
// own (contextTime, performanceTime) pair, so both modalities share one time base.
const AVTiming = (function () {
  const FRAME_TOLERANCE_S = 0.1;   // must match deployment/live/buffer.py FRAME_TOLERANCE_S

  // anchor: {contextTime (s), performanceTime (ms)} from AudioContext.getOutputTimestamp()
  function makeClock(anchor, originFrame, sampleRate) {
    // a zero performanceTime means the browser has not started the audio clock yet
    if (!anchor || !Number.isFinite(anchor.contextTime) || !Number.isFinite(anchor.performanceTime) || anchor.performanceTime <= 0) {
      throw new Error("INVALID_CLOCK_ANCHOR");
    }
    const originCtx = originFrame / sampleRate;
    return {
      sampleRate: sampleRate,
      originFrame: originFrame,
      // camera frame time (performance.now ms) -> seconds on the shared timeline
      perfToTimeline: function (perfMs) {
        return anchor.contextTime + (perfMs - anchor.performanceTime) / 1000 - originCtx;
      },
      // AudioContext frame index -> sample index on the shared timeline
      frameToSample: function (frame) {
        return Math.round(frame - originFrame);
      },
    };
  }

  // Keeps, for every whole second k, the camera frame captured closest to k seconds.
  // offer(t) says whether the frame at t should be kept (the caller then copies its pixels);
  // due(t) returns the seconds whose candidate can no longer be beaten, oldest first.
  function makeFramePicker(tolerance) {
    const tol = tolerance === undefined ? FRAME_TOLERANCE_S : tolerance;
    const best = new Map();   // k -> |t - k|
    let emittedUpTo = -1;     // seconds <= this have been emitted
    return {
      offer: function (t) {
        const k = Math.round(t);
        const d = Math.abs(t - k);
        if (k <= emittedUpTo || d > tol) return null;
        if (best.has(k) && best.get(k) <= d) return null;
        best.set(k, d);
        return k;
      },
      due: function (t) {
        const out = [];
        for (const k of Array.from(best.keys()).sort(function (a, b) { return a - b; })) {
          if (t > k + tol) { out.push(k); best.delete(k); emittedUpTo = Math.max(emittedUpTo, k); }
        }
        return out;
      },
      reset: function () { best.clear(); emittedUpTo = -1; },
    };
  }

  // Float32 [-1, 1] -> little-endian int16 PCM bytes. One scale for both signs, as FFmpeg's flt -> s16
  // conversion (round(x * 32768), clipped to the int16 range); the server divides by 32768 like soundfile.
  function floatToPcm16(samples) {
    const out = new Uint8Array(samples.length * 2);
    const view = new DataView(out.buffer);
    for (let i = 0; i < samples.length; i++) {
      const v = Math.round(Math.max(-1, Math.min(1, samples[i])) * 32768);
      view.setInt16(2 * i, Math.max(-32768, Math.min(32767, v)), true);
    }
    return out;
  }

  function bytesToBase64(bytes) {
    let s = "";
    const step = 0x8000;
    for (let i = 0; i < bytes.length; i += step) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
    }
    return typeof btoa === "function" ? btoa(s) : Buffer.from(bytes).toString("base64");
  }

  // Accumulates contiguous audio blocks; take() returns one chunk {start, samples} and clears.
  function makeAudioAccumulator() {
    let start = null, parts = [], length = 0;
    return {
      add: function (sampleIndex, block) {
        if (start !== null && sampleIndex !== start + length) {
          // discontinuity: the caller must flush first, so the gap reaches the server as a gap
          return false;
        }
        if (start === null) start = sampleIndex;
        parts.push(block); length += block.length;
        return true;
      },
      take: function () {
        if (start === null) return null;
        const samples = new Float32Array(length);
        let o = 0;
        for (const p of parts) { samples.set(p, o); o += p.length; }
        const chunk = { start: start, samples: samples };
        start = null; parts = []; length = 0;
        return chunk;
      },
      get length() { return length; },
      clear: function () { start = null; parts = []; length = 0; },
    };
  }

  // Readable message for getUserMedia failures (DOMException names)
  function permissionMessage(err) {
    const name = err && err.name;
    switch (name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Camera or microphone permission was denied. Allow both in the browser's site settings and try again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No camera or microphone was found. Connect both devices and try again.";
      case "NotReadableError":
      case "AbortError":
        return "The camera or microphone is in use by another application. Close it and try again.";
      case "NO_AUDIO_CLOCK":
        return "This browser does not provide the audio clock needed to align audio and video. Use a current version of Chrome or Edge.";
      case "INSECURE_CONTEXT":
        return "Live capture needs a secure page (https or localhost).";
      case "NO_AUDIO_TRACK":
        return "No microphone audio was received. Check that a microphone is connected and not disabled.";
      case "MICROPHONE_DISABLED":
        return "Unable to verify: the selected microphone track is disabled. Enable the microphone in browser and operating-system settings, then Reset and Record again.";
      case "MICROPHONE_TRACK_MUTED":
        return "Unable to verify: the selected microphone track is muted or temporarily unavailable. Check the hardware mute switch and selected browser input, then Reset and Record again.";
      case "AUDIO_CONTEXT_NOT_RUNNING":
        return "Unable to verify: the browser audio context is suspended or interrupted. Keep this tab active, then click Reset and Record again. No microphone samples were substituted.";
      case "NO_VIDEO_TRACK":
        return "No camera video was received. Check that a camera is connected and not disabled.";
      default:
        return "Live capture could not start (" + (name || "unknown error") + ").";
    }
  }

  const REQUEST_TIMEOUT_MS = 3000;
  const MAX_BACKLOG_SECONDS = 1.5;  // below the server's two-second push limit
  const MAX_PENDING_FRAMES = 4;
  const MAX_PENDING_CHUNKS = 8;
  function backlogAllowed(sampleRate, samples, chunks, frames) {
    return samples <= Math.floor(sampleRate * MAX_BACKLOG_SECONDS) &&
      chunks <= MAX_PENDING_CHUNKS && frames <= MAX_PENDING_FRAMES;
  }
  function withTimeout(promise, timeoutMs) {
    return new Promise(function (resolve, reject) {
      const timer = setTimeout(function () { reject(new Error("LIVE_REQUEST_TIMEOUT")); },
        timeoutMs === undefined ? REQUEST_TIMEOUT_MS : timeoutMs);
      Promise.resolve(promise).then(function (value) { clearTimeout(timer); resolve(value); },
        function (err) { clearTimeout(timer); reject(err); });
    });
  }
  return { FRAME_TOLERANCE_S, makeClock, makeFramePicker, floatToPcm16, bytesToBase64,
           makeAudioAccumulator, permissionMessage, backlogAllowed, withTimeout,
           REQUEST_TIMEOUT_MS, MAX_BACKLOG_SECONDS, MAX_PENDING_FRAMES };
})();

if (typeof module !== "undefined" && module.exports) { module.exports = AVTiming; }
