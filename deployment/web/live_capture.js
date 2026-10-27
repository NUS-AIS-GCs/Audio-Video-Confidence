// AValidate live capture controller. Runs as gr.HTML js_on_load (variables: element, server), after
// live_timing.js (AVTiming). Captures webcam + microphone from ONE getUserMedia stream, stamps both on
// the shared timeline, and pushes audio samples and one frame per second to the server buffer
// (deployment/live). It shows readiness and sync diagnostics only: it never shows a prediction.
(function () {
  const PUSH_INTERVAL_MS = 250;
  const MAX_FRAME_SIDE = 640;
  const JPEG_QUALITY = 0.9;
  const $ = function (sel) { return element.querySelector(sel); };
  const ui = {
    video: $(".av-preview"), start: $(".av-start"), stop: $(".av-stop"), check: $(".av-check"),
    state: $(".av-state"), msg: $(".av-msg"), grid: $(".av-grid"), diag: $(".av-diag"),
    meter: $(".av-meter-fill"), contract: $(".av-contract"),
  };
  let cap = null;   // everything owned by one running capture, released by stop()

  const WORKLET = `
    class AvCapture extends AudioWorkletProcessor {
      constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; this.first = null; }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch) {
          if (this.n + ch.length > this.buf.length) this.flush();
          if (this.n === 0) this.first = currentFrame;
          this.buf.set(ch, this.n); this.n += ch.length;
          if (this.n >= 1600) this.flush();
        }
        return true;
      }
      flush() {
        if (this.n === 0) return;
        this.port.postMessage({ frame: this.first, samples: this.buf.slice(0, this.n) });
        this.n = 0;
      }
    }
    registerProcessor("av-capture", AvCapture);`;

  function setState(text, kind) {
    ui.state.textContent = text;
    ui.state.dataset.kind = kind || "idle";
  }
  function setMessage(text, kind) {
    ui.msg.textContent = text || "";
    ui.msg.dataset.kind = kind || "info";
    ui.msg.hidden = !text;
  }
  function renderGrid(diag) {
    const segs = (diag && diag.segments) || [];
    const cells = ui.grid.children;
    for (let i = 0; i < cells.length; i++) {
      const s = segs[i];
      cells[i].dataset.status = s ? s.status : "pending";
      cells[i].title = s ? ("second " + s.second + ": " + s.status.replace(/_/g, " ") +
        (s.frame_offset_ms === null ? "" : " (frame offset " + s.frame_offset_ms + " ms)")) : "waiting";
    }
  }
  function renderDiag(d) {
    const c = d.counters || {}, cl = d.client || {};
    const rows = [
      ["Buffer", d.segments_valid + " / 10 valid seconds" + (d.ready ? " (ready)" : "")],
      ["Timeline", (d.timeline_s || 0).toFixed(1) + " s (audio " + (d.audio_s || 0).toFixed(1) + " s, video " + (d.video_s || 0).toFixed(1) + " s)"],
      ["Frame delivery lag", d.frame_lag_ms === null ? "n/a" : Math.round(d.frame_lag_ms) + " ms"],
      ["Worst frame offset", d.max_frame_offset_ms === null ? "n/a" : d.max_frame_offset_ms + " ms"],
      ["Audio", (d.capture_sample_rate / 1000) + " kHz, gaps " + (c.audio_gaps || 0) + (cl.audio_latency_ms !== undefined ? ", latency " + Math.round(cl.audio_latency_ms) + " ms" : "")],
      ["Video", (cl.video_fps ? Math.round(cl.video_fps) + " fps, " : "") + "frames " + (c.frames || 0) + ", off-grid " + (c.off_grid_frames || 0) + ", late " + (c.late_frames || 0) + (cl.dropped_video_frames ? ", browser-dropped " + cl.dropped_video_frames : "")],
    ];
    ui.diag.innerHTML = "";
    for (const r of rows) {
      const dt = document.createElement("dt"); dt.textContent = r[0];
      const dd = document.createElement("dd"); dd.textContent = r[1];
      ui.diag.append(dt, dd);
    }
  }

  async function start() {
    if (cap) return;
    setMessage(""); ui.contract.textContent = ""; renderGrid(null);
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMessage(AVTiming.permissionMessage({ name: "INSECURE_CONTEXT" }), "error"); return;
    }
    ui.start.disabled = true;
    setState("Requesting camera and microphone…", "busy");
    const c = { stream: null, ctx: null, node: null, session: null, timer: null, rvfc: null, raf: null,
                pushing: false, stopped: false, clock: null, audio: AVTiming.makeAudioAccumulator(),
                picker: AVTiming.makeFramePicker(), canvases: new Map(), frames: [], pendingAudio: [],
                client: {}, fpsCount: 0, fpsSince: performance.now() };
    cap = c;
    try {
      c.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      });
      if (!c.stream.getAudioTracks().length) throw { name: "NO_AUDIO_TRACK" };
      if (!c.stream.getVideoTracks().length) throw { name: "NO_VIDEO_TRACK" };
      for (const t of c.stream.getTracks()) {
        t.addEventListener("ended", function () { c.client.track_ended = true; stop("A camera or microphone was disconnected. Capture stopped.", "error"); });
        t.addEventListener("mute", function () { c.client[t.kind === "audio" ? "audio_muted" : "video_muted"] = true; });
        t.addEventListener("unmute", function () { c.client[t.kind === "audio" ? "audio_muted" : "video_muted"] = false; });
      }
      ui.video.srcObject = c.stream;
      await ui.video.play();
      const vs = c.stream.getVideoTracks()[0].getSettings();
      c.client.camera_width = vs.width || 0; c.client.camera_height = vs.height || 0;

      // Ask for 16 kHz directly; fall back to the device rate (the server resamples the window).
      try { c.ctx = new AudioContext({ sampleRate: 16000 }); c.src = c.ctx.createMediaStreamSource(c.stream); }
      catch (e) { if (c.ctx) c.ctx.close(); c.ctx = new AudioContext(); c.src = c.ctx.createMediaStreamSource(c.stream); }
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await c.ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      c.node = new AudioWorkletNode(c.ctx, "av-capture", { numberOfInputs: 1, numberOfOutputs: 1 });
      c.sink = c.ctx.createGain(); c.sink.gain.value = 0;   // silent sink keeps the node in the render graph
      c.src.connect(c.node); c.node.connect(c.sink); c.sink.connect(c.ctx.destination);
      if (c.ctx.state !== "running") await c.ctx.resume();

      const started = await server.live_start({ sample_rate: c.ctx.sampleRate });
      if (c.stopped) return;
      if (!started || started.status !== "STARTED") throw { name: (started && started.error_code) || "LIVE_START_FAILED" };
      c.session = started.session;

      c.node.port.onmessage = function (ev) {
        const m = ev.data;
        if (!c.clock) {   // the first audio sample defines the timeline origin
          c.clock = AVTiming.makeClock(c.ctx.getOutputTimestamp(), m.frame, c.ctx.sampleRate);
          setState("Capturing: filling the 10-second buffer…", "busy");
          watchFrames();
        }
        const idx = c.clock.frameToSample(m.frame);
        if (!c.audio.add(idx, m.samples)) { c.pendingAudio.push(c.audio.take()); c.audio.add(idx, m.samples); }
        let level = 0; for (let i = 0; i < m.samples.length; i++) level = Math.max(level, Math.abs(m.samples[i]));
        ui.meter.style.width = Math.min(100, Math.round(level * 140)) + "%";
      };
      c.timer = setInterval(push, PUSH_INTERVAL_MS);
      ui.stop.disabled = false;
      setState("Capturing: waiting for audio…", "busy");
    } catch (err) {
      const code = err && err.name;
      const known = ["NotAllowedError", "SecurityError", "NotFoundError", "OverconstrainedError",
                     "NotReadableError", "AbortError", "NO_AUDIO_TRACK", "NO_VIDEO_TRACK"];
      stop(known.indexOf(code) >= 0 ? AVTiming.permissionMessage(err) : "Live capture could not start (" + (code || "error") + ").", "error");
    }
  }

  function watchFrames() {
    const c = cap, v = ui.video;
    const onFrame = function (perfMs) {
      if (c.stopped) return;
      const t = c.clock.perfToTimeline(perfMs);
      c.fpsCount++;
      const k = c.picker.offer(t);
      if (k !== null) {   // copy the pixels now: this is the best frame so far for second k
        let cv = c.canvases.get(k);
        if (!cv) { cv = document.createElement("canvas"); c.canvases.set(k, cv); }
        const scale = Math.min(1, MAX_FRAME_SIDE / Math.max(v.videoWidth, v.videoHeight));
        cv.width = Math.round(v.videoWidth * scale); cv.height = Math.round(v.videoHeight * scale);
        cv.getContext("2d").drawImage(v, 0, 0, cv.width, cv.height);
        cv.dataset.t = String(t);
      }
      for (const due of c.picker.due(t)) {
        const cv = c.canvases.get(due); c.canvases.delete(due);
        if (cv) c.frames.push({ t: Number(cv.dataset.t), image: cv.toDataURL("image/jpeg", JPEG_QUALITY).split(",")[1] });
      }
    };
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      const loop = function (now, meta) {
        // captureTime is the camera's own timestamp (performance.now domain) where the browser has it
        onFrame(meta && Number.isFinite(meta.captureTime) ? meta.captureTime : now);
        c.rvfc = v.requestVideoFrameCallback(loop);
      };
      c.rvfc = v.requestVideoFrameCallback(loop);
    } else {
      const loop = function () { onFrame(performance.now()); c.raf = requestAnimationFrame(loop); };
      c.raf = requestAnimationFrame(loop);
    }
  }

  async function push() {
    const c = cap;
    if (!c || c.stopped || c.pushing || !c.session || !c.clock) return;
    const chunks = c.pendingAudio.splice(0);
    const last = c.audio.take(); if (last) chunks.push(last);
    const frames = c.frames.splice(0, 4);
    if (!chunks.length && !frames.length) return;
    const now = performance.now();
    c.client.video_fps = 1000 * c.fpsCount / Math.max(1, now - c.fpsSince);
    c.fpsCount = 0; c.fpsSince = now;
    const q = ui.video.getVideoPlaybackQuality && ui.video.getVideoPlaybackQuality();
    if (q) c.client.dropped_video_frames = q.droppedVideoFrames;
    c.client.audio_latency_ms = 1000 * ((c.ctx.baseLatency || 0) + (c.ctx.outputLatency || 0));
    c.pushing = true;
    try {
      const d = await server.live_push({
        session: c.session,
        audio: chunks.map(function (ch) { return { start: ch.start, pcm: AVTiming.bytesToBase64(AVTiming.floatToPcm16(ch.samples)) }; }),
        frames: frames, client: c.client,
      });
      if (c.stopped) return;
      if (!d || d.status !== "CAPTURING") {
        stop("Capture stopped by the server (" + ((d && d.error_code) || "no response") + ").", "error"); return;
      }
      renderGrid(d); renderDiag(d);
      ui.check.disabled = !d.ready;
      if (d.ready) setState("Buffer ready: 10 aligned seconds", "ok");
      else setState("Capturing: " + d.segments_valid + " / 10 valid seconds", "busy");
      if (c.client.audio_muted) setMessage("The microphone is muted, so those seconds are marked as muted audio.", "warn");
      else if (document.hidden) setMessage("The tab is in the background, so the camera may pause and seconds will be dropped.", "warn");
      else setMessage("");
    } catch (e) {
      if (!c.stopped) stop("Lost connection to the server. Capture stopped.", "error");
    } finally { c.pushing = false; }
  }

  async function check() {
    if (!cap || !cap.session) return;
    const r = await server.live_check({ session: cap.session });
    ui.contract.textContent = r && r.status === "CONTRACT_VALID"
      ? "Window passes the backend RawMediaInput contract: audio " + r.waveform_shape.join("×") + ", frames " + r.frames_shape.join("×") + ". No prediction is produced until the model is integrated."
      : "Window not accepted (" + ((r && r.error_code) || "error") + ").";
  }

  // Releases devices, audio graph, timers and the server session. Safe to call more than once.
  function stop(message, kind) {
    const c = cap; cap = null;
    if (c) {
      c.stopped = true;
      clearInterval(c.timer);
      if (c.rvfc !== null && ui.video.cancelVideoFrameCallback) ui.video.cancelVideoFrameCallback(c.rvfc);
      if (c.raf !== null) cancelAnimationFrame(c.raf);
      if (c.node) { c.node.port.onmessage = null; c.node.disconnect(); }
      if (c.sink) c.sink.disconnect();
      if (c.src) c.src.disconnect();
      if (c.ctx && c.ctx.state !== "closed") c.ctx.close();
      if (c.stream) for (const t of c.stream.getTracks()) t.stop();
      c.canvases.clear(); c.frames.length = 0;
      if (c.session) server.live_stop({ session: c.session }).catch(function () {});
    }
    ui.video.srcObject = null;
    ui.meter.style.width = "0%";
    // The server discards the buffer on stop, so the display must not keep showing it as ready.
    renderGrid(null); ui.diag.innerHTML = ""; ui.contract.textContent = "";
    ui.start.disabled = false; ui.stop.disabled = true; ui.check.disabled = true;
    setState(message && kind === "error" ? "Stopped" : "Idle", message && kind === "error" ? "error" : "idle");
    setMessage(message || "", kind);
  }

  ui.start.addEventListener("click", start);
  ui.stop.addEventListener("click", function () { stop(); });
  ui.check.addEventListener("click", check);
  window.addEventListener("pagehide", function () { stop(); });
  renderGrid(null);
  stop();
})();
