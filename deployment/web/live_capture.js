// AValidate live capture controller. Runs as gr.HTML js_on_load (variables: element, server), after
// live_timing.js (AVTiming). Captures webcam + microphone from ONE getUserMedia stream, stamps both on
// the shared timeline, and pushes audio samples and one frame per second to the server buffer
// (deployment/live). It shows readiness and sync diagnostics only: it never shows a prediction.
(function () {
  const PUSH_INTERVAL_MS = 250;
  const CLOCK_TRIES = 40;   // audio blocks tolerated while the browser audio clock starts, before giving up
  const MAX_FRAME_SIDE = 640;
  const JPEG_QUALITY = 0.9;
  const $ = function (sel) { return element.querySelector(sel); };
  const ui = {
    video: $(".av-preview"), start: $(".av-start"), record: $(".av-record"), stop: $(".av-stop"), check: $(".av-check"),
    state: $(".av-state"), msg: $(".av-msg"), grid: $(".av-grid"), diag: $(".av-diag"),
    meter: $(".av-meter-fill"), contract: $(".av-contract"),
    timer: $(".av-timer"), progress: $(".av-progress"), reset: $(".av-reset"),
  };
  let cap = null;   // everything owned by one running capture, released by stop()
  let inFlight = 0;
  let cleanupTimer = null;
  const CLEANUP_MESSAGE = "Releasing capture resources...";
  const WAITING_MESSAGE = "Waiting for outstanding requests to finish; reload this tab if the connection stays unavailable.";
  const bounded = AVTiming.withTimeout;
  // Gradio's component promises do not expose AbortSignal. Bound both the
  // logical wait and the number of actual outstanding calls across restarts.
  // A stalled call blocks another capture until it settles (or the tab reloads).
  function rpc(method, payload, onSettled) {
    if (inFlight >= 3) return Promise.reject(new Error("LIVE_REQUEST_LIMIT"));
    inFlight++;
    if (!cap) { ui.start.disabled = true; ui.record.disabled = true; }
    let promise;
    try { promise = Promise.resolve(server[method](payload)); }
    catch (e) { promise = Promise.reject(e); }
    function release() {
      inFlight--;
      if (!cap) {
        ui.start.disabled = inFlight > 0; ui.record.disabled = inFlight > 0;
        if (inFlight === 0) {
          clearTimeout(cleanupTimer); cleanupTimer = null;
          for (const suffix of [CLEANUP_MESSAGE, WAITING_MESSAGE]) {
            if (ui.msg.textContent.endsWith(suffix)) {
              setMessage(ui.msg.textContent.slice(0, -suffix.length).trim(), ui.msg.dataset.kind);
            }
          }
        }
      }
    }
    promise.then(function (r) {
      // A late start may create another cleanup RPC. Keep the controls disabled
      // until that request also settles, with no briefly enabled restart window.
      try { if (onSettled) onSettled(r); } finally { release(); }
    }, release);
    return bounded(promise);
  }
  function credentials(c) { return { session: c.session, tab_key: c.tabKey }; }
  function releaseStream(stream) { if (stream) for (const t of stream.getTracks()) t.stop(); }
  function capacity(c, extraSamples, extraFrames) {
    const samples = c.audio.length + c.pendingAudio.reduce(function (n, p) { return n + p.samples.length; }, 0);
    if (AVTiming.backlogAllowed(c.ctx.sampleRate, samples + (extraSamples || 0),
                               c.pendingAudio.length + 1, c.frames.length + (extraFrames || 0))) return true;
    stop("The connection cannot keep up with capture. Recording stopped; try again.", "error");
    return false;
  }

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

  // mode "record": stop by itself once ten aligned seconds are captured and checked; "rolling": run until Stop
  async function start(mode) {
    if (cap || inFlight) return;
    clearTimeout(cleanupTimer); cleanupTimer = null;
    setMessage(""); ui.contract.textContent = ""; ui.diag.innerHTML = ""; renderGrid(null);
    if (document.hidden) {
      setMessage("Return to this tab before starting camera and microphone capture.", "warn"); return;
    }
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMessage(AVTiming.permissionMessage({ name: "INSECURE_CONTEXT" }), "error"); return;
    }
    ui.start.disabled = true; ui.record.disabled = true;
    setState("Requesting camera and microphone…", "busy");
    const c = { mode: mode, stream: null, ctx: null, node: null, session: null, tabKey: null, timer: null, rvfc: null, raf: null,
                pushing: false, stopped: false, clock: null, audio: AVTiming.makeAudioAccumulator(),
                picker: AVTiming.makeFramePicker(), canvases: new Map(), frames: [], pendingAudio: [],
                client: {}, clockTries: 0, fpsCount: 0, fpsSince: performance.now() };
    cap = c;
    ui.stop.disabled = false;
    c.deadline = setTimeout(function () { if (cap === c) stop("Capture time limit reached. Start a new recording.", "warn"); },
      mode === "record" ? 60000 : 30 * 60 * 1000);
    try {
      const permission = navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      });
      // A late permission grant must not reopen devices after Stop or a timeout.
      permission.then(function (stream) { if (c.stopped) releaseStream(stream); }, function () {});
      c.stream = await bounded(permission, 30000);
      if (c.stopped) { releaseStream(c.stream); return; }
      if (!c.stream.getAudioTracks().length) throw { name: "NO_AUDIO_TRACK" };
      if (!c.stream.getVideoTracks().length) throw { name: "NO_VIDEO_TRACK" };
      for (const t of c.stream.getTracks()) {
        t.addEventListener("ended", function () {
          if (c.stopped || cap !== c) return;
          c.client.track_ended = true; stop("A camera or microphone was disconnected. Capture stopped.", "error");
        });
        t.addEventListener("mute", function () { c.client[t.kind === "audio" ? "audio_muted" : "video_muted"] = true; });
        t.addEventListener("unmute", function () { c.client[t.kind === "audio" ? "audio_muted" : "video_muted"] = false; });
      }
      ui.video.srcObject = c.stream;
      await bounded(ui.video.play());
      if (c.stopped) return;
      const vs = c.stream.getVideoTracks()[0].getSettings();
      c.client.camera_width = vs.width || 0; c.client.camera_height = vs.height || 0;

      // Ask for 16 kHz directly; fall back to the device rate (the server resamples the window).
      try { c.ctx = new AudioContext({ sampleRate: 16000 }); c.src = c.ctx.createMediaStreamSource(c.stream); }
      catch (e) { if (c.ctx) c.ctx.close(); c.ctx = new AudioContext(); c.src = c.ctx.createMediaStreamSource(c.stream); }
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      try { await bounded(c.ctx.audioWorklet.addModule(url)); }
      finally { URL.revokeObjectURL(url); }
      if (c.stopped) return;
      c.node = new AudioWorkletNode(c.ctx, "av-capture", { numberOfInputs: 1, numberOfOutputs: 1 });
      c.sink = c.ctx.createGain(); c.sink.gain.value = 0;   // silent sink keeps the node in the render graph
      c.src.connect(c.node); c.node.connect(c.sink); c.sink.connect(c.ctx.destination);
      if (c.ctx.state !== "running") await bounded(c.ctx.resume());
      if (c.stopped) return;
      if (c.ctx.addEventListener) c.ctx.addEventListener("statechange", function () {
        if (!c.stopped && cap === c && c.ctx.state !== "running") {
          stop("The audio clock stopped. Start a new recording to synchronize again.", "error");
        }
      });

      const starting = rpc("live_start", { sample_rate: c.ctx.sampleRate }, function (r) {
        if (c.stopped && r && r.status === "STARTED") {
          rpc("live_stop", { session: r.session, tab_key: r.tab_key }).catch(function () {});
        }
      });
      const started = await starting;
      if (c.stopped) return;
      if (!started || started.status !== "STARTED") throw { name: (started && started.error_code) || "LIVE_START_FAILED" };
      c.session = started.session;
      c.tabKey = started.tab_key;

      c.node.port.onmessage = function (ev) {
        if (c.stopped) return;
        const m = ev.data;
        if (!capacity(c, m.samples.length, 0)) return;
        if (!c.clock) {   // the first audio sample defines the timeline origin
          try { c.clock = AVTiming.makeClock(c.ctx.getOutputTimestamp(), m.frame, c.ctx.sampleRate); }
          catch (e) {   // clock not running yet (or not supported): drop this block and retry, then give up
            if (++c.clockTries > CLOCK_TRIES) stop(AVTiming.permissionMessage({ name: "NO_AUDIO_CLOCK" }), "error");
            return;
          }
          setState(c.mode === "record" ? "Recording: capturing 10 aligned seconds…" : "Capturing: filling the 10-second buffer…", "busy");
          watchFrames();
        }
        const idx = c.clock.frameToSample(m.frame);
        const elapsed = Math.max(0, (idx + m.samples.length) / c.ctx.sampleRate);
        if (ui.timer) ui.timer.textContent = (c.mode === "record" ? Math.min(10, elapsed) : elapsed).toFixed(1) + " s / 10 s";
        if (ui.progress) ui.progress.value = Math.min(10, elapsed);
        if (!c.audio.add(idx, m.samples)) {
          c.pendingAudio.push(c.audio.take());
          if (!capacity(c, m.samples.length, 0)) return;
          c.audio.add(idx, m.samples);
        }
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
      if (!c.stopped && cap === c) stop(known.indexOf(code) >= 0 ? AVTiming.permissionMessage(err) :
        "Live capture could not start (" + (code || "error") + ").", "error");
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
        if (cv) {
          if (!capacity(c, 0, 1)) return;
          c.frames.push({ t: Number(cv.dataset.t), image: cv.toDataURL("image/jpeg", JPEG_QUALITY).split(",")[1] });
        }
      }
    };
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      const loop = function (now, meta) {
        // captureTime is the camera's own timestamp (performance.now domain) where the browser has it
        onFrame(meta && Number.isFinite(meta.captureTime) ? meta.captureTime : now);
        if (!c.stopped) c.rvfc = v.requestVideoFrameCallback(loop);
      };
      c.rvfc = v.requestVideoFrameCallback(loop);
    } else {
      const loop = function () { onFrame(performance.now()); if (!c.stopped) c.raf = requestAnimationFrame(loop); };
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
      const d = await rpc("live_push", {
        ...credentials(c),
        audio: chunks.map(function (ch) { return { start: ch.start, pcm: AVTiming.bytesToBase64(AVTiming.floatToPcm16(ch.samples)) }; }),
        frames: frames, client: c.client,
      });
      if (c.stopped) return;
      if (!d || d.status !== "CAPTURING") {
        stop("Capture stopped by the server (" + ((d && d.error_code) || "no response") + ").", "error"); return;
      }
      renderGrid(d); renderDiag(d);
      if (c.mode === "record" && d.ready) { await finishRecording(c); return; }
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

  function describeCheck(r) {
    if (!r || r.status !== "CONTRACT_VALID") return "Window not accepted (" + ((r && r.error_code) || "error") + ").";
    const t = r.timing || {};
    return "Window passes the backend RawMediaInput contract: audio " + r.waveform_shape.join("×") +
      ", frames " + r.frames_shape.join("×") + ". Camera frames were captured within " + t.max_abs_frame_offset_ms +
      " ms of their second (limit " + t.frame_tolerance_ms + " ms)" +
      (t.clipped_samples ? "; " + t.clipped_samples + " audio samples were clipped" : "") +
      ". Live capture is not yet verified against the encoders' training input. " +
      "Model verification is unavailable until independent live-input and release approval; predictions remain null.";
  }

  async function check() {
    if (!cap || !cap.session) return null;
    const c = cap;
    if (c.checking) return null;
    c.checking = true; ui.check.disabled = true;
    try {
      const r = await rpc("live_check", credentials(c));
      if (!c.stopped) ui.contract.textContent = describeCheck(r);
      return r;
    } catch (e) {
      if (!c.stopped) stop("The window check timed out or lost its connection. Capture stopped.", "error");
      return null;
    } finally { c.checking = false; }
  }

  // Record mode: check the first complete window, then release the devices but keep the result on screen.
  async function finishRecording(c) {
    let r = null;
    try { r = await check(); } catch (e) { r = null; }
    if (c.stopped) return;
    const ok = r && r.status === "CONTRACT_VALID";
    stop(ok ? "" : "The recording could not be checked. Press Record to try again.", ok ? "" : "error", true);
    setState(ok ? "Recorded: 10 aligned seconds" : "Stopped", ok ? "ok" : "error");
  }

  // Releases devices, audio graph, timers and the server session. Safe to call more than once.
  // keepResult leaves the last grid, diagnostics and check on screen (a finished recording).
  function stop(message, kind, keepResult) {
    const c = cap; cap = null;
    if (c) {
      c.stopped = true;
      clearInterval(c.timer);
      clearTimeout(c.deadline);
      if (c.rvfc !== null && ui.video.cancelVideoFrameCallback) ui.video.cancelVideoFrameCallback(c.rvfc);
      if (c.raf !== null) cancelAnimationFrame(c.raf);
      if (c.node) { c.node.port.onmessage = null; c.node.disconnect(); }
      if (c.sink) c.sink.disconnect();
      if (c.src) c.src.disconnect();
      if (c.ctx && c.ctx.state !== "closed") c.ctx.close();
      releaseStream(c.stream);
      c.canvases.clear(); c.frames.length = 0;
      c.audio.clear(); c.pendingAudio.length = 0;
      if (c.session) rpc("live_stop", credentials(c)).catch(function () {});
    }
    ui.video.srcObject = null;
    ui.meter.style.width = "0%";
    // The server discards the buffer on stop, so the display must not keep showing it as ready.
    if (!keepResult) {
      renderGrid(null); ui.diag.innerHTML = ""; ui.contract.textContent = "";
      if (ui.timer) ui.timer.textContent = "0.0 s / 10 s";
      if (ui.progress) ui.progress.value = 0;
    }
    ui.start.disabled = inFlight > 0; ui.record.disabled = inFlight > 0; ui.stop.disabled = true; ui.check.disabled = true;
    setState(message && kind === "error" ? "Stopped" : "Idle", message && kind === "error" ? "error" : "idle");
    clearTimeout(cleanupTimer); cleanupTimer = null;
    setMessage((message || "") + (inFlight ? " " + CLEANUP_MESSAGE : ""), kind);
    if (inFlight) cleanupTimer = setTimeout(function () {
      if (!cap && inFlight) {
        setMessage((message || "") + " " + WAITING_MESSAGE, kind || "warn");
      }
    }, 1000);
  }

  ui.start.addEventListener("click", function () { start("rolling"); });
  ui.record.addEventListener("click", function () { start("record"); });
  ui.stop.addEventListener("click", function () { stop(); });
  if (ui.reset) ui.reset.addEventListener("click", function () { stop(); });
  ui.check.addEventListener("click", check);
  window.addEventListener("pagehide", function () { stop(); });
  document.addEventListener("visibilitychange", function () {
    if (document.hidden && cap) stop("Capture stopped because the tab is in the background. Start a new recording on return.", "warn");
  });
  renderGrid(null);
  stop();
})();
