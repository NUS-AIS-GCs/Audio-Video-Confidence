// AValidate live capture controller. Runs as gr.HTML js_on_load (variables: element, server), after
// live_timing.js (AVTiming). Captures webcam + microphone from ONE getUserMedia stream, stamps both on
// the shared timeline, and pushes audio samples and one frame per second to the server buffer
// (deployment/live). Public mode stays diagnostics-only; authenticated research
// mode displays signed results produced by the separately isolated worker.
(function () {
  const PUSH_INTERVAL_MS = 250;
  const MAX_FRAME_SIDE = 640;
  const JPEG_QUALITY = 0.9;
  const $ = function (sel) { return element.querySelector(sel); };
  const ui = {
    video: $(".av-preview"), start: $(".av-start"), record: $(".av-record"), stop: $(".av-stop"), check: $(".av-check"),
    state: $(".av-state"), msg: $(".av-msg"), grid: $(".av-grid"), diag: $(".av-diag"),
    meter: $(".av-meter-fill"), contract: $(".av-contract"),
    micStatus: $(".av-mic-status"), micDiag: $(".av-mic-diag"),
    timer: $(".av-timer"), progress: $(".av-progress"), reset: $(".av-reset"),
    consent: $(".av-research-consent"), researchResult: $(".av-research-result"),
  };
  const research = !!ui.consent;
  let researchJob = null;
  if (research) ui.start.hidden = true; // research exports only the first completed window
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
      constructor(options) {
        super(); this.buf = new Float32Array(2048); this.n = 0; this.first = null;
        this.total = 0; this.limit = options.processorOptions.limit || Infinity;
        this.nextFrame = null; this.startupEmpty = 0; this.failed = false;
      }
      process(inputs) {
        if (this.failed) return false;
        const ch = inputs[0] && inputs[0][0];
        if (this.nextFrame !== null && currentFrame !== this.nextFrame)
          return this.fail("AUDIO_CALLBACK_DISCONTINUITY");
        if (!ch || !ch.length) {
          if (this.nextFrame === null) { this.startupEmpty++; return true; }
          return this.fail("AUDIO_INPUT_MISSING");
        }
        if (ch) {
          this.nextFrame = currentFrame + ch.length;
          const block = ch.subarray(0, Math.max(0, this.limit - this.total));
          if (this.n + block.length > this.buf.length) this.flush();
          if (this.n === 0) this.first = currentFrame;
          this.buf.set(block, this.n); this.n += block.length; this.total += block.length;
          const done = this.total >= this.limit;
          // Deliver the first quantum immediately so frame zero is not missed
          // while waiting for the normal 100 ms transport block.
          if (this.total === block.length || this.n >= 1600 || done) this.flush(done);
          if (done) return false;
        }
        return true;
      }
      fail(code) {
        // Never join input from opposite sides of a missing quantum, and never
        // replace missing browser input with the zero-initialized buffer tail.
        this.flush(false);
        this.port.postMessage({capture_error:code, expected_frame:this.nextFrame, observed_frame:currentFrame});
        this.failed = true;
        return false;
      }
      flush(done) {
        if (this.n === 0) return;
        this.port.postMessage({ frame: this.first, samples: this.buf.slice(0, this.n), done: !!done,
          written_samples:this.n, startup_empty_input_quanta:this.startupEmpty });
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
  // Local, bounded metadata only. Never include PCM, stream/track objects,
  // device labels/IDs, request payloads, capabilities or credentials here.
  function micEvent(c, event) {
    if (!c.mic) return;
    const t = c.stream && c.stream.getAudioTracks()[0];
    const now = performance.now();
    c.mic.events.push({event: event, elapsed_ms: Math.round(now - c.began),
      last_delivered_sample_index:c.mic.nextSample, timeline_s:c.clock ? c.clock.perfToTimeline(now) : null,
      track:t ? {enabled:t.enabled, muted:t.muted, ready_state:t.readyState} : null,
      context_state:c.ctx ? c.ctx.state : null});
    if (c.mic.events.length > 16) c.mic.events.shift();
  }
  function renderMicrophone(c) {
    if (!c.mic) return;
    const m = c.mic, track = c.stream && c.stream.getAudioTracks()[0];
    if (!c.devicesReleased) {
      if (track) m.track = {enabled: track.enabled === true, muted: track.muted === true, ready_state: track.readyState};
      if (c.ctx) m.context_state = c.ctx.state;
    }
    const view = {
      category: m.category, phase: m.phase, track: m.track,
      // Preserve why capture failed, while making cleanup state explicit.
      capture_context_state: m.context_state,
      current_context_state: c.ctx ? c.ctx.state : null,
      current_track_ready_state: track ? track.readyState : null,
      devices_released: !!c.devicesReleased,
      sample_rate: c.ctx ? c.ctx.sampleRate : null,
      received_samples: m.samples, target_samples: c.ctx ? c.ctx.sampleRate * 10 : null,
      web_audio_rms: m.samples ? Math.sqrt(m.sumSquares / m.samples) : 0,
      latest_block_rms: m.latestRms, web_audio_nonzero_samples: m.floatNonzero,
      pcm16_nonzero_samples: m.pcmNonzero, server_accepted_audio_samples: m.accepted,
      zero_pcm_seconds_reported_by_server: m.zeroSeconds,
      zero_pcm_seconds_at_worklet_input: m.seconds.map((s,i) => s.samples === (c.ctx && c.ctx.sampleRate) && s.pcmNonzero === 0 ? i : null).filter(i => i !== null),
      continuity: {next_sample_index:m.nextSample, callbacks:m.callbacks,
        discontinuities:m.discontinuities, missing_input_errors:m.missingInput,
        last_expected_frame:m.lastExpectedFrame, last_observed_frame:m.lastObservedFrame,
        startup_empty_input_quanta:m.startupEmpty, all_emitted_samples_explicitly_written:m.allWritten,
        acquisition_complete:m.acquisitionComplete, finalized_before_acquisition_complete:m.prematureSeal},
      shared_clock:m.clockAnchor, sampled_video_frames:m.videoTimes,
      audio_observation_point:"Browser AudioWorklet input 0/channel 0, after the existing mono mix; hardware versus browser/driver silence is not distinguishable here.",
      server_seconds:m.serverSeconds, device_change_events:m.deviceChanges,
      browser_server_comparison:m.seconds.map(function (s,i) {
        const remote = m.serverSeconds.find(r => r.second === i);
        const complete = !!remote && !!c.ctx && s.samples === c.ctx.sampleRate && remote.complete;
        return {second:i, complete:complete,
          nonzero_counts_agree:complete ? s.pcmNonzero === remote.nonzero_samples : null,
          received_counts_agree:complete ? s.samples === remote.received_samples : null,
          rms_difference:complete ? Math.abs(Math.sqrt(s.pcmSquares/s.samples)-remote.rms) : null,
          peak_abs_difference:complete ? Math.abs(s.pcmPeak-remote.peak_abs) : null,
          sample_clock_bounds_agree:complete ? remote.start_sample === i*c.ctx.sampleRate &&
            remote.end_sample_exclusive === (i+1)*c.ctx.sampleRate : null};
      }),
      mute_events: m.mutes, unmute_events: m.unmutes, events: m.events,
      seconds: m.seconds.map(function (s, i) { return {second: i,
        start_sample:i*(c.ctx ? c.ctx.sampleRate : 0), end_sample_exclusive:(i+1)*(c.ctx ? c.ctx.sampleRate : 0),
        samples: s.samples, web_audio_nonzero_samples:s.floatNonzero,
        input_classification:!s.samples ? "no_samples_received" :
          !s.floatNonzero ? "zero_values_at_browser_audio_input" :
          !s.pcmNonzero ? "nonzero_input_quantized_to_zero_pcm16" : "nonzero_pcm_received",
        web_audio_rms: s.samples ? Math.sqrt(s.sumSquares / s.samples) : 0,
        peak_abs:s.peak, pcm16_nonzero_samples: s.pcmNonzero,
        pcm16_rms:s.samples ? Math.sqrt(s.pcmSquares / s.samples) : 0, pcm16_peak_abs:s.pcmPeak,
        first_observed_state:s.firstState, last_observed_state:s.lastState}; }),
    };
    if (ui.micDiag) ui.micDiag.textContent = JSON.stringify(view, null, 2);
    if (ui.micStatus) ui.micStatus.textContent = "Microphone: " + m.category.replace(/_/g, " ") +
      " · " + m.samples + (c.ctx ? " / " + c.ctx.sampleRate * 10 : "") + " samples" +
      " · RMS " + view.web_audio_rms.toExponential(2) +
      (m.samples ? " · " + m.pcmNonzero + " nonzero PCM16 samples" : "");
  }
  function inspectMicrophone(c) {
    const t = c.stream && c.stream.getAudioTracks()[0];
    renderMicrophone(c);
    if (!t || t.readyState === "ended") throw {name:"NO_AUDIO_TRACK"};
    if (t.enabled === false) throw {name:"MICROPHONE_DISABLED"};
    if (t.muted === true) throw {name:"MICROPHONE_TRACK_MUTED"};
  }
  function measureAudio(c, start, samples) {
    const m = c.mic, bytes = AVTiming.floatToPcm16(samples), pcm = new DataView(bytes.buffer);
    const t = c.stream.getAudioTracks()[0], state = {enabled:t.enabled, muted:t.muted,
      ready_state:t.readyState, context_state:c.ctx.state};
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      const square = samples[i] * samples[i], pcmValue = pcm.getInt16(i * 2, true)/32768, nonzero = pcmValue !== 0;
      sum += square; m.floatNonzero += samples[i] !== 0 ? 1 : 0;
      m.pcmNonzero += nonzero ? 1 : 0;
      const k = Math.floor((start + i) / c.ctx.sampleRate);
      if (k >= 0 && k < 10) {
        const sec = m.seconds[k];
        if (!sec.samples) sec.firstState = state;
        sec.lastState = state; sec.samples++; sec.sumSquares += square;
        sec.peak = Math.max(sec.peak, Math.abs(samples[i]));
        sec.floatNonzero += samples[i] !== 0 ? 1 : 0;
        sec.pcmNonzero += nonzero ? 1 : 0; sec.pcmSquares += pcmValue*pcmValue;
        sec.pcmPeak = Math.max(sec.pcmPeak, Math.abs(pcmValue));
      }
    }
    m.samples += samples.length; m.sumSquares += sum; m.latestRms = Math.sqrt(sum / samples.length);
    m.category = m.pcmNonzero ? "nonzero_pcm_received" :
      m.floatNonzero ? "signal_below_pcm16_resolution" : "zero_pcm_received";
    renderMicrophone(c);
    // A logarithmic DISPLAY meter makes quiet nonzero input visible. No audio
    // gain, samples, silence checks or inference thresholds are changed.
    ui.meter.style.width = (m.latestRms > 0 ? Math.max(2, Math.min(100,
      100 * (1 + 20 * Math.log10(m.latestRms) / 100))) : 0) + "%";
  }
  function serverAudioDiagnostic(c, d) {
    if (!d) return;
    if (Number.isFinite(d.accepted_audio_samples)) c.mic.accepted = d.accepted_audio_samples;
    if (Array.isArray(d.audio_seconds)) c.mic.serverSeconds = d.audio_seconds.slice(0,10);
    const segments = (d.buffer && d.buffer.segments) || [];
    c.mic.zeroSeconds = segments.filter(s => s.status === "muted_audio").map(s => s.second).slice(0, 10);
    renderMicrophone(c);
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
    if (research && !ui.consent.checked) {
      setMessage("Consent is required before camera and microphone recording.", "warn"); return;
    }
    if (research && !("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
      setMessage("Research capture requires actual video-frame callback metadata.", "error"); return;
    }
    if (researchJob) { await cancelResearch(); if (inFlight) return; }
    if (ui.researchResult) ui.researchResult.textContent = "";
    if (ui.micDiag) ui.micDiag.textContent = "";
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
                client: {}, fpsCount: 0, fpsSince: performance.now(), began: performance.now(),
                mic: {phase:"requesting_permission", category:"not_started", track:null, context_state:null,
                  samples:0, sumSquares:0, latestRms:0, floatNonzero:0, pcmNonzero:0, accepted:0,
                  nextSample:0, callbacks:0, discontinuities:0, missingInput:0, startupEmpty:0,
                  lastExpectedFrame:null, lastObservedFrame:null, allWritten:true,
                  acquisitionComplete:false, prematureSeal:false, deviceChanges:0,
                  clockAnchor:null, videoTimes:[], serverSeconds:[],
                  zeroSeconds:[], mutes:0, unmutes:0, events:[],
                  seconds:Array.from({length:10}, () => ({samples:0, sumSquares:0, peak:0,
                    floatNonzero:0, pcmNonzero:0, pcmSquares:0, pcmPeak:0, firstState:null, lastState:null}))} };
    cap = c;
    renderMicrophone(c);
    ui.stop.disabled = false;
    c.deadline = setTimeout(function () { if (cap === c) stop("Capture time limit reached. Start a new recording.", "warn"); },
      mode === "record" ? 30000 : 30 * 60 * 1000);
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
      micEvent(c, "permission_granted");
      c.deviceChange = function () {
        if (cap !== c || c.stopped || c.acquired) return;
        c.mic.deviceChanges++; micEvent(c, "device_change"); renderMicrophone(c);
      };
      if (navigator.mediaDevices.addEventListener) navigator.mediaDevices.addEventListener("devicechange", c.deviceChange);
      for (const t of c.stream.getTracks()) {
        t.addEventListener("ended", function () {
          if (c.stopped || c.acquired || cap !== c) return;
          if (t.kind === "audio") { c.mic.category = "microphone_device_unavailable"; micEvent(c, "microphone_ended"); }
          c.client.track_ended = true; stop("A camera or microphone was disconnected. Capture stopped.", "error");
        });
        c.client[t.kind === "audio" ? "audio_muted" : "video_muted"] = !!t.muted;
        t.addEventListener("mute", function () {
          if (cap !== c || c.stopped || c.acquired) return;
          if (t.kind === "audio") { c.mic.mutes++; c.mic.category = "microphone_track_muted"; micEvent(c, "microphone_mute"); }
          stop("Unable to verify: the " + t.kind + " track became muted. Enable the device and record again.", "error");
        });
        t.addEventListener("unmute", function () {
          if (cap === c && !c.stopped && !c.acquired) {
            c.client[t.kind === "audio" ? "audio_muted" : "video_muted"] = false;
            if (t.kind === "audio") { c.mic.unmutes++; micEvent(c, "microphone_unmute"); renderMicrophone(c); }
          }
        });
      }
      ui.video.srcObject = c.stream;
      ui.video.muted = true; ui.video.autoplay = true;
      await bounded(ui.video.play());
      if (c.stopped) return;
      inspectMicrophone(c);
      const vs = c.stream.getVideoTracks()[0].getSettings();
      c.client.camera_width = vs.width || 0; c.client.camera_height = vs.height || 0;

      // Ask for 16 kHz directly; fall back to the device rate (the server resamples the window).
      try { c.ctx = new AudioContext({ sampleRate: 16000 }); c.src = c.ctx.createMediaStreamSource(c.stream); }
      catch (e) { if (c.ctx) c.ctx.close(); c.ctx = new AudioContext(); c.src = c.ctx.createMediaStreamSource(c.stream); }
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      try { await bounded(c.ctx.audioWorklet.addModule(url)); }
      finally { URL.revokeObjectURL(url); }
      if (c.stopped) return;
      c.node = new AudioWorkletNode(c.ctx, "av-capture", research ?
        { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: "explicit", channelInterpretation: "speakers", processorOptions: {limit: mode === "record" ? c.ctx.sampleRate * 10 : 0} } :
        { numberOfInputs: 1, numberOfOutputs: 1, processorOptions: {limit: mode === "record" ? c.ctx.sampleRate * 10 : 0} });
      c.sink = c.ctx.createGain(); c.sink.gain.value = 0;   // silent sink keeps the node in the render graph
      c.sink.connect(c.ctx.destination);
      if (c.ctx.state !== "running") {
        micEvent(c, "context_resume_requested");
        try { await bounded(c.ctx.resume()); }
        catch (_) { throw {name:"AUDIO_CONTEXT_NOT_RUNNING"}; }
      }
      if (c.stopped) return;
      if (c.ctx.state !== "running") throw {name:"AUDIO_CONTEXT_NOT_RUNNING"};
      micEvent(c, "context_running"); renderMicrophone(c);
      // Prime the output clock BEFORE recording. Dropping the first quantum
      // while waiting for this clock used to remove 128 real samples and move
      // the origin, leaving a short window and potentially losing frame zero.
      try {
        await bounded(new Promise(function (resolve, reject) {
          function ready() {
            if (c.stopped) { reject(new Error("CANCELLED")); return; }
            const a = c.ctx.getOutputTimestamp();
            if (a && a.performanceTime > 0) { resolve(); return; }
            setTimeout(ready, 10);
          }
          ready();
        }));
      } catch (_) { throw {name:"NO_AUDIO_CLOCK"}; }
      if (c.stopped) return;
      if (c.ctx.addEventListener) c.ctx.addEventListener("statechange", function () {
        if (!c.stopped && !c.acquired && cap === c) micEvent(c, "context_state_change");
        if (!c.stopped && !c.acquired && cap === c && c.ctx.state !== "running") {
          c.mic.category = "audio_context_not_running"; micEvent(c, "context_interrupted");
          stop("Unable to verify: the browser audio context is " + c.ctx.state + ". Keep the tab active, then Reset and Record again.", "error");
        }
      });

      const startPayload = { sample_rate: c.ctx.sampleRate };
      if (research) {
        const settings = c.stream.getAudioTracks()[0].getSettings();
        startPayload.research = {consent: true, context_rate:c.ctx.sampleRate,
          device_rate:settings.sampleRate || 0, device_channels:settings.channelCount || 0,
          echo_cancellation:settings.echoCancellation, noise_suppression:settings.noiseSuppression,
          auto_gain_control:settings.autoGainControl, mono_mix:"explicit-speakers-mono",
          browser_resampler:"AudioContext-implementation-defined"};
      }
      const starting = rpc("live_start", startPayload, function (r) {
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
        if (c.stopped || c.acquired || cap !== c) return;
        try { inspectMicrophone(c); }
        catch (e) { c.mic.category = e.name.toLowerCase(); stop(AVTiming.permissionMessage(e), "error"); return; }
        const m = ev.data;
        if (m.capture_error) {
          c.mic.category = "audio_input_continuity_failure";
          c.mic.discontinuities++; c.mic.missingInput += m.capture_error === "AUDIO_INPUT_MISSING" ? 1 : 0;
          c.mic.lastExpectedFrame = m.expected_frame; c.mic.lastObservedFrame = m.observed_frame;
          micEvent(c, m.capture_error);
          stop("Unable to verify: microphone input or audio callback timing was interrupted. No gap was filled. Select a working input device, keep this tab active and record again. (" + m.capture_error + ")", "error");
          return;
        }
        c.mic.callbacks++;
        c.mic.startupEmpty = m.startup_empty_input_quanta || 0;
        c.mic.allWritten = c.mic.allWritten && m.written_samples === m.samples.length;
        if (!capacity(c, m.samples.length, 0)) return;
        if (!c.clock) {   // the first audio sample defines the timeline origin
          try {
            const a = c.ctx.getOutputTimestamp();
            c.clock = AVTiming.makeClock(a, m.frame, c.ctx.sampleRate);
            c.mic.clockAnchor = {origin_frame:m.frame, sample_rate:c.ctx.sampleRate,
              context_time:a.contextTime, performance_time_ms:a.performanceTime};
          }
          catch (e) { stop(AVTiming.permissionMessage({ name: "NO_AUDIO_CLOCK" }), "error"); return; }
          setState(c.mode === "record" ? "Recording: capturing 10 aligned seconds…" : "Capturing: filling the 10-second buffer…", "busy");
          watchFrames();
          clearTimeout(c.deadline);
          c.deadline = setTimeout(function () {
            if (cap === c && !c.acquired) stop("Unable to verify: the media clock stalled. Keep this tab active and check the microphone, then record again.", "error");
          }, c.mode === "record" ? 11000 : 30 * 60 * 1000);
        }
        const idx = c.clock.frameToSample(m.frame);
        if (idx !== c.mic.nextSample) {
          c.mic.category = "audio_message_continuity_failure"; c.mic.discontinuities++;
          c.mic.lastExpectedFrame = c.clock.originFrame + c.mic.nextSample; c.mic.lastObservedFrame = m.frame;
          micEvent(c, "AUDIO_MESSAGE_DISCONTINUITY");
          stop("Unable to verify: audio sample timestamps contain a gap or overlap. No samples were filled or reindexed. Keep the tab active and record again.", "error"); return;
        }
        c.mic.nextSample = idx + m.samples.length;
        c.mic.phase = "recording";
        measureAudio(c, idx, m.samples);
        const elapsed = Math.max(0, (idx + m.samples.length) / c.ctx.sampleRate);
        if (ui.timer) ui.timer.textContent = (c.mode === "record" ? Math.min(10, elapsed) : elapsed).toFixed(1) + " s / 10 s";
        if (ui.progress) ui.progress.value = Math.min(10, elapsed);
        if (!c.audio.add(idx, m.samples)) {
          c.pendingAudio.push(c.audio.take());
          if (!capacity(c, m.samples.length, 0)) return;
          c.audio.add(idx, m.samples);
        }
        if (m.done && c.mode === "record") {
          c.mic.acquisitionComplete = c.mic.nextSample === c.ctx.sampleRate * 10;
          c.mic.phase = "acquisition_complete"; micEvent(c, "ten_second_sampling_complete");
          c.finalClock = c.ctx.getOutputTimestamp();
          c.finalQuality = ui.video.getVideoPlaybackQuality && ui.video.getVideoPlaybackQuality();
          c.acquired = true;
          releaseDevices(c);
          ui.timer.textContent = "10.0 s / 10 s"; ui.progress.value = 10;
          setState("Processing your recording…", "busy");
          push();
        }
      };
      // Start acquisition only after the authenticated session and message
      // handler exist; no pre-session worklet backlog enters the window.
      const quality = ui.video.getVideoPlaybackQuality && ui.video.getVideoPlaybackQuality();
      c.initialDropped = quality ? quality.droppedVideoFrames : 0;
      inspectMicrophone(c);
      c.src.connect(c.node); c.node.connect(c.sink);
      c.timer = setInterval(push, PUSH_INTERVAL_MS);
      ui.stop.disabled = false;
      setState("Capturing: waiting for audio…", "busy");
    } catch (err) {
      const code = err && err.name;
      const known = ["NotAllowedError", "SecurityError", "NotFoundError", "OverconstrainedError",
                     "NotReadableError", "AbortError", "NO_AUDIO_TRACK", "NO_VIDEO_TRACK", "NO_AUDIO_CLOCK",
                     "MICROPHONE_DISABLED", "MICROPHONE_TRACK_MUTED", "AUDIO_CONTEXT_NOT_RUNNING"];
      c.mic.category = code === "NotAllowedError" || code === "SecurityError" ? "media_permission_denied" :
        code === "NotFoundError" || code === "NotReadableError" || code === "NO_AUDIO_TRACK" ? "microphone_device_unavailable" :
        (code || "capture_start_failed").toLowerCase();
      if (!c.stopped && cap === c) stop(known.indexOf(code) >= 0 ? AVTiming.permissionMessage(err) :
        "Live capture could not start (" + (code || "error") + ").", "error");
    }
  }

  function watchFrames() {
    const c = cap, v = ui.video;
    const onFrame = function (perfMs, meta) {
      if (c.stopped || c.acquired || cap !== c) return;
      const t = c.clock.perfToTimeline(perfMs);
      c.fpsCount++;
      const k = c.mode === "record" && t > 9.1 ? null : c.picker.offer(t);
      if (k !== null) {   // copy the pixels now: this is the best frame so far for second k
        let cv = c.canvases.get(k);
        if (!cv) { cv = document.createElement("canvas"); c.canvases.set(k, cv); }
        const scale = Math.min(1, MAX_FRAME_SIDE / Math.max(v.videoWidth, v.videoHeight));
        cv.width = Math.round(v.videoWidth * scale); cv.height = Math.round(v.videoHeight * scale);
        cv.getContext("2d").drawImage(v, 0, 0, cv.width, cv.height);
        cv.dataset.t = String(t);
        if (research) {
          cv.dataset.frameId = String(meta && meta.presentedFrames);
          cv.dataset.mediaTime = String(meta && meta.mediaTime);
          cv.dataset.timestampKind = meta && Number.isFinite(meta.captureTime) ? "captureTime" : "callbackNow";
        }
      }
      for (const due of c.picker.due(t)) {
        const cv = c.canvases.get(due); c.canvases.delete(due);
        if (cv) {
          if (!capacity(c, 0, 1)) return;
          const frame = { t: Number(cv.dataset.t), image: cv.toDataURL("image/jpeg", JPEG_QUALITY).split(",")[1] };
          if (research) Object.assign(frame,{frame_id:Number(cv.dataset.frameId),media_time:Number(cv.dataset.mediaTime),
                                           timestamp_kind:cv.dataset.timestampKind});
          c.frames.push(frame);
          if (due >= 0 && due < 10) c.mic.videoTimes[due] = {second:due,
            timestamp_s:frame.t, offset_ms:1000*(frame.t-due)};
        }
      }
    };
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      const loop = function (now, meta) {
        // captureTime is the camera's own timestamp (performance.now domain) where the browser has it
        onFrame(meta && Number.isFinite(meta.captureTime) ? meta.captureTime : now, meta);
        if (!c.stopped && !c.acquired) c.rvfc = v.requestVideoFrameCallback(loop);
      };
      c.rvfc = v.requestVideoFrameCallback(loop);
    } else {
      const loop = function () { onFrame(performance.now()); if (!c.stopped && !c.acquired) c.raf = requestAnimationFrame(loop); };
      c.raf = requestAnimationFrame(loop);
    }
  }

  async function push() {
    const c = cap;
    if (!c || c.stopped || c.pushing || !c.session || !c.clock) return;
    const chunks = c.pendingAudio.splice(0);
    const last = c.audio.take(); if (last) chunks.push(last);
    const frames = c.frames.splice(0, 4);
    if (!chunks.length && !frames.length) {
      if (c.acquired && !c.finalizing) { c.finalizing = true; await finishRecording(c); }
      return;
    }
    const now = performance.now();
    c.client.video_fps = 1000 * c.fpsCount / Math.max(1, now - c.fpsSince);
    c.fpsCount = 0; c.fpsSince = now;
    const q = c.finalQuality || (ui.video.getVideoPlaybackQuality && ui.video.getVideoPlaybackQuality());
    if (q) c.client.dropped_video_frames = q.droppedVideoFrames - (research ? c.initialDropped : 0);
    c.client.audio_latency_ms = 1000 * ((c.ctx.baseLatency || 0) + (c.ctx.outputLatency || 0));
    c.pushing = true;
    try {
      const payload = {
        ...credentials(c),
        audio: chunks.map(function (ch) { return { start: ch.start, pcm: AVTiming.bytesToBase64(AVTiming.floatToPcm16(ch.samples)) }; }),
        frames: frames, client: c.client,
      };
      if (research) {
        const a = c.finalClock || c.ctx.getOutputTimestamp();
        payload.research_clock = {context:a.contextTime,performance:a.performanceTime};
      }
      const d = await rpc("live_push", payload);
      if (c.stopped || cap !== c) return;
      serverAudioDiagnostic(c, d && (d.capture_progress || d.capture_diagnostics));
      if (!d || d.status !== "CAPTURING") {
        const diagnostic = research && d && d.capture_diagnostics;
        const code = (d && d.error_code) || "NO_RESPONSE";
        c.mic.category = code.includes("DEVICE") ? "microphone_or_camera_unavailable" : "capture_timing_or_sync_failure";
        const reason = code.includes("DROPPED") || code.includes("FRAME") ?
          "Camera frames were dropped or delayed. Keep this tab visible and close other camera apps, then record again." :
          code.includes("DEVICE") ? "Camera or microphone input became unavailable. Enable both devices, then record again." :
          code.includes("CLOCK") || code.includes("DISCONTINUITY") ?
          "The media clock was interrupted. Keep the tab active and reconnect the devices, then record again." :
          "Capture could not be accepted. Reset, check the local connection and both devices, then record again.";
        stop("Unable to verify: " + reason + " (" + code + ")", "error");
        // Keep only the server's sanitized final counters after cleanup. The
        // ordinary timer/grid reset must not erase the reason for rejection.
        if (diagnostic) ui.contract.textContent = "Capture diagnostic: " + JSON.stringify(diagnostic);
        return;
      }
      renderGrid(d); renderDiag(d);
      if (c.acquired) return;
      ui.check.disabled = !d.ready;
      if (d.ready) setState("Buffer ready: 10 aligned seconds", "ok");
      else setState("Capturing: " + d.segments_valid + " / 10 valid seconds", "busy");
      if (c.client.audio_muted) setMessage("The microphone is muted, so those seconds are marked as muted audio.", "warn");
      else if (document.hidden) setMessage("The tab is in the background, so the camera may pause and seconds will be dropped.", "warn");
      else setMessage("");
    } catch (e) {
      if (!c.stopped) stop("Lost connection to the server. Capture stopped.", "error");
    } finally {
      c.pushing = false;
      if (cap === c && c.acquired && !c.stopped && !c.finalizing) push();
    }
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
      const r = await rpc("live_check", {...credentials(c), complete: !!c.acquired});
      if (!c.stopped) ui.contract.textContent = describeCheck(r);
      return r;
    } catch (e) {
      if (!c.stopped) stop("The window check timed out or lost its connection. Capture stopped.", "error");
      return null;
    } finally { c.checking = false; }
  }

  // Record mode: check the first complete window, then release the devices but keep the result on screen.
  async function finishRecording(c) {
    c.mic.prematureSeal = !c.acquired || !c.mic.acquisitionComplete;
    if (research) {
      let sealed;
      try {
        sealed = await rpc("research_seal",{...credentials(c), complete: !!c.acquired},function (r) {
          if (c.stopped && r && r.status === "RESEARCH_CAPTURE_SEALED") {
            rpc("research_cancel",{ticket:r.ticket,cancel_key:r.cancel_key}).catch(function () {});
          }
        });
      } catch (e) { if (!c.stopped) stop("Research export failed or timed out.","error"); return; }
      if (c.stopped || cap !== c) return;
      serverAudioDiagnostic(c, sealed && sealed.capture_diagnostics);
      if (!sealed || sealed.status !== "RESEARCH_CAPTURE_SEALED") {
        const code = (sealed && sealed.error_code) || "LIVE_BUFFER_NOT_READY";
        const zeroBins = c.mic.zeroSeconds.length ? c.mic.zeroSeconds :
          c.mic.seconds.map((s,i) => s.samples === c.ctx.sampleRate && s.pcmNonzero === 0 ? i : null).filter(i => i !== null);
        const location = zeroBins.length ? "Zero PCM in " + zeroBins.map(i => i + "–" + (i+1) + " s (second " + i + ")").join(", ") + ". " : "";
        const quantizedBins = zeroBins.filter(i => c.mic.seconds[i] &&
          c.mic.seconds[i].samples === c.ctx.sampleRate && c.mic.seconds[i].floatNonzero > 0 && c.mic.seconds[i].pcmNonzero === 0);
        const quantizedReason = quantizedBins.length ? "Nonzero browser audio in second(s) " + quantizedBins.join(", ") +
          " became all-zero PCM16 in the existing validated conversion. The input level fell below PCM16 resolution. Select a working direct microphone input and keep audible input present for the full ten seconds, then record again." : "";
        const silentReason = quantizedReason || (c.mic.pcmNonzero ?
          "The server found an all-zero PCM second even though other samples were nonzero. Check the microphone input and hardware mute switch, then record again. Quiet nonzero audio is checked under the unchanged contract." :
          c.mic.floatNonzero ?
          "The microphone signal was nonzero but became digital silence in the existing PCM16 conversion. Check the selected input and increase the operating-system microphone input level, then record again." :
          "All captured microphone PCM samples were exactly zero. A live, enabled track can still receive silence from the device. Check the selected browser input, hardware mute switch and operating-system microphone input level, then record again.");
        c.mic.category = code.includes("MUTED") ? (quantizedBins.length ? "pcm16_quantized_zero_seconds" : c.mic.pcmNonzero ? "all_zero_pcm_second" :
          c.mic.floatNonzero ? "below_pcm16_resolution" : "all_zero_pcm") : "capture_timing_or_sync_failure";
        const reason = code.includes("MUTED") ? location + silentReason :
          code.includes("FRAME") ? "Camera frames were missing or delayed. Keep the tab visible, close other camera apps and record again." :
          "A complete synchronized window could not be validated. Keep this tab active, check both devices and record again.";
        stop("Unable to verify: " + reason + " (" + code + ")", "error");return;
      }
      c.mic.phase = "sealed"; micEvent(c, "capture_sealed"); renderMicrophone(c);
      researchJob={ticket:sealed.ticket,cancel_key:sealed.cancel_key,timer:null};
      ui.consent.checked=false;
      stop("Processing your recording in the isolated model worker…","info",true);
      setState("Processing your recording…", "busy");
      ui.stop.disabled=false;
      const job=researchJob;
      async function pollResearch() {
        if (researchJob!==job) return;
        try {
          const r=await rpc("research_poll",{ticket:job.ticket,cancel_key:job.cancel_key});
          if (researchJob!==job) return;
          if (r && r.status === "EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED" && r.production_predictions_enabled === false) {
            // Keep polling for ownership/expiry revocation, but preserve the
            // user's expanded evidence/diagnostics for this immutable result.
            if (job.renderedRequest !== r.request_id) {
              AVLiveResearch.render(ui.researchResult,r);
              job.renderedRequest=r.request_id;
              c.mic.phase = "verified"; renderMicrophone(c);
              setState("Results ready","ok");
              setMessage("");
            }
          } else if (!r || r.status !== "RESEARCH_REPLAY_PENDING") {
            stop("Research result unavailable, expired or revoked.","warn");return;
          }
        } catch (e) { if (researchJob===job) stop("Research session lost; result cleared.","error");return; }
        if (researchJob===job) job.timer=setTimeout(pollResearch,1000);
      }
      job.timer=setTimeout(pollResearch,500);
      return;
    }
    let r = null;
    try { r = await check(); } catch (e) { r = null; }
    if (c.stopped) return;
    const ok = r && r.status === "CONTRACT_VALID";
    stop(ok ? "" : "The recording could not be checked. Press Record to try again.", ok ? "" : "error", true);
    setState(ok ? "Recorded: 10 aligned seconds" : "Stopped", ok ? "ok" : "error");
  }

  // Releases devices, audio graph, timers and the server session. Safe to call more than once.
  function releaseDevices(c) {
    if (c.devicesReleased) return;
    renderMicrophone(c);
    c.devicesReleased = true;
    clearInterval(c.timer); clearTimeout(c.deadline);
    if (c.deviceChange && navigator.mediaDevices.removeEventListener)
      navigator.mediaDevices.removeEventListener("devicechange", c.deviceChange);
    if (c.rvfc !== null && ui.video.cancelVideoFrameCallback) ui.video.cancelVideoFrameCallback(c.rvfc);
    if (c.raf !== null) cancelAnimationFrame(c.raf);
    if (c.node) { c.node.port.onmessage = null; c.node.disconnect(); }
    if (c.sink) c.sink.disconnect();
    if (c.src) c.src.disconnect();
    if (c.ctx && c.ctx.state !== "closed") c.ctx.close();
    releaseStream(c.stream);
    ui.video.srcObject = null;
    ui.meter.style.width = "0%";
    micEvent(c, "devices_released"); renderMicrophone(c);
  }

  // keepResult leaves the last grid, diagnostics and check on screen (a finished recording).
  function stop(message, kind, keepResult) {
    if (research && !keepResult) cancelResearch();
    const c = cap; cap = null;
    if (c) {
      if (kind === "error") c.mic.phase = "invalid";
      else if (!keepResult) c.mic.phase = "cancelled";
      c.stopped = true;
      releaseDevices(c);
      renderMicrophone(c);
      c.canvases.clear(); c.frames.length = 0;
      c.audio.clear(); c.pendingAudio.length = 0;
      if (c.session) rpc("live_stop", credentials(c)).catch(function () {});
    }
    ui.video.srcObject = null;
    ui.meter.style.width = "0%";
    // The server discards the buffer on stop, so the display must not keep showing it as ready.
    if (!keepResult) {
      if (ui.consent && c) ui.consent.checked = false;
      renderGrid(null); ui.diag.innerHTML = ""; ui.contract.textContent = "";
      if (ui.timer) ui.timer.textContent = "0.0 s / 10 s";
      if (ui.progress) ui.progress.value = 0;
    }
    ui.start.disabled = inFlight > 0; ui.record.disabled = inFlight > 0; ui.stop.disabled = true; ui.check.disabled = true;
    setState(message && kind === "error" ? "Unable to verify" : "Idle", message && kind === "error" ? "error" : "idle");
    clearTimeout(cleanupTimer); cleanupTimer = null;
    setMessage((message || "") + (inFlight ? " " + CLEANUP_MESSAGE : ""), kind);
    if (inFlight) cleanupTimer = setTimeout(function () {
      if (!cap && inFlight) {
        setMessage((message || "") + " " + WAITING_MESSAGE, kind || "warn");
      }
    }, 1000);
  }

  async function cancelResearch() {
    const job=researchJob;researchJob=null;
    if (ui.researchResult) ui.researchResult.textContent="";
    if (job) {
      clearTimeout(job.timer);
      try { await rpc("research_cancel",{ticket:job.ticket,cancel_key:job.cancel_key}); } catch (e) {}
    }
  }

  ui.start.addEventListener("click", function () { start("rolling"); });
  ui.record.addEventListener("click", function () { start("record"); });
  ui.stop.addEventListener("click", function () { stop(); });
  if (ui.reset) ui.reset.addEventListener("click", function () {
    stop();
    if (ui.micDiag) ui.micDiag.textContent = "";
    if (ui.micStatus) ui.micStatus.textContent = "Microphone not started.";
  });
  ui.check.addEventListener("click", check);
  window.addEventListener("pagehide", function () { stop(); });
  document.addEventListener("visibilitychange", function () {
    if (document.hidden && cap) stop("Capture stopped because the tab is in the background. Start a new recording on return.", "warn");
  });
  renderGrid(null);
  stop();
})();
