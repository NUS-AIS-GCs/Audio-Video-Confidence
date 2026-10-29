// Controller-level tests with synthetic browser devices. No camera, server or model.
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const T = require('../../deployment/web/live_timing.js');
const web = path.join(__dirname, '../../deployment/web');
const tick = () => new Promise(resolve => setTimeout(resolve, 20));

function browser(options = {}) {
  const nodes = {};
  function node(selector) {
    const handlers = {};
    return nodes[selector] = {dataset: {}, style: {}, children: Array.from({length: 10}, () => ({dataset: {}})),
      textContent: '', innerHTML: '', addEventListener: (name, fn) => {handlers[name] = fn;},
      click: () => handlers.click(), handlers, append() {}};
  }
  for (const name of ['preview','start','record','stop','reset','timer','progress','check','state','msg','grid','diag','meter-fill','contract','mic-status','mic-diag']) node('.av-' + name);
  if (options.research) {
    node('.av-research-consent').checked = options.consent !== false;
    node('.av-research-result');
  }
  const video = nodes['.av-preview']; video.play = async () => {}; video.videoWidth = 640; video.videoHeight = 480;
  video.cancelVideoFrameCallback = () => {}; video.requestVideoFrameCallback = () => 1;
  const tracks = ['audio','video'].map(kind => ({kind, handlers:{}, enabled:kind==='audio'?options.enabled!==false:true,
    muted:kind==='audio'&&!!options.muted, readyState:'live', stopped: false, stop() {this.stopped = true;this.readyState='ended';},
    addEventListener(n,f) {this.handlers[n]=f;}, getSettings: () => ({width: 640, height: 480,
      sampleRate:48000,channelCount:2,echoCancellation:false,noiseSuppression:false,autoGainControl:false})}));
  const stream = {getTracks: () => tracks, getAudioTracks: () => [tracks[0]], getVideoTracks: () => [tracks[1]]};
  let worklet, interval, audioContext, permissionCalls=0; const stops = [], pushes = [], cancels=[];
  class Context {
    constructor() {audioContext=this;this.listeners={};this.sampleRate = 16000; this.state = options.contextState||'running'; this.audioWorklet = {addModule: async () => {}};}
    addEventListener(n,f){this.listeners[n]=f;}
    resume(){if(options.resumeReject)return Promise.reject(new Error('resume blocked'));if(!options.staySuspended)this.state='running';return Promise.resolve();}
    createMediaStreamSource() {return {connect() {}, disconnect() {}};}
    createGain() {return {gain: {}, connect() {}, disconnect() {}};}
    getOutputTimestamp() {return options.clock ? options.clock() : {contextTime: 1, performanceTime: 1000};}
    close() {this.state = 'closed'; return Promise.resolve();}
  }
  const server = {
    live_start: options.start || (async () => ({status: 'STARTED', session: 'server-session', tab_key: 'server-tab-key'})),
    live_push: data => {pushes.push(data); return options.push ? options.push(data) : Promise.resolve({status:'CAPTURING'});},
    live_stop: data => {stops.push(data); return options.stop ? options.stop(data) : Promise.resolve({status:'STOPPED'});},
    live_check: options.check || (async () => ({status:'NOT_READY', error_code:'LIVE_BUFFER_NOT_READY'})),
    research_seal: options.seal || (async()=>({status:'RESEARCH_CAPTURE_SEALED',ticket:'a'.repeat(64),cancel_key:'b'.repeat(64)})),
    research_poll: options.poll || (async()=>({status:'RESEARCH_REPLAY_PENDING',predictions:null})),
    research_cancel: async data=>{cancels.push(data);return {status:'RESEARCH_CAPTURE_REVOKED'};}
  };
  const context = {
    AVTiming: {...T, withTimeout: (p, ms) => T.withTimeout(p, ms === 30000 ? 100 : 10)},
    element: {querySelector: selector => nodes[selector]}, server,
    window: {isSecureContext: true, addEventListener() {}},
    navigator: {mediaDevices: {listeners:{},addEventListener(n,f){this.listeners[n]=f;},removeEventListener(n){delete this.listeners[n];},getUserMedia: () => {permissionCalls++;tracks.forEach(t=>{t.stopped=false;t.readyState='live';});return options.permissionName?Promise.reject({name:options.permissionName}):options.permission || Promise.resolve(stream);}}},
    AVLiveResearch:options.render?{render:options.render}:require('../../deployment/web/live_research_result.js'),
    AudioContext: Context,
    AudioWorkletNode: class {constructor() {this.port = {}; worklet = this;} connect() {} disconnect() {}},
    URL: {createObjectURL: () => 'blob:unit', revokeObjectURL() {}}, Blob,
    HTMLVideoElement: {prototype: {requestVideoFrameCallback() {}}},
    performance: {now: () => 1000},
    document: {hidden:false, listeners:{}, addEventListener(n,f){this.listeners[n]=f;},
      createElement: () => ({dataset:{},getContext: () => ({drawImage() {}})})},
    setInterval: fn => {interval = fn; return 1;}, clearInterval() {},
    setTimeout: (fn,ms) => setTimeout(fn,options.pollFast&&(ms===500||ms===1000)?5:ms===1000 && options.cleanupGrace ? options.cleanupGrace : ms),
    clearTimeout, cancelAnimationFrame() {}, requestAnimationFrame: () => 1,
  };
  vm.runInNewContext(fs.readFileSync(path.join(web, 'live_capture.js'), 'utf8'), context);
  return {nodes, tracks, stream, stops, pushes, cancels, get permissionCalls(){return permissionCalls;},
    get audioContext(){return audioContext;}, diagnostic:()=>JSON.parse(nodes['.av-mic-diag'].textContent),
    deviceChange:()=>context.navigator.mediaDevices.listeners.devicechange?.(),
    get deviceListenerCount(){return Object.keys(context.navigator.mediaDevices.listeners).length;},
    workletError:data=>worklet.port.onmessage?.({data}),
    background: () => {context.document.hidden=true;context.document.listeners.visibilitychange();},
    start: () => nodes['.av-start'].click(), stop: () => nodes['.av-stop'].click(),
    record: () => nodes['.av-record'].click(),
    push: () => interval(),
    audio: (i, done=false, amplitude=options.amplitude||0) => worklet.port.onmessage?.({data: {frame:16000 + i * 1600,
      samples:new Float32Array(1600).fill(amplitude), written_samples:1600, startup_empty_input_quanta:0, done}})};
}

test('stalled push times out, stops devices and includes both server capabilities', async () => {
  const b = browser({push: () => new Promise(() => {})});
  b.start(); await tick();
  b.audio(0); await b.push();
  assert.ok(b.tracks.every(t => t.stopped));
  assert.equal(b.nodes['.av-preview'].srcObject, null);
  assert.equal(b.pushes.length, 1);
  assert.equal(b.pushes[0].tab_key, 'server-tab-key');
  assert.equal(b.stops[0].session, 'server-session');
  assert.equal(b.nodes['.av-start'].disabled, true);
  b.start(); await tick(); assert.equal(b.pushes.length, 1); // no new capture while the real call is stalled
});

test('research consent is required before activating even fake devices', async () => {
  const b=browser({research:true,consent:false});
  await b.record();
  assert.equal(b.permissionCalls,0);
  assert.match(b.nodes['.av-msg'].textContent,/Consent is required/);
});

test('a research seal arriving after Stop is revoked and never displayed', async () => {
  let finish;
  const b=browser({research:true,push:async()=>({status:'CAPTURING',ready:true}),
    seal:()=>new Promise(resolve=>{finish=resolve;})});
  b.record();await tick();b.audio(0,true);const pending=b.push();await new Promise(r=>setTimeout(r,1));
  b.stop();finish({status:'RESEARCH_CAPTURE_SEALED',ticket:'a'.repeat(64),cancel_key:'b'.repeat(64)});
  await pending;await tick();
  assert.equal(b.cancels.length,1);
  assert.equal(b.nodes['.av-research-result'].textContent,'');
  assert.ok(b.tracks.every(t=>t.stopped));
});

test('Reset cancels sealed offline replay and clears its display', async () => {
  const b=browser({research:true,push:async()=>({status:'CAPTURING',ready:true})});
  b.record();await tick();b.audio(0,true);await b.push();await tick();
  assert.equal(b.nodes['.av-research-consent'].checked,false);
  b.nodes['.av-reset'].click();await tick();
  assert.equal(b.cancels.length,1);
  assert.equal(b.nodes['.av-research-result'].textContent,'');
  assert.ok(b.tracks.every(t=>t.stopped));
});

test('completed evidence stays expanded while authenticated revocation polling continues', async () => {
  let polls=0,renders=0;
  const b=browser({research:true,pollFast:true,push:async()=>({status:'CAPTURING',ready:true}),
    poll:async()=>{polls++;return {status:'EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED',production_predictions_enabled:false,request_id:'a'.repeat(32)};},
    render:element=>{renders++;element.textContent='expanded evidence';}});
  b.record();await tick();b.audio(0,true);await b.push();await tick();await new Promise(r=>setTimeout(r,35));
  assert.ok(polls>=2);assert.equal(renders,1);
  assert.equal(b.nodes['.av-research-result'].textContent,'expanded evidence');
  assert.match(b.nodes['.av-state'].textContent,/Results ready/);
  b.nodes['.av-reset'].click();await tick();
  assert.equal(b.cancels.length,1);assert.equal(b.nodes['.av-research-result'].textContent,'');
});

test('research limit rejection preserves sanitized counters after device cleanup', async () => {
  const diagnostic={stage:'capture_ingest',limit_kind:'audio_samples',sample_rate:16000,
    accepted_audio_samples:208000,proposed_audio_samples:212000,max_audio_samples:208000,
    buffer:{ready:false,segments_valid:0}};
  const b=browser({research:true,push:async()=>({status:'REJECTED',
    error_code:'LIVE_RESEARCH_CAPTURE_TOO_LONG',capture_diagnostics:diagnostic})});
  b.record();await tick();b.audio(0,true);await b.push();await tick();
  assert.ok(b.tracks.every(t=>t.stopped));
  assert.equal(b.nodes['.av-preview'].srcObject,null);
  assert.match(b.nodes['.av-contract'].textContent,/audio_samples/);
  assert.match(b.nodes['.av-contract'].textContent,/208000/);
  assert.equal(b.nodes['.av-research-result'].textContent,'');
  assert.equal(b.stops.length,1);
  b.nodes['.av-reset'].click();
  assert.equal(b.nodes['.av-contract'].textContent,'');
});

test('normal pending stop shows cleanup state without a stalled warning, then clears', async () => {
  let complete;
  const b = browser({stop: () => new Promise(resolve => {complete=resolve;})});
  b.start(); await tick(); b.stop();
  assert.match(b.nodes['.av-msg'].textContent, /Releasing capture resources/);
  assert.doesNotMatch(b.nodes['.av-msg'].textContent, /reload this tab/);
  assert.equal(b.nodes['.av-start'].disabled, true);
  complete({status:'STOPPED'}); await tick();
  assert.equal(b.nodes['.av-msg'].hidden, true);
  assert.equal(b.nodes['.av-start'].disabled, false);
});

test('actually stalled cleanup keeps restart blocked until the raw request settles', async () => {
  let complete;
  const b = browser({cleanupGrace:5,stop:()=>new Promise(resolve=>{complete=resolve;})});
  b.start(); await tick(); b.stop(); await tick();
  assert.match(b.nodes['.av-msg'].textContent, /Waiting for outstanding/);
  assert.equal(b.nodes['.av-start'].disabled, true);
  complete({status:'STOPPED'}); await tick();
  assert.equal(b.nodes['.av-msg'].hidden, true);
  assert.equal(b.nodes['.av-start'].disabled, false);
});

test('backgrounding releases devices and requires a new foreground recording', async () => {
  const b = browser(); b.start(); await tick(); b.background(); await tick();
  assert.ok(b.tracks.every(t=>t.stopped));
  assert.equal(b.nodes['.av-preview'].srcObject,null);
  assert.match(b.nodes['.av-msg'].textContent,/background/);
  b.start(); await tick();
  assert.match(b.nodes['.av-msg'].textContent,/Return to this tab/);
});

test('a device event from an old capture cannot stop a new capture', async () => {
  const b = browser(); b.start(); await tick();
  const oldEnded=b.tracks[0].handlers.ended;
  b.stop(); await tick(); b.start(); await tick();
  oldEnded();
  assert.equal(b.nodes['.av-stop'].disabled,false);
  assert.doesNotMatch(b.nodes['.av-msg'].textContent,/disconnected/);
  b.stop(); await tick();
});

test('current device disconnect cleans up both modalities', async () => {
  const b=browser(); b.start(); await tick(); b.tracks[1].handlers.ended(); await tick();
  assert.ok(b.tracks.every(t=>t.stopped));
  assert.match(b.nodes['.av-msg'].textContent,/disconnected/);
});

test('a late completed-window check cannot restore results after Reset', async () => {
  let complete;
  const b=browser({push:async()=>({status:'CAPTURING',ready:true,segments_valid:10,segments:[],
    frame_lag_ms:0,max_frame_offset_ms:0,capture_sample_rate:16000}),
    check:()=>new Promise(resolve=>{complete=resolve;})});
  b.record(); await tick(); b.audio(0,true); const pushing=b.push();
  await new Promise(resolve=>setTimeout(resolve,1)); b.nodes['.av-reset'].click();
  complete({status:'CONTRACT_VALID',waveform_shape:[1,160000],frames_shape:[1,10,224,224,3],timing:{}});
  await pushing; await tick();
  assert.equal(b.nodes['.av-contract'].textContent,'');
  assert.equal(b.nodes['.av-state'].textContent,'Idle');
  assert.equal(b.nodes['.av-progress'].value,0);
});

test('permission rejection and unavailable audio clock produce readable failures', async () => {
  const denied=browser({permission:Promise.reject({name:'NotAllowedError'})}); denied.start(); await tick();
  assert.match(denied.nodes['.av-msg'].textContent,/permission/i);
  const b=browser({clock:()=>({contextTime:0,performanceTime:0})}); b.start(); await tick();
  for(let i=0;i<41;i++) b.audio(i);
  await tick();
  assert.match(b.nodes['.av-msg'].textContent,/clock/i);
  assert.ok(b.tracks.every(t=>t.stopped));
});

test('capture timer follows audio sample time and Reset releases and clears capture', async () => {
  const b = browser();
  b.start(); await tick();
  for(let i=0;i<10;i++)b.audio(i);
  assert.equal(b.nodes['.av-timer'].textContent, '1.0 s / 10 s');
  assert.equal(b.nodes['.av-progress'].value, 1);
  b.nodes['.av-reset'].click(); await tick();
  assert.ok(b.tracks.every(t => t.stopped));
  assert.equal(b.nodes['.av-timer'].textContent, '0.0 s / 10 s');
  assert.equal(b.nodes['.av-progress'].value, 0);
  assert.equal(b.nodes['.av-contract'].textContent, '');
  assert.equal(b.nodes['.av-msg'].hidden, true);
});

test('backlog stops capture before a stalled request can accumulate unbounded media', async () => {
  const b = browser({push: () => new Promise(() => {})});
  b.start(); await tick(); b.audio(0);
  const pushing = b.push();
  for (let i = 1; i <= 16 && !b.tracks[0].stopped; i++) b.audio(i);
  assert.ok(b.tracks.every(t => t.stopped));
  assert.equal(b.nodes['.av-preview'].srcObject, null);
  await pushing;
  assert.equal(T.backlogAllowed(16000, 24001, 1, 0), false);
  assert.equal(T.backlogAllowed(16000, 1, 9, 0), false);
  assert.equal(T.backlogAllowed(16000, 1, 1, 5), false);
});

test('a permission grant after Stop immediately releases devices', async () => {
  let grant;
  const permission = new Promise(resolve => {grant = resolve;});
  const b = browser({permission}); b.start(); b.stop(); grant(b.stream); await tick();
  assert.ok(b.tracks.every(t => t.stopped));
  assert.equal(b.nodes['.av-preview'].srcObject, null);
  assert.equal(b.stops.length, 0);
});

test('a late backend start is cleaned up after Stop', async () => {
  let complete;
  const b = browser({start: () => new Promise(resolve => {complete = resolve;})});
  b.start(); await new Promise(resolve => setTimeout(resolve, 1));
  b.stop(); complete({status:'STARTED', session:'late-session', tab_key:'late-tab'}); await tick();
  assert.ok(b.tracks.every(t => t.stopped));
  assert.equal(b.stops.length, 1);
  assert.equal(b.stops[0].tab_key, 'late-tab');
});

test('preview uses local URLs, revokes old media, and handles clear, limits and unmount', () => {
  const listeners = {}, video = {hidden:true, pause(){}, removeAttribute(){}, load(){}, addEventListener(){}};
  const note = {textContent:''}, root = {dataset:{limit:'1024'}};
  const element = {isConnected:true, querySelector: s => s === 'video' ? video : s === '.av-preview-note' ? note : root};
  const revoked = []; let observer, created = 0;
  const doc = {body:{}, addEventListener:(n,f) => {listeners[n]=f;}, removeEventListener:n => {delete listeners[n];}};
  const context = {element, document:doc, window:{addEventListener(){},removeEventListener(){}},
    URL:{createObjectURL:() => 'blob:unit-' + ++created, revokeObjectURL:u => revoked.push(u)},
    MutationObserver:class {constructor(fn){observer=fn;} observe(){} disconnect(){}}};
  vm.runInNewContext(fs.readFileSync(path.join(web,'upload_preview.js'),'utf8'),context);
  const select = file => listeners.change({target:{matches:()=>true, files:[file]}});
  select({name:'a.mp4',size:100}); assert.equal(video.src,'blob:unit-1');
  select({name:'b.webm',size:100}); assert.deepEqual(revoked,['blob:unit-1']);
  listeners.click({target:{closest:()=>({getAttribute:()=> 'Remove',textContent:''})}});
  assert.equal(video.hidden,true); assert.equal(revoked.length,2);
  select({name:'too-large.mp4',size:1025}); assert.equal(created,2);
  select({name:'bad.txt',size:100}); assert.equal(created,2);
  listeners.drop({type:'drop',target:{closest:()=>true},dataTransfer:{files:[{name:'drop.mp4',size:100}]}});
  assert.equal(created,3);
  element.isConnected=false; observer();
  assert.equal(revoked.length,3); assert.equal(Object.keys(listeners).length,0);
});

test('the audio processor ends at exactly ten seconds without a server-ready response', () => {
  const source = fs.readFileSync(path.join(web,'live_capture.js'),'utf8').match(/const WORKLET = `([\s\S]*?)`;/)[1];
  const messages=[];let Processor;
  const context={Float32Array,Infinity,currentFrame:0,sampleRate:16000,
    AudioWorkletProcessor:class {constructor(){this.port={postMessage:m=>messages.push(m)};}},
    registerProcessor:(_name,ctor)=>{Processor=ctor;}};
  vm.runInNewContext(source,context);
  const processor=new Processor({processorOptions:{limit:160000}});
  let running=true,quanta=0;
  while(running&&quanta<1300){context.currentFrame=quanta*128;running=processor.process([[new Float32Array(128).fill(.1)]]);quanta++;}
  assert.equal(running,false);assert.equal(quanta,1250);
  assert.equal(messages.reduce((n,m)=>n+m.samples.length,0),160000);
  assert.equal(messages.filter(m=>m.done).length,1);
  assert.equal(messages[0].frame,0);assert.equal(messages[0].samples.length,128);
  for(let i=1;i<messages.length;i++)assert.equal(messages[i].frame,messages[i-1].frame+messages[i-1].samples.length);
});

test('device release happens before sealing and a server-ready response cannot extend acquisition', async () => {
  let seals=0,releasedAtSeal=false;
  const b=browser({research:true,push:async()=>({status:'CAPTURING',ready:false}),
    seal:async data=>{seals++;releasedAtSeal=b.tracks.every(t=>t.stopped);assert.equal(data.complete,true);
      return {status:'RESEARCH_CAPTURE_SEALED',ticket:'a'.repeat(64),cancel_key:'b'.repeat(64)};}});
  b.record();await tick();b.audio(0,true);await tick();await tick();
  assert.equal(seals,1);assert.equal(releasedAtSeal,true);
  const pushed=b.pushes.length; b.audio(1);await b.push();assert.equal(b.pushes.length,pushed);
  b.nodes['.av-reset'].click();await tick();
});

test('muting a track cancels acquisition with an actionable reason', async () => {
  const b=browser({research:true});b.record();await tick();
  b.tracks[0].handlers.mute();await tick();
  assert.ok(b.tracks.every(t=>t.stopped));assert.match(b.nodes['.av-msg'].textContent,/muted/);
  assert.equal(b.nodes['.av-research-result'].textContent,'');
});

test('disabled and muted microphone tracks fail before any samples are sent',async()=>{
  for(const options of [{enabled:false},{muted:true}]){
    const b=browser({research:true,...options});b.record();await tick();
    assert.equal(b.pushes.length,0);assert.ok(b.tracks.every(t=>t.stopped));
    assert.match(b.nodes['.av-msg'].textContent,options.enabled===false?/disabled/:/muted/);
    const d=b.diagnostic();assert.equal(d.received_samples,0);assert.equal(d.phase,'invalid');
    assert.equal(d.track.enabled,options.enabled!==false);assert.equal(d.track.muted,!!options.muted);
  }
});

test('permission denial, missing device and blocked audio context remain distinct',async()=>{
  for(const [options,category,reason] of [
    [{permissionName:'NotAllowedError'},'media_permission_denied',/permission/],
    [{permissionName:'NotFoundError'},'microphone_device_unavailable',/found|device/],
    [{contextState:'suspended',staySuspended:true},'audio_context_not_running',/context.*suspended/],
    [{contextState:'suspended',resumeReject:true},'audio_context_not_running',/context.*suspended/]
  ]){
    // Attach the permission rejection before yielding to the event loop.
    const b=browser({research:true,...options});b.record();await tick();
    assert.equal(b.diagnostic().category,category);assert.equal(b.diagnostic().received_samples,0);
    assert.match(b.nodes['.av-msg'].textContent,reason);
    if(!options.permissionName)assert.ok(b.tracks.every(t=>t.stopped));
    assert.equal(b.pushes.length,0);assert.equal(b.nodes['.av-preview'].srcObject,null);
  }
});

test('context suspension during sampling releases tracks with its own diagnostic',async()=>{
  const b=browser({research:true,amplitude:.001});b.record();await tick();b.audio(0);
  b.audioContext.state='suspended';b.audioContext.listeners.statechange();await tick();
  assert.equal(b.diagnostic().category,'audio_context_not_running');assert.equal(b.diagnostic().capture_context_state,'suspended');
  assert.equal(b.diagnostic().current_context_state,'closed');
  assert.equal(b.diagnostic().received_samples,1600);assert.ok(b.tracks.every(t=>t.stopped));
});

test('quiet nonzero PCM is measured and passed unchanged rather than rejected by an activity threshold',async()=>{
  const b=browser({research:true,amplitude:1/32768});b.record();await tick();b.audio(0);await b.push();
  const d=b.diagnostic();assert.equal(d.received_samples,1600);assert.equal(d.pcm16_nonzero_samples,1600);
  assert.equal(d.web_audio_rms,1/32768);assert.equal(d.category,'nonzero_pcm_received');
  assert.ok(parseFloat(b.nodes['.av-meter-fill'].style.width)>0);
  const pcm=Buffer.from(b.pushes[0].audio[0].pcm,'base64');
  for(let i=0;i<1600;i++)assert.equal(pcm.readInt16LE(i*2),1);
  const text=b.nodes['.av-mic-diag'].textContent;
  for(const secret of ['server-session','server-tab-key','cancel_key','deviceId','samples":[','pcm":'])assert.ok(!text.includes(secret));
  b.nodes['.av-reset'].click();assert.equal(b.nodes['.av-mic-diag'].textContent,'');await tick();
});

test('exact zero PCM and a nonzero signal below PCM16 resolution have different diagnoses',async()=>{
  for(const [amplitude,category,reason] of [[0,'all_zero_pcm',/All captured.*exactly zero/],[1e-8,'below_pcm16_resolution',/nonzero.*PCM16 conversion/]]){
    const b=browser({research:true,amplitude,seal:async()=>({status:'REJECTED',error_code:'LIVE_WINDOW_MUTED_AUDIO',
      capture_diagnostics:{accepted_audio_samples:1600,buffer:{segments:[{second:0,status:'muted_audio'}]}}})});
    b.record();await tick();b.audio(0,true);await tick();await tick();
    const d=b.diagnostic();assert.equal(d.category,category);assert.equal(d.pcm16_nonzero_samples,0);
    assert.equal(d.server_accepted_audio_samples,1600);assert.deepEqual(d.zero_pcm_seconds_reported_by_server,[0]);
    assert.match(b.nodes['.av-msg'].textContent,reason);assert.equal(b.nodes['.av-research-result'].textContent,'');
  }
});

test('one silent second is not mislabeled as a wholly muted microphone',async()=>{
  const b=browser({research:true,amplitude:.01,seal:async()=>({status:'REJECTED',error_code:'LIVE_WINDOW_MUTED_AUDIO'})});
  b.record();await tick();b.audio(0,true);await tick();await tick();
  assert.equal(b.diagnostic().category,'all_zero_pcm_second');assert.match(b.nodes['.av-msg'].textContent,/other samples were nonzero/);
});

test('mute and unmute events are counted without preserving track identity or audio',async()=>{
  const b=browser({research:true});b.record();await tick();
  b.tracks[0].handlers.unmute();assert.equal(b.diagnostic().unmute_events,1);
  b.tracks[0].muted=true;b.tracks[0].handlers.mute();await tick();
  const d=b.diagnostic();assert.equal(d.mute_events,1);assert.equal(d.track.muted,true);
  assert.ok(d.events.some(x=>x.event==='microphone_mute'));assert.ok(d.events.length<=16);
});

function processorHarness(limit) {
  const source=fs.readFileSync(path.join(web,'live_capture.js'),'utf8').match(/const WORKLET = `([\s\S]*?)`;/)[1];
  const messages=[];let Processor;
  const context={Float32Array,Infinity,currentFrame:0,AudioWorkletProcessor:class {constructor(){this.port={postMessage:m=>messages.push(m)};}},registerProcessor:(_,p)=>Processor=p};
  vm.runInNewContext(source,context);
  return {context,messages,processor:new Processor({processorOptions:{limit}})};
}

test('missing input quantum fails without concatenating samples across the absent input',()=>{
  const {context,messages,processor}=processorHarness(160000);
  context.currentFrame=0;assert.equal(processor.process([[new Float32Array(128).fill(.125)]]),true);
  context.currentFrame=128;processor.process([[new Float32Array(128).fill(.25)]]);
  context.currentFrame=256;assert.equal(processor.process([[]]),false);
  assert.equal(messages.at(-1).capture_error,'AUDIO_INPUT_MISSING');
  assert.equal(messages.filter(m=>m.samples).reduce((s,m)=>s+m.samples.length,0),256);
  assert.equal(messages.filter(m=>m.done).length,0);
  assert.equal(messages[1].frame,128);assert.equal(messages[1].samples.length,128);
  assert.ok(messages[1].samples.every(x=>x===.25));
});

test('render-frame gaps and overlaps fail before the later input is accepted',()=>{
  for(const frame of [256,64]){
    const h=processorHarness(160000);h.processor.process([[new Float32Array(128).fill(.125)]]);
    h.context.currentFrame=frame;assert.equal(h.processor.process([[new Float32Array(128).fill(.25)]]),false);
    assert.equal(h.messages.at(-1).capture_error,'AUDIO_CALLBACK_DISCONTINUITY');
    assert.equal(h.messages.filter(m=>m.samples).reduce((s,m)=>s+m.samples.length,0),128);
    assert.equal(h.messages.at(-1).expected_frame,128);assert.equal(h.messages.at(-1).observed_frame,frame);
  }
});

test('ten-second worklet boundaries preserve every provided sample and the exact zero-second location',()=>{
  for(const sr of [16000,22050,48000])for(const zeroSecond of [0,4,9]){
    const h=processorHarness(sr*10),expected=new Float32Array(sr*10);let running=true;
    for(let frame=0;running;frame+=128){
      const input=new Float32Array(128);
      for(let i=0;i<input.length;i++){
        const sample=frame+i,value=Math.floor(sample/sr)===zeroSecond?0:(1+sample%31)/32768;
        input[i]=value;if(sample<expected.length)expected[sample]=value;
      }
      h.context.currentFrame=frame;running=h.processor.process([[input]]);
      assert.ok(frame<sr*10+128);
    }
    const chunks=h.messages.filter(m=>m.samples),actual=new Float32Array(expected.length);let end=0;
    for(const m of chunks){assert.equal(m.frame,end);assert.equal(m.written_samples,m.samples.length);actual.set(m.samples,end);end+=m.samples.length;}
    assert.equal(end,sr*10);assert.deepEqual(actual,expected);assert.equal(chunks.filter(m=>m.done).length,1);
    assert.equal(chunks.at(-1).done,true);
    const accumulator=T.makeAudioAccumulator();
    for(const m of chunks)assert.equal(accumulator.add(m.frame,m.samples),true);
    const combined=accumulator.take();assert.equal(combined.start,0);assert.deepEqual(combined.samples,expected);
    const bytes=T.floatToPcm16(combined.samples),pcm=new DataView(bytes.buffer);
    for(let second=0;second<10;second++){
      let nonzero=0;for(let i=second*sr;i<(second+1)*sr;i++)nonzero+=pcm.getInt16(i*2,true)!==0?1:0;
      assert.equal(nonzero,second===zeroSecond?0:sr);
    }
  }
});

test('zero second 4 maps to samples 64000..79999 and is named in the rejection message',async()=>{
  let seals=0;
  const b=browser({research:true,seal:async()=>{seals++;return {status:'REJECTED',error_code:'LIVE_WINDOW_MUTED_AUDIO',
    capture_diagnostics:{accepted_audio_samples:160000,buffer:{segments:[{second:4,status:'muted_audio'}]}}};}});
  b.record();await tick();
  for(let i=0;i<100;i++){b.audio(i,i===99,i>=40&&i<50?0:.125);if(i%5===4)await b.push();}
  await tick();await tick();const d=b.diagnostic();
  assert.equal(seals,1);assert.equal(d.received_samples,160000);assert.equal(d.continuity.next_sample_index,160000);
  assert.equal(d.continuity.acquisition_complete,true);assert.equal(d.continuity.finalized_before_acquisition_complete,false);
  assert.equal(d.continuity.all_emitted_samples_explicitly_written,true);assert.equal(d.continuity.discontinuities,0);
  assert.deepEqual(d.zero_pcm_seconds_at_worklet_input,[4]);assert.deepEqual(d.zero_pcm_seconds_reported_by_server,[4]);
  for(const s of d.seconds){assert.equal(s.samples,16000);assert.equal(s.peak_abs,s.second===4?0:.125);assert.equal(s.web_audio_rms,s.second===4?0:.125);assert.equal(s.pcm16_nonzero_samples,s.second===4?0:16000);}
  assert.match(b.nodes['.av-msg'].textContent,/second 4/);assert.ok(b.tracks.every(t=>t.stopped));
});

test('controller refuses out-of-order audio messages and records device-change observations',async()=>{
  const b=browser({research:true,amplitude:.1});b.record();await tick();b.audio(0);
  b.deviceChange();assert.equal(b.diagnostic().device_change_events,1);
  assert.ok(b.diagnostic().events.some(e=>e.event==='device_change'&&e.track.enabled&&e.context_state==='running'));
  b.audio(2);await tick();const d=b.diagnostic();assert.equal(d.category,'audio_message_continuity_failure');
  assert.equal(d.received_samples,1600);assert.equal(d.continuity.discontinuities,1);
  assert.equal(b.deviceListenerCount,0);assert.ok(b.tracks.every(t=>t.stopped));assert.equal(b.nodes['.av-research-result'].textContent,'');
});

test('processor missing-input errors release devices and never finalize a capture',async()=>{
  let seals=0;const b=browser({research:true,seal:async()=>{seals++;return {};}});
  b.record();await tick();b.audio(0);b.workletError({capture_error:'AUDIO_INPUT_MISSING',expected_frame:17600,observed_frame:17600});await tick();
  assert.equal(seals,0);assert.equal(b.diagnostic().continuity.missing_input_errors,1);
  assert.equal(b.diagnostic().received_samples,1600);assert.ok(b.tracks.every(t=>t.stopped));
});

test('reported physical pattern with sub-PCM16 values in seconds 8 and 9 is diagnosed at conversion',async()=>{
  const b=browser({research:true,seal:async()=>({status:'REJECTED',error_code:'LIVE_WINDOW_MUTED_AUDIO',
    capture_diagnostics:{accepted_audio_samples:160000,buffer:{segments:[{second:8,status:'muted_audio'},{second:9,status:'muted_audio'}]}}})});
  b.record();await tick();
  for(let i=0;i<100;i++){
    const value=i>=90?5.420763503822147e-9:i>=80?1.893336729341907e-7:.01;
    b.audio(i,i===99,value);if(i%5===4)await b.push();
  }
  await tick();await tick();const d=b.diagnostic();
  assert.deepEqual(d.zero_pcm_seconds_at_worklet_input,[8,9]);assert.deepEqual(d.zero_pcm_seconds_reported_by_server,[8,9]);
  assert.equal(d.category,'pcm16_quantized_zero_seconds');
  for(const second of [8,9]){
    const s=d.seconds[second];assert.equal(s.samples,16000);assert.equal(s.web_audio_nonzero_samples,16000);
    assert.ok(s.web_audio_rms>0);assert.ok(s.peak_abs>0);assert.equal(s.pcm16_nonzero_samples,0);
    assert.equal(s.input_classification,'nonzero_input_quantized_to_zero_pcm16');
  }
  assert.match(b.nodes['.av-msg'].textContent,/second 8/);assert.match(b.nodes['.av-msg'].textContent,/second 9/);
  assert.match(b.nodes['.av-msg'].textContent,/PCM16 resolution/);assert.ok(b.tracks.every(t=>t.stopped));
});
