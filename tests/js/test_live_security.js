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
  for (const name of ['preview','start','record','stop','check','state','msg','grid','diag','meter-fill','contract']) node('.av-' + name);
  const video = nodes['.av-preview']; video.play = async () => {}; video.videoWidth = 640; video.videoHeight = 480;
  video.cancelVideoFrameCallback = () => {}; video.requestVideoFrameCallback = () => 1;
  const tracks = ['audio','video'].map(kind => ({kind, stopped: false, stop() {this.stopped = true;},
    addEventListener() {}, getSettings: () => ({width: 640, height: 480})}));
  const stream = {getTracks: () => tracks, getAudioTracks: () => [tracks[0]], getVideoTracks: () => [tracks[1]]};
  let worklet, interval; const stops = [], pushes = [];
  class Context {
    constructor() {this.sampleRate = 16000; this.state = 'running'; this.audioWorklet = {addModule: async () => {}};}
    createMediaStreamSource() {return {connect() {}, disconnect() {}};}
    createGain() {return {gain: {}, connect() {}, disconnect() {}};}
    getOutputTimestamp() {return {contextTime: 1, performanceTime: 1000};}
    close() {this.state = 'closed'; return Promise.resolve();}
  }
  const server = {
    live_start: options.start || (async () => ({status: 'STARTED', session: 'server-session', tab_key: 'server-tab-key'})),
    live_push: data => {pushes.push(data); return options.push ? options.push(data) : Promise.resolve({status:'CAPTURING'});},
    live_stop: data => {stops.push(data); return Promise.resolve({status:'STOPPED'});},
    live_check: async () => ({status:'NOT_READY', error_code:'LIVE_BUFFER_NOT_READY'})
  };
  const context = {
    AVTiming: {...T, withTimeout: (p, ms) => T.withTimeout(p, ms === 30000 ? 100 : 10)},
    element: {querySelector: selector => nodes[selector]}, server,
    window: {isSecureContext: true, addEventListener() {}},
    navigator: {mediaDevices: {getUserMedia: () => options.permission || Promise.resolve(stream)}},
    AudioContext: Context,
    AudioWorkletNode: class {constructor() {this.port = {}; worklet = this;} connect() {} disconnect() {}},
    URL: {createObjectURL: () => 'blob:unit', revokeObjectURL() {}}, Blob,
    HTMLVideoElement: {prototype: {requestVideoFrameCallback() {}}},
    performance: {now: () => 1000},
    document: {hidden:false, createElement: () => ({dataset:{},getContext: () => ({drawImage() {}})})},
    setInterval: fn => {interval = fn; return 1;}, clearInterval() {},
    setTimeout, clearTimeout, cancelAnimationFrame() {}, requestAnimationFrame: () => 1,
  };
  vm.runInNewContext(fs.readFileSync(path.join(web, 'live_capture.js'), 'utf8'), context);
  return {nodes, tracks, stream, stops, pushes,
    start: () => nodes['.av-start'].click(), stop: () => nodes['.av-stop'].click(),
    push: () => interval(),
    audio: i => worklet.port.onmessage({data: {frame:16000 + i * 1600, samples:new Float32Array(1600)}})};
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
