// Node tests for deployment/web/live_timing.js (run: node tests/js/test_live_timing.js; also run by pytest).
const assert = require("assert");
const path = require("path");
const T = require(path.join(__dirname, "..", "..", "deployment", "web", "live_timing.js"));

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test("clock maps camera time onto the audio timeline", () => {
  // AudioContext time 2.0 s was performance time 10 000 ms; the first audio sample was frame 32 000 @ 16 kHz (2.0 s)
  const c = T.makeClock({ contextTime: 2.0, performanceTime: 10000 }, 32000, 16000);
  assert.ok(Math.abs(c.perfToTimeline(10000) - 0) < 1e-9);
  assert.ok(Math.abs(c.perfToTimeline(10500) - 0.5) < 1e-9);
  assert.strictEqual(c.frameToSample(32000 + 1600), 1600);
  assert.throws(() => T.makeClock({ contextTime: NaN, performanceTime: 0 }, 0, 16000));
});

test("frame picker keeps the frame closest to each second and emits in order", () => {
  const p = T.makeFramePicker(0.1);
  const kept = {};
  const emitted = [];
  for (let i = 0; i <= 12 * 30; i++) {
    const t = i / 30 + 0.004 * Math.sin(i * 7.3);   // 30 fps with jitter
    const k = p.offer(t);
    if (k !== null) kept[k] = t;
    emitted.push(...p.due(t));
  }
  assert.deepStrictEqual(emitted, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  for (const k of emitted) assert.ok(Math.abs(kept[k] - k) <= 1 / 60 + 0.004 + 1e-9, "second " + k);
});

test("frame picker rejects frames outside the tolerance and late frames", () => {
  const p = T.makeFramePicker(0.1);
  assert.strictEqual(p.offer(0.3), null);
  assert.strictEqual(p.offer(1.05), 1);
  assert.strictEqual(p.offer(1.07), null);          // worse than the kept one
  assert.strictEqual(p.offer(0.98), 1);             // better
  assert.deepStrictEqual(p.due(1.2), [1]);
  assert.strictEqual(p.offer(1.0), null);           // second 1 already emitted
});

test("pcm16 encoding is little-endian and clipped", () => {
  const b = T.floatToPcm16(new Float32Array([0, 1, -1, 2, -2]));
  const v = new DataView(b.buffer);
  assert.deepStrictEqual([0, 1, 2, 3, 4].map((i) => v.getInt16(2 * i, true)), [0, 32767, -32768, 32767, -32768]);
  assert.strictEqual(T.bytesToBase64(new Uint8Array([1, 2, 3])), "AQID");
});

test("audio accumulator merges contiguous blocks and refuses a discontinuity", () => {
  const a = T.makeAudioAccumulator();
  assert.ok(a.add(0, new Float32Array(128)));
  assert.ok(a.add(128, new Float32Array(128)));
  assert.strictEqual(a.add(400, new Float32Array(128)), false);
  const c = a.take();
  assert.strictEqual(c.start, 0);
  assert.strictEqual(c.samples.length, 256);
  assert.strictEqual(a.take(), null);
});

test("permission errors get specific messages", () => {
  assert.match(T.permissionMessage({ name: "NotAllowedError" }), /permission was denied/);
  assert.match(T.permissionMessage({ name: "NotFoundError" }), /No camera or microphone/);
  assert.match(T.permissionMessage({ name: "NotReadableError" }), /in use/);
  assert.match(T.permissionMessage({ name: "INSECURE_CONTEXT" }), /secure page/);
  assert.match(T.permissionMessage({ name: "NO_AUDIO_TRACK" }), /microphone audio/);
  assert.match(T.permissionMessage({}), /could not start/);
});

test("tolerance matches the server buffer", () => {
  const fs = require("fs");
  const py = fs.readFileSync(path.join(__dirname, "..", "..", "deployment", "live", "buffer.py"), "utf8");
  const m = py.match(/FRAME_TOLERANCE_S = ([0-9.]+)/);
  assert.strictEqual(Number(m[1]), T.FRAME_TOLERANCE_S);
});

let failed = 0;
for (const [name, fn] of tests) {
  try { fn(); console.log("ok   " + name); }
  catch (e) { failed++; console.log("FAIL " + name + "\n     " + e.message); }
}
process.exit(failed ? 1 : 0);
