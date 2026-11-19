const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

test('recorder averages channels and flushes the final fragment exactly once', () => {
  const messages = [];
  let Recorder;
  const sandbox = {
    Float32Array, sampleRate: 16000,
    AudioWorkletProcessor: class {
      constructor() { this.port = {postMessage: value => messages.push(value)}; }
    },
    registerProcessor: (_, implementation) => { Recorder = implementation; },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/speech_translate/web/recorder.js'), 'utf8'), sandbox);
  const recorder = new Recorder();
  recorder.process([[new Float32Array(128).fill(.2), new Float32Array(128).fill(.6)]]);
  assert.equal(messages.length, 0);
  recorder.port.onmessage({data: 'finish'});
  assert.equal(messages.length, 2);
  const samples = new Float32Array(messages[0]);
  assert.equal(samples.length, 128);
  assert.ok(samples.every(value => Math.abs(value - .4) < 1e-6));
  assert.equal(messages[1], 'flushed');
  recorder.port.onmessage({data: 'finish'});
  assert.equal(messages.length, 3);
  assert.equal(messages[2], 'flushed');
  assert.equal(recorder.process([[new Float32Array(8000).fill(.8)]]), false);
  assert.equal(messages.length, 3);
});

test('44.1 kHz render quanta produce exact half-second frames and one lossless tail', () => {
  const chunks = [];
  let Recorder;
  const sandbox = {Float32Array, sampleRate: 44100,
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage: value => chunks.push(value)}; } },
    registerProcessor: (_, implementation) => { Recorder = implementation; },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/speech_translate/web/recorder.js'), 'utf8'), sandbox);
  const recorder = new Recorder();
  const signal = Float32Array.from({length: 44603}, (_, i) => Math.sin(i / 7));
  for (let offset = 0; offset < signal.length; offset += 128) {
    recorder.process([[signal.subarray(offset, offset + 128)]]);
  }
  recorder.port.onmessage({data: 'finish'});
  assert.equal(chunks.pop(), 'flushed');
  const parts = chunks.map(buffer => new Float32Array(buffer));
  assert.deepEqual(parts.map(part => part.length), [22050, 22050, 503]);
  assert.deepEqual(parts.flatMap(part => Array.from(part)), Array.from(signal));
});

test('finish timeout releases capture and submits already captured audio', async () => {
  const elements = new Map();
  const sent = [];
  let closed = 0, removed = 0;
  const sandbox = vm.createContext({
    document: {getElementById(id) {
      if (!elements.has(id)) elements.set(id, {files: [], classList: {toggle() {}}});
      return elements.get(id);
    }},
    fetch: async () => ({ok: true}),
    WebSocket: {OPEN: 1},
    setTimeout: callback => setTimeout(callback, 0), clearTimeout,
    sent,
    fakeContext: {state: 'running', close: async () => { closed++; }},
    fakePort: {addEventListener() {}, removeEventListener() { removed++; }, postMessage() {}},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/speech_translate/web/app.js'), 'utf8'), sandbox);
  vm.runInContext(`
    socket = {readyState: WebSocket.OPEN, send: value => sent.push(value)};
    context = fakeContext;
    recorder = {port: fakePort, disconnect() {}};
  `, sandbox);
  await vm.runInContext('finishRecording()', sandbox);
  assert.equal(closed, 1);
  assert.equal(removed, 1);
  assert.equal(sent[0], '{"type":"stop"}');
  assert.match(elements.get('error').textContent, /final audio fragment may be missing/);
});
