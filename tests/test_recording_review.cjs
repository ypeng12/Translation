const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const web = path.join(__dirname, '../src/speech_translate/web');
function runFile(name, context) {
  vm.runInContext(fs.readFileSync(path.join(web, name), 'utf8'), context);
}
function context(extras = {}) {
  const env = vm.createContext({Blob, ArrayBuffer, DataView, Float32Array, ...extras});
  runFile('recording-review.js', env);
  return env;
}

test('review WAV contains signed mono PCM and the last fractional frame', async () => {
  const capture = vm.runInContext('new MicrophoneRecording(44100)', context());
  capture.append(Float32Array.of(-2, -.5, 0).buffer);
  capture.append(Float32Array.of(.5, 2).buffer);
  const bytes = Buffer.from(await capture.toWav().arrayBuffer());
  assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
  assert.equal(bytes.readUInt32LE(4), bytes.length - 8);
  assert.equal(bytes.toString('ascii', 8, 12), 'WAVE');
  assert.equal(bytes.readUInt32LE(24), 44100);
  assert.equal(bytes.readUInt16LE(22), 1);
  assert.equal(bytes.readUInt16LE(34), 16);
  assert.equal(bytes.readUInt32LE(40), 10);
  assert.deepEqual([44, 46, 48, 50, 52].map(i => bytes.readInt16LE(i)),
    [-32768, -16384, 0, 16384, 32767]);
});

test('retains exactly two minutes and releases all review audio on overflow', () => {
  const capture = vm.runInContext('new MicrophoneRecording(8000)', context());
  assert.equal(capture.toWav(), null);
  capture.append(new Float32Array(8000 * 120).buffer);
  assert.equal(capture.toWav().size, 44 + 2 * 8000 * 120);
  capture.append(Float32Array.of(.25).buffer);
  capture.append(Float32Array.of(.1).buffer);
  assert.equal(capture.exceeded, true);
  assert.equal(capture.toWav(), null);
  assert.equal(capture.parts.length, 0);
});

test('review replaces captions only on success and clear revokes playback', async () => {
  const elements = new Map();
  const calls = [], revoked = [];
  let result, ok, pending;
  const document = {createElement: () => ({}), getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      files: [], classList: {toggle() {}}, value: '', replaceChildren() {},
      pause() {}, removeAttribute() {}, load() {}, querySelector() { return null; }, append() {},
    });
    return elements.get(id);
  }};
  const env = context({
    document,
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL: url => revoked.push(url)},
    fetch: async (url, req) => {
      if (url === '/api/health') return {ok: true};
      calls.push(req.body);
      if (pending) await pending;
      return {ok, json: async () => result};
    },
    FormData, setInterval, clearInterval,
  });
  runFile('app.js', env);
  vm.runInContext(`
    rows = [{english: 'Saved', translation: 'Guardado'}];
    captureForReview = new MicrophoneRecording(16000);
    captureForReview.append(Float32Array.of(.5).buffer);
    offerRecording();
  `, env);
  document.getElementById('strategy').value = 'fixed';
  ok = false; result = {detail: 'Unavailable'};
  await elements.get('review-action').onclick();
  assert.equal(vm.runInContext('rows[0].english', env), 'Saved');
  assert.equal(calls[0].get('strategy'), 'pause');
  assert.equal(calls[0].get('audio').size, 46);
  ok = true; result = {segments: []};
  await elements.get('review-action').onclick();
  assert.equal(vm.runInContext('rows[0].translation', env), 'Guardado');
  assert.deepEqual(revoked, []);
  let resolve;
  pending = new Promise(callback => { resolve = callback; });
  result = {segments: [{english: 'Reviewed', translation: 'Revisado'}]};
  const processing = elements.get('review-action').onclick();
  assert.equal(vm.runInContext('rows[0].english', env), 'Saved');
  assert.equal(elements.get('review-action').disabled, true);
  resolve();
  await processing;
  assert.equal(vm.runInContext('rows.length', env), 1);
  assert.equal(vm.runInContext('rows[0].english', env), 'Reviewed');
  assert.equal(elements.get('review-action').disabled, false);
  assert.equal(elements.get('review-player').src, 'blob:test');
  elements.get('clear').onclick();
  assert.deepEqual(revoked, ['blob:test']);
  assert.equal(vm.runInContext('reviewBlob', env), null);
});
