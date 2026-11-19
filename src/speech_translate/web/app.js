const $ = id => document.getElementById(id);
const languageNames = {es:'Spanish',fr:'French',de:'German'};
let rows = [], busy = false, socket, context, media, recorder, source;
let frames = [], inFlight = false, finishing = false, stopping = false, stopSent = false;
let completed = false;
let captureForReview = null, reviewBlob = null, reviewUrl = null;
function status(text, live = false) {
  $('status').textContent = text;
  $('status-dot').classList.toggle('recording', live);
}
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function lock(value) {
  busy = value;
  $('edit-caption').disabled = value || !rows.length;
  $('mic-mode').disabled = value;
  for (const id of ['language','strategy','clean','clear','audio-file','example-choice','translate-example']) $(id).disabled = value;
  $('record').disabled = value;
  $('review-action').disabled = value || !reviewBlob;
  $('translate-file').disabled = value || !$('audio-file').files.length;
}
function discardRecording() {
  captureForReview = null;
  reviewBlob = null;
  if (reviewUrl) {
    const player = $('review-player');
    player.pause();
    player.removeAttribute('src');
    player.load();
    URL.revokeObjectURL(reviewUrl);
    reviewUrl = null;
  }
  $('recording-review').hidden = true;
  $('review-action').disabled = true;
}
function offerRecording() {
  const capture = captureForReview;
  captureForReview = null;
  if (!capture) return;
  reviewBlob = capture.toWav();
  if (!reviewBlob && !capture.exceeded) return;
  $('recording-review').hidden = false;
  $('review-description').textContent = reviewBlob
    ? 'Listen back or translate the recording again with more context.'
    : 'Playback and review are available for recordings of up to two minutes.';
  $('review-player').hidden = !reviewBlob;
  $('review-action').hidden = !reviewBlob;
  if (reviewBlob) {
    reviewUrl = URL.createObjectURL(reviewBlob);
    $('review-player').src = reviewUrl;
  }
}
function reset(keepRecording = false) {
  if (!keepRecording) discardRecording();
  rows = [];
  $('edit-caption').disabled = true;
  $('english').replaceChildren(); $('translation').replaceChildren();
  $('count').textContent = '0 segments'; $('download').disabled = true;
  $('timing').textContent = 'Captions appear one speech segment at a time.'; error('');
}
function add(row) {
  if (!row.english && !row.translation) return;
  rows.push(row);
  for (const [id, key] of [['english','english'],['translation','translation']]) {
    $(id).querySelector('.placeholder')?.remove();
    const p = document.createElement('p'); p.textContent = row[key]; $(id).append(p);
    $(id).scrollTop = $(id).scrollHeight;
  }
  $('count').textContent = `${rows.length} segment${rows.length === 1 ? '' : 's'}`; $('download').disabled = false;
  if (Number.isFinite(row.compute_seconds)) $('timing').textContent = `Last segment processed in ${row.compute_seconds.toFixed(2)} seconds · Recording and queue wait excluded`;
}
function settings() { return {language:$('language').value, strategy:$('strategy').value, clean:$('clean').checked}; }
$('edit-caption').onclick = async () => {
  try {
    const {openCaptionEditor} = await import('./caption-editor.js');
    if (busy || !rows.length) return;
    openCaptionEditor(rows, (index, english, translation) => {
      rows[index].english = english;
      rows[index].translation = translation;
      $('english').children[index].textContent = english;
      $('translation').children[index].textContent = translation;
      status('Caption corrected and translated');
    });
  } catch {
    error('The caption editor could not open. Please try again.');
  }
};
$('language').onchange = () => {
  $('target-code').textContent = $('language').value.toUpperCase();
  $('target-name').textContent = languageNames[$('language').value] + ' translation';
  $('translation').lang = $('language').value;
  reset(true); status('Waiting for input');
};
$('clear').onclick = () => { reset(); status('Waiting for input'); };
$('audio-file').onchange = () => {
  $('file-name').textContent = $('audio-file').files[0]?.name || 'Choose an audio file';
  $('translate-file').disabled = !$('audio-file').files.length;
};
async function translateRecording(useExample, review = false) {
  if (busy) return;
  let recording = review ? reviewBlob : $('audio-file').files[0];
  if (!useExample && !recording) return;
  if (!useExample && recording.size > 32 * 1024 * 1024) {
    error('Choose an audio file of 32 MB or less.');
    return;
  }
  if (review) error('');
  else reset();
  lock(true);
  const started = Date.now();
  status(review ? 'Reviewing your recording…' : useExample ? 'Opening the sample recording…' : 'Sending your recording…');
  const progress = setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    $('timing').textContent = `Waiting for captions · ${seconds}s elapsed`;
  }, 1000);
  try {
    if (useExample) {
      const audioResponse = await fetch($('example-player').src);
      if (!audioResponse.ok) throw new Error('The sample could not be loaded. Try uploading a recording.');
      recording = await audioResponse.blob();
    }
    const body = new FormData();
    body.append('audio', recording, review ? 'microphone.wav' : useExample ? 'example.wav' : recording.name);
    const options = settings();
    if (review) options.strategy = 'pause';
    for (const [key,value] of Object.entries(options)) body.append(key, value);
    status('Processing audio. The first request for a language also loads its models.');
    const response = await fetch('/api/translate/audio', {method: 'POST', body});
    const result = await response.json().catch(() => null);
    if (!response.ok || !Array.isArray(result?.segments)) {
      throw new Error(typeof result?.detail === 'string' ? result.detail : 'The server could not finish this recording. Please try again.');
    }
    if (review) {
      if (recording !== reviewBlob) return;
      const complete = row => typeof row?.english === 'string' && typeof row.translation === 'string' &&
        Boolean(row.english.trim()) === Boolean(row.translation.trim());
      if (!result.segments.every(complete) || !result.segments.some(row => row.english.trim())) {
        throw new Error('No complete review was returned. Your existing captions are still available.');
      }
      reset(true);
    }
    $('timing').textContent = '';
    result.segments.forEach(add);
    status(review ? 'Recording review complete' : rows.length ? 'Translation complete' : 'No speech detected. Try another recording.');
  } catch (e) {
    $('timing').textContent = '';
    error(e.message);
    status('Translation did not finish');
  } finally {
    clearInterval(progress);
    lock(false);
  }
}
$('translate-file').onclick = () => translateRecording(false);
$('translate-example').onclick = () => translateRecording(true);
$('review-action').onclick = () => translateRecording(false, true);
function showInputLevel(samples) {
  if (!samples.length) return;
  let squared = 0, peak = 0;
  for (const sample of samples) {
    squared += sample * sample;
    peak = Math.max(peak, Math.abs(sample));
  }
  const db = 10 * Math.log10(squared / samples.length);
  $('mic-meter').value = Math.max(-60, db);
  let reading = db < -60 ? 'Below −60 dBFS' : `${Math.round(db)} dBFS`;
  if (peak >= 0.995) reading += ' · Peak near clipping. Lower the input gain.';
  else if (db < -45) reading += ' · If you are speaking, move closer.';
  $('mic-reading').textContent = reading;
}
async function releaseAudio() {
  $('mic-level').hidden = true;
  const oldContext = context, oldMedia = media, oldSource = source, oldRecorder = recorder;
  media = source = recorder = context = null;
  oldMedia?.getTracks().forEach(track => track.stop());
  oldSource?.disconnect(); oldRecorder?.disconnect();
  if (oldContext && oldContext.state !== 'closed') await oldContext.close();
}
function pump() {
  if (!socket || socket.readyState !== WebSocket.OPEN || inFlight) return;
  if (frames.length) { socket.send(frames.shift()); inFlight = true; }
  else if (stopping && !stopSent) { socket.send(JSON.stringify({type:'stop'})); stopSent = true; }
}
async function finishRecording() {
  if (stopping || finishing) return;
  finishing = true; $('stop').disabled = true;
  // Disconnect the source first, then request the worklet's last partial frame.
  source?.disconnect();
  // A message-channel round trip confirms all final PCM messages reached the main thread.
  const activeSocket = socket;
  try {
    if (recorder) await new Promise((resolve, reject) => {
      const port = recorder.port;
      const timer = setTimeout(() => {
        port.removeEventListener('message', flushed);
        reject(new Error('Microphone did not finish. The final audio fragment may be missing.'));
      }, 3000);
      function flushed(event) {
        if (event.data !== 'flushed') return;
        clearTimeout(timer); port.removeEventListener('message', flushed); resolve();
      }
      port.addEventListener('message', flushed);
      port.postMessage('finish');
    });
    if (socket === activeSocket) offerRecording();
  } catch (e) {
    if (socket === activeSocket) {
      captureForReview = null;
      error(e.message);
    }
  }
  if (socket !== activeSocket) return;
  await releaseAudio();
  if (socket !== activeSocket) return;
  stopping = true; status('Finishing the remaining caption…'); pump();
}
$('record').onclick = async () => {
  if (busy) return;
  $('example-player').pause();
  reset(); lock(true); status('Connecting to microphone…');
  finishing = stopping = stopSent = inFlight = false; frames = [];
  completed = false;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error('This browser does not support recording. Use a microphone-enabled browser via localhost or HTTPS.');
    const speechMode = $('mic-mode').value === 'speech';
    media = await navigator.mediaDevices.getUserMedia({
      audio: {channelCount: {ideal: 1}, echoCancellation: {ideal: speechMode},
        noiseSuppression: {ideal: speechMode}, autoGainControl: {ideal: speechMode}},
      video: false,
    });
    context = new AudioContext(); await context.resume();
    await context.audioWorklet.addModule('/assets/recorder.js');
    recorder = new AudioWorkletNode(context, 'pcm-recorder');
    source = context.createMediaStreamSource(media);
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/live`);
    const recordingSocket = socket;
    socket.onopen = () => socket.send(JSON.stringify({...settings(), sample_rate:context.sampleRate}));
    socket.onmessage = async ({data}) => {
      const message = JSON.parse(data);
      if (message.type === 'ready') {
        $('mic-level').hidden = false;
        $('mic-meter').value = -60;
        $('mic-reading').textContent = 'Waiting for audio';
        captureForReview = new MicrophoneRecording(context.sampleRate);
        source.connect(recorder); recorder.connect(context.destination);
        $('stop').hidden = false; $('stop').disabled = false; $('record').hidden = true;
        status($('strategy').value === 'pause' ? 'Listening · Captions update after a pause' : 'Listening · Captions update at each interval', true);
      } else if (message.type === 'segment') add(message);
      else if (message.type === 'ack') { inFlight = false; pump(); }
      else if (message.type === 'error') { error(message.message); socket.close(); }
      else if (message.type === 'done') { completed = true; status(rows.length ? 'Recording translation complete' : 'No speech detected'); socket.close(); }
    };
    recorder.port.onmessage = ({data}) => {
      if (socket !== recordingSocket || !(data instanceof ArrayBuffer)) return;
      showInputLevel(new Float32Array(data));
      captureForReview?.append(data);
      frames.push(data);
      if (frames.length > 60 && !stopping) { error('Processing cannot keep up with recording. Capture has stopped while existing audio is processed.'); finishRecording(); }
      pump();
    };
    socket.onerror = () => error('Connection interrupted. Reconnect and start a new recording.');
    socket.onclose = async () => {
      if (socket !== recordingSocket) return;
      captureForReview = null;
      await releaseAudio(); lock(false); $('record').hidden = false; $('stop').hidden = true;
      if (!completed) status('Recording interrupted. Some audio may not have been translated.');
      socket = null; frames = [];
    };
  } catch (e) {
    await releaseAudio(); error(e.message); status('Unable to start recording'); lock(false);
  }
};
$('stop').onclick = finishRecording;
$('download').onclick = () => {
  const text = rows.map((r,i) => `${i+1}\n${r.english}\n${r.translation}`).join('\n\n');
  const url = URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'bilingual-subtitles.txt'; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
};
fetch('/api/health').then(r => { if (!r.ok) throw new Error(); }).catch(() => { error('The service is unavailable. Refresh the page to retry.'); });
