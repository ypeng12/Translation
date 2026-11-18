const $ = id => document.getElementById(id);
const languageNames = {es:'Spanish',fr:'French',de:'German'};
let rows = [], busy = false, socket, context, media, recorder, source;
let frames = [], inFlight = false, finishing = false, stopping = false, stopSent = false;
function status(text, live = false) {
  $('status').textContent = text;
  $('status-dot').classList.toggle('recording', live);
}
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function lock(value) {
  busy = value;
  for (const id of ['language','strategy','clean','clear','audio-file']) $(id).disabled = value;
  $('record').disabled = value;
  $('translate-file').disabled = value || !$('audio-file').files.length;
}
function reset() {
  rows = [];
  $('english').replaceChildren(); $('translation').replaceChildren();
  $('count').textContent = '0 caption segments'; $('download').disabled = true;
  $('timing').textContent = 'Segment-level captions · The final fragment is submitted when recording stops'; error('');
}
function add(row) {
  if (!row.english && !row.translation) return;
  rows.push(row);
  for (const [id, key] of [['english','english'],['translation','translation']]) {
    $(id).querySelector('.placeholder')?.remove();
    const p = document.createElement('p'); p.textContent = row[key]; $(id).append(p);
    $(id).scrollTop = $(id).scrollHeight;
  }
  $('count').textContent = `${rows.length} caption segment${rows.length === 1 ? '' : 's'}`; $('download').disabled = false;
  if (row.compute_seconds !== undefined) $('timing').textContent = `Last segment processed in ${row.compute_seconds.toFixed(2)} seconds · Recording and queue wait excluded`;
}
function settings() { return {language:$('language').value, strategy:$('strategy').value, clean:$('clean').checked}; }
$('language').onchange = () => {
  $('target-code').textContent = $('language').value.toUpperCase();
  $('target-name').textContent = languageNames[$('language').value] + ' translation';
  reset(); status('Waiting for input');
};
$('clear').onclick = () => { reset(); status('Waiting for input'); };
$('audio-file').onchange = () => {
  $('file-name').textContent = $('audio-file').files[0]?.name || 'Choose an audio file';
  $('translate-file').disabled = !$('audio-file').files.length;
};
$('translate-file').onclick = async () => {
  const file = $('audio-file').files[0]; if (!file || busy) return;
  if (file.size > 32 * 1024 * 1024) { error('Please select an audio file smaller than 32 MB.'); return; }
  reset(); lock(true); status('Transcribing and translating. The first request may take longer…');
  const body = new FormData(); body.append('audio', file);
  for (const [key,value] of Object.entries(settings())) body.append(key, value);
  try {
    const response = await fetch('/api/translate/audio', {method:'POST',body});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Audio processing failed. Check the file format or server log.');
    result.segments.forEach(add);
    status(rows.length ? 'Translation complete' : 'No speech detected. Please try another recording.');
  } catch (e) { error(e.message); status('Processing did not finish'); }
  finally { lock(false); }
};
async function releaseAudio() {
  media?.getTracks().forEach(track => track.stop());
  source?.disconnect(); recorder?.disconnect();
  if (context && context.state !== 'closed') await context.close();
  media = source = recorder = context = null;
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
  if (recorder) await new Promise(resolve => {
    recorder.port.addEventListener('message', function flushed(event) {
      if (event.data === 'flushed') { recorder?.port.removeEventListener('message', flushed); resolve(); }
    });
    recorder.port.postMessage('finish');
  });
  await releaseAudio(); stopping = true; status('Finishing the remaining caption…'); pump();
}
$('record').onclick = async () => {
  if (busy) return;
  reset(); lock(true); status('Connecting to microphone…');
  finishing = stopping = stopSent = inFlight = false; frames = [];
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error('This browser does not support recording. Use a microphone-enabled browser via localhost or HTTPS.');
    media = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});
    context = new AudioContext(); await context.resume();
    await context.audioWorklet.addModule('/assets/recorder.js');
    recorder = new AudioWorkletNode(context, 'pcm-recorder');
    source = context.createMediaStreamSource(media);
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/live`);
    socket.onopen = () => socket.send(JSON.stringify({...settings(), sample_rate:context.sampleRate}));
    socket.onmessage = async ({data}) => {
      const message = JSON.parse(data);
      if (message.type === 'ready') {
        source.connect(recorder); recorder.connect(context.destination);
        $('stop').hidden = false; $('stop').disabled = false; $('record').hidden = true;
        status('Listening · Captions update after a pause', true);
      } else if (message.type === 'segment') add(message);
      else if (message.type === 'ack') { inFlight = false; pump(); }
      else if (message.type === 'error') { error(message.message); socket.close(); }
      else if (message.type === 'done') { status(rows.length ? 'Recording translation complete' : 'No speech detected'); socket.close(); }
    };
    recorder.port.onmessage = ({data}) => {
      if (!(data instanceof ArrayBuffer)) return;
      frames.push(data);
      if (frames.length > 60 && !stopping) { error('Processing cannot keep up with recording. Capture has stopped while existing audio is processed.'); finishRecording(); }
      pump();
    };
    socket.onerror = () => error('Connection interrupted. Confirm that the local server is running.');
    socket.onclose = async () => {
      await releaseAudio(); lock(false); $('record').hidden = false; $('stop').hidden = true;
      if (!stopSent) status('Recording interrupted. Unsubmitted audio was not saved.');
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
fetch('/api/health').then(r => { if (!r.ok) throw new Error(); }).catch(() => { error('Service is unavailable. Start the local backend first.'); });
