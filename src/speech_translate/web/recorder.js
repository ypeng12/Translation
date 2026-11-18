class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frames = [];
    this.length = 0;
    this.port.onmessage = ({data}) => {
      if (data === 'finish') { this.flush(); this.port.postMessage('flushed'); }
    };
  }
  flush() {
    if (!this.length) return;
    const audio = new Float32Array(this.length);
    let offset = 0;
    for (const frame of this.frames) { audio.set(frame, offset); offset += frame.length; }
    this.port.postMessage(audio.buffer, [audio.buffer]);
    this.frames = []; this.length = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (input?.length && input[0].length) {
      const mono = new Float32Array(input[0].length);
      for (const channel of input) for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / input.length;
      this.frames.push(mono); this.length += mono.length;
      if (this.length >= sampleRate / 2) this.flush();
    }
    return true;
  }
}
registerProcessor('pcm-recorder', PCMRecorder);
