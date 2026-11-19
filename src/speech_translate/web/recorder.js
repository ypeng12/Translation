class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Float32Array(Math.round(sampleRate / 2));
    this.used = 0;
    this.finished = false;
    this.port.onmessage = ({data}) => {
      if (data === 'finish') {
        this.finished = true;
        this.flush();
        this.port.postMessage('flushed');
      }
    };
  }
  flush() {
    if (!this.used) return;
    const capacity = this.samples.length;
    const audio = this.used === capacity ? this.samples : this.samples.slice(0, this.used);
    this.port.postMessage(audio.buffer, [audio.buffer]);
    this.used = 0;
    if (!this.finished) this.samples = new Float32Array(capacity);
  }
  process(inputs) {
    if (this.finished) return false;
    const input = inputs[0];
    if (input?.length && input[0].length) {
      for (let i = 0; i < input[0].length; i++) {
        let sum = 0;
        for (const channel of input) sum += channel[i];
        this.samples[this.used++] = sum / input.length;
        if (this.used === this.samples.length) this.flush();
      }
    }
    return true;
  }
}
registerProcessor('pcm-recorder', PCMRecorder);
