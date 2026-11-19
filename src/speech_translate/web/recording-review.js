/* PCM from the microphone worklet, retained only in memory for optional review. */
class MicrophoneRecording {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    this.sampleCount = 0;
    this.parts = [];
    this.exceeded = false;
  }

  append(frame) {
    if (this.exceeded) return;
    const samples = new Float32Array(frame);
    if (this.sampleCount + samples.length > this.sampleRate * 120) {
      // A partial file cannot safely replace captions for the entire session.
      this.parts = [];
      this.sampleCount = 0;
      this.exceeded = true;
      return;
    }
    const pcm = new DataView(new ArrayBuffer(samples.length * 2));
    for (let index = 0; index < samples.length; index++) {
      const sample = Math.min(1, Math.max(-1, samples[index]));
      pcm.setInt16(index * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
    }
    this.parts.push(pcm.buffer);
    this.sampleCount += samples.length;
  }

  toWav() {
    if (this.exceeded || !this.sampleCount) return null;
    const header = new DataView(new ArrayBuffer(44));
    const writeLabel = (offset, label) => {
      for (let i = 0; i < label.length; i++) header.setUint8(offset + i, label.charCodeAt(i));
    };
    const bytes = this.sampleCount * 2;
    writeLabel(0, 'RIFF');
    header.setUint32(4, 36 + bytes, true);
    writeLabel(8, 'WAVE');
    writeLabel(12, 'fmt ');
    header.setUint32(16, 16, true);
    header.setUint16(20, 1, true); // Linear PCM
    header.setUint16(22, 1, true); // Mono
    header.setUint32(24, this.sampleRate, true);
    header.setUint32(28, this.sampleRate * 2, true);
    header.setUint16(32, 2, true);
    header.setUint16(34, 16, true);
    writeLabel(36, 'data');
    header.setUint32(40, bytes, true);
    return new Blob([header.buffer, ...this.parts], {type: 'audio/wav'});
  }
}
