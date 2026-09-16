class TimingDetector extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.generation = options.processorOptions.generation;
    this.first = null; this.last = null; this.previous = 0; this.crossings = [];
  }
  process(inputs, outputs) {
    const input = inputs[0], output = outputs[0];
    for (let channel = 0; channel < output.length; channel++) {
      if (input[channel]) output[channel].set(input[channel]); else output[channel].fill(0);
    }
    const mono = input[0];
    if (!mono) return true;
    for (let offset = 0; offset < mono.length; offset++) {
      const sample = mono[offset], frame = currentFrame + offset;
      if (Math.abs(sample) > 0.1) { if (this.first === null) this.first = frame; this.last = frame; }
      if (this.first !== null && frame - this.first >= sampleRate * 0.01 && frame - this.first < sampleRate * 0.08 && this.previous <= 0 && sample > 0) {
        this.crossings.push(frame - 1 - this.previous / (sample - this.previous));
      }
      if (this.last !== null && frame - this.last >= sampleRate * 0.02) {
        const count = this.crossings.length;
        const hz = count > 1 ? (count - 1) * sampleRate / (this.crossings[count - 1] - this.crossings[0]) : null;
        this.port.postMessage({ generation: this.generation, firstSample: this.first, lastSample: this.last, sampleRate, hz,
          id: hz === null ? null : Math.round((hz - 600) / 20) });
        this.first = this.last = null; this.crossings = [];
      }
      this.previous = sample;
    }
    return true;
  }
}
registerProcessor('m1010-timing', TimingDetector);