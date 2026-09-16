declare const currentFrame: number;
declare const sampleRate: number;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options: AudioWorkletNodeOptions);
}
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void;

class ReplayAudioRecorder extends AudioWorkletProcessor {
  private readonly pcm: Float32Array;
  private used = 0;
  private firstFrame: number | null = null;
  private active = false;
  private consumed = false;
  private overflow = false;
  private discontinuity = false;
  private readonly blockLengths = new Set<number>();

  constructor(options: AudioWorkletNodeOptions) {
    super(options);
    this.pcm = new Float32Array(sampleRate * 80);
    this.port.onmessage = (event: MessageEvent<unknown>) => {
      if (event.data === 'start') {
        if (this.consumed) throw new Error('Audio recorder is single-use');
        this.consumed = true;
        this.active = true;
      } else if (event.data === 'finish') {
        this.consumed = true;
        this.active = false;
        this.port.postMessage({ type: 'audio-recording', sampleRate, firstFrame: this.firstFrame, samples: this.used,
          overflow: this.overflow, discontinuity: this.discontinuity, blockLengths: [...this.blockLengths],
          pcm: this.pcm.buffer }, [this.pcm.buffer]);
      }
    };
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const input = inputs[0], output = outputs[0];
    if (output) for (let channel = 0; channel < output.length; channel++) {
      const destination = output[channel], source = input?.[channel];
      if (destination) { if (source) destination.set(source); else destination.fill(0); }
    }
    const samples = input?.[0];
    if (!this.active || !samples) return true;
    this.firstFrame ??= currentFrame;
    if (currentFrame !== this.firstFrame + this.used) this.discontinuity = true;
    this.blockLengths.add(samples.length);
    if (this.used + samples.length > this.pcm.length) { this.overflow = true; this.active = false; return true; }
    this.pcm.set(samples, this.used); this.used += samples.length;
    return true;
  }
}

registerProcessor('m1010r-audio', ReplayAudioRecorder);
export {};