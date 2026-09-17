declare const currentFrame: number;
declare const sampleRate: number;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options: AudioWorkletNodeOptions);
}
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void;

type Completion = 'RENDER_TARGET_REACHED' | 'WATCHDOG_ABORT' | 'AUDIO_CONTEXT_SUSPENDED' |
  'PROCESSOR_ERROR' | 'DISCONTINUITY' | 'OVERFLOW' | 'HOST_ABORT' | 'MALFORMED_RESULT';

export class RenderClockRecorder extends AudioWorkletProcessor {
  private readonly pcm = new Float32Array(sampleRate * 80);
  private readonly blocks = new Uint32Array(sampleRate * 80);
  private readonly requestedEndFrame: number;
  private firstFrame: number | null = null;
  private actualEndFrame: number | null = null;
  private processedSamples = 0;
  private processedBlocks = 0;
  private lastHeartbeatEnd: number | null = null;
  private heartbeatCount = 0;
  private state: 'RECORDING' | Completion = 'RECORDING';
  private overflow = false;
  private discontinuity = false;

  constructor(options: AudioWorkletNodeOptions) {
    super(options);
    const settings: unknown = options.processorOptions;
    const end = settings && typeof settings === 'object' && 'targetEndFrame' in settings ? settings.targetEndFrame : undefined;
    if (typeof end !== 'number' || !Number.isSafeInteger(end) || end <= currentFrame || sampleRate !== 48000) {
      throw new Error('A future integer render-frame target at 48000 Hz is required');
    }
    this.requestedEndFrame = end;
    this.port.onmessage = (event: MessageEvent<unknown>) => {
      if (this.state !== 'RECORDING') return;
      const message = event.data;
      const reason = message && typeof message === 'object' && 'reason' in message ? message.reason : undefined;
      const command = message && typeof message === 'object' && 'type' in message ? message.type : undefined;
      if (command === 'snapshot-and-abort') {
        this.finish(reason === 'WATCHDOG_ABORT' || reason === 'AUDIO_CONTEXT_SUSPENDED' || reason === 'PROCESSOR_ERROR' ? reason : 'HOST_ABORT');
      } else if (message === 'finish' || command === 'finish') {
        this.finish('HOST_ABORT');
      } else {
        this.finish('MALFORMED_RESULT');
      }
    };
  }

  private finish(reason: Completion): void {
    if (this.state !== 'RECORDING') return;
    this.state = reason;
    this.port.postMessage({ type: 'render-recording', completionReason: reason,
      requestedEndFrame: this.requestedEndFrame, actualObservedEndFrame: this.actualEndFrame,
      firstFrame: this.firstFrame, processedSamples: this.processedSamples, processedBlocks: this.processedBlocks,
      sampleRate, blockLengths: this.blocks.buffer, heartbeatCount: this.heartbeatCount,
      overflow: this.overflow, discontinuity: this.discontinuity, pcm: this.pcm.buffer },
    [this.pcm.buffer, this.blocks.buffer]);
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const input = inputs[0], output = outputs[0];
    const length = output?.[0]?.length ?? input?.[0]?.length ?? 0;
    if (input?.some(channel => channel.length !== length) || output?.some(channel => channel.length !== length)) {
      this.finish('MALFORMED_RESULT'); return true;
    }
    if (output) for (let channel = 0; channel < output.length; channel++) {
      const destination = output[channel], source = input?.[channel];
      if (destination) { if (source) destination.set(source); else destination.fill(0); }
    }
    if (this.state !== 'RECORDING') return true;
    if (!length) return true;
    const start = currentFrame, end = start + length;
    if (this.actualEndFrame !== null && start !== this.actualEndFrame) {
      this.discontinuity = true; this.finish('DISCONTINUITY'); return true;
    }
    if (this.processedSamples + length > this.pcm.length || this.processedBlocks >= this.blocks.length) {
      this.overflow = true; this.finish('OVERFLOW'); return true;
    }
    this.firstFrame ??= start;
    const mono = input?.[0];
    if (mono && mono.length !== length) { this.finish('MALFORMED_RESULT'); return true; }
    if (mono) this.pcm.set(mono, this.processedSamples);
    else this.pcm.fill(0, this.processedSamples, this.processedSamples + length);
    this.blocks[this.processedBlocks++] = length;
    this.processedSamples += length;
    this.actualEndFrame = end;
    if (end >= this.requestedEndFrame) { this.finish('RENDER_TARGET_REACHED'); return true; }
    if (this.lastHeartbeatEnd === null || end - this.lastHeartbeatEnd >= sampleRate / 4) {
      this.lastHeartbeatEnd = end; this.heartbeatCount++;
      this.port.postMessage({ type: 'render-heartbeat', currentFrame: start, actualObservedEndFrame: end,
        sampleRate, processedBlocks: this.processedBlocks, processedSamples: this.processedSamples,
        blockLength: length, state: this.state, heartbeatOrdinal: this.heartbeatCount });
    }
    return true;
  }
}

registerProcessor('m1010ri-render', RenderClockRecorder);