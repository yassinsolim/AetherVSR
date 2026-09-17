import { acquireGpu, watchDeviceFailures, type GpuContext } from '../../src/core/gpu/device.js';
import { VideoPipeline, type PipelineOptions } from '../../src/core/pipeline.js';
import { BaselineScaler } from '../../src/core/upscale/baseline-scaler.js';
import { NeuralUpscaler, NEURAL_OPTIONAL_FEATURES } from '../../src/core/upscale/neural-upscaler.js';
import { loadModel } from '../../src/core/neural/model.js';
import type { RuntimeMode } from '../../src/core/upscale/runtime-controller.js';
import type { FrameTick, Upscaler } from '../../src/core/types.js';
import { RuntimeDriver } from '../../src/runtime.js';

export type DesktopState = 'empty' | 'ready' | 'playing' | 'paused' | 'unavailable' | 'error' | 'disposed';
export type DesktopSnapshot = ReturnType<DesktopSession['snapshot']>;

export interface DesktopSessionOptions {
  readonly onChange?: (snapshot: DesktopSnapshot) => void;
  readonly onFrame?: (tick: FrameTick, driver: RuntimeDriver) => void;
  readonly createPipeline?: (gpu: GpuContext, video: HTMLVideoElement, canvas: HTMLCanvasElement,
    upscaler: Upscaler, options: PipelineOptions) => VideoPipeline;
  readonly forceCopy?: boolean;
}

interface PendingOperation {
  readonly generation: number;
  promise: Promise<void>;
}

export class DesktopSession {
  private sourceGeneration = 0;
  private name: string | null = null;
  private url: string | null = null;
  private state: DesktopState = 'empty';
  private mode: RuntimeMode = 'auto';
  private ready = false;
  private frameEpoch = 0;
  private hasPlayed = false;
  private changingSource = false;
  private enhancementFailed = false;
  private error: string | null = null;
  private observerError: string | null = null;
  private notificationPending = false;
  private gpuContext: GpuContext | null = null;
  private capabilities: GpuContext['capabilities'] | null = null;
  private pipeline: VideoPipeline | null = null;
  private driver: RuntimeDriver | null = null;
  private unwatch: (() => void) | null = null;
  private setup: PendingOperation | null = null;
  private playRequest: PendingOperation | null = null;
  private readonly listeners: [string, EventListener][] = [];
  private readonly cleanupErrors: string[] = [];

  constructor(private readonly video: HTMLVideoElement, private readonly canvas: HTMLCanvasElement,
    private readonly options: DesktopSessionOptions = {}) {
    canvas.hidden = true;
    video.preload = 'metadata';
    video.autoplay = false;
    for (const event of ['loadedmetadata', 'durationchange', 'play', 'playing', 'pause', 'ended', 'seeking',
      'seeked', 'emptied', 'error', 'volumechange', 'ratechange']) {
      const listener = () => this.mediaEvent(event);
      video.addEventListener(event, listener);
      this.listeners.push([event, listener]);
    }
  }

  load(file: File): void {
    this.assertLive();
    this.sourceGeneration++;
    this.changingSource = true;
    this.playRequest = null;
    this.hide();
    this.cleanup(() => this.video.pause());
    this.releaseEnhancement();
    this.revokeUrl();
    this.hasPlayed = false;
    this.enhancementFailed = false;
    this.capabilities = null;
    this.error = null;
    this.name = file.name;
    try {
      this.url = URL.createObjectURL(file);
      this.video.src = this.url;
      this.video.load();
    } catch (error) {
      this.error = String(error);
      this.revokeUrl();
      this.cleanup(() => this.video.removeAttribute('src'));
    } finally {
      this.changingSource = false;
      this.notify();
    }
  }

  play(): Promise<void> {
    try {
      this.assertLive();
      if (this.url === null) throw new Error('Select a local video first');
    } catch (error) { return Promise.reject(error instanceof Error ? error : new Error(String(error))); }
    if (this.playRequest !== null) return this.playRequest.promise;
    if (!this.enhancementFailed) this.error = null;
    let resolveRequest!: () => void;
    let rejectRequest!: (error: unknown) => void;
    const promise = new Promise<void>((resolve, reject) => { resolveRequest = resolve; rejectRequest = reject; });
    const request: PendingOperation = { generation: this.sourceGeneration, promise };
    this.playRequest = request;
    if (this.video.paused) this.hide();
    let nativePlay: Promise<void>;
    try { nativePlay = this.video.play(); } catch (error) {
      nativePlay = Promise.reject(error instanceof Error ? error : new Error(String(error)));
    }
    const enhancement = this.current(request.generation) && this.playRequest === request
      ? this.ensureGpu() : Promise.resolve();
    void (async () => {
      try {
        try { await nativePlay; } catch (error) {
          if (!this.current(request.generation) || this.playRequest !== request) return;
          this.error = String(error);
          this.releaseEnhancement();
          throw error;
        }
        await enhancement;
        if (this.current(request.generation) && this.playRequest === request) this.driver?.syncActive();
      } finally {
        if (this.playRequest === request) {
          this.playRequest = null;
          this.notify();
        }
      }
    })().then(resolveRequest, rejectRequest);
    return request.promise;
  }

  pause(): void {
    this.assertLive();
    this.playRequest = null;
    this.video.pause();
    this.driver?.syncActive();
    this.notify();
  }

  seek(seconds: number): void {
    this.assertLive();
    this.finite(seconds);
    if (this.url === null || !Number.isFinite(this.video.duration) || this.video.duration < 0) {
      throw new RangeError('Seek requires finite media duration');
    }
    const generation = this.sourceGeneration;
    this.hide();
    this.notify();
    if (!this.current(generation)) return;
    this.video.currentTime = Math.min(this.video.duration, Math.max(0, seconds));
  }

  setVolume(volume: number): void {
    this.assertLive();
    this.finite(volume);
    this.video.volume = Math.min(1, Math.max(0, volume));
    this.notify();
  }

  setMuted(muted: boolean): void {
    this.assertLive();
    this.video.muted = muted;
    this.notify();
  }

  setRate(rate: number): void {
    this.assertLive();
    this.finite(rate);
    this.video.playbackRate = Math.min(4, Math.max(0.25, rate));
    this.notify();
  }

  setMode(mode: RuntimeMode): void {
    this.assertLive();
    if (mode !== 'auto' && mode !== 'neural' && mode !== 'baseline') throw new RangeError('Invalid runtime mode');
    if (mode === this.mode) return;
    this.mode = mode;
    this.hide();
    this.driver?.setMode(mode);
    this.notify();
  }

  get runtime(): RuntimeDriver | null { return this.driver; }
  get gpu(): GpuContext | null { return this.gpuContext; }

  private ensureGpu(): Promise<void> {
    if (this.state === 'disposed' || this.url === null || this.enhancementFailed || this.driver !== null) {
      return Promise.resolve();
    }
    if (this.setup !== null) return this.setup.promise;
    const operation: PendingOperation = { generation: this.sourceGeneration, promise: Promise.resolve() };
    this.setup = operation;
    operation.promise = this.setupGpu(operation);
    this.notify();
    return operation.promise;
  }

  private async setupGpu(operation: PendingOperation): Promise<void> {
    const current = () => this.current(operation.generation) && this.setup === operation;
    try {
      const gpu = await acquireGpu({ optionalFeatures: NEURAL_OPTIONAL_FEATURES });
      if (!current()) { this.cleanup(() => gpu.device.destroy()); return; }
      this.gpuContext = gpu;
      this.capabilities = gpu.capabilities;
      this.unwatch = watchDeviceFailures(gpu.device, message => {
        if (this.current(operation.generation) && this.gpuContext === gpu) this.failEnhancement(message);
      });
      const model = await loadModel('aethervsr://app/models/production.json');
      if (!current()) return;
      const upscaler = new BaselineScaler('catmull-rom');
      const pipelineOptions = { forceCopyImport: this.options.forceCopy ?? false };
      const pipeline = this.options.createPipeline
        ? this.options.createPipeline(gpu, this.video, this.canvas, upscaler, pipelineOptions)
        : new VideoPipeline(gpu, this.video, this.canvas, upscaler, pipelineOptions);
      if (!current()) { this.cleanup(() => pipeline.destroy()); return; }
      this.pipeline = pipeline;
      const driver = new RuntimeDriver(pipeline, this.video, gpu.capabilities.timestampQuery, this.mode);
      this.driver = driver;
      let signature = '';
      let tier = driver.snapshot().controller.tier;
      driver.onChange = state => {
        if (!this.currentRuntime(driver, operation.generation)) return;
        if (state.state === 'failed') { this.failEnhancement(state.reason); return; }
        if (state.tier !== tier) { tier = state.tier; this.hide(); }
        const next = `${state.state}:${state.tier}:${state.mode}`;
        if (next !== signature) { signature = next; this.notifySoon(); }
      };
      driver.onConfigure = () => {
        if (!this.currentRuntime(driver, operation.generation)) return;
        this.hide();
        this.notifySoon();
      };
      this.hide();
      driver.setNeuralFactory(() => new NeuralUpscaler(model));
      if (this.currentRuntime(driver, operation.generation)) driver.syncActive();
    } catch (error) {
      if (current()) this.failEnhancement(String(error));
    } finally {
      if (this.setup === operation) { this.setup = null; this.notify(); }
    }
  }

  private current(generation: number): boolean {
    return this.state !== 'disposed' && generation === this.sourceGeneration;
  }

  private currentRuntime(driver: RuntimeDriver, generation: number): boolean {
    return this.current(generation) && this.driver === driver && !this.enhancementFailed;
  }

  private hide(): void {
    this.ready = false;
    this.canvas.hidden = true;
    const epoch = ++this.frameEpoch;
    const driver = this.driver;
    if (driver === null) return;
    const generation = this.sourceGeneration;
    const pipeline = driver.pipeline;
    driver.onFrame = tick => {
      if (!this.currentRuntime(driver, generation) || epoch !== this.frameEpoch) return;
      this.options.onFrame?.(tick, driver);
      if (this.ready) return;
      if (this.video.paused || this.video.seeking || this.video.readyState < 2 || !pipeline.running || pipeline.error) return;
      if (tick.size.width !== this.video.videoWidth || tick.size.height !== this.video.videoHeight) return;
      this.ready = true;
      this.canvas.hidden = false;
      this.notifySoon();
    };
  }

  private mediaEvent(event: string): void {
    if (this.state === 'disposed' || this.changingSource) return;
    if (event === 'play' || event === 'seeking' || event === 'emptied' || event === 'error' || event === 'ratechange') this.hide();
    if (event === 'error') this.failEnhancement(this.video.error?.message || 'Native media error');
    if (event === 'pause' || event === 'ended') this.playRequest = null;
    if (event === 'playing') {
      this.hasPlayed = true;
      if (!this.enhancementFailed) this.error = null;
      void this.ensureGpu();
    }
    this.driver?.syncActive();
    this.notify();
  }

  private failEnhancement(message: string): void {
    this.enhancementFailed = true;
    this.error = message;
    this.releaseEnhancement();
    this.notify();
  }

  private releaseEnhancement(): void {
    this.setup = null;
    const driver = this.driver, pipeline = this.pipeline, gpu = this.gpuContext, unwatch = this.unwatch;
    this.driver = null;
    this.pipeline = null;
    this.gpuContext = null;
    this.unwatch = null;
    this.hide();
    if (driver !== null) {
      driver.onChange = null;
      driver.onFrame = null;
      driver.onConfigure = null;
    }
    if (unwatch !== null) this.cleanup(unwatch);
    if (driver !== null) this.cleanup(() => driver.destroy());
    if (pipeline !== null) this.cleanup(() => pipeline.destroy());
    if (gpu !== null) this.cleanup(() => gpu.device.destroy());
  }

  private notify(): void {
    try { this.options.onChange?.(this.snapshot()); } catch (error) { this.observerError = String(error); }
  }

  private notifySoon(): void {
    if (this.notificationPending || !this.options.onChange) return;
    this.notificationPending = true;
    queueMicrotask(() => {
      this.notificationPending = false;
      if (this.state !== 'disposed') this.notify();
    });
  }

  private finite(value: number): void {
    if (!Number.isFinite(value)) throw new RangeError('Value must be finite');
  }

  snapshot() {
    if (this.state !== 'disposed') {
      this.state = this.video.error !== null || (this.error !== null && !this.enhancementFailed) ? 'error'
        : this.enhancementFailed ? 'unavailable' : this.url === null ? 'empty'
        : !this.video.paused ? 'playing' : this.hasPlayed ? 'paused' : 'ready';
    }
    return {
      state: this.state,
      name: this.name,
      url: this.url,
      sourceGeneration: this.sourceGeneration,
      mode: this.mode,
      ready: this.ready,
      error: this.error ?? this.video.error?.message ?? null,
      video: { time: this.video.currentTime, duration: Number.isFinite(this.video.duration) ? this.video.duration : null,
        width: this.video.videoWidth, height: this.video.videoHeight, paused: this.video.paused,
        muted: this.video.muted, volume: this.video.volume, rate: this.video.playbackRate,
        seeking: this.video.seeking, ended: this.video.ended, readyState: this.video.readyState,
        error: this.video.error ? { code: this.video.error.code, message: this.video.error.message } : null },
      canvas: { width: this.canvas.width, height: this.canvas.height, visible: this.ready },
      runtime: this.driver?.snapshot() ?? null,
      capabilities: this.capabilities,
      pending: this.setup !== null || this.playRequest !== null,
      resources: { devices: this.gpuContext === null ? 0 : 1, pipelines: this.pipeline === null ? 0 : 1,
        drivers: this.driver === null ? 0 : 1, callbacks: this.pipeline?.running ? 1 : 0,
        objectUrls: this.url === null ? 0 : 1 },
      cleanupErrors: [...this.cleanupErrors],
      observerError: this.observerError,
    };
  }

  destroy(): void {
    if (this.state === 'disposed') return;
    this.state = 'disposed';
    this.sourceGeneration++;
    this.playRequest = null;
    this.releaseEnhancement();
    for (const [event, listener] of this.listeners.splice(0)) {
      this.cleanup(() => this.video.removeEventListener(event, listener));
    }
    this.cleanup(() => this.video.pause());
    this.revokeUrl();
    this.cleanup(() => this.video.removeAttribute('src'));
    this.cleanup(() => this.video.load());
    this.notify();
  }

  private assertLive(): void {
    if (this.state === 'disposed') throw new Error('Desktop session is disposed');
  }

  private revokeUrl(): void {
    const url = this.url;
    this.url = null;
    if (url !== null) this.cleanup(() => URL.revokeObjectURL(url));
  }

  private cleanup(action: () => void): void {
    try { action(); } catch (error) { this.cleanupErrors.push(String(error)); }
  }
}