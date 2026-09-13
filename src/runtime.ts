import { RuntimeController, type RuntimeMode, type RuntimeSnapshot } from './core/upscale/runtime-controller.js';
import { RuntimeSession } from './core/metrics/runtime-session.js';
import { BaselineScaler } from './core/upscale/baseline-scaler.js';
import type { VideoPipeline, PipelineGpuSample, PipelineConfiguration } from './core/pipeline.js';
import type { FrameTick, Upscaler } from './core/types.js';

export class RuntimeDriver {
  readonly controller: RuntimeController;
  readonly session: RuntimeSession;
  onChange: ((state: RuntimeSnapshot) => void) | null = null;
  onSample: ((sample: PipelineGpuSample) => void) | null = null;
  onFrame: ((tick: FrameTick) => void) | null = null;
  onConfigure: ((config: PipelineConfiguration) => void) | null = null;
  private factory: (() => Upscaler) | null = null;
  private active = false;
  private fatal = false;
  private loadGeneration = 0;
  private cadenceReady = false;
  private cadenceStarted = 0;
  private width = 0;
  private height = 0;
  private mediaTime: number | null = null;
  private readonly listeners: [EventTarget, string, EventListener][] = [];
  private readonly interval: ReturnType<typeof setInterval>;

  constructor(readonly pipeline: VideoPipeline, readonly video: HTMLVideoElement,
    private readonly timestamps: boolean, mode: RuntimeMode = 'auto',
    private readonly visible: () => boolean = () => document.visibilityState === 'visible') {
    const now = performance.now();
    this.controller = new RuntimeController({ mode, available: false, active: false }, now);
    this.session = new RuntimeSession(now);
    this.session.reset(now, this.quality(), this.loadGeneration);
    this.session.setActive(false, now);
    pipeline.onGpuSample = sample => {
      if (this.fatal || !this.active) return;
      this.session.recordGpu(sample);
      this.onSample?.(sample);
      this.apply(this.controller.record(sample, performance.now()));
    };
    pipeline.onConfiguration = config => {
      if (this.fatal) return;
      this.session.recordConfiguration(config);
      this.width = config.source.width;
      this.height = config.source.height;
      const now = performance.now();
      if (config.sourceChanged) {
        this.cadenceReady = false;
        this.cadenceStarted = this.controller.snapshot(now).activeMs;
        this.controller.setAvailable(false, now);
        this.controller.setWorkload(this.width, this.height, 60, now);
      }
      this.controller.bindGeneration(config.generation, now);
      this.onConfigure?.(config);
    };
    pipeline.onFrame = tick => {
      if (this.fatal) return;
      const now = performance.now();
      if (this.mediaTime !== null && tick.mediaTime < this.mediaTime) this.invalidate(now);
      this.mediaTime = tick.mediaTime;
      const fps = this.session.recordFrame(tick, this.quality(), this.loadGeneration,
        pipeline.currentUpscaler.neural ? 'neural' : 'baseline', video.playbackRate);
      const observed = this.session.snapshot(now);
      if (fps !== null) {
        this.controller.setWorkload(this.width, this.height, fps, now);
        this.pipeline.invalidateTiming();
        this.controller.bindGeneration(this.pipeline.timingGeneration, now);
      }
      if (!this.cadenceReady && (observed.cadence.cleanBaselineIntervals >= 30 ||
        this.controller.snapshot(now).activeMs - this.cadenceStarted >= 1500)) {
        this.cadenceReady = true;
        this.controller.setAvailable(this.timestamps && this.factory !== null, now);
      }
      this.onFrame?.(tick);
      this.apply(this.controller.snapshot(now));
    };
    for (const event of ['playing', 'pause', 'ended', 'seeking', 'seeked']) {
      this.listen(video, event, () => {
        if (event === 'seeking') this.invalidate(performance.now());
        this.syncActive();
      });
    }
    this.listen(video, 'loadstart', () => {
      this.loadGeneration++;
      this.newSource();
      this.syncActive();
    });
    this.listen(video, 'ratechange', () => this.newSource());
    this.listen(document, 'visibilitychange', () => this.syncActive());
    this.interval = setInterval(() => {
      if (this.fatal) return;
      if (pipeline.error) { this.fail('frame execution failed'); return; }
      this.apply(this.controller.tick(performance.now()));
    }, 100);
  }

  private listen(target: EventTarget, event: string, handler: EventListener): void {
    target.addEventListener(event, handler);
    this.listeners.push([target, event, handler]);
  }

  private quality() {
    if (typeof this.video.getVideoPlaybackQuality !== 'function') return null;
    const quality = this.video.getVideoPlaybackQuality();
    return { totalVideoFrames: quality.totalVideoFrames, droppedVideoFrames: quality.droppedVideoFrames,
      corruptedVideoFrames: quality.corruptedVideoFrames };
  }

  setNeuralFactory(factory: () => Upscaler): void {
    if (this.fatal) return;
    this.factory = factory;
    const now = performance.now();
    this.apply(this.controller.setAvailable(this.timestamps && this.cadenceReady, now));
  }

  setMode(mode: RuntimeMode): void {
    if (this.fatal) return;
    this.apply(this.controller.setMode(mode, performance.now()));
    this.bind();
  }

  force(on: boolean): void {
    if (!this.fatal) this.apply(this.controller.setForced(on, performance.now()));
  }

  syncActive(): void {
    if (this.fatal) return;
    const active = !this.video.paused && !this.video.ended && this.visible();
    if (active === this.active) return;
    this.active = active;
    const now = performance.now();
    this.session.setActive(active, now);
    this.apply(this.controller.setActive(active, now));
    this.bind();
    if (active) this.pipeline.start();
    else this.pipeline.stop();
  }

  private bind(now = performance.now()): void {
    this.pipeline.invalidateTiming();
    this.controller.bindGeneration(this.pipeline.timingGeneration, now);
  }

  private invalidate(now: number): void {
    this.controller.invalidate(now);
    this.session.discontinuity();
    this.bind(now);
  }

  private newSource(): void {
    if (this.fatal) return;
    const now = performance.now();
    this.mediaTime = null;
    this.cadenceReady = false;
    this.cadenceStarted = this.controller.snapshot(now).activeMs;
    this.session.sourceChanged();
    this.controller.setAvailable(false, now);
    this.controller.setWorkload(this.video.videoWidth, this.video.videoHeight, 60, now);
    this.apply(this.controller.snapshot(now));
    this.bind();
  }

  private apply(state: RuntimeSnapshot): void {
    if (this.fatal) return;
    if (state.tier === 'neural' && !this.pipeline.currentUpscaler.neural && this.factory && this.active) {
      this.pipeline.setUpscaler(this.factory());
      this.controller.bindGeneration(this.pipeline.timingGeneration, performance.now());
    } else if (state.tier === 'baseline' && this.pipeline.currentUpscaler.neural) {
      this.pipeline.setUpscaler(new BaselineScaler('catmull-rom'));
      this.controller.bindGeneration(this.pipeline.timingGeneration, performance.now());
    }
    this.onChange?.(this.controller.snapshot(performance.now()));
  }

  resetMeasurements(): void {
    const now = performance.now();
    this.pipeline.resetMeasurements();
    this.controller.bindGeneration(this.pipeline.timingGeneration, now);
    this.session.reset(now, this.quality(), this.loadGeneration, true);
  }

  snapshot() {
    const now = performance.now();
    return { controller: this.controller.snapshot(now), session: this.session.snapshot(now),
      actualTier: this.pipeline.currentUpscaler.neural ? 'neural' : 'baseline', running: this.pipeline.running };
  }

  fail(reason: string): void {
    if (this.fatal) return;
    this.fatal = true;
    const now = performance.now();
    this.pipeline.stop();
    this.pipeline.invalidateTiming();
    this.controller.fail(reason, now);
    this.session.setActive(false, now);
    clearInterval(this.interval);
    this.onChange?.(this.controller.snapshot(now));
  }

  destroy(): void {
    this.fail('disposed');
    for (const [target, event, listener] of this.listeners) target.removeEventListener(event, listener);
    this.pipeline.onGpuSample = null;
    this.pipeline.onFrame = null;
    this.pipeline.onConfiguration = null;
  }
}