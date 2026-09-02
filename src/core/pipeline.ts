import { FrameImporter } from './acquisition/frame-importer.js';
import { VideoFrameSource, type FrameClockKind, type PlaybackQuality } from './acquisition/video-source.js';
import type { GpuContext } from './gpu/device.js';
import { GpuTimer } from './metrics/gpu-timer.js';
import { RateMeter, SampleWindow } from './metrics/stats.js';
import { CanvasTarget } from './present/canvas-target.js';
import type { FrameTextureKind, FrameTick, Size, Upscaler } from './types.js';

const RATE_WINDOW_MS = 1000;
const SAMPLE_CAPACITY = 240;

/** Immutable snapshot for the diagnostic overlay and benchmark exports. */
export interface PipelineStats {
  readonly running: boolean;
  readonly clock: FrameClockKind;
  readonly importPath: FrameTextureKind;
  readonly upscalerId: string;
  readonly upscalerLabel: string;
  readonly neural: boolean;

  readonly sourceSize: Size;
  readonly targetSize: Size;
  readonly scaleFactor: number;

  /**
   * Frames the compositor presented per second over the trailing 1 s. Useful
   * live; too noisy to publish.
   */
  readonly sourceFps: number;
  /** Frames this pipeline processed per second over the trailing 1 s. */
  readonly renderFps: number;
  /**
   * Mean rates over the whole interval since the last reset. These are the
   * figures a benchmark should quote: a trailing-1 s window sampled at an
   * arbitrary instant can differ from a run's actual mean by several frames
   * per second.
   */
  readonly meanSourceFps: number;
  readonly meanRenderFps: number;
  /** Milliseconds of measurement since the last reset. */
  readonly elapsedMs: number;

  readonly framesRendered: number;
  /** Presented frames that never reached a callback, so were never upscaled. */
  readonly framesSkipped: number;
  readonly quality: PlaybackQuality;

  /**
   * Main-thread time per frame: the import call, command recording and
   * submission. This is not GPU time and not the whole frame's cost.
   */
  readonly cpuFrameMs: Aggregate;
  /** Measured GPU execution time of the upscale pass; null when unsupported. */
  readonly gpuPassMs: Aggregate | null;
  /** Delay between the UA presenting a frame and our callback running. */
  readonly callbackLatencyMs: Aggregate;
  /**
   * UA-reported decode *latency* (submit-to-ready, includes queueing), or null
   * when the browser never reported it. Not a per-frame cost; see
   * {@link FrameTick.decodeLatencyMs}.
   */
  readonly decodeLatencyMs: Aggregate | null;
}

export interface PipelineOptions {
  /**
   * Ignore `importExternalTexture()` and use the `copyExternalImageToTexture()`
   * path even where `importExternalTexture()` is available.
   *
   * The fallback exists for browsers that lack external textures, which means
   * it is the path least likely to be exercised on a developer's machine. This
   * flag makes it directly benchmarkable and testable on hardware that does
   * not need it.
   */
  readonly forceCopyImport?: boolean;
}

export interface Aggregate {
  readonly mean: number;
  readonly p50: number;
  readonly p95: number;
  readonly max: number;
  readonly samples: number;
}

const EMPTY_AGGREGATE: Aggregate = { mean: NaN, p50: NaN, p95: NaN, max: NaN, samples: 0 };

function summarise(window: SampleWindow): Aggregate {
  if (window.size === 0) return EMPTY_AGGREGATE;
  return {
    mean: window.mean(),
    p50: window.quantile(0.5),
    p95: window.quantile(0.95),
    max: window.max(),
    samples: window.size,
  };
}

const ZERO_QUALITY: PlaybackQuality = {
  totalVideoFrames: 0,
  droppedVideoFrames: 0,
  corruptedVideoFrames: 0,
};

/**
 * Decoder counters accumulated since a baseline.
 *
 * `getVideoPlaybackQuality()` counts from media load and resets only when the
 * element reloads, so reporting it raw next to a freshly reset frame count
 * compares two different time ranges.
 */
function qualityDelta(now: PlaybackQuality, baseline: PlaybackQuality): PlaybackQuality {
  return {
    totalVideoFrames: Math.max(0, now.totalVideoFrames - baseline.totalVideoFrames),
    droppedVideoFrames: Math.max(0, now.droppedVideoFrames - baseline.droppedVideoFrames),
    corruptedVideoFrames: Math.max(0, now.corruptedVideoFrames - baseline.corruptedVideoFrames),
  };
}

/**
 * Wires acquisition -> import -> upscale -> presentation together and owns the
 * per-frame hot path.
 *
 * Deliberate properties of `onTick`:
 * - no `await`, so the frame never spans a task boundary (a `GPUExternalTexture`
 *   would expire);
 * - no pixel readback of any kind;
 * - the only per-frame allocations are the ones WebGPU mandates: a command
 *   encoder, a swap-chain texture view, and — on the external path — a bind
 *   group, because external textures are single-frame-valid by specification.
 */
export class VideoPipeline {
  private readonly source: VideoFrameSource;
  private readonly importer: FrameImporter;
  private readonly target: CanvasTarget;
  private readonly timer: GpuTimer | null;

  private readonly sourceRate = new RateMeter(RATE_WINDOW_MS);
  private readonly renderRate = new RateMeter(RATE_WINDOW_MS);
  private readonly cpuFrame = new SampleWindow(SAMPLE_CAPACITY);
  private readonly gpuPass = new SampleWindow(SAMPLE_CAPACITY);
  private readonly callbackLatency = new SampleWindow(SAMPLE_CAPACITY);
  private readonly decode = new SampleWindow(SAMPLE_CAPACITY);

  private upscaler: Upscaler;
  private configuredSource: Size = { width: 0, height: 0 };
  private framesRendered = 0;
  private framesSkipped = 0;
  /**
   * Presented frames observed since reset, accumulated from `presentedDelta`.
   *
   * Not derivable as rendered + skipped: `presentedDelta` is 0 when the UA's
   * counter rewinds on a loop or seek, and the rAF fallback can report 0 when
   * no new frame was decoded, so that sum would overstate the presented rate.
   */
  private framesPresented = 0;
  /**
   * The opening tick's contribution, excluded from mean-rate numerators.
   *
   * `windowStart` is the opening tick's own timestamp, so that tick bounds the
   * window rather than occurring within it. Counting it would report N events
   * over N-1 intervals — 70 fps for seven ideal 60 Hz ticks spanning 100 ms.
   * This matches `RateMeter`, which subtracts its oldest sample for the same
   * reason.
   */
  private openingRendered = 0;
  private openingPresented = 0;
  /**
   * Start of the measurement window, or null while waiting for the first frame.
   *
   * Deliberately *not* set at reset time: a reset is issued before a clip is
   * loaded and before playback begins, and counting that dead time would
   * silently depress every mean frame rate we publish. The window opens on the
   * first processed tick instead.
   */
  private windowStart: number | null = null;
  /** Cumulative decoder counters at the last reset; stats report deltas. */
  private qualityBaseline: PlaybackQuality = ZERO_QUALITY;
  /** Media-load generation the baseline was taken in. */
  private qualityGeneration = 0;
  private lastError: unknown = null;

  constructor(
    private readonly gpu: GpuContext,
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    upscaler: Upscaler,
    options: PipelineOptions = {},
  ) {
    this.upscaler = upscaler;
    this.source = new VideoFrameSource(video);
    this.importer = new FrameImporter(
      gpu.device,
      video,
      gpu.capabilities.externalTexture && !options.forceCopyImport,
    );
    this.target = new CanvasTarget(canvas);
    this.timer = gpu.capabilities.timestampQuery
      ? new GpuTimer(gpu.device, (ms) => this.gpuPass.push(ms))
      : null;
  }

  get running(): boolean {
    return this.source.running;
  }

  /** Most recent hot-path exception, surfaced instead of being swallowed. */
  get error(): unknown {
    return this.lastError;
  }

  start(): void {
    if (this.source.running) return;
    this.source.start((tick) => this.onTick(tick));
  }

  stop(): void {
    this.source.stop();
  }

  /**
   * Swaps the processing stage without touching acquisition or presentation.
   * This is the seam a neural backend will arrive through; exercising it at
   * runtime is how we keep it honest.
   */
  setUpscaler(next: Upscaler): void {
    if (next === this.upscaler) return;
    this.upscaler.destroy();
    this.upscaler = next;
    // Force reconfiguration on the next tick.
    this.configuredSource = { width: 0, height: 0 };
    this.resetMeasurements();
  }

  get currentUpscaler(): Upscaler {
    return this.upscaler;
  }

  resetMeasurements(): void {
    this.sourceRate.reset();
    this.renderRate.reset();
    this.cpuFrame.reset();
    this.gpuPass.reset();
    this.callbackLatency.reset();
    this.decode.reset();
    this.framesRendered = 0;
    this.framesSkipped = 0;
    this.framesPresented = 0;
    this.openingRendered = 0;
    this.openingPresented = 0;
    // `getVideoPlaybackQuality()` counters are cumulative from media load and
    // cannot be zeroed, so snapshot them and report deltas. Without this the
    // reported drop count silently includes warm-up and any earlier playback,
    // which makes it incomparable with the frame counts beside it.
    this.qualityBaseline = this.source.quality();
    this.qualityGeneration = this.source.loadGeneration;
    this.windowStart = null;
    // Discard readbacks still in flight from the window being replaced.
    this.timer?.newEpoch();
  }

  /**
   * Decoder counters since the last reset.
   *
   * `getVideoPlaybackQuality()` counts from media load, and loading a new clip
   * silently zeroes it. A stale baseline would then clamp every count to zero
   * until the new clip overtook the old one's totals, so a counter that has
   * gone backwards is treated as a reload and the baseline is dropped.
   */
  private qualitySinceReset(): PlaybackQuality {
    // Keyed on the element's load generation rather than on noticing the
    // counter go backwards: a new clip can overtake the old total between two
    // 4 Hz polls, and the rewind would then never be observed at all.
    if (this.source.loadGeneration !== this.qualityGeneration) {
      this.qualityBaseline = ZERO_QUALITY;
      this.qualityGeneration = this.source.loadGeneration;
    }
    return qualityDelta(this.source.quality(), this.qualityBaseline);
  }

  /**
   * Frames per second averaged over the whole interval since reset. Returns 0
   * for a window too short to be meaningful rather than a huge spike.
   */
  private meanRate(frames: number, opening: number, nowMs: number): number {
    const elapsed = this.elapsed(nowMs);
    if (!(elapsed > 0)) return 0;
    return ((frames - opening) / elapsed) * 1000;
  }

  /** Milliseconds of measurement so far; 0 before the first frame arrives. */
  private elapsed(nowMs: number): number {
    return this.windowStart === null ? 0 : Math.max(0, nowMs - this.windowStart);
  }

  destroy(): void {
    this.stop();
    this.upscaler.destroy();
    this.importer.destroy();
    this.timer?.destroy();
    this.target.unconfigure();
  }

  private onTick(tick: FrameTick): void {
    if (tick.size.width === 0 || tick.size.height === 0) return;
    try {
      this.ensureConfigured(tick.size);

      if (this.windowStart === null) {
        this.windowStart = tick.now;
        this.openingRendered = 1;
        this.openingPresented = tick.presentedDelta;
      }
      this.sourceRate.mark(tick.now, tick.presentedDelta);
      this.renderRate.mark(tick.now);
      this.framesPresented += tick.presentedDelta;
      if (tick.presentedDelta > 1) this.framesSkipped += tick.presentedDelta - 1;
      this.callbackLatency.push(tick.now - tick.presentationTime);
      if (tick.decodeLatencyMs !== null) this.decode.push(tick.decodeLatencyMs);

      // Brackets everything this frame costs the main thread: the import call,
      // command recording, and submission. Not "command recording only".
      const frameStart = performance.now();
      const frame = this.importer.acquire(tick.size);
      const timing = this.timer?.begin() ?? null;
      const encoder = this.gpu.device.createCommandEncoder({ label: 'aethervsr:frame' });

      this.upscaler.encode({ encoder, frame, target: this.target.currentView(), timing });

      this.timer?.end(encoder);
      this.gpu.device.queue.submit([encoder.finish()]);
      this.timer?.afterSubmit();

      this.cpuFrame.push(performance.now() - frameStart);
      this.framesRendered++;
      this.lastError = null;
    } catch (err) {
      // A slot claimed by `begin()` is still marked busy if the frame threw
      // before `afterSubmit()`; releasing it keeps the pool from draining.
      this.timer?.abort();
      // A throwing frame loop that keeps rescheduling would spam the console
      // forever; record it and stop so the failure is visible and diagnosable.
      this.lastError = err;
      this.stop();
      throw err;
    }
  }

  private ensureConfigured(source: Size): void {
    if (this.configuredSource.width === source.width && this.configuredSource.height === source.height) {
      return;
    }
    const scale = this.upscaler.scaleFactor;
    const target: Size = { width: source.width * scale, height: source.height * scale };
    const format = this.gpu.capabilities.preferredCanvasFormat;

    this.target.configure(this.gpu.device, target, format);
    this.upscaler.configure({
      device: this.gpu.device,
      source,
      target,
      targetFormat: format,
      sourceKind: this.importer.kind,
    });
    this.configuredSource = source;
  }

  stats(nowMs: number): PipelineStats {
    return {
      running: this.source.running,
      clock: this.source.clock,
      importPath: this.importer.kind,
      upscalerId: this.upscaler.id,
      upscalerLabel: this.upscaler.label,
      neural: this.upscaler.neural,
      sourceSize: this.configuredSource,
      targetSize: this.target.size,
      scaleFactor: this.upscaler.scaleFactor,
      sourceFps: this.sourceRate.rate(nowMs),
      renderFps: this.renderRate.rate(nowMs),
      meanSourceFps: this.meanRate(this.framesPresented, this.openingPresented, nowMs),
      meanRenderFps: this.meanRate(this.framesRendered, this.openingRendered, nowMs),
      elapsedMs: this.elapsed(nowMs),
      framesRendered: this.framesRendered,
      framesSkipped: this.framesSkipped,
      quality: this.qualitySinceReset(),
      cpuFrameMs: summarise(this.cpuFrame),
      gpuPassMs: this.timer ? summarise(this.gpuPass) : null,
      callbackLatencyMs: summarise(this.callbackLatency),
      // Null means the browser never reported a decode latency, which must not
      // read as "decoding was free".
      decodeLatencyMs: this.decode.size > 0 ? summarise(this.decode) : null,
    };
  }
}
