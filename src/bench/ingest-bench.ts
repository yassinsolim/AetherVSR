import { FrameImporter } from '../core/acquisition/frame-importer.js';
import { VideoFrameSource } from '../core/acquisition/video-source.js';
import type { GpuContext } from '../core/gpu/device.js';
import { ExternalTextureIngest, type IngestFormat } from '../core/ingest/external-texture-ingest.js';
import { GpuTimer } from '../core/metrics/gpu-timer.js';
import { SampleWindow } from '../core/metrics/stats.js';
import { CanvasTarget } from '../core/present/canvas-target.js';
import type { FrameTexture, Size } from '../core/types.js';
import { BaselineScaler } from '../core/upscale/baseline-scaler.js';
import type { BaselineFilter } from '../core/upscale/baseline.wgsl.js';

/** How the frame reaches the scaler. */
export type IngestMode = 'direct' | 'ingest';

export interface IngestBenchConfig {
  readonly mode: IngestMode;
  readonly filter: BaselineFilter;
  readonly ingestFormat: IngestFormat;
}

export interface StageStats {
  readonly mean: number;
  readonly p50: number;
  readonly p95: number;
  readonly max: number;
  readonly samples: number;
}

export interface IngestBenchStats {
  readonly mode: IngestMode;
  readonly filter: BaselineFilter;
  readonly ingestFormat: IngestFormat;
  readonly sourceSize: Size;
  readonly targetSize: Size;
  readonly framesRendered: number;
  readonly elapsedMs: number;
  readonly meanRenderFps: number;
  /** GPU time of the ingest pass. Null in `direct` mode, which has no ingest. */
  readonly ingestMs: StageStats | null;
  /** GPU time of the upscale pass. */
  readonly scaleMs: StageStats;
  /** Sum of the two measured passes, or NaN when nothing was measured. */
  readonly totalGpuMs: number;
  /** Main-thread time: import call, command recording, submit. */
  readonly cpuFrameMs: StageStats;
}

const EMPTY: StageStats = { mean: NaN, p50: NaN, p95: NaN, max: NaN, samples: 0 };

function summarise(w: SampleWindow): StageStats {
  if (w.size === 0) return EMPTY;
  return { mean: w.mean(), p50: w.quantile(0.5), p95: w.quantile(0.95), max: w.max(), samples: w.size };
}

/**
 * Measures the cost of converting a `GPUExternalTexture` into an ordinary
 * texture, separately from the cost of the pass that consumes it.
 *
 * The experiment is a 2x2: `{direct, ingest} x {bilinear, catmull-rom}`.
 * Because bilinear takes one tap and Catmull-Rom takes nine, differencing the
 * two filters within a mode isolates the marginal cost of a single tap in that
 * sampling domain, while differencing the modes isolates the ingest pass. That
 * is what makes this a controlled measurement rather than two numbers that
 * bracket different work — the flaw that made the Milestone 1 comparison
 * suggestive but not conclusive.
 *
 * Both passes are timed with independent `timestamp-query` sets in the same
 * command encoder, so no CPU-side timing is involved.
 */
export class IngestBench {
  private readonly source: VideoFrameSource;
  private readonly importer: FrameImporter;
  private readonly target: CanvasTarget;
  private readonly ingest = new ExternalTextureIngest();
  private readonly ingestTimer: GpuTimer | null;
  private readonly scaleTimer: GpuTimer | null;

  private readonly ingestWindow = new SampleWindow(240);
  private readonly scaleWindow = new SampleWindow(240);
  private readonly cpuWindow = new SampleWindow(240);

  private scaler: BaselineScaler;
  private config: IngestBenchConfig;
  private configuredSource: Size = { width: 0, height: 0 };
  private framesRendered = 0;
  private windowStart: number | null = null;
  private openingRendered = 0;
  private lastError: unknown = null;

  constructor(
    private readonly gpu: GpuContext,
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    config: IngestBenchConfig,
  ) {
    this.config = config;
    this.scaler = new BaselineScaler(config.filter);
    this.source = new VideoFrameSource(video);
    this.importer = new FrameImporter(gpu.device, video, gpu.capabilities.externalTexture);
    this.target = new CanvasTarget(canvas);
    const timestamps = gpu.capabilities.timestampQuery;
    this.ingestTimer = timestamps ? new GpuTimer(gpu.device, (ms) => this.ingestWindow.push(ms)) : null;
    this.scaleTimer = timestamps ? new GpuTimer(gpu.device, (ms) => this.scaleWindow.push(ms)) : null;
  }

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

  /** Switches configuration and clears all measurements. */
  reconfigure(config: IngestBenchConfig): void {
    this.config = config;
    this.scaler.destroy();
    this.scaler = new BaselineScaler(config.filter);
    this.configuredSource = { width: 0, height: 0 };
    this.reset();
  }

  reset(): void {
    this.ingestWindow.reset();
    this.scaleWindow.reset();
    this.cpuWindow.reset();
    this.framesRendered = 0;
    this.windowStart = null;
    this.openingRendered = 0;
    this.ingestTimer?.newEpoch();
    this.scaleTimer?.newEpoch();
  }

  destroy(): void {
    this.stop();
    this.scaler.destroy();
    this.ingest.destroy();
    this.importer.destroy();
    this.ingestTimer?.destroy();
    this.scaleTimer?.destroy();
    this.target.unconfigure();
  }

  private onTick(tick: { now: number; size: Size }): void {
    if (tick.size.width === 0 || tick.size.height === 0) return;
    try {
      this.ensureConfigured(tick.size);

      const frameStart = performance.now();
      const frame = this.importer.acquire(tick.size);
      const encoder = this.gpu.device.createCommandEncoder({ label: 'aethervsr:bench' });

      let scalerInput: FrameTexture = frame;
      let ingestTiming = null;
      // Only claim a timestamp slot when the ingest stage will actually record
      // a pass; a passed-through `sampled` frame writes none, and resolving an
      // unwritten slot fabricates a duration.
      if (this.config.mode === 'ingest' && ExternalTextureIngest.writesPass(frame)) {
        ingestTiming = this.ingestTimer?.begin() ?? null;
        const view = this.ingest.encode(encoder, frame, ingestTiming);
        scalerInput = { kind: 'sampled', view };
      }

      const scaleTiming = this.scaleTimer?.begin() ?? null;
      this.scaler.encode({
        encoder,
        frame: scalerInput,
        target: this.target.currentView(),
        timing: scaleTiming,
      });

      if (ingestTiming) this.ingestTimer?.end(encoder);
      this.scaleTimer?.end(encoder);
      this.gpu.device.queue.submit([encoder.finish()]);
      if (ingestTiming) this.ingestTimer?.afterSubmit();
      this.scaleTimer?.afterSubmit();

      this.cpuWindow.push(performance.now() - frameStart);
      this.framesRendered++;
      if (this.windowStart === null) {
        this.windowStart = tick.now;
        this.openingRendered = 1;
      }
      this.lastError = null;
    } catch (err) {
      this.ingestTimer?.abort();
      this.scaleTimer?.abort();
      this.lastError = err;
      this.stop();
      throw err;
    }
  }

  private ensureConfigured(source: Size): void {
    if (this.configuredSource.width === source.width && this.configuredSource.height === source.height) {
      return;
    }
    const target: Size = { width: source.width * 2, height: source.height * 2 };
    const format = this.gpu.capabilities.preferredCanvasFormat;
    this.target.configure(this.gpu.device, target, format);

    if (this.config.mode === 'ingest') {
      this.ingest.configure({ device: this.gpu.device, size: source, format: this.config.ingestFormat });
    }

    this.scaler.configure({
      device: this.gpu.device,
      source,
      target,
      targetFormat: format,
      // In ingest mode the scaler reads an ordinary texture, which is the
      // whole point of the experiment.
      sourceKind: this.config.mode === 'ingest' ? 'sampled' : this.importer.kind,
    });
    this.configuredSource = source;
  }

  stats(nowMs: number): IngestBenchStats {
    const elapsed = this.windowStart === null ? 0 : Math.max(0, nowMs - this.windowStart);
    const ingest = this.config.mode === 'ingest' ? summarise(this.ingestWindow) : null;
    const scale = summarise(this.scaleWindow);
    return {
      mode: this.config.mode,
      filter: this.config.filter,
      ingestFormat: this.config.ingestFormat,
      sourceSize: this.configuredSource,
      targetSize: this.target.size,
      framesRendered: this.framesRendered,
      elapsedMs: elapsed,
      meanRenderFps:
        elapsed > 0 ? (Math.max(0, this.framesRendered - this.openingRendered) / elapsed) * 1000 : 0,
      ingestMs: ingest,
      scaleMs: scale,
      // NaN, not 0, when the upscale pass was never timed: AGENTS.md §2
      // requires "not measured" to stay distinguishable from a real zero.
      totalGpuMs:
        scale.samples === 0
          ? Number.NaN
          : (ingest && ingest.samples > 0 ? ingest.mean : 0) + scale.mean,
      cpuFrameMs: summarise(this.cpuWindow),
    };
  }
}
