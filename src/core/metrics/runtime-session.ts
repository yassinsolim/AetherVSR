import type { PlaybackQuality } from '../acquisition/video-source.js';
import type { PipelineConfiguration, PipelineGpuSample } from '../pipeline.js';
import type { FrameTick } from '../types.js';

export type { PlaybackQuality } from '../acquisition/video-source.js';

export interface RuntimeSummary {
  readonly count: number;
  readonly sum: number | null;
  readonly mean: number | null;
  readonly max: number | null;
  readonly p50: number | null;
  readonly p90: number | null;
  readonly p95: number | null;
  readonly overflowCount: number;
  readonly rejectedCount: number;
  readonly units: 'ms';
  readonly binWidthMs: 0.05;
  readonly ceilingMs: 200;
  readonly quantileScope: string;
}

const BINS_PER_MS = 20;
const FINITE_BINS = 4000;
const CADENCE_WINDOW = 30;
const CADENCES = [24, 25, 30, 50, 60, 120];

class Histogram {
  private readonly bins = new Float64Array(FINITE_BINS + 1);
  private count = 0;
  private sum = 0;
  private max: number | null = null;
  private rejectedCount = 0;

  reset(): void {
    this.bins.fill(0);
    this.count = 0;
    this.sum = 0;
    this.max = null;
    this.rejectedCount = 0;
  }

  add(ms: number): void {
    if (!Number.isFinite(ms) || ms < 0) {
      this.rejectedCount++;
      return;
    }
    const index = Math.min(FINITE_BINS, Math.max(0, Math.ceil(ms * BINS_PER_MS) - 1));
    this.bins[index] = this.bins[index]! + 1;
    this.count++;
    this.sum += ms;
    this.max = this.max === null ? ms : Math.max(this.max, ms);
  }

  private quantile(fraction: number): number | null {
    if (this.count === 0) return null;
    const rank = Math.ceil(this.count * fraction);
    let cumulative = 0;
    for (let index = 0; index < FINITE_BINS; index++) {
      cumulative += this.bins[index]!;
      if (cumulative >= rank) return (index + 1) / BINS_PER_MS;
    }
    return null;
  }

  snapshot(): RuntimeSummary {
    return {
      count: this.count,
      sum: this.count === 0 ? null : this.sum,
      mean: this.count === 0 ? null : this.sum / this.count,
      max: this.max,
      p50: this.quantile(0.5),
      p90: this.quantile(0.9),
      p95: this.quantile(0.95),
      overflowCount: this.bins[FINITE_BINS]!,
      rejectedCount: this.rejectedCount,
      units: 'ms',
      binWidthMs: 0.05,
      ceilingMs: 200,
      quantileScope: 'Nearest-rank upper-bin boundary; +/-0.05 ms quantization within 0..200 ms. Null when empty or rank is in overflow (>200 ms). Count/sum/mean/max use raw samples.',
    };
  }
}

type RuntimeGpuSample = Pick<PipelineGpuSample, 'ms' | 'neural' | 'generation'> &
  Partial<Omit<PipelineGpuSample, 'ms' | 'neural' | 'generation'>>;
type Tier = 'neural' | 'baseline';

function validQuality(quality: PlaybackQuality): boolean {
  return [quality.totalVideoFrames, quality.droppedVideoFrames, quality.corruptedVideoFrames]
    .every((value) => Number.isSafeInteger(value) && value >= 0);
}

function validTime(now: number): void {
  if (!Number.isFinite(now)) throw new RangeError('Time must be finite milliseconds.');
}

export class RuntimeSession {
  private active = true;
  private activeSince: number;
  private lastNow: number;
  private elapsedActiveMs = 0;
  private framesRendered = 0;
  private framesPresented = 0;
  private framesSkipped = 0;
  private decoderDrops = 0;
  private decoderFrames = 0;
  private decoderCorrupted = 0;
  private qualityAvailable = false;
  private qualityObserved = false;
  private qualitySamples = 0;
  private qualityMissingSamples = 0;
  private qualityRejectedSamples = 0;
  private lastQuality: PlaybackQuality | null = null;
  private loadGeneration = 0;
  private readonly neuralGpu = new Histogram();
  private readonly baselineGpu = new Histogram();
  private readonly callbackLatency = new Histogram();
  private readonly configuration = new Histogram();
  private configurationCount = 0;
  private latestConfiguration: PipelineConfiguration | null = null;
  private sourceFpsEstimate = 60;
  private readonly intervals = new Float64Array(CADENCE_WINDOW);
  private intervalCount = 0;
  private intervalCursor = 0;
  private cleanBaselineIntervals = 0;
  private previousMediaTime: number | null = null;
  private previousTier: Tier | null = null;
  private previousRate: number | null = null;

  constructor(now = 0) {
    validTime(now);
    this.activeSince = now;
    this.lastNow = now;
  }

  reset(now: number, quality: PlaybackQuality | null, loadGeneration: number, preserveCadence = false): void {
    const cadence = this.sourceFpsEstimate;
    validTime(now);
    this.activeSince = now;
    this.lastNow = now;
    this.elapsedActiveMs = 0;
    this.framesRendered = 0;
    this.framesPresented = 0;
    this.framesSkipped = 0;
    this.decoderDrops = 0;
    this.decoderFrames = 0;
    this.decoderCorrupted = 0;
    this.qualityAvailable = quality !== null && validQuality(quality);
    this.qualityObserved = this.qualityAvailable;
    this.qualitySamples = 0;
    this.qualityMissingSamples = 0;
    this.qualityRejectedSamples = 0;
    this.lastQuality = this.qualityAvailable && quality !== null ? { ...quality } : null;
    this.loadGeneration = loadGeneration;
    this.neuralGpu.reset();
    this.baselineGpu.reset();
    this.callbackLatency.reset();
    this.configuration.reset();
    this.configurationCount = 0;
    this.latestConfiguration = null;
    this.sourceChanged();
    if (preserveCadence) this.sourceFpsEstimate = cadence;
  }

  private checkTime(now: number): void {
    validTime(now);
    if (now < this.lastNow) throw new RangeError('Session timestamps must be monotonic.');
  }

  setActive(active: boolean, now: number): void {
    this.checkTime(now);
    this.lastNow = now;
    if (this.active === active) return;
    if (this.active) this.elapsedActiveMs += now - this.activeSince;
    this.activeSince = now;
    this.active = active;
    this.clearCadenceEvidence();
  }

  recordFrame(
    tick: FrameTick,
    quality: PlaybackQuality | null,
    loadGeneration: number,
    tier: Tier,
    playbackRate: number,
    observedAt = tick.now,
  ): number | null {
    this.checkTime(observedAt);
    if (!Number.isSafeInteger(tick.presentedDelta) || tick.presentedDelta < 0) {
      throw new RangeError('Presented delta must be a nonnegative integer.');
    }
    this.lastNow = observedAt;
    this.framesRendered++;
    this.framesPresented += tick.presentedDelta;
    this.framesSkipped += Math.max(0, tick.presentedDelta - 1);
    const cleanQuality = this.recordQuality(quality, loadGeneration);
    const frameMetadata = Number.isFinite(tick.presentationTime) &&
      Number.isFinite(tick.expectedDisplayTime) &&
      (tick.presentationTime !== tick.now || tick.expectedDisplayTime !== tick.now);
    if (frameMetadata) this.callbackLatency.add(Math.max(0, tick.now - tick.presentationTime));
    if (!this.active || !frameMetadata || loadGeneration !== this.loadGeneration ||
      !Number.isFinite(playbackRate) || playbackRate <= 0 ||
      !Number.isFinite(tick.mediaTime) || tick.presentedDelta === 0) {
      this.clearCadenceEvidence();
      return null;
    }
    const previousMediaTime = this.previousMediaTime;
    const previousTier = this.previousTier;
    const previousRate = this.previousRate;
    this.previousMediaTime = tick.mediaTime;
    this.previousTier = tier;
    this.previousRate = playbackRate;
    const mediaDelta = previousMediaTime === null ? 0 : tick.mediaTime - previousMediaTime;
    if (mediaDelta <= 0 || mediaDelta > 0.2 || previousRate !== playbackRate) {
      this.clearCadenceWindow();
      return null;
    }
    const interval = mediaDelta / tick.presentedDelta / playbackRate;
    if (!Number.isFinite(interval) || interval <= 0) {
      this.clearCadenceWindow();
      return null;
    }
    this.intervals[this.intervalCursor] = interval;
    this.intervalCursor = (this.intervalCursor + 1) % CADENCE_WINDOW;
    this.intervalCount = Math.min(CADENCE_WINDOW, this.intervalCount + 1);
    this.cleanBaselineIntervals = tier === 'baseline' && previousTier === 'baseline' &&
      cleanQuality && tick.presentedDelta === 1
      ? Math.min(CADENCE_WINDOW, this.cleanBaselineIntervals + 1) : 0;
    if (this.intervalCount < CADENCE_WINDOW) return null;
    const sorted = this.intervals.slice().sort();
    const median = (sorted[14]! + sorted[15]!) / 2;
    const measuredFps = 1 / median;
    if (!Number.isFinite(measuredFps)) return null;
    const nearest = CADENCES.reduce((best, candidate) =>
      Math.abs(candidate - measuredFps) < Math.abs(best - measuredFps) ? candidate : best);
    const estimate = Math.abs(nearest - measuredFps) / nearest <= 0.08
      ? nearest : Math.max(1, Math.round(measuredFps));
    if (estimate === this.sourceFpsEstimate ||
      (estimate < this.sourceFpsEstimate && this.cleanBaselineIntervals < CADENCE_WINDOW)) return null;
    this.sourceFpsEstimate = estimate;
    return estimate;
  }

  private recordQuality(quality: PlaybackQuality | null, generation: number): boolean {
    const sameLoad = generation === this.loadGeneration;
    const previousAvailable = this.qualityAvailable;
    if (generation > this.loadGeneration) {
      this.loadGeneration = generation;
      this.lastQuality = { totalVideoFrames: 0, droppedVideoFrames: 0, corruptedVideoFrames: 0 };
      this.sourceChanged();
    }
    this.qualityAvailable = false;
    if (quality === null) {
      this.qualityMissingSamples++;
      return false;
    }
    const previous = this.lastQuality;
    if (generation !== this.loadGeneration || !validQuality(quality) ||
      (previous !== null && (quality.totalVideoFrames < previous.totalVideoFrames ||
        quality.droppedVideoFrames < previous.droppedVideoFrames ||
        quality.corruptedVideoFrames < previous.corruptedVideoFrames))) {
      this.qualityRejectedSamples++;
      return false;
    }
    if (previous !== null) {
      this.decoderFrames += quality.totalVideoFrames - previous.totalVideoFrames;
      this.decoderDrops += quality.droppedVideoFrames - previous.droppedVideoFrames;
      this.decoderCorrupted += quality.corruptedVideoFrames - previous.corruptedVideoFrames;
    }
    this.qualitySamples++;
    this.qualityAvailable = true;
    this.qualityObserved = true;
    this.lastQuality = { ...quality };
    return sameLoad && previousAvailable && previous !== null &&
      quality.droppedVideoFrames === previous.droppedVideoFrames &&
      quality.corruptedVideoFrames === previous.corruptedVideoFrames;
  }

  recordGpu(sample: RuntimeGpuSample): void {
    (sample.neural ? this.neuralGpu : this.baselineGpu).add(sample.ms);
  }

  recordConfiguration(config: PipelineConfiguration): void {
    this.configurationCount++;
    this.configuration.add(config.configureMs);
    this.latestConfiguration = { ...config, source: { ...config.source }, target: { ...config.target } };
    if (config.sourceChanged) this.sourceChanged();
  }

  sourceChanged(): void {
    this.sourceFpsEstimate = 60;
    this.clearCadenceEvidence();
  }

  discontinuity(): void {
    this.clearCadenceEvidence();
  }

  private clearCadenceWindow(): void {
    this.intervalCount = 0;
    this.intervalCursor = 0;
    this.cleanBaselineIntervals = 0;
  }

  private clearCadenceEvidence(): void {
    this.clearCadenceWindow();
    this.previousMediaTime = null;
    this.previousTier = null;
    this.previousRate = null;
  }

  snapshot(now: number) {
    this.checkTime(now);
    const activeMs = this.elapsedActiveMs + (this.active ? now - this.activeSince : 0);
    return {
      scope: {
        session: 'Since constructor/explicit reset; stage and source changes preserve totals.',
        activeTime: 'Wall milliseconds from reset, stopped only by setActive(false); stalls count. Reset preserves active state; construction starts active.',
        frameRates: 'All recordFrame calls divided by activeMs, including the opening frame; null at zero activeMs. Caller records successful renders.',
        decoder: 'Observed monotonic load-counter deltas: reset/first exposure baselines current load; newer loads start at zero. Gaps are recovered when counters return, not imputed. Null until exposed; qualityAvailable describes the latest reading.',
        gpu: 'All finite nonnegative raw durations received since reset, cold samples included, classified by sample.neural. No generation, age, warmup or controller filtering.',
        callbackLatency: 'max(0, tick.now - tick.presentationTime), only with non-synthetic frame metadata; not GPU or decode time.',
        configuration: 'All reported synchronous configureMs durations since reset; excludes first-submit/driver work.',
        cadence: 'Median of 30 latest valid mediaDelta/presentedDelta/currentPlaybackRate intervals; no seek/loop/pause/rate-change crossing. Lower FPS requires 30 consecutive clean baseline-to-baseline intervals with exposed monotonic quality and no loss. Synthetic rAF metadata is excluded.',
      },
      active: this.active,
      activeMs,
      framesRendered: this.framesRendered,
      framesPresented: this.framesPresented,
      framesSkipped: this.framesSkipped,
      decoderDrops: this.qualityObserved ? this.decoderDrops : null,
      decoderFrames: this.qualityObserved ? this.decoderFrames : null,
      decoderCorrupted: this.qualityObserved ? this.decoderCorrupted : null,
      meanRenderedFps: activeMs > 0 ? this.framesRendered * 1000 / activeMs : null,
      meanPresentedFps: activeMs > 0 ? this.framesPresented * 1000 / activeMs : null,
      qualityAvailable: this.qualityAvailable,
      qualityObserved: this.qualityObserved,
      qualitySamples: this.qualitySamples,
      qualityMissingSamples: this.qualityMissingSamples,
      qualityRejectedSamples: this.qualityRejectedSamples,
      loadGeneration: this.loadGeneration,
      gpu: { neural: this.neuralGpu.snapshot(), baseline: this.baselineGpu.snapshot() },
      callbackLatency: this.callbackLatency.snapshot(),
      configuration: this.configuration.snapshot(),
      configurationCount: this.configurationCount,
      generation: this.latestConfiguration?.generation ?? null,
      source: this.latestConfiguration === null ? null : { ...this.latestConfiguration.source },
      target: this.latestConfiguration === null ? null : { ...this.latestConfiguration.target },
      sourceFpsEstimate: this.sourceFpsEstimate,
      cadence: { observations: this.intervalCount, cleanBaselineIntervals: this.cleanBaselineIntervals },
    };
  }
}