import { afterEach, describe, expect, it, vi } from 'vitest';
import type { RuntimeSnapshot } from '../src/core/upscale/runtime-controller.js';

function checkpoint(at: number, activeMs: number, rendered: number, presented = rendered) {
  return { at, runtime: {
    controller: { transitionCount: 0, probeCount: 0, failedProbeCount: 0, fallbackMs: 0 },
    actualTier: 'neural',
    session: { activeMs, framesRendered: rendered, framesPresented: presented, framesSkipped: presented - rendered,
      loadGeneration: 3, decoderDrops: 0, decoderFrames: rendered, decoderCorrupted: 0,
      qualitySamples: rendered, qualityMissingSamples: 0, qualityRejectedSamples: 0,
      gpu: { neural: { count: rendered }, baseline: { count: 0 } } },
  }, decoderQuality: [100 + presented, 2, 0] as (number | null)[], pipeline: { clock: 'rvfc' } };
}

function frame(observedAt: number, sequence: number, metadataAt = observedAt) {
  return [metadataAt, 0, 1, 4, 5, 'neural', 100 + sequence, 2, 0, 3, 7, sequence, observedAt, 3];
}

function fixture(drainMs = 0) {
  return { startedAt: 1000, endedAt: 2000, plannedEndAt: 2000, resetRequestedAt: 999,
    endOfDrainAt: 2000 + drainMs, sampleCountAtBoundary: 1,
    initial: checkpoint(1000, 50, 5), boundary: checkpoint(2000, 1050, 7),
    drained: checkpoint(2000 + drainMs, 1050 + drainMs, 7),
    raw: { frames: [frame(1100, 6), frame(1900, 7)],
      samples: [{ submittedAt: 1100, resolvedAt: 1105, sequence: 6, ms: 2, neural: true },
        { submittedAt: 1900, resolvedAt: 2000 + drainMs, sequence: 7, ms: 4, neural: true }],
      configurations: [] as { at: number; configureMs: number }[],
      transitions: [] as { id: number; atMs: number }[], refresh: [] as number[] },
  };
}

interface Summary {
  measured: boolean;
  elapsed: { sessionActiveMs: number; endedAt: number; boundaryCapturedAt: number; drainMs: number };
  counts: { rendered: number; presented: number; skipped: number; rawFrames: number;
    rawGpu: number; sessionGpu: number; gpuReceivedDuringDrain: number;
    decoderFrames: number | null; decoderDrops: number | null; decoderCorrupted: number | null };
  rates: { renderedFps: number | null; presentedFps: number | null };
  gpu: { all: Distribution; neural: Distribution; baseline: Distribution };
  rawGaps: { missingSequences: number[]; internalGpuSequenceGaps: number[][]; duplicateGpuSequences: number };
  decoder: Decoder;
  windows: { first120Seconds: WindowStats; last120Seconds: WindowStats };
}

interface Distribution {
  count: number;
  rejected: number;
  mean: number | null;
  p50: number | null;
  p90: number | null;
  p95: number | null;
  max: number | null;
}

interface Decoder {
  counts: { frames: number | null; drops: number | null; corrupted: number | null };
  observed: { frames: number | null; drops: number | null; corrupted: number | null };
  exact: boolean;
  boundsExact: boolean;
  bounds: { start: { at: number }; end: { at: number } };
  gaps: { reason: string }[];
}

interface WindowStats {
  startedAt: number;
  endedAt: number;
  elapsedMs: number;
  activeMs: number | null;
  counts: { rendered: number; presented: number; skipped: number; rawFrames: number; rawGpu: number };
  rates: Summary['rates'] & { scope: string };
  gpu: Summary['gpu'];
  decoder: Decoder;
  callbackLatency: Distribution;
  decodeLatency: Distribution;
}

const toolPath = '../tools/m9-live.mjs';
const { summarize, windowStats, installCapture } = await import(toolPath) as {
  summarize(this: void, result: ReturnType<typeof fixture>): Summary;
  windowStats(this: void, result: ReturnType<typeof fixture>, startedAt?: number, endedAt?: number): WindowStats;
  installCapture(this: void): void;
};

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('GPU-free live measurement summary', () => {
  it('uses boundary count and active-clock deltas without depressing FPS during drain', () => {
    const immediate = summarize(fixture());
    const delayed = summarize(fixture(100));
    expect(delayed.rates).toEqual(immediate.rates);
    expect(delayed.rates).toMatchObject({ renderedFps: 2, presentedFps: 2 });
    expect(delayed.elapsed).toMatchObject({ sessionActiveMs: 1000, drainMs: 100 });
    expect(delayed.counts).toMatchObject({ rendered: 2, rawFrames: 2, rawGpu: 2,
      sessionGpu: 2, gpuReceivedDuringDrain: 1 });
  });

  it('selects observed callbacks even when queued metadata precedes the start', () => {
    const result = fixture(100);
    result.raw.frames = [frame(999, 5, 1100), frame(1100, 6, 990), frame(1900, 7), frame(2001, 8, 1990)];
    expect(summarize(result)).toMatchObject({ counts: { rawFrames: 2, rendered: 2 },
      rawGaps: { missingSequences: [] } });
  });

  it('keeps actual timer and captured boundary times separate and uses the active clock', () => {
    const result = fixture();
    result.boundary.at += 3;
    result.boundary.runtime.session.activeMs += 3;
    expect(summarize(result)).toMatchObject({ elapsed: { endedAt: 2000, boundaryCapturedAt: 2003,
      sessionActiveMs: 1003 }, rates: { renderedFps: 2000 / 1003 } });
  });

  it('reports unmeasured rates when there is no boundary-active elapsed time', () => {
    const result = fixture();
    result.boundary.runtime.session.activeMs = result.initial.runtime.session.activeMs;
    expect(summarize(result).rates).toMatchObject({ renderedFps: null, presentedFps: null });
  });

  it('counts the decoder tail at the boundary, not the last frame or post-drain endpoint', () => {
    const result = fixture(100);
    result.initial.decoderQuality = [100, 2, 1];
    result.boundary.decoderQuality = [112, 5, 2];
    result.drained.decoderQuality = [119, 8, 4];
    result.raw.frames[0]![8] = 1;
    result.raw.frames[1]![8] = 1;
    expect(summarize(result)).toMatchObject({ counts: { decoderFrames: 12, decoderDrops: 3, decoderCorrupted: 1 },
      decoder: { exact: true, gaps: [] } });
  });

  it('preserves observed per-load deltas and exposes unobserved reload tails as gaps', () => {
    const result = fixture();
    result.initial.decoderQuality = [100, 2, 0];
    result.raw.frames[0]!.splice(6, 4, 110, 3, 0, 3);
    result.raw.frames[1]!.splice(6, 4, 200, 9, 3, 4);
    result.boundary.runtime.session.loadGeneration = 4;
    result.boundary.decoderQuality = [205, 10, 3];
    const summary = summarize(result);
    expect(summary.counts).toMatchObject({ decoderFrames: null, decoderDrops: null, decoderCorrupted: null });
    expect(summary.decoder.observed).toEqual({ frames: 215, drops: 11, corrupted: 3 });
    expect(summary.decoder.gaps).toContainEqual(expect.objectContaining({ reason: 'Prior-load tail is unobserved before the next load' }));
    expect(summary.decoder.exact).toBe(false);
  });

  it('marks skipped generations and missing quality endpoints instead of inventing zero', () => {
    const result = fixture();
    result.boundary.runtime.session.loadGeneration = 5;
    result.boundary.decoderQuality = [10, null, 0];
    const summary = summarize(result);
    expect(summary.counts.decoderFrames).toBeNull();
    expect(summary.decoder.gaps).toContainEqual(expect.objectContaining({ reason: 'Skipped or regressing load generations' }));
    result.initial.decoderQuality = [null, null, null];
    result.boundary.runtime.session.loadGeneration = 3;
    result.boundary.decoderQuality = [120, null, 0];
    expect(summarize(result).counts).toMatchObject({ decoderFrames: null, decoderDrops: null, decoderCorrupted: null });
  });

  it('uses the captured boundary load when a reload has not reached the session yet', () => {
    const result = fixture();
    Object.assign(result.boundary, { decoderLoadGeneration: 4 });
    result.boundary.decoderQuality = [3, 1, 0];
    expect(summarize(result).decoder).toMatchObject({ exact: false,
      counts: { frames: null }, observed: { frames: 5, drops: 1, corrupted: 0 } });
  });

  it('retains exact zero decoder deltas and rejects counter regressions', () => {
    const result = fixture();
    result.boundary.decoderQuality = [...result.initial.decoderQuality];
    expect(summarize(result).counts).toMatchObject({ decoderFrames: 0, decoderDrops: 0, decoderCorrupted: 0 });
    result.boundary.decoderQuality = [90, 1, 0];
    expect(summarize(result).counts).toMatchObject({ decoderFrames: null, decoderDrops: null, decoderCorrupted: 0 });
  });

  it('keeps submitted-time GPU selection, late readbacks, exact distributions and timing gaps', () => {
    const result = fixture(100);
    result.raw.samples = [
      { submittedAt: 999, resolvedAt: 1500, sequence: 5, ms: 100, neural: true },
      { submittedAt: 1000, resolvedAt: 1100, sequence: 6, ms: 2, neural: true },
      { submittedAt: 2000, resolvedAt: 2100, sequence: 8, ms: 6, neural: false },
      { submittedAt: 2000, resolvedAt: 2100, sequence: 8, ms: 6, neural: false },
      { submittedAt: 2001, resolvedAt: 2002, sequence: 9, ms: 100, neural: false },
    ];
    result.sampleCountAtBoundary = 2;
    result.boundary.at = 2003;
    const original = structuredClone(result);
    expect(summarize(result)).toMatchObject({
      gpu: { all: { count: 3, rejected: 0, mean: 14 / 3, p50: 6, p90: 6, p95: 6, max: 6 },
        neural: { count: 1, mean: 2 }, baseline: { count: 2, mean: 6 } },
      counts: { rawGpu: 3, gpuReceivedDuringDrain: 2 },
      rawGaps: { missingSequences: [7], internalGpuSequenceGaps: [[7, 7]], duplicateGpuSequences: 1 },
    });
    expect(result).toEqual(original);
  });

  it('uses outward quality neighbors for tail subwindows and labels the approximation', () => {
    const result = fixture();
    const stats = windowStats(result, 1500, 2000);
    expect(stats).toMatchObject({ elapsedMs: 500, activeMs: null, counts: { rendered: 1, rawGpu: 1 },
      rates: { renderedFps: 2 }, decoder: { exact: false, boundsExact: false,
        bounds: { start: { at: 1100 }, end: { at: 2000 } }, counts: { frames: 1, drops: 0, corrupted: 0 } } });
    expect(stats.rates.scope).toContain('wall-clock');
    expect(windowStats(result, 1500, 1800).decoder.bounds.end.at).toBe(1900);
  });

  it('reports exact endpoint subwindows and includes callback metadata only for rVFC', () => {
    const result = fixture();
    expect(windowStats(result, 1100, 1900)).toMatchObject({ decoder: { exact: true },
      callbackLatency: { count: 2, mean: 4 }, decodeLatency: { count: 2, mean: 3 } });
    result.boundary.pipeline.clock = 'raf';
    expect(windowStats(result).callbackLatency).toMatchObject({ count: 0, mean: null });
  });

  it('provides bounded first and last 120-second analyses, using raw rates for each cut', () => {
    const result = fixture();
    result.endedAt = result.plannedEndAt = 601000;
    result.endOfDrainAt = 601100;
    result.boundary = checkpoint(601000, 600050, 9);
    result.drained = checkpoint(601100, 600150, 9);
    result.raw.frames.push(frame(500000, 8), frame(600000, 9));
    const summary = summarize(result);
    expect(summary.rates.renderedFps).toBe(4 / 600);
    expect(summary.windows.first120Seconds).toMatchObject({ startedAt: 1000, endedAt: 121000,
      elapsedMs: 120000, counts: { rendered: 2 }, rates: { renderedFps: 2 / 120 } });
    expect(summary.windows.last120Seconds).toMatchObject({ startedAt: 481000, endedAt: 601000,
      elapsedMs: 120000, counts: { rendered: 2 }, rates: { renderedFps: 2 / 120 }, decoder: { exact: false } });
    expect(summarize(fixture()).windows.first120Seconds.rates).toEqual(summarize(fixture()).rates);
    expect(() => windowStats(result, 999, 2000)).toThrow('Window must lie within');
  });
});

type ControllerRow = Omit<RuntimeSnapshot, 'transitions'> & { at: number };

function mockCapture() {
  let now = 0;
  vi.spyOn(performance, 'now').mockImplementation(() => now);
  const controller: RuntimeSnapshot = { mode: 'auto', state: 'stable', tier: 'neural', reason: 'qualified',
    failMs: 10, recoverMs: 7, medianMs: 5, p90Ms: 6, samples: 30, generation: 7,
    fps: 60, width: 1280, height: 720, activeMs: 0, backoffMs: 2000, nextProbeAtMs: null,
    transitionCount: 0, transitions: [], discardedTransitions: 0, probeCount: 0,
    failedProbeCount: 0, fallbackMs: 0, forced: false };
  const windowStub = { addEventListener: vi.fn(), aethervsrRuntime: undefined as unknown,
    __m9Live: undefined as { data: { raw: { controller: ControllerRow[]; transitions: unknown[] } } } | undefined };
  vi.stubGlobal('window', windowStub);
  vi.stubGlobal('document', { visibilityState: 'visible', hasFocus: () => true,
    addEventListener: vi.fn(), getElementById: vi.fn() });
  vi.stubGlobal('requestAnimationFrame', vi.fn());
  vi.stubGlobal('navigator', { userAgent: 'unit-test' });
  vi.stubGlobal('screen', { width: 1200, height: 800, availWidth: 1200, availHeight: 800 });
  vi.stubGlobal('devicePixelRatio', 1);
  vi.stubGlobal('innerWidth', 1200);
  vi.stubGlobal('innerHeight', 800);
  const driver = { session: { latestConfiguration: null }, onChange: vi.fn<(value: RuntimeSnapshot) => void>() };
  const runtime = { ...fixture().initial.runtime, controller };
  installCapture();
  windowStub.aethervsrRuntime = { driver, snapshot: () => runtime, neural: () => null,
    video: { getVideoPlaybackQuality: () => ({ totalVideoFrames: 105, droppedVideoFrames: 2, corruptedVideoFrames: 0 }),
      addEventListener: vi.fn() },
    pipeline: { submissionSequence: 1, currentUpscaler: { neural: true, id: 'test-neural' },
      stats: () => ({ clock: 'rvfc' }),
      gpu: { device: { addEventListener: vi.fn(), lost: { then: vi.fn() } },
        capabilities: { timestampQuery: true }, adapterReport: { fallbackAdapter: false } } },
  };
  return { controller, driver, rows: windowStub.__m9Live!.data.raw.controller,
    notify(this: void, at: number) { now = at; driver.onChange(controller); } };
}

describe('controller capture with in-memory API stubs only', () => {
  it('bounds 36,000 changing frame notifications to one-second snapshots without transition arrays', () => {
    const { controller, rows, notify } = mockCapture();
    controller.transitions = Array.from({ length: 128 }, () => ({ atMs: 0, activeMs: 0,
      from: 'probing', to: 'fallback', tier: 'baseline', reason: 'test' }));
    controller.transitionCount = 128;
    for (let index = 1; index <= 36000; index++) {
      controller.samples = index % 120;
      controller.p90Ms = index % 10;
      controller.activeMs = Math.floor(index * 1000 / 60);
      controller.nextProbeAtMs = controller.activeMs + 2000;
      notify(controller.activeMs);
    }
    expect(rows).toHaveLength(601);
    expect(rows.every(row => !Object.hasOwn(row, 'transitions'))).toBe(true);
    expect(rows[0]).toMatchObject({ at: 0, samples: 30, p90Ms: 6 });
    expect(rows.at(-1)).toMatchObject({ at: 600000, samples: 0, p90Ms: 0 });
  });

  it('captures transitions and failed-probe backoff immediately without inventing cleared evidence', () => {
    const { controller, rows, notify } = mockCapture();
    controller.state = 'probing';
    controller.probeCount = 1;
    controller.p90Ms = 12;
    notify(10);
    expect(rows.at(-1)).toMatchObject({ at: 10, state: 'probing', p90Ms: 12, samples: 30, failMs: 10 });
    controller.state = 'fallback';
    controller.tier = 'baseline';
    controller.reason = 'probe failed';
    controller.failedProbeCount = 1;
    controller.backoffMs = 4000;
    controller.nextProbeAtMs = 4010;
    controller.samples = 0;
    controller.medianMs = controller.p90Ms = null;
    notify(20);
    expect(rows.at(-1)).toMatchObject({ at: 20, state: 'fallback', backoffMs: 4000,
      nextProbeAtMs: 4010, failedProbeCount: 1, samples: 0, p90Ms: null, recoverMs: 7 });
    controller.backoffMs = 8000;
    notify(30);
    controller.generation = 8;
    notify(40);
    controller.failMs = 11;
    notify(50);
    controller.reason = 'forced fallback';
    notify(60);
    expect(rows).toHaveLength(7);
    expect(rows.at(-1)).toMatchObject({ at: 60, generation: 8, backoffMs: 8000, failMs: 11 });
    expect(rows[1]).toMatchObject({ at: 10, p90Ms: 12, samples: 30 });
  });
});