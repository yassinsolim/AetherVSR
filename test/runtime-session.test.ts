import { describe, expect, it } from 'vitest';
import { RuntimeSession, type PlaybackQuality } from '../src/core/metrics/runtime-session.js';
import type { FrameTick } from '../src/core/types.js';

function quality(totalVideoFrames: number, droppedVideoFrames = 0, corruptedVideoFrames = 0): PlaybackQuality {
  return { totalVideoFrames, droppedVideoFrames, corruptedVideoFrames };
}

function frame(now: number, mediaTime = now / 1000, presentedDelta = 1): FrameTick {
  return {
    now, mediaTime, presentedDelta,
    size: { width: 1280, height: 720 },
    presentationTime: now - 1.021,
    expectedDisplayTime: now + 1,
    decodeLatencyMs: null,
  };
}

function stream(session: RuntimeSession) {
  let now = 0;
  let mediaTime = 0;
  let total = 0;
  let dropped = 0;
  return {
    step(options: {
      fps?: number; tier?: 'neural' | 'baseline'; rate?: number; presentedDelta?: number;
      drop?: number; exposed?: boolean; generation?: number; mediaDelta?: number;
      synthetic?: boolean;
    } = {}) {
      const { fps = 30, tier = 'baseline', rate = 1, presentedDelta = 1,
        drop = 0, exposed = true, generation = 0, synthetic = false } = options;
      now += 1000 * presentedDelta / fps;
      mediaTime += options.mediaDelta ?? presentedDelta * rate / fps;
      total += presentedDelta + drop;
      dropped += drop;
      const tick = frame(now, mediaTime, presentedDelta);
      return session.recordFrame(
        synthetic ? { ...tick, presentationTime: now, expectedDisplayTime: now } : tick,
        exposed ? quality(total, dropped) : null, generation, tier, rate,
      );
    },
    snapshot: () => session.snapshot(now),
  };
}

describe('RuntimeSession accounting', () => {
  it('starts at reset, includes the opening tick and stalls, and excludes explicit pauses only', () => {
    const session = new RuntimeSession(100);
    session.reset(100, quality(10, 2), 0);
    session.recordFrame(frame(600, 0, 3), quality(14, 3), 0, 'neural', 1);
    expect(session.snapshot(1100)).toMatchObject({
      activeMs: 1000, framesRendered: 1, framesPresented: 3, framesSkipped: 2,
      meanRenderedFps: 1, meanPresentedFps: 3, decoderFrames: 4, decoderDrops: 1,
    });
    session.setActive(false, 1100);
    expect(session.snapshot(9000).activeMs).toBe(1000);
    session.setActive(true, 9100);
    session.setActive(true, 9500);
    expect(session.snapshot(10100).activeMs).toBe(2000);
    session.setActive(false, 10100);
    session.reset(10200, null, 0);
    expect(session.snapshot(11000)).toMatchObject({ active: false, activeMs: 0, meanRenderedFps: null });
  });

  it('accumulates quality deltas once, including larger new-load counters and gaps', () => {
    const session = new RuntimeSession();
    session.reset(0, quality(100, 5, 1), 2);
    session.recordFrame(frame(10), quality(110, 7, 2), 2, 'neural', 1);
    session.recordFrame(frame(20), quality(110, 7, 2), 2, 'baseline', 1);
    session.recordFrame(frame(30), quality(200, 9, 3), 3, 'baseline', 1);
    session.recordFrame(frame(40), null, 3, 'baseline', 1);
    expect(session.snapshot(40)).toMatchObject({
      decoderFrames: 210, decoderDrops: 11, decoderCorrupted: 4, qualityAvailable: false,
    });
    session.recordFrame(frame(50), quality(205, 10, 3), 3, 'baseline', 1);
    session.recordFrame(frame(60), quality(205, 10, 3), 3, 'baseline', 1);
    expect(session.snapshot(60)).toMatchObject({
      decoderFrames: 215, decoderDrops: 12, decoderCorrupted: 4, qualityAvailable: true,
      qualitySamples: 5, qualityMissingSamples: 1,
    });
  });

  it('rejects regressions within a load without double counting recovery', () => {
    const session = new RuntimeSession();
    session.reset(0, quality(100, 5, 2), 1);
    session.recordFrame(frame(10), quality(110, 7, 3), 1, 'neural', 1);
    session.recordFrame(frame(20), quality(109, 7, 3), 1, 'neural', 1);
    session.recordFrame(frame(30), quality(120, 6, 3), 1, 'neural', 1);
    session.recordFrame(frame(40), quality(120, 8, 2), 1, 'neural', 1);
    session.recordFrame(frame(50), quality(120, 8, 4), 0, 'neural', 1);
    session.recordFrame(frame(60), quality(120, 8, 4), 1, 'neural', 1);
    expect(session.snapshot(60)).toMatchObject({
      decoderFrames: 20, decoderDrops: 3, decoderCorrupted: 2, qualityRejectedSamples: 4,
    });
  });

  it('distinguishes unsupported counters from exposed zero and baselines late exposure', () => {
    const session = new RuntimeSession();
    expect(session.snapshot(0)).toMatchObject({
      decoderFrames: null, decoderDrops: null, qualityAvailable: false, meanPresentedFps: null,
    });
    session.recordFrame(frame(10), null, 0, 'baseline', 1);
    session.recordFrame(frame(20), quality(500, 20), 0, 'baseline', 1);
    expect(session.snapshot(20)).toMatchObject({ decoderFrames: 0, decoderDrops: 0, qualityAvailable: true });
    session.recordFrame(frame(30), null, 1, 'baseline', 1);
    session.recordFrame(frame(40), quality(8, 2), 1, 'baseline', 1);
    expect(session.snapshot(40)).toMatchObject({ decoderFrames: 8, decoderDrops: 2 });
  });

  it('preserves lifetime counters and cold GPU timings across configurations and sources', () => {
    const session = new RuntimeSession();
    session.recordFrame(frame(10), quality(1), 0, 'neural', 1);
    session.recordGpu({ ms: 17, neural: true, generation: 1 });
    session.recordGpu({ ms: 2, neural: false, generation: 2 });
    session.recordConfiguration({
      configureMs: 1.13, source: { width: 1280, height: 720 },
      target: { width: 2560, height: 1440 }, generation: 2, sourceChanged: false,
    });
    session.recordConfiguration({
      configureMs: 1.17, source: { width: 1920, height: 1080 },
      target: { width: 3840, height: 2160 }, generation: 3, sourceChanged: true,
    });
    session.sourceChanged();
    session.recordGpu({ ms: 19, neural: true, generation: 1 });
    const snapshot = session.snapshot(1000);
    expect(snapshot).toMatchObject({ framesRendered: 1, activeMs: 1000, configurationCount: 2, generation: 3 });
    expect(snapshot.gpu.neural).toMatchObject({ count: 2, sum: 36, mean: 18, max: 19 });
    expect(snapshot.gpu.baseline).toMatchObject({ count: 1, mean: 2 });
    expect(snapshot.configuration).toMatchObject({ count: 2, mean: 1.15, p95: 1.2 });
    expect(snapshot.callbackLatency).toMatchObject({ count: 1, p50: 1.05 });
  });

  it('resets every session total and baselines the current load exactly once', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 31; sample++) feed.step();
    session.recordGpu({ ms: 18, neural: true, generation: 4, sequence: 1, submittedAt: -1, resolvedAt: 9999 });
    session.recordGpu({ ms: 1, neural: false, generation: 3 });
    session.recordConfiguration({ configureMs: 2, source: { width: 1, height: 1 },
      target: { width: 2, height: 2 }, generation: 4, sourceChanged: false });
    session.reset(2000, quality(31, 2, 1), 0);
    const snapshot = session.snapshot(2000);
    expect(snapshot).toMatchObject({ activeMs: 0, framesRendered: 0, framesPresented: 0,
      framesSkipped: 0, decoderFrames: 0, decoderDrops: 0, decoderCorrupted: 0,
      qualitySamples: 0, qualityMissingSamples: 0, qualityRejectedSamples: 0,
      generation: null, source: null, target: null, configurationCount: 0,
      sourceFpsEstimate: 60, cadence: { observations: 0, cleanBaselineIntervals: 0 } });
    for (const summary of [snapshot.gpu.neural, snapshot.gpu.baseline, snapshot.callbackLatency, snapshot.configuration]) {
      expect(summary).toMatchObject({ count: 0, sum: null, max: null, mean: null, p50: null,
        p90: null, p95: null, overflowCount: 0, rejectedCount: 0 });
    }
    session.recordFrame(frame(2100), quality(35, 3, 2), 0, 'neural', 1);
    expect(session.snapshot(3000)).toMatchObject({ decoderFrames: 4, decoderDrops: 1,
      decoderCorrupted: 1, meanRenderedFps: 1 });
  });

  it('reports real zero rates and latencies, excludes synthetic latency, and rejects time reversal', () => {
    const session = new RuntimeSession();
    expect(session.snapshot(100)).toMatchObject({ meanRenderedFps: 0, meanPresentedFps: 0 });
    session.recordFrame({ ...frame(100), presentationTime: 101 }, null, 0, 'baseline', 1);
    session.recordFrame({ ...frame(200), presentationTime: 200, expectedDisplayTime: 200 }, null, 0, 'baseline', 1);
    expect(session.snapshot(200).callbackLatency).toMatchObject({ count: 1, sum: 0, mean: 0, max: 0 });
    expect(() => session.recordFrame(frame(199), null, 0, 'neural', 1)).toThrow(RangeError);
    expect(() => session.setActive(false, 199)).toThrow(RangeError);
    expect(() => session.snapshot(NaN)).toThrow(RangeError);
    expect(session.snapshot(200).framesRendered).toBe(2);
  });

  it('rejects malformed quality and never repairs it by silently rebasing', () => {
    const session = new RuntimeSession();
    session.reset(0, quality(10, 2, 1), 0);
    session.recordFrame(frame(10), quality(NaN, 3, 1), 0, 'baseline', 1);
    session.recordFrame(frame(20), quality(20, -1, 1), 0, 'baseline', 1);
    session.recordFrame(frame(30), quality(20, 3, 1.5), 0, 'baseline', 1);
    session.recordFrame(frame(40), quality(20, 3, 2), 0, 'baseline', 1);
    expect(session.snapshot(40)).toMatchObject({ decoderFrames: 10, decoderDrops: 1,
      decoderCorrupted: 1, qualityRejectedSamples: 3, qualitySamples: 1 });
  });

  it('copies reset baselines, configuration metadata and snapshot geometry', () => {
    const session = new RuntimeSession();
    const baseline = { totalVideoFrames: 10, droppedVideoFrames: 2, corruptedVideoFrames: 0 };
    session.reset(0, baseline, 0);
    baseline.totalVideoFrames = 100;
    const config = { configureMs: 2, source: { width: 1, height: 1 },
      target: { width: 2, height: 2 }, generation: 1, sourceChanged: false };
    session.recordConfiguration(config);
    config.source.width = 99;
    config.target.width = 99;
    session.recordFrame(frame(10), quality(20, 2), 0, 'baseline', 1);
    const snapshot = session.snapshot(10);
    snapshot.source!.width = 50;
    snapshot.target!.width = 50;
    expect(session.snapshot(10)).toMatchObject({ decoderFrames: 10,
      source: { width: 1, height: 1 }, target: { width: 2, height: 2 } });
  });
});

describe('RuntimeSession bounded histograms', () => {
  it('keeps all raw counts, sums and maxima beyond 240 samples with quantized quantiles', () => {
    const session = new RuntimeSession();
    for (let sample = 0; sample < 10000; sample++) {
      session.recordGpu({ ms: sample < 9000 ? 1.021 : 2.071, neural: true, generation: sample });
    }
    const summary = session.snapshot(0).gpu.neural;
    expect(summary).toMatchObject({ count: 10000, p50: 1.05, p90: 1.05, p95: 2.1, max: 2.071 });
    expect(summary.sum).toBeCloseTo(9000 * 1.021 + 1000 * 2.071, 7);
    expect(summary.mean).toBeCloseTo((9000 * 1.021 + 1000 * 2.071) / 10000, 10);
    expect(summary.quantileScope).toContain('+/-0.05');
  });

  it('reports overflow honestly and leaves empty timing statistics null', () => {
    const session = new RuntimeSession();
    expect(session.snapshot(0).gpu.neural).toMatchObject({ count: 0, sum: null, mean: null, max: null, p50: null });
    for (const ms of [0, 0.05, 200, 200.001, 500]) session.recordGpu({ ms, neural: true, generation: 0 });
    for (const ms of [-1, NaN, Infinity]) session.recordGpu({ ms, neural: true, generation: 0 });
    expect(session.snapshot(0).gpu.neural).toMatchObject({
      count: 5, max: 500, p50: 200, p90: null, p95: null, overflowCount: 2, rejectedCount: 3,
    });
  });

  it('retains fixed histogram and cadence storage over a long session', () => {
    function storageLengths(value: object): number[] {
      const lengths: number[] = [];
      for (const child of Object.values(value) as unknown[]) {
        if (child instanceof Float64Array) lengths.push(child.length);
        else if (Array.isArray(child)) lengths.push(child.length);
        else if (child !== null && typeof child === 'object') lengths.push(...storageLengths(child));
      }
      return lengths.sort((left, right) => left - right);
    }
    const session = new RuntimeSession();
    const originalStorage = storageLengths(session);
    expect(originalStorage).toEqual([30, 4001, 4001, 4001, 4001]);
    const feed = stream(session);
    for (let sample = 0; sample < 20000; sample++) {
      feed.step({ fps: 60, tier: 'neural' });
      session.recordGpu({ ms: 4.321, neural: true, generation: sample });
    }
    expect(storageLengths(session)).toEqual(originalStorage);
    expect(feed.snapshot()).toMatchObject({ framesRendered: 20000,
      gpu: { neural: { count: 20000 } }, cadence: { observations: 30 } });
  });
});

describe('RuntimeSession cadence', () => {
  it('requires thirty clean baseline intervals after the opening frame to relax', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 30; sample++) expect(feed.step()).toBeNull();
    expect(feed.snapshot().cadence).toEqual({ observations: 29, cleanBaselineIntervals: 29 });
    expect(feed.step()).toBe(30);
    expect(feed.step()).toBeNull();
    expect(feed.snapshot().sourceFpsEstimate).toBe(30);
  });

  it('preserves the learned estimate only when requested at reset and tightens on a fresh neural window', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 31; sample++) feed.step();
    session.recordGpu({ ms: 20, neural: true, generation: 0 });
    const before = feed.snapshot();
    expect(before).toMatchObject({ sourceFpsEstimate: 30, framesRendered: 31,
      cadence: { observations: 30, cleanBaselineIntervals: 30 } });

    session.reset(before.activeMs, quality(31), 0, true);

    expect(feed.snapshot()).toMatchObject({ active: true, activeMs: 0, framesRendered: 0,
      decoderFrames: 0, sourceFpsEstimate: 30, gpu: { neural: { count: 0, sum: null } },
      cadence: { observations: 0, cleanBaselineIntervals: 0 } });
    for (let sample = 0; sample < 30; sample++) expect(feed.step({ tier: 'neural', fps: 50 })).toBeNull();
    expect(feed.snapshot()).toMatchObject({ sourceFpsEstimate: 30,
      cadence: { observations: 29, cleanBaselineIntervals: 0 } });
    expect(feed.step({ tier: 'neural', fps: 50 })).toBe(50);
    expect(feed.snapshot()).toMatchObject({ sourceFpsEstimate: 50, framesRendered: 31, decoderFrames: 31,
      cadence: { observations: 30, cleanBaselineIntervals: 0 } });
    const now = before.activeMs + feed.snapshot().activeMs;
    session.reset(now, quality(62), 0);
    expect(session.snapshot(now)).toMatchObject({ sourceFpsEstimate: 60, framesRendered: 0,
      cadence: { observations: 0, cleanBaselineIntervals: 0 } });
  });

  it('clears the cadence window and its opening anchor on discontinuity without resetting the estimate or totals', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 31; sample++) feed.step();
    session.recordGpu({ ms: 20, neural: true, generation: 0 });
    const before = feed.snapshot();
    expect(before.sourceFpsEstimate).toBe(30);

    session.discontinuity();

    expect(feed.snapshot()).toEqual({ ...before, cadence: { observations: 0, cleanBaselineIntervals: 0 } });
    expect(session.snapshot(before.activeMs + 1000).activeMs).toBeCloseTo(before.activeMs + 1000);
    for (let sample = 0; sample < 30; sample++) expect(feed.step({ tier: 'neural', fps: 50 })).toBeNull();
    expect(feed.snapshot()).toMatchObject({ sourceFpsEstimate: 30,
      cadence: { observations: 29, cleanBaselineIntervals: 0 } });
    expect(feed.step({ tier: 'neural', fps: 50 })).toBe(50);
    expect(feed.snapshot()).toMatchObject({ active: true, framesRendered: before.framesRendered + 31,
      sourceFpsEstimate: 50, gpu: before.gpu });
  });

  it('cannot relax on neural observations but can tighten after thirty valid intervals', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 90; sample++) expect(feed.step({ tier: 'neural' })).toBeNull();
    expect(feed.snapshot().sourceFpsEstimate).toBe(60);
    session.sourceChanged();
    for (let sample = 0; sample < 30; sample++) expect(feed.step({ tier: 'neural', fps: 120 })).toBeNull();
    expect(feed.step({ tier: 'neural', fps: 120 })).toBe(120);
  });

  it.each(['neural', 'drop', 'skip', 'missing'] as const)('restarts baseline certification after %s', (interruption) => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 30; sample++) feed.step();
    feed.step({ tier: interruption === 'neural' ? 'neural' : 'baseline',
      drop: interruption === 'drop' ? 1 : 0, presentedDelta: interruption === 'skip' ? 2 : 1,
      exposed: interruption !== 'missing' });
    expect(feed.snapshot().cadence.cleanBaselineIntervals).toBe(0);
    for (let sample = 0; sample < 29; sample++) expect(feed.step()).toBeNull();
    if (interruption === 'neural' || interruption === 'missing') expect(feed.step()).toBeNull();
    expect(feed.step()).toBe(30);
  });

  it('uses the true even median of normalized intervals rather than a mean or median FPS', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    feed.step();
    for (let sample = 0; sample < 14; sample++) feed.step({ mediaDelta: 0.005 });
    feed.step({ mediaDelta: 0.025 });
    feed.step({ mediaDelta: 0.04 });
    for (let sample = 0; sample < 13; sample++) feed.step({ mediaDelta: 0.16 });
    expect(feed.step({ mediaDelta: 0.16 })).toBe(30);
  });

  it('normalizes by presented delta and current playback rate without repeating observations', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 30; sample++) {
      expect(feed.step({ fps: 120, rate: 2, presentedDelta: 2, tier: 'neural' })).toBeNull();
    }
    expect(feed.snapshot().cadence.observations).toBe(29);
    expect(feed.step({ fps: 120, rate: 2, presentedDelta: 2, tier: 'neural' })).toBe(120);
  });

  it('does not certify cadence using unavailable counters or synthetic rAF metadata', () => {
    for (const synthetic of [false, true]) {
      const session = new RuntimeSession();
      const feed = stream(session);
      for (let sample = 0; sample < 90; sample++) expect(feed.step({ synthetic, exposed: synthetic })).toBeNull();
      expect(feed.snapshot().sourceFpsEstimate).toBe(60);
    }
  });

  it.each([0, -0.01, 0.201])('flushes a seek/loop/discontinuity interval of %s seconds', (mediaDelta) => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 30; sample++) feed.step();
    expect(feed.step({ mediaDelta })).toBeNull();
    expect(feed.snapshot().cadence).toEqual({ observations: 0, cleanBaselineIntervals: 0 });
    for (let sample = 0; sample < 29; sample++) expect(feed.step()).toBeNull();
    expect(feed.step()).toBe(30);
    expect(feed.step({ mediaDelta: -1 })).toBeNull();
    expect(feed.snapshot().sourceFpsEstimate).toBe(30);
  });

  it.each([0.5, 1.25, 2])('normalizes positive playback rate %s and flushes rate-crossing evidence', (rate) => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 30; sample++) feed.step();
    expect(feed.step({ rate })).toBeNull();
    expect(feed.snapshot().cadence.observations).toBe(0);
    for (let sample = 0; sample < 29; sample++) expect(feed.step({ rate })).toBeNull();
    expect(feed.step({ rate })).toBe(30);
  });

  it.each([0, -1, NaN, Infinity])('rejects invalid playback rate %s without changing the estimate', (rate) => {
    const session = new RuntimeSession();
    for (let sample = 1; sample <= 40; sample++) {
      expect(session.recordFrame(frame(sample * 30), quality(sample), 0, 'baseline', rate)).toBeNull();
    }
    expect(session.snapshot(1200)).toMatchObject({ sourceFpsEstimate: 60,
      cadence: { observations: 0, cleanBaselineIntervals: 0 } });
  });

  it.each([[23.976, 24], [25.01, 25], [29.97, 30], [49.95, 50], [59.94, 60],
    [119.88, 120], [27.4, 27], [36.7, 37], [90, 90], [21.9, 22], [22.1, 24]])(
    'snaps %s FPS to %s only inside the nearest standard rate tolerance', (fps, expected) => {
      const session = new RuntimeSession();
      const feed = stream(session);
      for (let sample = 0; sample < 30; sample++) expect(feed.step({ fps })).toBeNull();
      expect(feed.step({ fps })).toBe(expected === 60 ? null : expected);
      expect(feed.snapshot().sourceFpsEstimate).toBe(expected);
    },
  );

  it('cannot use apparently slower presentation to relax under ongoing decoder loss', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 90; sample++) expect(feed.step({ drop: 1 })).toBeNull();
    expect(feed.snapshot()).toMatchObject({ sourceFpsEstimate: 60, decoderDrops: 89,
      cadence: { observations: 30, cleanBaselineIntervals: 0 } });
    for (let sample = 0; sample < 29; sample++) expect(feed.step()).toBeNull();
    expect(feed.step()).toBe(30);
  });

  it('requires a fresh baseline window after neural loss but allows a constant historical drop total', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 60; sample++) expect(feed.step({ tier: 'neural', drop: 1 })).toBeNull();
    for (let sample = 0; sample < 30; sample++) expect(feed.step()).toBeNull();
    expect(feed.step()).toBe(30);
    expect(feed.snapshot().decoderDrops).toBe(59);
  });

  it('clears only cadence for explicit source changes and actual resolution changes, not tier swaps', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 31; sample++) feed.step();
    session.recordGpu({ ms: 12, neural: true, generation: 1 });
    const before = feed.snapshot();
    const config = { configureMs: 2, source: { width: 1280, height: 720 },
      target: { width: 2560, height: 1440 }, generation: 2, sourceChanged: false };
    session.recordConfiguration(config);
    expect(feed.snapshot()).toMatchObject({ sourceFpsEstimate: 30,
      cadence: { observations: 30, cleanBaselineIntervals: 30 } });
    session.sourceChanged();
    expect(feed.snapshot()).toMatchObject({ sourceFpsEstimate: 60,
      framesRendered: before.framesRendered, decoderFrames: before.decoderFrames, activeMs: before.activeMs,
      gpu: before.gpu, cadence: { observations: 0, cleanBaselineIntervals: 0 } });
    for (let sample = 0; sample < 30; sample++) expect(feed.step()).toBeNull();
    expect(feed.step()).toBe(30);
    session.recordConfiguration({ ...config, generation: 3,
      source: { width: 1920, height: 1080 }, sourceChanged: true });
    expect(feed.snapshot()).toMatchObject({ sourceFpsEstimate: 60, configurationCount: 2,
      framesRendered: 62, cadence: { observations: 0, cleanBaselineIntervals: 0 } });
    expect(feed.step()).toBeNull();
    expect(feed.snapshot().cadence.observations).toBe(0);
  });

  it('starts a new load at 60 without resetting session counts or counting a cross-load interval', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 31; sample++) feed.step();
    expect(feed.snapshot().sourceFpsEstimate).toBe(30);
    session.recordFrame(frame(2000, 10), quality(5, 1), 1, 'baseline', 1);
    expect(session.snapshot(2000)).toMatchObject({ framesRendered: 32, decoderFrames: 35, decoderDrops: 1,
      sourceFpsEstimate: 60, loadGeneration: 1, cadence: { observations: 0, cleanBaselineIntervals: 0 } });
    session.recordFrame(frame(2100, 10.1), quality(5, 1), 1, 'baseline', 1);
    expect(session.snapshot(2100).decoderFrames).toBe(35);
  });

  it('never uses paused frames or crosses a pause when certifying cadence', () => {
    const session = new RuntimeSession();
    const feed = stream(session);
    for (let sample = 0; sample < 30; sample++) feed.step();
    session.setActive(false, 1100);
    for (let sample = 1; sample <= 40; sample++) {
      expect(session.recordFrame(frame(1100 + sample * 30), quality(30 + sample), 0, 'baseline', 1)).toBeNull();
    }
    expect(session.snapshot(2400)).toMatchObject({ activeMs: 1100, sourceFpsEstimate: 60,
      cadence: { observations: 0, cleanBaselineIntervals: 0 } });
    session.setActive(true, 2500);
    for (let sample = 0; sample < 30; sample++) {
      expect(session.recordFrame(frame(2600 + sample * 1000 / 30, sample / 30),
        quality(71 + sample), 0, 'baseline', 1)).toBeNull();
    }
    expect(session.recordFrame(frame(3600, 1), quality(101), 0, 'baseline', 1)).toBe(30);
  });
});