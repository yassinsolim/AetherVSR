import { describe, expect, it } from 'vitest';
import { RuntimeController } from '../src/core/upscale/runtime-controller.js';
import type { RuntimeSample, RuntimeSnapshot } from '../src/core/upscale/runtime-controller.js';
import { BudgetGuard } from '../src/core/upscale/budget-guard.js';

interface ReplayTrace {
  started: number;
  ended: number;
  generation: number;
  workload: { width: number; height: number; fps: number };
  samples: RuntimeSample[];
  pressure?: unknown;
}

interface ReplaySide {
  state: string;
  tier: string;
  generation: number;
  fallbackCount: number;
  probeCount: number;
  routing: { offered: number; baseline: number; beforeSwitch: number; nonNeural: number; wrongGeneration: number };
  bindings: { atMs: number; generation: number; tier: string }[];
  transitions: { atMs: number; to: string; generation: number }[];
}

interface Candidate {
  maxMs: number | null;
  evaluations: number;
  threeWindowTriggerCount: number;
}

interface ReplayResult {
  current: ReplaySide & { final: RuntimeSnapshot };
  reference: ReplaySide & { metadata: { label: string; exactLegacy: boolean } };
  window: { tickCount: number; replayEnd: number; rawSamples: number };
  candidates: { median30: Candidate; p90of30: Candidate };
}

interface ReplayModule {
  readCalibration(this: void, file: string): ReplayTrace | null;
  replay(this: void, trace: ReplayTrace, dependencies: {
    RuntimeController: typeof RuntimeController;
    createReference: () => BudgetGuard;
    referenceMetadata: { label: string; exactLegacy: boolean };
  }): ReplayResult;
}

const toolPath = '../tools/m9-replay.mjs';
const { replay, readCalibration } = await import(toolPath) as ReplayModule;
const referenceMetadata = { label: 'corrected fixed10/7 BudgetGuard (CI comparator, NOT original c65f67a)', exactLegacy: false };
const dependencies = { RuntimeController, createReference: () => new BudgetGuard({ failMs: 10, recoverMs: 7 }), referenceMetadata };

function synthetic(duration = 6000, cost: (offset: number) => number = () => 6): ReplayTrace {
  const started = 10000;
  return { started, ended: started + duration, generation: 7, workload: { width: 1280, height: 720, fps: 60 },
    samples: Array.from({ length: duration / 20 }, (_, index) => ({ ms: cost(index * 20), generation: 7,
      sequence: index + 100, submittedAt: started + index * 20, resolvedAt: started + index * 20 + 5, neural: true })) };
}

describe('GPU-free calibration replay (CI uses corrected fixed10/7, exact legacy is CLI-only)', () => {
  it('excludes the known invalid device-loss calibration', () => {
    expect(readCalibration('results/m9-calibration-720p60-load4.json.gz')).toBeNull();
  });

  it('stays neural, keeps absolute arrival times, and binds a constant trace generation only once', () => {
    const trace = synthetic();
    const original = structuredClone(trace);
    const result = replay(trace, dependencies);
    expect(result.current).toMatchObject({ state: 'stable', tier: 'neural', fallbackCount: 0,
      generation: 7, routing: { offered: 300 } });
    expect(result.reference).toMatchObject({ fallbackCount: 0, metadata: referenceMetadata });
    expect(result.current.bindings).toEqual([{ atMs: trace.started, generation: 7, tier: 'neural' }]);
    expect(result.window).toMatchObject({ tickCount: 60, replayEnd: trace.ended, rawSamples: 300 });
    expect(trace).toEqual(original);
    expect(replay({ ...trace, samples: [...trace.samples].reverse() }, dependencies)).toEqual(result);
  });

  it('withholds virtual baseline submissions and pre-probe arrivals, then rebinds fresh probes once', () => {
    const trace = synthetic(12000, offset => offset < 4000 ? 30 : 6);
    const seen: RuntimeSample[] = [];
    class RecordingController extends RuntimeController {
      override record(sample: RuntimeSample, now = sample.resolvedAt): RuntimeSnapshot {
        seen.push({ ...sample });
        return super.record(sample, now);
      }
    }
    const first = replay(trace, dependencies);
    const probe = first.current.bindings.find(binding => binding.generation === 9)!;
    expect(probe).toBeDefined();
    trace.samples.push({ ms: 0.01, sequence: 99999, generation: 7, neural: true,
      submittedAt: probe.atMs - 10, resolvedAt: probe.atMs + 10 });
    const result = replay(trace, { ...dependencies, RuntimeController: RecordingController });
    expect(result.current.fallbackCount).toBeGreaterThan(0);
    expect(result.current.routing.baseline).toBeGreaterThan(0);
    expect(result.current.routing.beforeSwitch).toBeGreaterThan(0);
    expect(result.current.final.state).toBe('stable');
    expect(seen.some(sample => sample.sequence === 99999)).toBe(false);
    expect(seen.some(sample => sample.generation > 7)).toBe(true);
    for (const sample of seen) {
      const binding = result.current.bindings.find(binding => binding.generation === sample.generation)!;
      expect(binding.tier).toBe('neural');
      expect(sample.submittedAt).toBeGreaterThanOrEqual(binding.atMs);
    }
    expect(result.current.bindings.map(binding => binding.generation))
      .toEqual(result.current.bindings.map((_, index) => 7 + index));
    expect(result.reference.routing.baseline).toBeGreaterThan(0);
    expect(trace.samples.every(sample => sample.generation === 7)).toBe(true);
  });

  it('does not use cheap baseline or wrong-generation records to recover either comparator', () => {
    const trace = synthetic(10000, () => 30);
    trace.samples.push(...trace.samples.map(sample => ({ ...sample, ms: 0.01,
      sequence: sample.sequence + 10000, resolvedAt: sample.resolvedAt + 1, neural: false })));
    trace.samples.push(...trace.samples.filter(sample => sample.neural).map(sample => ({ ...sample,
      ms: 0.01, sequence: sample.sequence + 20000, resolvedAt: sample.resolvedAt + 2, generation: 6 })));
    const result = replay(trace, dependencies);
    for (const side of [result.current, result.reference]) {
      expect(side.routing.nonNeural).toBe(500);
      expect(side.routing.wrongGeneration).toBe(500);
      expect(side.fallbackCount).toBeGreaterThan(1);
      expect(side.transitions.some(transition => transition.to === 'stable')).toBe(false);
    }
  });

  it('ticks without GPU samples through warmup and probe deadlines', () => {
    const trace = { ...synthetic(8000), samples: [] };
    const result = replay(trace, dependencies);
    expect(result.current.transitions.map(transition => [transition.atMs, transition.to]))
      .toEqual([[12000, 'fallback'], [14000, 'probing'], [16000, 'fallback'], [18000, 'probing']]);
    expect(result.window.tickCount).toBe(80);
    expect(result.candidates.median30.maxMs).toBeNull();
  });

  it.each(['1080p30-normal', '1080p60-normal'])('keeps actual %s neural while fixed10/7 rejects it', name => {
    const trace = readCalibration(`results/m9-calibration-${name}.json.gz`)!;
    expect(trace).not.toBeNull();
    const result = replay(trace, dependencies);
    expect(result.current).toMatchObject({ state: 'stable', fallbackCount: 0, probeCount: 0 });
    expect(result.reference.fallbackCount).toBeGreaterThan(0);
    expect(result.candidates.median30.maxMs).toBeGreaterThan(10);
  });

  it('keeps the actual single-spike trace neural without dropping the spike', () => {
    const trace = readCalibration('results/m9-calibration-v2-720p60-spike.json.gz')!;
    expect(Math.max(...trace.samples.map(sample => sample.ms))).toBeCloseTo(41.375, 2);
    const result = replay(trace, dependencies);
    expect(result.current).toMatchObject({ state: 'stable', fallbackCount: 0, probeCount: 0 });
    expect(result.current.routing.offered).toBe(trace.samples.length);
    expect(result.reference.fallbackCount).toBe(0);
  });

  it('reports tail candidate triggers separately and never lets raw frame pressure steer replay', () => {
    const trace = readCalibration('results/m9-calibration-v3-720p60-tail.json.gz')!;
    const result = replay(trace, dependencies);
    expect(result.current.fallbackCount).toBe(0);
    expect(result.candidates.median30.threeWindowTriggerCount).toBe(0);
    expect(result.candidates.p90of30.threeWindowTriggerCount).toBeGreaterThan(0);
    const changedPressure = replay({ ...trace, pressure: { callbackMaxMs: 999999, drops: 999999 } }, dependencies);
    expect(changedPressure.current).toEqual(result.current);
    expect(changedPressure.reference).toEqual(result.reference);
  });
});