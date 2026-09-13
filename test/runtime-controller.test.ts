import { describe, expect, it } from 'vitest';
import { RuntimeController } from '../src/core/upscale/runtime-controller.js';
import type { RuntimeSample, RuntimeSnapshot } from '../src/core/upscale/runtime-controller.js';

function feed(controller: RuntimeController, ms: number, count: number, start: number, generation = 1, sequence = 0) {
  let result = controller.snapshot(start);
  for (let index = 0; index < count; index++) {
    const now = start + index * 10;
    result = controller.record({ ms, generation, sequence: sequence + index, submittedAt: now, resolvedAt: now, neural: true });
  }
  return result;
}

function repeated(ms: number, count = 35): number[] {
  return Array.from({ length: count }, () => ms);
}

function trace(options: ConstructorParameters<typeof RuntimeController>[0] = {}, origin = 0) {
  const controller = new RuntimeController(options, origin);
  let generation = 1;
  let sequence = 0;
  controller.bindGeneration(generation, origin);
  const sample = (ms: number, now: number, overrides: Partial<RuntimeSample> = {}) =>
    controller.record({ ms, generation, sequence: sequence++, submittedAt: now, resolvedAt: now, neural: true, ...overrides }, now);
  const bind = (next: number, now: number) => {
    generation = next;
    return controller.bindGeneration(next, now);
  };
  const run = (values: number[], start: number, step = 10) => {
    let result = controller.snapshot(start);
    for (const [index, ms] of values.entries()) result = sample(ms, start + index * step);
    return result;
  };
  const probe = (now = origin) => {
    controller.setForced(true, now);
    controller.setForced(false, now + 1);
    const result = controller.tick(now + 2000);
    bind(generation + 1, now + 2000);
    return result;
  };
  return { controller, sample, bind, run, probe };
}

function policy(snapshot: RuntimeSnapshot) {
  const { activeMs, fallbackMs, ...rest } = snapshot;
  expect(Number.isFinite(activeMs) && Number.isFinite(fallbackMs)).toBe(true);
  return rest;
}

describe('RuntimeController', () => {
  it('uses the calibrated thresholds and requires recovery during a probe', () => {
    const controller = new RuntimeController();
    expect(controller.snapshot(0)).toMatchObject({ mode: 'auto', state: 'warmup', tier: 'neural', failMs: 15, recoverMs: 13 });
    controller.bindGeneration(1, 0);
    expect(feed(controller, 14, 35, 150).state).toBe('stable');
    controller.setForced(true, 500);
    controller.setForced(false, 501);
    expect(controller.tick(2500).state).toBe('probing');
    controller.bindGeneration(2, 2500);
    expect(feed(controller, 14, 35, 2650, 2, 35).state).toBe('probing');
    expect(controller.tick(4500)).toMatchObject({ state: 'fallback', backoffMs: 4000, failedProbeCount: 1, nextProbeAtMs: 6500 });
  });

  it('binding a new generation cannot extend the original probe deadline', () => {
    const controller = new RuntimeController();
    controller.setForced(true, 0);
    controller.setForced(false, 1);
    controller.tick(2000);
    controller.bindGeneration(1, 3900);
    expect(controller.tick(4000)).toMatchObject({ state: 'fallback', failedProbeCount: 1, nextProbeAtMs: 6000 });
  });

  describe('AK-A: calibrated policy and continuous traces', () => {
    it.each(['auto', 'neural'] as const)('%s stays stable through continuous 9.8-boundary-scaled noise', mode => {
      const rig = trace({ mode });
      const result = rig.run(Array.from({ length: 4000 }, (_, index) => [14.4, 14.7, 15.2, 14.8, 14.6][index % 5]!), 0, 16);
      expect(result).toMatchObject({ mode, tier: 'neural', state: 'stable', samples: 30, medianMs: 14.7, transitionCount: 1, probeCount: 0 });
    });

    it.each([
      [60, 15, 13], [30, 24, 22], [24, 24, 22], [25, 24, 22], [50, 18, 15.6], [120, 7.5, 6.5], [240, 6, 5],
    ])('keeps the frozen formula at %s fps', (fps, failMs, recoverMs) => {
      const controller = new RuntimeController();
      expect(controller.setWorkload(1280, 720, fps, 0)).toMatchObject({ fps, failMs, recoverMs });
    });

    it.each([Number.NaN, Infinity, -Infinity, 0, -1])('uses 60 fps for unusable cadence %s', fps => {
      expect(new RuntimeController().setWorkload(1280, 720, fps, 0)).toMatchObject({ fps: 60, failMs: 15, recoverMs: 13 });
    });

    it.each([[0.01, 1, 24, 22], [10000, 240, 6, 5]])('clamps extreme positive cadence %s', (fps, clamped, failMs, recoverMs) => {
      expect(new RuntimeController().setWorkload(1280, 720, fps, 0)).toMatchObject({ fps: clamped, failMs, recoverMs });
    });

    it.each([[60, 15], [30, 24], [120, 7.5]])('accepts equality but rejects a strictly higher failure median at %s fps', (fps, boundary) => {
      for (const extra of [0, 0.000001]) {
        const rig = trace();
        rig.controller.setWorkload(1280, 720, fps, 0);
        rig.bind(2, 0);
        expect(rig.run(repeated(boundary + extra), 150).state).toBe(extra === 0 ? 'stable' : 'fallback');
      }
    });

    it('reports null for missing statistics and an exact even raw median and interpolated p90', () => {
      const rig = trace();
      expect(rig.controller.snapshot(0)).toMatchObject({ medianMs: null, p90Ms: null, samples: 0 });
      expect(rig.run([1, 2, 3, 10], 0)).toMatchObject({ medianMs: 2.5, p90Ms: 7.900000000000001, samples: 4, state: 'warmup' });
      expect(rig.run([...repeated(2, 15), ...repeated(10, 15)], 40, 1)).toMatchObject({ medianMs: 6, p90Ms: 10, samples: 30 });
    });
  });

  describe('AK-B: transient tolerance and sustained overload', () => {
    it('tolerates one 41.375 ms spike without using p90 to steer', () => {
      const rig = trace();
      rig.run(repeated(6), 150);
      expect(rig.sample(41.375, 500).state).toBe('stable');
      const result = rig.run(Array.from({ length: 600 }, (_, index) => index % 5 === 0 ? 26.423 : 6), 510);
      expect(result).toMatchObject({ state: 'stable', medianMs: 6, p90Ms: 26.423, transitionCount: 1 });
    });

    it('falls back within three active seconds of continuous overload onset', () => {
      const rig = trace();
      rig.run(repeated(6), 150);
      const result = rig.run(repeated(21, 40), 500, 16);
      expect(result).toMatchObject({ state: 'fallback', tier: 'baseline', backoffMs: 2000 });
      const fallback = result.transitions.find(transition => transition.to === 'fallback')!;
      expect(fallback.atMs - 500).toBeLessThanOrEqual(3000);
      expect(result.nextProbeAtMs).toBe(fallback.atMs + 2000);
    });
  });

  describe('AK-C: cold exclusion and independent predicate dwells', () => {
    it('retains the first three resolved samples in telemetry but never uses them in decisions', () => {
      const rig = trace();
      expect(rig.run(repeated(999, 3), 150)).toMatchObject({ samples: 3, medianMs: 999, state: 'warmup' });
      expect(rig.run(repeated(14, 29), 180).state).toBe('warmup');
      expect(rig.sample(14, 470).state).toBe('warmup');
      expect(rig.sample(14, 480).state).toBe('warmup');
      expect(rig.sample(14, 490).state).toBe('stable');
    });

    it('also excludes submissions before 150 active ms, including delayed cold deliveries', () => {
      const rig = trace();
      expect(rig.run(repeated(99, 30), 0, 4)).toMatchObject({ state: 'warmup', samples: 30 });
      rig.sample(99, 160, { submittedAt: 149 });
      expect(rig.run(repeated(14, 31), 170).state).toBe('warmup');
      expect(rig.sample(14, 480).state).toBe('stable');
    });

    it('requires three failure evaluations, and invalid samples never complete dwell', () => {
      const rig = trace();
      expect(rig.run(repeated(20, 34), 150).state).toBe('warmup');
      for (const ms of [0, -1, NaN, Infinity]) expect(rig.sample(ms, 481).state).toBe('warmup');
      expect(rig.sample(20, 490).state).toBe('fallback');
    });

    it.each([[27, 27], [24, 30]])('does not share failure/hysteresis dwell with recovery (%s, %s)', (first, second) => {
      const rig = trace();
      rig.probe();
      rig.run(repeated(40, 3), 2150);
      expect(rig.run([first, second, ...repeated(9, 14), ...repeated(40, 14)], 2180).state).toBe('probing');
      expect(rig.sample(9, 2480).state).toBe('probing');
      expect(rig.sample(9, 2490).state).toBe('probing');
      for (const ms of [NaN, 0, -1]) expect(rig.sample(ms, 2491).state).toBe('probing');
      expect(rig.sample(9, 2500).state).toBe('probing');
      expect(rig.sample(9, 2510).state).toBe('stable');
    });

    it('resets recovery dwell on a hysteresis evaluation', () => {
      const rig = trace();
      rig.probe();
      rig.run(repeated(40, 3), 2150);
      rig.run([9, 9, 18, ...repeated(9, 14), ...repeated(40, 13)], 2180);
      expect(rig.sample(40, 2480).state).toBe('probing');
      expect(rig.sample(9, 2490).state).toBe('probing');
      expect(rig.sample(9, 2500).state).toBe('probing');
      expect(rig.sample(9, 2510).state).toBe('probing');
      expect(rig.sample(9, 2520).state).toBe('stable');
    });
  });

  describe('AK-D: bounded probes and recovery latency', () => {
    it.each([13, 13.000001])('requires recovery-boundary qualification at %s ms without changing the selected tier', ms => {
      const rig = trace();
      expect(rig.probe()).toMatchObject({ state: 'probing', tier: 'neural' });
      const result = rig.run(repeated(ms), 2150);
      expect(result).toMatchObject({ state: ms === 13 ? 'stable' : 'probing', tier: 'neural', probeCount: 1 });
      if (ms === 13) expect(result.transitions.at(-1)).toMatchObject({ from: 'probing', to: 'stable', tier: 'neural' });
      else expect(rig.controller.tick(4000).state).toBe('fallback');
    });

    it('backs off from probe start through 2/4/8/16/30 seconds and resets only on confirmation', () => {
      const rig = trace();
      rig.probe();
      let startedAt = 2000;
      for (const [index, backoffMs] of [4000, 8000, 16000, 30000, 30000].entries()) {
        const result = rig.run(repeated(21), startedAt + 150);
        expect(result).toMatchObject({ state: 'fallback', backoffMs, nextProbeAtMs: startedAt + backoffMs, failedProbeCount: index + 1 });
        expect(rig.controller.tick(startedAt + backoffMs - 1).state).toBe('fallback');
        startedAt += backoffMs;
        expect(rig.controller.tick(startedAt)).toMatchObject({ state: 'probing', backoffMs });
        rig.bind(index + 3, startedAt);
      }
      expect(rig.run(repeated(9), startedAt + 150)).toMatchObject({ state: 'stable', backoffMs: 2000, nextProbeAtMs: null, failedProbeCount: 5, probeCount: 6 });
    });

    it('bounds an alternating unresolved probe and schedules no earlier than its late end', () => {
      const rig = trace();
      rig.probe();
      expect(rig.run(Array.from({ length: 180 }, (_, index) => index % 2 ? 9 : 19), 2150).state).toBe('probing');
      expect(rig.controller.tick(9000)).toMatchObject({ state: 'fallback', backoffMs: 4000, nextProbeAtMs: 9000, failedProbeCount: 1 });
      expect(rig.controller.tick(9000).state).toBe('probing');
    });

    it('keeps adversarial removal recovery under 35 active seconds at the ceiling', () => {
      const rig = trace();
      rig.probe();
      let startedAt = 2000;
      for (const backoff of [4000, 8000, 16000, 30000]) {
        rig.controller.tick(startedAt + 2000);
        startedAt += backoff;
        rig.controller.tick(startedAt);
      }
      const removedAt = startedAt + 1;
      rig.controller.tick(startedAt + 2250);
      const retryAt = startedAt + 30250;
      rig.controller.tick(retryAt);
      rig.bind(10, retryAt + 250);
      const result = rig.run(repeated(9), retryAt + 400);
      expect(result.state).toBe('stable');
      expect(result.transitions.at(-1)!.atMs - removedAt).toBeLessThanOrEqual(35000);
      expect(rig.run(repeated(9, 1000), retryAt + 750, 16).state).toBe('stable');
    });
  });

  describe('AK-E: absence, stable age, and explicit clocks', () => {
    it('times out startup and probes even with no samples, without catching up unseen trials', () => {
      const controller = new RuntimeController();
      expect(controller.tick(1999).state).toBe('warmup');
      expect(controller.tick(2000)).toMatchObject({ state: 'fallback', nextProbeAtMs: 4000 });
      expect(controller.tick(50000)).toMatchObject({ state: 'probing', probeCount: 1 });
      expect(controller.tick(51999).state).toBe('probing');
      expect(controller.tick(52000)).toMatchObject({ state: 'fallback', nextProbeAtMs: 54000 });
    });

    it('checks deadlines before a sample could confirm an expired trial', () => {
      const rig = trace();
      rig.run(repeated(9, 34), 150);
      expect(rig.sample(9, 2000)).toMatchObject({ state: 'fallback', samples: 0 });
    });

    it('expires stable evidence at exactly 2000 active ms since submission', () => {
      const rig = trace();
      rig.run(repeated(9), 150);
      rig.sample(9, 600, { submittedAt: 500 });
      expect(rig.controller.tick(2499).state).toBe('stable');
      expect(rig.controller.tick(2500)).toMatchObject({ state: 'fallback', reason: 'stable neural evidence expired' });
    });

    it('does not count a pre-construction gap or a suspended gap as active time', () => {
      const controller = new RuntimeController({ active: false }, 50000);
      expect(controller.tick(100000)).toMatchObject({ activeMs: 0, state: 'suspended', tier: 'neural' });
      controller.setActive(true, 100000);
      expect(controller.tick(101999).state).toBe('warmup');
      expect(controller.tick(102000)).toMatchObject({ state: 'fallback', activeMs: 2000 });
    });

    it('snapshot advances accounting but only tick or accepted records evaluate deadlines', () => {
      const controller = new RuntimeController();
      expect(controller.snapshot(2000)).toMatchObject({ state: 'warmup', activeMs: 2000 });
      expect(controller.tick(2000).state).toBe('fallback');
      expect(controller.snapshot(2500).fallbackMs).toBe(500);
    });

    it.each([NaN, Infinity, -Infinity, -1])('rejects invalid initial time %s', now => {
      expect(() => new RuntimeController({}, now)).toThrow(RangeError);
    });

    it.each(['setMode', 'setAvailable', 'setActive', 'setWorkload', 'invalidate', 'bindGeneration', 'setForced', 'fail', 'record', 'tick', 'snapshot'])('validates monotonic finite time in %s', method => {
      const controller = new RuntimeController({}, 100);
      const calls: Record<string, (now: number) => unknown> = {
        setMode: now => controller.setMode('baseline', now), setAvailable: now => controller.setAvailable(false, now),
        setActive: now => controller.setActive(false, now), setWorkload: now => controller.setWorkload(1280, 720, 60, now),
        invalidate: now => controller.invalidate(now), bindGeneration: now => controller.bindGeneration(1, now),
        setForced: now => controller.setForced(true, now), fail: now => controller.fail('fatal', now),
        record: now => controller.record({ ms: 1, generation: 1, sequence: 1, submittedAt: 100, resolvedAt: 100, neural: true }, now),
        tick: now => controller.tick(now), snapshot: now => controller.snapshot(now),
      };
      for (const now of [NaN, Infinity, -Infinity, 99]) expect(() => calls[method]!(now)).toThrow(RangeError);
      expect(controller.snapshot(100)).toMatchObject({ state: 'warmup', transitionCount: 0, activeMs: 0 });
    });
  });

  describe('AK-F: source identity, order, delivery age, and baseline rejection', () => {
    it.each([
      { ms: NaN }, { ms: Infinity }, { ms: -1 }, { ms: 0 }, { neural: false },
      { generation: 0 }, { generation: 2 }, { generation: NaN },
      { sequence: 34 }, { sequence: 33 }, { sequence: -1 }, { sequence: 35.5 }, { sequence: Infinity },
      { submittedAt: 489 }, { submittedAt: 502 }, { submittedAt: NaN },
      { resolvedAt: 489 }, { resolvedAt: 502 }, { resolvedAt: Infinity },
    ])('ignores invalid evidence without advancing dwell or sample counters: %j', overrides => {
      const rig = trace();
      const before = rig.run(repeated(14), 150);
      expect(policy(rig.sample(20, 501, overrides))).toEqual(policy(before));
    });

    it('accepts exactly 500 ms delivery age, rejects 501, and does not poison the sequence watermark', () => {
      const rig = trace();
      expect(rig.sample(9, 650, { submittedAt: 150, sequence: 1 })).toMatchObject({ samples: 1 });
      expect(rig.sample(99, 651, { submittedAt: 150, sequence: 999 })).toMatchObject({ samples: 1, medianMs: 9 });
      expect(rig.sample(9, 652, { sequence: 2 })).toMatchObject({ samples: 2 });
    });

    it('rejects excessive age between resolution and record processing, including past a trial deadline', () => {
      const rig = trace();
      const before = rig.controller.snapshot(0);
      expect(policy(rig.sample(9, 2100, { submittedAt: 150, resolvedAt: 200 }))).toEqual(policy(before));
      expect(rig.controller.tick(2100).state).toBe('fallback');
    });

    it('requires explicit binding and never adopts a generation from a record', () => {
      const rig = trace();
      rig.controller.invalidate(100);
      expect(rig.sample(9, 150)).toMatchObject({ generation: null, samples: 0 });
      rig.bind(2, 200);
      expect(rig.sample(9, 201, { generation: 1 })).toMatchObject({ generation: 2, samples: 0 });
      expect(rig.sample(9, 202)).toMatchObject({ generation: 2, samples: 1 });
    });

    it('retains sequence identity across binding, seeks, and actual source replacements', () => {
      const rig = trace();
      rig.sample(9, 0, { sequence: 100 });
      rig.controller.invalidate(1);
      rig.bind(2, 1);
      expect(rig.sample(9, 2, { sequence: 100 }).samples).toBe(0);
      expect(rig.sample(9, 3, { sequence: 101 }).samples).toBe(1);
      rig.controller.setWorkload(1920, 1080, 30, 4);
      rig.bind(3, 4);
      expect(rig.sample(9, 5, { sequence: 101 }).samples).toBe(0);
      expect(rig.sample(9, 6, { sequence: 102 }).samples).toBe(1);
    });

    it('rejects submissions predating a same-generation binding', () => {
      const rig = trace();
      rig.bind(1, 100);
      expect(rig.sample(9, 150, { submittedAt: 99 }).samples).toBe(0);
      expect(rig.sample(9, 151, { submittedAt: 100 }).samples).toBe(1);
    });

    it('baseline evidence and late neural deliveries cannot start or certify a probe', () => {
      const rig = trace();
      rig.controller.setForced(true, 0);
      rig.controller.setForced(false, 1);
      for (const neural of [false, true]) expect(rig.sample(1, 50000, { neural })).toMatchObject({ state: 'fallback', probeCount: 0, samples: 0 });
      expect(rig.controller.tick(50000).state).toBe('probing');
      rig.bind(2, 50000);
      for (let index = 0; index < 100; index++) rig.sample(1, 50200 + index, { neural: false });
      expect(rig.controller.tick(52000)).toMatchObject({ state: 'fallback', failedProbeCount: 1 });
    });
  });

  describe('AK-G: seek, workload, and generation reconfiguration', () => {
    it('ordinary stable seek rewarms against failure, not the stricter probe recovery threshold', () => {
      const rig = trace({ mode: 'neural' });
      rig.run(repeated(14), 150);
      expect(rig.controller.invalidate(500)).toMatchObject({ mode: 'neural', state: 'warmup', tier: 'neural', generation: null, samples: 0 });
      rig.bind(2, 500);
      expect(rig.run(repeated(14), 650)).toMatchObject({ state: 'stable', probeCount: 0 });
    });

    it('seek during a probe preserves original deadline and recovery qualification', () => {
      const rig = trace();
      rig.probe();
      rig.run(repeated(14), 2150);
      expect(rig.controller.invalidate(2500)).toMatchObject({ state: 'probing', tier: 'neural', backoffMs: 2000 });
      rig.bind(3, 2500);
      expect(rig.run(repeated(14), 2650).state).toBe('probing');
      expect(rig.controller.tick(4000)).toMatchObject({ state: 'fallback', failedProbeCount: 1, nextProbeAtMs: 6000 });
    });

    it('seek near the end of a probe cannot restart its cumulative deadline', () => {
      const rig = trace();
      rig.probe();
      rig.controller.invalidate(3999);
      rig.bind(3, 3999);
      expect(rig.controller.tick(4000).state).toBe('fallback');
    });

    it('same workload and repeated seek preserve failed-probe backoff and remaining wait', () => {
      const rig = trace();
      rig.controller.setWorkload(1280, 720, 60, 0);
      rig.probe();
      rig.run(repeated(21), 2150);
      expect(rig.controller.setWorkload(1280, 720, 60, 2500)).toMatchObject({ state: 'fallback', backoffMs: 4000, nextProbeAtMs: 6000 });
      expect(rig.controller.invalidate(2600)).toMatchObject({ state: 'fallback', backoffMs: 4000, nextProbeAtMs: 6000 });
      expect(rig.controller.setMode('neural', 2700)).toMatchObject({ state: 'fallback', backoffMs: 4000, nextProbeAtMs: 6000 });
      expect(rig.controller.tick(5999).state).toBe('fallback');
    });

    it.each([[1920, 720, 60], [1280, 1080, 60], [1280, 720, 30]])('resets backoff only for actual workload changes: %s/%s/%s', (width, height, fps) => {
      const rig = trace();
      rig.controller.setWorkload(1280, 720, 60, 0);
      rig.probe();
      rig.run(repeated(21), 2150);
      expect(rig.controller.setWorkload(width, height, fps, 2500)).toMatchObject({ state: 'warmup', backoffMs: 2000, generation: null, samples: 0, nextProbeAtMs: null });
    });

    it('covers 720 to 1080 to 720 without accepting old-generation evidence', () => {
      const rig = trace();
      for (const [index, width] of [1280, 1920, 1280].entries()) {
        const now = index * 1000;
        rig.controller.setWorkload(width, width === 1920 ? 1080 : 720, 60, now);
        rig.bind(index + 2, now);
        expect(rig.sample(1, now, { generation: index + 1 }).samples).toBe(0);
        expect(rig.run(repeated(9), now + 150).state).toBe('stable');
      }
    });

    it('binding clears window, cold count, and evidence age without extending startup', () => {
      const rig = trace();
      rig.run(repeated(14, 34), 150);
      expect(rig.bind(2, 1990)).toMatchObject({ state: 'warmup', samples: 0, medianMs: null });
      expect(rig.controller.tick(2000).state).toBe('fallback');
    });

    it('binding resets stable evidence age but requires fresh cold exclusion', () => {
      const rig = trace();
      rig.run(repeated(14), 150);
      rig.bind(2, 2400);
      expect(rig.run(repeated(99, 3), 2550).state).toBe('stable');
      expect(rig.controller.tick(4399).state).toBe('stable');
      expect(rig.controller.tick(4400).state).toBe('fallback');
    });
  });

  describe('AK-H: pause and background active-clock suspension', () => {
    it.each(['manual', 'availability', 'workload'] as const)('holds the selected baseline tier while suspended through %s eligibility changes', event => {
      const rig = trace();
      rig.controller.setForced(true, 0);
      rig.controller.setForced(false, 1);
      rig.controller.setActive(false, 100);
      if (event === 'manual') {
        rig.controller.setMode('baseline', 200);
        rig.controller.setMode('auto', 300);
      } else if (event === 'availability') {
        rig.controller.setAvailable(false, 200);
        rig.controller.setAvailable(true, 300);
      } else rig.controller.setWorkload(1920, 1080, 60, 300);
      expect(rig.controller.tick(10000)).toMatchObject({ state: 'suspended', tier: 'baseline', probeCount: 0, activeMs: 100 });
      expect(rig.controller.setActive(true, 10000)).toMatchObject({ state: 'warmup', tier: 'neural', generation: null, samples: 0 });
      expect(rig.controller.tick(11999).state).toBe('warmup');
      expect(rig.controller.tick(12000).state).toBe('fallback');
    });

    it('holds the selected neural tier, rejects samples, and rewarms on resume', () => {
      const rig = trace();
      rig.run(repeated(14), 150);
      expect(rig.controller.setActive(false, 500)).toMatchObject({ state: 'suspended', tier: 'neural', activeMs: 500 });
      expect(rig.sample(99, 50000)).toMatchObject({ state: 'suspended', samples: 30, activeMs: 500 });
      expect(rig.controller.tick(60000)).toMatchObject({ state: 'suspended', activeMs: 500, probeCount: 0 });
      expect(rig.controller.setActive(true, 60000)).toMatchObject({ state: 'warmup', tier: 'neural', generation: null, samples: 0 });
      rig.bind(2, 60000);
      expect(rig.run(repeated(14), 60150).state).toBe('stable');
    });

    it('freezes fallback waiting and fallback duration, then resumes on baseline', () => {
      const rig = trace();
      rig.controller.setForced(true, 0);
      rig.controller.setForced(false, 1);
      expect(rig.controller.setActive(false, 500)).toMatchObject({ state: 'suspended', tier: 'baseline', fallbackMs: 500 });
      expect(rig.controller.tick(10000)).toMatchObject({ activeMs: 500, fallbackMs: 500, probeCount: 0 });
      expect(rig.controller.setActive(true, 10000)).toMatchObject({ state: 'fallback', tier: 'baseline', nextProbeAtMs: 11500 });
      expect(rig.controller.tick(11499).state).toBe('fallback');
      expect(rig.controller.tick(11500)).toMatchObject({ state: 'probing', fallbackMs: 2000 });
    });

    it('preserves an interrupted probe and its remaining active deadline', () => {
      const rig = trace();
      rig.probe();
      rig.controller.setActive(false, 2100);
      rig.controller.tick(10000);
      expect(rig.controller.setActive(true, 10000)).toMatchObject({ state: 'probing', tier: 'neural', activeMs: 2100 });
      rig.bind(3, 10000);
      expect(rig.run(repeated(14), 10150).state).toBe('probing');
      expect(rig.controller.tick(11899).state).toBe('probing');
      expect(rig.controller.tick(11900)).toMatchObject({ state: 'fallback', backoffMs: 4000, nextProbeAtMs: 13900 });
    });

    it('does not allow hidden force, manual, availability, or source changes to initiate probes', () => {
      const rig = trace({ active: false });
      expect(rig.controller.setMode('baseline', 1)).toMatchObject({ state: 'suspended', tier: 'baseline' });
      rig.controller.setForced(true, 2);
      rig.controller.setMode('neural', 3);
      expect(rig.controller.setAvailable(false, 4)).toMatchObject({ state: 'suspended', tier: 'baseline' });
      rig.controller.setWorkload(1920, 1080, 30, 5);
      rig.controller.setAvailable(true, 6);
      rig.controller.setForced(false, 7);
      expect(rig.controller.tick(50000)).toMatchObject({ state: 'suspended', tier: 'baseline', probeCount: 0, activeMs: 0 });
      expect(rig.controller.setActive(true, 50000)).toMatchObject({ state: 'fallback', nextProbeAtMs: 52000 });
    });
  });

  describe('AK-I: manual intent, late availability, and force', () => {
    it.each(['auto', 'neural'] as const)('%s never probes when usable timestamps are unavailable', mode => {
      const rig = trace({ mode, available: false });
      rig.run(repeated(1), 150);
      expect(rig.controller.tick(100000)).toMatchObject({ state: 'unavailable', tier: 'baseline', probeCount: 0, samples: 0, nextProbeAtMs: null });
    });

    it('manual baseline survives late availability, invalidation, workload, pause, and bind callbacks', () => {
      const rig = trace({ available: false });
      rig.controller.setMode('baseline', 1);
      rig.controller.setAvailable(true, 2);
      rig.controller.invalidate(3);
      rig.controller.setWorkload(1920, 1080, 30, 4);
      rig.controller.setActive(false, 5);
      rig.bind(2, 6);
      rig.controller.setActive(true, 7);
      expect(rig.controller.tick(100000)).toMatchObject({ mode: 'baseline', state: 'manual-baseline', tier: 'baseline', probeCount: 0 });
    });

    it('cancels an in-flight probe on manual baseline and starts ordinary warmup only on explicit neural intent', () => {
      const rig = trace();
      rig.probe();
      expect(rig.controller.setMode('baseline', 2100)).toMatchObject({ state: 'manual-baseline', tier: 'baseline', nextProbeAtMs: null });
      rig.bind(3, 2200);
      expect(rig.run(repeated(1), 2350).state).toBe('manual-baseline');
      expect(rig.controller.tick(10000)).toMatchObject({ state: 'manual-baseline', probeCount: 1, failedProbeCount: 0 });
      expect(rig.controller.setMode('neural', 10001)).toMatchObject({ state: 'warmup', tier: 'neural' });
    });

    it('late availability honors force and current mode instead of overwriting them', () => {
      const rig = trace({ available: false });
      rig.controller.setForced(true, 1);
      expect(rig.controller.setAvailable(true, 2)).toMatchObject({ state: 'fallback', forced: true, tier: 'baseline' });
      rig.controller.setMode('baseline', 3);
      rig.controller.setWorkload(1280, 720, 60, 4);
      expect(rig.controller.setMode('neural', 5)).toMatchObject({ state: 'fallback', forced: true });
      expect(rig.controller.tick(50000)).toMatchObject({ state: 'fallback', probeCount: 0 });
      expect(rig.controller.setForced(false, 50000).state).toBe('fallback');
      expect(rig.controller.tick(50000).state).toBe('probing');
    });

    it('idempotent events preserve evidence and force release never shortens its wait', () => {
      const rig = trace();
      const stable = rig.run(repeated(14), 150);
      expect(policy(rig.controller.setMode('auto', 490))).toEqual(policy(stable));
      expect(policy(rig.controller.setAvailable(true, 490))).toEqual(policy(stable));
      expect(policy(rig.controller.setActive(true, 490))).toEqual(policy(stable));
      expect(policy(rig.controller.setForced(false, 490))).toEqual(policy(stable));
      rig.controller.setForced(true, 500);
      expect(rig.controller.setForced(true, 600).nextProbeAtMs).toBe(2500);
      expect(rig.controller.setForced(false, 700).nextProbeAtMs).toBe(2500);
      expect(rig.controller.tick(2499).state).toBe('fallback');
      expect(rig.controller.tick(2500).state).toBe('probing');
    });

    it('late availability can initiate warmup only for currently eligible intent', () => {
      const rig = trace({ available: false });
      rig.controller.setMode('neural', 1);
      expect(rig.controller.setAvailable(true, 10000)).toMatchObject({ mode: 'neural', state: 'warmup', tier: 'neural', generation: null });
      rig.bind(2, 10000);
      expect(rig.run(repeated(14), 10150).state).toBe('stable');
    });
  });

  describe('AK-J: terminal failures and input bounds', () => {
    it.each(['warmup', 'stable', 'fallback', 'probing', 'manual-baseline', 'suspended', 'unavailable'] as const)('failure is terminal from %s through every public mutator', state => {
      const rig = trace();
      if (state === 'stable') rig.run(repeated(9), 150);
      if (state === 'fallback') rig.controller.setForced(true, 0);
      if (state === 'probing') rig.probe();
      if (state === 'manual-baseline') rig.controller.setMode('baseline', 0);
      if (state === 'suspended') rig.controller.setActive(false, 0);
      if (state === 'unavailable') rig.controller.setAvailable(false, 0);
      const failed = rig.controller.fail('device lost', 3000);
      expect(failed).toMatchObject({ state: 'failed', tier: 'baseline', reason: 'device lost', nextProbeAtMs: null });
      const results = [
        rig.controller.setMode('auto', 4000), rig.controller.setMode('neural', 4001), rig.controller.setMode('baseline', 4002),
        rig.controller.setAvailable(true, 4003), rig.controller.setAvailable(false, 4004),
        rig.controller.setActive(false, 4005), rig.controller.setActive(true, 4006),
        rig.controller.setWorkload(1920, 1080, 30, 4007), rig.controller.invalidate(4008), rig.bind(2, 4009),
        rig.controller.setForced(true, 4010), rig.controller.setForced(false, 4011), rig.sample(1, 4012),
        rig.controller.fail('later failure', 4013), rig.controller.tick(100000), rig.controller.snapshot(200000),
      ];
      for (const result of results) expect(result).toEqual(failed);
    });

    it.each([-1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])('rejects invalid dimensions and generation %s', value => {
      const controller = new RuntimeController();
      expect(() => controller.bindGeneration(value, 0)).toThrow(RangeError);
      expect(() => controller.setWorkload(value, 720, 60, 0)).toThrow(RangeError);
      expect(() => controller.setWorkload(1280, value, 60, 0)).toThrow(RangeError);
      expect(controller.snapshot(0).transitionCount).toBe(0);
    });
  });

  describe('AK-K: bounded lifetime telemetry', () => {
    it('bounds the transition log to 128 while preserving lifetime totals and snapshot isolation', () => {
      const controller = new RuntimeController();
      for (let index = 1; index <= 200; index++) controller.setMode(index % 2 ? 'baseline' : 'auto', index);
      const result = controller.snapshot(200);
      expect(result).toMatchObject({ transitionCount: 200, discardedTransitions: 72 });
      expect(result.transitions).toHaveLength(128);
      expect(result.transitions[0]).toMatchObject({ atMs: 73, activeMs: 73, from: 'warmup', to: 'manual-baseline', tier: 'baseline' });
      result.transitions[0]!.reason = 'tampered';
      result.transitions.length = 0;
      expect(controller.snapshot(200).transitions).toHaveLength(128);
      expect(controller.snapshot(200).transitions[0]!.reason).toBe('manual baseline intent');
    });

    it('counts active fallback across tier switches but not manual baseline or suspension', () => {
      const rig = trace();
      rig.probe();
      rig.run(repeated(21), 2150);
      expect(rig.controller.setActive(false, 3000).fallbackMs).toBe(2510);
      rig.controller.setActive(true, 10000);
      expect(rig.controller.tick(13000)).toMatchObject({ state: 'probing', probeCount: 2, failedProbeCount: 1, fallbackMs: 5510 });
      rig.controller.setMode('baseline', 13000);
      expect(rig.controller.snapshot(14000)).toMatchObject({ fallbackMs: 5510, activeMs: 7000, probeCount: 2, failedProbeCount: 1 });
    });
  });
});