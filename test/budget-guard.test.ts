import { describe, expect, it } from 'vitest';
import { BudgetGuard, DEFAULT_BUDGET_GUARD } from '../src/core/upscale/budget-guard.js';

/**
 * The guard's whole job is to switch at the right moment and never dither, so
 * these tests are about *when* it changes state rather than about plumbing.
 */
const feed = (guard: BudgetGuard, ms: number, times: number): void => {
  for (let i = 0; i < times; i++) guard.record(ms);
};

describe('BudgetGuard', () => {
  it('stays neural while the stage is inside budget', () => {
    const guard = new BudgetGuard();
    feed(guard, 5.5, 100);
    expect(guard.current).toBe('neural');
  });

  it('does not judge before it has a full window', () => {
    const guard = new BudgetGuard({ window: 30 });
    // Wildly over budget, but only a handful of frames.
    for (let i = 0; i < 29; i++) {
      const d = guard.record(50);
      expect(d.state).toBe('neural');
      expect(d.reason).toBe('warming up');
    }
    expect(guard.record(50).state).toBe('fallback');
  });

  it('ignores isolated slow frames', () => {
    const guard = new BudgetGuard();
    feed(guard, 5, 29);
    // One catastrophic frame - a tab regaining focus, a cache miss - must not
    // move a median.
    guard.record(400);
    expect(guard.current).toBe('neural');
  });

  it('falls back on a sustained overrun', () => {
    const guard = new BudgetGuard();
    feed(guard, 5, 30);
    expect(guard.current).toBe('neural');
    feed(guard, 12, 30);
    expect(guard.current).toBe('fallback');
  });

  it('requires the recovery to hold for the dwell, not just qualify once', () => {
    const guard = new BudgetGuard({ window: 10, recoverDwell: 3 });
    feed(guard, 12, 10);
    expect(guard.current).toBe('fallback');

    // Feed good frames one at a time and find the first record whose median
    // clears the recovery threshold. Recovery must not happen on that record,
    // nor on the next, but on the third.
    let firstQualifying = -1;
    for (let i = 0; i < 10; i++) {
      const d = guard.record(5);
      if (firstQualifying < 0 && d.medianMs <= 7) {
        firstQualifying = i;
        expect(d.state).toBe('fallback');
      }
      if (firstQualifying >= 0 && i === firstQualifying + 1) expect(d.state).toBe('fallback');
      if (firstQualifying >= 0 && i === firstQualifying + 2) {
        expect(d.state).toBe('neural');
        expect(d.changed).toBe(true);
        break;
      }
    }
    expect(firstQualifying).toBeGreaterThanOrEqual(0);
    expect(guard.current).toBe('neural');
  });

  it('does not oscillate at the threshold', () => {
    // Sitting exactly between the two thresholds must produce no switching at
    // all: this is the case a single threshold would flap on every window.
    const guard = new BudgetGuard();
    const between = (DEFAULT_BUDGET_GUARD.fallbackMs + DEFAULT_BUDGET_GUARD.recoverMs) / 2;
    feed(guard, between, 30);
    let changes = 0;
    for (let i = 0; i < 500; i++) {
      if (guard.record(between).changed) changes++;
    }
    expect(changes).toBe(0);
    expect(guard.current).toBe('neural');
  });

  it('refuses thresholds that would guarantee oscillation', () => {
    expect(() => new BudgetGuard({ fallbackMs: 8, recoverMs: 8 })).toThrow(/oscillate/);
    expect(() => new BudgetGuard({ fallbackMs: 8, recoverMs: 9 })).toThrow(/oscillate/);
  });

  it('forces fallback on demand and releases it', () => {
    const guard = new BudgetGuard();
    feed(guard, 4, 30);
    expect(guard.current).toBe('neural');
    guard.setForced(true);
    const forced = guard.record(4);
    expect(forced.state).toBe('fallback');
    expect(forced.changed).toBe(true);
    expect(forced.reason).toMatch(/forced/);
    // Releasing must not snap straight back: the ordinary dwell still applies.
    guard.setForced(false);
    expect(guard.record(4).state).toBe('fallback');
    guard.record(4);
    expect(guard.record(4).state).toBe('neural');
  });

  it('reports the median it actually decided on', () => {
    const guard = new BudgetGuard({ window: 5 });
    for (const v of [1, 2, 3, 4, 100]) guard.record(v);
    expect(guard.record(3).medianMs).toBe(3);
  });

  it('ignores non-finite and non-positive samples', () => {
    const guard = new BudgetGuard({ window: 4 });
    guard.record(Number.NaN);
    guard.record(0);
    guard.record(-5);
    const d = guard.record(6);
    expect(d.samples).toBe(1);
  });
});
