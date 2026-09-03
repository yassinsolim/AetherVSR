import { describe, expect, it } from 'vitest';
import { BudgetGuard, DEFAULT_BUDGET_GUARD } from '../src/core/upscale/budget-guard.js';

/**
 * The guard's job is to switch at the right moment and never dither, so these
 * tests are about *when* it changes state. The regression that matters most is
 * the one independent review reproduced on live video: recovery judged from the
 * cheaper baseline's timings makes a permanently over-budget machine cycle
 * forever.
 */

/** Feeds n samples at a fixed cost, advancing a clock, and returns the last decision. */
function feed(guard: BudgetGuard, ms: number, n: number, startMs = 0, stepMs = 16.67) {
  let last = guard.record(ms, startMs);
  for (let i = 1; i < n; i++) last = guard.record(ms, startMs + i * stepMs);
  return last;
}

describe('BudgetGuard', () => {
  it('starts neural and stays there while the stage fits', () => {
    const guard = new BudgetGuard();
    const d = feed(guard, 5.5, 120);
    expect(d.state).toBe('neural');
    expect(d.changed).toBe(false);
    expect(d.medianMs).toBeCloseTo(5.5, 5);
  });

  it('will not act before it has a full window of evidence', () => {
    const guard = new BudgetGuard({ window: 30, dwell: 1 });
    const d = feed(guard, 999, 29);
    expect(d.state).toBe('neural');
    expect(d.reason).toContain('gathering evidence');
  });

  it('falls back once the median is over budget for the dwell', () => {
    const guard = new BudgetGuard({ window: 10, dwell: 3 });
    feed(guard, 12, 10);
    expect(guard.current).toBe('neural');
    const a = guard.record(12, 200);
    expect(a.changed).toBe(false);
    const b = guard.record(12, 220);
    expect(b.changed).toBe(true);
    expect(b.state).toBe('fallback');
  });

  it('ignores a single slow frame', () => {
    const guard = new BudgetGuard({ window: 30 });
    feed(guard, 5, 30);
    const d = guard.record(40, 600);
    expect(d.state).toBe('neural');
    expect(d.changed).toBe(false);
  });

  it('does not oscillate when the stage sits between the thresholds', () => {
    const guard = new BudgetGuard();
    let changes = 0;
    for (let i = 0; i < 500; i++) if (feed(guard, 8.5, 1, i * 16.67).changed) changes++;
    expect(changes).toBe(0);
    expect(guard.current).toBe('neural');
  });

  /**
   * The defect independent review found. A machine whose neural stage costs
   * 14 ms and whose baseline costs 3 ms must settle on the baseline. Feeding
   * the guard only neural samples - as the contract now requires - it does.
   */
  it('does not cycle when the stage is permanently over budget', () => {
    const guard = new BudgetGuard({ window: 5, dwell: 2 });
    let now = 0;
    let neuralFrames = 0;
    const total = 6000; // ~100 s at 60 fps
    for (let i = 0; i < total; i++) {
      now += 16.67;
      if (guard.current === 'neural') {
        neuralFrames++;
        // Only ever neural timings, and always over budget.
        guard.record(14, now);
      } else {
        guard.tick(now);
      }
    }
    expect(guard.current).toBe('fallback');
    // The backoff decays the probe rate, so the machine spends almost all of
    // its time on the baseline instead of flapping. The old implementation
    // sat at roughly half and half.
    expect(neuralFrames / total).toBeLessThan(0.05);
  });

  it('backs off further after each failed probe', () => {
    const guard = new BudgetGuard({ window: 4, dwell: 1, probeBackoffMs: 100, maxProbeBackoffMs: 10_000 });
    let now = 0;
    const gaps: number[] = [];
    let fellBackAt = 0;
    for (let round = 0; round < 4; round++) {
      // Drive it over budget until it falls back.
      while (guard.current === 'neural') {
        now += 16.67;
        if (guard.record(14, now).changed) fellBackAt = now;
      }
      // Wait for the probe.
      while (guard.current === 'fallback') {
        now += 16.67;
        if (guard.tick(now).changed) gaps.push(now - fellBackAt);
      }
    }
    for (let i = 1; i < gaps.length; i++) {
      expect(gaps[i] as number).toBeGreaterThan(gaps[i - 1] as number);
    }
  });

  it('recovers only after a probe measures the neural stage itself', () => {
    const guard = new BudgetGuard({ window: 4, dwell: 2, probeBackoffMs: 100 });
    let now = 0;
    while (guard.current === 'neural') {
      now += 16.67;
      guard.record(14, now);
    }
    expect(guard.current).toBe('fallback');

    // A cheap baseline running for a long time must never itself trigger
    // recovery - the guard is not fed those samples, and time alone only earns
    // a probe, not a return to service.
    now += 5_000;
    const probe = guard.tick(now);
    expect(probe.changed).toBe(true);
    expect(probe.probing).toBe(true);

    // The probe now measures the neural stage. Fast samples confirm it.
    const d = feed(guard, 5, 8, now + 16.67);
    expect(d.state).toBe('neural');
    expect(guard.isProbing).toBe(false);
  });

  /**
   * Confirmation review found this: a probe whose median sat between the
   * thresholds entered neither branch, so the guard stayed `probing` forever
   * and the network ran unconfirmed - bypassing the documented rule that
   * recovery requires dropping below recoverMs.
   */
  it('abandons a probe that never reaches the recovery threshold', () => {
    const guard = new BudgetGuard({ window: 4, dwell: 2, probePatience: 5, probeBackoffMs: 50 });
    let now = 0;
    while (guard.current === 'neural') {
      now += 16.67;
      guard.record(14, now);
    }
    now += 100;
    expect(guard.tick(now).changed).toBe(true);
    expect(guard.isProbing).toBe(true);
    // 8.5 ms is between recoverMs 7 and failMs 10: neither confirms nor fails.
    const d = feed(guard, 8.5, 60, now + 16.67);
    expect(guard.current).toBe('fallback');
    expect(guard.isProbing).toBe(false);
    expect(d.state).toBe('fallback');
  });

  /**
   * Sign-off review found the residual: patience was reset on every qualifying
   * evaluation, so an alternating stream - one good window, one bad, forever -
   * accumulated neither counter and probed without bound.
   */
  it('abandons a probe on an alternating stream that never sustains recovery', () => {
    const guard = new BudgetGuard({ window: 3, dwell: 3, probePatience: 60, probeBackoffMs: 50 });
    let now = 0;
    while (guard.current === 'neural') {
      now += 16.67;
      guard.record(14, now);
    }
    now += 100;
    expect(guard.tick(now).changed).toBe(true);
    let ended = false;
    for (let i = 0; i < 2000 && !ended; i++) {
      now += 16.67;
      // One qualifying window, one not, forever: neither counter accumulated
      // under the old reset, so the probe ran without bound.
      ended = guard.record(i % 2 === 0 ? 6.5 : 8.5, now).state === 'fallback';
    }
    expect(ended).toBe(true);
    expect(guard.isProbing).toBe(false);
  });

  it('a failing probe returns to fallback rather than sticking', () => {
    const guard = new BudgetGuard({ window: 4, dwell: 1, probeBackoffMs: 50 });
    let now = 0;
    while (guard.current === 'neural') {
      now += 16.67;
      guard.record(14, now);
    }
    now += 100;
    expect(guard.tick(now).changed).toBe(true);
    expect(guard.isProbing).toBe(true);
    const d = feed(guard, 14, 8, now + 16.67);
    expect(d.state).toBe('fallback');
  });

  it('forcing failure switches immediately and reports the change to the caller', () => {
    const guard = new BudgetGuard({ probeBackoffMs: 1_000 });
    feed(guard, 5, 40);
    expect(guard.current).toBe('neural');
    // The decision must say it changed, or the caller never swaps the stage and
    // the overlay reports a fallback that is not running.
    const forced = guard.setForced(true, 1_000);
    expect(forced.changed).toBe(true);
    expect(forced.state).toBe('fallback');
  });

  it('releasing a forced failure still waits out the probe backoff', () => {
    const guard = new BudgetGuard({ probeBackoffMs: 1_000 });
    feed(guard, 5, 40);
    guard.setForced(true, 1_000);
    guard.setForced(false, 1_100);
    // Only 100 ms have passed against a 1 s backoff.
    expect(guard.tick(1_200).changed).toBe(false);
    expect(guard.current).toBe('fallback');
    // Once the backoff elapses, a probe starts.
    expect(guard.tick(2_150).changed).toBe(true);
    expect(guard.isProbing).toBe(true);
  });

  it('averages the two middle samples on an even window', () => {
    const guard = new BudgetGuard({ window: 4, dwell: 99 });
    guard.record(1, 0);
    guard.record(2, 1);
    guard.record(3, 2);
    const d = guard.record(10, 3);
    // Sorted [1,2,3,10] -> (2+3)/2 = 2.5, not the upper middle 3.
    expect(d.medianMs).toBeCloseTo(2.5, 6);
  });

  it('rejects non-finite and non-positive samples', () => {
    const guard = new BudgetGuard({ window: 4 });
    guard.record(Number.NaN, 0);
    guard.record(0, 1);
    guard.record(-5, 2);
    const d = guard.record(6, 3);
    expect(d.samples).toBe(1);
  });

  it('defaults leave a gap between falling back and recovering', () => {
    expect(DEFAULT_BUDGET_GUARD.failMs).toBeGreaterThan(DEFAULT_BUDGET_GUARD.recoverMs);
  });
});
