import { describe, expect, it } from 'vitest';
import { FRAME_BUDGET_60HZ_MS, RateMeter, SampleWindow } from '../src/core/metrics/stats.js';

describe('SampleWindow', () => {
  it('reports NaN rather than 0 when no samples exist', () => {
    // A benchmark overlay must never show a confident zero for unmeasured work.
    const w = new SampleWindow(8);
    expect(w.mean()).toBeNaN();
    expect(w.quantile(0.5)).toBeNaN();
    expect(w.max()).toBeNaN();
    expect(w.last()).toBeNaN();
    expect(w.size).toBe(0);
  });

  it('evicts oldest samples and keeps the mean consistent with the retained window', () => {
    const w = new SampleWindow(3);
    for (const v of [1, 2, 3, 4, 5]) w.push(v);
    expect(w.size).toBe(3);
    // Retains 3, 4, 5 — not 1 and 2.
    expect(w.mean()).toBeCloseTo(4, 12);
    expect(w.last()).toBe(5);
    expect(w.max()).toBe(5);
  });

  it('keeps the running sum exact across many wrap-arounds', () => {
    const w = new SampleWindow(4);
    for (let i = 0; i < 1000; i++) w.push(i % 7);
    // Recompute independently from the last four pushed values.
    const tail = [996 % 7, 997 % 7, 998 % 7, 999 % 7];
    expect(w.mean()).toBeCloseTo(tail.reduce((a, b) => a + b, 0) / 4, 12);
  });

  it('interpolates quantiles linearly between order statistics', () => {
    const w = new SampleWindow(5);
    for (const v of [10, 20, 30, 40, 50]) w.push(v);
    expect(w.quantile(0)).toBe(10);
    expect(w.quantile(1)).toBe(50);
    expect(w.quantile(0.5)).toBe(30);
    // pos = 4 * 0.95 = 3.8 -> between 40 and 50.
    expect(w.quantile(0.95)).toBeCloseTo(48, 12);
  });

  it('sorts numerically, not lexicographically', () => {
    // The naive Array.prototype.sort() bug would put 100 before 9.
    const w = new SampleWindow(3);
    w.push(9);
    w.push(100);
    w.push(11);
    expect(w.quantile(0.5)).toBe(11);
    expect(w.max()).toBe(100);
  });

  it('ignores non-finite samples so one bad reading cannot poison the mean', () => {
    const w = new SampleWindow(4);
    w.push(10);
    w.push(Number.NaN);
    w.push(Number.POSITIVE_INFINITY);
    w.push(20);
    expect(w.size).toBe(2);
    expect(w.mean()).toBe(15);
  });

  it('rejects invalid capacities and quantiles', () => {
    expect(() => new SampleWindow(0)).toThrow(RangeError);
    expect(() => new SampleWindow(2.5)).toThrow(RangeError);
    expect(() => new SampleWindow(4).quantile(1.5)).toThrow(RangeError);
  });

  it('clears retained state on reset', () => {
    const w = new SampleWindow(3);
    w.push(1);
    w.reset();
    expect(w.size).toBe(0);
    expect(w.mean()).toBeNaN();
  });
});

describe('RateMeter', () => {
  it('measures a steady 60 Hz cadence as exactly 60 events per second', () => {
    const m = new RateMeter(1000);
    let t = 0;
    for (let i = 0; i < 120; i++) {
      m.mark(t);
      t += FRAME_BUDGET_60HZ_MS;
    }
    // Query at the instant of the last mark. A count-based estimator reports
    // 61 here because 61 marks fall inside a 1000 ms window at 60 Hz.
    expect(m.rate(t - FRAME_BUDGET_60HZ_MS)).toBeCloseTo(60, 6);
  });

  it('is accurate long before the window has filled', () => {
    // Dividing by the nominal window instead of the observed span would report
    // 6 fps here, and the overlay would look broken for the first second.
    const m = new RateMeter(1000);
    for (let i = 0; i < 7; i++) m.mark(i * FRAME_BUDGET_60HZ_MS);
    expect(m.rate(6 * FRAME_BUDGET_60HZ_MS)).toBeCloseTo(60, 6);
  });

  it('excludes events older than the window', () => {
    const m = new RateMeter(1000);
    for (let i = 0; i < 30; i++) m.mark(i * 10); // 30 events in the first 300 ms
    for (let i = 0; i < 10; i++) m.mark(5000 + i * 100); // 10 events much later
    // Only the recent burst is in the trailing second: 10 marks spanning
    // 900 ms is nine intervals of 100 ms, i.e. 10 events per second.
    expect(m.rate(5900)).toBeCloseTo(10, 6);
  });

  it('counts weighted marks, which is how skipped presented frames are attributed', () => {
    const unweighted = new RateMeter(1000);
    const weighted = new RateMeter(1000);
    for (let i = 0; i < 30; i++) {
      unweighted.mark(i * 33.3);
      weighted.mark(i * 33.3, 2);
    }
    const now = 29 * 33.3;
    expect(weighted.rate(now)).toBeCloseTo(unweighted.rate(now) * 2, 6);
  });

  it('returns 0 before any event and after a reset', () => {
    const m = new RateMeter(1000);
    expect(m.rate(0)).toBe(0);
    m.mark(10);
    m.mark(20);
    expect(m.rate(20)).toBeGreaterThan(0);
    m.reset();
    expect(m.rate(20)).toBe(0);
  });

  it('does not report a rate from a single sample, which has no measurable span', () => {
    const m = new RateMeter(1000);
    m.mark(100);
    expect(m.rate(100)).toBe(0);
  });

  it('rejects invalid construction', () => {
    expect(() => new RateMeter(0)).toThrow(RangeError);
    expect(() => new RateMeter(1000, 0)).toThrow(RangeError);
  });
});
