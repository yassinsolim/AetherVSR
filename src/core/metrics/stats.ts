/**
 * Pure, allocation-free statistics helpers for the per-frame hot path.
 *
 * Everything here is deliberately dependency-free and synchronous so it can be
 * unit tested without a GPU, a browser, or a video element.
 */

/**
 * Fixed-capacity ring buffer of `number` samples backed by a single
 * `Float64Array`. Pushing never allocates; quantile queries allocate one
 * scratch array that is reused across calls.
 */
export class SampleWindow {
  readonly capacity: number;
  private readonly samples: Float64Array;
  private readonly scratch: Float64Array;
  private write = 0;
  private filled = 0;
  private sum = 0;

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new RangeError(`SampleWindow capacity must be a positive integer, got ${capacity}`);
    }
    this.capacity = capacity;
    this.samples = new Float64Array(capacity);
    this.scratch = new Float64Array(capacity);
  }

  /** Number of samples currently retained (<= capacity). */
  get size(): number {
    return this.filled;
  }

  push(value: number): void {
    if (!Number.isFinite(value)) return;
    if (this.filled === this.capacity) {
      this.sum -= this.samples[this.write] as number;
    } else {
      this.filled++;
    }
    this.samples[this.write] = value;
    this.sum += value;
    this.write = (this.write + 1) % this.capacity;
  }

  reset(): void {
    this.write = 0;
    this.filled = 0;
    this.sum = 0;
  }

  /** Arithmetic mean, or NaN when empty. */
  mean(): number {
    return this.filled === 0 ? Number.NaN : this.sum / this.filled;
  }

  /** Most recently pushed sample, or NaN when empty. */
  last(): number {
    if (this.filled === 0) return Number.NaN;
    const idx = (this.write - 1 + this.capacity) % this.capacity;
    return this.samples[idx] as number;
  }

  /**
   * Linear-interpolated quantile in [0, 1]. Returns NaN when empty.
   *
   * Sorting is done on a preallocated scratch buffer; the cost is O(n log n)
   * but this is only ever called from the (throttled) overlay refresh, never
   * from the frame loop.
   */
  quantile(q: number): number {
    if (!(q >= 0 && q <= 1)) throw new RangeError(`quantile must be in [0,1], got ${q}`);
    if (this.filled === 0) return Number.NaN;
    const n = this.filled;
    const view = this.scratch.subarray(0, n);
    view.set(this.samples.subarray(0, n));
    view.sort();
    const pos = (n - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    const loValue = view[lo] as number;
    if (lo === hi) return loValue;
    const hiValue = view[hi] as number;
    return loValue + (hiValue - loValue) * (pos - lo);
  }

  max(): number {
    if (this.filled === 0) return Number.NaN;
    let m = -Infinity;
    for (let i = 0; i < this.filled; i++) {
      const v = this.samples[i] as number;
      if (v > m) m = v;
    }
    return m;
  }
}

/**
 * Sliding-window rate meter. `mark()` records one or more events at a
 * timestamp; `rate()` reports events per second over the trailing `windowMs`.
 *
 * The estimator is `(weight in window - weight of the oldest entry) / span`,
 * i.e. it counts the *intervals* covered by the retained samples rather than
 * the samples themselves. Counting samples over the window width overstates a
 * periodic signal by exactly one event (61 vsync marks fall inside a 1000 ms
 * window at 60 Hz), and dividing by the nominal window instead of the observed
 * span understates the rate until the window fills. This form is exact for a
 * periodic source, correct from the second sample onward, and — unlike a
 * sample count — scales linearly with `weight`, which is what makes weighted
 * marks usable for attributing skipped presented frames.
 */
export class RateMeter {
  private readonly times: Float64Array;
  private readonly weights: Float64Array;
  private write = 0;
  private count = 0;

  constructor(
    readonly windowMs: number,
    capacity = 256,
  ) {
    if (!(windowMs > 0)) throw new RangeError(`windowMs must be > 0, got ${windowMs}`);
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new RangeError(`capacity must be a positive integer, got ${capacity}`);
    }
    this.times = new Float64Array(capacity);
    this.weights = new Float64Array(capacity);
  }

  /** Record `weight` events that occurred at `nowMs`. */
  mark(nowMs: number, weight = 1): void {
    if (weight <= 0) return;
    this.times[this.write] = nowMs;
    this.weights[this.write] = weight;
    this.write = (this.write + 1) % this.times.length;
    if (this.count < this.times.length) this.count++;
  }

  reset(): void {
    this.write = 0;
    this.count = 0;
  }

  /** Events per second observed within `windowMs` of `nowMs`. */
  rate(nowMs: number): number {
    const cutoff = nowMs - this.windowMs;
    let total = 0;
    let oldest = Infinity;
    let oldestWeight = 0;
    for (let i = 0; i < this.count; i++) {
      const t = this.times[i] as number;
      if (t < cutoff) continue;
      total += this.weights[i] as number;
      if (t < oldest) {
        oldest = t;
        oldestWeight = this.weights[i] as number;
      }
    }
    const span = nowMs - oldest;
    if (!(span > 0)) return 0;
    return ((total - oldestWeight) / span) * 1000;
  }
}

/**
 * Wall-clock interval between frames at 60 Hz, in milliseconds.
 *
 * This is the total budget for decode + upscale + composite + present. The
 * upscale stage alone must stay well under it; see AGENTS.md.
 */
export const FRAME_BUDGET_60HZ_MS = 1000 / 60;
