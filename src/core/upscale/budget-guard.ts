/**
 * Decides when the neural stage must give way to the baseline scaler.
 *
 * Milestone 6 owns dynamic quality selection. What is needed here is a
 * mechanism that cannot oscillate, cannot trip on one slow frame, and can be
 * forced into failure on demand so its recovery path is exercised rather than
 * asserted.
 *
 * ## Only the neural stage can vouch for the neural stage
 *
 * An earlier version fed the guard whatever the pipeline was currently timing
 * and treated those samples as recovery evidence. That guarantees a cycle: once
 * the guard falls back, it is measuring the *cheaper* baseline, which clears the
 * recovery threshold immediately, so it returns to neural, blows the budget
 * again, and repeats. Independent review reproduced exactly that on live video -
 * five state changes in eighteen seconds.
 *
 * So `record` accepts neural samples only. While in fallback the guard measures
 * nothing and instead waits out a backoff, then *probes*: it re-enables the
 * neural stage and judges it on its own timings. Each failed probe doubles the
 * backoff to a ceiling, so a machine that genuinely cannot afford the network
 * settles into the baseline instead of flapping forever.
 */

/** Which stage the pipeline should currently be running. */
export type BudgetState = 'neural' | 'fallback';

export interface BudgetGuardOptions {
  /** Fall back when the median whole-stage GPU time exceeds this, in ms. */
  readonly failMs: number;
  /** Only recover when the median drops back below this, in ms. */
  readonly recoverMs: number;
  /** Frames of history the median is taken over. */
  readonly window: number;
  /** Consecutive qualifying evaluations required before a state change. */
  readonly dwell: number;
  /**
   * Total non-qualifying evaluations a probe may accumulate before it is
   * abandoned. Without this a probe whose median sits between `recoverMs` and
   * `failMs` never ends: it neither confirms nor fails, and the network runs
   * unconfirmed indefinitely while the guard reports `probing`.
   *
   * The count is cumulative, not consecutive. Clearing it on each qualifying
   * evaluation let an alternating stream - one good window, one bad, forever -
   * probe without bound, because neither counter ever accumulated.
   */
  readonly probePatience: number;
  /** Wait before the first probe after falling back, in ms. */
  readonly probeBackoffMs: number;
  /** Ceiling for the doubling backoff, in ms. */
  readonly maxProbeBackoffMs: number;
}

export interface BudgetDecision {
  readonly state: BudgetState;
  /** True only on the evaluation that changed the state. */
  readonly changed: boolean;
  /** Median over the current window, or NaN when there is no evidence yet. */
  readonly medianMs: number;
  /** How many samples that median was taken over. */
  readonly samples: number;
  /** True while the neural stage is running only to be measured. */
  readonly probing: boolean;
  /** Human-readable justification, shown in the diagnostic overlay. */
  readonly reason: string;
}

/**
 * 10 ms to fail and 7 ms to recover, against a 16.67 ms frame interval.
 *
 * The gap is the whole point: a single threshold makes the controller dither
 * whenever the stage sits near it. 10 ms leaves roughly 6 ms for decode,
 * composite and present; 7 ms is where the measured stage actually sits, so
 * recovery means "back to normal" rather than "briefly less bad".
 */
export const DEFAULT_BUDGET_GUARD: BudgetGuardOptions = {
  failMs: 10,
  recoverMs: 7,
  window: 30,
  dwell: 3,
  probePatience: 60,
  probeBackoffMs: 2_000,
  maxProbeBackoffMs: 30_000,
};

export class BudgetGuard {
  private readonly options: BudgetGuardOptions;
  private readonly samples: number[] = [];
  private state: BudgetState = 'neural';
  private qualifying = 0;
  private forced = false;
  private probing = false;
  private backoffMs: number;
  private probeAtMs = 0;
  private probeMisses = 0;

  constructor(options: Partial<BudgetGuardOptions> = {}) {
    this.options = { ...DEFAULT_BUDGET_GUARD, ...options };
    this.backoffMs = this.options.probeBackoffMs;
  }

  get current(): BudgetState {
    return this.state;
  }

  get isForced(): boolean {
    return this.forced;
  }

  get isProbing(): boolean {
    return this.probing;
  }

  /**
   * Forces the over-budget condition, so the fallback path can be exercised on
   * demand instead of being asserted. Releasing it does not jump straight back
   * to neural: the normal probe schedule applies, because a forced failure must
   * not have a faster recovery path than a real one.
   *
   * Returns a decision the caller must apply, exactly like `record` and `tick`.
   * Mutating the state without handing back a decision let the overlay report
   * "fallback" while the neural stage was still the thing running.
   */
  setForced(on: boolean, nowMs: number): BudgetDecision {
    if (on === this.forced) {
      return this.decide(false, on ? 'forced over-budget for testing' : 'within budget');
    }
    this.forced = on;
    if (on) {
      const changed = this.state !== 'fallback';
      this.enterFallback();
      // A forced failure schedules its probe like any other, so releasing the
      // force does not shortcut the backoff.
      this.probeAtMs = nowMs + this.backoffMs;
      return this.decide(changed, 'forced over-budget for testing');
    }
    return this.decide(false, this.state === 'fallback' ? 'force released, waiting for probe' : 'within budget');
  }

  /**
   * Records one whole-stage GPU sample **measured while the neural stage was
   * running**. Passing baseline timings here would reintroduce the oscillation
   * this class exists to prevent.
   */
  record(ms: number, nowMs: number): BudgetDecision {
    if (Number.isFinite(ms) && ms > 0) {
      this.samples.push(ms);
      if (this.samples.length > this.options.window) this.samples.shift();
    }

    if (this.forced) {
      return this.decide(false, 'forced over-budget for testing');
    }

    // Nothing to judge yet. Reporting a partial window as evidence is how a
    // controller trips on startup noise.
    if (this.samples.length < this.options.window) {
      return this.decide(false, `gathering evidence (${this.samples.length}/${this.options.window})`);
    }

    const median = this.median();

    if (median > this.options.failMs) {
      this.qualifying++;
      if (this.qualifying >= this.options.dwell) {
        const wasProbing = this.probing;
        this.enterFallback();
        if (wasProbing) {
          this.backoffMs = Math.min(this.backoffMs * 2, this.options.maxProbeBackoffMs);
          this.probeAtMs = nowMs + this.backoffMs;
        } else {
          this.probeAtMs = nowMs + this.backoffMs;
        }
        return this.decide(
          true,
          `neural stage over budget (median ${median.toFixed(2)} ms > ${this.options.failMs} ms)`,
        );
      }
      return this.decide(false, `over budget, ${this.options.dwell - this.qualifying} more to fall back`);
    }

    if (this.probing) {
      if (median <= this.options.recoverMs) {
        this.qualifying++;
        if (this.qualifying >= this.options.dwell) {
          this.probing = false;
          this.qualifying = 0;
          this.backoffMs = this.options.probeBackoffMs;
          return this.decide(
            true,
            `neural stage back within budget (median ${median.toFixed(2)} ms held under ${this.options.recoverMs} ms)`,
          );
        }
        return this.decide(false, `probe qualifying, ${this.options.dwell - this.qualifying} more to confirm`);
      }

      // Between the thresholds: not bad enough to fail outright, not good
      // enough to recover. A probe must still end, or the network runs
      // unconfirmed forever - so patience runs out and it goes back.
      this.qualifying = 0;
      this.probeMisses++;
      if (this.probeMisses >= this.options.probePatience) {
        this.enterFallback();
        this.backoffMs = Math.min(this.backoffMs * 2, this.options.maxProbeBackoffMs);
        this.probeAtMs = nowMs + this.backoffMs;
        return this.decide(
          true,
          `probe did not reach ${this.options.recoverMs} ms (median ${median.toFixed(2)} ms) - staying on the baseline`,
        );
      }
      return this.decide(false, `probing, median ${median.toFixed(2)} ms not yet under ${this.options.recoverMs} ms`);
    }

    this.qualifying = 0;
    return this.decide(false, 'within budget');
  }

  /**
   * Advances the fallback timer. Returns a decision that changes state when the
   * backoff has elapsed and it is time to re-enable the neural stage for
   * measurement. The caller must switch stages when `state` becomes `neural`.
   */
  tick(nowMs: number): BudgetDecision {
    if (this.forced || this.state !== 'fallback') {
      return this.decide(false, this.forced ? 'forced over-budget for testing' : 'within budget');
    }
    if (nowMs < this.probeAtMs) {
      const waitS = ((this.probeAtMs - nowMs) / 1000).toFixed(1);
      return this.decide(false, `using baseline, next probe in ${waitS} s`);
    }
    this.state = 'neural';
    this.probing = true;
    this.qualifying = 0;
    this.probeMisses = 0;
    this.samples.length = 0;
    return this.decide(true, 'probing whether the neural stage now fits');
  }

  /** Returns to the initial state. Used when the pipeline is reconfigured. */
  reset(): void {
    this.samples.length = 0;
    this.state = 'neural';
    this.qualifying = 0;
    this.probing = false;
    this.forced = false;
    this.backoffMs = this.options.probeBackoffMs;
    this.probeAtMs = 0;
  }

  private enterFallback(): void {
    this.state = 'fallback';
    this.probing = false;
    this.qualifying = 0;
    this.probeMisses = 0;
    this.samples.length = 0;
  }

  private decide(changed: boolean, reason: string): BudgetDecision {
    return {
      state: this.state,
      changed,
      medianMs: this.median(),
      samples: this.samples.length,
      probing: this.probing,
      reason,
    };
  }

  /**
   * True median: for an even window the two middle samples are averaged, which
   * is what `SampleWindow.quantile(0.5)` does elsewhere in the project. Taking
   * the upper middle instead let a 15-low/15-high window report the first high
   * value and cross a threshold the data did not support.
   */
  private median(): number {
    const n = this.samples.length;
    if (n === 0) return Number.NaN;
    const sorted = [...this.samples].sort((a, b) => a - b);
    const mid = n >> 1;
    return n % 2 === 1 ? (sorted[mid] as number) : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
  }
}
