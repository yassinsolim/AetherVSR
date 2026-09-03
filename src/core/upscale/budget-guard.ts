/**
 * Decides when the neural stage must give way to the baseline scaler.
 *
 * Deliberately simple. Milestone 6 owns dynamic quality selection; what is
 * needed here is a mechanism that cannot oscillate, cannot trip on one slow
 * frame, and can be forced into failure on demand so its recovery path is
 * exercised rather than assumed.
 *
 * ## Why a rolling median rather than a mean
 *
 * A single 40 ms frame - a tab regaining focus, a shader cache miss, another
 * application taking the GPU - would drag a mean over any threshold worth
 * setting. The median of a short window ignores that and still responds within
 * a few frames to a genuine change.
 *
 * ## Hysteresis
 *
 * Falling back and recovering use different thresholds. With one threshold, a
 * stage sitting exactly at the limit would alternate every window, and the
 * switching itself costs more than either choice. Recovery also requires the
 * measurement to hold for a dwell period, so a brief dip cannot pull the neural
 * stage back into a load it will immediately fail again.
 */
export interface BudgetGuardOptions {
  /** Fall back when the median stage time exceeds this, in milliseconds. */
  readonly fallbackMs: number;
  /** Return to neural only when the median is at or below this. */
  readonly recoverMs: number;
  /** Frames of history the median is taken over. */
  readonly window: number;
  /** Consecutive qualifying windows required before recovering. */
  readonly recoverDwell: number;
}

export const DEFAULT_BUDGET_GUARD: BudgetGuardOptions = {
  // 8 ms is the milestone's target for the whole stage. Falling back at 10
  // leaves room for ordinary variance without tolerating a stage that is
  // genuinely over budget.
  fallbackMs: 10,
  // Recovery needs a clear margin, not merely the absence of failure.
  recoverMs: 7,
  window: 30,
  recoverDwell: 3,
};

export type BudgetState = 'neural' | 'fallback';

export interface BudgetDecision {
  readonly state: BudgetState;
  /** True on the frame the state changed. */
  readonly changed: boolean;
  readonly medianMs: number;
  readonly samples: number;
  readonly reason: string;
}

export class BudgetGuard {
  private readonly options: BudgetGuardOptions;
  private readonly samples: number[] = [];
  private state: BudgetState = 'neural';
  private qualifyingWindows = 0;
  private forced = false;

  constructor(options: Partial<BudgetGuardOptions> = {}) {
    this.options = { ...DEFAULT_BUDGET_GUARD, ...options };
    if (this.options.recoverMs >= this.options.fallbackMs) {
      throw new Error(
        `recoverMs (${this.options.recoverMs}) must be below fallbackMs ` +
          `(${this.options.fallbackMs}) or the guard will oscillate`,
      );
    }
  }

  get current(): BudgetState {
    return this.state;
  }

  /** Forces the over-budget condition, so the fallback path can be exercised. */
  setForced(forced: boolean): void {
    this.forced = forced;
  }

  get isForced(): boolean {
    return this.forced;
  }

  reset(): void {
    this.samples.length = 0;
    this.qualifyingWindows = 0;
    this.state = 'neural';
  }

  /** Feeds one measured whole-stage time and returns the resulting decision. */
  record(stageMs: number): BudgetDecision {
    if (Number.isFinite(stageMs) && stageMs > 0) {
      this.samples.push(stageMs);
      if (this.samples.length > this.options.window) this.samples.shift();
    }

    if (this.forced) {
      const changed = this.state !== 'fallback';
      this.state = 'fallback';
      this.qualifyingWindows = 0;
      return {
        state: this.state,
        changed,
        medianMs: this.median(),
        samples: this.samples.length,
        reason: 'forced over-budget for testing',
      };
    }

    // Not enough history to judge: keep whatever is running rather than
    // switching on one or two frames.
    if (this.samples.length < this.options.window) {
      return {
        state: this.state,
        changed: false,
        medianMs: this.median(),
        samples: this.samples.length,
        reason: 'warming up',
      };
    }

    const median = this.median();
    if (this.state === 'neural') {
      if (median > this.options.fallbackMs) {
        this.state = 'fallback';
        this.qualifyingWindows = 0;
        return {
          state: this.state,
          changed: true,
          medianMs: median,
          samples: this.samples.length,
          reason: `median ${median.toFixed(2)} ms over ${this.options.fallbackMs} ms`,
        };
      }
      return {
        state: this.state,
        changed: false,
        medianMs: median,
        samples: this.samples.length,
        reason: 'within budget',
      };
    }

    // In fallback: the neural stage is no longer running, so the samples being
    // measured are the baseline's. They are still the right signal - the
    // question is whether this machine currently has the headroom - but
    // recovery must clear a lower bar and hold it.
    if (median <= this.options.recoverMs) {
      this.qualifyingWindows++;
      if (this.qualifyingWindows >= this.options.recoverDwell) {
        this.state = 'neural';
        this.qualifyingWindows = 0;
        return {
          state: this.state,
          changed: true,
          medianMs: median,
          samples: this.samples.length,
          reason: `median ${median.toFixed(2)} ms held under ${this.options.recoverMs} ms`,
        };
      }
    } else {
      this.qualifyingWindows = 0;
    }
    return {
      state: this.state,
      changed: false,
      medianMs: median,
      samples: this.samples.length,
      reason: `awaiting ${this.options.recoverDwell - this.qualifyingWindows} more qualifying windows`,
    };
  }

  private median(): number {
    if (this.samples.length === 0) return Number.NaN;
    const sorted = [...this.samples].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)] as number;
  }
}
