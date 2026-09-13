export type RuntimeMode = 'auto' | 'neural' | 'baseline';
export type RuntimeTier = 'neural' | 'baseline';
export type RuntimeState = 'warmup' | 'stable' | 'fallback' | 'probing' | 'manual-baseline' | 'suspended' | 'unavailable' | 'failed';

export interface RuntimeSample {
  ms: number;
  generation: number;
  sequence: number;
  submittedAt: number;
  resolvedAt: number;
  neural: boolean;
}

export interface RuntimeTransition {
  atMs: number;
  activeMs: number;
  from: RuntimeState;
  to: RuntimeState;
  tier: RuntimeTier;
  reason: string;
}

export interface RuntimeSnapshot {
  mode: RuntimeMode;
  tier: RuntimeTier;
  state: RuntimeState;
  reason: string;
  failMs: number;
  recoverMs: number;
  medianMs: number | null;
  p90Ms: number | null;
  samples: number;
  generation: number | null;
  fps: number;
  width: number;
  height: number;
  activeMs: number;
  backoffMs: number;
  nextProbeAtMs: number | null;
  transitionCount: number;
  transitions: RuntimeTransition[];
  discardedTransitions: number;
  probeCount: number;
  failedProbeCount: number;
  fallbackMs: number;
  forced: boolean;
}

type Phase =
  | { state: 'warmup'; deadline: number }
  | { state: 'probing'; startedAt: number; deadline: number }
  | { state: 'stable'; lastEvidenceAt: number }
  | { state: 'fallback'; nextProbeAt: number }
  | { state: 'manual-baseline' | 'unavailable' | 'failed' };

function quantile(values: readonly number[], fraction: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lower = sorted[Math.floor(position)]!;
  return lower + (sorted[Math.ceil(position)]! - lower) * (position % 1);
}

export class RuntimeController {
  private mode: RuntimeMode;
  private available: boolean;
  private activity: { active: true } | { active: false; tier: RuntimeTier };
  private forced = false;
  private phase: Phase = { state: 'warmup', deadline: 2000 };
  private reason = 'gathering neural evidence';
  private now: number;
  private activeMs = 0;
  private fallbackMs = 0;
  private width = 0;
  private height = 0;
  private fps = 60;
  private generation: number | null = null;
  private lastSequence = -1;
  private lastSubmittedAt = -1;
  private lastResolvedAt = -1;
  private cutoff: number;
  private coldAt = 0;
  private coldCount = 0;
  private raw: number[] = [];
  private evidence: number[] = [];
  private failing = 0;
  private recovering = 0;
  private confirming = 0;
  private backoffMs = 2000;
  private probeCount = 0;
  private failedProbeCount = 0;
  private transitionCount = 0;
  private transitions: RuntimeTransition[] = [];

  constructor(options: { mode?: RuntimeMode; available?: boolean; active?: boolean } = {}, now = 0) {
    if (!Number.isFinite(now) || now < 0) throw new RangeError('time must be finite and nonnegative');
    this.now = now;
    this.cutoff = now;
    this.mode = options.mode ?? 'auto';
    this.available = options.available ?? true;
    this.activity = (options.active ?? true) ? { active: true } : { active: false, tier: 'neural' };
    this.reconcile();
    if (!this.activity.active) this.activity.tier = this.tier();
  }

  setMode(mode: RuntimeMode, now: number): RuntimeSnapshot {
    return this.update(now, () => {
      if (mode === this.mode) return;
      this.mode = mode;
      this.flush();
      this.rewarm();
      this.reconcile();
    });
  }

  setAvailable(available: boolean, now: number): RuntimeSnapshot {
    return this.update(now, () => {
      if (available === this.available) return;
      this.available = available;
      this.flush();
      this.reconcile();
    });
  }

  setActive(active: boolean, now: number): RuntimeSnapshot {
    return this.update(now, () => {
      if (active === this.activity.active) return;
      this.activity = active ? { active: true } : { active: false, tier: this.tier() };
      if (active) {
        this.flush();
        this.rewarm();
      }
      this.reason = active ? 'resumed with fresh evidence' : 'active clock suspended';
    });
  }

  setWorkload(width: number, height: number, fps: number, now: number): RuntimeSnapshot {
    if (![width, height].every(value => Number.isSafeInteger(value) && value >= 0)) {
      throw new RangeError('dimensions must be nonnegative safe integers');
    }
    const cadence = Number.isFinite(fps) && fps > 0 ? Math.min(240, Math.max(1, fps)) : 60;
    return this.update(now, () => {
      if (width === this.width && height === this.height && cadence === this.fps) return;
      this.width = width;
      this.height = height;
      this.fps = cadence;
      this.backoffMs = 2000;
      this.flush();
      this.phase = { state: 'warmup', deadline: this.activeMs + 2000 };
      this.reason = 'workload changed; gathering neural evidence';
      this.reconcile();
    });
  }

  invalidate(now: number): RuntimeSnapshot {
    return this.update(now, () => {
      this.flush();
      this.rewarm();
    });
  }

  bindGeneration(generation: number, now: number): RuntimeSnapshot {
    if (!Number.isSafeInteger(generation) || generation < 0) throw new RangeError('invalid generation');
    return this.update(now, () => this.flush(generation));
  }

  setForced(on: boolean, now: number): RuntimeSnapshot {
    return this.update(now, () => {
      if (on === this.forced) return;
      this.forced = on;
      if (on) this.flush();
      this.reconcile();
      if (!on && this.phase.state === 'fallback') this.reason = 'force released; waiting for neural probe';
    });
  }

  fail(reason: string, now: number): RuntimeSnapshot {
    return this.update(now, () => {
      this.flush();
      this.phase = { state: 'failed' };
      this.reason = reason;
    });
  }

  record(sample: RuntimeSample, now = sample.resolvedAt): RuntimeSnapshot {
    return this.update(now, () => {
      if (!this.activity.active || this.tier() !== 'neural' || !sample.neural || this.generation === null ||
          sample.generation !== this.generation || !Number.isFinite(sample.ms) || sample.ms <= 0 ||
          !Number.isSafeInteger(sample.sequence) || sample.sequence <= this.lastSequence ||
          !Number.isFinite(sample.submittedAt) || !Number.isFinite(sample.resolvedAt) ||
          sample.submittedAt < this.cutoff || sample.submittedAt < this.lastSubmittedAt ||
          sample.resolvedAt < this.lastResolvedAt || sample.resolvedAt < sample.submittedAt ||
          sample.resolvedAt > now || now - sample.submittedAt > 500) return;
      this.checkDeadline();
      if (this.tier() !== 'neural') return;
      this.lastSequence = sample.sequence;
      this.lastSubmittedAt = sample.submittedAt;
      this.lastResolvedAt = sample.resolvedAt;
      this.raw.push(sample.ms);
      if (this.raw.length > 30) this.raw.shift();
      this.coldCount++;
      const submittedActive = this.activeMs - (now - sample.submittedAt);
      if (this.coldCount <= 3 || submittedActive - this.coldAt < 150) return;
      this.evidence.push(sample.ms);
      if (this.evidence.length > 30) this.evidence.shift();
      if (this.phase.state === 'stable') this.phase.lastEvidenceAt = submittedActive;
      if (this.evidence.length < 30) return;
      const median = quantile(this.evidence, 0.5)!;
      const failing = median > this.failMs();
      const recovering = median <= this.recoverMs();
      this.failing = failing ? this.failing + 1 : 0;
      this.recovering = recovering ? this.recovering + 1 : 0;
      this.confirming = !failing ? this.confirming + 1 : 0;
      if (this.failing >= 3) this.fallback('neural median exceeds failure threshold');
      else if ((this.phase.state === 'probing' && this.recovering >= 3) ||
               (this.phase.state === 'warmup' && this.confirming >= 3)) {
        this.phase = { state: 'stable', lastEvidenceAt: submittedActive };
        this.backoffMs = 2000;
        this.reason = 'neural evidence confirmed';
      }
    });
  }

  tick(now: number): RuntimeSnapshot {
    return this.update(now, () => {
      if (!this.activity.active) return;
      if (this.phase.state === 'fallback' && !this.forced && this.available && this.mode !== 'baseline' &&
          this.activeMs >= this.phase.nextProbeAt) {
        this.phase = { state: 'probing', startedAt: this.activeMs, deadline: this.activeMs + 2000 };
        this.probeCount++;
        this.flush();
        this.reason = 'probing neural recovery';
      } else this.checkDeadline();
    });
  }

  snapshot(now: number): RuntimeSnapshot {
    this.advance(now);
    return this.view();
  }

  private failMs(): number {
    return Math.min(24, Math.max(6, (0.90 * 1000) / this.fps));
  }

  private recoverMs(): number {
    return Math.min(22, Math.max(5, (0.78 * 1000) / this.fps));
  }

  private state(): RuntimeState {
    return this.phase.state === 'failed' || this.activity.active ? this.phase.state : 'suspended';
  }

  private tier(): RuntimeTier {
    if (!['warmup', 'stable', 'probing'].includes(this.phase.state)) return 'baseline';
    return this.activity.active ? 'neural' : this.activity.tier;
  }

  private advance(now: number): void {
    if (!Number.isFinite(now) || now < this.now) throw new RangeError('time must be finite and monotonic');
    if (this.activity.active && this.phase.state !== 'failed') {
      this.activeMs += now - this.now;
      if (this.phase.state === 'fallback') this.fallbackMs += now - this.now;
    }
    this.now = now;
  }

  private update(now: number, action: () => void): RuntimeSnapshot {
    this.advance(now);
    if (this.phase.state === 'failed') return this.view();
    const from = this.state();
    const tier = this.tier();
    action();
    if (!this.activity.active) this.activity.tier = this.tier();
    if (from !== this.state() || tier !== this.tier()) {
      this.transitionCount++;
      this.transitions.push({ atMs: now, activeMs: this.activeMs, from, to: this.state(), tier: this.tier(), reason: this.reason });
      if (this.transitions.length > 128) this.transitions.shift();
    }
    return this.view();
  }

  private flush(generation: number | null = null): void {
    this.generation = generation;
    this.cutoff = this.now;
    this.coldAt = this.activeMs;
    this.coldCount = 0;
    this.raw = [];
    this.evidence = [];
    this.failing = 0;
    this.recovering = 0;
    this.confirming = 0;
    if (this.phase.state === 'stable') this.phase.lastEvidenceAt = this.activeMs;
  }

  private rewarm(): void {
    if (this.phase.state === 'stable') {
      this.phase = { state: 'warmup', deadline: this.activeMs + 2000 };
      this.reason = 'rewarming selected neural stage';
    }
  }

  private reconcile(): void {
    if (this.mode === 'baseline') {
      this.phase = { state: 'manual-baseline' };
      this.reason = 'manual baseline intent';
    } else if (!this.available) {
      this.phase = { state: 'unavailable' };
      this.reason = 'usable GPU timestamps unavailable';
    } else if (this.forced) {
      if (this.phase.state !== 'fallback') this.fallback('forced fallback');
    } else if (this.phase.state === 'manual-baseline' || this.phase.state === 'unavailable') {
      this.phase = { state: 'warmup', deadline: this.activeMs + 2000 };
      this.reason = 'gathering neural evidence';
    }
  }

  private checkDeadline(): void {
    if (!this.activity.active) return;
    if ((this.phase.state === 'warmup' || this.phase.state === 'probing') && this.activeMs >= this.phase.deadline) {
      this.fallback('neural qualification deadline expired');
    } else if (this.phase.state === 'stable' && this.activeMs - this.phase.lastEvidenceAt >= 2000) {
      this.fallback('stable neural evidence expired');
    }
  }

  private fallback(reason: string): void {
    let nextProbeAt = this.activeMs + this.backoffMs;
    if (this.phase.state === 'probing') {
      this.failedProbeCount++;
      this.backoffMs = Math.min(30000, this.backoffMs * 2);
      nextProbeAt = Math.max(this.activeMs, this.phase.startedAt + this.backoffMs);
    }
    this.phase = { state: 'fallback', nextProbeAt };
    this.flush();
    this.reason = reason;
  }

  private view(): RuntimeSnapshot {
    return {
      mode: this.mode, tier: this.tier(), state: this.state(), reason: this.reason,
      failMs: this.failMs(), recoverMs: this.recoverMs(), medianMs: quantile(this.raw, 0.5),
      p90Ms: quantile(this.raw, 0.9), samples: this.raw.length, generation: this.generation,
      fps: this.fps, width: this.width, height: this.height, activeMs: this.activeMs,
      backoffMs: this.backoffMs,
      nextProbeAtMs: this.phase.state === 'fallback' ? this.now + Math.max(0, this.phase.nextProbeAt - this.activeMs) : null,
      transitionCount: this.transitionCount, transitions: this.transitions.map(transition => ({ ...transition })),
      discardedTransitions: this.transitionCount - this.transitions.length,
      probeCount: this.probeCount, failedProbeCount: this.failedProbeCount, fallbackMs: this.fallbackMs, forced: this.forced,
    };
  }
}