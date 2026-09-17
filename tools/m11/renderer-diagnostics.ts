import { VideoPipeline, type PipelineOptions, type PipelineGpuSample } from '../../src/core/pipeline.js';
import type { GpuContext } from '../../src/core/gpu/device.js';
import type { FrameTick, Upscaler } from '../../src/core/types.js';
import type { RuntimeDriver } from '../../src/runtime.js';
import type { DesktopSession, DesktopSessionOptions } from '../../apps/desktop/session.js';
import { IdentityProbe } from '../m1010r/probe.js';

interface Row {
  sequence: number; probe: number; index: number; neural: boolean; generation: number;
  before: number; submittedAt: number | null; opportunityAt: number | null;
  mediaTime: number | null; presentationTime: number | null; expectedDisplayTime: number | null;
  currentTime: number | null; sourceIdentity: number | null; identityValid: boolean;
  superseded: boolean; canvasVisible: boolean; visibility: string;
}

export class DesktopDiagnostics {
  private session: (() => DesktopSession) | null = null;
  private enabled = false;
  private readonly probes: IdentityProbe[] = [];
  private pending: Row | null = null;
  private sequence = 0;
  private latest = 0;
  private epoch = 0;
  private readonly animationFrames = new Set<number>();
  private observedDriver: RuntimeDriver | null = null;
  private rows: Row[] = [];
  private samples: PipelineGpuSample[] = [];
  private states: { at: number; actualTier: string; state: string; reason: string; generation: number }[] = [];
  private errors: string[] = [];
  private startAt: number | null = null;
  private stopAt: number | null = null;
  private stateTimer: ReturnType<typeof setInterval> | null = null;
  private gpu: GpuContext | null = null;
  private activeProbe: IdentityProbe | null = null;
  private scopes = 0;

  constructor(private readonly video: HTMLVideoElement, private readonly canvas: HTMLCanvasElement) {}

  bind(session: () => DesktopSession): void { this.session = session; }

  options(observe = false): DesktopSessionOptions {
    this.enabled = observe;
    return { createPipeline: (gpu, video, canvas, upscaler, options) => this.pipeline(gpu, video, canvas, upscaler, options),
      onFrame: (tick, driver) => this.frame(tick, driver) };
  }

  private pipeline(gpu: GpuContext, video: HTMLVideoElement, canvas: HTMLCanvasElement, initial: Upscaler, options: PipelineOptions): VideoPipeline {
    this.epoch++;
    if (this.stateTimer !== null) clearInterval(this.stateTimer);
    this.stateTimer = null;
    if (this.observedDriver) this.observedDriver.onSample = null;
    this.observedDriver = null;
    for (const handle of this.animationFrames) cancelAnimationFrame(handle);
    this.animationFrames.clear();
    while (this.scopes) { this.scopes--; void this.gpu?.device.popErrorScope().catch(() => {}); }
    this.sequence = 0; this.latest = 0; this.pending = null;
    this.probes.length = 0; this.activeProbe = null;
    this.rows = []; this.samples = []; this.states = []; this.errors = [];
    this.startAt = null; this.stopAt = null;
    this.gpu = gpu;
    if (!this.enabled) return new VideoPipeline(gpu, video, canvas, initial, options);
    const wrap = (inner: Upscaler) => {
      const ordinal = this.probes.length;
      const probe = new IdentityProbe(inner, index => {
        this.pending = { sequence: ++this.sequence, probe: ordinal, index, neural: inner.neural, generation: 0,
          before: performance.now(), submittedAt: null, opportunityAt: null, mediaTime: null, presentationTime: null,
          expectedDisplayTime: null, currentTime: null, sourceIdentity: null, identityValid: false,
          superseded: false, canvasVisible: false, visibility: document.visibilityState };
      });
      this.probes.push(probe); this.activeProbe = probe;
      return probe;
    };
    class ObservedPipeline extends VideoPipeline {
      override setUpscaler(next: Upscaler): void { super.setUpscaler(wrap(next)); }
    }
    return new ObservedPipeline(gpu, video, canvas, wrap(initial), options);
  }

  private frame(tick: FrameTick, driver: RuntimeDriver): void {
    if (!this.enabled || !this.pending) return;
    const row = this.pending; this.pending = null;
    row.submittedAt = performance.now(); row.mediaTime = tick.mediaTime;
    row.presentationTime = tick.presentationTime; row.expectedDisplayTime = tick.expectedDisplayTime;
    row.currentTime = this.video.currentTime; row.generation = driver.pipeline.timingGeneration;
    this.latest = row.sequence;
    this.rows.push(row);
    if (this.rows.length > 65536) throw new Error('Desktop diagnostic frame capacity exhausted');
    const epoch = this.epoch;
    const handle = requestAnimationFrame(() => {
      this.animationFrames.delete(handle);
      if (epoch !== this.epoch) return;
      row.opportunityAt = performance.now();
      row.superseded = this.latest !== row.sequence || driver.pipeline.timingGeneration !== row.generation;
      row.canvasVisible = !this.canvas.hidden;
    });
    this.animationFrames.add(handle);
  }

  begin(): void {
    const driver = this.session?.().runtime;
    if (!driver || !this.enabled || this.startAt !== null || !this.gpu) throw new Error('No idle instrumented runtime');
    this.startAt = performance.now(); this.stopAt = null;
    this.samples.length = 0; this.states.length = 0; this.errors.length = 0;
    for (const filter of ['internal', 'out-of-memory', 'validation'] as const) { this.gpu.device.pushErrorScope(filter); this.scopes++; }
    const epoch = this.epoch;
    this.observedDriver = driver;
    driver.onSample = sample => { if (epoch === this.epoch) this.samples.push(sample); };
    const observe = () => {
      if (epoch !== this.epoch || this.session?.().runtime !== driver) {
        this.errors.push('Observed runtime was released during measurement');
        if (this.stateTimer !== null) clearInterval(this.stateTimer);
        this.stateTimer = null;
        return;
      }
      const state = driver.snapshot();
      this.states.push({ at: performance.now(), actualTier: state.actualTier, state: state.controller.state,
        reason: state.controller.reason, generation: driver.pipeline.timingGeneration });
    };
    observe(); this.stateTimer = setInterval(observe, 100);
  }

  async finish() {
    const session = this.session?.(), driver = session?.runtime;
    this.stopAt = performance.now();
    if (this.stateTimer !== null) clearInterval(this.stateTimer); this.stateTimer = null;
    const startAt = this.startAt, stopAt = this.stopAt, epoch = this.epoch;
    const rows = this.rows, samples = this.samples, states = this.states, errors = this.errors;
    const gpu = this.gpu, probe = this.activeProbe, observedDriver = this.observedDriver;
    let scopes = this.scopes; this.scopes = 0;
    try {
      if (!driver || !gpu || !probe || startAt === null) throw new Error('Diagnostic runtime unavailable at finish; retaining partial evidence');
      session.pause();
      await driver.pipeline.drainTimings(); await gpu.device.queue.onSubmittedWorkDone();
      const identities = await probe.readAfterPause();
      if (epoch !== this.epoch) throw new Error('Source changed during diagnostic drain');
      const ordinal = this.probes.indexOf(probe);
      for (const row of rows) if (row.probe === ordinal) {
        row.sourceIdentity = identities[4 + row.index * 4] ?? null;
        row.identityValid = identities[5 + row.index * 4] === 1;
      }
    } catch (error) { errors.push(String(error)); }
    finally {
      while (scopes) {
        scopes--;
        try { const error = await gpu?.device.popErrorScope(); if (error) errors.push(error.message); }
        catch (error) { errors.push(String(error)); }
      }
      if (observedDriver) observedDriver.onSample = null;
      if (epoch === this.epoch) {
        this.observedDriver = null;
        for (const handle of this.animationFrames) cancelAnimationFrame(handle);
        this.animationFrames.clear();
      }
    }
    return { startAt, stopAt, durationMs: startAt === null ? null : stopAt - startAt,
      rows: rows.filter(row => row.submittedAt !== null && (startAt === null || row.submittedAt >= startAt)).map(row => ({ ...row })),
      samples: [...samples], states: [...states], errors: [...errors], snapshot: session?.snapshot() ?? null,
      scope: 'Same imported texture identity and next animation-frame opportunity, not physical scanout or speaker synchronization' };
  }
}