import { acquireGpu, watchDeviceFailures, type GpuContext } from '../core/gpu/device.js';
import type { PackedModel } from '../core/neural/model.js';
import { VideoPipeline } from '../core/pipeline.js';
import { BaselineScaler } from '../core/upscale/baseline-scaler.js';
import { NeuralUpscaler, NEURAL_OPTIONAL_FEATURES, type NeuralMemoryReport } from '../core/upscale/neural-upscaler.js';
import type { RuntimeMode, RuntimeState, RuntimeTier } from '../core/upscale/runtime-controller.js';
import { RuntimeDriver } from '../runtime.js';
import { inspectGeometry, type Rect } from './geometry.js';
import type { StatusCode } from './protocol.js';

export interface AttachmentOptions {
  mode: RuntimeMode;
  model: PackedModel;
  onFailure: (code: StatusCode, message: string) => void;
  forceCopy?: boolean;
  withheldFeatures?: GPUFeatureName[];
}

export interface AttachmentSnapshot {
  active: boolean;
  ready: boolean;
  suspendedReason: string | null;
  source: { w: number; h: number };
  backing: { w: number; h: number };
  cssRect: Rect | null;
  mode: RuntimeMode;
  current: RuntimeTier | null;
  controllerState: RuntimeState | null;
  controllerReason: string | null;
  gpuMs: { p50: number | null; p95: number | null; count: number };
  session: {
    framesRendered: number; framesPresented: number; framesSkipped: number;
    decoderDrops: number | null; meanRenderedFps: number | null; meanPresentedFps: number | null;
    activeMs: number; callbackP95: number | null;
  };
  infrastructure: {
    geometryCalls: number; geometryTotalMs: number; geometryMaxMs: number | null;
    refreshCalls: number; cleanupErrors: number;
  };
  resources: {
    device: 0 | 1; pipeline: 0 | 1; canvas: 0 | 1; resizeObservers: 0 | 1;
    listeners: number; geometryFrame: 0 | 1; frameCallback: 0 | 1;
  };
  memory: NeuralMemoryReport | null;
  features: { timestampQuery: boolean; shaderF16: boolean };
  precision: 'f16' | 'fp32' | null;
  scope: { gpuMs: string; session: string; callbackP95: string; geometryMs: string; resources: string; memory: string };
}

export class VideoAttachment {
  readonly canvas: HTMLCanvasElement;
  private context: GpuContext | null = null;
  private runtime: RuntimeDriver | null = null;
  private videoPipeline: VideoPipeline | null = null;
  private stage: NeuralUpscaler | null = null;
  private model: PackedModel | null;
  private failure: AttachmentOptions['onFailure'] | null;
  private readonly forceCopy: boolean;
  private readonly withheldFeatures: GPUFeatureName[];
  private mode: RuntimeMode;
  private disposed = false;
  private generation = 0;
  private starting: Promise<void> | null = null;
  private eligible = false;
  private outputReady = false;
  private suspendedReason: string | null = 'starting';
  private cssRect: Rect | null = null;
  private observer: ResizeObserver | null = null;
  private observedParent: Element | null = null;
  private geometryFrame: number | null = null;
  private unwatch: (() => void) | null = null;
  private readonly listeners: [EventTarget, string, EventListener, boolean][] = [];
  private styleKeys = new Set<string>();
  private readonly infrastructure: AttachmentSnapshot['infrastructure'] = {
    geometryCalls: 0, geometryTotalMs: 0, geometryMaxMs: null, refreshCalls: 0, cleanupErrors: 0,
  };
  private finalSnapshot: AttachmentSnapshot | null = null;

  constructor(readonly video: HTMLVideoElement, options: AttachmentOptions) {
    this.mode = options.mode;
    this.model = options.model;
    this.failure = options.onFailure;
    this.forceCopy = options.forceCopy ?? false;
    this.withheldFeatures = [...(options.withheldFeatures ?? [])];
    this.canvas = video.ownerDocument.createElement('canvas');
    this.canvas.width = 0;
    this.canvas.height = 0;
    this.canvas.setAttribute('aria-hidden', 'true');
    this.canvas.dataset.aethervsrM10 = crypto.randomUUID();
    for (const [name, value] of Object.entries({ all: 'initial', 'pointer-events': 'none',
      animation: 'none', transition: 'none', visibility: 'hidden' })) this.style(name, value);
  }

  get driver(): RuntimeDriver | null { return this.runtime; }
  get pipeline(): VideoPipeline | null { return this.videoPipeline; }
  get gpu(): GpuContext | null { return this.context; }

  async start(): Promise<void> {
    if (this.disposed) return;
    this.starting ??= this.initialize(++this.generation);
    await this.starting;
  }

  private async initialize(generation: number): Promise<void> {
    try {
      this.observe();
      this.checkGeometry();
      if (this.disposed) return;
      let gpu: GpuContext;
      try {
        gpu = await acquireGpu({ optionalFeatures: NEURAL_OPTIONAL_FEATURES,
          withheldFeatures: this.withheldFeatures });
      } catch (error) {
        if (!this.disposed && generation === this.generation) this.fail('webgpu-unavailable', this.message(error));
        return;
      }
      if (this.disposed || generation !== this.generation) {
        gpu.device.destroy();
        return;
      }
      this.context = gpu;
      if (this.failSource()) return;
      const unwatch = watchDeviceFailures(gpu.device, message => {
        if (this.video.mediaKeys || this.video.error || this.videoPipeline?.error) this.failExecution(this.videoPipeline?.error ?? message);
        else this.fail('device-lost', message);
      });
      if (this.disposed) { unwatch(); return; }
      this.unwatch = unwatch;
      this.videoPipeline = new VideoPipeline(gpu, this.video, this.canvas, new BaselineScaler('catmull-rom'),
        { forceCopyImport: this.forceCopy });
      const driver = new RuntimeDriver(this.videoPipeline, this.video, gpu.capabilities.timestampQuery,
        this.mode, () => this.visible());
      this.runtime = driver;
      driver.onChange = state => {
        if (this.disposed || this.failSource()) return;
        if (state.state === 'failed') this.failExecution(this.videoPipeline?.error ?? state.reason);
      };
      driver.onFrame = () => {
        if (this.disposed || this.failSource()) return;
        if (this.videoPipeline?.error) { this.failExecution(this.videoPipeline.error); return; }
        if (!this.visible() || !this.videoPipeline?.running || this.outputReady) return;
        this.outputReady = true;
        this.style('visibility', 'visible');
      };
      driver.setNeuralFactory(() => {
        this.stage = new NeuralUpscaler(this.model!, { passDiagnostics: true });
        return this.stage;
      });
      this.checkGeometry();
    } catch (error) {
      if (!this.disposed) this.failExecution(error);
      else throw error;
    }
  }

  setMode(mode: RuntimeMode): void {
    if (this.disposed || this.failSource()) return;
    this.mode = mode;
    try { this.runtime?.setMode(mode); } catch (error) { this.failExecution(error); }
  }

  refresh(): void {
    if (this.disposed) return;
    this.infrastructure.refreshCalls++;
    if (this.failSource()) return;
    const reason = this.unavailableReason();
    if (reason !== null) this.suspend(reason);
    if (this.geometryFrame !== null || this.disposed) return;
    const view = this.video.ownerDocument.defaultView;
    if (!view) { this.fail('unsupported-page', 'The video document has no window.'); return; }
    this.geometryFrame = view.requestAnimationFrame(() => {
      this.geometryFrame = null;
      if (this.disposed) return;
      try { this.checkGeometry(); } catch (error) { this.failExecution(error); }
    });
  }

  private unavailableReason(): string | null {
    const document = this.video.ownerDocument;
    if (!this.video.isConnected || this.video.readyState < 2 || this.video.videoWidth <= 0 || this.video.videoHeight <= 0) {
      return 'video-not-ready';
    }
    if (this.video.seeking) return 'video-seeking';
    if (this.video.paused) return 'video-paused';
    if (document.visibilityState !== 'visible') return 'document-hidden';
    if (document.pictureInPictureElement === this.video) return 'picture-in-picture';
    const fullscreen = document.fullscreenElement;
    if (fullscreen === this.video) return 'video-fullscreen';
    if (fullscreen && !fullscreen.contains(this.video)) return 'outside-fullscreen';
    return null;
  }

  private visible(): boolean {
    return !this.disposed && this.eligible && !this.video.mediaKeys && !this.video.error && this.unavailableReason() === null;
  }

  private style(name: string, value: string): void {
    if (this.canvas.style.getPropertyValue(name) !== value || this.canvas.style.getPropertyPriority(name) !== 'important') {
      this.canvas.style.setProperty(name, value, 'important');
    }
  }

  private hide(): void {
    this.outputReady = false;
    this.style('visibility', 'hidden');
  }

  private suspend(reason: string): void {
    this.eligible = false;
    this.suspendedReason = reason;
    this.hide();
    this.runtime?.syncActive();
  }

  private checkGeometry(): void {
    if (this.disposed) return;
    if (this.failSource()) return;
    this.observeParent();
    const reason = this.unavailableReason();
    if (reason !== null) { this.suspend(reason); return; }
    const started = performance.now();
    let geometry: ReturnType<typeof inspectGeometry>;
    try { geometry = inspectGeometry(this.video); }
    finally {
      const duration = performance.now() - started;
      this.infrastructure.geometryCalls++;
      this.infrastructure.geometryTotalMs += duration;
      this.infrastructure.geometryMaxMs = Math.max(this.infrastructure.geometryMaxMs ?? 0, duration);
    }
    if (!geometry.ok) {
      if (geometry.code === 'offscreen' || geometry.code === 'video-not-ready') this.suspend(geometry.code);
      else this.fail(geometry.code, geometry.reason);
      return;
    }
    this.cssRect = { ...geometry.rect };
    for (const key of this.styleKeys) {
      if (!(key in geometry.style)) this.canvas.style.removeProperty(key);
    }
    for (const [key, value] of Object.entries(geometry.style)) this.style(key, value);
    this.styleKeys = new Set(Object.keys(geometry.style));
    this.style('pointer-events', 'none');
    this.style('animation', 'none');
    this.style('transition', 'none');
    this.style('visibility', this.outputReady ? 'visible' : 'hidden');
    if (this.videoPipeline && (this.canvas.parentNode !== geometry.placement.parent || this.video.nextSibling !== this.canvas)) {
      geometry.placement.parent.insertBefore(this.canvas, geometry.placement.before);
    }
    this.eligible = true;
    this.suspendedReason = null;
    this.runtime?.syncActive();
  }

  private listen(target: EventTarget, type: string, callback: () => void, capture = false): void {
    const listener = () => { if (!this.disposed) callback(); };
    target.addEventListener(type, listener, capture);
    this.listeners.push([target, type, listener, capture]);
  }

  private observe(): void {
    const document = this.video.ownerDocument;
    const view = document.defaultView;
    if (!view) throw new Error('The video document has no window.');
    this.observer = new ResizeObserver(() => this.refresh());
    this.observer.observe(this.video);
    this.observeParent();
    this.listen(view, 'scroll', () => this.refresh(), true);
    this.listen(view, 'resize', () => this.refresh());
    for (const type of ['fullscreenchange', 'visibilitychange']) this.listen(document, type, () => this.refresh());
    for (const type of ['resize', 'loadeddata', 'loadstart', 'emptied']) {
      this.listen(this.video, type, () => { this.suspend('video-not-ready'); this.refresh(); });
    }
    for (const type of ['pause', 'seeking', 'playing', 'seeked']) this.listen(this.video, type, () => this.refresh());
    const tracks = this.video.textTracks;
    if (typeof tracks?.addEventListener === 'function') {
      for (const type of ['change', 'addtrack', 'removetrack']) this.listen(tracks, type, () => this.refresh());
    }
    for (const type of ['enterpictureinpicture', 'leavepictureinpicture']) this.listen(this.video, type, () => this.refresh());
    this.listen(this.video, 'error', () => { this.failSource(); });
    this.listen(this.video, 'encrypted', () => this.fail('protected-media', 'Encrypted media cannot be enhanced.'));
  }

  private observeParent(): void {
    const node = this.video.parentNode;
    const parent = this.video.parentElement ?? (node?.nodeType === 11 && 'host' in node ? (node as ShadowRoot).host : null);
    if (!this.observer || this.observedParent === parent) return;
    if (this.observedParent) this.observer.unobserve(this.observedParent);
    this.observedParent = parent;
    if (parent) this.observer.observe(parent);
  }

  private message(error: unknown): string {
    return (typeof error === 'object' && error !== null && 'message' in error && typeof error.message === 'string'
      ? error.message : String(error)).slice(0, 2048);
  }

  private failSource(): boolean {
    if (this.video.mediaKeys) {
      this.fail('protected-media', 'Encrypted media cannot be enhanced.');
      return true;
    }
    const error = this.video.error;
    if (!error) return false;
    const reasons: Record<number, [string, string]> = {
      1: ['MEDIA_ERR_ABORTED', 'Video loading was aborted.'],
      2: ['MEDIA_ERR_NETWORK', 'A network error interrupted video loading.'],
      3: ['MEDIA_ERR_DECODE', 'The video could not be decoded.'],
      4: ['MEDIA_ERR_SRC_NOT_SUPPORTED', 'The video source is not supported.'],
    };
    const [name, fallback] = reasons[error.code] ?? [`Media error (code ${error.code})`, 'The video failed.'];
    const detail = typeof error.message === 'string' && error.message.trim() ? error.message : fallback;
    this.fail('unsupported-media', `${name}: ${detail}`);
    return true;
  }

  private failExecution(error: unknown): void {
    if (this.failSource()) return;
    const actual = this.videoPipeline?.error ?? error;
    const security = typeof actual === 'object' && actual !== null && 'name' in actual && actual.name === 'SecurityError';
    this.fail(security ? 'cors-blocked' : 'error', this.message(actual));
  }

  private fail(code: StatusCode, message: string): void {
    if (this.disposed) return;
    const callback = this.failure;
    this.suspendedReason = code;
    this.destroy();
    callback?.(code, message.slice(0, 2048));
  }

  snapshot(): AttachmentSnapshot {
    if (this.finalSnapshot) return structuredClone(this.finalSnapshot);
    const state = this.runtime?.snapshot();
    const session = state?.session;
    const current = this.videoPipeline ? (this.videoPipeline.currentUpscaler.neural ? 'neural' : 'baseline') : null;
    const gpu = current === null ? null : session?.gpu[current];
    const stage = this.videoPipeline?.currentUpscaler === this.stage ? this.stage : null;
    return {
      active: !this.disposed && (state?.running ?? false), ready: !this.disposed && this.outputReady,
      suspendedReason: this.suspendedReason,
      source: { w: this.video.videoWidth, h: this.video.videoHeight },
      backing: { w: this.canvas.width, h: this.canvas.height },
      cssRect: this.cssRect ? { ...this.cssRect } : null, mode: this.mode, current,
      controllerState: state?.controller.state ?? null, controllerReason: state?.controller.reason ?? null,
      gpuMs: { p50: gpu?.p50 ?? null, p95: gpu?.p95 ?? null, count: gpu?.count ?? 0 },
      session: {
        framesRendered: session?.framesRendered ?? 0, framesPresented: session?.framesPresented ?? 0,
        framesSkipped: session?.framesSkipped ?? 0, decoderDrops: session?.decoderDrops ?? null,
        meanRenderedFps: session?.meanRenderedFps ?? null, meanPresentedFps: session?.meanPresentedFps ?? null,
        activeMs: session?.activeMs ?? 0, callbackP95: session?.callbackLatency.p95 ?? null,
      },
      infrastructure: { ...this.infrastructure }, resources: this.resources(),
      memory: stage?.memoryReport ? { ...stage.memoryReport } : null,
      features: { timestampQuery: this.context?.capabilities.timestampQuery ?? false,
        shaderF16: this.context?.device.features.has('shader-f16') ?? false },
      precision: stage?.memoryReport ? stage.resolvedPrecision : null,
      scope: {
        gpuMs: 'Whole upscale-stage GPU timestamps received over this session for the current tier, including cold samples; excludes decode/import/presentation. Quantiles are nearest-rank upper 0.05 ms bin boundaries; null when empty or above 200 ms.',
        session: 'Since attachment start; stage/source changes preserve totals. Frames count successful submissions, not validated GPU output. Mean rates divide all frames by active wall milliseconds; pauses and suspensions excluded, stalls included.',
        callbackP95: 'Session p95 of max(0, callback time minus presentation time), excluding synthetic metadata; 0.05 ms upper-bin quantization, null when empty or above 200 ms.',
        geometryMs: 'Synchronous wall time bracketing inspectGeometry, including style, bounding-rect and hit-test reads; total/max over geometryCalls since construction. Excludes canvas writes and driver work.',
        resources: 'Owned live handles; listeners counts attachment listeners plus the GPU watcher, excluding shared driver/acquisition listeners. frameCallback follows pipeline.running.',
        memory: 'Current neural-stage payload bytes only, excluding padding, timers and browser/driver overhead; not total device memory.',
      },
    };
  }

  private resources(): AttachmentSnapshot['resources'] {
    return {
      device: this.context ? 1 : 0, pipeline: this.videoPipeline ? 1 : 0, canvas: this.disposed ? 0 : 1,
      resizeObservers: this.observer ? 1 : 0, listeners: this.listeners.length + (this.unwatch ? 1 : 0),
      geometryFrame: this.geometryFrame === null ? 0 : 1, frameCallback: this.videoPipeline?.running ? 1 : 0,
    };
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.generation++;
    this.eligible = false;
    this.suspendedReason ??= 'disposed';
    const cleanup = (operation: () => void) => {
      try { operation(); } catch { this.infrastructure.cleanupErrors++; }
    };
    cleanup(() => this.hide());
    if (this.runtime) this.runtime.onChange = null;
    cleanup(() => this.runtime?.destroy());
    cleanup(() => { this.finalSnapshot = this.snapshot(); });
    cleanup(() => this.videoPipeline?.destroy());
    cleanup(() => this.unwatch?.());
    cleanup(() => this.context?.device.destroy());
    cleanup(() => {
      if (this.geometryFrame !== null) this.video.ownerDocument.defaultView?.cancelAnimationFrame(this.geometryFrame);
    });
    cleanup(() => this.observer?.disconnect());
    for (const [target, type, listener, capture] of this.listeners.splice(0)) {
      cleanup(() => target.removeEventListener(type, listener, capture));
    }
    cleanup(() => this.canvas.remove());
    this.runtime = null;
    this.videoPipeline = null;
    this.context = null;
    this.stage = null;
    this.model = null;
    this.failure = null;
    this.observer = null;
    this.observedParent = null;
    this.unwatch = null;
    this.geometryFrame = null;
    if (this.finalSnapshot) {
      this.finalSnapshot.resources = this.resources();
      this.finalSnapshot.infrastructure = { ...this.infrastructure };
      this.finalSnapshot.current = null;
      this.finalSnapshot.memory = null;
      this.finalSnapshot.precision = null;
    }
  }
}