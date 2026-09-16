import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GpuContext } from '../src/core/gpu/device.js';
import type { FrameTick, Size } from '../src/core/types.js';
import type { inspectGeometry } from '../src/extension/geometry.js';
import type { Admission, PresentationInput } from '../tools/m109-contract.js';
import type * as ContractModule from '../tools/m109-contract.js';
import { startMonitor } from '../tools/m109-monitor.js';
import type * as SubmissionModule from '../tools/m109-submission.js';

const mocks = vi.hoisted(() => {
  const construct = vi.fn();
  class FakePipeline {
    onFrame: ((tick: FrameTick) => void) | null = null;
    error: Error | null = null;
    running = false;
    submissionSequence = 0;
    timingGeneration = 0;
    constructor(_gpu: GpuContext, _video: HTMLVideoElement, readonly canvas: HTMLCanvasElement) {
      construct();
      instances.push(this);
    }
    readonly start = vi.fn(() => { this.running = true; });
    readonly stop = vi.fn(() => { this.running = false; });
    readonly destroy = vi.fn(() => { this.stop(); });
    readonly configure = vi.fn((source: Size) => {
      this.canvas.width = source.width * 2;
      this.canvas.height = source.height * 2;
      this.timingGeneration++;
    });
    emit(source: Size, fresh = true) {
      if (fresh) this.submissionSequence++;
      const now = this.submissionSequence * 16;
      this.onFrame?.({ now, mediaTime: now / 1000, size: { ...source }, presentedDelta: 1,
        presentationTime: now, expectedDisplayTime: now + 16, decodeLatencyMs: null });
    }
  }
  const instances: FakePipeline[] = [];
  return { FakePipeline, instances, construct, read: vi.fn<() => PresentationInput>(),
    admission: vi.fn<(input: PresentationInput) => Admission>(),
    proof: vi.fn<typeof inspectGeometry>(), acquire: vi.fn<() => Promise<GpuContext>>(),
    watch: vi.fn<(device: GPUDevice, fail: (message: string) => void) => () => void>(),
    unwatch: vi.fn(), release: vi.fn(), destroyDevice: vi.fn() };
});

vi.mock('../src/core/pipeline.js', () => ({ VideoPipeline: mocks.FakePipeline }));
vi.mock('../src/core/upscale/baseline-scaler.js', () => ({ BaselineScaler: class {} }));
vi.mock('../src/core/gpu/device.js', () => ({ acquireGpu: mocks.acquire, watchDeviceFailures: mocks.watch }));
vi.mock('../src/extension/geometry.js', () => ({ inspectGeometry: mocks.proof }));
vi.mock('../tools/m109-contract.js', async importOriginal => ({
  ...await importOriginal<typeof ContractModule>(),
  createContractReader: () => mocks.read, assessContract: mocks.admission,
}));
vi.mock('../tools/m109-submission.js', async importOriginal => {
  const actual = await importOriginal<typeof SubmissionModule>();
  return { ...actual, observeSuccessfulSubmissions: (...args: Parameters<typeof actual.observeSuccessfulSubmissions>) => {
    const release = actual.observeSuccessfulSubmissions(...args);
    return () => { mocks.release(); release(); };
  } };
});

class FakeEventTarget extends EventTarget {
  readonly listeners = new Map<string, Set<EventListenerOrEventListenerObject>>();
  override addEventListener(type: string, callback: EventListenerOrEventListenerObject | null, options?: AddEventListenerOptions | boolean) {
    super.addEventListener(type, callback, options);
    if (callback) {
      const callbacks = this.listeners.get(type) ?? new Set<EventListenerOrEventListenerObject>();
      callbacks.add(callback);
      this.listeners.set(type, callbacks);
    }
  }
  override removeEventListener(type: string, callback: EventListenerOrEventListenerObject | null, options?: EventListenerOptions | boolean) {
    super.removeEventListener(type, callback, typeof options === 'boolean' ? { capture: options } : options);
    if (callback) this.listeners.get(type)?.delete(callback);
  }
  get listenerCount() { return [...this.listeners.values()].reduce((total, callbacks) => total + callbacks.size, 0); }
}

const disposals: (() => unknown)[] = [];

function makeHarness() {
  for (const mock of [mocks.construct, mocks.read, mocks.admission, mocks.proof, mocks.acquire,
    mocks.watch, mocks.unwatch, mocks.release, mocks.destroyDevice]) mock.mockReset();
  mocks.instances.length = 0;
  const parent = {};
  const properties = new Map<string, string>();
  const canvas = { width: 300, height: 150, dataset: {} as Record<string, string>,
    parentNode: null as object | null, isConnected: false, setAttribute: vi.fn(), remove: vi.fn(),
    style: { setProperty: vi.fn((name: string, value: string) => { properties.set(name, value); }) } };
  const window = new FakeEventTarget();
  const document = Object.assign(new FakeEventTarget(), { createElement: vi.fn(() => canvas) });
  const video = Object.assign(new FakeEventTarget(), { ownerDocument: document, textTracks: new FakeEventTarget(),
    nextSibling: null as typeof canvas | null, after: vi.fn((output: typeof canvas) => {
      output.parentNode = parent; output.isConnected = true; video.nextSibling = output;
    }) });
  canvas.remove.mockImplementation(() => {
    canvas.isConnected = false; canvas.parentNode = null;
    if (video.nextSibling === canvas) video.nextSibling = null;
  });
  const targets = [window, document, video, video.textTracks];
  const frames = new Map<number, FrameRequestCallback>();
  let nextFrame = 0;
  vi.stubGlobal('window', window);
  vi.stubGlobal('document', document);
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frames.set(++nextFrame, callback); return nextFrame;
  });
  const cancel = vi.fn((frame: number) => { frames.delete(frame); });
  vi.stubGlobal('cancelAnimationFrame', cancel);
  const source = { width: 320, height: 180, url: 'fake-source' };
  const rect = { left: 40, top: 80, width: 640, height: 360 };
  const state: PresentationInput = { connected: true, ready: true, playing: true, seeking: false,
    protected: false, mediaError: false, documentVisible: true, nativeControls: false, showingTracks: false,
    pip: false, directFullscreen: false, outsideFullscreen: false, source, rect,
    viewport: { width: 1200, height: 760 }, fullscreen: 0, parent: 1, rootSupported: true,
    nonvideoTopLayer: false, output: null, video: {}, chain: [], branches: [], nonemptyText: false, overflow: false };
  mocks.read.mockImplementation(() => ({ ...state, source: { ...source }, output: {
    width: canvas.width, height: canvas.height, connected: canvas.isConnected,
    parent: canvas.parentNode === parent ? 1 : 0, followsVideo: video.nextSibling === canvas,
    pointerEvents: 'none', rect: { ...rect },
  } }));
  mocks.admission.mockImplementation(input => {
    const output = input.output;
    const reason = !input.playing ? 'video-paused' : output && (!output.connected || !output.followsVideo ||
      output.parent !== input.parent || output.width !== input.source.width * 2 ||
      output.height !== input.source.height * 2) ? 'unsupported-output' : null;
    return { outcome: reason ? 'UNSUPPORTED' : 'SUPPORTED', reason, fingerprint: JSON.stringify(input) };
  });
  mocks.proof.mockReturnValue({ ok: true, rect, clip: { ...rect }, style: {},
    objectFit: 'contain', objectPosition: '50% 50%', borderRadius: '0px',
    placement: { parent: parent as Element, before: null } });
  mocks.acquire.mockResolvedValue({ device: { destroy: mocks.destroyDevice } } as unknown as GpuContext);
  mocks.watch.mockReturnValue(mocks.unwatch);
  return { canvas, video, window, source, rect, state, frames, cancel, targets,
    visibility: () => properties.get('visibility'),
    render: () => {
      const pending = [...frames.values()]; frames.clear();
      for (const callback of pending) callback(16);
    },
    pipeline: () => {
      const pipeline = mocks.instances[0];
      if (!pipeline) throw new Error('Fake pipeline was not constructed');
      return pipeline;
    },
    start: async () => {
      const monitor = await startMonitor(video as unknown as HTMLVideoElement);
      disposals.push(() => monitor.dispose());
      return monitor;
    },
  };
}

afterEach(() => {
  for (const dispose of disposals.splice(0)) dispose();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('M10.9 monitor lifecycle with fake DOM and simulated submissions, not native presentation', () => {
  it('keeps zero backing hidden and requires two distinct authorized successes after warmup', async () => {
    const harness = makeHarness(), monitor = await harness.start(), pipeline = harness.pipeline();
    expect([harness.canvas.width, harness.canvas.height]).toEqual([0, 0]);
    expect(monitor.snapshot()).toMatchObject({ reason: 'backing-not-ready', authorized: false, visible: false, initializes: 1 });
    harness.render();
    expect(harness.visibility()).toBe('hidden');
    expect(monitor.snapshot().submissions).toBe(0);
    pipeline.configure(harness.source);
    pipeline.emit(harness.source);
    expect(monitor.snapshot()).toMatchObject({ authorized: true, visible: false, latest: { validForRecovery: false } });
    pipeline.emit(harness.source);
    expect(monitor.snapshot()).toMatchObject({ visible: false, submissions: 2, latest: { validForRecovery: true } });
    pipeline.emit(harness.source, false);
    harness.render();
    expect(monitor.snapshot()).toMatchObject({ visible: false, submissions: 2 });
    expect(harness.visibility()).toBe('hidden');
    pipeline.emit(harness.source);
    expect(monitor.snapshot()).toMatchObject({ visible: true, submissions: 3, proofCalls: 1 });
    expect(harness.visibility()).toBe('visible');
  });

  it('hides on an unannounced ABR change, reconfigures and recovers without another initialization', async () => {
    const harness = makeHarness(), monitor = await harness.start(), pipeline = harness.pipeline();
    pipeline.configure(harness.source); monitor.guard();
    pipeline.emit(harness.source); pipeline.emit(harness.source);
    const before = monitor.snapshot();
    expect(before.visible).toBe(true);
    Object.assign(harness.source, { width: 640, height: 360 });
    harness.render();
    expect(monitor.snapshot()).toMatchObject({ visible: false, authorized: false, reason: 'backing-not-ready',
      sourceGeneration: before.sourceGeneration + 1, initializes: 1 });
    expect(harness.visibility()).toBe('hidden');
    expect([harness.canvas.width, harness.canvas.height]).toEqual([640, 360]);
    expect(pipeline.running).toBe(true);
    pipeline.configure(harness.source); pipeline.emit(harness.source);
    expect(monitor.snapshot().latest?.validForRecovery).toBe(false);
    pipeline.emit(harness.source);
    expect(harness.visibility()).toBe('hidden');
    pipeline.emit(harness.source);
    expect(monitor.snapshot()).toMatchObject({ visible: true, initializes: 1, proofCalls: 2,
      latest: { backingWidth: 1280, backingHeight: 720, validForRecovery: true } });
    expect(pipeline.configure).toHaveBeenCalledTimes(2);
    expect(mocks.acquire).toHaveBeenCalledTimes(1);
    expect(mocks.construct).toHaveBeenCalledTimes(1);
  });

  it('does not repeat a failed full proof on stable guards and retries only after a change', async () => {
    const harness = makeHarness();
    const supported = mocks.proof.getMockImplementation()!;
    mocks.proof.mockReturnValue({ ok: false, code: 'unsupported-geometry', reason: 'fake rejected effect' });
    const monitor = await harness.start(), pipeline = harness.pipeline();
    for (let index = 0; index < 5; index++) { harness.render(); monitor.guard(); }
    expect(monitor.snapshot()).toMatchObject({ visible: false, authorized: false, proofCalls: 1, reason: 'unsupported-geometry' });
    expect(harness.visibility()).toBe('hidden');
    expect(mocks.proof).toHaveBeenCalledTimes(1);
    expect(pipeline.start).not.toHaveBeenCalled();
    mocks.proof.mockImplementation(supported);
    harness.rect.top++;
    harness.render();
    pipeline.configure(harness.source); monitor.guard();
    pipeline.emit(harness.source); pipeline.emit(harness.source);
    expect(monitor.snapshot()).toMatchObject({ visible: true, proofCalls: 2, initializes: 1 });
  });

  it('latches pipeline.error and never restarts after the underlying error is cleared', async () => {
    const harness = makeHarness(), monitor = await harness.start(), pipeline = harness.pipeline();
    pipeline.configure(harness.source); monitor.guard();
    pipeline.emit(harness.source); pipeline.emit(harness.source);
    pipeline.error = new Error('fake queue failure');
    harness.render();
    expect(monitor.snapshot()).toMatchObject({ error: 'Error: fake queue failure', reason: 'execution-error', visible: false, authorized: false });
    expect(harness.visibility()).toBe('hidden');
    expect(pipeline.running).toBe(false);
    const starts = pipeline.start.mock.calls.length, reads = mocks.read.mock.calls.length;
    pipeline.error = null;
    harness.video.dispatchEvent(new Event('playing'));
    harness.render(); monitor.guard();
    pipeline.emit(harness.source);
    expect(pipeline.start).toHaveBeenCalledTimes(starts);
    expect(mocks.read).toHaveBeenCalledTimes(reads);
    expect(monitor.snapshot().visible).toBe(false);
  });

  it('fails closed on synchronous reader, proof and pipeline-start exceptions', async () => {
    for (const boundary of ['reader', 'proof', 'start'] as const) {
      const harness = makeHarness(), monitor = await harness.start(), pipeline = harness.pipeline();
      const failure = () => { throw new Error(`fake ${boundary} failure`); };
      if (boundary === 'reader') mocks.read.mockImplementationOnce(failure);
      if (boundary === 'proof') { monitor.invalidate(); mocks.proof.mockImplementationOnce(failure); }
      if (boundary === 'start') pipeline.start.mockImplementationOnce(failure);
      expect(() => monitor.guard()).not.toThrow();
      expect(monitor.snapshot()).toMatchObject({ error: `Error: fake ${boundary} failure`, reason: 'execution-error', visible: false, authorized: false });
      expect(harness.visibility()).toBe('hidden');
      expect(pipeline.running).toBe(false);
      const starts = pipeline.start.mock.calls.length;
      harness.render(); monitor.guard();
      expect(pipeline.start).toHaveBeenCalledTimes(starts);
      monitor.dispose();
    }
  });

  it('rejects acquisition and synchronous construction failures while cleaning attached resources', async () => {
    for (const boundary of ['acquire', 'construct'] as const) {
      const harness = makeHarness(), failure = new Error(`fake ${boundary} failure`);
      if (boundary === 'acquire') mocks.acquire.mockRejectedValueOnce(failure);
      else mocks.construct.mockImplementationOnce(() => { throw failure; });
      await expect(harness.start()).rejects.toBe(failure);
      expect(harness.canvas.isConnected).toBe(false);
      expect(harness.canvas.remove).toHaveBeenCalledTimes(1);
      expect(harness.targets.map(target => target.listenerCount)).toEqual([0, 0, 0, 0]);
      expect(harness.frames.size).toBe(0);
      expect(mocks.destroyDevice).toHaveBeenCalledTimes(boundary === 'construct' ? 1 : 0);
    }
  });

  it('attempts every cleanup when GPU destroy throws and reports the unreleased device honestly', async () => {
    const harness = makeHarness(), monitor = await harness.start(), pipeline = harness.pipeline();
    pipeline.configure(harness.source); monitor.guard();
    pipeline.emit(harness.source); pipeline.emit(harness.source);
    const observer = pipeline.onFrame;
    expect(harness.targets.map(target => target.listenerCount)).toEqual([3, 2, 10, 3]);
    expect(harness.frames.size).toBe(1);
    mocks.destroyDevice.mockImplementation(() => { throw new Error('fake device destroy failure'); });
    const result = monitor.dispose();
    expect(result).toMatchObject({ error: 'Error: fake device destroy failure', initializes: 1,
      resources: { device: 1, pipeline: 0, canvas: 0, listeners: 0, frame: 0 } });
    for (const cleanup of [mocks.release, mocks.unwatch, pipeline.destroy, mocks.destroyDevice, harness.canvas.remove]) {
      expect(cleanup).toHaveBeenCalledTimes(1);
    }
    expect(pipeline.onFrame).not.toBe(observer);
    expect(harness.canvas.isConnected).toBe(false);
    expect(harness.visibility()).toBe('hidden');
    expect(harness.targets.map(target => target.listenerCount)).toEqual([0, 0, 0, 0]);
    expect(harness.cancel).toHaveBeenCalledTimes(1);
    expect(harness.frames.size).toBe(0);
    const starts = pipeline.start.mock.calls.length;
    harness.video.dispatchEvent(new Event('playing')); harness.render(); monitor.guard();
    pipeline.emit(harness.source);
    expect(monitor.snapshot().submissions).toBe(result.submissions.length);
    expect(pipeline.start).toHaveBeenCalledTimes(starts);
    expect(monitor.dispose()).toEqual(result);
    expect(harness.canvas.remove).toHaveBeenCalledTimes(1);
  });

  it('hides and stops on reordered output or post-configuration backing mismatch', async () => {
    for (const mutation of ['reorder', 'backing'] as const) {
      const harness = makeHarness(), monitor = await harness.start(), pipeline = harness.pipeline();
      pipeline.configure(harness.source); monitor.guard();
      pipeline.emit(harness.source); pipeline.emit(harness.source);
      expect(harness.visibility()).toBe('visible');
      const starts = pipeline.start.mock.calls.length;
      if (mutation === 'reorder') harness.video.nextSibling = null;
      else harness.canvas.width++;
      harness.render();
      expect(monitor.snapshot()).toMatchObject({ reason: 'unsupported-output', visible: false, authorized: false });
      expect(harness.visibility()).toBe('hidden');
      expect(pipeline.running).toBe(false);
      expect(pipeline.start).toHaveBeenCalledTimes(starts);
      expect(mocks.proof).toHaveBeenCalledTimes(1);
      if (mutation === 'reorder') harness.video.nextSibling = harness.canvas;
      else harness.canvas.width--;
      harness.render(); pipeline.emit(harness.source);
      expect(harness.visibility()).toBe('hidden');
      pipeline.emit(harness.source);
      expect(harness.visibility()).toBe('visible');
      expect(monitor.dispose().resources).toEqual({ device: 0, pipeline: 0, canvas: 0, listeners: 0, frame: 0 });
    }
  });
});