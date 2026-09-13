import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { GpuContext } from '../src/core/gpu/device.js';
import { GpuTimer } from '../src/core/metrics/gpu-timer.js';
import { VideoPipeline } from '../src/core/pipeline.js';
import type { EncodeContext, FrameTexture, FrameTextureKind, FrameTick, Size, Upscaler, UpscalerConfig } from '../src/core/types.js';

const mocks = vi.hoisted(() => ({
  deliver: null as ((tick: FrameTick) => void) | null,
  running: false,
  configureTarget: vi.fn(),
  configureImporter: vi.fn<(size: Size) => void>(),
  sampledView: null as GPUTextureView | null,
  acquire: vi.fn<() => FrameTexture>(),
  createImporter: vi.fn(),
  createTarget: vi.fn(),
  createTimer: vi.fn(),
  destroySource: vi.fn(),
  destroyImporter: vi.fn(),
  unconfigureTarget: vi.fn(),
  loadListeners: new Set<() => void>(),
}));

vi.mock('../src/core/acquisition/video-source.js', () => ({
  VideoFrameSource: class {
    readonly clock = 'rvfc';
    readonly loadGeneration = 0;
    private readonly listener = () => {};
    constructor() { mocks.loadListeners.add(this.listener); }
    get running() { return mocks.running; }
    quality() { return { totalVideoFrames: 0, droppedVideoFrames: 0, corruptedVideoFrames: 0 }; }
    start(callback: (tick: FrameTick) => void) {
      mocks.deliver = callback;
      mocks.running = true;
    }
    stop() { mocks.running = false; }
    destroy() {
      this.stop();
      mocks.deliver = null;
      mocks.loadListeners.delete(this.listener);
      mocks.destroySource();
    }
  },
}));

vi.mock('../src/core/acquisition/frame-importer.js', () => ({
  FrameImporter: class {
    readonly kind: FrameTextureKind;
    constructor(_device: GPUDevice, _video: HTMLVideoElement, external: boolean) {
      mocks.createImporter();
      this.kind = external ? 'external' : 'sampled';
    }
    readonly configure = mocks.configureImporter;
    get sampledView() { return this.kind === 'sampled' ? mocks.sampledView : null; }
    readonly acquire = mocks.acquire;
    readonly destroy = mocks.destroyImporter;
  },
}));

vi.mock('../src/core/present/canvas-target.js', () => ({
  CanvasTarget: class {
    size: Size = { width: 0, height: 0 };
    constructor() { mocks.createTarget(); }
    configure(_device: GPUDevice, size: Size) {
      mocks.configureTarget();
      this.size = size;
    }
    currentView() { return {}; }
    readonly unconfigure = mocks.unconfigureTarget;
  },
}));

function makeTick(width = 320, height = 180, now = 100): FrameTick {
  return {
    now, mediaTime: now / 1000, size: { width, height }, presentedDelta: 1,
    presentationTime: now - 2, expectedDisplayTime: now + 16, decodeLatencyMs: 3,
  };
}

function makeHarness(
  timestampQuery = true,
  externalTexture = false,
  beforeConstruct?: (gpu: GpuContext, upscaler: Upscaler) => void,
) {
  const pending: (() => void)[] = [];
  const clock = { now: 10 };
  vi.spyOn(performance, 'now').mockImplementation(() => clock.now);
  const epoch = vi.spyOn(GpuTimer.prototype, 'newEpoch');
  const encoder = {
    resolveQuerySet: vi.fn(), copyBufferToBuffer: vi.fn(), finish: vi.fn(() => ({})),
  };
  const device = {
    createQuerySet: vi.fn(() => {
      mocks.createTimer();
      return { destroy: vi.fn() };
    }),
    createBuffer: vi.fn(() => ({
      mapAsync: () => new Promise<void>((resolve) => pending.push(resolve)),
      getMappedRange: () => new BigInt64Array([0n, 2_000_000n]).buffer,
      unmap: vi.fn(), destroy: vi.fn(),
    })),
    createCommandEncoder: vi.fn(() => encoder),
    queue: { submit: vi.fn() },
  };
  const upscaler = {
    id: 'first', label: 'First', neural: true, scaleFactor: 2,
    configure: vi.fn<(config: UpscalerConfig) => void>(),
    encode: vi.fn<(context: EncodeContext) => void>(),
    destroy: vi.fn(),
  };
  const gpu = {
    device, capabilities: { timestampQuery, externalTexture, preferredCanvasFormat: 'rgba8unorm' },
  } as unknown as GpuContext;
  beforeConstruct?.(gpu, upscaler);
  const pipeline = new VideoPipeline(gpu, {} as HTMLVideoElement, {} as HTMLCanvasElement, upscaler);
  pipeline.start();
  return {
    pipeline, upscaler, device, clock, epoch,
    emit: (tick = makeTick()) => {
      if (mocks.deliver === null) throw new Error('Frame source was not started');
      mocks.deliver(tick);
    },
    settle: async () => {
      while (pending.length > 0) {
        pending.shift()?.();
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
      }
    },
  };
}

beforeEach(() => {
  mocks.deliver = null;
  mocks.running = false;
  mocks.loadListeners.clear();
  for (const mock of [
    mocks.createImporter, mocks.createTarget, mocks.createTimer,
    mocks.destroySource, mocks.destroyImporter, mocks.unconfigureTarget,
  ]) mock.mockReset();
  mocks.configureTarget.mockReset();
  mocks.sampledView = null;
  mocks.configureImporter.mockReset().mockImplementation(() => { mocks.sampledView ??= {} as GPUTextureView; });
  mocks.acquire.mockReset().mockImplementation(() => {
    if (mocks.sampledView === null) throw new Error('Importer was not configured');
    return { kind: 'sampled', view: mocks.sampledView };
  });
  vi.stubGlobal('GPUBufferUsage', { COPY_SRC: 0x0004, COPY_DST: 0x0008, QUERY_RESOLVE: 0x0200, MAP_READ: 0x0001 });
  vi.stubGlobal('GPUMapMode', { READ: 0x0001 });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('VideoPipeline teardown', () => {
  it('is terminal, clears observers, and releases each resource exactly once', async () => {
    const timerDestroy = vi.spyOn(GpuTimer.prototype, 'destroy');
    const { pipeline, upscaler, device, emit, settle } = makeHarness();
    emit();
    const stale = mocks.deliver!;
    const observer = vi.fn();
    pipeline.onGpuPassSample = observer;
    pipeline.onGpuSample = observer;
    pipeline.onFrame = observer;
    pipeline.onConfiguration = observer;
    expect(mocks.loadListeners.size).toBe(1);
    pipeline.destroy();
    pipeline.destroy();
    pipeline.start();
    pipeline.setUpscaler(upscaler);
    const incoming = { ...upscaler, id: 'late', destroy: vi.fn() };
    pipeline.setUpscaler(incoming);
    stale(makeTick());
    await settle();
    expect(pipeline.running).toBe(false);
    expect(pipeline.currentUpscaler).toBe(upscaler);
    expect(mocks.deliver).toBeNull();
    expect(mocks.loadListeners.size).toBe(0);
    expect(observer).not.toHaveBeenCalled();
    expect([pipeline.onGpuPassSample, pipeline.onGpuSample, pipeline.onFrame, pipeline.onConfiguration])
      .toEqual([null, null, null, null]);
    for (const cleanup of [
      mocks.destroySource, upscaler.destroy, mocks.destroyImporter,
      timerDestroy, mocks.unconfigureTarget, incoming.destroy,
    ]) expect(cleanup).toHaveBeenCalledTimes(1);
    expect(device.queue.submit).toHaveBeenCalledTimes(1);
    for (const allocation of [...device.createQuerySet.mock.results, ...device.createBuffer.mock.results]) {
      expect(allocation.type).toBe('return');
      if (allocation.type === 'return') expect(allocation.value.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('does not encode or submit after an onConfiguration observer destroys it', () => {
    const { pipeline, upscaler, device, emit } = makeHarness();
    const onFrame = vi.fn();
    pipeline.onFrame = onFrame;
    pipeline.onConfiguration = () => { pipeline.destroy(); pipeline.start(); };
    emit();
    expect(pipeline.running).toBe(false);
    expect(mocks.acquire).not.toHaveBeenCalled();
    expect(upscaler.encode).not.toHaveBeenCalled();
    expect(device.queue.submit).not.toHaveBeenCalled();
    expect(onFrame).not.toHaveBeenCalled();
  });

  it('marks disposal before cleanup reenters and attempts every cleanup despite failures', () => {
    const { pipeline, upscaler } = makeHarness();
    const error = new Error('source cleanup');
    const incoming = { ...upscaler, destroy: vi.fn() };
    mocks.destroySource.mockImplementation(() => {
      pipeline.destroy();
      pipeline.start();
      pipeline.setUpscaler(incoming);
      throw error;
    });
    upscaler.destroy.mockImplementation(() => { throw new Error('stage cleanup'); });
    mocks.destroyImporter.mockImplementation(() => { throw new Error('importer cleanup'); });
    const timerDestroy = vi.spyOn(GpuTimer.prototype, 'destroy').mockImplementation(() => { throw new Error('timer cleanup'); });
    mocks.unconfigureTarget.mockImplementation(() => { throw new Error('target cleanup'); });
    expect(() => pipeline.destroy()).toThrow(error);
    expect(() => pipeline.destroy()).not.toThrow();
    expect(pipeline.running).toBe(false);
    expect(mocks.loadListeners.size).toBe(0);
    for (const cleanup of [
      mocks.destroySource, upscaler.destroy, mocks.destroyImporter,
      timerDestroy, mocks.unconfigureTarget, incoming.destroy,
    ]) expect(cleanup).toHaveBeenCalledTimes(1);
  });

  it.each(['importer', 'target', 'timer'] as const)('unwinds a %s constructor failure without masking the original error', (failure) => {
    const error = new DOMException('construction failed', 'OperationError');
    const stageDestroy = vi.fn();
    const failingCall = { importer: mocks.createImporter, target: mocks.createTarget, timer: mocks.createTimer }[failure];
    failingCall.mockImplementation(() => { throw error; });
    mocks.destroySource.mockImplementation(() => { throw new Error('source cleanup'); });
    mocks.destroyImporter.mockImplementation(() => { throw new Error('importer cleanup'); });
    expect(() => makeHarness(true, false, (_gpu, upscaler) => { upscaler.destroy = stageDestroy; })).toThrow(error);
    expect(mocks.loadListeners.size).toBe(0);
    expect(mocks.destroySource).toHaveBeenCalledTimes(1);
    expect(mocks.destroyImporter).toHaveBeenCalledTimes(failure === 'importer' ? 0 : 1);
    expect(mocks.unconfigureTarget).toHaveBeenCalledTimes(failure === 'timer' ? 1 : 0);
    expect(stageDestroy).toHaveBeenCalledTimes(1);
    expect(mocks.running).toBe(false);
  });

  it('exposes the actual hot-path DOMException without allowing a stopped stale callback to resume', () => {
    const { pipeline, upscaler, device, emit } = makeHarness(false);
    const error = new DOMException('import denied', 'SecurityError');
    const stale = mocks.deliver!;
    mocks.acquire.mockImplementationOnce(() => { throw error; });
    expect(() => emit()).toThrow(error);
    expect(pipeline.error).toBe(error);
    stale(makeTick());
    expect(pipeline.error).toBe(error);
    expect(upscaler.encode).not.toHaveBeenCalled();
    expect(device.queue.submit).not.toHaveBeenCalled();
    expect(pipeline.running).toBe(false);
    pipeline.destroy();
  });
});

describe('VideoPipeline timing provenance', () => {
  it('prepares the sampled view before stage configuration and forwards the original frame', () => {
    const { pipeline, upscaler, emit } = makeHarness(false);
    expect(mocks.sampledView).toBeNull();
    upscaler.configure.mockImplementation((config) => {
      expect(config.sourceKind).toBe('sampled');
      expect(config.sampledSourceView).not.toBeNull();
      expect(config.sampledSourceView).toBe(mocks.sampledView);
      expect(mocks.acquire).not.toHaveBeenCalled();
    });
    emit();
    expect(mocks.configureImporter).toHaveBeenCalledExactlyOnceWith({ width: 320, height: 180 });
    expect(mocks.configureImporter.mock.invocationCallOrder[0]!).toBeLessThan(upscaler.configure.mock.invocationCallOrder[0]!);
    expect(upscaler.encode.mock.calls[0]?.[0].frame).toBe(mocks.acquire.mock.results[0]?.value);
    emit();
    expect(mocks.configureImporter).toHaveBeenCalledTimes(1);
    expect(upscaler.configure).toHaveBeenCalledTimes(1);

    upscaler.configure.mockReset();
    const resizedView = {} as GPUTextureView;
    mocks.configureImporter.mockImplementationOnce(() => { mocks.sampledView = resizedView; });
    emit(makeTick(640, 360));
    expect(mocks.configureImporter).toHaveBeenLastCalledWith({ width: 640, height: 360 });
    expect(upscaler.configure).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({
      source: { width: 640, height: 360 }, sampledSourceView: resizedView,
    }));
    expect(upscaler.encode.mock.calls.at(-1)?.[0].frame).toEqual({ kind: 'sampled', view: resizedView });
    pipeline.destroy();
  });

  it('configures external import without supplying a sampled view', () => {
    const { pipeline, upscaler, emit } = makeHarness(false, true);
    const frame: FrameTexture = { kind: 'external', texture: {} as GPUExternalTexture };
    mocks.acquire.mockReturnValue(frame);
    emit();
    expect(mocks.configureImporter).toHaveBeenCalledExactlyOnceWith({ width: 320, height: 180 });
    expect(upscaler.configure.mock.calls[0]?.[0]).toMatchObject({ sourceKind: 'external' });
    expect(upscaler.configure.mock.calls[0]?.[0]).not.toHaveProperty('sampledSourceView');
    expect(upscaler.encode.mock.calls[0]?.[0].frame).toBe(frame);
    pipeline.destroy();
  });

  it('drains the outer timer after stopping without changing epochs or allocating GPU resources', async () => {
    const { pipeline, device, epoch, emit, settle } = makeHarness();
    const onGpuSample = vi.fn<NonNullable<VideoPipeline['onGpuSample']>>();
    const onGpuPassSample = vi.fn();
    pipeline.onGpuSample = onGpuSample;
    pipeline.onGpuPassSample = onGpuPassSample;
    for (let index = 0; index < 4; index++) emit();
    pipeline.stop();
    const generation = pipeline.timingGeneration;
    const epochCalls = epoch.mock.calls.length;
    const drained = vi.fn();
    const draining = pipeline.drainTimings().then(drained);
    await Promise.resolve();
    expect(drained).not.toHaveBeenCalled();
    expect(onGpuSample).not.toHaveBeenCalled();

    await settle();
    await draining;
    expect(drained).toHaveBeenCalledExactlyOnceWith(undefined);
    expect(onGpuSample.mock.calls.map(([sample]) => [sample.sequence, sample.generation])).toEqual([
      [1, generation], [2, generation], [3, generation], [4, generation],
    ]);
    expect(onGpuPassSample.mock.calls).toEqual([[2], [2], [2], [2]]);
    expect(pipeline.stats(100).gpuPassMs).toMatchObject({ samples: 4 });
    expect(pipeline.running).toBe(false);
    expect(pipeline.timingGeneration).toBe(generation);
    expect(epoch).toHaveBeenCalledTimes(epochCalls);
    expect(device.createQuerySet).toHaveBeenCalledTimes(1);
    expect(device.createBuffer).toHaveBeenCalledTimes(8);
    pipeline.destroy();
  });

  it('resolves drainTimings when GPU timestamps are unavailable', async () => {
    const { pipeline, device, epoch } = makeHarness(false);
    pipeline.stop();
    await expect(pipeline.drainTimings()).resolves.toBeUndefined();
    expect(pipeline.timingGeneration).toBe(0);
    expect(epoch).not.toHaveBeenCalled();
    expect(device.createQuerySet).not.toHaveBeenCalled();
    expect(device.createBuffer).not.toHaveBeenCalled();
    pipeline.destroy();
  });

  it('snapshots provenance before encode and reports synchronous target, importer and stage configuration time', async () => {
    const { pipeline, upscaler, device, clock, epoch, emit, settle } = makeHarness();
    const onGpuSample = vi.fn();
    const onGpuPassSample = vi.fn();
    const onConfiguration = vi.fn();
    const onFrame = vi.fn(() => expect(device.queue.submit).toHaveBeenCalledTimes(1));
    pipeline.onGpuSample = onGpuSample;
    pipeline.onGpuPassSample = onGpuPassSample;
    pipeline.onConfiguration = onConfiguration;
    pipeline.onFrame = onFrame;
    expect(pipeline.timingGeneration).toBe(0);
    mocks.configureTarget.mockImplementation(() => {
      expect(pipeline.timingGeneration).toBe(1);
      expect(epoch).toHaveBeenCalledTimes(1);
      clock.now += 2;
    });
    mocks.configureImporter.mockImplementation(() => { clock.now += 4; });
    upscaler.configure.mockImplementation(() => { clock.now += 3; });
    mocks.acquire.mockImplementation(() => {
      clock.now += 1;
      return { kind: 'sampled', view: {} as GPUTextureView };
    });
    upscaler.encode.mockImplementation(() => { clock.now += 7; });
    const source = { width: 320, height: 180 };
    const tick = { ...makeTick(), size: source };
    emit(tick);
    expect(onFrame).toHaveBeenCalledExactlyOnceWith(tick);
    expect(onGpuSample).not.toHaveBeenCalled();
    source.width = 99;
    upscaler.id = 'mutated-after-submit';
    upscaler.neural = false;
    clock.now = 40;
    await settle();
    expect(onGpuSample).toHaveBeenCalledExactlyOnceWith({
      ms: 2, generation: 1, sequence: 1, submittedAt: 20, resolvedAt: 40,
      neural: true, upscalerId: 'first', source: { width: 320, height: 180 },
    });
    expect(onGpuPassSample).toHaveBeenCalledExactlyOnceWith(2);
    expect(onConfiguration).toHaveBeenCalledExactlyOnceWith({
      generation: 1, source: { width: 320, height: 180 }, target: { width: 640, height: 360 },
      configureMs: 9, sourceChanged: true,
    });
    expect(pipeline.stats(100).sourceSize).toEqual({ width: 320, height: 180 });
  });

  it('invalidates before resize configuration, dropping pending samples without clearing legacy metrics', async () => {
    const { pipeline, epoch, emit, settle } = makeHarness();
    const onGpuSample = vi.fn<NonNullable<VideoPipeline['onGpuSample']>>();
    const onGpuPassSample = vi.fn();
    pipeline.onGpuSample = onGpuSample;
    pipeline.onGpuPassSample = onGpuPassSample;
    emit();
    await settle();
    emit(makeTick(320, 180, 116));
    mocks.configureTarget.mockImplementation(() => {
      expect(pipeline.timingGeneration).toBe(2);
      expect(epoch).toHaveBeenCalledTimes(2);
    });
    const onConfiguration = vi.fn();
    pipeline.onConfiguration = onConfiguration;
    emit(makeTick(640, 360, 132));
    expect(onConfiguration).toHaveBeenCalledExactlyOnceWith({
      generation: 2, source: { width: 640, height: 360 }, target: { width: 1280, height: 720 },
      configureMs: 0, sourceChanged: true,
    });
    expect(pipeline.stats(132)).toMatchObject({
      framesRendered: 3, elapsedMs: 32, cpuFrameMs: { samples: 3 }, gpuPassMs: { samples: 1 },
      callbackLatencyMs: { samples: 3 }, decodeLatencyMs: { samples: 3 },
    });
    await settle();
    expect(onGpuSample.mock.calls.map(([sample]) => [sample.sequence, sample.generation])).toEqual([[1, 1], [3, 2]]);
    expect(onGpuPassSample.mock.calls).toEqual([[2], [2]]);
    expect(pipeline.stats(132).gpuPassMs).toMatchObject({ samples: 2 });
  });

  it('separates timing invalidation from legacy reset and keeps submission sequences across both', async () => {
    const { pipeline, epoch, emit, settle } = makeHarness();
    const onGpuSample = vi.fn();
    pipeline.onGpuSample = onGpuSample;
    emit();
    await settle();
    emit();
    pipeline.invalidateTiming();
    expect(pipeline.timingGeneration).toBe(2);
    expect(pipeline.stats(100)).toMatchObject({ framesRendered: 2, gpuPassMs: { samples: 1 } });
    await settle();
    expect(onGpuSample).toHaveBeenCalledTimes(1);
    emit();
    pipeline.resetMeasurements();
    expect(pipeline.timingGeneration).toBe(3);
    expect(epoch).toHaveBeenCalledTimes(3);
    expect(pipeline.stats(100)).toMatchObject({
      framesRendered: 0, framesSkipped: 0, elapsedMs: 0, cpuFrameMs: { samples: 0 }, gpuPassMs: { samples: 0 },
      callbackLatencyMs: { samples: 0 }, decodeLatencyMs: null,
    });
    await settle();
    expect(onGpuSample).toHaveBeenCalledTimes(1);
    emit();
    await settle();
    expect(onGpuSample).toHaveBeenLastCalledWith(expect.objectContaining({ sequence: 4, generation: 3 }));
    expect(mocks.configureTarget).toHaveBeenCalledTimes(1);
  });

  it.each([false, true])('distinguishes actual source changes from a stage swap (resized: %s)', async (resized) => {
    const { pipeline, upscaler, emit, settle } = makeHarness();
    const onGpuSample = vi.fn();
    pipeline.onGpuSample = onGpuSample;
    emit();
    pipeline.setUpscaler(upscaler);
    expect(pipeline.timingGeneration).toBe(1);
    const next = { ...upscaler, id: 'second', neural: false, scaleFactor: 3, configure: vi.fn(), destroy: vi.fn() };
    pipeline.setUpscaler(next);
    expect(pipeline.timingGeneration).toBe(2);
    expect(upscaler.destroy).toHaveBeenCalledTimes(1);
    expect(pipeline.stats(100).framesRendered).toBe(0);
    const onConfiguration = vi.fn();
    pipeline.onConfiguration = onConfiguration;
    const source = resized ? { width: 640, height: 360 } : { width: 320, height: 180 };
    emit(makeTick(source.width, source.height));
    await settle();
    const generation = resized ? 3 : 2;
    expect(onConfiguration).toHaveBeenCalledExactlyOnceWith({
      generation, source, target: { width: source.width * 3, height: source.height * 3 },
      configureMs: 0, sourceChanged: resized,
    });
    expect(onGpuSample).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({
      generation, sequence: 2, source, neural: false, upscalerId: 'second',
    }));
  });

  it.each([false, true])('reports every submitted frame even without a timing slot (timestamps: %s)', async (timestamps) => {
    const { pipeline, device, emit, settle } = makeHarness(timestamps);
    const onFrame = vi.fn();
    const onGpuSample = vi.fn();
    pipeline.onFrame = onFrame;
    pipeline.onGpuSample = onGpuSample;
    emit(makeTick(0, 0));
    for (let index = 0; index < 6; index++) emit(makeTick(320, 180, 100 + index * 16));
    expect(onFrame).toHaveBeenCalledTimes(6);
    expect(device.queue.submit).toHaveBeenCalledTimes(6);
    expect(pipeline.timingGeneration).toBe(1);
    await settle();
    expect(onGpuSample).toHaveBeenCalledTimes(timestamps ? 4 : 0);
    emit();
    await settle();
    expect(onFrame).toHaveBeenCalledTimes(7);
    if (timestamps) expect(onGpuSample).toHaveBeenLastCalledWith(expect.objectContaining({ sequence: 7 }));
    else expect(pipeline.stats(100).gpuPassMs).toBeNull();
    expect(device.createQuerySet).toHaveBeenCalledTimes(timestamps ? 1 : 0);
    expect(device.createBuffer).toHaveBeenCalledTimes(timestamps ? 8 : 0);
  });

  it.each(['target', 'importer', 'stage', 'encode', 'submit'] as const)('does not report a submitted frame after a %s failure', (failure) => {
    const { pipeline, upscaler, device, emit } = makeHarness();
    const error = new Error(failure);
    const onFrame = vi.fn();
    const onConfiguration = vi.fn();
    pipeline.onFrame = onFrame;
    pipeline.onConfiguration = onConfiguration;
    const failingCall = { target: mocks.configureTarget, importer: mocks.configureImporter, stage: upscaler.configure, encode: upscaler.encode, submit: device.queue.submit }[failure];
    failingCall.mockImplementationOnce(() => { throw error; });
    expect(() => emit()).toThrow(error);
    expect(onFrame).not.toHaveBeenCalled();
    expect(onConfiguration).toHaveBeenCalledTimes(['target', 'importer', 'stage'].includes(failure) ? 0 : 1);
    expect(pipeline.error).toBe(error);
    expect(pipeline.running).toBe(false);
    expect(pipeline.stats(100).framesRendered).toBe(0);
  });

  it('suppresses both raw callbacks when destruction precedes readback', async () => {
    const { pipeline, emit, settle } = makeHarness();
    const onGpuSample = vi.fn();
    const onGpuPassSample = vi.fn();
    pipeline.onGpuSample = onGpuSample;
    pipeline.onGpuPassSample = onGpuPassSample;
    emit();
    pipeline.destroy();
    await settle();
    expect(onGpuSample).not.toHaveBeenCalled();
    expect(onGpuPassSample).not.toHaveBeenCalled();
  });
});