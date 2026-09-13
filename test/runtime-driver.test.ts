import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { RuntimeDriver } from '../src/runtime.js';
import type { VideoPipeline, PipelineConfiguration, PipelineGpuSample } from '../src/core/pipeline.js';
import type { FrameTick, Upscaler } from '../src/core/types.js';
import type { RuntimeMode } from '../src/core/upscale/runtime-controller.js';

vi.mock('../src/core/upscale/baseline-scaler.js', () => ({
  BaselineScaler: vi.fn(function () { return createUpscaler(false); }),
}));

function createUpscaler(neural: boolean) {
  return {
    id: neural ? 'test-neural' : 'catmull-rom',
    label: neural ? 'Test neural' : 'Catmull-Rom',
    scaleFactor: 2,
    neural,
    configure: vi.fn<Upscaler['configure']>(),
    encode: vi.fn<Upscaler['encode']>(),
    destroy: vi.fn<Upscaler['destroy']>(),
  } satisfies Upscaler;
}

class FakeVideo extends EventTarget {
  paused = true;
  seeking = false;
  ended = false;
  playbackRate = 1;
  videoWidth = 1280;
  videoHeight = 720;
  quality = { totalVideoFrames: 0, droppedVideoFrames: 0, corruptedVideoFrames: 0 };
  getVideoPlaybackQuality() { return { ...this.quality }; }
}

const drivers: RuntimeDriver[] = [];
const clock = () => Date.now();

function createHarness(options: { mode?: RuntimeMode; timestamps?: boolean; visible?: () => boolean } = {}) {
  const documentTarget = Object.assign(new EventTarget(), { visibilityState: 'visible' });
  vi.stubGlobal('document', documentTarget);
  const video = new FakeVideo();
  const baseline = createUpscaler(false);
  const factory = vi.fn(() => createUpscaler(true));
  let needsConfiguration = false;
  const pipeline = {
    currentUpscaler: baseline,
    timingGeneration: 0,
    running: false,
    error: null as unknown,
    onFrame: null as VideoPipeline['onFrame'],
    onConfiguration: null as VideoPipeline['onConfiguration'],
    onGpuSample: null as VideoPipeline['onGpuSample'],
    start: vi.fn(() => { pipeline.running = true; }),
    stop: vi.fn(() => { pipeline.running = false; }),
    invalidateTiming: vi.fn(() => { pipeline.timingGeneration++; }),
    resetMeasurements: vi.fn(() => { pipeline.invalidateTiming(); }),
    setUpscaler: vi.fn((next: ReturnType<typeof createUpscaler>) => {
      if (next === pipeline.currentUpscaler) return;
      pipeline.currentUpscaler.destroy();
      pipeline.currentUpscaler = next;
      pipeline.resetMeasurements();
      needsConfiguration = true;
    }),
    destroy: vi.fn(),
  };
  const driver = new RuntimeDriver(pipeline as unknown as VideoPipeline, video as unknown as HTMLVideoElement,
    options.timestamps ?? true, options.mode ?? 'auto', options.visible);
  drivers.push(driver);
  const callbacks = {
    change: vi.fn<NonNullable<RuntimeDriver['onChange']>>(),
    sample: vi.fn<NonNullable<RuntimeDriver['onSample']>>(),
    frame: vi.fn<NonNullable<RuntimeDriver['onFrame']>>(),
    configure: vi.fn<NonNullable<RuntimeDriver['onConfigure']>>(),
  };
  driver.onChange = callbacks.change;
  driver.onSample = callbacks.sample;
  driver.onFrame = callbacks.frame;
  driver.onConfigure = callbacks.configure;
  let mediaTime = 0;
  let sequence = 0;

  function configure(sourceChanged = true): PipelineConfiguration {
    if (sourceChanged) pipeline.invalidateTiming();
    needsConfiguration = false;
    const config = {
      generation: pipeline.timingGeneration,
      source: { width: video.videoWidth, height: video.videoHeight },
      target: { width: video.videoWidth * 2, height: video.videoHeight * 2 },
      configureMs: 2,
      sourceChanged,
    };
    pipeline.onConfiguration?.(config);
    return config;
  }

  function frames(count: number, gpuMs?: number, options: { fps?: number; mediaDelta?: number } = {}) {
    let sample: PipelineGpuSample | null = null;
    for (let index = 0; index < count; index++) {
      vi.advanceTimersByTime(1000 / (options.fps ?? 50));
      if (needsConfiguration) configure(false);
      mediaTime += options.mediaDelta ?? video.playbackRate / (options.fps ?? 50);
      video.quality.totalVideoFrames++;
      sequence++;
      const tick: FrameTick = {
        now: clock(), mediaTime, presentedDelta: 1,
        size: { width: video.videoWidth, height: video.videoHeight },
        presentationTime: clock() - 1, expectedDisplayTime: clock() + 1,
        decodeLatencyMs: null,
      };
      sample = pipeline.currentUpscaler.neural ? {
        ms: gpuMs ?? 6, generation: pipeline.timingGeneration, sequence,
        submittedAt: clock(), resolvedAt: clock(), neural: true,
        upscalerId: pipeline.currentUpscaler.id, source: tick.size,
      } : null;
      pipeline.onFrame?.(tick);
      if (sample !== null && gpuMs !== undefined) pipeline.onGpuSample?.(sample);
    }
    return sample;
  }

  function play() {
    video.paused = false;
    video.dispatchEvent(new Event('playing'));
  }

  function startNeural() {
    configure();
    play();
    driver.setNeuralFactory(factory);
    expect(factory).not.toHaveBeenCalled();
    frames(31);
    expect(driver.snapshot()).toMatchObject({
      actualTier: 'neural', controller: { state: 'warmup', samples: 0, fps: 50 },
    });
  }

  return { driver, pipeline, video, documentTarget, baseline, factory, callbacks,
    configure, frames, play, startNeural };
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
  vi.setSystemTime(0);
  vi.spyOn(globalThis.performance, 'now').mockImplementation(clock);
});

afterEach(() => {
  for (const driver of drivers.splice(0)) driver.destroy();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('RuntimeDriver', () => {
  it('accepts a delivered rVFC timestamp preceding the resume event clock', () => {
    const { driver, pipeline, play, configure } = createHarness();
    configure();
    vi.advanceTimersByTime(100);
    play();
    expect(() => pipeline.onFrame?.({ now: 95, mediaTime: 1, presentedDelta: 1,
      size: { width: 1280, height: 720 }, presentationTime: 94, expectedDisplayTime: 110,
      decodeLatencyMs: null })).not.toThrow();
    expect(driver.snapshot().session.framesRendered).toBe(1);
  });

  it('keeps manual baseline intent when the neural factory arrives after clean cadence', () => {
    const { driver, pipeline, baseline, factory, configure, frames, play } = createHarness();
    configure();
    play();
    frames(31);
    expect(driver.snapshot()).toMatchObject({
      actualTier: 'baseline', running: true, controller: { state: 'unavailable', mode: 'auto' },
      session: { framesRendered: 31, cadence: { cleanBaselineIntervals: 30 } },
    });

    driver.setMode('baseline');
    driver.setNeuralFactory(factory);
    frames(10);
    vi.advanceTimersByTime(3000);

    expect(driver.snapshot()).toMatchObject({
      actualTier: 'baseline', controller: { state: 'manual-baseline', mode: 'baseline' },
    });
    expect(pipeline.currentUpscaler).toBe(baseline);
    expect(factory).not.toHaveBeenCalled();
    expect(pipeline.setUpscaler).not.toHaveBeenCalled();
    expect(baseline.destroy).not.toHaveBeenCalled();
    expect(pipeline.destroy).not.toHaveBeenCalled();
  });

  it('confirms a successful recovery probe without reconstructing the selected neural stage', () => {
    const { driver, pipeline, factory, frames, startNeural } = createHarness();
    startNeural();
    frames(40, 6);
    expect(driver.snapshot().controller).toMatchObject({ state: 'stable', samples: 30, medianMs: 6 });

    frames(40, 40);
    expect(driver.snapshot()).toMatchObject({ actualTier: 'baseline', controller: { state: 'fallback' } });
    expect(pipeline.setUpscaler).toHaveBeenCalledTimes(2);
    vi.advanceTimersByTime(2100);
    expect(driver.snapshot()).toMatchObject({
      actualTier: 'neural', controller: { state: 'probing', probeCount: 1 },
    });
    const selected = pipeline.currentUpscaler;
    expect(factory).toHaveBeenCalledTimes(2);

    frames(40, 6);

    expect(driver.snapshot().controller).toMatchObject({
      state: 'stable', samples: 30, medianMs: 6, probeCount: 1, failedProbeCount: 0,
    });
    expect(driver.snapshot().controller.transitions.at(-1)).toMatchObject({ from: 'probing', to: 'stable' });
    expect(pipeline.currentUpscaler).toBe(selected);
    expect(factory).toHaveBeenCalledTimes(2);
    expect(pipeline.setUpscaler).toHaveBeenCalledTimes(3);
    expect(selected.destroy).not.toHaveBeenCalled();
    expect(pipeline.destroy).not.toHaveBeenCalled();
  });

  it('waits on baseline when forced before the factory arrives and probes only after release', () => {
    const { driver, pipeline, baseline, factory, configure, frames, play } = createHarness();
    configure();
    play();
    frames(31);
    driver.force(true);
    expect(driver.snapshot().controller).toMatchObject({ state: 'unavailable', forced: true });

    driver.setNeuralFactory(factory);
    vi.advanceTimersByTime(5000);
    expect(driver.snapshot()).toMatchObject({
      actualTier: 'baseline', controller: { state: 'fallback', forced: true, probeCount: 0 },
    });
    expect(pipeline.currentUpscaler).toBe(baseline);
    expect(factory).not.toHaveBeenCalled();
    expect(pipeline.setUpscaler).not.toHaveBeenCalled();

    driver.force(false);
    expect(factory).not.toHaveBeenCalled();
    vi.advanceTimersByTime(100);
    expect(driver.snapshot()).toMatchObject({
      actualTier: 'neural', controller: { state: 'probing', forced: false, probeCount: 1 },
    });
    expect(factory).toHaveBeenCalledTimes(1);
    expect(baseline.destroy).toHaveBeenCalledTimes(1);
  });

  it('ignores late factories, mode changes, force, events, callbacks and timers after device failure', () => {
    const { driver, pipeline, video, documentTarget, factory, callbacks, frames, startNeural } = createHarness();
    startNeural();
    const sample = frames(10, 6)!;
    const tick = callbacks.frame.mock.lastCall![0];
    const configuration = callbacks.configure.mock.lastCall![0];
    const deliverFrame = pipeline.onFrame!;
    const deliverSample = pipeline.onGpuSample!;
    const deliverConfiguration = pipeline.onConfiguration!;
    const selected = pipeline.currentUpscaler;
    driver.fail('device lost');
    expect(driver.snapshot()).toMatchObject({
      running: false, controller: { state: 'failed', reason: 'device lost' }, session: { active: false },
    });
    expect(callbacks.change.mock.lastCall![0]).toMatchObject({ state: 'failed', reason: 'device lost' });
    expect(vi.getTimerCount()).toBe(0);
    const stopped = driver.snapshot();
    const mutations = {
      starts: pipeline.start.mock.calls.length,
      stops: pipeline.stop.mock.calls.length,
      swaps: pipeline.setUpscaler.mock.calls.length,
      invalidations: pipeline.invalidateTiming.mock.calls.length,
      resets: pipeline.resetMeasurements.mock.calls.length,
    };
    for (const callback of Object.values(callbacks)) callback.mockClear();
    const lateFactory = vi.fn(() => createUpscaler(true));

    driver.setNeuralFactory(lateFactory);
    driver.setMode('baseline');
    driver.setMode('neural');
    driver.force(true);
    driver.force(false);
    video.dispatchEvent(new Event('playing'));
    video.dispatchEvent(new Event('loadstart'));
    documentTarget.dispatchEvent(new Event('visibilitychange'));
    deliverFrame(tick);
    deliverSample(sample);
    deliverConfiguration(configuration);
    vi.advanceTimersByTime(10000);
    driver.fail('a later error');

    expect.soft(driver.snapshot()).toEqual(stopped);
    for (const callback of Object.values(callbacks)) expect.soft(callback).not.toHaveBeenCalled();
    expect(lateFactory).not.toHaveBeenCalled();
    expect(factory).toHaveBeenCalledTimes(1);
    expect(pipeline.start).toHaveBeenCalledTimes(mutations.starts);
    expect(pipeline.stop).toHaveBeenCalledTimes(mutations.stops);
    expect(pipeline.setUpscaler).toHaveBeenCalledTimes(mutations.swaps);
    expect(pipeline.invalidateTiming).toHaveBeenCalledTimes(mutations.invalidations);
    expect(pipeline.resetMeasurements).toHaveBeenCalledTimes(mutations.resets);
    expect(pipeline.currentUpscaler).toBe(selected);
    expect(selected.destroy).not.toHaveBeenCalled();
    expect(pipeline.destroy).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('freezes both active clocks while paused and resumes the same neural stage with fresh evidence', () => {
    const { driver, pipeline, video, factory, callbacks, frames, play, startNeural } = createHarness();
    startNeural();
    const pending = frames(40, 6)!;
    const selected = pipeline.currentUpscaler;
    expect(driver.snapshot().controller.state).toBe('stable');
    video.paused = true;
    video.dispatchEvent(new Event('pause'));
    const paused = driver.snapshot();
    expect(paused).toMatchObject({ running: false, controller: { state: 'suspended' }, session: { active: false } });
    callbacks.sample.mockClear();
    pipeline.onGpuSample!(pending);
    vi.advanceTimersByTime(10000);
    expect(driver.snapshot().session).toEqual(paused.session);
    expect(driver.snapshot().controller.activeMs).toBe(paused.controller.activeMs);
    expect(callbacks.sample).not.toHaveBeenCalled();
    expect(pipeline.stop).toHaveBeenCalledTimes(1);

    play();
    expect(driver.snapshot()).toMatchObject({
      running: true,
      controller: { state: 'warmup', samples: 0, activeMs: paused.controller.activeMs },
      session: { active: true, activeMs: paused.session.activeMs, framesRendered: paused.session.framesRendered },
    });
    expect(pipeline.timingGeneration).toBeGreaterThan(paused.controller.generation!);
    pipeline.onGpuSample!({ ...pending, resolvedAt: clock() });
    expect(driver.snapshot().controller.samples).toBe(0);
    frames(40, 6);
    expect(driver.snapshot().controller.state).toBe('stable');
    expect(driver.snapshot().session.activeMs).toBe(paused.session.activeMs + 800);
    expect(pipeline.currentUpscaler).toBe(selected);
    expect(factory).toHaveBeenCalledTimes(1);
    expect(selected.destroy).not.toHaveBeenCalled();
    expect(pipeline.start).toHaveBeenCalledTimes(2);
  });

  it('keeps backward-frame invalidation monotonic when performance.now advances inside the callback', () => {
    const { driver, pipeline, callbacks, frames, startNeural } = createHarness({ mode: 'neural' });
    startNeural();
    frames(40, 6);
    const before = driver.snapshot();
    const selected = pipeline.currentUpscaler;
    const previousMediaTime = callbacks.frame.mock.lastCall![0].mediaTime;
    const performanceNow = vi.spyOn(globalThis.performance, 'now').mockImplementation(() => {
      const now = clock();
      vi.setSystemTime(now + 1);
      return now;
    });

    expect(() => frames(1, undefined, { mediaDelta: -1 })).not.toThrow();

    performanceNow.mockImplementation(clock);
    expect(callbacks.frame.mock.lastCall![0].mediaTime).toBeLessThan(previousMediaTime);
    expect(driver.snapshot()).toMatchObject({
      actualTier: 'neural', running: true,
      controller: { mode: 'neural', state: 'warmup', samples: 0, generation: pipeline.timingGeneration },
      session: { active: true, framesRendered: before.session.framesRendered + 1,
        sourceFpsEstimate: 50, cadence: { observations: 0, cleanBaselineIntervals: 0 } },
    });
    expect(pipeline.timingGeneration).toBeGreaterThan(before.controller.generation!);
    expect(pipeline.currentUpscaler).toBe(selected);
    expect(pipeline.stop).not.toHaveBeenCalled();
  });

  it('counts seeking time as active while preserving mode and flushing cadence evidence', () => {
    const { driver, pipeline, video, frames, startNeural } = createHarness({ mode: 'neural' });
    startNeural();
    frames(40, 6);
    const before = driver.snapshot();
    const selected = pipeline.currentUpscaler;
    expect(before.session.cadence.observations).toBe(30);

    video.seeking = true;
    video.dispatchEvent(new Event('seeking'));

    expect(driver.snapshot()).toMatchObject({
      actualTier: 'neural', running: true,
      controller: { mode: 'neural', state: 'warmup', samples: 0, activeMs: before.controller.activeMs },
      session: { active: true, activeMs: before.session.activeMs,
        framesRendered: before.session.framesRendered, sourceFpsEstimate: 50,
        cadence: { observations: 0, cleanBaselineIntervals: 0 } },
    });
    vi.advanceTimersByTime(1000);
    expect(driver.snapshot()).toMatchObject({
      controller: { mode: 'neural', activeMs: before.controller.activeMs + 1000 },
      session: { active: true, activeMs: before.session.activeMs + 1000,
        framesRendered: before.session.framesRendered, cadence: { observations: 0, cleanBaselineIntervals: 0 } },
    });
    video.seeking = false;
    video.dispatchEvent(new Event('seeked'));
    frames(1);
    expect(driver.snapshot()).toMatchObject({
      controller: { mode: 'neural', activeMs: before.controller.activeMs + 1020 },
      session: { activeMs: before.session.activeMs + 1020, framesRendered: before.session.framesRendered + 1,
        cadence: { observations: 0, cleanBaselineIntervals: 0 } },
    });
    frames(30);
    expect(driver.snapshot().session.cadence).toEqual({ observations: 30, cleanBaselineIntervals: 0 });
    expect(pipeline.currentUpscaler).toBe(selected);
    expect(pipeline.stop).not.toHaveBeenCalled();
    expect(pipeline.start).toHaveBeenCalledTimes(1);
  });

  it('uses the injected visibility callback and freezes fallback retry time while hidden', () => {
    let visible = true;
    const visibility = vi.fn(() => visible);
    const { driver, pipeline, documentTarget, factory, startNeural } = createHarness({ visible: visibility });
    startNeural();
    driver.force(true);
    driver.force(false);
    vi.advanceTimersByTime(200);
    visible = false;
    documentTarget.dispatchEvent(new Event('visibilitychange'));
    const hidden = driver.snapshot();
    const remaining = hidden.controller.nextProbeAtMs! - clock();
    expect(hidden).toMatchObject({ running: false, controller: { state: 'suspended' }, session: { active: false } });
    expect(visibility).toHaveBeenLastCalledWith();
    vi.advanceTimersByTime(10000);
    expect(driver.snapshot().controller.activeMs).toBe(hidden.controller.activeMs);
    expect(driver.snapshot().session).toEqual(hidden.session);
    expect(factory).toHaveBeenCalledTimes(1);

    documentTarget.visibilityState = 'hidden';
    visible = true;
    documentTarget.dispatchEvent(new Event('visibilitychange'));
    expect(driver.snapshot()).toMatchObject({ running: true, controller: { state: 'fallback' } });
    expect(driver.snapshot().controller.nextProbeAtMs).toBe(clock() + remaining);
    vi.advanceTimersByTime(remaining - 1);
    expect(factory).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(101);
    expect(driver.snapshot().controller).toMatchObject({ state: 'probing', probeCount: 1 });
    expect(factory).toHaveBeenCalledTimes(2);
    expect(pipeline.start).toHaveBeenCalledTimes(2);
    expect(pipeline.stop).toHaveBeenCalledTimes(1);
  });

  it('clears source cadence and selected timing evidence and rejects old-generation samples', () => {
    const { driver, pipeline, video, callbacks, configure, frames, startNeural } = createHarness();
    startNeural();
    const oldSample = frames(40, 6)!;
    const before = driver.snapshot();
    expect(before.controller).toMatchObject({ state: 'stable', samples: 30 });
    expect(before.session.cadence.observations).toBe(30);
    video.videoWidth = 1920;
    video.videoHeight = 1080;
    const nextConfiguration = configure(true);
    expect(callbacks.configure).toHaveBeenLastCalledWith(nextConfiguration);
    expect(driver.snapshot()).toMatchObject({
      controller: { state: 'unavailable', width: 1920, height: 1080, fps: 60,
        samples: 0, medianMs: null, generation: nextConfiguration.generation },
      session: { framesRendered: before.session.framesRendered, sourceFpsEstimate: 60,
        cadence: { observations: 0, cleanBaselineIntervals: 0 },
        configurationCount: before.session.configurationCount + 1 },
    });
    pipeline.onGpuSample!({ ...oldSample, resolvedAt: clock() });
    expect(driver.snapshot().controller.samples).toBe(0);
    expect(driver.snapshot().session.gpu.neural.count).toBe(before.session.gpu.neural.count + 1);
    frames(1);
    expect(driver.snapshot().actualTier).toBe('baseline');
    frames(29);
    expect(driver.snapshot().actualTier).toBe('baseline');
    frames(1);
    expect(driver.snapshot().actualTier).toBe('neural');
    frames(1, 6);
    const fresh = driver.snapshot().controller;
    expect(fresh).toMatchObject({ state: 'warmup', samples: 1, medianMs: 6 });
    expect(fresh.generation).toBe(pipeline.timingGeneration);
    pipeline.onGpuSample!({ ...oldSample, sequence: 10000, submittedAt: clock(), resolvedAt: clock() });
    expect(driver.snapshot().controller).toEqual(fresh);
  });

  it('preserves rendered frames across a file-picker measurement reset and a new source load', () => {
    const { driver, pipeline, video, configure, frames, play } = createHarness({ mode: 'baseline' });
    configure();
    play();
    frames(31, undefined, { fps: 30 });
    const before = driver.snapshot();
    expect(before.session).toMatchObject({ framesRendered: 31, decoderFrames: 31, sourceFpsEstimate: 30 });
    const resetSession = vi.spyOn(driver.session, 'reset');
    const resetDriver = vi.spyOn(driver, 'resetMeasurements');

    pipeline.resetMeasurements();
    video.paused = true;
    video.quality = { totalVideoFrames: 0, droppedVideoFrames: 0, corruptedVideoFrames: 0 };
    video.dispatchEvent(new Event('loadstart'));
    configure(true);

    expect(driver.snapshot()).toMatchObject({
      controller: { mode: 'baseline', fps: 60, samples: 0 },
      session: { activeMs: before.session.activeMs, framesRendered: before.session.framesRendered,
        framesPresented: before.session.framesPresented, decoderFrames: before.session.decoderFrames,
        loadGeneration: before.session.loadGeneration, sourceFpsEstimate: 60,
        cadence: { observations: 0, cleanBaselineIntervals: 0 },
        configurationCount: before.session.configurationCount + 1 },
    });
    play();
    frames(2);
    expect(driver.snapshot().session).toMatchObject({
      framesRendered: before.session.framesRendered + 2, framesPresented: before.session.framesPresented + 2,
      decoderFrames: before.session.decoderFrames! + 2, qualityRejectedSamples: 0, loadGeneration: 1,
    });
    expect(pipeline.resetMeasurements).toHaveBeenCalledTimes(1);
    expect(resetDriver).not.toHaveBeenCalled();
    expect(resetSession).not.toHaveBeenCalled();
  });

  it.each(['auto', 'neural'] as const)(
    'preserves learned cadence on reset and tightens the overload policy from neural frames in %s mode', (mode) => {
      const { driver, pipeline, factory, configure, frames, play } = createHarness({ mode });
      configure();
      play();
      driver.setNeuralFactory(factory);
      frames(31, undefined, { fps: 30 });
      frames(40, 20, { fps: 30 });
      const before = driver.snapshot();
      const selected = pipeline.currentUpscaler;
      expect(before).toMatchObject({
        actualTier: 'neural', controller: { mode, state: 'stable', fps: 30, failMs: 24, medianMs: 20 },
        session: { sourceFpsEstimate: 30 },
      });

      driver.resetMeasurements();

      expect(driver.snapshot()).toMatchObject({
        actualTier: 'neural', controller: { mode, state: 'stable', fps: 30, failMs: 24, samples: 0, medianMs: null },
        session: { activeMs: 0, framesRendered: 0, sourceFpsEstimate: 30,
          cadence: { observations: 0, cleanBaselineIntervals: 0 } },
      });
      frames(30, 6);
      expect(driver.snapshot()).toMatchObject({
        controller: { fps: 30, failMs: 24 },
        session: { sourceFpsEstimate: 30, cadence: { observations: 29, cleanBaselineIntervals: 0 } },
      });
      frames(1, 6);
      expect(driver.snapshot()).toMatchObject({
        actualTier: 'neural', controller: { mode, state: 'warmup', fps: 50, failMs: 18, recoverMs: 15.6 },
        session: { sourceFpsEstimate: 50, cadence: { observations: 30, cleanBaselineIntervals: 0 } },
      });
      expect(pipeline.currentUpscaler).toBe(selected);
      expect(factory).toHaveBeenCalledTimes(1);
      frames(40, 20);
      expect(driver.snapshot()).toMatchObject({
        actualTier: 'baseline', controller: { mode, state: 'fallback', fps: 50, failMs: 18,
          reason: 'neural median exceeds failure threshold', probeCount: 0 },
      });
      expect(selected.destroy).toHaveBeenCalledTimes(1);
    },
  );

  it('resets session totals and selected evidence but preserves mode, transitions and probe counts', () => {
    const { driver, pipeline, video, frames, startNeural } = createHarness({ mode: 'neural' });
    startNeural();
    frames(40, 6);
    driver.force(true);
    driver.force(false);
    vi.advanceTimersByTime(2100);
    frames(40, 6);
    const before = driver.snapshot();
    expect(before.controller).toMatchObject({ mode: 'neural', state: 'stable', probeCount: 1 });
    expect(before.session.framesRendered).toBe(111);
    const selected = pipeline.currentUpscaler;
    const resets = pipeline.resetMeasurements.mock.calls.length;

    driver.resetMeasurements();

    const reset = driver.snapshot();
    expect(reset.controller).toEqual({ ...before.controller, generation: pipeline.timingGeneration,
      samples: 0, medianMs: null, p90Ms: null });
    expect(reset.session).toMatchObject({
      active: true, activeMs: 0, framesRendered: 0, framesPresented: 0, framesSkipped: 0,
      decoderFrames: 0, decoderDrops: 0, decoderCorrupted: 0, qualitySamples: 0,
      generation: null, source: null, target: null, configurationCount: 0,
      cadence: { observations: 0, cleanBaselineIntervals: 0 },
    });
    for (const summary of [reset.session.gpu.neural, reset.session.gpu.baseline,
      reset.session.callbackLatency, reset.session.configuration]) {
      expect(summary).toMatchObject({ count: 0, sum: null, mean: null, max: null });
    }
    expect(pipeline.resetMeasurements).toHaveBeenCalledTimes(resets + 1);
    expect(pipeline.timingGeneration).toBeGreaterThan(before.controller.generation!);
    expect(pipeline.currentUpscaler).toBe(selected);
    video.quality.totalVideoFrames += 2;
    video.quality.droppedVideoFrames++;
    frames(1, 6);
    expect(driver.snapshot().session).toMatchObject({
      activeMs: 20, framesRendered: 1, decoderFrames: 3, decoderDrops: 1, gpu: { neural: { count: 1 } },
    });
  });

  it('preserves session totals across stage swaps while destroying each replaced stage exactly once', () => {
    const { driver, pipeline, baseline, factory, callbacks, frames, startNeural } = createHarness();
    startNeural();
    frames(40, 6);
    const firstNeural = pipeline.currentUpscaler;
    const before = driver.snapshot().session;
    driver.setMode('baseline');
    const manualBaseline = pipeline.currentUpscaler;
    expect(driver.snapshot().session).toEqual(before);
    expect(firstNeural.destroy).toHaveBeenCalledTimes(1);
    frames(31);
    expect(driver.snapshot().session).toMatchObject({
      framesRendered: before.framesRendered + 31, decoderFrames: before.decoderFrames! + 31,
      configurationCount: before.configurationCount + 1, gpu: { neural: { count: before.gpu.neural.count } },
    });
    const manual = driver.snapshot().session;
    driver.setMode('neural');
    expect(driver.snapshot().session).toEqual(manual);
    expect(pipeline.currentUpscaler).not.toBe(firstNeural);
    frames(40, 6);
    expect(driver.snapshot().session).toMatchObject({
      framesRendered: 142, framesPresented: 142, decoderFrames: 142,
      activeMs: 2840, configurationCount: 4,
      gpu: { neural: { count: 80, sum: 480 }, baseline: { count: 0 } },
    });
    expect(callbacks.frame).toHaveBeenCalledTimes(142);
    expect(callbacks.sample).toHaveBeenCalledTimes(80);
    expect(callbacks.configure).toHaveBeenCalledTimes(4);
    expect(factory).toHaveBeenCalledTimes(2);
    expect(pipeline.setUpscaler).toHaveBeenCalledTimes(3);
    expect(baseline.destroy).toHaveBeenCalledTimes(1);
    expect(manualBaseline.destroy).toHaveBeenCalledTimes(1);
    expect(pipeline.currentUpscaler.destroy).not.toHaveBeenCalled();
    expect(pipeline.destroy).not.toHaveBeenCalled();
  });

  it('stays unavailable without GPU timestamps even with clean cadence and explicit neural intent', () => {
    const { driver, pipeline, factory, configure, frames, play } = createHarness({ timestamps: false });
    configure();
    play();
    driver.setNeuralFactory(factory);
    driver.setMode('neural');
    frames(100);
    driver.force(true);
    driver.force(false);
    vi.advanceTimersByTime(10000);
    expect(driver.snapshot()).toMatchObject({
      actualTier: 'baseline', running: true, controller: { state: 'unavailable', mode: 'neural', probeCount: 0 },
      session: { framesRendered: 100, gpu: { neural: { count: 0 } } },
    });
    expect(factory).not.toHaveBeenCalled();
    expect(pipeline.setUpscaler).not.toHaveBeenCalled();
  });
});