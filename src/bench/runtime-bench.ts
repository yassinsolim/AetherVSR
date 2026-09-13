import { acquireGpu, watchDeviceFailures } from '../core/gpu/device.js';
import { VideoPipeline, type PipelineConfiguration, type PipelineGpuSample } from '../core/pipeline.js';
import { loadModel } from '../core/neural/model.js';
import { NeuralUpscaler, NEURAL_OPTIONAL_FEATURES } from '../core/upscale/neural-upscaler.js';
import { BaselineScaler, UPSCALER_OPTIONAL_FEATURES } from '../core/upscale/baseline-scaler.js';
import { LoadedUpscaler, type RuntimeLoad } from './runtime-load.js';

export async function startCalibration(clip: string, neural = true) {
  if (!import.meta.env.DEV) throw new Error('Calibration is development-only');
  const gpu = await acquireGpu({ optionalFeatures: [...NEURAL_OPTIONAL_FEATURES, ...UPSCALER_OPTIONAL_FEATURES] });
  if (!gpu.capabilities.timestampQuery || gpu.adapterReport.fallbackAdapter) throw new Error('Hardware timestamps required');
  const model = await loadModel('/models/aethersr-c16d2.json');
  const video = document.createElement('video');
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.style.cssText = 'position:absolute;width:2px;height:2px;opacity:0.01';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;height:auto;max-height:95vh;object-fit:contain';
  document.body.replaceChildren(video, canvas);
  const load: RuntimeLoad = { passes: 0, frames: 0 };
  const stage = new NeuralUpscaler(model);
  const pipeline = new VideoPipeline(gpu, video, canvas,
    neural ? new LoadedUpscaler(stage, load) : new BaselineScaler('catmull-rom'));
  const samples: PipelineGpuSample[] = [];
  const frames: number[][] = [];
  const configurations: PipelineConfiguration[] = [];
  const refresh: number[] = [];
  const events: { at: number; event: string }[] = [];
  const errors: string[] = [];
  let started = performance.now();
  let refreshHandle = 0;
  const refreshTick = (now: number): void => {
    if (refresh.length < 240) refresh.push(now);
    refreshHandle = requestAnimationFrame(refreshTick);
  };
  refreshHandle = requestAnimationFrame(refreshTick);
  pipeline.onGpuSample = (sample) => samples.push(sample);
  pipeline.onFrame = (tick) => frames.push([tick.now, tick.mediaTime, tick.presentedDelta,
    tick.now - tick.presentationTime, tick.expectedDisplayTime - tick.now]);
  pipeline.onConfiguration = (configuration) => configurations.push(configuration);
  const unwatch = watchDeviceFailures(gpu.device, (message) => { errors.push(message); pipeline.stop(); });
  const visibility = (): void => { events.push({ at: performance.now(), event: document.visibilityState }); };
  const focus = (): void => { events.push({ at: performance.now(), event: document.hasFocus() ? 'focus' : 'blur' }); };
  document.addEventListener('visibilitychange', visibility);
  window.addEventListener('focus', focus);
  window.addEventListener('blur', focus);
  visibility();
  focus();
  video.src = clip;
  await new Promise<void>((resolve, reject) => {
    video.addEventListener('loadeddata', () => resolve(), { once: true });
    video.addEventListener('error', () => reject(new Error(video.error?.message)), { once: true });
    video.load();
  });
  await video.play();
  pipeline.start();
  return {
    async finish() {
      const ended = performance.now();
      const stats = pipeline.stats(ended);
      pipeline.stop();
      await pipeline.drainTimings();
      return { ...this.snapshot(), ended, stats };
    },
    async reset() {
      pipeline.stop();
      await pipeline.drainTimings();
      pipeline.resetMeasurements();
      samples.length = 0;
      frames.length = 0;
      events.length = 0;
      visibility();
      focus();
      started = performance.now();
      pipeline.start();
    },
    load(passes: number, count = Number.POSITIVE_INFINITY) {
      if (!Number.isInteger(passes) || passes < 0 || passes > 64) throw new Error('Load passes out of range');
      load.passes = passes;
      load.frames = count;
      events.push({ at: performance.now(), event: `load:${passes}:${count}` });
    },
    snapshot() {
      return { started, ended: performance.now(), samples, frames, configurations, refresh, events, errors,
        stats: pipeline.stats(performance.now()), memory: neural ? stage.memoryReport : null,
        precision: neural ? stage.resolvedPrecision : null, source: clip,
        environment: { userAgent: navigator.userAgent, adapter: gpu.adapterReport,
          timestampQuery: gpu.capabilities.timestampQuery, visibility: document.visibilityState,
          focus: document.hasFocus(), screen: { width: screen.width, height: screen.height } } };
    },
    stop() {
      video.pause();
      pipeline.destroy();
      cancelAnimationFrame(refreshHandle);
      document.removeEventListener('visibilitychange', visibility);
      window.removeEventListener('focus', focus);
      window.removeEventListener('blur', focus);
      unwatch();
      gpu.device.destroy();
    },
  };
}