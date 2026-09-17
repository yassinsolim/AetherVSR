import { acquireGpu, watchDeviceFailures } from '../../src/core/gpu/device.js';
import { VideoPipeline } from '../../src/core/pipeline.js';
import { RuntimeDriver } from '../../src/runtime.js';
import { BaselineScaler } from '../../src/core/upscale/baseline-scaler.js';
import { NeuralUpscaler, NEURAL_OPTIONAL_FEATURES } from '../../src/core/upscale/neural-upscaler.js';
import { loadModel } from '../../src/core/neural/model.js';
import { FrameImporter } from '../../src/core/acquisition/frame-importer.js';

const video = document.querySelector<HTMLVideoElement>('video')!;
const canvas = document.querySelector<HTMLCanvasElement>('canvas')!;
const file = document.querySelector<HTMLInputElement>('input[type=file]')!;
const output = document.querySelector<HTMLOutputElement>('output')!;
let objectUrl: string | null = null;
let activeCleanup: (() => void) | null = null;

file.addEventListener('change', () => {
  video.pause(); activeCleanup?.();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = file.files?.[0] ? URL.createObjectURL(file.files[0]) : null;
  if (objectUrl) { video.src = objectUrl; video.load(); }
});

async function smoke(forceCopy = false) {
  if (!objectUrl || video.readyState < 2) throw new Error('Select a decoded local video first');
  const errors: string[] = [];
  const gpu = await acquireGpu({ optionalFeatures: NEURAL_OPTIONAL_FEATURES });
  const unwatch = watchDeviceFailures(gpu.device, message => errors.push(message));
  for (const filter of ['internal', 'out-of-memory', 'validation'] as const) gpu.device.pushErrorScope(filter);
  let pendingScopes = 3;
  const drainErrors = async () => {
    while (pendingScopes > 0) {
      pendingScopes--;
      const error = await gpu.device.popErrorScope();
      if (error) errors.push(error.message);
    }
  };
  let pipeline: VideoPipeline | null = null, driver: RuntimeDriver | null = null;
  const cleanup = () => { video.pause(); driver?.destroy(); pipeline?.destroy(); unwatch(); gpu.device.destroy(); activeCleanup = null; };
  activeCleanup = cleanup;
  try {
    if (!gpu.capabilities.timestampQuery || !gpu.capabilities.externalTexture || gpu.adapterReport.fallbackAdapter) throw new Error('Required native WebGPU capabilities unavailable');
    const model = await loadModel('aethervsr://app/models/production.json');
    const neural = new NeuralUpscaler(model);
    pipeline = new VideoPipeline(gpu, video, canvas, new BaselineScaler('catmull-rom'), { forceCopyImport: forceCopy });
    driver = new RuntimeDriver(pipeline, video, true, 'neural');
    driver.setNeuralFactory(() => neural);
    const samples: { ms: number; generation: number; sequence: number }[] = [];
    driver.onSample = sample => { if (sample.neural) samples.push({ ms: sample.ms, generation: sample.generation, sequence: sample.sequence }); };
    await video.play(); driver.syncActive();
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => { clearInterval(poll); reject(new Error('No actual neural timestamp submissions within 12 seconds')); }, 12000);
      const poll = setInterval(() => {
        if (errors.length || pipeline?.error) { clearTimeout(timeout); clearInterval(poll); reject(new Error(errors.join('; ') || String(pipeline?.error))); }
        else if (samples.length >= 8 && pipeline?.currentUpscaler.neural) { clearTimeout(timeout); clearInterval(poll); resolve(); }
      }, 50);
    });
    video.pause(); driver.syncActive();
    await pipeline.drainTimings(); await gpu.device.queue.onSubmittedWorkDone();
    const extent = { width: video.videoWidth, height: video.videoHeight };
    const config = canvas.getContext('webgpu')!.getConfiguration();
    if (!config) throw new Error('No canvas GPU configuration');
    canvas.getContext('webgpu')!.configure({ ...config, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const frameTexture = gpu.device.createTexture({ size: [canvas.width, canvas.height], format: config.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const bytesPerRow = Math.ceil(canvas.width * 4 / 256) * 256;
    const readback = gpu.device.createBuffer({ size: bytesPerRow * canvas.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const importer = new FrameImporter(gpu.device, video, !forceCopy);
    try {
      const current = pipeline.currentUpscaler;
      importer.configure(extent);
      current.configure({ device: gpu.device, source: extent, target: { width: canvas.width, height: canvas.height },
        targetFormat: config.format, sourceKind: importer.kind, ...(importer.sampledView ? { sampledSourceView: importer.sampledView } : {}) });
      const encoder = gpu.device.createCommandEncoder();
      current.encode({ encoder, frame: importer.acquire(extent), target: frameTexture.createView(), timing: null });
      encoder.copyTextureToBuffer({ texture: frameTexture }, { buffer: readback, bytesPerRow }, [canvas.width, canvas.height]);
      gpu.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const pixels = new Uint8Array(readback.getMappedRange());
      let minimum = 255, maximum = 0;
      for (let row = 0; row < canvas.height; row++) for (let column = 0; column < canvas.width * 4; column++) {
        if (column % 4 === 3) continue;
        const value = pixels[row * bytesPerRow + column]!; minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
      }
      readback.unmap();
      await drainErrors();
      if (maximum - minimum < 32 || errors.length) throw new Error(`Invalid neural output: ${errors.join('; ')}`);
      return { outcome: 'PASS', forceCopy, input: extent, output: { width: canvas.width, height: canvas.height, minimum, maximum },
        capabilities: gpu.capabilities, adapter: gpu.adapterReport, upscaler: current.id, precision: neural.resolvedPrecision,
        gpuSamples: samples, runtime: driver.snapshot(), errors, audioAuthority: 'same HTMLVideoElement', userAgent: navigator.userAgent };
    } finally { importer.destroy(); readback.destroy(); frameTexture.destroy(); }
  } finally { try { await drainErrors(); } finally { cleanup(); } }
}

Object.assign(window, { m11Smoke: { run: smoke } });
document.querySelector('button')!.addEventListener('click', () => {
  void smoke().then(result => { output.value = JSON.stringify(result); }, error => { output.value = String(error); });
});
window.addEventListener('pagehide', () => { activeCleanup?.(); video.pause(); if (objectUrl) URL.revokeObjectURL(objectUrl); });