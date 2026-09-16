import { acquireGpu, watchDeviceFailures, type GpuContext } from '../../src/core/gpu/device.js';
import { VideoPipeline } from '../../src/core/pipeline.js';
import { BaselineScaler } from '../../src/core/upscale/baseline-scaler.js';
import { NeuralUpscaler, NEURAL_OPTIONAL_FEATURES } from '../../src/core/upscale/neural-upscaler.js';
import { loadModel } from '../../src/core/neural/model.js';
import { RuntimeDriver } from '../../src/runtime.js';
import type { RuntimeMode } from '../../src/core/upscale/runtime-controller.js';

interface DocumentPip {
  requestWindow(options: { width: number; height: number }): Promise<Window>;
}
interface ResearchGlobal {
  aethervsrRuntime?: { driver: RuntimeDriver };
  documentPictureInPicture?: DocumentPip;
  m1010?: ReturnType<typeof createResearch>;
}
const global = globalThis as unknown as ResearchGlobal;
const video = document.querySelector<HTMLVideoElement>('#source')!;
const canvas = document.querySelector<HTMLCanvasElement>('#output')!;
const stage = document.querySelector<HTMLElement>('#stage')!;
const controls = document.querySelector<HTMLElement>('#controls')!;
const status = document.querySelector<HTMLOutputElement>('#status')!;
const params = new URLSearchParams(location.search);
let gpu: GpuContext | null = null, pipeline: VideoPipeline | null = null, driver: RuntimeDriver | null = null;
let starting: Promise<void> | null = null, unwatch = () => {}, destroyed = false, failure: string | null = null;
let pipWindow: Window | null = null, objectUrl: string | null = null;
let pipGeneration = 0;
let mode: RuntimeMode = params.get('mode') === 'baseline' ? 'baseline' : params.get('mode') === 'neural' ? 'neural' : 'auto';
const events: { at: number; type: string; detail: unknown }[] = [];
const samples: { at: number; mediaTime: number; presented: number; width: number; height: number }[] = [];
const transitions: { at: number; tier: string; state: string; reason: string }[] = [];
let frame: number | null = null;
const record = (type: string, detail: unknown = null) => { events.push({ at: performance.now(), type, detail }); if (events.length > 1000) events.shift(); };
const fail = (error: unknown) => { failure ??= String(error); status.textContent = failure; driver?.fail(failure); pipeline?.stop(); record('error', failure); };

async function initialize() {
  if (destroyed) throw new Error('Player disposed');
  if (driver) return;
  gpu = await acquireGpu({ optionalFeatures: NEURAL_OPTIONAL_FEATURES });
  if (destroyed) { gpu.device.destroy(); gpu = null; return; }
  pipeline = new VideoPipeline(gpu, video, canvas, new BaselineScaler('catmull-rom'), { forceCopyImport: params.get('import') === 'copy' });
  driver = new RuntimeDriver(pipeline, video, gpu.capabilities.timestampQuery, mode);
  global.aethervsrRuntime = { driver };
  unwatch = watchDeviceFailures(gpu.device, fail);
  driver.onChange = state => {
    transitions.push({ at: performance.now(), tier: state.tier, state: state.state, reason: state.reason });
    if (transitions.length > 1000) transitions.shift();
    if (!failure) status.textContent = `${state.tier}: ${state.reason}`;
  };
  const model = await loadModel(new URL('models/production.json', location.href).href);
  if (destroyed) return;
  driver.setNeuralFactory(() => new NeuralUpscaler(model, { passDiagnostics: true }));
  driver.syncActive();
  record('initialized', { origin: location.origin, secure: isSecureContext, mode, copy: params.get('import') === 'copy', capabilities: gpu.capabilities });
}

async function play() {
  if (failure || destroyed) throw new Error(failure ?? 'Player disposed');
  await video.play();
  starting ??= initialize().catch(error => { fail(error); throw error; });
  await starting;
  driver?.syncActive();
}

function destroy() {
  if (destroyed) return snapshot();
  destroyed = true;
  pipGeneration++;
  const errors: string[] = [];
  for (const cleanup of [() => video.pause(), () => { if (frame !== null) video.cancelVideoFrameCallback(frame); frame = null; },
    unwatch, () => { driver?.destroy(); driver = null; }, () => { pipeline?.destroy(); pipeline = null; },
    () => { gpu?.device.destroy(); gpu = null; }, () => { pipWindow?.close(); pipWindow = null; },
    () => { if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = null; }]) {
    try { cleanup(); } catch (error) { errors.push(String(error)); }
  }
  delete global.aethervsrRuntime;
  record('destroyed', { errors });
  if (errors.length) failure ??= errors.join('; ');
  return snapshot();
}

function snapshot() {
  return { origin: location.origin, secure: isSecureContext, topLevel: window === window.top, destroyed, failure,
    source: { url: video.currentSrc, width: video.videoWidth, height: video.videoHeight, ready: video.readyState,
      mediaTime: video.currentTime, paused: video.paused, muted: video.muted, volume: video.volume, rate: video.playbackRate },
    canvas: { width: canvas.width, height: canvas.height, ownerOrigin: canvas.ownerDocument.location?.origin },
    runtime: driver?.snapshot() ?? null, capabilities: gpu?.capabilities ?? null,
    events: events.slice(), samples: samples.slice(), transitions: transitions.slice(),
    resources: { device: Number(gpu !== null), pipeline: Number(pipeline !== null), driver: Number(driver !== null),
      frameCallback: Number(frame !== null), pip: Number(pipWindow !== null), objectUrl: Number(objectUrl !== null) } };
}

function createResearch() {
  return { snapshot, destroy, play, getVideo: () => video,
    pause: () => { video.pause(); driver?.syncActive(); },
    setSource: async (url: string, cors: 'anonymous' | 'use-credentials' | null = null) => {
      video.pause(); driver?.syncActive();
      if (cors) video.crossOrigin = cors; else video.removeAttribute('crossorigin');
      video.srcObject = null; video.src = url; video.load(); record('source-change');
      return Promise.resolve();
    },
    setBlob: async (bytes: number[], type = 'video/mp4') => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type }));
      video.removeAttribute('crossorigin'); video.srcObject = null; video.src = objectUrl; video.load();
      return Promise.resolve();
    },
    setStream: (stream: MediaStream) => { video.pause(); driver?.syncActive(); video.removeAttribute('src'); video.srcObject = stream; record('stream-set', stream.getTracks().map(track => ({ kind: track.kind, settings: track.getSettings() }))); },
    clearObservations: () => { events.length = samples.length = transitions.length = 0; driver?.resetMeasurements(); },
  };
}

global.m1010 = createResearch();
for (const type of ['play', 'pause', 'seeking', 'seeked', 'loadeddata', 'resize', 'ratechange', 'volumechange', 'ended', 'emptied']) video.addEventListener(type, () => record(type));
video.addEventListener('error', () => { if (video.error) record('media-error', { code: video.error.code, message: video.error.message }); });
const observe = (at: number, metadata: VideoFrameCallbackMetadata) => {
  samples.push({ at, mediaTime: metadata.mediaTime, presented: metadata.presentedFrames, width: metadata.width, height: metadata.height });
  if (samples.length > 10000) samples.shift();
  frame = destroyed ? null : video.requestVideoFrameCallback(observe);
};
frame = video.requestVideoFrameCallback(observe);
document.querySelector('#play')!.addEventListener('click', () => { void play().catch(fail); });
document.querySelector('#pause')!.addEventListener('click', () => { video.pause(); driver?.syncActive(); });
document.querySelector<HTMLInputElement>('#seek')!.addEventListener('input', event => { video.currentTime = Number((event.target as HTMLInputElement).value); });
document.querySelector<HTMLInputElement>('#volume')!.addEventListener('input', event => { video.volume = Number((event.target as HTMLInputElement).value); });
document.querySelector<HTMLInputElement>('#mute')!.addEventListener('change', event => { video.muted = (event.target as HTMLInputElement).checked; });
const modeSelect = document.querySelector<HTMLSelectElement>('#mode')!; modeSelect.value = mode;
modeSelect.addEventListener('change', () => { mode = modeSelect.value as RuntimeMode; driver?.setMode(mode); });
document.querySelector('#fullscreen')!.addEventListener('click', () => { void (document.fullscreenElement ? document.exitFullscreen() : stage.requestFullscreen()).catch(fail); });
document.querySelector('#pip')!.addEventListener('click', () => {
  if (destroyed) return;
  if (!global.documentPictureInPicture) { record('pip-unavailable'); return; }
  const ticket = ++pipGeneration;
  void global.documentPictureInPicture.requestWindow({ width: 720, height: 480 }).then(window => {
    if (destroyed || ticket !== pipGeneration) { window.close(); record('pip-stale'); return; }
    pipWindow = window;
    const style = window.document.createElement('link'); style.rel = 'stylesheet'; style.href = new URL('player.css', location.href).href;
    window.document.head.append(style); window.document.body.append(stage, controls);
    record('pip-open', { origin: window.document.location.origin, canvasOwner: canvas.ownerDocument.location.origin });
    window.addEventListener('pagehide', () => {
      if (pipWindow !== window) return;
      if (!destroyed) { document.body.insertBefore(stage, video); document.body.append(controls); }
      pipWindow = null; record('pip-close');
    });
  }).catch(error => record('pip-error', String(error)));
});
document.querySelector('#return')!.addEventListener('click', () => { void chrome.runtime.sendMessage({ type: 'research.return' }); });
document.querySelector('#close')!.addEventListener('click', () => { destroy(); window.close(); });
window.addEventListener('pagehide', destroy, { once: true });