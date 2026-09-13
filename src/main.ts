import {
  acquireGpu,
  describeAdapter,
  watchDeviceFailures,
  type GpuContext,
} from './core/gpu/device.js';
import { VideoPipeline, type PipelineStats } from './core/pipeline.js';
import { BaselineScaler, UPSCALER_OPTIONAL_FEATURES } from './core/upscale/baseline-scaler.js';
import { NeuralUpscaler, NEURAL_OPTIONAL_FEATURES } from './core/upscale/neural-upscaler.js';
import type { RuntimeMode, RuntimeSnapshot } from './core/upscale/runtime-controller.js';
import { loadModel } from './core/neural/model.js';
import { RuntimeDriver } from './runtime.js';
import type { RuntimeLoad } from './bench/runtime-load.js';
import { DiagnosticOverlay, OVERLAY_INTERVAL_MS } from './ui/overlay.js';

/** Clip shipped with the repo so benchmarks are reproducible. See tools/. */
const DEFAULT_MODEL = '/models/aethersr-c16d2.json';
const DEFAULT_CLIP = '/media/aethervsr-testclip-720p30-vp9.webm';

function requireElement<T extends Element>(id: string, ctor: abstract new () => T): T {
  const el = document.getElementById(id);
  if (!(el instanceof ctor)) throw new Error(`#${id} is missing or not a ${ctor.name}`);
  return el;
}

const video = requireElement('source', HTMLVideoElement);
const canvas = requireElement('output', HTMLCanvasElement);
const status = requireElement('status', HTMLParagraphElement);
const stage = requireElement('stage', HTMLElement);
const fileInput = requireElement('file', HTMLInputElement);
const upscalerSelect = requireElement('upscaler', HTMLSelectElement);
const runtimeState = requireElement('runtime-state', HTMLElement);
const resetButton = requireElement('reset', HTMLButtonElement);
const exportButton = requireElement('export', HTMLButtonElement);

/**
 * True once the GPU device has failed unrecoverably.
 *
 * A lost device never comes back, so after one the harness must neither resume
 * submitting nor let a cheerful "playing ..." overwrite the explanation.
 */
let gpuFatal = false;

function setStatus(text: string, level: 'info' | 'warn' | 'error' = 'info'): void {
  // A fatal GPU failure is the last thing worth saying; nothing may bury it.
  if (gpuFatal && level !== 'error') return;
  status.textContent = text;
  status.dataset['level'] = level;
}

/**
 * Renders an unknown thrown value as something a person can act on.
 *
 * Interpolating a rejection directly yields `[object Object]` for anything
 * that is not an `Error`, which turns a diagnostic surface into a dead end.
 */
function describeError(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  try {
    return JSON.stringify(err) ?? 'unknown error';
  } catch {
    return 'unknown error';
  }
}

/**
 * Loads a clip and starts playback.
 *
 * `play()` must not be called in the same turn as the `src` assignment: the
 * pending load aborts the play request and the promise rejects with a
 * misleading `AbortError`. Wait for the element to signal it has data first,
 * and surface load and playback failures as distinct errors.
 */
async function loadClip(url: string): Promise<void> {
  video.src = url;
  video.load();
  await new Promise<void>((resolve, reject) => {
    const onReady = (): void => {
      cleanup();
      resolve();
    };
    const onError = (): void => {
      cleanup();
      reject(new Error(`could not load ${url}: ${video.error?.message ?? 'unknown media error'}`));
    };
    const cleanup = (): void => {
      video.removeEventListener('loadeddata', onReady);
      video.removeEventListener('error', onError);
    };
    video.addEventListener('loadeddata', onReady, { once: true });
    video.addEventListener('error', onError, { once: true });
  });
  await video.play();
}

function main(gpu: GpuContext): void {
  const overlay = new DiagnosticOverlay(gpu);
  stage.append(overlay.element);

  const params = new URLSearchParams(location.search);
  const requestedMode = params.get('mode');
  const initialMode: RuntimeMode =
    requestedMode === 'auto' || requestedMode === 'neural' || requestedMode === 'baseline'
      ? requestedMode
      : params.get('upscaler') === 'neural'
        ? 'neural'
        : params.get('filter') === 'bilinear'
          ? 'baseline'
          : 'auto';
  upscalerSelect.value = initialMode;
  const forceCopyImport = params.get('import') === 'copy';

  // Relative paths only: this is a dev harness, not a URL loader.
  const clipParam = params.get('clip');
  const clip = clipParam !== null && clipParam.startsWith('/') ? clipParam : DEFAULT_CLIP;

  const pipeline = new VideoPipeline(gpu, video, canvas, new BaselineScaler('catmull-rom'), {
    forceCopyImport,
  });
  const driver = new RuntimeDriver(pipeline, video, gpu.capabilities.timestampQuery, initialMode);

  // A discarded submission or a lost device never throws in the frame loop, so
  // without this the canvas can freeze while the frame counter keeps climbing.
  watchDeviceFailures(gpu.device, (message) => {
    gpuFatal = true;
    driver.fail(message);
    upscalerSelect.disabled = true;
    setStatus(`${message} — reload the page to recover`, 'error');
  });

  // What is actually playing, for the benchmark record. Not the startup clip:
  // the file picker replaces it.
  let activeClip = clip;
  let activeObjectUrl: string | null = null;

  let neuralInstance: NeuralUpscaler | null = null;
  const diagnosticLoad: RuntimeLoad = { passes: 0, frames: 0 };
  const loadEnabled = import.meta.env.DEV && params.get('runtime-bench') === '1';
  const neural = (): NeuralUpscaler | null =>
    pipeline.currentUpscaler.neural ? neuralInstance : null;
  let modelState: 'pending' | 'ready' | 'unavailable' = 'pending';
  let modelError = '';
  let lastRuntimeReason = '';

  function updateRuntime(state: RuntimeSnapshot): void {
    upscalerSelect.value = state.mode;
    const label = `${state.mode} · ${state.tier} · ${state.state}`;
    if (runtimeState.textContent !== label) runtimeState.textContent = label;
    if (gpuFatal) return;
    if (state.state === 'failed') {
      gpuFatal = true;
      upscalerSelect.disabled = true;
      setStatus(`${state.reason} — reload the page to recover`, 'error');
      return;
    }
    const reason = modelState === 'unavailable'
      ? `neural model unavailable (${modelError}); using Catmull-Rom baseline`
      : state.mode !== 'baseline' && modelState === 'pending'
        ? 'neural model loading; using Catmull-Rom baseline'
        : state.state === 'unavailable' && gpu.capabilities.timestampQuery
          ? 'collecting source cadence; using Catmull-Rom baseline'
          : state.reason;
    if (reason === lastRuntimeReason) return;
    lastRuntimeReason = reason;
    if (!video.error) {
      setStatus(reason, modelState === 'unavailable' || state.state === 'fallback' ||
        (state.mode !== 'baseline' && !gpu.capabilities.timestampQuery) ? 'warn' : 'info');
    }
  }

  driver.onChange = updateRuntime;
  updateRuntime(driver.snapshot().controller);

  void loadModel(DEFAULT_MODEL)
    .then(async (model) => {
      const wrapper = loadEnabled ? (await import('./bench/runtime-load.js')).LoadedUpscaler : null;
      modelState = 'ready';
      driver.setNeuralFactory(() => {
        neuralInstance = new NeuralUpscaler(model, { passDiagnostics: !loadEnabled });
        return wrapper ? new wrapper(neuralInstance, diagnosticLoad) : neuralInstance;
      });
    })
    .catch((err: unknown) => {
      modelState = 'unavailable';
      modelError = describeError(err);
      console.warn('neural model unavailable:', err);
      updateRuntime(driver.snapshot().controller);
    });

  upscalerSelect.addEventListener('change', () => {
    const mode = upscalerSelect.value;
    if (mode === 'auto' || mode === 'neural' || mode === 'baseline') {
      driver.setMode(mode);
    }
  });

  function budgetSnapshot(runtime: ReturnType<RuntimeDriver['snapshot']>) {
    return {
      state: runtime.controller.tier === 'neural' ? 'neural' : 'fallback',
      forced: runtime.controller.forced,
      probing: runtime.controller.state === 'probing',
      lastReason: runtime.controller.reason,
      lastMedianMs: runtime.controller.medianMs,
    };
  }

  const diagnostics = window as unknown as Record<string, unknown>;
  diagnostics['aethervsrNeuralStage'] = () => {
    const instance = neural();
    const runtime = driver.snapshot();
    return {
      id: instance?.id ?? null,
      timing: instance?.stageTiming ?? null,
      memory: instance?.memoryReport ?? null,
      runtime,
      budget: budgetSnapshot(runtime),
    };
  };

  if (import.meta.env.DEV) {
    diagnostics['aethervsrForceOverBudget'] = (on: boolean) => {
      driver.force(on);
      const budget = budgetSnapshot(driver.snapshot());
      return { forced: budget.forced, state: budget.state };
    };
    diagnostics['aethervsrRuntime'] = {
      snapshot: () => driver.snapshot(),
      reset: () => driver.resetMeasurements(),
      setMode: (mode: RuntimeMode) => driver.setMode(mode),
      force: (on: boolean) => driver.force(on),
      load: (passes: number, frames = Number.POSITIVE_INFINITY, every = 1) => {
        if (!loadEnabled || !Number.isInteger(passes) || passes < 0 || passes > 8 ||
            !Number.isInteger(every) || every < 1) throw new Error('Diagnostic load is unavailable or invalid');
        diagnosticLoad.passes = passes;
        diagnosticLoad.frames = frames;
        diagnosticLoad.every = every;
      },
      loseDevice: () => gpu.device.destroy(),
      pipeline,
      driver,
      video,
      neural,
    };
  }

  resetButton.addEventListener('click', () => driver.resetMeasurements());

  exportButton.addEventListener('click', () => {
    const payload = benchmarkRecord(
      gpu,
      pipeline.stats(performance.now()),
      activeClip,
      neural()?.resolvedPrecision ?? null,
      driver.snapshot(),
    );
    const json = JSON.stringify(payload, null, 2);
    void navigator.clipboard
      .writeText(json)
      .then(() => setStatus('benchmark JSON copied to clipboard'))
      .catch(() => setStatus('clipboard denied; benchmark JSON written to console', 'warn'));
    console.info(json);
  });

  // Guards against a slow load being overtaken by a newer selection.
  let loadGeneration = 0;

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const generation = ++loadGeneration;
    const previous = activeObjectUrl;
    const candidate = URL.createObjectURL(file);
    pipeline.resetMeasurements();

    void loadClip(candidate).then(
      () => {
        if (generation !== loadGeneration) {
          URL.revokeObjectURL(candidate);
          return;
        }
        // Commit only now: until playback actually started, the old clip is
        // still the one a benchmark export should name.
        activeObjectUrl = candidate;
        activeClip = `file picker: ${file.name}`;
        if (previous) URL.revokeObjectURL(previous);
        setStatus(`playing ${file.name}`);
      },
      (err: unknown) => {
        if (generation !== loadGeneration) {
          URL.revokeObjectURL(candidate);
          return;
        }
        // The element is still pointing at the candidate that just failed.
        // Detach it before revoking, so the harness never holds a dead URL, and
        // drop the previous blob too: nothing is playing now, and naming a
        // stale clip in a benchmark export would be a lie.
        video.removeAttribute('src');
        video.load();
        URL.revokeObjectURL(candidate);
        if (previous) URL.revokeObjectURL(previous);
        activeObjectUrl = null;
        activeClip = `none — last load failed (${file.name})`;
        if (!gpuFatal) setStatus(`could not play ${file.name}: ${describeError(err)}`, 'error');
      },
    );
  });

  window.addEventListener('pagehide', (event) => {
    // `persisted` means the document is going into the back/forward cache and
    // may be restored with this same video element; revoking now would leave
    // it pointing at a dead URL.
    if (event.persisted) return;
    if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
  });

  // The global `error` listener would fire for a clip we already reported on,
  // so only the load path reports media errors; this catches later failures.
  video.addEventListener('error', () => {
    if (!gpuFatal && video.error) setStatus(`video element error: ${video.error.message}`, 'error');
  });

  window.setInterval(() => {
    const stats = pipeline.stats(performance.now());
    overlay.update(stats);
    if (!gpuFatal && pipeline.error) driver.fail(`frame loop stopped: ${describeError(pipeline.error)}`);
  }, OVERLAY_INTERVAL_MS);

  void loadClip(clip).then(
    () => setStatus(`playing bundled test clip (${clip})`),
    (err: unknown) =>
      setStatus(`bundled clip unavailable (${describeError(err)}) — choose a local video file`, 'warn'),
  );
}

/**
 * Self-describing benchmark record.
 *
 * Everything here is observable from the page. Machine, SoC, OS build, browser
 * build and display refresh rate are **not** web-exposed and must be filled in
 * by hand; `manualFields` names them explicitly so a pasted record is never
 * mistaken for complete. See BENCHMARKS.md.
 */
function benchmarkRecord(
  gpu: GpuContext,
  stats: PipelineStats,
  clip: string,
  neuralPrecision: 'f16' | 'fp32' | null,
  runtime?: ReturnType<RuntimeDriver['snapshot']>,
): unknown {
  return {
    schema: 'aethervsr.benchmark/1',
    capturedAt: new Date().toISOString(),
    manualFields: [
      'machine and SoC',
      'OS name and build',
      'browser name and exact build number',
      'display refresh rate (measure it; do not assume 60)',
      'window state (must be frontmost)',
    ],
    environment: {
      userAgent: navigator.userAgent,
      adapter: describeAdapter(gpu.adapterReport),
      // Raw fields as well as the display string: Chrome masks `device` and
      // `description` unless developer features are on, and a benchmark record
      // must show which fields were actually disclosed.
      adapterInfo: {
        vendor: gpu.adapterReport.vendor,
        architecture: gpu.adapterReport.architecture,
        device: gpu.adapterReport.device,
        description: gpu.adapterReport.description,
        fallbackAdapter: gpu.adapterReport.fallbackAdapter,
      },
      adapterFeatures: gpu.adapterReport.features,
      limits: gpu.capabilities.reportedLimits,
      preferredCanvasFormat: gpu.capabilities.preferredCanvasFormat,
      devicePixelRatio: window.devicePixelRatio,
      timestampQuery: gpu.capabilities.timestampQuery,
      documentVisibility: document.visibilityState,
      documentHasFocus: document.hasFocus(),
    },
    clip: {
      src: clip,
      decodedSize: stats.sourceSize,
      // The element's own view of what it decoded, independent of rVFC.
      duration: Number.isFinite(video.duration) ? video.duration : null,
      playbackRate: video.playbackRate,
    },
    pipeline: {
      clock: stats.clock,
      importPath: stats.importPath,
      // `upscalerId` is identical in both precisions and `adapterFeatures`
      // reports the adapter's capability, not what the stage used, so without
      // this an fp32 run is indistinguishable from an f16 one at 1.4x the cost.
      neuralPrecision,
      upscaler: stats.upscalerId,
      neural: stats.neural,
      source: stats.sourceSize,
      target: stats.targetSize,
      scaleFactor: stats.scaleFactor,
    },
    window: {
      elapsedMsSinceReset: stats.elapsedMs,
      rateWindow: 'instantaneous fps are trailing 1 s; mean fps are over elapsedMsSinceReset',
      timingWindow: 'each timing aggregate covers the trailing 240 samples, not the whole run',
    },
    measurements: {
      meanPresentedFps: stats.meanSourceFps,
      meanRenderedFps: stats.meanRenderFps,
      instantaneousPresentedFps: stats.sourceFps,
      instantaneousRenderedFps: stats.renderFps,
      framesRendered: stats.framesRendered,
      framesSkipped: stats.framesSkipped,
      decoderSinceReset: stats.quality,
      gpuUpscaleMs: stats.gpuPassMs,
      // What the GPU timestamps actually bracket. On the copy fallback the
      // import is a queue operation outside our command encoder, so its cost
      // is NOT included and the two import paths must not be ranked against
      // each other on this number alone.
      gpuUpscaleMsScope:
        stats.importPath === 'external'
          ? 'upscale render pass; importExternalTexture adds no separate GPU pass of ours, but whether the browser copied internally was not observed'
          : 'upscale render pass only; EXCLUDES the copyExternalImageToTexture upload',
      cpuFrameMs: stats.cpuFrameMs,
      cpuFrameMsScope: 'main thread: import call, command recording and submit',
      callbackLatencyMs: stats.callbackLatencyMs,
      // Submit-to-ready latency reported by the UA, not a per-frame cost.
      uaDecodeLatencyMs: stats.decodeLatencyMs,
    },
    ...(runtime ? {
      runtime: {
        ...runtime,
        gpuTimingScope: 'whole upscale stage only; excludes decode, import/upload, presentation and display latency',
        legacyFilter: new URLSearchParams(location.search).get('filter') === 'bilinear'
          ? 'main-page bilinear harness removed; legacy filter maps to Catmull-Rom baseline unless mode or upscaler=neural overrides it; bilinear remains available in bench.html'
          : null,
      },
    } : {}),
  };
}

// `.catch` after `.then`, not a rejection callback: a synchronous throw inside
// `main()` (a missing element, or a canvas that cannot give a WebGPU context)
// must reach the status line too, not become an unhandled rejection.
// A device only exposes features requested at creation, and creation happens
// before an upscaler is chosen. Without shader-f16 here the neural backend
// silently runs in fp32 - twice the activation memory and, measured, roughly
// twice the convolution time - while still reporting itself as working.
/** Features the harness deliberately does not request, so fallbacks can be tested. */
const withheldFeatures = new Set(
  (new URLSearchParams(location.search).get('withhold') ?? '')
    .split(',')
    .map((f) => f.trim())
    .filter((f) => f.length > 0),
);

acquireGpu({
  withheldFeatures: withheldFeatures.has('timestamp-query') ? ['timestamp-query'] : [],
  // `?withhold=shader-f16` drops a feature from the request, so the device is
  // created genuinely without it. That exercises the real detection path -
  // `device.features.has('shader-f16')` returning false - rather than
  // short-circuiting it with an option, which would leave the branch that
  // every f16-less adapter actually takes untested. On hardware that grants
  // f16 there is otherwise no way to reach it, and that is how the stage once
  // ran fp32 silently at 9.30 ms with no error at any layer (ADR-0023).
  optionalFeatures: [...UPSCALER_OPTIONAL_FEATURES, ...NEURAL_OPTIONAL_FEATURES].filter(
    (f) => !withheldFeatures.has(f),
  ),
})
  .then(main)
  .catch((err: unknown) => {
    setStatus(describeError(err), 'error');
    console.error(err);
  });
