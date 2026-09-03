import {
  acquireGpu,
  describeAdapter,
  watchDeviceFailures,
  type GpuContext,
} from './core/gpu/device.js';
import { VideoPipeline, type PipelineStats } from './core/pipeline.js';
import { BaselineScaler, UPSCALER_OPTIONAL_FEATURES } from './core/upscale/baseline-scaler.js';
import { NeuralUpscaler, NEURAL_OPTIONAL_FEATURES } from './core/upscale/neural-upscaler.js';
import { BudgetGuard, type BudgetDecision } from './core/upscale/budget-guard.js';
import { loadModel, type PackedModel } from './core/neural/model.js';
import type { BaselineFilter } from './core/upscale/baseline.wgsl.js';
import { DiagnosticOverlay, OVERLAY_INTERVAL_MS } from './ui/overlay.js';

/** Clip shipped with the repo so benchmarks are reproducible. See tools/. */
const DEFAULT_MODEL = '/models/aethersr-c16d2.json';
const DEFAULT_CLIP = '/media/aethervsr-testclip-720p30-vp9.webm';

const FILTERS: readonly { readonly id: BaselineFilter; readonly label: string }[] = [
  { id: 'catmull-rom', label: 'Catmull-Rom bicubic (9-tap)' },
  { id: 'bilinear', label: 'Bilinear (hardware sampler)' },
];

/** Filter used unless `?filter=` selects another. */
const DEFAULT_FILTER: BaselineFilter = 'catmull-rom';

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

  // Query overrides exist so a benchmark run is reproducible from a URL rather
  // than from a sequence of clicks. See BENCHMARKS.md.
  const params = new URLSearchParams(location.search);
  const requested = FILTERS.find((f) => f.id === params.get('filter'));
  const initialFilter: BaselineFilter = requested?.id ?? DEFAULT_FILTER;
  const forceCopyImport = params.get('import') === 'copy';
  // Relative paths only: this is a dev harness, not a URL loader.
  const clipParam = params.get('clip');
  const clip = clipParam !== null && clipParam.startsWith('/') ? clipParam : DEFAULT_CLIP;

  const pipeline = new VideoPipeline(gpu, video, canvas, new BaselineScaler(initialFilter), {
    forceCopyImport,
  });

  // A discarded submission or a lost device never throws in the frame loop, so
  // without this the canvas can freeze while the frame counter keeps climbing.
  watchDeviceFailures(gpu.device, (message) => {
    gpuFatal = true;
    pipeline.stop();
    setStatus(`${message} — reload the page to recover`, 'error');
  });

  // What is actually playing, for the benchmark record. Not the startup clip:
  // the file picker replaces it.
  let activeClip = clip;
  let activeObjectUrl: string | null = null;

  for (const filter of FILTERS) {
    const option = document.createElement('option');
    option.value = filter.id;
    option.textContent = filter.label;
    option.selected = filter.id === initialFilter;
    upscalerSelect.append(option);
  }
  // The neural backend is offered only once its weights have loaded, so the
  // menu never lists a stage that would fail on selection.
  const NEURAL_VALUE = 'neural';
  let neuralModel: PackedModel | null = null;
  let neuralInstance: NeuralUpscaler | null = null;
  const modelParam = params.get('model');
  const modelUrl = modelParam !== null && modelParam.startsWith('/') ? modelParam : DEFAULT_MODEL;
  void loadModel(modelUrl)
    .then((model) => {
      neuralModel = model;
      const option = document.createElement('option');
      option.value = NEURAL_VALUE;
      option.textContent = `Neural C${model.features}D${model.depth} (2x)`;
      upscalerSelect.append(option);
      if (params.get('upscaler') === NEURAL_VALUE) {
        upscalerSelect.value = NEURAL_VALUE;
        neuralInstance = new NeuralUpscaler(model);
        pipeline.setUpscaler(neuralInstance);
      }
    })
    .catch((err: unknown) => {
      // A missing model is not a harness failure: the baseline still works and
      // saying so is more useful than an empty menu entry that throws.
      console.warn('neural model unavailable:', err);
    });

  upscalerSelect.addEventListener('change', () => {
    if (upscalerSelect.value === NEURAL_VALUE) {
      if (!neuralModel) return;
      userChoseNeural = true;
      guard.reset();
      neuralInstance = new NeuralUpscaler(neuralModel);
      pipeline.setUpscaler(neuralInstance);
      return;
    }
    const chosen = FILTERS.find((f) => f.id === upscalerSelect.value);
    if (!chosen) return;
    // Exercises the Milestone 1 seam: the processing stage is reconfigured
    // while acquisition and presentation keep running untouched.
    userChoseNeural = false;
    guard.reset();
    neuralInstance = null;
    pipeline.setUpscaler(new BaselineScaler(chosen.id));
  });

  // The neural stage measures itself per pass; expose it so a whole-stage
  // breakdown never has to be inferred from the pipeline's single span.
  (window as unknown as Record<string, unknown>)['aethervsrNeuralStage'] = () => ({
    id: neuralInstance?.id ?? null,
    timing: neuralInstance?.stageTiming ?? null,
    memory: neuralInstance?.memoryReport ?? null,
    budget: {
      state: guard.current,
      forced: guard.isForced,
      probing: guard.isProbing,
      lastReason: lastBudgetReason,
      lastMedianMs: lastBudgetMedian,
    },
  });

  // --- frame-budget fallback ------------------------------------------------
  // Watches the measured whole-stage time and drops to the baseline scaler when
  // the neural stage cannot hold the budget. Deliberately not a quality
  // controller - that is Milestone 6 - just a switch that cannot oscillate.
  const guard = new BudgetGuard();
  let lastBudgetReason = 'warming up';
  let lastBudgetMedian = Number.NaN;
  let userChoseNeural = params.get('upscaler') === NEURAL_VALUE;

  function applyBudgetDecision(decision: BudgetDecision): void {
    lastBudgetReason = decision.reason;
    lastBudgetMedian = decision.medianMs;
    if (!decision.changed) return;
    if (decision.state === 'fallback') {
      neuralInstance = null;
      pipeline.setUpscaler(new BaselineScaler(DEFAULT_FILTER));
      setStatus(`neural stage over budget (${decision.reason}) — using ${DEFAULT_FILTER}`, 'warn');
    } else {
      neuralInstance = new NeuralUpscaler(neuralModel!);
      pipeline.setUpscaler(neuralInstance);
      setStatus(decision.reason, 'info');
    }
  }

  (window as unknown as Record<string, unknown>)['aethervsrForceOverBudget'] = (on: boolean) => {
    applyBudgetDecision(guard.setForced(on, performance.now()));
    return { forced: guard.isForced, state: guard.current };
  };

  // Every resolved whole-stage sample, and only ever one measured while the
  // neural stage was the thing running. Handing the guard the baseline's
  // cheaper timings is what made an earlier version cycle.
  pipeline.onGpuPassSample = (ms) => {
    if (!userChoseNeural || !neuralModel || !neuralInstance) return;
    applyBudgetDecision(guard.record(ms, performance.now()));
  };

  // While in fallback nothing neural is being measured, so the only thing that
  // can change the state is the probe timer.
  setInterval(() => {
    if (!userChoseNeural || !neuralModel) return;
    if (guard.current !== 'fallback') return;
    applyBudgetDecision(guard.tick(performance.now()));
  }, 250);

  resetButton.addEventListener('click', () => pipeline.resetMeasurements());

  exportButton.addEventListener('click', () => {
    const payload = benchmarkRecord(gpu, pipeline.stats(performance.now()), activeClip);
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
        setStatus(`could not play ${file.name}: ${describeError(err)}`, 'error');
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

  video.addEventListener('playing', () => {
    // Never resume onto a lost device: submissions would be silently discarded
    // while the frame counter kept climbing.
    if (gpuFatal || pipeline.running) return;
    pipeline.start();
  });
  // The global `error` listener would fire for a clip we already reported on,
  // so only the load path reports media errors; this catches later failures.
  video.addEventListener('error', () => {
    if (video.error) setStatus(`video element error: ${video.error.message}`, 'error');
  });

  window.setInterval(() => {
    const stats = pipeline.stats(performance.now());
    overlay.update(stats);
    if (pipeline.error) setStatus(`frame loop stopped: ${describeError(pipeline.error)}`, 'error');
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
function benchmarkRecord(gpu: GpuContext, stats: PipelineStats, clip: string): unknown {
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
  };
}

// `.catch` after `.then`, not a rejection callback: a synchronous throw inside
// `main()` (a missing element, or a canvas that cannot give a WebGPU context)
// must reach the status line too, not become an unhandled rejection.
// A device only exposes features requested at creation, and creation happens
// before an upscaler is chosen. Without shader-f16 here the neural backend
// silently runs in fp32 - twice the activation memory and, measured, roughly
// twice the convolution time - while still reporting itself as working.
acquireGpu({
  optionalFeatures: [...UPSCALER_OPTIONAL_FEATURES, ...NEURAL_OPTIONAL_FEATURES],
})
  .then(main)
  .catch((err: unknown) => {
    setStatus(describeError(err), 'error');
    console.error(err);
  });
