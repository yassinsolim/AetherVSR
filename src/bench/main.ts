import { acquireGpu, describeAdapter, type GpuContext } from '../core/gpu/device.js';
import { ConvBench, type ConvCase, type ConvResult } from './conv-bench.js';
import { verifyConv, type ConvVerifyCase, type ConvVerifyResult } from './conv-verify.js';
import { deferred, delay } from './deferred.js';
import { probeOrt, type OrtProbeConfig, type OrtProbeResult } from './ort-bench.js';
import { evaluateScalers } from './quality-bench.js';
import type { QualityScores } from './quality.js';
import { IngestBench, type IngestBenchConfig, type IngestBenchStats } from './ingest-bench.js';
import type { BaselineFilter } from '../core/upscale/baseline.wgsl.js';

/**
 * Milestone 2 feasibility bench.
 *
 * Two experiments, both driven from automation rather than clicks so a run is
 * reproducible:
 *
 * - `window.aethervsrIngestBench(config, warmupMs, runMs)` measures the
 *   external-texture ingest pass separately from the pass that consumes it.
 * - `window.aethervsrConvBench(cases)` measures 3x3 convolution throughput.
 *
 * Nothing here is production code; it exists to produce numbers that decide
 * the Milestone 3 architecture.
 */

const CLIP = '/media/aethervsr-testclip-720p60-h264.mp4';

function requireElement<T extends Element>(id: string, ctor: abstract new () => T): T {
  const el = document.getElementById(id);
  if (!(el instanceof ctor)) throw new Error(`#${id} is missing or not a ${ctor.name}`);
  return el;
}

const video = requireElement('source', HTMLVideoElement);
const canvas = requireElement('output', HTMLCanvasElement);
const status = requireElement('status', HTMLParagraphElement);

/** Surfaces a hot-path failure captured by the bench as a real Error. */
function rethrow(err: unknown): void {
  if (err === null || err === undefined) return;
  if (err instanceof Error) throw err;
  // Not an Error: describe it without relying on Object's stringification,
  // which would render every plain object as "[object Object]".
  throw new Error(typeof err === 'string' ? err : JSON.stringify(err));
}

/**
 * Refuses to return an unmeasured result.
 *
 * A throttled or occluded tab suspends `requestVideoFrameCallback` entirely,
 * so the bench completes its timers having processed nothing and every
 * aggregate is NaN. Marshalled out of the page as JSON, NaN silently becomes
 * `null` and reads like a value. Failing here turns that into an obvious
 * error at the point of collection instead of a plausible-looking zero in a
 * results table.
 */
function assertMeasured(stats: IngestBenchStats, timestampQuery: boolean): void {
  if (stats.framesRendered === 0) {
    throw new Error(
      'bench collected 0 frames: requestVideoFrameCallback did not fire. ' +
        'The browser window must be visible and frontmost for the whole run.',
    );
  }
  if (timestampQuery && stats.scaleMs.samples === 0) {
    throw new Error('bench collected 0 GPU timestamp samples despite timestamp-query being available');
  }
  if (stats.mode === 'ingest' && timestampQuery && (stats.ingestMs?.samples ?? 0) === 0) {
    throw new Error('ingest mode collected 0 ingest-pass GPU samples');
  }
}

function setStatus(text: string): void {
  status.textContent = text;
}

async function loadClip(url: string): Promise<void> {
  video.src = url;
  video.load();
  const { promise, resolve, reject } = deferred<void>();
  const ok = (): void => {
    cleanup();
    resolve();
  };
  const bad = (): void => {
    cleanup();
    reject(new Error(video.error?.message ?? 'media error'));
  };
  const cleanup = (): void => {
    video.removeEventListener('loadeddata', ok);
    video.removeEventListener('error', bad);
  };
  video.addEventListener('loadeddata', ok, { once: true });
  video.addEventListener('error', bad, { once: true });
  await promise;
  await video.play();
}

function environment(gpu: GpuContext): unknown {
  return {
    userAgent: navigator.userAgent,
    adapter: describeAdapter(gpu.adapterReport),
    features: gpu.adapterReport.features,
    limits: gpu.capabilities.reportedLimits,
    timestampQuery: gpu.capabilities.timestampQuery,
    shaderF16: gpu.adapterReport.features.includes('shader-f16'),
    manualFields: ['machine and SoC', 'OS build', 'browser build', 'display refresh rate'],
  };
}

function main(gpu: GpuContext): void {
  setStatus('GPU ready — call window.aethervsrIngestBench() or window.aethervsrConvBench()');

  const w = window as unknown as Record<string, unknown>;
  w['aethervsrEnvironment'] = () => environment(gpu);

  w['aethervsrIngestBench'] = async (
    config: IngestBenchConfig,
    warmupMs = 4000,
    runMs = 20000,
  ): Promise<IngestBenchStats> => {
    await loadClip(CLIP);
    const bench = new IngestBench(gpu, video, canvas, config);
    try {
      bench.start();
      await delay(warmupMs);
      rethrow(bench.error);
      bench.reset();
      await delay(runMs);
      rethrow(bench.error);
      const stats = bench.stats(performance.now());
      assertMeasured(stats, gpu.capabilities.timestampQuery);
      return stats;
    } finally {
      bench.destroy();
    }
  };

  w['aethervsrConvVerify'] = async (cases: readonly ConvVerifyCase[]): Promise<ConvVerifyResult[]> => {
    setStatus(`verifying ${cases.length} convolution cases against a CPU reference…`);
    const results: ConvVerifyResult[] = [];
    for (const c of cases) results.push(await verifyConv(gpu.device, c));
    const failed = results.filter((r) => !r.passed).length;
    setStatus(failed === 0 ? 'convolution kernel verified' : `${failed} verification case(s) FAILED`);
    return results;
  };

  w['aethervsrConvBench'] = async (cases: readonly ConvCase[]): Promise<ConvResult[]> => {
    setStatus(`running ${cases.length} convolution cases…`);
    const bench = new ConvBench(gpu.device);
    const results = await bench.run(cases);
    setStatus('convolution bench complete');
    return results;
  };

  w['aethervsrQualityBench'] = async (
    filters: readonly BaselineFilter[] = ['bilinear', 'catmull-rom'],
  ): Promise<QualityScores[]> => {
    setStatus('evaluating upscaler image quality against a generated reference…');
    const results = await evaluateScalers(gpu.device, filters);
    setStatus('quality evaluation complete');
    return results;
  };

  w['aethervsrOrtProbe'] = async (config: OrtProbeConfig): Promise<OrtProbeResult> => {
    setStatus('probing ONNX Runtime Web (WebGPU EP) — this fetches a ~26 MB WASM artefact…');
    const result = await probeOrt(config);
    setStatus(result.error ? `ORT probe failed: ${result.error}` : 'ORT probe complete');
    return result;
  };

  // Surfaces a device-side validation failure that would otherwise only
  // discard submissions silently and leave the numbers looking plausible.
  gpu.device.addEventListener('uncapturederror', (event) => {
    const detail =
      'error' in event && event.error instanceof GPUValidationError
        ? event.error.message
        : 'uncaptured WebGPU error';
    setStatus(`WebGPU error: ${detail}`);
    console.error(detail);
  });
}

acquireGpu({ optionalFeatures: ['shader-f16'] })
  .then(main)
  .catch((err: unknown) => {
    setStatus(err instanceof Error ? err.message : String(err));
    console.error(err);
  });
