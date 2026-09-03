import { acquireGpu, describeAdapter, type GpuContext } from '../core/gpu/device.js';
import { ConvBench, type ConvCase, type ConvResult } from './conv-bench.js';
import { measureRoofline, type RooflineResult } from './roofline.js';
import { ChainBench, type ChainCase, type ChainResult } from './chain-bench.js';
import { verifyChain, type ChainVerifyCase, type ChainVerifyResult } from './chain-verify.js';
import { StemBench, type StemCase, type StemResult } from './stem-bench.js';
import { verifyStem, type StemVerifyCase, type StemVerifyResult } from './stem-verify.js';
import { BridgeBench, type BridgeBenchConfig } from './bridge-bench.js';
import { verifyPixelShuffle, type ShuffleVerifyCase, type ShuffleVerifyResult } from './shuffle-verify.js';
import { HeadBench, verifyUpsampleHead, type HeadCase, type HeadResult, type HeadVerifyResult } from './head-bench.js';
import { verifyConv, type ConvVerifyCase, type ConvVerifyResult } from './conv-verify.js';
import { deferred, delay } from './deferred.js';
import { probeOrt, type OrtProbeConfig, type OrtProbeResult } from './ort-bench.js';
import { evaluateScalers } from './quality-bench.js';
import type { QualityScores } from './quality.js';
import { IngestBench, type IngestBenchConfig, type IngestBenchStats } from './ingest-bench.js';
import type { BaselineFilter } from '../core/upscale/baseline.wgsl.js';
import { runTemporalBench, type TemporalBenchConfig, type TemporalSequenceResult } from './temporal-bench.js';

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
  // Only require ingest samples when an ingest pass actually ran. Without
  // external textures the frame arrives already sampleable and is passed
  // through, which is a supported configuration, not a failed measurement.
  if (stats.ingestPasses > 0 && timestampQuery && (stats.ingestMs?.samples ?? 0) === 0) {
    throw new Error('ingest passes were recorded but produced 0 GPU samples');
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

  w['aethervsrConvVerify'] = async (
    cases: readonly ConvVerifyCase[],
    // f32 cases land at ~1e-7. f16 accumulates over 9*inChannels MACs and needs
    // a tolerance sized to that; without this parameter the hook could not
    // verify the f16 kernel at all.
    tolerance = 1e-4,
  ): Promise<ConvVerifyResult[]> => {
    setStatus(`verifying ${cases.length} convolution cases against a CPU reference…`);
    const results: ConvVerifyResult[] = [];
    for (const c of cases) results.push(await verifyConv(gpu.device, c, tolerance));
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

  w['aethervsrChainBench'] = async (cases: readonly ChainCase[]): Promise<ChainResult[]> => {
    setStatus(`running ${cases.length} convolution chain cases…`);
    const bench = new ChainBench(gpu.device);
    const results = await bench.run(cases);
    setStatus('chain bench complete');
    return results;
  };

  w['aethervsrChainVerify'] = async (
    cases: readonly ChainVerifyCase[],
    tolerance?: number,
  ): Promise<ChainVerifyResult[]> => {
    setStatus(`verifying ${cases.length} chains against the CPU reference…`);
    const out: ChainVerifyResult[] = [];
    for (const c of cases) out.push(await verifyChain(gpu.device, c, tolerance));
    setStatus('chain verification complete');
    return out;
  };

  w['aethervsrStemBench'] = async (cases: readonly StemCase[]): Promise<StemResult[]> => {
    setStatus(`running ${cases.length} stem cases…`);
    const bench = new StemBench(gpu.device);
    const results = await bench.run(cases);
    setStatus('stem bench complete');
    return results;
  };

  w['aethervsrStemVerify'] = async (
    cases: readonly StemVerifyCase[],
    tolerance?: number,
  ): Promise<StemVerifyResult[]> => {
    setStatus(`verifying ${cases.length} stem cases…`);
    const out: StemVerifyResult[] = [];
    for (const c of cases) out.push(await verifyStem(gpu.device, c, tolerance));
    setStatus('stem verification complete');
    return out;
  };

  w['aethervsrBridgeBench'] = async (
    bridgeConfig: BridgeBenchConfig,
    warmupMs = 3000,
    runMs = 8000,
  ): Promise<unknown> => {
    setStatus('measuring external texture -> ingest -> packed activations…');
    await loadClip(CLIP);
    const bench = new BridgeBench(gpu, video, bridgeConfig);
    try {
      bench.start();
      await delay(warmupMs);
      bench.reset();
      await delay(runMs);
      const verification = await bench.captureAndVerify();
      const stats = bench.stats();
      setStatus('bridge bench complete');
      return { ...stats, verification };
    } finally {
      bench.destroy();
    }
  };

  w['aethervsrShuffleVerify'] = async (
    cases: readonly ShuffleVerifyCase[],
    tolerance?: number,
  ): Promise<ShuffleVerifyResult[]> => {
    setStatus(`verifying ${cases.length} pixel-shuffle cases…`);
    const out: ShuffleVerifyResult[] = [];
    for (const c of cases) out.push(await verifyPixelShuffle(gpu.device, c, tolerance));
    setStatus('pixel-shuffle verification complete');
    return out;
  };

  w['aethervsrHeadBench'] = async (cases: readonly HeadCase[]): Promise<HeadResult[]> => {
    setStatus(`running ${cases.length} reconstruction-head cases…`);
    const bench = new HeadBench(gpu.device);
    const results = await bench.run(cases);
    setStatus('head bench complete');
    return results;
  };

  w['aethervsrHeadVerify'] = async (
    cases: readonly { width: number; height: number; inChannels: number; useF16?: boolean }[],
    tolerance?: number,
  ): Promise<HeadVerifyResult[]> => {
    setStatus(`verifying ${cases.length} reconstruction-head cases…`);
    const out: HeadVerifyResult[] = [];
    for (const c of cases) out.push(await verifyUpsampleHead(gpu.device, c, tolerance));
    setStatus('head verification complete');
    return out;
  };

  w['aethervsrRoofline'] = async (useF16 = true): Promise<RooflineResult[]> => {
    setStatus('measuring achievable bandwidth and FMA throughput…');
    const results = await measureRoofline(gpu.device, useF16);
    setStatus('roofline probe complete');
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
    // The harness owns the only device on the page. Handing it to ORT is what
    // makes a GPU-resident input possible at all: a buffer allocated here has
    // to be visible to the runtime that reads it.
    const result = await probeOrt(
      config.inputLocation === 'gpu-buffer'
        ? { ...config, device: gpu.device, adapter: gpu.adapter }
        : config,
    );
    setStatus(result.error ? `ORT probe failed: ${result.error}` : 'ORT probe complete');
    return result;
  };

  w['aethervsrTemporalBench'] = async (
    temporalConfig?: TemporalBenchConfig,
  ): Promise<TemporalSequenceResult[]> => {
    setStatus('measuring temporal behaviour (frame-to-frame shimmer) of the baseline upscalers…');
    const results = await runTemporalBench(gpu.device, temporalConfig);
    setStatus('temporal baseline measurement complete');
    return results;
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

acquireGpu({
  // 'chromium-experimental-subgroup-matrix' is requested opportunistically and
  // is absent unless Chrome was launched with --enable-unsafe-webgpu, so it can
  // never be part of a shipped path. acquireGpu drops features the adapter does
  // not advertise, so asking is free.
  optionalFeatures: [
    'shader-f16',
    'subgroups',
    'chromium-experimental-subgroup-matrix' as GPUFeatureName,
  ],
  // Raised only if the adapter allows it. Any result that depends on more
  // than the 16384-byte guaranteed floor is labelled as such in BENCHMARKS.md.
  optionalLimits: { maxComputeWorkgroupStorageSize: 32768 },
})
  .then(main)
  .catch((err: unknown) => {
    setStatus(err instanceof Error ? err.message : String(err));
    console.error(err);
  });
