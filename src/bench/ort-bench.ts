/**
 * ONNX Runtime Web / WebGPU probe.
 *
 * Purpose is narrow: find out whether a mature inference runtime's convolution
 * kernels beat the hand-written WGSL in `conv.wgsl.ts` on this device, and
 * what it costs to get there. It is not an integration.
 *
 * Three things are measured separately, because they have completely different
 * consequences for a per-frame budget:
 *
 * - **session creation** — paid once, can be hidden behind a loading state
 * - **first inference** — shader compilation and allocation, paid once
 * - **steady-state inference** — the only figure that competes with our 16.67 ms
 *
 * The runtime is imported dynamically so the ~26 MB WASM artefact is fetched
 * only when this probe runs, never by the Milestone 1 harness.
 */

import type { Env, InferenceSession, Tensor as OrtTensor } from 'onnxruntime-web';
// Let Vite own these two artefacts and hand us final URLs. Without this the
// Emscripten glue's own dynamic import is rewritten by Vite's module pipeline
// and resolves to the dev server's HTML fallback.
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';

export interface OrtProbeConfig {
  readonly modelUrl: string;
  readonly channels: number;
  readonly height: number;
  readonly width: number;
  readonly iterations: number;
  /** Execution providers to request, in order. */
  readonly providers: readonly string[];
  /**
   * Where ORT should leave the output. `cpu` forces a GPU->CPU download of the
   * whole tensor inside `run()`; `gpu-buffer` keeps it resident. With a 56 MB
   * activation tensor the difference is most of the measurement.
   */
  readonly outputLocation: 'cpu' | 'gpu-buffer';
}

export interface OrtProbeResult {
  readonly available: boolean;
  readonly ortVersion: string | null;
  readonly providers: readonly string[];
  readonly sessionCreateMs: number;
  readonly firstInferenceMs: number;
  readonly steadyMedianMs: number;
  readonly steadyMeanMs: number;
  readonly steadyMinMs: number;
  readonly steadyMaxMs: number;
  readonly iterations: number;
  readonly macs: number;
  readonly gmacPerSecond: number;
  readonly outputLocation: string;
  /** Console output captured during session creation, for EP-placement clues. */
  readonly logs: readonly string[];
  readonly error: string | null;
}

/** The subset of the `onnxruntime-web/webgpu` entry point this probe uses. */
interface OrtModule {
  readonly InferenceSession: {
    create(uri: string, options: InferenceSession.SessionOptions): Promise<InferenceSession>;
  };
  readonly Tensor: new (
    type: 'float32',
    data: Float32Array,
    dims: readonly number[],
  ) => OrtTensor;
  readonly env: Env;
}

/**
 * Runs the probe. Timings are wall-clock around `session.run()`, which is the
 * honest end-to-end figure an application would experience — it includes
 * ORT's own dispatch and synchronisation, not just GPU execution.
 */
export async function probeOrt(config: OrtProbeConfig): Promise<OrtProbeResult> {
  const macs = config.width * config.height * config.channels * config.channels * 9;
  const logs: string[] = [];
  const base = {
    available: false,
    ortVersion: null,
    providers: config.providers,
    sessionCreateMs: NaN,
    firstInferenceMs: NaN,
    steadyMedianMs: NaN,
    steadyMeanMs: NaN,
    steadyMinMs: NaN,
    steadyMaxMs: NaN,
    iterations: 0,
    macs,
    gmacPerSecond: NaN,
    outputLocation: config.outputLocation,
    logs,
    error: null,
  } satisfies OrtProbeResult;

  let ort: OrtModule;
  try {
    // Dynamic import exception: `onnxruntime-web/webgpu` pulls a ~26 MB WASM
    // artefact. A static import would attach it to the bundle graph and make
    // every visitor to the Milestone 1 harness pay for a Milestone 2 probe.
    // The specifier is literal; only the *timing* is deliberately deferred.
    ort = await import('onnxruntime-web/webgpu');
  } catch (err) {
    return { ...base, error: `import failed: ${describe(err)}` };
  }

  // Explicit per-file URLs, not a prefix: ORT would otherwise construct the
  // paths itself and Vite would intercept the glue module import.
  ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };

  // ORT reports execution-provider placement only through its own logging;
  // there is no structured API for it, so capture the console during setup.
  const originalWarn = console.warn;
  const originalError = console.error;
  console.warn = (...args: unknown[]) => {
    logs.push(`warn: ${args.map(String).join(' ')}`);
    originalWarn(...args);
  };
  console.error = (...args: unknown[]) => {
    logs.push(`error: ${args.map(String).join(' ')}`);
    originalError(...args);
  };

  try {
    const createStart = performance.now();
    const session = await ort.InferenceSession.create(config.modelUrl, {
      executionProviders: config.providers,
      graphOptimizationLevel: 'all',
      preferredOutputLocation: config.outputLocation,
    });
    const sessionCreateMs = performance.now() - createStart;

    const elements = config.channels * config.height * config.width;
    const data = new Float32Array(elements);
    for (let i = 0; i < elements; i++) data[i] = Math.sin(i * 0.01) * 0.5;
    const input = new ort.Tensor('float32', data, [1, config.channels, config.height, config.width]);
    const feeds: Record<string, OrtTensor> = { input };

    const firstStart = performance.now();
    await session.run(feeds);
    const firstInferenceMs = performance.now() - firstStart;

    const samples: number[] = [];
    for (let i = 0; i < config.iterations; i++) {
      const t0 = performance.now();
      await session.run(feeds);
      samples.push(performance.now() - t0);
    }
    await session.release();

    samples.sort((a, b) => a - b);
    const median = samples[Math.floor(samples.length / 2)] ?? NaN;
    return {
      ...base,
      available: true,
      ortVersion: readVersion(ort),
      sessionCreateMs,
      firstInferenceMs,
      steadyMedianMs: median,
      steadyMeanMs: samples.reduce((a, b) => a + b, 0) / samples.length,
      steadyMinMs: samples[0] ?? NaN,
      steadyMaxMs: samples[samples.length - 1] ?? NaN,
      iterations: samples.length,
      gmacPerSecond: macs / (median / 1000) / 1e9,
    };
  } catch (err) {
    return { ...base, error: describe(err) };
  } finally {
    console.warn = originalWarn;
    console.error = originalError;
  }
}

function readVersion(ort: OrtModule): string | null {
  const versions: unknown = (ort.env as unknown as Record<string, unknown>)['versions'];
  if (versions && typeof versions === 'object' && 'web' in versions) {
    return String((versions as Record<string, unknown>)['web']);
  }
  return null;
}

function describe(err: unknown): string {
  if (err instanceof Error) return err.message;
  return typeof err === 'string' ? err : JSON.stringify(err);
}
