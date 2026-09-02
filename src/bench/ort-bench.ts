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
  /**
   * Median wall time around `session.run()`.
   *
   * **Scope depends on `outputLocation`.**
   *
   * With `cpu` the call must download the output, which forces GPU completion,
   * so the figure is end-to-end including both transfers.
   *
   * With `gpu-buffer` nothing inside `run()` waits — the native EP flushes by
   * submitting to the queue and returns. The probe therefore awaits
   * `queue.onSubmittedWorkDone()` on ORT's own device inside the timed
   * interval, making the figure a **queue-completion latency**: everything
   * submitted to that queue, which for a single-session idle page is this
   * inference plus the input upload, but is not guaranteed to be only that.
   * `fenced` records whether the fence was actually taken; when false the
   * figure is submission-side only.
   */
  readonly steadyMedianMs: number;
  readonly steadyMeanMs: number;
  readonly steadyMinMs: number;
  readonly steadyMaxMs: number;
  readonly iterations: number;
  /**
   * Node-to-execution-provider placement lines captured from ORT's verbose
   * log. The only way to observe a silent CPU fallback: there is no structured
   * API for it. Empty means nothing matched, not that placement was all-GPU.
   */
  readonly placement: readonly string[];
  readonly outputLocation: string;
  /** True when a GPU completion fence was inside the timed interval. */
  readonly fenced: boolean;
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
  // Deliberately no MAC/throughput figure. The probe cannot introspect the
  // loaded graph, so a caller-supplied channel count is not evidence that the
  // model matches, and a derived GMAC/s would be unfalsifiable. Compare
  // against the WGSL kernel on time, with the scope caveats in BENCHMARKS.md.
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
    placement: [],
    outputLocation: config.outputLocation,
    fenced: false,
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
  const originalLog = console.log;
  const originalWarn = console.warn;
  const originalError = console.error;
  console.log = (...args: unknown[]) => {
    const text = args.map(String).join(' ');
    // Keep only placement-relevant lines; verbose ORT logging is enormous.
    if (/placement|provider|fallback|kernel not found/i.test(text)) logs.push(`log: ${text}`);
    originalLog(...args);
  };
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
      // ORT emits node-to-execution-provider placement only at verbose level.
      // Without this the console capture below can never observe a silent CPU
      // fallback, which is the specific risk we are trying to detect.
      logSeverityLevel: 0,
    });
    const sessionCreateMs = performance.now() - createStart;

    // Resolved before the first inference so that run and the steady loop are
    // fenced identically.
    const ortDevice = config.outputLocation === 'gpu-buffer' ? await readOrtDevice(ort) : null;

    const elements = config.channels * config.height * config.width;
    const data = new Float32Array(elements);
    for (let i = 0; i < elements; i++) data[i] = Math.sin(i * 0.01) * 0.5;
    const input = new ort.Tensor('float32', data, [1, config.channels, config.height, config.width]);
    const feeds: Record<string, OrtTensor> = { input };

    const firstStart = performance.now();
    const firstOutputs = await session.run(feeds);
    // Fenced on the same terms as the steady loop, so the two are comparable.
    await ortDevice?.queue.onSubmittedWorkDone();
    const firstInferenceMs = performance.now() - firstStart;
    disposeOutputs(firstOutputs);

    // With a GPU-resident output nothing in `run()` waits for the GPU: the
    // native EP ends a run by submitting to the queue and returning. Awaiting
    // the device's own completion signal inside the timed interval turns a
    // submission latency into an inference latency. ORT exposes the device it
    // is using, which is the same one the fence must be taken on.
    const samples: number[] = [];
    for (let i = 0; i < config.iterations; i++) {
      const t0 = performance.now();
      const outputs = await session.run(feeds);
      await ortDevice?.queue.onSubmittedWorkDone();
      samples.push(performance.now() - t0);
      // ORT owns GPU-resident output buffers. Without disposal they accumulate
      // for the length of the run and eventually distort or exhaust memory.
      disposeOutputs(outputs);
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
      placement: logs.filter((l) => /Node placements|All nodes placed|kernel not found/i.test(l)),
      fenced: ortDevice !== null,
    };
  } catch (err) {
    return { ...base, error: describe(err) };
  } finally {
    console.log = originalLog;
    console.warn = originalWarn;
    console.error = originalError;
  }
}

/**
 * ORT's own `GPUDevice`, for taking a completion fence on the right queue.
 * Returns null if the runtime does not expose one, in which case the caller
 * must treat the timing as submission-side only.
 */
async function readOrtDevice(ort: OrtModule): Promise<GPUDevice | null> {
  const webgpu: unknown = (ort.env as unknown as Record<string, unknown>)['webgpu'];
  if (!webgpu || typeof webgpu !== 'object' || !('device' in webgpu)) return null;
  const device: unknown = await webgpu.device;
  return device instanceof GPUDevice ? device : null;
}

function readVersion(ort: OrtModule): string | null {
  const versions: unknown = (ort.env as unknown as Record<string, unknown>)['versions'];
  if (versions && typeof versions === 'object' && 'web' in versions) {
    return String((versions as Record<string, unknown>)['web']);
  }
  return null;
}

/**
 * Releases ORT-owned output tensors.
 *
 * Only meaningful for `gpu-buffer` outputs, where the tensor owns a GPUBuffer
 * that is not reclaimed by dropping the JS reference.
 */
function disposeOutputs(outputs: Record<string, OrtTensor>): void {
  for (const tensor of Object.values(outputs)) {
    const disposable = tensor as unknown as { dispose?: () => void };
    disposable.dispose?.();
  }
}

function describe(err: unknown): string {
  if (err instanceof Error) return err.message;
  return typeof err === 'string' ? err : JSON.stringify(err);
}
