/**
 * WebGPU adapter/device acquisition and introspection.
 *
 * This module owns every assumption about *what the browser exposes*, so the
 * rest of the pipeline can treat capabilities as plain data.
 */

/** Capability flags the pipeline branches on. */
export interface GpuCapabilities {
  /** `GPUDevice.importExternalTexture` exists and the device honours it. */
  readonly externalTexture: boolean;
  /** `timestamp-query` feature was requested and granted. */
  readonly timestampQuery: boolean;
  /** Canvas format reported by `navigator.gpu.getPreferredCanvasFormat()`. */
  readonly preferredCanvasFormat: GPUTextureFormat;
  readonly maxTextureDimension2D: number;
  /**
   * Limits worth recording alongside a benchmark. An external texture consumes
   * several sampled-texture and sampler bindings internally, and a neural
   * stage will be bounded by workgroup and storage-buffer limits, so these are
   * the numbers that explain a result on unfamiliar hardware.
   */
  readonly reportedLimits: Readonly<Record<string, number>>;
}

/**
 * Human-readable adapter identification, as far as the browser is willing to
 * disclose it. Every field is optional because Chrome masks most of this
 * unless the user opts in, and we must never fabricate values for a
 * benchmark report.
 */
export interface AdapterReport {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly features: readonly string[];
  /** True when the UA reported the adapter as software-backed. */
  readonly fallbackAdapter: boolean;
}

export interface GpuContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly capabilities: GpuCapabilities;
  readonly adapterReport: AdapterReport;
}

/**
 * Subscribes to the WebGPU failures that never reach a `try`/`catch`.
 *
 * Validation errors and device loss are delivered asynchronously: a bad
 * submission is discarded, the canvas stops updating, and the frame loop keeps
 * counting frames as though nothing happened. Without this, a broken pipeline
 * looks like a working one with a frozen picture.
 *
 * Returns an unsubscribe function.
 */
export function watchDeviceFailures(
  device: GPUDevice,
  onFailure: (message: string) => void,
): () => void {
  let live = true;

  const onUncaptured = (event: Event): void => {
    if (!live) return;
    // `GPUUncapturedErrorEvent` is not in every lib.dom; probe for the field.
    const detail =
      'error' in event && event.error instanceof GPUValidationError
        ? `WebGPU validation error: ${event.error.message}`
        : 'WebGPU uncaptured error (see console)';
    onFailure(detail);
  };

  device.addEventListener('uncapturederror', onUncaptured);
  void device.lost.then((info) => {
    if (!live) return;
    onFailure(`WebGPU device lost (${info.reason}): ${info.message}`);
  });

  return () => {
    live = false;
    device.removeEventListener('uncapturederror', onUncaptured);
  };
}

export class WebGpuUnavailableError extends Error {
  constructor(reason: string) {
    super(`WebGPU is unavailable: ${reason}`);
    this.name = 'WebGpuUnavailableError';
  }
}

/**
 * Features the harness itself wants. `timestamp-query` is what makes honest
 * performance reporting possible at all.
 */
const HARNESS_OPTIONAL_FEATURES: readonly GPUFeatureName[] = ['timestamp-query'];

/**
 * `isFallbackAdapter` migrated from `GPUAdapter` to `GPUAdapterInfo` mid-spec
 * and is absent from the current `@webgpu/types`, so it has to be probed at
 * runtime on whichever object the running browser put it on.
 */
function readFallbackFlag(source: object | undefined): boolean | undefined {
  if (!source || !('isFallbackAdapter' in source)) return undefined;
  const value = source.isFallbackAdapter;
  return typeof value === 'boolean' ? value : undefined;
}

/**
 * Limits recorded with every benchmark. Kept as an explicit list rather than
 * dumping `GPUSupportedLimits`, so a benchmark file stays diffable and does not
 * churn when the browser adds a limit we never reason about.
 */
const REPORTED_LIMIT_NAMES = [
  'maxTextureDimension2D',
  'maxSampledTexturesPerShaderStage',
  'maxSamplersPerShaderStage',
  'maxUniformBuffersPerShaderStage',
  'maxStorageBuffersPerShaderStage',
  'maxBindingsPerBindGroup',
  'maxUniformBufferBindingSize',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupStorageSize',
  'maxComputeWorkgroupsPerDimension',
  'minUniformBufferOffsetAlignment',
  'minStorageBufferOffsetAlignment',
] as const;

function collectLimits(limits: GPUSupportedLimits): Record<string, number> {
  const out: Record<string, number> = {};
  for (const name of REPORTED_LIMIT_NAMES) {
    const value = limits[name];
    if (typeof value === 'number') out[name] = value;
  }
  return out;
}

export interface AcquireGpuOptions {
  readonly powerPreference?: GPUPowerPreference;
  /**
   * Extra optional features to request, typically the union declared by the
   * available {@link Upscaler} backends. A device only exposes features that
   * were requested at creation, and creation necessarily happens before a
   * backend is chosen, so a backend that needs `shader-f16` must have said so
   * here rather than discovering the gap at shader-validation time.
   */
  readonly optionalFeatures?: readonly GPUFeatureName[];
}

export async function acquireGpu(options: AcquireGpuOptions = {}): Promise<GpuContext> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new WebGpuUnavailableError(
      'navigator.gpu is undefined. Use a Chromium 113+ / Safari 26+ build with hardware acceleration enabled.',
    );
  }

  const adapter = await navigator.gpu.requestAdapter(
    options.powerPreference ? { powerPreference: options.powerPreference } : {},
  );
  if (!adapter) {
    throw new WebGpuUnavailableError('navigator.gpu.requestAdapter() returned null (no compatible adapter).');
  }

  // Request only what the adapter actually advertises; asking for an
  // unsupported feature makes requestDevice() reject outright.
  const wanted = new Set([...HARNESS_OPTIONAL_FEATURES, ...(options.optionalFeatures ?? [])]);
  const requiredFeatures = [...wanted].filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures });

  const info: GPUAdapterInfo | undefined = adapter.info;
  const adapterReport: AdapterReport = {
    vendor: info?.vendor ?? '',
    architecture: info?.architecture ?? '',
    device: info?.device ?? '',
    description: info?.description ?? '',
    features: [...adapter.features].sort(),
    fallbackAdapter: readFallbackFlag(info) ?? readFallbackFlag(adapter) ?? false,
  };

  const capabilities: GpuCapabilities = {
    externalTexture: typeof device.importExternalTexture === 'function',
    timestampQuery: device.features.has('timestamp-query'),
    preferredCanvasFormat: navigator.gpu.getPreferredCanvasFormat(),
    maxTextureDimension2D: device.limits.maxTextureDimension2D,
    reportedLimits: collectLimits(device.limits),
  };

  return { adapter, device, capabilities, adapterReport };
}

/**
 * Collapses the adapter fields Chrome actually fills into one display string.
 * Chrome masks `vendor`/`architecture` on most platforms, so an empty result
 * is normal and must be shown as "not exposed" rather than guessed at.
 */
export function describeAdapter(report: AdapterReport): string {
  const parts = [report.vendor, report.architecture, report.device, report.description]
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (parts.length === 0) return 'not exposed by browser';
  return parts.join(' / ') + (report.fallbackAdapter ? ' (software fallback)' : '');
}
