import { buildConvShader, convMacCount, type Activation, type ConvShaderConfig } from './conv.wgsl.js';
import { buildTiledConvShader, tiledSharedBytes } from './conv-tiled.wgsl.js';
import { buildPackedConvShader, packedSharedBytes } from './conv-packed.wgsl.js';
import { buildBlockedConvShader, blockedSharedBytes } from './conv-blocked.wgsl.js';
import { buildMatrixConvShader, matrixWeightIndex } from './conv-matrix.wgsl.js';

/**
 * Which convolution kernel implementation to measure.
 *
 * - `naive`  - Milestone 2 kernel: every tap read from global storage.
 * - `tiled`  - stages each input channel through workgroup memory with a halo.
 *
 * Kept as separate implementations rather than a flag inside one shader so a
 * regression in the newer kernel can never silently become the baseline.
 */
export type ConvVariant = 'naive' | 'tiled' | 'packed' | 'blocked' | 'matrix';

export interface ConvCase {
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly inChannels: number;
  readonly outChannels: number;
  readonly tileX: number;
  readonly tileY: number;
  readonly blockX: number;
  readonly activation: Activation;
  readonly useF16: boolean;
  readonly residual: boolean;
  /** Defaults to `naive`, which is the Milestone 2 reference implementation. */
  readonly variant?: ConvVariant;
  /** Output channels per invocation, for the `blocked` variant. Defaults to 1. */
  readonly outBlock?: number;
  /** Output rows per invocation, for the `blocked` variant. Defaults to 1. */
  readonly blockY?: number;
  /** Weight memory order, for the `blocked` variant. Defaults to `oc-major`. */
  readonly weightLayout?: 'oc-major' | 'tap-major';
  /** Output rows per workgroup, for the `matrix` variant. Defaults to 1. */
  readonly rowsPerGroup?: number;
}

export interface ConvResult extends ConvCase {
  /** Validation/compilation problems. Non-empty means the numbers are invalid. */
  readonly diagnostics: readonly string[];
  /** False when the case could not be measured; ignore all timings then. */
  readonly valid: boolean;
  /** Median GPU time across timed iterations, in milliseconds. */
  readonly medianMs: number;
  readonly meanMs: number;
  readonly minMs: number;
  readonly maxMs: number;
  readonly iterations: number;
  readonly macs: number;
  /** Effective throughput in GMAC/s, from the median. */
  readonly gmacPerSecond: number;
  /** Activation + weight bytes touched per dispatch, lower bound. */
  readonly bytesMoved: number;
  /** Lower-bound effective bandwidth in GB/s, from the median. */
  readonly gbPerSecond: number;
  /** Total GPU buffer bytes allocated for this case. */
  readonly bufferBytes: number;
  /** Workgroup storage the generated shader declares. 0 for variants without any. */
  readonly sharedBytes: number;
  /**
   * True when the configuration fits this adapter only because a raised
   * `maxComputeWorkgroupStorageSize` was granted. Such a result is valid but
   * **not portable**, and must not be published as a portable figure.
   */
  readonly requiresRaisedLimit: boolean;
}

const NS_PER_MS = 1_000_000;

/**
 * The `maxComputeWorkgroupStorageSize` every WebGPU implementation must offer.
 *
 * Anything above this is a per-adapter bonus. A configuration that needs more
 * is not portable, however fast it is here.
 */
export const PORTABLE_WORKGROUP_STORAGE_BYTES = 16384;

/**
 * Measures 3x3 convolution throughput on the real device.
 *
 * ## Method
 *
 * Each case allocates its own buffers, compiles its own pipeline, runs
 * `warmup` untimed dispatches to force shader compilation and let clocks
 * settle, then runs `iterations` dispatches each bracketed by its own
 * `timestamp-query` pair. Every dispatch is a separate submission with its own
 * query resolve, so results are per-dispatch GPU times rather than an average
 * smeared over a batch.
 *
 * The reported figure is the **median**, not the mean: the first timed
 * iterations after warm-up are still occasionally slow, and a median is robust
 * to that without discarding data by hand.
 *
 * No CPU readback of activations ever occurs. The only mapped buffer is the
 * timestamp resolve target.
 */
export class ConvBench {
  constructor(
    private readonly device: GPUDevice,
    private readonly warmup = 8,
    private readonly iterations = 40,
    /**
     * Minimum wall-clock milliseconds of warm-up dispatches before timing.
     *
     * A fixed iteration count is not a warm-up, it is a warm-up for one
     * workload size. Eight dispatches of a 0.03 ms kernel is 0.25 ms of work,
     * which is nowhere near enough to bring the GPU off its idle clock: the
     * first measured pass of a session came out at 0.135 ms against a steady
     * 0.031 ms, a 4x error, entirely on the small configurations. Large
     * workloads never showed it because they ramp the clock themselves.
     *
     * Warming for a duration instead makes every configuration in a sweep
     * comparable without the caller having to know to throw a pass away.
     */
    private readonly warmupMs = 60,
  ) {}

  async run(cases: readonly ConvCase[]): Promise<ConvResult[]> {
    const results: ConvResult[] = [];
    for (const c of cases) {
      results.push(await this.runCase(c));
    }
    return results;
  }

  private async runCase(c: ConvCase): Promise<ConvResult> {
    const { device } = this;
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    const bytesPerElement = c.useF16 ? 2 : 4;
    const pixels = c.width * c.height;
    const inElements = pixels * c.inChannels;
    const outElements = pixels * c.outChannels;
    const weightElements = c.inChannels * c.outChannels * 9;

    const shaderConfig: ConvShaderConfig = {
      inChannels: c.inChannels,
      outChannels: c.outChannels,
      tileX: c.tileX,
      tileY: c.tileY,
      blockX: c.blockX,
      activation: c.activation,
      useF16: c.useF16,
      residual: c.residual,
    };

    const variant: ConvVariant = c.variant ?? 'naive';
    const earlyDiagnostics: string[] = [];
    let sharedBytes = 0;
    // Two separate questions, and conflating them is how an unshippable
    // configuration gets labelled portable.
    //
    // Exceeding the *device* limit is fatal - the pipeline will not build - and
    // is surfaced as a diagnostic rather than an exception because the sweeps
    // deliberately walk into configurations that do not fit, and an invalid row
    // carrying the reason is more useful than a missing row.
    //
    // Exceeding the *guaranteed* floor still runs here, because this adapter
    // grants a raised limit, but it would not run on a device that offers only
    // the WebGPU minimum. That is recorded on the result instead of being
    // asserted by hand in the documentation, which is what ADR-0020 requires
    // and what nothing was previously enforcing.
    let requiresRaisedLimit = false;
    const guardShared = (bytes: number): void => {
      sharedBytes = bytes;
      const limit = device.limits.maxComputeWorkgroupStorageSize;
      if (bytes > limit) {
        earlyDiagnostics.push(`workgroup storage ${bytes}B exceeds device limit ${limit}B`);
      } else if (bytes > PORTABLE_WORKGROUP_STORAGE_BYTES) {
        requiresRaisedLimit = true;
      }
    };

    let code: string;
    switch (variant) {
      case 'tiled':
        guardShared(tiledSharedBytes(shaderConfig));
        code = buildTiledConvShader(shaderConfig);
        break;
      case 'packed':
        guardShared(packedSharedBytes(shaderConfig));
        code = buildPackedConvShader(shaderConfig);
        break;
      case 'blocked': {
        const blockedConfig = {
          ...shaderConfig,
          outBlock: c.outBlock ?? 1,
          blockY: c.blockY ?? 1,
          weightLayout: c.weightLayout ?? ('oc-major' as const),
        };
        guardShared(blockedSharedBytes(blockedConfig));
        code = buildBlockedConvShader(blockedConfig);
        break;
      }
      case 'matrix':
        code = buildMatrixConvShader({
          inChannels: c.inChannels,
          outChannels: c.outChannels,
          rowsPerGroup: c.rowsPerGroup ?? 1,
          activation: c.activation,
          useF16: c.useF16,
        });
        break;
      case 'naive':
        code = buildConvShader(shaderConfig);
        break;
    }

    const module = device.createShaderModule({ label: `conv:${c.label}`, code });

    const storage = (elements: number, extraUsage = 0): GPUBuffer =>
      device.createBuffer({
        size: Math.max(4, Math.ceil((elements * bytesPerElement) / 4) * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
      });

    const input = storage(inElements);
    const weights = storage(weightElements);
    const biases = storage(c.outChannels);
    const output = storage(outElements);
    const dims = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(dims, 0, new Uint32Array([c.width, c.height]));

    // Deterministic non-zero data. Values matter for `tanh` timing on some
    // hardware, and all-zero input can be optimised in ways real data is not.
    //
    // The packed kernel reinterprets each vec4 as four channels of one pixel,
    // so feeding it the planar ramp would have it computing a differently
    // permuted convolution from the one the verifier checks. The addresses
    // touched, and therefore the timings, are identical either way — this is
    // about the harness and the verifier agreeing on what is being computed,
    // not about the numbers. Repacking happens once, outside the timed loop.
    const remap =
      variant === 'matrix'
        ? { input: undefined, weights: matrixWeightIndex(c.inChannels, c.outChannels) }
        : variant === 'packed' || variant === 'blocked'
        ? {
            input: packedActivationIndex(c.width * c.height),
            weights:
              variant === 'blocked' && c.weightLayout === 'tap-major'
                ? tapMajorWeightIndex(c.inChannels, c.outChannels)
                : packedWeightIndex(c.inChannels, c.outChannels),
          }
        : null;
    fillDeterministic(device, input, inElements, c.useF16, remap?.input);
    fillDeterministic(device, weights, weightElements, c.useF16, remap?.weights);
    fillDeterministic(device, biases, c.outChannels, c.useF16);

    const pipeline = device.createComputePipeline({
      label: `conv:${c.label}`,
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weights } },
        { binding: 2, resource: { buffer: biases } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: dims } },
      ],
    });

    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolve = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const staging = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const matrixRows = c.rowsPerGroup ?? 1;
    const groupsX =
      variant === 'matrix' ? Math.ceil(c.width / 8) : Math.ceil(c.width / (c.tileX * c.blockX));
    const groupsY =
      variant === 'matrix'
        ? Math.ceil(c.height / matrixRows)
        : Math.ceil(c.height / (c.tileY * (variant === 'blocked' ? (c.blockY ?? 1) : 1)));
    // The blocked variant folds `outBlock` output channels into one
    // invocation, so it needs proportionally fewer z-slices.
    const groupsZ =
      variant === 'blocked'
        ? c.outChannels / (c.outBlock ?? 1)
        : variant === 'matrix'
          ? c.outChannels / 8
          : c.outChannels;

    const dispatch = (withTimestamps: boolean): GPUCommandBuffer => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(
        withTimestamps
          ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
          : {},
      );
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
      pass.end();
      if (withTimestamps) {
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, staging, 0, 16);
      }
      return encoder.finish();
    };

    const compilation = await module.getCompilationInfo();
    const diagnostics = [
      ...earlyDiagnostics,
      ...compilation.messages
        .filter((m) => m.type !== 'info')
        .map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`),
    ];

    // Warm up *inside* the error scopes: a bind group or dispatch rejected at
    // submit time is exactly the failure mode that otherwise shows up only as
    // a zero-length timestamp span.
    for (let i = 0; i < this.warmup; i++) device.queue.submit([dispatch(false)]);
    // Then keep going until the clock has had time to ramp. Batched between
    // fences so a cheap kernel does not spend the whole budget round-tripping
    // to the CPU, and bounded so a pathologically slow case cannot hang.
    const warmupStart = performance.now();
    for (let batch = 0; batch < 256 && performance.now() - warmupStart < this.warmupMs; batch++) {
      for (let i = 0; i < 16; i++) device.queue.submit([dispatch(false)]);
      await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();

    const oom = await device.popErrorScope();
    if (oom) diagnostics.push(`out-of-memory ${oom.message}`);
    const validation = await device.popErrorScope();
    if (validation) diagnostics.push(`validation ${validation.message}`);

    const samples: number[] = [];
    for (let i = 0; i < this.iterations; i++) {
      device.queue.submit([dispatch(true)]);
      await staging.mapAsync(GPUMapMode.READ);
      const [begin, end] = new BigInt64Array(staging.getMappedRange().slice(0));
      staging.unmap();
      if (begin !== undefined && end !== undefined) {
        const ms = Number(end - begin) / NS_PER_MS;
        if (ms >= 0) samples.push(ms);
      }
    }

    for (const b of [input, weights, biases, output, dims, resolve, staging]) b.destroy();
    querySet.destroy();

    samples.sort((a, b) => a - b);
    const median = samples.length > 0 ? (samples[Math.floor(samples.length / 2)] as number) : NaN;
    // An all-zero timestamp span means the dispatch never executed — usually a
    // resource limit rejected the bind group. Reporting 0 ms as a result would
    // be reporting infinite throughput.
    const executed = median > 0;
    const valid = diagnostics.length === 0 && executed;
    if (!executed) diagnostics.push('dispatch produced zero-length GPU timestamps: it did not run');
    // The extent the dispatch actually covers, which is what was paid for.
    //
    // The naive kernel returns before accumulating when an invocation is out of
    // range, so only its blockX tail overhangs. Every other variant guards at
    // the store and accumulates regardless, so their overhang is the full
    // dispatch grid.
    const issued =
      variant === 'naive'
        ? { width: Math.ceil(c.width / c.blockX) * c.blockX, height: c.height }
        : variant === 'matrix'
          ? { width: groupsX * 8, height: groupsY * matrixRows }
          : {
              width: groupsX * c.tileX * c.blockX,
              height: groupsY * c.tileY * (variant === 'blocked' ? (c.blockY ?? 1) : 1),
            };
    const macs = convMacCount(issued.width, issued.height, c.inChannels, c.outChannels);
    // Lower bound: every activation element read once, written once, plus the
    // weights. Real traffic is higher because the 3x3 windows overlap and
    // cache behaviour is not modelled, so treat this as a floor.
    const bytesMoved = (inElements + outElements + weightElements) * bytesPerElement;
    // Every GPU buffer this case allocates: activations, weights, biases, the
    // 8-byte dims uniform and the two 16-byte timestamp buffers.
    const bufferBytes =
      (inElements + outElements + weightElements + c.outChannels) * bytesPerElement + 8 + 16 + 16;

    return {
      ...c,
      diagnostics,
      valid,
      medianMs: median,
      meanMs: samples.length > 0 ? samples.reduce((a, b) => a + b, 0) / samples.length : NaN,
      minMs: samples.length > 0 ? (samples[0] as number) : NaN,
      maxMs: samples.length > 0 ? (samples[samples.length - 1] as number) : NaN,
      iterations: samples.length,
      macs,
      gmacPerSecond: valid ? macs / (median / 1000) / 1e9 : NaN,
      bytesMoved,
      gbPerSecond: valid ? bytesMoved / (median / 1000) / 1e9 : NaN,
      bufferBytes,
      sharedBytes,
      requiresRaisedLimit,
    };
  }
}

/** Maps a planar element index to its destination index in the target layout. */
type LayoutRemap = (planarIndex: number) => number;

/**
 * Planar `[c][y][x]` -> grouped `[c/4][y][x][c%4]`, matching the packed shader.
 * Curried on the geometry so the per-element cost is one closure call.
 */
export function packedActivationIndex(pixels: number): LayoutRemap {
  return (i) => {
    const c = Math.floor(i / pixels);
    const p = i - c * pixels;
    return (Math.floor(c / 4) * pixels + p) * 4 + (c % 4);
  };
}

/** Planar `[oc][ic][k]` -> tap-major grouped `[ic/4][k][oc][ic%4]`. */
export function tapMajorWeightIndex(inChannels: number, outChannels: number): LayoutRemap {
  return (i) => {
    const k = i % 9;
    const ic = Math.floor(i / 9) % inChannels;
    const oc = Math.floor(i / (9 * inChannels)) % outChannels;
    return ((Math.floor(ic / 4) * 9 + k) * outChannels + oc) * 4 + (ic % 4);
  };
}

/** Planar `[oc][ic][k]` -> grouped `[oc][ic/4][k][ic%4]`. */
export function packedWeightIndex(inChannels: number, outChannels: number): LayoutRemap {
  const groups = inChannels / 4;
  return (i) => {
    const k = i % 9;
    const ic = Math.floor(i / 9) % inChannels;
    const oc = Math.floor(i / (9 * inChannels)) % outChannels;
    return ((oc * groups + Math.floor(ic / 4)) * 9 + k) * 4 + (ic % 4);
  };
}

/**
 * Writes a deterministic ramp so runs are comparable and the compiler cannot
 * fold the data away. f16 is written via the half-float bit pattern because
 * `writeBuffer` has no half-float view.
 *
 * `remap` relocates each planar element to the layout the kernel expects. The
 * value written for a given logical element is the same either way, so the
 * value distribution — and therefore anything data-dependent about the timing
 * — is unchanged; only where it lands moves.
 */
function fillDeterministic(
  device: GPUDevice,
  buffer: GPUBuffer,
  elements: number,
  useF16: boolean,
  remap?: LayoutRemap,
): void {
  const at = remap ?? ((i: number) => i);
  const value = (i: number): number => ((i % 17) - 8) / 16;
  if (useF16) {
    // Padded to an even element count: writeBuffer rejects a byte count that
    // is not a multiple of 4, which an odd number of f16 elements produces.
    const half = new Uint16Array(elements + (elements % 2));
    for (let i = 0; i < elements; i++) half[at(i)] = floatToHalf(value(i));
    device.queue.writeBuffer(buffer, 0, half);
    return;
  }
  const full = new Float32Array(elements);
  for (let i = 0; i < elements; i++) full[at(i)] = value(i);
  device.queue.writeBuffer(buffer, 0, full);
}

const halfConverter = new DataView(new ArrayBuffer(4));

/** IEEE-754 binary32 to binary16, round-toward-zero. Adequate for test data. */
export function floatToHalf(value: number): number {
  halfConverter.setFloat32(0, value);
  const bits = halfConverter.getUint32(0);
  const sign = (bits >>> 16) & 0x8000;
  let exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;

  if (exponent === 0xff) return sign | 0x7c00 | (mantissa !== 0 ? 0x200 : 0); // Inf/NaN
  exponent = exponent - 127 + 15;
  if (exponent >= 0x1f) return sign | 0x7c00; // overflow to Inf
  if (exponent <= 0) {
    if (exponent < -10) return sign; // underflow to zero
    mantissa |= 0x800000;
    const shift = 14 - exponent;
    return sign | (mantissa >>> shift);
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}
