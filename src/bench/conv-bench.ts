import { buildConvShader, convMacCount, type Activation, type ConvShaderConfig } from './conv.wgsl.js';

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
}

const NS_PER_MS = 1_000_000;

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

    const module = device.createShaderModule({
      label: `conv:${c.label}`,
      code: buildConvShader(shaderConfig),
    });

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
    fillDeterministic(device, input, inElements, c.useF16);
    fillDeterministic(device, weights, weightElements, c.useF16);
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

    const groupsX = Math.ceil(c.width / (c.tileX * c.blockX));
    const groupsY = Math.ceil(c.height / c.tileY);
    const groupsZ = c.outChannels;

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
    const diagnostics = compilation.messages
      .filter((m) => m.type !== 'info')
      .map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);

    // Warm up *inside* the error scopes: a bind group or dispatch rejected at
    // submit time is exactly the failure mode that otherwise shows up only as
    // a zero-length timestamp span.
    for (let i = 0; i < this.warmup; i++) device.queue.submit([dispatch(false)]);
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
    const macs = convMacCount(c.width, c.height, c.inChannels, c.outChannels);
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
    };
  }
}

/**
 * Writes a deterministic ramp so runs are comparable and the compiler cannot
 * fold the data away. f16 is written via the half-float bit pattern because
 * `writeBuffer` has no half-float view.
 */
function fillDeterministic(device: GPUDevice, buffer: GPUBuffer, elements: number, useF16: boolean): void {
  if (useF16) {
    // Padded to an even element count: writeBuffer rejects a byte count that
    // is not a multiple of 4, which an odd number of f16 elements produces.
    const half = new Uint16Array(elements + (elements % 2));
    for (let i = 0; i < elements; i++) half[i] = floatToHalf(((i % 17) - 8) / 16);
    device.queue.writeBuffer(buffer, 0, half);
    return;
  }
  const full = new Float32Array(elements);
  for (let i = 0; i < elements; i++) full[i] = ((i % 17) - 8) / 16;
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
