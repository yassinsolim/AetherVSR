import { floatToHalf, type ConvVariant } from './conv-bench.js';
import { buildConvShader, type Activation } from './conv.wgsl.js';
import { buildTiledConvShader } from './conv-tiled.wgsl.js';
import { buildPackedConvShader, packActivations, packWeights } from './conv-packed.wgsl.js';

export interface ConvVerifyCase {
  /** Verify the f16 variant of the shader instead of the f32 one. */
  readonly useF16?: boolean;
  readonly width: number;
  readonly height: number;
  readonly inChannels: number;
  readonly outChannels: number;
  readonly blockX: number;
  readonly activation: Activation;
  readonly residual: boolean;
  /** Which kernel implementation to check. Defaults to `naive`. */
  readonly variant?: ConvVariant;
  /** Workgroup shape. Defaults to 8x8. Tiling bugs are shape-dependent, so
   *  the tiled kernel must be verified at the shapes it is benchmarked at. */
  readonly tileX?: number;
  readonly tileY?: number;
}

export interface ConvVerifyResult extends ConvVerifyCase {
  /** WGSL compile diagnostics, and any validation error from pipeline creation. */
  readonly diagnostics: readonly string[];
  readonly maxAbsError: number;
  readonly meanAbsError: number;
  readonly referenceRange: readonly [number, number];
  readonly passed: boolean;
  readonly elements: number;
}

/**
 * Checks the convolution shader against a CPU reference.
 *
 * A throughput number from an incorrect kernel is worse than no number: it is
 * fast precisely because it is skipping work. This runs the same shader the
 * benchmark uses, on a small tensor, and compares every output element with a
 * straightforward CPU implementation of the same arithmetic.
 *
 * This deliberately reads GPU memory back to the CPU. That is forbidden in the
 * per-frame hot path and irrelevant here — this is an offline correctness
 * check, not a frame path.
 *
 * `useF16` verifies the half-precision variant against the same f32 CPU
 * reference. That comparison cannot use the f32 tolerance — half precision has
 * ~3 decimal digits and the accumulator rounds at every one of the
 * `9 * inChannels` MACs — so the caller must pass a tolerance appropriate to
 * the accumulation depth. The point is to catch a *wrong* f16 kernel, not to
 * pretend f16 is exact.
 */
export async function verifyConv(
  device: GPUDevice,
  c: ConvVerifyCase,
  tolerance = 1e-4,
): Promise<ConvVerifyResult> {
  const useF16 = c.useF16 ?? false;
  const bytesPerElement = useF16 ? 2 : 4;
  const pixels = c.width * c.height;
  const inElements = pixels * c.inChannels;
  const outElements = pixels * c.outChannels;
  const weightElements = c.inChannels * c.outChannels * 9;

  const inputData = new Float32Array(new ArrayBuffer(inElements * 4));
  for (let i = 0; i < inElements; i++) inputData[i] = Math.sin(i * 0.7) * 0.5;
  const weightData = new Float32Array(new ArrayBuffer(weightElements * 4));
  for (let i = 0; i < weightElements; i++) weightData[i] = Math.cos(i * 1.3) * 0.25;
  const biasData = new Float32Array(new ArrayBuffer(c.outChannels * 4));
  for (let i = 0; i < c.outChannels; i++) biasData[i] = (i % 3) * 0.1 - 0.1;

  const makeBuffer = (data: Float32Array<ArrayBuffer>, extra = 0): GPUBuffer => {
    const buf = device.createBuffer({
      size: Math.max(4, roundUp4(data.length * bytesPerElement)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra,
    });
    if (useF16) {
      // writeBuffer requires a byte count that is a multiple of 4, and an odd
      // number of f16 elements is not. Pad to an even element count.
      const half = new Uint16Array(data.length + (data.length % 2));
      for (let i = 0; i < data.length; i++) half[i] = floatToHalf(data[i] as number);
      device.queue.writeBuffer(buf, 0, half);
    } else {
      device.queue.writeBuffer(buf, 0, data);
    }
    return buf;
  };

  // The CPU reference always works in planar layout; only what the GPU is fed
  // changes. Comparing a packed kernel against a planar reference is the point
  // — a repacking bug then shows up as a numeric mismatch rather than hiding
  // inside a reference that was repacked the same wrong way.
  const packed = (c.variant ?? 'naive') === 'packed';
  const input = makeBuffer(
    packed ? packActivations(inputData, c.width, c.height, c.inChannels) : inputData,
  );
  const weights = makeBuffer(
    packed ? packWeights(weightData, c.inChannels, c.outChannels) : weightData,
  );
  const biases = makeBuffer(biasData);
  const output = device.createBuffer({
    size: Math.max(4, roundUp4(outElements * bytesPerElement)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const dims = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, new Uint32Array([c.width, c.height]));

  const tileX = c.tileX ?? 8;
  const tileY = c.tileY ?? 8;
  const shaderConfig = {
    inChannels: c.inChannels,
    outChannels: c.outChannels,
    tileX,
    tileY,
    blockX: c.blockX,
    activation: c.activation,
    useF16,
    residual: c.residual,
  };

  const buildShader = (): string => {
    switch (c.variant ?? 'naive') {
      case 'tiled':
        return buildTiledConvShader(shaderConfig);
      case 'packed':
        return buildPackedConvShader(shaderConfig);
      case 'naive':
        return buildConvShader(shaderConfig);
    }
  };

  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code: buildShader() });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });

  // A WGSL scope or type error makes pipeline creation fail; the dispatch then
  // writes nothing and the output buffer stays zero, which reads as a
  // plausible-but-wrong result. Surface it instead.
  const compilation = await module.getCompilationInfo();
  const diagnostics = compilation.messages
    .filter((m) => m.type !== 'info')
    .map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
  const validationError = await device.popErrorScope();
  if (validationError) diagnostics.push(`validation ${validationError.message}`);
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

  const readback = device.createBuffer({
    size: Math.max(4, roundUp4(outElements * bytesPerElement)),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(c.width / (tileX * c.blockX)),
    Math.ceil(c.height / tileY),
    c.outChannels,
  );
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, roundUp4(outElements * bytesPerElement));
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const rawBytes = readback.getMappedRange().slice(0);
  readback.unmap();
  const actual = useF16
    ? Float32Array.from(new Uint16Array(rawBytes).subarray(0, outElements), halfToFloat)
    : new Float32Array(rawBytes);

  const expected = referenceConv(c, inputData, weightData, biasData);

  let maxAbs = 0;
  let sumAbs = 0;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < outElements; i++) {
    const e = expected[i] as number;
    const diff = Math.abs((actual[i] as number) - e);
    if (diff > maxAbs) maxAbs = diff;
    sumAbs += diff;
    if (e < lo) lo = e;
    if (e > hi) hi = e;
  }

  for (const b of [input, weights, biases, output, dims, readback]) b.destroy();

  return {
    ...c,
    diagnostics,
    maxAbsError: maxAbs,
    meanAbsError: sumAbs / outElements,
    referenceRange: [lo, hi],
    passed: diagnostics.length === 0 && maxAbs <= tolerance,
    elements: outElements,
  };
}

/** Straightforward CPU 3x3 convolution with zero padding, planar layout. */
function referenceConv(
  c: ConvVerifyCase,
  input: Float32Array,
  weights: Float32Array,
  biases: Float32Array,
): Float32Array {
  const { width: W, height: H, inChannels: IC, outChannels: OC } = c;
  const out = new Float32Array(W * H * OC);
  for (let oc = 0; oc < OC; oc++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let acc = biases[oc] as number;
        for (let ic = 0; ic < IC; ic++) {
          const wBase = oc * IC * 9 + ic * 9;
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              const sx = x + kx - 1;
              const sy = y + ky - 1;
              if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
              acc += (weights[wBase + ky * 3 + kx] as number) * (input[ic * W * H + sy * W + sx] as number);
            }
          }
        }
        let value = acc;
        if (c.activation === 'relu') value = Math.max(value, 0);
        else if (c.activation === 'tanh') value = Math.tanh(value);
        if (c.residual) value += input[oc * W * H + y * W + x] as number;
        out[oc * W * H + y * W + x] = value;
      }
    }
  }
  return out;
}

/** Rounds a byte count up to the 4-byte granularity WebGPU copies require. */
function roundUp4(bytes: number): number {
  return Math.ceil(bytes / 4) * 4;
}

/** IEEE-754 binary16 to binary32, for reading back f16 GPU output. */
function halfToFloat(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}
