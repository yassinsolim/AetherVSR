import { buildBlockedConvShader } from '../core/neural/conv.wgsl.js';
import { buildRepackShader } from './repack.wgsl.js';
import { floatToHalf } from './conv-bench.js';
import { packActivations, packWeights, toTapMajorWeights } from './conv-packed.wgsl.js';
import type { ChainMode } from './chain-bench.js';
import type { Activation } from './conv.wgsl.js';

export interface ChainVerifyCase {
  readonly width: number;
  readonly height: number;
  readonly channels: number;
  readonly layers: number;
  readonly mode: ChainMode;
  readonly activation: Activation;
  readonly useF16?: boolean;
  readonly tileX: number;
  readonly tileY: number;
  readonly blockX: number;
  readonly blockY: number;
  readonly outBlock: number;
}

export interface ChainVerifyResult extends ChainVerifyCase {
  readonly diagnostics: readonly string[];
  readonly maxAbsError: number;
  readonly meanAbsError: number;
  readonly referenceRange: readonly [number, number];
  readonly passed: boolean;
  readonly elements: number;
}

/**
 * Checks a chained convolution against a CPU reference that runs the same
 * layers in the same order.
 *
 * The individual kernel is already verified elsewhere. What is unverified in a
 * chain is the *wiring*: whether ping-pong buffers alternate correctly, whether
 * a layer reads what its predecessor actually wrote, and whether the repack
 * pass lands where the next layer looks. Every one of those bugs produces
 * confident, plausible, wrong numbers — and a chain that reads its own output
 * buffer would even be *faster*, so timing alone would reward the bug.
 *
 * Distinct weights per layer are essential here. With shared weights a
 * transposed or off-by-one layer order can still agree with the reference.
 */
export async function verifyChain(
  device: GPUDevice,
  c: ChainVerifyCase,
  tolerance = 1e-4,
): Promise<ChainVerifyResult> {
  const useF16 = c.useF16 ?? false;
  const bytesPerElement = useF16 ? 2 : 4;
  const pixels = c.width * c.height;
  const activationElements = pixels * c.channels;
  const weightElements = c.channels * c.channels * 9;
  const diagnostics: string[] = [];

  // Distinct, deterministic, small-magnitude data per layer.
  const inputData = new Float32Array(new ArrayBuffer(activationElements * 4));
  for (let i = 0; i < activationElements; i++) inputData[i] = Math.sin(i * 0.7) * 0.25;
  const layerWeights: Float32Array<ArrayBuffer>[] = [];
  const layerBiases: Float32Array<ArrayBuffer>[] = [];
  for (let l = 0; l < c.layers; l++) {
    const w = new Float32Array(new ArrayBuffer(weightElements * 4));
    for (let i = 0; i < weightElements; i++) w[i] = Math.cos((i + l * 31) * 1.3) * 0.08;
    const b = new Float32Array(new ArrayBuffer(c.channels * 4));
    for (let i = 0; i < c.channels; i++) b[i] = ((i + l) % 3) * 0.02 - 0.02;
    layerWeights.push(w);
    layerBiases.push(b);
  }

  const upload = (data: Float32Array<ArrayBuffer>, extra = 0): GPUBuffer => {
    const buf = device.createBuffer({
      size: Math.max(16, Math.ceil((data.length * bytesPerElement) / 16) * 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra,
    });
    if (useF16) {
      const half = new Uint16Array(data.length + (data.length % 2));
      for (let i = 0; i < data.length; i++) half[i] = floatToHalf(data[i] as number);
      device.queue.writeBuffer(buf, 0, half);
    } else {
      device.queue.writeBuffer(buf, 0, data);
    }
    return buf;
  };

  const activationBytes = Math.max(16, Math.ceil((activationElements * bytesPerElement) / 16) * 16);
  const blank = (): GPUBuffer =>
    device.createBuffer({
      size: activationBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

  const ping = upload(packActivations(inputData, c.width, c.height, c.channels), GPUBufferUsage.COPY_SRC);
  const pong = blank();
  const scratch = c.mode === 'repack' ? blank() : null;

  const weightBuffers = layerWeights.map((w) =>
    upload(toTapMajorWeights(packWeights(w, c.channels, c.channels), c.channels, c.channels)),
  );
  const biasBuffers = layerBiases.map((b) => upload(b));

  const dims = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, new Uint32Array([c.width, c.height]));

  device.pushErrorScope('validation');
  const convModule = device.createShaderModule({
    code: buildBlockedConvShader({
      inChannels: c.channels,
      outChannels: c.channels,
      tileX: c.tileX,
      tileY: c.tileY,
      blockX: c.blockX,
      blockY: c.blockY,
      outBlock: c.outBlock,
      activation: c.activation,
      useF16,
      residual: false,
      weightLayout: 'tap-major',
      packedOutput: c.mode === 'packed',
    }),
  });
  const convPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: convModule, entryPoint: 'main' },
  });
  let repackPipeline: GPUComputePipeline | null = null;
  if (c.mode === 'repack') {
    const repackModule = device.createShaderModule({
      code: buildRepackShader({ channels: c.channels, useF16 }),
    });
    repackPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: repackModule, entryPoint: 'main' },
    });
  }
  const validationError = await device.popErrorScope();
  if (validationError) diagnostics.push(`validation ${validationError.message}`);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  const groupsX = Math.ceil(c.width / (c.tileX * c.blockX));
  const groupsY = Math.ceil(c.height / (c.tileY * c.blockY));
  const groupsZ = c.channels / c.outBlock;
  const repackGroupsX = Math.ceil((pixels * (c.channels / 4)) / 64);

  for (let l = 0; l < c.layers; l++) {
    const src = l % 2 === 0 ? ping : pong;
    const dst = l % 2 === 0 ? pong : ping;
    const convOut = c.mode === 'repack' ? (scratch as GPUBuffer) : dst;
    pass.setPipeline(convPipeline);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: convPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: weightBuffers[l] as GPUBuffer } },
          { binding: 2, resource: { buffer: biasBuffers[l] as GPUBuffer } },
          { binding: 3, resource: { buffer: convOut } },
          { binding: 4, resource: { buffer: dims } },
        ],
      }),
    );
    pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
    if (repackPipeline) {
      pass.setPipeline(repackPipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: repackPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: scratch as GPUBuffer } },
            { binding: 1, resource: { buffer: dst } },
            { binding: 2, resource: { buffer: dims } },
          ],
        }),
      );
      pass.dispatchWorkgroups(repackGroupsX, 1, 1);
    }
  }
  pass.end();

  const final = c.layers % 2 === 0 ? ping : pong;
  const readback = device.createBuffer({
    size: activationBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(final, 0, readback, 0, activationBytes);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const rawBytes = readback.getMappedRange().slice(0);
  readback.unmap();
  const grouped = useF16
    ? Float32Array.from(new Uint16Array(rawBytes).subarray(0, activationElements), halfToFloat)
    : new Float32Array(rawBytes).subarray(0, activationElements);

  // Every mode ends in the grouped layout, because that is what a chain must
  // produce for the next layer.
  const actual = unpack(grouped, pixels, c.channels);
  const expected = referenceChain(c, inputData, layerWeights, layerBiases);

  let maxAbs = 0;
  let sumAbs = 0;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < activationElements; i++) {
    const e = expected[i] as number;
    const diff = Math.abs((actual[i] as number) - e);
    if (diff > maxAbs) maxAbs = diff;
    sumAbs += diff;
    if (e < lo) lo = e;
    if (e > hi) hi = e;
  }

  for (const b of [ping, pong, dims, readback, ...weightBuffers, ...biasBuffers]) b.destroy();
  scratch?.destroy();

  return {
    ...c,
    diagnostics,
    maxAbsError: maxAbs,
    meanAbsError: sumAbs / activationElements,
    referenceRange: [lo, hi],
    passed: diagnostics.length === 0 && maxAbs <= tolerance,
    elements: activationElements,
  };
}

/** Runs the same layers on the CPU, in planar layout, in order. */
function referenceChain(
  c: ChainVerifyCase,
  input: Float32Array,
  weights: readonly Float32Array[],
  biases: readonly Float32Array[],
): Float32Array {
  const { width: W, height: H, channels: C } = c;
  let current = input;
  for (let l = 0; l < c.layers; l++) {
    const out = new Float32Array(W * H * C);
    const w = weights[l] as Float32Array;
    const b = biases[l] as Float32Array;
    for (let oc = 0; oc < C; oc++) {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let acc = b[oc] as number;
          for (let ic = 0; ic < C; ic++) {
            const wBase = oc * C * 9 + ic * 9;
            for (let ky = 0; ky < 3; ky++) {
              for (let kx = 0; kx < 3; kx++) {
                const sx = x + kx - 1;
                const sy = y + ky - 1;
                if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
                acc += (w[wBase + ky * 3 + kx] as number) * (current[ic * W * H + sy * W + sx] as number);
              }
            }
          }
          if (c.activation === 'relu') acc = Math.max(acc, 0);
          else if (c.activation === 'tanh') acc = Math.tanh(acc);
          out[oc * W * H + y * W + x] = acc;
        }
      }
    }
    current = out;
  }
  return current;
}

/** Grouped `[c/4][y][x][c%4]` -> planar `[c][y][x]`. */
function unpack(grouped: Float32Array, pixels: number, channels: number): Float32Array {
  const out = new Float32Array(pixels * channels);
  for (let c = 0; c < channels; c++) {
    const group = Math.floor(c / 4);
    const lane = c % 4;
    for (let p = 0; p < pixels; p++) out[c * pixels + p] = grouped[(group * pixels + p) * 4 + lane] as number;
  }
  return out;
}

/** IEEE-754 binary16 to binary32. */
function halfToFloat(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}
