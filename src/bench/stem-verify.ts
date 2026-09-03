import { buildStemShader, packStemWeights } from '../core/neural/stem.wgsl.js';
import { floatToHalf } from './conv-bench.js';

export interface StemVerifyCase {
  readonly width: number;
  readonly height: number;
  readonly outChannels: number;
  readonly useF16?: boolean;
  readonly blockX: number;
  readonly blockY: number;
  readonly tileX: number;
  readonly tileY: number;
  readonly activation: 'none' | 'relu';
  /** Kernel size, odd. Defaults to 3. */
  readonly kernel?: number;
  /** Per-channel input transform: `(sample - mean) * scale`. */
  readonly mean?: readonly [number, number, number];
  readonly scale?: readonly [number, number, number];
}

export interface StemVerifyResult extends StemVerifyCase {
  readonly diagnostics: readonly string[];
  readonly maxAbsError: number;
  readonly meanAbsError: number;
  readonly referenceRange: readonly [number, number];
  readonly passed: boolean;
  readonly elements: number;
}

/**
 * Checks the fused stem against a CPU reference.
 *
 * Three things can be wrong here that the convolution tests do not cover: the
 * texel fetch and its zero padding, the normalisation transform, and the
 * `[oc][tap]` vec4 weight packing with its zeroed alpha lane. The reference
 * works from the same 8-bit texel values the GPU sees, so a mismatch is a
 * kernel bug and not a disagreement about what the texture contained.
 */
export async function verifyStem(
  device: GPUDevice,
  c: StemVerifyCase,
  tolerance = 1e-4,
): Promise<StemVerifyResult> {
  const useF16 = c.useF16 ?? false;
  const bytesPerElement = useF16 ? 2 : 4;
  const pixels = c.width * c.height;
  const outElements = pixels * c.outChannels;
  const diagnostics: string[] = [];
  const mean = c.mean ?? [0, 0, 0];
  const scale = c.scale ?? [1, 1, 1];

  // Deterministic 8-bit texel content, held exactly so the reference and the
  // shader agree on the input bit for bit.
  const texels = new Uint8Array(pixels * 4);
  for (let p = 0; p < pixels; p++) {
    const x = p % c.width;
    const y = Math.floor(p / c.width);
    texels[p * 4 + 0] = (x * 37 + y * 11) & 0xff;
    texels[p * 4 + 1] = (x * 5 + y * 61) & 0xff;
    texels[p * 4 + 2] = (x * 97 + y * 3) & 0xff;
    texels[p * 4 + 3] = 200; // deliberately not 255: alpha must not contribute
  }
  const texture = device.createTexture({
    size: { width: c.width, height: c.height },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, texels, { bytesPerRow: c.width * 4 }, {
    width: c.width,
    height: c.height,
  });

  const kernel = c.kernel ?? 3;
  const taps = kernel * kernel;
  const weightPlanar = new Float32Array(new ArrayBuffer(c.outChannels * 3 * taps * 4));
  for (let i = 0; i < weightPlanar.length; i++) weightPlanar[i] = Math.cos(i * 1.7) * 0.2;
  const biasData = new Float32Array(new ArrayBuffer(c.outChannels * 4));
  for (let i = 0; i < c.outChannels; i++) biasData[i] = (i % 3) * 0.05 - 0.05;

  const upload = (data: Float32Array<ArrayBuffer>): GPUBuffer => {
    const buf = device.createBuffer({
      size: Math.max(16, Math.ceil((data.length * bytesPerElement) / 16) * 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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

  const weights = upload(packStemWeights(weightPlanar, c.outChannels, kernel));
  const biases = upload(biasData);
  const outBytes = Math.max(16, Math.ceil((outElements * bytesPerElement) / 16) * 16);
  const output = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  // Poison, so an output the dispatch never wrote fails rather than matching a
  // zero-initialised buffer against a relu reference that is often zero.
  if (useF16) {
    device.queue.writeBuffer(output, 0, new Uint16Array(outBytes / 2).fill(0xfbff));
  } else {
    device.queue.writeBuffer(output, 0, new Float32Array(outBytes / 4).fill(-1e4));
  }

  const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const paramData = new ArrayBuffer(48);
  new Uint32Array(paramData, 0, 2).set([c.width, c.height]);
  new Float32Array(paramData, 16, 4).set([mean[0], mean[1], mean[2], 0]);
  new Float32Array(paramData, 32, 4).set([scale[0], scale[1], scale[2], 0]);
  device.queue.writeBuffer(params, 0, paramData);

  device.pushErrorScope('validation');
  const module = device.createShaderModule({
    code: buildStemShader({
      outChannels: c.outChannels,
      blockX: c.blockX,
      blockY: c.blockY,
      tileX: c.tileX,
      tileY: c.tileY,
      useF16,
      activation: c.activation,
      kernel,
    }),
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  for (const m of (await module.getCompilationInfo()).messages) {
    if (m.type !== 'info') diagnostics.push(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
  }
  const validation = await device.popErrorScope();
  if (validation) diagnostics.push(`validation ${validation.message}`);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: { buffer: weights } },
        { binding: 2, resource: { buffer: biases } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    }),
  );
  pass.dispatchWorkgroups(
    Math.ceil(c.width / (c.tileX * c.blockX)),
    Math.ceil(c.height / (c.tileY * c.blockY)),
    1,
  );
  pass.end();

  const readback = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(output, 0, readback, 0, outBytes);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const raw = readback.getMappedRange().slice(0);
  readback.unmap();
  const grouped = useF16
    ? Float32Array.from(new Uint16Array(raw).subarray(0, outElements), halfToFloat)
    : new Float32Array(raw).subarray(0, outElements);

  // Grouped -> planar, so the comparison is against a plain reference.
  const actual = new Float32Array(outElements);
  for (let ch = 0; ch < c.outChannels; ch++) {
    const g = Math.floor(ch / 4);
    const lane = ch % 4;
    for (let p = 0; p < pixels; p++) actual[ch * pixels + p] = grouped[(g * pixels + p) * 4 + lane] as number;
  }

  const expected = referenceStem(c, texels, weightPlanar, biasData, mean, scale, kernel);

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

  for (const b of [weights, biases, output, params, readback]) b.destroy();
  texture.destroy();

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

/** CPU 3x3 RGB -> C_out convolution with zero padding, planar output. */
function referenceStem(
  c: StemVerifyCase,
  texels: Uint8Array,
  weights: Float32Array,
  biases: Float32Array,
  mean: readonly [number, number, number],
  scale: readonly [number, number, number],
  kernel: number,
): Float32Array {
  const { width: W, height: H, outChannels: OC } = c;
  const pad = (kernel - 1) / 2;
  const taps = kernel * kernel;
  const out = new Float32Array(W * H * OC);
  const sample = (x: number, y: number, ch: number): number => {
    if (x < 0 || y < 0 || x >= W || y >= H) return 0;
    const raw = (texels[(y * W + x) * 4 + ch] as number) / 255;
    return (raw - (mean[ch] as number)) * (scale[ch] as number);
  };
  for (let oc = 0; oc < OC; oc++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let acc = biases[oc] as number;
        for (let ic = 0; ic < 3; ic++) {
          for (let ky = 0; ky < kernel; ky++) {
            for (let kx = 0; kx < kernel; kx++) {
              const w = weights[(oc * 3 + ic) * taps + ky * kernel + kx] as number;
              const sx = x + kx - pad;
              const sy = y + ky - pad;
              // Out-of-range taps contribute zero, which is what the shader's
              // fetch returns.
              if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
              acc += w * sample(sx, sy, ic);
            }
          }
        }
        out[oc * W * H + y * W + x] = c.activation === 'relu' ? Math.max(acc, 0) : acc;
      }
    }
  }
  return out;
}

function halfToFloat(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}
