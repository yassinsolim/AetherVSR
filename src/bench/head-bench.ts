import {
  buildUpsampleHeadShader,
  packUpsampleHeadWeights,
} from '../core/neural/upsample-head.wgsl.js';
import { floatToHalf } from './conv-bench.js';

export interface HeadCase {
  readonly label: string;
  /** Low-resolution dimensions. Output is 2x. */
  readonly width: number;
  readonly height: number;
  readonly inChannels: number;
  readonly useF16: boolean;
  readonly blockX: number;
  readonly blockY: number;
  readonly tileX: number;
  readonly tileY: number;
}

export interface HeadResult extends HeadCase {
  readonly diagnostics: readonly string[];
  readonly valid: boolean;
  readonly medianMs: number;
  readonly minMs: number;
  readonly maxMs: number;
  readonly macs: number;
  readonly gmacPerSecond: number;
}

export interface HeadVerifyResult {
  readonly width: number;
  readonly height: number;
  readonly inChannels: number;
  readonly useF16: boolean;
  readonly diagnostics: readonly string[];
  readonly maxAbsError: number;
  readonly passed: boolean;
  /** True when the four sub-positions of a 2x2 block are not all identical. */
  readonly subPixelVaries: boolean;
  readonly outputPixels: number;
}

const NS_PER_MS = 1_000_000;

function makeWeights(inChannels: number): { planar: Float32Array<ArrayBuffer>; bias: Float32Array<ArrayBuffer> } {
  const planar = new Float32Array(new ArrayBuffer(3 * inChannels * 9 * 4));
  for (let i = 0; i < planar.length; i++) planar[i] = Math.cos(i * 1.37) * 0.06;
  const bias = new Float32Array(new ArrayBuffer(3 * 4));
  bias[0] = 0.05;
  bias[1] = -0.02;
  bias[2] = 0.01;
  return { planar, bias };
}

function upload(device: GPUDevice, data: Float32Array<ArrayBuffer>, useF16: boolean): GPUBuffer {
  const bpe = useF16 ? 2 : 4;
  const buf = device.createBuffer({
    size: Math.max(16, Math.ceil((data.length * bpe) / 16) * 16),
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
}

/**
 * CPU reference: nearest-upsample the low-resolution features, then a 3x3
 * convolution with zero padding, then clamp to [0,1].
 */
function referenceHead(
  features: Float32Array,
  W: number,
  H: number,
  inChannels: number,
  weights: Float32Array,
  bias: Float32Array,
): Float32Array {
  const OW = W * 2;
  const OH = H * 2;
  const out = new Float32Array(OW * OH * 3);
  for (let y = 0; y < OH; y++) {
    for (let x = 0; x < OW; x++) {
      for (let col = 0; col < 3; col++) {
        let acc = bias[col] as number;
        for (let ic = 0; ic < inChannels; ic++) {
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              // Offset in high-resolution space, then map down.
              const hx = x + kx - 1;
              const hy = y + ky - 1;
              if (hx < 0 || hy < 0 || hx >= OW || hy >= OH) continue;
              const sx = Math.floor(hx / 2);
              const sy = Math.floor(hy / 2);
              acc +=
                (weights[(col * inChannels + ic) * 9 + ky * 3 + kx] as number) *
                (features[ic * W * H + sy * W + sx] as number);
            }
          }
        }
        out[(y * OW + x) * 3 + col] = Math.min(1, Math.max(0, acc));
      }
    }
  }
  return out;
}

/** Checks the resize-convolution head against the CPU reference. */
export async function verifyUpsampleHead(
  device: GPUDevice,
  c: { width: number; height: number; inChannels: number; useF16?: boolean },
  tolerance = 1 / 255,
): Promise<HeadVerifyResult> {
  const useF16 = c.useF16 ?? false;
  const pixels = c.width * c.height;
  const elements = pixels * c.inChannels;
  const OW = c.width * 2;
  const OH = c.height * 2;
  const diagnostics: string[] = [];

  const planarFeatures = new Float32Array(elements);
  for (let i = 0; i < elements; i++) planarFeatures[i] = Math.sin(i * 0.53) * 0.4;
  const grouped = new Float32Array(new ArrayBuffer(elements * 4));
  for (let ch = 0; ch < c.inChannels; ch++) {
    const g = Math.floor(ch / 4);
    const lane = ch % 4;
    for (let p = 0; p < pixels; p++) {
      grouped[(g * pixels + p) * 4 + lane] = planarFeatures[ch * pixels + p] as number;
    }
  }
  const { planar, bias } = makeWeights(c.inChannels);

  const features = upload(device, grouped, useF16);
  const weights = upload(device, packUpsampleHeadWeights(planar, c.inChannels), useF16);
  const biases = upload(device, bias, useF16);
  const dst = device.createTexture({
    size: { width: OW, height: OH },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const params = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([c.width, c.height]));

  device.pushErrorScope('validation');
  const module = device.createShaderModule({
    code: buildUpsampleHeadShader({
      inChannels: c.inChannels,
      scale: 2,
      useF16,
      format: 'rgba8unorm',
      blockX: 2,
      blockY: 2,
      tileX: 8,
      tileY: 8,
    }),
  });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  for (const m of (await module.getCompilationInfo()).messages) {
    if (m.type !== 'info') diagnostics.push(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
  }

  const bytesPerRow = Math.ceil((OW * 4) / 256) * 256;
  const readback = device.createBuffer({
    size: bytesPerRow * OH,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: features } },
        { binding: 1, resource: { buffer: weights } },
        { binding: 2, resource: { buffer: biases } },
        { binding: 3, resource: dst.createView() },
        { binding: 4, resource: { buffer: params } },
      ],
    }),
  );
  pass.dispatchWorkgroups(Math.ceil(OW / (8 * 2)), Math.ceil(OH / (8 * 2)), 1);
  pass.end();
  encoder.copyTextureToBuffer({ texture: dst }, { buffer: readback, bytesPerRow }, {
    width: OW,
    height: OH,
  });
  device.queue.submit([encoder.finish()]);
  const validation = await device.popErrorScope();
  if (validation) diagnostics.push(`validation ${validation.message}`);

  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(readback.getMappedRange().slice(0));
  readback.unmap();

  const expected = referenceHead(planarFeatures, c.width, c.height, c.inChannels, planar, bias);
  let maxAbs = 0;
  let varies = false;
  for (let y = 0; y < OH; y++) {
    for (let x = 0; x < OW; x++) {
      for (let col = 0; col < 3; col++) {
        const want = expected[(y * OW + x) * 3 + col] as number;
        const got = (bytes[y * bytesPerRow + x * 4 + col] as number) / 255;
        const d = Math.abs(got - want);
        if (d > maxAbs) maxAbs = d;
      }
    }
  }
  // A resize-convolution head must produce four *different* values inside each
  // 2x2 block. If it does not, the upsampling has degenerated into nearest
  // neighbour and the learned head is doing nothing at sub-pixel scale.
  for (let y = 0; y + 1 < OH && !varies; y += 2) {
    for (let x = 0; x + 1 < OW && !varies; x += 2) {
      const a = bytes[y * bytesPerRow + x * 4] as number;
      const b = bytes[y * bytesPerRow + (x + 1) * 4] as number;
      const cc = bytes[(y + 1) * bytesPerRow + x * 4] as number;
      const d = bytes[(y + 1) * bytesPerRow + (x + 1) * 4] as number;
      if (a !== b || a !== cc || a !== d) varies = true;
    }
  }

  for (const b of [features, weights, biases, params, readback]) b.destroy();
  dst.destroy();

  return {
    width: c.width,
    height: c.height,
    inChannels: c.inChannels,
    useF16,
    diagnostics,
    maxAbsError: maxAbs,
    passed: diagnostics.length === 0 && maxAbs <= tolerance && varies,
    subPixelVaries: varies,
    outputPixels: OW * OH,
  };
}

/** Times the resize-convolution head. */
export class HeadBench {
  constructor(
    private readonly device: GPUDevice,
    private readonly iterations = 30,
    private readonly warmupMs = 60,
  ) {}

  async run(cases: readonly HeadCase[]): Promise<HeadResult[]> {
    const out: HeadResult[] = [];
    for (const c of cases) out.push(await this.runCase(c));
    return out;
  }

  private async runCase(c: HeadCase): Promise<HeadResult> {
    const { device } = this;
    const diagnostics: string[] = [];
    const bpe = c.useF16 ? 2 : 4;
    const pixels = c.width * c.height;
    const OW = c.width * 2;
    const OH = c.height * 2;

    device.pushErrorScope('validation');
    const features = device.createBuffer({
      size: Math.ceil((pixels * c.inChannels * bpe) / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const { planar, bias } = makeWeights(c.inChannels);
    const weights = upload(device, packUpsampleHeadWeights(planar, c.inChannels), c.useF16);
    const biases = upload(device, bias, c.useF16);
    const dst = device.createTexture({
      size: { width: OW, height: OH },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING,
    });
    const params = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([c.width, c.height]));

    const module = device.createShaderModule({
      label: `head:${c.label}`,
      code: buildUpsampleHeadShader({
        inChannels: c.inChannels,
        scale: 2,
        useF16: c.useF16,
        format: 'rgba8unorm',
        blockX: c.blockX,
        blockY: c.blockY,
        tileX: c.tileX,
        tileY: c.tileY,
      }),
    });
    const pipeline = device.createComputePipeline({
      label: `head:${c.label}`,
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: features } },
        { binding: 1, resource: { buffer: weights } },
        { binding: 2, resource: { buffer: biases } },
        { binding: 3, resource: dst.createView() },
        { binding: 4, resource: { buffer: params } },
      ],
    });

    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolve = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const gx = Math.ceil(OW / (c.tileX * c.blockX));
    const gy = Math.ceil(OH / (c.tileY * c.blockY));
    const encode = (timed: boolean): GPUCommandBuffer => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(
        timed ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {},
      );
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(gx, gy, 1);
      pass.end();
      if (timed) {
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, staging, 0, 16);
      }
      return encoder.finish();
    };

    const start = performance.now();
    for (let batch = 0; batch < 512 && performance.now() - start < this.warmupMs; batch++) {
      for (let i = 0; i < 8; i++) device.queue.submit([encode(false)]);
      await device.queue.onSubmittedWorkDone();
    }
    const validation = await device.popErrorScope();
    if (validation) diagnostics.push(`validation ${validation.message}`);
    for (const m of (await module.getCompilationInfo()).messages) {
      if (m.type !== 'info') diagnostics.push(`${m.type} ${m.message}`);
    }

    const samples: number[] = [];
    for (let i = 0; i < this.iterations; i++) {
      device.queue.submit([encode(true)]);
      await staging.mapAsync(GPUMapMode.READ);
      const [b0, b1] = new BigInt64Array(staging.getMappedRange().slice(0));
      staging.unmap();
      if (b0 !== undefined && b1 !== undefined) {
        const ms = Number(b1 - b0) / NS_PER_MS;
        if (ms > 0) samples.push(ms);
      }
    }

    for (const b of [features, weights, biases, params, resolve, staging]) b.destroy();
    dst.destroy();
    querySet.destroy();

    samples.sort((a, b) => a - b);
    const median = samples.length > 0 ? (samples[Math.floor(samples.length / 2)] as number) : NaN;
    const executed = median > 0;
    if (!executed) diagnostics.push('head produced zero-length GPU timestamps: it did not run');
    // Issued extent covers the dispatch grid, matching the convolution harness.
    const macs = gx * c.tileX * c.blockX * gy * c.tileY * c.blockY * c.inChannels * 3 * 9;

    return {
      ...c,
      diagnostics,
      valid: diagnostics.length === 0 && executed,
      medianMs: median,
      minMs: samples[0] ?? NaN,
      maxMs: samples[samples.length - 1] ?? NaN,
      macs,
      gmacPerSecond: executed ? macs / (median / 1000) / 1e9 : NaN,
    };
  }
}
