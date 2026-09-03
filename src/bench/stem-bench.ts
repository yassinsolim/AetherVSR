import { buildStemShader, packStemWeights } from '../core/neural/stem.wgsl.js';
import { buildRgbPackShader } from '../core/neural/rgb-pack.wgsl.js';
import { buildBlockedConvShader } from './conv-blocked.wgsl.js';
import { floatToHalf } from './conv-bench.js';
import { packWeights, toTapMajorWeights } from './conv-packed.wgsl.js';
import type { IngestFormat } from '../core/ingest/external-texture-ingest.js';

/**
 * How `RGB texture -> C_out packed activations` is performed.
 *
 * - `fused`   — one texture-native 3x3 convolution.
 * - `two-pass`— convert the texture to a packed C4 buffer, then run the
 *   generic C4 -> C_out convolution.
 */
export type StemMode = 'fused' | 'two-pass';

export interface StemCase {
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly outChannels: number;
  readonly mode: StemMode;
  readonly format: IngestFormat;
  readonly useF16: boolean;
  readonly blockX: number;
  readonly blockY: number;
  readonly tileX: number;
  readonly tileY: number;
  /** Kernel size, odd. Defaults to 3. */
  readonly kernel?: number;
}

export interface StemResult extends StemCase {
  readonly diagnostics: readonly string[];
  readonly valid: boolean;
  readonly medianMs: number;
  readonly meanMs: number;
  readonly minMs: number;
  readonly maxMs: number;
  readonly iterations: number;
  readonly dispatches: number;
  /** Persistent GPU bytes for this stage: activations, weights, biases. */
  readonly persistentBytes: number;
}

const NS_PER_MS = 1_000_000;

/**
 * Times the bridge from an ingest-format texture to packed neural activations.
 *
 * The source is a real texture in the format the production ingest stage
 * writes, filled once. It is not a `GPUExternalTexture`: the external-texture
 * conversion is already measured separately by `IngestBench`, and including it
 * here would conflate two independently variable costs. The combined entry cost
 * is reported by adding the two *measured* figures only where they are encoded
 * as separate passes, never by re-attributing one to the other.
 */
export class StemBench {
  constructor(
    private readonly device: GPUDevice,
    private readonly iterations = 30,
    private readonly warmupMs = 60,
  ) {}

  async run(cases: readonly StemCase[]): Promise<StemResult[]> {
    const out: StemResult[] = [];
    for (const c of cases) out.push(await this.runCase(c));
    return out;
  }

  private async runCase(c: StemCase): Promise<StemResult> {
    const { device } = this;
    device.pushErrorScope('validation');
    const diagnostics: string[] = [];
    const bytesPerElement = c.useF16 ? 2 : 4;
    const pixels = c.width * c.height;

    const src = device.createTexture({
      label: 'stem:source',
      size: { width: c.width, height: c.height },
      format: c.format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    writeGradient(device, src, c.width, c.height, c.format);

    const storage = (bytes: number): GPUBuffer =>
      device.createBuffer({
        size: Math.max(16, Math.ceil(bytes / 16) * 16),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

    const activations = storage(pixels * c.outChannels * bytesPerElement);
    const params = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    writeParams(device, params, c.width, c.height);

    const biasElements = c.outChannels;
    const biases = storage(biasElements * bytesPerElement);
    fillRamp(device, biases, biasElements, c.useF16, 0.01);

    let dispatches = 0;
    let persistentBytes = pixels * c.outChannels * bytesPerElement + biasElements * bytesPerElement;
    const encodeOps: ((pass: GPUComputePassEncoder) => void)[] = [];
    const owned: GPUBuffer[] = [activations, params, biases];

    if (c.mode === 'fused') {
      const kernel = c.kernel ?? 3;
      const taps = kernel * kernel;
      const weightElements = c.outChannels * taps * 4;
      const weights = storage(weightElements * bytesPerElement);
      const planar = new Float32Array(new ArrayBuffer(c.outChannels * 3 * taps * 4));
      for (let i = 0; i < planar.length; i++) planar[i] = Math.cos(i * 1.1) * 0.1;
      uploadFloats(device, weights, packStemWeights(planar, c.outChannels, kernel), c.useF16);
      owned.push(weights);
      persistentBytes += weightElements * bytesPerElement;

      const module = device.createShaderModule({
        label: `stem:fused:${c.label}`,
        code: buildStemShader({
          outChannels: c.outChannels,
          blockX: c.blockX,
          blockY: c.blockY,
          tileX: c.tileX,
          tileY: c.tileY,
          useF16: c.useF16,
          activation: 'relu',
          kernel,
        }),
      });
      const pipeline = device.createComputePipeline({
        label: `stem:fused:${c.label}`,
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: src.createView() },
          { binding: 1, resource: { buffer: weights } },
          { binding: 2, resource: { buffer: biases } },
          { binding: 3, resource: { buffer: activations } },
          { binding: 4, resource: { buffer: params } },
        ],
      });
      const gx = Math.ceil(c.width / (c.tileX * c.blockX));
      const gy = Math.ceil(c.height / (c.tileY * c.blockY));
      encodeOps.push((pass) => {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(gx, gy, 1);
      });
      dispatches = 1;
      for (const m of (await module.getCompilationInfo()).messages) {
        if (m.type !== 'info') diagnostics.push(`${m.type} ${m.message}`);
      }
    } else {
      // Pass 1: texture -> packed C4.
      const c4 = storage(pixels * 4 * bytesPerElement);
      owned.push(c4);
      persistentBytes += pixels * 4 * bytesPerElement;
      const packModule = device.createShaderModule({
        label: `stem:pack:${c.label}`,
        code: buildRgbPackShader({ useF16: c.useF16 }),
      });
      const packPipeline = device.createComputePipeline({
        label: `stem:pack:${c.label}`,
        layout: 'auto',
        compute: { module: packModule, entryPoint: 'main' },
      });
      const packGroup = device.createBindGroup({
        layout: packPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: src.createView() },
          { binding: 1, resource: { buffer: c4 } },
          { binding: 2, resource: { buffer: params } },
        ],
      });

      // Pass 2: generic C4 -> C_out convolution.
      const weightElements = 4 * c.outChannels * 9;
      const weights = storage(weightElements * bytesPerElement);
      const planar = new Float32Array(new ArrayBuffer(weightElements * 4));
      for (let i = 0; i < planar.length; i++) planar[i] = Math.cos(i * 1.1) * 0.1;
      uploadFloats(
        device,
        weights,
        toTapMajorWeights(packWeights(planar, 4, c.outChannels), 4, c.outChannels),
        c.useF16,
      );
      owned.push(weights);
      persistentBytes += weightElements * bytesPerElement;

      const dims = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(dims, 0, new Uint32Array([c.width, c.height]));
      owned.push(dims);

      const outBlock = Math.min(c.outChannels, 16);
      const convModule = device.createShaderModule({
        label: `stem:conv:${c.label}`,
        code: buildBlockedConvShader({
          inChannels: 4,
          outChannels: c.outChannels,
          tileX: c.tileX,
          tileY: c.tileY,
          blockX: c.blockX,
          blockY: c.blockY,
          outBlock,
          activation: 'relu',
          useF16: c.useF16,
          residual: false,
          weightLayout: 'tap-major',
          packedOutput: true,
        }),
      });
      const convPipeline = device.createComputePipeline({
        label: `stem:conv:${c.label}`,
        layout: 'auto',
        compute: { module: convModule, entryPoint: 'main' },
      });
      const convGroup = device.createBindGroup({
        layout: convPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: c4 } },
          { binding: 1, resource: { buffer: weights } },
          { binding: 2, resource: { buffer: biases } },
          { binding: 3, resource: { buffer: activations } },
          { binding: 4, resource: { buffer: dims } },
        ],
      });

      const packGroups = Math.ceil(pixels / 64);
      const gx = Math.ceil(c.width / (c.tileX * c.blockX));
      const gy = Math.ceil(c.height / (c.tileY * c.blockY));
      encodeOps.push((pass) => {
        pass.setPipeline(packPipeline);
        pass.setBindGroup(0, packGroup);
        pass.dispatchWorkgroups(packGroups, 1, 1);
      });
      encodeOps.push((pass) => {
        pass.setPipeline(convPipeline);
        pass.setBindGroup(0, convGroup);
        pass.dispatchWorkgroups(gx, gy, c.outChannels / outBlock);
      });
      dispatches = 2;
      for (const mod of [packModule, convModule]) {
        for (const m of (await mod.getCompilationInfo()).messages) {
          if (m.type !== 'info') diagnostics.push(`${m.type} ${m.message}`);
        }
      }
    }

    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolve = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const encode = (timed: boolean): GPUCommandBuffer => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(
        timed ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {},
      );
      for (const op of encodeOps) op(pass);
      pass.end();
      if (timed) {
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, staging, 0, 16);
      }
      return encoder.finish();
    };

    const start = performance.now();
    for (let batch = 0; batch < 512 && performance.now() - start < this.warmupMs; batch++) {
      for (let i = 0; i < 16; i++) device.queue.submit([encode(false)]);
      await device.queue.onSubmittedWorkDone();
    }
    const validation = await device.popErrorScope();
    if (validation) diagnostics.push(`validation ${validation.message}`);

    const samples: number[] = [];
    for (let i = 0; i < this.iterations; i++) {
      device.queue.submit([encode(true)]);
      await staging.mapAsync(GPUMapMode.READ);
      const [begin, end] = new BigInt64Array(staging.getMappedRange().slice(0));
      staging.unmap();
      if (begin !== undefined && end !== undefined) {
        const ms = Number(end - begin) / NS_PER_MS;
        if (ms > 0) samples.push(ms);
      }
    }

    for (const b of [...owned, resolve, staging]) b.destroy();
    src.destroy();
    querySet.destroy();

    samples.sort((a, b) => a - b);
    const median = samples.length > 0 ? (samples[Math.floor(samples.length / 2)] as number) : NaN;
    const executed = median > 0;
    if (!executed) diagnostics.push('stem produced zero-length GPU timestamps: it did not run');

    return {
      ...c,
      diagnostics,
      valid: diagnostics.length === 0 && executed,
      medianMs: median,
      meanMs: samples.length > 0 ? samples.reduce((a, b) => a + b, 0) / samples.length : NaN,
      minMs: samples.length > 0 ? (samples[0] as number) : NaN,
      maxMs: samples.length > 0 ? (samples[samples.length - 1] as number) : NaN,
      iterations: samples.length,
      dispatches,
      persistentBytes,
    };
  }
}

function writeParams(device: GPUDevice, buffer: GPUBuffer, width: number, height: number): void {
  const data = new ArrayBuffer(48);
  new Uint32Array(data, 0, 2).set([width, height]);
  new Float32Array(data, 16, 4).set([0, 0, 0, 0]);
  new Float32Array(data, 32, 4).set([1, 1, 1, 1]);
  device.queue.writeBuffer(buffer, 0, data);
}

/** Deterministic gradient, so the texture is neither uniform nor random. */
function writeGradient(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  format: IngestFormat,
): void {
  if (format === 'rgba8unorm') {
    const data = new Uint8Array(width * height * 4);
    for (let i = 0, p = 0; p < width * height; p++) {
      const x = p % width;
      const y = Math.floor(p / width);
      data[i++] = (x * 7) & 0xff;
      data[i++] = (y * 5) & 0xff;
      data[i++] = ((x + y) * 3) & 0xff;
      data[i++] = 255;
    }
    device.queue.writeTexture({ texture }, data, { bytesPerRow: width * 4 }, { width, height });
    return;
  }
  const data = new Uint16Array(width * height * 4);
  for (let i = 0, p = 0; p < width * height; p++) {
    const x = p % width;
    const y = Math.floor(p / width);
    data[i++] = floatToHalf(((x * 7) & 0xff) / 255);
    data[i++] = floatToHalf(((y * 5) & 0xff) / 255);
    data[i++] = floatToHalf((((x + y) * 3) & 0xff) / 255);
    data[i++] = floatToHalf(1);
  }
  device.queue.writeTexture({ texture }, data, { bytesPerRow: width * 8 }, { width, height });
}

function uploadFloats(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: Float32Array<ArrayBuffer>,
  useF16: boolean,
): void {
  if (useF16) {
    const half = new Uint16Array(data.length + (data.length % 2));
    for (let i = 0; i < data.length; i++) half[i] = floatToHalf(data[i] as number);
    device.queue.writeBuffer(buffer, 0, half);
    return;
  }
  device.queue.writeBuffer(buffer, 0, data);
}

function fillRamp(
  device: GPUDevice,
  buffer: GPUBuffer,
  elements: number,
  useF16: boolean,
  scale: number,
): void {
  const value = (i: number): number => (((i % 17) - 8) / 16) * scale;
  if (useF16) {
    const half = new Uint16Array(elements + (elements % 2));
    for (let i = 0; i < elements; i++) half[i] = floatToHalf(value(i));
    device.queue.writeBuffer(buffer, 0, half);
    return;
  }
  const full = new Float32Array(elements);
  for (let i = 0; i < elements; i++) full[i] = value(i);
  device.queue.writeBuffer(buffer, 0, full);
}
