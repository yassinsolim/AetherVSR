import { buildBlockedConvShader } from './conv-blocked.wgsl.js';
import { buildRepackShader } from './repack.wgsl.js';
import { floatToHalf } from './conv-bench.js';
import type { Activation } from './conv.wgsl.js';

/**
 * How consecutive convolution layers hand activations to one another.
 *
 * - `packed`  — each layer writes the grouped `vec4` layout the next one reads.
 * - `repack`  — each layer writes scalar planar and a separate pass converts it.
 *
 * The second exists because removing a pass is not automatically a win: the
 * packed store may cost the convolution more than the pass it saves. That is an
 * empirical question and this harness is how it gets answered.
 */
export type ChainMode = 'packed' | 'repack';

export interface ChainCase {
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly channels: number;
  readonly layers: number;
  readonly mode: ChainMode;
  readonly activation: Activation;
  /** Applies a residual add on every layer after the first. */
  readonly residual: boolean;
  readonly useF16: boolean;
  readonly tileX: number;
  readonly tileY: number;
  readonly blockX: number;
  readonly blockY: number;
  readonly outBlock: number;
}

export interface ChainResult extends ChainCase {
  readonly diagnostics: readonly string[];
  readonly valid: boolean;
  /** Median GPU time for the whole chain, in milliseconds. */
  readonly medianMs: number;
  readonly meanMs: number;
  readonly minMs: number;
  readonly maxMs: number;
  readonly iterations: number;
  /** Total issued MACs across every layer in the chain. */
  readonly macs: number;
  readonly gmacPerSecond: number;
  /** Persistent GPU bytes: activations, weights and biases. */
  readonly persistentBytes: number;
  /** Dispatches encoded per chain, including any repack passes. */
  readonly dispatches: number;
}

const NS_PER_MS = 1_000_000;

/**
 * Times a chain of convolution layers sharing preallocated ping-pong buffers.
 *
 * ## Why this is not N times the single-layer number
 *
 * A lone dispatch reads an input that nothing else just wrote, into caches that
 * nothing else just evicted. Layer *k* of a chain reads exactly what layer
 * *k-1* wrote, which is a different memory system state, and the whole working
 * set is now two full activation buffers instead of one. Whether that helps
 * (the data is hot) or hurts (it does not fit) is not predictable from the
 * isolated measurement, which is the entire reason this file exists.
 *
 * ## Structure
 *
 * Every buffer, bind group and pipeline is created in the constructor path and
 * reused. The timed region encodes one compute pass containing every layer's
 * dispatch back to back; WebGPU orders dispatches within a pass and makes each
 * one's writes visible to the next, so no explicit barrier is needed and none
 * is available to add.
 *
 * No CPU readback occurs. The only mapped buffer is the timestamp resolve
 * target.
 */
export class ChainBench {
  constructor(
    private readonly device: GPUDevice,
    private readonly iterations = 30,
    private readonly warmupMs = 60,
  ) {}

  async run(cases: readonly ChainCase[]): Promise<ChainResult[]> {
    const results: ChainResult[] = [];
    for (const c of cases) results.push(await this.runCase(c));
    return results;
  }

  private async runCase(c: ChainCase): Promise<ChainResult> {
    const { device } = this;
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');

    const bytesPerElement = c.useF16 ? 2 : 4;
    const pixels = c.width * c.height;
    const activationElements = pixels * c.channels;
    const activationBytes = activationElements * bytesPerElement;
    const weightElements = c.channels * c.channels * 9;
    const diagnostics: string[] = [];

    // --- persistent resources -------------------------------------------
    const storage = (bytes: number): GPUBuffer =>
      device.createBuffer({
        size: Math.max(16, Math.ceil(bytes / 16) * 16),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

    // Ping-pong pair. `repack` mode needs a third: the scalar planar staging
    // area a layer writes before the repack pass groups it.
    const ping = storage(activationBytes);
    const pong = storage(activationBytes);
    const scratch = c.mode === 'repack' ? storage(activationBytes) : null;

    const weights: GPUBuffer[] = [];
    const biases: GPUBuffer[] = [];
    for (let i = 0; i < c.layers; i++) {
      // Distinct weights per layer. Sharing one buffer would let the cache hold
      // it across the whole chain, which no real network enjoys.
      const w = storage(weightElements * bytesPerElement);
      const b = storage(c.channels * bytesPerElement);
      fillRamp(device, w, weightElements, c.useF16, 0.05);
      fillRamp(device, b, c.channels, c.useF16, 0.01);
      weights.push(w);
      biases.push(b);
    }
    fillRamp(device, ping, activationElements, c.useF16, 0.25);

    const dims = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dims, 0, new Uint32Array([c.width, c.height]));

    // --- pipelines --------------------------------------------------------
    const convCode = buildBlockedConvShader({
      inChannels: c.channels,
      outChannels: c.channels,
      tileX: c.tileX,
      tileY: c.tileY,
      blockX: c.blockX,
      blockY: c.blockY,
      outBlock: c.outBlock,
      activation: c.activation,
      useF16: c.useF16,
      residual: false,
      weightLayout: 'tap-major',
      packedOutput: c.mode === 'packed',
    });
    const convModule = device.createShaderModule({ label: `chain:conv:${c.label}`, code: convCode });
    const convPipeline = device.createComputePipeline({
      label: `chain:conv:${c.label}`,
      layout: 'auto',
      compute: { module: convModule, entryPoint: 'main' },
    });

    let repackPipeline: GPUComputePipeline | null = null;
    if (c.mode === 'repack') {
      const repackModule = device.createShaderModule({
        label: `chain:repack:${c.label}`,
        code: buildRepackShader({ channels: c.channels, useF16: c.useF16 }),
      });
      repackPipeline = device.createComputePipeline({
        label: `chain:repack:${c.label}`,
        layout: 'auto',
        compute: { module: repackModule, entryPoint: 'main' },
      });
    }

    // --- bind groups, one per layer, created once -------------------------
    const convGroups: GPUBindGroup[] = [];
    const repackGroups: GPUBindGroup[] = [];
    for (let i = 0; i < c.layers; i++) {
      const src = i % 2 === 0 ? ping : pong;
      const dst = i % 2 === 0 ? pong : ping;
      const convOut = c.mode === 'repack' ? (scratch as GPUBuffer) : dst;
      convGroups.push(
        device.createBindGroup({
          layout: convPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: src } },
            { binding: 1, resource: { buffer: weights[i] as GPUBuffer } },
            { binding: 2, resource: { buffer: biases[i] as GPUBuffer } },
            { binding: 3, resource: { buffer: convOut } },
            { binding: 4, resource: { buffer: dims } },
          ],
        }),
      );
      if (repackPipeline) {
        repackGroups.push(
          device.createBindGroup({
            layout: repackPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: scratch as GPUBuffer } },
              { binding: 1, resource: { buffer: dst } },
              { binding: 2, resource: { buffer: dims } },
            ],
          }),
        );
      }
    }

    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolve = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const groupsX = Math.ceil(c.width / (c.tileX * c.blockX));
    const groupsY = Math.ceil(c.height / (c.tileY * c.blockY));
    const groupsZ = c.channels / c.outBlock;
    const repackGroupsX = Math.ceil((pixels * (c.channels / 4)) / 64);

    const encodeChain = (timed: boolean): GPUCommandBuffer => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(
        timed ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {},
      );
      for (let i = 0; i < c.layers; i++) {
        pass.setPipeline(convPipeline);
        pass.setBindGroup(0, convGroups[i]);
        pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
        if (repackPipeline) {
          pass.setPipeline(repackPipeline);
          pass.setBindGroup(0, repackGroups[i]);
          pass.dispatchWorkgroups(repackGroupsX, 1, 1);
        }
      }
      pass.end();
      if (timed) {
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, staging, 0, 16);
      }
      return encoder.finish();
    };

    const compilation = await convModule.getCompilationInfo();
    for (const m of compilation.messages) {
      if (m.type !== 'info') diagnostics.push(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
    }

    // Duration-based warm-up, same reason as ConvBench: a fixed iteration count
    // is a warm-up for one workload size only.
    const start = performance.now();
    for (let batch = 0; batch < 256 && performance.now() - start < this.warmupMs; batch++) {
      for (let i = 0; i < 4; i++) device.queue.submit([encodeChain(false)]);
      await device.queue.onSubmittedWorkDone();
    }

    const oom = await device.popErrorScope();
    if (oom) diagnostics.push(`out-of-memory ${oom.message}`);
    const validation = await device.popErrorScope();
    if (validation) diagnostics.push(`validation ${validation.message}`);

    const samples: number[] = [];
    for (let i = 0; i < this.iterations; i++) {
      device.queue.submit([encodeChain(true)]);
      await staging.mapAsync(GPUMapMode.READ);
      const [begin, end] = new BigInt64Array(staging.getMappedRange().slice(0));
      staging.unmap();
      if (begin !== undefined && end !== undefined) {
        const ms = Number(end - begin) / NS_PER_MS;
        if (ms > 0) samples.push(ms);
      }
    }

    const persistentBytes =
      activationBytes * (c.mode === 'repack' ? 3 : 2) +
      (weightElements + c.channels) * bytesPerElement * c.layers;

    for (const b of [ping, pong, dims, resolve, staging, ...weights, ...biases]) b.destroy();
    scratch?.destroy();
    querySet.destroy();

    samples.sort((a, b) => a - b);
    const median = samples.length > 0 ? (samples[Math.floor(samples.length / 2)] as number) : NaN;
    const executed = median > 0;
    if (!executed) diagnostics.push('chain produced zero-length GPU timestamps: it did not run');
    const valid = diagnostics.length === 0 && executed;

    // Issued extent per layer, matching the dispatch grid, then summed.
    const issuedWidth = groupsX * c.tileX * c.blockX;
    const issuedHeight = groupsY * c.tileY * c.blockY;
    const macs = issuedWidth * issuedHeight * c.channels * c.channels * 9 * c.layers;

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
      persistentBytes,
      dispatches: c.layers * (c.mode === 'repack' ? 2 : 1),
    };
  }
}

/** Deterministic small-magnitude ramp, so a deep chain cannot overflow f16. */
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
