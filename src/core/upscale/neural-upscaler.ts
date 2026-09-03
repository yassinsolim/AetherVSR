import { ExternalTextureIngest, type IngestFormat } from '../ingest/external-texture-ingest.js';
import { buildStemShader } from '../neural/stem.wgsl.js';
import { buildBlockedConvShader } from '../neural/conv.wgsl.js';
import { buildUpsampleHeadShader } from '../neural/upsample-head.wgsl.js';
import { buildBlitShader } from '../neural/blit.wgsl.js';
import { weightBytes, type PackedModel } from '../neural/model.js';
import type { EncodeContext, Upscaler, UpscalerConfig } from '../types.js';

/**
 * Device features the neural backend wants.
 *
 * `shader-f16` is optional in the sense that the graph still produces correct
 * output without it, and load-bearing in the sense that without it the stage
 * costs about twice as much and twice the memory. It must be requested at
 * device creation, which happens before any upscaler exists, so it is exported
 * for the harness to include.
 */
export const NEURAL_OPTIONAL_FEATURES: readonly GPUFeatureName[] = ['shader-f16'];

/** Tunables that shape the dispatches, not the network. */
export interface NeuralUpscalerOptions {
  /** Half precision throughout. Falls back to f32 when unsupported. */
  readonly useF16: boolean;
  readonly ingestFormat: IngestFormat;
}

const DEFAULT_OPTIONS: NeuralUpscalerOptions = {
  useF16: true,
  ingestFormat: 'rgba8unorm',
};

/** Per-stage GPU times from the most recent resolved timestamp read. */
export interface NeuralStageTiming {
  readonly stemMs: number;
  readonly bodyMs: number;
  readonly headMs: number;
}

/** Persistent GPU allocation, reported rather than estimated. */
export interface NeuralMemoryReport {
  readonly weights: number;
  readonly activationPing: number;
  readonly activationPong: number;
  readonly ingestTexture: number;
  readonly outputTexture: number;
  readonly uniforms: number;
  readonly total: number;
}

/**
 * The neural upscaler: a second implementation of {@link Upscaler}, chosen at
 * runtime alongside the baseline scaler.
 *
 * ## Where the graph lives
 *
 * Ingest is owned here rather than by the pipeline. The baseline scaler samples
 * the external texture once per output pixel and needs no intermediate; a
 * multi-tap network does, and paying the external-texture conversion on every
 * tap would be the wrong shape entirely. Keeping ingest inside the stage that
 * needs it leaves acquisition and presentation untouched, which is the seam
 * Milestone 1 was built around.
 *
 * ## Allocation
 *
 * Every buffer, texture, pipeline, sampler and bind group is created in
 * {@link configure}. {@link encode} allocates nothing: it records passes into
 * the caller's encoder using resources that already exist. The one thing that
 * changes per frame is the external texture, whose bind group WebGPU requires
 * be rebuilt because the handle expires each task — that rebuild happens inside
 * `ExternalTextureIngest`, which already documented the constraint.
 *
 * ## No readback
 *
 * Nothing in `encode` maps a buffer. The timestamp resolve is copied into a
 * staging buffer and read on a later frame, never awaited inside the frame.
 */
export class NeuralUpscaler implements Upscaler {
  readonly id: string;
  readonly label: string;
  readonly scaleFactor = 2;
  readonly neural = true;

  private readonly options: NeuralUpscalerOptions;
  private readonly ingest = new ExternalTextureIngest();

  private device: GPUDevice | null = null;
  private useF16 = true;

  private ping: GPUBuffer | null = null;
  private pong: GPUBuffer | null = null;
  private stemWeights: GPUBuffer | null = null;
  private stemBias: GPUBuffer | null = null;
  private bodyWeights: GPUBuffer[] = [];
  private bodyBias: GPUBuffer[] = [];
  private headWeights: GPUBuffer | null = null;
  private headBias: GPUBuffer | null = null;
  private stemParams: GPUBuffer | null = null;
  private dims: GPUBuffer | null = null;
  private headParams: GPUBuffer | null = null;

  private stemPipeline: GPUComputePipeline | null = null;
  private bodyPipeline: GPUComputePipeline | null = null;
  private headPipeline: GPUComputePipeline | null = null;
  private blitPipeline: GPURenderPipeline | null = null;

  private stemGroup: GPUBindGroup | null = null;
  private bodyGroups: GPUBindGroup[] = [];
  private headGroup: GPUBindGroup | null = null;
  private blitGroup: GPUBindGroup | null = null;

  private outputTexture: GPUTexture | null = null;
  private sampler: GPUSampler | null = null;
  private lastIngestView: GPUTextureView | null = null;

  private querySet: GPUQuerySet | null = null;
  private resolveBuf: GPUBuffer | null = null;
  private stagingBuf: GPUBuffer | null = null;
  private reading = false;
  private pendingRead = false;
  private timing: NeuralStageTiming | null = null;
  private stemDispatch: [number, number] = [0, 0];
  private bodyDispatch: [number, number, number] = [0, 0, 0];
  private headDispatch: [number, number] = [0, 0];
  private memory: NeuralMemoryReport | null = null;

  constructor(
    private readonly model: PackedModel,
    options: Partial<NeuralUpscalerOptions> = {},
  ) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.id = `neural-${model.file.architecture}-c${model.features}d${model.depth}`;
    this.label = `Neural C${model.features}D${model.depth} (${model.file.architecture})`;
  }

  /** Persistent GPU bytes, available after {@link configure}. */
  get memoryReport(): NeuralMemoryReport | null {
    return this.memory;
  }

  /**
   * Per-stage GPU times from the most recently resolved frame.
   *
   * Read asynchronously and never awaited inside `encode`, so a frame is never
   * blocked on its own measurement. `totalMs` is one span from the first
   * timestamp to the last, not the sum of the stages: separate passes can
   * overlap and the gaps between them are real time.
   */
  get stageTiming(): NeuralStageTiming | null {
    return this.timing;
  }

  configure(config: UpscalerConfig): void {
    this.destroyResources();
    const { device, source } = config;
    this.device = device;
    // f16 is requested by default but the device decides; a model that silently
    // ran at a different precision than reported would make every numerical
    // comparison meaningless.
    this.useF16 = this.options.useF16 && device.features.has('shader-f16');

    const c = this.model.features;
    const bpe = this.useF16 ? 2 : 4;
    const pixels = source.width * source.height;
    const activationBytes = Math.ceil((pixels * c * bpe) / 16) * 16;

    this.ingest.configure({ device, size: source, format: this.options.ingestFormat });

    this.ping = device.createBuffer({
      label: 'aethervsr:nn:ping',
      size: activationBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.pong = device.createBuffer({
      label: 'aethervsr:nn:pong',
      size: activationBytes,
      usage: GPUBufferUsage.STORAGE,
    });

    const upload = (data: Float32Array<ArrayBuffer>, label: string): GPUBuffer => {
      const buf = device.createBuffer({
        label,
        size: Math.max(16, Math.ceil((data.length * bpe) / 16) * 16),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      if (this.useF16) {
        const half = new Uint16Array(data.length + (data.length % 2));
        for (let i = 0; i < data.length; i++) half[i] = floatToHalf(data[i] as number);
        device.queue.writeBuffer(buf, 0, half);
      } else {
        device.queue.writeBuffer(buf, 0, data);
      }
      return buf;
    };

    this.stemWeights = upload(this.model.stemWeights, 'aethervsr:nn:stem.w');
    this.stemBias = upload(this.model.stemBias, 'aethervsr:nn:stem.b');
    this.bodyWeights = this.model.bodyWeights.map((w, i) => upload(w, `aethervsr:nn:body${i}.w`));
    this.bodyBias = this.model.bodyBias.map((b, i) => upload(b, `aethervsr:nn:body${i}.b`));
    this.headWeights = upload(this.model.headWeights, 'aethervsr:nn:head.w');
    this.headBias = upload(this.model.headBias, 'aethervsr:nn:head.b');

    // Stem params carry the normalisation, so no separate pass applies it.
    this.stemParams = device.createBuffer({
      label: 'aethervsr:nn:stemparams',
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const sp = new ArrayBuffer(48);
    new Uint32Array(sp, 0, 2).set([source.width, source.height]);
    const mean = this.model.file.normalisation.mean;
    const scale = this.model.file.normalisation.scale;
    new Float32Array(sp, 16, 4).set([mean[0] ?? 0, mean[1] ?? 0, mean[2] ?? 0, 0]);
    new Float32Array(sp, 32, 4).set([scale[0] ?? 1, scale[1] ?? 1, scale[2] ?? 1, 0]);
    device.queue.writeBuffer(this.stemParams, 0, sp);

    this.dims = device.createBuffer({
      label: 'aethervsr:nn:dims',
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.dims, 0, new Uint32Array([source.width, source.height]));
    this.headParams = device.createBuffer({
      label: 'aethervsr:nn:headparams',
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.headParams, 0, new Uint32Array([source.width, source.height]));

    this.outputTexture = device.createTexture({
      label: 'aethervsr:nn:output',
      size: { width: source.width * 2, height: source.height * 2 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // --- pipelines ------------------------------------------------------
    const stemModule = device.createShaderModule({
      label: 'aethervsr:nn:stem',
      code: buildStemShader({
        outChannels: c,
        blockX: 2,
        blockY: 2,
        tileX: 8,
        tileY: 8,
        useF16: this.useF16,
        activation: 'tanh',
        kernel: 5,
      }),
    });
    this.stemPipeline = device.createComputePipeline({
      label: 'aethervsr:nn:stem',
      layout: 'auto',
      compute: { module: stemModule, entryPoint: 'main' },
    });

    const bodyModule = device.createShaderModule({
      label: 'aethervsr:nn:body',
      code: buildBlockedConvShader({
        inChannels: c,
        outChannels: c,
        tileX: 8,
        tileY: 4,
        blockX: 2,
        blockY: 2,
        outBlock: c,
        activation: 'tanh',
        useF16: this.useF16,
        residual: false,
        weightLayout: 'tap-major',
        packedOutput: true,
      }),
    });
    this.bodyPipeline = device.createComputePipeline({
      label: 'aethervsr:nn:body',
      layout: 'auto',
      compute: { module: bodyModule, entryPoint: 'main' },
    });

    const headModule = device.createShaderModule({
      label: 'aethervsr:nn:head',
      code: buildUpsampleHeadShader({
        inChannels: c,
        scale: 2,
        useF16: this.useF16,
        format: 'rgba8unorm',
        blockX: 2,
        blockY: 4,
        tileX: 8,
        tileY: 4,
        globalResidual: true,
      }),
    });
    this.headPipeline = device.createComputePipeline({
      label: 'aethervsr:nn:head',
      layout: 'auto',
      compute: { module: headModule, entryPoint: 'main' },
    });

    const blitModule = device.createShaderModule({
      label: 'aethervsr:nn:blit',
      code: buildBlitShader(config.targetFormat),
    });
    this.blitPipeline = device.createRenderPipeline({
      label: 'aethervsr:nn:blit',
      layout: 'auto',
      vertex: { module: blitModule, entryPoint: 'vs' },
      fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format: config.targetFormat }] },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    // --- bind groups ----------------------------------------------------
    // All but the stem's, which needs the ingest view and is built on the first
    // frame and then reused; the view is stable for a given configuration.
    for (let i = 0; i < this.model.depth; i++) {
      const src = i % 2 === 0 ? this.ping : this.pong;
      const dst = i % 2 === 0 ? this.pong : this.ping;
      this.bodyGroups.push(
        device.createBindGroup({
          label: `aethervsr:nn:body${i}`,
          layout: this.bodyPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: src } },
            { binding: 1, resource: { buffer: this.bodyWeights[i] as GPUBuffer } },
            { binding: 2, resource: { buffer: this.bodyBias[i] as GPUBuffer } },
            { binding: 3, resource: { buffer: dst } },
            { binding: 4, resource: { buffer: this.dims } },
          ],
        }),
      );
    }

    // The stem writes ping; after `depth` body layers the result is in ping for
    // even depth and pong for odd.
    this.blitGroup = device.createBindGroup({
      label: 'aethervsr:nn:blit',
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: this.outputTexture.createView() },
      ],
    });

    if (device.features.has('timestamp-query')) {
      // Six slots for the three compute stages. Ingest and present are
      // bracketed by the pipeline's own timer, which spans the whole stage.
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 6 });
      this.resolveBuf = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.stagingBuf = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    this.stemDispatch = [Math.ceil(source.width / 16), Math.ceil(source.height / 16)];
    this.bodyDispatch = [Math.ceil(source.width / 16), Math.ceil(source.height / 8), 1];
    this.headDispatch = [Math.ceil((source.width * 2) / 16), Math.ceil((source.height * 2) / 16)];

    const ingestBytes = pixels * (this.options.ingestFormat === 'rgba8unorm' ? 4 : 8);
    const outputBytes = pixels * 4 * 4;
    this.memory = {
      weights: weightBytes(this.model, this.useF16),
      activationPing: activationBytes,
      activationPong: activationBytes,
      ingestTexture: ingestBytes,
      outputTexture: outputBytes,
      uniforms: 48 + 8 + 8,
      total:
        weightBytes(this.model, this.useF16) +
        activationBytes * 2 +
        ingestBytes +
        outputBytes +
        64,
    };
    this.lastIngestView = null;
  }

  encode(ctx: EncodeContext): void {
    const { device } = this;
    if (!device || !this.stemPipeline || !this.bodyPipeline || !this.headPipeline) {
      throw new Error('NeuralUpscaler.encode called before configure()');
    }

    this.startTimestampRead();

    // 1. External texture -> ordinary texture, once per frame.
    const view = this.ingest.encode(
      ctx.encoder,
      ctx.frame,
      // The pipeline's own timer brackets the *whole* stage: it opens here and
      // closes on the present pass below. That is what its overlay row claims
      // to show, and giving it a single inner pass instead would be a quietly
      // wrong number. Slot pairing is legal across passes because the indices
      // are explicit.
      ctx.timing && ctx.frame.kind === 'external' && ctx.timing.beginIndex !== undefined
        ? { querySet: ctx.timing.querySet, beginIndex: ctx.timing.beginIndex }
        : null,
    );
    if (view !== this.lastIngestView) {
      this.stemGroup = device.createBindGroup({
        label: 'aethervsr:nn:stem',
        layout: this.stemPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: view },
          { binding: 1, resource: { buffer: this.stemWeights as GPUBuffer } },
          { binding: 2, resource: { buffer: this.stemBias as GPUBuffer } },
          { binding: 3, resource: { buffer: this.ping as GPUBuffer } },
          { binding: 4, resource: { buffer: this.stemParams as GPUBuffer } },
        ],
      });
      // The head reads the same view for its global residual, so both groups
      // are rebuilt together and only when the view actually changes.
      this.headGroup = device.createBindGroup({
        label: 'aethervsr:nn:head',
        layout: this.headPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.finalBuffer() } },
          { binding: 1, resource: { buffer: this.headWeights as GPUBuffer } },
          { binding: 2, resource: { buffer: this.headBias as GPUBuffer } },
          { binding: 3, resource: (this.outputTexture as GPUTexture).createView() },
          { binding: 4, resource: { buffer: this.headParams as GPUBuffer } },
          { binding: 5, resource: view },
        ],
      });
      this.lastIngestView = view;
    }

    const stemGroup = this.stemGroup;
    const headGroup = this.headGroup;
    const blitGroup = this.blitGroup;
    const blitPipeline = this.blitPipeline;
    if (!stemGroup || !headGroup || !blitGroup || !blitPipeline) {
      throw new Error('NeuralUpscaler.encode called before configure()');
    }

    // 2. Each stage in its own pass, so each can be timed. WebGPU orders
    //    passes within a command buffer and makes each one's writes visible to
    //    the next, so the ping-pong chain still needs no explicit barrier.
    const qs = this.querySet;
    // Spread rather than an optional property: `exactOptionalPropertyTypes`
    // makes `timestampWrites: undefined` a different thing from absent.
    const stamp = (begin: number, end: number): GPUComputePassDescriptor =>
      qs
        ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end } }
        : {};

    const stemPass = ctx.encoder.beginComputePass({
      label: 'aethervsr:nn:stem',
      ...stamp(0, 1),
    });
    stemPass.setPipeline(this.stemPipeline);
    stemPass.setBindGroup(0, stemGroup);
    stemPass.dispatchWorkgroups(this.stemDispatch[0], this.stemDispatch[1], 1);
    stemPass.end();

    const bodyPass = ctx.encoder.beginComputePass({
      label: 'aethervsr:nn:body',
      ...stamp(2, 3),
    });
    bodyPass.setPipeline(this.bodyPipeline);
    for (const group of this.bodyGroups) {
      bodyPass.setBindGroup(0, group);
      bodyPass.dispatchWorkgroups(this.bodyDispatch[0], this.bodyDispatch[1], this.bodyDispatch[2]);
    }
    bodyPass.end();

    const headPass = ctx.encoder.beginComputePass({
      label: 'aethervsr:nn:head',
      ...stamp(4, 5),
    });
    headPass.setPipeline(this.headPipeline);
    headPass.setBindGroup(0, headGroup);
    headPass.dispatchWorkgroups(this.headDispatch[0], this.headDispatch[1], 1);
    headPass.end();

    // 3. Present. The reconstruction writes an owned storage texture, which
    //    this copies to the swap-chain view. The blit's bind group references
    //    only owned resources, so nothing is allocated here.
    const blit = ctx.encoder.beginRenderPass({
      label: 'aethervsr:nn:present',
      colorAttachments: [
        { view: ctx.target, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' },
      ],
      ...(ctx.timing && ctx.timing.endIndex !== undefined
        ? {
            timestampWrites: {
              querySet: ctx.timing.querySet,
              endOfPassWriteIndex: ctx.timing.endIndex,
            },
          }
        : {}),
    });
    blit.setPipeline(blitPipeline);
    blit.setBindGroup(0, blitGroup);
    blit.draw(3);
    blit.end();

    this.resolveTimestamps(ctx.encoder);
  }

  /**
   * Copies the timestamp set out and reads it on a later turn.
   *
   * Never awaited inside the frame: a resolved read arrives whenever it
   * arrives, and a frame that skips a measurement is better than a frame that
   * waits for one.
   */
  private resolveTimestamps(encoder: GPUCommandEncoder): void {
    const qs = this.querySet;
    const resolve = this.resolveBuf;
    const staging = this.stagingBuf;
    if (!qs || !resolve || !staging) return;
    // Only encode a new copy when the previous one has been consumed. A buffer
    // that is still mapped, or still awaiting a map, must not appear in a
    // submission.
    if (this.pendingRead || this.reading) return;
    encoder.resolveQuerySet(qs, 0, 6, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, staging, 0, 48);
    this.pendingRead = true;
  }

  /**
   * Starts reading the previous frame's timestamps.
   *
   * Called at the *top* of `encode`, which is the only safe moment: the frame
   * that wrote them has been submitted, and nothing has yet been encoded into
   * the staging buffer for this frame. Mapping it during the frame that writes
   * it makes the submission fail with "used in submit while mapped", which is
   * exactly what happened the first time this was wired up.
   */
  private startTimestampRead(): void {
    const staging = this.stagingBuf;
    if (!staging || !this.pendingRead || this.reading) return;
    this.reading = true;
    void staging.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigInt64Array(staging.getMappedRange().slice(0));
      staging.unmap();
      const ms = (a: number, b: number): number => {
        const begin = t[a];
        const end = t[b];
        return begin !== undefined && end !== undefined ? Number(end - begin) / 1_000_000 : Number.NaN;
      };
      const stem = ms(0, 1);
      if (stem > 0) {
        this.timing = { stemMs: stem, bodyMs: ms(2, 3), headMs: ms(4, 5) };
      }
      this.reading = false;
      this.pendingRead = false;
    });
  }

  destroy(): void {
    this.destroyResources();
    this.ingest.destroy();
  }

  /** Where the trunk leaves its result: ping for even depth, pong for odd. */
  private finalBuffer(): GPUBuffer {
    const buf = this.model.depth % 2 === 0 ? this.ping : this.pong;
    if (!buf) throw new Error('activations not allocated');
    return buf;
  }

  private destroyResources(): void {
    for (const b of [
      this.ping,
      this.pong,
      this.stemWeights,
      this.stemBias,
      this.headWeights,
      this.headBias,
      this.stemParams,
      this.dims,
      this.headParams,
      ...this.bodyWeights,
      ...this.bodyBias,
    ]) {
      b?.destroy();
    }
    this.outputTexture?.destroy();
    this.querySet?.destroy();
    this.resolveBuf?.destroy();
    this.stagingBuf?.destroy();
    this.querySet = null;
    this.resolveBuf = null;
    this.stagingBuf = null;
    this.timing = null;
    this.pendingRead = false;
    this.reading = false;
    this.ping = null;
    this.pong = null;
    this.stemWeights = null;
    this.stemBias = null;
    this.headWeights = null;
    this.headBias = null;
    this.stemParams = null;
    this.dims = null;
    this.headParams = null;
    this.bodyWeights = [];
    this.bodyBias = [];
    this.bodyGroups = [];
    this.outputTexture = null;
    this.stemGroup = null;
    this.headGroup = null;
    this.blitGroup = null;
    this.lastIngestView = null;
    this.memory = null;
  }
}

/** IEEE-754 binary32 to binary16, round-toward-zero. */
function floatToHalf(value: number): number {
  converter.setFloat32(0, value);
  const bits = converter.getUint32(0);
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa !== 0 ? 0x200 : 0);
  const e = exponent - 127 + 15;
  if (e >= 0x1f) return sign | 0x7bff;
  if (e <= 0) return sign;
  return sign | (e << 10) | (mantissa >>> 13);
}

const converter = new DataView(new ArrayBuffer(4));
