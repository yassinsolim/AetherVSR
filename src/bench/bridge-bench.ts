import { VideoFrameSource } from '../core/acquisition/video-source.js';
import { FrameImporter } from '../core/acquisition/frame-importer.js';
import { ExternalTextureIngest, type IngestFormat } from '../core/ingest/external-texture-ingest.js';
import { buildStemShader, packStemWeights } from '../core/neural/stem.wgsl.js';
import { floatToHalf } from './conv-bench.js';
import { SampleWindow } from '../core/metrics/stats.js';
import type { GpuContext } from '../core/gpu/device.js';
import type { Size } from '../core/types.js';

export interface BridgeBenchConfig {
  readonly format: IngestFormat;
  readonly outChannels: number;
  readonly useF16: boolean;
  readonly blockX: number;
  readonly blockY: number;
  readonly tileX: number;
  readonly tileY: number;
  /**
   * Encode the stem this many times per frame.
   *
   * A discriminator, not a feature. If per-dispatch cost falls sharply as this
   * rises, the frame-paced cost is dominated by something that amortises -
   * clock ramp, or the barrier between the ingest render pass and the first
   * compute pass - rather than by the dispatch itself. If it stays flat, the
   * per-frame figure is the real one and tight-loop benchmarks are optimistic.
   */
  readonly repeats?: number;
}

export interface BridgeStats {
  readonly framesProcessed: number;
  /** External texture -> ingest texture. */
  readonly ingestMs: Summary;
  /** Ingest texture -> packed activations. */
  readonly stemMs: Summary;
  /**
   * One timestamp span covering both passes: begin of the ingest pass to end
   * of the stem pass. This is the entry cost, and it is measured rather than
   * obtained by adding the two figures above — those are separate passes and
   * the gap between them belongs to the total.
   */
  readonly combinedMs: Summary;
  readonly sourceSize: Size;
  readonly activationBytes: number;
}

export interface CaptureResult {
  readonly maxAbsError: number;
  readonly meanAbsError: number;
  readonly referenceRange: readonly [number, number];
  readonly elements: number;
  readonly texelRange: readonly [number, number];
  readonly texelSample: readonly number[];
}

export interface Summary {
  readonly mean: number;
  readonly p50: number;
  readonly p95: number;
  readonly max: number;
  readonly samples: number;
}

const NS_PER_MS = 1_000_000;

/**
 * Measures the real entry path: `GPUExternalTexture -> ingest -> packed
 * neural activations`, driven by an actual decoding video.
 *
 * Everything here runs on frames the decoder produced, through the production
 * ingest stage, so the cost includes whatever colour conversion the decoder
 * surface actually requires. Nothing is read back in the timed loop.
 */
export class BridgeBench {
  private readonly source: VideoFrameSource;
  private readonly importer: FrameImporter;
  private readonly ingest = new ExternalTextureIngest();
  private readonly ingestWindow = new SampleWindow(240);
  private readonly stemWindow = new SampleWindow(240);
  private readonly combinedWindow = new SampleWindow(240);

  private pipeline: GPUComputePipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private activations: GPUBuffer | null = null;
  private weights: GPUBuffer | null = null;
  private biases: GPUBuffer | null = null;
  private params: GPUBuffer | null = null;
  private querySet: GPUQuerySet | null = null;
  private resolve: GPUBuffer | null = null;
  private staging: GPUBuffer | null = null;
  private reading = false;
  private configuredSize: Size = { width: 0, height: 0 };
  private frames = 0;
  private lastIngestView: GPUTextureView | null = null;
  private dumpPipeline: GPUComputePipeline | null = null;
  /** Planar [oc][ic][tap] stem weights, kept for the CPU reference. */
  private stemWeights = new Float32Array(0);
  private pendingCapture: {
    resolve: (r: CaptureResult) => void;
    reject: (e: unknown) => void;
  } | null = null;

  constructor(
    private readonly gpu: GpuContext,
    video: HTMLVideoElement,
    private readonly config: BridgeBenchConfig,
  ) {
    this.source = new VideoFrameSource(video);
    this.importer = new FrameImporter(gpu.device, video, gpu.capabilities.externalTexture);
  }

  start(): void {
    if (!this.source.running) this.source.start((tick) => this.onTick(tick));
  }

  stop(): void {
    this.source.stop();
  }

  reset(): void {
    this.ingestWindow.reset();
    this.stemWindow.reset();
    this.combinedWindow.reset();
    this.frames = 0;
  }

  stats(): BridgeStats {
    const pixels = this.configuredSize.width * this.configuredSize.height;
    return {
      framesProcessed: this.frames,
      ingestMs: summarise(this.ingestWindow),
      stemMs: summarise(this.stemWindow),
      combinedMs: summarise(this.combinedWindow),
      sourceSize: this.configuredSize,
      activationBytes: pixels * this.config.outChannels * (this.config.useF16 ? 2 : 4),
    };
  }

  /**
   * Reads back one frame's ingest texture and the activations the stem produced
   * from it, and checks the stem against a CPU reference computed from those
   * exact texels.
   *
   * This is the only honest way to verify the combined path. The decoder's
   * colour conversion is not reproducible on the CPU, so the reference cannot
   * start from the encoded video; it starts from what ingest actually produced.
   * The check is therefore "did the stem consume the real ingest output
   * correctly", not a restatement of the synthetic-texture test.
   *
   * Readback happens here and nowhere in the timed path.
   */
  async captureAndVerify(): Promise<CaptureResult> {
    if (this.pendingCapture) throw new Error('a capture is already in flight');
    const done = new Promise<CaptureResult>((resolve, reject) => {
      this.pendingCapture = { resolve, reject };
    });
    if (!this.source.running) throw new Error('captureAndVerify requires the frame loop to be running');
    return done;
  }

  destroy(): void {
    this.stop();
    this.ingest.destroy();
    this.importer.destroy();
    for (const b of [this.activations, this.weights, this.biases, this.params, this.resolve, this.staging]) {
      b?.destroy();
    }
    this.querySet?.destroy();
  }

  private onTick(tick: { size: Size }): void {
    const { device } = this.gpu;
    const size = tick.size;
    if (size.width === 0 || size.height === 0) return;
    if (size.width !== this.configuredSize.width || size.height !== this.configuredSize.height) {
      this.configure(size);
    }
    // Deliberately not requiring `bindGroup` here: it can only be built once
    // the ingest pass has produced a view, so demanding it up front would
    // return on the first tick and never create it.
    const { pipeline, querySet, resolve, staging } = this;
    if (!pipeline || !querySet || !resolve || !staging) return;

    const frame = this.importer.acquire(size);
    const encoder = device.createCommandEncoder({ label: 'aethervsr:bridge' });

    // Slots 0/1 bracket ingest, 2/3 bracket the stem. The combined span is
    // slot 3 minus slot 0, which includes any gap between the passes.
    const view = this.ingest.encode(
      encoder,
      frame,
      frame.kind === 'external' ? { querySet, beginIndex: 0, endIndex: 1 } : null,
    );
    if (view !== this.lastIngestView) {
      this.rebindStem(view);
      this.lastIngestView = view;
    }

    const pass = encoder.beginComputePass({
      label: 'aethervsr:bridge:stem',
      timestampWrites: { querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 },
    });
    const bindGroup = this.bindGroup;
    if (!bindGroup) {
      pass.end();
      return;
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const gx = Math.ceil(size.width / (this.config.tileX * this.config.blockX));
    const gy = Math.ceil(size.height / (this.config.tileY * this.config.blockY));
    const repeats = Math.max(1, this.config.repeats ?? 1);
    for (let i = 0; i < repeats; i++) pass.dispatchWorkgroups(gx, gy, 1);

    // A capture rides along on a real frame: same encoder, same pass, same
    // view the stem just consumed.
    const capture = this.pendingCapture ? this.beginCapture(pass, view, size) : null;
    if (capture) device.pushErrorScope('validation');
    pass.end();

    if (frame.kind === 'external' && !this.reading) {
      encoder.resolveQuerySet(querySet, 0, 4, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, staging, 0, 32);
    }
    if (capture) capture.copy(encoder);
    device.queue.submit([encoder.finish()]);
    this.frames++;
    if (capture) {
      const scope = device.popErrorScope();
      void this.finishCapture(capture, size, scope);
    }

    if (frame.kind === 'external' && !this.reading) {
      this.reading = true;
      void staging.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(staging.getMappedRange().slice(0));
        staging.unmap();
        const [t0, t1, t2, t3] = t;
        if (t0 !== undefined && t1 !== undefined && t2 !== undefined && t3 !== undefined) {
          const ingestMs = Number(t1 - t0) / NS_PER_MS;
          const stemMs = Number(t3 - t2) / NS_PER_MS;
          const combinedMs = Number(t3 - t0) / NS_PER_MS;
          if (ingestMs > 0) this.ingestWindow.push(ingestMs);
          if (stemMs > 0) this.stemWindow.push(stemMs);
          if (combinedMs > 0) this.combinedWindow.push(combinedMs);
        }
        this.reading = false;
      });
    }
  }

  /**
   * Encodes the texel dump alongside the live stem dispatch and returns the
   * buffers to read once the frame has been submitted.
   */
  private beginCapture(
    pass: GPUComputePassEncoder,
    view: GPUTextureView,
    size: Size,
  ): { texelBuffer: GPUBuffer; texelReadback: GPUBuffer; actReadback: GPUBuffer; copy: (e: GPUCommandEncoder) => void } {
    const { device } = this.gpu;
    const c = this.config;
    const pixels = size.width * size.height;
    const params = this.params as GPUBuffer;

    if (!this.dumpPipeline) {
      const module = device.createShaderModule({
        label: 'aethervsr:bridge:dump',
        code: `
struct D { width: u32, height: u32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> d: D;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= d.width * d.height) { return; }
  dst[i] = textureLoad(src, vec2i(i32(i % d.width), i32(i / d.width)), 0);
}`,
      });
      this.dumpPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
    const dumpPipeline = this.dumpPipeline;

    const texelBuffer = device.createBuffer({
      size: pixels * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const texelReadback = device.createBuffer({
      size: pixels * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const actBytes = Math.ceil((pixels * c.outChannels * (c.useF16 ? 2 : 4)) / 16) * 16;
    const actReadback = device.createBuffer({
      size: actBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    pass.setPipeline(dumpPipeline);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: dumpPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: view },
          { binding: 1, resource: { buffer: texelBuffer } },
          { binding: 2, resource: { buffer: params } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(pixels / 64), 1, 1);

    const activations = this.activations as GPUBuffer;
    return {
      texelBuffer,
      texelReadback,
      actReadback,
      copy: (e) => {
        e.copyBufferToBuffer(texelBuffer, 0, texelReadback, 0, pixels * 16);
        e.copyBufferToBuffer(activations, 0, actReadback, 0, actBytes);
      },
    };
  }

  private async finishCapture(
    capture: { texelBuffer: GPUBuffer; texelReadback: GPUBuffer; actReadback: GPUBuffer },
    size: Size,
    scope: Promise<GPUError | null>,
  ): Promise<void> {
    const pending = this.pendingCapture;
    this.pendingCapture = null;
    if (!pending) return;
    const c = this.config;
    const pixels = size.width * size.height;
    const outElements = pixels * c.outChannels;
    try {
      const err = await scope;
      if (err) throw new Error(`capture frame failed validation: ${err.message}`);
      await capture.texelReadback.mapAsync(GPUMapMode.READ);
      const texelRaw = capture.texelReadback.getMappedRange().slice(0);
      capture.texelReadback.unmap();
      await capture.actReadback.mapAsync(GPUMapMode.READ);
      const actRaw = capture.actReadback.getMappedRange().slice(0);
      capture.actReadback.unmap();

      const grouped = c.useF16
        ? Float32Array.from(new Uint16Array(actRaw).subarray(0, outElements), halfToFloat)
        : new Float32Array(actRaw).subarray(0, outElements);
      const actual = new Float32Array(outElements);
      for (let ch = 0; ch < c.outChannels; ch++) {
        const g = Math.floor(ch / 4);
        const lane = ch % 4;
        for (let p = 0; p < pixels; p++) {
          actual[ch * pixels + p] = grouped[(g * pixels + p) * 4 + lane] as number;
        }
      }

      const texels = new Float32Array(texelRaw);
      const rgb = new Float32Array(pixels * 3);
      let texLo = Infinity;
      let texHi = -Infinity;
      for (let p = 0; p < pixels; p++) {
        for (let k = 0; k < 3; k++) {
          const v = texels[p * 4 + k] as number;
          rgb[p * 3 + k] = v;
          if (v < texLo) texLo = v;
          if (v > texHi) texHi = v;
        }
      }
      // A uniformly zero capture makes every comparison below trivially agree.
      // Refuse rather than report a perfect score.
      if (texHi <= texLo) {
        throw new Error(
          `captured ingest texels are uniform (${texLo}); the comparison would be vacuous`,
        );
      }

      const expected = referenceStemFromRgb(rgb, size.width, size.height, c.outChannels, this.stemWeights);
      let maxAbs = 0;
      let sumAbs = 0;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < outElements; i++) {
        const e = expected[i] as number;
        const d = Math.abs((actual[i] as number) - e);
        if (d > maxAbs) maxAbs = d;
        sumAbs += d;
        if (e < lo) lo = e;
        if (e > hi) hi = e;
      }
      pending.resolve({
        maxAbsError: maxAbs,
        meanAbsError: sumAbs / outElements,
        referenceRange: [lo, hi],
        elements: outElements,
        texelRange: [texLo, texHi],
        texelSample: Array.from(texels.subarray(4096, 4104)),
      });
    } catch (err) {
      pending.reject(err);
    } finally {
      capture.texelBuffer.destroy();
      capture.texelReadback.destroy();
      capture.actReadback.destroy();
    }
  }

  private configure(size: Size): void {
    const { device } = this.gpu;
    const c = this.config;
    this.ingest.configure({ device, size, format: c.format });

    for (const b of [this.activations, this.weights, this.biases, this.params]) b?.destroy();

    const bytesPerElement = c.useF16 ? 2 : 4;
    const pixels = size.width * size.height;
    this.activations = device.createBuffer({
      label: 'aethervsr:bridge:activations',
      size: Math.ceil((pixels * c.outChannels * bytesPerElement) / 16) * 16,
      // COPY_SRC so a verification capture can read the activations out. The
      // frame loop never copies from it; without this the capture frame fails
      // to finish and *every* pass in it silently does nothing, which is how a
      // uniformly zero comparison first appeared to pass.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const planar = new Float32Array(new ArrayBuffer(c.outChannels * 3 * 9 * 4));
    for (let i = 0; i < planar.length; i++) planar[i] = Math.cos(i * 1.1) * 0.1;
    this.stemWeights = planar;
    this.weights = uploadFloats(device, packStemWeights(planar, c.outChannels), c.useF16);
    const bias = new Float32Array(new ArrayBuffer(c.outChannels * 4));
    this.biases = uploadFloats(device, bias, c.useF16);

    this.params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const p = new ArrayBuffer(48);
    new Uint32Array(p, 0, 2).set([size.width, size.height]);
    new Float32Array(p, 16, 4).set([0, 0, 0, 0]);
    new Float32Array(p, 32, 4).set([1, 1, 1, 1]);
    device.queue.writeBuffer(this.params, 0, p);

    const module = device.createShaderModule({
      label: 'aethervsr:bridge:stem',
      code: buildStemShader({
        outChannels: c.outChannels,
        blockX: c.blockX,
        blockY: c.blockY,
        tileX: c.tileX,
        tileY: c.tileY,
        useF16: c.useF16,
        activation: 'relu',
      }),
    });
    this.pipeline = device.createComputePipeline({
      label: 'aethervsr:bridge:stem',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });

    if (!this.querySet) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
      this.resolve = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.staging = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    this.configuredSize = size;
    this.lastIngestView = null;
  }

  private rebindStem(view: GPUTextureView): void {
    const { device } = this.gpu;
    if (!this.pipeline || !this.activations || !this.weights || !this.biases || !this.params) return;
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: { buffer: this.weights } },
        { binding: 2, resource: { buffer: this.biases } },
        { binding: 3, resource: { buffer: this.activations } },
        { binding: 4, resource: { buffer: this.params } },
      ],
    });
  }
}

function summarise(w: SampleWindow): Summary {
  return {
    mean: w.mean(),
    p50: w.quantile(0.5),
    p95: w.quantile(0.95),
    max: w.quantile(1),
    samples: w.size,
  };
}

function uploadFloats(device: GPUDevice, data: Float32Array<ArrayBuffer>, useF16: boolean): GPUBuffer {
  const bytesPerElement = useF16 ? 2 : 4;
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
}

/** CPU 3x3 RGB -> C_out convolution, zero padding, relu, planar output. */
function referenceStemFromRgb(
  rgb: Float32Array,
  W: number,
  H: number,
  outChannels: number,
  weights: Float32Array,
): Float32Array {
  const out = new Float32Array(W * H * outChannels);
  for (let oc = 0; oc < outChannels; oc++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let acc = 0;
        for (let ic = 0; ic < 3; ic++) {
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              const sx = x + kx - 1;
              const sy = y + ky - 1;
              if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
              acc +=
                (weights[(oc * 3 + ic) * 9 + ky * 3 + kx] as number) *
                (rgb[(sy * W + sx) * 3 + ic] as number);
            }
          }
        }
        out[oc * W * H + y * W + x] = Math.max(acc, 0);
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
