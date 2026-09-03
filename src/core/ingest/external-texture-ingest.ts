import type { FrameTexture, PassTiming, Size } from '../types.js';
import { buildIngestShader } from './ingest.wgsl.js';

/**
 * Storage formats the ingest pass can write.
 *
 * `rgba8unorm` matches the source's 8-bit precision and is the cheapest in
 * bandwidth. `rgba16float` costs twice the bandwidth but gives a neural stage
 * headroom for intermediate values outside [0,1] without a separate
 * normalisation pass, and pairs with `shader-f16` arithmetic.
 */
export type IngestFormat = 'rgba8unorm' | 'rgba16float';

export interface IngestConfig {
  readonly device: GPUDevice;
  /** Decoded video dimensions. The ingest pass never rescales. */
  readonly size: Size;
  readonly format: IngestFormat;
}

/**
 * Converts a `GPUExternalTexture` into an ordinary sampleable texture, once
 * per frame, entirely on the GPU.
 *
 * ## Why this exists
 *
 * `texture_external` can only be read with `textureSampleBaseClampToEdge`, and
 * on a multi-planar decoder surface every such read performs plane sampling
 * and colour conversion. A single-tap consumer does not care. A neural stage
 * reads its input many times — a 3x3 convolution alone is nine reads per
 * output pixel, and a real network stacks dozens of layers — so paying that
 * conversion per read is the wrong shape entirely.
 *
 * This stage pays it exactly once and hands downstream passes a plain
 * `texture_2d<f32>`, which is also the only form a multi-pass ping-pong graph
 * can write back into.
 *
 * ## Guarantees
 *
 * - GPU-resident end to end: no `mapAsync`, no CPU pixel access.
 * - No per-frame GPU resource allocation except the bind group, which is
 *   unavoidable because an external texture expires each task.
 * - Separately timeable: {@link encode} accepts its own {@link PassTiming}, so
 *   the ingest cost is attributable rather than folded into a downstream pass.
 * - The output texture is `TEXTURE_BINDING | RENDER_ATTACHMENT`, so it can be
 *   both sampled by a convolution and used as a ping-pong render target.
 */
export class ExternalTextureIngest {
  private device: GPUDevice | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private layout: GPUBindGroupLayout | null = null;
  private sampler: GPUSampler | null = null;
  private texture: GPUTexture | null = null;
  private textureView: GPUTextureView | null = null;
  private size: Size = { width: 0, height: 0 };
  private format: IngestFormat = 'rgba8unorm';

  /** The ingested texture view. Valid after the first {@link encode}. */
  get view(): GPUTextureView {
    if (!this.textureView) throw new Error('ExternalTextureIngest: configure() has not run');
    return this.textureView;
  }

  get outputFormat(): IngestFormat {
    return this.format;
  }

  get outputSize(): Size {
    return this.size;
  }

  configure(config: IngestConfig): void {
    this.release();
    const { device, size, format } = config;
    this.device = device;
    this.size = size;
    this.format = format;

    this.layout = device.createBindGroupLayout({
      label: 'aethervsr:ingest:layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, externalTexture: {} },
      ],
    });

    const module = device.createShaderModule({
      label: 'aethervsr:ingest:shader',
      code: buildIngestShader(),
    });

    this.pipeline = device.createRenderPipeline({
      label: 'aethervsr:ingest:pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.sampler = device.createSampler({
      label: 'aethervsr:ingest:sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.texture = device.createTexture({
      label: 'aethervsr:ingest:target',
      size: { width: size.width, height: size.height },
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.textureView = this.texture.createView();
  }

  /**
   * True when {@link encode} will actually record a pass for this frame kind.
   *
   * A `sampled` frame is already an ordinary texture and is passed straight
   * through, so callers must not claim a timestamp slot for it: the slot would
   * be resolved having never been written, yielding a fabricated duration.
   */
  static writesPass(frame: FrameTexture): boolean {
    return frame.kind === 'external';
  }

  /**
   * Records the conversion pass. Returns the view downstream passes sample.
   *
   * A `sampled` frame is returned untouched — ingesting it would be a
   * pointless copy — and in that case `timing` must be null, because no pass
   * is recorded to write the timestamps into. Note the returned view then has
   * the *source's* format, not {@link outputFormat}.
   */
  encode(encoder: GPUCommandEncoder, frame: FrameTexture, timing: PassTiming | null): GPUTextureView {
    if (frame.kind === 'sampled') {
      if (timing !== null) {
        throw new Error(
          'ExternalTextureIngest: a sampled frame records no pass, so it cannot write timestamps',
        );
      }
      return frame.view;
    }

    const { device, pipeline, layout, sampler, textureView } = this;
    if (!device || !pipeline || !layout || !sampler || !textureView) {
      throw new Error('ExternalTextureIngest.encode called before configure()');
    }

    // External textures expire each task, so this bind group cannot be cached.
    const bindGroup = device.createBindGroup({
      label: 'aethervsr:ingest:bindgroup',
      layout,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: frame.texture },
      ],
    });

    const pass = encoder.beginRenderPass({
      label: 'aethervsr:ingest:pass',
      colorAttachments: [
        {
          view: textureView,
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: 'store',
        },
      ],
      ...(timing
        ? {
            // Either end may be omitted, so a caller can open a span here and
            // close it in a later pass - which is how a multi-pass stage
            // reports one whole-stage figure rather than a slice of itself.
            timestampWrites: {
              querySet: timing.querySet,
              ...(timing.beginIndex !== undefined
                ? { beginningOfPassWriteIndex: timing.beginIndex }
                : {}),
              ...(timing.endIndex !== undefined ? { endOfPassWriteIndex: timing.endIndex } : {}),
            },
          }
        : {}),
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    return textureView;
  }

  destroy(): void {
    this.release();
    this.device = null;
  }

  private release(): void {
    this.texture?.destroy();
    this.texture = null;
    this.textureView = null;
    this.pipeline = null;
    this.layout = null;
    this.sampler = null;
  }
}
