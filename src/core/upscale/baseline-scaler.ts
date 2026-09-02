import type { EncodeContext, Upscaler, UpscalerConfig } from '../types.js';
import { buildBaselineShader, type BaselineFilter } from './baseline.wgsl.js';

const UNIFORM_BYTES = 16; // vec2f srcSize + vec2f invSrcSize

/**
 * Optional WebGPU features the upscalers in this module want enabled.
 *
 * Device creation happens before any `Upscaler` exists, and a feature the
 * adapter supports but nobody requested is absent from `device.features` and
 * will fail shader validation. Backends therefore declare their requirements
 * here so `acquireGpu()` can request the union up front — this is the
 * negotiation point a neural backend needing `shader-f16` or `subgroups` will
 * add to, without touching acquisition or presentation.
 *
 * The baseline needs none: it is plain f32 sampling.
 */
export const UPSCALER_OPTIONAL_FEATURES: readonly GPUFeatureName[] = [];

const LABELS: Record<BaselineFilter, string> = {
  bilinear: 'Bilinear (GPU sampler)',
  'catmull-rom': 'Catmull-Rom bicubic (9-tap)',
};

/**
 * Milestone 1 reference upscaler: a single fullscreen render pass that
 * resamples the imported video frame into the output texture.
 *
 * This deliberately contains no learned component. It exists to prove the
 * acquisition -> upscale -> presentation path, and to be the quality and
 * performance baseline every future neural backend is measured against.
 */
export class BaselineScaler implements Upscaler {
  readonly id: string;
  readonly label: string;
  readonly scaleFactor: number;
  readonly neural = false;

  private device: GPUDevice | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private layout: GPUBindGroupLayout | null = null;
  private sampler: GPUSampler | null = null;
  private uniforms: GPUBuffer | null = null;
  /** Reused on the `sampled` path only; see {@link resolveBindGroup}. */
  private cachedBindGroup: GPUBindGroup | null = null;

  constructor(
    readonly filter: BaselineFilter,
    scaleFactor = 2,
  ) {
    this.id = `baseline-${filter}`;
    this.label = LABELS[filter];
    this.scaleFactor = scaleFactor;
  }

  configure(config: UpscalerConfig): void {
    this.releaseGpuObjects();
    const { device, source, targetFormat, sourceKind } = config;
    this.device = device;

    this.layout = device.createBindGroupLayout({
      label: `aethervsr:${this.id}:layout`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        sourceKind === 'external'
          ? { binding: 1, visibility: GPUShaderStage.FRAGMENT, externalTexture: {} }
          : {
              binding: 1,
              visibility: GPUShaderStage.FRAGMENT,
              texture: { sampleType: 'float', viewDimension: '2d' },
            },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const module = device.createShaderModule({
      label: `aethervsr:${this.id}:shader`,
      code: buildBaselineShader(sourceKind, this.filter),
    });

    this.pipeline = device.createRenderPipeline({
      label: `aethervsr:${this.id}:pipeline`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: targetFormat }] },
      primitive: { topology: 'triangle-list' },
    });

    // Clamp-to-edge matches `textureSampleBaseClampToEdge`, so the external
    // and fallback paths produce identical pixels at the frame border.
    this.sampler = device.createSampler({
      label: `aethervsr:${this.id}:sampler`,
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.uniforms = device.createBuffer({
      label: `aethervsr:${this.id}:uniforms`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      this.uniforms,
      0,
      new Float32Array([source.width, source.height, 1 / source.width, 1 / source.height]),
    );
  }

  encode(ctx: EncodeContext): void {
    const { device, pipeline, layout, sampler, uniforms } = this;
    if (!device || !pipeline || !layout || !sampler || !uniforms) {
      throw new Error('BaselineScaler.encode called before configure()');
    }

    const bindGroup = this.resolveBindGroup(device, layout, sampler, uniforms, ctx);

    const pass = ctx.encoder.beginRenderPass({
      label: `aethervsr:${this.id}:pass`,
      colorAttachments: [
        {
          view: ctx.target,
          // The draw covers every pixel, so the previous contents are never
          // read. `clear` lets a tile-based GPU skip loading the framebuffer.
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: 'store',
        },
      ],
      ...(ctx.timing
        ? {
            timestampWrites: {
              querySet: ctx.timing.querySet,
              beginningOfPassWriteIndex: ctx.timing.beginIndex,
              endOfPassWriteIndex: ctx.timing.endIndex,
            },
          }
        : {}),
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  /**
   * A `GPUExternalTexture` imported from an `HTMLVideoElement` expires in an
   * automatic-expiry task shortly after import, so a bind group referencing it
   * is built fresh inside every frame's encode.
   *
   * The specification does permit reuse: a repeated import of an unchanged
   * frame returns the *same* object, un-expired, and bind groups referencing it
   * stay valid. We deliberately do not exploit that. Chromium replaces the
   * object whenever the decoded frame advances, and under rVFC every callback
   * is a new frame by definition, so an identity-keyed cache would hit only
   * while playback is stalled — no measurable gain in exchange for reasoning
   * about resource expiry on the hot path.
   *
   * The copy fallback binds a texture view we own for the whole session and so
   * builds exactly one bind group.
   */
  private resolveBindGroup(
    device: GPUDevice,
    layout: GPUBindGroupLayout,
    sampler: GPUSampler,
    uniforms: GPUBuffer,
    ctx: EncodeContext,
  ): GPUBindGroup {
    if (ctx.frame.kind === 'sampled' && this.cachedBindGroup) return this.cachedBindGroup;

    const resource: GPUBindingResource =
      ctx.frame.kind === 'external' ? ctx.frame.texture : ctx.frame.view;
    const bindGroup = device.createBindGroup({
      label: `aethervsr:${this.id}:bindgroup`,
      layout,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource },
        { binding: 2, resource: { buffer: uniforms } },
      ],
    });
    if (ctx.frame.kind === 'sampled') this.cachedBindGroup = bindGroup;
    return bindGroup;
  }

  destroy(): void {
    this.releaseGpuObjects();
    this.device = null;
  }

  private releaseGpuObjects(): void {
    this.uniforms?.destroy();
    this.uniforms = null;
    this.pipeline = null;
    this.layout = null;
    this.sampler = null;
    this.cachedBindGroup = null;
  }
}
