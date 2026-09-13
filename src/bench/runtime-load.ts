import type { EncodeContext, Upscaler, UpscalerConfig } from '../core/types.js';

export interface RuntimeLoad {
  passes: number;
  frames: number;
}

export class LoadedUpscaler implements Upscaler {
  readonly neural = true;
  private pipeline: GPUComputePipeline | null = null;
  private buffer: GPUBuffer | null = null;
  private binding: GPUBindGroup | null = null;

  constructor(private readonly inner: Upscaler, private readonly load: RuntimeLoad) {
    if (!import.meta.env.DEV) throw new Error('Runtime load is development-only');
  }

  get id(): string { return this.inner.id; }
  get label(): string { return this.inner.label; }
  get scaleFactor(): number { return this.inner.scaleFactor; }

  configure(config: UpscalerConfig): void {
    this.inner.configure(config);
    this.buffer?.destroy();
    this.buffer = config.device.createBuffer({ size: 32768, usage: GPUBufferUsage.STORAGE });
    this.pipeline = config.device.createComputePipeline({
      layout: 'auto',
      compute: { module: config.device.createShaderModule({ code: `
        @group(0) @binding(0) var<storage, read_write> values: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) index: vec3u) {
          var value = values[index.x];
          for (var iteration = 0u; iteration < 1024u; iteration++) {
            value = fract(value * 1.001 + 0.00001);
          }
          values[index.x] = value;
        }` }), entryPoint: 'main' },
    });
    this.binding = config.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.buffer } }],
    });
  }

  encode(context: EncodeContext): void {
    const passes = this.load.frames > 0 ? this.load.passes : 0;
    if (passes === 0) { this.inner.encode(context); return; }
    this.load.frames--;
    this.inner.encode({ ...context, timing: context.timing
      ? { querySet: context.timing.querySet,
        ...(context.timing.beginIndex === undefined ? {} : { beginIndex: context.timing.beginIndex }) }
      : null });
    for (let index = 0; index < passes; index++) {
      const pass = context.encoder.beginComputePass({
        label: 'm9:diagnostic-load',
        ...(index === passes - 1 && context.timing?.endIndex !== undefined
          ? { timestampWrites: { querySet: context.timing.querySet, endOfPassWriteIndex: context.timing.endIndex } }
          : {}),
      });
      pass.setPipeline(this.pipeline!);
      pass.setBindGroup(0, this.binding);
      pass.dispatchWorkgroups(128);
      pass.end();
    }
  }

  destroy(): void {
    this.inner.destroy();
    this.buffer?.destroy();
    this.buffer = null;
    this.pipeline = null;
    this.binding = null;
  }
}