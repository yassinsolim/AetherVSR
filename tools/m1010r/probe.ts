import type { EncodeContext, Upscaler, UpscalerConfig } from '../../src/core/types.js';

export const PROBE_CAPACITY = 65536;

export function probeShader(external: boolean): string {
  const declaration = external ? 'texture_external' : 'texture_2d<f32>';
  const sample = external ? 'textureSampleBaseClampToEdge(source, filtering, coordinates)' : 'textureSampleLevel(source, filtering, coordinates, 0.0)';
  return `
struct Records { count: atomic<u32>, unused: array<u32, 3>, rows: array<vec4<u32>>, }
@group(0) @binding(0) var source: ${declaration};
@group(0) @binding(1) var filtering: sampler;
@group(0) @binding(2) var<storage, read_write> records: Records;
@group(0) @binding(3) var<uniform> extent: vec4<f32>;
fn cell(horizontal: f32, vertical: f32) -> u32 {
  let coordinates = vec2<f32>(horizontal + 0.5, vertical + 0.5) / extent.xy;
  let color = ${sample};
  let luma = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
  if (luma <= 0.25) { return 0u; }
  if (luma >= 0.75) { return 1u; }
  return 2u;
}
@compute @workgroup_size(1) fn main() {
  let slot = atomicAdd(&records.count, 1u);
  if (slot >= ${PROBE_CAPACITY}u) { return; }
  var valid = cell(8.0,24.0)==1u && cell(8.0,56.0)==1u && cell(32.0,24.0)==0u && cell(32.0,56.0)==0u;
  var identity = 0u;
  for (var bit = 0u; bit < 16u; bit++) {
    let horizontal = f32((bit+2u)*24u+8u);
    let top = cell(horizontal,24.0); let bottom = cell(horizontal,56.0);
    valid = valid && top < 2u && bottom < 2u && top != bottom;
    identity = identity | ((top & 1u) << bit);
  }
  records.rows[slot] = vec4<u32>(identity,select(0u,1u,valid),u32(extent.x),u32(extent.y));
}`;
}

export class IdentityProbe implements Upscaler {
  private device: GPUDevice | null = null;
  private records: GPUBuffer | null = null;
  private readback: GPUBuffer | null = null;
  private dimensions: GPUBuffer | null = null;
  private layout: GPUBindGroupLayout | null = null;
  private compute: GPUComputePipeline | null = null;
  private sampler: GPUSampler | null = null;
  private sampled: GPUBindGroup | null = null;
  encoded = 0;

  constructor(readonly inner: Upscaler, readonly beforeEncode: (index: number) => void) {}
  get id() { return this.inner.id; }
  get label() { return this.inner.label; }
  get scaleFactor() { return this.inner.scaleFactor; }
  get neural() { return this.inner.neural; }

  configure(config: UpscalerConfig): void {
    this.inner.configure(config);
    this.device = config.device;
    const bytes = 16 + PROBE_CAPACITY * 16;
    this.records ??= config.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.readback ??= config.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.dimensions ??= config.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    config.device.queue.writeBuffer(this.dimensions, 0, new Float32Array([config.source.width, config.source.height, 0, 0]));
    this.sampler ??= config.device.createSampler({ minFilter: 'nearest', magFilter: 'nearest' });
    const external = config.sourceKind === 'external';
    this.layout = config.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, ...(external ? { externalTexture: {} } : { texture: { sampleType: 'float' as const } }) },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ] });
    this.compute = config.device.createComputePipeline({ layout: config.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module: config.device.createShaderModule({ code: probeShader(external) }), entryPoint: 'main' } });
    this.sampled = config.sampledSourceView ? this.bind(config.sampledSourceView) : null;
  }

  private bind(source: GPUTextureView | GPUExternalTexture): GPUBindGroup {
    if (!this.device || !this.layout || !this.records || !this.dimensions || !this.sampler) throw new Error('Unconfigured identity probe');
    return this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: source }, { binding: 1, resource: this.sampler },
      { binding: 2, resource: { buffer: this.records } }, { binding: 3, resource: { buffer: this.dimensions } },
    ] });
  }

  encode(context: EncodeContext): void {
    if (!this.compute || this.encoded >= PROBE_CAPACITY) throw new Error('Identity diagnostic capacity/configuration');
    this.beforeEncode(this.encoded++);
    this.inner.encode(context);
    const group = context.frame.kind === 'external' ? this.bind(context.frame.texture) : this.sampled;
    if (!group) throw new Error('Missing sampled identity input');
    const pass = context.encoder.beginComputePass({ label: 'm1010r:identity' });
    pass.setPipeline(this.compute); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
  }

  async readAfterPause(): Promise<Uint32Array> {
    if (!this.device || !this.records || !this.readback) throw new Error('No identity records');
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.records, 0, this.readback, 0, 16 + PROBE_CAPACITY * 16);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const rows = new Uint32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    if (rows[0] !== this.encoded) throw new Error('GPU and CPU identity record counts differ');
    return rows;
  }

  destroy(): void {
    this.inner.destroy(); this.records?.destroy(); this.readback?.destroy(); this.dimensions?.destroy();
    this.compute = null; this.sampled = null;
    this.records = null; this.readback = null; this.dimensions = null; this.device = null;
  }
}