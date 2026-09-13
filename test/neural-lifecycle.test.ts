import { describe, expect, it, vi } from 'vitest';
import { NeuralUpscaler } from '../src/core/upscale/neural-upscaler.js';
import type { PackedModel } from '../src/core/neural/model.js';
import type { EncodeContext, FrameTexture, UpscalerConfig } from '../src/core/types.js';

vi.stubGlobal('GPUBufferUsage', {
  MAP_READ: 0x0001, COPY_SRC: 0x0004, COPY_DST: 0x0008,
  UNIFORM: 0x0040, STORAGE: 0x0080, QUERY_RESOLVE: 0x0200,
});
vi.stubGlobal('GPUTextureUsage', {
  TEXTURE_BINDING: 0x0004, STORAGE_BINDING: 0x0008, RENDER_ATTACHMENT: 0x0010,
});
vi.stubGlobal('GPUShaderStage', { FRAGMENT: 0x0002 });
vi.stubGlobal('GPUMapMode', { READ: 0x0001 });

function makeBuffer() {
  const bytes = new BigInt64Array([0n, 1_000_000n, 0n, 2_000_000n, 0n, 3_000_000n]).buffer;
  let resolveMap = (): void => { throw new Error('no pending map'); };
  let rejectMap = (): void => { throw new Error('no pending map'); };
  return {
    mapAsync: vi.fn(() => new Promise<void>((resolve, reject) => {
      resolveMap = resolve;
      rejectMap = () => reject(new Error('map failed'));
    })),
    getMappedRange: vi.fn(() => bytes),
    unmap: vi.fn(),
    destroy: vi.fn(),
    settle: async (reject = false) => {
      if (reject) rejectMap();
      else resolveMap();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    },
  };
}

function makeHarness(sourceKind: UpscalerConfig['sourceKind'] = 'external') {
  const createView = vi.fn(() => ({}));
  const device = {
    features: new Set(['timestamp-query']),
    queue: { writeBuffer: vi.fn() },
    createBuffer: vi.fn<(descriptor: GPUBufferDescriptor) => ReturnType<typeof makeBuffer>>(makeBuffer),
    createTexture: vi.fn(() => ({ createView, destroy: vi.fn() })),
    createQuerySet: vi.fn(() => ({ destroy: vi.fn() })),
    createShaderModule: vi.fn(() => ({})),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({ getBindGroupLayout: () => ({}) })),
    createRenderPipeline: vi.fn(() => ({ getBindGroupLayout: () => ({}) })),
    createSampler: vi.fn(() => ({})),
    createBindGroup: vi.fn((descriptor: GPUBindGroupDescriptor) => ({ descriptor })),
  };
  const makePass = () => ({
    setPipeline: vi.fn(), setBindGroup: vi.fn(), dispatchWorkgroups: vi.fn(),
    draw: vi.fn(), end: vi.fn(),
  });
  const encoder = {
    beginComputePass: vi.fn((descriptor: GPUComputePassDescriptor) => ({ ...makePass(), descriptor })),
    beginRenderPass: vi.fn((descriptor: GPURenderPassDescriptor) => ({ ...makePass(), descriptor })),
    resolveQuerySet: vi.fn(), copyBufferToBuffer: vi.fn(),
  };
  const weights = new Float32Array(16);
  const model = {
    features: 4, depth: 1,
    file: { architecture: 'aethersr-resizeconv', normalisation: { mean: [0, 0, 0], scale: [1, 1, 1] } },
    stemWeights: weights, stemBias: weights, bodyWeights: [weights], bodyBias: [weights],
    headWeights: weights, headBias: weights,
  } as unknown as PackedModel;
  const upscaler = new NeuralUpscaler(model, { useF16: false });
  const sampledSourceView = {} as GPUTextureView;
  const config: UpscalerConfig = {
    device: device as unknown as GPUDevice, source: { width: 16, height: 16 },
    target: { width: 32, height: 32 }, targetFormat: 'rgba8unorm', sourceKind,
    ...(sourceKind === 'sampled' ? { sampledSourceView } : {}),
  };
  const frame: FrameTexture = sourceKind === 'external'
    ? { kind: 'external', texture: {} as GPUExternalTexture }
    : { kind: 'sampled', view: sampledSourceView };
  const context: EncodeContext = {
    encoder: encoder as unknown as GPUCommandEncoder, frame, target: {} as GPUTextureView,
    timing: { querySet: {} as GPUQuerySet, beginIndex: 0, endIndex: 1 },
  };
  const allocationCounts = () => Object.fromEntries<number>([
    ...Object.entries(device).filter(([name]) => name.startsWith('create')),
    ['createView', createView] as const,
  ].map(([name, mock]) => [name, (mock as typeof createView).mock.calls.length]));
  const group = (label: string) => {
    const descriptor = device.createBindGroup.mock.calls.slice().reverse().find(
      ([entry]) => entry.label === `aethervsr:nn:${label}`,
    )?.[0];
    if (!descriptor) throw new Error(`missing group: ${label}`);
    return Array.from(descriptor.entries);
  };
  const staging = () => {
    const index = device.createBuffer.mock.calls
      .map(([descriptor]) => descriptor.usage & GPUBufferUsage.MAP_READ)
      .lastIndexOf(GPUBufferUsage.MAP_READ);
    const result = device.createBuffer.mock.results[index];
    if (result?.type !== 'return') throw new Error('missing staging buffer');
    return result.value;
  };
  const startRead = (input = context) => {
    upscaler.encode(input);
    upscaler.encode(input);
    return staging();
  };
  return { upscaler, config, context, device, encoder, createView, allocationCounts, group, startRead };
}

describe('NeuralUpscaler resource lifecycle', () => {
  it('preallocates external-path groups and views on every configuration', () => {
    const harness = makeHarness();
    const { upscaler, config, context, device, encoder, createView, group } = harness;
    let previousSource: GPUBindingResource | undefined;
    for (let generation = 0; generation < 3; generation++) {
      upscaler.configure(config);
      const sourceView = group('stem')[0]?.resource;
      expect(sourceView).toBe(createView.mock.results[generation * 2]?.value);
      expect(sourceView).not.toBe(previousSource);
      expect(group('head')[5]?.resource).toBe(sourceView);
      expect(group('head')[3]?.resource).toBe(group('blit')[1]?.resource);
      const counts = harness.allocationCounts();
      for (let frameIndex = 0; frameIndex < 3; frameIndex++) upscaler.encode(context);
      expect(harness.allocationCounts()).toEqual({ ...counts, createBindGroup: counts.createBindGroup! + 3 });
      expect(device.createBindGroup.mock.calls.slice(-3).map(([entry]) => entry.label))
        .toEqual(Array(3).fill('aethervsr:ingest:bindgroup'));
      expect(encoder.beginRenderPass.mock.calls.at(-2)?.[0].timestampWrites).toEqual({
        querySet: context.timing?.querySet, beginningOfPassWriteIndex: 0,
      });
      expect(encoder.beginRenderPass.mock.calls.at(-1)?.[0].timestampWrites).toEqual({
        querySet: context.timing?.querySet, endOfPassWriteIndex: 1,
      });
      previousSource = sourceView;
    }
    upscaler.destroy();
  });

  it('preallocates copy-import groups for the supplied view on every configuration', () => {
    const harness = makeHarness('sampled');
    const { upscaler, config, context, encoder, group } = harness;
    for (let generation = 0; generation < 3; generation++) {
      const view = generation === 0 ? config.sampledSourceView! : {} as GPUTextureView;
      const source = { width: 16 * (generation + 1), height: 16 };
      upscaler.configure({ ...config, source, target: { width: source.width * 2, height: 32 }, sampledSourceView: view });
      expect(group('stem')[0]?.resource).toBe(view);
      expect(group('head')[5]?.resource).toBe(view);
      expect(group('head')[3]?.resource).toBe(group('blit')[1]?.resource);
      const counts = harness.allocationCounts();
      for (let frameIndex = 0; frameIndex < 3; frameIndex++) {
        upscaler.encode({ ...context, frame: { kind: 'sampled', view } });
        expect(harness.allocationCounts()).toEqual(counts);
      }
      expect(encoder.beginComputePass.mock.calls.at(-3)?.[0].timestampWrites).toEqual({
        querySet: context.timing?.querySet, beginningOfPassWriteIndex: 0,
      });
      expect(encoder.beginRenderPass.mock.calls).toHaveLength((generation + 1) * 3);
      expect(encoder.beginRenderPass.mock.calls.at(-1)?.[0].timestampWrites).toEqual({
        querySet: context.timing?.querySet, endOfPassWriteIndex: 1,
      });
    }
    upscaler.destroy();
  });

  it('rejects legacy sampled configuration without a prepared source view', () => {
    const harness = makeHarness();
    const counts = harness.allocationCounts();
    expect(() => harness.upscaler.configure({ ...harness.config, sourceKind: 'sampled' }))
      .toThrow(/requires sampledSourceView/);
    expect(() => harness.upscaler.encode(harness.context)).toThrow(/before configure/);
    expect(harness.allocationCounts()).toEqual(counts);
  });

  it('rejects a changed sampled view instead of lazily allocating groups', () => {
    const harness = makeHarness('sampled');
    const { upscaler, config, context, encoder, group } = harness;
    upscaler.configure(config);
    const counts = harness.allocationCounts();
    const changed = { ...context, frame: { kind: 'sampled' as const, view: {} as GPUTextureView } };
    expect(() => upscaler.encode(changed)).toThrow(/source view is not prepared/);
    expect(harness.allocationCounts()).toEqual(counts);
    expect(encoder.beginComputePass).not.toHaveBeenCalled();
    expect(encoder.beginRenderPass).not.toHaveBeenCalled();
    expect(group('stem')[0]?.resource).toBe(config.sampledSourceView);
    upscaler.encode(context);
    expect(harness.allocationCounts()).toEqual(counts);
    upscaler.destroy();
    expect(() => upscaler.encode(context)).toThrow(/configure\(\)/);
    expect(harness.allocationCounts()).toEqual(counts);
    upscaler.destroy();
  });

  it.each([false, true])('ignores old readbacks after reconfigure/destroy (reject=%s)', async (reject) => {
    const { upscaler, config, startRead, encoder, context } = makeHarness();
    upscaler.configure(config);
    const first = startRead();
    upscaler.configure(config);
    const second = startRead();
    upscaler.destroy();
    upscaler.destroy();
    upscaler.configure(config);
    const current = startRead();
    const copies = encoder.copyBufferToBuffer.mock.calls.length;
    for (const old of [second, first]) {
      expect(old.destroy).toHaveBeenCalledExactlyOnceWith();
      await old.settle(reject);
      expect(old.getMappedRange).not.toHaveBeenCalled();
      expect(old.unmap).not.toHaveBeenCalled();
      expect(upscaler.stageTiming).toBeNull();
      expect(current.unmap).not.toHaveBeenCalled();
      upscaler.encode(context);
      expect(current.mapAsync).toHaveBeenCalledTimes(1);
      expect(encoder.copyBufferToBuffer).toHaveBeenCalledTimes(copies);
    }
    await current.settle();
    expect(current.unmap).toHaveBeenCalledTimes(1);
    expect(upscaler.stageTiming?.body.samples).toBe(1);
    upscaler.encode(context);
    expect(encoder.copyBufferToBuffer).toHaveBeenCalledTimes(copies + 1);
    upscaler.encode(context);
    upscaler.destroy();
    upscaler.destroy();
    await current.settle(reject);
    expect(upscaler.stageTiming).toBeNull();
    expect(current.getMappedRange).toHaveBeenCalledTimes(1);
    expect(current.unmap).toHaveBeenCalledTimes(1);
  });

  it.each(['reject', 'range-error'] as const)('cleans up %s and resumes diagnostics', async (failure) => {
    const { upscaler, config, context, encoder, startRead } = makeHarness();
    upscaler.configure(config);
    const staging = startRead();
    if (failure === 'range-error') staging.getMappedRange.mockImplementationOnce(() => {
      throw new Error('range unavailable');
    });
    await staging.settle(failure === 'reject');
    expect(staging.unmap).toHaveBeenCalledTimes(1);
    expect(upscaler.stageTiming).toBeNull();
    upscaler.encode(context);
    expect(encoder.copyBufferToBuffer).toHaveBeenCalledTimes(2);
    upscaler.encode(context);
    await staging.settle();
    expect(staging.unmap).toHaveBeenCalledTimes(2);
    expect(upscaler.stageTiming?.body.samples).toBe(1);
    upscaler.destroy();
  });

  it.each([false, true])('uses the originating sampled stem timing flag (whole-stage timing=%s)', async (wholeStage) => {
    const { upscaler, config, context, startRead } = makeHarness('sampled');
    upscaler.configure(config);
    const staging = startRead({ ...context, timing: wholeStage ? context.timing : null });
    upscaler.encode({ ...context, timing: wholeStage ? null : context.timing });
    await staging.settle();
    expect(upscaler.stageTiming?.stem === null).toBe(wholeStage);
    expect(upscaler.stageTiming?.body.samples).toBe(1);
    upscaler.destroy();
  });
});