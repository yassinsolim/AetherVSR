import { afterEach, describe, expect, it, vi } from 'vitest';
import { acquireGpu, watchDeviceFailures } from '../src/core/gpu/device.js';

it('actually withholds harness timestamp queries from the device request', async () => {
  const requestDevice = vi.fn().mockResolvedValue({ features: new Set(), limits: {} });
  vi.stubGlobal('navigator', { gpu: { requestAdapter: () => Promise.resolve({
    requestDevice, features: new Set(['timestamp-query', 'shader-f16']), limits: {}, info: {},
  }), getPreferredCanvasFormat: () => 'rgba8unorm' } });
  try {
    const gpu = await acquireGpu({ optionalFeatures: ['shader-f16'], withheldFeatures: ['timestamp-query'] });
    expect(requestDevice).toHaveBeenCalledWith({ requiredFeatures: ['shader-f16'] });
    expect(gpu.capabilities.timestampQuery).toBe(false);
  } finally { vi.unstubAllGlobals(); }
});

function createDevice() {
  type LostInfo = Pick<GPUDeviceLostInfo, 'reason' | 'message'>;
  let lose!: (info: LostInfo) => void;
  const lost = new Promise<LostInfo>((resolve) => { lose = resolve; });
  const target = Object.assign(new EventTarget(), { lost });
  const add = vi.spyOn(target, 'addEventListener');
  const remove = vi.spyOn(target, 'removeEventListener');
  return { target, device: target as unknown as GPUDevice, lose, add, remove };
}

describe('watchDeviceFailures', () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it('preserves validation, generic-error and device-loss messages while subscribed', async () => {
    class ValidationError extends Error {}
    vi.stubGlobal('GPUValidationError', ValidationError);
    const { target, device, lose } = createDevice();
    const callback = vi.fn();
    const unsubscribe = watchDeviceFailures(device, callback);
    target.dispatchEvent(Object.assign(new Event('uncapturederror'), { error: new ValidationError('bad binding') }));
    target.dispatchEvent(new Event('uncapturederror'));
    lose({ reason: 'unknown', message: 'device unavailable' });
    await device.lost;
    expect(callback.mock.calls).toEqual([
      ['WebGPU validation error: bad binding'],
      ['WebGPU uncaptured error (see console)'],
      ['WebGPU device lost (unknown): device unavailable'],
    ]);
    unsubscribe();
  });

  it.each([false, true])('ignores device loss and stale listeners after unsubscribe (loss already queued: %s)', async (queued) => {
    const { target, device, lose, add, remove } = createDevice();
    const callback = vi.fn();
    const unsubscribe = watchDeviceFailures(device, callback);
    const stale = add.mock.calls[0]![1] as EventListener;
    if (queued) lose({ reason: 'destroyed', message: 'closed' });
    unsubscribe();
    unsubscribe();
    if (!queued) lose({ reason: 'destroyed', message: 'closed' });
    await device.lost;
    stale(new Event('uncapturederror'));
    target.dispatchEvent(new Event('uncapturederror'));
    expect(callback).not.toHaveBeenCalled();
    expect(remove).toHaveBeenCalledExactlyOnceWith('uncapturederror', stale);
  });

  it('repeatedly detaches without silencing a later attachment to the same device', async () => {
    const { device, target, lose, add, remove } = createDevice();
    const callback = vi.fn();
    const detachments: (() => void)[] = [];
    for (let index = 0; index < 3; index++) {
      const unsubscribe = watchDeviceFailures(device, callback);
      detachments.push(unsubscribe);
      unsubscribe();
    }
    const current = vi.fn();
    const unsubscribe = watchDeviceFailures(device, current);
    for (const detach of detachments) detach();
    target.dispatchEvent(new Event('uncapturederror'));
    lose({ reason: 'unknown', message: 'lost' });
    await device.lost;
    expect(callback).not.toHaveBeenCalled();
    expect(current.mock.calls).toEqual([
      ['WebGPU uncaptured error (see console)'], ['WebGPU device lost (unknown): lost'],
    ]);
    unsubscribe();
    expect(add).toHaveBeenCalledTimes(4);
    expect(remove.mock.calls).toEqual(add.mock.calls);
  });

  it('can unsubscribe inside a failure callback and suppress subsequent device loss', async () => {
    const { device, target, lose, remove } = createDevice();
    const callback = vi.fn(() => unsubscribe());
    const unsubscribe = watchDeviceFailures(device, callback);
    target.dispatchEvent(new Event('uncapturederror'));
    target.dispatchEvent(new Event('uncapturederror'));
    lose({ reason: 'unknown', message: 'lost' });
    await device.lost;
    expect(callback).toHaveBeenCalledTimes(1);
    expect(remove).toHaveBeenCalledTimes(1);
  });
});