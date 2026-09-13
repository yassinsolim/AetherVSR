import { expect, it, vi } from 'vitest';
import { acquireGpu } from '../src/core/gpu/device.js';

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