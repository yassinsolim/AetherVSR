import { describe, expect, it, vi } from 'vitest';
import { GpuTimer } from '../src/core/metrics/gpu-timer.js';

/**
 * These tests exercise the timer's *slot pool*, which is our state machine, not
 * WebGPU's. The device stand-in below records nothing and asserts nothing about
 * GPU behaviour; it exists only so the pool can be driven deterministically.
 *
 * The bug this guards against is specific and was real: a frame that throws
 * between `begin()` and `afterSubmit()` used to leave its slot marked busy, so
 * a handful of failures silently disabled GPU timing for the whole session —
 * the overlay would keep showing stale numbers with no indication why.
 */

// `GPUBufferUsage` is a namespace of spec constants, not behaviour. Node has no
// WebGPU globals, so the real values are declared here; stubbing them is not
// mocking an API, it is supplying two integers the module ORs together.
vi.stubGlobal('GPUBufferUsage', {
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  QUERY_RESOLVE: 0x0200,
  MAP_READ: 0x0001,
});
vi.stubGlobal('GPUMapMode', { READ: 0x0001 });

interface FakeBuffer {
  destroyed: boolean;
  mapAsync: () => Promise<void>;
  getMappedRange: () => ArrayBuffer;
  unmap: () => void;
  destroy: () => void;
}

function makeDevice(options: { readonly deltaNs?: bigint; readonly reject?: boolean } = {}): {
  device: GPUDevice;
  settle: (count?: number) => Promise<void>;
} {
  const pending: (() => void)[] = [];

  const makeBuffer = (): FakeBuffer => {
    const bytes = new ArrayBuffer(16);
    const view = new BigInt64Array(bytes);
    view[0] = 0n;
    view[1] = options.deltaNs ?? 0n;
    return {
      destroyed: false,
      mapAsync: () =>
        new Promise<void>((resolve, reject) => {
          pending.push(() => (options.reject ? reject(new Error('device lost')) : resolve()));
        }),
      getMappedRange: () => bytes,
      unmap: () => {},
      destroy() {
        this.destroyed = true;
      },
    };
  };

  const device = {
    createQuerySet: () => ({ destroy: () => {} }),
    createBuffer: makeBuffer,
  } as unknown as GPUDevice;

  return {
    device,
    settle: async (count = pending.length) => {
      for (let index = 0; index < count; index++) {
        const next = pending.shift();
        next?.();
        // Let the .then/.catch/.finally chain run to completion.
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
      }
    },
  };
}

const encoder = {
  resolveQuerySet: () => {},
  copyBufferToBuffer: () => {},
} as unknown as GPUCommandEncoder;

describe('GpuTimer slot pool', () => {
  it('keeps drain pending until all four readbacks publish their contexts', async () => {
    const { device, settle } = makeDevice({ deltaNs: 2_000_000n });
    const onSample = vi.fn();
    const timer = new GpuTimer<{ readonly sequence: number }>(device, onSample);
    for (const sequence of [1, 2, 3, 4]) {
      expect(timer.begin({ sequence })).not.toBeNull();
      timer.end(encoder);
      timer.afterSubmit();
    }
    expect(timer.begin({ sequence: 5 })).toBeNull();

    const drained = vi.fn();
    const draining = timer.drain().then(drained);
    await Promise.resolve();
    expect(drained).not.toHaveBeenCalled();
    expect(onSample).not.toHaveBeenCalled();

    await settle(1);
    expect(onSample).toHaveBeenCalledExactlyOnceWith(2, { sequence: 1 });
    expect(drained).not.toHaveBeenCalled();

    await settle();
    await draining;
    expect(drained).toHaveBeenCalledExactlyOnceWith(undefined);
    expect(onSample.mock.calls).toEqual([
      [2, { sequence: 1 }], [2, { sequence: 2 }], [2, { sequence: 3 }], [2, { sequence: 4 }],
    ]);
    expect(timer.begin({ sequence: 5 })).not.toBeNull();
    timer.abort();
    timer.destroy();
  });

  it('resolves drain cleanly when pending readbacks reject', async () => {
    const { device, settle } = makeDevice({ reject: true });
    const onSample = vi.fn();
    const timer = new GpuTimer(device, onSample, 1);
    timer.begin();
    timer.end(encoder);
    timer.afterSubmit();
    const draining = timer.drain();
    await settle();
    await expect(draining).resolves.toBeUndefined();
    expect(onSample).not.toHaveBeenCalled();
    expect(timer.last).toBeNaN();
    expect(timer.begin()).not.toBeNull();
    timer.abort();
    timer.destroy();
  });

  it('resolves an empty drain without waiting for a readback', async () => {
    const { device } = makeDevice();
    const timer = new GpuTimer(device, vi.fn());
    await expect(timer.drain()).resolves.toBeUndefined();
    timer.destroy();
  });

  it('does not wait for readbacks submitted after drain was called', async () => {
    const { device, settle } = makeDevice({ deltaNs: 1_000_000n });
    const onSample = vi.fn();
    const timer = new GpuTimer<number>(device, onSample, 2);
    timer.begin(1);
    timer.end(encoder);
    timer.afterSubmit();
    const draining = timer.drain();
    timer.begin(2);
    timer.end(encoder);
    timer.afterSubmit();

    await settle(1);
    await expect(draining).resolves.toBeUndefined();
    expect(onSample).toHaveBeenCalledExactlyOnceWith(1, 1);
    await settle();
    await timer.drain();
    expect(onSample).toHaveBeenLastCalledWith(1, 2);
    timer.destroy();
  });

  it('preserves each begin context while multiple readbacks are pending', async () => {
    const { device, settle } = makeDevice({ deltaNs: 2_000_000n });
    const onSample = vi.fn();
    const timer = new GpuTimer<{ readonly sequence: number }>(device, onSample, 2);

    for (const sequence of [1, 2]) {
      timer.begin({ sequence });
      timer.end(encoder);
      timer.afterSubmit();
    }
    expect(onSample).not.toHaveBeenCalled();
    await settle();

    expect(onSample.mock.calls).toEqual([[2, { sequence: 1 }], [2, { sequence: 2 }]]);
    timer.begin({ sequence: 3 });
    timer.abort();
    timer.begin({ sequence: 4 });
    timer.end(encoder);
    timer.afterSubmit();
    await settle();
    expect(onSample).toHaveBeenLastCalledWith(2, { sequence: 4 });
  });

  it('defaults an omitted context to undefined', async () => {
    const { device, settle } = makeDevice({ deltaNs: 1_000_000n });
    const onSample = vi.fn();
    const timer = new GpuTimer(device, onSample, 1);
    timer.begin();
    timer.end(encoder);
    timer.afterSubmit();
    await settle();
    expect(onSample).toHaveBeenCalledExactlyOnceWith(1, undefined);
  });

  it('publishes only the current epoch context when old and new readbacks settle', async () => {
    const { device, settle } = makeDevice({ deltaNs: 3_000_000n });
    const onSample = vi.fn();
    const timer = new GpuTimer<number>(device, onSample, 2);
    timer.begin(1);
    timer.end(encoder);
    timer.afterSubmit();
    timer.newEpoch();
    timer.begin(2);
    timer.end(encoder);
    timer.afterSubmit();
    await settle();
    expect(onSample).toHaveBeenCalledExactlyOnceWith(3, 2);
  });

  it('does not publish a pending readback after destruction even if mapping succeeds', async () => {
    const { device, settle } = makeDevice({ deltaNs: 4_000_000n });
    const onSample = vi.fn();
    const timer = new GpuTimer<string>(device, onSample, 1);
    timer.begin('pending');
    timer.end(encoder);
    timer.afterSubmit();
    timer.destroy();
    await settle();
    expect(onSample).not.toHaveBeenCalled();
    expect(timer.last).toBeNaN();
    expect(timer.begin('destroyed')).toBeNull();
  });

  it('hands out distinct query indices per pooled slot', () => {
    const { device } = makeDevice();
    const timer = new GpuTimer(device, () => {}, 3);
    const seen = new Set<number>();
    for (let i = 0; i < 3; i++) {
      const timing = timer.begin();
      expect(timing).not.toBeNull();
      seen.add(timing?.beginIndex ?? -1);
      timer.end(encoder);
      timer.afterSubmit();
    }
    expect(seen.size).toBe(3);
  });

  it('returns null instead of blocking when every slot is awaiting readback', () => {
    const { device } = makeDevice();
    const timer = new GpuTimer(device, () => {}, 2);
    for (let i = 0; i < 2; i++) {
      timer.begin();
      timer.end(encoder);
      timer.afterSubmit();
    }
    expect(timer.begin()).toBeNull();
  });

  it('recycles slots once readback completes', async () => {
    const { device, settle } = makeDevice({ deltaNs: 2_000_000n });
    const samples: number[] = [];
    const timer = new GpuTimer(device, (ms) => samples.push(ms), 2);

    for (let i = 0; i < 2; i++) {
      timer.begin();
      timer.end(encoder);
      timer.afterSubmit();
    }
    expect(timer.begin()).toBeNull();

    await settle();

    expect(samples).toEqual([2, 2]);
    expect(timer.begin()).not.toBeNull();
  });

  it('releases a slot when the frame aborts before submitting', () => {
    // The pipeline calls abort() from its catch block. Without it the pool
    // drains and GPU timing dies permanently after a few bad frames.
    const { device } = makeDevice();
    const timer = new GpuTimer(device, () => {}, 1);

    for (let i = 0; i < 5; i++) {
      expect(timer.begin()).not.toBeNull();
      timer.abort();
    }
    expect(timer.begin()).not.toBeNull();
  });

  it('does not double-release when abort follows a completed frame', async () => {
    const { device, settle } = makeDevice({ deltaNs: 1_000_000n });
    const timer = new GpuTimer(device, () => {}, 1);
    timer.begin();
    timer.end(encoder);
    timer.afterSubmit();
    timer.abort(); // no active slot; must be a no-op
    expect(timer.begin()).toBeNull();
    await settle();
    expect(timer.begin()).not.toBeNull();
  });

  it('recycles the slot when readback rejects', async () => {
    const { device, settle } = makeDevice({ reject: true });
    const onSample = vi.fn();
    const timer = new GpuTimer(device, onSample, 1);
    timer.begin();
    timer.end(encoder);
    timer.afterSubmit();

    await settle();

    expect(onSample).not.toHaveBeenCalled();
    expect(timer.begin()).not.toBeNull();
  });

  it('ignores a negative interval rather than publishing it', async () => {
    const { device, settle } = makeDevice({ deltaNs: -5_000_000n });
    const onSample = vi.fn();
    const timer = new GpuTimer(device, onSample, 1);
    timer.begin();
    timer.end(encoder);
    timer.afterSubmit();

    await settle();

    expect(onSample).not.toHaveBeenCalled();
    expect(timer.last).toBeNaN();
  });

  it('drops readbacks that were started before a new measurement epoch', async () => {
    // Without epochs, a stats reset or an upscaler swap would be contaminated
    // by up to poolSize samples measured under the previous configuration.
    const { device, settle } = makeDevice({ deltaNs: 9_000_000n });
    const samples: number[] = [];
    const timer = new GpuTimer(device, (ms) => samples.push(ms), 2);

    timer.begin();
    timer.end(encoder);
    timer.afterSubmit();

    timer.newEpoch();
    await settle();

    expect(samples).toEqual([]);
    expect(timer.last).toBeNaN();
  });

  it('keeps measuring after a new epoch', async () => {
    const { device, settle } = makeDevice({ deltaNs: 3_000_000n });
    const samples: number[] = [];
    const timer = new GpuTimer(device, (ms) => samples.push(ms), 2);

    timer.newEpoch();
    timer.begin();
    timer.end(encoder);
    timer.afterSubmit();
    await settle();

    expect(samples).toEqual([3]);
  });

  it('releases epoch-discarded slots back to the pool', async () => {
    const { device, settle } = makeDevice({ deltaNs: 1_000_000n });
    const timer = new GpuTimer(device, () => {}, 1);
    timer.begin();
    timer.end(encoder);
    timer.afterSubmit();
    timer.newEpoch();
    await settle();
    expect(timer.begin()).not.toBeNull();
  });

  it('end() and afterSubmit() are safe with no claimed slot', () => {
    const { device } = makeDevice();
    const timer = new GpuTimer(device, () => {}, 1);
    expect(() => {
      timer.end(encoder);
      timer.afterSubmit();
      timer.abort();
    }).not.toThrow();
  });
});
