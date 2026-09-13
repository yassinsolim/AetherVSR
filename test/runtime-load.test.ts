import { expect, it, vi } from 'vitest';
import { LoadedUpscaler } from '../src/bench/runtime-load.js';
import type { EncodeContext, Upscaler } from '../src/core/types.js';

it('repeats only the requested frames and brackets all encodes once', () => {
  const encode = vi.fn();
  const inner = { encode, id: 'test', label: 'test', scaleFactor: 2 } as unknown as Upscaler;
  const load = { passes: 3, frames: 1 };
  const wrapped = new LoadedUpscaler(inner, load);
  const context = { timing: { querySet: {} as GPUQuerySet, beginIndex: 2, endIndex: 3 } } as EncodeContext;
  wrapped.encode(context);
  expect(encode).toHaveBeenCalledTimes(4);
  const timings = encode.mock.calls.map(call => (call[0] as EncodeContext).timing);
  expect(timings.map(timing => timing?.beginIndex)).toEqual([2, undefined, undefined, undefined]);
  expect(timings.map(timing => timing?.endIndex)).toEqual([undefined, undefined, undefined, 3]);
  wrapped.encode(context);
  expect(encode).toHaveBeenCalledTimes(5);
  expect(encode.mock.lastCall?.[0]).toEqual(context);
});