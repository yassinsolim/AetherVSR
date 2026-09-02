import { describe, expect, it } from 'vitest';
import { convMacCount } from '../src/bench/conv.wgsl.js';

/**
 * `convMacCount` is the denominator of every published GMAC/s figure, so an
 * error here silently mis-scales throughput rather than failing loudly.
 */
describe('convMacCount', () => {
  it('counts 9 MACs per output pixel per input/output channel pair', () => {
    // 3x3 kernel over a 10x10 image, 2 in -> 3 out.
    expect(convMacCount(10, 10, 2, 3, 1)).toBe(10 * 10 * 2 * 3 * 9);
  });

  it('is unaffected by register blocking when the width divides evenly', () => {
    const unblocked = convMacCount(1280, 720, 16, 16, 1);
    for (const blockX of [2, 4, 8]) {
      expect(convMacCount(1280, 720, 16, 16, blockX)).toBe(unblocked);
    }
  });

  it('counts the tail invocation that accumulates outputs it never stores', () => {
    // At blockX=4 a width of 1281 needs ceil(1281/4)=321 invocations per row,
    // each accumulating 4 outputs — 1284 columns of issued work. Only the
    // stores are guarded, so the MACs are still issued and must be counted or
    // throughput is overstated.
    expect(convMacCount(1281, 8, 2, 2, 4)).toBe(1284 * 8 * 2 * 2 * 9);
    expect(convMacCount(1281, 8, 2, 2, 4)).toBeGreaterThan(convMacCount(1281, 8, 2, 2, 1));
  });

  it('never counts fewer MACs than the useful output requires', () => {
    for (const width of [1, 7, 13, 1279, 1280, 1281]) {
      for (const blockX of [1, 2, 4, 8]) {
        expect(convMacCount(width, 4, 3, 3, blockX)).toBeGreaterThanOrEqual(
          width * 4 * 3 * 3 * 9,
        );
      }
    }
  });

  it('defaults to no blocking when blockX is omitted', () => {
    expect(convMacCount(1281, 8, 2, 2)).toBe(1281 * 8 * 2 * 2 * 9);
  });
});
