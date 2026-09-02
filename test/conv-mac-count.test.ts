import { describe, expect, it } from 'vitest';
import { convMacCount } from '../src/bench/conv.wgsl.js';

/**
 * `convMacCount` is the denominator of every GMAC/s figure this project
 * publishes, so an error here silently mis-scales an entire milestone rather
 * than failing loudly.
 *
 * Its contract is deliberately narrow: it multiplies out an *issued* extent
 * that the caller derives from the same numbers it passes to
 * `dispatchWorkgroups`. It does not infer the extent from a block size,
 * because how far a dispatch overhangs the image depends on which kernel
 * variant is running — the naive one returns before accumulating, every other
 * one guards only the store.
 */
describe('convMacCount', () => {
  it('is 9 * inC MACs per output pixel per output channel', () => {
    expect(convMacCount(0, 4, 2, 3)).toBe(0);
    expect(convMacCount(8, 4, 2, 3)).toBe(8 * 4 * 2 * 3 * 9);
    expect(convMacCount(1, 1, 1, 1)).toBe(9);
  });

  it('counts padded border taps, because the shader still issues them', () => {
    // A 1x1 image still runs all nine taps; eight read zero. Counting only the
    // useful one would report nine times the real throughput.
    expect(convMacCount(1, 1, 1, 1)).toBe(9);
  });

  it('scales linearly in every dimension', () => {
    const base = convMacCount(16, 16, 4, 4);
    expect(convMacCount(32, 16, 4, 4)).toBe(base * 2);
    expect(convMacCount(16, 32, 4, 4)).toBe(base * 2);
    expect(convMacCount(16, 16, 8, 4)).toBe(base * 2);
    expect(convMacCount(16, 16, 4, 8)).toBe(base * 2);
  });

  it('is zero when any extent is zero', () => {
    expect(convMacCount(0, 16, 4, 4)).toBe(0);
    expect(convMacCount(16, 0, 4, 4)).toBe(0);
    expect(convMacCount(16, 16, 0, 4)).toBe(0);
  });

  it('reflects a dispatch that overhangs the image', () => {
    // 854 wide dispatched in 32-pixel steps covers 864. The overhang is
    // accumulated and only its stores are discarded, so it is real issued work
    // and must appear in the denominator. Passing the image width instead
    // would understate the extent by 1.2% and understate throughput with it.
    const issuedWidth = Math.ceil(854 / 32) * 32;
    expect(issuedWidth).toBe(864);
    expect(convMacCount(issuedWidth, 480, 16, 16)).toBeGreaterThan(
      convMacCount(854, 480, 16, 16),
    );
    expect(convMacCount(issuedWidth, 480, 16, 16) / convMacCount(854, 480, 16, 16)).toBeCloseTo(
      864 / 854,
      10,
    );
  });

  it('reflects a vertical overhang the same way', () => {
    // The blocked kernel computes blockY rows per invocation, so height rounds
    // to tileY * blockY. 540 dispatched in 8-row steps covers 544.
    const issuedHeight = Math.ceil(540 / 8) * 8;
    expect(issuedHeight).toBe(544);
    expect(convMacCount(960, issuedHeight, 12, 12)).toBe(960 * 544 * 12 * 12 * 9);
  });

  it('agrees with the published headline workload', () => {
    // 1280x720 C16 with an 8x4 workgroup and 2x2 blocking divides exactly, so
    // the issued extent is the image extent and the published 2.1234 GMAC
    // stands.
    expect(Math.ceil(1280 / (8 * 2)) * 8 * 2).toBe(1280);
    expect(Math.ceil(720 / (4 * 2)) * 4 * 2).toBe(720);
    expect(convMacCount(1280, 720, 16, 16) / 1e9).toBeCloseTo(2.1234, 3);
  });
});
