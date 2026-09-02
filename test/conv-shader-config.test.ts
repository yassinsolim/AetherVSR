import { describe, expect, it } from 'vitest';
import { buildTiledConvShader, tiledSharedBytes } from '../src/bench/conv-tiled.wgsl.js';
import { buildPackedConvShader, packedSharedBytes } from '../src/bench/conv-packed.wgsl.js';
import { buildBlockedConvShader, blockedSharedBytes } from '../src/bench/conv-blocked.wgsl.js';
import { buildMatrixConvShader } from '../src/bench/conv-matrix.wgsl.js';

/**
 * These assertions guard the properties of a generated shader that the type
 * system cannot see and that a GPU will not complain about — it will just
 * compute something else. Numerical correctness is checked on the real device
 * by `verifyConv`; what is checked here is the shape of what gets generated,
 * and the refusals.
 *
 * The refusals matter as much as the code. Every rejected configuration here
 * is one that could otherwise be padded into "working" while quietly changing
 * the MAC count the throughput figures are computed from.
 */

const base = {
  inChannels: 16,
  outChannels: 16,
  tileX: 8,
  tileY: 8,
  blockX: 4,
  blockY: 2,
  outBlock: 8,
  activation: 'relu' as const,
  useF16: true,
  residual: false,
  weightLayout: 'oc-major' as const,
};

const count = (src: string, needle: string): number => src.split(needle).length - 1;

describe('workgroup storage accounting', () => {
  it('matches the array the tiled shader actually declares', () => {
    const cfg = { ...base, blockX: 4, tileX: 16, tileY: 8 };
    const bytes = tiledSharedBytes(cfg);
    // (16*4+2) * (8+2) = 660 elements of f16.
    expect(bytes).toBe(660 * 2);
    expect(buildTiledConvShader(cfg)).toContain('array<f16, 660>');
  });

  it('counts a vec4 element as four for the packed shader', () => {
    const cfg = { ...base, tileX: 8, tileY: 8, blockX: 4 };
    expect(packedSharedBytes(cfg)).toBe(tiledSharedBytes(cfg) * 4);
  });

  it('grows the staged tile vertically with blockY', () => {
    const one = blockedSharedBytes({ ...base, blockY: 1 });
    const two = blockedSharedBytes({ ...base, blockY: 2 });
    // Halo is a fixed 2 rows, so doubling blockY is less than doubling bytes.
    expect(two).toBeGreaterThan(one);
    expect(two).toBeLessThan(one * 2);
    // (8*4+2) * (8*2+2) vec4<f16> = 34*18*8 bytes.
    expect(two).toBe(34 * 18 * 8);
  });

  it('reports f32 as twice f16', () => {
    expect(blockedSharedBytes({ ...base, useF16: false })).toBe(blockedSharedBytes(base) * 2);
  });
});

describe('blocked shader structure', () => {
  it('emits one accumulator per output channel, row and column', () => {
    for (const [ob, bx, by] of [
      [8, 4, 2],
      [16, 2, 2],
      [1, 1, 1],
      [4, 2, 4],
    ] as const) {
      const src = buildBlockedConvShader({ ...base, outBlock: ob, blockX: bx, blockY: by });
      expect(count(src, ': f16 = biases['), ` ob${ob} b${bx}x${by}`).toBe(ob * bx * by);
    }
  });

  it('emits one dot product per accumulator per tap', () => {
    const src = buildBlockedConvShader({ ...base, outBlock: 4, blockX: 2, blockY: 2 });
    // The 3x3 taps are a runtime loop, so the unrolled body holds one dot per
    // accumulator. Fewer would mean an accumulator is silently never updated.
    expect(count(src, '+= dot(')).toBe(4 * 2 * 2);
  });

  it('loads each staged value once and reuses it across output channels', () => {
    // The whole justification for output-channel blocking. One load per
    // (row, column), not one per (row, column, output channel).
    const src = buildBlockedConvShader({ ...base, outBlock: 8, blockX: 2, blockY: 2 });
    expect(count(src, 'let v')).toBe(2 * 2);
    expect(count(src, '+= dot(')).toBe(8 * 2 * 2);
  });

  it('switches weight indexing with the layout and nothing else', () => {
    const oc = buildBlockedConvShader({ ...base, weightLayout: 'oc-major' });
    const tap = buildBlockedConvShader({ ...base, weightLayout: 'tap-major' });
    expect(oc).toContain('IN_GROUPS * 9u + wTap');
    expect(tap).toContain('wTap * OUT_C');
    expect(tap).not.toContain('IN_GROUPS * 9u + wTap');
    expect(count(oc, '+= dot(')).toBe(count(tap, '+= dot('));
  });

  it('adds the residual tap only when asked, unpacked from the grouped layout', () => {
    expect(buildBlockedConvShader({ ...base, residual: false })).not.toContain('+ input[');
    const res = buildBlockedConvShader({ ...base, residual: true });
    expect(count(res, '+ input[')).toBe(base.outBlock * base.blockX * base.blockY);
    expect(res).toContain('/ 4u) * W * H');
  });

  it('enables f16 only for the half-precision variant', () => {
    expect(buildBlockedConvShader({ ...base, useF16: true }).startsWith('enable f16;')).toBe(true);
    expect(buildBlockedConvShader({ ...base, useF16: false })).not.toContain('enable f16;');
  });

  it('emits the requested activation and no other', () => {
    expect(buildBlockedConvShader({ ...base, activation: 'relu' })).toContain('max(');
    expect(buildBlockedConvShader({ ...base, activation: 'tanh' })).toContain('tanh(');
    const none = buildBlockedConvShader({ ...base, activation: 'none' });
    expect(none).not.toContain('max(acc');
    expect(none).not.toContain('tanh(acc');
  });

  it('barriers on both sides of the staging loop', () => {
    // Two per input-channel iteration. Dropping the leading one is the classic
    // bug: a fast invocation refills the tile while a slow one still reads it,
    // and it only shows up under load.
    const src = buildBlockedConvShader(base);
    const body = src.slice(src.indexOf('for (var cg'), src.indexOf('if (ocBase >= OUT_C)'));
    expect(count(body, 'workgroupBarrier()')).toBe(2);
  });
});

describe('configuration refusals', () => {
  it('rejects channel counts the vec4 layouts cannot represent', () => {
    expect(() => buildPackedConvShader({ ...base, inChannels: 6 })).toThrow(/multiple of 4|% 4/);
    expect(() => buildBlockedConvShader({ ...base, inChannels: 6 })).toThrow(/% 4/);
  });

  it('rejects an output block that does not divide the output channels', () => {
    expect(() => buildBlockedConvShader({ ...base, outChannels: 12, outBlock: 8 })).toThrow(
      /outChannels % outBlock/,
    );
    expect(() => buildBlockedConvShader({ ...base, outChannels: 12, outBlock: 6 })).not.toThrow();
  });

  it('rejects a non-positive blockY', () => {
    expect(() => buildBlockedConvShader({ ...base, blockY: 0 })).toThrow(/blockY/);
    expect(() => buildBlockedConvShader({ ...base, blockY: 1.5 })).toThrow(/blockY/);
  });

  it('rejects a residual between mismatched channel counts', () => {
    expect(() =>
      buildBlockedConvShader({ ...base, inChannels: 8, outChannels: 16, residual: true, outBlock: 8 }),
    ).toThrow(/residual/);
  });

  it('rejects matrix tiles that do not divide by 8 rather than padding them', () => {
    const m = { inChannels: 16, outChannels: 16, rowsPerGroup: 1, activation: 'relu' as const, useF16: true };
    expect(() => buildMatrixConvShader({ ...m, inChannels: 12 })).toThrow(/multiples of 8/);
    expect(() => buildMatrixConvShader({ ...m, outChannels: 4 })).toThrow(/multiples of 8/);
    expect(() => buildMatrixConvShader(m)).not.toThrow();
  });
});

describe('matrix shader structure', () => {
  it('derives the K-slice count from the input channels', () => {
    const src = buildMatrixConvShader({
      inChannels: 16,
      outChannels: 16,
      rowsPerGroup: 1,
      activation: 'relu',
      useF16: true,
    });
    // 9 taps x 16 channels / 8 = 18.
    expect(src).toContain('const SLICES: u32 = 18u;');
  });

  it('requests a workgroup that is a whole number of subgroups', () => {
    const src = buildMatrixConvShader({
      inChannels: 8,
      outChannels: 8,
      rowsPerGroup: 4,
      activation: 'none',
      useF16: false,
    });
    expect(src).toContain('@workgroup_size(32)');
    expect(src).toContain('enable chromium_experimental_subgroup_matrix;');
  });
});
