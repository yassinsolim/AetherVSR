import { describe, expect, it } from 'vitest';
import {
  packedActivationIndex,
  packedWeightIndex,
  tapMajorWeightIndex,
} from '../src/bench/conv-bench.js';
import { packActivations, packWeights, toTapMajorWeights } from '../src/bench/conv-packed.wgsl.js';
import { matrixWeightIndex, toMatrixWeights } from '../src/bench/conv-matrix.wgsl.js';

/**
 * Every layout used by the convolution kernels is a permutation of the planar
 * one. Two properties follow, and both catch real bugs:
 *
 * - It must be a bijection. Index arithmetic that collides leaves some
 *   destination slots unwritten, and an unwritten slot is a silent zero — the
 *   kernel still runs and still produces plausible output.
 * - The benchmark and the verifier must agree. The benchmark relocates
 *   elements through an index function; the verifier rebuilds whole arrays.
 *   If those two drift, the harness measures a different computation from the
 *   one that was checked, which is the worst failure this project can have
 *   because nothing looks wrong.
 */

/** Asserts `at` maps 0..n-1 onto 0..n-1 with no collisions and no gaps. */
function expectPermutation(at: (i: number) => number, n: number): void {
  const seen = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const j = at(i);
    expect(Number.isInteger(j), `index ${i} mapped to non-integer ${j}`).toBe(true);
    expect(j, `index ${i} mapped out of range`).toBeGreaterThanOrEqual(0);
    expect(j, `index ${i} mapped out of range`).toBeLessThan(n);
    expect(seen[j], `collision at destination ${j}`).toBe(0);
    seen[j] = 1;
  }
}

/** A distinct value per planar index, so a permutation is fully observable. */
function ramp(n: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = i + 1;
  return a;
}

/** Applies an index remap the way the benchmark's buffer fill does. */
function applyRemap(src: Float32Array, at: (i: number) => number): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[at(i)] = src[i] as number;
  return out;
}

describe('packed activation layout', () => {
  it('is a permutation for every supported channel count', () => {
    for (const [w, h, c] of [
      [4, 3, 4],
      [7, 5, 8],
      [16, 9, 12],
      [3, 3, 16],
    ] as const) {
      expectPermutation(packedActivationIndex(w * h), w * h * c);
    }
  });

  it('places channel c of pixel p at group c/4, lane c%4', () => {
    const pixels = 6;
    const at = packedActivationIndex(pixels);
    // Channel 5, pixel 2 -> group 1, lane 1.
    expect(at(5 * pixels + 2)).toBe((1 * pixels + 2) * 4 + 1);
    // Channel 0, pixel 0 stays at the origin.
    expect(at(0)).toBe(0);
  });

  it('agrees with the verifier repack', () => {
    const [w, h, c] = [5, 4, 8];
    const planar = ramp(w * h * c);
    expect(Array.from(applyRemap(planar, packedActivationIndex(w * h)))).toEqual(
      Array.from(packActivations(planar, w, h, c)),
    );
  });
});

describe('packed weight layout', () => {
  it('is a permutation', () => {
    for (const [inC, outC] of [
      [4, 4],
      [8, 4],
      [12, 8],
      [16, 16],
    ] as const) {
      expectPermutation(packedWeightIndex(inC, outC), inC * outC * 9);
    }
  });

  it('agrees with the verifier repack', () => {
    const [inC, outC] = [8, 4];
    const planar = ramp(inC * outC * 9);
    expect(Array.from(applyRemap(planar, packedWeightIndex(inC, outC)))).toEqual(
      Array.from(packWeights(planar, inC, outC)),
    );
  });
});

describe('tap-major weight layout', () => {
  it('is a permutation', () => {
    for (const [inC, outC] of [
      [4, 8],
      [8, 8],
      [16, 16],
      [12, 4],
    ] as const) {
      expectPermutation(tapMajorWeightIndex(inC, outC), inC * outC * 9);
    }
  });

  it('makes one tap\u2019s weights for consecutive output channels adjacent', () => {
    // This is the entire point of the layout: at a fixed (channel group, tap),
    // stepping the output channel must step one vec4 forward.
    const [inC, outC] = [16, 16];
    const at = tapMajorWeightIndex(inC, outC);
    const planarIndex = (oc: number, ic: number, k: number): number => (oc * inC + ic) * 9 + k;
    for (let oc = 0; oc + 1 < outC; oc++) {
      expect(at(planarIndex(oc + 1, 4, 7)) - at(planarIndex(oc, 4, 7))).toBe(4);
    }
  });

  it('agrees with the verifier repack applied after grouping', () => {
    const [inC, outC] = [8, 8];
    const planar = ramp(inC * outC * 9);
    const viaIndex = applyRemap(planar, tapMajorWeightIndex(inC, outC));
    const viaArrays = toTapMajorWeights(packWeights(planar, inC, outC), inC, outC);
    expect(Array.from(viaIndex)).toEqual(Array.from(viaArrays));
  });
});

describe('subgroup-matrix weight layout', () => {
  it('is a permutation', () => {
    for (const [inC, outC] of [
      [8, 8],
      [16, 16],
      [16, 8],
      [8, 16],
    ] as const) {
      expectPermutation(matrixWeightIndex(inC, outC), inC * outC * 9);
    }
  });

  it('keeps each 8-wide K slice inside a single tap', () => {
    // The gather in the shader assumes this: one slice differs only by input
    // channel, never by tap. If the ordering changed, the shader would read
    // the wrong pixels and still produce numbers.
    const [inC, outC] = [16, 16];
    const at = matrixWeightIndex(inC, outC);
    const sliceOf = (planarIdx: number): number => Math.floor((at(planarIdx) / 64) % ((9 * inC) / 8));
    for (let tap = 0; tap < 9; tap++) {
      const slices = new Set<number>();
      for (let ic = 0; ic < inC; ic++) slices.add(sliceOf((0 * inC + ic) * 9 + tap));
      // 16 channels at 8 per slice is exactly two slices, and no other tap
      // may share them.
      expect(slices.size).toBe(inC / 8);
    }
  });

  it('agrees with the array repack', () => {
    const [inC, outC] = [8, 16];
    const planar = ramp(inC * outC * 9);
    expect(Array.from(applyRemap(planar, matrixWeightIndex(inC, outC)))).toEqual(
      Array.from(toMatrixWeights(planar, inC, outC)),
    );
  });
});
