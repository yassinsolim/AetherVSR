import { describe, expect, it } from 'vitest';
import {
  FrameDifferenceAccumulator,
  motionCompensatedRegion,
  motionCompensatedResidual,
  motionCompensatedTemporalVariance,
  sequenceFrameToFrameDifference,
  staticSceneResidual,
  temporalVariance,
  type IntegerShift,
} from '../src/bench/temporal.js';

const WIDTH = 4;
const HEIGHT = 4;

/** Builds an RGBA8 frame whose luma at (x, y) is `value(x, y)`, alpha 255. */
function makeFrame(width: number, height: number, value: (x: number, y: number) => number): Uint8Array {
  const frame = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = value(x, y);
      const o = (y * width + x) * 4;
      // Equal R/G/B makes luma exactly `v` regardless of the BT.601 weights.
      frame[o] = v;
      frame[o + 1] = v;
      frame[o + 2] = v;
      frame[o + 3] = 255;
    }
  }
  return frame;
}

/** A frame with a distinct value per pixel, so a shift is actually detectable. */
function rampFrame(width: number, height: number, originX = 0, originY = 0): Uint8Array {
  return makeFrame(width, height, (x, y) => (10 + (x + originX) * 5 + (y + originY) * 3) % 256);
}

describe('sequenceFrameToFrameDifference / staticSceneResidual', () => {
  it('is exactly zero for byte-identical frames', () => {
    const frame = rampFrame(WIDTH, HEIGHT);
    const result = staticSceneResidual([frame, frame, frame], WIDTH, HEIGHT);
    expect(result.meanAbsoluteLuma).toBe(0);
    expect(result.rmsLuma).toBe(0);
    expect(result.pixels).toBe(WIDTH * HEIGHT * 2);
  });

  it('reports a known constant offset exactly', () => {
    const a = makeFrame(WIDTH, HEIGHT, () => 100);
    const b = makeFrame(WIDTH, HEIGHT, () => 107);
    const result = sequenceFrameToFrameDifference([a, b], WIDTH, HEIGHT);
    expect(result.meanAbsoluteLuma).toBeCloseTo(7, 10);
    expect(result.rmsLuma).toBeCloseTo(7, 10);
  });

  it('rejects a sequence shorter than two frames', () => {
    const frame = rampFrame(WIDTH, HEIGHT);
    expect(() => sequenceFrameToFrameDifference([frame], WIDTH, HEIGHT)).toThrow(RangeError);
  });
});

describe('FrameDifferenceAccumulator.addMotionCompensated', () => {
  it('fully cancels a known integer shift with the correct displacement', () => {
    const previous = rampFrame(WIDTH, HEIGHT, 0, 0);
    // `current` is `previous` translated right by one output pixel: the
    // pixel formerly at x now appears at x + 1, so current(x+1, y) === previous(x, y).
    const current = rampFrame(WIDTH, HEIGHT, -1, 0);
    const shift: IntegerShift = { x: 1, y: 0 };

    const accumulator = new FrameDifferenceAccumulator();
    accumulator.addMotionCompensated(previous, current, WIDTH, HEIGHT, shift);
    const result = accumulator.result();

    expect(result.meanAbsoluteLuma).toBe(0);
    expect(result.rmsLuma).toBe(0);
    // The non-overlapping leading column is excluded, not padded.
    expect(result.pixels).toBe((WIDTH - 1) * HEIGHT);
  });

  it('does not cancel the same shifted pair under the wrong displacement', () => {
    const previous = rampFrame(WIDTH, HEIGHT, 0, 0);
    const current = rampFrame(WIDTH, HEIGHT, -1, 0);
    const wrongShift: IntegerShift = { x: 2, y: 0 };

    const accumulator = new FrameDifferenceAccumulator();
    accumulator.addMotionCompensated(previous, current, WIDTH, HEIGHT, wrongShift);
    const result = accumulator.result();

    expect(result.meanAbsoluteLuma).toBeGreaterThan(0);
  });

  it('rejects a non-integer shift', () => {
    const previous = rampFrame(WIDTH, HEIGHT);
    const current = rampFrame(WIDTH, HEIGHT);
    const accumulator = new FrameDifferenceAccumulator();
    expect(() => accumulator.addMotionCompensated(previous, current, WIDTH, HEIGHT, { x: 0.5, y: 0 })).toThrow(
      RangeError,
    );
  });
});

describe('motionCompensatedResidual', () => {
  it('fully cancels a sequence translating by a constant known integer shift', () => {
    const shift: IntegerShift = { x: 1, y: 0 };
    const frames = [
      rampFrame(WIDTH, HEIGHT, 0, 0),
      rampFrame(WIDTH, HEIGHT, -1, 0),
      rampFrame(WIDTH, HEIGHT, -2, 0),
    ];
    const result = motionCompensatedResidual(frames, WIDTH, HEIGHT, shift);
    expect(result.meanAbsoluteLuma).toBe(0);
    expect(result.rmsLuma).toBe(0);
  });

  it('leaves a nonzero residual under the wrong known shift', () => {
    const frames = [rampFrame(WIDTH, HEIGHT, 0, 0), rampFrame(WIDTH, HEIGHT, -1, 0)];
    const result = motionCompensatedResidual(frames, WIDTH, HEIGHT, { x: 0, y: 1 });
    expect(result.meanAbsoluteLuma).toBeGreaterThan(0);
  });
});

describe('temporalVariance', () => {
  it('is exactly zero for a constant sequence', () => {
    const frame = rampFrame(WIDTH, HEIGHT);
    const result = temporalVariance([frame, frame, frame, frame], WIDTH, HEIGHT);
    expect(result.meanLumaVariance).toBe(0);
    expect(result.rmsLumaDeviation).toBe(0);
    expect(result.frames).toBe(4);
  });

  it('matches the hand-computed population variance for a two-value alternation', () => {
    // Every pixel alternates between 100 and 116: population variance = ((8)^2 + (-8)^2) / 2 = 64.
    const low = makeFrame(WIDTH, HEIGHT, () => 100);
    const high = makeFrame(WIDTH, HEIGHT, () => 116);
    const result = temporalVariance([low, high, low, high], WIDTH, HEIGHT);
    expect(result.meanLumaVariance).toBeCloseTo(64, 6);
    expect(result.rmsLumaDeviation).toBeCloseTo(8, 6);
  });

  it('rejects a sequence shorter than two frames', () => {
    const frame = rampFrame(WIDTH, HEIGHT);
    expect(() => temporalVariance([frame], WIDTH, HEIGHT)).toThrow(RangeError);
  });
});

describe('motionCompensatedRegion / motionCompensatedTemporalVariance', () => {
  it('shrinks the region by the accumulated shift across the sequence', () => {
    const region = motionCompensatedRegion(WIDTH, HEIGHT, 3, { x: 1, y: 0 });
    // Total displacement over 3 frames (2 steps) is 2 output pixels.
    expect(region).toEqual({ x: 0, y: 0, width: WIDTH - 2, height: HEIGHT });
  });

  it('is exactly zero for a perfectly translating constant-content sequence', () => {
    const shift: IntegerShift = { x: 1, y: 0 };
    const frames = [
      rampFrame(WIDTH, HEIGHT, 0, 0),
      rampFrame(WIDTH, HEIGHT, -1, 0),
      rampFrame(WIDTH, HEIGHT, -2, 0),
    ];
    const result = motionCompensatedTemporalVariance(frames, WIDTH, HEIGHT, shift);
    expect(result.meanLumaVariance).toBe(0);
  });

  it('rejects a shift that leaves no common region', () => {
    expect(() => motionCompensatedRegion(WIDTH, HEIGHT, 2, { x: WIDTH, y: 0 })).toThrow(RangeError);
  });
});
