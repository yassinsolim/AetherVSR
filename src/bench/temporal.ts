/**
 * Pure temporal metrics for the offline upscaler evaluation.
 *
 * All differences use BT.601 luma in 8-bit output-code-value units (LSB),
 * and intentionally ignore alpha because the benchmark render target always
 * writes alpha = 255. The metrics describe output stability; they are not a
 * substitute for still-frame quality. In particular, a filter that retains
 * more high-frequency detail legitimately has a larger raw frame-to-frame
 * difference while an image moves. Read these values as shimmer per unit of
 * retained detail beside Milestone 2's still-frame scores: Catmull-Rom
 * (19.45 dB PSNR / 0.925 SSIM) and bilinear (17.51 dB / 0.862).
 */

/** A whole-output-pixel translation of content from one frame to the next. */
export interface IntegerShift {
  readonly x: number;
  readonly y: number;
}

/** A rectangular pixel region in an output frame. */
export interface TemporalRegion {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** Consecutive-frame difference, measured on luma code values. */
export interface FrameDifferenceStats {
  /** Mean absolute luma difference, in 8-bit output-code-value LSB. */
  readonly meanAbsoluteLuma: number;
  /** Root mean square luma difference, in 8-bit output-code-value LSB. */
  readonly rmsLuma: number;
  /** Number of luma pixels included across all compared frame pairs. */
  readonly pixels: number;
}

/** Per-pixel temporal variance, aggregated across an aligned output region. */
export interface TemporalVarianceStats {
  /** Mean population variance of per-pixel luma, in output-code-value LSB². */
  readonly meanLumaVariance: number;
  /** Square root of the mean variance, in output-code-value LSB. */
  readonly rmsLumaDeviation: number;
  /** Region whose pixels contributed to the aggregate. */
  readonly region: TemporalRegion;
  /** Number of frames contributing to each pixel's variance. */
  readonly frames: number;
}

const CHANNELS_PER_PIXEL = 4;
const DEFAULT_SHIFT: IntegerShift = { x: 0, y: 0 };

/** ITU-R BT.601 luma, matching the project's still-frame quality convention. */
function luma(frame: Uint8Array, pixel: number): number {
  const offset = pixel * CHANNELS_PER_PIXEL;
  return (
    0.299 * (frame[offset] as number) +
    0.587 * (frame[offset + 1] as number) +
    0.114 * (frame[offset + 2] as number)
  );
}

function assertDimensions(width: number, height: number): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`frame dimensions must be positive integers, got ${width}x${height}`);
  }
}

function assertFrame(frame: Uint8Array, width: number, height: number): void {
  const expectedLength = width * height * CHANNELS_PER_PIXEL;
  if (frame.length !== expectedLength) {
    throw new RangeError(`expected ${expectedLength} RGBA bytes, got ${frame.length}`);
  }
}

function assertIntegerShift(shift: IntegerShift): void {
  if (!Number.isInteger(shift.x) || !Number.isInteger(shift.y)) {
    throw new RangeError(`motion compensation requires an integer output-pixel shift, got ${shift.x},${shift.y}`);
  }
}

function assertSequence(frames: readonly Uint8Array[], width: number, height: number): void {
  if (frames.length < 2) {
    throw new RangeError('temporal metrics require at least two frames');
  }
  assertDimensions(width, height);
  for (const frame of frames) assertFrame(frame, width, height);
}

function frameDifferenceFromSums(absoluteSum: number, squaredSum: number, pixels: number): FrameDifferenceStats {
  if (pixels === 0) throw new RangeError('frame comparison has no overlapping pixels');
  return {
    meanAbsoluteLuma: absoluteSum / pixels,
    rmsLuma: Math.sqrt(squaredSum / pixels),
    pixels,
  };
}

/**
 * Streaming accumulator for consecutive-frame differences.
 *
 * This is useful to the GPU harness because it can score a frame as soon as
 * its benchmark-only readback arrives instead of retaining a whole sequence.
 */
export class FrameDifferenceAccumulator {
  private absoluteSum = 0;
  private squaredSum = 0;
  private pixels = 0;

  /** Adds an uncompensated comparison over every output pixel. */
  add(previous: Uint8Array, current: Uint8Array, width: number, height: number): void {
    assertDimensions(width, height);
    assertFrame(previous, width, height);
    assertFrame(current, width, height);

    const pixelCount = width * height;
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      const difference = luma(current, pixel) - luma(previous, pixel);
      this.absoluteSum += Math.abs(difference);
      this.squaredSum += difference * difference;
    }
    this.pixels += pixelCount;
  }

  /**
   * Adds a comparison after shifting the previous frame by `shift`.
   *
   * A positive x means that content moved right from `previous` to `current`:
   * current(x, y) is compared with previous(x - shift.x, y - shift.y). The
   * non-overlapping edge is excluded rather than padded, so an entering edge
   * cannot masquerade as temporal instability.
   */
  addMotionCompensated(
    previous: Uint8Array,
    current: Uint8Array,
    width: number,
    height: number,
    shift: IntegerShift,
  ): void {
    assertDimensions(width, height);
    assertFrame(previous, width, height);
    assertFrame(current, width, height);
    assertIntegerShift(shift);

    const startX = Math.max(0, shift.x);
    const endX = Math.min(width, width + shift.x);
    const startY = Math.max(0, shift.y);
    const endY = Math.min(height, height + shift.y);
    if (startX >= endX || startY >= endY) {
      throw new RangeError(`shift ${shift.x},${shift.y} leaves no overlapping output pixels`);
    }

    for (let y = startY; y < endY; y++) {
      for (let x = startX; x < endX; x++) {
        const currentPixel = y * width + x;
        const previousPixel = (y - shift.y) * width + x - shift.x;
        const difference = luma(current, currentPixel) - luma(previous, previousPixel);
        this.absoluteSum += Math.abs(difference);
        this.squaredSum += difference * difference;
      }
    }
    this.pixels += (endX - startX) * (endY - startY);
  }

  result(): FrameDifferenceStats {
    return frameDifferenceFromSums(this.absoluteSum, this.squaredSum, this.pixels);
  }
}

/**
 * Difference between one pair of upscaled output frames.
 *
 * This includes real image movement as well as any shimmer, so it cannot by
 * itself identify a temporal defect.
 */
export function frameToFrameDifference(
  previous: Uint8Array,
  current: Uint8Array,
  width: number,
  height: number,
): FrameDifferenceStats {
  const accumulator = new FrameDifferenceAccumulator();
  accumulator.add(previous, current, width, height);
  return accumulator.result();
}

/** Aggregates the ordinary difference over every consecutive pair in a sequence. */
export function sequenceFrameToFrameDifference(
  frames: readonly Uint8Array[],
  width: number,
  height: number,
): FrameDifferenceStats {
  assertSequence(frames, width, height);
  const accumulator = new FrameDifferenceAccumulator();
  for (let index = 1; index < frames.length; index++) {
    accumulator.add(frames[index - 1] as Uint8Array, frames[index] as Uint8Array, width, height);
  }
  return accumulator.result();
}

/**
 * Residual for an intentionally static sequence.
 *
 * It is the ordinary consecutive-frame difference because there is no motion
 * to subtract. A non-zero result for byte-identical input is a harness or GPU
 * determinism defect, not a property to explain away as acceptable flicker.
 */
export function staticSceneResidual(
  frames: readonly Uint8Array[],
  width: number,
  height: number,
): FrameDifferenceStats {
  return sequenceFrameToFrameDifference(frames, width, height);
}

/**
 * Residual after exact integer-pixel motion compensation across a sequence.
 *
 * The shift is the known content displacement from each frame to the next in
 * output pixels. It separates a uniform translation from output changes, but
 * cannot compensate non-integer shifts, spatially varying camera motion, or
 * aliasing already present in the low-resolution input.
 */
export function motionCompensatedResidual(
  frames: readonly Uint8Array[],
  width: number,
  height: number,
  shiftPerFrame: IntegerShift,
): FrameDifferenceStats {
  assertSequence(frames, width, height);
  assertIntegerShift(shiftPerFrame);
  const accumulator = new FrameDifferenceAccumulator();
  for (let index = 1; index < frames.length; index++) {
    accumulator.addMotionCompensated(
      frames[index - 1] as Uint8Array,
      frames[index] as Uint8Array,
      width,
      height,
      shiftPerFrame,
    );
  }
  return accumulator.result();
}

/**
 * Online population-variance accumulator for a fixed output region.
 *
 * `add` can take a source offset to align a moving frame with a common output
 * coordinate system. The caller must choose a region that stays in bounds for
 * every offset. Storage is Float64: a constant sequence must accumulate to
 * exactly zero variance (the harness's own determinism control), and Float32
 * mean/m2 storage rounds every intermediate update, which reintroduces a
 * spurious ~1e-6 relative residual for that exact case. Accumulation uses
 * Welford's numerically stable update rather than subtracting two large sums.
 */
export class TemporalVarianceAccumulator {
  private readonly mean: Float64Array;
  private readonly m2: Float64Array;
  private frameCount = 0;

  constructor(
    private readonly width: number,
    private readonly height: number,
    readonly region: TemporalRegion = { x: 0, y: 0, width, height },
  ) {
    assertDimensions(width, height);
    if (
      !Number.isInteger(region.x) ||
      !Number.isInteger(region.y) ||
      !Number.isInteger(region.width) ||
      !Number.isInteger(region.height) ||
      region.x < 0 ||
      region.y < 0 ||
      region.width <= 0 ||
      region.height <= 0 ||
      region.x + region.width > width ||
      region.y + region.height > height
    ) {
      throw new RangeError(`invalid temporal region ${region.x},${region.y} ${region.width}x${region.height}`);
    }
    const pixels = region.width * region.height;
    this.mean = new Float64Array(pixels);
    this.m2 = new Float64Array(pixels);
  }

  /** Adds one frame, optionally sampled at an integer offset into that frame. */
  add(frame: Uint8Array, offset: IntegerShift = DEFAULT_SHIFT): void {
    assertFrame(frame, this.width, this.height);
    assertIntegerShift(offset);
    if (
      this.region.x + offset.x < 0 ||
      this.region.y + offset.y < 0 ||
      this.region.x + this.region.width + offset.x > this.width ||
      this.region.y + this.region.height + offset.y > this.height
    ) {
      throw new RangeError(`offset ${offset.x},${offset.y} places the temporal region outside its frame`);
    }

    const nextFrameCount = this.frameCount + 1;
    let accumulatorPixel = 0;
    for (let y = 0; y < this.region.height; y++) {
      const sourceY = this.region.y + y + offset.y;
      for (let x = 0; x < this.region.width; x++) {
        const sourcePixel = sourceY * this.width + this.region.x + x + offset.x;
        const sample = luma(frame, sourcePixel);
        const oldMean = this.mean[accumulatorPixel] as number;
        const delta = sample - oldMean;
        const nextMean = oldMean + delta / nextFrameCount;
        this.mean[accumulatorPixel] = nextMean;
        this.m2[accumulatorPixel] =
          (this.m2[accumulatorPixel] as number) + delta * (sample - nextMean);
        accumulatorPixel++;
      }
    }
    this.frameCount = nextFrameCount;
  }

  result(): TemporalVarianceStats {
    if (this.frameCount < 2) {
      throw new RangeError('temporal variance requires at least two frames');
    }

    let varianceSum = 0;
    for (let pixel = 0; pixel < this.m2.length; pixel++) {
      varianceSum += Math.max(0, (this.m2[pixel] as number) / this.frameCount);
    }
    const meanLumaVariance = varianceSum / this.m2.length;
    return {
      meanLumaVariance,
      rmsLumaDeviation: Math.sqrt(meanLumaVariance),
      region: this.region,
      frames: this.frameCount,
    };
  }
}

/** Population temporal variance without motion compensation. */
export function temporalVariance(
  frames: readonly Uint8Array[],
  width: number,
  height: number,
): TemporalVarianceStats {
  assertSequence(frames, width, height);
  const accumulator = new TemporalVarianceAccumulator(width, height);
  for (const frame of frames) accumulator.add(frame);
  return accumulator.result();
}

/**
 * Intersection of output coordinates that remain valid while every frame is
 * aligned to frame zero with the supplied constant integer translation.
 */
export function motionCompensatedRegion(
  width: number,
  height: number,
  frameCount: number,
  shiftPerFrame: IntegerShift,
): TemporalRegion {
  assertDimensions(width, height);
  if (!Number.isInteger(frameCount) || frameCount < 2) {
    throw new RangeError(`motion compensation requires at least two frames, got ${frameCount}`);
  }
  assertIntegerShift(shiftPerFrame);

  const totalX = (frameCount - 1) * shiftPerFrame.x;
  const totalY = (frameCount - 1) * shiftPerFrame.y;
  const x = Math.max(0, -totalX);
  const y = Math.max(0, -totalY);
  const right = Math.min(width, width - totalX);
  const bottom = Math.min(height, height - totalY);
  if (x >= right || y >= bottom) {
    throw new RangeError(`shift ${shiftPerFrame.x},${shiftPerFrame.y} leaves no common temporal region`);
  }
  return { x, y, width: right - x, height: bottom - y };
}

/**
 * Population variance after aligning every frame to frame zero.
 *
 * This is meaningful for a translating sequence with an integer output shift.
 * Do not apply it to non-integer or spatially varying motion: interpolation
 * would itself add a temporal filter and contaminate the measurement.
 */
export function motionCompensatedTemporalVariance(
  frames: readonly Uint8Array[],
  width: number,
  height: number,
  shiftPerFrame: IntegerShift,
): TemporalVarianceStats {
  assertSequence(frames, width, height);
  const region = motionCompensatedRegion(width, height, frames.length, shiftPerFrame);
  const accumulator = new TemporalVarianceAccumulator(width, height, region);
  for (let index = 0; index < frames.length; index++) {
    accumulator.add(frames[index] as Uint8Array, {
      x: index * shiftPerFrame.x,
      y: index * shiftPerFrame.y,
    });
  }
  return accumulator.result();
}
