import type { Size } from '../core/types.js';
import { BaselineScaler } from '../core/upscale/baseline-scaler.js';
import type { Upscaler } from '../core/types.js';
import type { BaselineFilter } from '../core/upscale/baseline.wgsl.js';
import {
  DEFAULT_TEMPORAL_SEQUENCE_CONFIG,
  generateTemporalSequence,
  type TemporalSequenceConfig,
  type TemporalSequenceKind,
} from '../../tools/temporal-sequences.js';
import {
  motionCompensatedResidual,
  motionCompensatedTemporalVariance,
  sequenceFrameToFrameDifference,
  staticSceneResidual,
  temporalVariance,
  type FrameDifferenceStats,
  type IntegerShift,
  type TemporalVarianceStats,
} from './temporal.js';

/**
 * Part P — the temporal baseline.
 *
 * Drives the real `BaselineScaler` (both `bilinear` and `catmull-rom`, at an
 * exact 2x scale) over the three deterministic sequences from
 * `tools/temporal-sequences.ts`, and scores the *upscaled output* with the
 * pure metrics in `temporal.ts`. This is a measurement harness for the two
 * existing non-neural upscalers, not an implementation of temporal
 * super-resolution: no motion compensation network, no frame interpolation
 * and no temporal filtering is added to the render path itself.
 *
 * Every frame's output is read back to the CPU to compute metrics. That is a
 * benchmark-only readback, forbidden in the per-frame production path by
 * AGENTS.md §4, and legitimate here: this function runs offline, once per
 * synthetic frame, never inside a `requestVideoFrameCallback`.
 */

export interface MotionCompensatedTemporalStats {
  readonly residual: FrameDifferenceStats;
  readonly variance: TemporalVarianceStats;
  /** Number of *input* frames spanned by one compensation step. */
  readonly strideFrames: number;
  /** Shift applied per compensation step, in output pixels; see temporal.ts for the sign convention. */
  readonly shiftOutputPixels: IntegerShift;
}

export interface TemporalSequenceResult {
  readonly sequence: TemporalSequenceKind;
  readonly filter: string;
  readonly frameCount: number;
  readonly outputSize: Size;
  /** Raw consecutive-frame difference on the upscaled output; includes real motion. */
  readonly frameToFrame: FrameDifferenceStats;
  /** Raw per-pixel temporal variance on the upscaled output; includes real motion. */
  readonly rawTemporalVariance: TemporalVarianceStats;
  /** Present only when an integer output-pixel shift could align the sequence. */
  readonly motionCompensated: MotionCompensatedTemporalStats | null;
  readonly notes: readonly string[];
}

export interface TemporalBenchConfig {
  readonly filters: readonly BaselineFilter[];
  /** Extra stages to evaluate alongside the baselines, e.g. the neural model. */
  readonly extra?: readonly { readonly label: string; readonly upscaler: Upscaler }[];
  readonly sequences: readonly TemporalSequenceKind[];
  readonly sequenceConfig: TemporalSequenceConfig;
  readonly outputSize: Size;
}

export const DEFAULT_TEMPORAL_BENCH_CONFIG: TemporalBenchConfig = {
  filters: ['bilinear', 'catmull-rom'],
  sequences: ['static', 'translating', 'camera-motion'],
  sequenceConfig: DEFAULT_TEMPORAL_SEQUENCE_CONFIG,
  outputSize: { width: 2560, height: 1440 },
};

const SHIMMER_VS_DETAIL_NOTE =
  'A higher-frequency-preserving filter legitimately shows a larger frame-to-frame difference while the scene moves, simply because it reconstructs more detail for the difference to act on. Read these numbers as shimmer per unit of retained detail next to the Milestone 2 still-frame scores (Catmull-Rom 19.45 dB PSNR / 0.925 SSIM, bilinear 17.51 dB / 0.862 SSIM), not instead of them.';

/**
 * Smallest frame stride (1..frameCount-1) at which `shiftPerInputFrame`
 * accumulates to a nonzero integer number of output pixels, or `null` if no
 * such stride exists within the sequence length.
 */
function findIntegerStride(shiftPerInputFrame: number, frameCount: number): { stride: number; totalShift: number } | null {
  for (let stride = 1; stride < frameCount; stride++) {
    const total = shiftPerInputFrame * stride;
    const rounded = Math.round(total);
    if (rounded !== 0 && Math.abs(total - rounded) < 1e-6) return { stride, totalShift: rounded };
  }
  return null;
}

function buildSequenceResult(
  filter: string,
  sequence: TemporalSequenceKind,
  outputs: readonly Uint8Array[],
  sequenceConfig: TemporalSequenceConfig,
  outputSize: Size,
): TemporalSequenceResult {
  const { width, height } = outputSize;
  const scaleFactor = outputSize.width / sequenceConfig.width;
  const notes: string[] = [];
  let motionCompensated: MotionCompensatedTemporalStats | null = null;
  let frameToFrame: FrameDifferenceStats;

  if (sequence === 'static') {
    frameToFrame = staticSceneResidual(outputs, width, height);
    notes.push(
      'input is byte-identical across the whole sequence, so this residual is the harness/GPU-determinism control: it should be exactly zero, and any nonzero value here is a defect in the render path, not baseline flicker to be explained away.',
    );
  } else {
    frameToFrame = sequenceFrameToFrameDifference(outputs, width, height);
    notes.push(SHIMMER_VS_DETAIL_NOTE);
  }

  const rawTemporalVariance = temporalVariance(outputs, width, height);

  if (sequence === 'translating') {
    // Panning the sampled window right by `panPxPerFrame` LR pixels per frame
    // makes the *content* appear to move left in output space, so the true
    // output-pixel displacement per input frame is negative; see the
    // shift-sign convention documented on `FrameDifferenceAccumulator` in
    // temporal.ts.
    const perFrameOutputShift = -sequenceConfig.panPxPerFrame * scaleFactor;
    const stride = findIntegerStride(perFrameOutputShift, outputs.length);
    if (stride) {
      const strided = outputs.filter((_, index) => index % stride.stride === 0);
      const shift: IntegerShift = { x: stride.totalShift, y: 0 };
      motionCompensated = {
        residual: motionCompensatedResidual(strided, width, height, shift),
        variance: motionCompensatedTemporalVariance(strided, width, height, shift),
        strideFrames: stride.stride,
        shiftOutputPixels: shift,
      };
      notes.push(
        `the ground-truth per-input-frame output-pixel displacement is ${perFrameOutputShift} px ` +
          `(source pan ${sequenceConfig.panPxPerFrame} px/frame x ${scaleFactor}x scale), not an integer, ` +
          `so it cannot be cancelled between consecutive frames. Instead this harness compares output frames ` +
          `${stride.stride} apart, whose accumulated displacement is exactly ${stride.totalShift} integer ` +
          'output pixel(s), and reports the motion-compensated figures over that strided subsequence.',
      );
    } else {
      notes.push(
        `the ground-truth per-input-frame output-pixel displacement is ${perFrameOutputShift} px, not an ` +
          `integer, and no integer-pixel stride was found within ${outputs.length} frames, so motion ` +
          'compensation could not be measured for this configuration.',
      );
    }
  }

  if (sequence === 'camera-motion') {
    notes.push(
      'motion compensation is not attempted: the transform combines a spatially varying zoom with a sub-pixel ' +
        'pan, so no single integer-pixel shift aligns consecutive frames. The frame-to-frame difference and ' +
        'temporal variance above are raw and mix real scene motion with any shimmer; this harness cannot ' +
        'separate them further for this sequence.',
    );
  }

  return {
    sequence,
    filter,
    frameCount: outputs.length,
    outputSize,
    frameToFrame,
    rawTemporalVariance,
    motionCompensated,
    notes,
  };
}

export async function runTemporalBench(
  device: GPUDevice,
  config: TemporalBenchConfig = DEFAULT_TEMPORAL_BENCH_CONFIG,
): Promise<TemporalSequenceResult[]> {
  const { filters, sequences, sequenceConfig, outputSize } = config;
  if (outputSize.width !== sequenceConfig.width * 2 || outputSize.height !== sequenceConfig.height * 2) {
    throw new Error(
      `temporal bench requires an exact 2x scale: got ${sequenceConfig.width}x${sequenceConfig.height} -> ` +
        `${outputSize.width}x${outputSize.height}`,
    );
  }

  const format: GPUTextureFormat = 'rgba8unorm';
  const lrTexture = device.createTexture({
    label: 'temporal:lr',
    size: { width: sequenceConfig.width, height: sequenceConfig.height },
    format,
    usage:
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const lrView = lrTexture.createView();

  const target = device.createTexture({
    label: 'temporal:sr',
    size: outputSize,
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const targetView = target.createView();

  const unpaddedBytesPerRow = outputSize.width * 4;
  const bytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;
  const readback = device.createBuffer({
    label: 'temporal:readback',
    size: bytesPerRow * outputSize.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  async function renderAndReadback(scaler: Upscaler, frame: ImageData): Promise<Uint8Array> {
    const bitmap = await createImageBitmap(frame);
    device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: lrTexture },
      { width: sequenceConfig.width, height: sequenceConfig.height },
    );
    bitmap.close();

    const encoder = device.createCommandEncoder();
    scaler.encode({ encoder, frame: { kind: 'sampled', view: lrView }, target: targetView, timing: null });
    encoder.copyTextureToBuffer(
      { texture: target },
      { buffer: readback, bytesPerRow, rowsPerImage: outputSize.height },
      { width: outputSize.width, height: outputSize.height },
    );
    device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const tight = new Uint8Array(unpaddedBytesPerRow * outputSize.height);
    for (let y = 0; y < outputSize.height; y++) {
      tight.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + unpaddedBytesPerRow), y * unpaddedBytesPerRow);
    }
    return tight;
  }

  const results: TemporalSequenceResult[] = [];

  const stages: { label: string; upscaler: Upscaler; owned: boolean }[] = [
    ...filters.map((f) => ({ label: f, upscaler: new BaselineScaler(f), owned: true })),
    ...(config.extra ?? []).map((e) => ({ label: e.label, upscaler: e.upscaler, owned: false })),
  ];

  for (const stage of stages) {
    const filter = stage.label;
    const scaler = stage.upscaler;
    scaler.configure({
      device,
      source: { width: sequenceConfig.width, height: sequenceConfig.height },
      target: outputSize,
      targetFormat: format,
      sourceKind: 'sampled',
    });

    for (const sequence of sequences) {
      const inputFrames = generateTemporalSequence(sequence, sequenceConfig);
      const outputs: Uint8Array[] = [];
      for (const frame of inputFrames) outputs.push(await renderAndReadback(scaler, frame));
      results.push(buildSequenceResult(filter, sequence, outputs, sequenceConfig, outputSize));
    }

    if (stage.owned) scaler.destroy();
  }

  lrTexture.destroy();
  target.destroy();
  readback.destroy();
  return results;
}
