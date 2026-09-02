/**
 * Deterministic low-resolution input sequences for the Part P temporal
 * baseline benchmark (`src/bench/temporal-bench.ts`).
 *
 * Frames are synthesised procedurally, never decoded from an encoded clip, so
 * there is no decoder variance to control and every run is bit-for-bit
 * reproducible from this module alone.
 *
 * Content reuses `generateReference` from `src/bench/quality.ts` — the same
 * radial zone plate / diagonal-bar / gradient image the still-frame quality
 * bench scores against, which already contains frequencies at and near
 * Nyquist — rendered once onto a "world" canvas somewhat larger than the
 * viewport. Each output frame samples a window of that single world image
 * through an explicit per-frame affine transform (translate for panning,
 * scale-about-centre for zooming) using a hand-written bilinear resampler.
 * `drawImage` is deliberately not used: its resampling is browser-defined,
 * and the whole point of this generator is that motion is an exact,
 * controlled floating-point transform, matching the determinism rationale in
 * `quality.ts`'s `boxDownsample2x`.
 */
import { generateReference } from '../src/bench/quality.js';

/** The three temporal probes Part P measures. */
export type TemporalSequenceKind = 'static' | 'translating' | 'camera-motion';

export interface TemporalSequenceConfig {
  readonly frameCount: number;
  readonly width: number;
  readonly height: number;
  /** Sub-pixel horizontal translation per frame, in input (LR) pixels. */
  readonly panPxPerFrame: number;
  /** Fractional shrink of the sampled world window per frame (zoom-in rate). */
  readonly zoomPerFrame: number;
}

export const DEFAULT_TEMPORAL_SEQUENCE_CONFIG: TemporalSequenceConfig = {
  frameCount: 24,
  width: 1280,
  height: 720,
  panPxPerFrame: 0.25,
  zoomPerFrame: 0.0015,
};

/**
 * Half-extent, in pixels, added on every side of the viewport when building
 * the world canvas. Comfortably covers the default config's motion budget
 * (24 frames * 0.25 px/frame pan = 6 px, plus zoom, which needs *less* margin
 * than pan since it shrinks the sampled window towards the centre). A caller
 * who scales `panPxPerFrame` or `frameCount` up far enough can still exceed
 * this; `sampleBilinear` clamps to the world edge in that case rather than
 * throwing, which matches the real scaler's clamp-to-edge sampler but will
 * read the same edge pixel repeatedly. That would show up as a drop in the
 * measured temporal variance near the border, not a crash, so keep the
 * default margin in mind before enlarging motion parameters.
 */
const WORLD_MARGIN = 48;

function buildWorld(config: TemporalSequenceConfig): ImageData {
  return generateReference(config.width + WORLD_MARGIN * 2, config.height + WORLD_MARGIN * 2);
}

/** Bilinear sample of `world` at fractional coordinates, clamped to its edges. */
function sampleBilinear(world: ImageData, wx: number, wy: number, out: Uint8ClampedArray, outOffset: number): void {
  const maxX = world.width - 1;
  const maxY = world.height - 1;
  const cx = Math.min(Math.max(wx, 0), maxX);
  const cy = Math.min(Math.max(wy, 0), maxY);
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, maxX);
  const y1 = Math.min(y0 + 1, maxY);
  const fx = cx - x0;
  const fy = cy - y0;
  const data = world.data;
  for (let c = 0; c < 4; c++) {
    const p00 = data[(y0 * world.width + x0) * 4 + c] as number;
    const p10 = data[(y0 * world.width + x1) * 4 + c] as number;
    const p01 = data[(y1 * world.width + x0) * 4 + c] as number;
    const p11 = data[(y1 * world.width + x1) * 4 + c] as number;
    const top = p00 + (p10 - p00) * fx;
    const bottom = p01 + (p11 - p01) * fx;
    out[outOffset + c] = top + (bottom - top) * fy;
  }
}

/**
 * Renders one output frame by sampling `world` through the transform for
 * `kind` at `frameIndex`: `static` never moves (frame index is ignored by the
 * caller, which always passes 0), `translating` pans by a constant sub-pixel
 * step, and `camera-motion` shrinks the sampled window about its centre
 * (a zoom-in) while also panning, so the sub-pixel phase of the content
 * varies across the frame instead of uniformly.
 */
function renderFrame(
  world: ImageData,
  config: TemporalSequenceConfig,
  kind: TemporalSequenceKind,
  frameIndex: number,
): ImageData {
  const { width, height } = config;
  const out = new Uint8ClampedArray(width * height * 4);
  const worldCenterX = world.width / 2;
  const worldCenterY = world.height / 2;
  const viewCenterX = width / 2;
  const viewCenterY = height / 2;

  const scale = kind === 'camera-motion' ? 1 - config.zoomPerFrame * frameIndex : 1;
  const panX = kind === 'static' ? 0 : config.panPxPerFrame * frameIndex;

  for (let y = 0; y < height; y++) {
    const wy = worldCenterY + (y - viewCenterY) * scale;
    for (let x = 0; x < width; x++) {
      const wx = worldCenterX + (x - viewCenterX) * scale + panX;
      sampleBilinear(world, wx, wy, out, (y * width + x) * 4);
    }
  }
  return new ImageData(out, width, height);
}

/**
 * Generates a full deterministic input sequence.
 *
 * `static` renders a single frame once and repeats the same `ImageData`
 * object for every entry, so the sequence is not merely pixel-equal but
 * literally the same bytes — the strongest input the static control can
 * offer for isolating scaler-introduced flicker from generator noise.
 */
export function generateTemporalSequence(
  kind: TemporalSequenceKind,
  config: TemporalSequenceConfig = DEFAULT_TEMPORAL_SEQUENCE_CONFIG,
): ImageData[] {
  const world = buildWorld(config);
  if (kind === 'static') {
    const frame = renderFrame(world, config, kind, 0);
    return new Array<ImageData>(config.frameCount).fill(frame);
  }
  const frames: ImageData[] = [];
  for (let frameIndex = 0; frameIndex < config.frameCount; frameIndex++) {
    frames.push(renderFrame(world, config, kind, frameIndex));
  }
  return frames;
}
