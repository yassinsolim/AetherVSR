import type { Activation } from './conv.wgsl.js';

export interface TiledConvShaderConfig {
  readonly inChannels: number;
  readonly outChannels: number;
  /** Workgroup dimensions in *invocations*, not output pixels. */
  readonly tileX: number;
  readonly tileY: number;
  /** Output pixels each invocation computes along x. */
  readonly blockX: number;
  readonly activation: Activation;
  readonly useF16: boolean;
  readonly residual: boolean;
}

/** Bytes of workgroup storage the generated shader will declare. */
export function tiledSharedBytes(config: TiledConvShaderConfig): number {
  const tileW = config.tileX * config.blockX + 2;
  const tileH = config.tileY + 2;
  return tileW * tileH * (config.useF16 ? 2 : 4);
}

/**
 * 3x3 convolution with cooperative workgroup tiling.
 *
 * ## What changes versus the naive kernel
 *
 * The naive kernel reads its 3x3 window straight from global storage for every
 * output pixel, every input channel: `9 * inChannels` global loads per output.
 * Neighbouring outputs overlap by two columns and two rows, so almost all of
 * that traffic is redundant and only the cache absorbs it.
 *
 * Here each workgroup cooperatively stages one input channel's tile — plus a
 * one-pixel halo on every side — into workgroup memory, then every invocation
 * reads its window from there. Global loads per output fall from
 * `9 * inChannels` to
 *
 * ```
 *   (tileX * blockX + 2) * (tileY + 2)
 *   ---------------------------------- * inChannels
 *        (tileX * blockX) * tileY
 * ```
 *
 * which for a 64x8 output tile is ≈1.29 per channel against 9 — a ~7x
 * reduction in *issued* global loads. Whether that converts into time depends
 * on how much of the redundancy the cache was already absorbing, which is
 * precisely what the measurement is for.
 *
 * ## Correctness structure
 *
 * Two barriers per input channel. The first is easy to forget and is the one
 * that matters: without a barrier *before* the store loop, a fast invocation
 * can overwrite tile memory that a slow one is still reading from the previous
 * channel's accumulation. The second orders the stores against the reads.
 *
 * Padding semantics are unchanged — out-of-bounds taps read zero — and are
 * applied while filling the tile so the accumulation loop stays branch-free.
 */
export function buildTiledConvShader(config: TiledConvShaderConfig): string {
  const { inChannels, outChannels, tileX, tileY, blockX, activation, useF16, residual } = config;

  if (residual && inChannels !== outChannels) {
    throw new Error(`residual requires inChannels === outChannels, got ${inChannels} -> ${outChannels}`);
  }

  const T = useF16 ? 'f16' : 'f32';
  const enable = useF16 ? 'enable f16;\n' : '';
  const tileW = tileX * blockX + 2;
  const tileH = tileY + 2;
  const threads = tileX * tileY;

  const activate = (expr: string): string => {
    switch (activation) {
      case 'relu':
        return `max(${expr}, ${T}(0.0))`;
      case 'tanh':
        return `tanh(${expr})`;
      case 'none':
        return expr;
    }
  };

  const accDecl = Array.from({ length: blockX }, (_, i) => `  var acc${i}: ${T} = bias;`).join('\n');

  // Each blocked output reads a 3x3 window whose left edge is its own column.
  const macBody = Array.from(
    { length: blockX },
    (_, i) => `        acc${i} += w * tile[rowBase + localX + ${i}u + u32(kx)];`,
  ).join('\n');

  const stores = Array.from({ length: blockX }, (_, i) => {
    const value = residual
      ? `${activate(`acc${i}`)} + input[outPlane + outY * W + outX + ${i}u]`
      : activate(`acc${i}`);
    return `  if (outX + ${i}u < W) { output[outPlane + outY * W + outX + ${i}u] = ${value}; }`;
  }).join('\n');

  return /* wgsl */ `${enable}
const IN_C: u32 = ${inChannels}u;
const OUT_C: u32 = ${outChannels}u;
const BLOCK_X: u32 = ${blockX}u;
const TILE_W: u32 = ${tileW}u;   // staged columns, including the 1px halo
const TILE_H: u32 = ${tileH}u;   // staged rows, including the 1px halo
const TILE_N: u32 = ${tileW * tileH}u;
const THREADS: u32 = ${threads}u;

struct Dims {
  width: u32,
  height: u32,
};

@group(0) @binding(0) var<storage, read> input: array<${T}>;
@group(0) @binding(1) var<storage, read> weights: array<${T}>;
@group(0) @binding(2) var<storage, read> biases: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> tile: array<${T}, ${tileW * tileH}>;

@compute @workgroup_size(${tileX}, ${tileY}, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) lindex: u32,
  @builtin(workgroup_id) wid: vec3u,
) {
  let W = dims.width;
  let H = dims.height;

  // Origin of this workgroup's *output* region, in pixels.
  let originX = i32(wid.x * ${tileX}u * BLOCK_X);
  let originY = i32(wid.y * ${tileY}u);
  let outC = wid.z;

  let localX = lid.x * BLOCK_X;
  let outX = u32(originX) + localX;
  let outY = u32(originY) + lid.y;

  let bias = biases[outC];
${accDecl}

  let outPlane = outC * W * H;
  let wOutBase = outC * IN_C * 9u;

  for (var ic: u32 = 0u; ic < IN_C; ic = ic + 1u) {
    let inBase = ic * W * H;

    // Ordered against the previous channel's reads, not just its writes: a
    // fast invocation must not refill the tile while a slow one is still
    // accumulating from it.
    workgroupBarrier();
    for (var i: u32 = lindex; i < TILE_N; i = i + THREADS) {
      let tx = i % TILE_W;
      let ty = i / TILE_W;
      let sx = originX + i32(tx) - 1;
      let sy = originY + i32(ty) - 1;
      var v: ${T} = ${T}(0.0);
      // Zero padding, applied once here rather than per tap.
      if (sx >= 0 && sy >= 0 && sx < i32(W) && sy < i32(H)) {
        v = input[inBase + u32(sy) * W + u32(sx)];
      }
      tile[i] = v;
    }
    workgroupBarrier();

    let wBase = wOutBase + ic * 9u;
    for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
      let rowBase = (lid.y + u32(ky)) * TILE_W;
      for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
        let w = weights[wBase + u32(ky * 3 + kx)];
${macBody}
      }
    }
  }

  if (outY >= H || outC >= OUT_C) {
    return;
  }
${stores}
}
`;
}
