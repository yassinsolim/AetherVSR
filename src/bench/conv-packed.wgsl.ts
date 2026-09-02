import type { Activation } from './conv.wgsl.js';
import type { TiledConvShaderConfig } from './conv-tiled.wgsl.js';

export type PackedConvShaderConfig = TiledConvShaderConfig;

/** Bytes of workgroup storage the packed shader declares. */
export function packedSharedBytes(config: PackedConvShaderConfig): number {
  const tileW = config.tileX * config.blockX + 2;
  const tileH = config.tileY + 2;
  return tileW * tileH * (config.useF16 ? 8 : 16);
}

/**
 * Tiled 3x3 convolution with the input channel axis packed into `vec4`.
 *
 * ## Layout
 *
 * Activations move from planar `[c][y][x]` to channel-grouped
 * `[c/4][y][x][c%4]`, addressed as `array<vec4<T>>`. Weights move from
 * `[oc][ic][k]` to `[oc][ic/4][k]`, also `vec4`, so the inner product over four
 * input channels is one `dot`.
 *
 * The output stays planar with one output channel per workgroup z-slice. That
 * is deliberate: packing the output as well means computing four output
 * channels per invocation, which is a different optimisation with its own
 * register-pressure trade-off, and mixing the two would make neither
 * attributable.
 *
 * ## What this is expected to buy
 *
 * Four scalar loads and four scalar multiply-adds become one vector load and
 * one `dot`. On a scalar-ALU GPU the arithmetic is unchanged and only the
 * load/index overhead falls; the measurement decides which regime this is.
 *
 * ## Constraint
 *
 * `inChannels` must be a multiple of 4. Padding to a multiple of 4 would work
 * but would silently inflate the MAC count and make throughput figures
 * flattering, so unsupported widths are rejected instead.
 */
export function buildPackedConvShader(config: PackedConvShaderConfig): string {
  const { inChannels, outChannels, tileX, tileY, blockX, activation, useF16, residual } = config;

  if (inChannels % 4 !== 0) {
    throw new Error(`packed variant requires inChannels % 4 === 0, got ${inChannels}`);
  }
  if (residual && inChannels !== outChannels) {
    throw new Error(`residual requires inChannels === outChannels, got ${inChannels} -> ${outChannels}`);
  }

  const T = useF16 ? 'f16' : 'f32';
  const V = `vec4<${T}>`;
  const enable = useF16 ? 'enable f16;\n' : '';
  const groups = inChannels / 4;
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
  const macBody = Array.from(
    { length: blockX },
    (_, i) => `        acc${i} += dot(w, tile[rowBase + localX + ${i}u + u32(kx)]);`,
  ).join('\n');

  const stores = Array.from({ length: blockX }, (_, i) => {
    // The residual tap has to be unpacked back out of the grouped layout.
    const value = residual
      ? `${activate(`acc${i}`)} + input[(outC / 4u) * W * H + outY * W + outX + ${i}u][outC % 4u]`
      : activate(`acc${i}`);
    return `  if (outX + ${i}u < W) { output[outPlane + outY * W + outX + ${i}u] = ${value}; }`;
  }).join('\n');

  return /* wgsl */ `${enable}
const IN_GROUPS: u32 = ${groups}u;   // inChannels / 4
const OUT_C: u32 = ${outChannels}u;
const BLOCK_X: u32 = ${blockX}u;
const TILE_W: u32 = ${tileW}u;
const TILE_N: u32 = ${tileW * tileH}u;
const THREADS: u32 = ${threads}u;

struct Dims {
  width: u32,
  height: u32,
};

@group(0) @binding(0) var<storage, read> input: array<${V}>;
@group(0) @binding(1) var<storage, read> weights: array<${V}>;
@group(0) @binding(2) var<storage, read> biases: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> tile: array<${V}, ${tileW * tileH}>;

@compute @workgroup_size(${tileX}, ${tileY}, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) lindex: u32,
  @builtin(workgroup_id) wid: vec3u,
) {
  let W = dims.width;
  let H = dims.height;

  let originX = i32(wid.x * ${tileX}u * BLOCK_X);
  let originY = i32(wid.y * ${tileY}u);
  let outC = wid.z;

  let localX = lid.x * BLOCK_X;
  let outX = u32(originX) + localX;
  let outY = u32(originY) + lid.y;

  let bias = biases[outC];
${accDecl}

  let outPlane = outC * W * H;
  let wOutBase = outC * IN_GROUPS * 9u;

  for (var cg: u32 = 0u; cg < IN_GROUPS; cg = cg + 1u) {
    let inBase = cg * W * H;

    workgroupBarrier();
    for (var i: u32 = lindex; i < TILE_N; i = i + THREADS) {
      let sx = originX + i32(i % TILE_W) - 1;
      let sy = originY + i32(i / TILE_W) - 1;
      var v: ${V} = ${V}(${T}(0.0));
      if (sx >= 0 && sy >= 0 && sx < i32(W) && sy < i32(H)) {
        v = input[inBase + u32(sy) * W + u32(sx)];
      }
      tile[i] = v;
    }
    workgroupBarrier();

    let wBase = wOutBase + cg * 9u;
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

/**
 * Repacks planar `[c][y][x]` activations into grouped `[c/4][y][x][c%4]`.
 *
 * Used by the verifier and by any harness feeding the packed kernel. Kept
 * here, next to the shader that defines the layout, so the two cannot drift.
 */
export function packActivations(
  planar: Float32Array,
  width: number,
  height: number,
  channels: number,
): Float32Array<ArrayBuffer> {
  if (channels % 4 !== 0) throw new Error(`packActivations requires channels % 4 === 0, got ${channels}`);
  const pixels = width * height;
  const out = new Float32Array(new ArrayBuffer(planar.length * 4));
  for (let c = 0; c < channels; c++) {
    const group = Math.floor(c / 4);
    const lane = c % 4;
    for (let p = 0; p < pixels; p++) {
      out[(group * pixels + p) * 4 + lane] = planar[c * pixels + p] as number;
    }
  }
  return out;
}

/** Repacks planar `[oc][ic][k]` weights into grouped `[oc][ic/4][k][ic%4]`. */
export function packWeights(
  planar: Float32Array,
  inChannels: number,
  outChannels: number,
): Float32Array<ArrayBuffer> {
  if (inChannels % 4 !== 0) throw new Error(`packWeights requires inChannels % 4 === 0, got ${inChannels}`);
  const groups = inChannels / 4;
  const out = new Float32Array(new ArrayBuffer(planar.length * 4));
  for (let oc = 0; oc < outChannels; oc++) {
    for (let ic = 0; ic < inChannels; ic++) {
      const group = Math.floor(ic / 4);
      const lane = ic % 4;
      for (let k = 0; k < 9; k++) {
        out[((oc * groups + group) * 9 + k) * 4 + lane] = planar[(oc * inChannels + ic) * 9 + k] as number;
      }
    }
  }
  return out;
}

export type { Activation };
