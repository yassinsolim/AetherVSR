import type { Activation } from './activation.js';

/**
 * The convolution the production graph runs.
 *
 * Self-contained rather than extending the benchmark's tiled/packed config
 * types: those describe Milestone 3 experiments that exist only to be measured
 * against this one, and production code must not depend on `src/bench`.
 */
export interface BlockedConvShaderConfig {
  readonly inChannels: number;
  readonly outChannels: number;
  /** Workgroup dimensions in invocations. */
  readonly tileX: number;
  readonly tileY: number;
  /** Output pixels each invocation computes. */
  readonly blockX: number;
  readonly activation: Activation;
  readonly useF16: boolean;
  readonly residual: boolean;
  /** Output channels each invocation accumulates simultaneously. */
  readonly outBlock: number;
  /** Output rows each invocation computes. 1 restores pure horizontal blocking. */
  readonly blockY: number;
  /**
   * Memory order of the weight tensor.
   *
   * - `oc-major`  - `[oc][ic/4][k]`, the natural order. The outBlock weights
   *   needed at one tap sit `inGroups * 9` vec4s apart.
   * - `tap-major` - `[ic/4][k][oc]`, so those same outBlock weights are
   *   contiguous and load as one run.
   *
   * Weights are static, so any repacking happens once at model load and never
   * in the per-frame path. That makes layout free to choose and therefore
   * worth measuring rather than assuming.
   */
  readonly weightLayout: 'oc-major' | 'tap-major';
  /**
   * Emit `array<vec4<T>>` in the same grouped `[c/4][y][x][c%4]` layout the
   * kernel *reads*, instead of scalar planar `[c][y][x]`.
   *
   * This is what makes layers chainable at all. With scalar output the next
   * layer cannot consume this one without a repack pass, because the two
   * layouts disagree.
   *
   * Requires `outBlock % 4 === 0`, so that one invocation owns whole vec4s and
   * can store them entire. Storing single components of a vec4 would work but
   * compiles to masked stores, which is the thing worth measuring rather than
   * assuming.
   */
  readonly packedOutput: boolean;
}

/** Bytes of workgroup storage the blocked shader declares. */
export function blockedSharedBytes(config: BlockedConvShaderConfig): number {
  const tileW = config.tileX * config.blockX + 2;
  const tileH = config.tileY * config.blockY + 2;
  return tileW * tileH * (config.useF16 ? 8 : 16);
}

/**
 * Tiled, vec4-packed 3x3 convolution accumulating several output channels per
 * invocation.
 *
 * ## Why
 *
 * With one output channel per invocation, every value read from the staged
 * tile feeds exactly one multiply-add. The tile is then re-read, in full, by
 * every other workgroup in the z dimension — `outChannels` times over.
 *
 * Holding `outBlock` accumulators lets one shared-memory read feed `outBlock`
 * dot products, and cuts the number of z-slices — and therefore the number of
 * times the input is staged at all — by the same factor. The cost is
 * `blockX * outBlock` live accumulators plus `outBlock` weight vectors, which
 * is where occupancy starts to suffer.
 *
 * That trade-off is the measurement: this file makes the knob exist, it does
 * not assert which setting wins.
 */
export function buildBlockedConvShader(config: BlockedConvShaderConfig): string {
  const { inChannels, outChannels, tileX, tileY, blockX, blockY, outBlock, activation, useF16, residual } =
    config;
  const weightLayout = config.weightLayout;

  if (inChannels % 4 !== 0) {
    throw new Error(`blocked variant requires inChannels % 4 === 0, got ${inChannels}`);
  }
  if (outChannels % outBlock !== 0) {
    throw new Error(`blocked variant requires outChannels % outBlock === 0, got ${outChannels} % ${outBlock}`);
  }
  if (residual && inChannels !== outChannels) {
    throw new Error(`residual requires inChannels === outChannels, got ${inChannels} -> ${outChannels}`);
  }
  if (!Number.isInteger(blockY) || blockY < 1) {
    throw new Error(`blocked variant requires blockY >= 1, got ${blockY}`);
  }
  if (config.packedOutput && outBlock % 4 !== 0) {
    throw new Error(`packedOutput requires outBlock % 4 === 0, got ${outBlock}`);
  }
  if (config.packedOutput && outChannels % 4 !== 0) {
    throw new Error(`packedOutput requires outChannels % 4 === 0, got ${outChannels}`);
  }

  const T = useF16 ? 'f16' : 'f32';
  const V = `vec4<${T}>`;
  const enable = useF16 ? 'enable f16;\n' : '';
  const groups = inChannels / 4;
  const tileW = tileX * blockX + 2;
  const tileH = tileY * blockY + 2;
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

  // acc[j][m][i]: output channel ocBase+j, row outY+m, pixel outX+i. Flattened
  // so the compiler sees plain scalars rather than a dynamically indexed array,
  // which on Metal is the difference between registers and thread-local memory.
  const each = <R,>(n: number, f: (k: number) => R): R[] => Array.from({ length: n }, (_, k) => f(k));

  const accDecl = each(outBlock, (j) =>
    each(blockY, (m) =>
      each(blockX, (i) => `  var acc${j}_${m}_${i}: ${T} = biases[ocBase + ${j}u];`).join('\n'),
    ).join('\n'),
  ).join('\n');

  const weightLoads = each(outBlock, (j) =>
    weightLayout === 'tap-major'
      ? `        let w${j} = weights[wTap * OUT_C + ocBase + ${j}u];`
      : `        let w${j} = weights[(ocBase + ${j}u) * IN_GROUPS * 9u + wTap];`,
  ).join('\n');

  // One staged value feeds outBlock dot products. With blockY > 1 the vertical
  // taps of adjacent output rows overlap, so a tile row staged for row m is
  // re-read for rows m-1 and m-2 from the same shared storage while the
  // accumulators for all of them stay live in registers.
  const macBody = each(blockY, (m) =>
    each(blockX, (i) => {
      const v = `        let v${m}_${i} = tile[rowBase + ${m}u * TILE_W + localX + ${i}u + u32(kx)];`;
      const dots = each(outBlock, (j) => `        acc${j}_${m}_${i} += dot(w${j}, v${m}_${i});`).join('\n');
      return `${v}\n${dots}`;
    }).join('\n'),
  ).join('\n');

  const packedOutput = config.packedOutput;

  const stores = each(blockY, (m) => {
    let body: string;
    if (packedOutput) {
      // One invocation owns outBlock consecutive channels, so it owns
      // outBlock/4 whole vec4s and stores each in one go.
      body = each(outBlock / 4, (g) =>
        each(blockX, (i) => {
          const lanes = each(4, (l) => {
            const j = g * 4 + l;
            const oc = `(ocBase + ${j}u)`;
            return residual
              ? `${activate(`acc${j}_${m}_${i}`)} + input[(${oc} / 4u) * W * H + row * W + outX + ${i}u][${oc} % 4u]`
              : activate(`acc${j}_${m}_${i}`);
          }).join(', ');
          return (
            `        if (outX + ${i}u < W) { ` +
            `output[(ocBase / 4u + ${g}u) * W * H + row * W + outX + ${i}u] = ${V}(${lanes}); }`
          );
        }).join('\n'),
      ).join('\n');
    } else {
      body = each(outBlock, (j) =>
        each(blockX, (i) => {
          const oc = `(ocBase + ${j}u)`;
          const value = residual
            ? `${activate(`acc${j}_${m}_${i}`)} + input[(${oc} / 4u) * W * H + row * W + outX + ${i}u][${oc} % 4u]`
            : activate(`acc${j}_${m}_${i}`);
          return `        if (outX + ${i}u < W) { output[${oc} * W * H + row * W + outX + ${i}u] = ${value}; }`;
        }).join('\n'),
      ).join('\n');
    }
    return `  {\n    let row = outY + ${m}u;\n    if (row < H) {\n${body}\n    }\n  }`;
  }).join('\n');

  return /* wgsl */ `${enable}
const IN_GROUPS: u32 = ${groups}u;
const OUT_C: u32 = ${outChannels}u;
const OUT_BLOCK: u32 = ${outBlock}u;
const BLOCK_X: u32 = ${blockX}u;
const BLOCK_Y: u32 = ${blockY}u;
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
@group(0) @binding(3) var<storage, read_write> output: array<${packedOutput ? V : T}>;
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
  let originY = i32(wid.y * ${tileY}u * BLOCK_Y);
  let ocBase = wid.z * OUT_BLOCK;

  let localX = lid.x * BLOCK_X;
  let outX = u32(originX) + localX;
  let outY = u32(originY) + lid.y * BLOCK_Y;

${accDecl}

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

    for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
      let rowBase = (lid.y * BLOCK_Y + u32(ky)) * TILE_W;
      for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
        let wTap = cg * 9u + u32(ky * 3 + kx);
${weightLoads}
${macBody}
      }
    }
  }

  if (ocBase >= OUT_C) {
    return;
  }
${stores}
}
`;
}
