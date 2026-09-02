export interface StemShaderConfig {
  /** Output feature width. Must be a multiple of 4 for the packed layout. */
  readonly outChannels: number;
  /** Output pixels per invocation along x and y. */
  readonly blockX: number;
  readonly blockY: number;
  readonly tileX: number;
  readonly tileY: number;
  readonly useF16: boolean;
  /** `relu` matches most SR stems; `none` leaves the projection linear. */
  readonly activation: 'none' | 'relu';
}

/**
 * Fused 3x3 convolution straight from the ingest texture into packed
 * activations: `RGB texture -> C_out feature tensor`, one pass.
 *
 * ## Why fused
 *
 * The obvious route is two passes — convert the texture to a packed C4 buffer,
 * then run the generic C4->C16 convolution. That works, and it is measured
 * against this, but it writes and re-reads a whole activation buffer for no
 * arithmetic reason.
 *
 * Reading the texture directly costs nine `textureLoad`s per output pixel, and
 * those nine feed *every* output channel — with `outChannels` 16 that is 144
 * multiply-accumulates per texel fetched. The stem is nowhere near
 * bandwidth-bound, so folding the conversion into it should be close to free.
 * Should be: the point of writing both is to find out.
 *
 * ## Weight layout
 *
 * `array<vec4<T>>` indexed `[oc * 9 + tap]`, each vector holding
 * `(wr, wg, wb, 0)`. A texture fetch already returns `vec4(r, g, b, a)`, so one
 * `dot` consumes a whole tap for one output channel and the alpha lane falls
 * out at zero weight. No channel loop, no swizzling.
 *
 * ## Padding
 *
 * Zero padding, matching `Conv2d(..., padding=1)` in PyTorch and the interior
 * convolutions. Clamp-to-edge would avoid a dark border ring but would then
 * disagree with the reference implementation the weights were trained under,
 * which is the more expensive kind of wrong.
 */
export function buildStemShader(config: StemShaderConfig): string {
  const { outChannels, blockX, blockY, tileX, tileY, useF16, activation } = config;

  if (outChannels % 4 !== 0) {
    throw new Error(`stem requires outChannels % 4 === 0, got ${outChannels}`);
  }
  if (!Number.isInteger(blockX) || blockX < 1 || !Number.isInteger(blockY) || blockY < 1) {
    throw new Error(`stem requires blockX and blockY >= 1, got ${blockX}x${blockY}`);
  }

  const T = useF16 ? 'f16' : 'f32';
  const V = `vec4<${T}>`;
  const enable = useF16 ? 'enable f16;\n' : '';
  const groups = outChannels / 4;
  const each = <R,>(n: number, f: (k: number) => R): R[] => Array.from({ length: n }, (_, k) => f(k));

  const activate = (expr: string): string =>
    activation === 'relu' ? `max(${expr}, ${T}(0.0))` : expr;

  // acc[oc][m][i]. Flattened to scalars so they stay in registers.
  const accDecl = each(outChannels, (oc) =>
    each(blockY, (m) =>
      each(blockX, (i) => `  var acc${oc}_${m}_${i}: ${T} = biases[${oc}u];`).join('\n'),
    ).join('\n'),
  ).join('\n');

  // One fetch per (row, column, tap), reused across every output channel.
  const macBody = each(blockY, (m) =>
    each(blockX, (i) => {
      const fetch = `      let s${m}_${i} = fetch(baseX + ${i}i, baseY + ${m}i, kx, ky);`;
      const dots = each(outChannels, (oc) => `      acc${oc}_${m}_${i} += dot(w${oc}, s${m}_${i});`).join(
        '\n',
      );
      return `${fetch}\n${dots}`;
    }).join('\n'),
  ).join('\n');

  const weightLoads = each(
    outChannels,
    (oc) => `      let w${oc} = weights[${oc}u * 9u + tap];`,
  ).join('\n');

  const stores = each(blockY, (m) => {
    const body = each(groups, (g) =>
      each(blockX, (i) => {
        const lanes = each(4, (l) => activate(`acc${g * 4 + l}_${m}_${i}`)).join(', ');
        return (
          `      if (x + ${i}u < W) { ` +
          `output[${g}u * W * H + y * W + x + ${i}u] = ${V}(${lanes}); }`
        );
      }).join('\n'),
    ).join('\n');
    return `  {\n    let y = outY + ${m}u;\n    if (y < H) {\n${body}\n    }\n  }`;
  }).join('\n');

  return /* wgsl */ `${enable}
const OUT_C: u32 = ${outChannels}u;
const BLOCK_X: u32 = ${blockX}u;
const BLOCK_Y: u32 = ${blockY}u;

struct Params {
  width: u32,
  height: u32,
  // Per-channel input transform applied before the convolution:
  //   value = (sample - mean) * scale
  // Networks trained on [0,1] RGB use mean 0, scale 1. Networks trained with a
  // dataset mean subtracted carry it here so no separate pass is needed.
  mean: vec4<f32>,
  scale: vec4<f32>,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${V}>;
@group(0) @binding(2) var<storage, read> biases: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${V}>;
@group(0) @binding(4) var<uniform> params: Params;

var<private> W: u32;
var<private> H: u32;

// Zero outside the image, matching Conv2d(padding=1).
fn fetch(px: i32, py: i32, kx: i32, ky: i32) -> ${V} {
  let sx = px + kx - 1;
  let sy = py + ky - 1;
  if (sx < 0 || sy < 0 || sx >= i32(W) || sy >= i32(H)) {
    return ${V}(${T}(0.0));
  }
  let raw = textureLoad(src, vec2i(sx, sy), 0);
  let norm = (raw - params.mean) * params.scale;
  // Alpha is forced to zero rather than trusted: a decoder surface may deliver
  // anything there, and the weight lane that would multiply it is also zero, so
  // this only guards against NaN propagating out of an undefined alpha.
  return ${V}(${T}(norm.r), ${T}(norm.g), ${T}(norm.b), ${T}(0.0));
}

@compute @workgroup_size(${tileX}, ${tileY}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  W = params.width;
  H = params.height;

  let x = gid.x * BLOCK_X;
  let outY = gid.y * BLOCK_Y;
  if (x >= W || outY >= H) {
    return;
  }
  let baseX = i32(x);
  let baseY = i32(outY);

${accDecl}

  for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
    for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
      let tap = u32(ky * 3 + kx);
${weightLoads}
${macBody}
    }
  }

${stores}
}
`;
}

/**
 * Planar `[oc][ic][tap]` stem weights (ic = 3, RGB) -> `[oc][tap]` vec4 with the
 * unused alpha lane zeroed.
 *
 * Runs once at model load. The alpha lane is explicitly zero so the fetch's
 * alpha can never contribute regardless of what the decoder put there.
 */
export function packStemWeights(planar: Float32Array, outChannels: number): Float32Array<ArrayBuffer> {
  const expected = outChannels * 3 * 9;
  if (planar.length !== expected) {
    throw new Error(`stem weights must be ${expected} elements (oc*3*9), got ${planar.length}`);
  }
  const out = new Float32Array(new ArrayBuffer(outChannels * 9 * 4 * 4));
  for (let oc = 0; oc < outChannels; oc++) {
    for (let ic = 0; ic < 3; ic++) {
      for (let tap = 0; tap < 9; tap++) {
        out[(oc * 9 + tap) * 4 + ic] = planar[(oc * 3 + ic) * 9 + tap] as number;
      }
    }
  }
  return out;
}
