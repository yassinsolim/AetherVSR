export interface UpsampleHeadShaderConfig {
  /** Low-resolution feature width. Multiple of 4. */
  readonly inChannels: number;
  /** Upscale factor. 2 for this milestone. */
  readonly scale: number;
  readonly useF16: boolean;
  readonly format: 'rgba8unorm' | 'rgba16float';
  /** Output pixels per invocation along each axis, at high resolution. */
  readonly blockX: number;
  readonly blockY: number;
  readonly tileX: number;
  readonly tileY: number;
  /**
   * Add a nearest-upsampled copy of the source image to the head's output, so
   * the network learns a residual rather than the whole picture.
   *
   * Standard practice in super-resolution and worth far more than its cost
   * here: the head already runs at high resolution, so this is one extra
   * texture fetch per output pixel. Nearest rather than bilinear because it is
   * exactly reproducible in the reference implementation and exactly
   * shift-equivariant in whole low-resolution pixels, which the temporal gate
   * checks.
   */
  readonly globalResidual: boolean;
}

/**
 * Resize-convolution reconstruction: nearest-neighbour upsample of the
 * low-resolution feature map, followed by a 3x3 convolution to RGB, evaluated
 * directly at high resolution in one pass.
 *
 * ## Relationship to sub-pixel convolution
 *
 * This computes the same *class* of function as a sub-pixel convolution head.
 * With nearest upsampling, the 3x3 window of a high-resolution output lands on
 * a different set of low-resolution taps depending on the output's parity in x
 * and y, so the four positions inside each 2x2 block apply four distinct
 * effective filters — which is exactly what a depth-to-space head produces from
 * its r^2 channel groups.
 *
 * It gets there differently. There is no learned kernel producing r^2 channels
 * and no periodic reshuffle; there is a fixed resize followed by an ordinary
 * convolution. Resize-then-convolve is the long-standing alternative
 * formulation, and is also the standard remedy for the checkerboard artefacts
 * that learned upsamplers are prone to — which matters here because
 * checkerboarding is precisely what the Milestone 3 temporal gate would fail.
 *
 * ## Cost
 *
 * The arithmetic moves to high resolution, so it is `scale^2` times the MAC
 * count of an equivalent low-resolution head: `4 * H * W * inChannels * 3 * 9`.
 * Against that, it replaces *both* the low-resolution head and the separate
 * shuffle pass, and it never materialises the intermediate `r^2 * 3` channel
 * tensor. Which side wins is measured, not assumed.
 *
 * Neighbouring outputs share most of their taps, so the reads are heavily
 * cache-resident despite being evaluated per high-resolution pixel.
 */
export function buildUpsampleHeadShader(config: UpsampleHeadShaderConfig): string {
  const { inChannels, scale, useF16, format, blockX, blockY, tileX, tileY } = config;
  const globalResidual = config.globalResidual;

  if (inChannels % 4 !== 0) {
    throw new Error(`upsample head requires inChannels % 4 === 0, got ${inChannels}`);
  }
  if (scale !== 2) {
    throw new Error(`upsample head currently assumes scale 2, got ${scale}`);
  }

  const T = useF16 ? 'f16' : 'f32';
  const V = `vec4<${T}>`;
  const enable = useF16 ? 'enable f16;\n' : '';
  const groups = inChannels / 4;
  const each = <R,>(n: number, f: (k: number) => R): R[] => Array.from({ length: n }, (_, k) => f(k));

  // acc[colour][m][i], flattened to stay in registers.
  const accDecl = each(3, (col) =>
    each(blockY, (m) =>
      each(blockX, (i) => `      var acc${col}_${m}_${i}: ${T} = biases[${col}u];`).join('\n'),
    ).join('\n'),
  ).join('\n');

  const macBody = each(blockY, (m) =>
    each(blockX, (i) => {
      // Taps are offset in *high-resolution* space and only then mapped down.
      // Offsetting after the divide would give both pixels of a pair identical
      // taps, collapsing the head to nearest-neighbour.
      const fetch =
        `        let v${m}_${i} = fetch(i32(outX + ${i}u) + kx - 1, i32(outY + ${m}u) + ky - 1, cg);`;
      const dots = each(
        3,
        (col) => `        acc${col}_${m}_${i} += dot(w${col}, v${m}_${i});`,
      ).join('\n');
      return `${fetch}\n${dots}`;
    }).join('\n'),
  ).join('\n');


  const stores = each(blockY, (m) =>
    each(blockX, (i) => {
      const base = globalResidual
        ? `+ residual(i32(outX + ${i}u), i32(outY + ${m}u))`
        : '';
      const c =
        `vec4f(f32(acc0_${m}_${i}), f32(acc1_${m}_${i}), f32(acc2_${m}_${i}), 0.0) ` +
        `${base} + vec4f(0.0, 0.0, 0.0, 1.0)`;
      return (
        `    if (outX + ${i}u < OW && outY + ${m}u < OH) { ` +
        `textureStore(dst, vec2i(i32(outX + ${i}u), i32(outY + ${m}u)), ` +
        `clamp(${c}, vec4f(0.0), vec4f(1.0))); }`
      );
    }).join('\n'),
  ).join('\n');

  const weightLoads = each(
    3,
    (col) => `      let w${col} = weights[(${col}u * IN_GROUPS + cg) * 9u + u32(ky * 3 + kx)];`,
  ).join('\n');

  return /* wgsl */ `${enable}
const IN_GROUPS: u32 = ${groups}u;

struct Params {
  width: u32,   // low-resolution width
  height: u32,  // low-resolution height
};

@group(0) @binding(0) var<storage, read> features: array<${V}>;
@group(0) @binding(1) var<storage, read> weights: array<${V}>;
@group(0) @binding(2) var<storage, read> biases: array<${T}>;
@group(0) @binding(3) var dst: texture_storage_2d<${format}, write>;
@group(0) @binding(4) var<uniform> params: Params;
${globalResidual ? '@group(0) @binding(5) var source: texture_2d<f32>;' : ''}

var<private> W: u32;
var<private> H: u32;
var<private> OW: u32;
var<private> OH: u32;

// Zero outside the *upsampled* image, matching Conv2d(padding=1) applied to the
// nearest-upsampled tensor. Coordinates arrive in high-resolution space and are
// mapped down here, which is what makes the four positions inside a 2x2 block
// see different low-resolution taps.
fn fetch(hx: i32, hy: i32, cg: u32) -> ${V} {
  if (hx < 0 || hy < 0 || hx >= i32(OW) || hy >= i32(OH)) {
    return ${V}(${T}(0.0));
  }
  let lx = u32(hx) / 2u;
  let ly = u32(hy) / 2u;
  return features[cg * W * H + ly * W + lx];
}

${
  globalResidual
    ? `// Nearest upsample of the source: the low-resolution pixel under this
// output. Alpha is dropped so only RGB is added.
fn residual(hx: i32, hy: i32) -> vec4f {
  let t = textureLoad(source, vec2i(hx / 2, hy / 2), 0);
  return vec4f(t.r, t.g, t.b, 0.0);
}`
    : ''
}

@compute @workgroup_size(${tileX}, ${tileY}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  W = params.width;
  H = params.height;
  OW = W * 2u;
  OH = H * 2u;

  let outX = gid.x * ${blockX}u;
  let outY = gid.y * ${blockY}u;
  if (outX >= OW || outY >= OH) {
    return;
  }

${accDecl}

  for (var cg: u32 = 0u; cg < IN_GROUPS; cg = cg + 1u) {
    for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
      for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
${weightLoads}
${macBody}
      }
    }
  }

${stores}
}
`;
}

/**
 * Planar `[colour][inChannels][tap]` head weights -> `[colour][ic/4][tap]` vec4.
 *
 * Runs once at model load, never per frame.
 */
export function packUpsampleHeadWeights(
  planar: Float32Array,
  inChannels: number,
): Float32Array<ArrayBuffer> {
  const expected = 3 * inChannels * 9;
  if (planar.length !== expected) {
    throw new Error(`head weights must be ${expected} elements (3*inC*9), got ${planar.length}`);
  }
  const groups = inChannels / 4;
  const out = new Float32Array(new ArrayBuffer(3 * groups * 9 * 4 * 4));
  for (let col = 0; col < 3; col++) {
    for (let ic = 0; ic < inChannels; ic++) {
      const g = Math.floor(ic / 4);
      const lane = ic % 4;
      for (let tap = 0; tap < 9; tap++) {
        out[((col * groups + g) * 9 + tap) * 4 + lane] = planar[(col * inChannels + ic) * 9 + tap] as number;
      }
    }
  }
  return out;
}
