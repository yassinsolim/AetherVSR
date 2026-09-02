/** Activation applied after the convolution. */
export type Activation = 'none' | 'relu' | 'tanh';

export interface ConvShaderConfig {
  /** Input channels. */
  readonly inChannels: number;
  /** Output channels. */
  readonly outChannels: number;
  /** Workgroup tile in output pixels. */
  readonly tileX: number;
  readonly tileY: number;
  /** Output pixels each invocation computes along x. Register blocking. */
  readonly blockX: number;
  readonly activation: Activation;
  /** Emit `f16` arithmetic. Requires the `shader-f16` device feature. */
  readonly useF16: boolean;
  /** Add the input to the output (requires inChannels === outChannels). */
  readonly residual: boolean;
}

/**
 * Builds a 3x3 convolution compute shader over NCHW-ish planar storage
 * buffers.
 *
 * ## Layout
 *
 * Activations are stored channel-planar: `data[c * W * H + y * W + x]`. Planar
 * beats interleaved here because a 3x3 convolution reads a 3x3 window from
 * *every* input channel, so consecutive invocations along x read consecutive
 * addresses within a plane — coalesced. Weights are `[outC][inC][3][3]`.
 *
 * ## Register blocking
 *
 * Each invocation computes `blockX` horizontally adjacent output pixels. The
 * 3x3 windows of adjacent outputs overlap by two columns, so the loaded input
 * values are reused `blockX` times instead of being re-fetched. This is the
 * single most effective optimisation for a bandwidth-bound convolution and it
 * is why the harness is parameterised on it rather than hard-coding 1.
 *
 * ## Precision
 *
 * `useF16` switches the accumulator and storage element type to `f16`. That
 * halves activation bandwidth and, on hardware with double-rate f16 ALUs, can
 * roughly double arithmetic throughput. It is a real numerical change, not a
 * free win, so the harness measures both and the caller compares.
 */
export function buildConvShader(config: ConvShaderConfig): string {
  const { inChannels, outChannels, tileX, tileY, blockX, activation, useF16, residual } = config;
  const enable = useF16 ? 'enable f16;\n' : '';
  const T = useF16 ? 'f16' : 'f32';
  const acc = useF16 ? 'f16' : 'f32';

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

  // Unrolled accumulator declarations, one per horizontally blocked output.
  const accDecl = Array.from({ length: blockX }, (_, i) => `  var acc${i}: ${acc} = bias;`).join('\n');
  if (residual && inChannels !== outChannels) {
    throw new Error(
      `residual requires inChannels === outChannels, got ${inChannels} -> ${outChannels}`,
    );
  }

  const accStore = Array.from({ length: blockX }, (_, i) => {
    // The residual reads the *output* channel's input plane. `outPlane` is
    // valid here; the per-channel `inBase` is scoped inside the accumulation
    // loop and referencing it out here does not compile.
    const value = residual
      ? `${activate(`acc${i}`)} + input[outPlane + outY * W + outX + ${i}u]`
      : activate(`acc${i}`);
    const guard = `if (outX + ${i}u < W) { output[outPlane + outY * W + outX + ${i}u] = ${value}; }`;
    return `  ${guard}`;
  }).join('\n');

  const macBody = Array.from(
    { length: blockX },
    (_, i) => `        acc${i} += w * sampleAt(inBase, i32(outX) + ${i} + kx - 1, i32(outY) + ky - 1);`,
  ).join('\n');

  return /* wgsl */ `${enable}
const IN_C: u32 = ${inChannels}u;
const OUT_C: u32 = ${outChannels}u;
const BLOCK_X: u32 = ${blockX}u;

struct Dims {
  width: u32,
  height: u32,
};

@group(0) @binding(0) var<storage, read> input: array<${T}>;
@group(0) @binding(1) var<storage, read> weights: array<${T}>;
@group(0) @binding(2) var<storage, read> biases: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<private> W: u32;
var<private> H: u32;

// Zero padding at the border, matching the usual SR convention.
fn sampleAt(planeBase: u32, x: i32, y: i32) -> ${T} {
  if (x < 0 || y < 0 || x >= i32(W) || y >= i32(H)) {
    return ${T}(0.0);
  }
  return input[planeBase + u32(y) * W + u32(x)];
}

@compute @workgroup_size(${tileX}, ${tileY}, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  W = dims.width;
  H = dims.height;

  let outX = gid.x * BLOCK_X;
  let outY = gid.y;
  let outC = gid.z;
  if (outY >= H || outX >= W || outC >= OUT_C) {
    return;
  }

  let bias = biases[outC];
${accDecl}

  let outPlane = outC * W * H;
  let wOutBase = outC * IN_C * 9u;

  for (var ic: u32 = 0u; ic < IN_C; ic = ic + 1u) {
    let inBase = ic * W * H;
    let wBase = wOutBase + ic * 9u;
    for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
      for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
        let w = weights[wBase + u32(ky * 3 + kx)];
${macBody}
      }
    }
  }

${accStore}
}
`;
}

/**
 * Multiply-accumulate count actually issued by one dispatch.
 *
 * A 3x3 convolution performs `9 * inC` MACs per output pixel per output
 * channel. The arguments are the *issued* extent, not the image extent, because
 * throughput must be computed from the work the GPU actually performed:
 *
 * - Border pixels do fewer useful MACs because of zero padding, but the shader
 *   still issues the reads and multiplies.
 * - Every kernel here accumulates its whole spatial block and guards only the
 *   *stores*, so a dispatch that overhangs the image still pays for the
 *   overhang. How far it overhangs depends on the variant's dispatch geometry,
 *   which is why the caller computes the extent rather than this function
 *   guessing from `blockX` alone.
 *
 * Understating the extent understates throughput; overstating it flatters it.
 * Neither is acceptable, so the caller derives both extents from the same
 * numbers it passes to `dispatchWorkgroups`.
 */
export function convMacCount(
  issuedWidth: number,
  issuedHeight: number,
  inChannels: number,
  outChannels: number,
): number {
  return issuedWidth * issuedHeight * inChannels * outChannels * 9;
}
