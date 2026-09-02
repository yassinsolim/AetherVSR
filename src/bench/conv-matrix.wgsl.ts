import type { Activation } from './conv.wgsl.js';

export interface MatrixConvShaderConfig {
  readonly inChannels: number;
  readonly outChannels: number;
  /** Output rows one workgroup handles. Amortises the weight loads. */
  readonly rowsPerGroup: number;
  readonly activation: Activation;
  readonly useF16: boolean;
}

/** The only subgroup-matrix shape Dawn exposes on Metal: 8x8x8, f32 or f16. */
export const MATRIX_DIM = 8;

/**
 * Workgroup storage the matrix shader declares: the staged right-hand operand
 * and the result landing zone, one 8x8 tile each.
 */
export function matrixSharedBytes(useF16: boolean): number {
  return 2 * MATRIX_DIM * MATRIX_DIM * (useF16 ? 2 : 4);
}

/**
 * 3x3 convolution as an implicit GEMM on `chromium_experimental_subgroup_matrix`.
 *
 * ## Why implicit rather than im2col
 *
 * The textbook route is im2col: materialise the 9x-expanded activation matrix
 * and call a GEMM. At 1280x720 with 16 channels that expansion is 265 MB in
 * f16 — larger than the 128 MiB default storage binding limit, and it would
 * add roughly half a gigabyte of pure buffer traffic before any arithmetic
 * happens. So the patch gather is done inline, into workgroup memory, one
 * K-slice at a time.
 *
 * ## Mapping
 *
 * With `M` = 8 output channels, `N` = 8 output pixels, `K` = 9 * inChannels:
 *
 * - `k` is ordered as `tap * inChannels + ic`, so each 8-wide K-slice sits
 *   inside a single 3x3 tap and differs only by input channel. That makes the
 *   staged right-hand matrix eight channels of eight horizontally adjacent
 *   pixels — a contiguous run per channel, not a scattered gather.
 * - The left-hand matrix is weights, pre-arranged offline into exactly the
 *   order `subgroupMatrixLoad` wants, so no runtime shuffling is needed.
 *
 * ## Syntax note
 *
 * Chrome 152 ships the *older* Dawn spelling: a boolean `col_major` parameter
 * on load/store, and `subgroupMatrixMultiplyAccumulate` templated on the
 * result *component* type. The in-progress W3C proposal renames the extension
 * to `subgroup_matrix` and replaces the boolean with a `row_major`/`col_major`
 * template argument, so this shader is written against a moving target and is
 * expected to need rewriting. That is a portability cost, recorded here rather
 * than discovered later.
 *
 * ## Requirements
 *
 * `workgroup_size.x` must be a multiple of the adapter's subgroup size (32 on
 * this device), and both channel counts must be multiples of 8 so the tiles
 * divide exactly. Partial tiles are rejected rather than padded: padding would
 * add MACs that the throughput figure would then take credit for.
 */
export function buildMatrixConvShader(config: MatrixConvShaderConfig): string {
  const { inChannels, outChannels, rowsPerGroup, activation, useF16 } = config;

  if (inChannels % MATRIX_DIM !== 0 || outChannels % MATRIX_DIM !== 0) {
    throw new Error(
      `matrix variant requires channels to be multiples of ${MATRIX_DIM}, got ${inChannels} -> ${outChannels}`,
    );
  }

  const T = useF16 ? 'f16' : 'f32';
  const enable = `enable chromium_experimental_subgroup_matrix;\n${useF16 ? 'enable f16;\n' : ''}`;
  const slices = (9 * inChannels) / MATRIX_DIM;

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

  return /* wgsl */ `${enable}
alias Lhs = subgroup_matrix_left<${T}, ${MATRIX_DIM}, ${MATRIX_DIM}>;
alias Rhs = subgroup_matrix_right<${T}, ${MATRIX_DIM}, ${MATRIX_DIM}>;
alias Acc = subgroup_matrix_result<${T}, ${MATRIX_DIM}, ${MATRIX_DIM}>;

const IN_C: u32 = ${inChannels}u;
const OUT_C: u32 = ${outChannels}u;
const SLICES: u32 = ${slices}u;
const D: u32 = ${MATRIX_DIM}u;
const ROWS: u32 = ${rowsPerGroup}u;

struct Dims {
  width: u32,
  height: u32,
};

@group(0) @binding(0) var<storage, read> input: array<${T}>;
@group(0) @binding(1) var<storage, read> weights: array<${T}>;
@group(0) @binding(2) var<storage, read> biases: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<uniform> dims: Dims;

// Right-hand operand: 8 input channels x 8 horizontally adjacent pixels.
var<workgroup> staged: array<${T}, ${MATRIX_DIM * MATRIX_DIM}>;
// Result landing zone: 8 output channels x 8 pixels.
var<workgroup> resolved: array<${T}, ${MATRIX_DIM * MATRIX_DIM}>;

@compute @workgroup_size(32)
fn main(
  @builtin(local_invocation_index) lindex: u32,
  @builtin(workgroup_id) wid: vec3u,
) {
  let W = dims.width;
  let H = dims.height;

  let baseX = wid.x * D;
  let ocBase = wid.z * D;
  let wGroupBase = wid.z * SLICES * D * D;

  for (var r: u32 = 0u; r < ROWS; r = r + 1u) {
    let y = wid.y * ROWS + r;

    var acc: Acc = Acc();

    for (var s: u32 = 0u; s < SLICES; s = s + 1u) {
      // Each slice lies inside one tap: k = tap * IN_C + ic.
      let kBase = s * D;
      let tap = kBase / IN_C;
      let ic0 = kBase % IN_C;
      let dy = i32(tap / 3u) - 1;
      let dx = i32(tap % 3u) - 1;
      let sy = i32(y) + dy;

      workgroupBarrier();
      // 32 invocations stage 64 values: row = channel, column = pixel.
      for (var i: u32 = lindex; i < D * D; i = i + 32u) {
        let kk = i / D;
        let n = i % D;
        let sx = i32(baseX + n) + dx;
        var v: ${T} = ${T}(0.0);
        if (sx >= 0 && sy >= 0 && sx < i32(W) && sy < i32(H)) {
          v = input[(ic0 + kk) * W * H + u32(sy) * W + u32(sx)];
        }
        staged[i] = v;
      }
      workgroupBarrier();

      // Weights are pre-arranged into load order, so this is a straight read.
      let lhs = subgroupMatrixLoad<Lhs>(&weights, wGroupBase + s * D * D, false, D);
      let rhs = subgroupMatrixLoad<Rhs>(&staged, 0u, false, D);
      acc = subgroupMatrixMultiplyAccumulate(lhs, rhs, acc);
    }

    workgroupBarrier();
    subgroupMatrixStore(&resolved, 0u, acc, false, D);
    workgroupBarrier();

    if (y < H) {
      for (var i: u32 = lindex; i < D * D; i = i + 32u) {
        let m = i / D;
        let n = i % D;
        let x = baseX + n;
        if (x < W) {
          let sum = resolved[i] + biases[ocBase + m];
          output[(ocBase + m) * W * H + y * W + x] = ${activate('sum')};
        }
      }
    }
  }
}
`;
}

/**
 * Planar `[oc][ic][k]` weights -> subgroup-matrix load order.
 *
 * Destination is `[oc/8][slice][m][kk]`, matching a row-major
 * `subgroup_matrix_left` (8 rows of output channel, 8 columns of K) loaded with
 * stride 8. Static, so this runs once at load time.
 */
export function matrixWeightIndex(
  inChannels: number,
  outChannels: number,
): (planarIndex: number) => number {
  const slices = (9 * inChannels) / MATRIX_DIM;
  return (i) => {
    const tap = i % 9;
    const ic = Math.floor(i / 9) % inChannels;
    const oc = Math.floor(i / (9 * inChannels)) % outChannels;
    const k = tap * inChannels + ic;
    const slice = Math.floor(k / MATRIX_DIM);
    const kk = k % MATRIX_DIM;
    const ocBlock = Math.floor(oc / MATRIX_DIM);
    const m = oc % MATRIX_DIM;
    return ((ocBlock * slices + slice) * MATRIX_DIM + m) * MATRIX_DIM + kk;
  };
}

/** Applies {@link matrixWeightIndex} to a planar weight array. */
export function toMatrixWeights(
  planar: Float32Array,
  inChannels: number,
  outChannels: number,
): Float32Array<ArrayBuffer> {
  const at = matrixWeightIndex(inChannels, outChannels);
  const out = new Float32Array(new ArrayBuffer(planar.length * 4));
  for (let i = 0; i < planar.length; i++) out[at(i)] = planar[i] as number;
  return out;
}
