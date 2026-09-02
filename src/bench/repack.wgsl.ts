export interface RepackShaderConfig {
  readonly channels: number;
  readonly useF16: boolean;
}

/**
 * Scalar planar `[c][y][x]` -> grouped `[c/4][y][x][c%4]`.
 *
 * The conversion a chain needs when its convolution writes planar output but
 * the next layer reads grouped. One invocation produces one whole `vec4` by
 * gathering four channels of the same pixel, which are `pixels` elements apart
 * in the source — a strided gather, four times per output vector.
 *
 * That stride is the reason this pass is not free and the reason the packed
 * store inside the convolution is worth measuring against it. Removing a pass
 * is not automatically a win; doing the same work inside a kernel that was
 * previously bandwidth-bound can cost more than the pass it replaces.
 */
export function buildRepackShader(config: RepackShaderConfig): string {
  const { channels, useF16 } = config;
  if (channels % 4 !== 0) {
    throw new Error(`repack requires channels % 4 === 0, got ${channels}`);
  }
  const T = useF16 ? 'f16' : 'f32';
  const V = `vec4<${T}>`;
  const enable = useF16 ? 'enable f16;\n' : '';

  return /* wgsl */ `${enable}
const GROUPS: u32 = ${channels / 4}u;

struct Dims {
  width: u32,
  height: u32,
};

@group(0) @binding(0) var<storage, read> planar: array<${T}>;
@group(0) @binding(1) var<storage, read_write> grouped: array<${V}>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pixels = dims.width * dims.height;
  let total = pixels * GROUPS;
  let i = gid.x;
  if (i >= total) {
    return;
  }
  // Destination vec4 i covers channel group i / pixels at pixel i % pixels.
  let group = i / pixels;
  let p = i % pixels;
  let base = group * 4u * pixels + p;
  grouped[i] = ${V}(
    planar[base],
    planar[base + pixels],
    planar[base + 2u * pixels],
    planar[base + 3u * pixels],
  );
}
`;
}
