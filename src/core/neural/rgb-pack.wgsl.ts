export interface RgbPackShaderConfig {
  readonly useF16: boolean;
}

/**
 * `RGB texture -> packed C4 activation buffer`, one texel per invocation.
 *
 * The first half of the unfused stem route: convert here, then run the generic
 * C4 -> C16 convolution. It exists to be measured against
 * {@link buildStemShader}, which does both in one pass.
 *
 * The fourth channel is zero rather than the decoder's alpha. Feeding an
 * undefined alpha into a network trained on three channels would be a silent
 * correctness bug, and a constant zero is free.
 */
export function buildRgbPackShader(config: RgbPackShaderConfig): string {
  const T = config.useF16 ? 'f16' : 'f32';
  const V = `vec4<${T}>`;
  const enable = config.useF16 ? 'enable f16;\n' : '';

  return /* wgsl */ `${enable}
struct Params {
  width: u32,
  height: u32,
  mean: vec4<f32>,
  scale: vec4<f32>,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<${V}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let W = params.width;
  let H = params.height;
  let i = gid.x;
  if (i >= W * H) {
    return;
  }
  let x = i % W;
  let y = i / W;
  let raw = textureLoad(src, vec2i(i32(x), i32(y)), 0);
  let norm = (raw - params.mean) * params.scale;
  output[i] = ${V}(${T}(norm.r), ${T}(norm.g), ${T}(norm.b), ${T}(0.0));
}
`;
}
