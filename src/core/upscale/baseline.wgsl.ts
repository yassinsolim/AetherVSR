import type { FrameTextureKind } from '../types.js';

/** Reconstruction filters implemented by the baseline scaler. */
export type BaselineFilter = 'bilinear' | 'catmull-rom';

/**
 * Builds the baseline scaler's WGSL.
 *
 * Two axes of specialisation, both resolved at pipeline-creation time so the
 * frame loop never branches:
 *
 * - `sourceKind` selects between `texture_external` (the import path) and
 *   `texture_2d<f32>` (copy fallback). These are different WGSL types with
 *   different sampling builtins, so they cannot share one shader module.
 * - `filter` selects the reconstruction kernel.
 *
 * Filtering happens in the video's own transfer-encoded (non-linear) domain,
 * matching what `<video>` compositing and every other browser scaler does.
 * See DECISIONS.md ADR-0006 for why, and why the neural stage will revisit it.
 */
export function buildBaselineShader(sourceKind: FrameTextureKind, filter: BaselineFilter): string {
  const isExternal = sourceKind === 'external';

  const textureDecl = isExternal
    ? '@group(0) @binding(1) var srcTex: texture_external;'
    : '@group(0) @binding(1) var srcTex: texture_2d<f32>;';

  // `textureSampleBaseClampToEdge` is the only sampling builtin accepted for
  // external textures; it also gives us free edge clamping for the wide
  // Catmull-Rom footprint, so the 2D path uses a clamp-to-edge sampler to
  // match behaviour exactly.
  const sampleFn = isExternal
    ? 'return textureSampleBaseClampToEdge(srcTex, srcSampler, uv);'
    : 'return textureSampleLevel(srcTex, srcSampler, uv, 0.0);';

  return /* wgsl */ `
struct Uniforms {
  // Decoded source dimensions in pixels.
  srcSize: vec2f,
  // 1.0 / srcSize, precomputed on the CPU once per configuration.
  invSrcSize: vec2f,
};

@group(0) @binding(0) var srcSampler: sampler;
${textureDecl}
@group(0) @binding(2) var<uniform> u: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

// Single oversized triangle covering the viewport. Avoids a vertex buffer,
// avoids the diagonal seam of a two-triangle quad, and costs one draw call
// with three vertices.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  const corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let xy = corners[index];
  var out: VertexOutput;
  out.position = vec4f(xy, 0.0, 1.0);
  // Clip space is y-up, texture space is y-down.
  out.uv = vec2f((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
  return out;
}

fn sampleSource(uv: vec2f) -> vec4f {
  ${sampleFn}
}

${filter === 'catmull-rom' ? CATMULL_ROM : ''}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  ${
    filter === 'catmull-rom'
      ? 'let rgb = clamp(catmullRom(in.uv).rgb, vec3f(0.0), vec3f(1.0));'
      : 'let rgb = sampleSource(in.uv).rgb;'
  }
  return vec4f(rgb, 1.0);
}
`;
}

/**
 * Nine-tap Catmull-Rom (B = 0, C = 0.5 cubic) evaluated through the hardware
 * bilinear unit.
 *
 * The separable 4x4 kernel needs sixteen texel reads; pairing the two inner
 * taps per axis and letting the sampler interpolate between them collapses
 * that to nine filtered reads with identical output. The kernel has negative
 * lobes, so the caller clamps the result.
 */
const CATMULL_ROM = /* wgsl */ `
fn catmullRom(uv: vec2f) -> vec4f {
  let samplePos = uv * u.srcSize;
  // Centre of the texel containing the sample position.
  let texPos1 = floor(samplePos - 0.5) + 0.5;
  let f = samplePos - texPos1;
  let f2 = f * f;
  let f3 = f2 * f;

  let w0 = -0.5 * f3 + f2 - 0.5 * f;
  let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
  let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
  let w3 = 0.5 * f3 - 0.5 * f2;

  // Fuse taps 1 and 2 into one bilinear fetch positioned by their weight ratio.
  let w12 = w1 + w2;
  let offset12 = w2 / w12;

  let uv0 = (texPos1 - 1.0) * u.invSrcSize;
  let uv3 = (texPos1 + 2.0) * u.invSrcSize;
  let uv12 = (texPos1 + offset12) * u.invSrcSize;

  var acc = vec4f(0.0);
  acc += sampleSource(vec2f(uv0.x, uv0.y)) * (w0.x * w0.y);
  acc += sampleSource(vec2f(uv12.x, uv0.y)) * (w12.x * w0.y);
  acc += sampleSource(vec2f(uv3.x, uv0.y)) * (w3.x * w0.y);

  acc += sampleSource(vec2f(uv0.x, uv12.y)) * (w0.x * w12.y);
  acc += sampleSource(vec2f(uv12.x, uv12.y)) * (w12.x * w12.y);
  acc += sampleSource(vec2f(uv3.x, uv12.y)) * (w3.x * w12.y);

  acc += sampleSource(vec2f(uv0.x, uv3.y)) * (w0.x * w3.y);
  acc += sampleSource(vec2f(uv12.x, uv3.y)) * (w12.x * w3.y);
  acc += sampleSource(vec2f(uv3.x, uv3.y)) * (w3.x * w3.y);
  return acc;
}
`;
