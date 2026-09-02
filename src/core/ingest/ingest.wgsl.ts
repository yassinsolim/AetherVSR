/**
 * WGSL for the external-texture ingest pass.
 *
 * One fullscreen triangle, one `textureSampleBaseClampToEdge` per output
 * pixel, writing into an ordinary colour attachment. That single tap is the
 * entire point: it pays the external-texture sampling cost exactly once,
 * after which any number of downstream passes read a plain `texture_2d<f32>`.
 */
export function buildIngestShader(): string {
  return /* wgsl */ `
@group(0) @binding(0) var srcSampler: sampler;
@group(0) @binding(1) var srcTex: texture_external;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  const corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let xy = corners[index];
  var out: VertexOutput;
  out.position = vec4f(xy, 0.0, 1.0);
  out.uv = vec2f((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  // Sampled at 1:1 with the destination, so the sampler performs no
  // resampling; this measures conversion and bandwidth, not filtering.
  return vec4f(textureSampleBaseClampToEdge(srcTex, srcSampler, in.uv).rgb, 1.0);
}
`;
}
