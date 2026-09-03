/**
 * Fullscreen copy from the reconstruction texture to the presentation target.
 *
 * The reconstruction is a compute pass, so it writes a storage texture rather
 * than a colour attachment. This moves that result to the swap-chain view.
 *
 * Writing the swap chain directly from the compute pass would save the copy,
 * but requires configuring the canvas with STORAGE_BINDING and rebuilding a
 * bind group every frame as the swap-chain texture rotates. This keeps the
 * upscaler's contract identical to the baseline scaler's - render into the
 * target you were handed - and allocates nothing per frame, at the cost of one
 * read and one write of the output image.
 *
 * Nearest sampling: source and destination are the same size, so any filtering
 * would only soften an exact copy.
 */
export function buildBlitShader(format: GPUTextureFormat): string {
  return /* wgsl */ `
// target format: ${format}
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
  // One oversized triangle covering the viewport; cheaper than two triangles
  // and avoids a seam along the shared diagonal.
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  var out: VsOut;
  out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(uv.x, 1.0 - uv.y);
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  return textureSample(src, samp, in.uv);
}
`;
}
