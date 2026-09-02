export interface PixelShuffleShaderConfig {
  /** Output colour channels. 3 for RGB. */
  readonly outChannels: number;
  /** Upscale factor. 2 for this milestone. */
  readonly scale: number;
  /** Input activations are f16. */
  readonly useF16: boolean;
  /** Storage format of the destination texture. */
  readonly format: 'rgba8unorm' | 'rgba16float';
}

/**
 * Depth-to-space reconstruction: packed `C = outChannels * scale^2` activations
 * at low resolution -> an RGB texture at `scale x` resolution.
 *
 * ## Layout
 *
 * PyTorch's `PixelShuffle(r)` maps `(C*r*r, H, W) -> (C, H*r, W*r)` with
 *
 * ```
 *   out[c][h*r + i][w*r + j] = in[c*r*r + i*r + j][h][w]
 * ```
 *
 * With `C = 3` and `r = 2` the input has 12 channels, and the grouped `vec4`
 * layout lines up exactly: channels 0-3 are group 0, 4-7 group 1, 8-11 group 2,
 * so each colour occupies one whole group and the four sub-pixel positions are
 * that group's four lanes. One invocation reads three `vec4`s — one per colour
 * — and writes the whole `r x r` block. Nothing is gathered twice.
 *
 * ## Why compute rather than a fragment pass
 *
 * A fragment shader over the high-resolution target would run one invocation
 * per *output* pixel, and each would read three whole `vec4`s to use three of
 * their twelve lanes. At 2560x1440 that is ~88 MB read per frame against ~22 MB
 * here, on a device measured at 126 GB/s. The compute form reads each activation
 * exactly once.
 */
export function buildPixelShuffleShader(config: PixelShuffleShaderConfig): string {
  const { outChannels, scale, useF16, format } = config;

  if (outChannels !== 3) {
    throw new Error(`pixel shuffle currently assumes 3 colour channels, got ${outChannels}`);
  }
  if (scale !== 2) {
    throw new Error(`pixel shuffle currently assumes scale 2, got ${scale}`);
  }

  const T = useF16 ? 'f16' : 'f32';
  const V = `vec4<${T}>`;
  const enable = useF16 ? 'enable f16;\n' : '';

  return /* wgsl */ `${enable}
struct Params {
  width: u32,   // low-resolution width
  height: u32,  // low-resolution height
};

@group(0) @binding(0) var<storage, read> activations: array<${V}>;
@group(0) @binding(1) var dst: texture_storage_2d<${format}, write>;
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
  let plane = W * H;

  // One group per colour: lane (iy*2 + ix) is the sub-pixel at that offset.
  let r = activations[i];
  let g = activations[plane + i];
  let b = activations[2u * plane + i];

  for (var iy: u32 = 0u; iy < 2u; iy = iy + 1u) {
    for (var ix: u32 = 0u; ix < 2u; ix = ix + 1u) {
      let lane = iy * 2u + ix;
      let c = vec4f(f32(r[lane]), f32(g[lane]), f32(b[lane]), 1.0);
      // The network is trained against [0,1] targets; anything outside is a
      // clipped highlight or an undershoot, not a value the display can show.
      textureStore(dst, vec2i(i32(x * 2u + ix), i32(y * 2u + iy)), clamp(c, vec4f(0.0), vec4f(1.0)));
    }
  }
}
`;
}

/**
 * Reorders a PyTorch `[outChannels*r*r][inChannels][tap]` head weight tensor so
 * that the shuffle above reads whole groups.
 *
 * PyTorch orders the head's output channels as `c*r*r + i*r + j`, which is
 * already exactly group-major with the sub-pixel index as the lane. So this is
 * the identity — but it is written out and tested rather than assumed, because
 * a silent transposition here produces a picture that looks almost right and
 * shimmers under motion.
 */
export function pixelShuffleHeadChannelOrder(outChannels: number, scale: number): number[] {
  const order: number[] = [];
  for (let c = 0; c < outChannels; c++) {
    for (let i = 0; i < scale; i++) {
      for (let j = 0; j < scale; j++) {
        order.push(c * scale * scale + i * scale + j);
      }
    }
  }
  return order;
}
