/**
 * Deterministic image-quality evaluation for upscalers.
 *
 * ## Methodology
 *
 * ```
 *   high-resolution reference  (generated, 2560x1440)
 *          | controlled 2x box downsample, exact, in JS
 *   low-resolution input       (1280x720)
 *          | the upscaler under test, on the GPU
 *   reconstruction             (2560x1440)
 *          | compare against the reference
 *   PSNR / SSIM
 * ```
 *
 * The reference is *generated*, not decoded from a compressed file, and the
 * downsample is an exact integer box filter implemented here rather than
 * delegated to `drawImage`, whose resampling is browser-defined. That matters:
 * comparing against a reference produced by an unknown compression and
 * resampling chain measures compression-artefact restoration and
 * super-resolution together, and the two cannot then be separated.
 *
 * A 2x box downsample is also exactly invertible in the low-frequency limit,
 * so a perfect upscaler would score infinitely well — the metric therefore
 * measures how much high-frequency detail a scaler fails to reconstruct, which
 * is the quantity of interest.
 */

export interface QualityScores {
  readonly label: string;
  /** Peak signal-to-noise ratio over luma, in dB. Higher is better. */
  readonly psnrLuma: number;
  /** PSNR over RGB channels jointly, in dB. */
  readonly psnrRgb: number;
  /** Mean structural similarity over 8x8 luma windows, in [0, 1]. */
  readonly ssimLuma: number;
  readonly width: number;
  readonly height: number;
}

/** ITU-R BT.601 luma, matching the convention used by most SR literature. */
function luma(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * Deterministic high-resolution test image.
 *
 * Content is chosen so that a downsample genuinely destroys information an
 * upscaler must invent: a radial zone plate sweeping to Nyquist, hard diagonal
 * edges at several angles, fine gratings, and smooth gradients as a control
 * region where every scaler should score well.
 */
export function generateReference(width: number, height: number): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  const cx = width * 0.32;
  const cy = height * 0.5;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const dx = x - cx;
      const dy = y - cy;

      // Zone plate: frequency rises with radius, reaching the sampling limit.
      const zone = 127.5 * (1 + Math.cos((dx * dx + dy * dy) * 0.0009));
      // Diagonal bars, period varying across the image.
      const period = 3 + ((x / width) * 13) | 0;
      const diag = ((x + y) % period) < period / 2 ? 255 : 0;
      // Smooth gradient control region.
      const grad = (x / width) * 255;

      const region = x / width;
      let value: number;
      if (region < 0.45) value = zone;
      else if (region < 0.75) value = diag;
      else value = grad;

      // Per-channel variation so chroma is exercised, not just luma.
      data[i] = value;
      data[i + 1] = (value * 0.7 + (y / height) * 76) % 256;
      data[i + 2] = (value * 0.4 + 128) % 256;
      data[i + 3] = 255;
    }
  }
  return new ImageData(data, width, height);
}

/**
 * Exact 2x box downsample: each output pixel is the mean of a 2x2 block.
 *
 * Averaging is done in the encoded (sRGB-ish) domain, matching how the
 * pipeline filters — see DECISIONS.md ADR-0006. Doing it in linear light here
 * while the upscaler works in the encoded domain would introduce a systematic
 * bias unrelated to scaler quality.
 */
export function boxDownsample2x(source: ImageData): ImageData {
  const w = source.width >> 1;
  const h = source.height >> 1;
  const out = new Uint8ClampedArray(w * h * 4);
  const src = source.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      for (let c = 0; c < 4; c++) {
        const a = src[(2 * y * source.width + 2 * x) * 4 + c] as number;
        const b = src[(2 * y * source.width + 2 * x + 1) * 4 + c] as number;
        const d = src[((2 * y + 1) * source.width + 2 * x) * 4 + c] as number;
        const e = src[((2 * y + 1) * source.width + 2 * x + 1) * 4 + c] as number;
        out[o + c] = (a + b + d + e) / 4;
      }
    }
  }
  return new ImageData(out, w, h);
}

/** PSNR in dB for 8-bit data. Returns Infinity for identical inputs. */
function psnrFromMse(mse: number): number {
  if (mse <= 0) return Infinity;
  return 10 * Math.log10((255 * 255) / mse);
}

/**
 * Compares a reconstruction against the reference.
 *
 * `reconstruction` is RGBA8 in the same dimensions as `reference`.
 */
export function score(label: string, reference: ImageData, reconstruction: Uint8Array): QualityScores {
  const { width, height } = reference;
  const ref = reference.data;
  const n = width * height;

  const refLuma = new Float32Array(n);
  const recLuma = new Float32Array(n);
  let seRgb = 0;
  let seLuma = 0;

  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const rr = ref[o] as number;
    const rg = ref[o + 1] as number;
    const rb = ref[o + 2] as number;
    const cr = reconstruction[o] as number;
    const cg = reconstruction[o + 1] as number;
    const cb = reconstruction[o + 2] as number;

    seRgb += (rr - cr) ** 2 + (rg - cg) ** 2 + (rb - cb) ** 2;

    const rl = luma(rr, rg, rb);
    const cl = luma(cr, cg, cb);
    refLuma[i] = rl;
    recLuma[i] = cl;
    seLuma += (rl - cl) ** 2;
  }

  return {
    label,
    psnrRgb: psnrFromMse(seRgb / (n * 3)),
    psnrLuma: psnrFromMse(seLuma / n),
    ssimLuma: meanSsim(refLuma, recLuma, width, height),
    width,
    height,
  };
}

/**
 * Mean SSIM over non-overlapping 8x8 luma windows.
 *
 * The canonical formulation uses an 11x11 Gaussian window; 8x8 uniform windows
 * are the common fast approximation and are adequate here because the metric
 * is used to *rank* scalers on one fixed image, not to publish an absolute
 * value comparable with the literature. Stated plainly so nobody later
 * compares these numbers against a paper's SSIM.
 */
function meanSsim(a: Float32Array, b: Float32Array, width: number, height: number): number {
  const win = 8;
  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  let total = 0;
  let windows = 0;

  for (let wy = 0; wy + win <= height; wy += win) {
    for (let wx = 0; wx + win <= width; wx += win) {
      let sumA = 0;
      let sumB = 0;
      let sumAA = 0;
      let sumBB = 0;
      let sumAB = 0;
      for (let y = 0; y < win; y++) {
        for (let x = 0; x < win; x++) {
          const i = (wy + y) * width + wx + x;
          const va = a[i] as number;
          const vb = b[i] as number;
          sumA += va;
          sumB += vb;
          sumAA += va * va;
          sumBB += vb * vb;
          sumAB += va * vb;
        }
      }
      const count = win * win;
      const muA = sumA / count;
      const muB = sumB / count;
      const varA = sumAA / count - muA * muA;
      const varB = sumBB / count - muB * muB;
      const cov = sumAB / count - muA * muB;
      const ssim =
        ((2 * muA * muB + C1) * (2 * cov + C2)) /
        ((muA * muA + muB * muB + C1) * (varA + varB + C2));
      total += ssim;
      windows++;
    }
  }
  return windows === 0 ? NaN : total / windows;
}
