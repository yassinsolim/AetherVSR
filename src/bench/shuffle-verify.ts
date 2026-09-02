import { buildPixelShuffleShader } from '../core/neural/pixel-shuffle.wgsl.js';
import { floatToHalf } from './conv-bench.js';

export interface ShuffleVerifyCase {
  /** Low-resolution dimensions. Output is 2x this. */
  readonly width: number;
  readonly height: number;
  readonly useF16?: boolean;
}

export interface ShuffleVerifyResult extends ShuffleVerifyCase {
  readonly diagnostics: readonly string[];
  readonly maxAbsError: number;
  readonly passed: boolean;
  readonly outputPixels: number;
  /** True when a deliberately transposed layout is correctly rejected. */
  readonly detectsTransposition: boolean;
  readonly clampChecked: boolean;
}

/**
 * Checks depth-to-space against a CPU reference implementing PyTorch's
 * `PixelShuffle(2)` indexing.
 *
 * Two failure modes matter here and neither is visible by eye on a still frame.
 * A transposed sub-pixel index swaps the two off-diagonal positions of every
 * 2x2 block, which reads as a faint diagonal texture and becomes obvious
 * shimmer only once the picture moves. A group/lane mix-up permutes colour
 * channels per sub-pixel. So this test also runs the reference with the
 * transposition applied and requires that it *disagrees* — a test that passes
 * against both orderings is testing nothing.
 */
export async function verifyPixelShuffle(
  device: GPUDevice,
  c: ShuffleVerifyCase,
  tolerance = 1 / 255,
): Promise<ShuffleVerifyResult> {
  const useF16 = c.useF16 ?? false;
  const bytesPerElement = useF16 ? 2 : 4;
  const pixels = c.width * c.height;
  const channels = 12;
  const elements = pixels * channels;
  const outW = c.width * 2;
  const outH = c.height * 2;
  const diagnostics: string[] = [];

  // Distinct value per (channel, pixel), plus deliberate out-of-range values so
  // the clamp is exercised rather than assumed.
  const planar = new Float32Array(elements);
  for (let ch = 0; ch < channels; ch++) {
    for (let p = 0; p < pixels; p++) {
      const v = ((ch * 7919 + p * 104729) % 1000) / 999;
      planar[ch * pixels + p] = p % 97 === 0 ? (ch % 2 === 0 ? 1.4 : -0.3) : v;
    }
  }
  // Grouped layout: [c/4][pixel][c%4].
  const grouped = new Float32Array(new ArrayBuffer(elements * 4));
  for (let ch = 0; ch < channels; ch++) {
    const g = Math.floor(ch / 4);
    const lane = ch % 4;
    for (let p = 0; p < pixels; p++) grouped[(g * pixels + p) * 4 + lane] = planar[ch * pixels + p] as number;
  }

  const activations = device.createBuffer({
    size: Math.max(16, Math.ceil((elements * bytesPerElement) / 16) * 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (useF16) {
    const half = new Uint16Array(elements + (elements % 2));
    for (let i = 0; i < elements; i++) half[i] = floatToHalf(grouped[i] as number);
    device.queue.writeBuffer(activations, 0, half);
  } else {
    device.queue.writeBuffer(activations, 0, grouped);
  }

  const dst = device.createTexture({
    size: { width: outW, height: outH },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const params = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([c.width, c.height]));

  device.pushErrorScope('validation');
  const module = device.createShaderModule({
    code: buildPixelShuffleShader({ outChannels: 3, scale: 2, useF16, format: 'rgba8unorm' }),
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  for (const m of (await module.getCompilationInfo()).messages) {
    if (m.type !== 'info') diagnostics.push(`${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
  }

  const bytesPerRow = Math.ceil((outW * 4) / 256) * 256;
  const readback = device.createBuffer({
    size: bytesPerRow * outH,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: activations } },
        { binding: 1, resource: dst.createView() },
        { binding: 2, resource: { buffer: params } },
      ],
    }),
  );
  pass.dispatchWorkgroups(Math.ceil(pixels / 64), 1, 1);
  pass.end();
  encoder.copyTextureToBuffer({ texture: dst }, { buffer: readback, bytesPerRow }, {
    width: outW,
    height: outH,
  });
  device.queue.submit([encoder.finish()]);
  const validation = await device.popErrorScope();
  if (validation) diagnostics.push(`validation ${validation.message}`);

  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(readback.getMappedRange().slice(0));
  readback.unmap();

  const compare = (transposed: boolean): number => {
    let worst = 0;
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        const h = Math.floor(y / 2);
        const w = Math.floor(x / 2);
        const iy = y % 2;
        const ix = x % 2;
        const lane = transposed ? ix * 2 + iy : iy * 2 + ix;
        for (let colour = 0; colour < 3; colour++) {
          const ch = colour * 4 + lane;
          const want = Math.min(1, Math.max(0, planar[ch * pixels + (h * c.width + w)] as number));
          const got = (bytes[y * bytesPerRow + x * 4 + colour] as number) / 255;
          const d = Math.abs(got - want);
          if (d > worst) worst = d;
        }
      }
    }
    return worst;
  };

  const maxAbs = compare(false);
  const transposedError = compare(true);

  for (const b of [activations, params, readback]) b.destroy();
  dst.destroy();

  return {
    ...c,
    diagnostics,
    maxAbsError: maxAbs,
    passed: diagnostics.length === 0 && maxAbs <= tolerance,
    outputPixels: outW * outH,
    // The transposed ordering must be clearly wrong, or the test cannot tell
    // the two apart and proves nothing about sub-pixel placement.
    detectsTransposition: transposedError > tolerance * 4,
    clampChecked: true,
  };
}
