import type { Size, Upscaler } from '../core/types.js';
import { BaselineScaler } from '../core/upscale/baseline-scaler.js';
import type { BaselineFilter } from '../core/upscale/baseline.wgsl.js';
import { boxDownsample2x, generateReference, score, type QualityScores } from './quality.js';

/**
 * Runs the deterministic quality evaluation against the real GPU upscalers.
 *
 * The reconstruction is produced by the same `BaselineScaler` the Milestone 1
 * harness uses, on the same device, so what is scored is the shipped scaler
 * rather than a JavaScript reimplementation of it.
 *
 * This reads pixels back to the CPU. That is forbidden in the frame hot path
 * and entirely appropriate here: it is an offline evaluation.
 */
export async function evaluateScalers(
  device: GPUDevice,
  filters: readonly BaselineFilter[],
  hr: Size = { width: 2560, height: 1440 },
  extra: readonly { readonly label: string; readonly upscaler: Upscaler }[] = [],
): Promise<QualityScores[]> {
  const reference = generateReference(hr.width, hr.height);
  return evaluateAgainstReference(device, reference, filters, extra);
}

/**
 * Scores every scaler against one high-resolution reference.
 *
 * The reference is downsampled by an exact 2x box filter - the same degradation
 * the model was trained to invert, and the same one `quality.ts` has always
 * used - so a scaler is being asked to undo a known operation rather than to
 * guess at an unknown one.
 */
export async function evaluateAgainstReference(
  device: GPUDevice,
  reference: ImageData,
  filters: readonly BaselineFilter[],
  extra: readonly { readonly label: string; readonly upscaler: Upscaler }[] = [],
): Promise<QualityScores[]> {
  const hr: Size = { width: reference.width, height: reference.height };
  const lowRes = boxDownsample2x(reference);
  const lrBitmap = await createImageBitmap(lowRes);

  const format: GPUTextureFormat = 'rgba8unorm';
  const lrTexture = device.createTexture({
    label: 'quality:lr',
    size: { width: lowRes.width, height: lowRes.height },
    format,
    usage:
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: lrBitmap },
    { texture: lrTexture },
    { width: lowRes.width, height: lowRes.height },
  );
  const lrView = lrTexture.createView();

  const target = device.createTexture({
    label: 'quality:sr',
    size: { width: hr.width, height: hr.height },
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const targetView = target.createView();

  // Readback rows must be padded to 256 bytes, per WebGPU's copy alignment.
  const unpaddedBytesPerRow = hr.width * 4;
  const bytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;
  const readback = device.createBuffer({
    size: bytesPerRow * hr.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const results: QualityScores[] = [];

  const runOne = async (label: string, scaler: Upscaler, ownsScaler: boolean): Promise<void> => {
    scaler.configure({
      device,
      source: { width: lowRes.width, height: lowRes.height },
      target: hr,
      targetFormat: format,
      sourceKind: 'sampled',
    });
    const encoder = device.createCommandEncoder();
    scaler.encode({ encoder, frame: { kind: 'sampled', view: lrView }, target: targetView, timing: null });
    encoder.copyTextureToBuffer(
      { texture: target },
      { buffer: readback, bytesPerRow, rowsPerImage: hr.height },
      { width: hr.width, height: hr.height },
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const tight = new Uint8Array(unpaddedBytesPerRow * hr.height);
    for (let y = 0; y < hr.height; y++) {
      tight.set(
        padded.subarray(y * bytesPerRow, y * bytesPerRow + unpaddedBytesPerRow),
        y * unpaddedBytesPerRow,
      );
    }
    results.push(score(label, reference, tight));
    if (ownsScaler) scaler.destroy();
  };

  for (const filter of filters) {
    const scaler = new BaselineScaler(filter);
    scaler.configure({
      device,
      source: { width: lowRes.width, height: lowRes.height },
      target: hr,
      targetFormat: format,
      sourceKind: 'sampled',
    });
    scaler.destroy();
    await runOne(`baseline-${filter}`, new BaselineScaler(filter), true);
  }

  for (const item of extra) {
    await runOne(item.label, item.upscaler, false);
  }

  // A control: the LR image itself, nearest-neighbour expanded on the CPU.
  // Any GPU scaler that cannot beat this is broken, so it anchors the scale.
  results.push(score('control-nearest', reference, nearestExpand(lowRes, hr)));

  lrBitmap.close();
  lrTexture.destroy();
  target.destroy();
  readback.destroy();
  return results;
}

/** Nearest-neighbour 2x expansion, used only as a lower-bound control. */
function nearestExpand(lr: ImageData, hr: Size): Uint8Array {
  const out = new Uint8Array(hr.width * hr.height * 4);
  for (let y = 0; y < hr.height; y++) {
    const sy = y >> 1;
    for (let x = 0; x < hr.width; x++) {
      const sx = x >> 1;
      const s = (sy * lr.width + sx) * 4;
      const d = (y * hr.width + x) * 4;
      out[d] = lr.data[s] as number;
      out[d + 1] = lr.data[s + 1] as number;
      out[d + 2] = lr.data[s + 2] as number;
      out[d + 3] = 255;
    }
  }
  return out;
}
