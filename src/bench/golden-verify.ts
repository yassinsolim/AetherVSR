import { buildStemShader } from '../core/neural/stem.wgsl.js';
import { buildBlockedConvShader } from '../core/neural/conv.wgsl.js';
import { buildUpsampleHeadShader } from '../core/neural/upsample-head.wgsl.js';
import { packModel, type ModelFile, type PackedModel } from '../core/neural/model.js';
import { floatToHalf } from './conv-bench.js';

/** The reference activations exported by `tools/export-golden.py`. */
export interface GoldenVectors {
  readonly model: string;
  readonly modelSha256: string;
  readonly features: number;
  readonly depth: number;
  readonly width: number;
  readonly height: number;
  readonly input: readonly number[];
  readonly stages: Record<string, readonly number[]>;
  readonly output: readonly number[];
}

export interface StageComparison {
  readonly stage: string;
  readonly maxAbsError: number;
  readonly meanAbsError: number;
  readonly referenceRange: readonly [number, number];
  readonly elements: number;
  /** Present only when a non-finite value was seen; its absence means none. */
  readonly nonFinite?: number;
  readonly passed: boolean;
}

export interface GoldenResult {
  readonly useF16: boolean;
  readonly tolerance: number;
  readonly stages: readonly StageComparison[];
  readonly output: StageComparison;
  readonly passed: boolean;
  readonly diagnostics: readonly string[];
}

/**
 * Runs the WGSL graph on the golden input and compares every stage against the
 * PyTorch reference.
 *
 * Stage-by-stage rather than end-to-end on purpose. A whole-network comparison
 * tells you the picture is wrong; this tells you *which layer* first diverged,
 * which is the difference between a five-minute fix and an afternoon of
 * bisecting a shader by eye. It also catches the case where two errors cancel.
 *
 * The input is fed as an ordinary texture rather than through ingest, because
 * what is under test here is the network, not the decoder bridge - that has its
 * own verification against real decoder output.
 */
export async function verifyGolden(
  device: GPUDevice,
  modelFile: ModelFile,
  golden: GoldenVectors,
  useF16: boolean,
  tolerance = useF16 ? 5e-2 : 1e-3,
): Promise<GoldenResult> {
  const model = packModel(modelFile);
  const diagnostics: string[] = [];
  const { width: W, height: H } = golden;
  const c = model.features;
  const pixels = W * H;
  const bpe = useF16 ? 2 : 4;

  if (model.features !== golden.features || model.depth !== golden.depth) {
    throw new Error(
      `golden vectors are for C${golden.features}D${golden.depth}, model is ` +
        `C${model.features}D${model.depth}`,
    );
  }

  device.pushErrorScope('validation');

  // Input as an 8-bit texture, matching what ingest hands the stem. The golden
  // input is quantised the same way on the reference side by construction:
  // export writes float32, so we quantise here and compare against a reference
  // recomputed from the quantised values would be circular. Instead the input
  // texture is float32 to keep the comparison about the graph, not about 8-bit.
  const srcTex = device.createTexture({
    size: { width: W, height: H },
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const texels = new Float32Array(pixels * 4);
  for (let p = 0; p < pixels; p++) {
    for (let ch = 0; ch < 3; ch++) texels[p * 4 + ch] = golden.input[ch * pixels + p] as number;
    texels[p * 4 + 3] = 1;
  }
  device.queue.writeTexture({ texture: srcTex }, texels, { bytesPerRow: W * 16 }, {
    width: W,
    height: H,
  });

  const upload = (data: Float32Array<ArrayBuffer>): GPUBuffer => {
    const buf = device.createBuffer({
      size: Math.max(16, Math.ceil((data.length * bpe) / 16) * 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (useF16) {
      const half = new Uint16Array(data.length + (data.length % 2));
      for (let i = 0; i < data.length; i++) half[i] = floatToHalf(data[i] as number);
      device.queue.writeBuffer(buf, 0, half);
    } else {
      device.queue.writeBuffer(buf, 0, data);
    }
    return buf;
  };

  const activationBytes = Math.ceil((pixels * c * bpe) / 16) * 16;
  const makeActivation = (): GPUBuffer =>
    device.createBuffer({
      size: activationBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  const ping = makeActivation();
  const pong = makeActivation();

  const stemW = upload(model.stemWeights);
  const stemB = upload(model.stemBias);
  const bodyW = model.bodyWeights.map(upload);
  const bodyB = model.bodyBias.map(upload);
  const headW = upload(model.headWeights);
  const headB = upload(model.headBias);

  const stemParams = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const sp = new ArrayBuffer(48);
  new Uint32Array(sp, 0, 2).set([W, H]);
  new Float32Array(sp, 16, 4).set([0, 0, 0, 0]);
  new Float32Array(sp, 32, 4).set([1, 1, 1, 1]);
  device.queue.writeBuffer(stemParams, 0, sp);
  const dims = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, new Uint32Array([W, H]));

  const outTex = device.createTexture({
    size: { width: W * 2, height: H * 2 },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  const stemPipe = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: buildStemShader({
          outChannels: c,
          blockX: 2,
          blockY: 2,
          tileX: 8,
          tileY: 8,
          useF16,
          activation: 'tanh',
          kernel: 5,
        }),
      }),
      entryPoint: 'main',
    },
  });
  const bodyPipe = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: buildBlockedConvShader({
          inChannels: c,
          outChannels: c,
          tileX: 8,
          tileY: 4,
          blockX: 2,
          blockY: 2,
          outBlock: c,
          activation: 'tanh',
          useF16,
          residual: false,
          weightLayout: 'tap-major',
          packedOutput: true,
        }),
      }),
      entryPoint: 'main',
    },
  });
  const headPipe = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: buildUpsampleHeadShader({
          inChannels: c,
          scale: 2,
          useF16,
          format: 'rgba8unorm',
          blockX: 2,
          blockY: 4,
          tileX: 8,
          tileY: 4,
          globalResidual: true,
        }),
      }),
      entryPoint: 'main',
    },
  });

  // One readback buffer per stage, so every intermediate can be compared.
  const stageNames = ['stem', ...Array.from({ length: model.depth }, (_, i) => `body.${i}`)];
  const stageReadbacks = stageNames.map(() =>
    device.createBuffer({ size: activationBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
  );
  const outBytesPerRow = Math.ceil((W * 2 * 4) / 256) * 256;
  const outReadback = device.createBuffer({
    size: outBytesPerRow * H * 2,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(stemPipe);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: stemPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: { buffer: stemW } },
        { binding: 2, resource: { buffer: stemB } },
        { binding: 3, resource: { buffer: ping } },
        { binding: 4, resource: { buffer: stemParams } },
      ],
    }),
  );
  pass.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16), 1);
  pass.end();
  encoder.copyBufferToBuffer(ping, 0, stageReadbacks[0] as GPUBuffer, 0, activationBytes);

  for (let i = 0; i < model.depth; i++) {
    const src = i % 2 === 0 ? ping : pong;
    const dst = i % 2 === 0 ? pong : ping;
    const bodyPass = encoder.beginComputePass();
    bodyPass.setPipeline(bodyPipe);
    bodyPass.setBindGroup(
      0,
      device.createBindGroup({
        layout: bodyPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: bodyW[i] as GPUBuffer } },
          { binding: 2, resource: { buffer: bodyB[i] as GPUBuffer } },
          { binding: 3, resource: { buffer: dst } },
          { binding: 4, resource: { buffer: dims } },
        ],
      }),
    );
    bodyPass.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 8), 1);
    bodyPass.end();
    encoder.copyBufferToBuffer(dst, 0, stageReadbacks[i + 1] as GPUBuffer, 0, activationBytes);
  }

  const finalBuffer = model.depth % 2 === 0 ? ping : pong;
  const headPass = encoder.beginComputePass();
  headPass.setPipeline(headPipe);
  headPass.setBindGroup(
    0,
    device.createBindGroup({
      layout: headPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: finalBuffer } },
        { binding: 1, resource: { buffer: headW } },
        { binding: 2, resource: { buffer: headB } },
        { binding: 3, resource: outTex.createView() },
        { binding: 4, resource: { buffer: dims } },
        // The head's global residual reads the source image directly.
        { binding: 5, resource: srcTex.createView() },
      ],
    }),
  );
  headPass.dispatchWorkgroups(Math.ceil((W * 2) / 16), Math.ceil((H * 2) / 16), 1);
  headPass.end();
  encoder.copyTextureToBuffer({ texture: outTex }, { buffer: outReadback, bytesPerRow: outBytesPerRow }, {
    width: W * 2,
    height: H * 2,
  });
  device.queue.submit([encoder.finish()]);

  const validation = await device.popErrorScope();
  if (validation) diagnostics.push(`validation ${validation.message}`);

  const comparisons: StageComparison[] = [];
  for (let i = 0; i < stageNames.length; i++) {
    const name = stageNames[i] as string;
    const buf = stageReadbacks[i] as GPUBuffer;
    await buf.mapAsync(GPUMapMode.READ);
    const raw = buf.getMappedRange().slice(0);
    buf.unmap();
    const elements = pixels * c;
    const grouped = useF16
      ? Float32Array.from(new Uint16Array(raw).subarray(0, elements), halfToFloat)
      : new Float32Array(raw).subarray(0, elements);
    const actual = new Float32Array(elements);
    for (let ch = 0; ch < c; ch++) {
      const g = Math.floor(ch / 4);
      const lane = ch % 4;
      for (let p = 0; p < pixels; p++) actual[ch * pixels + p] = grouped[(g * pixels + p) * 4 + lane] as number;
    }
    comparisons.push(compare(name, actual, golden.stages[name] ?? [], tolerance));
  }

  await outReadback.mapAsync(GPUMapMode.READ);
  const outBytes = new Uint8Array(outReadback.getMappedRange().slice(0));
  outReadback.unmap();
  const OW = W * 2;
  const OH = H * 2;
  const actualOut = new Float32Array(OW * OH * 3);
  for (let y = 0; y < OH; y++) {
    for (let x = 0; x < OW; x++) {
      for (let ch = 0; ch < 3; ch++) {
        actualOut[(y * OW + x) * 3 + ch] = (outBytes[y * outBytesPerRow + x * 4 + ch] as number) / 255;
      }
    }
  }
  // The output is quantised to 8 bits by the storage texture, so it gets its
  // own tolerance: anything tighter would be measuring the texture format.
  const outputCmp = compare('output', actualOut, golden.output, Math.max(tolerance, 1.5 / 255));

  for (const b of [
    ping,
    pong,
    stemW,
    stemB,
    headW,
    headB,
    stemParams,
    dims,
    outReadback,
    ...bodyW,
    ...bodyB,
    ...stageReadbacks,
  ]) {
    b.destroy();
  }
  srcTex.destroy();
  outTex.destroy();

  return {
    useF16,
    tolerance,
    stages: comparisons,
    output: outputCmp,
    passed: diagnostics.length === 0 && comparisons.every((s) => s.passed) && outputCmp.passed,
    diagnostics,
  };
}

function compare(
  stage: string,
  actual: Float32Array,
  expected: readonly number[],
  tolerance: number,
): StageComparison {
  // A truncated reference must fail loudly. Comparing the overlap silently
  // turns a corrupt or partially-written vector file into a pass over whatever
  // prefix happens to agree - which is precisely the regression this verifier
  // exists to catch.
  if (actual.length !== expected.length || expected.length === 0) {
    return {
      stage,
      maxAbsError: Number.POSITIVE_INFINITY,
      meanAbsError: Number.POSITIVE_INFINITY,
      referenceRange: [Number.NaN, Number.NaN],
      elements: 0,
      passed: false,
    };
  }
  const n = expected.length;
  let maxAbs = 0;
  let nonFinite = 0;
  let sumAbs = 0;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const e = expected[i] as number;
    const a = actual[i] as number;
    const d = Math.abs(a - e);
    // NaN fails every comparison, so `d > maxAbs` silently skips it and a
    // NaN-poisoned activation used to score a flawless match. Count them.
    if (!Number.isFinite(e) || !Number.isFinite(a) || !Number.isFinite(d)) nonFinite++;
    if (d > maxAbs) maxAbs = d;
    sumAbs += d;
    if (e < lo) lo = e;
    if (e > hi) hi = e;
  }
  // A stage whose reference is constant cannot discriminate anything; say so
  // rather than reporting a flawless match.
  const degenerate = !(hi > lo);
  return {
    stage,
    maxAbsError: maxAbs,
    meanAbsError: n > 0 ? sumAbs / n : Number.NaN,
    ...(nonFinite > 0 ? { nonFinite } : {}),
    referenceRange: [lo, hi],
    elements: n,
    passed: n > 0 && nonFinite === 0 && !degenerate && maxAbs <= tolerance,
  };
}

function halfToFloat(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

export type { PackedModel };
