import { packStemWeights } from './stem.wgsl.js';
import { packUpsampleHeadWeights } from './upsample-head.wgsl.js';

/**
 * On-disk model format for AetherVSR.
 *
 * Deliberately a plain JSON document rather than a binary container. The whole
 * network is a few tens of thousands of parameters, the file is under a
 * megabyte, and being able to read the provenance and licence fields without
 * tooling is worth more here than the bytes saved.
 *
 * Every field that would be needed to reproduce or audit the weights is part of
 * the format, not a README note: the architecture and its version, the exact
 * degradation the model was trained to invert, the normalisation folded into
 * the stem, the corpus and its licence policy, and a digest of the corpus
 * contents.
 */
export interface ModelFile {
  readonly architecture: string;
  readonly architectureVersion: number;
  readonly scale: number;
  readonly inChannels: number;
  readonly outChannels: number;
  readonly features: number;
  readonly depth: number;
  readonly precision: string;
  readonly layers: readonly ModelLayer[];
  readonly normalisation: {
    readonly mean: readonly number[];
    readonly scale: readonly number[];
    readonly range: string;
  };
  readonly degradation: string;
  readonly training: Record<string, unknown>;
  readonly provenance: string;
  readonly parameters: number;
  readonly weights: Record<string, readonly number[]>;
  readonly sha256: string;
}

export interface ModelLayer {
  readonly name: string;
  readonly type: 'conv' | 'nearest';
  readonly kernel?: number;
  readonly in?: number;
  readonly out?: number;
  readonly padding?: number;
  readonly activation?: 'tanh' | 'relu' | 'none';
  readonly scale?: number;
  readonly clamp?: readonly number[];
}

/** The architecture id this loader understands. */
export const SUPPORTED_ARCHITECTURE = 'aethersr-resizeconv';
export const SUPPORTED_ARCHITECTURE_VERSION = 1;

/**
 * Weights already rearranged into the exact orders the WGSL kernels read.
 *
 * This happens **once, at model load**. Repacking per frame would be both
 * pointless — the weights are static — and expensive enough to matter against
 * a 16.67 ms budget.
 */
export interface PackedModel {
  readonly file: ModelFile;
  readonly features: number;
  readonly depth: number;
  /** `[oc][tap]` vec4, alpha lane zeroed. */
  readonly stemWeights: Float32Array<ArrayBuffer>;
  readonly stemBias: Float32Array<ArrayBuffer>;
  /** One entry per body layer, tap-major grouped vec4. */
  readonly bodyWeights: readonly Float32Array<ArrayBuffer>[];
  readonly bodyBias: readonly Float32Array<ArrayBuffer>[];
  /** `[colour][ic/4][tap]` vec4. */
  readonly headWeights: Float32Array<ArrayBuffer>;
  readonly headBias: Float32Array<ArrayBuffer>;
}

function requireArray(
  weights: Record<string, readonly number[]>,
  key: string,
  expected: number,
): Float32Array<ArrayBuffer> {
  const raw = weights[key];
  if (!raw) throw new Error(`model is missing tensor '${key}'`);
  if (raw.length !== expected) {
    throw new Error(`tensor '${key}' has ${raw.length} elements, expected ${expected}`);
  }
  const out = new Float32Array(new ArrayBuffer(expected * 4));
  out.set(raw);
  return out;
}

/**
 * Grouped `[oc][ic/4][tap]` -> tap-major `[ic/4][tap][oc]`, both as vec4.
 *
 * Duplicated from the benchmark's helpers rather than imported, because those
 * live under `src/bench` and the production path must not depend on benchmark
 * code. The two are checked against each other by test.
 */
function packBodyWeights(planar: Float32Array, channels: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(new ArrayBuffer(channels * channels * 9 * 4));
  for (let oc = 0; oc < channels; oc++) {
    for (let ic = 0; ic < channels; ic++) {
      const g = Math.floor(ic / 4);
      const lane = ic % 4;
      for (let tap = 0; tap < 9; tap++) {
        // Destination is tap-major: [(cg*9 + tap) * outC + oc][lane]
        out[((g * 9 + tap) * channels + oc) * 4 + lane] =
          planar[(oc * channels + ic) * 9 + tap] as number;
      }
    }
  }
  return out;
}

/**
 * Validates a model document and rearranges its weights for the GPU.
 *
 * Validation is strict on purpose. A model whose declared shape disagrees with
 * its tensors will otherwise produce a running, plausible, wrong picture, and
 * the failure will be attributed to the network rather than to the file.
 */
export function packModel(file: ModelFile): PackedModel {
  if (file.architecture !== SUPPORTED_ARCHITECTURE) {
    throw new Error(
      `unsupported architecture '${file.architecture}', expected '${SUPPORTED_ARCHITECTURE}'`,
    );
  }
  if (file.architectureVersion !== SUPPORTED_ARCHITECTURE_VERSION) {
    throw new Error(
      `unsupported architecture version ${file.architectureVersion}, ` +
        `expected ${SUPPORTED_ARCHITECTURE_VERSION}`,
    );
  }
  if (file.scale !== 2) throw new Error(`only scale 2 is implemented, got ${file.scale}`);
  if (file.inChannels !== 3 || file.outChannels !== 3) {
    throw new Error(`expected RGB in and out, got ${file.inChannels} -> ${file.outChannels}`);
  }
  const c = file.features;
  if (!Number.isInteger(c) || c % 4 !== 0) {
    throw new Error(`feature width must be a positive multiple of 4, got ${c}`);
  }
  if (!Number.isInteger(file.depth) || file.depth < 0) {
    throw new Error(`depth must be a non-negative integer, got ${file.depth}`);
  }
  const mean = file.normalisation.mean;
  const scale = file.normalisation.scale;
  if (mean.length < 3 || scale.length < 3) {
    throw new Error('normalisation must supply three mean and three scale values');
  }

  const stemPlanar = requireArray(file.weights, 'stem.weight', c * 3 * 25);
  const stemBias = requireArray(file.weights, 'stem.bias', c);
  const bodyWeights: Float32Array<ArrayBuffer>[] = [];
  const bodyBias: Float32Array<ArrayBuffer>[] = [];
  for (let i = 0; i < file.depth; i++) {
    bodyWeights.push(packBodyWeights(requireArray(file.weights, `body.${i}.weight`, c * c * 9), c));
    bodyBias.push(requireArray(file.weights, `body.${i}.bias`, c));
  }
  const headPlanar = requireArray(file.weights, 'head.weight', 3 * c * 9);
  const headBias = requireArray(file.weights, 'head.bias', 3);

  return {
    file,
    features: c,
    depth: file.depth,
    stemWeights: packStemWeights(stemPlanar, c, 5),
    stemBias,
    bodyWeights,
    bodyBias,
    headWeights: packUpsampleHeadWeights(headPlanar, c),
    headBias,
  };
}

/** Fetches and packs a model document. */
export async function loadModel(url: string): Promise<PackedModel> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`model fetch failed: ${response.status} ${response.statusText}`);
  return packModel((await response.json()) as ModelFile);
}

/** Persistent GPU bytes the packed weights will occupy at a given precision. */
export function weightBytes(model: PackedModel, useF16: boolean): number {
  const bpe = useF16 ? 2 : 4;
  let total = model.stemWeights.length + model.stemBias.length;
  for (const w of model.bodyWeights) total += w.length;
  for (const b of model.bodyBias) total += b.length;
  total += model.headWeights.length + model.headBias.length;
  return total * bpe;
}
