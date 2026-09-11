import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

/**
 * Binds the committed golden vectors to the model they were exported from.
 *
 * `verifyGolden` used to check only that channels and depth agreed. Every model
 * this project has ever shipped is C16D2, so vectors exported from one set of
 * weights would happily "verify" a completely different set and report a WebGPU
 * parity that had never been tested. That is exactly what had happened: the
 * committed vectors carried modelSha256 7c5922... while the deployed model was
 * 9154d949..., and nothing failed.
 *
 * Milestone 7's whole claim is that a fused reparameterized model is
 * numerically identical to the deployed graph, and that claim is only worth
 * anything if the reference it is checked against belongs to the model under
 * test. This runs without a GPU so it fails in CI rather than in a browser.
 */

type Golden = {
  model: string;
  modelSha256: string;
  features: number;
  depth: number;
  stages: Record<string, number[]>;
  input: number[];
  output: number[];
};
type Model = { sha256: string; features: number; depth: number; parameters: number };

function load<T>(rel: string): T {
  return JSON.parse(readFileSync(new URL(rel, import.meta.url), 'utf8')) as T;
}

describe('golden vectors', () => {
  const golden = load<Golden>('../public/models/golden-c16d2.json');
  const model = load<Model>('../public/models/aethersr-c16d2.json');

  it('were exported from the model that is actually shipped', () => {
    expect(golden.modelSha256).toBe(model.sha256);
  });

  it('describe the shipped architecture', () => {
    expect(golden.features).toBe(model.features);
    expect(golden.depth).toBe(model.depth);
    expect(model.parameters).toBe(6291);
  });

  it('cover every stage the runtime executes', () => {
    expect(Object.keys(golden.stages).sort()).toEqual(['body.0', 'body.1', 'stem']);
  });

  it('carry a 2x output for the recorded input', () => {
    // 3 channels at 24x16 in, 3 channels at 48x32 out: the scale-2 contract.
    expect(golden.input).toHaveLength(3 * 24 * 16);
    expect(golden.output).toHaveLength(3 * 48 * 32);
  });
});
