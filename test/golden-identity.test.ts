import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

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

  it('retains M13 initialization failures and rejects incomplete paired evidence', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
    import { tmpdir } from 'node:os';
    import { join } from 'node:path';
    import { recordAttempt, validateWebGPU } from './tools/m13/qualify.mjs';
    const directory = mkdtempSync(join(tmpdir(), 'm13-initialization-'));
    try {
      const header = { sourceCommit: 'a'.repeat(40), id: 'failure-control' };
      assert.throws(() => recordAttempt(directory, header, () => { throw new Error('injected probe failure'); }), /injected/);
      const bytes = readFileSync(join(directory, 'failure.json'));
      const failed = JSON.parse(bytes);
      assert.equal(failed.sourceCommit, header.sourceCommit);
      assert.equal(failed.phase, 'initialization'); assert.equal(failed.outcome, 'FAIL');
      assert.equal(failed.cleanup, 'no app launched');
      assert.throws(() => recordAttempt(directory, header, () => { throw new Error('second failure'); }), /EEXIST/);
      assert.deepEqual(readFileSync(join(directory, 'failure.json')), bytes);
      const runs = ['f32', 'f16'].map(precision => ({ schema: 'aethervsr.m13.webgpu-golden/1', precision, outcome: 'PASS',
        summary: { passed: true }, finalFloat: { passed: true }, rgbaAgreement: { passed: true },
        stages: ['stem', 'body.0', 'body.1', 'final'].map(name => ({ name })) }));
      const value = { schema: 'aethervsr.m13.webgpu-capture/1', outcome: 'PASS', adapter: { fallbackAdapter: false }, errors: [], runs };
      validateWebGPU(value);
      for (const mutate of [value => value.runs.pop(), value => value.runs.reverse(), value => value.adapter.fallbackAdapter = true,
        value => value.errors.push('GPU error'), value => value.runs[0].stages.pop(), value => value.runs[0].rgbaAgreement.passed = false]) {
        const wrong = structuredClone(value); mutate(wrong); assert.throws(() => validateWebGPU(wrong));
      }
    } finally { rmSync(directory, { recursive: true }); }
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));
});
