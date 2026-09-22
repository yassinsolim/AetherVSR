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

  it('independently detects corrupt M13 tensors and timing statistics', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { compareStage, timingSummary, verifyBlob, verifySourceIdentity } from './tools/m13/report.mjs';
    import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
    import { tmpdir } from 'node:os';
    import { join } from 'node:path';
    import { createHash } from 'node:crypto';
    const directory = mkdtempSync(join(tmpdir(), 'm13-provenance-'));
    try {
      const path = join(directory, 'binary'); const bytes = Buffer.from('retained binary bytes');
      writeFileSync(path, bytes, { flag: 'wx' });
      const artifact = { path, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
      verifyBlob(artifact);
      writeFileSync(path, Buffer.from('corrupt')); assert.throws(() => verifyBlob(artifact));
      rmSync(path); assert.throws(() => verifyBlob(artifact));
    } finally { rmSync(directory, { recursive: true }); }
    const sourceCommit = 'f370d6ccac90779aa1028589f08904a5e59dcaf2', sourceTree = 'a'.repeat(40);
    const provenance = [{ sourceCommit, sourceTree, webgpuBuild: { sourceCommit, sourceDirty: false } },
      { sourceCommit }, { sourceCommit, sourceTree }, { sourceCommit }];
    verifySourceIdentity(...provenance);
    for (const mutate of [value => value[0].sourceCommit = 'bad', value => value[1].sourceCommit = 'bad',
      value => value[2].sourceCommit = 'bad', value => value[3].sourceCommit = 'bad',
      value => value[0].webgpuBuild.sourceCommit = 'bad', value => value[0].webgpuBuild.sourceDirty = true,
      value => value[2].sourceTree = 'bad']) {
      const wrong = structuredClone(provenance); mutate(wrong); assert.throws(() => verifySourceIdentity(...wrong));
    }
    const valid = [0, 0.25, 0.5, 1];
    assert(compareStage('stem', valid, valid, 2, 2, 1, 0.001).passed);
    for (const value of [NaN, Infinity, -Infinity, 10]) {
      const bad = [...valid]; bad[1] = value;
      const result = compareStage('stem', bad, valid, 2, 2, 1, 0.001);
      assert.equal(result.passed, false); assert.equal(result.failingElements, 1); assert.equal(result.edgeFailures, 1);
    }
    assert.throws(() => compareStage('stem', [], valid, 2, 2, 1, 0.001));
    assert.throws(() => compareStage('stem', valid, valid, Number.MAX_VALUE, 2, 1, 0.001));
    const samples = Array.from({ length: 60 }, (_, index) => ({ startSeconds: index + 1, endSeconds: index + 1.5, milliseconds: 500 }));
    const series = { name: 'synthetic', samples, observationWindowMS: 60000, measuredSamples: 60,
      status: 'measured', statisticsMS: { p50: 500, p95: 500, max: 500 } };
    assert.deepEqual(timingSummary(series).statisticsMS, series.statisticsMS);
    for (const mutate of [value => value.samples.pop(), value => value.samples[0].milliseconds = 0,
      value => value.samples[0].startSeconds = null, value => value.statisticsMS.p95 = 0, value => value.observationWindowMS = 100]) {
      const bad = structuredClone(series); mutate(bad); assert.throws(() => timingSummary(bad));
    }
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));
});
