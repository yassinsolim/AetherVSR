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

  it('keeps the Phase-1.5 candidate selection and binding thresholds fixed', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { CANDIDATES, selectCandidate, classify, validateInterval, matchConfiguration } from './tools/m13/optimization-report.mjs';
    import { boundedProcess, nextAttempt, failureClassification, pinObservationFailed } from './tools/m13/optimize.mjs';
    const run = (p50, p95) => ({ outcome: 'PASS', validSamples: 60, thermalStateBefore: 0, statisticsMS: { p50, p95, max: p95 + 1 } });
    assert.equal(classify([run(8, 10), run(8, 10)], true), 'METAL REALTIME BACKEND QUALIFIED');
    assert.equal(classify([run(8.0001, 10), run(8, 10)], true), 'METAL REALTIME BACKEND PARTIAL');
    assert.equal(classify([run(8, 10.0001), run(8, 16.6699)], true), 'METAL REALTIME BACKEND PARTIAL');
    for (const pair of [[run(8, 16.67), run(8, 10)], [run(8, 10)], [run(8, 10), { ...run(8, 10), validSamples: 59 }],
      [run(8, 10), { ...run(8, 10), thermalStateBefore: 1 }], [run(8, 10), run(NaN, 10)]]) {
      assert.equal(classify(pair, true), 'METAL REALTIME BACKEND NOT QUALIFIED');
    }
    assert.equal(classify([run(8, 10), run(8, 10)], false), 'METAL REALTIME BACKEND NOT QUALIFIED');
    const rows = CANDIDATES.map(candidate => ({ candidate, outcome: 'PASS', validSamples: 20,
      statisticsMS: { p50: 5, p95: 6, max: 7 }, configuration: { configuredBufferBytes: 100 } }));
    assert.equal(selectCandidate(rows), 'B'); rows[1].configuration.configuredBufferBytes = 99;
    assert.equal(selectCandidate(rows), 'C'); rows[2].statisticsMS.p50 = 4;
    assert.equal(selectCandidate(rows), 'D'); rows[3].statisticsMS.p95 = 5;
    assert.equal(selectCandidate(rows), 'E'); rows[3].outcome = 'FAIL';
    assert.equal(selectCandidate(rows), 'D'); assert.throws(() => selectCandidate(rows.slice(1)));
    assert.equal(selectCandidate(rows.map(row => ({ ...row, outcome: 'FAIL' }))), null);
    assert.equal(validateInterval({ startSeconds: 1, endSeconds: 1.5, milliseconds: 500 }), 500);
    for (const sample of [{ startSeconds: 0, endSeconds: 1, milliseconds: 1000 }, { startSeconds: 1, endSeconds: 1.5, milliseconds: 0 },
      { startSeconds: 1, endSeconds: Infinity, milliseconds: 500 }, { startSeconds: 2, endSeconds: 1, milliseconds: 1 }]) assert.throws(() => validateInterval(sample));
    assert.throws(() => matchConfiguration({ candidate: 'B' }, { candidate: 'F' }, true));
    assert.equal(nextAttempt([]), 1);
    assert.equal(nextAttempt([{ outcome: 'NOT_STARTED', began: false, thermalState: 1 }]), 2);
    for (const value of [{ outcome: 'PASS' }, { outcome: 'FAIL' }, { outcome: 'NOT_STARTED', began: true, thermalState: 1 },
      { outcome: 'NOT_STARTED', began: false, thermalState: 0 }]) assert.throws(() => nextAttempt([value]));
    const processFailure = { exitCode: 1, expired: false, signal: null };
    assert.deepEqual(failureClassification(processFailure, { outcome: 'FAIL', error: 'numerical mismatch' }, []), { outcome: 'FAIL', unsafe: false });
    assert.equal(failureClassification({ ...processFailure, expired: true, signal: 'SIGKILL' }, null, []).unsafe, true);
    assert.equal(failureClassification({ exitCode: 0, expired: false, signal: null }, null, [], true).unsafe, true);
    const deferred = { outcome: 'NOT_STARTED', began: false, thermalState: 1 };
    assert.deepEqual(failureClassification(processFailure, deferred, [{ phase: 'preflight', thermalState: 1 }]), { outcome: 'NOT_STARTED', unsafe: false });
    assert.throws(() => failureClassification(processFailure, { ...deferred, thermalState: 0 }, [{ phase: 'preflight', thermalState: 1 }]));
    assert.throws(() => failureClassification({ exitCode: 0, expired: false, signal: null }, null, []));
    const pinned = [{ path: 'source', bytes: 1, sha256: 'original' }];
    assert.equal(pinObservationFailed({ status: '', files: pinned }, pinned), false);
    const dirtyDuringRun = { status: ' M source', files: [{ path: 'source', bytes: 1, sha256: 'changed' }] };
    assert.equal(pinObservationFailed(dirtyDuringRun, pinned), true);
    assert.deepEqual(failureClassification({ exitCode: 0, expired: false, signal: null }, null, [], pinObservationFailed(dirtyDuringRun, pinned)),
      { outcome: 'FAIL', unsafe: true });
    assert.equal(pinObservationFailed({ status: '', files: [{ path: 'source', missing: true }] }, pinned), true);
    const normal = await boundedProcess(process.execPath, ['-e', 'process.stdout.write("done")'], { deadlineMS: 2000 });
    assert.equal(normal.exitCode, 0); assert.equal(normal.expired, false); assert.equal(normal.stdout, 'done');
    const blocked = await boundedProcess(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { deadlineMS: 100 });
    assert.equal(blocked.expired, true); assert.equal(blocked.signal, 'SIGKILL');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));

  it('rejects incomplete optimized manifests and corrupted timing progress', () => execFileSync(process.execPath, ['--input-type=module', '-e', String.raw`
    import assert from 'node:assert/strict';
    import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
    import { join } from 'node:path'; import { tmpdir } from 'node:os';
    import { CANDIDATES, validateInventory, inspectTiming, ref } from './tools/m13/optimization-report.mjs';
    const directory = mkdtempSync(join(tmpdir(), 'm13-timing-control-'));
    try {
      const json = (name, value) => writeFileSync(join(directory, name), JSON.stringify(value));
      for (const precision of ['f32', 'f16']) json(precision + '.json', { precision });
      const manifest = { candidates: CANDIDATES, sourceFiles: [{ path: 'source', sha256: 'a', bytes: 1 }], resources: [],
        qualifications: CANDIDATES.flatMap(candidate => ['f32', 'f16'].map(precision => ({ candidate, precision }))),
        references: ['f32', 'f16'].map(precision => ref(join(directory, precision + '.json'))) };
      validateInventory(manifest, manifest.sourceFiles, []);
      for (const mutate of [value => value.qualifications.pop(), value => value.qualifications = [], value => value.sourceFiles = [],
        value => value.references.reverse(), value => value.qualifications.reverse()]) {
        const bad = structuredClone(manifest); mutate(bad); assert.throws(() => validateInventory(bad, manifest.sourceFiles, []));
      }
      json('qualification.json', {});
      const interval = index => ({ startSeconds: index + 1, endSeconds: index + 1.5, milliseconds: 500 });
      const warmups = Array.from({ length: 5 }, (_, index) => ({ phase: 'warmup', index, thermalState: 0, interval: interval(index) }));
      const samples = Array.from({ length: 20 }, (_, index) => ({ phase: 'sample', index, thermalState: 0, interval: interval(index + 5) }));
      const events = [{ phase: 'preflight', index: 0, thermalState: 0 }, { phase: 'begin', index: 0, thermalState: 0 }, ...warmups, ...samples];
      const configuration = { candidate: 'F', precision: 'f16', diagnostic: false, extent: { width: 1280, height: 720 } };
      const value = { schema: 'aethervsr.m13.phase15-timing/1', outcome: 'PASS', configuration,
        inputFloat32Sha256: 'e43b54d00e708a81b1e537e5345e72195300b5ab3dde75f2e05204d273953c19', qualificationSha256: ref(join(directory, 'qualification.json')).sha256,
        timing: { mode: 'explore', warmups: 5, requestedSamples: 20, thermalStateBefore: 0, thermalStateAfter: 0,
          wholeGraph: { samples: samples.map(value => value.interval), measuredSamples: 20, status: 'measured', observationWindowMS: 20000,
            statisticsMS: { p50: 500, p95: 500, max: 500 } } },
        postAlphaOpaque: true, postDiagnosticRGBAEqual: true, postRGBABytes: 1280 * 720 * 16,
        postStages: ['stem', 'body.0', 'body.1', 'final'].map(stage => ({ stage, elements: stage === 'final' ? 11059200 : 14745600, nonFinite: 0, outsideRange: 0 })) };
      const progress = rows => writeFileSync(join(directory, 'progress.jsonl'), rows.map(value => JSON.stringify(value)).join('\n') + '\n');
      json('timing.json', value); progress(events); assert.equal(inspectTiming(join(directory, 'timing.json'), 'explore').outcome, 'PASS');
      for (const mutate of [rows => rows[2].interval.startSeconds = 0, rows => rows[2].interval.milliseconds = 1,
        rows => rows[2].interval.endSeconds = null, rows => rows.pop(), rows => rows[0].thermalState = 1,
        rows => rows[2].interval = interval(20)]) {
        const bad = structuredClone(events); mutate(bad); progress(bad); assert.throws(() => inspectTiming(join(directory, 'timing.json'), 'explore'));
      }
      progress(events); const wrong = structuredClone(value); wrong.timing.wholeGraph.statisticsMS.p95 = 1;
      json('timing.json', wrong); assert.throws(() => inspectTiming(join(directory, 'timing.json'), 'explore'));
    } finally { rmSync(directory, { recursive: true }); }
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));
});
