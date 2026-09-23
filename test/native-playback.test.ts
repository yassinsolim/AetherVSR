import { execFileSync } from 'node:child_process';
import { it } from 'vitest';

it('rejects invalid native cadence, identity, timing, mode and cleanup evidence', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
  import assert from 'node:assert/strict';
  import { analyze, zeroResources, prerequisiteIDs, CASES, studyStatus, checkGeometry } from './tools/m13/playback.mjs';
  const zero = { activeOutputs: 0, configuredSlots: 0, displayLinks: 0, occupiedSlots: 0, pixelBuffers: 0, presentationCommands: 0,
    processingSlots: 0, textureCaches: 0, textureWrappers: 0, retainedOwners: { decodedSampleOwners: 0, leasedPixelBuffers: 0, liveTextureWrappers: 0 } };
  assert(zeroResources(zero)); for (const value of [{}, [], { occupiedSlots: 0 }, { ...zero, textureCaches: 1 }]) assert(!zeroResources(value));
  const live = { ...zero, activeOutputs: 1, configuredSlots: 2, displayLinks: 1, textureCaches: 1 };
  const result = { outcome: 'PASS', measurementFinished: true, error: null, diagnosticFailure: null, cleanup: zero };
  const rows = [{ kind: 'invalidate', generation: 1, host: 9 }, { kind: 'measurement-start', host: 15 }, { kind: 'measurement-end', host: 75 }];
  for (let index = 0; index < 3900; index++) {
    const host = 10 + index / 60, start = host - 0.01, ingestEnd = start + 0.001, networkStart = ingestEnd + 0.0001,
      networkEnd = networkStart + 0.004, presentStart = networkEnd + 0.0002, presentEnd = presentStart + 0.001;
    const gpuIngestMS = (ingestEnd - start) * 1000, gpuNeuralMS = (networkEnd - networkStart) * 1000, gpuPresentationMS = (presentEnd - presentStart) * 1000;
    rows.push({ kind: 'frame', host: host + 0.001, opportunityHost: host, pts: index / 60, playerTime: index / 60 + 0.02, ageMS: 20,
      sequence: index + 1, frameGeneration: 1, generation: 1, itemIdentity: 'item-1', itemMatched: true,
      sourceFile: 'aethervsr-testclip-720p60-h264.mp4', neural: true, rate: 1, timeControl: 2,
      invalidOutputFlags: 0, outputValidation: 'hidden-final-gpu',
      submissionHost: start - 0.001, completionHost: networkEnd + 0.0001, presentationCompleteHost: host + 0.001,
      gpuIngestStart: start, gpuIngestEnd: ingestEnd, gpuNetworkStart: networkStart, gpuNetworkEnd: networkEnd,
      gpuPresentationStart: presentStart, gpuPresentationEnd: presentEnd, gpuIngestMS, gpuNeuralMS, gpuPresentationMS,
      gpuFramePathMS: gpuIngestMS + gpuNeuralMS + gpuPresentationMS, gpuProcessingSpanMS: (networkEnd - start) * 1000, resources: live });
  }
  for (let host = 15; host < 75; host++) rows.push({ kind: 'heartbeat', host, resources: live });
  assert.equal(analyze(rows, result, 60, 60, 'neural').outcome, 'PASS');
  for (const mutate of [value => value[400].pts = value[399].pts, value => value[400].ageMS = 0,
    value => value[400].itemIdentity = 'other', value => value[400].gpuIngestStart = 0,
    value => value[400].neural = false, value => value[400].resources = {}, value => value[400].frameGeneration = 99,
    value => value[400].invalidOutputFlags = 1, value => delete value[400].invalidOutputFlags,
    value => value.push({ kind: 'invalidate', host: 20, generation: 2, reason: 'mode' })]) {
    const bad = structuredClone(rows); mutate(bad); assert.equal(analyze(bad, result, 60, 60, 'neural').outcome, 'FAIL');
  }
  assert.equal(analyze(rows, { ...result, cleanup: {} }, 60, 60, 'neural').outcome, 'FAIL');
  assert.equal(analyze(rows.filter(row => row.kind !== 'frame' || row.sequence % 2 === 0), result, 60, 60, 'neural').outcome, 'FAIL');
  assert(prerequisiteIDs('neural30').includes('baseline30'));
  assert(prerequisiteIDs('neural60b').includes('neural60a'));
  assert(prerequisiteIDs('soak').includes('neural60c'));
  assert.throws(() => prerequisiteIDs('soak-retry'));
  const outcomes = Object.fromEntries(Object.keys(CASES).map(id => [id, 'NOT_RUN']));
  assert.equal(studyStatus(outcomes), 'INCOMPLETE');
  assert.throws(() => studyStatus({ ...outcomes, neural30: 'PASS' }));
  assert.equal(studyStatus({ ...outcomes, lifecycle: 'PASS', parity30: 'PASS', baseline30: 'FAIL' }), 'FAIL');
  assert.equal(studyStatus(Object.fromEntries(Object.keys(CASES).map(id => [id, 'PASS']))), 'PASS');
  assert.throws(() => studyStatus({ ...outcomes, retry: 'PASS' }));
  const geometry = [{ kind: 'fullscreen-enter', host: 3 }, { kind: 'fullscreen-exit', host: 5 },
    ...[[1000,600],[600,400],[1000,600],[1400,900],[1000,600]].map((drawable, index) => ({ kind: 'frame', opportunityHost: index + 1, drawable, networkOutput: [2560,1440] })),
    ...[[600,400],[1000,600],[1400,900],[1000,600]].map(drawable => ({ kind: 'drawable-resize', drawable, viewPoints: drawable.map(value => value / 2), backingScale: 2 }))];
  checkGeometry(geometry);
  const fixed = structuredClone(geometry); fixed.filter(row => row.kind === 'frame').forEach(row => row.drawable = [1000,600]);
  assert.throws(() => checkGeometry(fixed));
  const changedOutput = structuredClone(geometry); changedOutput.find(row => row.kind === 'frame').networkOutput = [1280,720];
  assert.throws(() => checkGeometry(changedOutput));
`], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));

it('rejects substituted parity snapshots and redirected evidence paths', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
  import assert from 'node:assert/strict';
  import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
  import { tmpdir } from 'node:os';
  import { join, relative } from 'node:path';
  import { reference, verifySavedSnapshot } from './tools/m13/playback.mjs';
  const directory = mkdtempSync(join(tmpdir(), 'aethervsr-parity-check-'));
  try {
    const path = join(directory, 'result.json'), other = join(directory, 'other.json');
    const result = { outcome: 'PASS', pts: 1 };
    writeFileSync(path, JSON.stringify(result)); writeFileSync(other, JSON.stringify(result));
    const entry = { reference: reference(relative(process.cwd(), path)), result };
    assert.deepEqual(verifySavedSnapshot(entry, path), result);
    assert.throws(() => verifySavedSnapshot(entry, other));
    writeFileSync(path, JSON.stringify({ ...result, pts: 2 }));
    assert.throws(() => verifySavedSnapshot(entry, path));
    assert.throws(() => verifySavedSnapshot({ ...entry, reference: reference(relative(process.cwd(), path)) }, path));
  } finally { rmSync(directory, { recursive: true }); }
`], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));

it('replays a consumed timeout as FAIL and leaves later trials gated', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
  import assert from 'node:assert/strict';
  import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
  import { tmpdir } from 'node:os';
  import { join, relative } from 'node:path';
  import { reference, evaluateCase, checkCase, studyStatus, CASES } from './tools/m13/playback.mjs';
  const directory = mkdtempSync(join(tmpdir(), 'aethervsr-timeout-check-'));
  try {
    const output = join(directory, 'lifecycle'); mkdirSync(output);
    const manifest = { sourceCommit: 'fixture', binary: { sha256: 'fixture' }, clips: [{ path: 'public/media/aethervsr-testclip-720p30-h264.mp4' }] };
    writeFileSync(join(directory, 'manifest.json'), JSON.stringify(manifest));
    writeFileSync(join(output, 'launch.json'), JSON.stringify({ id: 'lifecycle', sourceCommit: manifest.sourceCommit, binary: manifest.binary,
      clip: manifest.clips[0], manifest: reference(relative(process.cwd(), join(directory, 'manifest.json'))) }));
    writeFileSync(join(output, 'process.log'), 'diagnostic timeout');
    writeFileSync(join(output, 'process.json'), JSON.stringify({ exitCode: null, signal: 'SIGKILL', error: 'timeout',
      log: reference(relative(process.cwd(), join(output, 'process.log'))) }));
    const checked = evaluateCase(output, 'lifecycle'); assert.equal(checked.outcome, 'FAIL'); assert.equal(checked.observation, null);
    writeFileSync(join(output, 'result.json'), JSON.stringify(checked));
    assert.deepEqual(checkCase(directory, 'lifecycle', manifest), checked);
    const outcomes = Object.fromEntries(Object.keys(CASES).map(id => [id, id === 'lifecycle' ? 'FAIL' : 'NOT_RUN']));
    assert.equal(studyStatus(outcomes), 'FAIL'); assert.throws(() => studyStatus({ ...outcomes, baseline30: 'PASS' }));
  } finally { rmSync(directory, { recursive: true }); }
`], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));