import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ROOT, cachePath, digest, verifyReference } from './build.mjs';
import { CALIBRATION_CASES } from './study.mjs';

const DIRECTORY = '.cache/m1010r/calibration-01';
const SOURCE = '90dc459d2a144f201b7a84e77e9eff8aebbb7b6c';
export const VERDICT = 'CONTROLLED REPLAY PATH NOT QUALIFIED';
const fields = (value, names) => Object.fromEntries(names.filter(name => value && Object.hasOwn(value, name)).map(name => [name, value[name]]));

export function referenceEvidence(path) {
  const absolute = cachePath(path), bytes = readFileSync(absolute);
  return { path: relative(ROOT, absolute), bytes: bytes.length, sha256: digest(bytes) };
}

export function controlDiagnosis(record, bytes) {
  assert.equal(record.sampleRate, 48000);
  assert.equal(bytes.length, record.samples * 4);
  assert(Number.isSafeInteger(record.firstFrame) && record.firstFrame >= 0);
  let allFinite = true, nonzeroSamples = 0;
  for (let offset = 0; offset < bytes.length; offset += 4) {
    const sample = bytes.readFloatLE(offset);
    allFinite &&= Number.isFinite(sample);
    if (sample !== 0) nonzeroSamples++;
  }
  const end = record.firstFrame + record.samples;
  const scheduled = record.scheduled.map(row => ({ startFrame: row.startFrame, endExclusiveFrame: row.startFrame + row.samples,
    referenceStart: row.referenceStart, samples: row.samples, gain: row.gain }));
  return { firstFrame: record.firstFrame, endExclusiveFrame: end, samples: record.samples, sampleRate: record.sampleRate,
    sampleCoverageMs: record.samples / record.sampleRate * 1000, allFinite, nonzeroSamples,
    scheduled, endsBeforeFirstScheduledSignal: end <= Math.min(...scheduled.map(row => row.startFrame)),
    overlapWithScheduledSignals: scheduled.some(row => Math.max(record.firstFrame, row.startFrame) < Math.min(end, row.endExclusiveFrame)),
    expectedWindows: record.expectedWindows, verifiedWindows: record.verifiedWindows,
    maximumSampleError: record.maximumSampleError, unverifiedWindows: record.errors.length,
    interpretation: 'Sample coverage is not wall-clock duration or physical latency; missing scheduled windows are not measured waveform errors.' };
}

export function createReplayReport() {
  const rawReferences = new Map();
  const retain = reference => { verifyReference(reference); rawReferences.set(reference.path, reference); return reference; };
  const visit = value => {
    if (!value || typeof value !== 'object') return;
    if (typeof value.path === 'string' && typeof value.sha256 === 'string' && typeof value.bytes === 'number') retain(value);
    for (const child of Object.values(value)) visit(child);
  };
  const read = path => {
    const reference = retain(referenceEvidence(path));
    const value = JSON.parse(verifyReference(reference).toString('utf8'));
    visit(value);
    return { reference, value };
  };
  const { reference: stateReference, value: state } = read(`${DIRECTORY}/state.json`);
  assert.equal(state.pin.sourceCommit, SOURCE);
  assert.equal(state.status, 'STOPPED');
  assert.equal(state.activeExperimentId, null); assert.equal(state.requiredNextManualAction, null);
  assert.deepEqual(state.completedExperimentIds, CALIBRATION_CASES.map(row => row.id));
  const experiments = CALIBRATION_CASES.map(({ id, fps, repeat }) => {
    const reference = retain(state.rawArtifacts[id]);
    const envelope = JSON.parse(verifyReference(reference).toString('utf8'));
    assert.equal(envelope.id, id); assert.deepEqual(envelope.pin, state.pin); visit(envelope.result);
    const result = envelope.result;
    if (result.outcome === 'NOT_RUN') return { id, fps, repeat, outcome: 'NOT_RUN', reference, reason: result.reason };
    assert.equal(id, 'calibration-30-1'); assert.equal(result.outcome, 'UNRESOLVED');
    const report = result.report;
    const analysisReference = retain(state.analysisArtifacts[id]);
    const analysis = JSON.parse(verifyReference(analysisReference).toString('utf8'));
    assert.equal(analysis.id, id); assert.deepEqual(analysis.pin, state.pin);
    assert.deepEqual(analysis.result.raw, reference); assert.equal(analysis.result.outcome, 'UNRESOLVED');
    const control = controlDiagnosis(report.audioControl, verifyReference(result.controlAudioPcm));
    assert(control.endsBeforeFirstScheduledSignal && !control.overlapWithScheduledSignals && control.allFinite);
    assert.equal(report.frames.length, 0); assert.equal(report.callbacks.length, 0);
    assert.equal(report.gpuSamples.length, 0); assert.equal(report.audio, null);
    assert.equal(control.verifiedWindows, 0); assert.equal(control.maximumSampleError, null);
    const browser = result.environment.browser;
    return { id, fps, repeat, outcome: 'UNRESOLVED', reference, analysisReference,
      startedAt: result.startedAt, endedAt: result.endedAt, pageClosed: result.pageClosed,
      environment: { machine: result.environment.machine, os: result.environment.os,
        displayRefreshHz: result.environment.displayRefreshHz, browser: {
          version: browser.version, executableSha256: browser.executableSha256,
          policy: browser.policy, requestedFlags: browser.flags, actualCommandLine: browser.commandLine,
          manifest: browser.manifest } },
      native: { state: report.state, primaryErrors: report.errors,
        control, frames: report.frames.length, callbacks: report.callbacks.length, gpuSamples: report.gpuSamples.length,
        mediaAudio: 'NOT_RUN', graph: report.graph, capabilities: report.capabilities,
        cleanup: fields(report.cleanup, ['pipeline', 'probe', 'device', 'audioNodes', 'observers', 'timers',
          'pendingCompletions', 'audioContext', 'videoPaused', 'completed']),
        secondaryCollectionErrors: result.errors.map(error => fields(error, ['stage', 'message'])) },
      originalAnalysis: fields(analysis.result, ['outcome', 'summary', 'candidateTiming', 'neural']),
      postRunDiagnosis: 'The host-timer finish occurred before the recording reached the scheduled audio sample frames. Audio-clock-driven completion is unverified; decoder/OS/device startup causation is not established.' };
  });
  const media = read('.cache/m1010r/media-02/media.json');
  const shifts = read('.cache/m1010r/shift-final-01/validation.json');
  assert.equal(shifts.value.mediaManifestSha256, digest(JSON.stringify(media.value)));
  assert.equal(media.value.seconds, 70);
  for (const asset of media.value.assets) {
    assert.equal(asset.validation.video.frames, asset.fps * 70);
    assert.equal(asset.validation.video.allIdentitiesAndPtsExact, true);
    assert.equal(asset.validation.audio.decodedSamples, 3360000);
  }
  const frozenBytes = readFileSync(resolve(ROOT, 'results/m10.10-product.json'));
  assert.equal(digest(frozenBytes), 'b7d110addf560efae427d160331ff012d4613ce2a21d38e86e2199d952d3920d');
  const gatedNames = ['timingAcceptance', 'authorityHandover', 'audioOverlapGap', 'returnToPage', 'failureRollback',
    'permissionRefetch', 'sourceReplacement', 'workerRestart', 'multiplePlayers', 'inputProvenanceRegression',
    'raw30Cadence', 'raw60Cadence', 'avSync', 'drift', 'seekRecovery', 'pauseResume', 'rate125',
    'neural720p60', 'tenMinuteSoak'];
  return { schemaVersion: 1, study: 'M10.10R', verdict: VERDICT, status: 'CLOSED_NOT_QUALIFIED',
    scope: 'Progressive replay-only research; no product implementation, no physical timing claim',
    baseline: { sourceCommit: state.pin.baseline, branch: 'main', clean: true, ciRunId: 35086643662,
      ciGate: 'success', ciFusion: 'success', vitestFiles: 45, vitestPassed: 1249, optInSkipped: 2,
      ffmpegFixturePassed: 8, pythonPassed: 311, trackedBytes: 53855518, capBytes: 54525952 },
    frozenM1010: { status: 'INCOMPLETE_RESEARCH_CHECKPOINT', decision: 'NO CONTROLLED WEB PRODUCT PATH QUALIFIED',
      index: { path: 'results/m10.10-product.json', bytes: frozenBytes.length, sha256: digest(frozenBytes) } },
    nativeCheckpoint: { state: stateReference, pin: state.pin, status: state.status, attempted: 1,
      notRun: 5, completedRecordsAreNotMeasurements: true, manualAction: null },
    experiments,
    groundTruth: { manifest: media.reference, seconds: media.value.seconds, ffmpeg: media.value.ffmpeg,
      signal: media.value.signal, assets: media.value.assets.map(asset => ({ fps: asset.fps, media: asset.media, pcm: asset.pcm,
        video: asset.validation.video, audio: fields(asset.validation.audio, ['sampleRate', 'decodedSamples', 'firstPts',
          'timeBase', 'firstSampleMediaTime', 'endExclusiveMediaTime', 'decodedFrames', 'decodedPcmBytes', 'decodedPcmSha256', 'ptsFramesSha256']),
        impulseCount: asset.validation.audio.impulses.length })), shifts: { reference: shifts.reference,
          controls: shifts.value.controls.map(row => fields(row, ['fps', 'requestedAudioShiftSeconds', 'observedAudioShiftSeconds',
            'expectedVideoMinusAudioShiftMs', 'videoPtsUnchanged', 'matchedInteriorSamples', 'referenceFirstMediaTime', 'shiftedFirstMediaTime', 'media'])) } },
    qualification: { instrument: 'UNRESOLVED', digitalUncertaintyBoundMs: null,
      candidateEligibility: false, ...Object.fromEntries(gatedNames.map(name => [name, 'NOT_RUN'])), m1011: 'NOT_STARTED' },
    nextDirection: 'One separately authorized replay-instrument investigation: sample-clock-driven recording completion with an independent wall-clock watchdog; no automatic retry or capture/overlay pivot.',
    limitations: ['No media timing window began; zero frames is an observed count, not a throughput measurement.',
      'No native authority, Return, permission, worker-restart, generation or multi-session gate was earned.',
      'The measured apparatus has a known wall-clock completion defect and is not qualified for reuse.',
      'Audio startup/scheduling/OS causation is unresolved; no speaker/scanout claim.',
      'Post-measurement test-only failure regression is not a native fix or requalification.'],
    rawReferences: [...rawReferences.values()].sort((left, right) => left.path.localeCompare(right.path)) };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert(process.argv.length <= 3 && (process.argv[2] === undefined || process.argv[2] === '--check'));
  const path = resolve(ROOT, 'results/m10.10r-replay.json');
  const bytes = Buffer.from(JSON.stringify(createReplayReport(), null, 2) + '\n');
  assert(bytes.length <= 180000, 'Compact replay report exceeded its prospective budget');
  if (process.argv[2] === '--check') assert.deepEqual(readFileSync(path), bytes, 'Replay report is not reproducible');
  else writeFileSync(path, bytes);
  console.log(JSON.stringify({ path: relative(ROOT, path), bytes: bytes.length, sha256: digest(bytes), verified: true }));
}