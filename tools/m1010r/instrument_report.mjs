import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ROOT, digest, verifyReference } from './build.mjs';
import { referenceEvidence } from './report.mjs';

export const INSTRUMENT_VERDICT = 'DIGITAL TIMING INSTRUMENT NOT QUALIFIED';
const DIRECTORY = '.cache/m1010r/ri-01';
const SOURCE = '6e0c4b741c581ac615e8b4675dd723cc729c1a59';
const IDS = ['ri-30-1', 'ri-60-1', 'ri-30-2', 'ri-60-2'];
const pick = (value, keys) => Object.fromEntries(keys.filter(key => value && Object.hasOwn(value, key)).map(key => [key, value[key]]));
const milliseconds = stats => Object.fromEntries(Object.entries(stats).map(([key, value]) => [key, key === 'count' || value === null ? value : value * 1000]));

export function renderEvidence(render, bytes) {
  const terminal = render.terminal;
  assert(terminal && terminal.sampleRate === 48000);
  const blocks = terminal.blockLengths;
  assert.equal(blocks.length, terminal.processedBlocks);
  assert.equal(blocks.reduce((sum, length) => sum + length, 0), terminal.processedSamples);
  assert.equal(terminal.actualObservedEndFrame - terminal.firstFrame, terminal.processedSamples);
  assert.equal(bytes.length, terminal.processedSamples * 4);
  let finite = true, nonzeroSamples = 0;
  for (let offset = 0; offset < bytes.length; offset += 4) {
    const value = bytes.readFloatLE(offset);
    finite &&= Number.isFinite(value);
    if (value !== 0) nonzeroSamples++;
  }
  const reached = terminal.completionReason === 'RENDER_TARGET_REACHED';
  if (reached) assert(terminal.actualObservedEndFrame >= terminal.requestedEndFrame);
  return { completionReason: terminal.completionReason, requestedEndFrame: terminal.requestedEndFrame,
    firstFrame: terminal.firstFrame, actualObservedEndFrame: terminal.actualObservedEndFrame,
    processedSamples: terminal.processedSamples, processedBlocks: terminal.processedBlocks,
    sampleRate: terminal.sampleRate, sampleCoverageMs: terminal.processedSamples / 48,
    blockLengths: Object.fromEntries([...new Set(blocks)].sort((left, right) => left - right).map(length => [length, blocks.filter(value => value === length).length])),
    lastBlockLength: blocks.at(-1), completionOvershootFrames: reached ? terminal.actualObservedEndFrame - terminal.requestedEndFrame : null,
    offendingNextFrame: null, discontinuityMagnitudeFrames: null, finite, nonzeroSamples,
    heartbeatCount: render.heartbeats.length, reportedHeartbeatCount: terminal.heartbeatCount,
    watchdogFired: render.watchdogFired, timedOut: render.timedOut, overflow: terminal.overflow,
    discontinuity: terminal.discontinuity, hostErrors: render.errors,
    contextStates: render.contextEvents.map(event => event.state),
    interpretation: 'Half-open retained sample range, not wall duration or physical latency; the offending next frame is not recorded.' };
}

export function createInstrumentReport() {
  const references = new Map();
  const keep = reference => { const bytes = verifyReference(reference); references.set(reference.path, reference); return bytes; };
  const visit = value => {
    if (!value || typeof value !== 'object') return;
    if (typeof value.path === 'string' && typeof value.sha256 === 'string' && typeof value.bytes === 'number') keep(value);
    for (const child of Object.values(value)) visit(child);
  };
  const stateRef = referenceEvidence(`${DIRECTORY}/state.json`);
  const state = JSON.parse(keep(stateRef));
  assert.equal(state.pin.sourceCommit, SOURCE); assert.equal(state.status, 'STOPPED');
  assert.equal(state.activeExperimentId, null); assert.equal(state.requiredNextManualAction, null);
  assert.deepEqual(state.order, IDS); assert.deepEqual(state.completedExperimentIds, IDS);
  assert.equal(digest(readFileSync(resolve(ROOT, 'docs/M10.10RI-PREREGISTRATION.md'))), state.pin.preregistrationSha256);
  visit(state);
  const experiments = IDS.map(id => {
    const raw = state.rawArtifacts[id], envelope = JSON.parse(keep(raw));
    assert.equal(envelope.id, id); assert.deepEqual(envelope.pin, state.pin); visit(envelope.result);
    const result = envelope.result;
    if (id !== IDS[0]) { assert.equal(result.outcome, 'NOT_RUN'); return { id, outcome: 'NOT_RUN', raw, reason: result.reason }; }
    const analysisRef = state.analysisArtifacts[id], saved = JSON.parse(keep(analysisRef));
    assert.deepEqual(saved.pin, state.pin); assert.equal(saved.id, id);
    assert.deepEqual(saved.result.raw, raw); assert.equal(saved.result.outcome, 'FAIL');
    const analysis = saved.result, report = result.report;
    assert.equal(report.instrument, 'M10.10RI'); assert.equal(report.state, 'UNRESOLVED');
    assert.equal(analysis.control.passed, true); assert.equal(analysis.control.verifiedWindows, 180);
    assert.equal(analysis.control.maximumSampleError, 0);
    assert.equal(report.renderAudio.terminal.completionReason, 'DISCONTINUITY');
    assert.equal(analysis.timing.count, 0); assert.equal(analysis.uncertaintyBoundMs, null);
    assert.equal(analysis.repeatability.validatedPairCount, 0); assert.equal(analysis.physicalLatencyBoundMs, null);
    const control = renderEvidence(report.audioControl.render, keep(result.controlAudioPcm));
    const media = renderEvidence(report.renderAudio, keep(result.audioPcm));
    const clock = render => ({ receiveResidualLowerMs: milliseconds(render.clock.lowerSeconds),
      receiveResidualUpperMs: milliseconds(render.clock.upperSeconds),
      negativeGuardMs: render.clock.negativeGuardSeconds * 1000, positiveDeliveryBoundMs: null,
      renderHeartbeatDeltaFrames: render.heartbeat.renderFrameDeltas, hostDeliveryRateHz: null,
      scope: 'Host AudioContext time minus retained render endpoint at message receipt; includes asynchronous delivery and further rendering, not clock error or physical latency.' });
    const browser = result.environment.browser;
    return { id, outcome: 'FAIL', nativeState: report.state, raw, analysis: analysisRef,
      startedAt: result.startedAt, endedAt: result.endedAt, environment: {
        machine: result.environment.machine, os: result.environment.os, physicalRefresh: result.environment.displayRefreshHz,
        browser: pick(browser, ['version', 'executableSha256', 'policy', 'commandLine', 'manifest']) },
      control: { render: control, scheduled: report.audioControl.scheduled,
        tailAfterFinalSignalFrames: analysis.control.render.range.tailFrames,
        expectedWindows: analysis.control.expectedWindows, verifiedWindows: analysis.control.verifiedWindows,
        maximumSampleError: analysis.control.maximumSampleError, independentWaveformPassed: analysis.control.windows.passed,
        pcm: result.controlAudioPcm, clocks: clock(analysis.control.render) },
      media: { render: media, renderWindow: report.renderWindow, pcm: result.audioPcm,
        callbacks: report.callbacks.length, successfulSubmissions: report.frames.length, gpuSamples: report.gpuSamples.length,
        selectedTimingRows: analysis.timing.count, validCoverage: analysis.timing.validCoverage,
        absolutePhase: analysis.timing.absoluteUnsubtractedPhase, driftIntervalMs: analysis.timing.firstToLastDriftIntervalMs,
        digitalUncertaintyBoundMs: analysis.uncertaintyBoundMs, physicalLatencyBoundMs: analysis.physicalLatencyBoundMs,
        callbackReadyStates: report.callbacks.map(row => row.readyState),
        callbacksVisibleAndFocused: report.callbacks.every(row => row.visibility === 'visible' && row.focused),
        clocks: clock(analysis.mediaRender), audioMatching: pick(analysis.audio,
          ['matchedWindowCount', 'validWindowCount', 'invalidWindowCount', 'invalidReasons', 'empiricalOffsetResidualSamples', 'residualSemantics']),
        globalWaveform: analysis.fullWaveform },
      originalAnalysisSummary: analysis.summary, nativeErrors: report.errors, runnerErrors: result.errors,
      repeatability: analysis.repeatability, cleanup: report.cleanup, pageClosed: result.pageClosed,
      postRunInterpretation: [
        'Short control passed once; it is not repeatability or full instrument qualification.',
        'Media process observed a discontinuous next block; its start/direction/magnitude were not serialized and cannot be reconstructed from target shortfall.',
        'audio_metadata_mismatch includes the disqualifying discontinuity flag; audio and terminal metadata agree on that flag.',
        'foreground_not_verified includes first-callback readiness=1; all callbacks were visible/focused. This is not evidence of a hidden interval.',
        'Global waveform and timing-window checks lack required duration; they do not demonstrate waveform corruption.',
      ] };
  });
  const oldRaw = referenceEvidence('.cache/m1010r/calibration-01/calibration-30-1.json');
  assert.equal(oldRaw.sha256, 'a80e452ca7d45cbb0abe50f1e985bbd128656f0393feb9cde871fd22ad83b1bf');
  const old = JSON.parse(keep(oldRaw));
  const oldPcm = old.result.controlAudioPcm, oldBytes = keep(oldPcm);
  assert.equal(oldPcm.sha256, '18619b678a5c207a971a0aa931604f48162e307c57ecdec450d5f095fe9f32c7');
  assert.equal(old.result.report.audioControl.firstFrame, 0); assert.equal(oldBytes.length, 7168 * 4);
  for (let offset = 0; offset < oldBytes.length; offset += 4) assert.equal(oldBytes.readFloatLE(offset), 0);
  const historical = [
    ['results/m10.10r-replay.json', 'c2d87b03e4c12314e03975cb1c0dbbab8003ce464f8b15e3d45fffbefb8ad7d1'],
    ['results/m10.10-product.json', 'b7d110addf560efae427d160331ff012d4613ce2a21d38e86e2199d952d3920d'],
  ].map(([path, expected]) => { const bytes = readFileSync(resolve(ROOT, path)); assert.equal(digest(bytes), expected); return { path, bytes: bytes.length, sha256: expected }; });
  return { schemaVersion: 1, milestone: 'M10.10RI', verdict: INSTRUMENT_VERDICT,
    scope: 'Finite instrument study only; no replay product qualification or physical timing claim',
    baseline: { sourceCommit: state.pin.baseline, cleanMain: true, startingCiRunId: 35100333935,
      ciGate: 'success', ciFusion: 'success', vitestFiles: 47, vitestPassed: 1337, optInSkipped: 3,
      pythonPassed: 497, optInPassed: 40, trackedBytes: 54183335, approvedCapBytes: 54525952 },
    checkpoint: { state: stateRef, pin: state.pin, status: 'STOPPED', attempted: 1, notRun: 3,
      completedIdsAreTerminalRecordsNotMeasurements: true, requiredNextManualAction: null },
    experiments, qualification: { shortControl: 'ONE_NATIVE_CONTROL_PASS', fullInstrument: 'FAIL',
      repeatability: 'NOT_RUN', digitalUncertaintyBoundMs: null, physicalLatencyBoundMs: null,
      futureReplayPrerequisites: 'NOT_AUTHORIZED', neural: 'NOT_RUN', productAuthority: 'NOT_RUN', m1011: 'NOT_STARTED' },
    frozen: { m1010: 'INCOMPLETE RESEARCH CHECKPOINT / NO CONTROLLED WEB PRODUCT PATH QUALIFIED',
      m1010r: 'CONTROLLED REPLAY PATH NOT QUALIFIED', originalFailedRaw: oldRaw, originalFailedPcm: oldPcm, indexes: historical },
    nextStep: 'Owner product-direction decision: scientifically valid weaker audio-only claim, cooperating-site/API, native desktop, or discontinuation. No automatic RI.2 or further native revision.',
    rawReferences: [...references.values()].sort((left, right) => left.path.localeCompare(right.path)) };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert(process.argv.length <= 3 && (process.argv[2] === undefined || process.argv[2] === '--check'));
  const path = resolve(ROOT, 'results/m10.10ri-instrument.json');
  const bytes = Buffer.from(JSON.stringify(createInstrumentReport(), null, 2) + '\n');
  assert(bytes.length <= 60000, 'RI compact evidence exceeds remaining planned budget');
  if (process.argv[2] === '--check') assert.deepEqual(readFileSync(path), bytes);
  else writeFileSync(path, bytes);
  console.log(JSON.stringify({ path: 'results/m10.10ri-instrument.json', bytes: bytes.length, sha256: digest(bytes), verified: true }));
}