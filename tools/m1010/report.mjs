import assert from 'node:assert/strict';
import { readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { ROOT, sha256 } from '../m10-fixtures.mjs';

const rawRoot = join(ROOT, '.cache/m1010');
const legacyNames = ['frozen-controls', 'surface-01', 'media-api-01', 'media-api-02', 'navigation-01', 'navigation-02', 'sender-context-01',
  ...[1, 2, 3, 4].flatMap(index => [`manual-0${index}`, `manual-0${index}.progress`])];
const studyNames = ['replay-01', 'replay-input-01', 'timing-floor-01', 'tab-fidelity-01', 'self-capture-01', 'self-capture-02', 'self-capture-03'];
const pick = (value, keys) => Object.fromEntries(keys.filter(key => value?.[key] !== undefined).map(key => [key, value[key]]));

export function fileEvidence(path, expected) {
  const absolute = realpathSync(path);
  assert(absolute.startsWith(`${rawRoot}/`), 'Evidence outside ignored study storage');
  const bytes = readFileSync(absolute), reference = { path: relative(ROOT, absolute), bytes: bytes.length, sha256: sha256(bytes) };
  if (expected) { assert.equal(reference.bytes, expected.bytes, 'Evidence byte count changed'); assert.equal(reference.sha256, expected.sha256, 'Evidence hash changed'); }
  return reference;
}

export function referencedEvidence(value, directory, collected = new Map()) {
  if (!value || typeof value !== 'object') return [...collected.values()];
  if (typeof value.path === 'string' && typeof value.sha256 === 'string' && Number.isSafeInteger(value.bytes)) {
    const reference = fileEvidence(resolve(value.path.startsWith('.cache/') ? ROOT : directory, value.path), value);
    collected.set(reference.path, reference);
  }
  for (const child of Object.values(value)) referencedEvidence(child, directory, collected);
  return [...collected.values()].sort((left, right) => left.path.localeCompare(right.path));
}

const playback = value => pick(value, ['playable', 'error', 'width', 'height', 'muted', 'paused', 'readyState', 'mediaError', 'cadence']);
const track = value => pick(value, ['kind', 'constructor', 'tag', 'readyState', 'muted', 'cropTo', 'restrictTo']);
const observation = value => value ? pick(value, ['instruction', 'observable', 'requestedAt', 'observedAt']) : null;
const resources = value => value ? pick(value, ['tracks', 'peers', 'objectUrls', 'device', 'pipeline', 'driver', 'frameCallback', 'pip', 'objectUrl']) : undefined;
const cleanup = value => value ? {
  ...pick(value, ['error', 'pending', 'captureState', 'paused', 'sourceAttribute']),
  ...(value.resources ? { resources: resources(value.resources) } : {}),
  ...(Object.hasOwn(value, 'srcObject') ? { hasSrcObject: value.srcObject !== null } : {}),
  ...(value.displayTracks ? { displayTracks: value.displayTracks.map(track) } : {}),
  ...(value.stoppedTracks ? { stoppedTracks: value.stoppedTracks.map(value => ({ before: track(value.before), after: track(value.after) })) } : {}),
} : null;

export function summarizeExperiment(id, result) {
  const summary = pick(result, ['outcome', 'executionStatus', 'reason', 'error', 'scope', 'limitation', 'sameBytes', 'brokerEnforced', 'exception']);
  if (result.credentialsOmitted) summary.credentialsOmitted = pick(result.credentialsOmitted, ['fetchable', 'status', 'bytes', 'sha256', 'credentials', 'error']);
  if (result.requestDelta) summary.requestDelta = result.requestDelta.map(value => pick(value, ['method', 'path', 'status', 'cookie', 'authenticated', 'origin']));
  if (id === 'R1-host-grant') summary.grant = pick(result.grants, ['origins', 'permissions']);
  if (id === 'R1-F') Object.assign(summary, { removed: observation(result.removed), futureRefetchRejected: result.futureRefetchRejected,
    fetchAfter: pick(result.fetchAfter, ['fetchable', 'status', 'error', 'credentials']) });
  if (id === 'consent') Object.assign(summary, { initiallyGranted: result.initiallyGranted, observation: observation(result.observation),
    observedGrant: result.observation?.value?.granted ?? null });
  if (id === 'invocation') Object.assign(summary, { observation: observation(result), sourceTabId: result.value?.sourceTabId ?? null });
  if (id === 'setup-failure') Object.assign(summary, pick(result, ['name', 'phase', 'chooserTrialsStarted', 'recordOrigin', 'observedAt', 'terminationError', 'terminationScope']));
  if (result.fetch) summary.fetch = pick(result.fetch, ['ok', 'fetchable', 'status', 'bytes', 'sha256', 'error']);
  if (result.playback) summary.playback = playback(result.playback);
  if (result.comparisons) summary.comparisons = result.comparisons;
  if (result.cleanup) summary.cleanup = cleanup(result.cleanup);
  if (result.cadenceSummary) summary.cadence = result.cadenceSummary;
  if (id === 'R1-lifecycle') {
    summary.cycles = result.cycles.map(value => ({ ...pick(value, ['index', 'asset', 'outcome', 'authority', 'error']),
      source: pick(value.source, ['sourceClass', 'intrinsic']), seek: pick(value.seek, ['requestedTime', 'currentTime']), resumed: playback(value.resumed), cleanup: cleanup(value.cleanup) }));
    summary.scope = result.scope;
  }
  if (id.startsWith('floor-')) {
    summary.timing = result.summary;
    summary.environment = result.environment;
    summary.media = pick(result.media, ['path', 'bytes', 'sha256', 'fps', 'seconds', 'analysis']);
    summary.boundary = 'Local source and preview floor only; not candidate latency, physical scanout, speaker output or neural performance';
  }
  if (id.startsWith('R3-') && result.samples) summary.samples = result.samples.map(value => ({
    ...pick(value, ['time', 'seek', 'geometry', 'mapping', 'difference']),
    source: pick(value.source, ['width', 'height', 'originClean', 'pixels', 'copiedPixels', 'errors']),
    acquired: pick(value.acquired, ['width', 'height', 'originClean', 'pixels', 'copiedPixels', 'errors']),
    readiness: value.readiness ? { ...playback(value.readiness), callbacks: value.readiness.frames.length } : null,
  }));
  if (id === 'cleanup' && result.player) summary.teardown = { player: cleanup(result.player), source: {
    ...pick(result.source, ['captureState', 'paused', 'readyState']), displayTracks: result.source?.displayTracks?.map(track) ?? [],
    hasSelection: result.source?.selection != null } };
  if (id.startsWith('R6-')) {
    Object.assign(summary, pick(result, ['identity', 'generation', 'captureError', 'markerErrors']));
    if (id === 'R6-cancel') summary.rejectionScope = result.captureError
      ? 'Requested cancellation; native rejection observed, API cannot distinguish Cancel from denial'
      : 'Requested cancellation was not observed; capture occurred instead. Raw rejectionScope wording is contradicted by its own capture evidence and is not adopted.';
    summary.request = pick(result.choice, ['instruction', 'requestedAt', 'observedAt']);
    summary.observedTracks = result.choice?.value?.displayTracks?.map(value => ({ ...track(value), settings: pick(value.settings,
      ['width', 'height', 'frameRate', 'screenPixelRatio', 'displaySurface', 'resizeMode', 'sampleRate', 'channelCount', 'latency']) })) ?? [];
    summary.identityFrame = pick(result.identityFrame, ['width', 'height', 'frames', 'pixels', 'scope', 'stateTransitionFreshness', 'cadence']);
    summary.method = result.method ? { ...pick(result.method, ['name', 'factory', 'outcome', 'error']), track: track(result.method.track) } : null;
    summary.scopeObservations = result.scopes?.length ?? 0;
  }
  return summary;
}

export function createEvidenceIndex() {
  const legacy = legacyNames.map(name => {
    const path = join(rawRoot, `${name}.json`), value = JSON.parse(readFileSync(path));
    return { ...fileEvidence(path), sourceCommit: value.identity?.sourceCommit ?? value.sourceCommit ?? null,
      browser: pick(value.browser, ['version', 'executableSha256']), phase: value.phase ?? null,
      provenanceScope: 'Only fields present in this artifact; null or absent attribution is not inherited from neighboring runs',
      ...(name === 'surface-01' ? { surfaces: value.results.map(row => ({ ...pick(row, ['surface', 'import', 'outcome']), comparisons: row.comparisons })) } : {}) };
  });
  const studies = studyNames.map(name => {
    const directory = join(rawRoot, name), statePath = join(directory, 'state.json'), state = JSON.parse(readFileSync(statePath));
    assert.equal(state.activeExperimentId, null, 'Cannot index an active experiment');
    assert.equal(state.requiredNextManualAction, null, 'Cannot index pending consent');
    assert.equal(new Set(state.completedExperimentIds).size, state.completedExperimentIds.length);
    assert.deepEqual(Object.keys(state.rawArtifacts).sort(), [...state.completedExperimentIds].sort());
    const experiments = state.completedExperimentIds.map(id => {
      assert(/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/.test(id));
      const path = join(directory, `${id}.json`), reference = fileEvidence(path, state.rawArtifacts[id]), value = JSON.parse(readFileSync(path));
      assert.equal(value.experimentId, id); assert.deepEqual(value.pin, state.pin);
      return { id, artifact: reference, completedAt: value.completedAt, rawReferences: referencedEvidence(value.result, directory), summary: summarizeExperiment(id, value.result) };
    });
    return { name, state: fileEvidence(statePath), pin: state.pin, candidateState: state.candidateState, experiments };
  });
  return { schemaVersion: 1, milestone: 'M10.10', status: 'INCOMPLETE_RESEARCH_CHECKPOINT',
    decision: 'NO CONTROLLED WEB PRODUCT PATH QUALIFIED', decisionScope: 'Mandatory gates not earned; not proof of platform impossibility',
    protocol: 'docs/M10.10-PRODUCT-PREREGISTRATION.md',
    qualification: { absoluteInstrumentErrorBoundMs: null, candidateLatency: 'NOT_RUN', candidateAvSync: 'NOT_RUN',
      neuralPairedPerformance: 'NOT_RUN', publicCensus: 'NOT_RUN', documentPipNative: 'NOT_RUN', m1011: 'NOT_STARTED' },
    unavailableMetric: 'not measured; null is used for unavailable numeric values, never zero',
    rawStorage: 'Ignored local .cache/m1010; this index is verifiable only where retained raw artifacts are available',
    legacy, studies };
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) {
  const output = resolve(ROOT, process.argv.slice(2).find(value => value !== '--check') ?? 'results/m10.10-product.json');
  assert(dirname(output) === join(ROOT, 'results'), 'Only compact results output is permitted');
  const text = `${JSON.stringify(createEvidenceIndex(), null, 2)}\n`;
  assert(Buffer.byteLength(text) < 300000, 'Compact evidence budget exceeded');
  if (process.argv.includes('--check')) assert.equal(readFileSync(output, 'utf8'), text, 'Tracked evidence is stale');
  else writeFileSync(output, text);
  console.log(JSON.stringify({ output: relative(ROOT, output), bytes: Buffer.byteLength(text), sha256: sha256(Buffer.from(text)), verified: true }));
}