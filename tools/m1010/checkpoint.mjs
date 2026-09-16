import assert from 'node:assert/strict';
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { ROOT, sha256 } from '../m10-fixtures.mjs';

export function openCheckpoint(directory, pin) {
  const root = resolve(ROOT, directory);
  assert(root.startsWith(join(ROOT, '.cache/m1010/')), 'Checkpoint must remain in ignored research storage');
  assert(typeof pin.studyVersion === 'string' && pin.studyVersion.length > 0);
  assert(/^[0-9a-f]{40}$/.test(pin.sourceCommit));
  assert(/^[0-9a-f]{64}$/.test(pin.browserExecutableSha256));
  mkdirSync(root, { recursive: true });
  const path = join(root, 'state.json');
  const state = existsSync(path) ? JSON.parse(readFileSync(path)) : {
    schemaVersion: 1, pin: structuredClone(pin), completedExperimentIds: [], rawArtifacts: {},
    candidateState: {}, requiredNextManualAction: null, activeExperimentId: null,
  };
  assert.equal(state.schemaVersion, 1);
  assert.deepEqual(state.pin, pin, 'Study/source/browser identity changed; explicit new study required');
  const validateId = id => assert(typeof id === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/.test(id), 'Invalid experiment ID');
  const artifact = id => { validateId(id); return join(root, `${id}.json`); };
  const verifyReferences = value => {
    if (!value || typeof value !== 'object') return;
    if (typeof value.path === 'string' && typeof value.sha256 === 'string' && Number.isSafeInteger(value.bytes)) {
      const referenced = resolve(value.path.startsWith('.cache/') ? ROOT : root, value.path);
      assert(referenced.startsWith(join(ROOT, '.cache/m1010/')), 'Raw reference outside ignored study storage');
      const bytes = readFileSync(referenced);
      assert.equal(bytes.length, value.bytes, 'Retained raw byte count changed');
      assert.equal(sha256(bytes), value.sha256, 'Retained raw artifact changed');
    }
    for (const child of Object.values(value)) verifyReferences(child);
  };
  const read = id => {
    const bytes = readFileSync(artifact(id)), report = JSON.parse(bytes);
    assert.equal(report.experimentId, id); assert.deepEqual(report.pin, pin);
    verifyReferences(report.result);
    return { report, reference: { path: `${id}.json`, bytes: bytes.length, sha256: sha256(bytes) } };
  };
  const save = () => {
    const temporary = join(root, `.state-${randomUUID()}.json`);
    writeFileSync(temporary, JSON.stringify(state, null, 2), { flag: 'wx' }); renameSync(temporary, path);
  };
  assert.equal(new Set(state.completedExperimentIds).size, state.completedExperimentIds.length);
  assert.deepEqual(Object.keys(state.rawArtifacts).sort(), [...state.completedExperimentIds].sort());
  for (const id of state.completedExperimentIds) assert.deepEqual(read(id).reference, state.rawArtifacts[id], 'Retained artifact changed');
  for (const name of readdirSync(root).filter(name => /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\.json$/.test(name) && name !== 'state.json')) {
    const id = name.slice(0, -5);
    if (state.completedExperimentIds.includes(id)) continue;
    const retained = read(id);
    assert.equal(state.activeExperimentId, id, 'Unregistered artifact cannot be adopted');
    state.completedExperimentIds.push(id); state.rawArtifacts[id] = retained.reference;
    state.activeExperimentId = null; state.requiredNextManualAction = null;
  }
  save();
  return {
    snapshot: () => structuredClone(state),
    has: id => state.completedExperimentIds.includes(id),
    read: id => {
      assert(state.completedExperimentIds.includes(id), 'Experiment is not complete');
      const retained = read(id); assert.deepEqual(retained.reference, state.rawArtifacts[id]);
      return retained.report.result;
    },
    begin(id) {
      validateId(id); assert(!state.completedExperimentIds.includes(id), 'Completed experiment is immutable');
      assert(state.activeExperimentId === null || state.activeExperimentId === id, 'Unfinished experiment must be resolved first');
      state.activeExperimentId = id; save();
    },
    complete(id, result) {
      assert.equal(state.activeExperimentId, id); assert(!state.completedExperimentIds.includes(id));
      writeFileSync(artifact(id), JSON.stringify({ experimentId: id, pin, completedAt: new Date().toISOString(), result }, null, 2), { flag: 'wx' });
      state.completedExperimentIds.push(id); state.rawArtifacts[id] = read(id).reference;
      state.activeExperimentId = null; state.requiredNextManualAction = null; save();
    },
    manual(action) {
      assert(action === null || (typeof action.instruction === 'string' && typeof action.observable === 'string'));
      state.requiredNextManualAction = action; save();
    },
    candidate(id, value) { validateId(id); state.candidateState[id] = structuredClone(value); save(); },
  };
}