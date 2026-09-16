import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { endianness } from 'node:os';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openNativeChrome } from '../m9-browser.mjs';
import { nativeWindow } from '../m105-accounting.mjs';
import { environment } from '../m1010/study.mjs';
import { ROOT, DEFAULT_EXTENSION, PROVENANCE_FILE, cachePath, digest, git, verifyBuild, verifyReference } from './build.mjs';

export const FROZEN_BASELINE = '5b1a313e1d603b46471aebb70594e044a67d44d6';
export const DEFAULT_CALIBRATION = '.cache/m1010r/calibration-01';
export const ANALYZER = 'tools/m1010r/analyze.py';
export const PYTHON = '.cache/m8-venv/bin/python';
export const CALIBRATION_CASES = Object.freeze([1, 2, 3].flatMap(repeat => [30, 60].map(fps =>
  Object.freeze({ id: `calibration-${fps}-${repeat}`, fps, repeat }))));
const ids = CALIBRATION_CASES.map(entry => entry.id);
const json = value => `${JSON.stringify(value, null, 2)}\n`;
const errorInfo = error => ({ message: String(error), stack: error?.stack ?? null });
const reference = path => {
  const bytes = readFileSync(cachePath(path));
  return { path: relative(ROOT, path), bytes: bytes.length, sha256: digest(bytes) };
};

export function studyIdentity(outdir = DEFAULT_EXTENSION) {
  assert.equal(git(['status', '--porcelain=v1', '--untracked-files=all']), '', 'Review and freeze clean source before launching calibration');
  git(['diff', '--exit-code', FROZEN_BASELINE, '--', 'src', 'models', 'public/models', 'tools/build-extension.mjs',
    'tools/m9-browser.mjs', 'tools/m105-accounting.mjs', 'package-lock.json', 'npm-shrinkwrap.json', 'yarn.lock', 'pnpm-lock.yaml']);
  const original = JSON.parse(git(['show', `${FROZEN_BASELINE}:package.json`]));
  const current = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'));
  delete original.scripts; delete current.scripts;
  assert.deepEqual(current, original, 'Only package scripts may change; dependencies and install policy are frozen');
  assert.equal(endianness(), 'LE', 'Native Float32 recording requires little-endian storage');
  const identity = verifyBuild(outdir), sourceCommit = git(['rev-parse', 'HEAD']);
  assert.equal(identity.provenance.sourceCommit, sourceCommit); assert.equal(identity.provenance.sourceDirty, false);
  assert(Object.keys(identity.provenance.sourceFiles).length > 0, 'Missing build-input hashes');
  for (const [name, expected] of Object.entries(identity.provenance.sourceFiles)) {
    const path = resolve(ROOT, name);
    assert(path.startsWith(ROOT + '/'), 'Build input outside repository');
    const bytes = readFileSync(path);
    assert.deepEqual({ bytes: bytes.length, sha256: digest(bytes) }, expected, `Build input changed: ${name}`);
  }
  return { ...identity, sourceCommit, baseline: FROZEN_BASELINE, build: reference(join(identity.directory, PROVENANCE_FILE)) };
}

function writeArtifact(directory, name, value) {
  const path = cachePath(join(directory, name));
  writeFileSync(path, Buffer.isBuffer(value) ? value : json(value), { flag: 'wx' });
  return reference(path);
}

function verifyReferences(value) {
  if (!value || typeof value !== 'object') return;
  if (typeof value.path === 'string' && 'sha256' in value && 'bytes' in value) verifyReference(value);
  for (const child of Object.values(value)) verifyReferences(child);
}

export function openCalibrationCheckpoint(directory, pin) {
  const root = cachePath(directory);
  mkdirSync(root, { recursive: true });
  const statePath = join(root, 'state.json');
  const retained = existsSync(statePath);
  if (!retained) assert.equal(readdirSync(root).length, 0, 'Fresh calibration directory must be empty');
  const state = retained ? JSON.parse(readFileSync(statePath, 'utf8')) : {
    schemaVersion: 1, pin, order: ids, status: 'READY', completedExperimentIds: [], rawArtifacts: {}, analysisArtifacts: {},
    activeExperimentId: null, requiredNextManualAction: null, stopReason: null, interruption: null, runnerError: null,
  };
  assert.equal(state.schemaVersion, 1); assert.deepEqual(state.pin, pin, 'Study/source/build/browser/analyzer identity changed');
  assert.deepEqual(state.order, ids); assert.equal(state.requiredNextManualAction, null);
  assert(['READY', 'RUNNING', 'COMPLETE', 'STOPPED'].includes(state.status));
  assert.deepEqual(state.completedExperimentIds, ids.slice(0, state.completedExperimentIds.length), 'Completed IDs must be a fixed-order prefix');
  const readEnvelope = (id, analysis = false) => {
    assert(ids.includes(id), 'Unknown calibration ID');
    const path = cachePath(join(root, `${id}${analysis ? '-analysis' : ''}.json`));
    const envelope = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(envelope.id, id); assert.deepEqual(envelope.pin, pin);
    verifyReferences(envelope.result);
    return { envelope, reference: reference(path) };
  };
  const save = () => {
    const temporary = join(root, `.state-${randomUUID()}.json`);
    writeFileSync(temporary, json(state), { flag: 'wx' }); renameSync(temporary, statePath);
  };
  const registerRaw = id => {
    const raw = readEnvelope(id);
    if (state.rawArtifacts[id]) assert.deepEqual(raw.reference, state.rawArtifacts[id], 'Immutable raw JSON changed');
    state.rawArtifacts[id] = raw.reference;
    return raw;
  };
  for (const id of Object.keys(state.rawArtifacts)) {
    assert(state.completedExperimentIds.includes(id) || state.activeExperimentId === id);
    registerRaw(id);
  }
  for (const id of state.completedExperimentIds) {
    assert(state.rawArtifacts[id], 'Missing completed raw reference');
    const raw = registerRaw(id).envelope.result;
    const interrupted = state.interruption?.path === relative(ROOT, join(root, `${id}-interrupted.json`));
    if (!['NOT_RUN', 'INTERRUPTED'].includes(raw.outcome) && !interrupted) assert(state.analysisArtifacts[id], 'Missing completed analysis');
    if (state.analysisArtifacts[id]) {
      const analysis = readEnvelope(id, true);
      assert.deepEqual(analysis.reference, state.analysisArtifacts[id], 'Immutable analysis changed');
      assert.deepEqual(analysis.envelope.result.raw, state.rawArtifacts[id]);
      assert(['PASS', 'FAIL', 'UNRESOLVED'].includes(analysis.envelope.result.outcome));
      if (!state.stopReason) assert.equal(analysis.envelope.result.outcome, 'PASS', 'Non-PASS analysis cannot resume binding collection');
    } else if (!state.stopReason) assert.fail('Non-analyzed attempt cannot resume binding collection');
  }
  for (const id of Object.keys(state.analysisArtifacts)) {
    assert(state.completedExperimentIds.includes(id) || state.activeExperimentId === id);
    assert.deepEqual(readEnvelope(id, true).reference, state.analysisArtifacts[id]);
  }
  verifyReferences(state.interruption); verifyReferences(state.runnerError);
  for (const id of ids) if (!state.completedExperimentIds.includes(id) && state.activeExperimentId !== id) {
    assert(!readdirSync(root).some(name => name === `${id}.json` || name.startsWith(`${id}-`)), 'Unregistered attempt artifacts cannot be reused');
  }
  const stop = reason => {
    state.stopReason ??= reason; state.status = 'STOPPED'; state.requiredNextManualAction = null; save();
    for (const id of ids.slice(state.completedExperimentIds.length)) {
      assert.equal(state.activeExperimentId, null);
      state.activeExperimentId = id; save();
      writeArtifact(root, `${id}.json`, { pin, id, result: { outcome: 'NOT_RUN', reason: state.stopReason } });
      registerRaw(id); state.completedExperimentIds.push(id); state.activeExperimentId = null; save();
    }
  };
  const interrupt = (reason, outcome = 'INTERRUPTED') => {
    const id = state.activeExperimentId;
    if (id !== null) {
      assert.equal(id, ids[state.completedExperimentIds.length]);
      if (!existsSync(join(root, `${id}.json`))) {
        const rawReferences = readdirSync(root).filter(name => name.startsWith(`${id}-`) && !name.endsWith('-analysis.json'))
          .map(name => reference(cachePath(join(root, name))));
        writeArtifact(root, `${id}.json`, { pin, id, result: { outcome, reason, rawReferences } });
      }
      registerRaw(id);
      if (existsSync(join(root, `${id}-analysis.json`))) {
        const analysis = readEnvelope(id, true);
        assert.deepEqual(analysis.envelope.result.raw, state.rawArtifacts[id]);
        state.analysisArtifacts[id] = analysis.reference;
      }
      const name = `${id}-interrupted.json`;
      if (!existsSync(join(root, name))) writeArtifact(root, name, { pin, id, result: { outcome, reason, raw: state.rawArtifacts[id] } });
      const interruption = JSON.parse(readFileSync(join(root, name), 'utf8'));
      assert.deepEqual(interruption.pin, pin); assert.equal(interruption.id, id); verifyReferences(interruption.result);
      state.interruption = reference(join(root, name));
      state.completedExperimentIds.push(id); state.activeExperimentId = null;
    }
    stop(reason);
  };
  if (!retained) save();
  if (state.activeExperimentId !== null) interrupt({ id: state.activeExperimentId, outcome: 'INTERRUPTED', message: 'Process ended with an active attempt; no rerun permitted' });
  else if (state.stopReason) stop(state.stopReason);
  else if (state.completedExperimentIds.length === ids.length) assert.equal(state.status, 'COMPLETE');
  return {
    snapshot: () => structuredClone(state),
    begin(id) {
      assert.equal(state.status, 'READY', 'Only a ready checkpoint can begin an attempt');
      assert.equal(state.stopReason, null); assert.equal(state.activeExperimentId, null);
      assert.equal(id, ids[state.completedExperimentIds.length]);
      state.activeExperimentId = id; state.status = 'RUNNING'; save();
    },
    raw(id, result) {
      assert.equal(state.activeExperimentId, id);
      writeArtifact(root, `${id}.json`, { pin, id, result }); registerRaw(id); save();
      return state.rawArtifacts[id];
    },
    complete(id, analysis) {
      assert.equal(state.activeExperimentId, id); assert(['PASS', 'FAIL', 'UNRESOLVED'].includes(analysis.outcome));
      assert(state.rawArtifacts[id], 'Raw evidence is required before analysis completion');
      const result = { ...analysis, raw: state.rawArtifacts[id] };
      state.analysisArtifacts[id] = writeArtifact(root, `${id}-analysis.json`, { pin, id, result });
      state.completedExperimentIds.push(id); state.activeExperimentId = null;
      state.status = state.completedExperimentIds.length === ids.length ? 'COMPLETE' : 'READY';
      if (result.outcome !== 'PASS') {
        state.stopReason = { id, outcome: result.outcome, reason: result.summary ?? result.error ?? 'Analyzer did not pass' };
        state.status = 'STOPPED';
      }
      save();
      if (state.stopReason) stop(state.stopReason);
    },
    fail(error) {
      const reason = { id: state.activeExperimentId, outcome: 'UNRESOLVED', ...errorInfo(error) };
      state.runnerError = writeArtifact(root, `runner-error-${randomUUID()}.json`, { pin, id: 'runner-error', result: reason });
      interrupt(reason, 'UNRESOLVED');
    },
  };
}

async function bounded(operation, milliseconds, label, signal) {
  let timer, aborted;
  try {
    return await Promise.race([operation, new Promise((resolveValue, reject) => {
      aborted = () => reject(signal.reason);
      timer = setTimeout(() => reject(new Error(`${label}: ${milliseconds}ms timeout`)), milliseconds);
      signal?.addEventListener('abort', aborted, { once: true });
      if (signal?.aborted) aborted();
    })]);
  } finally { clearTimeout(timer); if (aborted) signal?.removeEventListener('abort', aborted); }
}

export function analyzeCalibration(raw, { executable = join(ROOT, PYTHON), script = join(ROOT, ANALYZER) } = {}) {
  let stdout = '';
  try {
    const envelope = JSON.parse(verifyReference(raw).toString('utf8'));
    verifyReferences(envelope.result);
    stdout = execFileSync(executable, [script, cachePath(raw.path)], { cwd: ROOT, encoding: 'utf8',
      maxBuffer: 32 * 1024 * 1024, timeout: 120000, killSignal: 'SIGKILL' });
    verifyReference(raw); verifyReferences(envelope.result);
    const analysis = JSON.parse(stdout);
    assert(analysis && typeof analysis === 'object' && !Array.isArray(analysis));
    assert(['PASS', 'FAIL', 'UNRESOLVED'].includes(analysis.outcome), 'Analyzer must emit a recognized outcome');
    return analysis;
  } catch (error) {
    return { outcome: 'UNRESOLVED', summary: 'Offline analyzer could not produce a valid verdict', error: errorInfo(error),
      stdout: stdout || String(error.stdout ?? ''), stderr: String(error.stderr ?? '') };
  }
}

function acquireLock() {
  const path = cachePath('.cache/m1010r/calibration.lock');
  mkdirSync(join(ROOT, '.cache/m1010r'), { recursive: true });
  if (existsSync(path)) {
    const previous = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(previous.generator, 'm1010r-calibration'); assert(Number.isSafeInteger(previous.pid) && previous.pid > 0);
    try { process.kill(previous.pid, 0); throw new Error('Another calibration runner is active'); }
    catch (error) { if (error.code !== 'ESRCH') throw error; }
    unlinkSync(path);
  }
  const token = randomUUID();
  writeFileSync(path, json({ generator: 'm1010r-calibration', pid: process.pid, token }), { flag: 'wx' });
  return () => { assert.equal(JSON.parse(readFileSync(path, 'utf8')).token, token); unlinkSync(path); };
}

async function collect(page, native, entry, directory, result, signal) {
  const recordError = (stage, error) => result.errors.push({ stage, ...errorInfo(error) });
  try {
    await bounded(page.bringToFront(), 10000, 'Foreground', signal);
    result.window = await bounded(nativeWindow(page, native.context), 10000, 'Native window', signal);
    const url = new URL(`chrome-extension://${native.extensionId}/calibration.html`);
    url.searchParams.set('fps', String(entry.fps)); url.searchParams.set('media', `media/replay-${entry.fps}.mp4`);
    result.url = url.href;
    await bounded(page.goto(url.href, { timeout: 10000 }), 10000, 'Navigation', signal);
    await bounded(page.waitForFunction(() => typeof window.m1010rCalibration?.snapshot === 'function', undefined,
      { timeout: 10000 }), 10000, 'Calibration API', signal);
    await bounded(page.locator('#play').click({ timeout: 10000 }), 10000, 'Ordinary Play button', signal);
    await bounded(page.waitForFunction(() => ['RECORDED', 'UNRESOLVED'].includes(document.getElementById('status')?.value),
      undefined, { timeout: 100000, polling: 250 }), 101000, 'Collection and drain', signal);
  } catch (error) { recordError('collection', error); }
  try {
    result.report = await bounded(page.evaluate(() => window.m1010rCalibration.snapshot()), 10000, 'Raw snapshot');
    result.rawSnapshot = writeArtifact(directory, `${entry.id}-snapshot.json`, result.report);
  } catch (error) { recordError('snapshot', error); }
  for (const [control, name, key] of [[false, 'audio', 'audio'], [true, 'control-audio', 'audioControl']]) {
    try {
      const base64 = await bounded(page.evaluate(control => window.m1010rCalibration.audioBase64(control), control), 10000, name);
      assert(typeof base64 === 'string', 'Missing PCM base64');
      const bytes = Buffer.from(base64, 'base64');
      const pcm = writeArtifact(directory, `${entry.id}-${name}.f32le`, bytes);
      result[control ? 'controlAudioPcm' : 'audioPcm'] = pcm;
      if (result.report?.[key]) result.report[key].pcm = pcm;
      assert(bytes.length > 0 && bytes.length % 4 === 0 && bytes.toString('base64') === base64, 'Invalid PCM encoding');
      assert.equal(bytes.length, result.report?.[key]?.samples * 4, 'PCM length differs from recording metadata');
    } catch (error) { recordError(name, error); }
  }
  try { await bounded(page.close(), 10000, 'Page close'); result.pageClosed = true; }
  catch (error) { recordError('page close', error); }
  result.endedAt = new Date().toISOString();
  result.outcome = result.report?.state === 'RECORDED' && result.errors.length === 0 ? 'RECORDED' : 'UNRESOLVED';
}

export async function runCalibrationStudy(directory = DEFAULT_CALIBRATION, { outdir = DEFAULT_EXTENSION } = {}) {
  assert.equal(resolve(process.cwd()), ROOT, 'Run the native helper from the repository root');
  const identity = studyIdentity(outdir), root = cachePath(directory);
  assert(root !== identity.directory && !root.startsWith(identity.directory + '/') && !identity.directory.startsWith(root + '/'), 'Study and package directories must be disjoint');
  const analyzerPath = join(ROOT, ANALYZER), pythonPath = join(ROOT, PYTHON);
  assert(existsSync(analyzerPath) && existsSync(pythonPath), 'Main must provide and review analyze.py and the existing Python environment before launch');
  process.env.PLAYWRIGHT_BROWSERS_PATH ??= join(ROOT, '.cache/m9/browsers');
  const { chromium } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
  const executable = process.env.M9_CHROME_EXECUTABLE_PATH ?? chromium.executablePath();
  const pin = { studyVersion: 'M10.10R-calibration-1', sourceCommit: identity.sourceCommit, baseline: FROZEN_BASELINE,
    buildSha256: identity.build.sha256, modelSha256: identity.provenance.modelSha256,
    mediaManifestSha256: identity.provenance.mediaManifestSha256, browserExecutable: executable,
    browserExecutableSha256: digest(readFileSync(executable)), analyzerSha256: digest(readFileSync(analyzerPath)),
    pythonExecutableSha256: digest(readFileSync(pythonPath)), order: ids };
  const release = acquireLock(), abort = new AbortController();
  const onInterrupt = () => abort.abort(new Error('Runner interrupted by SIGINT/SIGTERM; no repeat allowed'));
  let checkpoint, native, machine, browserInfo;
  process.once('SIGINT', onInterrupt); process.once('SIGTERM', onInterrupt);
  try {
    checkpoint = openCalibrationCheckpoint(root, pin);
    if (['COMPLETE', 'STOPPED'].includes(checkpoint.snapshot().status)) return checkpoint.snapshot();
    for (const entry of CALIBRATION_CASES.slice(checkpoint.snapshot().completedExperimentIds.length)) {
      checkpoint.begin(entry.id); console.log(JSON.stringify({ running: entry.id }));
      const result = { outcome: 'UNRESOLVED', startedAt: new Date().toISOString(), fps: entry.fps, repeat: entry.repeat,
        scope: 'No-neural instrument calibration; not candidate qualification or physical A/V latency',
        build: identity.build, media: identity.provenance.mediaManifest.assets.find(asset => asset.fps === entry.fps),
        environment: null, report: null, rawSnapshot: null, audioPcm: null, controlAudioPcm: null, pageClosed: false, errors: [] };
      try {
        if (abort.signal.aborted) throw abort.signal.reason;
        const currentIdentity = studyIdentity(outdir);
        assert.equal(currentIdentity.sourceCommit, pin.sourceCommit, 'Source pin changed between attempts');
        assert.deepEqual(currentIdentity.build, identity.build, 'Package changed between attempts');
        machine ??= environment(); result.environment = { ...machine, displayRefreshHz: 'not measured', browser: browserInfo ?? null };
        if (!native) {
          const flags = [`--load-extension=${identity.directory}`];
          native = await openNativeChrome(flags);
          assert.equal(digest(readFileSync(native.executable)), pin.browserExecutableSha256);
          browserInfo = { executable: native.executable, executableSha256: pin.browserExecutableSha256, flags,
            launcherSha256: digest(readFileSync(join(ROOT, 'tools/m9-browser.mjs'))),
            policy: 'Default autoplay; temporary profile; no owner invocation, permission override or chooser' };
          result.environment.browser = browserInfo;
          native.context.setDefaultTimeout(10000);
          const matches = worker => /^chrome-extension:\/\/[a-p]{32}\/service-worker\.js$/.test(worker.url());
          const worker = native.context.serviceWorkers().find(matches) ?? await native.context.waitForEvent('serviceworker', { predicate: matches, timeout: 10000 });
          native.extensionId = new URL(worker.url()).hostname;
          const installed = await bounded(worker.evaluate(() => chrome.runtime.getManifest()), 10000, 'Installed manifest', abort.signal);
          assert.equal(installed.name, 'AetherVSR M10.10R Calibration'); assert.equal(installed.version, '0.0.1');
          for (const key of ['permissions', 'optional_permissions', 'host_permissions', 'optional_host_permissions', 'web_accessible_resources']) assert(!(key in installed));
          const cdp = await native.browser.newBrowserCDPSession();
          try {
            const version = await cdp.send('Browser.getVersion');
            const processes = await cdp.send('SystemInfo.getProcessInfo');
            const browserProcess = processes.processInfo.find(process => process.type === 'browser');
            assert(Number.isSafeInteger(browserProcess?.id) && browserProcess.id > 0);
            Object.assign(browserInfo, { version: await native.browser.version(), protocol: version,
              commandLine: execFileSync('ps', ['-ww', '-p', String(browserProcess.id), '-o', 'command='], { encoding: 'utf8' }).trim(),
              workerUrl: worker.url(), manifest: installed });
          } finally { await cdp.detach(); }
        }
        result.environment.browser = browserInfo;
        const page = await native.context.newPage();
        await collect(page, native, entry, root, result, abort.signal);
      } catch (error) { result.errors.push({ stage: 'setup', ...errorInfo(error) }); result.endedAt = new Date().toISOString(); }
      const raw = checkpoint.raw(entry.id, result);
      let analysis;
      if (!result.pageClosed || abort.signal.aborted) analysis = { outcome: 'UNRESOLVED', summary: 'Setup, page closure or interruption prevents offline acceptance', errors: result.errors };
      else {
        assert.equal(digest(readFileSync(analyzerPath)), pin.analyzerSha256, 'Analyzer changed during collection');
        assert.equal(digest(readFileSync(pythonPath)), pin.pythonExecutableSha256, 'Analyzer interpreter changed');
        analysis = analyzeCalibration(raw);
        if (result.outcome !== 'RECORDED' && analysis.outcome === 'PASS') analysis = { ...analysis, analyzerOutcome: 'PASS', outcome: 'UNRESOLVED', summary: 'Analyzer PASS cannot override recorder/collection errors' };
      }
      checkpoint.complete(entry.id, analysis);
      console.log(JSON.stringify({ completed: entry.id, outcome: analysis.outcome, controlError: result.errors.length ? result.errors : null }));
      if (analysis.outcome !== 'PASS') break;
    }
  } catch (error) {
    checkpoint?.fail(error);
    throw error;
  } finally {
    try { await native?.close(); }
    catch (error) { checkpoint?.fail(error); throw error; }
    finally { process.removeListener('SIGINT', onInterrupt); process.removeListener('SIGTERM', onInterrupt); release(); }
  }
  return checkpoint.snapshot();
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    assert(process.argv.length <= 4, 'Usage: node tools/m1010r/study.mjs [directory] [extension-directory]');
    const state = await runCalibrationStudy(process.argv[2], { outdir: process.argv[3] });
    console.log(JSON.stringify({ completed: state.status, state: join(process.argv[2] ?? DEFAULT_CALIBRATION, 'state.json') }));
    if (state.status !== 'COMPLETE') process.exitCode = 1;
  } catch (error) { console.error(JSON.stringify({ controlError: errorInfo(error) })); process.exitCode = 1; }
}