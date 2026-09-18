import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, lstatSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { arch, hostname, platform, release } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyzeArm, installRecorder, killOwnedGroup, localPath, verifyPrerequisites } from './playback.mjs';
import { waitForObservation } from './parity.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
export const SOAK_DURATION_MS = 600000;
export const SOAK_WARMUP_MS = 5000;
export const SOAK_ARMS = Object.freeze([
  Object.freeze({ id: 'raw60-soak', raw: true, fps: 60, minDurationMs: SOAK_DURATION_MS }),
  Object.freeze({ id: 'neural60-soak', raw: false, fps: 60, minDurationMs: SOAK_DURATION_MS }),
]);
const VERSION = '44.4.1';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', timeout: 10000 }).trim();
const pin = path => { const bytes = readFileSync(localPath(path)); return { path: relative(ROOT, resolve(ROOT, path)), bytes: bytes.length, sha256: digest(bytes) }; };

export function soakOutput(directory, root = ROOT) {
  const output = localPath(directory, root), base = resolve(root, '.cache/m11');
  assert.equal(dirname(output), base, 'Output must be a direct child of .cache/m11');
  assert(!existsSync(output), 'Immutable attempt already exists');
  return output;
}

function containedFile(path, root) {
  const absolute = localPath(path, root), base = resolve(root, '.cache/m11');
  assert(absolute.startsWith(base + sep), 'Long media must stay under .cache/m11');
  assert(lstatSync(absolute).isFile(), 'Long media must be a regular file');
  assert(!lstatSync(dirname(absolute)).isSymbolicLink(), 'Long media parent must not be a symlink');
  return absolute;
}

export function validateLongMediaManifest(manifest, mediaPath, root = ROOT, runProbe = null) {
  assert.equal(manifest?.schema, 'aethervsr.m11.long-media/1');
  assert(Number.isInteger(manifest.seconds) && manifest.seconds >= 610, 'Long media must be at least 610 seconds');
  assert.equal(manifest.fps, 60); assert.equal(manifest.width, 1280); assert.equal(manifest.height, 720);
  assert(Number.isSafeInteger(manifest.frames), 'Exact frame count required');
  assert(typeof manifest.bytes === 'number' && Number.isSafeInteger(manifest.bytes) && manifest.bytes > 0);
  assert(/^[a-f0-9]{64}$/.test(manifest.sha256));
  const path = containedFile(mediaPath, root);
  if (manifest.path !== undefined) assert.equal(manifest.path, relative(root, path));
  const bytes = readFileSync(path); assert.equal(bytes.length, manifest.bytes); assert.equal(digest(bytes), manifest.sha256);
  if (manifest.durationSeconds !== undefined) {
    assert(typeof manifest.durationSeconds === 'number' && Number.isFinite(manifest.durationSeconds));
    assert(manifest.durationSeconds >= 605.5, 'Declared duration must cover warmup plus 600 seconds');
    assert(Math.abs(manifest.durationSeconds * manifest.fps - manifest.frames) <= 1, 'Declared duration is not frame-aligned');
  } else assert(manifest.frames === manifest.seconds * manifest.fps, 'Exact frame count required');
  if (manifest.probe !== undefined) {
    assert(Array.isArray(manifest.probe.command) && manifest.probe.command.length > 0);
    assert(typeof manifest.probe.output === 'string' || /^[a-f0-9]{64}$/.test(manifest.probe.outputSha256));
    if (runProbe) {
      const output = runProbe(manifest.probe.command, path);
      if (manifest.probe.output !== undefined) assert.equal(output, manifest.probe.output);
      if (manifest.probe.outputSha256 !== undefined) assert.equal(digest(Buffer.from(output)), manifest.probe.outputSha256);
    }
  }
  return { ...manifest, path: relative(root, path), durationMs: Math.round((manifest.durationSeconds ?? manifest.seconds) * 1000) };
}

function readJsonReference(reference) {
  assert(reference?.path && Number.isSafeInteger(reference.bytes) && /^[a-f0-9]{64}$/.test(reference.sha256));
  const actual = pin(reference.path); assert.deepEqual(actual, reference); return JSON.parse(readFileSync(localPath(reference.path), 'utf8'));
}

export function verifyShortSoakPrerequisites(short, current, ancestor = null, runGit = git) {
  assert.equal(short?.schema, 'aethervsr.m11.short-playback/1'); assert.equal(short.verdict, 'PASS');
  assert.equal(short.arms?.length, 8);
  const parity = readJsonReference(short.prerequisites?.parity), journeys = readJsonReference(short.prerequisites?.journeys);
  const shortCommit = short.sourceBefore?.commit;
  assert(/^[a-f0-9]{40}$/.test(shortCommit) && /^[a-f0-9]{40}$/.test(current));
  if (ancestor !== null) assert.equal(ancestor, shortCommit, 'Explicit ancestor must be the short result source');
  if (shortCommit !== current) { proofAncestor(shortCommit, ancestor); runGit(['merge-base', '--is-ancestor', shortCommit, current]); }
  const proof = verifyPrerequisites(parity, journeys, current, parity.sourceBefore.commit, runGit);
  assert.equal(shortCommit, proof.runnerCommit); assert.deepEqual(short.sourceAfter, short.sourceBefore);
  assert.deepEqual(short.packageAfter, short.packageBefore);
  assert.deepEqual(short.pinsAfter?.payloadSha256, short.packageBefore?.payloadSha256);
  for (const arm of short.arms) {
    assert.equal(arm.verdict, 'PASS');
    const observation = readJsonReference(arm.observation), analysis = readJsonReference(arm.analysis), metadata = readJsonReference(arm.metadata);
    const expected = analyzeArm({ ...metadata, observation }, { raw: arm.raw, fps: arm.fps, minDurationMs: 60000 });
    assert.deepEqual(expected, analysis); assert.equal(analysis.outcome, 'PASS');
  }
  return proof;
}

function proofAncestor(shortCommit, ancestor) {
  assert.equal(ancestor, shortCommit, 'Explicit ancestor must be the short result source');
  return shortCommit;
}

function nativeState({ app, BrowserWindow, screen }) {
  const window = BrowserWindow.getAllWindows()[0];
  return { at: new Date().toISOString(), versions: process.versions, profile: app.getPath('userData'), count: BrowserWindow.getAllWindows().length,
    bounds: window?.getBounds(), contentBounds: window?.getContentBounds(), visible: window?.isVisible(), focused: window?.isFocused(),
    gpu: app.getGPUFeatureStatus(), display: window && screen.getDisplayMatching(window.getBounds()),
    processes: app.getAppMetrics().map(({ pid, type, cpu, memory }) => ({ pid, type, cpu, memory })) };
}

export async function runSoak(output, { shortPath = '.cache/m11/playback-01/result.json', mediaPath, mediaManifest, ancestor = null } = {}) {
  const directory = soakOutput(output), report = { schema: 'aethervsr.m11.long-soak/1', output: relative(ROOT, directory), verdict: 'FAIL', startedAt: new Date().toISOString(),
    arms: SOAK_ARMS.map(spec => ({ ...spec, verdict: 'NOT_RUN', reason: 'Earlier gate not completed' })), errors: [],
    bounds: { runMs: 1320000, warmupMs: 9000, observationMs: SOAK_DURATION_MS, closeMs: 10000 },
    host: { hostname: hostname(), platform: platform(), release: release(), arch: arch(), node: process.versions },
    scope: 'Software visual-lag, callback, submission, GPU timestamp and available process/resource snapshots only. No physical scanout, A/V or audio synchronization, thermal, watts or GPU-memory claim; process metrics are opening/midpoint/closing snapshots, not summed physical memory.' };
  assert(mediaPath && mediaManifest); const manifest = validateLongMediaManifest(JSON.parse(readFileSync(localPath(mediaManifest), 'utf8')), mediaPath, ROOT,
    (command, path) => execFileSync(command[0], command.slice(1).map(value => value === '{path}' ? path : value), { encoding: 'utf8', timeout: 30000 }));
  const current = git(['rev-parse', 'HEAD']); verifyShortSoakPrerequisites(JSON.parse(readFileSync(localPath(shortPath), 'utf8')), current, ancestor);
  const write = (name, value) => { const bytes = Buffer.from(JSON.stringify(value, null, 2) + '\n'); const path = join(directory, name); writeFileSync(path, bytes, { flag: 'wx' }); return { path: relative(ROOT, path), bytes: bytes.length, sha256: digest(bytes) }; };
  mkdirSync(directory); report.media = { ...manifest, file: pin(mediaPath) }; report.prerequisites = pin(shortPath);
  let app = null, page = null, child = null, exited = null, stopping = false;
  const abort = reason => { if (!stopping) { stopping = true; report.errors.push(String(reason)); } };
  const sigint = () => abort('SIGINT'), sigterm = () => abort('SIGTERM'); process.once('SIGINT', sigint); process.once('SIGTERM', sigterm);
  const watchdog = setTimeout(() => abort('Whole soak run deadline'), report.bounds.runMs);
  const close = async record => {
    if (!app) return;
    try { record.cleanup = await page.evaluate(() => window.m11Playback?.cleanup()); } catch (error) { record.errors.push(`Recorder cleanup: ${error}`); }
    try { await app.evaluate(({ BrowserWindow }) => { if (BrowserWindow.getAllWindows().length !== 1) throw Error('Unexpected window count'); setImmediate(() => BrowserWindow.getAllWindows()[0].close()); }); record.exit = await exited; }
    catch (error) { record.errors.push(`Natural close failed: ${error}`); record.forcedKill = true; try { record.groupKilled = killOwnedGroup(child.pid, record.processGroup); } catch (failure) { record.errors.push(`Owned group cleanup: ${failure}`); } }
    if (record.profile && existsSync(record.profile)) { try { rmSync(record.profile, { recursive: true }); } catch (error) { record.errors.push(`Profile cleanup: ${error}`); } }
    record.profileRemoved = !record.profile || !existsSync(record.profile); app = null; page = null; child = null;
  };
  try {
    const { buildDesktop } = await import('../../apps/desktop/build.mjs'); const build = localPath('.cache/m11/soak-app'); if (!existsSync(build)) await buildDesktop({ diagnostic: true, outdir: build });
    const executable = realpathSync((await import('electron')).default); const { _electron } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
    for (const spec of SOAK_ARMS) {
      if (stopping) break; const active = report.arms.find(row => row.id === spec.id); active.verdict = 'FAIL'; const profile = join(directory, `${spec.id}-profile`);
      const record = { errors: [], console: [], profile, profileRemoved: false, mediaManifest: manifest };
      let fatal = false;
      try {
        mkdirSync(profile); const env = { ...process.env, AETHERVSR_TEST_PROFILE: profile }; for (const name of ['NODE_OPTIONS', 'ELECTRON_RUN_AS_NODE', 'ELECTRON_EXTRA_LAUNCH_ARGS']) delete env[name];
        app = await _electron.launch({ executablePath: executable, args: [build], env, timeout: 30000 }); child = app.process(); exited = new Promise(done => child.once('close', (code, signal) => done({ code, signal })));
        record.processGroup = Number(execFileSync('ps', ['-o', 'pgid=', '-p', String(child.pid)], { encoding: 'utf8', timeout: 5000 }).trim()); assert.equal(record.processGroup, child.pid);
        page = await app.firstWindow({ timeout: 10000 }); page.setDefaultTimeout(10000); page.on('pageerror', error => { record.errors.push(String(error)); record.fatalError = true; });
        page.on('crash', () => { record.errors.push('Renderer crashed'); record.fatalError = true; }); page.on('console', message => { if (message.type() === 'error') record.errors.push(message.text()); });
        await page.waitForURL('aethervsr://app/index.html'); await app.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setBounds({ width: 1280, height: 720 }); window.show(); window.focus(); }); await page.bringToFront();
        await waitForObservation(() => page.evaluate(() => typeof window.m11Desktop !== 'undefined')); await page.evaluate(installRecorder, { raw: spec.raw, durationMs: SOAK_DURATION_MS });
        await page.locator('#file').setInputFiles(resolve(ROOT, mediaPath)); record.selected = await page.evaluate(() => window.m11Playback.selected); assert.equal(record.selected.bytes, manifest.bytes); assert.equal(record.selected.sha256, manifest.sha256);
        await waitForObservation(() => page.evaluate(() => window.m11Desktop.video.readyState >= 2)); record.warm = await page.evaluate(() => window.m11Playback.warm()); record.nativeBefore = await app.evaluate(nativeState);
        const midpoint = setTimeout(async () => { try { record.nativeMiddle = await app.evaluate(nativeState); } catch (error) { record.errors.push(`Midpoint snapshot: ${error}`); } }, SOAK_DURATION_MS / 2);
        record.observation = await page.evaluate(() => window.m11Playback.start()); clearTimeout(midpoint); record.nativeAfter = await app.evaluate(nativeState);
      } catch (error) { fatal = true; record.errors.push(String(error)); try { record.observation ??= await page?.evaluate(() => window.m11Playback?.partial()); } catch (failure) { record.errors.push(String(failure)); } }
      finally {
        if (!record.observation) record.observation = { complete: false, errors: record.errors, unavailable: true }; active.observation = write(`${spec.id}-observation.json`, record.observation); await close(record);
        const analysis = analyzeArm(record, spec); active.analysis = write(`${spec.id}-analysis.json`, analysis); const { observation: _observation, ...metadata } = record; active.metadata = write(`${spec.id}-metadata.json`, metadata); active.verdict = analysis.outcome; if (!analysis.criteria.cleanup) fatal = true;
      }
      if (fatal) { active.reason = 'Fatal setup, acquisition, or cleanup failure; no retry'; break; }
    }
    assert.equal(git(['rev-parse', 'HEAD']), current, 'Runner source changed during soak');
    validateLongMediaManifest(JSON.parse(readFileSync(localPath(mediaManifest), 'utf8')), mediaPath, ROOT,
      (command, path) => execFileSync(command[0], command.slice(1).map(value => value === '{path}' ? path : value), { encoding: 'utf8', timeout: 30000 }));
    report.verdict = !stopping && report.arms.every(row => row.verdict === 'PASS') ? 'PASS' : 'FAIL';
  } catch (error) { report.errors.push(String(error)); }
  finally { clearTimeout(watchdog); if (app) await close({ errors: report.errors, profile: null }); report.finishedAt = new Date().toISOString(); write('result.json', report); process.off('SIGINT', sigint); process.off('SIGTERM', sigterm); }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === '--help') console.log('node tools/m11/soak.mjs OUTPUT SHORT_RESULT LONG_MEDIA MANIFEST [EXACT_EVIDENCE_ANCESTOR]');
  else { assert(process.argv.length >= 6 && process.argv.length <= 7, 'Expected OUTPUT SHORT_RESULT LONG_MEDIA MANIFEST [EXACT_EVIDENCE_ANCESTOR]'); const report = await runSoak(process.argv[2], { shortPath: process.argv[3], mediaPath: process.argv[4], mediaManifest: process.argv[5], ancestor: process.argv[6] ?? null }); console.log(JSON.stringify({ output: report.output, verdict: report.verdict, arms: report.arms.map(({ id, verdict }) => ({ id, verdict })), errors: report.errors })); process.exitCode = report.verdict === 'PASS' ? 0 : 1; }
}