import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, lstatSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { arch, hostname, platform, release } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MODEL_SHA256, summarizeParity, waitForObservation } from './parity.mjs';
import { verifyJourneyParity } from './journeys.mjs';
import { analyzePlayback } from './metrics.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url)), BUILD = '.cache/m11/playback-app', VERSION = '44.4.1';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', timeout: 10000 }).trim();
const finite = value => typeof value === 'number' && Number.isFinite(value);
const PAYLOAD = ['index.html', 'main.cjs', 'models/production.json', 'package.json', 'player.css', 'renderer.js'];
const JOURNEYS = ['play', 'pause', 'resume', 'forward', 'backward', 'audio', 'baseline', 'neural', 'auto', 'rate',
  'resize', 'fullscreen', 'replace', 'device-loss', 'recover', 'security', 'close',
  'close-paused', 'close-baseline', 'close-neural', 'close-seek', 'close-replacement'];
export const MEDIA = Object.freeze([
  Object.freeze({ fps: 30, path: '.cache/m1010r/media-02/replay-30.mp4', sha256: 'e006f3d5381d73b1ca5739f5e74216f4bf0c312f28621634d258a770822a8f9b' }),
  Object.freeze({ fps: 60, path: '.cache/m1010r/media-02/replay-60.mp4', sha256: '5171a8b6da7303c8c409167f4191b7330f41120fcd89d7e3aa0df9483b237b29' }),
]);
export const SHORT = Object.freeze(['raw30', 'neural30', 'raw60-1', 'neural60-1', 'neural60-2', 'raw60-2', 'raw60-3', 'neural60-3']
  .map(id => Object.freeze({ id, raw: id.startsWith('raw'), fps: id.includes('30') ? 30 : 60 })));

export function killOwnedGroup(pid, group, signal = process.kill.bind(process)) {
  assert(Number.isSafeInteger(pid) && pid > 1 && pid !== process.pid && group === pid, 'Unverified owned process group');
  try { signal(-pid, 'SIGKILL'); return true; }
  catch (error) { if (error.code === 'ESRCH') return false; throw error; }
}
export function recoverableWarmupFailure(error, environment, spec, fatalError = false) {
  const video = environment?.video, session = environment?.session;
  return !spec.raw && !fatalError && String(error).includes('M11_NEURAL_WARMUP:') &&
    environment?.visibility === 'visible' && environment.focused === true && video?.paused === false &&
    video.ended === false && video.error === null && video.time > 0 && video.width === 1280 && video.height === 720 &&
    video.rate === 1 && video.muted === false && video.volume === 1 && session?.mode === 'auto' &&
    !session.error && !session.observerError && session.cleanupErrors?.length === 0 && session.runtime?.running === true &&
    ['warmup', 'stable', 'fallback', 'probing'].includes(session.runtime.controller?.state) &&
    ['neural', 'baseline'].includes(session.runtime.actualTier) &&
    ['devices', 'pipelines', 'drivers', 'callbacks', 'objectUrls'].every(key => session.resources?.[key] === 1);
}

export function localPath(path, root = ROOT) {
  assert(typeof path === 'string' && path.length && !path.includes('\\') && !path.split('/').includes('..'), 'Noncanonical path');
  const absolute = resolve(root, path), base = realpathSync(root);
  assert(absolute.startsWith(resolve(root) + sep), 'Path outside workspace');
  let parent = absolute;
  while (!existsSync(parent)) {
    try { assert(!lstatSync(parent).isSymbolicLink(), 'Dangling symlink'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    parent = dirname(parent);
  }
  assert.equal(realpathSync(parent), resolve(base, relative(root, parent)), 'Symlinked path');
  return absolute;
}
export function playbackOutput(directory, root = ROOT) {
  const output = localPath(directory, root), build = resolve(root, BUILD);
  assert.equal(dirname(output), resolve(root, '.cache/m11'), 'Output must be a direct child of .cache/m11');
  assert(output !== build && !output.startsWith(build + sep) && !build.startsWith(output + sep), 'Build overlap');
  return output;
}
export function shortMedia(fps, path = MEDIA.find(row => row.fps === fps)?.path) {
  const media = MEDIA.find(row => row.fps === fps);
  assert(media && path === media.path, 'Exact short media path required');
  return media;
}
export function verifyPayload(pin) {
  assert(pin?.diagnostic === true && pin.sourceDirty === false && pin.electron === VERSION && pin.modelSha256 === MODEL_SHA256, 'Invalid diagnostic pin');
  assert.deepEqual(Object.keys(pin.files ?? {}).sort(), PAYLOAD);
  for (const value of Object.values(pin.files)) assert(Number.isSafeInteger(value.bytes) && value.bytes > 0 && /^[a-f0-9]{64}$/.test(value.sha256), 'Invalid payload digest');
  assert.equal(pin.payloadSha256, hash(JSON.stringify(pin.files)), 'Payload manifest digest mismatch');
  assert(Array.isArray(pin.inputs) && pin.inputs.includes('apps/desktop/renderer.ts') && !pin.inputs.includes('apps/desktop/smoke.ts'));
  assert.equal(pin.files['models/production.json'].sha256, MODEL_SHA256);
  return pin.files;
}
export function verifyPrerequisites(parity, journeys, current, ancestor = null, runGit = git) {
  const evidence = parity?.sourceBefore?.commit;
  assert(/^[a-f0-9]{40}$/.test(evidence) && /^[a-f0-9]{40}$/.test(current), 'Exact commits required');
  verifyJourneyParity(parity, evidence);
  assert(summarizeParity(parity.cases).parityPrerequisitePassed);
  assert.equal(journeys?.schema, 'aethervsr.m11.journeys/1'); assert.equal(journeys.verdict, 'PASS');
  assert.equal(journeys.parityPrerequisitePassed, true); assert.deepEqual(journeys.errors, []);
  assert.equal(journeys.sourceBefore?.commit, evidence); assert.deepEqual(journeys.sourceAfter, journeys.sourceBefore);
  assert.deepEqual(journeys.cases?.map(row => row.id), JOURNEYS); assert(journeys.cases.every(row => row.verdict === 'PASS'));
  assert(journeys.cleanup?.naturalExits === true && journeys.cleanup.profilesRemoved === true);
  assert.equal(journeys.applications?.length, 6);
  for (const app of journeys.applications) {
    assert.deepEqual(app.errors, []); assert(app.profileRemoved === true && !app.forcedCleanup);
    assert(app.exit?.code === 0 && app.exit.signal === null && app.native?.versions?.electron === VERSION);
  }
  for (const report of [parity, journeys]) {
    verifyPayload(report.packageBefore); assert.equal(report.packageBefore.sourceCommit, evidence);
    assert.deepEqual(report.packageAfter, report.packageBefore);
  }
  assert.deepEqual(journeys.packageBefore.files, parity.packageBefore.files, 'Prerequisite player payload differs');
  assert(/^[a-f0-9]{64}$/.test(journeys.binary?.sha256), 'Missing prerequisite binary hash');
  assert.equal(journeys.binary?.sha256, parity.binaries?.electron?.sha256);
  if (current !== evidence) {
    assert.equal(ancestor, evidence, 'Explicit exact evidence ancestor required');
    runGit(['merge-base', '--is-ancestor', evidence, current]);
    const paths = [...new Set(['apps/desktop', 'src', 'public/models', 'package.json', 'package-lock.json',
      'tools/m11/renderer-diagnostics.ts', 'tools/m11/parity.mjs', 'tools/m11/journeys.mjs', ...parity.packageBefore.inputs])];
    for (const path of paths) assert(typeof path === 'string' && !path.startsWith('/') && !path.split('/').includes('..'), 'Invalid input path');
    assert.equal(runGit(['diff', '--name-only', evidence, current, '--', ...paths]), '', 'Measured player source changed');
  } else assert(ancestor === null || ancestor === evidence, 'Wrong evidence ancestor');
  return { evidenceCommit: evidence, runnerCommit: current, equivalence: current === evidence ? 'same HEAD' : 'explicit ancestor plus unchanged inputs and payload' };
}

export function qualityDelta(before, after) {
  const result = {};
  for (const key of ['totalVideoFrames', 'droppedVideoFrames', 'corruptedVideoFrames']) {
    result[key] = Number.isSafeInteger(before?.[key]) && before[key] >= 0 && Number.isSafeInteger(after?.[key]) && after[key] >= before[key]
      ? after[key] - before[key] : null;
  }
  return result;
}
export function analyzeRaw(record, { fps, minDurationMs = 60000 } = {}) {
  assert(fps === 30 || fps === 60);
  const start = record?.startAt, stop = record?.stopAt, duration = finite(start) && finite(stop) && stop > start ? stop - start : null;
  const rows = (record?.rows ?? []).filter(row => finite(row.at) && row.at >= start && row.at < stop);
  const valid = row => ['at', 'mediaTime', 'presentationTime', 'expectedDisplayTime'].every(key => finite(row[key]) && row[key] >= 0) &&
    Number.isSafeInteger(row.presentedFrames) && row.presentedFrames >= 0 && row.width === 1280 && row.height === 720 && row.visibility === 'visible' && row.focused === true;
  let gaps = 0, regressions = 0;
  for (let index = 1; index < rows.length; index++) {
    const previous = rows[index - 1], row = rows[index];
    if (row.at <= previous.at || row.mediaTime <= previous.mediaTime || row.presentedFrames <= previous.presentedFrames) regressions++;
    if (Number.isSafeInteger(row.presentedFrames) && Number.isSafeInteger(previous.presentedFrames)) gaps += Math.max(0, row.presentedFrames - previous.presentedFrames - 1);
  }
  const windows = {};
  if (duration !== null) for (const [name, from, to] of [['overall', start, stop], ['first20s', start, Math.min(stop, start + 20000)],
    ['middle20s', Math.max(start, (start + stop) / 2 - 10000), Math.min(stop, (start + stop) / 2 + 10000)], ['final20s', Math.max(start, stop - 20000), stop]]) {
    const selected = rows.filter(row => row.at >= from && row.at < to);
    windows[name] = { startAt: from, stopAt: to, callbacks: selected.length, callbackFps: selected.length * 1000 / (to - from) };
  }
  const criteria = { duration: duration !== null && duration >= minDurationMs, acquisition: rows.length > 0 && rows.every(valid) && regressions === 0,
    rowCoverage: Array.isArray(record?.rows) && record.rows.every(row => finite(row.at)),
    noPipeline: record?.pipelineAbsent === true, complete: record?.complete === true,
    errors: Array.isArray(record?.errors) && record.errors.length === 0 };
  return { outcome: Object.values(criteria).every(Boolean) ? 'PASS' : 'FAIL', criteria,
    metrics: { durationMs: duration, windows, callbackPresentedGaps: rows.length ? gaps : null, regressions,
      qualityDelta: qualityDelta(record?.qualityBefore, record?.qualityAfter), gpuMs: null, textureIdentity: null,
      scope: 'Native video callback acquisition only; no GPU pipeline, texture readback, physical scanout or A/V claim. Decoder drops and callback presented gaps are distinct, never summed. Raw cadence does not qualify neural throughput.' } };
}

export function installRecorder({ raw }) {
  const api = window.m11Desktop, video = api.video, session = raw ? api.session() : api.replace({ observe: true });
  if (session.runtime || session.gpu || session.snapshot().pending) throw Error('Recorder must precede GPU setup');
  if (raw) { session.destroy(); document.querySelector('#empty').remove(); api.canvas.hidden = true; }
  const record = { raw, rows: [], errors: [], events: [], warmup: [], complete: false, startAt: null, stopAt: null,
    pipelineAbsent: raw, qualityBefore: null, qualityAfter: null };
  let phase = 'idle', callback = null, timer = null, warmTimer = null, warmDeadline = null, url = null, finishPromise = null;
  let driver = null, oldSample = null, oldChange = null, stableAt = null, lastSample = null, resolveWarm, rejectWarm;
  const quality = () => {
    if (!video.getVideoPlaybackQuality) return null;
    const value = video.getVideoPlaybackQuality();
    return { at: performance.now(), creationTime: value.creationTime, totalVideoFrames: value.totalVideoFrames,
      droppedVideoFrames: value.droppedVideoFrames, corruptedVideoFrames: value.corruptedVideoFrames };
  };
  const environment = () => ({ at: performance.now(), visibility: document.visibilityState, focused: document.hasFocus(),
    width: innerWidth, height: innerHeight, outerWidth, outerHeight, x: screenX, y: screenY, dpr: devicePixelRatio,
    userAgent: navigator.userAgent, screen: { width: screen.width, height: screen.height },
    video: { time: video.currentTime, duration: video.duration, width: video.videoWidth, height: video.videoHeight,
      rate: video.playbackRate, muted: video.muted, volume: video.volume, paused: video.paused, ended: video.ended, loop: video.loop,
      error: video.error ? { code: video.error.code, message: video.error.message } : null },
    session: session.snapshot(), adapter: session.gpu?.adapterReport ?? null,
    stats: session.runtime?.pipeline.stats?.(performance.now()) ?? null,
    features: session.gpu ? [...session.gpu.device.features] : [], canvas: { width: api.canvas.width, height: api.canvas.height, hidden: api.canvas.hidden } });
  const available = () => document.visibilityState === 'visible' && document.hasFocus() && !video.paused && !video.ended && !video.error &&
    video.videoWidth === 1280 && video.videoHeight === 720 && video.playbackRate === 1 && !video.muted && video.volume === 1 && !video.loop;
  const neural = () => {
    const state = session.snapshot();
    return state.mode === 'auto' && state.ready && !state.pending && !state.error && !state.observerError && !state.cleanupErrors.length &&
      state.runtime?.running && state.runtime.actualTier === 'neural' && state.runtime.controller.state === 'stable' &&
      state.canvas.width === 2560 && state.canvas.height === 1440 && !api.canvas.hidden;
  };
  const detachWarm = () => {
    clearInterval(warmTimer); clearTimeout(warmDeadline); warmTimer = null; warmDeadline = null;
    if (driver) { driver.onSample = oldSample; driver.onChange = oldChange; }
  };
  const note = event => {
    if (phase !== 'observing') return;
    const value = { type: event.type, at: performance.now(), visibility: document.visibilityState, focused: document.hasFocus() };
    if (record.events.length < 1024) record.events.push(value);
    if (['blur', 'resize', 'visibilitychange', 'pause', 'ended', 'seeking', 'ratechange', 'volumechange', 'error'].includes(event.type)) record.errors.push(`Observation event: ${event.type}`);
  };
  const listeners = [[window, 'blur'], [window, 'resize'], [document, 'visibilitychange'],
    ...['pause', 'ended', 'seeking', 'ratechange', 'volumechange', 'error'].map(type => [video, type])];
  for (const [target, type] of listeners) target.addEventListener(type, note);
  let selectedResolve;
  const selected = new Promise(done => { selectedResolve = done; });
  const input = document.querySelector('#file');
  const onFile = async event => {
    if (raw) event.stopImmediatePropagation();
    try {
      const file = input.files[0];
      if (!file) throw Error('No selected File');
      const bytes = await file.arrayBuffer(), digest = await crypto.subtle.digest('SHA-256', bytes);
      record.file = { name: file.name, bytes: bytes.byteLength, sha256: Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('') };
      if (raw) { url = URL.createObjectURL(file); video.src = url; video.load(); input.value = ''; }
      selectedResolve(record.file);
    } catch (error) { record.errors.push(String(error)); selectedResolve({ error: String(error) }); }
  };
  input.addEventListener('change', onFile, { capture: true, once: true });
  const frame = (_now, metadata) => {
    callback = null;
    if (phase !== 'observing') return;
    if (session.runtime || session.gpu) record.pipelineAbsent = false;
    const number = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
    record.rows.push({ at: performance.now(), mediaTime: number(metadata.mediaTime), presentationTime: number(metadata.presentationTime),
      expectedDisplayTime: number(metadata.expectedDisplayTime), presentedFrames: number(metadata.presentedFrames),
      width: number(metadata.width), height: number(metadata.height), visibility: document.visibilityState, focused: document.hasFocus() });
    if (record.rows.length >= 65536) { void stop('Raw callback capacity'); return; }
    callback = video.requestVideoFrameCallback(frame);
  };
  const partial = () => {
    const diagnostics = api.diagnostics;
    return { ...record, ...(raw ? {} : { startAt: diagnostics.startAt, stopAt: diagnostics.stopAt ?? record.stopAt,
      rows: diagnostics.rows ?? [], samples: diagnostics.samples ?? [], states: diagnostics.states ?? [],
      errors: [...record.errors, ...(diagnostics.errors ?? []), 'Partial diagnostic readback; not qualified'], snapshot: session.snapshot() }), complete: false };
  };
  const stop = reason => {
    if (finishPromise) return finishPromise;
    phase = 'finished'; detachWarm(); clearTimeout(timer);
    if (callback !== null) video.cancelVideoFrameCallback(callback); callback = null;
    record.stopAt = performance.now(); record.after = environment(); record.qualityAfter = quality();
    record.complete = reason === 'timer' && record.startAt !== null && record.stopAt - record.startAt >= 60000;
    if (reason !== 'timer') record.errors.push(reason);
    rejectWarm?.(Error(reason));
    finishPromise = (async () => {
      if (raw) { video.pause(); return record; }
      video.pause();
      const observation = await api.diagnostics.finish();
      return { ...record, ...observation, diagnosticsErrors: observation.errors, errors: [...record.errors, ...observation.errors], complete: record.complete };
    })();
    return finishPromise;
  };
  window.m11Playback = {
    selected, environment, partial, stop,
    warm() {
      if (phase !== 'idle') throw Error('Single warmup only');
      phase = 'warming'; record.playAt = performance.now();
      const promise = new Promise((done, reject) => { resolveWarm = done; rejectWarm = reject; });
      const observe = () => {
        const now = performance.now();
        if (!raw && session.runtime && !driver) {
          driver = session.runtime; oldSample = driver.onSample; oldChange = driver.onChange;
          driver.onSample = sample => {
            oldSample?.(sample);
            if (sample.neural && Number.isFinite(sample.ms) && sample.ms >= 0 && Number.isFinite(sample.resolvedAt) &&
              Number.isFinite(sample.submittedAt) && sample.submittedAt >= record.playAt && sample.resolvedAt >= sample.submittedAt &&
              sample.resolvedAt <= performance.now() && Number.isSafeInteger(sample.sequence) && sample.sequence > 0 &&
              sample.generation === driver.pipeline.timingGeneration) lastSample = { ...sample };
          };
          driver.onChange = state => { oldChange?.(state); if (state.state !== 'stable' || !neural()) stableAt = null; };
        }
        const good = available() && (raw ? !session.gpu && !session.runtime : driver === session.runtime && neural() && lastSample !== null && lastSample.generation === driver.pipeline.timingGeneration && now - lastSample.resolvedAt <= 500);
        if (!good) stableAt = null; else stableAt ??= now;
        record.warmup.push({ at: now, good, stableAt, sample: lastSample, runtime: session.runtime?.snapshot() ?? null });
        if (stableAt !== null && now - stableAt >= 5000) resolveWarm({ playAt: record.playAt, stableAt, readyAt: now });
      };
      warmDeadline = setTimeout(() => { rejectWarm(Error(`${raw ? 'Raw warmup' : 'M11_NEURAL_WARMUP:'} exceeded 9000ms from play`)); detachWarm(); stableAt = null; }, 9000);
      warmTimer = setInterval(observe, 25);
      Promise.resolve(raw ? video.play() : session.play()).catch(rejectWarm);
      return promise;
    },
    start() {
      const now = performance.now();
      if (phase !== 'warming' || stableAt === null || now - stableAt < 5000 || now - record.playAt > 9000 || !available() ||
        (!raw && (!neural() || driver !== session.runtime || !lastSample || now - lastSample.resolvedAt > 500)) || video.duration - video.currentTime < 60.5) throw Error('No bounded five-second actual warmup');
      detachWarm(); record.before = environment(); record.qualityBefore = quality();
      if (!raw) api.diagnostics.begin();
      record.startAt = raw ? performance.now() : api.diagnostics.startAt;
      phase = 'observing';
      const result = new Promise((done, reject) => { timer = setTimeout(() => { stop('timer').then(done, reject); }, 60000); });
      if (raw) callback = video.requestVideoFrameCallback(frame);
      return result;
    },
    cleanup() {
      phase = 'finished'; detachWarm(); clearTimeout(timer);
      if (callback !== null) video.cancelVideoFrameCallback(callback); callback = null;
      for (const [target, type] of listeners) target.removeEventListener(type, note);
      input.removeEventListener('change', onFile, { capture: true });
      video.pause(); session.destroy();
      if (url !== null) { URL.revokeObjectURL(url); url = null; }
      video.removeAttribute('src'); video.load();
      return { snapshot: session.snapshot(), callbackCancelled: callback === null, urlRevoked: url === null,
        sourceReleased: video.getAttribute('src') === null, paused: video.paused, runtimeReleased: !session.runtime && !session.gpu };
    },
  };
}

function filePin(path) {
  const bytes = readFileSync(localPath(path));
  return { path: relative(ROOT, resolve(ROOT, path)), bytes: bytes.length, sha256: hash(bytes) };
}
function sourcePin() {
  assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '', 'Clean tracked runner required');
  git(['ls-files', '--error-unmatch', 'tools/m11/playback.mjs']);
  return { commit: git(['rev-parse', 'HEAD']), tree: git(['rev-parse', 'HEAD^{tree}']), runner: filePin('tools/m11/playback.mjs') };
}
function referencePins(value, pins = new Map()) {
  if (!value || typeof value !== 'object') return pins;
  if (typeof value.path === 'string' && typeof value.sha256 === 'string') {
    const pin = filePin(value.path); assert.equal(pin.sha256, value.sha256, `Evidence changed: ${value.path}`);
    if (value.bytes !== undefined) assert.equal(pin.bytes, value.bytes);
    if (pins.has(pin.path)) assert.deepEqual(pins.get(pin.path), pin);
    pins.set(pin.path, pin);
  }
  for (const [key, item] of Object.entries(value)) if (key !== 'binaries' && key !== 'binary') referencePins(item, pins);
  return pins;
}
function nativeState({ app, BrowserWindow, screen }) {
  const windows = BrowserWindow.getAllWindows(), window = windows[0];
  return { at: new Date().toISOString(), versions: process.versions, argv: process.argv, profile: app.getPath('userData'), count: windows.length,
    bounds: window?.getBounds(), contentBounds: window?.getContentBounds(), visible: window?.isVisible(), focused: window?.isFocused(),
    preferences: window?.webContents.getLastWebPreferences(), gpu: app.getGPUFeatureStatus(), display: window && screen.getDisplayMatching(window.getBounds()),
    processes: app.getAppMetrics().map(({ pid, type, cpu, memory }) => ({ pid, type, cpu, memory })) };
}
export function analyzeArm(record, spec) {
  const data = record.observation;
  const performance = spec.raw ? analyzeRaw(data, { fps: spec.fps }) : analyzePlayback(data, { fps: spec.fps, requireNeural: true, minDurationMs: 60000 });
  const before = data?.before, after = data?.after, native = record.nativeBefore, last = record.nativeAfter;
  const criteria = { complete: data?.complete === true, errors: record.errors?.length === 0 && data?.errors?.length === 0,
    warmup: finite(record.warm?.stableAt) && record.warm.readyAt - record.warm.stableAt >= 5000 && data?.startAt - record.warm.playAt <= 9000,
    environment: [before, after].every(value => value?.visibility === 'visible' && value.focused === true && value.video?.width === 1280 && value.video.height === 720 &&
      value.video.rate === 1 && value.video.muted === false && value.video.volume === 1 && value.video.loop === false && !value.video.ended && value.video.error === null),
    native: [native, last].every(value => value?.versions?.electron === VERSION && value.count === 1 && value.visible === true && value.focused === true && value.bounds?.width === 1280 && value.bounds.height === 720),
    stableWindow: !!native && !!last && JSON.stringify(native.bounds) === JSON.stringify(last.bounds) && JSON.stringify(native.display) === JSON.stringify(last.display) &&
      JSON.stringify(native.versions) === JSON.stringify(last.versions) && before?.dpr === after?.dpr && before?.width === after?.width && before?.height === after?.height,
    cleanup: record.cleanup?.runtimeReleased === true && record.cleanup.callbackCancelled === true && record.cleanup.urlRevoked === true &&
      record.cleanup.sourceReleased === true && record.cleanup.paused === true && record.cleanup.snapshot?.cleanupErrors?.length === 0 &&
      ['devices', 'pipelines', 'drivers', 'callbacks', 'objectUrls'].every(key => record.cleanup.snapshot?.resources?.[key] === 0) &&
      record.exit?.code === 0 && record.exit.signal === null && record.profileRemoved === true && !record.forcedKill };
  return { outcome: performance.outcome === 'PASS' && Object.values(criteria).every(Boolean) ? 'PASS' : 'FAIL', criteria, performance,
    qualityDelta: qualityDelta(data?.qualityBefore, data?.qualityAfter) };
}

export async function runShort(directory, { parityPath = '.cache/m11/parity-04/result.json', journeysPath = '.cache/m11/journeys-03/result.json', ancestor = null } = {}) {
  const output = playbackOutput(directory), build = localPath(BUILD), parityFile = localPath(parityPath), journeyFile = localPath(journeysPath);
  assert(!existsSync(output), 'Immutable attempt already exists');
  for (const file of [parityFile, journeyFile]) assert(!file.startsWith(output + sep) && !file.startsWith(build + sep), 'Evidence/output overlap');
  git(['check-ignore', '-q', relative(ROOT, output)]); git(['check-ignore', '-q', BUILD]);
  mkdirSync(dirname(output), { recursive: true }); mkdirSync(output);
  const report = { schema: 'aethervsr.m11.short-playback/1', output: relative(ROOT, output), verdict: 'FAIL', startedAt: new Date().toISOString(),
    arms: SHORT.map(spec => ({ ...spec, verdict: 'NOT_RUN', reason: 'Earlier gate not completed' })), errors: [],
    bounds: { runMs: 720000, launchMs: 30000, warmupMs: 9000, observationMs: 60000, drainMs: 10000, closeMs: 10000 },
    host: { hostname: hostname(), platform: platform(), release: release(), arch: arch(), node: process.versions },
    scope: 'Instrumented native Electron Auto; raw is a test-injected no-pipeline video, not an app UI mode. GPU timestamps bracket core upscale, not CPU recording. Callback/texture/rAF software lag is not physical A/V or scanout. Process metrics are opening/closing snapshots outside observation, not summed physical memory; observer cost not measured. Thermal/watts/physical refresh: not measured. No retries. Soak is not run by this command.' };
  const write = (name, value) => {
    const bytes = Buffer.from(JSON.stringify(value, null, 2) + '\n'), path = localPath(join(output, name));
    writeFileSync(path, bytes, { flag: 'wx' }); return { path: relative(ROOT, path), bytes: bytes.length, sha256: hash(bytes) };
  };
  let abort, stopping = false, app = null, page = null, child = null, exited = null, active = null, record = null, saved = false;
  const cancelled = new Promise((_, reject) => { abort = reject; }); cancelled.catch(() => {});
  const interrupt = reason => { if (stopping) return; stopping = true; report.errors.push(String(reason)); abort(Error(String(reason))); };
  const sigint = () => interrupt('SIGINT'), sigterm = () => interrupt('SIGTERM');
  process.once('SIGINT', sigint); process.once('SIGTERM', sigterm);
  const watchdog = setTimeout(() => interrupt('Whole short run deadline'), report.bounds.runMs);
  const limit = async (operation, ms = 10000, cleanup = false) => {
    let timer;
    try { return await Promise.race([operation, new Promise((_, reject) => { timer = setTimeout(() => reject(Error(`Operation deadline ${ms}ms`)), ms); }), ...(cleanup ? [] : [cancelled])]); }
    finally { clearTimeout(timer); }
  };
  const save = () => { if (saved) return; report.finishedAt = new Date().toISOString(); write('result.json', report); saved = true; };
  const close = async () => {
    if (!app) return;
    try { record.cleanup = await limit(page.evaluate(() => window.m11Playback?.cleanup()), 5000, true); }
    catch (error) { record.errors.push(`Recorder cleanup: ${error}`); }
    try {
      await limit((async () => {
        await app.evaluate(({ BrowserWindow }) => {
          if (BrowserWindow.getAllWindows().length !== 1) throw Error('Unexpected window count at close');
          setImmediate(() => BrowserWindow.getAllWindows()[0].close());
        });
        record.exit = await exited;
      })(), 10000, true);
      assert(record.exit.code === 0 && record.exit.signal === null, 'Nonzero native exit');
    } catch (error) {
      record.errors.push(`Natural close failed: ${error}`); record.forcedKill = true;
      try { record.groupKilled = killOwnedGroup(child.pid, record.processGroup); }
      catch (failure) { record.errors.push(`Owned group cleanup: ${failure}`); }
      try { record.exit = await limit(exited, 2000, true); } catch (failure) { record.errors.push(String(failure)); }
    }
    if (record.exit) {
      try { rmSync(localPath(record.profile), { recursive: true }); record.profileRemoved = !existsSync(record.profile); }
      catch (error) { record.errors.push(`Profile cleanup: ${error}`); }
    }
    app = null; page = null; child = null;
  };
  let pins, packagePin;
  try {
    report.sourceBefore = sourcePin();
    const parity = JSON.parse(readFileSync(parityFile)), journeys = JSON.parse(readFileSync(journeyFile));
    assert.equal(resolve(ROOT, journeys.parity?.path ?? ''), parityFile, 'Journey parity reference differs');
    assert.equal(filePin(parityFile).sha256, journeys.parity.sha256);
    report.prerequisites = { parity: filePin(parityFile), journeys: filePin(journeyFile),
      proof: verifyPrerequisites(parity, journeys, report.sourceBefore.commit, ancestor) };
    pins = [...referencePins([parity, journeys]).values(), report.prerequisites.parity, report.prerequisites.journeys];
    report.media = MEDIA.map(media => { const pin = filePin(shortMedia(media.fps).path); assert.equal(pin.sha256, media.sha256); return { ...media, bytes: pin.bytes }; });
    const modelPin = filePin('public/models/aethersr-c16d2.json'); assert.equal(modelPin.sha256, MODEL_SHA256); pins.push(modelPin);
    for (const path of ['package.json', 'node_modules/electron/package.json']) {
      const manifest = JSON.parse(readFileSync(localPath(path))); assert.equal(manifest.devDependencies?.electron ?? manifest.version, VERSION);
    }
    if (platform() === 'darwin') report.host.hardware = Object.fromEntries([
      ['model', 'sysctl', ['-n', 'hw.model']], ['chip', 'sysctl', ['-n', 'machdep.cpu.brand_string']], ['memoryBytes', 'sysctl', ['-n', 'hw.memsize']], ['os', 'sw_vers', []],
    ].map(([name, command, args]) => [name, execFileSync(command, args, { encoding: 'utf8', timeout: 5000 }).trim()]));
    const { buildDesktop } = await import('../../apps/desktop/build.mjs');
    if (!existsSync(build)) await limit(buildDesktop({ diagnostic: true, outdir: BUILD }), 60000);
    packagePin = () => {
      const pin = JSON.parse(readFileSync(localPath(join(build, 'build-provenance.json')))); verifyPayload(pin);
      assert.equal(pin.sourceCommit, report.sourceBefore.commit, 'Existing build has another source pin; never overwritten');
      assert.deepEqual(pin.files, parity.packageBefore.files, 'Built player differs from parity payload');
      assert.deepEqual(pin.files, journeys.packageBefore.files, 'Built player differs from journey payload');
      assert.deepEqual(pin.inputs, parity.packageBefore.inputs, 'Built inputs differ');
      for (const [name, expected] of Object.entries(pin.files)) { const actual = filePin(join(build, name)); assert.equal(actual.sha256, expected.sha256); assert.equal(actual.bytes, expected.bytes); }
      return pin;
    };
    report.packageBefore = packagePin();
    const executable = realpathSync((await import('electron')).default);
    report.binariesBefore = { electron: filePin(executable), runnerNode: { path: realpathSync(process.execPath), sha256: hash(readFileSync(process.execPath)) } };
    assert.equal(report.binariesBefore.electron.sha256, journeys.binary.sha256);
    const verifyPins = () => {
      assert.deepEqual(sourcePin(), report.sourceBefore); assert.deepEqual(packagePin(), report.packageBefore);
      for (const pin of [...pins, ...report.media, report.binariesBefore.electron]) { const actual = filePin(pin.path); assert.equal(actual.sha256, pin.sha256); if (pin.bytes !== undefined) assert.equal(actual.bytes, pin.bytes); }
      assert.equal(hash(readFileSync(report.binariesBefore.runnerNode.path)), report.binariesBefore.runnerNode.sha256);
      return { at: new Date().toISOString(), source: report.sourceBefore, payloadSha256: report.packageBefore.payloadSha256, clips: report.media, binaries: report.binariesBefore };
    };
    const { _electron } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
    for (const spec of SHORT) {
      assert(!stopping); active = report.arms.find(row => row.id === spec.id); active.verdict = 'FAIL'; delete active.reason;
      record = { errors: [], console: [], profile: join(output, `${spec.id}-profile`), profileRemoved: false };
      exited = null;
      let fatal = false;
      try {
        record.pinsBefore = verifyPins(); mkdirSync(record.profile);
        const env = { ...process.env, AETHERVSR_TEST_PROFILE: record.profile };
        for (const name of ['NODE_OPTIONS', 'ELECTRON_RUN_AS_NODE', 'ELECTRON_EXTRA_LAUNCH_ARGS']) delete env[name];
        app = await _electron.launch({ executablePath: executable, args: [build], env, timeout: 30000 });
        child = app.process(); exited = child.exitCode !== null || child.signalCode !== null ? Promise.resolve({ code: child.exitCode, signal: child.signalCode })
          : new Promise(done => child.once('close', (code, signal) => done({ code, signal })));
        record.processGroup = Number(execFileSync('ps', ['-o', 'pgid=', '-p', String(child.pid)], { encoding: 'utf8', timeout: 5000 }).trim());
        assert.equal(record.processGroup, child.pid, 'Electron must own its process group');
        assert(!stopping); page = await limit(app.firstWindow({ timeout: 10000 })); page.setDefaultTimeout(10000);
        const owned = record;
        page.on('pageerror', error => { owned.errors.push(String(error)); owned.fatalError = true; });
        page.on('crash', () => { owned.errors.push('Renderer crashed'); owned.fatalError = true; });
        page.on('console', message => { if (owned.console.length < 512) owned.console.push({ type: message.type(), text: message.text() }); if (message.type() === 'error') owned.errors.push(message.text()); });
        await limit(page.waitForURL('aethervsr://app/index.html'));
        await limit(app.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.setBounds({ width: 1280, height: 720 }); window.show(); window.focus(); }));
        await limit(page.bringToFront());
        await limit(waitForObservation(() => page.evaluate(() => typeof window.m11Desktop !== 'undefined')));
        await limit(page.evaluate(installRecorder, { raw: spec.raw }));
        await limit(page.locator('#file').setInputFiles(join(ROOT, shortMedia(spec.fps).path)));
        record.selected = await limit(page.evaluate(() => window.m11Playback.selected));
        const media = report.media.find(row => row.fps === spec.fps);
        assert.equal(record.selected.sha256, media.sha256); assert.equal(record.selected.bytes, media.bytes);
        await limit(waitForObservation(() => page.evaluate(() => window.m11Desktop.video.readyState >= 2)));
        record.warm = await limit(page.evaluate(() => window.m11Playback.warm()), 10000);
        record.nativeBefore = await limit(app.evaluate(nativeState));
        const native = record.nativeBefore, prefs = native.preferences;
        assert.equal(native.versions.electron, VERSION); assert.equal(native.profile, record.profile);
        assert(prefs.sandbox && prefs.contextIsolation && prefs.webSecurity && !prefs.nodeIntegration && !prefs.webviewTag && !prefs.allowRunningInsecureContent && !prefs.experimentalFeatures);
        assert(native.focused && native.visible && native.count === 1 && native.bounds.width === 1280 && native.bounds.height === 720);
        record.observation = await limit(page.evaluate(() => window.m11Playback.start()), 70000);
        active.observation = write(`${spec.id}-observation.json`, record.observation);
        record.nativeAfter = await limit(app.evaluate(nativeState));
        record.pinsAfter = verifyPins();
        if (record.fatalError || (!spec.raw && (record.observation.diagnosticsErrors?.length || ['unavailable', 'error'].includes(record.observation.snapshot?.state) || !record.observation.snapshot?.resources?.pipelines))) fatal = true;
      } catch (error) {
        fatal = true; record.errors.push(String(error));
        if (page) {
          try {
            record.failureEnvironment = await limit(page.evaluate(() => window.m11Playback?.environment()), 2000, true);
            record.warmupMiss = recoverableWarmupFailure(error, record.failureEnvironment, spec, record.fatalError);
            if (record.warmupMiss) fatal = false;
          } catch (failure) { record.errors.push(String(failure)); }
        }
        if (page && !record.observation) {
          try { record.observation = await limit(page.evaluate(() => {
            window.m11Desktop?.video.pause();
            void window.m11Playback?.stop('Interrupted/deadline/exception').catch(() => {});
            return window.m11Playback?.partial();
          }), 2000, true); }
          catch (failure) {
            record.errors.push(String(failure));
          }
        }
      } finally {
        if (!active.observation) active.observation = write(`${spec.id}-observation.json`, record.observation ?? { complete: false, errors: record.errors, unavailable: true });
        await close();
        if (!record.profileRemoved && existsSync(record.profile) && !record.exit && !exited) { rmSync(localPath(record.profile), { recursive: true }); record.profileRemoved = !existsSync(record.profile); }
        const analysis = analyzeArm(record, spec); active.analysis = write(`${spec.id}-analysis.json`, analysis);
        const { observation: _observation, ...metadata } = record; active.metadata = write(`${spec.id}-metadata.json`, metadata);
        active.verdict = analysis.outcome;
        if (!analysis.criteria.cleanup) fatal = true;
      }
      if (fatal || stopping) { active.reason = 'Fatal setup, acquisition, readback or cleanup failure; no retry'; break; }
    }
    report.pinsAfter = verifyPins(); report.sourceAfter = sourcePin(); report.packageAfter = packagePin();
    report.verdict = !stopping && report.arms.every(row => row.verdict === 'PASS') ? 'PASS' : 'FAIL';
  } catch (error) { report.errors.push(String(error)); }
  finally {
    clearTimeout(watchdog);
    if (app) { try { await close(); } catch (error) { report.errors.push(String(error)); } }
    if (report.errors.length || stopping) report.verdict = 'FAIL';
    save(); process.off('SIGINT', sigint); process.off('SIGTERM', sigterm);
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === '--help') console.log('node tools/m11/playback.mjs short NEW_OUTPUT [EXACT_EVIDENCE_ANCESTOR]\nFixed eight-arm short matrix only. Clean committed runner, native parity/journeys and byte-identical player payload required. Uses parity-04 and journeys-03 by default; runShort API accepts explicit paths. No retries or automatic soak. Existing build is verified, never overwritten.');
  else {
    assert(process.argv[2] === 'short' && process.argv[3] && process.argv.length <= 5, 'Expected short NEW_OUTPUT [EXACT_EVIDENCE_ANCESTOR]; see --help');
    const report = await runShort(process.argv[3], { ancestor: process.argv[4] ?? null });
    console.log(JSON.stringify({ output: report.output, verdict: report.verdict, arms: report.arms.map(({ id, verdict }) => ({ id, verdict })), errors: report.errors }));
    process.exitCode = report.verdict === 'PASS' ? 0 : 1;
  }
}