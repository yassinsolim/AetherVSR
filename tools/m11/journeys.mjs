import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { arch, hostname, platform, release } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MODEL_SHA256, summarizeParity } from './parity.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url)), BUILD = '.cache/m11/journey-app', VERSION = '44.4.1';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', timeout: 10000 }).trim();
const MEDIA = [
  ['.cache/m1010r/media-02/replay-30.mp4', 'e006f3d5381d73b1ca5739f5e74216f4bf0c312f28621634d258a770822a8f9b'],
  ['.cache/m1010r/media-02/replay-60.mp4', '5171a8b6da7303c8c409167f4191b7330f41120fcd89d7e3aa0df9483b237b29'],
];
const CLOSE = ['paused', 'baseline', 'neural', 'seek', 'replacement'];
const CASES = ['play', 'pause', 'resume', 'forward', 'backward', 'audio', 'baseline', 'neural', 'auto', 'rate',
  'resize', 'fullscreen', 'replace', 'device-loss', 'recover', 'security', 'close', ...CLOSE.map(mode => `close-${mode}`)];
export function observedAudibility(value, wanted) {
  assert.equal(typeof wanted, 'boolean');
  assert.equal(typeof value, 'boolean', 'Native audibility unavailable; mandatory audio gate unverified');
  return value === wanted;
}
function local(path) {
  let parent = path;
  while (!existsSync(parent)) parent = dirname(parent);
  assert.equal(realpathSync(parent), resolve(realpathSync(ROOT), relative(ROOT, parent)), 'Symlinked path');
  return path;
}
export function journeyOutput(directory = '.cache/m11/journeys-01') {
  assert(typeof directory === 'string' && directory.length > 0);
  const output = resolve(ROOT, directory), build = resolve(ROOT, BUILD);
  assert(output.startsWith(resolve(ROOT, '.cache/m11') + sep), 'Output outside .cache/m11');
  assert(output !== build && !output.startsWith(build + sep) && !build.startsWith(output + sep), 'Build overlap');
  return local(output);
}
export function verifyJourneyParity(parity, commit) {
  assert.equal(parity?.schema, 'aethervsr.m11.paused-parity/1');
  assert.equal(parity.verdict, 'PASS'); assert.equal(parity.parityPrerequisitePassed, true);
  assert.deepEqual(parity.errors, []); assert(summarizeParity(parity.cases).parityPrerequisitePassed);
  assert.equal(parity.sourceBefore?.commit, commit); assert.deepEqual(parity.sourceAfter, parity.sourceBefore);
  assert.equal(parity.expected?.modelSha256, MODEL_SHA256);
  assert.equal(parity.electronEnvironment?.versions?.electron, VERSION);
  assert.equal(parity.packageBefore?.diagnostic, true); assert.equal(parity.packageBefore?.sourceDirty, false);
  assert.equal(parity.packageBefore?.sourceCommit, commit); assert.deepEqual(parity.packageAfter, parity.packageBefore);
  for (const kind of ['chrome', 'electron', 'server']) assert.equal(parity.cleanup?.[kind], true);
  return true;
}
function sourcePin() {
  assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '', 'Reviewed clean HEAD required');
  git(['ls-files', '--error-unmatch', 'tools/m11/journeys.mjs']);
  return { commit: git(['rev-parse', 'HEAD']), tree: git(['rev-parse', 'HEAD^{tree}']) };
}
function snapshot() {
  const { video, canvas } = window.m11Desktop, session = window.m11Desktop.session();
  const stats = session.runtime?.pipeline.stats(performance.now());
  return { session: session.snapshot(), src: video.currentSrc, sameVideo: video === window.m11Journey.video,
    frames: stats?.framesRendered, generation: session.runtime?.pipeline.timingGeneration,
    features: session.gpu ? [...session.gpu.device.features] : [], adapter: session.gpu?.adapterReport,
    hidden: canvas.hidden, rect: canvas.getBoundingClientRect().toJSON(), background: getComputedStyle(canvas).backgroundImage,
    visible: document.visibilityState, focused: document.hasFocus(), width: innerWidth, height: innerHeight,
    dpr: devicePixelRatio, fullscreen: !!document.fullscreenElement, userAgent: navigator.userAgent };
}
function nativeState({ app, BrowserWindow, screen }) {
  const windows = BrowserWindow.getAllWindows(), window = windows[0];
  return { versions: process.versions, argv: process.argv, profile: app.getPath('userData'), count: windows.length,
    bounds: window?.getBounds(), visible: window?.isVisible(), fullscreen: window?.isFullScreen(),
    preferences: window?.webContents.getLastWebPreferences(), gpu: app.getGPUFeatureStatus(),
    display: window && screen.getDisplayMatching(window.getBounds()),
    processes: app.getAppMetrics().map(({ pid, type, cpu, memory }) => ({ pid, type, cpu, memory })) };
}
function seek(delta) {
  const video = window.m11Desktop.video, range = document.querySelector('#seek');
  const before = video.currentTime, target = Math.max(0, Math.min(video.duration - 1, before + delta));
  range.value = String(target); range.dispatchEvent(new Event('input', { bubbles: true })); range.dispatchEvent(new Event('change', { bubbles: true }));
  return { before, target, hidden: window.m11Desktop.canvas.hidden, seeking: video.seeking, at: performance.now() };
}

export async function runJourneys(directory = '.cache/m11/journeys-01', parityPath = '.cache/m11/parity-01/result.json') {
  const output = journeyOutput(directory), parityFile = local(resolve(ROOT, parityPath));
  assert(!existsSync(output), 'New attempt directory required'); assert(!parityFile.startsWith(output + sep));
  git(['check-ignore', '-q', relative(ROOT, output)]); git(['check-ignore', '-q', BUILD]); mkdirSync(output, { recursive: true });
  const report = { schema: 'aethervsr.m11.journeys/1', output: relative(ROOT, output), startedAt: new Date().toISOString(),
    verdict: 'FAIL', parityPrerequisitePassed: false, cases: CASES.map(id => ({ id, verdict: 'NOT_RUN' })), applications: [], errors: [],
    host: { hostname: hostname(), platform: platform(), release: release(), arch: arch() },
    scope: 'Digital lifecycle only. Performance, physical audio/AV sync and physical refresh: not measured. App metrics are instantaneous, not summed physical memory. Close requests follow recorded control state; overlap with decoder/GPU work is not proven.' };
  let app, page, child, exited, record, active, built, executable, stopping = false;
  const limit = async (promise, ms = 10000) => {
    let timer;
    try { return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(Error(`${active?.id ?? 'preflight/cleanup'} deadline ${ms}ms`)), ms); })]); }
    finally { clearTimeout(timer); }
  };
  const raw = (name, value) => {
    const bytes = Buffer.isBuffer(value) ? value : Buffer.from(JSON.stringify(value, null, 2) + '\n');
    writeFileSync(join(output, name), bytes, { flag: 'wx' });
    return { path: relative(ROOT, join(output, name)), bytes: bytes.length, sha256: hash(bytes) };
  };
  const snap = () => limit(page.evaluate(snapshot));
  const wait = (predicate, argument, timeout = 10000) => page.waitForFunction(predicate, argument, { timeout }).then(handle => handle.dispose());
  const ready = async (tier = null) => {
    await wait(tier => { const state = window.m11Desktop.session().snapshot(); if (state.error) throw Error(state.error); return state.ready && !state.pending && state.runtime?.running && (!tier || state.runtime.actualTier === tier); }, tier, 20000);
    const state = await snap(); assert(state.sameVideo && !state.hidden && state.focused && state.visible === 'visible');
    assert.equal(state.session.observerError, null); assert.deepEqual(state.session.cleanupErrors, []);
    assert(state.session.video.width === 1280 && state.session.video.height === 720);
    assert.deepEqual(state.session.canvas, { width: 2560, height: 1440, visible: true });
    assert.deepEqual(state.session.resources, { devices: 1, pipelines: 1, drivers: 1, callbacks: 1, objectUrls: 1 }); return state;
  };
  const quiet = async (old = false, ms = 500, lost = false) => {
    await page.evaluate(() => { delete window.m11Journey.quiet; });
    await wait(({ old, ms, lost }) => {
      const test = window.m11Journey, session = window.m11Desktop.session(), pipeline = old ? test.old.pipeline : session.runtime?.pipeline;
      if (!lost && !pipeline) throw Error('Expected stopped pipeline is missing');
      const now = performance.now(), frames = pipeline?.stats(now).framesRendered ?? null;
      test.quiet ??= { start: now, frames }; test.quiet.end = now;
      if (pipeline?.running || frames !== test.quiet.frames || (lost && (session.gpu || session.runtime || !window.m11Desktop.canvas.hidden || session.snapshot().state !== 'unavailable'))) throw Error('Stopped pipeline advanced or retried');
      return now - test.quiet.start >= ms;
    }, { old, ms, lost }, ms + 3000);
    return page.evaluate(() => window.m11Journey.quiet);
  };
  const file = async index => {
    await page.locator('#file').setInputFiles(join(ROOT, MEDIA[index][0]));
    await wait(name => { const state = window.m11Desktop.session().snapshot(); return state.name === name && state.video.width === 1280 && state.video.height === 720 && state.video.readyState >= 1; }, MEDIA[index][0].split('/').at(-1)); return snap();
  };
  const audible = async wanted => {
    const probes = []; active.audio ??= []; active.audio.push({ wanted, probes });
    for (let index = 0; index < 40; index++) {
      const value = await limit(app.evaluate(({ BrowserWindow }) => { const contents = BrowserWindow.getAllWindows()[0].webContents; return typeof contents.isCurrentlyAudible === 'function' ? contents.isCurrentlyAudible() : 'not measured'; }));
      probes.push({ at: new Date().toISOString(), value }); if (observedAudibility(value, wanted)) return;
      await wait(start => performance.now() - start >= 100, await page.evaluate(() => performance.now()), 1000);
    }
    throw Error(`Native audibility never became ${wanted}`);
  };
  const packagePin = () => {
    const provenance = JSON.parse(readFileSync(join(ROOT, BUILD, 'build-provenance.json'))); assert.deepEqual(provenance, built.provenance);
    assert.equal(provenance.sourceCommit, report.sourceBefore.commit); assert(!provenance.sourceDirty && provenance.diagnostic && provenance.electron === VERSION && provenance.modelSha256 === MODEL_SHA256);
    assert(provenance.inputs.includes('apps/desktop/renderer.ts') && !provenance.inputs.includes('apps/desktop/smoke.ts'));
    for (const [name, pin] of Object.entries(provenance.files)) { const bytes = readFileSync(local(join(ROOT, BUILD, name))); assert.equal(bytes.length, pin.bytes); assert.equal(hash(bytes), pin.sha256); } return provenance;
  };
  const close = async () => {
    record.last = await snap(); record.events.push(...await page.evaluate(() => window.m11Journey.events)); record.nativeLast = await limit(app.evaluate(nativeState));
    record.closeRequestedAt = new Date().toISOString();
    await limit(app.evaluate(({ BrowserWindow }) => { if (BrowserWindow.getAllWindows().length !== 1) throw Error('Extra window'); setImmediate(() => BrowserWindow.getAllWindows()[0].close()); }), 5000);
    record.exit = await limit(exited, 5000); assert.equal(record.exit.code, 0); assert.equal(record.exit.signal, null);
    app = null; page = null; child = null; rmSync(record.profile, { recursive: true }); record.profileRemoved = !existsSync(record.profile);
  };
  const attempt = async (id, operation) => {
    assert(!stopping); active = report.cases.find(row => row.id === id); active.startedAt = new Date().toISOString();
    try { const evidence = await limit(operation(), 60000); if (evidence !== undefined) active.evidence = evidence; assert.deepEqual(record.errors, []); active.verdict = 'PASS'; }
    catch (error) { active.verdict = 'FAIL'; active.error = String(error); throw error; }
    finally { active.endedAt = new Date().toISOString(); if (page) { try { active.last = await snap(); } catch (error) { active.snapshotError = String(error); } } active.artifact = raw(`${id}.json`, active); }
  };
  const interrupt = () => { stopping = true; report.errors.push('Interrupted or whole-run deadline'); child?.kill('SIGTERM'); };
  process.once('SIGINT', interrupt); process.once('SIGTERM', interrupt); const watchdog = setTimeout(interrupt, 300000);
  try {
    report.sourceBefore = sourcePin(); const parityBytes = readFileSync(parityFile), parity = JSON.parse(parityBytes);
    report.parity = { path: relative(ROOT, parityFile), sha256: hash(parityBytes) }; report.parityPrerequisitePassed = verifyJourneyParity(parity, report.sourceBefore.commit);
    report.media = MEDIA.map(([path, sha256]) => { const bytes = readFileSync(join(ROOT, path)); assert.equal(hash(bytes), sha256); return { path, sha256, bytes: bytes.length }; });
    for (const path of ['package.json', 'node_modules/electron/package.json']) { const manifest = JSON.parse(readFileSync(join(ROOT, path))); assert.equal(manifest.devDependencies?.electron ?? manifest.version, VERSION); }
    if (platform() === 'darwin') report.host.hardware = Object.fromEntries([['model', 'sysctl', ['-n', 'hw.model']], ['chip', 'sysctl', ['-n', 'machdep.cpu.brand_string']], ['os', 'sw_vers', []]].map(([name, command, args]) => [name, execFileSync(command, args, { encoding: 'utf8', timeout: 5000 }).trim()]));
    for (const name of ['main.cjs', 'renderer.js', 'index.html', 'player.css', 'package.json', 'build-provenance.json', 'models/production.json']) local(join(ROOT, BUILD, name));
    const { buildDesktop } = await import('../../apps/desktop/build.mjs'); built = await limit(buildDesktop({ diagnostic: true, outdir: BUILD }), 60000); report.packageBefore = packagePin();
    executable = (await import('electron')).default; report.binary = { path: realpathSync(executable), sha256: hash(readFileSync(executable)) }; assert.equal(report.binary.sha256, parity.binaries?.electron?.sha256);
    const { _electron } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
    const { PNG } = await import('../../.cache/m9/node_modules/playwright-core/lib/utilsBundle.js');
    const launch = async id => {
      assert(!app && !stopping); record = { id, profile: join(output, `${id}-profile`), errors: [], console: [], events: [], expected: [] }; report.applications.push(record); mkdirSync(record.profile);
      const env = { ...process.env, AETHERVSR_TEST_PROFILE: record.profile }; for (const name of ['NODE_OPTIONS', 'ELECTRON_RUN_AS_NODE', 'ELECTRON_EXTRA_LAUNCH_ARGS']) delete env[name];
      app = await _electron.launch({ executablePath: executable, args: [join(ROOT, BUILD)], env, timeout: 30000 }); child = app.process();
      exited = child.exitCode !== null || child.signalCode !== null ? Promise.resolve({ code: child.exitCode, signal: child.signalCode }) : new Promise(done => child.once('exit', (code, signal) => done({ code, signal, at: new Date().toISOString() })));
      page = await app.firstWindow({ timeout: 10000 }); page.setDefaultTimeout(10000); const owned = record;
      page.on('pageerror', error => owned.errors.push(String(error))); page.on('crash', () => owned.errors.push('Renderer crashed'));
      page.on('console', message => { const text = message.text(), url = message.location().url;
        const expected = (owned.expected.includes(url) && /Failed to load resource/.test(text)) || (owned.securityProbe && text.includes('https://m11.invalid') && /Content Security Policy/.test(text));
        owned.console.push({ type: message.type(), text, url, expected: !!expected }); if (message.type() === 'error' && !expected) owned.errors.push(text);
      });
      page.on('requestfailed', request => owned.events.push({ type: 'requestfailed', url: request.url(), error: request.failure() }));
      await page.waitForURL('aethervsr://app/index.html'); await page.bringToFront(); record.native = await limit(app.evaluate(nativeState));
      const native = record.native, prefs = native.preferences; assert.equal(native.versions.electron, VERSION); assert.equal(native.profile, record.profile); assert(native.count === 1 && native.visible);
      assert(prefs.sandbox && prefs.contextIsolation && prefs.webSecurity && !prefs.nodeIntegration && !prefs.webviewTag && !prefs.allowRunningInsecureContent && !prefs.experimentalFeatures);
      record.security = await page.evaluate(() => {
        if (document.querySelectorAll('video').length !== 1 || document.querySelectorAll('canvas').length !== 1) throw Error('Duplicate media');
        window.m11Journey = { video: window.m11Desktop.video, events: [], csp: [] };
        for (const type of ['loadedmetadata', 'play', 'playing', 'pause', 'seeking', 'seeked', 'emptied', 'volumechange', 'ratechange', 'error']) window.m11Desktop.video.addEventListener(type, () => {
          if (window.m11Journey.events.length >= 512) throw Error('Event capacity'); window.m11Journey.events.push({ type, at: performance.now(), state: window.m11Desktop.session().snapshot(), hidden: window.m11Desktop.canvas.hidden });
        });
        document.addEventListener('securitypolicyviolation', event => window.m11Journey.csp.push({ directive: event.effectiveDirective, uri: event.blockedURI }));
        return { secure: isSecureContext, require: typeof window.require, process: typeof window.process, electron: typeof window.electron, gpu: typeof navigator.gpu };
      });
      assert(record.security.secure && record.security.gpu !== 'undefined'); for (const name of ['require', 'process', 'electron']) assert.equal(record.security[name], 'undefined');
    };
    await attempt('play', async () => { await launch('primary'); active.metadata = await file(0); assert(active.metadata.session.video.paused); await page.locator('#play').click(); const state = await ready('neural'); assert(!state.session.video.muted && state.session.video.volume === 1); await audible(true); return state; });
    await attempt('pause', async () => { await page.locator('#play').click(); assert((await snap()).session.video.paused); return quiet(); });
    await attempt('resume', async () => { const before = await snap(); await page.locator('#play').click(); const after = await ready(); assert(after.frames > before.frames || (after.generation !== before.generation && after.frames > 0)); return { before, after }; });
    for (const [id, delta] of [['forward', 10], ['backward', -3]]) await attempt(id, async () => { active.action = await page.evaluate(seek, delta); assert(active.action.hidden && active.action.seeking); const after = await ready(); assert(!after.session.video.seeking && Math.abs(after.session.video.time - active.action.target) < 3); return after; });
    await attempt('audio', async () => {
      await page.locator('#volume').evaluate(range => { range.value = '0.4'; range.dispatchEvent(new Event('input', { bubbles: true })); }); assert.equal((await snap()).session.video.volume, 0.4);
      await page.locator('#mute').click(); assert((await snap()).session.video.muted); await audible(false);
      await page.locator('#mute').click(); const state = await snap(); assert(!state.session.video.muted && state.session.video.volume === 0.4); await audible(true); return state;
    });
    for (const mode of ['baseline', 'neural', 'auto']) await attempt(mode, async () => { await page.locator('#mode').selectOption(mode); const state = await ready(mode === 'auto' ? null : mode); assert.equal(state.session.runtime.controller.mode, mode); return state; });
    await attempt('rate', async () => {
      active.scope = 'Session API, no rate UI'; active.evidence = [];
      for (const rate of [1.25, 1]) {
        await page.evaluate(rate => window.m11Desktop.session().setRate(rate), rate);
        const state = await ready(); assert.equal(state.session.video.rate, rate); active.evidence.push(state);
      }
    });
    await attempt('resize', async () => {
      active.evidence = [];
      for (const [width, height] of [[1100, 760], [640, 480], [1280, 850], [800, 600], [1000, 700], [640, 480], [1100, 760]]) {
        await app.evaluate(({ BrowserWindow }, bounds) => BrowserWindow.getAllWindows()[0].setBounds(bounds), { width, height }); await wait(width => outerWidth === width, width);
        const state = await ready(), native = await app.evaluate(nativeState); assert(native.bounds.width === width && native.bounds.height === height && state.background === 'none');
        const row = { state, native }; active.evidence.push(row);
        if (active.evidence.length <= 2) {
          const bytes = await page.screenshot(); row.screenshot = raw(`visible-${width}x${height}.png`, bytes);
          const image = PNG.sync.read(bytes), scale = image.width / state.width, rect = state.rect; let minimum = 255, maximum = 0;
          for (let top = 1; top < 32; top++) for (let left = 1; left < 32; left++) {
            const offset = (Math.floor((rect.y + rect.height * top / 32) * scale) * image.width + Math.floor((rect.x + rect.width * left / 32) * scale)) * 4;
            const value = Math.max(...image.data.subarray(offset, offset + 3)); minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
          }
          row.pixels = { minimum, maximum, samples: 961, scope: 'Offline screenshot canvas region, not parity' }; assert(maximum > 32 && maximum - minimum > 8, 'Blank output');
        }
      }
    });
    await attempt('fullscreen', async () => {
      active.evidence = [];
      for (const entered of [true, false]) {
        await page.locator('#fullscreen').click(); await wait(entered => !!document.fullscreenElement === entered, entered);
        active.evidence.push({ state: await ready(), native: await app.evaluate(nativeState) });
      }
    });
    await attempt('replace', async () => {
      const before = await snap(); record.expected.push(before.src); await page.evaluate(() => { window.m11Journey.old = window.m11Desktop.session().runtime; });
      active.metadata = await file(1); assert(active.metadata.session.sourceGeneration > before.session.sourceGeneration && active.metadata.src !== before.src);
      active.revoked = await page.evaluate(async url => { try { const response = await fetch(url); await response.body?.cancel(); return false; } catch { return true; } }, before.src); assert(active.revoked);
      active.oldStopped = await quiet(true); await page.locator('#play').click(); const state = await ready();
      assert(await page.evaluate(() => window.m11Journey.old !== window.m11Desktop.session().runtime && window.m11Journey.old.pipeline.disposed && !window.m11Journey.old.pipeline.running)); await page.evaluate(() => { delete window.m11Journey.old; }); return state;
    });
    await attempt('device-loss', async () => {
      await page.evaluate(() => window.m11Desktop.session().gpu.device.destroy()); await wait(() => window.m11Desktop.session().snapshot().state === 'unavailable' && window.m11Desktop.canvas.hidden);
      active.failed = await snap(); assert(!active.failed.session.video.paused && active.failed.session.resources.drivers === 0);
      assert.equal(active.failed.session.resources.devices, 0); assert.equal(active.failed.session.resources.pipelines, 0);
      active.noRetry = await quiet(false, 1000, true); const after = await snap(); assert(after.session.video.time > active.failed.session.video.time); return after;
    });
    await attempt('recover', async () => { await file(0); await page.locator('#play').click(); return ready('neural'); });
    await attempt('security', async () => {
      record.securityProbe = true; assert(await page.evaluate(() => window.open('https://m11.invalid/') === null));
      await app.evaluate(({ BrowserWindow }) => { globalThis.m11Navigation = new Promise(done => BrowserWindow.getAllWindows()[0].webContents.once('will-navigate', (event, url) => done({ url, prevented: event.defaultPrevented }))); });
      await page.evaluate(() => { const link = document.createElement('a'); link.href = 'https://m11.invalid/'; document.body.append(link); link.click(); link.remove(); });
      active.navigation = await limit(app.evaluate(() => globalThis.m11Navigation)); assert(active.navigation.prevented);
      active.networkBlocked = await page.evaluate(async () => { const frame = document.createElement('iframe'); frame.id = 'journey-frame'; frame.src = 'https://m11.invalid/frame'; frame.hidden = true; document.body.append(frame); try { await fetch('https://m11.invalid/network'); return false; } catch { return true; } }); assert(active.networkBlocked);
      await wait(() => ['frame-src', 'connect-src'].every(directive => window.m11Journey.csp.some(row => row.directive === directive))); active.csp = await page.evaluate(() => window.m11Journey.csp);
      await page.evaluate(() => document.querySelector('#journey-frame').remove());
      const native = await app.evaluate(nativeState); assert.equal(native.count, 1); assert.equal(page.url(), 'aethervsr://app/index.html'); record.securityProbe = false; return native;
    });
    await attempt('close', close);
    for (const mode of CLOSE) await attempt(`close-${mode}`, async () => {
      await launch(mode); await file(1); await page.locator('#mode').selectOption(mode === 'baseline' ? 'baseline' : 'neural'); await page.locator('#play').click(); await ready(mode === 'baseline' ? 'baseline' : 'neural');
      if (mode === 'paused') { await page.locator('#play').click(); assert((await snap()).session.video.paused); await quiet(); }
      if (mode === 'seek') { active.action = await page.evaluate(seek, 10); assert(active.action.hidden && active.action.seeking); }
      if (mode === 'replacement') await page.locator('#file').setInputFiles(join(ROOT, MEDIA[0][0])); await close();
    });
    report.verdict = 'PASS';
  } catch (error) { report.errors.push(String(error)); }
  finally {
    stopping = true; clearTimeout(watchdog);
    if (app) {
      record.forcedCleanup = true;
      try { record.last = await snap(); record.events.push(...await limit(page.evaluate(() => window.m11Journey?.events ?? []))); } catch (error) { record.snapshotError = String(error); }
      try { await limit(app.close(), 5000); } catch (error) { report.errors.push(String(error)); child?.kill('SIGKILL'); }
      try { record.cleanupExit = await limit(exited, 5000); } catch (error) { report.errors.push(String(error)); }
    }
    if (record && !record.profileRemoved && (!app || record.cleanupExit)) { try { rmSync(record.profile, { recursive: true }); record.profileRemoved = !existsSync(record.profile); } catch (error) { report.errors.push(String(error)); } }
    try {
      if (report.sourceBefore) { report.sourceAfter = sourcePin(); assert.deepEqual(report.sourceAfter, report.sourceBefore); }
      if (built) report.packageAfter = packagePin(); if (executable) assert.equal(hash(readFileSync(executable)), report.binary.sha256);
      if (report.parity) assert.equal(hash(readFileSync(parityFile)), report.parity.sha256);
      for (const [path, sha256] of MEDIA) if (report.media) assert.equal(hash(readFileSync(join(ROOT, path))), sha256);
    } catch (error) { report.errors.push(String(error)); }
    report.cleanup = { profilesRemoved: report.applications.every(row => row.profileRemoved), naturalExits: report.applications.every(row => row.exit?.code === 0 && !row.forcedCleanup) };
    if (report.errors.length || report.applications.some(row => row.errors.length) || report.cases.some(row => row.verdict !== 'PASS') || !report.cleanup.profilesRemoved || !report.cleanup.naturalExits) report.verdict = 'FAIL';
    report.endedAt = new Date().toISOString(); raw('result.json', report); process.off('SIGINT', interrupt); process.off('SIGTERM', interrupt);
  }
  return report;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === '--help') console.log('node tools/m11/journeys.mjs [new-output-directory] [native-parity-result.json]\nSeparate native authorization, clean reviewed HEAD, Electron 44.4.1 and same-HEAD parity PASS required. No performance runs or retries.');
  else { assert(process.argv.length <= 4 && process.argv.slice(2).every(value => !value.startsWith('--'))); const report = await runJourneys(process.argv[2], process.argv[3]); console.log(JSON.stringify({ output: report.output, verdict: report.verdict, parityPrerequisitePassed: report.parityPrerequisitePassed, errors: report.errors })); process.exitCode = report.verdict === 'PASS' ? 0 : 1; }
}