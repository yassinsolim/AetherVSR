import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { arch, hostname, platform, release } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { captureTask, cleanupRuntime, compareCaptures, MODEL_SHA256, pauseRuntime, runtimeState,
  seekPaused, SOURCE_SHA256, TIMES } from '../m10-output-parity.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const MEDIA = 'public/media/m9/720p60.mp4';
const MODEL = 'public/models/aethersr-c16d2.json';
const BUILD = '.cache/m11/parity-app';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', timeout: 10000 }).trim();

export { MODEL_SHA256, SOURCE_SHA256 };
export const PARITY_CASES = Object.freeze([false, true].flatMap(forceCopy => TIMES.map(time =>
  Object.freeze({ id: `${forceCopy ? 'copy' : 'external'}-${time}`, forceCopy, time }))));

export async function waitForObservation(observe, timeout = 10000) {
  assert(Number.isFinite(timeout) && timeout > 0);
  let timer;
  const expired = new Promise((_, reject) => { timer = setTimeout(() => reject(Error(`Observation deadline ${timeout}ms`)), timeout); });
  try {
    for (;;) {
      const value = await Promise.race([observe(), expired]);
      if (value) return value;
      await Promise.race([new Promise(resolve => setTimeout(resolve, 50)), expired]);
    }
  } finally { clearTimeout(timer); }
}

export function compareParityCase(spec, desktop, harness, modelJsonSha256) {
  assert(PARITY_CASES.some(value => value.id === spec.id && value.time === spec.time && value.forceCopy === spec.forceCopy),
    'Unknown parity case');
  assert(/^[a-f0-9]{64}$/.test(modelJsonSha256), 'Pinned model JSON digest required');
  const comparison = compareCaptures(desktop?.capture, harness?.capture);
  comparison.checks.expectedCase = [desktop, harness].every(row => row.capture.requestedTime === spec.time &&
    row.capture.currentTime === spec.time && row.capture.importPath === (spec.forceCopy ? 'sampled' : 'external') &&
    row.capture.modelJsonSha256 === modelJsonSha256 && row.capture.options?.passDiagnostics === true);
  comparison.checks.seekIdentity = [desktop, harness].every(row => row.seek?.requestedTime === spec.time &&
    row.seek.currentTime === spec.time && row.seek.source === row.capture.source &&
    Number.isFinite(row.seek.metadata?.mediaTime) && Math.abs(row.seek.metadata.mediaTime - spec.time) <= 1 / 60 &&
    row.seek.metadata.width === 1280 && row.seek.metadata.height === 720);
  comparison.checks.decodedMediaTimeEqual = desktop.seek?.metadata?.mediaTime === harness.seek?.metadata?.mediaTime;
  comparison.checks.foreground = [desktop, harness].every(row => row.capture.environment?.visibility === 'visible' &&
    row.capture.environment.focused === true);
  comparison.checks.noGpuErrors = [desktop, harness].every(row => Array.isArray(row.capture.gpuErrors) && row.capture.gpuErrors.length === 0);
  comparison.verdict = Object.values(comparison.checks).every(Boolean) ? 'PASS' : 'FAIL';
  return comparison;
}

export function summarizeParity(cases) {
  const completeMatrix = Array.isArray(cases) && cases.length === PARITY_CASES.length && PARITY_CASES.every((spec, index) =>
    ['id', 'time', 'forceCopy'].every(key => cases[index]?.[key] === spec[key]));
  const passed = Array.isArray(cases) ? cases.filter(row => row?.verdict === 'PASS' && row.comparison?.verdict === 'PASS' &&
    Object.keys(row.comparison.checks ?? {}).length > 0 && Object.values(row.comparison.checks).every(value => value === true)).length : 0;
  return { required: PARITY_CASES.length, passed, failed: Array.isArray(cases) ? cases.filter(row => row?.verdict === 'FAIL').length : 0,
    notRun: Array.isArray(cases) ? cases.filter(row => row?.verdict === 'NOT_RUN').length : 0,
    parityPrerequisitePassed: completeMatrix && passed === PARITY_CASES.length };
}

export function parityOutput(directory = '.cache/m11/parity-01') {
  assert(typeof directory === 'string' && directory.length > 0, 'An output directory is required');
  const output = resolve(ROOT, directory), build = resolve(ROOT, BUILD);
  assert(output.startsWith(resolve(ROOT, '.cache/m11') + sep) || output.startsWith(resolve(ROOT, '.cache/m12') + sep), 'Output must be under validation cache');
  assert(output !== build && !output.startsWith(build + sep) && !build.startsWith(output + sep), 'Output overlaps diagnostic build');
  return output;
}

function workspaceParent(path) {
  let parent = dirname(path);
  while (!existsSync(parent)) parent = dirname(parent);
  const actual = realpathSync(parent), root = realpathSync(ROOT);
  assert(actual === root || actual.startsWith(root + sep), 'Output parent escaped workspace');
  assert.equal(actual, resolve(root, relative(ROOT, parent)), 'Symlinked output parent forbidden');
  if (existsSync(path)) assert.equal(realpathSync(path), resolve(root, relative(ROOT, path)), 'Symlinked output forbidden');
}

function sourcePin() {
  assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '', 'Reviewed clean HEAD required before native parity');
  const paths = git(['ls-files', '-z', '--', 'src', 'apps/desktop', 'tools/m11', 'tools/m9-browser.mjs',
    'tools/m10-output-parity.mjs', 'tools/m10-browser.mjs', 'tools/m105-accounting.mjs',
    'tools/m10-performance.mjs', 'tools/m10-fixtures.mjs', 'tools/build-extension.mjs',
    'index.html', 'vite.config.ts', 'package.json', 'package-lock.json']).split('\0').filter(Boolean);
  assert(paths.includes('tools/m11/parity.mjs'), 'Parity runner must be committed before native execution');
  assert.equal(hash(readFileSync(join(ROOT, MODEL))), MODEL_SHA256, 'Production model changed');
  assert.equal(hash(readFileSync(join(ROOT, MEDIA))), SOURCE_SHA256, 'Exact parity asset required; no fallback');
  return { commit: git(['rev-parse', 'HEAD']), files: Object.fromEntries(paths.map(path => [path, hash(readFileSync(join(ROOT, path)))])),
    modelSha256: MODEL_SHA256, mediaSha256: SOURCE_SHA256 };
}

function windowState() {
  return { visibility: document.visibilityState, focused: document.hasFocus(), width: innerWidth, height: innerHeight,
    outerWidth, outerHeight, x: screenX, y: screenY, dpr: devicePixelRatio,
    screen: { width: screen.width, height: screen.height }, userAgent: navigator.userAgent };
}

function watchSelectedFile() {
  globalThis.m11ParitySelectedFile = new Promise(done => {
    document.querySelector('#file').addEventListener('change', async event => {
      try {
        const file = event.target.files[0];
        const bytes = await file.arrayBuffer();
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        done({ name: file.name, bytes: bytes.byteLength,
          sha256: Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('') });
      } catch (error) { done({ error: String(error) }); }
    }, { once: true, capture: true });
  });
}

export async function runParity(directory = '.cache/m11/parity-01') {
  const output = parityOutput(directory);
  workspaceParent(output); workspaceParent(resolve(ROOT, BUILD));
  assert(!existsSync(output), 'Parity attempt directory must be new');
  git(['check-ignore', '-q', relative(ROOT, output)]);
  git(['check-ignore', '-q', BUILD]);
  mkdirSync(dirname(output), { recursive: true }); mkdirSync(output);
  const report = { schema: 'aethervsr.m11.paused-parity/1', startedAt: new Date().toISOString(), verdict: 'FAIL',
    output: relative(ROOT, output), expected: { modelSha256: MODEL_SHA256, mediaSha256: SOURCE_SHA256 },
    measurement: { performance: 'not measured', physicalRefreshRate: 'not measured', byteDeltas: 'not measured' },
    scope: 'Six desktop-versus-standalone paused cases. Actual runtime importer, production C16D2 and canvas presenter; unchanged M10 captureTask normalizes full RGBA8 digests with no tolerance. Offline temporary COPY_SRC and one synchronous pipeline frame plus readback copy, not an uninstrumented presentation or performance benchmark. Startup video loops only to reach neural readiness. Chrome closes before Electron launches. A parity pass alone does not qualify lifecycle, performance, audio or MVP readiness.',
    bounds: { runMs: 180000, startupMs: 60000, operationMs: 10000, neuralReadyMs: 20000, inPageCaptureMs: 3000, teardownMs: 45000 },
    host: { hostname: hostname(), platform: platform(), release: release(), arch: arch() },
    cases: PARITY_CASES.map(spec => ({ ...spec, verdict: 'NOT_RUN', harness: null, desktop: null, comparison: null })),
    sides: { harness: [], desktop: [] }, errors: [], cleanup: {}, binaries: {}, phase: 'preflight' };
  let server, native, application, built, bounded, watchdog, finishing, saved = false;
  const raw = (name, value) => {
    const bytes = JSON.stringify(value, null, 2) + '\n';
    writeFileSync(join(output, name), bytes, { flag: 'wx' });
    return { path: relative(ROOT, join(output, name)), bytes: Buffer.byteLength(bytes), sha256: hash(bytes) };
  };
  const fail = error => { report.verdict = 'FAIL'; report.errors.push(String(error)); };
  const save = () => {
    if (saved) return;
    report.summary = summarizeParity(report.cases);
    report.parityPrerequisitePassed = report.verdict === 'PASS' && report.errors.length === 0 && report.summary.parityPrerequisitePassed;
    report.finishedAt = new Date().toISOString();
    raw('result.json', report); saved = true;
  };
  const step = async (operation, label, milliseconds = 10000) => {
    assert(!finishing, 'Parity attempt is stopping');
    report.phase = label;
    const result = await bounded(operation, milliseconds, label);
    assert(!finishing, 'Parity attempt stopped');
    return result;
  };
  const acquire = async (factory, label) => {
    let abandoned = false;
    const pending = Promise.resolve().then(factory).then(async resource => {
      if (abandoned || finishing) { await resource.close(); throw new Error(`Late ${label} closed`); }
      return resource;
    });
    try { return await step(pending, label, 60000); }
    catch (error) { abandoned = true; throw error; }
  };
  const closeOwned = async kind => {
    const resource = kind === 'chrome' ? native : kind === 'electron' ? application : server;
    if (!resource) return;
    try { await bounded(resource.close(), 10000, `Close ${kind}`); report.cleanup[kind] = true; }
    catch (error) {
      report.cleanup[kind] = false;
      if (kind === 'electron') application.process().kill('SIGKILL');
      throw error;
    }
    if (kind === 'chrome') native = null;
    else if (kind === 'electron') application = null;
    else server = null;
  };
  const verifyPackage = () => {
    assert(built, 'No diagnostic package');
    const provenance = JSON.parse(readFileSync(join(ROOT, built.directory, 'build-provenance.json'), 'utf8'));
    assert.deepEqual(provenance, built.provenance, 'Desktop provenance changed');
    assert.equal(provenance.diagnostic, true); assert.equal(provenance.sourceDirty, false);
    assert.equal(provenance.sourceCommit, report.sourceBefore.commit); assert.equal(provenance.modelSha256, MODEL_SHA256);
    assert(provenance.inputs.includes('apps/desktop/renderer.ts') && !provenance.inputs.includes('apps/desktop/smoke.ts'), 'Actual renderer required');
    for (const [path, pin] of Object.entries(provenance.files)) {
      const bytes = readFileSync(join(ROOT, built.directory, path));
      assert.equal(bytes.length, pin.bytes); assert.equal(hash(bytes), pin.sha256, `Desktop artifact changed: ${path}`);
    }
    return provenance;
  };
  const finish = () => {
    if (finishing) return finishing;
    finishing = (async () => {
      clearTimeout(watchdog);
      const hardStop = setTimeout(() => { fail('Hard teardown deadline; owned-resource cleanup incomplete'); save(); process.exit(1); }, 45000);
      try {
        for (const kind of ['chrome', 'electron', 'server']) {
          try { await closeOwned(kind); } catch (error) { fail(error); }
        }
        try {
          if (report.sourceBefore) {
            report.sourceAfter = sourcePin(); assert.deepEqual(report.sourceAfter, report.sourceBefore, 'Source changed during parity');
          }
          if (built) report.packageAfter = verifyPackage();
          for (const binary of Object.values(report.binaries)) assert.equal(hash(readFileSync(binary.path)), binary.sha256, 'Browser executable changed');
        } catch (error) { fail(error); }
        save();
      } finally { clearTimeout(hardStop); }
    })();
    return finishing;
  };
  const interrupt = () => { fail('Interrupted; no retry'); void finish().finally(() => process.exit(1)); };
  process.once('SIGINT', interrupt); process.once('SIGTERM', interrupt);
  watchdog = setTimeout(() => { fail('Whole parity run deadline'); void finish().finally(() => process.exit(1)); }, 180000);
  let activeRow;
  try {
    report.sourceBefore = sourcePin();
    const { bounded: deadline } = await import('../m10-browser.mjs'); bounded = deadline;
    const { buildDesktop } = await import('../../apps/desktop/build.mjs');
    built = await step(buildDesktop({ diagnostic: true, outdir: BUILD }), 'Build actual desktop renderer', 60000);
    report.packageBefore = verifyPackage();
    const modelJsonSha256 = hash(JSON.stringify(JSON.parse(readFileSync(join(ROOT, MODEL), 'utf8'))));
    report.expected.modelJsonSha256 = modelJsonSha256;
    if (platform() === 'darwin') report.host.hardware = {
      model: execFileSync('sysctl', ['-n', 'hw.model'], { encoding: 'utf8', timeout: 5000 }).trim(),
      chip: execFileSync('sysctl', ['-n', 'machdep.cpu.brand_string'], { encoding: 'utf8', timeout: 5000 }).trim(),
      memoryBytes: Number(execFileSync('sysctl', ['-n', 'hw.memsize'], { encoding: 'utf8', timeout: 5000 })),
      os: execFileSync('sw_vers', [], { encoding: 'utf8', timeout: 5000 }).trim() };
    const { createServer } = await import('vite');
    server = await acquire(() => createServer({ root: ROOT, mode: 'benchmark',
      server: { host: '127.0.0.1', port: 5209, strictPort: true } }), 'Create owned Vite server');
    await step(server.listen(), 'Listen on strict parity port');
    const origin = 'http://127.0.0.1:5209';
    const servedPins = async () => {
      const read = async path => {
        const response = await fetch(origin + path, { signal: AbortSignal.timeout(10000) });
        assert(response.ok, `Harness HTTP failure: ${path}`); return Buffer.from(await response.arrayBuffer());
      };
      const html = await read('/');
      assert(html.includes('/src/main.ts') && !html.includes('/@vite/client'), 'Actual benchmark-mode root harness required');
      const pins = { index: hash(html) };
      for (const path of ['src/main.ts', 'src/runtime.ts', 'src/core/pipeline.ts']) {
        const source = readFileSync(join(ROOT, path), 'utf8');
        assert((await read(`/${path}?raw`)).toString().startsWith(`export default ${JSON.stringify(source)}`), 'Wrong served harness source');
        pins[path] = hash(source);
      }
      for (const [path, expected] of [['/media/m9/720p60.mp4', SOURCE_SHA256], ['/models/aethersr-c16d2.json', MODEL_SHA256]]) {
        pins[path] = hash(await read(path)); assert.equal(pins[path], expected, `Served asset differs: ${path}`);
      }
      return pins;
    };
    report.harnessServer = { origin, owned: true, before: await step(servedPins(), 'Pin served harness', 60000) };
    const { openNativeChrome } = await import('../m9-browser.mjs');
    const { nativeWindow } = await import('../m105-accounting.mjs');
    native = await acquire(() => openNativeChrome([], { profileDirectory: join(output, 'chrome-profile') }), 'Start native standalone Chrome');
    report.binaries.chrome = { path: native.executable, sha256: hash(readFileSync(native.executable)), version: native.browser.version() };
    const capture = async (page, spec, side) => {
      activeRow = report.cases.find(row => row.id === spec.id);
      const row = activeRow[side] = { startedAt: new Date().toISOString() };
      try {
        row.seek = await step(page.evaluate(seekPaused, { time: spec.time }), `${side} ${spec.id} paused seek`);
        row.seekArtifact = raw(`${side}-${spec.id}-seek.json`, row.seek);
        row.capture = await step(page.evaluate(captureTask(), { extension: false, forceCopy: spec.forceCopy,
          time: spec.time, sourceURL: row.seek.source, modelJsonSha256 }), `${side} ${spec.id} actual capture`);
        row.captureArtifact = raw(`${side}-${spec.id}-capture.json`, row.capture);
        if (side === 'desktop') {
          activeRow.comparison = compareParityCase(spec, row, activeRow.harness, modelJsonSha256);
          activeRow.verdict = activeRow.comparison.verdict;
          assert.equal(activeRow.verdict, 'PASS', `Exact parity mismatch: ${spec.id}; performance blocked; no retry`);
        }
      } catch (error) { row.error = String(error); activeRow.verdict = 'FAIL'; throw error; }
      finally { row.finishedAt = new Date().toISOString(); }
    };
    const prepare = async (page, side, forceCopy, evidence) => {
      await step(waitForObservation(async () => (await page.evaluate(runtimeState, { extension: false })).ready, 20000),
        `${side} actual neural readiness`, 22000);
      evidence.ready = await step(page.evaluate(runtimeState, { extension: false }), `${side} ready evidence`);
      assert(!evidence.ready.error); assert.equal(evidence.ready.importPath, forceCopy ? 'sampled' : 'external');
      if (side === 'desktop') await step(page.evaluate(() => window.m11Desktop.session().pause()), 'Pause desktop session');
      evidence.pause = await step(page.evaluate(pauseRuntime, { extension: false }), `${side} stop acquisition`);
      assert(evidence.pause.paused && !evidence.pause.running);
      evidence.windowBefore = await step(page.evaluate(windowState), `${side} opening window`);
      assert(evidence.windowBefore.visibility === 'visible' && evidence.windowBefore.focused, 'Foreground parity window required');
      assert.deepEqual(sourcePin(), report.sourceBefore, 'Source changed before capture');
    };
    const closeSide = async (page, side, evidence) => {
      try {
        if (side === 'desktop') evidence.cleanup = await bounded(page.evaluate(() => {
          const session = window.m11Desktop.session(), driver = session.runtime;
          session.destroy();
          return { snapshot: session.snapshot(), runtimeReleased: session.runtime === null, gpuReleased: session.gpu === null,
            pipelineDisposed: !driver || driver.pipeline.disposed, paused: window.m11Desktop.video.paused,
            sourceReleased: window.m11Desktop.video.getAttribute('src') === null };
        }), 10000, 'Destroy desktop session');
        else evidence.cleanup = await bounded(page.evaluate(cleanupRuntime, { extension: false }), 10000, 'Destroy harness runtime');
        if (side === 'desktop') assert(evidence.cleanup.runtimeReleased && evidence.cleanup.gpuReleased &&
          evidence.cleanup.pipelineDisposed && evidence.cleanup.paused && evidence.cleanup.sourceReleased, 'Desktop resources not released');
        else assert(evidence.cleanup.stopped && evidence.cleanup.destroyed, 'Harness resources not released');
      } catch (error) { fail(`${side} runtime cleanup: ${error}`); }
    };
    for (const forceCopy of [false, true]) {
      activeRow = report.cases.find(row => row.forceCopy === forceCopy);
      const evidence = { forceCopy, errors: [] }; report.sides.harness.push(evidence);
      const page = await step(native.context.newPage(), 'Create standalone page');
      page.setDefaultTimeout(10000);
      page.on('pageerror', error => evidence.errors.push(String(error))); page.on('crash', () => evidence.errors.push('Page crashed'));
      try {
        await step(page.bringToFront(), 'Foreground standalone before nativeWindow');
        evidence.placement = await step(nativeWindow(page, native.context), 'Bounded native standalone window');
        await step(page.goto(`${origin}/?clip=/media/m9/720p60.mp4&mode=neural${forceCopy ? '&import=copy' : ''}`,
          { waitUntil: 'domcontentloaded', timeout: 15000 }), 'Load standalone harness', 17000);
        await prepare(page, 'harness', forceCopy, evidence);
        for (const spec of PARITY_CASES.filter(row => row.forceCopy === forceCopy)) await capture(page, spec, 'harness');
        evidence.windowAfter = await step(page.evaluate(windowState), 'Standalone closing window');
        assert.deepEqual(evidence.windowAfter, evidence.windowBefore, 'Standalone window/focus changed');
        assert.equal(evidence.errors.length, 0, 'Standalone execution errors');
      } finally { await closeSide(page, 'harness', evidence); await bounded(page.close(), 3000, 'Close standalone page'); }
      assert.equal(report.errors.length, 0, 'Standalone cleanup failed');
    }
    report.harnessServer.after = await step(servedPins(), 'Recheck served harness', 60000);
    assert.deepEqual(report.harnessServer.after, report.harnessServer.before, 'Served harness changed');
    await closeOwned('chrome'); await closeOwned('server');
    activeRow = report.cases[0];
    const executable = (await import('electron')).default;
    const { _electron } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
    const automation = readFileSync(join(ROOT, '.cache/m9/node_modules/playwright/package.json'));
    report.automation = { version: JSON.parse(automation.toString()).version, packageSha256: hash(automation),
      source: '.cache/m9/node_modules/playwright' };
    report.binaries.electron = { path: executable, sha256: hash(readFileSync(executable)) };
    mkdirSync(join(output, 'electron-profile'));
    application = await acquire(() => _electron.launch({ executablePath: executable, args: [resolve(ROOT, built.directory)],
      env: { ...process.env, AETHERVSR_TEST_PROFILE: join(output, 'electron-profile') }, timeout: 30000 }), 'Start actual Electron renderer');
    report.electronEnvironment = await step(application.evaluate(({ app, BrowserWindow }) => ({ versions: process.versions,
      platform: process.platform, arch: process.arch, argv: process.argv, gpu: app.getGPUFeatureStatus(),
      preferences: BrowserWindow.getAllWindows()[0].webContents.getLastWebPreferences() })), 'Record actual Electron environment');
    assert.equal(report.electronEnvironment.versions.electron, built.provenance.electron, 'Electron build/runtime version mismatch');
    const page = await step(application.firstWindow({ timeout: 10000 }), 'Acquire Electron window');
    page.setDefaultTimeout(10000);
    await step(page.waitForURL('aethervsr://app/index.html', { timeout: 10000 }), 'Actual local desktop URL');
    await step(page.bringToFront(), 'Foreground desktop');
    report.desktopSecurity = await step(page.evaluate(() => ({ secure: isSecureContext, node: typeof window.require,
      process: typeof window.process, electron: typeof window.electron, gpu: typeof navigator.gpu })), 'Verify sandboxed renderer');
    assert(report.desktopSecurity.secure && report.desktopSecurity.gpu !== 'undefined');
    for (const name of ['node', 'process', 'electron']) assert.equal(report.desktopSecurity[name], 'undefined');
    const preferences = report.electronEnvironment.preferences;
    assert(preferences.sandbox && preferences.contextIsolation && preferences.webSecurity && !preferences.nodeIntegration, 'Insecure Electron preferences');
    const desktopErrors = [];
    page.on('pageerror', error => desktopErrors.push(String(error))); page.on('crash', () => desktopErrors.push('Page crashed'));
    report.desktopErrors = desktopErrors;
    for (const forceCopy of [false, true]) {
      activeRow = report.cases.find(row => row.forceCopy === forceCopy);
      const evidence = { forceCopy }; report.sides.desktop.push(evidence);
      try {
        await step(page.evaluate(forceCopy => {
          window.m11Desktop.replace({ forceCopy, observe: false }); window.m11Desktop.video.loop = true;
        }, forceCopy), 'Replace desktop session without observer');
        await step(page.evaluate(watchSelectedFile), 'Observe actual selected File bytes');
        await step(page.locator('#file').setInputFiles(join(ROOT, MEDIA)), 'Select exact local MP4 through file input');
        evidence.selectedFile = await step(page.evaluate(() => globalThis.m11ParitySelectedFile), 'Pin selected local File bytes');
        assert.equal(evidence.selectedFile.sha256, SOURCE_SHA256, 'Selected file differs');
        await step(page.locator('#mode').selectOption('neural'), 'Select Prefer neural');
        await step(page.locator('#play').click(), 'Start actual desktop playback');
        await prepare(page, 'desktop', forceCopy, evidence);
        verifyPackage();
        for (const spec of PARITY_CASES.filter(row => row.forceCopy === forceCopy)) await capture(page, spec, 'desktop');
        evidence.windowAfter = await step(page.evaluate(windowState), 'Desktop closing window');
        assert.deepEqual(evidence.windowAfter, evidence.windowBefore, 'Desktop window/focus changed');
        assert.equal(desktopErrors.length, 0, 'Desktop execution errors');
      } finally { await closeSide(page, 'desktop', evidence); }
      assert.equal(report.errors.length, 0, 'Desktop cleanup failed');
    }
    assert(summarizeParity(report.cases).parityPrerequisitePassed, 'Six exact paused comparisons required before performance');
    report.verdict = 'PASS';
  } catch (error) {
    if (activeRow) { activeRow.verdict = 'FAIL'; activeRow.error = String(error); }
    fail(error);
  } finally { await finish(); process.off('SIGINT', interrupt); process.off('SIGTERM', interrupt); }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length === 3 && process.argv[2] === '--help') {
    console.log('node tools/m11/parity.mjs [.cache/m11/parity-01]\nNative execution requires separate authorization, reviewed clean HEAD and prior desktop smoke gate. Six paused external/copy cases at 1, 2, 3 seconds; no performance runs or retries.');
  } else {
    assert(process.argv.length <= 3 && !process.argv[2]?.startsWith('--'), 'Use one new ignored attempt directory');
    const report = await runParity(process.argv[2]);
    console.log(JSON.stringify({ output: report.output, verdict: report.verdict, summary: report.summary, errors: report.errors }));
    process.exitCode = report.verdict === 'PASS' ? 0 : 1;
  }
}