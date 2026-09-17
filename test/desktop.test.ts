import { describe, expect, it } from 'vitest';
import { allowedPermission, allowedRequest, APP_ORIGIN, APP_URL, ASSETS, CONTENT_SECURITY_POLICY, localAsset, SECURE_PREFERENCES } from '../apps/desktop/security.js';

describe('desktop static asset and renderer boundaries', () => {
  it('observes the frame-navigation event that the desktop actually blocks', async () => {
    const { execFileSync } = await import('node:child_process');
    execFileSync(process.execPath, ['--input-type=module', '-e', `
      import assert from 'node:assert/strict';import{armNavigation}from'./tools/m11/journeys.mjs';
      let handler;const BrowserWindow={getAllWindows:()=>[{webContents:{once(event,callback){assert.equal(event,'will-frame-navigate');handler=callback}}}]};
      for(const prevented of [true,false]){
        armNavigation({BrowserWindow});handler({url:'https://m11.invalid/',defaultPrevented:prevented});
        assert.deepEqual(await globalThis.m11Navigation,{url:'https://m11.invalid/',prevented});
      }
    `], { encoding: 'utf8' });
  });

  it('waits for native fullscreen events instead of immediate requested state', async () => {
    const { execFileSync } = await import('node:child_process');
    execFileSync(process.execPath, ['--input-type=module', '-e', `
      import assert from 'node:assert/strict';import{armFullscreen}from'./tools/m11/journeys.mjs';
      const handlers=new Map();const BrowserWindow={getAllWindows:()=>[{once:(event,handler)=>handlers.set(event,handler)}]};
      for(const entered of [true,false]){
        armFullscreen({BrowserWindow},entered);const state=globalThis.m11Fullscreen;
        assert.equal(state.observed,false);assert.equal(state.at,null);
        assert.equal(state.event,entered?'enter-full-screen':'leave-full-screen');
        handlers.get(state.event)();assert.equal(state.observed,true);assert(!Number.isNaN(Date.parse(state.at)));
      }
    `], { encoding: 'utf8' });
  });

  it('uses bounded direct observations without dynamic page evaluation', async () => {
    const { execFileSync } = await import('node:child_process');
    execFileSync(process.execPath, ['--input-type=module', '-e', `
      import assert from 'node:assert/strict';import{readFileSync}from'node:fs';
      import{waitForObservation}from'./tools/m11/parity.mjs';
      const values=[false,false,true];assert(await waitForObservation(async()=>values.shift(),500));
      assert.equal(values.length,0);
      await assert.rejects(waitForObservation(async()=>false,10),/Observation deadline/);
      await assert.rejects(waitForObservation(async()=>{throw Error('observation failed')},100),/observation failed/);
      for(const file of ['parity','journeys'])assert(!readFileSync('tools/m11/'+file+'.mjs','utf8').includes('waitForFunction'));
    `], { encoding: 'utf8' });
  });

  it('does not pass mandatory audio gates when native audibility is unavailable', async () => {
    const { execFileSync } = await import('node:child_process');
    execFileSync(process.execPath, ['--input-type=module', '-e', `
      import assert from 'node:assert/strict';import{observedAudibility}from'./tools/m11/journeys.mjs';
      assert(observedAudibility(true,true));assert(observedAudibility(false,false));
      assert.equal(observedAudibility(false,true),false);
      for(const value of ['not measured',undefined,null,0])assert.throws(()=>observedAudibility(value,true),/unverified/);
    `], { encoding: 'utf8' });
  });

  it('allows fullscreen only from the exact app main frame and denies every privileged permission', () => {
    expect(allowedPermission('fullscreen', APP_URL, true, APP_ORIGIN)).toBe(true);
    expect(allowedPermission('fullscreen', APP_URL, true, APP_URL)).toBe(true);
    for (const permission of ['media', 'display-capture', 'fileSystem', 'openExternal', 'automatic-fullscreen', 'notifications']) {
      expect(allowedPermission(permission, APP_URL, true, APP_ORIGIN)).toBe(false);
    }
    expect(allowedPermission('fullscreen', APP_URL, false, APP_ORIGIN)).toBe(false);
    expect(allowedPermission('fullscreen', undefined, true, APP_ORIGIN)).toBe(false);
    expect(allowedPermission('fullscreen', 'https://example.com', true, APP_ORIGIN)).toBe(false);
    expect(allowedPermission('fullscreen', APP_URL, true, 'https://example.com')).toBe(false);
  });

  it('allows only exact app assets with GET/HEAD and same-origin or headerless requests', () => {
    for (const path of Object.keys(ASSETS)) {
      expect(localAsset(`${APP_ORIGIN}${path}`)).not.toBeNull();
      expect(localAsset(`${APP_ORIGIN}${path}`, 'HEAD', APP_ORIGIN)).not.toBeNull();
    }
    for (const url of [APP_URL + '?file=/etc/passwd', APP_URL + '#remote', 'aethervsr://app/../index.html',
      'aethervsr://app/%69ndex.html', 'aethervsr://user@app/index.html', 'aethervsr://app:80/index.html',
      'file:///etc/passwd', 'https://example.com/', 'aethervsr://other/index.html']) expect(localAsset(url)).toBeNull();
    expect(localAsset(APP_URL, 'POST')).toBeNull();
    expect(localAsset(APP_URL, 'GET', 'https://example.com')).toBeNull();
    expect(localAsset(APP_URL, 'GET', 'null')).toBeNull();
  });

  it('allows app-owned blob resources without enabling arbitrary files or remote documents', () => {
    expect(allowedRequest('blob:aethervsr://app/01234567-0123-4123-8123-0123456789ab')).toBe(true);
    for (const url of ['blob:https://example.com/id', 'blob:null/id', 'file:///video.mp4', 'https://example.com/video.mp4',
      'aethervsr://app/unknown.js', 'data:text/html,hello']) expect(allowedRequest(url)).toBe(false);
  });

  it('requires sandboxed isolated rendering and restrictive local content', () => {
    expect(SECURE_PREFERENCES).toEqual({ nodeIntegration: false, contextIsolation: true, sandbox: true,
      webSecurity: true, webviewTag: false, allowRunningInsecureContent: false, experimentalFeatures: false });
    for (const directive of ["default-src 'none'", "script-src 'self'", "media-src blob:", "frame-src 'none'", "worker-src 'none'", "base-uri 'none'", "form-action 'none'"]) expect(CONTENT_SECURITY_POLICY).toContain(directive);
    expect(CONTENT_SECURITY_POLICY).not.toMatch(/unsafe|https:/);
  });
});

async function checkDesktopBuild(source: string): Promise<void> {
  const { spawnSync } = await import('node:child_process');
  const output = spawnSync(process.execPath, ['--input-type=module', '-e', String.raw`
    import assert from 'node:assert/strict';
    import { createHash } from 'node:crypto';
    import { execFileSync } from 'node:child_process';
    import { lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
    import { createRequire } from 'node:module';
    import { basename, dirname, join, relative, resolve } from 'node:path';
    import { runInNewContext } from 'node:vm';
    import { buildDesktop } from './apps/desktop/build.mjs';
    const ROOT = process.cwd(), cache = join(ROOT, '.cache/m11');
    const publicModel = readFileSync(join(ROOT, 'public/models/aethersr-c16d2.json'));
    const hash = bytes => createHash('sha256').update(bytes).digest('hex');
    const pins = () => ({ sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim(),
      sourceDirty: execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim() !== '' });
    const initialPins = pins();
    const assets = ${JSON.stringify(ASSETS)};
    mkdirSync(cache, { recursive: true });
    const owned = mkdtempSync(join(cache, 'build-test-')), identity = lstatSync(owned);
    function snapshot(directory, prefix = '') {
      const files = [];
      for (const name of readdirSync(directory).sort()) {
        const path = join(directory, name), entry = lstatSync(path), key = prefix + name;
        assert(!entry.isSymbolicLink(), key);
        if (entry.isDirectory()) files.push(...snapshot(path, key + '/'));
        else { assert(entry.isFile(), key); files.push([key, readFileSync(path)]); }
      }
      return files;
    }
    async function build(diagnostic, name) {
      assert(['normal', 'diagnostic'].includes(name));
      const directory = join(owned, name), outdir = relative(ROOT, directory);
      const result = await buildDesktop({ diagnostic, outdir });
      assert.equal(result.directory, outdir);
      const files = new Map(snapshot(directory));
      const provenance = JSON.parse(files.get('build-provenance.json').toString('utf8'));
      assert.deepEqual(provenance, result.provenance);
      assert.equal(provenance.generator, 'aethervsr-desktop');
      assert.equal(provenance.diagnostic, diagnostic);
      assert.deepEqual(pins(), initialPins, 'Build source pins changed during the test');
      for (const [key, value] of Object.entries(initialPins)) assert.equal(provenance[key], value, key);
      assert.equal(provenance.electron, JSON.parse(readFileSync(join(ROOT, 'node_modules/electron/package.json'))).version);
      assert.equal(publicModel.length, 140467);
      assert.deepEqual(files.get('models/production.json'), publicModel, 'Entire production model, including metadata');
      assert.deepEqual(readFileSync(join(ROOT, 'public/models/aethersr-c16d2.json')), publicModel);
      assert.equal(provenance.modelSha256, hash(publicModel));
      const payloadNames = [...Object.values(assets).map(asset => asset.file), 'main.cjs', 'package.json'].sort();
      assert.deepEqual([...files.keys()].sort(), [...payloadNames, 'build-provenance.json'].sort());
      assert.deepEqual(Object.keys(provenance.files).sort(), payloadNames);
      const manifest = Object.fromEntries(payloadNames.map(name => [name, { bytes: files.get(name).length, sha256: hash(files.get(name)) }]));
      assert.deepEqual(provenance.files, manifest);
      assert.equal(provenance.payloadSha256, hash(JSON.stringify(manifest)));
      assert.equal(provenance.bytes, payloadNames.reduce((sum, name) => sum + files.get(name).length, 0));
      const app = JSON.parse(files.get('package.json').toString('utf8'));
      assert.equal(app.private, true);
      assert.equal(app.main, 'main.cjs');
      assert.notEqual(app.type, 'module');
      assert(files.has(app.main));
      assert(provenance.inputs.includes('apps/desktop/main.ts'));
      assert(provenance.inputs.includes('apps/desktop/renderer.ts'));
      return { directory, files, renderer: files.get('renderer.js').toString('utf8'), main: files.get('main.cjs').toString('utf8') };
    }
    function rendererBoundary(renderer) {
      assert.doesNotMatch(renderer, /["'](?:node:|electron["'])/);
      assert.doesNotMatch(renderer, /\b(?:import|export)\s+(?:[^;\n]*?\s+from\s*)?["']|\b(?:import|require)\s*\(\s*["']/);
      assert.doesNotMatch(renderer, /\b(?:ipcRenderer|ipcMain|contextBridge|__dirname|__filename)\b/);
      assert.doesNotMatch(renderer, /\b(?:process|global|Buffer)\s*[.\[(]|\brequire\s*\(/);
    }
    async function secureMain(output, diagnostic) {
      let ready, options, loaded, handler, opening;
      const paths = [], events = new Map();
      const electron = {
        app: { setName() {}, setPath(...args) { paths.push(args); }, on() {}, quit() {},
          exit(code) { throw new Error('Main failed: ' + code); },
          whenReady() { return { then(callback) { return ready = Promise.resolve().then(callback); } }; } },
        protocol: { registerSchemesAsPrivileged() {} },
        session: { defaultSession: { setPermissionCheckHandler() {}, setPermissionRequestHandler() {}, on() {},
          webRequest: { onBeforeRequest() {} }, protocol: { handle(scheme, callback) { assert.equal(scheme, 'aethervsr'); handler = callback; } } } },
        BrowserWindow: class {
          constructor(value) { options = value; }
          webContents = { setWindowOpenHandler(callback) { opening = callback; }, on(name, callback) { events.set(name, callback); } };
          on() {} show() {} destroy() {}
          loadURL(url) { loaded = url; return Promise.resolve(); }
        },
      };
      const require = createRequire(import.meta.url);
      runInNewContext(output.main, { __dirname: output.directory, Response, Uint8Array, console,
        process: { env: { AETHERVSR_TEST_PROFILE: 'test-profile-not-a-real-directory' } },
        require(name) {
          assert(['electron', 'node:fs/promises', 'node:path'].includes(name), 'Unexpected main import: ' + name);
          return name === 'electron' ? electron : require(name);
        } }, { timeout: 1000 });
      await ready;
      assert.equal(loaded, ${JSON.stringify(APP_URL)});
      assert.deepEqual(JSON.parse(JSON.stringify(options.webPreferences)), { ...${JSON.stringify(SECURE_PREFERENCES)}, devTools: diagnostic });
      assert(!Object.hasOwn(options.webPreferences, 'preload'));
      assert.equal(opening().action, 'deny');
      assert.deepEqual(paths, diagnostic ? [['userData', 'test-profile-not-a-real-directory']] : []);
      for (const event of ['will-navigate', 'will-frame-navigate', 'will-redirect', 'will-attach-webview']) {
        let prevented = false;
        events.get(event)({ url: 'https://example.com', preventDefault() { prevented = true; } }, 'https://example.com');
        assert(prevented, event);
      }
      for (const [path, asset] of Object.entries(assets)) {
        const response = await handler(new Request(${JSON.stringify(APP_ORIGIN)} + path));
        assert.equal(response.status, 200);
        assert.equal(response.headers.get('Content-Security-Policy'), ${JSON.stringify(CONTENT_SECURITY_POLICY)});
        assert.equal(response.headers.get('Content-Type'), asset.type);
        assert.equal(response.headers.get('X-Content-Type-Options'), 'nosniff');
        assert.deepEqual(Buffer.from(await response.arrayBuffer()), output.files.get(asset.file));
        const head = await handler(new Request(${JSON.stringify(APP_ORIGIN)} + path, { method: 'HEAD' }));
        assert.equal(head.status, 200); assert.equal((await head.arrayBuffer()).byteLength, 0);
      }
      for (const request of [new Request(${JSON.stringify(APP_URL)}, { method: 'POST' }),
        new Request(${JSON.stringify(APP_URL)}, { headers: { Origin: 'https://example.com' } }),
        ...['/main.cjs', '/package.json', '/build-provenance.json', '/missing', '/%69ndex.html'].map(path => new Request(${JSON.stringify(APP_ORIGIN)} + path))]) {
        assert.equal((await handler(request)).status, 403);
      }
    }
    try {
      ${source}
    } finally {
      assert.equal(dirname(owned), resolve(cache));
      assert.match(basename(owned), /^build-test-[A-Za-z0-9]+$/);
      const current = lstatSync(owned);
      assert(current.isDirectory() && !current.isSymbolicLink());
      assert.equal(current.dev, identity.dev); assert.equal(current.ino, identity.ino);
      rmSync(owned, { recursive: true });
    }
    console.log('checked real desktop builds without native launches');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 30000 });
  if (output.error) throw output.error;
  expect(output.status, output.stderr).toBe(0);
  expect(output.stdout.trim()).toBe('checked real desktop builds without native launches');
}

describe('desktop real build boundaries (no native launches)', () => {
  it('builds the product twice with identical file bytes including provenance and the unchanged public model', async () => checkDesktopBuild(`
    const first = await build(false, 'normal');
    const second = await build(false, 'normal');
    assert.deepEqual([...first.files.keys()], [...second.files.keys()]);
    for (const [name, bytes] of first.files) assert.deepEqual(second.files.get(name), bytes, name);
  `), 30000);

  it('builds a normal renderer without diagnostic or Node bridges and a secure local CommonJS main', async () => checkDesktopBuild(`
    const normal = await build(false, 'normal');
    rendererBoundary(normal.renderer);
    await secureMain(normal, false);
    const forbidden = ['m11Desktop', 'aethervsrRuntime', 'm1010r:identity', 'readAfterPause', 'injectLoss'];
    assert.deepEqual({ renderer: forbidden.filter(marker => normal.renderer.includes(marker)),
      main: ['AETHERVSR_TEST_PROFILE'].filter(marker => normal.main.includes(marker)) },
      { renderer: [], main: [] }, 'Diagnostic strings retained in normal build');
  `), 30000);

  it('builds distinct diagnostics only with the explicit flag while retaining renderer isolation', async () => checkDesktopBuild(`
    const normal = await build(false, 'normal');
    const diagnostic = await build(true, 'diagnostic');
    assert.notDeepEqual(diagnostic.files.get('renderer.js'), normal.files.get('renderer.js'));
    assert.notDeepEqual(diagnostic.files.get('main.cjs'), normal.files.get('main.cjs'));
    for (const marker of ['m11Desktop', 'aethervsrRuntime', 'm1010r:identity', 'readAfterPause'])
      assert(diagnostic.renderer.includes(marker), 'Missing diagnostic marker: ' + marker);
    assert(diagnostic.main.includes('AETHERVSR_TEST_PROFILE'));
    rendererBoundary(diagnostic.renderer);
    await secureMain(diagnostic, true);
    for (const name of ['index.html', 'player.css', 'models/production.json', 'package.json'])
      assert.deepEqual(diagnostic.files.get(name), normal.files.get(name), name);
    assert.deepEqual(new Map(snapshot(normal.directory)), normal.files, 'Diagnostic build overwrote normal output');
  `), 30000);
});

async function checkDesktopParity(source: string): Promise<void> {
  const { execFileSync } = await import('node:child_process');
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { PARITY_CASES, compareParityCase, summarizeParity, parityOutput, runParity } from './tools/m11/parity.mjs';
    const modelJsonSha256 = 'a'.repeat(64);
    const frame = (width, height) => ({ width, height, byteCount: width * height * 4, sha256: 'b'.repeat(64),
      nonuniform: true, alpha: { minimum: 255, maximum: 255, opaquePixels: width * height } });
    const evidence = (spec, source) => ({
      seek: { requestedTime: spec.time, currentTime: spec.time, source, metadata: { mediaTime: spec.time, width: 1280, height: 720 } },
      capture: { requestedTime: spec.time, currentTime: spec.time, source, paused: true, singleSubmission: true,
        input: frame(1280, 720), output: frame(2560, 1440), precision: 'f16', importPath: spec.forceCopy ? 'sampled' : 'external',
        upscalerId: 'mock-not-hardware-evidence', canvasFormat: 'bgra8unorm', modelJsonSha256, packedWeightsSha256: 'c'.repeat(64),
        options: { passDiagnostics: true }, counters: { before: { sequence: 4, framesRendered: 4 }, after: { sequence: 5, framesRendered: 5 } },
        environment: { visibility: 'visible', focused: true }, gpuErrors: [] }
    });
    ${source}
    console.log('checked without native launches');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 15000 });
  expect(output.trim()).toBe('checked without native launches');
}

describe('M11 paused desktop parity (offline evidence tests)', () => {
  it('requires all three external and three copy cases at exactly 1, 2, 3 seconds', async () => checkDesktopParity(`
    assert.deepEqual(PARITY_CASES.map(row => row.id), ['external-1','external-2','external-3','copy-1','copy-2','copy-3']);
    assert.equal(summarizeParity([]).parityPrerequisitePassed, false);
    const rows = PARITY_CASES.map(spec => ({ ...spec, verdict: 'PASS',
      comparison: compareParityCase(spec, evidence(spec, 'blob:desktop'), evidence(spec, 'http://harness/video'), modelJsonSha256) }));
    assert.equal(summarizeParity(rows).parityPrerequisitePassed, true);
    assert.equal(summarizeParity(rows.slice(0, 5)).parityPrerequisitePassed, false);
    assert.equal(summarizeParity([...rows.slice(0, 5), rows[0]]).parityPrerequisitePassed, false);
    rows[3].verdict = 'FAIL'; rows[4].verdict = rows[5].verdict = 'NOT_RUN';
    assert.deepEqual(summarizeParity(rows), { required: 6, passed: 3, failed: 1, notRun: 2, parityPrerequisitePassed: false });
  `));

  it('rejects byte, PTS, model, import, focus, option and submission mismatches without tolerance', async () => checkDesktopParity(`
    const spec = PARITY_CASES[0];
    for (const change of [row => row.capture.output.sha256 = 'd'.repeat(64), row => row.capture.input.sha256 = 'd'.repeat(64),
      row => row.seek.metadata.mediaTime += 1 / 120, row => row.capture.modelJsonSha256 = 'd'.repeat(64),
      row => row.capture.importPath = 'sampled', row => row.capture.options.passDiagnostics = false,
      row => row.capture.environment.focused = false, row => row.capture.environment.visibility = 'hidden',
      row => row.capture.counters.after.sequence++, row => row.capture.gpuErrors.push('validation error')]) {
      const desktop = evidence(spec, 'blob:desktop'); change(desktop);
      assert.equal(compareParityCase(spec, desktop, evidence(spec, 'http://harness/video'), modelJsonSha256).verdict, 'FAIL');
    }
  `));

  it('rejects matching but wrong cases, missing captures and missing decoded metadata', async () => checkDesktopParity(`
    const spec = PARITY_CASES[0];
    const desktop = evidence(PARITY_CASES[1], 'blob:desktop'), harness = evidence(PARITY_CASES[1], 'http://harness/video');
    assert.equal(compareParityCase(spec, desktop, harness, modelJsonSha256).verdict, 'FAIL');
    delete desktop.seek; delete harness.seek;
    assert.equal(compareParityCase(spec, desktop, harness, modelJsonSha256).verdict, 'FAIL');
    assert.throws(() => compareParityCase(spec, {}, harness, modelJsonSha256), /Missing actual capture/);
    assert.throws(() => compareParityCase({ ...spec, time: 4 }, desktop, harness, modelJsonSha256), /Unknown parity case/);
    assert.throws(() => compareParityCase(spec, desktop, harness, ''), /Pinned model/);
  `));

  it('rejects escaped, shared-build and malformed output or ledger inputs before any launch', async () => checkDesktopParity(`
    assert(parityOutput().endsWith('/.cache/m11/parity-01'));
    for (const directory of ['', 'results/parity', '.cache/m11', '.cache/m11/../outside',
      '.cache/m11/parity-app', '.cache/m11/parity-app/nested']) {
      assert.throws(() => parityOutput(directory));
      await assert.rejects(() => runParity(directory));
    }
    for (const rows of [null, undefined, {}, [null], PARITY_CASES.map(spec => ({ ...spec, verdict: 'PASS' }))])
      assert.equal(summarizeParity(rows).parityPrerequisitePassed, false);
  `));

  it('keeps native orchestration serial, actual-renderer-only and separate from performance', async () => checkDesktopParity(`
    const source = runParity.toString();
    assert(source.includes('buildDesktop({ diagnostic: true, outdir: BUILD })'));
    assert(!source.includes('renderer:'));
    assert(source.includes("page.locator('#file').setInputFiles(join(ROOT, MEDIA))"));
    assert(source.includes('window.m11Desktop.replace({ forceCopy, observe: false })'));
    assert(source.includes('page.evaluate(captureTask(), { extension: false'));
    assert(source.includes('page.evaluate(seekPaused, { time: spec.time })'));
    assert(source.indexOf('page.bringToFront()') < source.indexOf('nativeWindow(page, native.context)'));
    assert(source.indexOf("await closeOwned('chrome')") < source.indexOf('_electron.launch('));
    assert(source.indexOf("await closeOwned('server')") < source.indexOf('_electron.launch('));
    assert(source.includes('port: 5209, strictPort: true'));
    assert(source.includes("{ flag: 'wx' }"));
    assert(source.includes("performance: 'not measured'"));
    assert(!source.includes('setViewportSize'));
    assert(!source.includes('smoke.run'));
  `));
});