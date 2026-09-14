import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { ROOT, sha256, startFixtures } from './m10-fixtures.mjs';
import { runPerformance, distribution } from './m10-performance.mjs';
import { bounded, until } from './m10-browser.mjs';
import { installScheduling, installOwnedCost, nativeWindow, deliverySummary, validateScheduling } from './m105-accounting.mjs';

export const REVISIONS = { m9: '2c62a2cf27decc6c88c8636ee85be1e76544a63b', m10: 'acb50f8e69aa2893e221d654edb7d22e7876636e' };
export const ARMS = ['m9-harness', 'm10-harness', 'bare', 'idle', 'no-runtime', 'matched-harness', 'neural', 'baseline', 'legacy-neural'];
const git = (root, ...args) => execFileSync('git', ['-C', root, ...args], { encoding: 'utf8', timeout: 10000 }).trim();

export function parsePlan(value) {
  assert(Array.isArray(value) && value.length > 0 && value.length <= 100, 'Expected 1..100 fixed cases');
  const ids = new Set();
  return value.map(item => {
    assert(item && typeof item === 'object' && Object.keys(item).every(key => ['id', 'arm', 'durationMs', 'diagnostics'].includes(key)), 'Unknown case field');
    assert(typeof item.id === 'string' && /^[\w-]{1,80}$/.test(item.id) && !ids.has(item.id), 'Invalid/duplicate id'); ids.add(item.id);
    assert(ARMS.includes(item.arm), 'Unknown arm');
    assert(Number.isInteger(item.durationMs) && item.durationMs >= 1000 && item.durationMs <= 600000, 'Invalid window');
    const diagnostics = item.diagnostics ?? (item.arm === 'legacy-neural' ? 'legacy' : 'standard');
    assert(['standard', 'lean', 'owned', 'legacy'].includes(diagnostics), 'Unknown diagnostics');
    assert(item.arm !== 'legacy-neural' || diagnostics === 'legacy', 'Legacy replication must use legacy instrumentation');
    assert(diagnostics !== 'owned' || ['neural', 'baseline', 'no-runtime'].includes(item.arm), 'Owned timing requires activated extension');
    const kind = item.arm.endsWith('harness') ? 'harness' : item.arm === 'bare' ? 'no-extension' : item.arm === 'idle' ? 'installed-idle' : item.arm === 'baseline' ? 'extension-baseline' : 'extension-auto';
    return { ...item, diagnostics, kind, noRuntime: item.arm === 'no-runtime', warmupMs: 5000 };
  });
}

export async function revisionServer(name, port) {
  const root = join(ROOT, '.cache/m105', name);
  assert.equal(git(root, 'rev-parse', 'HEAD'), REVISIONS[name]);
  assert.equal(git(root, 'status', '--porcelain'), '');
  const html = readFileSync(join(ROOT, 'tools/m105-fixture.html'));
  const server = await createServer({ root, mode: 'benchmark', server: { host: '127.0.0.1', port, strictPort: true },
    plugins: [{ name: 'm105-matched-fixture', configureServer(instance) {
      instance.middlewares.use((request, response, next) => {
        if (new URL(request.url, 'http://local').pathname !== '/m105') return next();
        response.setHeader('Content-Type', 'text/html; charset=utf-8'); response.end(html);
      });
    } }] });
  await server.listen();
  const origin = `http://127.0.0.1:${port}`;
  const verify = async () => {
    assert.equal(git(root, 'rev-parse', 'HEAD'), REVISIONS[name]); assert.equal(git(root, 'status', '--porcelain'), '');
    const pins = { revision: REVISIONS[name], fixture: sha256(html) };
    const fixture = await fetch(`${origin}/m105`, { signal: AbortSignal.timeout(10000) });
    assert(fixture.ok); assert.equal(sha256(Buffer.from(await fixture.arrayBuffer())), pins.fixture, 'Matched HTML route was replaced');
    for (const path of ['src/main.ts', 'src/runtime.ts', 'src/core/pipeline.ts', 'src/core/upscale/runtime-controller.ts']) {
      const text = readFileSync(join(root, path), 'utf8');
      const response = await fetch(`${origin}/${path}?raw`, { signal: AbortSignal.timeout(10000) });
      assert(response.ok); assert((await response.text()).startsWith(`export default ${JSON.stringify(text)}`), `Wrong source server: ${name}/${path}`);
      pins[path] = sha256(text);
    }
    for (const path of ['models/aethersr-c16d2.json', 'media/m9/720p60.mp4']) {
      const response = await fetch(`${origin}/${path}`, { signal: AbortSignal.timeout(10000) }); assert(response.ok);
      pins[path] = sha256(Buffer.from(await response.arrayBuffer()));
      assert.equal(pins[path], sha256(readFileSync(join(ROOT, 'public', path))), `Media/model differs: ${name}/${path}`);
    }
    return pins;
  };
  try { return { root, origin, pins: await verify(), verify, close: () => server.close() }; }
  catch (error) { await server.close(); throw error; }
}

function frozenBuild(root) {
  assert.equal(git(ROOT, 'status', '--porcelain'), '', 'Commit apparatus before measurements');
  assert.equal(git(root, 'status', '--porcelain'), ''); assert.equal(git(root, 'rev-parse', 'HEAD'), REVISIONS.m10);
  const directory = join(root, 'dist-extension');
  const provenance = JSON.parse(readFileSync(join(directory, 'build-provenance.json')));
  assert.equal(provenance.sourceCommit, REVISIONS.m10); assert.equal(provenance.sourceDirty, false);
  const manifest = JSON.parse(readFileSync(join(directory, 'manifest.json')));
  assert.deepEqual(manifest.permissions, ['activeTab', 'scripting', 'storage']);
  const names = ['content.js', 'manifest.json', 'models/production.json', 'popup.css', 'popup.html', 'popup.js', 'service-worker.js'];
  assert.deepEqual(Object.keys(provenance.files).sort(), names);
  for (const [path, record] of Object.entries(provenance.files)) {
    const bytes = readFileSync(join(directory, path)); assert.equal(bytes.length, record.bytes); assert.equal(sha256(bytes), record.sha256);
  }
  assert.equal(provenance.bundleSha256, sha256(names.map(name => `${name}\0${provenance.files[name].sha256}\n`).join('')));
  assert.equal(provenance.files['models/production.json'].sha256, 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a');
  assert(!readFileSync(join(directory, 'content.js'), 'utf8').includes('__AETHERVSR_EXTENSION_TEST__'));
  assert.equal(provenance.bundleSha256, '724507ec7d3e7a8c9a4ce7bf0fc772f2d2de9f56c1218693db52d6cba195258a');
  return { directory, provenance, manifest, apparatusCommit: git(ROOT, 'rev-parse', 'HEAD') };
}

export async function runComparison(casesPath, prefix) {
  let m9, m10;
  try {
    m9 = await revisionServer('m9', 5186); m10 = await revisionServer('m10', 5187);
    const pins = { m9: m9.pins, m10: m10.pins, 'models/aethersr-c16d2.json': m10.pins['models/aethersr-c16d2.json'] };
    return await runPerformance(casesPath, prefix, {
      parseCases: parsePlan, verifyBuild: () => frozenBuild(m10.root), accounting: item => item.diagnostics !== 'legacy',
      rawByteLimit: 12 * 1024 * 1024,
      sources: ['tools/m105-accounting.mjs', 'tools/m105-compare.mjs', 'tools/m105-fixture.html'],
      description: { milestone: 'M10.5', acceptance: 'Exploratory attribution/replication, not final acceptance',
        sourceRoots: { m9: m9.root, m10: m10.root }, pins,
        comparison: 'Original M9/M10 root harness UI retained. All other arms use byte-identical matched HTML, 640x360 at (40,120), white background, no controls; 2560x1440 runtime output. Standard scheduling/metadata observer common; owned callback instrumentation only in separately labelled diagnostic arms.',
        metric: 'Historical combined retained; unique frame loss/overlap unavailable',
        placement: 'Native 1200x820 viewport at desktop(40,40), cleared device metrics override, no focus forcing during capture',
        rawPolicy: 'Full raw local only, compact hash-indexed summaries for Git; 12MiB local compressed-file limit is not a repository-cap change' },
      startFixtures: async () => { const fixtures = await startFixtures({ mse: false }); return { ...fixtures, legacyUrl: fixtures.url, url: `${m10.origin}/m105` }; },
      harnessServer: async () => ({ origin: m10.origin, owned: true, pins,
        verify: async () => ({ m9: await m9.verify(), m10: await m10.verify(), 'models/aethersr-c16d2.json': m10.pins['models/aethersr-c16d2.json'] }), close: async () => {} }),
      url: (item, fixtures) => item.arm === 'legacy-neural' ? `${fixtures.legacyUrl}?case=custom`
        : item.arm === 'm9-harness' ? `${m9.origin}/?mode=auto&clip=/media/m9/720p60.mp4`
        : item.arm === 'm10-harness' ? `${m10.origin}/?mode=auto&clip=/media/m9/720p60.mp4`
          : `${m10.origin}/m105?consumer=${item.arm === 'matched-harness' ? 'harness' : 'none'}`,
      preparePage: async (page, native, item) => {
        const placement = await nativeWindow(page, native.context);
        if (item.diagnostics === 'legacy') {
          await page.setViewportSize({ width: 1200, height: 820 });
          placement.deviceMetricsOverride = true; placement.scope = 'M10 viewport-emulated 1200x820 protocol; native window pinned to built-in display. Screen API is not physical screen geometry.';
        }
        if (!['lean', 'legacy'].includes(item.diagnostics)) await page.addInitScript(installScheduling);
        const workerEvents = native.m105WorkerEvents = [];
        for (const worker of native.context.serviceWorkers()) worker.once('close', () => workerEvents.push({ at: Date.now(), type: 'worker-close' }));
        native.context.on('serviceworker', worker => workerEvents.push({ at: Date.now(), type: 'worker-start', url: worker.url() }));
        return placement;
      },
      beforeActivation: async (native, page, item, popup) => {
        if (!item.noRuntime && item.diagnostics !== 'owned') return;
        const func = item.diagnostics === 'owned' ? installOwnedCost : function() { navigator.gpu.requestAdapter = async () => null; return { denied: true }; };
        const inject = new Function('tab', `return chrome.scripting.executeScript({target:{tabId:tab},world:'ISOLATED',func:${func.toString()},args:[${JSON.stringify({ noRuntime: item.noRuntime })}]});`);
        await native.workerEval(inject, popup.tabId);
      },
      ready: async (native, page, item, result) => {
        if (item.noRuntime) await until(() => native.isolated(page, () => globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)].status()), value => value.code === 'webgpu-unavailable');
        result.pageGeometry = await page.evaluate(() => ({ video: document.querySelector('video').getBoundingClientRect().toJSON(),
          canvases: [...document.querySelectorAll('canvas')].map(canvas => ({ rect: canvas.getBoundingClientRect().toJSON(), width: canvas.width, height: canvas.height })),
          background: getComputedStyle(document.body).backgroundColor, controls: document.querySelector('video').controls,
          screen: { width: screen.width, height: screen.height, availLeft: screen.availLeft, availTop: screen.availTop, availWidth: screen.availWidth, availHeight: screen.availHeight, dpr: devicePixelRatio, x: screenX, y: screenY } }));
        if (item.diagnostics === 'legacy') assert.equal(result.pageGeometry.screen.dpr, 1, 'Legacy DPR changed');
        if (!['m9-harness', 'm10-harness', 'legacy-neural'].includes(item.arm)) {
          const geometry = result.pageGeometry;
          assert.equal(geometry.video.x, 40); assert.equal(geometry.video.y, 120); assert.equal(geometry.video.width, 640); assert.equal(geometry.video.height, 360);
          assert.equal(geometry.controls, false); assert.equal(geometry.background, 'rgb(255, 255, 255)');
          for (const canvas of geometry.canvases) { assert.equal(canvas.width, 2560); assert.equal(canvas.height, 1440); assert.deepEqual(canvas.rect, geometry.video); }
        }
      },
      collect: async (native, page, item, result) => {
        let scheduling = null, owned = null;
        const closingScreen = await page.evaluate(() => ({ width: screen.width, height: screen.height, availLeft: screen.availLeft, availTop: screen.availTop, availWidth: screen.availWidth, availHeight: screen.availHeight, dpr: devicePixelRatio, x: screenX, y: screenY }));
        assert.deepEqual(closingScreen, result.pageGeometry.screen, 'Window/display changed');
        if (!['lean', 'legacy'].includes(item.diagnostics)) {
          await bounded(page.evaluate(() => globalThis[Symbol.for('aethervsr.m105.scheduling')].done), 3000);
          scheduling = await page.evaluate(() => { const data = globalThis[Symbol.for('aethervsr.m105.scheduling')]; return { ...data, done: undefined }; });
          validateScheduling(scheduling);
        }
        if (item.diagnostics === 'owned') owned = await native.isolated(page, () => { const data = globalThis[Symbol.for('aethervsr.m105.owned')]; return { ...data, restore: undefined }; });
        return { scheduling, owned, workerEvents: native.m105WorkerEvents, sourceCommit: item.arm === 'm9-harness' ? REVISIONS.m9 : REVISIONS.m10,
          bundleSha256: item.kind.startsWith('extension-') || item.kind === 'installed-idle' ? frozenBuild(m10.root).provenance.bundleSha256 : null };
      },
      summarize: raw => ({ ...deliverySummary(raw), rafIntervalsMs: distribution(raw.scheduling?.raf.map(row => row[2]) ?? []),
        longTasks: raw.scheduling?.tasks ?? null, mediaEvents: raw.scheduling?.events ?? null, owned: raw.owned,
        workerEvents: raw.workerEvents, schedulingOverflow: raw.scheduling?.overflow ?? null }),
    });
  } finally { await m9?.close(); await m10?.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert(process.argv.length === 4 && existsSync(process.argv[2]), 'Usage: node tools/m105-compare.mjs CASES.json .cache/m105/PREFIX');
  await runComparison(resolve(process.argv[2]), resolve(process.argv[3]));
}