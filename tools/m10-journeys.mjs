import assert from 'node:assert/strict';
import { existsSync, mkdirSync, openSync, writeSync, ftruncateSync, closeSync, readFileSync } from 'node:fs';
import { resolve, dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { hostname, platform, release, arch, cpus } from 'node:os';
import { execFileSync } from 'node:child_process';
import { ROOT, sha256, startFixtures } from './m10-fixtures.mjs';
import { openExtension, snapshotExtensionStatus, until, verifyBuild } from './m10-browser.mjs';

class Unverified extends Error {}
function pageSnapshot(extensionId) {
  const rect = element => { const bounds = element.getBoundingClientRect(); return { left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height }; };
  const main = document.querySelector('main').cloneNode(true); main.querySelectorAll('canvas').forEach(canvas => canvas.remove());
  return { href: location.href, visibility: document.visibilityState, focused: document.hasFocus(), originalHTML: main.innerHTML,
    mainTestHook: typeof globalThis.__AETHERVSR_EXTENSION_TEST__, mainSingleton: typeof globalThis[Symbol.for(`aethervsr.m10.document.${extensionId}`)],
    fullscreen: document.fullscreenElement?.tagName ?? null, pip: !!document.pictureInPictureElement,
    persisted: document.documentElement.dataset.pageshowPersisted ?? null,
    videos: [...document.querySelectorAll('video')].map(video => ({ src: video.getAttribute('src'), currentSrc: video.currentSrc, crossorigin: video.getAttribute('crossorigin'),
      paused: video.paused, controls: video.controls, mediaKeys: !!video.mediaKeys, width: video.videoWidth, height: video.videoHeight, readyState: video.readyState, rect: rect(video) })),
    canvases: [...document.querySelectorAll('canvas')].map(canvas => { const style = getComputedStyle(canvas); return { uuid: canvas.dataset.aethervsrM10 ?? null,
      siblingOfVideo: canvas.previousElementSibling?.tagName === 'VIDEO', videoRect: canvas.previousElementSibling?.tagName === 'VIDEO' ? rect(canvas.previousElementSibling) : null,
      rect: rect(canvas), width: canvas.width, height: canvas.height,
      pointerEvents: style.pointerEvents, visibility: style.visibility, objectFit: style.objectFit, clipPath: style.clipPath, radius: style.borderRadius }; }) };
}
function resourceCheck(status, stopped = false) {
  const details = status.details; assert(details?.infrastructure, 'Missing resource diagnostics');
  const counts = details.infrastructure;
  assert(counts.maximumConcurrent <= 1); assert.equal(counts.created - counts.destroyed, stopped ? 0 : status.owner ? 1 : 0);
  if (stopped) {
    assert.equal(status.code, 'inactive'); assert.equal(status.owner, null); assert.equal(status.current, null); assert.equal(details.attachment, null);
    assert.equal(status.enabled, false); assert.equal(details.discoveryActive, false); assert.equal(details.timerCount, 0);
    if (details.lastTeardown) assert(Object.values(details.lastTeardown).every(value => value === 0), 'Live resources after disable');
  } else if (status.code === 'active') {
    for (const key of ['device', 'pipeline', 'canvas', 'resizeObservers']) assert.equal(details.attachment.resources[key], 1, key);
    assert.equal(details.attachment.infrastructure.cleanupErrors, 0);
  }
}

export async function runJourneys(output, testBuild = false) {
  output = resolve(output); assert(!existsSync(output), 'Never overwrite raw evidence');
  const report = { schemaVersion: 1, started: new Date().toISOString(), testBuild, performance: 'not measured',
    scope: 'Native unpacked-extension lifecycle assertions. Popup focus interrupts visibility; status timing counters are diagnostic snapshots, not benchmarks. No pixel readback or parity claim.',
    manual: { visual: 'UNVERIFIED: review local screenshots for object-fit, clipping, radii and caption stacking', encryptedStream: 'UNVERIFIED: ClearKey test attaches MediaKeys to clear media only', displayRefreshRate: 'not measured' },
    machine: { hostname: hostname(), os: platform(), release: release(), arch: arch(), cpu: cpus()[0]?.model ?? 'not measured',
      osVersion: platform() === 'darwin' ? execFileSync('sw_vers', ['-productVersion'], { encoding: 'utf8' }).trim() : release() }, journeys: [], events: [] };
  let descriptor, native, fixtures, watchdog;
  const save = () => { if (descriptor === undefined) return; const bytes = Buffer.from(`${JSON.stringify(report)}\n`); writeSync(descriptor, bytes, 0, bytes.length, 0); ftruncateSync(descriptor, bytes.length); };
  const emit = (name, data) => { report.events.push({ name, at: new Date().toISOString(), data }); save(); };
  const cleanup = async () => { try { await native?.close(); } finally { await fixtures?.close(); } };
  const abort = async signal => { report.fatal = `Interrupted: ${signal}`; save(); const timer = setTimeout(() => process.exit(2), 10000); try { await cleanup(); } finally { clearTimeout(timer); process.exit(2); } };
  const interrupted = () => { void abort('signal'); };
  try {
    report.build = verifyBuild(testBuild);
    mkdirSync(dirname(output), { recursive: true }); descriptor = openSync(output, 'wx'); save();
    report.harness = Object.fromEntries(['tools/m10-journeys.mjs', 'tools/m10-browser.mjs', 'tools/m10-fixtures.mjs', 'tools/m10-fixtures/index.html', 'tools/m9-browser.mjs']
      .map(name => [name, sha256(readFileSync(join(ROOT, name)))]));
    process.once('SIGINT', interrupted); process.once('SIGTERM', interrupted);
    watchdog = setTimeout(() => { void abort('15 minute watchdog'); }, 900000);
    fixtures = await startFixtures(); report.media = fixtures.evidence;
    native = await openExtension(report.build, emit); native.context.setDefaultTimeout(10000); native.context.setDefaultNavigationTimeout(15000);
    const screenshots = join(ROOT, '.cache/m10/screenshots', `${Date.now()}-${sha256(output).slice(0, 8)}`); mkdirSync(screenshots, { recursive: true });
    const dom = page => page.evaluate(pageSnapshot, native.extensionId);
    const ready = page => page.waitForFunction(() => [...document.querySelectorAll('video')].every(video => video.readyState >= 2 && video.videoWidth > 0));
    const popup = async (page, fn) => { const panel = await native.popup(page); try { return await fn(panel); } finally { await panel.dismiss(); } };
    const enable = page => popup(page, async panel => { const status = await panel.click('#enable'); assert(status.enabled, status.message); return panel.tabId; });
    const status = async (id, accept = value => value.code === 'active') => {
      const value = await until(() => native.inspect(id), accept, 20000); resourceCheck(value); emit('status', value); return value;
    };
    const disable = async (page, original) => {
      const value = await popup(page, panel => panel.click('#disable')); resourceCheck(value, true);
      const restored = await dom(page); assert.equal(restored.canvases.length, 0); assert.equal(restored.originalHTML, original.originalHTML, 'Page DOM/styles changed');
      emit('disabled-restored', { status: value, dom: restored }); return value;
    };
    const action = async (page, name, optional = false) => {
      await page.locator(`[data-action="${name}"]`).first().click();
      await page.waitForFunction(() => ['done', 'error'].includes(document.querySelector('#result').dataset.state));
      const value = await page.locator('#result').evaluate(element => ({ state: element.dataset.state, action: element.dataset.action, message: element.textContent }));
      emit('page-owned-action', value); assert.equal(value.action, name);
      if (value.state === 'error') { if (optional) throw new Unverified(value.message); throw new Error(value.message); }
    };
    const screenshot = async (page, name) => { const path = join(screenshots, `${name.replaceAll('/', '-')}.png`); await page.screenshot({ path }); emit('screenshot', { path: relative(ROOT, path), sha256: sha256(readFileSync(path)), verdict: 'UNVERIFIED manual pixels' }); };
    const isolatedStatus = async page => snapshotExtensionStatus({ ok: true, status: await native.isolated(page, () => globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)].status()) });
    const active = async (page, id) => {
      const value = await status(id); const view = await dom(page);
      assert.equal(view.visibility, 'visible'); assert.equal(view.focused, true); assert.equal(view.mainTestHook, 'undefined'); assert.equal(view.mainSingleton, 'undefined');
      assert.equal(view.canvases.length, 1); const canvas = view.canvases[0];
      assert.match(canvas.uuid, /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i); assert(canvas.siblingOfVideo);
      assert.equal(canvas.pointerEvents, 'none'); assert.equal(canvas.visibility, 'visible'); assert(canvas.width > 0 && canvas.height > 0);
      const expected = value.details.attachment.cssRect;
      for (const key of ['left', 'top', 'width', 'height']) { assert(Math.abs(canvas.rect[key] - expected[key]) < 1, key); assert(Math.abs(canvas.rect[key] - canvas.videoRect[key]) < 1, `Original video ${key}`); }
      emit('active-dom', view); return value;
    };
    let sequence = 0;
    const journey = async (name, fixture, body) => {
      const item = { name, fixture, verdict: 'RUNNING', started: new Date().toISOString() }; report.journeys.push(item); save();
      const page = await native.context.newPage(); page.on('pageerror', error => emit('page-error', { name, error: String(error) }));
      try {
        await page.goto(`${fixtures.url}?case=${encodeURIComponent(fixture)}&journey=${++sequence}`); await page.bringToFront(); await ready(page);
        const original = await dom(page); assert.equal(original.canvases.length, 0); emit('journey-before', { name, dom: original });
        await body(page, original); item.verdict = 'PASS';
      } catch (error) {
        item.verdict = error instanceof Unverified ? 'UNVERIFIED' : 'FAIL'; item.error = String(error);
        try { emit('failure-dom', { name, dom: await dom(page) }); await screenshot(page, `${sequence}-failure`); } catch (captureError) { item.captureError = String(captureError); }
      } finally { await page.close().catch(error => { item.cleanupError = String(error); item.verdict = 'FAIL'; }); item.finished = new Date().toISOString(); save(); }
      console.log(`${item.verdict} ${name}`);
    };
    const accessDenied = async page => {
      const id = await native.tabId(page);
      const result = await native.workerEval(async tabId => { try { await chrome.scripting.executeScript({ target: { tabId, frameIds: [0] }, world: 'ISOLATED', func: () => location.href }); return { denied: false }; } catch (error) { return { denied: true, error: String(error) }; } }, id);
      assert(result.denied, 'Unexpected access without action grant'); assert.equal((await dom(page)).canvases.length, 0); emit('negative-access', result);
    };
    await journey('installed but unactivated', 'custom', async page => { await page.waitForFunction(() => document.querySelector('video').getVideoPlaybackQuality().totalVideoFrames >= 3); await accessDenied(page); assert.equal((await dom(page)).mainTestHook, 'undefined'); });
    await journey('actual activation, isolated model, duplicate enable', 'custom', async (page, original) => {
      const id = await enable(page); const before = await active(page, id);
      const model = await native.isolated(page, async () => { const result = await chrome.runtime.sendMessage({ type: 'm10.model' }); if (!result.ok) return result;
        const bytes = new TextEncoder().encode(result.modelJson); const digest = await crypto.subtle.digest('SHA-256', bytes);
        return { ok: true, sha256: result.sha256, bytes: bytes.length, observedHash: [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join(''), testHook: typeof globalThis.__AETHERVSR_EXTENSION_TEST__ }; });
      assert(model.ok); assert.equal(model.observedHash, report.build.provenance.modelSha256); assert.equal(model.sha256, model.observedHash); assert.equal(model.bytes, report.build.provenance.modelBytes);
      assert.equal(model.testHook, testBuild ? 'object' : 'undefined'); emit('top-isolated-model', model);
      const duplicate = await popup(page, panel => panel.request('m10.enable'));
      assert.equal(duplicate.owner, before.owner); assert.equal(duplicate.details.infrastructure.created, before.details.infrastructure.created); resourceCheck(duplicate); emit('duplicate-from-real-popup', duplicate);
      await disable(page, original);
    });
    for (const fixture of ['custom', 'cors', 'contain', 'cover', 'clipped', 'radius', 'translated', 'mse', 'controls/captions']) {
      await journey(`supported ${fixture}`, fixture, async (page, original) => {
        const id = await enable(page); await active(page, id);
        if (fixture === 'cors') assert.equal((await dom(page)).videos[0].crossorigin, 'anonymous');
        if (['contain', 'cover'].includes(fixture)) assert.equal((await dom(page)).canvases[0].objectFit, fixture);
        if (fixture === 'controls/captions') {
          const pointer = await page.locator('[data-action="pause"]').evaluate(button => { const rect = button.getBoundingClientRect(); return document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2)?.closest('[data-action]')?.dataset.action; });
          assert.equal(pointer, 'pause'); await action(page, 'pause'); assert((await dom(page)).videos[0].paused);
          await action(page, 'play'); assert(!(await dom(page)).videos[0].paused);
        }
        await screenshot(page, `${sequence}-${fixture}-white`); await page.evaluate(() => { document.body.style.background = 'black'; }); await screenshot(page, `${sequence}-${fixture}-black`);
        await disable(page, original);
      });
    }
    for (const [fixture, expected] of [['native', 'unsupported-controls'], ['nocors', 'cors-blocked'], ['iframe', 'unsupported-frame'], ['sameiframe', 'unsupported-frame']]) {
      await journey(`explicit unsupported ${fixture}`, fixture, async (page, original) => {
        const id = await enable(page); const value = await status(id, item => item.code === expected); assert.equal(value.owner, null); assert.equal((await dom(page)).canvases.length, 0);
        if (fixture.includes('iframe')) {
          assert(value.embeddedFrames > 0); assert.equal(value.candidates, 0);
          const frames = await until(() => page.frames().filter(frame => frame !== page.mainFrame() && frame.url().includes('/fixture?case=custom')), frames => frames.length === 1);
          for (const frame of frames) { await frame.waitForSelector('main'); await ready(frame); const view = await frame.evaluate(pageSnapshot, native.extensionId); assert.equal(view.canvases.length, 0); assert.equal(view.mainTestHook, 'undefined'); emit('uninspected-frame-dom', view); }
        }
        if (fixture === 'nocors') assert.equal((await dom(page)).videos[0].crossorigin, null);
        await disable(page, original);
      });
    }
    await journey('multiple video single owner', 'multiple', async (page, original) => { const id = await enable(page); const value = await active(page, id); assert.equal(value.candidates, 2); assert.equal(value.details.infrastructure.maximumConcurrent, 1); await disable(page, original); });
    await journey('late inserted video', 'late', async page => { const id = await enable(page); await status(id, item => item.code === 'no-video'); await action(page, 'insert'); await ready(page); const original = await dom(page); await active(page, id); await disable(page, original); });
    await journey('SPA replacement and source change', 'spa', async page => {
      const id = await enable(page); const before = await active(page, id); await action(page, 'route'); await ready(page);
      await status(id, item => item.code === 'active' && item.owner !== before.owner); await action(page, 'source'); await ready(page);
      const original = await dom(page); await active(page, id); await disable(page, original);
    });
    await journey('ClearKey MediaKeys detection only', 'drm', async (page, original) => { await action(page, 'drm', true); assert((await dom(page)).videos[0].mediaKeys); const id = await enable(page); await status(id, item => item.code === 'protected-media'); await disable(page, original); });
    await journey('20 enable-disable resource and DOM cycles', 'custom', async (page, original) => {
      let previous = 0;
      for (let cycle = 1; cycle <= 20; cycle++) { const id = await enable(page); await active(page, id); const value = await disable(page, original);
        assert.equal(value.details.infrastructure.created, previous + 1); assert.equal(value.details.infrastructure.destroyed, previous + 1); previous++;
        emit('cycle', { cycle, infrastructure: value.details.infrastructure, lastTeardown: value.details.lastTeardown }); }
    });
    await journey('hostile overlay removal stays bounded', 'custom', async (page, original) => {
      const id = await enable(page); const before = await active(page, id); await page.evaluate(() => document.querySelector('canvas[data-aethervsr-m10]').remove());
      const removed = await status(id, item => item.code === 'unsupported-geometry' && item.owner === null);
      const start = Date.now(); while (Date.now() - start < 1500) { const value = await native.inspect(id); assert.equal(value.details.infrastructure.created, before.details.infrastructure.created); assert.equal((await dom(page)).canvases.length, 0); await new Promise(done => setTimeout(done, 100)); }
      emit('removal-observation', { observedMs: Date.now() - start, status: removed }); await disable(page, original);
    });
    for (const kind of ['fullscreen', 'directfs', 'pip']) await journey(`page-owned ${kind}`, kind === 'pip' ? 'pip' : 'fullscreen', async (page, original) => {
      const id = await enable(page); await active(page, id); await action(page, kind, true);
      await page.waitForFunction(kind => kind === 'pip' ? !!document.pictureInPictureElement : !!document.fullscreenElement, kind);
      if (kind === 'fullscreen') { await active(page, id); assert(await page.evaluate(() => document.fullscreenElement.contains(document.querySelector('canvas[data-aethervsr-m10]')))); }
      else { const reason = kind === 'pip' ? 'picture-in-picture' : 'video-fullscreen'; await status(id, item => item.code === 'suspended' && item.details.attachment?.suspendedReason === reason); assert.equal((await dom(page)).canvases[0].visibility, 'hidden'); }
      await screenshot(page, `${sequence}-${kind}`);
      await page.evaluate(async () => { if (document.pictureInPictureElement) await document.exitPictureInPicture(); if (document.fullscreenElement) await document.exitFullscreen(); });
      if ((await dom(page)).videos[0].paused) await action(page, 'play');
      await active(page, id); await disable(page, original);
    });
    await journey('native service-worker stop and restart', 'custom', async (page, original) => {
      const id = await enable(page); await active(page, id); await native.stopWorker(page); await until(() => native.workerRunning(), running => !running);
      const before = await isolatedStatus(page);
      const continued = await until(() => isolatedStatus(page), item => item.details.attachment.session.framesRendered > before.details.attachment.session.framesRendered + 2);
      assert.equal(await native.workerRunning(), false, 'Worker restarted before popup query');
      assert.equal(continued.owner, before.owner); assert.deepEqual(continued.details.infrastructure.created, before.details.infrastructure.created); emit('content-with-worker-stopped', continued);
      const resumed = await popup(page, panel => panel.request('m10.status')); assert.equal(resumed.owner, before.owner); assert.equal(resumed.details.infrastructure.created, before.details.infrastructure.created);
      await until(() => native.workerRunning()); emit('worker-restarted-popup-status', resumed); await disable(page, original);
    });
    await journey('unenabled second tab remains untouched', 'custom', async (page, original) => {
      const id = await enable(page); const before = await active(page, id); const other = await native.context.newPage();
      try { await other.goto(`${fixtures.url}?case=custom&second=1`); await ready(other); await accessDenied(other); const view = await dom(other); assert.equal(view.mainTestHook, 'undefined'); emit('second-tab', view); }
      finally { await other.close(); }
      await page.bringToFront(); const resumed = await active(page, id); assert.equal(resumed.owner, before.owner); await disable(page, original);
    });
    await journey('active origin navigation revokes access', 'custom', async page => {
      const id = await enable(page); await active(page, id); await page.goto('http://localhost:5183/fixture?case=custom&active-navigation=1');
      await ready(page); await accessDenied(page); const view = await dom(page); assert.equal(view.mainTestHook, 'undefined'); emit('active-origin-navigation', view);
    });
    await journey('disabled back navigation preserves only mode', 'custom', async (page, original) => {
      const id = await enable(page); await active(page, id); await popup(page, panel => panel.click('input[value="baseline"]')); await disable(page, original);
      await page.goto('http://localhost:5183/fixture?case=custom&origin-change=1'); await ready(page); await accessDenied(page);
      await page.goBack(); await ready(page); const restored = await dom(page); assert.equal(restored.canvases.length, 0);
      const value = await popup(page, panel => panel.request('m10.status')); assert.equal(value.enabled, false); assert.equal(value.mode, 'baseline');
      emit('back-navigation', { status: value, dom: restored, bfcache: restored.persisted === 'true' ? 'observed persisted pageshow' : 'UNVERIFIED: browser did not restore BFCache' });
    });
    if (testBuild) for (const fixture of ['cors', 'nocors', 'custom']) await journey(`isolated diagnostics ${fixture}`, fixture, async (page, original) => {
      const id = await enable(page); await status(id, item => fixture === 'nocors' ? item.code === 'cors-blocked' : item.code === 'active'); await disable(page, original);
      const options = fixture === 'custom' ? { withheldFeatures: ['timestamp-query', 'shader-f16'] } : { forceCopy: true };
      const configured = await native.isolated(page, options => { const hook = globalThis.__AETHERVSR_EXTENSION_TEST__; if (hook.status().enabled) throw new Error('Configuration requires disabled state'); hook.configure(options); return { status: hook.status(), attachment: hook.attachment() }; }, options);
      assert.equal(configured.attachment, null); emit('test-only-configuration', { options, ...configured });
      await popup(page, panel => panel.click('input[value="auto"]')); await enable(page);
      const value = await status(id, item => fixture === 'nocors' ? item.code === 'cors-blocked' : item.code === 'active');
      if (fixture === 'cors') { const path = await native.isolated(page, () => globalThis.__AETHERVSR_EXTENSION_TEST__.attachment().pipeline.stats(performance.now()).importPath); assert.equal(path, 'sampled'); emit('forced-copy-path', path); }
      if (fixture === 'custom') { assert.equal(value.mode, 'auto'); assert.equal(value.current, 'baseline'); assert.equal(value.details.attachment.controllerState, 'unavailable'); assert.equal(value.details.attachment.features.timestampQuery, false); assert.equal(value.details.attachment.features.shaderF16, false); }
      await disable(page, original);
    });
    const finalBuild = verifyBuild(testBuild, relative(ROOT, output).startsWith('..') ? undefined : output);
    assert.equal(finalBuild.provenanceSha256, report.build.provenanceSha256);
  } catch (error) { report.fatal = String(error); }
  finally {
    clearTimeout(watchdog); process.off('SIGINT', interrupted); process.off('SIGTERM', interrupted);
    try { await cleanup(); } catch (error) { report.cleanupError = String(error); }
    report.requests = fixtures?.requests ?? []; report.finished = new Date().toISOString();
    report.verdict = report.fatal || report.cleanupError || report.journeys.some(item => item.verdict === 'FAIL') ? 'FAIL' : report.journeys.some(item => item.verdict === 'UNVERIFIED') ? 'UNVERIFIED' : 'PASS_LIFECYCLE_ONLY';
    if (descriptor === undefined) { mkdirSync(dirname(output), { recursive: true }); descriptor = openSync(output, 'wx'); }
    save(); closeSync(descriptor);
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [output, ...flags] = process.argv.slice(2);
  if (!output || output.startsWith('--') || flags.some(flag => flag !== '--test-build') || flags.length > 1) throw new Error('Usage: node tools/m10-journeys.mjs output.json [--test-build]');
  const report = await runJourneys(output, flags.includes('--test-build'));
  console.log(JSON.stringify({ output: resolve(output), verdict: report.verdict, journeys: report.journeys.length, fatal: report.fatal ?? null }));
  if (report.verdict !== 'PASS_LIFECYCLE_ONLY') process.exitCode = 1;
}