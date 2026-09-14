import assert from 'node:assert/strict';
import { existsSync, mkdirSync, openSync, writeSync, ftruncateSync, closeSync, readFileSync } from 'node:fs';
import { resolve, dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { hostname, platform, release, arch, cpus } from 'node:os';
import { execFileSync } from 'node:child_process';
import { ROOT, sha256, startFixtures } from './m10-fixtures.mjs';
import { bounded, openExtension, OperationTimeout, snapshotExtensionStatus, until, verifyBuild, Unverified } from './m10-browser.mjs';

export async function withJourneyWatchdog(body, shutdown, milliseconds = 60000) {
  const controller = new AbortController();
  const task = Promise.resolve().then(() => body(controller.signal));
  try { return await bounded(task, milliseconds, 'Journey watchdog'); }
  catch (error) {
    if (error instanceof OperationTimeout) {
      controller.abort(error);
      try { await shutdown(); }
      finally { await bounded(task.catch(() => {}), 10000, 'Drain cancelled journey after browser shutdown'); }
    }
    throw error;
  }
}
export function assertPageErrors(errors, fixture, observedCodes) {
  const taintedVideo = "SecurityError: Failed to execute 'importExternalTexture' on 'GPUDevice': Video element is tainted by cross-origin data and may not be loaded.";
  const taintedCopy = "SecurityError: Failed to execute 'copyExternalImageToTexture' on 'GPUQueue': Video element is tainted by cross-origin data and may not be loaded.";
  const classified = errors.map(error => ({ error, expected: fixture === 'nocors' && observedCodes.has('cors-blocked') && [taintedVideo, taintedCopy].includes(error) }));
  assert.deepEqual(classified.filter(item => !item.expected), [], 'Unexpected page errors');
  return classified;
}
export async function pageOwnedAction(page, name, optional = false, emit = () => {}) {
  const started = Date.now();
  await bounded(page.evaluate(() => {
    const key = Symbol.for('aethervsr.m10.actions');
    if (globalThis[key]) return;
    const events = globalThis[key] = [];
    const record = event => {
      events.push({ type: event.type, at: performance.now(), trusted: event.isTrusted,
        action: event.target.closest?.('[data-action]')?.dataset.action ?? null,
        activation: navigator.userActivation.isActive, focused: document.hasFocus(), visibility: document.visibilityState,
        fullscreen: document.fullscreenElement?.tagName ?? null, pip: !!document.pictureInPictureElement,
        pipWindow: event.pictureInPictureWindow ? { width: event.pictureInPictureWindow.width, height: event.pictureInPictureWindow.height } : null });
      if (events.length > 100) events.shift();
    };
    for (const type of ['click', 'fullscreenchange', 'fullscreenerror', 'enterpictureinpicture', 'leavepictureinpicture', 'visibilitychange']) document.addEventListener(type, record, true);
    for (const type of ['focus', 'blur']) window.addEventListener(type, record);
  }), 3000, 'Install page action diagnostics');
  let failure;
  try {
    await page.locator(`[data-action="${name}"]`).first().click({ timeout: 5000 });
    await page.waitForFunction(name => {
      const result = document.querySelector('#result');
      return result.dataset.action === name && ['done', 'error'].includes(result.dataset.state);
    }, name, { timeout: 5000 });
  } catch (error) { failure = error; }
  const value = await bounded(page.evaluate(() => {
    const element = document.querySelector('#result');
    return { state: element.dataset.state, action: element.dataset.action, message: element.textContent,
      events: globalThis[Symbol.for('aethervsr.m10.actions')], fullscreen: document.fullscreenElement?.tagName ?? null,
      pip: !!document.pictureInPictureElement, focused: document.hasFocus(), visibility: document.visibilityState };
  }), 3000, 'Read page action diagnostics');
  emit('page-owned-action', { ...value, elapsedMs: Date.now() - started,
    timingScope: 'Wall clock from diagnostic setup through trusted click and fixture result; not GPU timing', error: failure ? String(failure) : null });
  if (failure) throw failure;
  assert.equal(value.action, name);
  if (value.state === 'error') { if (optional) throw new Unverified(value.message); throw new Error(value.message); }
  if (['fullscreen', 'directfs'].includes(name) && value.fullscreen === null) throw new Unverified('Fullscreen request fulfilled but fullscreen is no longer active; inspect page-owned-action events');
  return value;
}
function pageSnapshot(extensionId) {
  const rect = element => { const bounds = element.getBoundingClientRect(); return { left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height }; };
  const main = document.querySelector('main').cloneNode(true); main.querySelectorAll('canvas').forEach(canvas => canvas.remove());
  return { href: location.href, visibility: document.visibilityState, focused: document.hasFocus(), originalHTML: main.innerHTML,
    mainTestHook: typeof globalThis.__AETHERVSR_EXTENSION_TEST__, mainSingleton: typeof globalThis[Symbol.for(`aethervsr.m10.document.${extensionId}`)],
    fullscreen: document.fullscreenElement?.tagName ?? null, pip: !!document.pictureInPictureElement,
    actionEvents: globalThis[Symbol.for('aethervsr.m10.actions')] ?? [],
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
  const report = { schemaVersion: 1, started: new Date().toISOString(), verdict: 'UNVERIFIED', completion: 'RUNNING', testBuild, performance: 'not measured',
    scope: 'Native unpacked-extension lifecycle assertions. Popup focus interrupts visibility; status timing counters are diagnostic snapshots, not benchmarks. No pixel readback or parity claim.',
    manual: { visual: 'UNVERIFIED: review local screenshots for object-fit, clipping, radii and caption stacking', encryptedStream: 'UNVERIFIED: ClearKey test attaches MediaKeys to clear media only', displayRefreshRate: 'not measured' },
    machine: { hostname: hostname(), os: platform(), release: release(), arch: arch(), cpu: cpus()[0]?.model ?? 'not measured',
      osVersion: platform() === 'darwin' ? execFileSync('sw_vers', ['-productVersion'], { encoding: 'utf8', timeout: 3000 }).trim() : release() }, journeys: [], events: [] };
  let descriptor, native, fixtures;
  const save = () => { if (descriptor === undefined) return; const bytes = Buffer.from(`${JSON.stringify(report)}\n`); writeSync(descriptor, bytes, 0, bytes.length, 0); ftruncateSync(descriptor, bytes.length); };
  const emit = (name, data) => { report.events.push({ name, at: new Date().toISOString(), data }); save(); };
  const cleanup = async () => { try { await native?.close(); } finally { await fixtures?.close(); } };
  const abort = async signal => {
    report.fatal = `Interrupted: ${signal}`; report.verdict = 'UNVERIFIED'; report.completion = 'INTERRUPTED';
    for (const item of report.journeys.filter(item => item.verdict === 'RUNNING')) { item.verdict = 'UNVERIFIED'; item.error = report.fatal; }
    save(); const timer = setTimeout(() => process.exit(2), 10000);
    try { await cleanup(); } finally { clearTimeout(timer); process.exit(2); }
  };
  const interrupted = () => { void abort('signal'); };
  try {
    mkdirSync(dirname(output), { recursive: true }); descriptor = openSync(output, 'wx'); save();
    process.once('SIGINT', interrupted); process.once('SIGTERM', interrupted);
    report.build = verifyBuild(testBuild, relative(ROOT, output).startsWith('..') ? undefined : output);
    report.harness = Object.fromEntries(['tools/m10-journeys.mjs', 'tools/m10-browser.mjs', 'tools/m10-fixtures.mjs', 'tools/m10-fixtures/index.html', 'tools/m9-browser.mjs']
      .map(name => [name, sha256(readFileSync(join(ROOT, name)))]));
    fixtures = await startFixtures(); report.media = fixtures.evidence;
    native = await openExtension(report.build, emit); native.context.setDefaultTimeout(10000); native.context.setDefaultNavigationTimeout(10000);
    const screenshots = join(ROOT, '.cache/m10/screenshots', `${Date.now()}-${sha256(output).slice(0, 8)}`); mkdirSync(screenshots, { recursive: true });
    let journeySignal;
    let observedCodes;
    const dom = page => bounded(page.evaluate(pageSnapshot, native.extensionId), 3000, 'Page snapshot');
    const ready = page => page.waitForFunction(() => [...document.querySelectorAll('video')].every(video => video.readyState >= 2 && video.videoWidth > 0));
    const popup = async (page, fn) => { const panel = await native.popup(page); try { return await fn(panel); } finally { await panel.dismiss(); } };
    const enable = page => popup(page, async panel => { const status = await panel.click('#enable'); assert(status.enabled, status.message); return panel.tabId; });
    const status = async (id, accept = value => value.code === 'active') => {
      const value = await until(() => native.inspect(id), accept, 10000, journeySignal); observedCodes.add(value.code); resourceCheck(value); emit('status', value); return value;
    };
    const disable = async (page, original) => {
      const value = await popup(page, panel => panel.click('#disable')); resourceCheck(value, true);
      const restored = await dom(page); assert.equal(restored.canvases.length, 0); assert.equal(restored.originalHTML, original.originalHTML, 'Page DOM/styles changed');
      emit('disabled-restored', { status: value, dom: restored }); return value;
    };
    const action = (page, name, optional = false) => pageOwnedAction(page, name, optional, emit);
    const screenshot = async (page, name) => { const path = join(screenshots, `${name.replaceAll('/', '-')}.png`); await page.screenshot({ path, timeout: 3000 }); emit('screenshot', { path: relative(ROOT, path), sha256: sha256(readFileSync(path)), verdict: 'UNVERIFIED manual pixels' }); };
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
      let page;
      const errors = []; observedCodes = new Set();
      try {
        await withJourneyWatchdog(async signal => {
          journeySignal = signal;
          page = await bounded(native.context.newPage(), 5000, 'Create journey page'); signal.throwIfAborted();
          page.on('pageerror', error => { errors.push(String(error)); emit('page-error', { name, error: String(error) }); });
          await page.goto(`${fixtures.url}?case=${encodeURIComponent(fixture)}&journey=${++sequence}`); await bounded(page.bringToFront(), 3000, 'Focus journey'); await ready(page);
          signal.throwIfAborted();
          const original = await dom(page); assert.equal(original.canvases.length, 0); emit('journey-before', { name, dom: original });
          await body(page, original); signal.throwIfAborted();
        }, async () => {
          report.aborted = `Timed-out journey: ${name}; remaining journeys not run`;
          item.verdict = 'UNVERIFIED'; emit('journey-shutdown', { name, reason: report.aborted });
          await native.close();
        });
        item.verdict = 'PASS';
      } catch (error) {
        item.verdict = error instanceof Unverified ? 'UNVERIFIED' : 'FAIL'; item.error = String(error);
        if (page && !page.isClosed()) try { emit('failure-dom', { name, dom: await dom(page) }); await screenshot(page, `${sequence}-failure`); } catch (captureError) { item.captureError = String(captureError); }
      } finally {
        try { if (page) await bounded(page.close(), 3000, 'Close journey page'); }
        catch (error) { item.cleanupError = String(error); item.verdict = 'FAIL'; report.aborted = `Journey page cleanup failed: ${name}`; await native.close(); }
        item.pageErrors = errors;
        try { item.pageErrors = assertPageErrors(errors, fixture, observedCodes); }
        catch (error) { item.verdict = 'FAIL'; item.error = [item.error, String(error)].filter(Boolean).join('\n'); }
        item.finished = new Date().toISOString(); save();
      }
      console.log(`${item.verdict} ${name}`);
      if (report.aborted) throw new Unverified(report.aborted);
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
      try { await page.waitForFunction(kind => kind === 'pip' ? !!document.pictureInPictureElement : !!document.fullscreenElement, kind, { timeout: 3000 }); }
      catch (error) { throw new Unverified(`Native ${kind} did not remain active: ${error}`); }
      if (kind !== 'pip') {
        try { await native.captureFullscreenWindow(page, join(screenshots, `${sequence}-${kind}-native-window.png`)); }
        catch (error) {
          report.manual.nativeFullscreenCapture = `UNVERIFIED: native OS capture unavailable; DOM/fullscreen assertions and browser raster are separate evidence. ${String(error)}`;
          emit('native-capture-unavailable', { kind, reason: String(error) });
        }
      }
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
      await page.goBack({ waitUntil: 'commit' }); await ready(page); const restored = await dom(page); assert.equal(restored.canvases.length, 0);
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
  } catch (error) { report.fatal = String(error); report.fatalVerdict = error instanceof Unverified ? 'UNVERIFIED' : 'FAIL'; }
  finally {
    process.off('SIGINT', interrupted); process.off('SIGTERM', interrupted);
    try { await cleanup(); } catch (error) { report.cleanupError = String(error); }
    report.requests = fixtures?.requests ?? []; report.finished = new Date().toISOString(); report.completion = report.fatal ? 'INTERRUPTED' : 'FINISHED';
    report.verdict = report.fatalVerdict === 'FAIL' || report.cleanupError || report.journeys.some(item => item.verdict === 'FAIL') ? 'FAIL' : report.fatal || report.journeys.some(item => item.verdict === 'UNVERIFIED') ? 'UNVERIFIED' : 'PASS_LIFECYCLE_ONLY';
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