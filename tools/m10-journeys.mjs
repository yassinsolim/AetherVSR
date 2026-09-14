import assert from 'node:assert/strict';
import { existsSync, mkdirSync, openSync, writeSync, ftruncateSync, closeSync, readFileSync } from 'node:fs';
import { resolve, dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { hostname, platform, release, arch, cpus } from 'node:os';
import { execFileSync } from 'node:child_process';
import { CASES, ROOT, sha256, startFixtures } from './m10-fixtures.mjs';
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
export async function pageOwnedAction(page, name, optional = false, emit = () => {}, index = 0, resultTimeout = 5000) {
  const started = Date.now();
  await bounded(page.evaluate(() => {
    const key = Symbol.for('aethervsr.m10.actions');
    if (globalThis[key]) return;
    const events = globalThis[key] = [];
    const record = event => {
      events.push({ type: event.type, at: performance.now(), trusted: event.isTrusted,
        action: event.composedPath().find(node => node.dataset?.action)?.dataset.action ?? null,
        activation: navigator.userActivation.isActive, focused: document.hasFocus(), visibility: document.visibilityState,
        fullscreen: document.fullscreenElement?.tagName ?? null, pip: !!document.pictureInPictureElement,
        pipWindow: event.pictureInPictureWindow ? { width: event.pictureInPictureWindow.width, height: event.pictureInPictureWindow.height } : null });
      if (events.length > 100) events.shift();
    };
    for (const type of ['click', 'fullscreenchange', 'fullscreenerror', 'enterpictureinpicture', 'leavepictureinpicture', 'visibilitychange']) document.addEventListener(type, record, true);
    for (const type of ['focus', 'blur']) window.addEventListener(type, record);
  }), 3000, 'Install page action diagnostics');
  const previousClicks = await bounded(page.evaluate(() => globalThis[Symbol.for('aethervsr.m10.actions')].filter(event => event.type === 'click').at(-1)?.at ?? -1), 3000, 'Previous page click');
  let failure;
  try {
    await page.locator(`[data-action="${name}"]`).nth(index).click({ timeout: 5000 });
    await page.waitForFunction(name => {
      const result = document.querySelector('#result');
      return result.dataset.action === name && ['done', 'error'].includes(result.dataset.state);
    }, name, { timeout: resultTimeout });
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
  assert(value.events.some(event => event.type === 'click' && event.action === name && event.trusted && event.at > previousClicks), 'Missing trusted page-owned click');
  if (value.state === 'error') { if (optional) throw new Unverified(value.message); throw new Error(value.message); }
  if (['fullscreen', 'directfs'].includes(name) && value.fullscreen === null) throw new Unverified('Fullscreen request fulfilled but fullscreen is no longer active; inspect page-owned-action events');
  return value;
}
export function pageSnapshot(extensionId) {
  const rect = element => { const bounds = element.getBoundingClientRect(); return { left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height }; };
  const elements = [];
  const visit = root => { for (const element of root.querySelectorAll('*')) { elements.push(element); if (element.shadowRoot) visit(element.shadowRoot); } };
  visit(document);
  const clone = node => {
    if (node.nodeType === Node.ELEMENT_NODE && node.matches('canvas[data-aethervsr-m10]:not([data-m10-page-owned])')) return null;
    const copy = node.cloneNode(false);
    for (const child of node.childNodes) { const nested = clone(child); if (nested) copy.appendChild(nested); }
    if (node.shadowRoot) {
      const shadow = document.createElement('template'); shadow.setAttribute('data-m10-shadow-snapshot', 'open');
      for (const child of node.shadowRoot.childNodes) { const nested = clone(child); if (nested) shadow.content.appendChild(nested); }
      copy.appendChild(shadow);
    }
    return copy;
  };
  const main = clone(document.querySelector('main'));
  return { href: location.href, visibility: document.visibilityState, focused: document.hasFocus(), originalHTML: main.innerHTML,
    mainTestHook: typeof globalThis.__AETHERVSR_EXTENSION_TEST__, mainSingleton: typeof globalThis[Symbol.for(`aethervsr.m10.document.${extensionId}`)],
    fullscreen: document.fullscreenElement?.tagName ?? null, pip: !!document.pictureInPictureElement,
    actionEvents: globalThis[Symbol.for('aethervsr.m10.actions')] ?? [],
    persisted: document.documentElement.dataset.pageshowPersisted ?? null,
    videos: elements.filter(element => element.tagName === 'VIDEO').map(video => ({ src: video.getAttribute('src'), currentSrc: video.currentSrc, crossorigin: video.getAttribute('crossorigin'),
      fixtureId: video.dataset.fixtureVideo, shadow: video.getRootNode() !== document,
      paused: video.paused, controls: video.controls, mediaKeys: !!video.mediaKeys, width: video.videoWidth, height: video.videoHeight, readyState: video.readyState,
      currentTime: video.currentTime, decodedFrames: video.getVideoPlaybackQuality().totalVideoFrames,
      visible: video.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }), rect: rect(video) })),
    canvases: elements.filter(element => element.tagName === 'CANVAS').map(canvas => { const style = getComputedStyle(canvas); return { uuid: canvas.dataset.aethervsrM10 ?? null,
      pageOwned: canvas.hasAttribute('data-m10-page-owned'), shadow: canvas.getRootNode() !== document,
      videoId: canvas.previousElementSibling?.dataset.fixtureVideo ?? null, videoSource: canvas.previousElementSibling?.currentSrc ?? null,
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
function detachedCheck(value, expectedCode) {
  resourceCheck(value);
  if (expectedCode) assert.equal(value.code, expectedCode);
  assert.equal(value.owner, null); assert.equal(value.current, null); assert.equal(value.details.attachment, null);
  assert(value.details.lastTeardown, 'Missing teardown evidence');
  for (const name of ['device', 'pipeline', 'canvas', 'resizeObservers', 'listeners', 'geometryFrame', 'frameCallback']) assert.equal(value.details.lastTeardown[name], 0, `Missing or live teardown resource: ${name}`);
  for (const [name, count] of Object.entries(value.details.lastTeardown)) assert.equal(count, 0, `Live ${name} after teardown`);
}

export async function runJourneys(output, testBuild = false, only = null) {
  output = resolve(output); assert(!existsSync(output), 'Never overwrite raw evidence');
  const report = { schemaVersion: 1, started: new Date().toISOString(), verdict: 'UNVERIFIED', completion: 'RUNNING', testBuild, only, performance: 'not measured',
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
    const ready = page => until(() => dom(page), view => view.videos.every(video => video.readyState >= 2 && video.width > 0), 10000, journeySignal);
    const popup = async (page, fn) => { const panel = await native.popup(page); try { return await fn(panel); } finally { await panel.dismiss(); } };
    const enable = page => popup(page, async panel => { const status = await panel.click('#enable'); assert(status.enabled, status.message); return panel.tabId; });
    const status = async (id, accept = value => value.code === 'active') => {
      const value = await until(() => native.inspect(id), accept, 10000, journeySignal); observedCodes.add(value.code); resourceCheck(value); emit('status', value); return value;
    };
    const disable = async (page, original) => {
      const value = await popup(page, panel => panel.click('#disable')); resourceCheck(value, true);
      const restored = await dom(page); assert.equal(restored.canvases.filter(canvas => !canvas.pageOwned).length, 0); assert.equal(restored.originalHTML, original.originalHTML, 'Page DOM/styles changed');
      emit('disabled-restored', { status: value, dom: restored }); return value;
    };
    const action = (page, name, optional = false, index = 0) => pageOwnedAction(page, name, optional, emit, index);
    const screenshot = async (page, name) => { const path = join(screenshots, `${name.replaceAll('/', '-')}.png`); await page.screenshot({ path, timeout: 3000 }); emit('screenshot', { path: relative(ROOT, path), sha256: sha256(readFileSync(path)), verdict: 'UNVERIFIED manual pixels' }); };
    const isolatedStatus = async page => snapshotExtensionStatus({ ok: true, status: await native.isolated(page, () => globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)].status()) });
    const active = async (page, id) => {
      const value = await status(id, state => state.code === 'active' && state.details?.attachment?.ready);
      const view = await until(() => dom(page), state => state.canvases.some(canvas => !canvas.pageOwned && canvas.visibility === 'visible'), 5000, journeySignal);
      assert.equal(view.visibility, 'visible'); assert.equal(view.focused, true); assert.equal(view.mainTestHook, 'undefined'); assert.equal(view.mainSingleton, 'undefined');
      const owned = view.canvases.filter(canvas => !canvas.pageOwned); assert.equal(owned.length, 1); const canvas = owned[0];
      assert.match(canvas.uuid, /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i); assert(canvas.siblingOfVideo);
      assert.equal(canvas.pointerEvents, 'none'); assert.equal(canvas.visibility, 'visible'); assert(canvas.width > 0 && canvas.height > 0);
      const expected = value.details.attachment.cssRect;
      for (const key of ['left', 'top', 'width', 'height']) { assert(Math.abs(canvas.rect[key] - expected[key]) < 1, key); assert(Math.abs(canvas.rect[key] - canvas.videoRect[key]) < 1, `Original video ${key}`); }
      emit('active-dom', view); return value;
    };
    const stable = async (id, check, milliseconds = 1700) => {
      const started = performance.now(); const samples = [];
      do {
        journeySignal.throwIfAborted();
        const value = await native.inspect(id); observedCodes.add(value.code); resourceCheck(value); await check(value);
        samples.push({ elapsedMs: performance.now() - started, code: value.code, owner: value.owner, infrastructure: value.details.infrastructure });
        await new Promise(done => setTimeout(done, 100));
      } while (performance.now() - started < milliseconds);
      const value = await native.inspect(id); resourceCheck(value); await check(value);
      samples.push({ elapsedMs: performance.now() - started, code: value.code, owner: value.owner, infrastructure: value.details.infrastructure });
      emit('stable-observation', { tabId: id, requestedMs: milliseconds, observedMs: performance.now() - started, samples,
        scope: 'Sampled lifecycle stability from first read through final read; 100 ms pauses plus inspection latency, not continuous tracing or a performance benchmark' });
      return value;
    };
    const originalPlaying = async (page, source) => {
      const before = await dom(page); assert.equal(before.videos.length, 1);
      const view = await until(() => dom(page), view => view.videos[0]?.decodedFrames > before.videos[0].decodedFrames, 5000, journeySignal);
      const video = view.videos[0]; assert.equal(video.paused, false); assert.equal(video.visible, true);
      assert(video.rect.width > 0 && video.rect.height > 0); assert.equal(video.currentSrc, source);
      emit('original-playing-visible', { before: before.videos[0], after: video }); return view;
    };
    const controlsReachable = async page => {
      const controls = await page.locator('main [data-action="pause"]').evaluateAll(buttons => buttons.map(button => {
        const bounds = button.getBoundingClientRect(); const root = button.getRootNode();
        return { hit: root.elementFromPoint(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2) === button,
          visible: button.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }), width: bounds.width, height: bounds.height };
      }));
      assert(controls.length > 0); assert(controls.every(control => control.hit && control.visible && control.width > 0 && control.height > 0), 'Page controls hidden or covered');
      emit('page-controls-reachable', controls);
    };
    let sequence = 0;
    const journey = async (name, fixture, body) => {
      if (only !== null && name !== only) return;
      assert(CASES.includes(fixture), `Missing root fixture: ${fixture}`);
      const item = { name, fixture, verdict: 'RUNNING', started: new Date().toISOString() }; report.journeys.push(item); save();
      let page;
      const errors = []; observedCodes = new Set();
      const pages = new Set();
      const observePage = created => { pages.add(created); created.on('pageerror', error => { errors.push(String(error)); emit('page-error', { name, url: created.url(), error: String(error) }); }); };
      native.context.on('page', observePage);
      try {
        await withJourneyWatchdog(async signal => {
          journeySignal = signal;
          page = await bounded(native.context.newPage(), 5000, 'Create journey page'); signal.throwIfAborted();
          await page.goto(`${fixtures.url}?case=${encodeURIComponent(fixture)}&journey=${++sequence}`); await bounded(page.bringToFront(), 3000, 'Focus journey');
          if (fixture !== 'open-shadow-late') await ready(page);
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
        for (const [index, failed] of [...pages].entries()) if (!failed.isClosed()) try { emit('failure-dom', { name, url: failed.url(), dom: await dom(failed) }); await screenshot(failed, `${sequence}-failure-${index}`); } catch (captureError) { item.captureError = String(captureError); }
      } finally {
        native.context.off('page', observePage);
        try { for (const created of pages) if (!created.isClosed()) await bounded(created.close(), 3000, 'Close journey page'); }
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
    for (const fixture of ['custom', 'cors', 'contain', 'cover', 'clipped', 'radius', 'translated', 'mse', 'controls/captions', 'caption-after-auto-passive', 'caption-before-positive-passive',
      'rounded-ancestor', 'size-container', 'positioned-ancestor']) {
      await journey(`supported ${fixture}`, fixture, async (page, original) => {
        const id = await enable(page); await active(page, id);
        if (fixture === 'cors') assert.equal((await dom(page)).videos[0].crossorigin, 'anonymous');
        if (['contain', 'cover'].includes(fixture)) assert.equal((await dom(page)).canvases[0].objectFit, fixture);
        if (['controls/captions', 'rounded-ancestor', 'size-container'].includes(fixture) || fixture.startsWith('caption-')) {
          await controlsReachable(page);
          const caption = await page.locator('.caption').evaluate(element => ({ visible: element.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }), pointerEvents: getComputedStyle(element).pointerEvents, zIndex: getComputedStyle(element).zIndex }));
          assert.equal(caption.visible, true); assert.equal(caption.pointerEvents, 'none'); emit('passive-caption', { fixture, ...caption });
          const pointer = await page.locator('[data-action="pause"]').evaluate(button => { const rect = button.getBoundingClientRect(); return document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2)?.closest('[data-action]')?.dataset.action; });
          assert.equal(pointer, 'pause'); await action(page, 'pause'); assert((await dom(page)).videos[0].paused);
          await action(page, 'play'); assert(!(await dom(page)).videos[0].paused);
        }
        if (['rounded-ancestor', 'size-container'].includes(fixture)) {
          const view = await dom(page); assert.equal(view.canvases[0].radius, fixture === 'rounded-ancestor' ? '12px' : '24px');
          await page.evaluate(() => { document.body.style.minHeight = '1600px'; scrollTo(0, 80); });
          await active(page, id); await controlsReachable(page); await screenshot(page, `${sequence}-${fixture}-scroll`);
          await page.evaluate(() => { scrollTo(0, 0); document.body.style.minHeight = ''; });
          await action(page, 'fullscreen');
          assert(await page.locator('.controls').evaluate(element => document.fullscreenElement.contains(element)));
          assert(await page.locator('.caption').evaluate(element => document.fullscreenElement.contains(element)));
          await active(page, id); await controlsReachable(page); await screenshot(page, `${sequence}-${fixture}-fullscreen`);
          await page.evaluate(() => document.exitFullscreen()); await active(page, id);
        }
        await screenshot(page, `${sequence}-${fixture}-white`); await page.evaluate(() => { document.body.style.background = 'black'; }); await screenshot(page, `${sequence}-${fixture}-black`);
        await disable(page, original);
      });
    }
    for (const [fixture, expected] of [['native', 'unsupported-controls'], ['nocors', 'cors-blocked'], ['iframe', 'unsupported-frame'], ['sameiframe', 'unsupported-frame'], ['caption-before-auto-passive', 'unsupported-controls'], ['rounded-offset', 'unsupported-geometry']]) {
      await journey(`explicit unsupported ${fixture}`, fixture, async (page, original) => {
        const id = await enable(page); const value = await status(id, item => item.code === expected); assert.equal(value.owner, null); assert.equal((await dom(page)).canvases.length, 0);
        if (fixture.includes('iframe')) {
          assert(value.embeddedFrames > 0); assert.equal(value.candidates, 0);
          const frames = await until(() => page.frames().filter(frame => frame !== page.mainFrame() && frame.url().includes('/fixture?case=custom')), frames => frames.length === 1);
          for (const frame of frames) { await frame.waitForSelector('main'); await ready(frame); const view = await frame.evaluate(pageSnapshot, native.extensionId); assert.equal(view.canvases.length, 0); assert.equal(view.mainTestHook, 'undefined'); emit('uninspected-frame-dom', view); }
        }
        if (fixture === 'nocors') assert.equal((await dom(page)).videos[0].crossorigin, null);
        if (fixture === 'caption-before-auto-passive') {
          await controlsReachable(page); await action(page, 'pause'); assert.equal((await dom(page)).videos[0].paused, true);
          await action(page, 'play'); await originalPlaying(page, original.videos[0].currentSrc);
          await stable(id, async state => { assert.equal(state.code, expected); assert.equal(state.owner, null); assert.equal(state.details.infrastructure.created, value.details.infrastructure.created); assert.equal((await dom(page)).canvases.length, 0); });
        }
        await screenshot(page, `${sequence}-unsupported-${fixture}`);
        await disable(page, original);
      });
    }
    for (const kind of ['style', 'z-index', 'passive-caption']) await journey(`geometry guard ${kind} mutation`, 'caption-after-auto-passive', async (page, original) => {
      const expectedCode = kind === 'passive-caption' ? 'unsupported-controls' : 'unsupported-geometry';
      const mutate = ({ kind, preview }) => {
        const main = preview ? document.querySelector('main').cloneNode(true) : document.querySelector('main');
        const video = main.querySelector('video');
        if (kind === 'style') video.style.transform = 'rotate(8deg)';
        else if (kind === 'z-index') video.style.zIndex = '1';
        else video.before(main.querySelector('.caption'));
        return main.innerHTML;
      };
      const expected = { ...original, originalHTML: await page.evaluate(mutate, { kind, preview: true }) };
      const id = await enable(page); const before = await active(page, id); await controlsReachable(page);
      await page.evaluate(mutate, { kind, preview: false });
      const rejected = await status(id, value => value.code === expectedCode && value.owner === null);
      detachedCheck(rejected, expectedCode);
      assert.equal(rejected.details.infrastructure.created, before.details.infrastructure.created);
      await stable(id, async value => { detachedCheck(value, expectedCode); assert.equal(value.details.infrastructure.created, before.details.infrastructure.created); assert.equal((await dom(page)).canvases.length, 0); });
      await controlsReachable(page); await action(page, 'pause'); assert.equal((await dom(page)).videos[0].paused, true);
      await action(page, 'play'); await originalPlaying(page, original.videos[0].currentSrc);
      await screenshot(page, `${sequence}-guard-${kind}`); await disable(page, expected);
    });
    await journey('scroll, offscreen suspension, and native resize', 'custom', async (page, original) => {
      const id = await enable(page); const before = await active(page, id); await screenshot(page, `${sequence}-normal`);
      await page.evaluate(() => { document.body.style.minHeight = '2400px'; });
      await page.mouse.move(900, 500); await page.mouse.wheel(0, 120);
      await page.waitForFunction(() => scrollY > 0 && scrollY < 200);
      await active(page, id); await screenshot(page, `${sequence}-scroll`);
      await page.mouse.wheel(0, 1000);
      const hidden = await status(id, value => value.code === 'suspended' && value.details.attachment?.suspendedReason === 'offscreen');
      assert.equal(hidden.owner, before.owner); assert.equal((await dom(page)).canvases[0].visibility, 'hidden');
      assert.equal(hidden.details.attachment.active, false); assert.equal(hidden.details.attachment.resources.frameCallback, 0);
      await page.mouse.wheel(0, -2400); await page.waitForFunction(() => scrollY === 0); await active(page, id);
      const session = await bounded(native.context.newCDPSession(page), 3000, 'Native resize session'); let window;
      try {
        window = await bounded(session.send('Browser.getWindowForTarget'), 3000, 'Read native window');
        assert.equal(window.bounds.windowState, 'normal');
        const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }));
        const bounds = { width: Math.max(900, window.bounds.width - 160), height: Math.max(650, window.bounds.height - 80) };
        await bounded(session.send('Browser.setWindowBounds', { windowId: window.windowId, bounds }), 3000, 'Resize native Chrome window');
        await page.waitForFunction(viewport => innerWidth !== viewport.width || innerHeight !== viewport.height, viewport);
        const resized = await active(page, id); assert.equal(resized.owner, before.owner);
        assert.equal(resized.details.infrastructure.created, before.details.infrastructure.created);
        emit('native-resize', { before: window, requested: bounds, after: await bounded(session.send('Browser.getWindowForTarget'), 3000, 'Read resized window') });
        await controlsReachable(page); await screenshot(page, `${sequence}-native-resize`);
      } finally {
        try { if (window) await bounded(session.send('Browser.setWindowBounds', { windowId: window.windowId, bounds: window.bounds }), 3000, 'Restore native window'); }
        finally { await bounded(session.detach(), 1000, 'Detach native resize session'); }
      }
      await active(page, id); await disable(page, original);
    });
    await journey('multiple video single owner', 'multiple', async (page, original) => { const id = await enable(page); const value = await active(page, id); assert.equal(value.candidates, 2); assert.equal(value.details.infrastructure.maximumConcurrent, 1); await disable(page, original); });
    await journey('same-element removal and reinsertion', 'reinsert', async (page, original) => {
      const id = await enable(page); const before = await active(page, id); const source = original.videos[0].currentSrc;
      const oldUuid = (await dom(page)).canvases[0].uuid;
      await page.evaluate(() => { const video = document.querySelector('video'); globalThis[Symbol.for('aethervsr.m10.identity')] = { video, parent: video.parentElement, canvas: video.nextElementSibling }; });
      await action(page, 'remove');
      const removed = await status(id, value => value.code === 'no-video' && value.owner === null); detachedCheck(removed, 'no-video');
      assert.equal(removed.details.infrastructure.destroyed, before.details.infrastructure.destroyed + 1);
      await stable(id, async value => {
        detachedCheck(value, 'no-video'); assert.equal(value.details.infrastructure.created, before.details.infrastructure.created);
        const view = await dom(page); assert.equal(view.videos.length, 0); assert.equal(view.canvases.length, 0);
        assert(await page.evaluate(() => { const saved = globalThis[Symbol.for('aethervsr.m10.identity')]; return !saved.video.isConnected && !saved.canvas.isConnected; }));
      });
      await action(page, 'reinsert'); await ready(page);
      assert(await page.evaluate(() => { const saved = globalThis[Symbol.for('aethervsr.m10.identity')]; return saved.video === document.querySelector('video') && saved.parent === saved.video.parentElement; }));
      if ((await dom(page)).videos[0].paused) await action(page, 'play');
      const restored = await active(page, id); assert.equal(restored.owner, before.owner, 'Same original must keep its identity ID');
      assert.equal(restored.details.infrastructure.created, before.details.infrastructure.created + 1);
      assert.notEqual((await dom(page)).canvases[0].uuid, oldUuid, 'Reinsertion must use a fresh output');
      await stable(id, value => { assert.equal(value.owner, before.owner); assert.equal(value.details.infrastructure.created, restored.details.infrastructure.created); assert.equal(value.details.infrastructure.destroyed, removed.details.infrastructure.destroyed); });
      await originalPlaying(page, source); await screenshot(page, `${sequence}-reinsert`); await disable(page, original);
    });
    await journey('dynamic playback and size owner hysteresis', 'multiple', async page => {
      const dwell = async (owner, triggeredAt) => {
        const transition = await native.isolated(page, owner => globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)].transitions.filter(item => item.owner === owner && item.reason === 'attached').at(-1), owner);
        assert(transition, 'Missing actual owner-attachment transition'); assert(transition.at - triggeredAt >= 750, 'Owner switched before challenger hysteresis');
        emit('owner-hysteresis', { triggeredAt, transition, elapsedMs: transition.at - triggeredAt, scope: 'Same-document monotonic time from trusted playback click or page size mutation to actual attachment transition; includes discovery scheduling, not GPU timing' });
      };
      await page.evaluate(() => {
        const main = document.querySelector('main'); main.style.cssText = 'display:flex;gap:16px;align-items:flex-start';
        for (const [index, video] of [...main.querySelectorAll('video')].entries()) {
          const size = index === 0 ? 'width:320px;height:180px' : 'width:240px;height:135px';
          video.style.cssText = size; video.parentElement.style.cssText = size;
        }
      });
      const original = await dom(page);
      const expected = { ...original, originalHTML: await page.evaluate(() => {
        const main = document.querySelector('main').cloneNode(true); const video = main.querySelectorAll('video')[1];
        video.style.cssText = 'width:480px;height:270px'; video.parentElement.style.cssText = video.style.cssText; return main.innerHTML;
      }) };
      const id = await enable(page); const first = await active(page, id);
      assert.equal(first.candidates, 2); assert.equal((await dom(page)).canvases[0].videoId, '0');
      const paused = await action(page, 'pause', false, 0);
      const second = await status(id, value => value.code === 'active' && value.owner !== first.owner);
      await dwell(second.owner, paused.events.filter(event => event.type === 'click' && event.action === 'pause').at(-1).at);
      assert.equal(second.details.infrastructure.created, first.details.infrastructure.created + 1);
      assert.equal(second.details.infrastructure.destroyed, first.details.infrastructure.destroyed + 1);
      assert.equal((await dom(page)).canvases[0].videoId, '1'); await active(page, id);
      await screenshot(page, `${sequence}-owner-playback-switch`);
      await action(page, 'play', false, 0);
      const returned = await status(id, value => value.code === 'active' && value.owner === first.owner);
      assert.equal(returned.details.infrastructure.created, second.details.infrastructure.created + 1);
      const resizedAt = await page.evaluate(() => { const at = performance.now(); const video = document.querySelectorAll('video')[1]; video.style.cssText = 'width:480px;height:270px'; video.parentElement.style.cssText = video.style.cssText; return at; });
      const resized = await status(id, value => value.code === 'active' && value.owner === second.owner);
      await dwell(resized.owner, resizedAt);
      assert.equal(resized.details.infrastructure.created, returned.details.infrastructure.created + 1);
      await stable(id, value => { assert.equal(value.owner, second.owner); assert.equal(value.details.infrastructure.created, resized.details.infrastructure.created); });
      await active(page, id); await controlsReachable(page); await screenshot(page, `${sequence}-owner-size-switch`);
      await disable(page, expected);
      assert.deepEqual((await dom(page)).videos.map(video => video.currentSrc), original.videos.map(video => video.currentSrc));
    });
    await journey('video reparent and old-parent removal', 'reinsert', async (page, original) => {
      const id = await enable(page); const before = await active(page, id);
      await page.evaluate(() => { const video = document.querySelector('video'); globalThis[Symbol.for('aethervsr.m10.identity')] = { video, parent: video.parentElement }; });
      await action(page, 'reparent');
      assert(await page.evaluate(() => { const saved = globalThis[Symbol.for('aethervsr.m10.identity')]; return !saved.parent.isConnected && saved.video === document.querySelector('video') && saved.video.parentElement !== saved.parent; }));
      const result = await until(async () => ({ status: await native.inspect(id), view: await dom(page) }), result =>
        result.status.code === 'unsupported-geometry' && result.status.owner === null ||
        result.status.code === 'active' && result.view.canvases.length === 1 && result.view.canvases[0].siblingOfVideo, 10000, journeySignal);
      const settled = result.status; resourceCheck(settled); observedCodes.add(settled.code);
      assert(settled.details.infrastructure.created <= before.details.infrastructure.created + 1, 'Repeated reparent recreation');
      if (settled.code === 'active') { assert.equal(settled.owner, before.owner); await active(page, id); }
      else { detachedCheck(settled, 'unsupported-geometry'); assert.equal(result.view.canvases.length, 0); }
      await stable(id, async value => {
        assert.equal(value.owner, settled.owner); assert.equal(value.details.infrastructure.created, settled.details.infrastructure.created);
        assert.equal(value.details.infrastructure.destroyed, settled.details.infrastructure.destroyed);
        if (!settled.owner) { detachedCheck(value, 'unsupported-geometry'); assert.equal((await dom(page)).canvases.length, 0); }
      });
      emit('reparent-outcome', { outcome: settled.owner ? 'stable recovery' : 'explicit bounded unsupported', status: settled });
      await originalPlaying(page, original.videos[0].currentSrc); await controlsReachable(page);
      await screenshot(page, `${sequence}-reparent`); await disable(page, original);
    });
    await journey('late inserted video', 'late', async page => { const id = await enable(page); await status(id, item => item.code === 'no-video'); await action(page, 'insert'); await ready(page); const original = await dom(page); await active(page, id); await disable(page, original); });
    await journey('late-ready open-shadow media and listener teardown', 'open-shadow-late', async (page, original) => {
      assert.equal(original.videos.length, 1); assert.equal(original.videos[0].shadow, true); assert.equal(original.videos[0].src, null);
      const expected = { ...original, originalHTML: await page.evaluate(html => {
        const fragment = document.createElement('template'); fragment.innerHTML = html;
        fragment.content.querySelector('template[data-m10-shadow-snapshot]').content.querySelector('video').setAttribute('src', '/media/same.mp4');
        return fragment.innerHTML;
      }, original.originalHTML) };
      const held = []; let released = false; let pendingAction;
      const routeMedia = route => { if (released) return route.continue(); held.push(route); };
      const pattern = '**/media/same.mp4'; await page.route(pattern, routeMedia);
      try {
        await page.evaluate(() => {
          const root = document.querySelector('[data-open-shadow]').shadowRoot; const video = root.querySelector('video');
          const state = globalThis[Symbol.for('aethervsr.m10.shadow-ready')] = { events: [], mutations: [], armed: false };
          const descriptor = Object.getOwnPropertyDescriptor(video, 'play'); const play = video.play;
          const completion = new Promise(resolve => { state.finish = resolve; });
          video.play = function (...args) { return play.apply(this, args).then(async result => { await completion; return result; }); };
          const rootEvent = event => { if (event.target === video) state.events.push({ scope: 'shadow', type: event.type, trusted: event.isTrusted, composed: event.composed }); };
          const documentEvent = event => state.events.push({ scope: 'document', type: event.type, trusted: event.isTrusted, composed: event.composed });
          root.addEventListener('loadeddata', rootEvent, true); document.addEventListener('loadeddata', documentEvent, true);
          state.observer = new MutationObserver(records => {
            if (!state.armed) return;
            for (const record of records) {
              if (record.target.matches?.('canvas[data-aethervsr-m10]')) continue;
              if (record.type === 'childList' && [...record.addedNodes, ...record.removedNodes].every(node => node.matches?.('canvas[data-aethervsr-m10]'))) continue;
              state.mutations.push({ type: record.type, target: record.target.nodeName, attribute: record.attributeName });
            }
          });
          for (const target of [document, root]) state.observer.observe(target, { childList: true, subtree: true, attributes: true, characterData: true });
          state.restorePlay = () => { if (descriptor) Object.defineProperty(video, 'play', descriptor); else delete video.play; };
          state.cleanup = () => { state.finish(); state.restorePlay(); state.observer.disconnect(); root.removeEventListener('loadeddata', rootEvent, true); document.removeEventListener('loadeddata', documentEvent, true); };
        });
        const id = await enable(page); const unloaded = await status(id, value => value.owner === null && value.code === 'unsupported-media');
        assert.equal(unloaded.candidates, 1); assert.equal(unloaded.details.infrastructure.created, 0);
        const listeners = () => native.isolated(page, () => {
          const adapter = globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)];
          const root = document.querySelector('[data-open-shadow]').shadowRoot;
          if (!Array.isArray(adapter.listeners)) throw new Error('Discovery listener diagnostics unavailable');
          return { total: adapter.listeners.length, rootTypes: adapter.listeners.filter(([target]) => target === root).map(([, type]) => type).sort() };
        });
        const listening = await listeners(); assert.equal(listening.rootTypes.filter(type => type === 'loadeddata').length, 1); emit('shadow-root-listeners', listening);
        pendingAction = pageOwnedAction(page, 'ready', false, emit, 0, 15000).then(value => ({ value }), error => ({ error }));
        await until(() => held.length, count => count > 0, 5000, journeySignal);
        await status(id, value => value.owner === null && value.code === 'unsupported-media' && value.details.timerCount === 1 && value.details.infrastructure.discoveryCalls > unloaded.details.infrastructure.discoveryCalls);
        assert((await dom(page)).videos[0].readyState < 2);
        await page.evaluate(() => { const state = globalThis[Symbol.for('aethervsr.m10.shadow-ready')]; state.observer.takeRecords(); state.mutations.length = 0; state.armed = true; });
        released = true;
        await bounded(Promise.all(held.splice(0).map(route => route.continue())), 3000, 'Release real shadow video response');
        const attached = await active(page, id); assert.equal(attached.details.infrastructure.created, 1);
        const evidence = await page.evaluate(() => { const state = globalThis[Symbol.for('aethervsr.m10.shadow-ready')]; return { events: state.events, mutations: state.mutations, resultState: document.querySelector('#result').dataset.state }; });
        assert(evidence.events.some(event => event.scope === 'shadow' && event.type === 'loadeddata' && event.trusted && !event.composed));
        assert.equal(evidence.events.filter(event => event.scope === 'document').length, 0);
        assert.deepEqual(evidence.mutations, [], 'Page DOM mutation after readiness release could mask missing shadow-root discovery');
        assert.equal(evidence.resultState, 'pending'); emit('real-shadow-readiness-without-dom-mutation', evidence);
        await page.evaluate(() => { const state = globalThis[Symbol.for('aethervsr.m10.shadow-ready')]; state.armed = false; state.restorePlay(); state.finish(); });
        const actionResult = await pendingAction; if (actionResult.error) throw actionResult.error;
        assert.equal((await dom(page)).canvases[0].shadow, true); await screenshot(page, `${sequence}-open-shadow`);
        const stopped = await disable(page, expected); assert.deepEqual(await listeners(), { total: 0, rootTypes: [] });
        await action(page, 'ready'); await ready(page); await originalPlaying(page, new URL('/media/same.mp4', page.url()).href);
        await stable(id, async value => {
          resourceCheck(value, true); assert.equal(value.details.infrastructure.discoveryCalls, stopped.details.infrastructure.discoveryCalls);
          assert.equal(value.details.infrastructure.created, stopped.details.infrastructure.created); assert.equal((await dom(page)).canvases.length, 0);
          assert.deepEqual(await listeners(), { total: 0, rootTypes: [] });
        });
      } finally {
        if (!page.isClosed()) await bounded(page.evaluate(() => { const key = Symbol.for('aethervsr.m10.shadow-ready'); globalThis[key]?.cleanup(); delete globalThis[key]; }), 3000, 'Restore readiness instrumentation');
        released = true; await bounded(Promise.all(held.splice(0).map(route => route.abort().catch(() => {}))), 3000, 'Release held media requests');
        if (!page.isClosed()) await bounded(page.unroute(pattern, routeMedia), 3000, 'Remove readiness route');
        if (pendingAction) await bounded(pendingAction, 16000, 'Drain page-owned readiness action');
      }
    });
    await journey('SPA replacement, authoritative source change, and back', 'spa', async (page, original) => {
      const id = await enable(page); const before = await active(page, id);
      await page.evaluate(() => { globalThis[Symbol.for('aethervsr.m10.route-videos')] = [document.querySelector('video')]; });
      await action(page, 'route'); await ready(page);
      const routed = await status(id, item => item.code === 'active' && item.owner !== before.owner);
      const routeView = await dom(page); assert.notEqual(routeView.href, original.href);
      await page.evaluate(() => { globalThis[Symbol.for('aethervsr.m10.route-videos')].push(document.querySelector('video')); });
      await action(page, 'source'); await ready(page);
      const changedURL = new URL('/media/same.mp4?changed=1', page.url()).href;
      await until(() => dom(page), view => view.videos[0].currentSrc === changedURL && view.canvases[0]?.videoSource === changedURL, 10000, journeySignal);
      const changed = await active(page, id); assert.equal(changed.owner, routed.owner);
      const sourceView = await originalPlaying(page, changedURL); assert.notEqual(sourceView.videos[0].currentSrc, routeView.videos[0].currentSrc);
      assert.equal(sourceView.videos[0].src, '/media/same.mp4?changed=1');
      assert.deepEqual(changed.details.attachment.source, { w: sourceView.videos[0].width, h: sourceView.videos[0].height });
      await action(page, 'back');
      await until(() => dom(page), view => view.href === original.href && view.videos.length === 1 && view.videos[0].currentSrc === original.videos[0].currentSrc, 10000, journeySignal);
      const returned = await status(id, value => value.code === 'active' && value.owner !== before.owner && value.owner !== routed.owner);
      assert(await page.evaluate(() => globalThis[Symbol.for('aethervsr.m10.route-videos')].every(video => !video.isConnected && video !== document.querySelector('video'))));
      assert.equal(returned.details.infrastructure.created, changed.details.infrastructure.created + 1);
      assert.equal(returned.details.infrastructure.destroyed, changed.details.infrastructure.destroyed + 1);
      await stable(id, value => { assert.equal(value.owner, returned.owner); assert.equal(value.details.infrastructure.created, returned.details.infrastructure.created); });
      await active(page, id); await originalPlaying(page, original.videos[0].currentSrc); await disable(page, original);
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
    await journey('two enabled tabs with independent modes and suspension', 'custom', async (page, original) => {
      const checkPopup = (target, tabId, mode, enabled) => popup(target, async panel => {
        assert.equal(panel.tabId, tabId); const value = await panel.request('m10.status');
        assert.equal(value.mode, mode); assert.equal(value.enabled, enabled);
        const ui = await panel.evaluate(`({mode:document.querySelector('input[name="mode"]:checked')?.value,enableDisabled:document.querySelector('#enable').disabled,disableDisabled:document.querySelector('#disable').disabled})`);
        assert.deepEqual(ui, { mode, enableDisabled: enabled, disableDisabled: !enabled }); emit('tab-local-popup', { tabId, status: value, ui });
      });
      await popup(page, panel => panel.click('input[value="baseline"]'));
      const firstId = await enable(page); const first = await active(page, firstId); assert.equal(first.mode, 'baseline'); assert.equal(first.current, 'baseline');
      assert.equal(first.details.infrastructure.maximumConcurrent, 1);
      const other = await bounded(native.context.newPage(), 3000, 'Create second enabled tab');
      await other.goto(`${fixtures.url}?case=custom&second-enabled=${sequence}`); await ready(other); const otherOriginal = await dom(other);
      await popup(other, panel => panel.click('input[value="auto"]'));
      const secondId = await enable(other); assert.notEqual(secondId, firstId); const second = await active(other, secondId); assert.equal(second.mode, 'auto');
      assert.equal(second.details.infrastructure.maximumConcurrent, 1);
      const suspended = await status(firstId, value => value.enabled && value.code === 'suspended' && value.details.attachment?.suspendedReason === 'document-hidden');
      assert.equal(suspended.owner, first.owner); assert.equal((await dom(page)).visibility, 'hidden'); assert.equal((await dom(page)).canvases[0].visibility, 'hidden');
      await stable(firstId, value => {
        assert.equal(value.enabled, true); assert.equal(value.mode, 'baseline'); assert.equal(value.owner, first.owner);
        assert.equal(value.details.attachment.suspendedReason, 'document-hidden'); assert.equal(value.details.attachment.active, false);
        assert.equal(value.details.attachment.resources.frameCallback, 0); assert.equal(value.details.attachment.session.framesRendered, suspended.details.attachment.session.framesRendered);
      });
      await checkPopup(other, secondId, 'auto', true); await other.bringToFront(); await active(other, secondId);
      await checkPopup(page, firstId, 'baseline', true); await page.bringToFront(); await active(page, firstId);
      const otherSuspended = await status(secondId, value => value.enabled && value.code === 'suspended' && value.details.attachment?.suspendedReason === 'document-hidden');
      assert.equal(otherSuspended.owner, second.owner); assert.equal(otherSuspended.details.attachment.resources.frameCallback, 0);
      assert.equal((await dom(other)).visibility, 'hidden'); assert.equal((await dom(other)).canvases[0].visibility, 'hidden');
      await disable(page, original); await checkPopup(page, firstId, 'baseline', false);
      await other.bringToFront(); const continued = await active(other, secondId);
      assert.equal(continued.enabled, true); assert.equal(continued.mode, 'auto'); assert.equal(continued.owner, second.owner);
      assert.equal(continued.details.infrastructure.created, second.details.infrastructure.created);
      assert.equal(continued.details.infrastructure.destroyed, second.details.infrastructure.destroyed);
      await until(() => native.inspect(secondId), value => value.details.attachment.session.framesRendered > continued.details.attachment.session.framesRendered + 2, 5000, journeySignal);
      await checkPopup(other, secondId, 'auto', true); await active(other, secondId); await screenshot(other, `${sequence}-other-tab-continues`);
      resourceCheck(await native.inspect(firstId), true); await disable(other, otherOriginal);
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
    if (testBuild) await journey('test-only actual GPU device loss stays terminal', 'custom', async (page, original) => {
      const id = await enable(page); const before = await active(page, id);
      try {
        const destroyed = await native.isolated(page, () => {
          const hook = globalThis.__AETHERVSR_EXTENSION_TEST__; const attachment = hook?.attachment();
          if (!hook?.status().enabled || !attachment?.gpu?.device) throw new Error('No active test-build GPU attachment');
          const key = Symbol.for('aethervsr.m10.device-loss');
          const adapterPrototype = Object.getPrototypeOf(attachment.gpu.adapter);
          const gpuPrototype = Object.getPrototypeOf(navigator.gpu);
          const requestDevice = Object.getOwnPropertyDescriptor(adapterPrototype, 'requestDevice');
          const requestAdapter = Object.getOwnPropertyDescriptor(gpuPrototype, 'requestAdapter');
          if (!requestDevice?.value || !requestAdapter?.value) throw new Error('GPU acquisition methods unavailable for observation');
          const observation = globalThis[key] = { adapterRequests: 0, deviceRequests: 0, attachment, restore() {
            Object.defineProperty(adapterPrototype, 'requestDevice', requestDevice); Object.defineProperty(gpuPrototype, 'requestAdapter', requestAdapter);
          } };
          Object.defineProperty(adapterPrototype, 'requestDevice', { ...requestDevice, value: function (...args) { observation.deviceRequests++; return requestDevice.value.apply(this, args); } });
          Object.defineProperty(gpuPrototype, 'requestAdapter', { ...requestAdapter, value: function (...args) { observation.adapterRequests++; return requestAdapter.value.apply(this, args); } });
          attachment.gpu.device.destroy(); return { method: 'GPUDevice.destroy', owner: hook.status().owner };
        });
        emit('actual-device-destroy', destroyed);
        const lost = await status(id, value => value.code === 'device-lost' && value.owner === null); detachedCheck(lost, 'device-lost');
        assert.equal(lost.details.infrastructure.destroyed, before.details.infrastructure.destroyed + 1);
        await stable(id, async value => {
          detachedCheck(value, 'device-lost'); assert.equal(value.details.infrastructure.created, before.details.infrastructure.created);
          const observed = await native.isolated(page, () => { const observation = globalThis[Symbol.for('aethervsr.m10.device-loss')]; return {
            adapterRequests: observation.adapterRequests, deviceRequests: observation.deviceRequests,
            attachmentNull: globalThis.__AETHERVSR_EXTENSION_TEST__.attachment() === null, resources: observation.attachment.snapshot().resources,
          }; });
          assert.equal(observed.attachmentNull, true); assert.equal(observed.adapterRequests, 0); assert.equal(observed.deviceRequests, 0);
          assert(Object.values(observed.resources).every(count => count === 0)); assert.equal((await dom(page)).canvases.length, 0);
          emit('device-loss-no-reacquisition', observed);
        }, 1800);
        await originalPlaying(page, original.videos[0].currentSrc); await screenshot(page, `${sequence}-device-lost-original`); await disable(page, original);
      } finally {
        if (!page.isClosed()) await native.isolated(page, () => { const key = Symbol.for('aethervsr.m10.device-loss'); globalThis[key]?.restore(); delete globalThis[key]; });
      }
    });
    await journey('native extension reload and fresh popup reinjection', 'custom', async page => {
      await page.evaluate(() => {
        const spoof = document.createElement('canvas'); spoof.dataset.aethervsrM10 = crypto.randomUUID(); spoof.dataset.m10PageOwned = 'true';
        spoof.width = 8; spoof.height = 8; spoof.style.cssText = 'position:fixed;right:8px;top:8px;pointer-events:none';
        document.querySelector('main').append(spoof);
        globalThis[Symbol.for('aethervsr.m10.reload-identity')] = { spoof, video: document.querySelector('video') };
      });
      const original = await dom(page); const id = await enable(page); await active(page, id);
      const oldUuid = (await dom(page)).canvases.find(canvas => !canvas.pageOwned).uuid;
      await page.evaluate(() => { const saved = globalThis[Symbol.for('aethervsr.m10.reload-identity')]; saved.output = saved.video.nextElementSibling; });
      const reloaded = await native.reloadExtension(page, journeySignal);
      assert.equal(reloaded.extensionId, native.extensionId); assert.equal(reloaded.oldWorkerClosed, true); assert.equal(reloaded.freshWorkerHandle, true);
      assert.notEqual(reloaded.previousTargetId, reloaded.freshTargetId);
      await until(() => dom(page), view => view.canvases.filter(canvas => !canvas.pageOwned).length === 0, 6000, journeySignal);
      const cleaned = await dom(page); assert.equal(cleaned.originalHTML, original.originalHTML); assert.equal(cleaned.canvases.length, 1); assert.equal(cleaned.canvases[0].pageOwned, true);
      assert(await page.evaluate(() => { const saved = globalThis[Symbol.for('aethervsr.m10.reload-identity')]; return !saved.output.isConnected && saved.spoof.isConnected && saved.spoof === document.querySelector('canvas[data-m10-page-owned]') && saved.video === document.querySelector('video'); }));
      await originalPlaying(page, original.videos[0].currentSrc); await screenshot(page, `${sequence}-reload-cleaned-original`);
      const activatedId = await popup(page, async panel => {
        assert.equal(panel.tabId, id); const inactive = await panel.request('m10.status'); assert.equal(inactive.enabled, false); assert.equal(inactive.owner, null);
        const controls = await panel.evaluate(`({enableDisabled:document.querySelector('#enable').disabled,disableDisabled:document.querySelector('#disable').disabled})`);
        assert.deepEqual(controls, { enableDisabled: false, disableDisabled: true });
        const enabled = await panel.click('#enable'); assert.equal(enabled.enabled, true); emit('fresh-popup-reactivation', { tabId: panel.tabId, inactive, enabled });
        return panel.tabId;
      });
      assert.equal(activatedId, id); const fresh = await active(page, id);
      assert.equal(fresh.details.infrastructure.created, 1); assert.equal(fresh.details.infrastructure.destroyed, 0); assert.equal(fresh.details.infrastructure.maximumConcurrent, 1);
      const freshView = await dom(page); assert.equal(freshView.canvases.length, 2); assert.notEqual(freshView.canvases.find(canvas => !canvas.pageOwned).uuid, oldUuid);
      assert(await page.evaluate(() => { const saved = globalThis[Symbol.for('aethervsr.m10.reload-identity')]; return saved.video.nextElementSibling !== saved.output && saved.spoof.isConnected; }));
      await stable(id, value => { assert.equal(value.owner, fresh.owner); assert.equal(value.details.infrastructure.created, 1); assert.equal(value.details.infrastructure.destroyed, 0); });
      await screenshot(page, `${sequence}-reload-reactivated`); await disable(page, original);
      assert(await page.evaluate(() => { const saved = globalThis[Symbol.for('aethervsr.m10.reload-identity')]; return saved.spoof.isConnected && document.querySelector('canvas') === saved.spoof; }));
    });
    assert(report.journeys.length > 0, 'No journey matched the requested selection');
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
  let testBuild = false, only = null;
  for (let index = 0; index < flags.length; index++) {
    if (flags[index] === '--test-build' && !testBuild) testBuild = true;
    else if (flags[index] === '--only' && only === null && flags[index + 1] && !flags[index + 1].startsWith('--')) only = flags[++index];
    else throw new Error('Invalid or duplicate journey argument');
  }
  if (!output || output.startsWith('--')) throw new Error('Usage: node tools/m10-journeys.mjs output.json [--test-build] [--only "journey name"]');
  const report = await runJourneys(output, testBuild, only);
  console.log(JSON.stringify({ output: resolve(output), verdict: report.verdict, journeys: report.journeys.length, fatal: report.fatal ?? null }));
  if (report.verdict !== 'PASS_LIFECYCLE_ONLY') process.exitCode = 1;
}