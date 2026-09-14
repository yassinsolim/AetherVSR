import assert from 'node:assert/strict';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ROOT, startFixtures, sha256 } from './m10-fixtures.mjs';
import { attach, bounded, openExtension, until, verifyBuild } from './m10-browser.mjs';
import { pageSnapshot } from './m10-journeys.mjs';

export async function runReload(output, mechanism, earlyResume) {
  output = resolve(output);
  assert(['runtime', 'ui'].includes(mechanism), 'Use runtime or ui reload');
  assert(relative(resolve(ROOT, '.cache/m105'), output) && !relative(resolve(ROOT, '.cache/m105'), output).startsWith('..'), 'Local evidence directory required');
  assert(!existsSync(output), 'Never overwrite reload evidence'); mkdirSync(dirname(output), { recursive: true });
  const report = { schemaVersion: 1, mechanism, earlyResume, verdict: 'UNVERIFIED', events: [],
    scope: 'Actual installed whole-package reload. Runtime request is unshipped diagnostic code; UI uses original Chrome reload control. Same-document and refresh-assisted behavior are distinct. Resource counters are references, not measured GPU residency.',
    sourceSha256: sha256(readFileSync(fileURLToPath(import.meta.url))), started: new Date().toISOString() };
  const emit = (name, data) => report.events.push({ name, at: new Date().toISOString(), data });
  let native, fixtures, session, browserCDP, stopped = false;
  const pending = new Map();
  try {
    report.build = verifyBuild(false); fixtures = await startFixtures({ mse: false });
    native = await openExtension(report.build, emit);
    const page = await native.context.newPage(); page.setDefaultTimeout(5000);
    await page.goto(`${fixtures.url}?case=custom`);
    await page.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
    const panel = await native.popup(page);
    let tabId;
    try { tabId = panel.tabId; await panel.click('#enable'); await panel.click('input[value="baseline"]'); }
    finally { await panel.dismiss(); }
    report.before = await until(() => native.inspect(tabId), state => state.code === 'active' && state.current === 'baseline');
    await page.evaluate(() => {
      const video = document.querySelector('video'), canvas = video.nextElementSibling;
      const spoof = document.createElement('canvas'); spoof.dataset.aethervsrM10 = crypto.randomUUID(); spoof.dataset.m10PageOwned = 'true';
      spoof.style.cssText = 'position:fixed;right:8px;top:8px;width:8px;height:8px;pointer-events:none'; document.querySelector('main').append(spoof);
      globalThis[Symbol.for('aethervsr.m105.reload-page')] = { video, canvas, spoof };
    });
    report.original = await page.evaluate(pageSnapshot, native.extensionId);
    session = await native.context.newCDPSession(page); const worlds = [];
    session.on('Runtime.executionContextCreated', event => worlds.push(event.context));
    await session.send('Runtime.enable');
    const world = await until(async () => {
      for (const candidate of worlds.filter(value => value.auxData?.isDefault === false)) {
        const result = await session.send('Runtime.evaluate', { contextId: candidate.id, expression: `typeof chrome !== 'undefined' && chrome.runtime?.id === ${JSON.stringify(native.extensionId)}`, returnByValue: true });
        if (result.result?.value) return candidate;
      }
      return null;
    });
    const oldEvaluate = async expression => {
      try {
        const value = await bounded(session.send('Runtime.evaluate', { contextId: world.id, expression, returnByValue: true, awaitPromise: true }), 3000, 'Old isolated world');
        return value.exceptionDetails ? { exception: value.exceptionDetails.text, description: value.exceptionDetails.exception?.description } : value.result.value;
      } catch (error) { return { unavailable: String(error) }; }
    };
    await oldEvaluate(`globalThis[Symbol.for('aethervsr.m105.old')] = {manager:globalThis[Symbol.for('aethervsr.m10.document.'+chrome.runtime.id)],attachment:globalThis[Symbol.for('aethervsr.m10.document.'+chrome.runtime.id)].attachment}; true`);
    browserCDP = await native.browser.newBrowserCDPSession();
    const targets = async () => (await bounded(browserCDP.send('Target.getTargets', { filter: [{}] }), 3000, 'Reload targets')).targetInfos;
    const workerURL = `chrome-extension://${native.extensionId}/service-worker.js`;
    const oldTarget = (await targets()).find(target => target.type === 'service_worker' && target.url === workerURL);
    assert(oldTarget); report.oldTargetId = oldTarget.targetId; report.oldWorld = world;
    const resume = target => {
      if (stopped || target.type !== 'service_worker' || target.url !== workerURL || target.targetId === oldTarget.targetId || pending.has(target.targetId)) return;
      emit('fresh-worker-observed', target);
      const task = (async () => {
        if (!earlyResume) return;
        const connection = await attach(browserCDP, target.targetId);
        try {
          await connection.send('Runtime.runIfWaitingForDebugger');
          emit('fresh-worker-explicit-resume', { targetId: target.targetId, value: await connection.evaluate('({id:chrome.runtime.id,url:location.href})') });
        } finally { await connection.close(); }
      })().catch(error => emit('fresh-worker-resume-error', { targetId: target.targetId, error: String(error) }));
      pending.set(target.targetId, task);
    };
    browserCDP.on('Target.targetCreated', event => resume(event.targetInfo));
    browserCDP.on('Target.targetInfoChanged', event => resume(event.targetInfo));
    await browserCDP.send('Target.setDiscoverTargets', { discover: true });
    if (mechanism === 'runtime') {
      report.request = await native.workerEval(() => { setTimeout(() => chrome.runtime.reload(), 0); return { id: chrome.runtime.id, method: 'unshipped worker diagnostic runtime.reload' }; });
    } else {
      const extensions = await native.context.newPage(); await extensions.goto('chrome://extensions/');
      const developer = extensions.locator('extensions-toolbar #devMode');
      if (await developer.getAttribute('aria-pressed') === 'false') await developer.click();
      const item = extensions.locator(`extensions-item[id="${native.extensionId}"]`);
      await item.locator('#dev-reload-button').click({ timeout: 5000 });
      report.request = { method: 'Trusted Chrome extensions page Reload control', extensionId: native.extensionId };
      await extensions.close(); await page.bringToFront();
    }
    await until(async () => (await targets()).every(target => target.targetId !== oldTarget.targetId), Boolean, 5000);
    report.oldWorkerDead = true;
    for (const target of await targets()) resume(target);
    report.oldCleanup = await until(() => oldEvaluate(`(()=>{const old=globalThis[Symbol.for('aethervsr.m105.old')];return {runtimeId:chrome.runtime?.id??null,status:old.manager.status(),resources:old.attachment?.snapshot().resources??null};})()`), value => value.unavailable || value.exception || value.status?.enabled === false, 5000).catch(error => ({ error: String(error) }));
    report.oldPrivilege = await oldEvaluate(`(async()=>{try{const result=await chrome.runtime.sendMessage({type:'m10.model'});return {privileged:result?.ok===true};}catch(error){return {privileged:false,error:String(error)};}})()`);
    assert.notEqual(report.oldPrivilege?.privileged, true, 'Old world retained model privilege');
    report.afterReload = await page.evaluate(pageSnapshot, native.extensionId);
    const freshWorker = await until(async () => (await targets()).find(target => target.type === 'service_worker' && target.url === workerURL && target.targetId !== oldTarget.targetId), Boolean, 10000);
    resume(freshWorker);
    await bounded(Promise.all([...pending.values()]), 5000, 'Fresh worker resume');
    report.workerReadyBeforeAction = await native.workerEval(() => ({ id: chrome.runtime.id, url: location.href }));
    assert.deepEqual(report.workerReadyBeforeAction, { id: native.extensionId, url: workerURL });
    emit('fresh-worker-ready-before-action', { targetId: freshWorker.targetId, ...report.workerReadyBeforeAction });
    const reenable = async () => {
      const probe = bounded(browserCDP.send('Browser.getVersion'), 3000, 'Concurrent browser liveness').then(value => emit('browser-liveness', { product: value.product })).catch(error => emit('browser-liveness-error', String(error)));
      const action = await native.popup(page);
      try {
        report.freshInactive = await action.request('m10.status');
        assert.equal(report.freshInactive.mode, 'baseline', 'Origin mode did not persist');
        assert.equal(report.freshInactive.enabled, false, 'Reload auto-enabled the document');
        await action.click('#enable');
      } finally { await action.dismiss(); await probe; }
      report.freshActive = await until(() => native.inspect(tabId), value => value.code === 'active' && value.current === 'baseline', 10000);
      assert.equal(report.freshActive.details.infrastructure.maximumConcurrent, 1);
      assert.equal(report.freshActive.details.infrastructure.created - report.freshActive.details.infrastructure.destroyed, 1);
    };
    try { await reenable(); report.recovery = 'same-document'; }
    catch (error) {
      report.sameDocumentError = String(error);
      await page.reload({ timeout: 10000 }); await page.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
      await reenable(); report.recovery = 'page-refresh-required';
    }
    const freshTargets = await targets(); report.newWorkers = freshTargets.filter(target => target.type === 'service_worker' && target.url === workerURL);
    assert.equal(report.newWorkers.length, 1); assert.notEqual(report.newWorkers[0].targetId, oldTarget.targetId);
    if (report.recovery === 'same-document') {
      report.identity = await page.evaluate(() => { const saved = globalThis[Symbol.for('aethervsr.m105.reload-page')]; return {
        sameVideo: saved.video === document.querySelector('video'), oldCanvasDetached: !saved.canvas.isConnected,
        spoofPreserved: saved.spoof.isConnected, newCanvas: saved.video.nextElementSibling !== saved.canvas,
        paused: saved.video.paused, quality: saved.video.getVideoPlaybackQuality().totalVideoFrames,
      }; });
      assert(report.identity.sameVideo && report.identity.oldCanvasDetached && report.identity.spoofPreserved && report.identity.newCanvas && !report.identity.paused);
      assert(report.identity.quality > report.original.videos[0].decodedFrames);
      if (report.oldCleanup.resources) assert(Object.values(report.oldCleanup.resources).every(count => count === 0));
    }
    const disable = await native.popup(page);
    try { report.disabled = await disable.click('#disable'); } finally { await disable.dismiss(); }
    assert.equal(report.disabled.details.timerCount, 0); assert.equal(report.disabled.details.discoveryActive, false);
    assert(Object.values(report.disabled.details.lastTeardown).every(count => count === 0));
    report.afterDisable = await page.evaluate(pageSnapshot, native.extensionId);
    if (report.recovery === 'same-document') assert.equal(report.afterDisable.originalHTML, report.original.originalHTML);
    report.verdict = report.recovery === 'same-document' ? 'PASS' : 'PASS_REFRESH_REQUIRED';
    assert.deepEqual(verifyBuild(false), report.build);
  } catch (error) { report.error = String(error); }
  finally {
    stopped = true;
    await bounded(Promise.allSettled([...pending.values()]), 4000, 'Resume task cleanup').catch(error => emit('resume-cleanup', String(error)));
    await session?.detach().catch(() => {}); await browserCDP?.detach().catch(() => {});
    await native?.close().catch(error => { report.cleanupError = String(error); report.verdict = 'UNVERIFIED'; });
    await fixtures?.close(); report.finished = new Date().toISOString(); writeFileSync(output, JSON.stringify(report, null, 2), { flag: 'wx' });
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert(process.argv.length >= 4 && process.argv.length <= 5 && (!process.argv[4] || process.argv[4] === '--early-resume'), 'Usage: node tools/m105-reload.mjs OUTPUT runtime|ui [--early-resume]');
  const report = await runReload(process.argv[2], process.argv[3], process.argv[4] === '--early-resume');
  console.log(JSON.stringify({ verdict: report.verdict, error: report.error, recovery: report.recovery }));
  if (!report.verdict.startsWith('PASS')) process.exitCode = 1;
}