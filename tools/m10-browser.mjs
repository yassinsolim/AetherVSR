import assert from 'node:assert/strict';
import { readFileSync, readdirSync, lstatSync, realpathSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, relative } from 'node:path';
import { transform } from 'esbuild';
import { openNativeChrome } from './m9-browser.mjs';
import { ROOT, sha256 } from './m10-fixtures.mjs';

const protocol = await transform(readFileSync(join(ROOT, 'src/extension/protocol.ts'), 'utf8'), { loader: 'ts', format: 'esm' });
const { parseExtensionResponse, MODEL_SHA256, MODEL_BYTES } = await import(`data:text/javascript;base64,${Buffer.from(protocol.code).toString('base64')}`);
export class Unverified extends Error {}
export class OperationTimeout extends Unverified {}
export function snapshotExtensionStatus(raw) {
  const parsed = parseExtensionResponse(raw);
  assert(parsed?.ok, `Invalid/failed extension response: ${JSON.stringify(raw)}`);
  return parsed.status;
}
export async function bounded(operation, milliseconds = 10000, label = 'Operation') {
  let timer;
  try { return await Promise.race([operation, new Promise((_, reject) => { timer = setTimeout(() => reject(new OperationTimeout(`${label} timed out after ${milliseconds}ms`)), milliseconds); })]); }
  finally { clearTimeout(timer); }
}
export async function until(read, accept = Boolean, timeout = 10000, signal) {
  const end = Date.now() + timeout;
  let last;
  do {
    signal?.throwIfAborted();
    last = await bounded(Promise.resolve().then(read), Math.max(1, end - Date.now()));
    signal?.throwIfAborted();
    if (accept(last)) return last;
    await new Promise(done => setTimeout(done, 100));
  } while (Date.now() < end);
  throw new Error(`Condition timed out: ${JSON.stringify(last)}`);
}
export function verifyBuild(testBuild = false, ignoredOutput) {
  const directory = join(ROOT, testBuild ? 'dist-extension-test' : 'dist-extension');
  const provenance = JSON.parse(readFileSync(join(directory, 'build-provenance.json'), 'utf8'));
  const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', timeout: 10000 }).trim();
  const dirty = git(['status', '--porcelain', '--untracked-files=normal', '--', '.', ...(ignoredOutput ? [`:(exclude)${relative(ROOT, ignoredOutput)}`] : [])]);
  assert.equal(dirty, '', `Root clean gate required; freeze/commit and build separately before running:\n${dirty}`);
  assert.equal(provenance.sourceDirty, false, 'Dirty-source builds cannot produce M10 evidence');
  assert.equal(provenance.sourceCommit, git(['rev-parse', 'HEAD']), 'Build is not from the current source commit');
  assert.equal(provenance.schemaVersion, 1); assert.equal(provenance.generator, 'aethervsr-m10');
  assert.equal(provenance.buildKind, testBuild ? 'test' : 'production');
  const names = ['content.js', 'manifest.json', 'models/production.json', 'popup.css', 'popup.html', 'popup.js', 'service-worker.js'];
  assert.deepEqual(Object.keys(provenance.files).sort(), names);
  const actual = readdirSync(directory, { recursive: true }).filter(name => { const info = lstatSync(join(directory, name)); assert(!info.isSymbolicLink()); return info.isFile(); }).sort();
  assert.deepEqual(actual, ['build-provenance.json', ...names].sort());
  for (const name of names) { const bytes = readFileSync(join(directory, name)); assert.deepEqual(provenance.files[name], { sha256: sha256(bytes), bytes: bytes.length }, name); }
  assert.equal(provenance.bundleSha256, sha256(names.map(name => `${name}\0${provenance.files[name].sha256}\n`).join('')));
  assert.equal(provenance.totalBytes, names.reduce((sum, name) => sum + provenance.files[name].bytes, 0));
  assert.equal(provenance.modelSha256, MODEL_SHA256); assert.equal(provenance.modelBytes, MODEL_BYTES);
  assert.deepEqual(provenance.files['models/production.json'], { sha256: MODEL_SHA256, bytes: MODEL_BYTES });
  const manifest = JSON.parse(readFileSync(join(directory, 'manifest.json'), 'utf8'));
  assert.equal(manifest.version, provenance.manifestVersion);
  assert.equal(manifest.background.service_worker, 'service-worker.js'); assert.equal(manifest.action.default_popup, 'popup.html');
  assert.deepEqual([...manifest.permissions].sort(), ['activeTab', 'scripting', 'storage']);
  for (const key of ['host_permissions', 'optional_host_permissions', 'content_scripts', 'web_accessible_resources']) assert(!manifest[key]?.length, key);
  assert.equal(readFileSync(join(directory, 'content.js'), 'utf8').includes('__AETHERVSR_EXTENSION_TEST__'), testBuild);
  return { directory, provenance, manifest, provenanceSha256: sha256(readFileSync(join(directory, 'build-provenance.json'))) };
}

const cdpSend = (session, method, params = {}, timeout = 5000) => bounded(session.send(method, params), timeout, `CDP ${method}`);
async function attach(browserCDP, targetId) {
  const { sessionId } = await cdpSend(browserCDP, 'Target.attachToTarget', { targetId, flatten: false });
  let nextId = 0;
  const pending = new Map();
  const receive = event => {
    if (event.sessionId !== sessionId) return;
    const message = JSON.parse(event.message); const operation = pending.get(message.id);
    if (!operation) return;
    clearTimeout(operation.timer); pending.delete(message.id);
    if (message.error) operation.reject(new Error(JSON.stringify(message.error))); else operation.resolve(message.result);
  };
  browserCDP.on('Target.receivedMessageFromTarget', receive);
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    const fail = error => { clearTimeout(pending.get(id)?.timer); pending.delete(id); reject(error); };
    pending.set(id, { resolve, reject, timer: setTimeout(() => fail(new OperationTimeout(`CDP timeout: ${method}`)), 10000) });
    cdpSend(browserCDP, 'Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id, method, params }) }).catch(fail);
  });
  const evaluate = async expression => {
    const value = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (value.exceptionDetails) throw new Error(JSON.stringify(value.exceptionDetails));
    return value.result.value;
  };
  return { send, evaluate, async close() {
    browserCDP.off('Target.receivedMessageFromTarget', receive);
    for (const operation of pending.values()) { clearTimeout(operation.timer); operation.reject(new Error('CDP session closed')); }
    pending.clear(); await cdpSend(browserCDP, 'Target.detachFromTarget', { sessionId }, 1000).catch(() => {});
  } };
}

export async function openExtension(build, record = () => {}) {
  const flags = ['--autoplay-policy=no-user-gesture-required', '--window-size=1280,900', `--load-extension=${build.directory}`,
    `--disable-extensions-except=${build.directory}`, '--enable-unsafe-extension-debugging'];
  const native = await openNativeChrome(flags);
  try {
    const { browser, context } = native;
    const browserCDP = await bounded(browser.newBrowserCDPSession(), 3000, 'Browser CDP session');
    const extensionId = await until(async () => (await cdpSend(browserCDP, 'Extensions.getExtensions')).extensions.find(item =>
      item.path && existsSync(item.path) && realpathSync(item.path) === realpathSync(build.directory))?.id);
    assert(extensionId, 'Exact unpacked build not installed');
    const workerURL = `chrome-extension://${extensionId}/service-worker.js`;
    const targets = () => cdpSend(browserCDP, 'Target.getTargets', { filter: [{}] });
    const workerTarget = await until(async () => (await targets()).targetInfos.find(target => target.type === 'service_worker' && target.url === workerURL));
    const startup = await attach(browserCDP, workerTarget.targetId);
    try { await startup.send('Runtime.runIfWaitingForDebugger'); } finally { await startup.close(); }
    const worker = () => until(() => context.serviceWorkers().find(candidate => candidate.url() === workerURL));
    const workerEval = async (fn, arg) => bounded((await worker()).evaluate(fn, arg), 5000, 'Service-worker evaluation');
    const tabId = async page => {
      await bounded(page.bringToFront(), 3000, 'Focus action tab');
      return workerEval(async () => { const tabs = await chrome.tabs.query({ active: true, currentWindow: true }); if (tabs.length !== 1) throw new Error('No unique active tab'); return tabs[0].id; });
    };
    const sessions = new Set();
    async function popup(page) {
      await bounded(page.bringToFront(), 3000, 'Focus popup tab');
      const outer = await until(async () => (await cdpSend(browserCDP, 'Target.getTargets', { filter: [{ type: 'tab', exclude: false }, { exclude: true }] })).targetInfos.find(target => target.url === page.url()));
      const before = new Set((await targets()).targetInfos.map(target => target.targetId));
      await cdpSend(browserCDP, 'Extensions.triggerAction', { id: extensionId, targetId: outer.targetId });
      const popupURL = `chrome-extension://${extensionId}/popup.html`;
      const target = await until(async () => (await targets()).targetInfos.find(candidate => !before.has(candidate.targetId) &&
        (candidate.url === popupURL || (candidate.type === 'other' && candidate.url === ''))));
      const connection = await attach(browserCDP, target.targetId); sessions.add(connection);
      try {
        await connection.send('Runtime.runIfWaitingForDebugger');
        await until(() => connection.evaluate(`document.readyState !== 'loading' && location.href === ${JSON.stringify(popupURL)} && document.body.getAttribute('aria-busy') === 'false'`));
        const tabs = await connection.evaluate('chrome.tabs.query({ active: true, currentWindow: true })');
        assert.equal(tabs.length, 1, 'Native popup has no unique active tab');
        assert.equal(tabs[0].url, page.url(), 'Native popup targets a different page');
        const id = tabs[0].id;
        record('native-action', { method: 'Extensions.triggerAction', outerTargetId: outer.targetId, popupTargetId: target.targetId, popupURL, tabId: id });
        const request = async type => snapshotExtensionStatus(await connection.evaluate(`chrome.runtime.sendMessage(${JSON.stringify({ type, tabId: id })})`));
        const click = async selector => {
          const point = await connection.evaluate(`(() => { const element = document.querySelector(${JSON.stringify(selector)}); if (!element || element.disabled) throw new Error('Unavailable popup control'); const rect = element.getBoundingClientRect(); return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }; })()`);
          await connection.send('Input.dispatchMouseEvent', { type: 'mousePressed', ...point, button: 'left', clickCount: 1 });
          await connection.send('Input.dispatchMouseEvent', { type: 'mouseReleased', ...point, button: 'left', clickCount: 1 });
          await until(() => connection.evaluate(`document.body.getAttribute('aria-busy') === 'false'`));
          record('native-popup-click', { selector, tabId: id, ui: await connection.evaluate(`Object.fromEntries(['state', 'message', 'source'].map(id => [id, document.getElementById(id).textContent]))`) });
          return request('m10.status');
        };
        return { ...connection, tabId: id, request, click, async dismiss() {
          const started = Date.now();
          try {
            const result = await bounded(browserCDP.send('Target.closeTarget', { targetId: target.targetId }), 5000, 'Close native popup');
            assert.equal(result.success, true, 'Native popup target did not close');
            await until(async () => (await targets()).targetInfos.every(candidate => candidate.targetId !== target.targetId), Boolean, 3000);
            await bounded(page.bringToFront(), 3000, 'Focus page after popup dismissal');
            record('native-popup-dismissed', { popupTargetId: target.targetId, method: 'Target.closeTarget', elapsedMs: Date.now() - started });
          } finally { await bounded(connection.close(), 1500, 'Detach popup session'); sessions.delete(connection); }
        } };
      } catch (error) { await connection.close(); sessions.delete(connection); throw error; }
    }
    const inspect = async id => snapshotExtensionStatus(await workerEval(async tab => {
      const record = (await chrome.storage.session.get(`m10.document.${tab}`))[`m10.document.${tab}`];
      if (!record) throw new Error('No actual popup activation registration');
      return chrome.tabs.sendMessage(tab, { type: 'm10.inspect' }, { frameId: 0, documentId: record.documentId });
    }, id));
    const isolated = async (page, fn, arg) => {
      const session = await bounded(context.newCDPSession(page), 3000, 'Isolated-world session'); const worlds = [];
      session.on('Runtime.executionContextCreated', event => worlds.push(event.context));
      try {
        const { frameTree } = await cdpSend(session, 'Page.getFrameTree'); await cdpSend(session, 'Runtime.enable');
        const world = await until(async () => {
          for (const candidate of worlds.filter(item => item.auxData?.frameId === frameTree.frame.id && item.auxData?.isDefault === false)) {
            const identity = await cdpSend(session, 'Runtime.evaluate', { contextId: candidate.id,
              expression: `typeof chrome !== 'undefined' && chrome.runtime?.id === ${JSON.stringify(extensionId)}`, returnByValue: true });
            if (identity.result?.value === true) return candidate;
          }
          return null;
        });
        const result = await cdpSend(session, 'Runtime.evaluate', { contextId: world.id, expression: `(${fn.toString()})(${JSON.stringify(arg ?? null)})`, returnByValue: true, awaitPromise: true });
        if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
        return result.result.value;
      } finally { await bounded(session.detach(), 1000, 'Detach isolated-world session'); }
    };
    const stopWorker = async page => {
      const started = Date.now();
      const target = (await targets()).targetInfos.find(item => item.type === 'service_worker' && item.url === workerURL);
      if (!target) throw new Unverified('No running extension worker to stop');
      const session = await bounded(context.newCDPSession(page), 3000, 'Worker-stop observer'); const versions = new Map();
      session.on('ServiceWorker.workerVersionUpdated', event => event.versions.forEach(version => versions.set(version.versionId, version)));
      try {
        await cdpSend(session, 'ServiceWorker.enable', {}, 2000);
        const result = await cdpSend(browserCDP, 'Target.closeTarget', { targetId: target.targetId }, 3000);
        record('worker-stop-request', { method: 'Target.closeTarget', workerURL, targetId: target.targetId, result });
        if (!result.success) {
          const version = await until(() => [...versions.values()].find(item => item.scriptURL === workerURL && item.runningStatus === 'running'), Boolean, 2000);
          record('worker-stop-request', { method: 'ServiceWorker.stopWorker', workerURL, versionId: version.versionId });
          await cdpSend(session, 'ServiceWorker.stopWorker', { versionId: version.versionId }, 3000);
        }
        await until(async () => (await targets()).targetInfos.every(item => item.type !== 'service_worker' || item.url !== workerURL), Boolean, 3000);
        const evidence = { workerURL, targetId: target.targetId, elapsedMs: Date.now() - started,
          versions: [...versions.values()].filter(item => item.scriptURL === workerURL), targetAbsent: true };
        record('worker-stopped', evidence); return evidence;
      } catch (error) {
        record('worker-stop-unverified', { workerURL, targetId: target.targetId, elapsedMs: Date.now() - started, error: String(error) });
        if (error instanceof OperationTimeout) throw error;
        throw new Unverified(`Actual service-worker shutdown unverified: ${error}`);
      } finally { await bounded(session.detach(), 1000, 'Detach worker-stop observer').catch(() => {}); }
    };
    const captureFullscreenWindow = async (page, path) => {
      const target = (await targets()).targetInfos.find(item => item.type === 'page' && item.url === page.url());
      assert(target, 'Fullscreen page target missing');
      const window = await cdpSend(browserCDP, 'Browser.getWindowForTarget', { targetId: target.targetId });
      record('native-fullscreen-window', window);
      if (window.bounds.windowState !== 'fullscreen') throw new Unverified('Chrome window is not in native fullscreen');
      if (process.platform !== 'darwin') throw new Unverified('Native-window screenshot capture is not implemented for this OS');
      assert(!existsSync(path), 'Never overwrite native-window evidence');
      const { left, top, width, height } = window.bounds;
      try { execFileSync('screencapture', ['-x', '-R', `${left},${top},${width},${height}`, path], { timeout: 3000 }); }
      catch (error) { throw new Unverified(`Native-window screenshot unavailable; check normal OS screen-recording permission: ${error}`); }
      const after = await cdpSend(browserCDP, 'Browser.getWindowForTarget', { targetId: target.targetId });
      const fullscreen = await bounded(page.evaluate(() => document.fullscreenElement?.tagName ?? null), 3000, 'Fullscreen after native screenshot');
      const evidence = { path: relative(ROOT, path), sha256: sha256(readFileSync(path)), window, after, fullscreen,
        scope: 'macOS screen capture of Chrome window bounds; manual visual review UNVERIFIED' };
      record('native-fullscreen-screenshot', evidence);
      if (after.bounds.windowState !== 'fullscreen' || !fullscreen) throw new Unverified('Fullscreen exited during native-window capture');
      return evidence;
    };
    record('browser', { version: await cdpSend(browserCDP, 'Browser.getVersion'), executable: native.executable,
      executableSha256: sha256(readFileSync(native.executable)), flags, extensionId, workerURL, launch: 'Native Chrome; default context; noDefaults:true; no focus emulation' });
    let closing;
    return { ...native, extensionId, workerEval, tabId, popup, inspect, isolated, stopWorker, captureFullscreenWindow,
      workerRunning: async () => (await targets()).targetInfos.some(target => target.type === 'service_worker' && target.url === workerURL),
      close() { return closing ??= (async () => { try { await Promise.all([...sessions].map(session => bounded(session.close(), 1500).catch(() => {}))); } finally { await native.close(); } })(); } };
  } catch (error) { await native.close(); throw error; }
}