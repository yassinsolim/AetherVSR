import assert from 'node:assert/strict';
import { readFileSync, existsSync, mkdirSync, openSync, writeSync, ftruncateSync, closeSync } from 'node:fs';
import { resolve, dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { hostname, platform, release, arch, cpus } from 'node:os';
import { execFileSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { ROOT, sha256 } from './m10-fixtures.mjs';
import { bounded, openExtension, OperationTimeout, until, verifyBuild, Unverified } from './m10-browser.mjs';

const SITES = {
  plyr: { url: 'https://plyr.io/', player: '.plyr', kind: 'Public custom-control player demo' },
  videojs: { url: 'https://videojs.org/', player: '.video-js', kind: 'Independent public custom-control player demo' },
  shaka: { url: 'https://shaka-project.github.io/shaka-player-release/demo/', player: '.shaka-video-container',
    assetTitle: 'Big Buck Bunny: the Dark Truths', kind: 'Public clear adaptive-streaming demo selected through its existing asset card' },
};
const NEGATIVE = new Set(['inactive', 'permission-required', 'no-video', 'unsupported', 'unsupported-page',
  'unsupported-geometry', 'unsupported-media', 'unsupported-controls', 'unsupported-frame', 'protected-media',
  'cors-blocked', 'webgpu-unavailable', 'device-lost', 'error']);
const CORS_ERRORS = [
  "SecurityError: Failed to execute 'importExternalTexture' on 'GPUDevice': Video element is tainted by cross-origin data and may not be loaded.",
  "SecurityError: Failed to execute 'copyExternalImageToTexture' on 'GPUQueue': Video element is tainted by cross-origin data and may not be loaded.",
];

function redact(value) {
  return String(value).replace(/(?:https?:\/\/|chrome-extension:\/\/|blob:|data:|file:\/\/)[^\s<>"']+/gi, address => {
    try {
      const url = new URL(address);
      if (url.protocol === 'chrome-extension:') return address;
      if (url.origin !== 'null' && url.pathname === '/' && !url.search && !url.hash && !url.username && !url.password) return url.origin;
      return `${url.origin === 'null' ? url.protocol : url.origin}/[redacted]`;
    }
    catch { return '[redacted URL]'; }
  }).slice(0, 2048);
}

function safeEvidence(value) {
  if (typeof value === 'string') return redact(value);
  if (Array.isArray(value)) return value.map(safeEvidence);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, safeEvidence(entry)]));
  return value;
}

export function siteSnapshot(extensionId) {
  const rect = element => { const bounds = element.getBoundingClientRect(); return { left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height }; };
  const properties = ['display', 'visibility', 'opacity', 'position', 'z-index', 'object-fit', 'object-position',
    'overflow-x', 'overflow-y', 'transform', 'filter', 'backdrop-filter', 'perspective', 'contain', 'will-change',
    'clip-path', 'mask-image', 'border-radius', 'border-width', 'padding', 'mix-blend-mode', 'isolation', 'pointer-events',
    'content-visibility', 'container-type', 'overflow-clip-margin', 'translate', 'rotate', 'scale', 'clip',
    '-webkit-mask-image', 'zoom', 'border-top-left-radius', 'border-top-right-radius',
    'border-bottom-right-radius', 'border-bottom-left-radius', 'border-top-width', 'border-right-width',
    'border-bottom-width', 'border-left-width', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left'];
  const style = element => { const computed = getComputedStyle(element); return Object.fromEntries(properties.map(key => [key, computed.getPropertyValue(key)])); };
  const source = video => {
    try { const url = new URL(video.currentSrc); return { present: true, scheme: url.protocol, origin: url.origin === 'null' ? null : url.origin }; }
    catch { return { present: false, scheme: null, origin: null }; }
  };
  const videos = [...document.querySelectorAll('video')].map((video, index) => {
    const ancestors = []; let ancestor = video.parentElement;
    while (ancestor && ancestors.length < 32) {
      ancestors.push({ tag: ancestor.tagName, id: ancestor.id, classes: typeof ancestor.className === 'string' ? ancestor.className : null,
        assignedSlot: ancestor.assignedSlot?.tagName ?? null, offsetParent: ancestor.offsetParent?.tagName ?? null,
        clientLeft: ancestor.clientLeft, clientTop: ancestor.clientTop, clientWidth: ancestor.clientWidth, clientHeight: ancestor.clientHeight,
        offsetWidth: ancestor.offsetWidth, offsetHeight: ancestor.offsetHeight, rect: rect(ancestor), style: style(ancestor) });
      ancestor = ancestor.parentElement;
    }
    const quality = video.getVideoPlaybackQuality?.();
    return { index, rect: rect(video), style: style(video), ancestors, ancestorsTruncated: !!ancestor,
      source: source(video), controls: video.controls, crossOrigin: video.crossOrigin, mediaKeys: !!video.mediaKeys,
      width: video.videoWidth, height: video.videoHeight, readyState: video.readyState, networkState: video.networkState,
      paused: video.paused, ended: video.ended, seeking: video.seeking, currentTime: video.currentTime,
      duration: Number.isFinite(video.duration) ? video.duration : null, playbackRate: video.playbackRate,
      error: video.error ? { code: video.error.code } : null,
      seekable: Array.from({ length: video.seekable.length }, (_, range) => ({ start: video.seekable.start(range), end: video.seekable.end(range) })),
      captions: [...video.textTracks].map(track => ({ kind: track.kind, mode: track.mode })),
      decodedFrames: quality?.totalVideoFrames ?? null, droppedFrames: quality?.droppedVideoFrames ?? null };
  });
  return { at: performance.now(), origin: location.origin, visibility: document.visibilityState, focused: document.hasFocus(),
    viewport: { width: innerWidth, height: innerHeight, devicePixelRatio }, scroll: { x: scrollX, y: scrollY },
    fullscreen: document.fullscreenElement?.tagName ?? null, pip: !!document.pictureInPictureElement,
    mainTestHook: typeof globalThis.__AETHERVSR_EXTENSION_TEST__,
    mainSingleton: typeof globalThis[Symbol.for(`aethervsr.m10.document.${extensionId}`)], videos,
    canvases: [...document.querySelectorAll('canvas[data-aethervsr-m10]')].map(canvas => ({
      uuid: canvas.dataset.aethervsrM10, rect: rect(canvas), style: style(canvas), width: canvas.width, height: canvas.height,
      videoIndex: [...document.querySelectorAll('video')].indexOf(canvas.previousElementSibling),
    })) };
}

function geometryCheck(status, dom, owner) {
  assert.equal(dom.mainTestHook, 'undefined', 'Public page sees a test hook');
  assert.equal(dom.mainSingleton, 'undefined', 'Public page sees extension singleton');
  assert.equal(dom.visibility, 'visible', 'Public test tab hidden');
  assert.equal(dom.focused, true, 'Public test tab unfocused');
  assert.equal(status.owner, owner, 'Original owner changed');
  assert.equal(status.enabled, true);
  assert(status.code === 'active' || status.code === 'suspended', `Pipeline is ${status.code}`);
  const details = status.details; const attachment = details?.attachment;
  assert(attachment, 'Missing attachment diagnostics');
  assert.equal(details.infrastructure.maximumConcurrent, 1);
  assert.equal(details.infrastructure.created - details.infrastructure.destroyed, 1);
  assert.equal(attachment.infrastructure.cleanupErrors, 0);
  for (const key of ['device', 'pipeline', 'canvas', 'resizeObservers']) assert.equal(attachment.resources[key], 1, key);
  assert.equal(dom.canvases.length, 1, 'Expected exactly one owned canvas');
  const canvas = dom.canvases[0]; const video = dom.videos[canvas.videoIndex];
  assert(video, 'Canvas is not next to an original video');
  assert.equal(video.mediaKeys, false); assert.equal(video.controls, false);
  assert.equal(canvas.style['pointer-events'], 'none');
  assert(canvas.width > 0 && canvas.height > 0);
  if (attachment.active && attachment.ready && !attachment.suspendedReason) {
    assert.equal(canvas.style.visibility, 'visible');
    for (const key of ['left', 'top', 'width', 'height']) {
      assert(Math.abs(canvas.rect[key] - attachment.cssRect[key]) < 1, `Canvas/status geometry ${key}`);
      assert(Math.abs(canvas.rect[key] - video.rect[key]) < 1, `Canvas/original geometry ${key}`);
    }
  } else assert.equal(canvas.style.visibility, 'hidden', 'Suspended output remains visible');
  return video;
}

async function visibleButton(scope, name) {
  const buttons = scope.getByRole('button', { name });
  for (let index = 0; index < await buttons.count(); index++) {
    const button = buttons.nth(index);
    if (await button.isVisible() && await button.isEnabled()) return button;
  }
  return null;
}

async function consent(page, record) {
  const explicit = await visibleButton(page, /^(Reject all cookies|Accept all cookies|Allow all cookies|Only necessary cookies|Continue without accepting)$/i);
  if (explicit) { await explicit.click({ timeout: 3000 }); record('cookie-consent', { method: 'Visible cookie-specific button' }); return; }
  const banners = page.getByRole('dialog').or(page.locator('#onetrust-banner-sdk, .cc-window, [aria-label*="cookie" i]'));
  for (let index = 0; index < await banners.count(); index++) {
    const banner = banners.nth(index);
    if (!await banner.isVisible() || !/cookie/i.test(await banner.innerText({ timeout: 1000 }))) continue;
    const button = await visibleButton(banner, /^(Reject all|Accept all|Accept|Agree|Allow all|Continue|Got it|OK)$/i);
    if (button) { await button.click({ timeout: 3000 }); record('cookie-consent', { method: 'Visible cookie banner button' }); return; }
  }
}

export function playerCommand(action) {
  assert(['play', 'pause'].includes(action), 'Unknown player command');
  return action === 'pause' ? /^Pause(?: Video)?(?:, .+)?$/i : /^(Play(?: Video)?|Replay|Restart)(?:, .+)?$/i;
}

async function playerButton(page, site, action) {
  const name = playerCommand(action);
  const player = page.locator(site.player).filter({ has: page.locator('video') }).first();
  if (!await player.count()) throw new Unverified('Original custom player container not found');
  await player.hover({ timeout: 3000 });
  const button = await visibleButton(player, name);
  if (!button) throw new Unverified(`No visible original ${action} button; no scripted autoplay fallback`);
  await button.click({ timeout: 3000 });
}

function publicConfig(path) {
  const config = path ? JSON.parse(readFileSync(resolve(path), 'utf8')) : {};
  assert(config && typeof config === 'object' && !Array.isArray(config));
  assert(Object.keys(config).every(key => ['sites', 'durationMs'].includes(key)), 'Config accepts only sites and durationMs');
  const durationMs = config.durationMs ?? 180000; const sites = config.sites ?? ['plyr', 'videojs'];
  assert(Number.isSafeInteger(durationMs) && durationMs > 0 && durationMs <= 600000, 'durationMs must be 1..600000');
  assert(Array.isArray(sites) && sites.length > 0 && sites.every(id => Object.hasOwn(SITES, id)) && new Set(sites).size === sites.length,
    'sites must be unique entries from plyr, videojs, shaka');
  return { durationMs, sites };
}

function errorObserver(page, extensionId, item) {
  const extensionOrigin = `chrome-extension://${extensionId}/`;
  const add = (kind, message, location = '') => {
    const origins = [...new Set((`${location}\n${message}`.match(/(?:https?:\/\/|chrome-extension:\/\/)[^\s)<>"']+/g) ?? [])
      .map(address => { try { const url = new URL(address); return url.protocol === 'chrome-extension:' ? `chrome-extension://${url.host}` : url.origin; } catch { return null; } }).filter(Boolean))];
    const cors = CORS_ERRORS.find(text => message.split('\n')[0] === text);
    const extension = location.startsWith(extensionOrigin) || message.includes(extensionOrigin) || /\baethervsr\b/i.test(message);
    const gpu = /GPUValidationError|GPUDevice|GPUQueue|WebGPU|importExternalTexture|copyExternalImageToTexture/.test(message);
    const category = cors ? 'cors-import' : extension ? 'extension' : gpu ? 'gpu-unattributed'
      : origins.length && origins.every(origin => /^https?:/.test(origin)) ? 'third-party-page' : 'unattributed';
    item.errors.counts[category] = (item.errors.counts[category] ?? 0) + 1;
    const detail = { kind, category, origins, message: category === 'third-party-page'
      ? 'Third-party script error; message and URL paths not retained' : redact(message.split('\n')[0]) };
    if (item.errors.samples.length < 80) item.errors.samples.push(detail); else item.errors.omittedSamples++;
  };
  const pageError = error => add('pageerror', String(error.stack ?? error));
  const consoleError = message => {
    const location = message.location().url; const text = message.text();
    if (message.type() === 'error' || message.type() === 'warning' && (location.startsWith(extensionOrigin) || /\baethervsr\b/i.test(text))) {
      add(`console-${message.type()}`, text, location);
    }
  };
  page.on('pageerror', pageError); page.on('console', consoleError);
  return () => { page.off('pageerror', pageError); page.off('console', consoleError); };
}

export async function nativeResize(native, page, width) {
  const session = await bounded(native.context.newCDPSession(page), 3000, 'Native resize CDP');
  const send = (method, params = {}) => bounded(session.send(method, params), 3000, method);
  try {
    const before = await send('Browser.getWindowForTarget');
    assert.equal(before.bounds.windowState, 'normal', 'Only resize a normal native window');
    await send('Browser.setWindowBounds', { windowId: before.windowId, bounds: { left: 40, top: 40, width, height: 900 } });
    return await until(async () => {
      const value = await send('Browser.getWindowForTarget'); return value.bounds.width === width ? value : null;
    }, Boolean, 3000);
  } finally { await bounded(session.detach(), 1000, 'Detach native resize CDP'); }
}

async function runtimeEvidence(native, page) {
  return native.isolated(page, () => {
    const adapter = globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)];
    const attachment = adapter?.attachment;
    if (!attachment?.driver || !attachment.pipeline || !attachment.gpu) return null;
    const stats = attachment.pipeline.stats(performance.now());
    return { owner: adapter.status().owner, runtime: attachment.driver.snapshot(),
      importPath: stats.importPath, upscalerId: stats.upscalerId, source: stats.sourceSize, target: stats.targetSize,
      features: [...attachment.gpu.device.features], capabilities: attachment.gpu.capabilities };
  });
}

export async function runSites(output, configPath) {
  const config = publicConfig(configPath);
  output = resolve(output);
  assert.equal(dirname(output), join(ROOT, '.cache/m10'), 'Public evidence belongs in ignored .cache/m10');
  assert(/^sites[^/]*\.json$/.test(relative(dirname(output), output)), 'Use a sites*.json evidence filename');
  assert(!existsSync(output), 'Never overwrite public-site evidence');
  execFileSync('git', ['check-ignore', '--quiet', output], { cwd: ROOT, timeout: 3000 });
  const report = { schemaVersion: 1, started: new Date().toISOString(), completion: 'RUNNING', publicClaim: 'UNVERIFIED', config,
    scope: 'Public-site integration only, not the root M10 release verdict or a calibrated performance benchmark. Two successful independent sites for READY, one for PARTIAL.',
    measurementScope: 'Exact RuntimeSession snapshots since attachment start, including startup, source loads, seeks and replay. Active rates exclude pause/suspension but include stalls. Frames are successful submissions, not pixel validation. GPU quantiles retain the source snapshot scopes.',
    performanceGate: 'not measured: serialize separately with root performance runs',
    mediaPolicy: 'Clear demo playback through existing player controls; no accounts, EME/CDM changes, media attribute/style changes, import rescue, or media URL persistence. No public screenshots or copyrighted media are saved.',
    sourceScope: 'Only currentSrc scheme and origin are retained. A blob origin identifies its creator, not the segment origin. Codec, asset identity, DRM clearance, and HTTP CORS headers are not inferred from a successful document load.',
    manualVisualReview: 'not measured: DOM geometry and owner checks do not establish visual quality',
    psnr: 'not measured', displayRefreshRate: 'not measured',
    machine: { hostname: hostname(), os: platform(), release: release(), arch: arch(), cpu: cpus()[0]?.model ?? 'not measured',
      osVersion: platform() === 'darwin' ? execFileSync('sw_vers', ['-productVersion'], { encoding: 'utf8', timeout: 3000 }).trim() : release() },
    sites: [], events: [] };
  let descriptor, native, current, activeAbort, interrupted = false;
  const save = () => {
    if (descriptor === undefined) return;
    const bytes = Buffer.from(`${JSON.stringify(report)}\n`);
    writeSync(descriptor, bytes, 0, bytes.length, 0); ftruncateSync(descriptor, bytes.length);
  };
  const emit = (name, data) => {
    const safe = safeEvidence(data);
    report.events.push({ name, site: current?.id ?? null, at: new Date().toISOString(), data: safe });
    if (name === 'browser') report.browser = safe;
    save();
  };
  const onSignal = () => {
    interrupted = true; report.completion = 'INTERRUPTED'; report.publicClaim = 'UNVERIFIED';
    report.fatal = 'Run interrupted; no public claim'; save();
    activeAbort?.abort(new Unverified('Run interrupted'));
    void native?.close().catch(error => { report.fatal = `Interrupted cleanup: ${redact(error.message ?? error)}`; save(); });
  };
  try {
    mkdirSync(dirname(output), { recursive: true }); descriptor = openSync(output, 'wx'); save();
    process.once('SIGINT', onSignal); process.once('SIGTERM', onSignal);
    const build = verifyBuild(false);
    report.build = build;
    report.harness = Object.fromEntries(['tools/m10-sites.mjs', 'tools/m10-browser.mjs', 'tools/m10-fixtures.mjs', 'tools/m9-browser.mjs']
      .map(name => [name, sha256(readFileSync(join(ROOT, name)))]));
    report.configSha256 = configPath ? sha256(readFileSync(resolve(configPath))) : null;
    native = await openExtension(build, emit);
    assert(!interrupted, 'Interrupted during browser startup');
    native.context.setDefaultTimeout(3000); native.context.setDefaultNavigationTimeout(20000);
    report.launcherFlags = ['--remote-debugging-port=0', '--user-data-dir=<fresh temporary profile, deleted on close>',
      '--no-first-run', '--no-default-browser-check', '--use-mock-keychain', '--disable-background-networking', '--disable-component-update', '--disable-sync'];
    report.launcherFlagsScope = 'Fixed native launcher arguments from hashed m9-browser; browser.flags records the actual helper arguments. No focus emulation or synthetic host privilege.';
    for (const id of config.sites) {
      if (interrupted) throw new Unverified('Run interrupted');
      const site = SITES[id];
      const item = current = { id, siteURL: site.url, kind: site.kind, started: new Date().toISOString(), verdict: 'RUNNING', supported: false,
        sourceCommit: build.provenance.sourceCommit, bundleSha256: build.provenance.bundleSha256, provenanceSha256: build.provenanceSha256,
        buildKind: build.provenance.buildKind, modelSha256: build.provenance.modelSha256,
        browser: report.browser, launcherFlags: report.launcherFlags, machine: report.machine,
        durationMs: config.durationMs, pageBudgetMs: config.durationMs + 90000,
        observedCodes: [], samples: [], scenarios: [], errors: { counts: {}, samples: [], omittedSamples: 0 },
        finalRuntimeSession: 'not measured', initialEligibility: 'not measured', importPath: 'not measured', features: 'not measured',
        fullscreen: 'not measured: optional native-control scenario omitted', manualVisualReview: report.manualVisualReview };
      report.sites.push(item); save();
      let page, tabId, stopErrors, owner, lastStatus, lastDOM, timedOut = false;
      const codes = new Set(); const controller = new AbortController();
      activeAbort = controller;
      const deadline = performance.now() + item.pageBudgetMs;
      const record = (name, data) => emit(name, data);
      const dom = async () => {
        lastDOM = safeEvidence(await bounded(page.evaluate(siteSnapshot, native.extensionId), 3000, 'Read public video geometry'));
        assert.equal(lastDOM.origin, new URL(site.url).origin, 'Public page left the allowlisted origin');
        return lastDOM;
      };
      const inspect = async () => {
        controller.signal.throwIfAborted();
        lastStatus = await native.inspect(tabId); codes.add(lastStatus.code); item.observedCodes = [...codes];
        return lastStatus;
      };
      const panelAction = async selector => {
        const panel = await native.popup(page);
        try {
          tabId ??= panel.tabId; assert.equal(panel.tabId, tabId, 'Action targets a different tab');
          const status = await panel.click(selector); codes.add(status.code); item.observedCodes = [...codes]; return status;
        } finally { await panel.dismiss(); }
      };
      const negative = async status => {
        item.rejection = { status: safeEvidence(status), dom: await dom(), noRescueAttempted: true };
        item.verdict = ['error', 'device-lost'].includes(status.code) ? 'FAIL' : 'NEGATIVE'; save();
        throw new Unverified(`Actual extension rejection: ${status.code}`);
      };
      const check = async (label, allowSuspended = false) => {
        const status = await inspect();
        if (NEGATIVE.has(status.code)) await negative(status);
        const view = await dom(); const video = geometryCheck(status, view, owner);
        assert.equal(view.canvases[0].videoIndex, 0, 'Owner is not the first original video');
        if (!allowSuspended && !video.ended) assert(status.details.attachment.active && !status.details.attachment.suspendedReason, 'Pipeline unexpectedly suspended');
        const evidence = { label, at: new Date().toISOString(), status: safeEvidence(status), dom: view };
        item.samples.push(evidence); save(); return { status, dom: view, video };
      };
      const captureRuntime = async key => {
        const value = await runtimeEvidence(native, page);
        if (!value) throw new Unverified('Exact production RuntimeSession unavailable');
        assert.equal(value.owner, owner, 'Session belongs to a different owner');
        item[key] = value.runtime.session;
        item[`${key}Context`] = { owner: value.owner, controller: value.runtime.controller, importPath: value.importPath,
          upscalerId: value.upscalerId, source: value.source, target: value.target, features: value.features, capabilities: value.capabilities,
          collector: 'Read-only production isolated world; attachment.driver.snapshot().session, not PipelineStats rolling timing windows' };
        item.importPath = value.importPath; save(); return value.runtime.session;
      };
      const work = async () => {
        page = await bounded(native.context.newPage(), 5000, 'Create public site tab');
        stopErrors = errorObserver(page, native.extensionId, item);
        const response = await page.goto(site.url, { waitUntil: 'domcontentloaded', timeout: 20000 });
        item.documentHTTPStatus = response?.status() ?? 'not measured';
        assert(response?.ok(), 'Public document did not load successfully');
        await bounded(page.bringToFront(), 3000, 'Focus public page');
        item.initialWindow = await nativeResize(native, page, 1200);
        await consent(page, record);
        if (site.assetTitle) {
          const card = page.locator('.asset-card').filter({ has: page.getByRole('heading', { name: site.assetTitle, exact: true }) }).first();
          await card.getByRole('button', { name: /^Play$/i }).click({ timeout: 15000 });
          record('public-asset-selection', { title: site.assetTitle, method: 'Trusted click on existing public asset-card Play control' });
        }
        let metadataPlayClicked = false;
        await until(async () => {
          await consent(page, record);
          const ready = await page.evaluate(() => {
            const video = document.querySelector('video'); if (!video) return null;
            return { loaded: video.readyState >= 2 && video.videoWidth > 0, paused: video.paused, mediaKeys: !!video.mediaKeys, failed: !!video.error };
          });
          if (ready?.mediaKeys || ready?.failed || ready?.loaded) return true;
          if (!metadataPlayClicked) {
            const player = page.locator(site.player).filter({ has: page.locator('video') }).first();
            const play = await visibleButton(player, playerCommand('play'));
            if (play) { await play.click({ timeout: 3000 }); metadataPlayClicked = true; }
          }
          return false;
        }, Boolean, 45000, controller.signal);
        try {
          await until(async () => {
            const player = page.locator(site.player).filter({ has: page.locator('video') }).first();
            return !!(await visibleButton(player, playerCommand('play')) || await visibleButton(player, playerCommand('pause')));
          }, Boolean, 5000, controller.signal);
          item.customPlayerSettled = true;
        } catch (error) {
          if (controller.signal.aborted || error instanceof OperationTimeout) throw error;
          item.customPlayerSettled = false;
        }
        item.beforeEnable = await dom();
        assert(item.beforeEnable.videos.length > 0, 'No top-document video');
        assert.equal(item.beforeEnable.canvases.length, 0, 'Enhancement present before action grant');
        if (!item.beforeEnable.videos[0].mediaKeys && !item.beforeEnable.videos[0].error && item.beforeEnable.videos[0].paused) {
          if (item.customPlayerSettled) {
            await playerButton(page, site, 'play');
            await until(() => page.locator('video').first().evaluate(video => !video.paused && video.readyState >= 2), Boolean, 5000, controller.signal);
          } else item.playbackPreparation = 'No settled original custom control; inspect actual extension eligibility without mutating native controls';
        }
        item.initialEligibility = { before: await dom(), statusAtEnable: safeEvidence(await panelAction('#enable')),
          transport: 'Extensions.triggerAction -> native popup Input.dispatchMouseEvent #enable; m10.inspect sent by actual service worker to registered documentId/frameId 0' };
        item.tabId = tabId;
        if (NEGATIVE.has(item.initialEligibility.statusAtEnable.code)) await negative(item.initialEligibility.statusAtEnable);
        item.registration = await native.workerEval(async tab => (await chrome.storage.session.get(`m10.document.${tab}`))[`m10.document.${tab}`], tabId);
        assert.equal(item.registration?.tabId, tabId); assert.equal(item.registration?.origin, new URL(site.url).origin);
        assert.equal(typeof item.registration?.documentId, 'string', 'No actual popup document registration');
        let previous;
        await until(async () => {
          const status = await inspect(); item.initialEligibility.latestStatus = safeEvidence(status); save();
          if (NEGATIVE.has(status.code)) return true;
          const attachment = status.details?.attachment;
          const stable = status.code === 'active' && status.current === 'neural' && attachment?.controllerState === 'stable' &&
            attachment.active && !attachment.suspendedReason && attachment.session.framesRendered > 30;
          const progressing = stable && previous?.owner === status.owner && previous.frames < attachment.session.framesRendered;
          previous = stable ? { owner: status.owner, frames: attachment.session.framesRendered } : null;
          return progressing;
        }, Boolean, 20000, controller.signal);
        if (NEGATIVE.has(lastStatus.code)) await negative(lastStatus);
        assert.equal(lastStatus.mode, 'auto', 'Unexpected non-default public mode');
        owner = lastStatus.owner; assert(owner);
        item.firstSupported = { status: safeEvidence(lastStatus), dom: await dom() };
        geometryCheck(lastStatus, item.firstSupported.dom, owner);
        item.supported = true; item.features = lastStatus.details.attachment.features;
        const first = await captureRuntime('initialRuntimeSession');
        const started = performance.now(); let lastFrameAt = started, frames = first.framesRendered, actionIndex = 0, scrollBack;
        const originalPaused = paused => until(() => page.locator('video').first().evaluate(video => video.paused), value => value === paused, 4000, controller.signal);
        const resumeProgress = async () => {
          const before = (await inspect()).details?.attachment?.session.framesRendered ?? 0;
          await until(async () => {
            const status = await inspect(); if (NEGATIVE.has(status.code)) await negative(status);
            return status.owner === owner && status.details?.attachment?.active && status.details.attachment.session.framesRendered > before + 2;
          }, Boolean, 6000, controller.signal);
        };
        const actions = [
          { at: 30000, name: 'relative-seek', run: async () => {
            const result = await page.locator('video').first().evaluate(video => {
              if (!Number.isFinite(video.duration) || video.duration <= 2 || !video.seekable.length) throw new Error('Original video has no safe finite seek range');
              const from = video.currentTime; let target = null;
              for (let index = 0; index < video.seekable.length; index++) {
                const lower = video.seekable.start(index) + 0.25, upper = Math.min(video.seekable.end(index), video.duration) - 1;
                if (upper <= lower || from < lower || from > video.seekable.end(index)) continue;
                const forward = Math.min(from + 5, upper), backward = Math.max(lower, from - 5);
                const candidate = forward > from + 0.25 ? forward : backward;
                if (Math.abs(candidate - from) > 0.25) { target = candidate; break; }
              }
              if (target === null) throw new Error('No safe relative seek available');
              video.currentTime = target; return { method: 'Original HTMLVideoElement.currentTime, bounded by seekable range', from, target };
            });
            result.observed = await until(() => page.locator('video').first().evaluate(video => ({ seeking: video.seeking, time: video.currentTime })),
              value => !value.seeking && Math.abs(value.time - result.target) < 2, 5000, controller.signal);
            return result;
          } },
          { at: 50000, name: 'original-pause-resume', run: async () => {
            await playerButton(page, site, 'pause'); await originalPaused(true);
            await until(async () => {
              const status = await inspect(); if (NEGATIVE.has(status.code)) await negative(status);
              return !status.details?.attachment?.active && status.details?.attachment?.suspendedReason === 'video-paused';
            }, Boolean, 4000, controller.signal);
            const paused = await check('original-paused', true);
            assert.equal(paused.status.details.attachment.active, false, 'Pipeline ignored original pause');
            const pausedFrames = paused.status.details.attachment.session.framesRendered;
            await delay(1500, undefined, { signal: controller.signal });
            const held = await inspect(); assert.equal(held.details.attachment.session.framesRendered, pausedFrames, 'Frames advanced during settled original pause');
            await playerButton(page, site, 'play'); await originalPaused(false);
            return { method: 'Visible original Pause and Play buttons', pausedFrames, holdMs: 1500 };
          } },
          { at: 75000, name: 'scroll-down', suspended: true, run: async () => {
            scrollBack = await page.evaluate(() => ({ x: scrollX, y: scrollY }));
            const result = await page.evaluate(() => { const from = scrollY; scrollBy({ top: innerHeight * 0.3, behavior: 'instant' }); return { from, requestedDelta: innerHeight * 0.3, to: scrollY }; });
            if (Math.abs(result.to - result.from) < 1) {
              scrollBack.notApplicable = 'Document is not scrollable in this player view; no forced layout adjustment';
              return { ...result, notApplicable: scrollBack.notApplicable };
            }
            return result;
          } },
          { at: 80000, name: 'scroll-back', run: async () => {
            if (scrollBack.notApplicable) return { notApplicable: scrollBack.notApplicable };
            const result = await page.evaluate(position => { scrollTo({ left: position.x, top: position.y, behavior: 'instant' }); return { x: scrollX, y: scrollY }; }, scrollBack);
            assert(Math.abs(result.x - scrollBack.x) < 1 && Math.abs(result.y - scrollBack.y) < 1, 'Original scroll position not restored');
            return result;
          } },
          { at: 100000, name: 'resize-1024', run: () => nativeResize(native, page, 1024) },
          { at: 115000, name: 'resize-1200', run: () => nativeResize(native, page, 1200) },
          { at: 145000, name: 'original-fullscreen', run: async () => {
            const player = page.locator(site.player).filter({ has: page.locator('video') }).first();
            await player.hover({ timeout: 3000 });
            const button = await visibleButton(player, /^(Enter )?Full\s*screen(?: mode)?$/i);
            if (!button) { item.fullscreen = 'not measured: no visible original fullscreen button'; return { attempted: false, reason: item.fullscreen }; }
            await button.click({ timeout: 3000 });
            try {
              await page.waitForFunction(() => document.fullscreenElement !== null, undefined, { timeout: 4000 });
              const direct = await page.evaluate(() => document.fullscreenElement instanceof HTMLVideoElement);
              await until(async () => {
                const state = await inspect();
                return direct ? state.details?.attachment?.suspendedReason === 'video-fullscreen' : state.details?.attachment?.active && state.details.attachment.ready;
              }, Boolean, 4000, controller.signal);
              const result = await check('original-fullscreen', direct);
              if (!direct) assert(await page.evaluate(() => document.fullscreenElement.contains(document.querySelector('canvas[data-aethervsr-m10]'))));
              item.fullscreen = direct ? 'PASS: direct-video fullscreen shows original' : 'PASS: original container fullscreen retains enhancement';
              return { attempted: true, direct, owner: result.status.owner, fullscreen: result.dom.fullscreen };
            } finally {
              await page.evaluate(async () => { if (document.fullscreenElement) await document.exitFullscreen(); });
              await page.bringToFront();
            }
          } },
        ];
        while (performance.now() - started < config.durationMs) {
          controller.signal.throwIfAborted();
          const sample = await check('observation', actionIndex === 3);
          if (sample.video.ended) {
            const replay = { name: 'original-ended-replay', startedMs: performance.now() - started, verdict: 'RUNNING' };
            item.scenarios.push(replay); save();
            await playerButton(page, site, 'play'); await originalPaused(false); await resumeProgress();
            const after = await check('original-ended-replay-after', actionIndex === 3);
            replay.owner = after.status.owner; replay.current = after.status.current; replay.geometry = after.dom.canvases;
            replay.finishedMs = performance.now() - started; replay.verdict = 'PASS';
          }
          const elapsed = performance.now() - started;
          const action = actions[actionIndex];
          if (action && elapsed >= action.at) {
            const scenario = { name: action.name, scheduledMs: action.at, startedMs: elapsed, verdict: 'RUNNING' };
            item.scenarios.push(scenario); save();
            await check(`${action.name}-before`, action.name === 'scroll-back');
            scenario.result = safeEvidence(await action.run());
            if (!action.suspended) await resumeProgress();
            else await delay(250, undefined, { signal: controller.signal });
            const after = await check(`${action.name}-after`, action.suspended);
            scenario.owner = after.status.owner; scenario.current = after.status.current; scenario.geometry = after.dom.canvases;
            scenario.finishedMs = performance.now() - started; scenario.verdict = scenario.result?.notApplicable ? 'NOT_APPLICABLE' : 'PASS'; actionIndex++;
          }
          const count = lastStatus.details.attachment.session.framesRendered;
          if (count > frames) { frames = count; lastFrameAt = performance.now(); }
          assert(performance.now() - lastFrameAt < 15000, 'No successful pipeline frame progress for 15 seconds; no automatic rescue');
          item.observedMs = performance.now() - started;
          await delay(Math.min(1000, Math.max(0, config.durationMs - item.observedMs)), undefined, { signal: controller.signal });
        }
        const final = await check('final'); const finalSession = await captureRuntime('finalRuntimeSession');
        item.observedMs = performance.now() - started;
        assert(finalSession.framesRendered > first.framesRendered + 30, 'Only initial frame success, no sustained frame progress');
        item.finalStatus = safeEvidence(final.status);
        item.verdict = config.durationMs >= 180000 && actionIndex === actions.length ? 'PASS' : 'UNVERIFIED';
        if (item.verdict === 'UNVERIFIED') item.reason = 'Short diagnostic run cannot establish the required three-minute public success';
      };
      const task = Promise.resolve().then(work);
      try { await bounded(task, Math.max(1, deadline - performance.now()), 'Public page watchdog'); }
      catch (error) {
        if (error instanceof OperationTimeout) {
          timedOut = true; controller.abort(error); report.fatal = `Public page watchdog/operation timeout: ${id}; remaining sites not run`;
          await native.close();
          await bounded(task.catch(() => {}), 5000, 'Drain public page after native browser cleanup').catch(() => {});
        }
        if (item.verdict === 'RUNNING' || timedOut || interrupted) item.verdict = error instanceof Unverified || interrupted ? 'UNVERIFIED' : 'FAIL';
        item.error = redact(error.message ?? error);
        item.lastStatus = safeEvidence(lastStatus ?? null); item.lastDOM = lastDOM ?? null;
        for (const scenario of item.scenarios.filter(value => value.verdict === 'RUNNING')) { scenario.verdict = 'FAIL'; scenario.error = item.error; }
      } finally {
        const cleanup = async () => {
          if (!timedOut && !interrupted && page && !page.isClosed()) {
            if (owner && item.finalRuntimeSession === 'not measured' && lastStatus?.details?.attachment) {
              try { await captureRuntime('finalRuntimeSession'); }
              catch (error) { item.runtimeCaptureError = redact(error.message ?? error); if (item.verdict === 'PASS') item.verdict = 'UNVERIFIED'; }
            }
            if (tabId !== undefined) {
              item.disabled = safeEvidence(await panelAction('#disable'));
              assert.equal(item.disabled.code, 'inactive'); assert.equal(item.disabled.enabled, false); assert.equal(item.disabled.owner, null);
              const details = item.disabled.details; assert(details, 'Missing teardown diagnostics');
              assert.equal(details.infrastructure.created, details.infrastructure.destroyed); assert.equal(details.timerCount, 0); assert.equal(details.discoveryActive, false);
              if (details.lastTeardown) assert(Object.values(details.lastTeardown).every(count => count === 0), 'Live resource after disable');
              item.afterDisable = await dom(); assert.equal(item.afterDisable.canvases.length, 0, 'Owned output remains after disable');
            }
          }
          if (page && !page.isClosed()) await bounded(page.close(), 3000, 'Close public site tab');
        };
        const cleanupTask = cleanup();
        try { await bounded(cleanupTask, Math.max(1, deadline - performance.now()), 'Public page teardown deadline'); }
        catch (error) {
          controller.abort(error); item.verdict = 'FAIL'; item.cleanupError = redact(error.message ?? error);
          report.fatal = `Public tab teardown failed: ${id}; remaining sites not run`;
          await native.close(); await bounded(cleanupTask.catch(() => {}), 5000, 'Drain public teardown').catch(() => {});
        }
        stopErrors?.(); activeAbort = undefined;
        item.errors.corsCorrelated = !item.errors.counts['cors-import'] || codes.has('cors-blocked');
        item.cors = { blocked: codes.has('cors-blocked'), observedImportPath: item.importPath,
          exactImportErrors: item.errors.counts['cors-import'] ?? 0, alternativeImportRequested: false,
          scope: 'Actual production route and status only; document HTTP success does not establish media origin-clean eligibility' };
        if (item.errors.counts.extension) { item.verdict = 'FAIL'; item.errorVerdict = 'Explicit extension errors'; }
        else if (item.errors.counts.unattributed || item.errors.counts['gpu-unattributed'] || !item.errors.corsCorrelated) {
          if (item.verdict === 'PASS') item.verdict = 'UNVERIFIED'; item.errorVerdict = 'Unattributed errors require review; not suppressed';
        } else item.errorVerdict = 'Only origin-attributed third-party errors or exact CORS import errors correlated with actual cors-blocked status';
        item.finished = new Date().toISOString(); save();
      }
      console.log(`${item.verdict} ${id}; successful public sites so far: ${report.sites.filter(site => site.verdict === 'PASS').length}`);
      if (report.fatal || interrupted) throw new Unverified(report.fatal ?? 'Run interrupted');
    }
    report.completion = 'COMPLETE';
  } catch (error) {
    report.fatal ??= redact(error.message ?? error);
    report.completion = interrupted ? 'INTERRUPTED' : 'ABORTED';
  } finally {
    try { await native?.close(); }
    catch (error) { report.fatal = `Browser cleanup failed: ${redact(error.message ?? error)}`; report.completion = 'ABORTED'; }
    try {
      const after = verifyBuild(false);
      if (report.build) {
        assert.equal(after.provenanceSha256, report.build.provenanceSha256, 'Build changed during public integration');
        assert.deepEqual(after.provenance, report.build.provenance, 'Bundle/source changed during public integration');
        report.buildAfter = after; report.sameCleanBuild = true;
      }
    } catch (error) { report.sameCleanBuild = false; report.buildError = redact(error.message ?? error); }
    report.successfulSites = report.sites.filter(item => item.verdict === 'PASS').length;
    report.publicClaim = !report.fatal && report.sameCleanBuild && report.completion === 'COMPLETE'
      ? report.successfulSites >= 2 ? 'READY' : report.successfulSites === 1 ? 'PARTIAL' : 'UNVERIFIED' : 'UNVERIFIED';
    report.finished = new Date().toISOString(); save();
    process.off('SIGINT', onSignal); process.off('SIGTERM', onSignal);
    if (descriptor !== undefined) { closeSync(descriptor); descriptor = undefined; }
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.length === 1 && args[0] === '--help') {
    console.log('Usage: node tools/m10-sites.mjs --run-public --output .cache/m10/sites-RUN.json [--config .cache/m10/sites-config.json]\nConfig: {"durationMs":180000,"sites":["plyr","videojs"]}. Root must serialize this native GPU run with performance. No screenshots, build, or commit.');
  } else {
    try {
      assert(args.includes('--run-public'), 'Public GPU execution requires explicit --run-public after root serialization');
      const options = {};
      for (let index = 0; index < args.length; index++) {
        if (args[index] === '--run-public') continue;
        assert(['--output', '--config'].includes(args[index]) && args[index + 1] && !args[index + 1].startsWith('--'), 'Invalid CLI options; use --help');
        assert(!Object.hasOwn(options, args[index]), 'Duplicate CLI option'); options[args[index]] = args[++index];
      }
      assert(options['--output'], '--output is required; existing evidence is never overwritten');
      const report = await runSites(options['--output'], options['--config']);
      console.log(`${report.publicClaim}: ${report.successfulSites} successful public sites; ${relative(ROOT, resolve(options['--output']))}`);
      process.exitCode = report.publicClaim === 'READY' ? 0 : 2;
    } catch (error) { console.error(redact(error.message ?? error)); process.exitCode = 2; }
  }
}