import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import { startFixtures, sha256 } from './m10-fixtures.mjs';
import { openNativeChrome } from './m9-browser.mjs';
import { installVideoObserver, installRuntimeRecorder, validateCapture, summarizeCapture, distribution, counterDelta } from './m10-performance.mjs';

export function installScheduling(options = {}) {
  const data = globalThis[Symbol.for('aethervsr.m105.scheduling')] = { raf: [], tasks: [], events: [], stimuli: [], overflow: false };
  let active = false, handle, started, previous;
  let complete;
  data.done = new Promise(done => { complete = done; });
  const timers = [], listeners = [], observers = [];
  const push = (rows, row) => { if (rows.length < 160000) rows.push(row); else data.overflow = true; };
  const screenState = () => ({ width: screen.width, height: screen.height, availWidth: screen.availWidth,
    availHeight: screen.availHeight, availLeft: screen.availLeft, availTop: screen.availTop,
    screenX, screenY, outerWidth, outerHeight, innerWidth, innerHeight, devicePixelRatio,
    colorDepth: screen.colorDepth, isExtended: screen.isExtended ?? null, physicalRefreshHz: null });
  const raf = now => { if (active) { push(data.raf, [performance.now(), now, previous === undefined ? null : now - previous]); previous = now; } handle = requestAnimationFrame(raf); };
  handle = requestAnimationFrame(raf);
  data.supportedEntryTypes = PerformanceObserver.supportedEntryTypes;
  const collect = entries => {
    for (const entry of entries) if (started !== undefined && entry.startTime >= started && (data.ended === undefined || entry.startTime < data.ended)) {
      push(data.tasks, { at: entry.startTime, ms: entry.duration, name: entry.name,
        attribution: entry.attribution?.map(value => ({ name: value.name, containerType: value.containerType })) ?? [] });
    }
  };
  if (data.supportedEntryTypes.includes('longtask')) {
    const observer = new PerformanceObserver(list => collect(list.getEntries()));
    observer.observe({ type: 'longtask', buffered: false }); observers.push(observer);
  }
  const on = (target, type, callback) => { target.addEventListener(type, callback); listeners.push([target, type, callback]); };
  on(document, 'DOMContentLoaded', () => {
    const video = document.querySelector('video');
    for (const type of ['waiting', 'stalled', 'seeking', 'seeked', 'emptied', 'loadeddata', 'resize', 'ended']) on(video, type, () => {
      if (active) push(data.events, { at: performance.now(), type, mediaTime: video.currentTime, readyState: video.readyState, quality: video.getVideoPlaybackQuality().toJSON?.() ?? { total: video.getVideoPlaybackQuality().totalVideoFrames, dropped: video.getVideoPlaybackQuality().droppedVideoFrames } });
    });
  });
  on(window, 'aethervsr:m10:performance:start', () => {
    active = true; started = performance.now(); data.started = started; data.opening = screenState();
    for (const stimulus of options.stalls ?? []) timers.push(setTimeout(() => {
      const before = performance.now();
      while (performance.now() - before < stimulus.ms) {}
      push(data.stimuli, { requestedAtMs: stimulus.at, requestedMs: stimulus.ms, at: before, ended: performance.now() });
    }, stimulus.at));
  });
  on(window, 'aethervsr:m10:performance:end', () => { data.ended = performance.now(); active = false; data.closing = screenState(); for (const timer of timers) clearTimeout(timer); cancelAnimationFrame(handle); });
  on(window, 'aethervsr:m10:performance:complete', () => {
    for (const [target, type, callback] of listeners) target.removeEventListener(type, callback);
    setTimeout(() => { for (const observer of observers) { collect(observer.takeRecords()); observer.disconnect(); } complete(); }, 0);
  });
}

export function installOwnedCost(options = {}) {
  const key = Symbol.for('aethervsr.m105.owned');
  if (globalThis[key]) throw new Error('Duplicate diagnostic wrapper');
  const data = globalThis[key] = { active: false, totalMs: 0, callbacks: 0, categories: {}, depth: 0,
    scope: 'Synchronous isolated-world entry callbacks and message calls; outermost total includes shared core JS but excludes GPU work, browser internals, unwrapped async continuations and instrumentation bookkeeping. Category times may nest; never sum them.' };
  const restores = [], cache = new WeakMap();
  const wrap = (name, callback) => {
    if (typeof callback !== 'function' && (typeof callback !== 'object' || callback === null)) return callback;
    let names = cache.get(callback);
    if (!names) { names = new Map(); cache.set(callback, names); }
    if (names.has(name)) return names.get(name);
    const wrapped = function(...args) {
      const call = () => typeof callback === 'function' ? Reflect.apply(callback, this, args) : Reflect.apply(callback.handleEvent, callback, args);
      if (!data.active) return call();
      const before = performance.now(); data.depth++;
      try { return call(); }
      finally {
        const elapsed = performance.now() - before; data.depth--;
        const bucket = data.categories[name] ??= { calls: 0, totalMs: 0, maxMs: 0 };
        bucket.calls++; bucket.totalMs += elapsed; bucket.maxMs = Math.max(bucket.maxMs, elapsed);
        if (data.depth === 0) { data.totalMs += elapsed; data.callbacks++; }
      }
    };
    names.set(name, wrapped); return wrapped;
  };
  const replace = (owner, name, method) => { const original = owner[name]; owner[name] = method(original); restores.push(() => { owner[name] = original; }); };
  const add = EventTarget.prototype.addEventListener, remove = EventTarget.prototype.removeEventListener;
  replace(EventTarget.prototype, 'addEventListener', original => function(type, callback, settings) { return original.call(this, type, wrap(`event:${type}`, callback), settings); });
  replace(EventTarget.prototype, 'removeEventListener', original => function(type, callback, settings) { return original.call(this, type, cache.get(callback)?.get(`event:${type}`) ?? callback, settings); });
  for (const name of ['MutationObserver', 'ResizeObserver']) {
    if (typeof globalThis[name] === 'function') replace(globalThis, name, Original => class extends Original { constructor(callback) { super(wrap(name, callback)); } });
  }
  for (const name of ['setTimeout', 'setInterval']) replace(globalThis, name, original => function(callback, delay, ...args) { return original(wrap(`${name}:${delay ?? 0}`, callback), delay, ...args); });
  replace(HTMLVideoElement.prototype, 'requestVideoFrameCallback', original => function(callback) { return original.call(this, wrap('rVFC', callback)); });
  const message = chrome.runtime.onMessage;
  replace(message, 'addListener', original => function(callback) { return original.call(this, wrap('runtime:onMessage', callback)); });
  replace(message, 'removeListener', original => function(callback) { return original.call(this, cache.get(callback)?.get('runtime:onMessage') ?? callback); });
  replace(chrome.runtime, 'sendMessage', original => function(...args) { return wrap('runtime:sendMessage', original).apply(this, args); });
  if (options.noRuntime) replace(navigator.gpu, 'requestAdapter', () => async () => null);
  const start = () => { data.active = true; data.started = performance.now(); };
  const end = () => { data.ended = performance.now(); data.active = false; };
  add.call(window, 'aethervsr:m10:performance:start', start);
  add.call(window, 'aethervsr:m10:performance:end', end);
  data.restore = () => {
    if (data.active) throw new Error('Cannot restore during recording');
    for (const restore of restores.reverse()) restore();
    remove.call(window, 'aethervsr:m10:performance:start', start); remove.call(window, 'aethervsr:m10:performance:end', end);
  };
  return { installed: true, noRuntime: !!options.noRuntime };
}

export function deliverySummary(raw) {
  const summary = summarizeCapture(raw), video = raw.video, frames = raw.runtime.frames;
  const callbacks = counterDelta(video.opening.callbacks, video.closing.callbacks);
  const presented = counterDelta(video.opening.presented, video.closing.presented);
  const gaps = video.rows.reduce((sum, row) => sum + Math.max(0, (row[4] ?? 1) - 1), 0);
  const available = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
  const percent = (value, total) => !available(value) || !available(total) || total <= 0 ? null : 100 * value / total;
  const attempts = raw.runtime.attempts?.length ?? null;
  const intervals = video.rows.slice(1).map((row, index) => {
    const previous = video.rows[index];
    return row[2] >= previous[2] && row[11] === previous[11] ? (row[2] - previous[2]) * 1000 : null;
  });
  return { decoderDropPercent: percent(summary.decoderDrops, summary.decoderFrames),
    qualityLabel: 'Playback-quality reported drops, not decoder-only or unique display loss',
    callbackGapPercent: percent(gaps, presented), callbacks, presented, gaps,
    callbackBoundaryResidual: presented === null ? null : presented - callbacks - gaps,
    submitted: summary.rendered, submissionAttempts: attempts,
    submissionDeficit: attempts === null || summary.rendered === null ? null : attempts - summary.rendered,
    renderedVsPresentedDeficitPercent: summary.pipelineLossPercent,
    m10CombinedPercent: summary.rendered === null || !available(summary.skipped) || !available(summary.decoderDrops) ? null : percent(summary.skipped + summary.decoderDrops, summary.presented),
    callbackIntervalsMs: distribution(video.rows.slice(1).map((row, index) => row[0] - video.rows[index][0])),
    mediaIntervalsMs: distribution(intervals),
    mediaIntervalExclusions: { candidates: intervals.length, backwardOrGenerationBoundary: intervals.filter(value => value === null).length,
      scope: 'Callback-sampled PTS differences, not every decoded/displayed frame; loop rewinds and load-generation boundaries excluded explicitly' },
    exactUniqueLostFrames: null, exactOverlap: null,
    identityScope: 'PTS, source generation and callback/submission sequence identify observations, not a shared native frame ID; counters have separate boundaries and asynchronous publication',
    runtimeSamples: frames.length };
}

export async function nativeWindow(page, context) {
  const cdp = await context.newCDPSession(page);
  try {
    await cdp.send('Emulation.clearDeviceMetricsOverride');
    const { windowId } = await cdp.send('Browser.getWindowForTarget');
    await cdp.send('Browser.setWindowBounds', { windowId, bounds: { left: 40, top: 40, width: 1280, height: 900, windowState: 'normal' } });
    const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }));
    await cdp.send('Browser.setWindowBounds', { windowId, bounds: { left: 40, top: 40, width: 1280 + 1200 - viewport.width, height: 900 + 820 - viewport.height } });
    await page.waitForFunction(() => innerWidth === 1200 && innerHeight === 820, undefined, { timeout: 5000 });
    return { bounds: await cdp.send('Browser.getWindowBounds', { windowId }), deviceMetricsOverride: false,
      scope: 'Native window at desktop (40,40), actual 1200x820 content; no forced focus during capture; Screen API is observable geometry, not measured physical refresh' };
  } finally { await cdp.detach(); }
}

export function validateScheduling(data, stalls = []) {
  assert.equal(data.overflow, false, 'Scheduling buffer overflow');
  assert.equal(data.stimuli.length, stalls.length, 'Missing diagnostic stimulus');
  for (let index = 0; index < stalls.length; index++) {
    const row = data.stimuli[index];
    assert.equal(row.requestedAtMs, stalls[index].at); assert.equal(row.requestedMs, stalls[index].ms);
    assert(row.at >= data.started && row.ended <= data.ended && row.ended - row.at >= stalls[index].ms, 'Stimulus outside observation window or shorter than requested');
  }
  assert.deepEqual(data.opening, data.closing, 'Display/window geometry changed during observation');
}

export async function runCounterProbe(prefix) {
  assert(!existsSync(`${prefix}.json`), 'Never overwrite evidence');
  mkdirSync(dirname(prefix), { recursive: true });
  const report = { source: execFileSync('git', ['rev-parse', 'HEAD']).toString().trim(),
    scope: 'Native counter/stall experiment, not final acceptance; 12s windows after 5s warmup, control then 50/100/200ms task stalls at 3/6/9s; fixed bounds, default GPU security',
    scriptSha256: sha256(readFileSync(fileURLToPath(import.meta.url))), results: [] };
  const fixtures = await startFixtures({ mse: false });
  report.media = fixtures.evidence;
  let native;
  try {
    for (const kind of ['control', 'stalls']) {
      const row = { kind, verdict: 'UNVERIFIED', errors: [] }; report.results.push(row);
      native = await openNativeChrome(['--autoplay-policy=no-user-gesture-required', '--window-position=40,40', '--window-size=1280,900']);
      const page = await native.context.newPage();
      row.window = await nativeWindow(page, native.context);
      const cdp = await native.context.newCDPSession(page);
      row.browser = await cdp.send('Browser.getVersion');
      await cdp.detach();
      page.on('pageerror', error => row.errors.push(String(error)));
      await page.addInitScript(installVideoObserver, { accounting: true });
      const stalls = kind === 'stalls' ? [{ at: 3000, ms: 50 }, { at: 6000, ms: 100 }, { at: 9000, ms: 200 }] : [];
      await page.addInitScript(installScheduling, { stalls });
      await page.goto(`${fixtures.url}?case=custom`); await page.bringToFront();
      await page.waitForFunction(() => globalThis[Symbol.for('aethervsr.m10.performance.video')]?.callbacks >= 2);
      const item = { kind: 'no-extension', durationMs: 12000, warmupMs: 5000, accounting: true };
      await page.evaluate(installRuntimeRecorder, item);
      await page.evaluate(() => globalThis[Symbol.for('aethervsr.m10.performance.video')].done);
      await page.evaluate(() => globalThis[Symbol.for('aethervsr.m105.scheduling')].done);
      const raw = await page.evaluate(() => {
        const video = globalThis[Symbol.for('aethervsr.m10.performance.video')];
        return { runtime: globalThis[Symbol.for('aethervsr.m10.performance.runtime')],
          video: { opening: video.opening, closing: video.closing, rows: video.rows, events: video.events, overflow: video.overflow },
          scheduling: globalThis[Symbol.for('aethervsr.m105.scheduling')] };
      });
      const bytes = gzipSync(JSON.stringify(raw));
      row.raw = { path: `${prefix}.${kind}.json.gz`, sha256: sha256(bytes), bytes: bytes.length };
      writeFileSync(row.raw.path, bytes, { flag: 'wx' });
      try { validateCapture(raw, item); validateScheduling(raw.scheduling, stalls); assert.deepEqual(row.errors, []); row.verdict = 'VALID'; }
      catch (error) { row.error = String(error); }
      row.accounting = deliverySummary(raw); row.scheduling = { opening: raw.scheduling.opening, closing: raw.scheduling.closing, rafIntervalsMs: distribution(raw.scheduling.raf.map(value => value[2])), tasks: raw.scheduling.tasks, stimuli: raw.scheduling.stimuli };
      console.log(JSON.stringify({ kind, verdict: row.verdict, accounting: row.accounting, scheduling: row.scheduling }));
      await native.close(); native = null;
    }
  } finally { await native?.close(); await fixtures.close(); writeFileSync(`${prefix}.json`, JSON.stringify(report, null, 2), { flag: 'wx' }); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert(process.argv.length === 3, 'Usage: node tools/m105-accounting.mjs .cache/m105/probe-N');
  await runCounterProbe(resolve(process.argv[2]));
}