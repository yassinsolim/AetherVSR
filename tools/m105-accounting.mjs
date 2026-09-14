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
  const timers = [], listeners = [], observers = [];
  const push = (rows, row) => { if (rows.length < 160000) rows.push(row); else data.overflow = true; };
  const screenState = () => ({ width: screen.width, height: screen.height, availWidth: screen.availWidth,
    availHeight: screen.availHeight, availLeft: screen.availLeft, availTop: screen.availTop,
    screenX, screenY, outerWidth, outerHeight, innerWidth, innerHeight, devicePixelRatio,
    colorDepth: screen.colorDepth, isExtended: screen.isExtended ?? null, physicalRefreshHz: null });
  const raf = now => { if (active) { push(data.raf, [performance.now(), now, previous === undefined ? null : now - previous]); previous = now; } handle = requestAnimationFrame(raf); };
  handle = requestAnimationFrame(raf);
  data.supportedEntryTypes = PerformanceObserver.supportedEntryTypes;
  if (data.supportedEntryTypes.includes('longtask')) {
    const observer = new PerformanceObserver(list => { for (const entry of list.getEntries()) if (started !== undefined && entry.startTime >= started) push(data.tasks, { at: entry.startTime, ms: entry.duration, name: entry.name, attribution: entry.attribution?.map(value => ({ name: value.name, containerType: value.containerType })) ?? [] }); });
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
    for (const observer of observers) observer.disconnect();
    for (const [target, type, callback] of listeners) target.removeEventListener(type, callback);
  });
}

export function deliverySummary(raw) {
  const summary = summarizeCapture(raw), video = raw.video, frames = raw.runtime.frames;
  const callbacks = counterDelta(video.opening.callbacks, video.closing.callbacks);
  const presented = counterDelta(video.opening.presented, video.closing.presented);
  const gaps = video.rows.reduce((sum, row) => sum + Math.max(0, (row[4] ?? 1) - 1), 0);
  const percent = (value, total) => value === null || !(total > 0) ? null : 100 * value / total;
  const attempts = raw.runtime.attempts?.length ?? null;
  return { decoderDropPercent: percent(summary.decoderDrops, summary.decoderFrames),
    qualityLabel: 'Playback-quality reported drops, not decoder-only or unique display loss',
    callbackGapPercent: percent(gaps, presented), callbacks, presented, gaps,
    callbackBoundaryResidual: presented === null ? null : presented - callbacks - gaps,
    submitted: summary.rendered, submissionAttempts: attempts,
    submissionDeficit: attempts === null || summary.rendered === null ? null : attempts - summary.rendered,
    renderedVsPresentedDeficitPercent: summary.pipelineLossPercent,
    m10CombinedPercent: summary.rendered === null ? null : percent(summary.skipped + summary.decoderDrops, summary.presented),
    callbackIntervalsMs: distribution(video.rows.slice(1).map((row, index) => row[0] - video.rows[index][0])),
    mediaIntervalsMs: distribution(video.rows.slice(1).map((row, index) => {
      const previous = video.rows[index];
      return row[2] >= previous[2] && row[11] === previous[11] ? (row[2] - previous[2]) * 1000 : null;
    })),
    exactUniqueLostFrames: null, exactOverlap: null,
    identityScope: 'PTS, source generation and callback/submission sequence identify observations, not a shared native frame ID; counters have separate boundaries and asynchronous publication',
    runtimeSamples: frames.length };
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
      await page.setViewportSize({ width: 1200, height: 820 });
      const cdp = await native.context.newCDPSession(page);
      row.browser = await cdp.send('Browser.getVersion');
      const window = await cdp.send('Browser.getWindowForTarget');
      await cdp.send('Browser.setWindowBounds', { windowId: window.windowId, bounds: { left: 40, top: 40, width: 1280, height: 900, windowState: 'normal' } });
      row.window = await cdp.send('Browser.getWindowBounds', { windowId: window.windowId });
      page.on('pageerror', error => row.errors.push(String(error)));
      await page.addInitScript(installVideoObserver, { accounting: true });
      await page.addInitScript(installScheduling, { stalls: kind === 'stalls' ? [{ at: 3000, ms: 50 }, { at: 6000, ms: 100 }, { at: 9000, ms: 200 }] : [] });
      await page.goto(`${fixtures.url}?case=custom`); await page.bringToFront();
      await page.waitForFunction(() => globalThis[Symbol.for('aethervsr.m10.performance.video')]?.callbacks >= 2);
      const item = { kind: 'no-extension', durationMs: 12000, warmupMs: 5000, accounting: true };
      await page.evaluate(installRuntimeRecorder, item);
      await page.evaluate(() => globalThis[Symbol.for('aethervsr.m10.performance.video')].done);
      const raw = await page.evaluate(() => {
        const video = globalThis[Symbol.for('aethervsr.m10.performance.video')];
        return { runtime: globalThis[Symbol.for('aethervsr.m10.performance.runtime')],
          video: { opening: video.opening, closing: video.closing, rows: video.rows, events: video.events, overflow: video.overflow },
          scheduling: globalThis[Symbol.for('aethervsr.m105.scheduling')] };
      });
      const bytes = gzipSync(JSON.stringify(raw));
      row.raw = { path: `${prefix}.${kind}.json.gz`, sha256: sha256(bytes), bytes: bytes.length };
      writeFileSync(row.raw.path, bytes, { flag: 'wx' });
      try { validateCapture(raw, item); assert.deepEqual(row.errors, []); row.verdict = 'VALID'; }
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