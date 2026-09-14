import assert from 'node:assert/strict';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { arch, platform, release, totalmem } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { ROOT, sha256, startFixtures } from './m10-fixtures.mjs';
import { bounded, openExtension, until, verifyBuild } from './m10-browser.mjs';
import { openNativeChrome } from './m9-browser.mjs';

export const KINDS = ['no-extension', 'installed-idle', 'extension-baseline', 'extension-auto', 'harness'];
export function parseCases(value) {
  assert(Array.isArray(value) && value.length > 0 && value.length <= 100, 'Expected 1..100 cases');
  const ids = new Set();
  return value.map(item => {
    assert(item && typeof item === 'object' && !Array.isArray(item), 'Invalid case');
    assert(Object.keys(item).every(key => ['id', 'kind', 'durationMs', 'warmupMs'].includes(key)), 'Unknown case field');
    assert(typeof item.id === 'string' && /^[a-zA-Z0-9_-]{1,80}$/.test(item.id) && !ids.has(item.id), 'Invalid/duplicate case id');
    assert(KINDS.includes(item.kind), 'Unknown performance kind');
    const durationMs = item.durationMs ?? (item.kind === 'extension-auto' ? 600000 : 30000);
    const warmupMs = item.warmupMs ?? 5000;
    assert(Number.isInteger(durationMs) && durationMs >= 1000 && durationMs <= 600000, 'Invalid durationMs');
    assert(durationMs <= 60000 || item.kind === 'extension-auto', 'Long windows require the actual Auto extension');
    assert(Number.isInteger(warmupMs) && warmupMs >= 5000 && warmupMs <= 60000, 'Invalid warmupMs');
    ids.add(item.id);
    return { id: item.id, kind: item.kind, durationMs, warmupMs };
  });
}
export function quantile(values, probability) {
  assert(Number.isFinite(probability) && probability >= 0 && probability <= 1, 'Invalid quantile');
  const sorted = values.filter(value => typeof value === 'number' && Number.isFinite(value)).sort((left, right) => left - right);
  if (!sorted.length) return null;
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  return sorted[lower] + (sorted[Math.ceil(index)] - sorted[lower]) * (index - lower);
}
export function counterDelta(opening, closing) {
  return Number.isFinite(opening) && Number.isFinite(closing) && opening >= 0 && closing >= opening ? closing - opening : null;
}
export function windowRate(opening, closing, durationMs) {
  const delta = counterDelta(opening, closing);
  return delta !== null && Number.isFinite(durationMs) && durationMs > 0 ? delta * 1000 / durationMs : null;
}
export function distribution(values) {
  const finite = values.filter(value => typeof value === 'number' && Number.isFinite(value));
  return { count: finite.length, p50: quantile(finite, 0.5), p95: quantile(finite, 0.95),
    mean: finite.length ? finite.reduce((total, value) => total + value, 0) / finite.length : null,
    max: finite.length ? finite.reduce((maximum, value) => Math.max(maximum, value), -Infinity) : null };
}

export function installVideoObserver() {
  const key = Symbol.for('aethervsr.m10.performance.video');
  const prefix = 'aethervsr:m10:performance:';
  const observer = globalThis[key] = { rows: [], events: [], callbacks: 0, presented: null, overflow: false };
  let recording = false, handle, progress, previous = null;
  observer.done = new Promise(done => { observer.complete = done; });
  const quality = video => {
    const value = video.getVideoPlaybackQuality?.();
    return value ? { total: value.totalVideoFrames, dropped: value.droppedVideoFrames } : null;
  };
  const foreground = () => ({ visibility: document.visibilityState, focused: document.hasFocus() });
  const snapshot = () => ({ at: performance.now(), callbacks: observer.callbacks, presented: observer.presented,
    quality: quality(observer.video), ...foreground(), paused: observer.video.paused, source: observer.video.currentSrc });
  const push = (rows, row) => { if (rows.length < 160000) rows.push(row); else observer.overflow = true; };
  document.addEventListener('DOMContentLoaded', () => {
    const video = observer.video = document.querySelector('video');
    if (!video?.requestVideoFrameCallback) throw new Error('Native rVFC video required');
    const tick = (now, metadata) => {
      const observedAt = performance.now();
      observer.callbacks++;
      const delta = previous === null ? null : metadata.presentedFrames - previous;
      previous = observer.presented = metadata.presentedFrames;
      if (recording) {
        const counts = quality(video);
        push(observer.rows, [observedAt, now, metadata.mediaTime, metadata.presentedFrames, delta,
          now - metadata.presentationTime, counts?.total ?? null, counts?.dropped ?? null, performance.now() - observedAt]);
      }
      handle = video.requestVideoFrameCallback(tick);
    };
    handle = video.requestVideoFrameCallback(tick);
    for (const type of ['visibilitychange', 'blur', 'focus', 'pagehide', 'loadstart', 'error', 'ratechange', 'pause', 'play']) {
      const target = ['blur', 'focus', 'pagehide'].includes(type) ? window : ['loadstart', 'error', 'ratechange', 'pause', 'play'].includes(type) ? video : document;
      target.addEventListener(type, () => { if (recording) push(observer.events, { at: performance.now(), type, ...foreground() }); });
    }
  }, { once: true });
  window.addEventListener(`${prefix}start`, () => {
    observer.opening = snapshot(); recording = true;
    progress = setInterval(() => console.info(`M10 performance: ${Math.round((performance.now() - observer.opening.at) / 1000)}s observed`), 60000);
  });
  window.addEventListener(`${prefix}end`, () => {
    observer.closing = snapshot(); recording = false; clearInterval(progress);
    observer.video.cancelVideoFrameCallback(handle);
  });
  window.addEventListener(`${prefix}complete`, () => observer.complete());
}

export function installRuntimeRecorder(options) {
  const key = Symbol.for('aethervsr.m10.performance.runtime');
  const prefix = 'aethervsr:m10:performance:';
  const manager = options.extension ? globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)] : null;
  const attachment = manager?.attachment;
  const driver = attachment?.driver ?? (options.kind === 'harness' ? globalThis.aethervsrRuntime?.driver : null);
  const pipeline = driver?.pipeline;
  const video = driver?.video ?? document.querySelector('video');
  if (!video || ((options.extension || options.kind === 'harness') && !pipeline)) throw new Error('Actual runtime missing');
  if (pipeline && typeof pipeline.onFrame !== 'function') throw new Error('Shared driver callback missing');
  if (manager?.testAccess !== undefined) throw new Error('Production build required');
  const record = globalThis[key] = { timeOrigin: performance.timeOrigin, samples: [], frames: [], driverCpuRows: [], states: [], configurations: [], overflow: false, error: null };
  const previous = driver ? { onSample: driver.onSample, onFrame: driver.onFrame, onChange: driver.onChange, onConfigure: driver.onConfigure } : {};
  const previousPipelineFrame = pipeline?.onFrame;
  let started = null, ended = null, finishing = false, lastState = null, timer, deadline;
  let inPipelineFrame = false, ownedCallbackMs = null, pendingState = null, pendingConfiguration = null;
  const push = (rows, row) => { if (rows.length < 160000) rows.push(row); else record.overflow = true; };
  const snapshot = () => ({ at: performance.now(), runtime: driver?.snapshot() ?? null,
    status: manager?.status() ?? null, stats: pipeline?.stats(performance.now()) ?? null,
    adapter: (attachment?.gpu ?? pipeline?.gpu)?.adapterReport ?? null,
    visibility: document.visibilityState, focused: document.hasFocus(),
    viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio },
    video: { width: video.videoWidth, height: video.videoHeight, paused: video.paused, rate: video.playbackRate,
      rect: video.getBoundingClientRect().toJSON(), source: video.currentSrc },
    canvasCount: document.querySelectorAll('canvas').length });
  const state = value => {
    const identity = `${value.state}:${value.tier}`;
    if (started !== null && ended === null && identity !== lastState) {
      push(record.states, [performance.now(), value.state, value.tier, value.reason]); lastState = identity;
    }
  };
  if (driver) {
    driver.onSample = sample => {
      previous.onSample?.call(driver, sample);
      if (started !== null && sample.submittedAt >= started && (ended === null || sample.submittedAt < ended))
        push(record.samples, [sample.ms, sample.submittedAt, sample.resolvedAt, sample.sequence, sample.generation, Number(sample.neural)]);
    };
    driver.onFrame = tick => {
      const before = performance.now();
      try { previous.onFrame?.call(driver, tick); }
      finally { ownedCallbackMs = previous.onFrame ? performance.now() - before : null; }
    };
    driver.onChange = value => {
      previous.onChange?.call(driver, value);
      if (inPipelineFrame) pendingState = value;
      else state(value);
    };
    const configuration = value => { if (started !== null && ended === null) push(record.configurations, [performance.now(), value.configureMs, value.generation]); };
    driver.onConfigure = value => {
      previous.onConfigure?.call(driver, value);
      if (inPipelineFrame) pendingConfiguration = value;
      else configuration(value);
    };
    pipeline.onFrame = tick => {
      const neural = Number(pipeline.currentUpscaler.neural);
      ownedCallbackMs = null; pendingState = null; pendingConfiguration = null; inPipelineFrame = true;
      const before = performance.now();
      try { previousPipelineFrame.call(pipeline, tick); }
      finally {
        const after = performance.now();
        inPipelineFrame = false;
        if (started !== null && ended === null) {
          push(record.driverCpuRows, [before, after - before]);
          const quality = video.getVideoPlaybackQuality?.();
          push(record.frames, [before, tick.now, tick.mediaTime, tick.presentedDelta,
            tick.presentationTime === tick.now && tick.expectedDisplayTime === tick.now ? null : Math.max(0, tick.now - tick.presentationTime), quality?.totalVideoFrames ?? null,
            quality?.droppedVideoFrames ?? null, pipeline.cpuFrame.last(), ownedCallbackMs, neural]);
        }
        if (pendingState) state(pendingState);
        if (pendingConfiguration) configuration(pendingConfiguration);
      }
    };
  }
  const finish = async error => {
    if (finishing) return;
    finishing = true; clearTimeout(timer); clearTimeout(deadline);
    try {
      ended = performance.now(); record.ended = ended;
      record.closing = snapshot();
      window.dispatchEvent(new Event(`${prefix}end`));
      if (pipeline) pipeline.onFrame = previousPipelineFrame;
      pipeline?.stop();
      if (pipeline) {
        let drainTimer;
        try { await Promise.race([pipeline.drainTimings(), new Promise((_, reject) => { drainTimer = setTimeout(() => reject(new Error('GPU drain timeout')), 3000); })]); }
        finally { clearTimeout(drainTimer); }
      }
      if (error) throw error;
      if (pipeline?.error) throw new Error(String(pipeline.error));
      if (manager && manager.attachment !== attachment) throw new Error('Attachment replaced during capture');
    } catch (failure) { record.error = String(failure); }
    finally {
      if (pipeline) pipeline.onFrame = previousPipelineFrame;
      try { video.pause(); driver?.syncActive(); } catch (failure) { record.error ??= String(failure); }
      if (driver) Object.assign(driver, previous);
      record.completedAt = performance.now();
      window.dispatchEvent(new Event(`${prefix}complete`));
    }
  };
  const begin = () => {
    try {
      const runtime = driver?.snapshot();
      if (runtime && (!runtime.running || (options.kind === 'extension-baseline'
        ? runtime.actualTier !== 'baseline' || runtime.controller.state !== 'manual-baseline'
        : runtime.actualTier !== 'neural' || runtime.controller.state !== 'stable'))) throw new Error('Runtime not stable at window opening');
      if (document.visibilityState !== 'visible' || !document.hasFocus() || video.paused) throw new Error('Foreground playing video required');
      record.opening = snapshot();
      window.dispatchEvent(new Event(`${prefix}start`));
      record.started = started = performance.now();
      if (runtime) state(runtime.controller);
      timer = setTimeout(() => { void finish(); }, options.durationMs);
    } catch (error) { void finish(error); }
  };
  timer = setTimeout(begin, options.warmupMs);
  deadline = setTimeout(() => { void finish(new Error('Capture deadline exceeded')); }, options.warmupMs + options.durationMs + 10000);
  return { installed: true, runtime: !!driver };
}

function activeTime(runtime) {
  const opening = runtime.opening.runtime, closing = runtime.closing.runtime;
  return opening?.controller?.activeMs !== undefined || closing?.controller?.activeMs !== undefined
    ? counterDelta(opening?.controller?.activeMs, closing?.controller?.activeMs)
    : counterDelta(opening?.session?.activeMs, closing?.session?.activeMs);
}

function validateActivity({ runtime, video }) {
  assert(video.events.every(event => event.visibility === 'visible' && event.focused && event.type === 'focus'), 'Foreground/source/playback integrity changed');
  for (const boundary of [video.opening, video.closing, runtime.opening, runtime.closing]) assert(boundary?.focused && boundary.visibility === 'visible', 'Invalid foreground boundary');
  assert(video.opening.paused === false && video.closing.paused === false &&
    runtime.opening.video.paused === false && runtime.closing.video.paused === false, 'Paused window boundary');
}

function timingSummary(runtime) {
  return {
    gpuMs: distribution(runtime.samples.map(row => row[0])),
    neuralGpuMs: distribution(runtime.samples.filter(row => row[5] === 1).map(row => row[0])),
    baselineGpuMs: distribution(runtime.samples.filter(row => row[5] === 0).map(row => row[0])),
    coreCpuMs: distribution(runtime.frames.map(row => row[7])), adapterOnFrameMs: distribution(runtime.frames.map(row => row[8])),
    driverCpuMs: distribution(runtime.driverCpuRows.map(row => row[1])),
    callbackLatencyMs: distribution(runtime.frames.map(row => row[4])),
  };
}

export function windowSummary(raw, started, ended) {
  const { runtime, video } = raw;
  assert(Number.isFinite(started) && Number.isFinite(ended) && started >= runtime.started && ended <= runtime.ended && ended > started, 'Invalid slice window');
  validateActivity(raw);
  if (runtime.opening.runtime) assert(activeTime(runtime) >= runtime.ended - runtime.started, 'Insufficient active time for slices');
  const durationMs = ended - started;
  const slice = rows => {
    const from = rows.findIndex(row => row[0] >= started), to = rows.findIndex(row => row[0] >= ended);
    const rowRange = [from < 0 ? rows.length : from, to < 0 ? rows.length : to];
    const selected = rows.slice(...rowRange);
    return { rows: selected, coverage: { rowRange, firstObservedAt: selected[0]?.[0] ?? null, lastObservedAt: selected.at(-1)?.[0] ?? null } };
  };
  const native = slice(video.rows), frames = slice(runtime.frames), driverCpu = slice(runtime.driverCpuRows);
  const sampleIndices = [];
  runtime.samples.forEach((row, index) => { if (row[1] >= started && row[1] < ended) sampleIndices.push(index); });
  const samples = sampleIndices.map(index => runtime.samples[index]);
  const extent = index => samples.reduce((range, row) => [range[0] === null ? row[index] : Math.min(range[0], row[index]),
    range[1] === null ? row[index] : Math.max(range[1], row[index])], [null, null]);
  const sum = (rows, index) => rows.every(row => Number.isFinite(row[index]) && row[index] >= 0)
    ? rows.reduce((total, row) => total + row[index], 0) : null;
  const rendered = runtime.opening.runtime ? frames.rows.length : null;
  const presented = runtime.opening.runtime ? sum(frames.rows, 3) : null;
  const nativePresented = sum(native.rows, 4);
  const openingIndex = video.rows.findLastIndex(row => row[0] <= started);
  const closingIndex = video.rows.findIndex(row => row[0] >= ended);
  const decoderBoundary = (index, boundary) => index < 0
    ? { at: boundary.at, total: boundary.quality?.total ?? null, dropped: boundary.quality?.dropped ?? null, rowIndex: null }
    : { at: video.rows[index][0], total: video.rows[index][6], dropped: video.rows[index][7], rowIndex: index };
  const opening = decoderBoundary(openingIndex, video.opening), closing = decoderBoundary(closingIndex, video.closing);
  assert(opening.at <= started && closing.at >= ended, 'Decoder observations do not bracket slice');
  const decoded = counterDelta(opening.total, closing.total), dropped = counterDelta(opening.dropped, closing.dropped);
  return { started, ended, durationMs, activeMs: durationMs,
    coverage: { video: native.coverage, frames: frames.coverage, driverCpuRows: driverCpu.coverage,
      samples: { rowIndices: sampleIndices, submittedAt: extent(1), resolvedAt: extent(2) } },
    nativeCallbacks: native.rows.length, nativePresented, rendered, presented,
    skipped: presented !== null && rendered !== null ? presented - rendered : null,
    nativeCallbackFps: native.rows.length * 1000 / durationMs,
    nativePresentedFps: nativePresented === null ? null : nativePresented * 1000 / durationMs,
    renderedFps: rendered === null ? null : rendered * 1000 / durationMs,
    runtimePresentedFps: presented === null ? null : presented * 1000 / durationMs,
    pipelineLossPercent: presented > 0 && rendered !== null ? 100 * (presented - rendered) / presented : null,
    decoder: { opening, closing, coverageMs: closing.at - opening.at, frames: decoded, drops: dropped,
      dropPercent: decoded > 0 && dropped !== null ? 100 * dropped / decoded : null },
    ...timingSummary({ samples, frames: frames.rows, driverCpuRows: driverCpu.rows }),
    nativeCallbackLatencyMs: distribution(native.rows.map(row => row[5])), observerBookkeepingMs: distribution(native.rows.map(row => row[8])) };
}

export function summarizeCapture(raw) {
  const { runtime, video } = raw;
  const durationMs = runtime.ended - runtime.started;
  const nativeDurationMs = video.closing.at - video.opening.at;
  const opening = runtime.opening.runtime?.session, closing = runtime.closing.runtime?.session;
  const dropped = counterDelta(video.opening.quality?.dropped, video.closing.quality?.dropped);
  const decoded = counterDelta(video.opening.quality?.total, video.closing.quality?.total);
  const rendered = counterDelta(opening?.framesRendered, closing?.framesRendered);
  const presented = counterDelta(opening?.framesPresented, closing?.framesPresented);
  const measured = (calls, totalMs) => ({ calls, totalMs, wallFraction: totalMs === null ? null : totalMs / durationMs, msPerSecond: totalMs === null ? null : totalMs * 1000 / durationMs,
    meanMs: calls > 0 && totalMs !== null ? totalMs / calls : null });
  const infrastructure = (owner, callsKey, timeKey) => {
    const before = runtime.opening.status?.details?.[owner], after = runtime.closing.status?.details?.[owner];
    const first = owner === 'attachment' ? before?.infrastructure : before;
    const last = owner === 'attachment' ? after?.infrastructure : after;
    return measured(counterDelta(first?.[callsKey], last?.[callsKey]), counterDelta(first?.[timeKey], last?.[timeKey]));
  };
  const discovery = infrastructure('infrastructure', 'discoveryCalls', 'discoveryMs');
  const discoveryGeometry = infrastructure('infrastructure', 'geometryCalls', 'geometryMs');
  const attachmentGeometry = infrastructure('attachment', 'geometryCalls', 'geometryTotalMs');
  const combined = [discovery, discoveryGeometry, attachmentGeometry];
  const sum = key => combined.every(value => value[key] !== null) ? combined.reduce((total, value) => total + value[key], 0) : null;
  return { durationMs, nativeDurationMs, activeMs: activeTime(runtime),
    nativeCallbackFps: windowRate(video.opening.callbacks, video.closing.callbacks, nativeDurationMs),
    nativePresentedFps: windowRate(video.opening.presented, video.closing.presented, nativeDurationMs),
    renderedFps: windowRate(opening?.framesRendered, closing?.framesRendered, durationMs),
    runtimePresentedFps: windowRate(opening?.framesPresented, closing?.framesPresented, durationMs),
    rendered, presented, skipped: counterDelta(opening?.framesSkipped, closing?.framesSkipped), decoderDrops: dropped, decoderFrames: decoded,
    decoderDropPercent: decoded > 0 && dropped !== null ? 100 * dropped / decoded : null,
    pipelineLossPercent: presented > 0 && rendered !== null ? 100 * (presented - rendered) / presented : null,
    ...timingSummary(runtime), nativeCallbackLatencyMs: distribution(video.rows.map(row => row[5])),
    observerBookkeepingMs: distribution(video.rows.map(row => row[8])), configureMs: distribution(runtime.configurations.map(row => row[1])),
    discovery, discoveryGeometry, attachmentGeometry, combinedInfrastructure: measured(sum('calls'), sum('totalMs')),
    attachmentGeometryLifetimeMaxMs: runtime.closing.status?.details?.attachment?.infrastructure?.geometryMaxMs ?? null,
    windowStats: durationMs >= 240000 ? { first120s: windowSummary(raw, runtime.started, runtime.started + 120000),
      last120s: windowSummary(raw, runtime.ended - 120000, runtime.ended) } : null };
}

export function validateCapture(raw, item) {
  const { runtime, video } = raw;
  assert(!runtime.error, runtime.error); assert(!runtime.overflow && !video.overflow, 'Trace capacity exceeded');
  validateActivity(raw);
  assert(runtime.ended - runtime.started >= item.durationMs && runtime.ended - runtime.started <= item.durationMs + 1000, 'Window timer overrun');
  assert.equal(video.closing.source, video.opening.source, 'Video source changed');
  assert(video.rows.length > 0 && video.rows.every(row => row[4] === null || row[4] >= 0), 'Video counter reset/no callbacks');
  assert.equal(counterDelta(video.opening.callbacks, video.closing.callbacks), video.rows.length, 'Incomplete observer trace');
  for (const boundary of [runtime.opening, runtime.closing]) {
    assert.equal(boundary.viewport.width, 1200); assert.equal(boundary.viewport.height, 820);
    assert(boundary.video.width === 1280 && boundary.video.height === 720 && boundary.video.rate === 1 && !boundary.video.paused, 'Unexpected source state');
    assert.equal(boundary.canvasCount, boundary.runtime ? 1 : 0);
    if (item.kind !== 'harness') assert(boundary.video.rect.width === 640 && boundary.video.rect.height === 360, 'Fixture CSS changed');
  }
  if (runtime.opening.runtime) {
    assert(activeTime(runtime) >= item.durationMs, 'Insufficient active time');
    assert(runtime.closing.runtime?.running && runtime.closing.runtime.controller.state !== 'failed', 'Runtime stopped');
    assert.equal(counterDelta(runtime.opening.runtime.session.framesRendered, runtime.closing.runtime.session.framesRendered), runtime.frames.length, 'Incomplete runtime trace');
    assert.equal(runtime.driverCpuRows.length, runtime.frames.length, 'Incomplete driver CPU trace');
  }
}

export async function harnessServer() {
  const origin = 'http://127.0.0.1:5173';
  let owned;
  try { await fetch(origin, { signal: AbortSignal.timeout(2000) }); }
  catch (error) {
    if (error.cause?.code !== 'ECONNREFUSED') throw error;
    const { createServer } = await import('vite');
    owned = await createServer({ root: ROOT, mode: 'benchmark', server: { host: '127.0.0.1', port: 5173, strictPort: true } });
    try { await owned.listen(); } catch (failure) { await owned.close(); throw failure; }
  }
  const verify = async () => {
    const read = async path => { const response = await fetch(`${origin}${path}`, { signal: AbortSignal.timeout(10000) }); assert(response.ok, path); return Buffer.from(await response.arrayBuffer()); };
    const html = await read('/');
    assert(html.includes('/src/main.ts') && !html.includes('/@vite/client'), 'Port 5173 must serve this root index in benchmark mode');
    const hashes = { index: sha256(html) };
    for (const path of ['src/main.ts', 'src/runtime.ts', 'src/core/pipeline.ts']) {
      const source = readFileSync(join(ROOT, path), 'utf8');
      assert((await read(`/${path}?raw`)).toString().startsWith(`export default ${JSON.stringify(source)}`), `Wrong root server: ${path}`);
      hashes[path] = sha256(source);
    }
    for (const path of ['media/m9/720p60.mp4', 'models/aethersr-c16d2.json']) {
      hashes[path] = sha256(await read(`/${path}`));
      assert.equal(hashes[path], sha256(readFileSync(join(ROOT, 'public', path))), `Served bytes differ: ${path}`);
    }
    return hashes;
  };
  try { return { origin, owned: !!owned, pins: await verify(), verify, close: async () => { await owned?.close(); } }; }
  catch (error) { await owned?.close(); throw error; }
}

function machineInfo() {
  const info = { os: platform(), release: release(), arch: arch(), memoryBytes: totalmem(), displayRefreshRate: 'not measured' };
  if (platform() === 'darwin') {
    info.osVersion = execFileSync('sw_vers', ['-productVersion'], { encoding: 'utf8', timeout: 3000 }).trim();
    const hardware = JSON.parse(execFileSync('system_profiler', ['SPHardwareDataType', '-json'], { encoding: 'utf8', timeout: 15000 })).SPHardwareDataType[0];
    info.hardware = { model: hardware.machine_model ?? null, chip: hardware.chip_type ?? null, memory: hardware.physical_memory ?? null };
  }
  return info;
}

const SCOPES = {
  window: 'After native popup dismissal, readiness and >=5s warmup. Start/end in browser performance.now; exact opening/closing counter deltas. Drain, pause, screenshots and teardown excluded. Native observer boundaries are synchronous same-task events; its own duration is reported.',
  rates: 'Native rVFC delivery and presented counter rates are browser video observations, not pipeline rendering. RuntimeSession rendered counts successful submissions, not GPU output verification. Control processing metrics are null (not measured).',
  gpu: 'Raw upscale-stage timestamps, filtered by submittedAt inside the window, including cold samples and readbacks received during <=3s post-window drain. Excludes decode/import/presentation; dropped timer slots are not fabricated.',
  quantiles: 'Linear interpolation at (n-1)*p over finite raw samples. Chrome default timestamp quantization retained; not directly comparable with M9 unquantized GPU measurements. No acceptance tolerances until calibration.',
  cpu: 'Per-frame pipeline.cpuFrame.last(): existing import/encode/submit wall-time sample, equivalent to stats.cpuFrameMs.last, without per-frame stats sorting. Excludes configuration, driver/controller, DOM and GPU execution.',
  driverCpu: 'Identical performance.now bracket around the original pipeline.onFrame shared RuntimeDriver handler in harness/extension, after submit. Includes session/cadence/controller and original adapter callback; its nested adapterOnFrameMs is not added again. Bounded [observedAt, ms] rows are appended in finally after the bracket, as are frame/state/configuration records. No added per-frame snapshot/JSON clone. Excludes core import/encode/submit, independent timers and GPU work; default clock quantization and minimal wrapper/timer overhead remain.',
  adapter: 'Chained driver.onFrame brackets only the original attachment output-visibility/failure callback, not all extension CPU. Geometry/discovery deltas use existing counters, no added per-frame layout queries. Lifetime geometry max is NOT a window maximum.',
  startup: 'readyMs includes navigation, actual popup activation/dismissal and readiness polling; separate from measurement. Opening configuration histogram covers runtime startup. Model initialization alone not measured.',
  observer: 'Same plain init-script rVFC observer in every mode. Bookkeeping timer excludes rescheduling and final append overhead. Runtime hooks add instrumentation to active pipeline cases only.',
  activeTime: 'Opening/closing runtime controller.activeMs delta (session.activeMs only when controller clock absent), before stop/drain/pause. Must meet requested duration without rounding/tolerance. Start follows opening snapshot, so active-clock coverage can be slightly longer. Controls report null. Pause/play and non-focus integrity events invalidate capture.',
  windows: 'For captures >=240s: first/last non-overlapping 120s slices of [started, ended), excluding drain. Native/frame/driver CPU rows selected by observedAt (frames at shared-driver entry); half-open rowRange indexes reference raw arrays. Presented/skipped deltas belong to delivered callbacks, including their preceding interval. GPU samples selected by submittedAt, retaining raw rowIndices, submission/resolution extents and exact neural flags, regardless of readback arrival order or drain. Decoder deltas use nearest outward native observations (opening/closing snapshots at edges), with explicit timestamps, row indexes and coverageMs; never interpolated or claimed as exact 120s decoder counts. Slice activeMs is continuous foreground-playing wall time established by activity boundaries/events and full runtime active delta, not interpolated controller snapshots.',
  comparison: 'Harness uses the actual root index with normal UI/stats, unchanged CSS. Extension fixture video is 640x360 CSS. Same CFR bytes, 1200x820 viewport; not a display-CSS-matched comparison.',
};

export async function runPerformance(casesPath, outputPrefix) {
  const casesBytes = readFileSync(resolve(casesPath)), cases = parseCases(JSON.parse(casesBytes));
  const prefix = resolve(outputPrefix);
  assert(relative(join(ROOT, '.cache'), prefix) && !relative(join(ROOT, '.cache'), prefix).startsWith('..'), 'Output prefix must be inside root .cache');
  const outputs = [`${prefix}.json.gz`, ...cases.flatMap(item => [`${prefix}.${item.id}.json.gz`, `${prefix}.${item.id}.png`])];
  assert(outputs.every(path => !existsSync(path)), 'Never overwrite output evidence');
  const build = verifyBuild(false);
  const sourcePins = Object.fromEntries(['tools/m10-performance.mjs', 'tools/m10-browser.mjs', 'tools/m10-fixtures.mjs', 'tools/m10-fixtures/index.html', 'tools/m9-browser.mjs']
    .map(path => [path, sha256(readFileSync(join(ROOT, path)))]));
  const report = { schemaVersion: 1, started: new Date().toISOString(), completion: 'UNVERIFIED', acceptance: 'not evaluated; calibration required',
    casesSha256: sha256(casesBytes), cases, sourcePins, build, scopes: SCOPES, machine: machineInfo(), results: [] };
  let fixtures, server, native, interrupted = false;
  const cleanup = async () => { try { await native?.close(); } finally { try { await fixtures?.close(); } finally { await server?.close(); } } };
  const abort = () => { interrupted = true; void cleanup().catch(error => { report.cleanupError = String(error); }); };
  const write = (path, value) => { const bytes = gzipSync(JSON.stringify(value)); writeFileSync(path, bytes, { flag: 'wx' }); return { path: relative(ROOT, path), bytes: bytes.length, sha256: sha256(bytes) }; };
  process.once('SIGINT', abort); process.once('SIGTERM', abort);
  try {
    mkdirSync(dirname(prefix), { recursive: true });
    fixtures = await startFixtures({ mse: false });
    assert.equal(fixtures.evidence.path, 'public/media/m9/720p60.mp4', 'Exact CFR fixture required, no fallback');
    const stream = fixtures.evidence.probe.streams.find(item => item.codec_type === 'video');
    assert(stream?.width === 1280 && stream.height === 720 && stream.r_frame_rate === '60/1' && stream.avg_frame_rate === '60/1', '720p60 CFR metadata required');
    report.media = fixtures.evidence;
    if (cases.some(item => item.kind === 'harness')) {
      server = await harnessServer(); report.harness = { owned: server.owned, pins: server.pins };
      assert.equal(server.pins['models/aethersr-c16d2.json'], build.provenance.modelSha256, 'Harness production model mismatch');
    }
    for (const item of cases) {
      assert(!interrupted, 'Interrupted');
      const active = item.kind.startsWith('extension-'), installed = active || item.kind === 'installed-idle';
      const result = { case: item, events: [], errors: [], completion: 'UNVERIFIED' }; report.results.push(result);
      const record = (name, data) => result.events.push({ name, data });
      const flags = ['--autoplay-policy=no-user-gesture-required', '--window-size=1280,900'];
      native = installed ? await openExtension(build, record) : await openNativeChrome(flags);
      try {
        native.context.setDefaultTimeout(10000); native.context.setDefaultNavigationTimeout(10000);
        const cdp = await native.browser.newBrowserCDPSession();
        try { result.browser = { version: await bounded(cdp.send('Browser.getVersion')), executableSha256: sha256(readFileSync(native.executable)), flags: installed ? result.events.find(event => event.name === 'browser').data.flags : flags }; }
        finally { await cdp.detach(); }
        const page = await native.context.newPage(); await page.setViewportSize({ width: 1200, height: 820 });
        page.on('pageerror', error => result.errors.push(String(error)));
        page.on('console', message => {
          if (message.type() === 'error' && result.errors.length < 100) result.errors.push(message.text());
          if (message.text().startsWith('M10 performance:')) console.log(`${item.id}: ${message.text()}`);
        });
        await page.addInitScript(installVideoObserver);
        const readyStarted = Date.now();
        await page.goto(item.kind === 'harness' ? `${server.origin}/?mode=auto&clip=/media/m9/720p60.mp4` : `${fixtures.url}?case=custom`);
        await page.bringToFront();
        await page.waitForFunction(() => globalThis[Symbol.for('aethervsr.m10.performance.video')]?.callbacks >= 2);
        if (installed) {
          result.idle = await native.workerEval(async () => ({ scripts: await chrome.scripting.getRegisteredContentScripts(), storage: await chrome.storage.session.get(null) }));
          assert.equal(result.idle.scripts.length, 0); assert.equal(Object.keys(result.idle.storage).length, 0);
          assert.equal(await page.locator('canvas').count(), 0);
          result.idle.scope = 'Before any action: no canvas, no registered content scripts, empty session storage and verified no automatic manifest injection. No page injection used to inspect idle; no independent MO/device instrumentation.';
        }
        if (active) {
          const popup = await native.popup(page);
          try { assert((await popup.click('#enable')).enabled); if (item.kind === 'extension-baseline') await popup.click('input[value="baseline"]'); }
          finally { await popup.dismiss(); }
        }
        const evaluate = (fn, arg) => active ? native.isolated(page, fn, arg) : bounded(page.evaluate(fn, arg));
        if (active || item.kind === 'harness') await until(() => evaluate(extension => {
          const driver = extension ? globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)]?.attachment?.driver : globalThis.aethervsrRuntime?.driver;
          return driver?.snapshot() ?? null;
        }, active), value => value?.running && (item.kind === 'extension-baseline' ? value.actualTier === 'baseline' && value.controller.state === 'manual-baseline' : value.actualTier === 'neural' && value.controller.state === 'stable'), 20000);
        result.readyMs = Date.now() - readyStarted; result.modelInitializationMs = null;
        await evaluate(installRuntimeRecorder, { ...item, extension: active });
        await bounded(page.evaluate(() => globalThis[Symbol.for('aethervsr.m10.performance.video')].done), item.warmupMs + item.durationMs + 20000, 'Performance window');
        const runtime = await evaluate(() => globalThis[Symbol.for('aethervsr.m10.performance.runtime')]);
        const video = await page.evaluate(() => { const value = globalThis[Symbol.for('aethervsr.m10.performance.video')]; return { opening: value.opening, closing: value.closing, rows: value.rows, events: value.events, overflow: value.overflow }; });
        const raw = { case: item, sourceCommit: build.provenance.sourceCommit, bundleSha256: build.provenance.bundleSha256,
          casesSha256: report.casesSha256, mediaSha256: report.media.sha256, runtime, video, columns: { samples: ['ms', 'submittedAt', 'resolvedAt', 'sequence', 'generation', 'neural'],
          driverCpuRows: ['observedAt', 'ms'],
          frames: ['observedAt', 'now', 'mediaTime', 'presentedDelta', 'latencyMs', 'qualityTotal', 'qualityDropped', 'coreCpuMs', 'adapterOnFrameMs', 'neural'],
          video: ['observedAt', 'now', 'mediaTime', 'presentedFrames', 'presentedDelta', 'latencyMs', 'qualityTotal', 'qualityDropped', 'observerMs'] } };
        result.raw = write(`${prefix}.${item.id}.json.gz`, raw);
        assert(result.raw.bytes <= 3 * 1024 * 1024, 'Raw trace exceeds 3 MiB evidence budget; retained as unverified');
        assert.deepEqual(result.errors, [], 'Unexpected page errors');
        validateCapture(raw, item);
        result.summary = summarizeCapture(raw);
        result.boundaries = { opening: runtime.opening, closing: runtime.closing, nativeOpening: video.opening, nativeClosing: video.closing };
        await page.screenshot({ path: `${prefix}.${item.id}.png`, timeout: 5000 });
        result.screenshot = { path: relative(ROOT, `${prefix}.${item.id}.png`), sha256: sha256(readFileSync(`${prefix}.${item.id}.png`)), scope: 'Post-window local fixture/harness viewport; manual visual review not performed' };
        if (active) result.teardown = await evaluate(() => {
          const manager = globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)];
          const before = performance.now(); manager.stop(); const elapsedMs = performance.now() - before;
          return { elapsedMs, status: manager.status(), scope: 'Synchronous manager.stop only, after window/drain/pause/screenshot; not frame CPU or popup roundtrip' };
        });
        if (result.teardown) assert(Object.values(result.teardown.status.details.lastTeardown).every(value => value === 0), 'Teardown resources remain');
        result.completion = 'CAPTURED';
        console.log(`${item.id}: captured; native callback FPS ${result.summary.nativeCallbackFps ?? 'not measured'}, rendered FPS ${result.summary.renderedFps ?? 'not measured'}`);
      } finally { await native.close(); native = undefined; }
    }
    assert(!interrupted, 'Interrupted');
    assert.deepEqual(verifyBuild(false), build, 'Build/source changed during capture');
    for (const [path, digest] of Object.entries(sourcePins)) assert.equal(sha256(readFileSync(join(ROOT, path))), digest, `Runner source changed: ${path}`);
    assert.equal(sha256(readFileSync(resolve(casesPath))), report.casesSha256, 'Cases changed');
    assert.equal(sha256(readFileSync(join(ROOT, fixtures.evidence.path))), fixtures.evidence.sha256, 'Media changed');
    if (server) assert.deepEqual(await server.verify(), server.pins, 'Harness server changed');
    report.completion = 'CAPTURED';
  } catch (error) { report.error = String(error); throw error; }
  finally {
    try { await cleanup(); }
    catch (error) { report.completion = 'UNVERIFIED'; report.cleanupError = String(error); throw error; }
    finally { process.off('SIGINT', abort); process.off('SIGTERM', abort); report.ended = new Date().toISOString(); write(`${prefix}.json.gz`, report); }
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [casesPath, outputPrefix, ...extra] = process.argv.slice(2);
  if (!casesPath || !outputPrefix || extra.length) {
    console.error('Usage: node tools/m10-performance.mjs CASES.json .cache/m10/OUTPUT_PREFIX'); process.exitCode = 2;
  } else await runPerformance(casesPath, outputPrefix).catch(error => { console.error(String(error)); process.exitCode = 1; });
}