import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const modelFile = 'public/models/aethersr-c16d2.json';
const fixedModelSha256 = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
const sourcePaths = ['src', 'index.html', 'vite.config.ts', 'tsconfig.json', 'package.json',
  'package-lock.json', 'tools/m9-live.mjs', 'tools/m9-runtime.mjs'];
const flags = ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis',
  '--disable-dawn-features=timestamp_quantization', '--autoplay-policy=no-user-gesture-required',
  '--window-position=0,0', '--window-size=1280,900'];
const progressPrefix = 'M9_LIVE_PROGRESS ';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const digest = file => hash(readFileSync(file));
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trimEnd();
const gitPaths = (...args) => git(...args).split('\0').filter(Boolean);
const integer = (value, minimum) => Number.isSafeInteger(value) && value >= minimum;

export function parseCases(bytes) {
  const cases = JSON.parse(bytes.toString());
  if (!Array.isArray(cases) || !cases.length) throw new Error('Cases must be a nonempty JSON array');
  const names = new Set();
  const modes = ['auto', 'neural', 'baseline'];
  return cases.map(config => {
    if (!config || typeof config !== 'object' || Array.isArray(config)) throw new Error('Invalid case');
    const allowed = ['name', 'clip', 'durationMs', 'warmupMs', 'mode', 'actions', 'expected'];
    if (Object.keys(config).some(key => !allowed.includes(key))) throw new Error(`Unknown case field: ${config.name}`);
    if (typeof config.name !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(config.name) || names.has(config.name)) {
      throw new Error('Case names must be unique filename-safe identifiers');
    }
    names.add(config.name);
    if (typeof config.clip !== 'string' || !/^\/media\/[a-zA-Z0-9][a-zA-Z0-9._/-]*$/.test(config.clip)) {
      throw new Error(`Invalid media path: ${config.name}`);
    }
    if (config.clip.slice(1).split('/').some(part => !part || part === '.' || part === '..')) throw new Error('Unsafe media path');
    if (!integer(config.durationMs, 1) || config.durationMs > 2147483647) throw new Error(`Invalid duration: ${config.name}`);
    const warmupMs = config.warmupMs ?? 3000;
    const mode = config.mode ?? 'auto';
    if (!integer(warmupMs, 3000) || warmupMs >= 20000 || !modes.includes(mode)) throw new Error(`Invalid warmup/mode: ${config.name}`);
    if (config.expected !== undefined && (!config.expected || typeof config.expected !== 'object' || Array.isArray(config.expected) ||
        Object.entries(config.expected).some(([key, value]) => !['width', 'height', 'fps'].includes(key) || !integer(value, 1)))) {
      throw new Error(`Invalid expected geometry/cadence: ${config.name}`);
    }
    if (config.actions !== undefined && !Array.isArray(config.actions)) throw new Error('Actions must be an array');
    let previousAt = -1;
    const actions = (config.actions ?? []).map(action => {
      if (!action || !integer(action.atMs, 0) || action.atMs <= previousAt || action.atMs >= config.durationMs) {
        throw new Error(`Actions must be strictly time-ordered within the window: ${config.name}`);
      }
      previousAt = action.atMs;
      const fields = { load: ['passes', 'frames', 'every'], force: ['on'], setMode: ['mode'], loseDevice: [] };
      if (!Object.hasOwn(fields, action.type) || Object.keys(action).some(key => !['type', 'atMs', ...fields[action.type]].includes(key))) {
        throw new Error(`Invalid action: ${config.name}`);
      }
      if (action.type === 'load') {
        if (!integer(action.passes, 0) || action.passes > 8 ||
            action.frames != null && !integer(action.frames, 1) || !integer(action.every ?? 1, 1)) throw new Error('Invalid load');
        return { ...action, frames: action.frames ?? null, every: action.every ?? 1 };
      }
      if (action.type === 'force' && typeof action.on !== 'boolean') throw new Error('force requires boolean on');
      if (action.type === 'setMode' && !modes.includes(action.mode)) throw new Error('Invalid setMode');
      return { ...action };
    });
    return { ...config, mode, warmupMs, actions };
  });
}

export function distribution(values) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  const quantile = fraction => {
    if (!sorted.length) return null;
    const position = (sorted.length - 1) * fraction;
    const lower = sorted[Math.floor(position)];
    return lower + (sorted[Math.ceil(position)] - lower) * (position % 1);
  };
  return { count: sorted.length, rejected: values.length - sorted.length,
    mean: sorted.length ? sorted.reduce((sum, value) => sum + value, 0) / sorted.length : null,
    p50: quantile(0.5), p90: quantile(0.9), p95: quantile(0.95), max: sorted.at(-1) ?? null };
}

function provenance(casesFile, mediaFile) {
  const head = git('rev-parse', 'HEAD');
  const files = gitPaths('ls-files', '-z', '--', ...sourcePaths);
  const source = Object.fromEntries(files.map(file => [file, digest(resolve(root, file))]));
  const untracked = gitPaths('ls-files', '--others', '--exclude-standard', '-z', '--', ...sourcePaths);
  const dirty = git('status', '--porcelain=v1', '--untracked-files=no');
  const issues = [];
  if (dirty) issues.push('Tracked worktree/index is not clean');
  if (untracked.length) issues.push(`Untracked executed source: ${untracked.join(', ')}`);
  if (!files.includes('tools/m9-live.mjs') || !files.includes('src/main.ts')) issues.push('Runner/main must be tracked in HEAD');
  for (const file of files) {
    try {
      if (hash(execFileSync('git', ['show', `${head}:${file}`], { cwd: root })) !== source[file]) issues.push(`Source differs from HEAD: ${file}`);
    } catch { issues.push(`Source absent from HEAD: ${file}`); }
  }
  const modelSha256 = digest(resolve(root, modelFile));
  if (modelSha256 !== fixedModelSha256) issues.push('Production model changed');
  if (git('rev-parse', 'HEAD') !== head) issues.push('HEAD changed while collecting provenance');
  return { at: new Date().toISOString(), head, source, dirty, untracked, issues, modelSha256,
    mediaSha256: digest(mediaFile), casesSha256: digest(casesFile) };
}

function changed(before, after) {
  return ['head', 'source', 'modelSha256', 'mediaSha256', 'casesSha256']
    .filter(key => JSON.stringify(before[key]) !== JSON.stringify(after[key])).map(key => `${key} changed`);
}

export function makePlan(casesFile, prefix) {
  const bytes = readFileSync(casesFile);
  return { casesSha256: hash(bytes), cases: parseCases(bytes).map(config => {
    const url = new URL('http://127.0.0.1:5173/');
    url.searchParams.set('mode', config.mode);
    url.searchParams.set('clip', config.clip);
    const diagnosticLoad = config.actions.some(action => action.type === 'load');
    if (diagnosticLoad) url.searchParams.set('runtime-bench', '1');
    return { config, url: url.href, diagnosticLoad,
      diagnostics: diagnosticLoad ? 'inner pass diagnostics disabled by main load wrapper' : 'production-normal inner diagnostics',
      mediaFile: resolve(root, `public${config.clip}`), output: resolve(`${prefix}-${config.name}.json.gz`),
      screenshot: resolve(root, '.cache/m9', `${basename(prefix)}-${config.name}.png`) };
  }) };
}

function installCapture() {
  const raw = { samples: [], frames: [], configurations: [], transitions: [], events: [], errors: [], refresh: [], actions: [] };
  const data = { installedAt: performance.now(), timeOrigin: performance.timeOrigin, attachedAt: null,
    phase: 'startup', raw, transitionGaps: [], firstConfigurationExcluded: null };
  const error = (kind, value) => raw.errors.push({ at: performance.now(), phase: data.phase, kind, message: String(value?.message ?? value) });
  const state = () => ({ visibility: document.visibilityState, focus: document.hasFocus() });
  const event = kind => raw.events.push({ at: performance.now(), phase: data.phase, kind, ...state() });
  document.addEventListener('visibilitychange', () => event('visibilitychange'));
  window.addEventListener('blur', () => event('blur'));
  window.addEventListener('focus', () => event('focus'));
  window.addEventListener('error', value => error('window.error', value.message));
  window.addEventListener('unhandledrejection', value => error('unhandledrejection', value.reason));
  event('document-start');
  const refresh = time => {
    raw.refresh.push(time);
    if (raw.refresh.length < 241) requestAnimationFrame(refresh);
  };
  requestAnimationFrame(refresh);
  const clone = value => structuredClone(value);
  let api;
  let transitionCursor = 0;
  let loadGeneration = 0;
  const quality = () => {
    const value = api.video.getVideoPlaybackQuality?.();
    return value ? [value.totalVideoFrames, value.droppedVideoFrames, value.corruptedVideoFrames] : [null, null, null];
  };
  const memory = () => {
    const neural = api.neural();
    return { at: performance.now(), actualTier: api.pipeline.currentUpscaler.neural ? 'neural' : 'baseline',
      stageId: api.pipeline.currentUpscaler.id, neuralId: neural?.id ?? null,
      precision: neural?.resolvedPrecision ?? null, memory: clone(neural?.memoryReport ?? null) };
  };
  const collectTransitions = controller => {
    const first = controller.transitionCount - controller.transitions.length + 1;
    if (first > transitionCursor + 1) data.transitionGaps.push({ from: transitionCursor + 1, to: first - 1, at: performance.now() });
    controller.transitions.forEach((transition, index) => {
      const id = first + index;
      if (id > transitionCursor) raw.transitions.push({ id, ...transition });
    });
    transitionCursor = controller.transitionCount;
  };
  const checkpoint = () => {
    const runtime = api.snapshot();
    collectTransitions(runtime.controller);
    return { at: performance.now(), runtime: clone(runtime), environment: state(), decoderQuality: quality(),
      memory: memory(), pipeline: clone(api.pipeline.stats(performance.now())),
      status: { text: document.getElementById('status')?.textContent, level: document.getElementById('status')?.dataset.level } };
  };
  function attach(value) {
    if (api) throw new Error('Runtime hook replaced after capture installation');
    api = value;
    data.attachedAt = performance.now();
    const initial = api.snapshot();
    loadGeneration = initial.session.loadGeneration;
    data.firstConfigurationExcluded = initial.session.configurationCount > 0;
    data.configurationBeforeAttach = { count: initial.session.configurationCount,
      latest: clone(api.driver.session.latestConfiguration ?? null),
      scope: 'Only the latest prior configuration can be recovered; never included as a captured configure callback.' };
    if (!Number.isSafeInteger(api.pipeline.submissionSequence)) throw new Error('Submission sequence diagnostics unavailable');
    collectTransitions(initial.controller);
    const gpu = api.pipeline.gpu;
    if (!gpu?.device || !gpu?.capabilities || !gpu?.adapterReport) throw new Error('Pipeline GPU diagnostics unavailable');
    data.environment = { userAgent: navigator.userAgent, adapter: clone(gpu.adapterReport),
      timestampQuery: gpu.capabilities.timestampQuery, hardware: gpu.adapterReport.fallbackAdapter === false,
      capabilities: clone(gpu.capabilities), screen: { width: screen.width, height: screen.height,
        availWidth: screen.availWidth, availHeight: screen.availHeight, devicePixelRatio },
      viewport: { width: innerWidth, height: innerHeight }, displayRefreshHz: 'not measured' };
    gpu.device.addEventListener('uncapturederror', value => error('gpu.uncapturederror', value.error));
    void gpu.device.lost.then(value => error('gpu.device.lost', `${value.reason}: ${value.message}`));
    api.video.addEventListener('loadstart', () => { loadGeneration++; event('media-loadstart'); });
    api.video.addEventListener('error', () => error('video.error', api.video.error));
    for (const name of ['pause', 'playing', 'seeking', 'seeked', 'ended', 'ratechange']) api.video.addEventListener(name, () => event(`video-${name}`));
    const previous = { sample: api.driver.onSample, frame: api.driver.onFrame,
      configure: api.driver.onConfigure, change: api.driver.onChange };
    api.driver.onSample = sample => {
      raw.samples.push({ ...sample, source: { ...sample.source }, phase: data.phase });
      previous.sample?.(sample);
    };
    api.driver.onFrame = tick => {
      raw.frames.push([tick.now, tick.mediaTime, tick.presentedDelta, tick.now - tick.presentationTime,
        tick.expectedDisplayTime - tick.now, api.pipeline.currentUpscaler.neural ? 'neural' : 'baseline',
        ...quality(), loadGeneration, api.pipeline.timingGeneration, api.pipeline.submissionSequence,
        performance.now(), tick.decodeLatencyMs]);
      previous.frame?.(tick);
    };
    api.driver.onConfigure = config => {
      const record = { at: performance.now(), phase: data.phase, ...clone(config), memoryAtCallback: memory() };
      raw.configurations.push(record);
      queueMicrotask(() => { record.memoryAfterTask = memory(); });
      previous.configure?.(config);
    };
    api.driver.onChange = controller => { collectTransitions(controller); previous.change?.(controller); };
    data.attached = checkpoint();
  }
  Object.defineProperty(window, 'aethervsrRuntime', { configurable: true, get: () => api, set: attach });
  window.__m9Live = { data, checkpoint, error, get api() { return api; } };
}

async function measure(settings) {
  const capture = window.__m9Live;
  const { data, checkpoint, error } = capture;
  const wait = milliseconds => new Promise(resolveWait => setTimeout(resolveWait, milliseconds));
  const deadline = performance.now() + 20000;
  const timers = [];
  let progress;
  let boundary;
  let drained;
  const bounded = async (promise, milliseconds, label) => {
    let timer;
    try { return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} timed out`)), milliseconds);
    })]); } finally { clearTimeout(timer); }
  };
  const ready = () => {
    const api = capture.api;
    if (!api || !data.environment) return false;
    const snapshot = api.snapshot();
    if (snapshot.controller.state === 'failed' || api.pipeline.error) throw new Error('Runtime failed during readiness');
    const expected = settings.expected ?? {};
    return api.video.readyState >= 2 && !api.video.paused && snapshot.running && snapshot.session.active &&
      snapshot.session.framesRendered > 0 && data.raw.samples.length > 0 &&
      ['stable', 'fallback', 'manual-baseline'].includes(snapshot.controller.state) &&
      snapshot.actualTier === snapshot.controller.tier &&
      (expected.width === undefined || api.video.videoWidth === expected.width) &&
      (expected.height === undefined || api.video.videoHeight === expected.height) &&
      (expected.fps === undefined || snapshot.controller.fps === expected.fps);
  };
  const awaitReady = async () => {
    while (!ready()) {
      if (performance.now() >= deadline) throw new Error('Stable/eligible runtime readiness timed out after 20 s');
      await wait(25);
    }
  };
  try {
    await awaitReady();
    if (!data.environment.timestampQuery || !data.environment.hardware) throw new Error('Hardware timestamp-query required');
    data.warmupStartedAt = performance.now();
    await wait(settings.warmupMs);
    await awaitReady();
    data.setup = checkpoint();
    if (!data.setup.environment.focus || data.setup.environment.visibility !== 'visible') throw new Error('Setup not visible/focused');
    const api = capture.api;
    api.pipeline.stop();
    await bounded(api.pipeline.drainTimings(), 10000, 'Initial timestamp drain');
    data.startup = checkpoint();
    data.startup.rawCounts = Object.fromEntries(Object.entries(data.raw).map(([key, rows]) => [key, rows.length]));
    data.resetRequestedAt = performance.now();
    api.reset();
    data.resetCompletedAt = performance.now();
    data.phase = 'measured';
    data.startedAt = performance.now();
    data.initial = checkpoint();
    data.plannedEndAt = data.startedAt + settings.durationMs;
    api.pipeline.start();
    data.restartedAt = performance.now();
    for (const action of settings.actions) {
      timers.push(setTimeout(() => {
        const entry = { ...action, scheduledAt: data.startedAt + action.atMs, at: performance.now() };
        data.raw.actions.push(entry);
        try {
          if (action.type === 'load') api.load(action.passes, action.frames ?? Infinity, action.every);
          else if (action.type === 'setMode') api.setMode(action.mode);
          else if (action.type === 'force') api.force(action.on);
          else api.loseDevice();
          entry.completedAt = performance.now();
          entry.controller = structuredClone(api.snapshot().controller);
        } catch (value) { entry.error = String(value); error('action', value); }
      }, Math.max(0, data.startedAt + action.atMs - performance.now())));
    }
    progress = setInterval(() => console.log('M9_LIVE_PROGRESS ' + JSON.stringify({ name: settings.name,
      elapsedMs: performance.now() - data.startedAt,
      frames: data.raw.frames.length - data.startup.rawCounts.frames,
      samples: data.raw.samples.length - data.startup.rawCounts.samples,
      actualTier: api.pipeline.currentUpscaler.neural ? 'neural' : 'baseline' })), 60000);
    await wait(Math.max(0, data.plannedEndAt - performance.now()));
    data.endedAt = performance.now();
    boundary = checkpoint();
    data.sampleCountAtBoundary = data.raw.samples.length;
    api.pipeline.stop();
    data.stoppedAt = performance.now();
    data.phase = 'drain';
    timers.forEach(clearTimeout);
    clearInterval(progress);
    await bounded(api.pipeline.drainTimings(), 10000, 'Final timestamp drain');
    data.endOfDrainAt = performance.now();
    drained = checkpoint();
  } catch (value) {
    error('runner', value);
    timers.forEach(clearTimeout);
    clearInterval(progress);
    if (capture.api) {
      capture.api.pipeline.stop();
      try { await bounded(capture.api.pipeline.drainTimings(), 10000, 'Failure timestamp drain'); }
      catch (failure) { error('drain', failure); }
      data.endOfDrainAt = performance.now();
      try { drained = checkpoint(); } catch (failure) { error('snapshot', failure); }
    }
  }
  data.phase = 'stopped';
  return { ...data, boundary, drained, returnedAt: performance.now() };
}

export function summarize(result) {
  const { startedAt, endedAt, raw, initial, boundary, drained } = result;
  if (!initial || !boundary || !drained) return { measured: false, reason: 'No complete measurement boundary and drain' };
  const samples = raw.samples.filter(sample => sample.submittedAt >= startedAt && sample.submittedAt <= endedAt);
  const frames = raw.frames.filter(frame => frame[0] >= startedAt && frame[0] <= endedAt);
  const configurations = raw.configurations.filter(config => config.at >= startedAt && config.at <= endedAt);
  const before = initial.runtime.controller;
  const after = boundary.runtime.controller;
  const transitions = raw.transitions.filter(transition => transition.id > before.transitionCount && transition.atMs >= startedAt && transition.atMs <= endedAt);
  const session = drained.runtime.session;
  const timed = new Set(samples.map(sample => sample.sequence));
  const missingSequences = frames.filter(frame => !timed.has(frame[11])).map(frame => frame[11]);
  const gaps = [];
  const sequences = [...timed].sort((left, right) => left - right);
  for (let index = 1; index < sequences.length; index++) {
    if (sequences[index] > sequences[index - 1] + 1) gaps.push([sequences[index - 1] + 1, sequences[index] - 1]);
  }
  return { measured: true, units: 'ms unless named otherwise', quantiles: 'Exact linear interpolation at (n - 1) * p on raw values; no histogram rounding. Null means not measured.',
    elapsed: { requestedMs: result.plannedEndAt - startedAt, boundaryMs: endedAt - startedAt,
      boundaryLatenessMs: endedAt - result.plannedEndAt, drainMs: result.endOfDrainAt - endedAt,
      resetToDrainMs: result.endOfDrainAt - result.resetRequestedAt, sessionActiveMs: session.activeMs,
      sessionClockBeyondBoundaryMs: session.activeMs - (endedAt - startedAt) },
    counts: { rendered: session.framesRendered, presented: session.framesPresented, skipped: session.framesSkipped,
      decoderDrops: session.decoderDrops, decoderFrames: session.decoderFrames, decoderCorrupted: session.decoderCorrupted,
      qualitySamples: session.qualitySamples, qualityMissingSamples: session.qualityMissingSamples,
      qualityRejectedSamples: session.qualityRejectedSamples, rawFrames: frames.length, rawGpu: samples.length,
      sessionGpu: session.gpu.neural.count + session.gpu.baseline.count,
      gpuReceivedDuringDrain: raw.samples.slice(result.sampleCountAtBoundary).filter(sample => sample.submittedAt >= startedAt && sample.submittedAt <= endedAt).length },
    rates: { renderedFps: session.meanRenderedFps, presentedFps: session.meanPresentedFps,
      scope: 'Post-drain RuntimeSession counts / activeMs; includes initial reset/start gap and final drain, not just the requested timer duration.' },
    gpu: { all: distribution(samples.map(sample => sample.ms)),
      neural: distribution(samples.filter(sample => sample.neural).map(sample => sample.ms)),
      baseline: distribution(samples.filter(sample => !sample.neural).map(sample => sample.ms)),
      scope: 'Outer GPU timestamps bracket stage execution, including diagnostic load when enabled. SubmittedAt selects the window; delayed readbacks retained. Excludes decode/import/compositor and CPU encode time.' },
    callbackLatency: distribution(boundary.pipeline.clock === 'rvfc' ? frames.map(frame => frame[3]) : []),
    expectedDisplayDelay: distribution(boundary.pipeline.clock === 'rvfc' ? frames.map(frame => frame[4]) : []),
    configure: distribution(configurations.map(config => config.configureMs)),
    decoderEndpoints: { initial: initial.decoderQuality, firstFrame: frames[0]?.slice(6, 10) ?? null,
      lastFrame: frames.at(-1)?.slice(6, 10) ?? null, boundary: boundary.decoderQuality, drained: drained.decoderQuality },
    rawGaps: { missingSequences, internalGpuSequenceGaps: gaps, duplicateGpuSequences: samples.length - timed.size,
      scope: 'Untimed rendered submissions are retained as gaps, never imputed. Frame timingGeneration is observed after driver processing, not necessarily encode generation.' },
    controller: { initial: before, final: after, finalActualTier: boundary.runtime.actualTier,
      transitionCount: after.transitionCount - before.transitionCount, transitions,
      probeCount: after.probeCount - before.probeCount, failedProbeCount: after.failedProbeCount - before.failedProbeCount,
      fallbackMs: after.fallbackMs - before.fallbackMs,
      falseFallback: { value: null, status: 'not inferred; requires separate analysis' },
      scope: 'Counters subtract initial controller snapshot; final state is sampled BEFORE stop/drain. Raw transitions additionally filter by absolute time and transition id.' },
    refreshDispatchIntervals: distribution(raw.refresh.slice(1).map((time, index) => time - raw.refresh[index])),
    refreshScope: 'First 241 document rAF callbacks only; dispatch cadence is not a measured display refresh rate.' };
}

function invalidReasons(result, summary, errors) {
  const reasons = [];
  if (!summary.measured) reasons.push('Incomplete window');
  if (!result.environment?.timestampQuery || !result.environment?.hardware) reasons.push('Hardware GPU timestamps unavailable');
  if (errors.length || result.raw.errors.length) reasons.push('Browser/page/GPU/runner errors');
  if (result.raw.events.some(event => event.kind === 'blur' || event.kind === 'visibilitychange' && event.visibility !== 'visible')) {
    reasons.push('Blur/hidden event since document installation, including startup and drain');
  }
  for (const key of ['setup', 'initial', 'boundary', 'drained']) {
    const point = result[key];
    if (!point?.environment.focus || point.environment.visibility !== 'visible') reasons.push(`${key} not focused/visible`);
    if (point?.status.level === 'error' || point?.runtime.controller.state === 'failed') reasons.push(`${key} runtime failed`);
  }
  if (result.setup?.pipeline.clock !== 'rvfc') reasons.push('Real video-frame callback metadata unavailable');
  if (summary.measured) {
    if (!summary.counts.rawGpu || !summary.counts.rendered) reasons.push('No measured frames/timestamps');
    if (summary.gpu.all.rejected || result.raw.samples.some(sample => sample.ms < 0)) reasons.push('Invalid raw GPU durations');
    if (summary.counts.rawFrames !== summary.counts.rendered || summary.counts.rawGpu !== summary.counts.sessionGpu) reasons.push('Raw/session counts disagree');
    if (summary.controller.transitionCount !== summary.controller.transitions.length) reasons.push('Transition count/capture mismatch');
    if (summary.rawGaps.duplicateGpuSequences) reasons.push('Duplicate GPU submission sequences');
    if (result.raw.actions.length !== result.expectedActions) reasons.push('Scheduled action did not execute');
  }
  return reasons;
}

async function bounded(promise, milliseconds, label) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${milliseconds} ms`)), milliseconds);
  })]); } finally { clearTimeout(timer); }
}

function machineEnvironment() {
  const hardware = JSON.parse(execFileSync('system_profiler', ['SPHardwareDataType', 'SPDisplaysDataType', '-json'], { encoding: 'utf8' }));
  return { machine: hardware.SPHardwareDataType.map(row => ({ name: row.machine_name, chip: row.chip_type, memory: row.physical_memory })),
    displays: hardware.SPDisplaysDataType.map(row => ({ gpu: row.sppci_model,
      displays: row.spdisplays_ndrvs?.map(display => ({ name: display._name, resolution: display._spdisplays_resolution,
        refresh: display.spdisplays_refreshRate ?? display._spdisplays_refreshRate ?? 'not measured' })) })),
    os: execFileSync('sw_vers', ['-productVersion'], { encoding: 'utf8' }).trim(),
    osBuild: execFileSync('sw_vers', ['-buildVersion'], { encoding: 'utf8' }).trim(), node: process.version, arch: process.arch };
}

function mediaMetadata(file) {
  try {
    return JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries',
      'stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate:format=duration', '-of', 'json', file], { encoding: 'utf8' }));
  } catch (error) { return { status: 'not measured', reason: error.message }; }
}

async function main(args) {
  const dryRun = args.includes('--dry-run');
  const positional = args.filter(arg => arg !== '--dry-run');
  if (positional.length !== 2 || positional.some(arg => arg.startsWith('--'))) {
    throw new Error('Usage: node tools/m9-live.mjs cases.json output-prefix [--dry-run]');
  }
  const [casesArgument, prefix] = positional;
  const casesFile = resolve(casesArgument);
  const plan = makePlan(casesFile, prefix);
  for (const entry of plan.cases) {
    if (!existsSync(entry.mediaFile)) throw new Error(`Missing media: ${entry.mediaFile}`);
    if (existsSync(entry.output) || existsSync(entry.screenshot)) throw new Error(`Refusing to overwrite ${entry.output} or ${entry.screenshot}`);
  }
  const first = provenance(casesFile, plan.cases[0].mediaFile);
  if (dryRun) {
    console.log(JSON.stringify({ dryRun: true, browserLaunched: false, filesWritten: false, head: first.head,
      casesSha256: plan.casesSha256, modelSha256: first.modelSha256, measurementReady: !first.issues.length,
      blockers: first.issues, cases: plan.cases }, null, 2));
    return;
  }
  if (first.issues.length) throw new Error(`Commit the executed sources before measurement: ${first.issues.join('; ')}`);
  process.env.PLAYWRIGHT_BROWSERS_PATH = resolve(root, '.cache/m9/browsers');
  const playwrightFile = resolve(root, '.cache/m9/node_modules/playwright/package.json');
  const playwright = { version: JSON.parse(readFileSync(playwrightFile, 'utf8')).version, packageSha256: digest(playwrightFile) };
  const { chromium } = await import(pathToFileURL(resolve(root, '.cache/m9/node_modules/playwright/index.mjs')).href);
  const environment = machineEnvironment();
  const browser = await chromium.launch({ headless: false, args: flags, timeout: 20000 });
  try {
    for (const entry of plan.cases) {
      const before = provenance(casesFile, entry.mediaFile);
      const earlyIssues = [...before.issues, ...changed({ ...first, mediaSha256: before.mediaSha256 }, before)];
      if (before.casesSha256 !== plan.casesSha256) earlyIssues.push('Cases changed since parsing');
      if (earlyIssues.length) throw new Error(earlyIssues.join('; '));
      const page = await browser.newPage({ viewport: { width: 1200, height: 820 } });
      const errors = [];
      const recordError = (kind, message) => errors.push({ at: new Date().toISOString(), kind, message });
      page.on('pageerror', error => recordError('pageerror', error.message));
      page.on('crash', () => recordError('crash', 'Page crashed'));
      page.on('requestfailed', request => recordError('requestfailed', `${request.url()}: ${request.failure()?.errorText}`));
      page.on('response', response => { if (response.status() >= 400) recordError('http', `${response.status()} ${response.url()}`); });
      page.on('console', message => {
        if (message.text().startsWith(progressPrefix)) console.log(message.text());
        else if (message.type() === 'error') recordError('console.error', message.text());
      });
      let result = { raw: { samples: [], frames: [], configurations: [], transitions: [], events: [], errors: [], actions: [], refresh: [] } };
      console.log(`START ${entry.config.name} ${new Date().toISOString()} ${entry.url}`);
      try {
        await page.addInitScript(installCapture);
        await page.goto(entry.url, { waitUntil: 'domcontentloaded', timeout: 20000 });
        await page.bringToFront();
        result = await bounded(page.evaluate(measure, entry.config), entry.config.durationMs + 65000, 'Page measurement');
      } catch (error) {
        recordError('runner', error.message);
        try {
          result = await bounded(page.evaluate(async () => {
            const capture = window.__m9Live;
            if (!capture) return null;
            capture.api?.pipeline.stop();
            return structuredClone(capture.data);
          }), 3000, 'Partial evidence capture') ?? result;
        } catch (failure) { recordError('capture', failure.message); }
      }
      result.expectedActions = entry.config.actions.length;
      const summary = summarize(result);
      const reasons = invalidReasons(result, summary, errors);
      let after;
      try { after = provenance(casesFile, entry.mediaFile); reasons.push(...after.issues, ...changed(before, after)); }
      catch (error) { reasons.push(`Post-run provenance failed: ${error.message}`); }
      let screenshot = null;
      try {
        if (!result.drained) throw new Error('No completed drain; screenshot omitted');
        mkdirSync(dirname(entry.screenshot), { recursive: true });
        writeFileSync(entry.screenshot, await bounded(page.screenshot({ timeout: 5000 }), 6000, 'Stopped screenshot'), { flag: 'wx' });
        screenshot = entry.screenshot;
      } catch (error) { reasons.push(`Screenshot: ${error.message}`); }
      const artifact = { schema: 'aethervsr.m9-live/1', phase: 'main-index-live', config: entry.config,
        url: entry.url, diagnosticLoad: entry.diagnosticLoad, diagnostics: entry.diagnostics, valid: !reasons.length,
        invalidReasons: reasons, errors, provenance: { before, after }, measuredAt: new Date().toISOString(),
        ...environment, browser: browser.version(), executable: chromium.executablePath(), flags, headless: false, playwright,
        media: mediaMetadata(entry.mediaFile), screenshot,
        scope: { frameColumns: ['time', 'mediaTime', 'presentedDelta', 'callbackLatencyMs', 'expectedDelayMs', 'actualTier',
          'totalVideoFrames', 'droppedVideoFrames', 'corruptedVideoFrames', 'loadGeneration', 'generationAtObservation', 'submissionSequence', 'observedAt', 'decodeLatencyMs'],
        timebase: 'All page times except mediaTime are performance.now milliseconds; mediaTime is seconds. Node error times are ISO wall clock.',
        diagnosticsContract: 'DEV hook plus ordinary TypeScript-private fields pipeline.gpu, pipeline.submissionSequence and driver.session.latestConfiguration; capture fails if required provenance is absent.',
        capture: 'Hooks installed by intercepting the DEV API assignment before eligibility. All startup/configure/drain rows retained, no reset truncation. Prior configuration is explicitly marked if capture was late.',
        measurement: 'Stop acquisition only, drain timestamps, reset, restart. At actual timer boundary snapshot controller, stop acquisition, drain outside frame callbacks, snapshot session. No pixel readbacks during capture.',
        missing: 'Null means not measured, not zero. Raw startup errors and all blur/hidden events invalidate performance. No automatic performance verdicts or policy tuning.' },
        summary, result };
      mkdirSync(dirname(entry.output), { recursive: true });
      writeFileSync(entry.output, gzipSync(JSON.stringify(artifact)), { flag: 'wx' });
      console.log(JSON.stringify({ name: entry.config.name, valid: artifact.valid, invalidReasons: reasons,
        output: entry.output, screenshot: artifact.screenshot, summary }));
      await bounded(page.close(), 5000, 'Page close');
      if (!artifact.valid) throw new Error(`Invalid measurement ${entry.config.name}; raw evidence retained at ${entry.output}`);
    }
  } finally {
    try { await bounded(browser.close(), 5000, 'Browser close'); }
    catch (error) { console.error(`${error.message}; exiting runner to avoid a hanging transport. Browser cleanup unverified.`); process.exit(1); }
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1; });
}