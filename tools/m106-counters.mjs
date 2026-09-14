import assert from 'node:assert/strict';
import { counterDelta, distribution } from './m10-performance.mjs';

export const COLUMNS = ['observedAt', 'callbackNow', 'mediaTime', 'presentationTime', 'expectedDisplayTime',
  'presentedFrames', 'presentedDelta', 'qualityTotal', 'qualityDropped', 'currentTime', 'readyState', 'networkState', 'sourceGeneration'];

export function installNativeObserver(options) {
  const key = Symbol.for('aethervsr.m106.native');
  if (globalThis[key]) throw new Error('Duplicate native observer');
  if (!['none', 'lean', 'rich'].includes(options.mode)) throw new Error('Unknown native observer mode');
  const data = globalThis[key] = { mode: options.mode, rows: [], events: [], failures: [], raf: [], tasks: [], overflow: false,
    invalid: null, callbacks: options.mode === 'none' ? null : 0, presented: null, gaps: options.mode === 'none' ? null : 0 };
  let video, previous = null, generation = 0, active = false, frameHandle, rafHandle, timer, done;
  const listeners = [], observers = [];
  const push = (target, value) => { if (target.length >= 160000) data.overflow = true; else target.push(value); };
  const on = (target, type, callback) => { target.addEventListener(type, callback); listeners.push([target, type, callback]); };
  const quality = () => { const value = video.getVideoPlaybackQuality?.(); return value ? { total: value.totalVideoFrames, dropped: value.droppedVideoFrames, at: value.creationTime } : null; };
  const snapshot = () => ({ at: performance.now(), callbacks: data.callbacks, presented: data.presented, gaps: data.gaps,
    quality: quality(), currentTime: video.currentTime, duration: Number.isFinite(video.duration) ? video.duration : null,
    readyState: video.readyState, networkState: video.networkState, paused: video.paused, ended: video.ended, mediaError: video.error?.code ?? null,
    rate: video.playbackRate, source: video.currentSrc, generation, visibility: document.visibilityState, focused: document.hasFocus() });
  const taskEntries = entries => { for (const entry of entries) if (data.opening && entry.startTime >= data.opening.at && (!data.closing || entry.startTime < data.closing.at)) push(data.tasks, { at: entry.startTime, ms: entry.duration, name: entry.name }); };
  const finish = reason => {
    if (!active) return;
    data.closing = snapshot(); active = false; data.invalid ??= reason ?? null;
    window.dispatchEvent(new Event('aethervsr:m106:end'));
    clearTimeout(timer);
    if (frameHandle !== undefined) video.cancelVideoFrameCallback(frameHandle);
    if (rafHandle !== undefined) cancelAnimationFrame(rafHandle);
    for (const [target, type, callback] of listeners) target.removeEventListener(type, callback);
    setTimeout(() => {
      for (const observer of observers) { taskEntries(observer.takeRecords()); observer.disconnect(); }
      done();
    }, 0);
  };
  data.done = new Promise(resolve => { done = resolve; });
  data.start = durationMs => {
    if (!video || active || data.opening) throw new Error('Observer is not startable');
    if (document.visibilityState !== 'visible' || !document.hasFocus() || video.paused || video.readyState < 2) throw new Error('Native foreground decoded playback required');
    if (options.mode !== 'none' && data.callbacks < 2) throw new Error('Need opening callback baseline');
    data.opening = snapshot(); active = true;
    window.dispatchEvent(new Event('aethervsr:m106:start'));
    timer = setTimeout(() => finish(), durationMs);
    return data.opening;
  };
  on(document, 'DOMContentLoaded', () => {
    video = document.querySelector('video');
    if (!video || !video.getVideoPlaybackQuality || !video.requestVideoFrameCallback) throw new Error('Native video counters unavailable');
    data.ready = true;
    if (options.mode !== 'none') {
      const frame = (now, metadata) => {
        const observedAt = performance.now(), delta = previous === null ? null : metadata.presentedFrames - previous;
        data.callbacks++; data.presented = metadata.presentedFrames; previous = metadata.presentedFrames;
        if (active) {
          if (delta === null || delta < 1) { data.invalid = 'Nonmonotonic native callback counter'; finish(data.invalid); return; }
          data.gaps += Math.max(0, delta - 1);
          const counts = quality();
          push(data.rows, [observedAt, now, metadata.mediaTime, metadata.presentationTime, metadata.expectedDisplayTime ?? null,
            metadata.presentedFrames, delta, counts?.total ?? null, counts?.dropped ?? null, video.currentTime,
            video.readyState, video.networkState, generation]);
        }
        frameHandle = video.requestVideoFrameCallback(frame);
      };
      frameHandle = video.requestVideoFrameCallback(frame);
    }
    for (const type of ['waiting', 'stalled', 'seeking', 'seeked', 'ended', 'playing', 'loadstart', 'error', 'ratechange', 'pause', 'play']) {
      on(video, type, () => {
        if (type === 'loadstart') { generation++; previous = null; }
        if (!active) return;
        push(data.events, { type, ...snapshot() });
        if (type === 'error') push(data.failures, 'Original media error');
        if (['loadstart', 'ratechange', 'pause', 'play'].includes(type) && !video.error) finish(`Media event: ${type}`);
      });
    }
    for (const [target, type] of [[window, 'blur'], [window, 'pagehide'], [window, 'resize'], [document, 'visibilitychange']]) {
      on(target, type, () => { if (active) { push(data.events, { type, ...snapshot() }); finish(`Integrity event: ${type}`); } });
    }
    if (options.mode === 'rich') {
      let lastRaf = null;
      const raf = now => {
        if (active) {
          push(data.raf, [performance.now(), now, lastRaf === null ? null : now - lastRaf]);
          lastRaf = now;
          const marker = document.getElementById('m106-marker');
          if (marker) marker.style.transform = `translateX(${Math.floor((now - data.opening.at) * 0.12) % 600}px)`;
        }
        rafHandle = requestAnimationFrame(raf);
      };
      rafHandle = requestAnimationFrame(raf);
      data.supportedEntryTypes = PerformanceObserver.supportedEntryTypes;
      if (data.supportedEntryTypes.includes('longtask')) {
        const observer = new PerformanceObserver(list => taskEntries(list.getEntries()));
        observer.observe({ type: 'longtask' }); observers.push(observer);
      }
    }
  });
}

export function installRuntimeCounters() {
  const manager = globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)];
  if (!manager?.attachment) throw new Error('Actual extension attachment required');
  const attachment = manager.attachment, pipeline = attachment.pipeline, driver = attachment.driver;
  const data = globalThis[Symbol.for('aethervsr.m106.runtime')] = { frames: [], states: [], attempts: pipeline ? 0 : null, overflow: false,
    teardownSnapshot: () => attachment.snapshot() };
  let active = false, lastState = null;
  const snapshot = () => ({ at: performance.now(), status: manager.status(), runtime: driver?.snapshot() ?? null,
    attempts: data.attempts, pipelineError: pipeline?.error ? String(pipeline.error) : null, sameAttachment: manager.attachment === attachment });
  const previousTick = pipeline?.onTick, previousFrame = pipeline?.onFrame, previousChange = driver?.onChange;
  if (pipeline) {
    pipeline.onTick = function(tick) { if (active) data.attempts++; return previousTick.call(this, tick); };
    pipeline.onFrame = function(tick) {
      const at = performance.now(), neural = Number(pipeline.currentUpscaler.neural);
      try { previousFrame?.call(this, tick); }
      finally {
        if (active) { if (data.frames.length < 160000) data.frames.push([at, tick.presentedDelta, neural]); else data.overflow = true; }
      }
    };
    driver.onChange = function(state) {
      previousChange?.call(this, state);
      if (active && `${state.state}:${state.tier}` !== lastState) { lastState = `${state.state}:${state.tier}`; data.states.push([performance.now(), state.state, state.tier]); }
    };
  }
  const start = () => {
    data.opening = snapshot();
    const state = data.opening.runtime?.controller;
    lastState = state ? `${state.state}:${state.tier}` : null;
    active = true;
  };
  const end = () => {
    data.closing = snapshot(); active = false;
    if (pipeline && attachment.pipeline === pipeline) { pipeline.onTick = previousTick; pipeline.onFrame = previousFrame; driver.onChange = previousChange; }
    window.removeEventListener('aethervsr:m106:start', start); window.removeEventListener('aethervsr:m106:end', end);
  };
  window.addEventListener('aethervsr:m106:start', start); window.addEventListener('aethervsr:m106:end', end);
  return { pipeline: !!pipeline, controller: !!driver, status: manager.status() };
}

const ratio = (numerator, denominator) => Number.isFinite(numerator) && numerator !== null && Number.isFinite(denominator) && denominator > 0 ? 100 * numerator / denominator : null;
const session = boundary => boundary?.runtime?.session;

export function summarizeNative(raw) {
  const data = raw.native, durationMs = data.closing.at - data.opening.at;
  const callbacks = counterDelta(data.opening.callbacks, data.closing.callbacks);
  const presented = counterDelta(data.opening.presented, data.closing.presented);
  const gaps = counterDelta(data.opening.gaps, data.closing.gaps);
  const qualityTotal = counterDelta(data.opening.quality?.total, data.closing.quality?.total);
  const qualityDrops = counterDelta(data.opening.quality?.dropped, data.closing.quality?.dropped);
  const runtime = raw.runtime;
  const submitted = counterDelta(session(runtime?.opening)?.framesRendered, session(runtime?.closing)?.framesRendered);
  const runtimePresented = counterDelta(session(runtime?.opening)?.framesPresented, session(runtime?.closing)?.framesPresented);
  const runtimeSkipped = counterDelta(session(runtime?.opening)?.framesSkipped, session(runtime?.closing)?.framesSkipped);
  const runtimeDecoderDrops = counterDelta(session(runtime?.opening)?.decoderDrops, session(runtime?.closing)?.decoderDrops);
  const attempts = counterDelta(runtime?.opening?.attempts, runtime?.closing?.attempts);
  const rate = count => count !== null && durationMs > 0 ? count * 1000 / durationMs : null;
  const window = (start, end) => {
    const rows = data.rows.filter(row => row[0] >= start && row[0] < end);
    const frames = runtime?.frames.filter(row => row[0] >= start && row[0] < end) ?? null;
    const milliseconds = end - start;
    return { durationMs: milliseconds, callbacks: rows.length, presented: rows.reduce((sum, row) => sum + row[6], 0),
      nativeCallbackFps: rows.length * 1000 / milliseconds, nativePresentedFps: rows.reduce((sum, row) => sum + row[6], 0) * 1000 / milliseconds,
      submitted: frames?.length ?? null, renderedFps: frames ? frames.length * 1000 / milliseconds : null,
      runtimePresentedFps: frames ? frames.reduce((sum, row) => sum + row[1], 0) * 1000 / milliseconds : null,
      scope: 'Half-open native rVFC-entry rows and runtime post-submit hook-entry rows, respectively. The first presented delta may include an interval preceding the slice; row counts do not. Not physical display count.' };
  };
  return { durationMs, callbacks, presented, gaps, qualityTotal, qualityDrops,
    nativeCombinedPercent: gaps !== null && qualityDrops !== null ? ratio(gaps + qualityDrops, presented) : null,
    historicalRuntimeCombinedPercent: runtimeSkipped !== null && qualityDrops !== null ? ratio(runtimeSkipped + qualityDrops, runtimePresented) : null,
    runtimeSessionCombinedPercent: runtimeSkipped !== null && runtimeDecoderDrops !== null ? ratio(runtimeSkipped + runtimeDecoderDrops, runtimePresented) : null,
    runtimeDecoderDrops,
    qualityDropPercent: ratio(qualityDrops, qualityTotal), callbackGapPercent: ratio(gaps, presented),
    nativeCallbackFps: rate(callbacks), nativePresentedFps: rate(presented), qualityTotalFps: rate(qualityTotal),
    submitted, runtimePresented, runtimeSkipped, renderedFps: rate(submitted), runtimePresentedFps: rate(runtimePresented),
    attempts, submissionDeficit: attempts !== null && submitted !== null ? attempts - submitted : null,
    callbackBoundaryResidual: presented !== null && callbacks !== null && gaps !== null ? presented - callbacks - gaps : null,
    runtimeNativeAlignment: runtimePresented !== null ? { presented: runtimePresented - presented, skipped: runtimeSkipped - gaps, submittedMinusCallbacks: submitted - callbacks } : null,
    callbackIntervalsMs: distribution(data.rows.slice(1).map((row, index) => row[0] - data.rows[index][0])),
    latencyMs: distribution(data.rows.map(row => row[0] - row[3])), rafIntervalsMs: distribution(data.raf?.map(row => row[2]) ?? []),
    first120: data.mode !== 'none' && durationMs >= 240000 ? window(data.opening.at, data.opening.at + 120000) : null,
    last120: data.mode !== 'none' && durationMs >= 240000 ? window(data.closing.at - 120000, data.closing.at) : null,
    importedFrames: null, uniquePhysicalLostFrames: null, counterOverlap: null,
    scopes: {
      nativeCombinedPercent: 'Native callback gaps plus boundary-read quality drops, divided by metadata presented delta. Potentially overlapping observations, not unique lost frames.',
      historicalRuntimeCombinedPercent: 'Historical M10/M10.5 capture formula: RuntimeSession skipped delta plus common boundary-read quality drops, divided by RuntimeSession presented delta.',
      runtimeSessionCombinedPercent: 'Exact RuntimeSession accumulation: skipped delta plus decoderDrops delta, divided by presented delta. Quality is sampled on accepted post-submit frames; snapshot() does not refresh quality.',
      attempts: 'Actual pipeline tick entries. Successful whole-window submissions use RuntimeSession frame deltas; interval slices use post-submit hook rows. Imports and physical presentation are not independently measured.',
    } };
}

export function validateNative(raw, expectedMs) {
  const data = raw.native, summary = summarizeNative(raw);
  assert.equal(data.invalid, null, `Invalid observation: ${data.invalid}`); assert.equal(data.overflow, false);
  assert(summary.durationMs >= expectedMs && summary.durationMs <= expectedMs + 1000, 'Native window length');
  for (const boundary of [data.opening, data.closing]) {
    assert.equal(boundary.visibility, 'visible'); assert.equal(boundary.focused, true);
    assert(boundary.paused === false || boundary === data.closing && boundary.mediaError != null, 'Unexpected original pause');
    assert.equal(boundary.rate, 1); assert(boundary.quality);
  }
  assert(data.opening.readyState >= 2);
  assert.equal(data.opening.source, data.closing.source); assert.equal(data.opening.generation, data.closing.generation);
  assert(summary.qualityTotal !== null && summary.qualityDrops !== null, 'Quality reset or unavailable');
  if (data.mode === 'none') assert.equal(summary.nativeCombinedPercent, null);
  else {
    assert.equal(summary.callbacks, data.rows.length); assert.equal(summary.callbackBoundaryResidual, 0);
    assert.equal(summary.gaps, data.rows.reduce((sum, row) => sum + row[6] - 1, 0));
    assert(data.rows.every(row => row[6] >= 1 && row[12] === data.opening.generation));
    assert(summary.presented !== null && summary.callbacks !== null, 'Native counter unavailable');
  }
  if (raw.runtime?.opening?.runtime) {
    assert.equal(raw.runtime.overflow, false); assert.equal(raw.runtime.frames.length, summary.submitted);
  }
  return summary;
}

export function pairedBounds(values, critical = 2.3533634348) {
  assert(values.length === 4 && values.every(Number.isFinite), 'Four finite run-level observations required');
  const mean = values.reduce((sum, value) => sum + value, 0) / 4;
  const deviation = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / 3);
  return { count: 4, values, mean, sd: deviation, min: Math.min(...values), max: Math.max(...values),
    lower95: mean - critical * deviation / 2, upper95: mean + critical * deviation / 2,
    twoSided95: [mean - 3.1824463053 * deviation / 2, mean + 3.1824463053 * deviation / 2],
    scope: 't bounds over four sessions/block contrasts, not frames. Normality/independence assumptions; one-sided95 equals two-sided90 endpoints.' };
}

export function observerComparison(results) {
  assert.equal(results.length,9,'Three frozen observer blocks required');
  const order=['none','lean','rich','lean','rich','none','rich','none','lean'];
  assert.deepEqual(results.map(result=>result.case.mode),order);
  assert(results.every(result=>result.case.phase==='observer'&&result.case.arm==='A'&&result.completion==='CAPTURED'));
  const blocks=[];
  for(let index=0;index<3;index++){
    const group=Object.fromEntries(results.slice(index*3,index*3+3).map(result=>[result.case.mode,result.summary]));
    for(const value of Object.values(group))assert(Number.isFinite(value.qualityTotalFps)&&Number.isFinite(value.qualityDropPercent));
    assert.equal(group.none.nativeCallbackFps,null);assert.equal(group.none.nativeCombinedPercent,null);
    blocks.push({block:index+1,
      leanMinusNoneQualityTotalFps:group.lean.qualityTotalFps-group.none.qualityTotalFps,
      richMinusNoneQualityTotalFps:group.rich.qualityTotalFps-group.none.qualityTotalFps,
      leanMinusNoneQualityDropPercentagePoints:group.lean.qualityDropPercent-group.none.qualityDropPercent,
      richMinusNoneQualityDropPercentagePoints:group.rich.qualityDropPercent-group.none.qualityDropPercent,
      richMinusLeanCallbackFps:group.rich.nativeCallbackFps-group.lean.nativeCallbackFps,
      richMinusLeanCombinedPercentagePoints:group.rich.nativeCombinedPercent-group.lean.nativeCombinedPercent});
  }
  return {blocks,observerCompatible:null,callbackDistortionBound:null,readinessNormalizationJustified:false,
    scope:'Descriptive paired blocks, not equivalence. Boundary quality measures exist in all three modes; no-observer callback loss is unobservable. Neither stable quality totals nor rich-minus-lean differences bound total callback distortion versus uninstrumented playback.'};
}

export function floorDecision(blocks, observerCompatible) {
  assert(blocks.length === 4);
  const missing = blocks.flatMap((block, index) => ['A', 'B', 'C', 'D'].flatMap(arm =>
    ['nativeCombinedPercent', arm === 'A' || arm === 'B' ? 'nativeCallbackFps' : 'renderedFps']
      .filter(field => !Number.isFinite(block?.[arm]?.[field])).map(field => `${index + 1}:${arm}:${field}`)));
  if (missing.length) return { case: 'CASE D', missing, readinessADRPermitted: false,
    scope: 'Missing or nonfinite run-level operands cannot support either harm or noninferiority.' };
  const native = pairedBounds(blocks.map(block => block.A.nativeCombinedPercent));
  const contrasts = {};
  for (const arm of ['B', 'C', 'D']) contrasts[arm] = {
    combined: pairedBounds(blocks.map(block => block[arm].nativeCombinedPercent - block.A.nativeCombinedPercent)),
    cadence: pairedBounds(blocks.map(block => block.A.nativeCallbackFps - (arm === 'B' ? block[arm].nativeCallbackFps : block[arm].renderedFps))),
  };
  const low = native.values.every(value => value <= 1) && native.upper95 <= 1;
  const high = native.values.every(value => value > 1) && native.lower95 > 1;
  const harm = ['C', 'D'].some(arm => contrasts[arm].combined.lower95 > 0.5 || contrasts[arm].cadence.lower95 > 0.5);
  const equivalent = Object.values(contrasts).every(value => value.combined.upper95 <= 0.5 && value.cadence.upper95 <= 0.5);
  const safety = blocks.every(block => ['C', 'D'].every(arm => block[arm].safety === true));
  const controls = blocks.every(block=>block.B.safety===true);
  return { case: low ? 'CASE A' : high && harm ? 'CASE B' : high && equivalent && safety && controls && observerCompatible === true ? 'CASE C' : 'CASE D',
    native, contrasts, high, low, harm, equivalent, safety, controls, observerCompatible,
    readinessADRPermitted: high && equivalent && safety && controls && observerCompatible === true && !harm,
    scope: 'Observable-counter degradation decision, not unique physical loss or automatic READY. Missing observer justification cannot authorize normalization.' };
}