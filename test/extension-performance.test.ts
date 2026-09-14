import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

function check(source: string): void {
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { Script, createContext } from 'node:vm';
    import { parseCases, quantile, distribution, counterDelta, windowRate,
      installVideoObserver, installRuntimeRecorder, summarizeCapture, validateCapture, windowSummary } from './tools/m10-performance.mjs';
    ${source}
    console.log('checked without browser');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 15000 });
  expect(output.trim()).toBe('checked without browser');
}

const fixture = `
  let clock = 0, nextTimer = 0, nextFrame = 0, pausedAt = null, stoppedAt = null, inactiveMs = 0, snapshots = 0;
  let drainResolve;
  const timers = new Map(), frameCallbacks = new Map(), calls = [];
  const target = () => {
    const listeners = new Map();
    return { addEventListener(type, callback) { const list = listeners.get(type) ?? []; list.push(callback); listeners.set(type, list); },
      removeEventListener(type, callback) { listeners.set(type, (listeners.get(type) ?? []).filter(value => value !== callback)); },
      dispatchEvent(event) { for (const callback of listeners.get(event.type) ?? []) callback(event); } };
  };
  const video = Object.assign(target(), { paused: false, playbackRate: 1, videoWidth: 1280, videoHeight: 720, currentSrc: 'http://127.0.0.1:5183/media/same.mp4',
    requestVideoFrameCallback(callback) { frameCallbacks.set(++nextFrame, callback); return nextFrame; },
    cancelVideoFrameCallback(handle) { frameCallbacks.delete(handle); },
    quality: { totalVideoFrames: 100, droppedVideoFrames: 2 },
    getVideoPlaybackQuality() { return this.quality; },
    getBoundingClientRect() { return { toJSON() { return { width: 640, height: 360 }; } }; },
    pause() { this.paused = true; pausedAt = clock; this.dispatchEvent({ type: 'pause' }); },
    play() { this.paused = false; this.dispatchEvent({ type: 'play' }); } });
  const counters = { framesRendered: 100, framesPresented: 110, framesSkipped: 10 };
  const controller = { state: 'stable', tier: 'neural', reason: 'test' };
  const pipeline = { currentUpscaler: { neural: true }, running: true, error: null,
    cpuFrame: { last: () => 0.25 }, stats: () => ({ cpuFrameMs: { last: 0.25 } }),
    stop() { this.running = false; stoppedAt = clock; },
    drainTimings: () => new Promise(done => { drainResolve = done; }) };
  const driver = { video, pipeline, snapshot() { snapshots++; return { session: { ...counters, activeMs: clock - inactiveMs },
      controller: { ...controller, activeMs: clock - inactiveMs }, actualTier: controller.tier, running: pipeline.running }; },
    syncActive() { calls.push('syncActive'); },
    onFrame(tick) { assert.equal(this, driver); calls.push('frame'); clock += 0.5; },
    onSample(sample) { assert.equal(this, driver); calls.push('sample'); },
    onChange(value) { assert.equal(this, driver); calls.push('change'); },
    onConfigure(value) { assert.equal(this, driver); calls.push('configure'); } };
  const originals = { onFrame: driver.onFrame, onSample: driver.onSample, onChange: driver.onChange, onConfigure: driver.onConfigure };
  pipeline.onFrame = function(tick) {
    assert.equal(this, pipeline); calls.push('driver-before'); clock += 1;
    driver.onFrame?.(tick);
    calls.push('driver-after'); clock += 2;
    driver.onChange?.(controller);
  };
  const originalPipelineFrame = pipeline.onFrame;
  const manager = { attachment: { driver, gpu: { adapterReport: { device: 'mock, not hardware evidence' } } },
    status: () => ({ details: { infrastructure: { discoveryCalls: 3, discoveryMs: 2, geometryCalls: 4, geometryMs: 1 },
      attachment: { infrastructure: { geometryCalls: 5, geometryTotalMs: 2, geometryMaxMs: 0.75 } } } }) };
  const document = Object.assign(target(), { visibilityState: 'visible', hasFocus: () => true,
    querySelector: () => video, querySelectorAll: () => [ {} ] });
  const window = target();
  const context = createContext({ document, window, console: { info() {} }, performance: { now: () => clock },
    innerWidth: 1200, innerHeight: 820, devicePixelRatio: 2, chrome: { runtime: { id: 'production' } },
    Event: class { constructor(type) { this.type = type; } }, queueMicrotask,
    setTimeout(callback, delay) { timers.set(++nextTimer, { callback, at: clock + delay }); return nextTimer; },
    clearTimeout(handle) { timers.delete(handle); }, setInterval() { return -1; }, clearInterval() {} });
  context.manager = manager;
  context.aethervsrRuntime = { driver };
  new Script('globalThis[Symbol.for("aethervsr.m10.document.production")] = manager').runInContext(context);
  new Script('(' + installVideoObserver.toString() + ')()').runInContext(context);
  document.dispatchEvent({ type: 'DOMContentLoaded' });
  const observer = new Script('globalThis[Symbol.for("aethervsr.m10.performance.video")]').runInContext(context);
  const nativeFrame = (presented = 10) => {
    const [handle, callback] = frameCallbacks.entries().next().value; frameCallbacks.delete(handle);
    callback(clock, { mediaTime: clock / 1000, presentedFrames: presented, presentationTime: clock - 1 });
  };
  nativeFrame();
  const install = (kind = 'extension-auto', durationMs = 30000) => {
    const options = { extension: kind.startsWith('extension-'), kind, warmupMs: 5000, durationMs };
    new Script('(' + installRuntimeRecorder.toString() + ')(' + JSON.stringify(options) + ')').runInContext(context);
    return new Script('globalThis[Symbol.for("aethervsr.m10.performance.runtime")]').runInContext(context);
  };
  const advance = async milliseconds => {
    clock = milliseconds;
    for (const [handle, timer] of [...timers]) if (timer.at <= clock && timers.has(handle)) { timers.delete(handle); timer.callback(); }
    for (let iteration = 0; iteration < 8; iteration++) await Promise.resolve();
  };
  const sample = (submittedAt, ms = 2) => ({ ms, submittedAt, resolvedAt: clock, sequence: 1, generation: 1, neural: true });
  const runtimeFrame = () => { counters.framesRendered++; counters.framesPresented++; pipeline.onFrame({ now: clock, mediaTime: clock / 1000, presentedDelta: 1, presentationTime: clock - 1, expectedDisplayTime: clock + 1 }); };
`;

describe('M10 performance protocol helpers (no browser)', () => {
  it('preserves supplied case order, rejects malformed input and limits long runs to the production extension', () => check(`
    assert.deepEqual(parseCases([{id:'pair3_b',kind:'harness'},{id:'pair3_a',kind:'extension-auto'}]).map(item => item.id), ['pair3_b','pair3_a']);
    assert.equal(parseCases([{id:'long',kind:'extension-auto'}])[0].durationMs, 600000);
    assert.equal(parseCases([{id:'control',kind:'installed-idle'}])[0].durationMs, 30000);
    for (const value of [[], {}, [null], [{id:'../x',kind:'harness'}], [{id:'x',kind:'fake'}],
      [{id:'x',kind:'harness',warmupMs:4999}], [{id:'x',kind:'harness',durationMs:NaN}],
      [{id:'x',kind:'harness',durationMs:600000}], [{id:'x',kind:'extension-auto',durationMs:600001}],
      [{id:'x',kind:'harness',unsafeGPU:true}], [{id:'x',kind:'harness'},{id:'x',kind:'harness'}]]) assert.throws(() => parseCases(value));
  `));

  it('uses finite-only linear percentiles and preserves unavailable values', () => check(`
    assert.equal(quantile([null, undefined, NaN, Infinity], .95), null);
    assert.equal(quantile([0, 10, 20, null, NaN], .95), 19);
    assert.equal(quantile([0, 10], .5), 5);
    assert.equal(quantile([7], .95), 7);
    assert.throws(() => quantile([1], Infinity));
    assert.equal(distribution(Array(160000).fill(2)).max, 2);
    assert.deepEqual(distribution([null]), {count:0,p50:null,p95:null,mean:null,max:null});
    assert.equal(windowRate(100, 1900, 30000), 60);
    for (const pair of [[null,0],[0,NaN],[10,9],[-1,3]]) assert.equal(counterDelta(...pair), null);
    assert.equal(windowRate(0,1,0), null);
  `));

  it('chains callbacks with their receiver, retains only in-window GPU samples and freezes boundaries before drain', () => check(fixture + `
    const record = install();
    driver.onSample(sample(100)); runtimeFrame();
    await advance(5000);
    assert.equal(record.frames.length, 0); assert.equal(record.samples.length, 0);
    await advance(5010); nativeFrame(11); runtimeFrame();
    driver.onSample(sample(4999)); driver.onSample(sample(5000));
    assert.equal(record.frames.length, 1); assert.equal(record.frames[0][7], .25); assert.equal(record.frames[0][8], .5);
    assert.deepEqual(Array.from(record.driverCpuRows[0]), [5010, 3.5]);
    assert.equal(record.samples.length, 1);
    await advance(35000);
    assert.equal(stoppedAt, 35000); assert.equal(pausedAt, null);
    assert.equal(pipeline.onFrame, originalPipelineFrame);
    video.quality.droppedVideoFrames = 99;
    await advance(35050); driver.onSample(sample(34999)); driver.onSample(sample(35000)); drainResolve();
    await advance(35050); await observer.done;
    assert.equal(record.ended, 35000); assert.equal(pausedAt, 35050);
    assert.equal(observer.events.length, 0);
    assert.equal(observer.closing.quality.dropped, 2);
    assert.equal(record.samples.length, 2);
    for (const name of Object.keys(originals)) assert.equal(driver[name], originals[name]);
    assert(calls.includes('syncActive'));
    const raw = {runtime: record, video: observer}; validateCapture(raw, {kind:'extension-auto',durationMs:30000});
    const summary = summarizeCapture(raw);
    assert.equal(summary.renderedFps, 1/30); assert.equal(summary.decoderDrops, 0);
    assert.equal(summary.driverCpuMs.mean, 3.5);
    assert.equal(summary.discovery.calls, 0); assert.equal(summary.discovery.meanMs, null);
    assert.equal(summary.attachmentGeometryLifetimeMaxMs, .75);
    record.closing.status.details.infrastructure.discoveryCalls += 2;
    record.closing.status.details.infrastructure.discoveryMs += 6;
    const infrastructure = summarizeCapture(raw).combinedInfrastructure;
    assert.equal(infrastructure.calls, 2); assert.equal(infrastructure.totalMs, 6);
    assert.equal(infrastructure.meanMs, 3); assert.equal(infrastructure.wallFraction, 6/30000);
    observer.events.push({type:'blur',visibility:'visible',focused:false});
    assert.throws(() => validateCapture(raw, {kind:'extension-auto',durationMs:30000}), /integrity/);
  `));

  it('reports native callbacks for controls without inventing pipeline FPS or GPU/CPU samples', () => check(fixture + `
    document.querySelectorAll = () => [];
    const record = install('no-extension');
    await advance(5000); await advance(5010); nativeFrame(12); await advance(35000); await observer.done;
    validateCapture({runtime:record, video:observer}, {kind:'no-extension',durationMs:30000});
    const summary = summarizeCapture({runtime:record,video:observer});
    assert.equal(summary.nativeCallbackFps, 1/30); assert.equal(summary.nativePresentedFps, 2/30);
    assert.equal(summary.renderedFps, null); assert.equal(summary.runtimePresentedFps, null);
    assert.equal(summary.gpuMs.p95, null); assert.equal(summary.coreCpuMs.mean, null);
    assert.deepEqual(summary.driverCpuMs, {count:0,p50:null,p95:null,mean:null,max:null});
    assert.equal(summary.discovery.totalMs, null); assert.equal(summary.pipelineLossPercent, null);
    assert.equal(summary.combinedInfrastructure.totalMs, null);
  `));

  it('times out a timing drain, pauses and restores original callbacks while marking the result invalid', () => check(fixture + `
    const record = install(); await advance(5000); await advance(35000); await advance(38000); await observer.done;
    assert.match(record.error, /GPU drain timeout/); assert.equal(pausedAt, 38000);
    assert.equal(driver.onSample, originals.onSample);
    assert.equal(pipeline.onFrame, originalPipelineFrame);
    assert.throws(() => validateCapture({runtime:record,video:observer}, {kind:'extension-auto',durationMs:30000}));
  `));

  it.each(['harness', 'extension-auto'])('times the enclosing %s handler without double-counting nested CPU or recorder work', kind => check(fixture + `
    const kind = ${JSON.stringify(kind)};
    const record = install(kind);
    await advance(5000);
    const openingSnapshots = snapshots;
    const bookkeeping = rows => {
      rows.push = function(...values) {
        assert.equal(clock, 5013.5 + bookkeepingWrites * 20);
        bookkeepingWrites++; clock += 20;
        return Array.prototype.push.apply(this, values);
      };
    };
    let bookkeepingWrites = 0;
    bookkeeping(record.driverCpuRows); bookkeeping(record.frames); bookkeeping(record.states); bookkeeping(record.configurations);
    controller.state = 'warming';
    await advance(5010); runtimeFrame();
    assert.equal(bookkeepingWrites, 3);
    assert.equal(record.driverCpuRows[0][1], 3.5);
    const summaryRows = record.frames;
    assert.equal(summaryRows[0][8], .5); assert.equal(summaryRows[0][7], .25);
    assert.equal(record.driverCpuRows.length, 1);
    assert.equal(snapshots, openingSnapshots);
    assert.deepEqual(calls.slice(-4), ['driver-before', 'frame', 'driver-after', 'change']);
    await advance(35000); drainResolve(); await advance(35001); await observer.done;
    assert.equal(pipeline.onFrame, originalPipelineFrame);
  `));

  it('requires the full requested active duration before drain without rounding a sub-600s window up', () => check(fixture + `
    const originalSnapshot = driver.snapshot;
    driver.snapshot = function() { const value = originalSnapshot.call(this); clock += .125; return value; };
    const record = install('extension-auto', 600000);
    await advance(5000); assert(record.started > record.opening.at);
    await advance(5010); nativeFrame(11); runtimeFrame();
    await advance(record.started + 600000);
    assert.equal(summarizeCapture({runtime:record,video:observer}).activeMs, 600000.125);
    inactiveMs = -3000;
    drainResolve(); await advance(record.ended + 2000); await observer.done;
    const raw = {runtime:record,video:observer}, item = {kind:'extension-auto',durationMs:600000};
    validateCapture(raw, item);
    assert.equal(summarizeCapture(raw).activeMs, 600000.125);
    assert.equal(summarizeCapture(raw).durationMs, 600000);
    record.closing.runtime.controller.activeMs = record.opening.runtime.controller.activeMs + 599999.95;
    assert.throws(() => validateCapture(raw, item), /active time/);
    assert.throws(() => windowSummary(raw, record.started, record.started + 120000), /active time/);
    delete record.opening.runtime.controller.activeMs; delete record.closing.runtime.controller.activeMs;
    validateCapture(raw, item);
    record.closing.runtime.session.activeMs = record.opening.runtime.session.activeMs + 599999.95;
    assert.throws(() => validateCapture(raw, item), /active time/);
    record.closing.runtime.session.activeMs = NaN;
    assert.throws(() => validateCapture(raw, item), /active time/);
  `));

  it('ends a long capture immediately after real focus loss without counting the remaining deadline', () => check(fixture + `
    const record = install('extension-auto', 600000);
    await advance(5000); await advance(10000); nativeFrame(11); runtimeFrame();
    const invalidAt = clock;
    document.hasFocus = () => false;
    window.dispatchEvent({type:'blur'});
    await advance(invalidAt);
    assert.equal(record.ended, invalidAt); assert.equal(stoppedAt, invalidAt);
    drainResolve(); await advance(invalidAt); await observer.done;
    assert.match(record.error, /integrity event: blur/);
    assert.equal(record.ended - record.started, invalidAt - 5000);
    assert.equal(observer.events.length, 1);
    assert.equal(observer.events[0].type, 'blur');
    assert.throws(() => validateCapture({runtime:record,video:observer}, {kind:'extension-auto',durationMs:600000}), /integrity/);
    const completedAt = record.completedAt;
    window.dispatchEvent({type:'blur'}); await advance(605000);
    assert.equal(record.completedAt, completedAt);
  `));

  it.each(['pause', 'play'])('rejects an observed %s even when both boundaries are playing', event => check(fixture + `
    const record = install(); await advance(5000); await advance(5010); nativeFrame(11); runtimeFrame();
    video.dispatchEvent({type:${JSON.stringify(event)}});
    await advance(35000); drainResolve(); await advance(35001); await observer.done;
    assert.equal(observer.events.length, 1);
    assert.equal(observer.events[0].type, ${JSON.stringify(event)});
    const raw = {runtime:record,video:observer};
    assert.throws(() => validateCapture(raw, {kind:'extension-auto',durationMs:30000}), /integrity/);
    assert.throws(() => windowSummary(raw, 5000, 15000), /integrity/);
  `));

  it('slices first/last shader samples by submission and frame/CPU rows by observation with outward decoder coverage', () => check(fixture + `
    const record = install('extension-auto', 600000); await advance(5000);
    for (const [at, total, dropped, presented] of [[5010,101,2,11],[124990,140,3,12],[125000,150,4,13],
      [484990,700,7,14],[485010,720,8,15],[604990,850,9,16]]) {
      await advance(at); video.quality.totalVideoFrames = total; video.quality.droppedVideoFrames = dropped;
      nativeFrame(presented); runtimeFrame();
    }
    await advance(605000);
    driver.onSample(sample(485000, 30));
    driver.onSample({...sample(5000, 2), neural:false});
    driver.onSample(sample(124999, 6));
    driver.onSample(sample(125000, 1000));
    driver.onSample(sample(484999, 2000));
    driver.onSample(sample(604999, 50));
    driver.onSample(sample(605000, 9999));
    await advance(605050); drainResolve(); await advance(605050); await observer.done;
    const raw = {runtime:record,video:observer};
    validateCapture(raw, {kind:'extension-auto',durationMs:600000});
    const before = JSON.stringify(raw);
    const summary = summarizeCapture(raw), first = summary.windowStats.first120s, last = summary.windowStats.last120s;
    assert.equal(JSON.stringify(raw), before);
    assert.equal(first.durationMs, 120000); assert.equal(last.activeMs, 120000);
    assert.equal(first.started, 5000); assert.equal(first.ended, 125000);
    assert.equal(last.started, 485000); assert.equal(last.ended, 605000);
    assert.deepEqual(first.coverage.frames.rowRange, [0,2]); assert.deepEqual(last.coverage.frames.rowRange, [4,6]);
    assert.deepEqual(first.coverage.driverCpuRows.rowRange, [0,2]); assert.deepEqual(last.coverage.driverCpuRows.rowRange, [4,6]);
    assert.deepEqual(first.coverage.video.rowRange, [0,2]);
    assert.equal(first.coverage.frames.firstObservedAt, 5010); assert.equal(last.coverage.frames.lastObservedAt, 604990);
    assert.deepEqual(first.coverage.samples.rowIndices, [1,2]); assert.deepEqual(last.coverage.samples.rowIndices, [0,5]);
    assert.deepEqual(first.coverage.samples.submittedAt, [5000,124999]);
    assert.deepEqual(last.coverage.samples.resolvedAt, [605000,605000]);
    assert.equal(first.gpuMs.mean, 4); assert.equal(first.neuralGpuMs.mean, 6); assert.equal(first.baselineGpuMs.mean, 2);
    assert.equal(last.gpuMs.mean, 40); assert.equal(last.baselineGpuMs.mean, null);
    assert.equal(first.driverCpuMs.count, 2); assert.equal(last.driverCpuMs.mean, 3.5);
    assert.equal(first.nativeCallbacks, 2); assert.equal(last.rendered, 2); assert.equal(first.renderedFps, 2/120);
    assert.equal(first.decoder.opening.rowIndex, null); assert.equal(first.decoder.closing.rowIndex, 2);
    assert.equal(first.decoder.frames, 50); assert.equal(first.decoder.drops, 2);
    assert.equal(last.decoder.opening.at, 484990); assert.equal(last.decoder.closing.at, 605000);
    assert.equal(last.decoder.coverageMs, 120010); assert.equal(last.decoder.frames, 150); assert.equal(last.decoder.drops, 2);
    const empty = windowSummary(raw, 200000, 300000);
    assert.equal(empty.rendered, 0); assert.equal(empty.driverCpuMs.count, 0); assert.equal(empty.driverCpuMs.p95, null);
    assert.deepEqual(empty.coverage.frames.rowRange, [3,3]); assert.deepEqual(empty.coverage.samples.rowIndices, []);
    assert.equal(empty.coverage.frames.firstObservedAt, null);
    assert.throws(() => windowSummary(raw, 4999, 125000), /slice window/);
    assert.throws(() => windowSummary(raw, 5000, 605001), /slice window/);
    assert.throws(() => windowSummary(raw, 125000, 125000), /slice window/);
  `));

  it('records enclosing CPU even when the original handler throws and captures the pre-handler tier', () => check(fixture + `
    const failure = new Error('original handler failed');
    pipeline.onFrame = function(tick) {
      originalPipelineFrame.call(this, tick);
      pipeline.currentUpscaler.neural = false;
      throw failure;
    };
    const throwingHandler = pipeline.onFrame;
    const record = install(); await advance(5000); await advance(5010);
    assert.throws(runtimeFrame, error => error === failure);
    assert.equal(record.driverCpuRows[0][1], 3.5); assert.equal(record.frames[0][9], 1);
    await advance(35000); assert.equal(pipeline.onFrame, throwingHandler);
    drainResolve(); await advance(35001); await observer.done;
  `));

  it('rejects a test-build hook and an unstable opening without treating either as evidence', () => check(fixture + `
    manager.testAccess = () => ({}); assert.throws(() => install(), /Production/); delete manager.testAccess;
    controller.state = 'fallback';
    const record = install(); await advance(5000); drainResolve(); await advance(5001); await observer.done;
    assert.match(record.error, /not stable/); assert.equal(record.started, undefined);
    assert.equal(driver.onFrame, originals.onFrame);
  `));
});