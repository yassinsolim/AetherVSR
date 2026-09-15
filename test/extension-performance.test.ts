import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

function check(source: string): void {
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { Script, createContext } from 'node:vm';
    import { parseCases, quantile, distribution, counterDelta, windowRate,
      installVideoObserver, installRuntimeRecorder, summarizeCapture, validateCapture, windowSummary } from './tools/m10-performance.mjs';
    import { deliverySummary, validateScheduling, installScheduling, installOwnedCost } from './tools/m105-accounting.mjs';
    import { parsePlan, ARMS } from './tools/m105-compare.mjs';
    import { summarizeTrace } from './tools/m105-trace.mjs';
    import { playerCommand } from './tools/m10-sites.mjs';
    import { validateOldShutdown } from './tools/m105-reload.mjs';
    import { parseFinalCases, deliveryGates } from './tools/m105-final.mjs';
    import { COST_ORDER, compareCosts, costBounds } from './tools/m107-performance.mjs';
    import { transitionVerdict } from './tools/m107-presentation.mjs';
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
  const install = (kind = 'extension-auto', durationMs = 30000, accounting = false) => {
    const options = { extension: kind.startsWith('extension-'), kind, warmupMs: 5000, durationMs, accounting };
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
  it('requires measured M10.7 watchdog costs and a valid neural reference', () => check(`
    const make=()=>COST_ORDER.map(item=>({case:item,valid:true,summary:{nativeCallbackFps:59,renderedFps:59,core:{p95:.1},driver:{p95:.1},geometryGuardMsPerSecond:1,
      tierStable:true,probes:0,transitions:0,ownerStable:true,error:null,deficit:0,visibleFraction:1,guard:{count:100,p95:.1,max:.2}}}));
    let rows=make();assert.equal(compareCosts(rows)[2].pass,true);
    rows.find(row=>row.case.strategy===2).summary.guard={count:0,p95:null,max:null};assert.equal(compareCosts(rows)[2].pass,false);
    rows=make();rows[0].summary.tierStable=false;assert.equal(compareCosts(rows)[2].pass,false);
    rows=make();rows[0].summary.probes=1;assert.equal(compareCosts(rows)[2].pass,false);
    rows=make();delete rows.find(row=>row.case.strategy===2).summary.probes;assert.equal(compareCosts(rows)[2].pass,false);
    rows=make();rows.find(row=>row.case.strategy===1).summary.transitions=1;assert.equal(compareCosts(rows)[1].pass,false);
    assert.equal(costBounds([0,0,0,0,0,0]).upper95,0);assert.throws(()=>costBounds([0,0]));
  `));

  it('rejects visible negative transitions and stale output geometry even when rectangles align', () => check(`
    const row={reason:'submitted:after',canvasVisible:true,mismatch:false,geometryGeneration:2,appliedGeometryGeneration:2,outputGeometryGeneration:2,sourceGeneration:1,outputGeneration:1,
      intrinsic:{width:640,height:360},backing:{width:1280,height:720},videoRect:{left:0,top:0,width:640,height:360},canvasRect:{left:0,top:0,width:640,height:360}};
    const trace={overflow:false,rows:[100,200,600,700].map(at=>({...row,at}))};
    const actions=[{name:'style-shift',startedAt:0,finishedAt:10}];assert.equal(transitionVerdict(trace,actions).pass,true);
    assert.equal(transitionVerdict(trace,actions,true).pass,false);
    trace.rows[0].outputGeometryGeneration=1;assert.equal(transitionVerdict(trace,actions).pass,false);
  `));

  it('keeps the 1% final loss and sustained58fps gates exact with no missing-data or rounding pass', () => check(`
    const summary={activeMs:600000,durationMs:600000,renderedFps:58,runtimePresentedFps:58,windowStats:{last120s:{renderedFps:58,runtimePresentedFps:58}}};
    const metrics={m10CombinedPercent:1,submissionDeficit:0};
    assert(Object.values(deliveryGates(summary,metrics)).every(Boolean));
    assert.equal(deliveryGates({...summary,activeMs:599999.99},metrics).duration,false);
    assert.equal(deliveryGates(summary,{...metrics,m10CombinedPercent:1.000001}).combinedLoss,false);
    assert.equal(deliveryGates({...summary,windowStats:{last120s:{renderedFps:57.999,runtimePresentedFps:59}}},metrics).lastRenderedFps,false);
    assert.equal(deliveryGates(summary,{m10CombinedPercent:null,submissionDeficit:null}).combinedLoss,false);
    assert.equal(deliveryGates({...summary,renderedFps:Infinity},metrics).renderedFps,false);
  `));

  it('fixes final candidate duration and mode without reusing short or baseline trials', () => check(`
    assert.equal(parseFinalCases([{id:'final-1',kind:'extension-auto',durationMs:600000,warmupMs:5000}]).length,1);
    assert.throws(()=>parseFinalCases([{id:'short',kind:'extension-auto',durationMs:30000}]));
    assert.throws(()=>parseFinalCases([{id:'other',kind:'harness',durationMs:30000}]));
    assert.throws(()=>parseFinalCases([{id:'warm',kind:'extension-auto',durationMs:600000,warmupMs:10000}]));
  `));

  it('requires observed privilege denial and complete old-world teardown rather than missing-data success', () => check(`
    const resources={device:0,pipeline:0,canvas:0,resizeObservers:0,listeners:0,geometryFrame:0,frameCallback:0};
    const cleanup={resources,status:{enabled:false,details:{discoveryActive:false,timerCount:0,infrastructure:{created:1,destroyed:1},lastTeardown:{...resources}}}};
    validateOldShutdown(cleanup,{privileged:false});
    assert.throws(()=>validateOldShutdown(cleanup,{unavailable:'timeout'}));
    assert.throws(()=>validateOldShutdown({...cleanup,resources:undefined},{privileged:false}));
    assert.throws(()=>validateOldShutdown({...cleanup,resources:{...resources,listeners:1}},{privileged:false}));
    assert.throws(()=>validateOldShutdown(cleanup,{privileged:true}));
  `));

  it('recognizes original player command labels with media titles without matching unrelated actions', () => check(`
    assert(playerCommand('play').test('Play, View From A Blue Moon'));
    assert(playerCommand('pause').test('Pause, Sample clip'));
    assert(playerCommand('play').test('Play Video'));assert(playerCommand('play').test('Replay'));
    assert(!playerCommand('play').test('Play next video'));assert(!playerCommand('play').test('Pause'));
    assert.throws(()=>playerCommand('enable'));
  `));

  it('summarizes nested Chrome trace slices without treating drop batches as unique frame identities', () => check(`
    const name='VideoFrameCallbackRequesterImpl::ExecuteVideoFrameCallbacks';
    const rows=summarizeTrace({traceEvents:[
      {name,ph:'B',pid:1,tid:2,ts:1000},{name:'nested',ph:'B',pid:1,tid:2,ts:1100},
      {ph:'E',pid:1,tid:2,ts:1200},{ph:'E',pid:1,tid:2,ts:2000},
      {name,ph:'X',pid:1,tid:2,ts:17000,dur:500},
      {name:'VideoFramesDropped',ph:'I',pid:1,tid:3,ts:18000,args:{count:4,id:7}}
    ]});
    assert.equal(rows[0].events,2);assert.equal(rows[0].durationMs.p50,.75);assert.equal(rows[0].intervalsMs.p50,16);
    assert.equal(rows[1].dropBatchCount,4);assert.deepEqual(rows[1].playerIds,[7]);assert.equal(rows[1].durationMs.p50,null);
    assert.throws(()=>summarizeTrace({traceEvents:[{name:'VideoFramesDropped',ph:'I',pid:1,tid:1,ts:1,args:{}}]}));
  `));

  it('validates controlled arms without accepting hidden fields or weakening historical case rules', () => check(`
    const plan = parsePlan(ARMS.map((arm,index)=>({id:'case'+index,arm,durationMs:30000})));
    assert.equal(plan[4].noRuntime,true); assert.equal(plan[7].kind,'extension-baseline');
    assert.equal(parsePlan([{id:'long',arm:'matched-harness',durationMs:600000}])[0].kind,'harness');
    for(const row of [{id:'x',arm:'neural',durationMs:0},{id:'x',arm:'neural',durationMs:1000,threshold:0},
      {id:'x',arm:'bare',durationMs:1000,diagnostics:'owned'}]) assert.throws(()=>parsePlan([row]));
  `));

  it('drains delayed long tasks by start-time boundaries and removes scheduling observers', () => check(fixture + `
    let pending = [], disconnected = false;
    context.PerformanceObserver = class { static supportedEntryTypes = ['longtask'];
      constructor(callback) { this.callback = callback; } observe() {} takeRecords() { return pending.splice(0); } disconnect() { disconnected = true; } };
    Object.assign(context, { screen:{width:1400,height:1000}, screenX:40,screenY:40,outerWidth:1200,outerHeight:900,
      requestAnimationFrame:() => 1,cancelAnimationFrame() {} });
    new Script('(' + installScheduling.toString() + ')()').runInContext(context);
    const data = new Script('globalThis[Symbol.for("aethervsr.m105.scheduling")]').runInContext(context);
    await advance(100); window.dispatchEvent({type:'aethervsr:m10:performance:start'});
    pending.push({startTime:90,duration:100,name:'boundary-before'}, {startTime:790,duration:100,name:'inside'}, {startTime:810,duration:100,name:'after'});
    await advance(800); window.dispatchEvent({type:'aethervsr:m10:performance:end'});
    window.dispatchEvent({type:'aethervsr:m10:performance:complete'}); await advance(801); await data.done;
    assert.equal(data.tasks.length, 1); assert.equal(data.tasks[0].at, 790); assert.equal(data.tasks[0].ms, 100);
    assert(disconnected);
  `));

  it('preserves diagnostic listener identity, receiver and non-overlapping outer CPU accounting', () => check(`
    const realm = createContext({assert});
    new Script(\`
      let clock=0;
      globalThis.performance={now:()=>clock};
      class EventTarget { constructor(){this.listeners=new Map();} addEventListener(type,fn){const rows=this.listeners.get(type)??new Set();rows.add(fn);this.listeners.set(type,rows);} removeEventListener(type,fn){this.listeners.get(type)?.delete(fn);} dispatchEvent(event){for(const fn of this.listeners.get(event.type)??[])fn.call(this,event);} }
      globalThis.EventTarget=EventTarget;globalThis.window=new EventTarget();
      globalThis.HTMLVideoElement=class extends EventTarget {requestVideoFrameCallback(fn){this.callback=fn;return 1;}};
      globalThis.MutationObserver=class {constructor(fn){this.callback=fn;}};
      globalThis.setTimeout=globalThis.setInterval=(fn)=>fn;
      globalThis.queueMicrotask=(fn)=>{globalThis.pendingMicrotask=fn;};
      globalThis.navigator={gpu:{requestAdapter:async()=>({real:true})}};
      globalThis.chrome={runtime:{onMessage:{addListener(fn){this.fn=fn;},removeListener(fn){if(this.fn===fn)this.fn=null;}},sendMessage(){clock+=2;return 7;}}};
      globalThis.target=new EventTarget();globalThis.original=EventTarget.prototype.addEventListener;
    \`).runInContext(realm);
    new Script('(' + installOwnedCost.toString() + ')({noRuntime:true})').runInContext(realm);
    new Script(\`
      const data=globalThis[Symbol.for('aethervsr.m105.owned')];
      function listener(){assert.equal(this,target);clock+=3;assert.equal(chrome.runtime.sendMessage(),7);clock+=4;}
      target.addEventListener('work',listener);window.dispatchEvent({type:'aethervsr:m10:performance:start'});
      target.dispatchEvent({type:'work'});assert.equal(data.totalMs,9);assert.equal(data.categories['runtime:sendMessage'].totalMs,2);assert.equal(data.categories['event:work'].totalMs,9);
      target.removeEventListener('work',listener);target.dispatchEvent({type:'work'});assert.equal(data.callbacks,1);
      queueMicrotask(()=>{clock+=5;});pendingMicrotask();assert.equal(data.categories.microtask.totalMs,5);assert.equal(data.totalMs,14);
      window.dispatchEvent({type:'aethervsr:m10:performance:end'});data.restore();assert.equal(EventTarget.prototype.addEventListener,original);
    \`).runInContext(realm);
  `));

  it('separates diagnostic tick attempts from successful post-submit callbacks and restores the tick method', () => check(fixture + `
    let submit = true;
    pipeline.source = {loadGeneration:3};
    const originalTick = pipeline.onTick = function(tick) {
      assert.equal(this, pipeline);
      if (submit) { counters.framesRendered++; counters.framesPresented++; this.onFrame(tick); }
      else this.error = 'diagnostic failed attempt';
    };
    const record = install('extension-auto', 30000, true);
    await advance(5000);
    const tick = {now:5010,mediaTime:2,presentedDelta:1,presentationTime:5008,expectedDisplayTime:5020};
    await advance(5010); pipeline.onTick(tick);
    submit = false; await advance(5030); pipeline.onTick({...tick,now:5030});
    assert.equal(record.frames.length, 1); assert.equal(record.attempts.length, 2);
    assert.deepEqual(Array.from(record.frames[0].slice(10)), [5008,5020,3,1]);
    assert.deepEqual(Array.from(record.attempts[0].slice(4)), [3,0,1,null]);
    assert.deepEqual(Array.from(record.attempts[1].slice(4)), [3,1,1,'diagnostic failed attempt']);
    await advance(35000); drainResolve(); await advance(35001); await observer.done;
    assert.equal(pipeline.onTick, originalTick);
    assert.equal(deliverySummary({runtime:record,video:observer}).submissionDeficit, 1);
  `));

  it('reports media-time intervals in milliseconds and retains unknown overlap and submission identity', () => check(fixture + `
    const record = install('no-extension');
    await advance(5000); await advance(5010); nativeFrame(11);
    await advance(5030); nativeFrame(13); await advance(35000); await observer.done;
    const summary = deliverySummary({ runtime: record, video: observer });
    assert(Math.abs(summary.mediaIntervalsMs.p50 - 20) < 1e-9);
    assert.equal(summary.gaps, 1);
    assert.equal(summary.callbackBoundaryResidual, 0);
    assert.equal(summary.exactOverlap, null);
    assert.equal(summary.submissionDeficit, null);
    assert.equal(summary.m10CombinedPercent, null);
    const active = install();
    active.opening = { runtime: { session: {framesRendered:0,framesPresented:0,framesSkipped:0},controller:{activeMs:0} } };
    active.closing = { runtime: { session: {framesRendered:1,framesPresented:2,framesSkipped:1},controller:{activeMs:30000} } };
    active.started = 0; active.ended = 30000;
    observer.closing.quality = null;
    assert.equal(deliverySummary({ runtime: active, video: observer }).m10CombinedPercent, null);
  `));

  it('requires complete in-window stimuli, bounded scheduling buffers and stable display observations', () => check(`
    const data = { overflow:false, started:10, ended:1000, opening:{width:1200}, closing:{width:1200}, stimuli:[{at:100,ended:150,requestedAtMs:90,requestedMs:50}] };
    validateScheduling(data, [{at:90,ms:50}]);
    assert.throws(() => validateScheduling({...data,overflow:true},[{at:90,ms:50}]));
    assert.throws(() => validateScheduling({...data,stimuli:[]},[{at:90,ms:50}]));
    assert.throws(() => validateScheduling({...data,ended:149},[{at:90,ms:50}]));
    assert.throws(() => validateScheduling({...data,closing:{width:800}},[{at:90,ms:50}]));
  `));

  it('adds diagnostic metadata without inventing frame identity or changing legacy columns', () => check(fixture + `
    const diagnosticContext = createContext({ ...context });
    new Script('(' + installVideoObserver.toString() + ')({accounting:true})').runInContext(diagnosticContext);
    document.dispatchEvent({ type: 'DOMContentLoaded' });
    const diagnostic = new Script('globalThis[Symbol.for("aethervsr.m10.performance.video")]').runInContext(diagnosticContext);
    window.dispatchEvent({ type: 'aethervsr:m10:performance:start' });
    for (const [handle, callback] of [...frameCallbacks]) {
      frameCallbacks.delete(handle);
      callback(100, { mediaTime: 2, presentedFrames: 123, presentationTime: 90, expectedDisplayTime: 110 });
    }
    assert.equal(diagnostic.rows.length, 1);
    assert.deepEqual(Array.from(diagnostic.rows[0].slice(9)), [90,110,0,1,null]);
    video.dispatchEvent({ type: 'loadstart' });
    for (const [handle, callback] of [...frameCallbacks]) {
      frameCallbacks.delete(handle);
      callback(120, { mediaTime: 0, presentedFrames: 1, presentationTime: 115, expectedDisplayTime: 130 });
    }
    assert.equal(diagnostic.rows[1][4], null);
    assert.equal(diagnostic.rows[1][11], 1);
    assert.equal(diagnostic.rows[1][12], 2);
  `));

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