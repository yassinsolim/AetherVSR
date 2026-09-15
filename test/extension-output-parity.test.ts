import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

function check(source: string): void {
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { createHash, webcrypto } from 'node:crypto';
    import { createContext, Script } from 'node:vm';
    import { normalizeRgba8, summarizeRgba8, compareCaptures, parseArgs, TIMES,
      captureTask, runtimeState, seekPaused } from './tools/m10-output-parity.mjs';
    const hash = bytes => createHash('sha256').update(bytes).digest('hex');
    ${source}
    console.log('checked without browser');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 15000 });
  expect(output.trim()).toBe('checked without browser');
}

const replayFixture = `
  const calls = [], resources = [];
  let scopes = 0, pendingCallbacks = 0, skipCounter = false, gpuFailure = false, copyFailure = false;
  let originalCallbacksInvoked = 0;
  const video = { paused: true, seeking: false, readyState: 4, currentTime: 1, videoWidth: 1280, videoHeight: 720,
    currentSrc: 'http://localhost/media.mp4', getVideoPlaybackQuality: () => ({ totalVideoFrames: 70, droppedVideoFrames: 0 }) };
  const texture = { id: 'actual-swap-chain' };
  const model = { features: 16, depth: 2, file: { architecture: 'mock-not-hardware', sha256: 'declared' },
    stemWeights: new Float32Array([1, 2]), stemBias: new Float32Array([3]),
    bodyWeights: [new Float32Array([4]), new Float32Array([5])], bodyBias: [new Float32Array([6]), new Float32Array([7])],
    headWeights: new Float32Array([8]), headBias: new Float32Array([9]) };
  const stage = { neural: true, id: 'actual-stage', resolvedPrecision: 'f16', model, options: { passDiagnostics: true },
    encode(view) { assert.equal(this, stage); assert.equal(view, texture); calls.push('encode'); } };
  const importer = { kind: 'external', acquire() { assert.equal(this, importer); calls.push('import'); } };
  const device = { features: new Set(['shader-f16']), addEventListener() {}, removeEventListener() {},
    pushErrorScope() { scopes++; }, async popErrorScope() { scopes--; return gpuFailure ? { message: 'GPU validation failure' } : null; },
    queue: { async onSubmittedWorkDone() { calls.push('drain'); }, submit(commands) {
      calls.push(commands[0].copy ? 'copy-submit' : 'frame-submit');
    } },
    createBuffer({size}) {
      assert.equal(pipeline.running, false, 'Readback allocation outside frame');
      const data = new Uint8Array(size);
      for (let offset = 0; offset < size; offset += 4) data.set([30, 20, (offset / 4) % 251, 255], offset);
      const buffer = { destroyed: false, mapped: false,
        async mapAsync() { assert.equal(pipeline.running, false); assert.equal(pendingCallbacks, 0); calls.push('map'); this.mapped = true; },
        getMappedRange() { assert(this.mapped); return data.buffer; }, unmap() { this.mapped = false; },
        destroy() { this.destroyed = true; this.mapped = false; } };
      resources.push(buffer); return buffer;
    },
    createCommandEncoder() { return { copyTextureToBuffer(source) {
      assert.equal(source.texture, texture); if (copyFailure) throw new Error('copy failed');
    }, finish: () => ({ copy: true }) }; }
  };
  const originalConfig = { device, format: 'bgra8unorm', usage: 16, alphaMode: 'opaque', colorSpace: 'srgb' };
  let actualConfig = originalConfig;
  const context = { getConfiguration: () => actualConfig, configure(config) { actualConfig = config; calls.push('configure:' + config.usage); },
    getCurrentTexture() { return texture; } };
  const target = { context, size: { width: 2560, height: 1440 }, currentView: () => context.getCurrentTexture() };
  const callback = () => { originalCallbacksInvoked++; };
  const pipeline = { gpu: { device, adapterReport: { device: 'mock; not hardware evidence' } }, currentUpscaler: stage, importer, target,
    configuredSource: {width: 1280, height: 720}, running: false, error: null, disposed: false,
    submissionSequence: 10, framesRendered: 10, framesPresented: 20, timingGeneration: 3,
    source: { loadGeneration: 1, quality: video.getVideoPlaybackQuality },
    onFrame: callback, onConfiguration: callback, onGpuSample: callback, onGpuPassSample: callback,
    async drainTimings() {}, start() { calls.push('start'); this.running = true; pendingCallbacks++; },
    stop() { calls.push('stop'); this.running = false; pendingCallbacks = 0; },
    onTick(tick) {
      assert(this.running); assert.equal(tick.presentedDelta, 0); assert.equal(tick.mediaTime, 1);
      importer.acquire(); stage.encode(target.currentView()); device.queue.submit([{copy: false}]);
      if (!skipCounter) { this.submissionSequence++; this.framesRendered++; }
      this.onFrame?.(tick);
    }
  };
  const driver = { pipeline, video, snapshot: () => ({actualTier: 'neural', running: pipeline.running}) };
  const manager = { attachment: {driver}, status: () => ({enabled: true}), testAccess: undefined };
  const inputData = new Uint8ClampedArray(1280 * 720 * 4);
  for (let offset = 0; offset < inputData.length; offset += 4) inputData.set([(offset / 4) % 251, 20, 30, 255], offset);
  const document = { visibilityState: 'visible', hasFocus: () => true, createElement(name) {
    assert.equal(name, 'canvas');
    return { getContext(type) { assert.equal(type, '2d'); return { drawImage(source) { assert.equal(source, video); calls.push('offline-input'); },
      getImageData() { return {data: inputData}; } }; } };
  } };
  const sandbox = createContext({ document, crypto: webcrypto, TextEncoder, setTimeout, clearTimeout,
    performance: {now: () => 10}, navigator: {userAgent: 'mock, not Chrome'}, innerWidth: 1280, innerHeight: 900, devicePixelRatio: 1,
    GPUTextureUsage: {RENDER_ATTACHMENT: 16, COPY_SRC: 1}, GPUBufferUsage: {COPY_DST: 8, MAP_READ: 1}, GPUMapMode: {READ: 1},
    chrome: {runtime: {id: 'installed'}}, manager, aethervsrRuntime: {driver} });
  new Script('globalThis[Symbol.for("aethervsr.m10.document.installed")] = manager').runInContext(sandbox);
  sandbox.options = { extension: true, time: 1, forceCopy: false, sourceURL: video.currentSrc, modelJsonSha256: hash(JSON.stringify(model.file)) };
  const capture = () => new Script('(' + captureTask().toString() + ')(options)').runInContext(sandbox);
  const restored = () => {
    assert.equal(pipeline.running, false); assert.equal(pendingCallbacks, 0); assert.equal(originalCallbacksInvoked, 0);
    assert.equal(actualConfig, originalConfig); assert.equal(scopes, 0);
    for (const name of ['onFrame', 'onConfiguration', 'onGpuSample', 'onGpuPassSample']) assert.equal(pipeline[name], callback);
    assert(resources.every(buffer => buffer.destroyed && !buffer.mapped));
  };
`;

describe('M10 output parity offline helpers (no browser evidence)', () => {
  it('waits for the requested paused frame when an earlier rVFC callback is delivered first', () => check(`
    const callbacks=new Map(),timers=new Map();let next=0,listenerCount=0;
    class Video extends EventTarget {
      paused=true;seeking=false;readyState=4;currentTime=0;currentSrc='local';
      requestVideoFrameCallback(callback){const handle=++next;callbacks.set(handle,callback);return handle;}
      cancelVideoFrameCallback(handle){callbacks.delete(handle);}
      addEventListener(...args){listenerCount++;super.addEventListener(...args);}
      removeEventListener(...args){listenerCount--;super.removeEventListener(...args);}
    }
    const video=new Video();
    const sandbox=createContext({document:{querySelectorAll:()=>[video]},setTimeout(callback){const handle=++next;timers.set(handle,callback);return handle;},clearTimeout(handle){timers.delete(handle);}});
    const pending=new Script('('+seekPaused.toString()+')({time:1})').runInContext(sandbox);
    const deliver=mediaTime=>{const [handle,callback]=callbacks.entries().next().value;callbacks.delete(handle);callback(10,{mediaTime,presentedFrames:60,width:1280,height:720});};
    deliver(.75);video.dispatchEvent(new Event('seeked'));assert.equal(callbacks.size,1);deliver(1);
    const result=await pending;assert.equal(result.metadata.mediaTime,1);assert.equal(result.discardedMetadataCallbacks,1);
    assert.equal(callbacks.size,0);assert.equal(timers.size,0);assert.equal(listenerCount,0);
    const failed=new Script('('+seekPaused.toString()+')({time:2})').runInContext(sandbox);
    deliver(1);video.dispatchEvent(new Event('seeked'));[...timers.values()][0]();
    await assert.rejects(failed,/deadline/);assert.equal(callbacks.size,0);assert.equal(timers.size,0);assert.equal(listenerCount,0);
  `));

  it('strips aligned row padding and explicitly normalizes BGRA to RGBA', () => check(`
    const raw = new Uint8Array(512).fill(99);
    raw.set([3, 2, 1, 255, 6, 5, 4, 255]); raw.set([9, 8, 7, 255, 12, 11, 10, 255], 256);
    assert.deepEqual([...normalizeRgba8(raw, 2, 2, 256, 'bgra8unorm')], [1,2,3,255,4,5,6,255,7,8,9,255,10,11,12,255]);
    assert.deepEqual([...normalizeRgba8(raw, 2, 2, 256, 'rgba8unorm')], [3,2,1,255,6,5,4,255,9,8,7,255,12,11,10,255]);
  `));

  it('rejects unsupported formats and invalid extents without fallback', () => check(`
    for (const format of ['rgba16float', 'rgba8unorm-srgb', '', null]) assert.throws(() => normalizeRgba8(new Uint8Array(4), 1, 1, 4, format));
    for (const dimensions of [[0,1,4], [1,0,4], [1.5,1,4], [1,1,3], [1,1,8], [8192,8192,32768]])
      assert.throws(() => normalizeRgba8(new Uint8Array(4), ...dimensions, 'rgba8unorm'));
  `));

  it('hashes every byte, reports five samples, and distinguishes uniform RGB from varying alpha', () => check(`
    const bytes = new Uint8Array([1,2,3,255, 4,5,6,255, 7,8,9,255, 10,11,12,255]);
    const result = await summarizeRgba8(bytes, 2, 2);
    assert.equal(result.sha256, hash(bytes)); assert.equal(result.byteCount, 16); assert.equal(result.samples.length, 5);
    assert.deepEqual(result.minimum, [1,2,3,255]); assert.deepEqual(result.maximum, [10,11,12,255]);
    assert.equal(result.alpha.opaquePixels, 4); assert.equal(result.nonuniform, true);
    const uniform = await summarizeRgba8(new Uint8Array([1,2,3,0, 1,2,3,255]), 2, 1);
    assert.equal(uniform.nonuniform, false); assert.equal(uniform.alpha.opaquePixels, 1);
    await assert.rejects(() => summarizeRgba8(bytes, 3, 2));
  `));

  it('requires an exclusive ignored output location and explicit test-build copy opt-in', () => check(`
    assert.deepEqual(TIMES, [1,2,3]);
    const base = ['--out', '.cache/m10/output-parity/unit-test'];
    assert.equal(parseArgs(base).forceCopy, false);
    assert.equal(parseArgs([...base, '--test-build', '--force-copy']).forceCopy, true);
    for (const args of [[], ['--out'], ['--out','results/parity.json'], ['--out','.cache/m10/output-parity'],
      [...base,'--force-copy'], [...base,'--test-build','--test-build'], [...base,'--unsafe-gpu'], [...base,'--out','another']])
      assert.throws(() => parseArgs(args));
  `));

  it('serializes into the actual isolated-world path and submits exactly one pipeline frame before offline mapping', () => check(replayFixture + `
    const result = await capture(); restored();
    assert.equal(result.singleSubmission, true); assert.equal(result.frameCallbacks, 1);
    assert.equal(result.counters.after.sequence - result.counters.before.sequence, 1);
    assert.equal(result.counters.after.framesPresented, result.counters.before.framesPresented);
    assert.equal(result.output.format, 'RGBA8'); assert.equal(result.output.byteCount, 2560 * 1440 * 4);
    assert.equal(result.output.samples[0].rgba.join(','), '0,20,30,255');
    assert(calls.indexOf('import') < calls.indexOf('encode'));
    assert(calls.indexOf('frame-submit') < calls.indexOf('copy-submit'));
    assert(calls.indexOf('copy-submit') < calls.indexOf('map'));
    assert.equal(calls.filter(value => value === 'frame-submit').length, 1);
    assert.equal(calls.filter(value => value === 'copy-submit').length, 1);
  `));

  it('compares installed and harness captures, preserving hash mismatches as FAIL without fabricated deltas', () => check(replayFixture + `
    const extension = await capture(); restored(); sandbox.options.extension = false;
    const harness = await capture(); restored();
    assert.equal(compareCaptures(extension, harness).verdict, 'PASS');
    for (const field of ['input', 'output']) {
      const mismatch = structuredClone(harness); mismatch[field].sha256 = 'a'.repeat(64);
      const comparison = compareCaptures(extension, mismatch);
      assert.equal(comparison.verdict, 'FAIL'); assert.equal(comparison.maxByteDelta, null); assert.equal(comparison.differentBytes, null);
    }
    for (const field of ['precision', 'importPath', 'canvasFormat', 'packedWeightsSha256', 'modelJsonSha256']) {
      assert.equal(compareCaptures(extension, {...harness, [field]: 'different'}).verdict, 'FAIL');
    }
    const empty = structuredClone(harness); empty.output.nonuniform = false;
    assert.equal(compareCaptures(extension, empty).verdict, 'FAIL');
    const transparent = structuredClone(harness); transparent.output.alpha.minimum = 0;
    assert.equal(compareCaptures(extension, transparent).verdict, 'FAIL');
    assert.throws(() => compareCaptures(extension, null));
    assert.throws(() => compareCaptures(extension, {...harness, output: undefined}));
  `));

  it('fails on missing submission counters and restores the real pipeline', () => check(replayFixture + `
    skipCounter = true;
    await assert.rejects(capture, /exactly one/); restored();
  `));

  it('restores callbacks, canvas usage and readback resources on copy and GPU validation errors', () => check(replayFixture + `
    copyFailure = true;
    await assert.rejects(capture, /copy failed/); restored();
    copyFailure = false; gpuFailure = true;
    await assert.rejects(capture, /GPU validation failure/); restored();
  `));

  it('rejects a missing pipeline, unreadable canvas, wrong input, and wrong live model before replay', () => check(replayFixture + `
    const tick = pipeline.onTick; pipeline.onTick = undefined;
    await assert.rejects(capture, /onTick unavailable/); pipeline.onTick = tick;
    originalConfig.format = 'rgba16float'; await assert.rejects(capture, /configuration/); restored(); originalConfig.format = 'bgra8unorm';
    video.currentTime = 2; await assert.rejects(capture, /identity.*source changed/); restored(); video.currentTime = 1;
    sandbox.options.modelJsonSha256 = '0'.repeat(64); await assert.rejects(capture, /Live stage model/); restored();
    assert.equal(calls.filter(value => value === 'frame-submit').length, 0);
  `));

  it('keeps forced-copy and production build checks explicit in serialized browser helpers', () => check(replayFixture + `
    importer.kind = 'sampled'; await assert.rejects(capture, /no fallback/); restored();
    sandbox.options.forceCopy = true; const result = await capture(); restored(); assert.equal(result.importPath, 'sampled');
    const state = () => new Script('(' + runtimeState.toString() + ')({extension:true,testBuild:false})').runInContext(sandbox);
    assert.equal(state().ready, true); manager.testAccess = () => ({});
    assert.throws(state, /Wrong installed build kind/);
  `));
});