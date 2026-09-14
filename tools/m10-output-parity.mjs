import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { arch, hostname, platform, release } from 'node:os';
import { dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const TIMES = [1, 2, 3];
export const MODEL_SHA256 = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
export const SOURCE_SHA256 = '8d81acbe164da1d62b7d0d02a3cc66915c96e8aa90d45cac34d818fc33df1d4a';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

export function parseArgs(args) {
  const options = { testBuild: false, forceCopy: false, out: null };
  const seen = new Set();
  for (let index = 0; index < args.length; index++) {
    const flag = args[index];
    assert(!seen.has(flag), `Duplicate argument: ${flag}`); seen.add(flag);
    if (flag === '--test-build') options.testBuild = true;
    else if (flag === '--force-copy') options.forceCopy = true;
    else if (flag === '--out') {
      options.out = args[++index];
      assert(options.out && !options.out.startsWith('--'), '--out requires a path');
    } else throw new Error(`Unknown argument: ${flag}`);
  }
  assert(options.out, 'Required: --out .cache/m10/output-parity/<new-run>');
  assert(!options.forceCopy || options.testBuild, '--force-copy requires --test-build');
  options.out = resolve(ROOT, options.out);
  const base = resolve(ROOT, '.cache/m10/output-parity');
  assert(options.out.startsWith(`${base}${sep}`), 'Output must be a new directory under .cache/m10/output-parity');
  return options;
}

export function normalizeRgba8(bytes, width, height, bytesPerRow, format) {
  if (!['rgba8unorm', 'bgra8unorm'].includes(format)) throw new Error(`Unsupported actual canvas format: ${format}`);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 4096 * 2160 ||
      !Number.isInteger(bytesPerRow) || bytesPerRow < width * 4 || bytes.length !== bytesPerRow * height)
    throw new Error('Invalid readback extent');
  const rgba = new Uint8Array(width * height * 4);
  const blueFirst = format === 'bgra8unorm';
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const source = row * bytesPerRow + column * 4;
      const target = (row * width + column) * 4;
      rgba[target] = bytes[source + (blueFirst ? 2 : 0)];
      rgba[target + 1] = bytes[source + 1];
      rgba[target + 2] = bytes[source + (blueFirst ? 0 : 2)];
      rgba[target + 3] = bytes[source + 3];
    }
  }
  return rgba;
}

export async function summarizeRgba8(bytes, width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || bytes.length !== width * height * 4)
    throw new Error('Invalid RGBA8 frame');
  const minimum = [255, 255, 255, 255], maximum = [0, 0, 0, 0];
  let nonuniform = false, opaquePixels = 0;
  for (let offset = 0; offset < bytes.length; offset += 4) {
    for (let channel = 0; channel < 4; channel++) {
      const value = bytes[offset + channel];
      if (value < minimum[channel]) minimum[channel] = value;
      if (value > maximum[channel]) maximum[channel] = value;
      if (channel < 3 && value !== bytes[channel]) nonuniform = true;
    }
    if (bytes[offset + 3] === 255) opaquePixels++;
  }
  const positions = [[0, 0], [width - 1, 0], [Math.floor(width / 2), Math.floor(height / 2)], [0, height - 1], [width - 1, height - 1]];
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return { width, height, byteCount: bytes.length, format: 'RGBA8',
    sha256: Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join(''),
    minimum, maximum, nonuniform, alpha: { minimum: minimum[3], maximum: maximum[3], opaquePixels },
    samples: positions.map(([column, row]) => ({ column, row, rgba: Array.from(bytes.subarray((row * width + column) * 4, (row * width + column) * 4 + 4)) })) };
}

export function compareCaptures(extension, harness) {
  const checks = {};
  for (const side of [extension, harness]) {
    assert(side && typeof side === 'object', 'Missing actual capture');
    for (const field of ['input', 'output']) {
      const frame = side[field];
      assert(frame && /^[a-f0-9]{64}$/.test(frame.sha256) && frame.byteCount === frame.width * frame.height * 4, `Missing ${field} evidence`);
    }
  }
  for (const field of ['input', 'output']) {
    checks[`${field}Extent`] = extension[field].width === harness[field].width && extension[field].height === harness[field].height;
    checks[`${field}Hash`] = extension[field].sha256 === harness[field].sha256;
    checks[`${field}NonuniformOpaque`] = [extension, harness].every(side => side[field].nonuniform === true &&
      side[field].alpha.minimum === 255 && side[field].alpha.maximum === 255 && side[field].alpha.opaquePixels === side[field].width * side[field].height);
  }
  checks.sourceExtent = [extension, harness].every(side => side.input.width === 1280 && side.input.height === 720);
  checks.outputExtent2x = [extension, harness].every(side => side.output.width === 2560 && side.output.height === 1440);
  for (const field of ['requestedTime', 'currentTime', 'precision', 'importPath', 'upscalerId', 'canvasFormat', 'modelJsonSha256', 'packedWeightsSha256'])
    checks[field] = extension[field] !== undefined && extension[field] === harness[field];
  checks.stageOptions = extension.options !== undefined && JSON.stringify(extension.options) === JSON.stringify(harness.options);
  checks.pausedSingleSubmission = [extension, harness].every(side => side.paused && side.singleSubmission === true &&
    side.counters.after.sequence - side.counters.before.sequence === 1 && side.counters.after.framesRendered - side.counters.before.framesRendered === 1);
  return { verdict: Object.values(checks).every(Boolean) ? 'PASS' : 'FAIL', checks,
    comparison: 'Full normalized RGBA8 SHA-256 equality; no tolerance', differentBytes: null, maxByteDelta: null,
    byteDeltaScope: 'not measured; full frames remain inside their browser worlds, only digests and five samples leave' };
}

export function runtimeState({ extension, testBuild }) {
  const manager = extension ? globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)] : null;
  const driver = extension ? manager?.attachment?.driver : globalThis.aethervsrRuntime?.driver;
  const pipeline = driver?.pipeline;
  if (extension && manager && (typeof manager.testAccess === 'function') !== testBuild) throw new Error('Wrong installed build kind');
  return { ready: !!pipeline && pipeline.currentUpscaler.neural && pipeline.framesRendered > 0 &&
      pipeline.configuredSource.width === 1280 && pipeline.configuredSource.height === 720,
    runtime: driver?.snapshot() ?? null, status: manager?.status() ?? null,
    precision: pipeline?.currentUpscaler.resolvedPrecision ?? null, importPath: pipeline?.importer.kind ?? null,
    error: pipeline?.error ? String(pipeline.error) : null };
}

export async function pauseRuntime({ extension }) {
  const manager = extension ? globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)] : null;
  const driver = extension ? manager?.attachment?.driver : globalThis.aethervsrRuntime?.driver;
  if (!driver?.pipeline || !driver.pipeline.currentUpscaler.neural) throw new Error('Actual neural pipeline unavailable');
  const { video, pipeline } = driver;
  video.pause(); driver.syncActive(); pipeline.stop();
  return { paused: video.paused, running: pipeline.running, runtime: driver.snapshot(), source: video.currentSrc,
    frameSequence: pipeline.submissionSequence, framesRendered: pipeline.framesRendered };
}

export async function seekPaused({ time }) {
  const videos = [...document.querySelectorAll('video')];
  if (videos.length !== 1) throw new Error('Expected one original page video');
  const video = videos[0];
  if (!video.paused || !video.requestVideoFrameCallback) throw new Error('Paused native video with rVFC required');
  return new Promise((done, reject) => {
    let metadata = null, sought = false, handle;
    const finish = error => {
      if (!error && (!metadata || !sought)) return;
      clearTimeout(timer); video.cancelVideoFrameCallback(handle);
      video.removeEventListener('seeked', seeked); video.removeEventListener('error', failure);
      if (error) reject(error);
      else if (!video.paused || video.seeking || video.readyState < 2 || Math.abs(video.currentTime - time) > 1e-6 ||
        !Number.isFinite(metadata.mediaTime) || Math.abs(metadata.mediaTime - time) > 1 / 60 || metadata.width !== 1280 || metadata.height !== 720)
        reject(new Error('Paused seek/rVFC integrity failed'));
      else done({ requestedTime: time, currentTime: video.currentTime, metadata, source: video.currentSrc });
    };
    const seeked = () => { sought = true; finish(); };
    const failure = () => finish(new Error(video.error?.message ?? 'Video error'));
    const timer = setTimeout(() => finish(new Error('Paused seek/rVFC deadline')), 3000);
    video.addEventListener('seeked', seeked); video.addEventListener('error', failure);
    handle = video.requestVideoFrameCallback((now, value) => { metadata = { now, mediaTime: value.mediaTime,
      presentedFrames: value.presentedFrames, width: value.width, height: value.height }; finish(); });
    try { video.currentTime = time; } catch (error) { finish(error); }
  });
}

export async function capturePausedFrame(options, normalize, summarize) {
  const manager = options.extension ? globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)] : null;
  const attachment = manager?.attachment;
  const driver = options.extension ? attachment?.driver : globalThis.aethervsrRuntime?.driver;
  const pipeline = driver?.pipeline;
  if (!pipeline || typeof pipeline.onTick !== 'function') throw new Error('Actual pipeline/private onTick unavailable');
  const { video } = driver, stage = pipeline.currentUpscaler, importer = pipeline.importer, target = pipeline.target;
  const device = pipeline.gpu.device, context = target.context;
  let timer;
  const deadline = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Offline GPU capture deadline')), 3000); });
  const wait = promise => Promise.race([promise, deadline]);
  const callbacks = Object.fromEntries(['onFrame', 'onConfiguration', 'onGpuSample', 'onGpuPassSample'].map(name => [name, pipeline[name]]));
  const originalConfig = context.getConfiguration();
  let readback, changed = false, scopeCount = 0;
  const gpuErrors = [];
  const gpuError = event => gpuErrors.push(event.error.message);
  const counters = () => ({ sequence: pipeline.submissionSequence, framesRendered: pipeline.framesRendered,
    framesPresented: pipeline.framesPresented, sourceGeneration: pipeline.source.loadGeneration,
    quality: pipeline.source.quality() });
  const guard = () => {
    if (!video.paused || video.seeking || video.readyState < 2 || Math.abs(video.currentTime - options.time) > 1e-6 ||
        video.videoWidth !== 1280 || video.videoHeight !== 720 || video.currentSrc !== options.sourceURL ||
        pipeline.running || pipeline.error || pipeline.currentUpscaler !== stage || pipeline.importer !== importer || pipeline.target !== target ||
        (options.extension && manager.attachment !== attachment)) throw new Error('Actual paused pipeline identity/source changed');
    if (!stage.neural || !['fp32', 'f16'].includes(stage.resolvedPrecision) || importer.kind !== (options.forceCopy ? 'sampled' : 'external'))
      throw new Error('Neural precision/import prerequisite missing; no fallback permitted');
    if (pipeline.configuredSource.width !== 1280 || pipeline.configuredSource.height !== 720 || target.size.width !== 2560 || target.size.height !== 1440)
      throw new Error('Existing pipeline must already be configured at 1280x720 -> 2560x1440');
  };
  const sha = async bytes => Array.from(new Uint8Array(await wait(crypto.subtle.digest('SHA-256', bytes))), value => value.toString(16).padStart(2, '0')).join('');
  device.addEventListener('uncapturederror', gpuError);
  try {
    guard();
    if (!originalConfig || originalConfig.device !== device || originalConfig.usage !== GPUTextureUsage.RENDER_ATTACHMENT ||
        originalConfig.alphaMode !== 'opaque' || originalConfig.colorSpace !== 'srgb' || !['rgba8unorm', 'bgra8unorm'].includes(originalConfig.format))
      throw new Error('Actual production canvas configuration is not readable by this apparatus');
    await wait(device.queue.onSubmittedWorkDone()); await wait(pipeline.drainTimings()); guard();
    const model = stage.model;
    if (model?.features !== 16 || model?.depth !== 2) throw new Error('Actual C16D2 model missing');
    const modelJsonSha256 = await sha(new TextEncoder().encode(JSON.stringify(model.file)));
    if (modelJsonSha256 !== options.modelJsonSha256) throw new Error('Live stage model differs from pinned production JSON');
    const tensors = [model.stemWeights, model.stemBias, ...model.bodyWeights, ...model.bodyBias, model.headWeights, model.headBias];
    const packed = new Uint8Array(tensors.reduce((total, tensor) => total + tensor.byteLength, 0));
    let cursor = 0;
    for (const tensor of tensors) { packed.set(new Uint8Array(tensor.buffer, tensor.byteOffset, tensor.byteLength), cursor); cursor += tensor.byteLength; }
    const packedWeightsSha256 = await sha(packed);
    const sourceCanvas = document.createElement('canvas'); sourceCanvas.width = 1280; sourceCanvas.height = 720;
    const sourceContext = sourceCanvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true });
    if (!sourceContext) throw new Error('Offline origin-clean source readback unavailable');
    sourceContext.drawImage(video, 0, 0);
    const input = await wait(summarize(new Uint8Array(sourceContext.getImageData(0, 0, 1280, 720).data), 1280, 720));
    const width = target.size.width, height = target.size.height;
    const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
    for (const filter of ['validation', 'out-of-memory', 'internal']) { device.pushErrorScope(filter); scopeCount++; }
    readback = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    changed = true; context.configure({ ...originalConfig, usage: originalConfig.usage | GPUTextureUsage.COPY_SRC });
    let frameCallbacks = 0;
    pipeline.onFrame = () => { frameCallbacks++; };
    pipeline.onConfiguration = null; pipeline.onGpuSample = null; pipeline.onGpuPassSample = null;
    guard();
    const before = counters(), generation = pipeline.timingGeneration;
    const texture = context.getCurrentTexture();
    const now = performance.now();
    try {
      pipeline.start();
      pipeline.onTick({ now, mediaTime: video.currentTime, size: { width: 1280, height: 720 }, presentedDelta: 0,
        presentationTime: now, expectedDisplayTime: now, decodeLatencyMs: null });
    } finally { pipeline.stop(); }
    if (context.getCurrentTexture() !== texture) throw new Error('Actual presenter changed swap-chain texture');
    const encoder = device.createCommandEncoder({ label: 'm10:offline-output-readback' });
    encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow }, { width, height });
    device.queue.submit([encoder.finish()]);
    const after = counters();
    if (after.sequence - before.sequence !== 1 || after.framesRendered - before.framesRendered !== 1 || frameCallbacks !== 1 ||
        after.framesPresented !== before.framesPresented || after.sourceGeneration !== before.sourceGeneration || pipeline.timingGeneration !== generation)
      throw new Error('Replay did not submit exactly one unchanged-configuration pipeline frame');
    guard();
    await wait(readback.mapAsync(GPUMapMode.READ));
    const rgba = normalize(new Uint8Array(readback.getMappedRange()), width, height, bytesPerRow, originalConfig.format);
    readback.unmap();
    while (scopeCount) { scopeCount--; const error = await wait(device.popErrorScope()); if (error) gpuErrors.push(error.message); }
    const output = await wait(summarize(rgba, width, height));
    guard();
    if (pipeline.submissionSequence !== after.sequence || pipeline.source.loadGeneration !== before.sourceGeneration || gpuErrors.length)
      throw new Error(`Capture changed or GPU error: ${gpuErrors.join('; ')}`);
    return { requestedTime: options.time, currentTime: video.currentTime, paused: video.paused, source: video.currentSrc,
      precision: stage.resolvedPrecision, importPath: importer.kind, upscalerId: stage.id, modelJsonSha256, packedWeightsSha256,
      model: { features: model.features, depth: model.depth, declaredSha256: model.file.sha256 }, options: stage.options,
      canvasFormat: originalConfig.format, canvasUsage: { production: originalConfig.usage, diagnostic: originalConfig.usage | GPUTextureUsage.COPY_SRC },
      bytesPerRow, input, output, counters: { before, after, afterReadback: counters() }, frameCallbacks, singleSubmission: true,
      runtime: driver.snapshot(), gpuErrors, adapter: pipeline.gpu.adapterReport, deviceFeatures: [...device.features],
      environment: { userAgent: navigator.userAgent, visibility: document.visibilityState, focused: document.hasFocus(),
        viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio } } };
  } finally {
    clearTimeout(timer);
    pipeline.stop(); Object.assign(pipeline, callbacks);
    try { if (changed) context.configure(originalConfig); }
    finally {
      readback?.destroy(); device.removeEventListener('uncapturederror', gpuError);
      while (scopeCount) { scopeCount--; void device.popErrorScope().catch(() => {}); }
    }
  }
}

export function captureTask() {
  return new Function('options', `return (${capturePausedFrame.toString()})(options, ${normalizeRgba8.toString()}, ${summarizeRgba8.toString()});`);
}

export function cleanupRuntime({ extension }) {
  const manager = extension ? globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)] : null;
  const driver = extension ? manager?.attachment?.driver : globalThis.aethervsrRuntime?.driver;
  const errors = [];
  const attempt = operation => { try { operation(); } catch (error) { errors.push(String(error)); } };
  attempt(() => { driver?.video.pause(); driver?.pipeline.stop(); });
  if (extension) attempt(() => manager?.stop());
  else {
    attempt(() => driver?.destroy()); attempt(() => driver?.pipeline.destroy()); attempt(() => driver?.pipeline.gpu.device.destroy());
  }
  if (errors.length) throw new Error(errors.join('; '));
  return { stopped: !driver?.pipeline.running, destroyed: driver?.pipeline.disposed ?? null, status: manager?.status() ?? null };
}

export async function run(options) {
  const relativeOutput = relative(ROOT, options.out);
  execFileSync('git', ['check-ignore', '-q', relativeOutput], { cwd: ROOT, timeout: 5000 });
  mkdirSync(dirname(options.out), { recursive: true });
  assert(realpathSync(dirname(options.out)).startsWith(`${realpathSync(ROOT)}${sep}`), 'Output parent escaped workspace');
  mkdirSync(options.out);
  const resultPath = resolve(options.out, 'result.json');
  const report = { schema: 'aethervsr.m10.output-parity/1', startedAt: new Date().toISOString(), verdict: 'FAIL',
    kind: options.testBuild ? (options.forceCopy ? 'test-build forced-copy diagnostic' : 'test-build external diagnostic') : 'installed production external parity',
    options, times: TIMES, expected: { modelSha256: MODEL_SHA256, sourceSha256: SOURCE_SHA256 },
    apparatus: 'Actual installed content singleton in its extension isolated world versus actual root index harness aethervsrRuntime.driver.pipeline. Original page videos paused at identical seeks; offline 2D sRGB input hashes. Already-configured importer/upscaler/presenter reused. Real GPUCanvasContext temporarily gains COPY_SRC, aligned staging buffer preallocated. start/onTick/stop synchronously replays one frame; scheduled acquisition cancelled before task ends. Pipeline callbacks temporarily suppressed (onFrame counted) so synthetic tick does not steer runtime. A second same-task GPU submission copies the actual presented texture, then mapAsync runs outside any frame callback. Restores callbacks/configuration and stops; destroys runtimes at side teardown. NOT an uninstrumented frame capture or a timing benchmark; no M9 parity substitution.',
    measurement: { performance: 'not measured', byteDeltas: 'not measured' },
    bounds: { runMs: 180000, startupMs: 60000, operationMs: 10000, captureInPageMs: 3000, teardownMs: 45000,
      buildGatePerGitCommandMs: 10000 },
    machine: { hostname: hostname(), platform: platform(), release: release(), arch: arch(), displayRefreshRate: 'not measured' },
    events: [], errors: [], sides: {}, comparisons: [] };
  let native, fixtures, server, build, saved = false, finishing = false;
  const save = () => {
    if (saved) return;
    report.finishedAt = new Date().toISOString();
    writeFileSync(resultPath, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' }); saved = true;
  };
  const record = (type, data) => {
    if (report.events.length >= 200) throw new Error('Evidence event bound exceeded');
    report.events.push({ at: new Date().toISOString(), type, data });
  };
  const fail = error => { report.verdict = 'FAIL'; report.errors.push(String(error)); };
  let watchdog, forcedExit;
  let verifyBuild, bounded;
  const finish = async () => {
    if (finishing) return; finishing = true; clearTimeout(watchdog);
    forcedExit = setTimeout(() => { fail('Hard teardown deadline; cleanup incomplete'); save(); process.exit(1); }, 45000);
    for (const resource of [native, fixtures, server]) if (resource) {
      try { await bounded(resource.close(), resource === native ? 8000 : 3000, 'Close owned resource'); } catch (error) { fail(error); }
    }
    try { report.buildAfter = verifyBuild?.(options.testBuild); assert(build, 'No verified opening build'); assert.deepEqual(report.buildAfter, build, 'Build/HEAD changed'); }
    catch (error) { fail(error); }
    save(); clearTimeout(forcedExit);
  };
  const interrupt = () => { fail('Interrupted'); void finish().finally(() => process.exit(1)); };
  process.once('SIGINT', interrupt); process.once('SIGTERM', interrupt);
  watchdog = setTimeout(() => { fail('Whole-run deadline'); void finish().finally(() => process.exit(1)); }, 180000);
  try {
    const browserTools = await import('./m10-browser.mjs');
    ({ verifyBuild, bounded } = browserTools);
    const { openExtension, until } = browserTools;
    const { harnessServer } = await import('./m10-performance.mjs');
    const { startFixtures } = await import('./m10-fixtures.mjs');
    build = report.buildBefore = verifyBuild(options.testBuild);
    const modelBytes = readFileSync(resolve(ROOT, 'public/models/aethersr-c16d2.json'));
    assert.equal(hash(modelBytes), MODEL_SHA256);
    assert.equal(hash(readFileSync(resolve(ROOT, 'public/media/m9/720p60.mp4'))), SOURCE_SHA256);
    const modelJsonSha256 = hash(JSON.stringify(JSON.parse(modelBytes.toString())));
    report.expected.modelJsonSha256 = modelJsonSha256;
    const acquire = async (create, label) => {
      let abandoned = false;
      const pending = Promise.resolve().then(create).then(async resource => {
        if (abandoned || finishing) { await resource.close(); throw new Error(`Late ${label} closed`); }
        return resource;
      });
      try { return await bounded(pending, 60000, label); } catch (error) { abandoned = true; throw error; }
    };
    server = await acquire(harnessServer, 'Root harness startup');
    report.harness = { origin: server.origin, owned: server.owned, pins: server.pins };
    fixtures = await acquire(() => startFixtures({ mse: false }), 'Fixture startup');
    report.fixture = fixtures.evidence;
    assert.equal(fixtures.evidence.sha256, SOURCE_SHA256, 'Fixture fallback forbidden');
    native = await acquire(() => openExtension(build, record), 'Native installed-extension startup');
    for (const extension of [true, false]) {
      const side = extension ? 'extension' : 'harness';
      const evidence = report.sides[side] = { frames: [], browserErrors: [] };
      const page = await bounded(native.context.newPage(), 5000, 'New page');
      page.on('pageerror', error => { if (evidence.browserErrors.length < 100) evidence.browserErrors.push(String(error)); });
      page.on('crash', () => evidence.browserErrors.push('Page crashed'));
      const evaluate = (fn, arg) => bounded(extension ? native.isolated(page, fn, arg) : page.evaluate(fn, arg), 10000, `${side} evaluation`);
      try {
        const url = extension ? `${fixtures.url}?case=custom` : `${server.origin}/?clip=/media/m9/720p60.mp4&mode=neural${options.forceCopy ? '&import=copy' : ''}`;
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
        await bounded(page.bringToFront(), 3000, 'Focus parity page');
        if (extension) {
          const popup = await bounded(native.popup(page), 20000, 'Actual popup');
          try {
            assert.equal((await popup.request('m10.status')).enabled, false, 'Fresh page unexpectedly enabled');
            assert.equal((await popup.click('#enable')).enabled, true);
            if (options.forceCopy) {
              assert.equal((await popup.click('#disable')).enabled, false);
              await native.isolated(page, () => {
                const access = globalThis.__AETHERVSR_EXTENSION_TEST__;
                if (!access || access.status().enabled) throw new Error('Disabled installed test build required');
                access.configure({ forceCopy: true });
              });
              assert.equal((await popup.click('#enable')).enabled, true);
            }
            evidence.activation = await popup.click('input[value="neural"]');
          } finally { await bounded(popup.dismiss(), 8000, 'Dismiss actual popup'); }
        }
        evidence.ready = await until(() => evaluate(runtimeState, { extension, testBuild: options.testBuild }), state => state.ready && !state.error, 20000);
        assert.equal(evidence.ready.importPath, options.forceCopy ? 'sampled' : 'external');
        evidence.pause = await evaluate(pauseRuntime, { extension });
        assert(evidence.pause.paused && !evidence.pause.running);
        const sourceURL = extension ? new URL('/media/same.mp4', fixtures.url).href : `${server.origin}/media/m9/720p60.mp4`;
        evidence.servedSource = await bounded(page.evaluate(async sourceURL => {
          const response = await fetch(sourceURL, { cache: 'no-store', signal: AbortSignal.timeout(3000) });
          if (!response.ok) throw new Error(`Actual source HTTP ${response.status}`);
          const bytes = await response.arrayBuffer();
          return { url: response.url, bytes: bytes.byteLength, sha256: Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('') };
        }, sourceURL), 5000, 'Actual source hash');
        assert.equal(evidence.servedSource.sha256, SOURCE_SHA256);
        for (const time of TIMES) {
          const row = { time }; evidence.frames.push(row);
          row.seek = await bounded(page.evaluate(seekPaused, { time }), 5000, 'Paused seek');
          row.capture = await evaluate(captureTask(), { extension, time, forceCopy: options.forceCopy, modelJsonSha256, sourceURL });
        }
        assert.equal(evidence.browserErrors.length, 0, 'Page execution errors');
      } finally {
        try { evidence.cleanup = await evaluate(cleanupRuntime, { extension }); }
        catch (error) { fail(`${side} cleanup: ${error}`); }
        await bounded(page.close(), 2000, 'Close parity page');
      }
    }
    for (let index = 0; index < TIMES.length; index++) {
      const extension = report.sides.extension.frames[index], harness = report.sides.harness.frames[index];
      const comparison = compareCaptures(extension.capture, harness.capture);
      comparison.decodedMediaTimeEqual = extension.seek.metadata.mediaTime === harness.seek.metadata.mediaTime;
      if (!comparison.decodedMediaTimeEqual) comparison.verdict = 'FAIL';
      report.comparisons.push({ time: TIMES[index], ...comparison });
    }
    report.harness.after = await bounded(server.verify(), 20000, 'Root harness postcheck');
    assert.deepEqual(report.harness.after, server.pins, 'Root served bytes changed');
    report.verdict = report.errors.length === 0 && report.comparisons.every(row => row.verdict === 'PASS') ? 'PASS' : 'FAIL';
  } catch (error) { fail(error); }
  finally { await finish(); process.off('SIGINT', interrupt); process.off('SIGTERM', interrupt); }
  console.log(`${report.verdict}: ${resultPath}`);
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.includes('--help')) console.log('node tools/m10-output-parity.mjs --out .cache/m10/output-parity/<new-run> [--test-build [--force-copy]]\nRequires separately frozen clean HEAD and matching build. Launches native Chrome with default security flags. Fixed paused seeks: 1, 2, 3 seconds.');
  else run(parseArgs(process.argv.slice(2))).then(report => {
    process.exitCode = report.verdict === 'PASS' ? 0 : 1;
    const exit = setTimeout(() => process.exit(process.exitCode), 1000); exit.unref();
  }).catch(error => { console.error(error); process.exitCode = 1; });
}