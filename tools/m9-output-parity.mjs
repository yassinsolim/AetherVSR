import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus, hostname, release } from 'node:os';
import { dirname, posix, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { createServer } from 'vite';
import { openNativeChrome } from './m9-browser.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const origin = 'http://localhost:5173';
const oldRef = 'c65f67a';
const sourceFile = 'src/core/upscale/neural-upscaler.ts';
const modelFile = 'public/models/aethersr-c16d2.json';
const productionSha256 = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
const clipFile = 'public/media/m9/720p60.mp4';
const clipSha256 = '8d81acbe164da1d62b7d0d02a3cc66915c96e8aa90d45cac34d818fc33df1d4a';
const cache = resolve(root, '.cache/m9/output-parity');
const resultFile = resolve(root, 'results/m9-output-parity.json');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trimEnd();
const digest = file => hash(readFileSync(resolve(root, file)));

function sourceSnapshot() {
  const files = git('ls-files', '-z', 'src', 'bench.html', 'vite.config.ts', 'tools/m9-browser.mjs')
    .split('\0').filter(Boolean);
  return {
    head: git('rev-parse', 'HEAD'),
    sourceSha256: Object.fromEntries(files.map(file => [file, digest(file)])),
    trackedDiff: git('diff', '--name-only', 'HEAD'),
    modelSha256: digest(modelFile),
    clipSha256: digest(clipFile),
  };
}

function historicalModule(source) {
  const transpiled = ts.transpileModule(source, {
    fileName: sourceFile,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
    reportDiagnostics: true,
  });
  assert.equal(transpiled.diagnostics?.length ?? 0, 0, 'Historical TypeScript transpilation failed');
  const imports = [];
  const code = transpiled.outputText.replace(/\bfrom\s+(['"])(\.[^'"]+)\1/g,
    (original, quote, specifier) => {
      const dependency = posix.normalize(posix.join(posix.dirname(sourceFile), specifier))
        .replace(/\.js$/, '.ts');
      assert.ok(dependency.startsWith('src/core/'), `Unexpected historical dependency: ${dependency}`);
      assert.ok(existsSync(resolve(root, dependency)), `Missing dependency: ${dependency}`);
      imports.push({ specifier, url: `${origin}/${dependency}`, file: dependency,
        currentSha256: digest(dependency), oldSha256: hash(execFileSync('git', ['show', `${oldRef}:${dependency}`], { cwd: root })) });
      return `from ${quote}${origin}/${dependency}${quote}`;
    });
  assert.ok(imports.length > 0, 'No historical imports were remapped');
  return { code, imports, transpiledSha256: hash(transpiled.outputText), blobModuleSha256: hash(code) };
}

async function runBrowser(page, oldModule, before, report) {
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push({ kind: 'pageerror', message: error.message }));
  page.on('console', message => {
    if (message.type() === 'error') browserErrors.push({ kind: 'console', message: message.text(), ...message.location() });
  });
  await page.goto(`${origin}/bench.html`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForFunction(() => typeof window.aethervsrEnvironment === 'function', null, { timeout: 30000 });
  report.initialPage = await page.evaluate(() => ({
    videos: [...document.querySelectorAll('video')].map(video => ({
      paused: video.paused, currentTime: video.currentTime, currentSrc: video.currentSrc,
      readyState: video.readyState, autoplay: video.autoplay,
    })),
    scripts: [...document.scripts].map(script => script.src),
    status: document.querySelector('#status')?.textContent,
  }));
  assert.ok(report.initialPage.videos.every(video => video.paused && !video.currentSrc && video.currentTime === 0),
    'Benchmark page unexpectedly loaded or played video');
  assert.ok(!report.initialPage.scripts.some(script => script.includes('/@vite/client')), 'Server is not in benchmark mode');
  report.browserSetup = await page.evaluate(async ({ code, origin, productionSha256, sourceHashes }) => {
    const sha256 = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
      .map(value => value.toString(16).padStart(2, '0')).join('');
    const servedSources = {};
    for (const [file, expected] of Object.entries(sourceHashes)) {
      if (!file.startsWith('src/core/')) continue;
      const module = await import(`${origin}/${file}?raw`);
      const actual = await sha256(new TextEncoder().encode(module.default));
      if (actual !== expected) throw new Error(`Served source mismatch: ${file}`);
      servedSources[file] = actual;
    }
    const blobUrl = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
    let OldNeural;
    try { OldNeural = (await import(blobUrl)).NeuralUpscaler; }
    finally { URL.revokeObjectURL(blobUrl); }
    const { NeuralUpscaler: NewNeural } = await import(`${origin}/src/core/upscale/neural-upscaler.ts`);
    const { packModel } = await import(`${origin}/src/core/neural/model.ts`);
    const response = await fetch('/models/aethersr-c16d2.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Model fetch failed: ${response.status}`);
    const modelBytes = await response.arrayBuffer();
    const modelSha256 = await sha256(modelBytes);
    if (modelSha256 !== productionSha256) throw new Error('Browser production model SHA mismatch');
    const modelFile = JSON.parse(new TextDecoder().decode(modelBytes));
    const model = packModel(modelFile);
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const features = ['shader-f16', 'timestamp-query'].filter(feature => adapter.features.has(feature));
    const device = await adapter.requestDevice({ requiredFeatures: features,
      requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize } });
    const gpuErrors = [];
    device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
    device.lost.then(info => { if (info.reason !== 'destroyed') gpuErrors.push(`Device lost: ${info.message}`); });
    window.m9Parity = { OldNeural, NewNeural, model, device, sha256, gpuErrors, stages: [], resources: [] };
    const info = adapter.info;
    return { userAgent: navigator.userAgent, adapter: { vendor: info.vendor, architecture: info.architecture,
      device: info.device, description: info.description }, adapterFeatures: [...adapter.features],
      deviceFeatures: [...device.features], modelSha256, declaredModelSha256: modelFile.sha256,
      model: { architecture: modelFile.architecture, features: model.features, depth: model.depth },
      servedCoreSources: servedSources, deviceLimits: { maxBufferSize: device.limits.maxBufferSize,
        maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize },
      visibility: document.visibilityState, devicePixelRatio, screen: { width: screen.width, height: screen.height },
    };
  }, { code: oldModule.code, origin, productionSha256, sourceHashes: before.sourceSha256 });

  report.video = await page.evaluate(async ({ url }) => {
    const video = document.querySelector('#source');
    const waitFor = (eventName, start) => new Promise((resolveEvent, reject) => {
      const timer = setTimeout(() => finish(new Error(`Video ${eventName} timeout`)), 20000);
      const success = () => finish();
      const failure = () => finish(new Error(video.error?.message ?? 'Video decode error'));
      const finish = error => {
        clearTimeout(timer);
        video.removeEventListener(eventName, success);
        video.removeEventListener('error', failure);
        if (error) reject(error); else resolveEvent();
      };
      video.addEventListener(eventName, success, { once: true });
      video.addEventListener('error', failure, { once: true });
      start();
    });
    video.pause();
    video.loop = false;
    await waitFor('loadeddata', () => { video.src = url; video.load(); });
    video.pause();
    await waitFor('seeked', () => { video.currentTime = 3; });
    video.pause();
    if (!video.paused || video.readyState < 2 || video.seeking) throw new Error('Video is not ready and paused');
    if (video.videoWidth !== 1280 || video.videoHeight !== 720) throw new Error('Expected a 1280x720 CFR clip');
    window.m9Parity.video = video;
    return { url: video.currentSrc, requestedTime: 3, currentTime: video.currentTime, readyState: video.readyState,
      paused: video.paused, width: video.videoWidth, height: video.videoHeight, duration: video.duration,
      playbackCalls: 0, inputScope: 'One decoded frame of one paused HTMLVideoElement, shared by old and new; no VideoFrame or canvas conversion' };
  }, { url: `/${clipFile.replace(/^public\//, '')}` });

  const precisions = report.browserSetup.deviceFeatures.includes('shader-f16') ? ['fp32', 'f16'] : ['fp32'];
  report.skipped = precisions.includes('f16') ? [] : [{ precision: 'f16', reason: 'shader-f16 unsupported' }];
  for (const precision of precisions) {
    await page.evaluate(({ precision }) => {
      const state = window.m9Parity;
      state.stages = [new state.OldNeural(state.model, { useF16: precision === 'f16' }),
        new state.NewNeural(state.model, { useF16: precision === 'f16' })];
    }, { precision });
    for (const [configIndex, [width, height]] of [[17, 13], [33, 19]].entries()) {
      console.log(`Comparing ${precision} sampled ${width}x${height}, configuration ${configIndex + 1}`);
      report.cases.push(await capturePair(page, { precision, kind: 'sampled', width, height, configIndex }));
    }
    await page.evaluate(({ precision }) => {
      const state = window.m9Parity;
      state.stages.forEach(stage => stage.destroy());
      state.stages = [new state.OldNeural(state.model, { useF16: precision === 'f16' }),
        new state.NewNeural(state.model, { useF16: precision === 'f16' })];
    }, { precision });
    console.log(`Comparing ${precision} external 1280x720 -> 2560x1440, one paused frame`);
    report.cases.push(await capturePair(page, { precision, kind: 'external', width: 1280, height: 720, configIndex: 0 }));
    await page.evaluate(() => {
      window.m9Parity.stages.forEach(stage => stage.destroy());
      window.m9Parity.stages = [];
    });
  }
  report.gpuErrors = await page.evaluate(() => window.m9Parity.gpuErrors);
  const favicon404 = error => error.kind === 'console' && error.url === `${origin}/favicon.ico`
    && error.message === 'Failed to load resource: the server responded with a status of 404 (Not Found)';
  report.nonBlockingBrowserErrors = browserErrors.filter(favicon404);
  report.nonBlockingBrowserErrorPolicy = 'Only an exact /favicon.ico 404 is non-blocking; retained here, no application resources or GPU errors excluded';
  report.browserErrors = browserErrors.filter(error => !favicon404(error));
}

async function capturePair(page, parameters) {
  return page.evaluate(async ({ precision, kind, width, height, configIndex }) => {
    const state = window.m9Parity;
    const { device, stages, video, sha256 } = state;
    await device.queue.onSubmittedWorkDone();
    for (const filter of ['validation', 'out-of-memory', 'internal']) device.pushErrorScope(filter);
    const source = { width, height };
    const target = { width: width * 2, height: height * 2 };
    const bytesPerRow = Math.ceil(target.width * 4 / 256) * 256;
    const resources = state.resources;
    const own = resource => { resources.push(resource); return resource; };
    let inputBytes;
    let sampledSourceView;
    if (kind === 'sampled') {
      inputBytes = new Uint8Array(width * height * 4);
      for (let row = 0; row < height; row++) {
        for (let column = 0; column < width; column++) {
          const offset = (row * width + column) * 4;
          const border = row === 0 || column === 0 || row === height - 1 || column === width - 1;
          inputBytes.set(border ? [column % 2 ? 255 : 0, row % 2 ? 0 : 255, (column + row) % 2 ? 255 : 0, 255]
            : [(column * 37 + row * 17 + configIndex * 41) % 256,
              (column * 11 + row * 53 + configIndex * 67) % 256,
              (column * 71 + row * 7 + configIndex * 97) % 256, 255], offset);
        }
      }
      const texture = own(device.createTexture({ size: source, format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }));
      sampledSourceView = texture.createView();
      device.queue.writeTexture({ texture }, inputBytes, { bytesPerRow: width * 4 }, source);
    }
    const outputs = stages.map(() => {
      const texture = own(device.createTexture({ size: target, format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }));
      return { texture, view: texture.createView(), readback: own(device.createBuffer({
        size: bytesPerRow * target.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })) };
    });
    for (const stage of stages) {
      stage.configure({ device, source, target, targetFormat: 'rgba8unorm', sourceKind: kind,
        ...(kind === 'sampled' ? { sampledSourceView } : {}) });
      if (stage.resolvedPrecision !== precision) throw new Error(`Unexpected precision: ${stage.resolvedPrecision}`);
    }
    const beforeTime = video.currentTime;
    if (!video.paused || video.seeking || video.readyState < 2) throw new Error('Video advanced or became unavailable');
    for (const [index, stage] of stages.entries()) {
      const encoder = device.createCommandEncoder();
      const frame = kind === 'sampled' ? { kind, view: sampledSourceView }
        : { kind, texture: device.importExternalTexture({ source: video, colorSpace: 'srgb' }) };
      stage.encode({ encoder, frame, target: outputs[index].view, timing: null });
      encoder.copyTextureToBuffer({ texture: outputs[index].texture },
        { buffer: outputs[index].readback, bytesPerRow }, target);
      device.queue.submit([encoder.finish()]);
    }
    const afterSubmitTime = video.currentTime;
    await device.queue.onSubmittedWorkDone();
    const gpuErrors = [];
    for (let scope = 0; scope < 3; scope++) {
      const error = await device.popErrorScope();
      if (error) gpuErrors.push(error.message);
    }
    if (gpuErrors.length) throw new Error(`GPU validation failed: ${gpuErrors.join('; ')}`);
    await Promise.all(outputs.map(output => output.readback.mapAsync(GPUMapMode.READ)));
    const pixels = outputs.map(output => {
      const mapped = new Uint8Array(output.readback.getMappedRange());
      const packed = new Uint8Array(target.width * target.height * 4);
      for (let row = 0; row < target.height; row++) {
        packed.set(mapped.subarray(row * bytesPerRow, row * bytesPerRow + target.width * 4), row * target.width * 4);
      }
      output.readback.unmap();
      return packed;
    });
    let maxByteDelta = 0;
    let differentBytes = 0;
    let sumByteDelta = 0;
    let borderMaxByteDelta = 0;
    let borderDifferentBytes = 0;
    const channelMaxByteDelta = [0, 0, 0, 0];
    const firstDifferences = [];
    for (let index = 0; index < pixels[0].length; index++) {
      const delta = Math.abs(pixels[0][index] - pixels[1][index]);
      const pixel = Math.floor(index / 4);
      const column = pixel % target.width;
      const row = Math.floor(pixel / target.width);
      const border = column < 2 || row < 2 || column >= target.width - 2 || row >= target.height - 2;
      maxByteDelta = Math.max(maxByteDelta, delta);
      channelMaxByteDelta[index % 4] = Math.max(channelMaxByteDelta[index % 4], delta);
      sumByteDelta += delta;
      if (border) borderMaxByteDelta = Math.max(borderMaxByteDelta, delta);
      if (delta) {
        differentBytes++;
        if (border) borderDifferentBytes++;
        if (firstDifferences.length < 8) firstDifferences.push({ column, row, channel: index % 4,
          old: pixels[0][index], current: pixels[1][index], delta });
      }
    }
    const outputEvidence = await Promise.all(pixels.map(async bytes => {
      const minimum = [255, 255, 255, 255];
      const maximum = [0, 0, 0, 0];
      for (let index = 0; index < bytes.length; index++) {
        minimum[index % 4] = Math.min(minimum[index % 4], bytes[index]);
        maximum[index % 4] = Math.max(maximum[index % 4], bytes[index]);
      }
      const coordinates = [[0, 0], [target.width - 1, 0], [0, target.height - 1],
        [target.width - 1, target.height - 1], [Math.floor(target.width / 2), Math.floor(target.height / 2)]];
      return { sha256: await sha256(bytes), minimum, maximum,
        nonUniformRgb: minimum.slice(0, 3).some((value, channel) => value !== maximum[channel]),
        opaqueAlpha: minimum[3] === 255 && maximum[3] === 255,
        samples: coordinates.map(([column, row]) => ({ column, row,
          rgba: [...bytes.slice((row * target.width + column) * 4, (row * target.width + column) * 4 + 4)] })) };
    }));
    const videoStable = beforeTime === afterSubmitTime && beforeTime === video.currentTime && video.paused && !video.seeking;
    const result = { precision, kind, configuration: configIndex + 1, source, target,
      sampledInputSha256: inputBytes ? await sha256(inputBytes) : null,
      newSampledView: kind === 'sampled', sameStageInstancesReconfigured: kind === 'sampled' && configIndex === 1,
      encodesPerStageThisConfiguration: 1, passDiagnostics: 'default true (old unconditional when timestamp-query exists)',
      timing: null, rgbaBytesCompared: pixels[0].length, maxByteDelta, differentBytes,
      meanByteDelta: sumByteDelta / pixels[0].length, channelMaxByteDelta,
      border: { outputPixelWidth: 2, maxByteDelta: borderMaxByteDelta, differentBytes: borderDifferentBytes },
      firstDifferences, outputs: { old: outputEvidence[0], current: outputEvidence[1] },
      video: { beforeTime, afterSubmitTime, afterReadbackTime: video.currentTime, paused: video.paused, stable: videoStable },
      gpuErrors, passed: differentBytes === 0 && videoStable && outputEvidence.every(output => output.nonUniformRgb && output.opaqueAlpha) };
    resources.splice(0).forEach(resource => resource.destroy());
    return result;
  }, parameters);
}

async function main() {
  const report = { schemaVersion: 1, startedAt: new Date().toISOString(), passed: false,
    scope: 'Offline old/new GPU RGBA8 output parity only, not a quality, performance, sustained playback, or timestamp-lifetime result',
    acceptance: { maxByteDelta: 0, rule: 'Exact equality of every RGBA8 byte; no one-LSB tolerance' },
    inputPattern: 'Opaque RGBA8: alternating saturated RGB borders; independent deterministic RGB interior formulas in this runner',
    readback: 'Targets and readback buffers allocated before encode; each import/bind/encode/copy/submit synchronous; map only after both submissions and queue completion',
    performance: 'not measured', cases: [], errors: [] };
  let server;
  let native;
  let page;
  let before;
  let watchdog;
  try {
    before = sourceSnapshot();
    assert.equal(before.modelSha256, productionSha256, 'Frozen production model mismatch');
    assert.equal(before.clipSha256, clipSha256, 'Frozen CFR clip mismatch');
    git('check-ignore', '--', resolve(cache, 'old-neural-upscaler.mjs'));
    const source = execFileSync('git', ['show', `${oldRef}:${sourceFile}`], { cwd: root, encoding: 'utf8' });
    const oldModule = historicalModule(source);
    mkdirSync(cache, { recursive: true });
    writeFileSync(resolve(cache, 'old-neural-upscaler.mjs'), oldModule.code);
    report.sourceBefore = before;
    report.sources = { old: { ref: oldRef, commit: git('rev-parse', oldRef), file: sourceFile, sha256: hash(source) },
      current: { head: before.head, file: sourceFile, sha256: before.sourceSha256[sourceFile],
        claim: 'Direct working-file SHA, not a clean-tree claim; other tracked working changes are recorded separately' },
      typescriptVersion: ts.version, transpiledSha256: oldModule.transpiledSha256, blobModuleSha256: oldModule.blobModuleSha256,
      transform: 'TypeScript ES2022/ESNext transpilation; only relative from specifiers remapped to absolute Vite /src/core/*.ts URLs',
      sharedDependencies: oldModule.imports, runnerSha256: digest('tools/m9-output-parity.mjs') };
    report.machine = { hostname: hostname(), cpu: cpus()[0]?.model, os: execFileSync('sw_vers', { encoding: 'utf8' }).trim(),
      kernel: release(), hardwareModel: execFileSync('sysctl', ['-n', 'hw.model'], { encoding: 'utf8' }).trim(),
      node: process.version, displayRefreshRate: 'not measured' };
    report.clip = { file: clipFile, sha256: before.clipSha256,
      probe: JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration', '-of', 'json', resolve(root, clipFile)], { encoding: 'utf8' })) };
    const stream = report.clip.probe.streams[0];
    assert.equal(stream.width, 1280);
    assert.equal(stream.height, 720);
    assert.equal(stream.r_frame_rate, stream.avg_frame_rate, 'Clip frame-rate metadata does not agree');
    let existing;
    try { existing = await fetch(`${origin}/bench.html`, { signal: AbortSignal.timeout(3000) }); }
    catch (error) { if (error.cause?.code !== 'ECONNREFUSED') throw error; }
    if (existing) {
      assert.ok(existing.ok, 'Existing server did not serve bench.html');
      assert.ok(!(await existing.text()).includes('/@vite/client'), 'Existing server must use benchmark mode');
      report.server = { origin, mode: 'benchmark', ownership: 'reused; left running' };
    } else {
      server = await createServer({ root, mode: 'benchmark', server: { port: 5173, strictPort: true } });
      await server.listen();
      report.server = { origin, mode: 'benchmark', ownership: 'runner-owned; close on completion' };
    }
    native = await openNativeChrome(['--enable-unsafe-webgpu', '--window-size=1280,900']);
    report.browser = { version: native.browser.version(), executable: native.executable,
      launch: 'openNativeChrome: mock keychain, native default context, CDP noDefaults=true' };
    page = await native.context.newPage();
    watchdog = setTimeout(() => { report.errors.push('Browser run exceeded 180 seconds'); void page.close().catch(() => {}); }, 180000);
    await runBrowser(page, oldModule, before, report);
    report.passed = report.cases.length >= 3 && report.cases.every(test => test.passed)
      && report.gpuErrors.length === 0 && report.browserErrors.length === 0 && report.errors.length === 0;
  } catch (error) {
    report.errors.push(error.stack ?? String(error));
    report.passed = false;
  } finally {
    clearTimeout(watchdog);
    try {
      if (page && !page.isClosed()) await page.evaluate(() => {
        const state = window.m9Parity;
        if (!state) return;
        state.video?.pause();
        state.stages.forEach(stage => stage.destroy());
        state.resources.forEach(resource => resource.destroy());
        state.device.destroy();
      });
    } catch (error) { report.errors.push(`GPU cleanup: ${error.message}`); }
    try { await native?.close(); report.nativeChromeClosed = !!native; }
    catch (error) { report.errors.push(`Chrome cleanup: ${error.message}`); }
    try { await server?.close(); }
    catch (error) { report.errors.push(`Server cleanup: ${error.message}`); }
    if (before) {
      try {
        report.sourceAfter = sourceSnapshot();
        assert.deepEqual(report.sourceAfter, before, 'Source, model, clip or tracked diff changed during capture');
        report.sourceAndModelInvariant = true;
      } catch (error) { report.errors.push(error.message); report.sourceAndModelInvariant = false; }
    }
    report.passed &&= report.errors.length === 0 && report.sourceAndModelInvariant === true;
    report.completedAt = new Date().toISOString();
    writeFileSync(resultFile, `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({ passed: report.passed, cases: report.cases.map(test => ({ precision: test.precision,
      kind: test.kind, source: test.source, maxByteDelta: test.maxByteDelta, differentBytes: test.differentBytes, passed: test.passed })),
      errors: report.errors, result: resultFile, nativeChromeClosed: report.nativeChromeClosed }, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
}

await main();