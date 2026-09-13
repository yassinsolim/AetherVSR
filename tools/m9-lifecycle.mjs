import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { closeSync, existsSync, openSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const modelPath = '/models/aethersr-c16d2.json';
const productionSha256 = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
const clip720 = '/media/aethervsr-testclip-720p60-h264.mp4';
const clip1080 = '/media/m9/1080p60.mp4';
const timeout = 45000;
const digest = file => createHash('sha256').update(readFileSync(file)).digest('hex');
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
const check = (condition, message) => assert.ok(condition, message);

async function snapshot(page) {
  return page.evaluate(() => {
    const api = window.aethervsrRuntime;
    const video = api?.video;
    return {
      runtime: api?.snapshot() ?? null,
      stats: api?.pipeline.stats(performance.now()) ?? null,
      memory: api?.neural()?.memoryReport ?? null,
      precision: api?.neural()?.resolvedPrecision ?? null,
      timingGeneration: api?.pipeline.timingGeneration ?? null,
      status: document.querySelector('#status')?.textContent,
      statusLevel: document.querySelector('#status')?.dataset.level,
      menu: { value: document.querySelector('#upscaler')?.value,
        disabled: document.querySelector('#upscaler')?.disabled },
      canvas: { width: document.querySelector('#output')?.width,
        height: document.querySelector('#output')?.height },
      video: video ? { width: video.videoWidth, height: video.videoHeight,
        currentTime: video.currentTime, duration: video.duration, paused: video.paused,
        loop: video.loop, src: video.currentSrc, playbackRate: video.playbackRate } : null,
      visibility: document.visibilityState,
      focused: document.hasFocus(),
      now: performance.now(),
      evidence: window.m9Lifecycle ?? null,
    };
  });
}

async function observe(page) {
  await page.waitForFunction(() => Boolean(window.aethervsrRuntime), null, { timeout });
  await page.evaluate(() => {
    const api = window.aethervsrRuntime;
    const initial = api.snapshot();
    const evidence = window.m9Lifecycle = {
      states: [], configurations: [], samples: 0, staleSamples: 0,
      continuityErrors: [], lastFrames: initial.session.framesRendered,
      minimumFrames: initial.session.framesRendered, loopEvents: 0,
      environment: { userAgent: navigator.userAgent, platform: navigator.platform,
        devicePixelRatio, screen: { width: screen.width, height: screen.height },
        displayRefreshRate: 'not measured' },
    };
    const previousChange = api.driver.onChange;
    api.driver.onChange = state => {
      previousChange?.(state);
      const last = evidence.states.at(-1);
      if (!last || ['state', 'mode', 'tier', 'generation', 'reason'].some(key => last[key] !== state[key])) {
        const { transitions, ...current } = state;
        evidence.states.push({ ...current, observedAt: performance.now(), retainedTransitions: transitions.length });
      }
    };
    const previousConfigure = api.driver.onConfigure;
    api.driver.onConfigure = config => {
      previousConfigure?.(config);
      evidence.configurations.push(structuredClone(config));
    };
    const previousSample = api.driver.onSample;
    api.driver.onSample = sample => {
      previousSample?.(sample);
      evidence.samples++;
      if (sample.generation !== api.snapshot().controller.generation) evidence.staleSamples++;
      evidence.lastSample = structuredClone(sample);
    };
    let lastMediaTime = api.video.currentTime;
    const previousFrame = api.driver.onFrame;
    api.driver.onFrame = tick => {
      previousFrame?.(tick);
      const frames = api.driver.session.snapshot(performance.now()).framesRendered;
      evidence.lastTick = structuredClone(tick);
      if (frames < evidence.lastFrames || (evidence.lastFrames > 0 && frames === 0)) {
        evidence.continuityErrors.push({ before: evidence.lastFrames, after: frames });
      }
      evidence.lastFrames = frames;
      if (frames > 0) evidence.minimumFrames = evidence.minimumFrames > 0
        ? Math.min(evidence.minimumFrames, frames) : frames;
      if (tick.mediaTime < lastMediaTime) evidence.loopEvents++;
      lastMediaTime = tick.mediaTime;
    };
  });
}

async function advance(page, count, minimumMs = 0) {
  const before = await snapshot(page);
  await page.waitForFunction(({ frames, count, start, minimumMs }) => {
    const state = window.aethervsrRuntime.snapshot();
    return state.session.framesRendered >= frames + count && performance.now() - start >= minimumMs;
  }, { frames: before.runtime.session.framesRendered, count, start: before.now, minimumMs }, { timeout });
  const after = await snapshot(page);
  check(after.runtime.session.framesRendered >= before.runtime.session.framesRendered + count,
    `Expected at least ${count} additional rendered frames`);
  assert.deepEqual(after.evidence.continuityErrors, [], 'Session frame count regressed');
  return { before, after, elapsedMs: after.now - before.now, requestedFrames: count,
    scope: 'Browser wall time and cumulative rendered-frame advancement; NOT a performance benchmark' };
}

function baselineOnly(before, after) {
  const states = after.evidence.states.slice(before.evidence.states.length);
  check(after.runtime.actualTier === 'baseline', 'Actual pipeline must be baseline');
  check(states.every(state => state.tier === 'baseline' && state.state !== 'probing'),
    'Baseline-only interval entered neural or probing');
  check(after.runtime.controller.state !== 'probing', 'Unexpected probe at interval end');
  assert.equal(after.runtime.controller.probeCount, before.runtime.controller.probeCount,
    'Baseline-only interval started a probe');
}

async function manualBaseline(page) {
  await page.selectOption('#upscaler', 'baseline');
  const interval = await advance(page, 180, 3000);
  baselineOnly(interval.before, interval.after);
  check(interval.after.runtime.controller.mode === 'baseline', 'Manual baseline mode changed');
  check(interval.after.menu.value === 'baseline', 'Mode menu disagrees with controller');
  return interval;
}

async function stable(page) {
  await page.waitForFunction(() => {
    const state = window.aethervsrRuntime.snapshot();
    return state.controller.mode === 'neural' && state.controller.state === 'stable' &&
      state.actualTier === 'neural' && state.running;
  }, null, { timeout });
  return snapshot(page);
}

function intent(before, after, mode) {
  assert.equal(after.runtime.controller.mode, mode, 'Manual intent changed');
  assert.equal(after.menu.value, mode, 'Menu lost manual intent');
  check(after.evidence.states.slice(before.evidence.states.length).every(state => state.mode === mode),
    'An intermediate state changed manual intent');
  check(after.runtime.session.framesRendered >= before.runtime.session.framesRendered,
    'Cumulative session frames regressed');
  assert.deepEqual(after.evidence.continuityErrors, []);
}

async function seek(page, nearEnd = false) {
  await page.evaluate(async ({ nearEnd, timeout }) => {
    const video = window.aethervsrRuntime.video;
    if (!Number.isFinite(video.duration) || video.duration < 2) throw new Error('Clip must be at least 2 s');
    const destination = nearEnd ? video.duration - 0.8
      : video.currentTime < video.duration / 2 ? video.duration / 2 : video.duration / 4;
    await new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer);
        video.removeEventListener('seeked', finished);
        video.removeEventListener('error', failed);
      };
      const finished = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error('Video failed during seek')); };
      const timer = setTimeout(() => { cleanup(); reject(new Error('Timed out waiting for real seeked event')); }, timeout);
      video.addEventListener('seeked', finished, { once: true });
      video.addEventListener('error', failed, { once: true });
      video.currentTime = destination;
    });
  }, { nearEnd, timeout });
}

async function seekAndLoop(page, mode) {
  const before = await snapshot(page);
  await seek(page);
  await advance(page, 30);
  const sought = await snapshot(page);
  intent(before, sought, mode);
  check(sought.timingGeneration > before.timingGeneration, 'Seek must invalidate timing generation');
  if (mode === 'neural') await stable(page);
  check(sought.video.loop, 'Bundled app video must actually loop');
  await seek(page, true);
  const positioned = await snapshot(page);
  await page.waitForFunction(({ wraps, duration }) => {
    const api = window.aethervsrRuntime;
    return window.m9Lifecycle.loopEvents > wraps && api.video.currentTime < duration / 2 && !api.video.seeking;
  }, { wraps: positioned.evidence.loopEvents, duration: positioned.video.duration }, { timeout });
  await advance(page, 30);
  if (mode === 'neural') await stable(page);
  const after = await snapshot(page);
  intent(before, after, mode);
  check(after.timingGeneration > positioned.timingGeneration, 'Loop must invalidate timing generation');
  if (mode === 'baseline') baselineOnly(before, after);
  return { before, sought, positioned, after, scope: 'Real seeked events and a subsequent natural media-time wrap' };
}

function dimensions(state, width, height) {
  const source = { width, height };
  const target = { width: width * 2, height: height * 2 };
  assert.deepEqual(state.runtime.session.source, source, 'Session source metadata');
  assert.deepEqual(state.stats.sourceSize, source, 'Pipeline source metadata');
  assert.deepEqual(state.runtime.session.target, target, 'Session output metadata');
  assert.deepEqual(state.stats.targetSize, target, 'Pipeline output metadata');
  assert.deepEqual(state.canvas, target, 'Actual output canvas backing dimensions');
  assert.equal(state.video.width, width);
  assert.equal(state.video.height, height);
  assert.equal(state.runtime.controller.width, width);
  assert.equal(state.runtime.controller.height, height);
  assert.equal(state.runtime.controller.generation, state.timingGeneration, 'Controller generation binding');
}

function memoryGeometry(state, width, height) {
  check(['f16', 'fp32'].includes(state.precision), 'Neural precision must be exposed');
  check(state.memory !== null, 'Configured neural memory must be exposed');
  const activation = width * height * 16 * (state.precision === 'f16' ? 2 : 4);
  assert.equal(state.memory.activationPing, activation, 'C16 ping bytes');
  assert.equal(state.memory.activationPong, activation, 'C16 pong bytes');
  assert.equal(state.memory.outputTexture, width * height * 16, '2x RGBA output bytes');
  const { total, ...parts } = state.memory;
  check(Object.values(parts).every(bytes => Number.isSafeInteger(bytes) && bytes > 0), 'Memory component bytes');
  assert.equal(total, Object.values(parts).reduce((sum, bytes) => sum + bytes, 0), 'Memory total');
}

async function replaceSource(page, clip, width, height) {
  const before = await snapshot(page);
  const pageUrl = page.url();
  const absolute = resolve(root, `public${clip}`);
  await page.setInputFiles('#file', absolute);
  await page.waitForFunction(({ width, height, load }) => {
    const api = window.aethervsrRuntime;
    const state = api.snapshot();
    return api.video.currentSrc.startsWith('blob:') && api.video.videoWidth === width &&
      api.video.videoHeight === height && state.session.source?.width === width &&
      state.session.loadGeneration > load && !api.video.paused;
  }, { width, height, load: before.runtime.session.loadGeneration }, { timeout });
  await stable(page);
  const after = await snapshot(page);
  assert.equal(page.url(), pageUrl, 'File picker must not navigate or reload');
  check(after.evidence.states.length >= before.evidence.states.length, 'Observation realm was replaced');
  check(after.runtime.session.framesRendered > before.runtime.session.framesRendered, 'Source switch reset/stopped frames');
  check(after.runtime.session.configurationCount > before.runtime.session.configurationCount,
    'Source change did not configure resources');
  check(after.runtime.session.generation > before.runtime.session.generation, 'Configuration generation did not advance');
  check(after.timingGeneration > before.timingGeneration, 'Timing generation did not advance');
  dimensions(after, width, height);
  memoryGeometry(after, width, height);
  intent(before, after, 'neural');
  const held = await advance(page, 60);
  assert.equal(held.after.runtime.session.configurationCount, held.before.runtime.session.configurationCount,
    'Steady playback unexpectedly reconfigured resources');
  assert.deepEqual(held.after.memory, after.memory, 'Steady playback changed owned memory');
  return { absolute, sha256: digest(absolute), before, after, held };
}

async function elapsed(page, milliseconds) {
  return page.evaluate(async milliseconds => {
    const start = performance.now();
    await new Promise(resolve => setTimeout(resolve, milliseconds));
    return { requestedMs: milliseconds, elapsedMs: performance.now() - start,
      scope: 'Real browser timer, not fake time; elapsed duration is not a performance result' };
  }, milliseconds);
}

function suspended(before, after) {
  assert.equal(before.runtime.controller.state, 'suspended');
  assert.equal(after.runtime.controller.state, 'suspended');
  assert.equal(after.runtime.running, false);
  assert.equal(after.runtime.session.active, false);
  for (const key of ['activeMs', 'backoffMs', 'probeCount', 'failedProbeCount', 'fallbackMs', 'mode', 'tier']) {
    assert.equal(after.runtime.controller[key], before.runtime.controller[key], `Suspension changed ${key}`);
  }
  assert.equal(after.runtime.session.framesRendered, before.runtime.session.framesRendered, 'Suspended rendering');
  assert.equal(after.runtime.session.configurationCount, before.runtime.session.configurationCount, 'Suspended configure');
  assert.equal(after.runtime.session.activeMs, before.runtime.session.activeMs, 'Suspended session clock');
  if (before.runtime.controller.nextProbeAtMs !== null) {
    check(after.runtime.controller.nextProbeAtMs !== null, 'Suspension lost pending probe');
    const remainingBefore = before.runtime.controller.nextProbeAtMs - before.now;
    const remainingAfter = after.runtime.controller.nextProbeAtMs - after.now;
    check(Math.abs(remainingAfter - remainingBefore) < 5, 'Suspension consumed backoff wait');
  }
  check(after.evidence.states.slice(before.evidence.states.length).every(state => state.state === 'suspended'),
    'Controller escaped suspension during a nonactive interval');
}

async function pauseResume(page, fallback) {
  await page.evaluate(fallback => {
    const api = window.aethervsrRuntime;
    if (fallback) { api.force(true); api.force(false); }
    api.video.pause();
  }, fallback);
  await page.waitForFunction(() => window.aethervsrRuntime.snapshot().controller.state === 'suspended', null, { timeout });
  const before = await snapshot(page);
  if (fallback) check(before.runtime.controller.nextProbeAtMs !== null, 'Need a real pending fallback probe');
  const quiet = await elapsed(page, 250);
  const after = await snapshot(page);
  check(quiet.elapsedMs >= 200, 'Pause interval must span at least 200 real milliseconds');
  suspended(before, after);
  await page.evaluate(() => window.aethervsrRuntime.video.play());
  await advance(page, 30);
  const resumed = await stable(page);
  intent(before, resumed, 'neural');
  check(resumed.runtime.controller.activeMs > after.runtime.controller.activeMs, 'Active clock did not resume');
  return { fallback, before, quiet, after, resumed };
}

async function background(page) {
  await page.evaluate(() => {
    const api = window.aethervsrRuntime;
    api.force(true);
  });
  const cover = await page.context().newPage();
  const cdp = await page.context().newCDPSession(page);
  let frozen = false;
  try {
    await cover.goto('about:blank');
    await cover.bringToFront();
    await page.waitForFunction(() => document.visibilityState === 'hidden' &&
      window.aethervsrRuntime.snapshot().controller.state === 'suspended', null, { polling: 100, timeout });
    await page.evaluate(() => window.aethervsrRuntime.force(false));
    const before = await snapshot(page);
    assert.equal(before.runtime.controller.forced, false, 'Hidden probe must not be suppressed by force');
    check(before.runtime.controller.nextProbeAtMs !== null, 'Hidden test requires a pending probe');
    await cdp.send('Page.setWebLifecycleState', { state: 'frozen' });
    frozen = true;
    const quiet = await elapsed(cover, 250);
    await cdp.send('Page.setWebLifecycleState', { state: 'active' });
    frozen = false;
    const after = await snapshot(page);
    assert.equal(after.visibility, 'hidden', 'Thaw must still be in a genuinely hidden tab');
    check(quiet.elapsedMs >= 200, 'Frozen interval must span at least 200 real milliseconds');
    suspended(before, after);
    await page.bringToFront();
    await page.waitForFunction(() => document.visibilityState === 'visible' &&
      window.aethervsrRuntime.snapshot().running, null, { timeout });
    await advance(page, 30);
    const resumed = await stable(page);
    intent(before, resumed, 'neural');
    return { before, quiet, after, resumed, scope: 'Actual hidden tab, CDP frozen/active, then foreground resume' };
  } finally {
    if (frozen) await cdp.send('Page.setWebLifecycleState', { state: 'active' }).catch(() => {});
    await cdp.detach();
    await cover.close();
    await page.bringToFront();
  }
}

async function capture(page, stem, label, verifyPixels = false) {
  const screenFile = `${stem}.${label}.png`;
  writeFileSync(screenFile, await page.screenshot({ fullPage: true }), { flag: 'wx' });
  const result = { screenFile, snapshot: await snapshot(page) };
  if (!verifyPixels) return result;
  const canvas = page.locator('#output');
  const box = await canvas.boundingBox();
  check(box && box.width > 0 && box.height > 0, 'Output canvas is not visible');
  const png = await canvas.screenshot();
  const canvasFile = `${stem}.${label}.canvas.png`;
  writeFileSync(canvasFile, png, { flag: 'wx' });
  const pixels = await page.evaluate(async bytes => {
    const image = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type: 'image/png' }));
    try {
      const decoded = new OffscreenCanvas(image.width, image.height);
      const context = decoded.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Screenshot PNG decoder unavailable');
      context.drawImage(image, 0, 0);
      const data = context.getImageData(0, 0, image.width, image.height).data;
      let nonBlack = 0;
      let minimum = 255;
      let maximum = 0;
      const colors = new Set();
      for (let offset = 0; offset < data.length; offset += 4) {
        const red = data[offset], green = data[offset + 1], blue = data[offset + 2];
        const brightness = Math.max(red, green, blue);
        if (brightness > 16 && data[offset + 3] > 0) nonBlack++;
        minimum = Math.min(minimum, brightness);
        maximum = Math.max(maximum, brightness);
        if (colors.size < 256) colors.add((red << 16) | (green << 8) | blue);
      }
      return { width: image.width, height: image.height, nonBlack, minimum, maximum,
        distinctColorsCapped256: colors.size, totalPixels: image.width * image.height,
        scope: 'Decoded Playwright canvas-element screenshot PNG, outside the render callback; no WebGPU pixel readback' };
    } finally { image.close(); }
  }, [...png]);
  result.canvasFile = canvasFile;
  result.pixels = pixels;
  check(pixels.nonBlack / pixels.totalPixels > 0.01 && pixels.maximum - pixels.minimum > 16 &&
    pixels.distinctColorsCapped256 > 16, `Blank or near-uniform canvas screenshot: ${JSON.stringify(pixels)}`);
  return result;
}

async function fatalDevice(page, quietMs) {
  await stable(page);
  await advance(page, 10);
  await page.evaluate(() => {
    const api = window.aethervsrRuntime;
    const evidence = window.m9Lifecycle;
    const frame = api.pipeline.onFrame;
    const sample = api.pipeline.onGpuSample;
    const configure = api.pipeline.onConfiguration;
    const tick = structuredClone(evidence.lastTick);
    const timing = structuredClone(evidence.lastSample);
    const configuration = structuredClone(evidence.configurations.at(-1));
    if (!tick || !timing || !configuration) throw new Error('Missing real callback payloads for late-delivery check');
    evidence.lateDeliveries = 0;
    evidence.lateFactoryCalls = 0;
    const released = new Promise(resolve => { window.m9ReleaseLate = resolve; });
    window.m9LateDone = released.then(() => {
      frame(tick);
      sample(timing);
      configure(configuration);
      evidence.lateDeliveries += 3;
      api.driver.setNeuralFactory(() => {
        evidence.lateFactoryCalls++;
        throw new Error('A late model factory must never execute after failure');
      });
    });
    api.loseDevice();
  });
  await page.waitForFunction(() => window.aethervsrRuntime.snapshot().controller.state === 'failed' &&
    document.querySelector('#upscaler').disabled && document.querySelector('#status').dataset.level === 'error',
  null, { timeout });
  const before = await snapshot(page);
  check(/reload/i.test(before.status), 'Device-loss status must explain recovery');
  await page.evaluate(async () => {
    const api = window.aethervsrRuntime;
    api.force(true);
    api.force(false);
    for (const mode of ['baseline', 'auto', 'neural']) api.setMode(mode);
    window.m9ReleaseLate();
    await window.m9LateDone;
    api.video.pause();
    await api.video.play();
  });
  const quiet = await elapsed(page, Math.ceil(quietMs));
  const after = await snapshot(page);
  assert.deepEqual(after.runtime, before.runtime, 'Late work or elapsed timers changed the failed runtime');
  assert.equal(after.status, before.status, 'Late work overwrote fatal status');
  assert.equal(after.statusLevel, 'error');
  assert.equal(after.menu.disabled, true);
  assert.equal(after.menu.value, before.menu.value);
  assert.equal(after.evidence.lateDeliveries, 3);
  assert.equal(after.evidence.lateFactoryCalls, 0);
  assert.equal(after.evidence.samples, before.evidence.samples, 'Failed driver accepted late timing callbacks');
  assert.equal(after.runtime.session.configurationCount, before.runtime.session.configurationCount);
  assert.equal(after.runtime.session.framesRendered, before.runtime.session.framesRendered);
  return { before, quiet, after,
    scope: 'Real device.destroy; held promise delivers previously observed callback payloads and a late factory; quiet window uses this run\'s measured manual-baseline interval' };
}

function fingerprint() {
  const files = git('ls-files', '-z', '--cached', '--others', '--exclude-standard', '--',
    'src', 'index.html', 'package.json', 'package-lock.json', 'vite.config.ts', 'tsconfig.json',
    'tools/m9-lifecycle.mjs', 'test/runtime-controller.test.ts', 'test/runtime-driver.test.ts').split('\0').filter(Boolean);
  return { head: git('rev-parse', 'HEAD'),
    source: Object.fromEntries([...new Set(files)].sort().map(file => [file, digest(resolve(root, file))])),
    productionModel: { file: modelPath, sha256: digest(resolve(root, `public${modelPath}`)) },
    media: Object.fromEntries([clip720, clip1080].map(clip => [clip, digest(resolve(root, `public${clip}`))])) };
}

function idleCalibration() {
  const rows = execFileSync('ps', ['-axo', 'pid=,ppid=,command='], { encoding: 'utf8' }).split('\n')
    .map(line => line.match(/^\s*(\d+)\s+(\d+)\s+(.+)$/)).filter(Boolean)
    .map(match => ({ pid: Number(match[1]), parent: Number(match[2]), command: match[3] }));
  const ancestors = new Set([process.pid]);
  let parent = process.ppid;
  while (parent && !ancestors.has(parent)) {
    ancestors.add(parent);
    parent = rows.find(row => row.pid === parent)?.parent;
  }
  const conflicts = rows.filter(row => !ancestors.has(row.pid) &&
    /tools\/m9-(?:runtime|calibration|lifecycle)\.mjs(?:\s|$)|startCalibration\s*\(|runtime-bench\.ts/.test(row.command));
  check(conflicts.length === 0, `Concurrent M9 calibration/lifecycle process detected (PIDs ${conflicts.map(row => row.pid).join(', ')})`);
  return { checkedAt: new Date().toISOString(), conflictingProcesses: conflicts.length,
    scope: 'Process argv scan plus required operator idle declaration; cannot detect arbitrary code in another browser' };
}

function environment() {
  const result = { node: process.version, platform: process.platform, arch: process.arch,
    displayRefreshRate: 'not measured', gpuPerformance: 'not measured' };
  if (process.platform === 'darwin') {
    result.os = execFileSync('sw_vers', ['-productVersion'], { encoding: 'utf8' }).trim();
    result.osBuild = execFileSync('sw_vers', ['-buildVersion'], { encoding: 'utf8' }).trim();
    const hardware = JSON.parse(execFileSync('system_profiler',
      ['SPHardwareDataType', 'SPDisplaysDataType', '-json'], { encoding: 'utf8', timeout: 30000 }));
    result.machine = hardware.SPHardwareDataType.map(row => ({ name: row.machine_name,
      chip: row.chip_type, memory: row.physical_memory }));
    result.displays = hardware.SPDisplaysDataType.map(row => ({ gpu: row.sppci_model,
      displays: row.spdisplays_ndrvs?.map(display => ({ name: display._name, resolution: display._spdisplays_resolution })) }));
  }
  return result;
}

const usage = `Usage: node tools/m9-lifecycle.mjs [output.json] --run --code-clean-gate=passed --calibration-idle
Default output: results/m9-lifecycle.json (exclusive wx; screenshots beside JSON).
Run only after the root operator requests execution and declares the code gate passed and calibration idle.
Start the root Vite DEV app separately. This script neither starts a server nor installs anything.
Environment: M9_BASE_URL (default http://127.0.0.1:5173/), M9_CHROME_EXECUTABLE_PATH (optional),
PLAYWRIGHT_BROWSERS_PATH (default <repo>/.cache/m9/browsers).
Uses isolated .cache/m9 Playwright, one new headful browser, temporary profile, serial GPU pages.
Static check only: node --check tools/m9-lifecycle.mjs`;

async function run() {
  const args = process.argv.slice(2);
  if (args.includes('--help')) { console.log(usage); return; }
  const flags = ['--run', '--code-clean-gate=passed', '--calibration-idle'];
  const positional = args.filter(arg => !arg.startsWith('--'));
  check(positional.length <= 1 && args.every(arg => !arg.startsWith('--') || flags.includes(arg)), usage);
  check(flags.every(flag => args.includes(flag)), usage);
  const output = resolve(root, positional[0] ?? 'results/m9-lifecycle.json');
  const stem = output.replace(/\.json$/i, '');
  const descriptor = openSync(output, 'wx');
  const names = ['manual-baseline', 'baseline-seek-loop', 'prefer-neural-desktop', 'neural-seek-loop',
    'source-picker-round-trip', 'pause-resume-neural', 'pause-resume-backoff', 'hidden-frozen-resume',
    'mobile-canvas', 'fatal-device-loss', 'timestamps-unavailable', 'delayed-model-manual-race'];
  const report = { schema: 'aethervsr.m9-lifecycle/1', kind: 'lifecycle correctness, NOT a benchmark',
    startedAt: new Date().toISOString(), valid: false,
    gating: { rootExecutionRequested: true, codeCleanGate: 'operator-declared passed; not rerun by this script',
      calibrationIdle: 'operator-declared; process scan also required', checks: [] },
    tests: names.map(name => ({ name, status: 'not-run' })), pages: [], screenshots: [], errors: [],
    limits: [
      'GPU/CPU performance and display refresh rate: not measured. Raw app snapshots carry their own timing scopes; observer hooks perturb execution.',
      'Pixel checks prove nonblank screenshot raster, not super-resolution quality or absence of all visual defects. Desktop/mobile PNGs need human review.',
      'Mobile means viewport resize in desktop Chromium, not a physical mobile GPU.',
      'Exhaustive stale-generation rejection is covered separately by test/runtime-controller.test.ts (720 to 1080 to 720) and test/runtime-driver.test.ts (old-generation samples); not rerun here.',
      'Late callbacks use real saved payloads behind an explicit promise barrier; this is not an exhaustive proof of every asynchronous race.',
      'Process scan cannot police unknown external GPU users; keep calibration and other GPU workloads idle throughout.',
      'Owned-memory reports/configuration counters do not measure browser/driver memory or prove all resources were freed.',
    ] };
  let browser;
  let context;
  let page;
  let current;
  let pageRecord;
  const fail = error => {
    const failure = { message: error.message ?? String(error), stack: error.stack ?? null };
    report.errors.push(failure);
    if (current?.status === 'running') { current.status = 'failed'; current.error = failure; }
  };
  try {
    report.gating.checks.push(idleCalibration());
    report.sourceBefore = fingerprint();
    assert.equal(report.sourceBefore.productionModel.sha256, productionSha256, 'Production model changed');
    report.environment = environment();
    const base = new URL(process.env.M9_BASE_URL ?? 'http://127.0.0.1:5173/');
    check(['http:', 'https:'].includes(base.protocol) && ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname) &&
      base.pathname === '/' && !base.search && !base.hash && !base.username && !base.password,
    'M9_BASE_URL must be the local root DEV app origin');
    process.env.PLAYWRIGHT_BROWSERS_PATH ??= resolve(root, '.cache/m9/browsers');
    const { chromium } = await import(pathToFileURL(resolve(root, '.cache/m9/node_modules/playwright/index.mjs')).href);
    const executablePath = process.env.M9_CHROME_EXECUTABLE_PATH
      ? resolve(process.env.M9_CHROME_EXECUTABLE_PATH) : chromium.executablePath();
    check(existsSync(executablePath), `Chrome executable missing: ${executablePath}`);
    const launchArgs = ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis',
      '--disable-dawn-features=timestamp_quantization', '--autoplay-policy=no-user-gesture-required',
      '--window-position=0,0', '--window-size=1280,900'];
    report.environment.launch = { executablePath, args: launchArgs, headless: false,
      baseUrl: base.href, browsersPath: process.env.PLAYWRIGHT_BROWSERS_PATH,
      playwright: JSON.parse(readFileSync(resolve(root, '.cache/m9/node_modules/playwright/package.json'), 'utf8')).version,
      profile: 'New temporary Playwright profile; no personal Chrome connection' };
    report.gating.checks.push(idleCalibration());
    browser = await chromium.launch({ headless: false, executablePath, args: launchArgs });
    report.environment.browser = browser.version();

    const freshPage = async () => {
      if (context) await context.close();
      context = await browser.newContext({ viewport: { width: 1200, height: 820 }, serviceWorkers: 'block' });
      page = await context.newPage();
      page.setDefaultTimeout(timeout);
      page.setDefaultNavigationTimeout(timeout);
      pageRecord = { page: report.pages.length + 1, errors: [], console: [], documents: [] };
      report.pages.push(pageRecord);
      const record = pageRecord;
      page.on('pageerror', error => record.errors.push(error.message));
      page.on('console', message => {
        if (['error', 'warning'].includes(message.type())) record.console.push({ type: message.type(), text: message.text() });
      });
      page.on('framenavigated', frame => { if (frame === page.mainFrame()) record.documents.push(frame.url()); });
      return page;
    };
    const navigate = async (params, waitModel = true) => {
      const target = new URL(base);
      target.search = new URLSearchParams({ clip: clip720, ...params }).toString();
      let model;
      const responsePromise = waitModel ? page.waitForResponse(new URL(modelPath, base).href)
        .then(async response => {
          check(response.ok(), `Model HTTP ${response.status()}`);
          return { url: response.url(), sha256: createHash('sha256').update(await response.body()).digest('hex') };
        }).catch(error => { model = { error: error.message }; }) : null;
      await page.goto(target.href, { waitUntil: 'domcontentloaded' });
      await page.bringToFront();
      await observe(page);
      if (waitModel) {
        model = (await responsePromise) ?? model;
        pageRecord.model = model;
        assert.equal(model?.sha256, productionSha256, `Served production model mismatch: ${JSON.stringify(model)}`);
      }
      await page.waitForFunction(() => window.aethervsrRuntime.snapshot().session.framesRendered > 0, null, { timeout });
    };
    const step = async (name, action) => {
      current = report.tests.find(test => test.name === name);
      current.status = 'running';
      console.log(`START ${name}`);
      report.gating.checks.push(idleCalibration());
      current.evidence = await action();
      check(pageRecord.errors.length === 0, `Page errors: ${pageRecord.errors.join('; ')}`);
      current.capture = await capture(page, stem, name);
      report.screenshots.push(current.capture.screenFile);
      current.status = 'passed';
      console.log(`PASS ${name}`);
      return current.evidence;
    };

    await freshPage();
    await navigate({ mode: 'baseline' });
    const manual = await step('manual-baseline', () => manualBaseline(page));
    await step('baseline-seek-loop', () => seekAndLoop(page, 'baseline'));
    let knownMemory;
    await step('prefer-neural-desktop', async () => {
      await page.selectOption('#upscaler', 'neural');
      const qualified = await stable(page);
      dimensions(qualified, 1280, 720);
      memoryGeometry(qualified, 1280, 720);
      knownMemory = qualified.memory;
      const interval = await advance(page, 60);
      assert.equal(interval.after.runtime.session.configurationCount, qualified.runtime.session.configurationCount);
      const image = await capture(page, stem, 'desktop-pixels', true);
      return { qualified, interval, image };
    });
    await step('neural-seek-loop', () => seekAndLoop(page, 'neural'));
    await step('source-picker-round-trip', async () => {
      const documents = pageRecord.documents.length;
      const larger = await replaceSource(page, clip1080, 1920, 1080);
      const returned = await replaceSource(page, clip720, 1280, 720);
      assert.deepEqual(returned.after.memory, knownMemory, 'Returned 720p owned memory differs from initial known neural memory');
      assert.equal(pageRecord.documents.length, documents, 'File replacement reloaded the app realm');
      return { larger, returned, expected720Memory: knownMemory };
    });
    await step('pause-resume-neural', () => pauseResume(page, false));
    await step('pause-resume-backoff', () => pauseResume(page, true));
    await step('hidden-frozen-resume', () => background(page));
    await step('mobile-canvas', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      const interval = await advance(page, 30);
      intent(interval.before, interval.after, 'neural');
      const image = await capture(page, stem, 'mobile-pixels', true);
      await page.setViewportSize({ width: 1200, height: 820 });
      return { interval, image };
    });
    await step('fatal-device-loss', () => fatalDevice(page, manual.elapsedMs));

    await freshPage();
    await navigate({ mode: 'auto', withhold: 'timestamp-query' });
    await step('timestamps-unavailable', async () => {
      await page.waitForFunction(() => window.aethervsrRuntime.snapshot().controller.state === 'unavailable', null, { timeout });
      const interval = await advance(page, 121, 2200);
      baselineOnly(interval.before, interval.after);
      assert.equal(interval.after.runtime.controller.mode, 'auto');
      assert.equal(interval.after.runtime.controller.state, 'unavailable');
      assert.equal(interval.after.stats.gpuPassMs, null, 'Timestamp feature was not actually withheld');
      assert.equal(interval.after.runtime.session.gpu.neural.count, 0);
      assert.equal(interval.after.runtime.session.gpu.baseline.count, 0);
      check(/timestamp/i.test(interval.after.status), 'Unavailable status must explain timestamps');
      return interval;
    });

    await freshPage();
    await step('delayed-model-manual-race', async () => {
      let release;
      let signalHeld;
      let signalDelivered;
      const held = new Promise(resolve => { signalHeld = resolve; });
      const barrier = new Promise(resolve => { release = resolve; });
      const delivered = new Promise(resolve => { signalDelivered = resolve; });
      let cancelled = false;
      let served;
      const requested = page.waitForRequest(new URL(modelPath, base).href, { timeout })
        .then(() => null, error => error.message);
      await page.route(new URL(modelPath, base).href, async route => {
        signalHeld();
        await barrier;
        try {
          if (cancelled) { await route.abort(); return; }
          const response = await route.fetch({ timeout });
          check(response.ok(), `Held model HTTP ${response.status()}`);
          const body = await response.body();
          served = { url: response.url(), sha256: createHash('sha256').update(body).digest('hex') };
          assert.equal(served.sha256, productionSha256, 'Delayed production model mismatch');
          await route.fulfill({ response, body });
          signalDelivered({ served });
        } catch (error) {
          signalDelivered({ error: error.message });
          await route.abort().catch(() => {});
        }
      });
      try {
        await navigate({ mode: 'auto' }, false);
        const requestError = await requested;
        check(requestError === null, `Production model request was not observed: ${requestError}`);
        await held;
        await page.selectOption('#upscaler', 'baseline');
        const before = await snapshot(page);
        check(before.runtime.controller.mode === 'baseline' && before.runtime.actualTier === 'baseline',
          'Manual baseline must be selected before releasing the model response');
        release();
        const response = await delivered;
        check(!response.error, `Delayed model delivery failed: ${response.error}`);
        pageRecord.model = served;
        const interval = await advance(page, 180, 3000);
        baselineOnly(before, interval.after);
        intent(before, interval.after, 'baseline');
        const manualResult = { before, interval, served,
          barrier: 'Model request observed and explicitly held until real UI baseline selection; no timeout releases the model' };
        await page.selectOption('#upscaler', 'neural');
        const qualified = await stable(page);
        return { manualResult, qualified,
          positiveControl: 'Explicit neural selection then qualifies, proving the released model was usable' };
      } finally {
        cancelled = true;
        release();
        await page.unrouteAll({ behavior: 'wait' });
      }
    });
  } catch (error) {
    fail(error);
  } finally {
    try {
      report.sourceAfter = fingerprint();
      if (report.sourceBefore) assert.deepEqual(report.sourceAfter, report.sourceBefore,
        'HEAD, working source, production model, or media changed during lifecycle testing');
      report.sourceUnchanged = Boolean(report.sourceBefore);
    } catch (error) { report.sourceUnchanged = false; fail(error); }
    if (report.errors.length && page && !page.isClosed()) {
      try {
        report.failureEvidence = await capture(page, stem, 'failure');
        report.screenshots.push(report.failureEvidence.screenFile);
      } catch (error) {
        report.failureCaptureError = error.message;
        try { report.failureSnapshot = await snapshot(page); } catch (error) { report.failureSnapshotError = error.message; }
      }
    }
    try { await browser?.close(); } catch (error) { fail(error); }
    report.finishedAt = new Date().toISOString();
    report.counts = { total: report.tests.length,
      passed: report.tests.filter(test => test.status === 'passed').length,
      failed: report.tests.filter(test => test.status === 'failed').length,
      notRun: report.tests.filter(test => test.status === 'not-run').length };
    report.valid = report.errors.length === 0 && report.counts.passed === report.counts.total && report.sourceUnchanged;
    try { writeFileSync(descriptor, `${JSON.stringify(report, null, 2)}\n`); } finally { closeSync(descriptor); }
    console.log(JSON.stringify({ output, valid: report.valid, counts: report.counts, errors: report.errors.map(error => error.message) }));
    if (!report.valid) process.exitCode = 1;
  }
}

await run().catch(error => {
  console.error(error.stack ?? error);
  process.exitCode = 1;
});