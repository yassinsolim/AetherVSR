import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { compareStage } from './report.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const BASELINE = 'e381bd9366803fe91c048a97bbe5088844627f10';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }).trim();
const read = path => JSON.parse(readFileSync(path));
const write = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
export const reference = path => { const bytes = readFileSync(resolve(ROOT, path)); return { path, bytes: bytes.length, sha256: hash(bytes) }; };
const verify = ref => assert.deepEqual(reference(ref.path), ref);
export function verifySavedSnapshot(entry, path) {
  assert.deepEqual(entry.reference, reference(relative(ROOT, resolve(path))));
  const value = read(path); assert.deepEqual(entry.result, value); return value;
}
const clean = () => assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '', 'Clean committed source required');
const clip = fps => `public/media/aethervsr-testclip-720p${fps}-h264.mp4`;
const files = () => git(['ls-files', 'native', 'tools/m13/playback.mjs', 'tools/m13/report.mjs', 'test/native-playback.test.ts', 'docs/M13-PHASE2-PLAN.md', 'public/models/aethersr-c16d2.json', 'public/models/golden-c16d2.json']).split('\n').map(reference);
const sourcesUnchanged = manifest => {
  clean(); assert.deepEqual(files(), manifest.sources); verify(manifest.binary); manifest.resources.forEach(verify); manifest.clips.forEach(verify);
  assert.equal(git(['diff', '--name-only', manifest.sourceCommit, 'HEAD', '--', ...manifest.sources.map(ref => ref.path)]), '');
};

export function percentiles(values) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return { count: 0, p50: null, p95: null, max: null, min: null };
  const quantile = fraction => { const position = (sorted.length - 1) * fraction, lower = Math.floor(position), upper = Math.ceil(position); return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower); };
  return { count: sorted.length, p50: quantile(0.5), p95: quantile(0.95), min: sorted[0], max: sorted.at(-1) };
}

export function zeroResources(value) {
  return resourceSchema(value) && Object.entries(value).every(([key, count]) => key === 'retainedOwners' ? Object.values(count).every(value => value === 0) : count === 0);
}

const RESOURCE_KEYS = ['activeOutputs', 'configuredSlots', 'displayLinks', 'occupiedSlots', 'pixelBuffers', 'presentationCommands', 'processingSlots', 'textureCaches', 'textureWrappers'];
const OWNER_KEYS = ['decodedSampleOwners', 'leasedPixelBuffers', 'liveTextureWrappers'];
export function resourceSchema(value) {
  return value && !Array.isArray(value) && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...RESOURCE_KEYS, 'retainedOwners'].sort()) &&
    RESOURCE_KEYS.every(key => Number.isInteger(value[key]) && value[key] >= 0) && value.retainedOwners && !Array.isArray(value.retainedOwners) &&
    JSON.stringify(Object.keys(value.retainedOwners).sort()) === JSON.stringify([...OWNER_KEYS].sort()) && OWNER_KEYS.every(key => Number.isInteger(value.retainedOwners[key]) && value.retainedOwners[key] >= 0);
}

function compareRGBA(actual, expected, width, height, precision) {
  const pixels = width * height;
  assert.equal(actual.length, pixels * 4);
  const planar = new Float32Array(pixels * 3);
  for (let pixel = 0; pixel < pixels; pixel++) {
    assert.equal(actual[pixel * 4 + 3], 255);
    for (let channel = 0; channel < 3; channel++) planar[channel * pixels + pixel] = actual[pixel * 4 + channel] / 255;
  }
  return compareStage('final', planar, expected, width, height, 3, Math.max(precision === 'f32' ? 0.001 : 0.05, 1.5 / 255));
}

export function parityCheck(path, target) {
  const value = read(path); assert.equal(value.schema, 'aethervsr.m13.phase2-paused-parity/1'); assert.equal(value.outcome, 'PASS');
  assert.equal(value.modelSHA, reference('public/models/aethersr-c16d2.json').sha256);
  assert(Number.isFinite(value.pts) && Math.abs(value.pts - target) <= 2 / 30);
  assert.deepEqual(value.runs.map(run => run.precision), ['f32', 'f16']);
  assert.equal(value.color.width, 1280); assert.equal(value.color.height, 720);
  assert([875704438, 875704422].includes(value.color.format));
  assert.equal(value.color.matrix, 'ITU_R_709_2'); assert.equal(value.color.primaries, 'ITU_R_709_2'); assert.equal(value.color.transfer, 'ITU_R_709_2');
  assert(['Left', 'Center'].includes(value.color.chroma));
  assert.deepEqual(value.planes.map(plane => plane.file), ['plane-0.bin', 'plane-1.bin']);
  const load = artifact => {
    assert(!artifact.file.includes('/') && !artifact.file.includes('..'));
    const bytes = readFileSync(join(dirname(path), artifact.file)); assert.equal(bytes.length, artifact.bytes); assert.equal(hash(bytes), artifact.sha256); return bytes;
  };
  const planes = value.planes.map(load), width = value.color.width, height = value.color.height, pixels = width * height;
  for (let plane = 0; plane < 2; plane++) {
    assert.equal(value.planes[plane].height, height / (plane + 1)); assert(value.planes[plane].stride >= width);
    assert.equal(value.planes[plane].bytes, value.planes[plane].stride * value.planes[plane].height);
  }
  const expectedInput = new Float64Array(pixels * 3), video = value.color.format === 875704438, centered = value.color.chroma === 'Center';
  for (let row = 0; row < height; row++) { for (let column = 0; column < width; column++) {
    const horizontal = column / 2 - (centered ? 0.25 : 0), vertical = row / 2 - 0.25, left = Math.floor(horizontal), top = Math.floor(vertical);
    const alpha = horizontal - left, beta = vertical - top;
    const chroma = channel => {
      const code = (column, row) => planes[1][Math.min(height / 2 - 1, Math.max(0, row)) * value.planes[1].stride + Math.min(width / 2 - 1, Math.max(0, column)) * 2 + channel];
      return (code(left, top) * (1 - alpha) + code(left + 1, top) * alpha) * (1 - beta) + (code(left, top + 1) * (1 - alpha) + code(left + 1, top + 1) * alpha) * beta;
    };
    const luminance = (planes[0][row * value.planes[0].stride + column] - (video ? 16 : 0)) / (video ? 219 : 255);
    const cb = (chroma(0) - 128) / (video ? 224 : 255), cr = (chroma(1) - 128) / (video ? 224 : 255);
    const red = luminance + 2 * (1 - 0.2126) * cr, blue = luminance + 2 * (1 - 0.0722) * cb, green = (luminance - 0.2126 * red - 0.0722 * blue) / 0.7152;
    for (const [channel, code] of [red, green, blue].entries()) expectedInput[channel * pixels + row * width + column] = Math.min(1, Math.max(0, code));
  } }
  for (const run of value.runs) {
    const precision = run.precision, names = ['stem', 'body.0', 'body.1', 'final'];
    assert.equal(run.invalidOutputFlags, 0);
    assert.deepEqual(run.shape, [width, height]); assert.equal(run.stageLayout, 'CHW'); assert.equal(run.inputLayout, 'HWC4'); assert.equal(run.referenceInputLayout, 'CHW');
    const files = [`${precision}-input.f32le`, `${precision}-reference-input.f32le`, ...names.flatMap(name => [`${precision}-${name}.f32le`, `${precision}-reference-${name}.f32le`]), `${precision}-output.rgba8`, `${precision}-reference-output.rgba8`];
    assert.deepEqual(run.artifacts.map(artifact => artifact.file), files);
    const raw = new Map(run.artifacts.map(artifact => [artifact.file, load(artifact)]));
    const float = name => { const bytes = raw.get(name); assert.equal(bytes.length % 4, 0); return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4); };
    const actualInput = float(files[0]), referenceInput = float(files[1]); assert.equal(actualInput.length, pixels * 4); assert.equal(referenceInput.length, pixels * 3);
    let maximum = 0, sum = 0;
    for (let pixel = 0; pixel < pixels; pixel++) {
      assert.equal(actualInput[pixel * 4 + 3], 0);
      for (let channel = 0; channel < 3; channel++) {
        const expected = expectedInput[channel * pixels + pixel], actual = actualInput[pixel * 4 + channel];
        assert(Number.isFinite(actual)); assert.equal(referenceInput[channel * pixels + pixel], Math.fround(expected));
        const error = Math.abs(actual - expected); assert(error <= 0.00001); maximum = Math.max(maximum, error); sum += error;
      }
    }
    assert.equal(run.inputFailures, 0); assert(Math.abs(run.inputMaximumError - maximum) < 1e-12); assert(Math.abs(run.inputMeanError - sum / (pixels * 3)) < 1e-12);
    for (const [index, name] of names.entries()) {
      const final = name === 'final'; const checked = compareStage(name, float(`${precision}-${name}.f32le`), float(`${precision}-reference-${name}.f32le`), width * (final ? 2 : 1), height * (final ? 2 : 1), final ? 3 : 16, precision === 'f16' ? 0.05 : 0.001);
      assert.deepEqual(checked, run.stages[index]); assert(checked.passed);
    }
    const actualRGBA = raw.get(`${precision}-output.rgba8`), expectedRGBA = raw.get(`${precision}-reference-output.rgba8`);
    assert.equal(hash(actualRGBA), run.integratedRGBAHash); assert.equal(hash(expectedRGBA), run.referenceRGBAHash);
    assert.deepEqual(compareRGBA(actualRGBA, float(`${precision}-reference-final.f32le`), width * 2, height * 2, precision), run.rgbaVsFloat);
    const normalized = new Float32Array(pixels * 12);
    for (let pixel = 0; pixel < pixels * 4; pixel++) {
      assert.equal(expectedRGBA[pixel * 4 + 3], 255);
      for (let channel = 0; channel < 3; channel++) normalized[channel * pixels * 4 + pixel] = expectedRGBA[pixel * 4 + channel] / 255;
    }
    assert.deepEqual(compareRGBA(actualRGBA, normalized, width * 2, height * 2, precision), run.rgbaVsRGBA);
    assert(run.rgbaVsFloat.passed && run.rgbaVsRGBA.passed && run.outcome === 'PASS');
  }
  return value;
}

export function analyze(rows, result, fps, seconds, mode) {
  const failures = [];
  const require = (condition, reason) => { if (!condition) failures.push(reason); };
  require(result.outcome === 'PASS' && result.measurementFinished === true && result.error === null && result.diagnosticFailure === null, 'runtime did not complete cleanly');
  require(zeroResources(result.cleanup), 'owned resource leak after disposal');
  const starts = rows.filter(row => row.kind === 'measurement-start'), ends = rows.filter(row => row.kind === 'measurement-end');
  if (starts.length !== 1 || ends.length !== 1) return { outcome: 'FAIL', failures: [...failures, 'missing/duplicate measurement boundaries'] };
  const start = starts[0].host, end = ends[0].host, duration = end - start;
  require(Number.isFinite(duration) && duration >= seconds && duration <= seconds + 0.5, 'observation duration invalid');
  const frames = rows.filter(row => row.kind === 'frame' && row.opportunityHost >= start && row.opportunityHost < end);
  const allFrames = rows.filter(row => row.kind === 'frame'), warmup = allFrames.filter(row => row.opportunityHost < start);
  require(warmup.length >= (fps === 60 ? 58 : 29) * 4 && start - allFrames[0]?.opportunityHost >= 5 && start - allFrames[0]?.opportunityHost < 5.5, 'invalid actual-frame warmup');
  require(['baseline', 'neural'].includes(mode) && allFrames.every(row => row.neural === (mode === 'neural') && row.sourceFile === clip(fps).split('/').at(-1)), 'mode/source mismatch');
  require(rows.filter(row => row.host >= start && row.host < end && row.kind === 'invalidate').every(row => ['item-transition', 'sample-discontinuity'].includes(row.reason)), 'unregistered controls during measurement');
  const final = frames.filter(row => row.opportunityHost >= end - 20), first = frames.filter(row => row.opportunityHost < start + 20);
  const floor = fps === 60 ? 58 : 29, interval = 1000 / fps;
  const cadence = frames.length / duration, finalCadence = final.length / 20;
  require(cadence >= floor, 'overall useful cadence below floor'); require(finalCadence >= floor, 'final20s useful cadence below floor');
  require(new Set(frames.map(row => `${row.frameGeneration}:${row.sequence}`)).size === frames.length, 'repeated rendered identity');
  const generations = new Map(), sequence = new Set();
  for (const row of frames) {
    require(Number.isInteger(row.frameGeneration) && Number.isInteger(row.sequence) && Number.isFinite(row.pts) && row.pts >= 0 && Number.isFinite(row.playerTime) && typeof row.itemIdentity === 'string', 'invalid source timestamps/identity');
    const previous = generations.get(row.frameGeneration);
    require(!previous || previous.itemIdentity === row.itemIdentity && row.pts > previous.pts, 'duplicate/backward PTS or mixed item');
    require(!sequence.has(row.sequence), 'repeated sequence'); sequence.add(row.sequence); generations.set(row.frameGeneration, row);
    require(rows.some(event => event.kind === 'invalidate' && event.generation === row.frameGeneration && event.host <= row.opportunityHost), 'unattributed frame generation');
    require(Number.isFinite(row.ageMS) && Math.abs(row.ageMS - (row.playerTime - row.pts) * 1000) < 1e-7, 'fabricated software age');
  }
  const ages = percentiles(frames.map(row => row.ageMS)), firstAge = percentiles(first.map(row => row.ageMS)), finalAge = percentiles(final.map(row => row.ageMS));
  require(ages.count >= frames.length * 0.99 && frames.length > 0, 'missing age observations');
  require(ages.p95 !== null && ages.p95 <= interval * 2 && ages.max <= 250 && ages.min >= -interval * 2, 'software age bound');
  require(firstAge.p50 !== null && finalAge.p50 !== null && finalAge.p50 - firstAge.p50 <= interval, 'software age drift');
  const windows = Array.from({ length: Math.floor(duration / 5) }, (_, index) => percentiles(frames.filter(row => row.opportunityHost >= start + index * 5 && row.opportunityHost < start + (index + 1) * 5).map(row => row.ageMS)).p50);
  for (let index = 3; index < windows.length; index++) require(![index, index - 1, index - 2].every(position => windows[position] !== null && windows[position - 1] !== null && windows[position] - windows[position - 1] > interval), 'three-window age growth');
  require(frames.every(row => row.itemMatched && row.frameGeneration === row.generation && row.rate === 1 && row.timeControl === 2), 'media authority/generation mismatch');
  require(frames.every(row => row.invalidOutputFlags === 0 && row.outputValidation === (mode === 'neural' ? 'hidden-final-gpu' : 'format-validated-unorm-baseline')), 'invalid or unverified output');
  require(frames.every(row => row.completionHost >= row.submissionHost && row.opportunityHost >= row.completionHost && row.presentationCompleteHost >= row.opportunityHost), 'host ordering');
  require(rows.filter(row => row.kind === 'error').length === 0, 'terminal error');
  const heartbeats = rows.filter(row => row.kind === 'heartbeat' && row.host >= start && row.host < end);
  require(heartbeats.length >= seconds - 3, 'missing independent display heartbeats');
  const resources = [...heartbeats, ...frames].map(row => row.resources);
  require(resources.every(value => resourceSchema(value) && value.occupiedSlots <= 2 && value.pixelBuffers <= 2 && value.textureWrappers <= 4 &&
    value.processingSlots <= 2 && value.presentationCommands <= 2 && value.activeOutputs === 1 && value.displayLinks === 1 &&
    value.configuredSlots === 2 && value.textureCaches === 1 && Object.values(value.retainedOwners).every(count => count <= 4)), 'unbounded owned resources');
  const gpu = Object.fromEntries(['gpuIngestMS', 'gpuNeuralMS', 'gpuPresentationMS', 'gpuProcessingSpanMS', 'gpuFramePathMS'].map(key => [key, percentiles(frames.map(row => row[key]))]));
  require(Object.values(gpu).every(value => value.count === frames.length && value.min > 0), 'invalid/missing GPU timing');
  require(frames.every(row => Math.abs(row.gpuFramePathMS - row.gpuIngestMS - row.gpuNeuralMS - row.gpuPresentationMS) < 1e-9), 'GPU active-work scope mismatch');
  require(frames.every(row => [['gpuIngestStart', 'gpuIngestEnd', 'gpuIngestMS'], ['gpuNetworkStart', 'gpuNetworkEnd', 'gpuNeuralMS'], ['gpuPresentationStart', 'gpuPresentationEnd', 'gpuPresentationMS']].every(([begin, finish, metric]) => Number.isFinite(row[begin]) && row[begin] > 0 && Number.isFinite(row[finish]) && row[finish] > row[begin] && Math.abs((row[finish] - row[begin]) * 1000 - row[metric]) < 1e-9) && row.gpuNetworkStart >= row.gpuIngestEnd && row.gpuPresentationStart >= row.gpuNetworkEnd && Math.abs((row.gpuNetworkEnd - row.gpuIngestStart) * 1000 - row.gpuProcessingSpanMS) < 1e-9), 'invalid GPU command endpoints');
  const headroom = gpu.gpuFramePathMS.p50 <= 10 && gpu.gpuFramePathMS.p95 <= 12;
  return { schema: 'aethervsr.m13.phase2-analysis/1', fps, mode, requestedSeconds: seconds, durationSeconds: duration,
    usefulFrames: frames.length, cadence, final20Frames: final.length, finalCadence, floor,
    age: ages, first20Age: firstAge, final20Age: finalAge, ageMedianDriftMS: finalAge.p50 - firstAge.p50, fiveSecondAgeMedians: windows,
    gpu, preferredGPUHeadroomPass: headroom, gpuScope: 'per-frame sum of actual ingest, network and presentation-render GPU durations; queue/display gaps excluded; processingSpan separately records ingest-start through network-end',
    loops: rows.filter(row => row.kind === 'loop' && row.host >= start && row.host < end).length,
    heartbeatCount: heartbeats.length, first: frames[0] ?? null, last: frames.at(-1) ?? null,
    cleanup: result.cleanup, failures: [...new Set(failures)], outcome: failures.length === 0 ? 'PASS' : 'FAIL' };
}

export function freeze(id) {
  clean(); assert.match(id, /^[a-z0-9-]+$/);
  assert.equal(git(['diff', '--name-only', BASELINE, 'HEAD', '--', 'native/macos', 'src/core', 'apps/desktop', 'public/models',
    'docs/M13-PHASE1-METAL.md', 'docs/M13-PHASE1.5-METAL-OPTIMIZATION.md', 'results/m13-phase1-metal.json', 'results/m13-phase15-metal.json']), '');
  const directory = resolve(ROOT, '.cache/m13', id); assert(!existsSync(directory)); mkdirSync(directory);
  const sourceCommit = git(['rev-parse', 'HEAD']); write(join(directory, 'attempt.json'), { sourceCommit, baseline: BASELINE });
  try {
    const scratch = resolve(ROOT, '.cache/m13/phase2-swift'), env = { ...process.env, CLANG_MODULE_CACHE_PATH: resolve(ROOT, '.cache/m13/phase2-clang'), SWIFTPM_MODULECACHE_OVERRIDE: resolve(ROOT, '.cache/m13/phase2-swift-cache'), AETHERVSR_METAL_TESTS: '1' };
    for (const [name, args] of [['tests', ['test']], ['build', ['build', '-c', 'release']]]) {
      const result = spawnSync('xcrun', ['swift', ...args, '--package-path', 'native', '--scratch-path', scratch, '--jobs', '2'], { cwd: ROOT, env, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
      writeFileSync(join(directory, name + '.log'), (result.stdout ?? '') + (result.stderr ?? ''), { flag: 'wx' }); assert.equal(result.status, 0, name);
    }
    const program = join(directory, 'program'); mkdirSync(program);
    copyFileSync(join(scratch, 'release/aether-player'), join(program, 'aether-player'));
    for (const name of readdirSync(join(scratch, 'release')).filter(name => name.endsWith('.bundle'))) cpSync(join(scratch, 'release', name), join(program, name), { recursive: true, errorOnExist: true, force: false });
    const resources = readdirSync(program, { recursive: true, withFileTypes: true }).filter(entry => entry.isFile() && entry.name !== 'aether-player').map(entry => reference(relative(ROOT, join(entry.parentPath, entry.name))));
    const manifest = { schema: 'aethervsr.m13.phase2-manifest/1', baseline: BASELINE, sourceCommit, sourceTree: git(['rev-parse', 'HEAD^{tree}']),
      sources: files(), binary: reference(relative(ROOT, join(program, 'aether-player'))), resources, clips: [30, 60].map(fps => reference(clip(fps))),
      os: execFileSync('sw_vers', { encoding: 'utf8' }), hardware: execFileSync('sysctl', ['-n', 'hw.model'], { encoding: 'utf8' }).trim(),
      swift: execFileSync('xcrun', ['swift', '--version'], { encoding: 'utf8' }), tests: reference(relative(ROOT, join(directory, 'tests.log'))) };
    sourcesUnchanged(manifest); write(join(directory, 'manifest.json'), manifest); console.log(JSON.stringify({ sourceCommit, directory: relative(ROOT, directory), outcome: 'PASS' }));
  } catch (error) { write(join(directory, 'failure.json'), { sourceCommit, error: String(error) }); throw error; }
}

export const CASES = {
  lifecycle: ['--lifecycle', 30, 0], parity30: ['--parity', 30, 0], parity60: ['--parity', 60, 0],
  baseline30: ['baseline', 30, 60], neural30: ['neural', 30, 60],
  baseline60a: ['baseline', 60, 60], neural60a: ['neural', 60, 60], neural60b: ['neural', 60, 60], baseline60b: ['baseline', 60, 60],
  baseline60c: ['baseline', 60, 60], neural60c: ['neural', 60, 60], soak: ['neural', 60, 600],
};

export function prerequisiteIDs(id) {
  const keys = Object.keys(CASES), index = keys.indexOf(id); assert(index >= 0);
  if (id === 'lifecycle') return [];
  if (id === 'parity30') return ['lifecycle'];
  if (id === 'parity60') return ['lifecycle', 'parity30'];
  return ['lifecycle', 'parity30', ...(CASES[id][1] === 60 ? ['parity60'] : []), ...keys.slice(3, index)];
}

export function checkGeometry(rows) {
  const frames = rows.filter(row => row.kind === 'frame'); assert(frames.length > 0);
  assert(frames.every(row => JSON.stringify(row.networkOutput) === '[2560,1440]' && Array.isArray(row.drawable) && row.drawable.length === 2 && row.drawable.every(value => Number.isSafeInteger(value) && value > 0)));
  const initial = frames[0].drawable, resize = rows.filter(row => row.kind === 'drawable-resize');
  assert(resize.length >= 4);
  for (const row of resize) {
    assert(Number.isFinite(row.backingScale) && row.backingScale > 0);
    assert.deepEqual(row.drawable, row.viewPoints.map(value => Math.round(value * row.backingScale)));
  }
  assert(frames.some(row => row.drawable[0] < initial[0] && row.drawable[1] < initial[1]));
  const entry = rows.find(row => row.kind === 'fullscreen-enter'), exit = rows.find(row => row.kind === 'fullscreen-exit');
  assert(entry && exit && exit.host > entry.host);
  assert(frames.some(row => row.opportunityHost >= entry.host && row.opportunityHost < exit.host && JSON.stringify(row.drawable) !== JSON.stringify(initial)));
  assert(frames.some(row => row.opportunityHost >= exit.host && JSON.stringify(row.drawable) === JSON.stringify(initial)));
}

export function evaluateCase(output, id) {
  let checked;
  try {
    const spec = CASES[id], process = read(join(output, 'process.json'));
    assert.equal(process.exitCode, 0); assert.equal(process.signal, null); assert.equal(process.error, null);
    const observation = read(join(output, 'native/result.json'));
    const rows = readFileSync(join(output, 'native/events.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
    if (spec[2]) checked = analyze(rows, observation, spec[1], spec[2], spec[0]);
    else {
    assert.equal(observation.outcome, 'PASS'); assert.equal(observation.measurementFinished, true); assert.equal(observation.diagnosticFailure, null);
    assert.equal(observation.error, id === 'lifecycle' ? 'diagnostic-software-boundary' : null);
    assert(zeroResources(observation.cleanup)); assert.equal(observation.checks.length, id === 'lifecycle' ? 17 : 3);
    assert(observation.checks.every(value => value.passed));
    assert.deepEqual(rows.filter(row => row.kind === 'check').map(({ name, passed }) => ({ name, passed })), observation.checks);
    if (id === 'lifecycle') checkGeometry(rows);
    const parity = id.startsWith('parity') ? [1, 2, 3].map(target => {
        const path = join(output, `native/frame-${target}/result.json`);
        return { reference: reference(relative(ROOT, path)), result: parityCheck(path, target) };
    }) : null;
    checked = { outcome: 'PASS', checks: observation.checks, cleanup: observation.cleanup, parity };
    }
  } catch (error) { checked = { outcome: 'FAIL', error: String(error) }; }
  for (const [key, path] of Object.entries({ launch: 'launch.json', process: 'process.json', observation: 'native/result.json', events: 'native/events.jsonl' })) {
    checked[key] = existsSync(join(output, path)) ? reference(relative(ROOT, join(output, path))) : null;
  }
  return checked;
}

export function checkCase(directory, id, manifest) {
  const spec = CASES[id], output = join(directory, id), saved = read(join(output, 'result.json'));
  assert.deepEqual(saved.launch, reference(relative(ROOT, join(output, 'launch.json'))));
  assert.deepEqual(saved.process, reference(relative(ROOT, join(output, 'process.json'))));
  const launch = read(join(output, 'launch.json')), process = read(join(output, 'process.json'));
  assert.deepEqual(process.log, reference(relative(ROOT, join(output, 'process.log'))));
  assert.equal(launch.id, id); assert.equal(launch.sourceCommit, manifest.sourceCommit);
  assert.deepEqual(launch.manifest, reference(relative(ROOT, join(directory, 'manifest.json'))));
  assert.deepEqual(launch.binary, manifest.binary); assert.deepEqual(launch.clip, manifest.clips.find(value => value.path === clip(spec[1])));
  if (saved.parity) for (const target of [1, 2, 3]) verifySavedSnapshot(saved.parity[target - 1], join(output, `native/frame-${target}/result.json`));
  assert.deepEqual(evaluateCase(output, id), saved);
  return saved;
}

export function run(study, id) {
  clean(); const directory = resolve(ROOT, study); assert(directory.startsWith(resolve(ROOT, '.cache/m13') + '/')); assert.equal(realpathSync(directory), directory);
  const manifest = read(join(directory, 'manifest.json')); sourcesUnchanged(manifest);
  const spec = CASES[id]; assert(spec, 'Unregistered playback case');
  for (const prerequisite of prerequisiteIDs(id)) assert.equal(checkCase(directory, prerequisite, manifest).outcome, 'PASS', prerequisite);
  const output = join(directory, id); assert(!existsSync(output), 'Case already consumed; no favorable rerun'); mkdirSync(output);
  const launch = { schema: 'aethervsr.m13.phase2-launch/1', id, invocationCommit: git(['rev-parse', 'HEAD']), manifest: reference(relative(ROOT, join(directory, 'manifest.json'))),
    sourceCommit: manifest.sourceCommit, binary: manifest.binary, clip: manifest.clips.find(value => value.path === clip(spec[1])), startedAt: new Date().toISOString() };
  write(join(output, 'launch.json'), launch);
  const native = join(output, 'native'); const args = spec[0].startsWith('--') ? [spec[0], clip(spec[1]), native] : ['--measure', spec[0], clip(spec[1]), native, String(spec[2])];
  const result = spawnSync(resolve(ROOT, manifest.binary.path), args, { cwd: ROOT, encoding: 'utf8', timeout: (spec[2] || 180) * 1000 + 60000, killSignal: 'SIGKILL', maxBuffer: 16 * 1024 * 1024 });
  writeFileSync(join(output, 'process.log'), (result.stdout ?? '') + (result.stderr ?? ''), { flag: 'wx' });
  write(join(output, 'process.json'), { exitCode: result.status, signal: result.signal, error: result.error?.message ?? null,
    log: reference(relative(ROOT, join(output, 'process.log'))), finishedAt: new Date().toISOString() });
  const checked = evaluateCase(output, id);
  write(join(output, 'result.json'), checked);
  sourcesUnchanged(manifest);
  console.log(JSON.stringify({ id, outcome: checked.outcome, cadence: checked.cadence ?? null, finalCadence: checked.finalCadence ?? null,
    age: checked.age ?? null, headroom: checked.preferredGPUHeadroomPass ?? null, failures: checked.failures ?? checked.error ?? [], result: reference(relative(ROOT, join(output, 'result.json'))) }));
  return checked;
}

export function studyStatus(outcomes) {
  assert.deepEqual(Object.keys(outcomes), Object.keys(CASES));
  assert(Object.values(outcomes).every(value => ['PASS', 'FAIL', 'NOT_RUN'].includes(value)));
  for (const [id, outcome] of Object.entries(outcomes)) {
    if (outcome !== 'NOT_RUN') for (const prerequisite of prerequisiteIDs(id)) assert.equal(outcomes[prerequisite], 'PASS', `${id} enabled without ${prerequisite}`);
  }
  return Object.values(outcomes).every(value => value === 'PASS') ? 'PASS' : Object.values(outcomes).some(value => value === 'FAIL') ? 'FAIL' : 'INCOMPLETE';
}

export function deriveStudy(study) {
  const directory = resolve(ROOT, study), manifestPath = join(directory, 'manifest.json'), manifest = read(manifestPath);
  assert(directory.startsWith(resolve(ROOT, '.cache/m13') + '/')); assert.equal(realpathSync(directory), directory);
  assert.equal(manifest.schema, 'aethervsr.m13.phase2-manifest/1'); assert.equal(manifest.baseline, BASELINE);
  sourcesUnchanged(manifest); verify(manifest.tests);
  const cases = Object.fromEntries(Object.entries(CASES).map(([id, [mode, fps, seconds]]) => {
    const path = join(directory, id, 'result.json');
    if (!existsSync(path)) {
      assert(!existsSync(join(directory, id)), `Unresolved consumed case: ${id}`);
      return [id, { outcome: 'NOT_RUN', mode, fps, seconds, reason: 'Registered trial not enabled or not yet executed' }];
    }
    const result = checkCase(directory, id, manifest);
    return [id, { ...result, reference: reference(relative(ROOT, path)) }];
  }));
  const evidenceOutcome = studyStatus(Object.fromEntries(Object.entries(cases).map(([id, result]) => [id, result.outcome])));
  for (const [id, result] of Object.entries(cases)) {
    if (result.outcome === 'NOT_RUN') result.blockedBy = prerequisiteIDs(id).filter(prerequisite => cases[prerequisite].outcome !== 'PASS');
  }
  return { schema: 'aethervsr.m13.phase2-study/1', baseline: BASELINE, sourceCommit: manifest.sourceCommit,
    manifest: reference(relative(ROOT, manifestPath)), binary: manifest.binary, clips: manifest.clips,
    environment: { os: manifest.os, hardware: manifest.hardware, swift: manifest.swift }, cases, evidenceOutcome,
    timingScope: 'GPU command endpoints after completion; per-frame active durations exclude display/queue gaps; neural command includes validation audits',
    ageScope: 'originating AVPlayerItem time minus exact decoded PTS at software presentation opportunity; not physical scanout latency',
    audio: 'Committed fixtures contain no audio track; AVPlayer authority/control checks do not qualify audible playback or speaker synchronization' };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === 'freeze') freeze(process.argv[3]);
  else if (process.argv[2] === 'run') { const result = run(process.argv[3], process.argv[4]); if (result.outcome !== 'PASS') process.exitCode = 1; }
  else if (['report', 'check'].includes(process.argv[2])) {
    const result = deriveStudy(process.argv[3]);
    if (process.argv[2] === 'report') write(resolve(ROOT, process.argv[4]), result);
    else assert.deepEqual(read(resolve(ROOT, process.argv[4])), result);
    console.log(JSON.stringify({ replay: 'PASS', evidenceOutcome: result.evidenceOutcome, sourceCommit: result.sourceCommit,
      cases: Object.fromEntries(Object.entries(result.cases).map(([id, value]) => [id, value.outcome])) }));
  } else throw new Error('Use playback.mjs freeze ID | run STUDY CASE | report STUDY OUTPUT | check STUDY REPORT');
}