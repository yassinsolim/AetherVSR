import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const MEASURED_COMMIT = 'f370d6ccac90779aa1028589f08904a5e59dcaf2';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const reference = path => { const bytes = readFileSync(resolve(ROOT, path)); return { path, bytes: bytes.length, sha256: hash(bytes) }; };
const read = path => JSON.parse(readFileSync(resolve(ROOT, path)));
export const verifyBlob = record => assert.deepEqual(reference(record.path), record);
const verify = record => { verifyBlob(record); return read(record.path); };
const names = ['stem', 'body.0', 'body.1', 'final'];
const operations = ['stem.conv', 'stem', 'body.0.conv', 'body.0', 'body.1.conv', 'body.1', 'nearest.features', 'head', 'nearest.input', 'residual', 'final', 'rgba'];
const floatBytes = values => { const bytes = Buffer.alloc(values.length * 4); values.forEach((value, index) => bytes.writeFloatLE(value, index * 4)); return bytes; };
const floats = bytes => { assert.equal(bytes.length % 4, 0); return Array.from({ length: bytes.length / 4 }, (_, index) => bytes.readFloatLE(index * 4)); };

export function verifySourceIdentity(parity, performance, attempt, timingAttempt) {
  for (const value of [parity, performance, attempt, timingAttempt, parity.webgpuBuild]) assert.equal(value.sourceCommit, MEASURED_COMMIT);
  assert.match(parity.sourceTree, /^[0-9a-f]{40}$/);
  assert.equal(attempt.sourceTree, parity.sourceTree);
  assert.equal(parity.webgpuBuild.sourceDirty, false);
}

export function compareStage(stage, actual, expected, width, height, channels, tolerance) {
  assert(Number.isSafeInteger(width) && width > 0 && width <= 2560);
  assert(Number.isSafeInteger(height) && height > 0 && height <= 1440);
  assert(Number.isSafeInteger(channels) && channels > 0 && channels <= 16);
  assert(Number.isFinite(tolerance) && tolerance >= 0);
  const elements = width * height * channels;
  assert.equal(actual.length, elements); assert.equal(expected.length, elements);
  let maxAbsError = 0, total = 0, failingElements = 0, nonFinite = 0, edgeFailures = 0, bitIdentical = true, worstIndex;
  for (let index = 0; index < elements; index++) {
    const difference = Math.abs(actual[index] - expected[index]);
    bitIdentical &&= Object.is(actual[index], expected[index]);
    const invalid = !Number.isFinite(actual[index]) || !Number.isFinite(expected[index]) || !Number.isFinite(difference);
    if (invalid) nonFinite++;
    else {
      total += difference;
      if (difference > maxAbsError) { maxAbsError = difference; worstIndex = index; }
    }
    if (invalid || difference > tolerance) {
      failingElements++;
      const pixel = index % (width * height), column = pixel % width, row = Math.floor(pixel / width);
      if (column < 2 || row < 2 || column >= width - 2 || row >= height - 2) edgeFailures++;
    }
  }
  return { stage, elements, tolerance, maxAbsError: nonFinite ? null : maxAbsError, meanAbsError: nonFinite ? null : total / elements,
    failingElements, nonFinite, bitIdentical, ...(worstIndex === undefined ? {} : { worstIndex }), edgeFailures,
    interiorFailures: failingElements - edgeFailures, passed: failingElements === 0 };
}

export function timingSummary(series) {
  assert.equal(series.samples.length, 60);
  assert(Number.isFinite(series.observationWindowMS) && series.observationWindowMS > 0);
  const values = series.samples.map(sample => {
    assert(Number.isFinite(sample.startSeconds) && sample.startSeconds > 0);
    assert(Number.isFinite(sample.endSeconds) && sample.endSeconds > sample.startSeconds);
    const duration = (sample.endSeconds - sample.startSeconds) * 1000;
    assert.equal(sample.milliseconds, duration);
    return duration;
  }).sort((left, right) => left - right);
  for (let index = 1; index < series.samples.length; index++) assert(series.samples[index].startSeconds >= series.samples[index - 1].endSeconds);
  assert(series.observationWindowMS >= (series.samples.at(-1).endSeconds - series.samples[0].startSeconds) * 1000);
  const percentile = fraction => {
    const position = (values.length - 1) * fraction, lower = Math.floor(position), upper = Math.ceil(position);
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
  };
  const statisticsMS = { p50: percentile(0.5), p95: percentile(0.95), max: values.at(-1) };
  assert.deepEqual(series.statisticsMS, statisticsMS);
  assert.equal(series.measuredSamples, 60); assert.equal(series.status, 'measured');
  return { name: series.name, measuredSamples: 60, observationWindowMS: series.observationWindowMS, statisticsMS };
}

export function derive() {
  const parityPath = '.cache/m13/parity-02/result.json', timingPath = '.cache/m13/timing-01/result.json';
  const parity = read(parityPath), performance = read(timingPath);
  assert.equal(parity.schema, 'aethervsr.m13.qualification/1'); assert.equal(parity.outcome, 'PASS');
  assert.equal(performance.schema, 'aethervsr.m13.performance/1'); assert.equal(performance.outcome, 'PASS');
  assert.equal(parity.sourceCommit, performance.sourceCommit);
  assert.deepEqual(performance.qualification, reference(parityPath));
  assert.deepEqual(performance.nativeBinary, parity.nativeBinary); assert.deepEqual(performance.shader, parity.shader);
  const attempt = verify(parity.attempt), timingAttempt = verify(performance.attempt);
  verifySourceIdentity(parity, performance, attempt, timingAttempt);
  verifyBlob(parity.nativeBinary); verifyBlob(parity.shader);
  const buildDirectory = `.cache/m12/m13-${attempt.id}-webgpu-app`;
  assert.match(attempt.id, /^[a-z0-9][a-z0-9-]{0,63}$/);
  assert.deepEqual(read(join(buildDirectory, 'build-provenance.json')), parity.webgpuBuild);
  for (const [file, record] of Object.entries(parity.webgpuBuild.files)) {
    const path = join(buildDirectory, file);
    assert(resolve(ROOT, path).startsWith(resolve(ROOT, buildDirectory) + '/'));
    verifyBlob({ path, ...record });
  }
  assert.equal(hash(JSON.stringify(parity.webgpuBuild.files)), parity.webgpuBuild.payloadSha256);
  assert.deepEqual(attempt.environment, timingAttempt.environment);
  assert.deepEqual(reference('public/models/aethersr-c16d2.json'), attempt.model);
  assert.deepEqual(reference('public/models/golden-c16d2.json'), attempt.golden);
  assert.equal(attempt.model.sha256, 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a');
  assert.equal(attempt.golden.sha256, '7ffe8d5c26ef02f605c04e9057dd5ef9767dc83e84695211eff39f04a1e3a759');
  const model = read(attempt.model.path), golden = read(attempt.golden.path);
  assert.equal(model.sha256, '9154d9490b02d8fdf136edd990217cfb6f0e9956ed928da7179d7781832c8e02');
  assert.equal(golden.modelSha256, model.sha256);
  const inputHash = hash(floatBytes(golden.input)), width = golden.width, height = golden.height;
  const tiled = Buffer.alloc(1280 * 720 * 3 * 4);
  for (let channel = 0; channel < 3; channel++) {
    for (let row = 0; row < 720; row++) {
      for (let column = 0; column < 1280; column++) {
        tiled.writeFloatLE(golden.input[channel * width * height + (row % height) * width + column % width],
          ((channel * 720 + row) * 1280 + column) * 4);
      }
    }
  }
  const tiledHash = hash(tiled);
  const outputPixels = width * height * 4;
  const expected = names.map(name => name === 'final'
    ? Array.from({ length: outputPixels * 3 }, (_, index) => Math.fround(golden.output[(index % outputPixels) * 3 + Math.floor(index / outputPixels)]))
    : golden.stages[name].map(Math.fround));
  const rgbaPlanar = bytes => {
    assert.equal(bytes.length, outputPixels * 4);
    assert(Array.from({ length: outputPixels }, (_, pixel) => bytes[pixel * 4 + 3]).every(alpha => alpha === 255));
    return Array.from({ length: outputPixels * 3 }, (_, index) => Math.fround(bytes[(index % outputPixels) * 4 + Math.floor(index / outputPixels)] / 255));
  };
  const rawWebGPU = verify(parity.rawWebGPU);
  assert.equal(rawWebGPU.outcome, 'PASS'); assert.deepEqual(rawWebGPU.errors, []);
  assert.deepEqual(parity.runs.map(run => run.precision), ['f32', 'f16']);
  assert.deepEqual(performance.runs.map(run => run.precision), ['f32', 'f16']);
  const runs = parity.runs.map((run, precisionIndex) => {
    const webgpu = verify(run.webgpu), native = verify(run.native), tolerance = run.precision === 'f32' ? 0.001 : 0.05;
    assert.deepEqual(webgpu, rawWebGPU.runs[precisionIndex]); assert.deepEqual(native, run.nativeSummary);
    assert.equal(native.modelIdentity, model.sha256); assert.equal(webgpu.modelIdentity, model.sha256);
    for (const value of [native, webgpu]) {
      assert.equal(value.precision, run.precision); assert.equal(value.outcome, 'PASS');
      assert.equal(value.modelBytesSha256, attempt.model.sha256); assert.equal(value.goldenBytesSha256, attempt.golden.sha256);
      assert.equal(value.inputFloat32Sha256, inputHash);
    }
    assert.equal(native.webgpu.referenceSha256, run.webgpu.sha256);
    assert.deepEqual(native.artifacts.map(value => value.file), [...names.map(name => name + '.f32le'), 'output.rgba8']);
    const tensors = native.artifacts.map(record => {
      const path = join(dirname(run.native.path), record.file);
      assert.deepEqual(reference(path), { path, bytes: record.bytes, sha256: record.sha256 });
      return readFileSync(resolve(ROOT, path));
    });
    const comparisons = names.map((name, index) => {
      const final = name === 'final', stageWidth = width * (final ? 2 : 1), stageHeight = height * (final ? 2 : 1), channels = final ? 3 : 16;
      const gpu = webgpu.stages[index];
      assert.equal(gpu.name, name); assert.equal(gpu.layout, 'CHW');
      assert.deepEqual([gpu.width, gpu.height, gpu.channels], [stageWidth, stageHeight, channels]);
      const actual = floats(tensors[index]);
      const compare = (left, right) => compareStage(name, left, right, stageWidth, stageHeight, channels, tolerance);
      const goldenResult = compare(actual, expected[index]), backend = compare(actual, gpu.values), webgpuGolden = compare(gpu.values, expected[index]);
      assert.deepEqual(goldenResult, native.stages[index]); assert.deepEqual(backend, native.webgpu.stages[index]);
      assert(goldenResult.passed && backend.passed && webgpuGolden.passed);
      return { golden: goldenResult, webgpu: webgpuGolden, backend };
    });
    const normalized = rgbaPlanar(tensors[4]), webgpuNormalized = rgbaPlanar(webgpu.rgba);
    const compareRGBA = right => compareStage('final', normalized, right, width * 2, height * 2, 3, Math.max(tolerance, 1.5 / 255));
    assert.deepEqual(compareRGBA(expected[3]), native.normalizedRGBA);
    assert.deepEqual(compareRGBA(webgpuNormalized), native.webgpu.normalizedRGBA);
    assert(native.normalizedRGBA.passed && native.webgpu.normalizedRGBA.passed);
    const agreement = compareStage('final', webgpuNormalized, webgpu.stages[3].values, width * 2, height * 2, 3, 1.5 / 255);
    assert(agreement.passed);
    assert.equal(tensors[4].equals(Buffer.from(webgpu.rgba)), native.webgpu.rgbaByteIdentical);
    const timingRun = performance.runs[precisionIndex], timing = verify(timingRun.timing), timingCorrectness = verify(timingRun.correctness);
    for (const record of timingCorrectness.artifacts) {
      verifyBlob({ path: join(dirname(timingRun.correctness.path), record.file), bytes: record.bytes, sha256: record.sha256 });
    }
    assert.equal(timing.precision, run.precision); assert.equal(timing.outcome, 'PASS');
    assert.equal(timing.device, native.device); assert.equal(native.device, 'Apple M5');
    assert.equal(native.unifiedMemory, true);
    assert.deepEqual(timing.inputExtent, { width: 1280, height: 720 }); assert.equal(timing.inputFloat32Sha256, tiledHash);
    assert.equal(timing.modelBytesSha256, attempt.model.sha256); assert.equal(timing.goldenBytesSha256, attempt.golden.sha256);
    assert.equal(timing.modelIdentity, model.sha256);
    assert.deepEqual(timingCorrectness, native); assert.equal(timing.correctnessSha256, timingRun.correctness.sha256);
    assert.equal(timing.timings.warmupIterations, 10); assert.equal(timing.timings.measuredIterations, 60);
    assert.deepEqual(timing.timings.isolatedStages.map(value => value.name), operations);
    const wholeGraph = timingSummary(timing.timings.wholeGraph), isolatedStages = timing.timings.isolatedStages.map(timingSummary);
    assert.deepEqual(timingRun.wholeGraph, wholeGraph.statisticsMS); assert.equal(timingRun.wholeWindowMS, wholeGraph.observationWindowMS);
    assert.deepEqual(timing.postTimingStages.map(value => value.stage), names);
    assert.deepEqual(timing.postTimingStages.map(value => value.elements), [14745600, 14745600, 14745600, 11059200]);
    assert(timing.postTimingStages.every(value => value.nonFinite === 0 && value.outsideRange === 0));
    assert(timing.postTimingRGBA.passed && timing.postTimingRGBA.nonFinite === 0);
    return { precision: run.precision, tolerance, device: native.device, unifiedMemory: native.unifiedMemory,
      raw: { webgpu: run.webgpu, native: run.native, tensors: native.artifacts, timing: timingRun.timing },
      comparisons, normalizedRGBA: { golden: native.normalizedRGBA, backend: native.webgpu.normalizedRGBA,
        byteIdentical: native.webgpu.rgbaByteIdentical, diagnosticHeadAgreement: agreement },
      performance: { wholeGraph, isolatedStages, inputExtent: timing.inputExtent, inputFloat32Sha256: timing.inputFloat32Sha256,
        configuredBufferBytes: timing.timings.configuredBufferBytes, thermalState: [timing.thermalStateBefore, timing.thermalStateAfter],
        startedAt: timing.startedAt, finishedAt: timing.finishedAt, postTimingStages: timing.postTimingStages, postTimingRGBA: timing.postTimingRGBA } };
  });
  const earlier = ['.cache/m13/f32-01/result.json', '.cache/m13/parity-01/result.json'].map(reference);
  assert(earlier.every(record => read(record.path).outcome === 'PASS'));
  return { schema: 'aethervsr.m13.phase1-derived/1', baseline: '1fc5953496fd029262c51dd1266a6b8c3b4923a5', sourceCommit: parity.sourceCommit,
    raw: { parity: reference(parityPath), performance: reference(timingPath), earlier },
    model: attempt.model, modelIdentity: model.sha256, golden: attempt.golden, inputFloat32Sha256: inputHash,
    nativeBinary: parity.nativeBinary, shader: parity.shader, environment: parity.environment,
    webgpu: { adapter: parity.adapter, versions: parity.browser.versions, payloadSha256: parity.webgpuBuild.payloadSha256 },
    runs, numericAndTimingOutcome: 'PASS', performanceScope: 'Unfused offline tensor graph only; no real-time player or equivalent-scope WebGPU speedup claim',
    optimizationSweep: 'not performed', phase2: 'not implemented' };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = derive();
  if (process.argv[2] === '--check') assert.deepEqual(read(process.argv[3]), result);
  else writeFileSync(resolve(ROOT, process.argv[2]), JSON.stringify(result) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ outcome: 'PASS', sourceCommit: result.sourceCommit, precisions: result.runs.map(run => run.precision), output: relative(ROOT, resolve(ROOT, process.argv[3] ?? process.argv[2])) }));
}