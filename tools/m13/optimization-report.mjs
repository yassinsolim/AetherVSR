import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { compareStage } from './report.mjs';

export const CANDIDATES = Object.freeze(['B', 'C', 'D', 'E', 'F', 'G']);
export const hash = bytes => createHash('sha256').update(bytes).digest('hex');
export const ref = path => { const bytes = readFileSync(path); return { path, bytes: bytes.length, sha256: hash(bytes) }; };
export const read = path => JSON.parse(readFileSync(path));
export const verify = record => { assert.deepEqual(ref(record.path), record); return readFileSync(record.path); };
const names = ['stem', 'body.0', 'body.1', 'final'];
const floatValues = bytes => { assert.equal(bytes.length % 4, 0); return Array.from({ length: bytes.length / 4 }, (_, index) => bytes.readFloatLE(index * 4)); };
const floatHash = values => { const bytes = Buffer.alloc(values.length * 4); values.forEach((value, index) => bytes.writeFloatLE(value, index * 4)); return hash(bytes); };

export function validateInventory(manifest, sources, resources) {
  assert.deepEqual(manifest.candidates, CANDIDATES);
  assert.deepEqual(manifest.sourceFiles, sources);
  assert.deepEqual(manifest.resources, resources);
  assert.equal(new Set(manifest.sourceFiles.map(file => file.path)).size, sources.length);
  assert.deepEqual(manifest.qualifications.map(value => `${value.candidate}-${value.precision}`),
    CANDIDATES.flatMap(candidate => ['f32', 'f16'].map(precision => `${candidate}-${precision}`)));
  assert.equal(manifest.references.length, 2);
  assert.deepEqual(manifest.references.map(value => read(value.path).precision), ['f32', 'f16']);
}

export function matchConfiguration(actual, expected, timing = false) {
  for (const key of ['candidate', 'precision', 'device', 'modelBytesSha256', 'shaderSha256', 'weightHashes', 'stem', 'body', 'head', 'pipelineLimits']) {
    assert.deepEqual(actual[key], expected[key], `Changed candidate configuration: ${key}`);
  }
  assert.equal(actual.diagnostic, !timing);
  assert.deepEqual(actual.extent, timing ? { width: 1280, height: 720 } : expected.extent);
  const pixels = actual.extent.width * actual.extent.height, bytesPerElement = actual.precision === 'f16' ? 2 : 4;
  const fixedWeights = (16 * 25 * 4 + 2 * 4 * 9 * 16 * 4 + 4 * 9 * 3 * 4 + 51) * bytesPerElement;
  const activationBytes = pixels * 16 * bytesPerElement;
  const fused = ['F', 'G'].includes(actual.candidate);
  const bufferBytes = fixedWeights + 12 + pixels * 16 + activationBytes * 2 + (timing ? 0 : activationBytes * 3) +
    (fused ? 0 : activationBytes * 4 + pixels * 64) + (timing ? 16 : pixels * 64) + Math.ceil(actual.extent.width * 8 / 256) * 256 * actual.extent.height * 2;
  assert.equal(actual.configuredBufferBytes, bufferBytes);
  assert.equal(actual.outputTextureBytes, pixels * 16);
}

export function validateInterval(sample) {
  assert(sample && Number.isFinite(sample.startSeconds) && sample.startSeconds > 0);
  assert(Number.isFinite(sample.endSeconds) && sample.endSeconds > sample.startSeconds);
  const duration = (sample.endSeconds - sample.startSeconds) * 1000;
  assert(Number.isFinite(duration) && duration > 0); assert.equal(duration, sample.milliseconds);
  return duration;
}

export function inspectQualification(path, referencePath) {
  const result = read(path), webgpu = read(referencePath), golden = read('public/models/golden-c16d2.json');
  assert.equal(result.schema, 'aethervsr.m13.phase15-golden/1'); assert.equal(result.outcome, 'PASS');
  assert.equal(result.configuration.modelBytesSha256, ref('public/models/aethersr-c16d2.json').sha256);
  assert.equal(result.goldenBytesSha256, ref('public/models/golden-c16d2.json').sha256);
  assert.equal(result.modelIdentity, golden.modelSha256); assert.equal(result.modelIdentity, webgpu.modelIdentity);
  assert.equal(result.webgpuReferenceSha256, ref(referencePath).sha256);
  assert.equal(result.inputFloat32Sha256, floatHash(golden.input));
  assert.equal(webgpu.inputFloat32Sha256, result.inputFloat32Sha256);
  assert.equal(webgpu.modelBytesSha256, result.configuration.modelBytesSha256);
  assert.equal(webgpu.goldenBytesSha256, result.goldenBytesSha256);
  assert.equal(webgpu.outcome, 'PASS'); assert.equal(webgpu.precision, result.configuration.precision);
  assert.deepEqual(result.artifacts.map(value => value.file), [...names.map(name => name + '.f32le'), 'output.rgba8', 'fast.rgba8']);
  const raw = result.artifacts.map(value => verify({ path: join(dirname(path), value.file), bytes: value.bytes, sha256: value.sha256 }));
  const tolerance = result.configuration.precision === 'f16' ? 0.05 : 0.001;
  assert(['f16', 'f32'].includes(result.configuration.precision));
  assert.equal(result.fastRGBAByteIdentical, true); assert(raw[4].equals(raw[5]));
  const width = golden.width, height = golden.height, pixels = width * height * 4;
  const expectedFinal = Array.from({ length: pixels * 3 }, (_, index) => Math.fround(golden.output[(index % pixels) * 3 + Math.floor(index / pixels)]));
  const errors = names.map((name, index) => {
    const final = name === 'final', stageWidth = width * (final ? 2 : 1), stageHeight = height * (final ? 2 : 1), channels = final ? 3 : 16;
    const actual = floatValues(raw[index]), expected = final ? expectedFinal : golden.stages[name].map(Math.fround), reference = webgpu.stages[index];
    assert.deepEqual([reference.name, reference.width, reference.height, reference.channels, reference.layout], [name, stageWidth, stageHeight, channels, 'CHW']);
    const compare = (left, right) => compareStage(name, left, right, stageWidth, stageHeight, channels, tolerance);
    const trusted = compare(actual, expected), parity = compare(actual, reference.values);
    assert.deepEqual(trusted, result.golden[index]); assert.deepEqual(parity, result.webgpu[index]);
    assert(trusted.passed && parity.passed && compare(reference.values, expected).passed);
    return { trusted, parity };
  });
  const rgba = bytes => {
    assert.equal(bytes.length, pixels * 4);
    assert(Array.from({ length: pixels }, (_, index) => bytes[index * 4 + 3]).every(alpha => alpha === 255));
    return Array.from({ length: pixels * 3 }, (_, index) => Math.fround(bytes[(index % pixels) * 4 + Math.floor(index / pixels)] / 255));
  };
  const compareRGBA = expected => compareStage('final', rgba(raw[4]), expected, width * 2, height * 2, 3, Math.max(tolerance, 1.5 / 255));
  assert.deepEqual(compareRGBA(expectedFinal), result.normalizedGolden);
  assert.deepEqual(compareRGBA(rgba(webgpu.rgba)), result.normalizedWebGPU);
  assert(result.normalizedGolden.passed && result.normalizedWebGPU.passed);
  assert.equal(result.webgpuRGBAByteIdentical, raw[4].equals(Buffer.from(webgpu.rgba)));
  return { candidate: result.configuration.candidate, precision: result.configuration.precision, configuration: result.configuration,
    errors, normalizedGolden: result.normalizedGolden, normalizedWebGPU: result.normalizedWebGPU, fastRGBAByteIdentical: true,
    webgpuRGBAByteIdentical: result.webgpuRGBAByteIdentical, outcome: 'PASS', raw: ref(path), reference: ref(referencePath) };
}

export function inspectTiming(path, mode, expectedConfiguration) {
  const result = read(path), timing = result.timing, count = mode === 'binding' ? 60 : 20, warmups = mode === 'binding' ? 10 : 5;
  assert.equal(result.schema, 'aethervsr.m13.phase15-timing/1'); assert.equal(result.outcome, 'PASS');
  assert(['explore', 'binding'].includes(mode));
  if (expectedConfiguration) matchConfiguration(result.configuration, expectedConfiguration, true);
  assert.equal(timing.mode, mode); assert.equal(timing.requestedSamples, count); assert.equal(timing.warmups, warmups);
  assert.equal(timing.thermalStateBefore, 0); assert.equal(result.configuration.precision, 'f16'); assert.equal(result.configuration.diagnostic, false);
  assert.deepEqual(result.configuration.extent, { width: 1280, height: 720 });
  assert.equal(result.inputFloat32Sha256, 'e43b54d00e708a81b1e537e5345e72195300b5ab3dde75f2e05204d273953c19');
  assert.equal(result.qualificationSha256, ref(join(dirname(path), 'qualification.json')).sha256);
  const series = timing.wholeGraph;
  assert.equal(series.samples.length, count); assert.equal(series.measuredSamples, count); assert.equal(series.status, 'measured');
  const values = series.samples.map(validateInterval).sort((left, right) => left - right);
  const percentile = fraction => { const position = (count - 1) * fraction, lower = Math.floor(position), upper = Math.ceil(position); return values[lower] + (values[upper] - values[lower]) * (position - lower); };
  const statisticsMS = { p50: percentile(0.5), p95: percentile(0.95), max: values.at(-1) };
  assert.deepEqual(statisticsMS, series.statisticsMS);
  assert(Number.isFinite(series.observationWindowMS) && series.observationWindowMS >= (series.samples.at(-1).endSeconds - series.samples[0].startSeconds) * 1000);
  for (let index = 1; index < count; index++) assert(series.samples[index].startSeconds >= series.samples[index - 1].endSeconds);
  const progressPath = join(dirname(path), 'progress.jsonl');
  const events = readFileSync(progressPath, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.equal(events.length, 2 + warmups + count);
  assert.equal(events[0].phase, 'preflight'); assert.equal(events[0].thermalState, 0);
  assert.equal(events[1].phase, 'begin'); assert.equal(events[1].thermalState, 0);
  assert(events.every(event => Number.isInteger(event.thermalState) && event.thermalState >= 0 && event.thermalState <= 3));
  const warm = events.slice(2, warmups + 2), sampled = events.slice(warmups + 2);
  for (const [index, event] of warm.entries()) { assert.equal(event.phase, 'warmup'); assert.equal(event.index, index); validateInterval(event.interval); }
  for (const [index, event] of sampled.entries()) { assert.equal(event.phase, 'sample'); assert.equal(event.index, index); assert.deepEqual(event.interval, series.samples[index]); }
  const intervals = events.slice(2).map(event => event.interval);
  for (let index = 1; index < intervals.length; index++) assert(intervals[index].startSeconds >= intervals[index - 1].endSeconds);
  assert.equal(result.postAlphaOpaque, true); assert.equal(result.postDiagnosticRGBAEqual, true); assert.equal(result.postRGBABytes, 1280 * 720 * 16);
  assert.deepEqual(result.postStages.map(value => value.stage), names);
  assert.deepEqual(result.postStages.map(value => value.elements), [14745600, 14745600, 14745600, 11059200]);
  assert(result.postStages.every(value => value.nonFinite === 0 && value.outsideRange === 0));
  return { candidate: result.configuration.candidate, mode, configuration: result.configuration, statisticsMS, validSamples: count,
    warmups, observationWindowMS: series.observationWindowMS, thermalStateBefore: timing.thermalStateBefore, thermalStateAfter: timing.thermalStateAfter,
    thermalStates: events.map(value => value.thermalState), startedAt: result.startedAt, finishedAt: result.finishedAt,
    raw: ref(path), progress: ref(progressPath), postStages: result.postStages, postDiagnosticRGBAEqual: true, outcome: 'PASS' };
}

export function selectCandidate(rows) {
  assert.deepEqual(rows.map(row => row.candidate), CANDIDATES);
  const eligible = rows.filter(row => row.outcome === 'PASS' && row.validSamples === 20);
  if (eligible.length === 0) return null;
  for (const row of eligible) {
    assert(['p50', 'p95', 'max'].every(key => Number.isFinite(row.statisticsMS[key]) && row.statisticsMS[key] > 0));
    assert(Number.isSafeInteger(row.configuration.configuredBufferBytes) && row.configuration.configuredBufferBytes > 0);
  }
  return [...eligible].sort((left, right) => left.statisticsMS.p95 - right.statisticsMS.p95 || left.statisticsMS.p50 - right.statisticsMS.p50 ||
    left.configuration.configuredBufferBytes - right.configuration.configuredBufferBytes || CANDIDATES.indexOf(left.candidate) - CANDIDATES.indexOf(right.candidate))[0].candidate;
}

export function classify(runs, numericalPass) {
  if (!numericalPass || runs.length !== 2 || runs.some(run => run.outcome !== 'PASS' || run.validSamples !== 60 ||
      run.thermalStateBefore !== 0 || !['p50', 'p95', 'max'].every(key => Number.isFinite(run.statisticsMS?.[key]) && run.statisticsMS[key] > 0))) return 'METAL REALTIME BACKEND NOT QUALIFIED';
  if (runs.every(run => run.statisticsMS.p50 <= 8 && run.statisticsMS.p95 <= 10)) return 'METAL REALTIME BACKEND QUALIFIED';
  if (runs.every(run => run.statisticsMS.p95 < 16.67)) return 'METAL REALTIME BACKEND PARTIAL';
  return 'METAL REALTIME BACKEND NOT QUALIFIED';
}