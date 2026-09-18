import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { join, resolve } from 'node:path';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const M11 = resolve(ROOT, '.cache/m11');
const MODEL = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
const ELECTRON = '44.4.1';
const MEDIA = {
  replay30: 'e006f3d5381d73b1ca5739f5e74216f4bf0c312f28621634d258a770822a8f9b',
  replay60: '5171a8b6da7303c8c409167f4191b7330f41120fcd89d7e3aa0df9483b237b29',
};
const hash = value => createHash('sha256').update(value).digest('hex');
const readJson = path => JSON.parse(readFileSync(path, 'utf8'));
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();

export const VENDORS = Object.freeze(['apple', 'nvidia', 'amd', 'intel']);
export const M12_SOURCE = git(['rev-parse', 'HEAD']);

export function currentSource() {
  assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '');
  assert.equal(git(['rev-parse', 'HEAD']), M12_SOURCE);
  assert.equal(hash(readFileSync(join(ROOT, 'public/models/aethersr-c16d2.json'))), MODEL);
  assert.equal(JSON.parse(readFileSync(join(ROOT, 'node_modules/electron/package.json'))).version, ELECTRON);
  assert.equal(git(['diff', 'bbe127e193961909bc98d421aee47bf0047811a9', '--', 'apps', 'src', 'public/models', 'package.json', 'package-lock.json', 'tools/m11']), '');
  return { commit: M12_SOURCE, modelSha256: MODEL, electron: ELECTRON };
}

function evidence(name) {
  const path = join(M11, name);
  assert(existsSync(path), `Missing retained M11 evidence: ${name}`);
  const bytes = readFileSync(path);
  return { path: `.cache/m11/${name}`, bytes: bytes.length, sha256: hash(bytes), result: readJson(path) };
}

export function buildAppleEnvelope() {
  const source = currentSource();
  const smoke = evidence('smoke-01/result.json').result;
  const parity = evidence('parity-04/result.json').result;
  const journeys = evidence('journeys-03/result.json').result;
  const short = evidence('playback-01/result.json').result;
  const soak = evidence('soak-01/result.json').result;
  const replacement = evidence('soak-replacement-1/result.json').result;
  assert.equal(smoke.outcome, 'PASS');
  assert.equal(parity.verdict, 'PASS'); assert.equal(parity.parityPrerequisitePassed, true);
  assert.equal(journeys.verdict, 'PASS'); assert.equal(journeys.cases.length, 22);
  assert.equal(short.verdict, 'PASS'); assert(short.arms.every(row => row.verdict === 'PASS'));
  assert.equal(soak.arms.find(row => row.id === 'neural60-soak').verdict, 'PASS');
  assert.equal(replacement.arms.find(row => row.id === 'raw60-soak-replacement-1').verdict, 'PASS');
  return {
    vendor: 'apple', machineVerdict: 'UNRESOLVED', source, environment: smoke.host,
    adapter: { class: 'Apple', selected: smoke.environment?.gpu ?? null, physicalSession: true },
    security: smoke.security,
    capabilities: smoke.runs.map(run => ({ forceCopy: run.forceCopy, upscaler: run.upscaler, gpuSamples: run.gpuSamples.length })),
    goldenVectors: { modelIdentity: 'PASS', stageExecution: 'UNRESOLVED', reason: 'No committed native stage-by-stage runner existed in M11; no result fabricated.' },
    parity: { evidence: '.cache/m11/parity-04/result.json', cases: 6, verdict: 'PASS' },
    lifecycle: { evidence: '.cache/m11/journeys-03/result.json', cases: 22, verdict: 'PASS' },
    short: { evidence: '.cache/m11/playback-01/result.json', arms: 8, verdict: 'PASS' },
    long: { rawOriginal: 'FAIL_FOCUS_FALSE_RETAINED', rawReplacement: 'PASS', neural: 'PASS', evidence: ['.cache/m11/soak-01/result.json', '.cache/m11/soak-replacement-1/result.json'] },
    notes: ['M12 reuses M11 native evidence because apps/src/model/package/runtime bytes are unchanged since M11 publication.', 'Stage-by-stage native golden execution remains unresolved and is not silently counted as PASS.'],
  };
}

export function buildUnavailableEnvelope(vendor) {
  assert(VENDORS.includes(vendor) && vendor !== 'apple');
  return { vendor, machineVerdict: 'NOT_RUN_HARDWARE_UNAVAILABLE', reason: 'No qualifying physical local machine was available; software or CI adapters are not substitutes.' };
}

export function buildEnvelope() {
  const vendors = [buildAppleEnvelope(), ...VENDORS.filter(vendor => vendor !== 'apple').map(buildUnavailableEnvelope)];
  return { schema: 'aethervsr.m12.cross-vendor-envelope/1', source: currentSource(), vendors,
    overallVerdict: 'CROSS-VENDOR DESKTOP PARTIAL', bindingClasses: VENDORS,
    publicClaim: 'M11 Apple M5 desktop validation remains valid; NVIDIA, AMD and Intel cross-vendor validation is not yet claimed.',
    generatedAt: new Date().toISOString() };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert(process.argv.length === 3, 'Expected output path under results/');
  const output = resolve(ROOT, process.argv[2]); assert(output.startsWith(resolve(ROOT, 'results') + '/'));
  const envelope = buildEnvelope(); writeFileSync(output, JSON.stringify(envelope, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ output: process.argv[2], overallVerdict: envelope.overallVerdict, vendors: envelope.vendors.map(vendor => [vendor.vendor, vendor.machineVerdict]) }));
}
