import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { arch, cpus, hostname, platform, totalmem, release } from 'node:os';
import { fileURLToPath } from 'node:url';
import { join, resolve } from 'node:path';
const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const MODEL = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
const MEDIA = { replay30: 'e006f3d5381d73b1ca5739f5e74216f4bf0c312f28621634d258a770822a8f9b', replay60: '5171a8b6da7303c8c409167f4191b7330f41120fcd89d7e3aa0df9483b237b29' };
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const command = (file, args) => execFileSync(process.execPath, [file, ...args], { cwd: ROOT, encoding: 'utf8', stdio: 'pipe' });
export function machineEnvironment() {
  const electron = JSON.parse(readFileSync(join(ROOT, 'node_modules/electron/package.json'))).version;
  return { hostname: hostname(), platform: platform(), release: release(), arch: arch(), cpu: cpus()[0]?.model ?? 'not measured', ramBytes: totalmem(), electron, node: process.versions.node, sourceCommit: git(['rev-parse', 'HEAD']), dirty: git(['status', '--porcelain', '--untracked-files=normal']) !== '', modelSha256: MODEL, display: 'not measured', driver: 'not measured' };
}
export function machineOutput(directory) { const output = resolve(ROOT, directory); assert(output.startsWith(resolve(ROOT, '.cache/m12.1') + '/')); return output; }
export async function validateMachine(directory = '.cache/m12.1/machine-01', { runGolden = true } = {}) {
  const output = machineOutput(directory); assert(!existsSync(output)); mkdirSync(output, { recursive: true });
  const environment = machineEnvironment(); assert.equal(environment.dirty, false); assert.equal(hash(join(ROOT, 'public/models/aethersr-c16d2.json')), MODEL);
  const media = Object.fromEntries(Object.entries(MEDIA).map(([name, expected]) => [name, { sha256: expected, status: 'pinned' }]));
  const write = (name, value) => { writeFileSync(join(output, name), JSON.stringify(value, null, 2) + '\n', { flag: 'wx' }); };
  write('hashes.json', { sourceCommit: environment.sourceCommit, modelSha256: MODEL, media });
  const stages = ['environment', 'adapter', 'stage-golden', 'parity', 'short-performance', 'lifecycle', 'raw-soak', 'neural-soak'];
  write('machine-environment.json', environment); write('adapter-capabilities.json', { status: 'NOT_RUN', reason: 'Populate from stage-golden adapter result' });
  for (const stage of stages.filter(stage => !['environment', 'adapter'].includes(stage))) write(`${stage}.json`, { stage, status: 'NOT_RUN', reason: 'Explicit staged workflow; run only after prior binding gate passes' });
  if (runGolden) {
    const goldenDirectory = `.cache/m12.1/${directory.split('/').at(-1)}-golden`;
    command('tools/m12/golden.mjs', [goldenDirectory]);
    const golden = JSON.parse(readFileSync(join(ROOT, goldenDirectory, 'result.json'), 'utf8'));
    write('stage-golden.json', golden); write('adapter-capabilities.json', { status: 'PASS', adapter: golden.result.adapter, features: golden.result.features }); stages[2] = 'stage-golden:PASS'; stages[1] = 'adapter:PASS';
  }
  const artifacts = Object.fromEntries(['machine-environment.json', 'adapter-capabilities.json', 'stage-golden.json', 'parity.json', 'short-performance.json', 'lifecycle.json', 'raw-soak.json', 'neural-soak.json', 'hashes.json'].filter(name => existsSync(join(output, name))).map(name => [name, { bytes: readFileSync(join(output, name)).length, sha256: hash(join(output, name)) }]));
  const envelope = { schema: 'aethervsr.m12.1.machine-envelope/1', environment, media, stages, artifacts, next: 'Review stage-golden result before parity; never skip the funnel' };
  write('final-machine-envelope.json', envelope); return envelope;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) { assert(process.argv.length === 3); const result = await validateMachine(process.argv[2]); console.log(JSON.stringify({ output: process.argv[2], sourceCommit: result.environment.sourceCommit, platform: result.environment.platform, next: result.next })); }
