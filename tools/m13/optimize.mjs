import assert from 'node:assert/strict';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, realpathSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildDesktop } from '../../apps/desktop/build.mjs';
import { validateWebGPU } from './qualify.mjs';
import { derive as phase1Evidence } from './report.mjs';
import { CANDIDATES, ref, read, verify, inspectQualification, inspectTiming, selectCandidate, classify, validateInventory, matchConfiguration, hash } from './optimization-report.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const CACHE = join(ROOT, '.cache/m13');
const PLAN = 'docs/M13-PHASE1.5-OPTIMIZATION-PLAN.md';
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }).trim();
const command = (name, args) => execFileSync(name, args, { cwd: ROOT, encoding: 'utf8' }).trim();
const write = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
const clean = () => assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '', 'Commit before experiments');
const inside = path => {
  const absolute = resolve(ROOT, path);
  assert(absolute.startsWith(CACHE + '/'));
  if (existsSync(absolute)) assert.equal(realpathSync(absolute), absolute);
  return absolute;
};
const identifier = id => assert.match(id, /^[a-z0-9][a-z0-9-]{0,63}$/);
const artifact = path => ref(relative(ROOT, path));
const environment = () => ({ machine: command('sysctl', ['-n', 'hw.model']), memoryBytes: Number(command('sysctl', ['-n', 'hw.memsize'])),
  os: command('sw_vers', ['-productVersion']), osBuild: command('sw_vers', ['-buildVersion']), swift: command('xcrun', ['swift', '--version']),
  xcode: command('xcodebuild', ['-version']), sdk: command('xcrun', ['--show-sdk-version']), node: process.version,
  inference: 'offline fixed tensor; no browser/codec/import/presentation/display cadence', compiler: 'MSL3.0 fastMathEnabled=false FP_CONTRACT=OFF' });

function sourceFiles() {
  return git(['ls-files', 'native/macos', 'tools/m13/optimize.mjs', 'tools/m13/optimization-report.mjs', PLAN,
    'tools/m13/qualify.mjs', 'tools/m13/report.mjs', 'tools/build-extension.mjs',
    'src', 'apps/desktop', 'package.json', 'package-lock.json',
    'public/models/aethersr-c16d2.json', 'public/models/golden-c16d2.json']).split('\n').map(ref);
}

function resources(program) {
  return readdirSync(program, { recursive: true, withFileTypes: true }).filter(entry => entry.isFile() && entry.name !== 'aether-metal-opt')
    .map(entry => artifact(join(entry.parentPath, entry.name))).sort((left, right) => left.path.localeCompare(right.path));
}

function verifyPins(manifest) {
  clean(); assert.equal(git(['rev-parse', `${manifest.sourceCommit}^{tree}`]), manifest.sourceTree);
  execFileSync('git', ['merge-base', '--is-ancestor', manifest.sourceCommit, 'HEAD'], { cwd: ROOT });
  assert.deepEqual(sourceFiles(), manifest.sourceFiles);
  assert.equal(git(['diff', '--name-only', manifest.sourceCommit, 'HEAD', '--', ...manifest.sourceFiles.map(file => file.path)]), '');
  assert.deepEqual(resources(resolve(ROOT, dirname(manifest.binary.path))), manifest.resources);
  for (const file of [manifest.binary, ...manifest.resources, ...manifest.references, manifest.referenceProvenance]) verify(file);
  const original = read(manifest.referenceProvenance.path);
  assert.equal(original.sourceCommit, 'f370d6ccac90779aa1028589f08904a5e59dcaf2');
  assert.deepEqual(original.runs.map(run => run.webgpu), manifest.references);
}

export function boundedProcess(executable, args, { cwd = ROOT, env = process.env, deadlineMS = 120000 } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(executable, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'], detached: process.platform !== 'win32' });
    const out = [], err = [];
    let expired = false;
    const timer = setTimeout(() => {
      expired = true;
      if (child.pid) {
        try {
          if (process.platform === 'win32') child.kill('SIGKILL');
          else process.kill(-child.pid, 'SIGKILL');
        } catch (error) { if (error.code !== 'ESRCH') reject(error); }
      }
    }, deadlineMS);
    child.stdout.on('data', bytes => out.push(bytes)); child.stderr.on('data', bytes => err.push(bytes));
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.on('close', (exitCode, signal) => {
      clearTimeout(timer);
      resolvePromise({ exitCode, signal, expired, stdout: Buffer.concat(out).toString(), stderr: Buffer.concat(err).toString() });
    });
  });
}

function study(path) {
  clean(); const directory = inside(path), manifest = read(join(directory, 'manifest.json'));
  assert.equal(manifest.schema, 'aethervsr.m13.phase15-manifest/1'); assert.equal(manifest.outcome, 'PASS');
  verifyPins(manifest);
  validateInventory(manifest, sourceFiles(), resources(resolve(ROOT, dirname(manifest.binary.path))));
  for (const item of manifest.qualifications) {
    const actual = inspectQualification(item.raw.path, item.reference.path); assert.deepEqual(actual, item);
    assert.equal(item.configuration.candidate, item.candidate); assert.equal(item.configuration.precision, item.precision);
    assert.deepEqual(item.reference, manifest.references.find(value => read(value.path).precision === item.precision));
    matchConfiguration(item.configuration, item.configuration);
  }
  return { directory, manifest };
}

export function nextAttempt(prior) {
  assert(prior.every(value => value.outcome === 'NOT_STARTED' && value.began === false && value.thermalState > 0 && value.thermalState <= 3),
    'Consumed or interrupted attempts cannot be replaced');
  return prior.length + 1;
}

export function failureClassification(process, failure, progress, inspectionFailed = false) {
  assert(process.exitCode !== 0 || process.expired || process.signal !== null || inspectionFailed, 'Successful process requires successful inspection');
  const thermal = failure?.outcome === 'NOT_STARTED' && failure.began === false && process.exitCode !== 0 && !process.expired && process.signal === null && !inspectionFailed;
  if (thermal) {
    assert.equal(progress.length, 1); assert.equal(progress[0].phase, 'preflight');
    assert.equal(failure.thermalState, progress[0].thermalState); nextAttempt([failure]);
  }
  return { outcome: thermal ? 'NOT_STARTED' : 'FAIL',
    unsafe: Boolean(process.expired || process.signal || !failure || inspectionFailed || /Metal execution failed/.test(failure.error)) };
}

function expectedPins(manifest, reference) {
  return [...manifest.sourceFiles, manifest.binary, ...manifest.resources, ...manifest.references, manifest.referenceProvenance, reference];
}

export function pinObservationFailed(observation, expected) {
  if (observation.status !== '') return true;
  try { assert.deepEqual(observation.files, expected); return false; }
  catch { return true; }
}

function observePins(manifest, reference) {
  return { status: git(['status', '--porcelain', '--untracked-files=normal']), files: expectedPins(manifest, reference).map(value => {
    try { return ref(value.path); } catch { return { path: value.path, missing: true }; }
  }) };
}

function checkedNativeEvidence(output, manifest, mode, candidate, precision, referencePath) {
  const qualification = inspectQualification(relative(ROOT, join(output, 'qualification.json')), relative(ROOT, referencePath));
  const frozen = manifest.qualifications?.find(item => item.candidate === candidate && item.precision === precision);
  matchConfiguration(qualification.configuration, frozen?.configuration ?? qualification.configuration);
  assert.equal(qualification.candidate, candidate); assert.equal(qualification.precision, precision);
  const timing = mode === 'qualify' ? null : inspectTiming(relative(ROOT, join(output, 'timing.json')), mode, qualification.configuration);
  return { qualification, timing };
}

function inspectRun(directory, manifest, mode, candidate, precision, referencePath) {
  const resolution = read(join(directory, 'resolution.json'));
  const value = JSON.parse(verify(resolution.attempt));
  assert.equal(resolution.outcome, value.outcome);
  const launch = JSON.parse(verify(value.launch)), process = JSON.parse(verify(value.process)); verify(process.log);
  assert.deepEqual([launch.mode, launch.candidate, launch.precision, launch.deadlineMS], [mode, candidate, precision, 120000]);
  assert.deepEqual(launch.binary, manifest.binary); assert.deepEqual(launch.reference, artifact(referencePath));
  assert.equal(launch.sourceCommit, manifest.sourceCommit);
  const pinFailure = pinObservationFailed(JSON.parse(verify(value.pinObservation)), expectedPins(manifest, launch.reference));
  if (value.outcome !== 'PASS') {
    assert.notEqual(value.outcome, 'NOT_STARTED');
    const failure = value.nativeFailure ? JSON.parse(verify(value.nativeFailure)) : null;
    const progress = value.progress ? verify(value.progress).toString().trim().split('\n').filter(Boolean).map(line => JSON.parse(line)) : [];
    let inspectionFailed = pinFailure;
    if (process.exitCode === 0 && !process.expired && process.signal === null) {
      try { checkedNativeEvidence(join(dirname(resolve(ROOT, value.launch.path)), 'native'), manifest, mode, candidate, precision, referencePath); }
      catch { inspectionFailed = true; }
    }
    const derived = failureClassification(process, failure, progress, inspectionFailed);
    assert.equal(value.outcome, derived.outcome); assert.equal(value.unsafe, derived.unsafe);
    return value;
  }
  assert.equal(pinFailure, false);
  assert.equal(process.exitCode, 0); assert.equal(process.expired, false); assert.equal(process.signal, null);
  const checked = checkedNativeEvidence(join(dirname(resolve(ROOT, value.launch.path)), 'native'), manifest, mode, candidate, precision, referencePath);
  assert.deepEqual(checked.qualification, value.qualification); assert.deepEqual(checked.timing, value.timing);
  return value;
}

async function nativeRun(directory, manifest, mode, candidate, precision, referencePath) {
  verifyPins(manifest);
  if (existsSync(join(directory, 'resolution.json'))) return inspectRun(directory, manifest, mode, candidate, precision, referencePath);
  if (!existsSync(directory)) mkdirSync(directory);
  const priorDirectories = readdirSync(directory).filter(name => /^attempt-\d{3}$/.test(name)).sort();
  const prior = priorDirectories.map(name => {
    const value = read(join(directory, name, 'result.json'));
    const failure = JSON.parse(verify(value.nativeFailure));
    assert.deepEqual([value.outcome, value.began, value.thermalState], [failure.outcome, failure.began, failure.thermalState]);
    const progress = verify(value.progress).toString().trim().split('\n').map(line => JSON.parse(line));
    assert.equal(progress.length, 1); assert.equal(progress[0].phase, 'preflight'); assert(progress[0].thermalState > 0);
    const launch = JSON.parse(verify(value.launch)), process = JSON.parse(verify(value.process)); verify(process.log);
    assert.deepEqual([launch.mode, launch.candidate, launch.precision], [mode, candidate, precision]);
    assert.deepEqual(launch.binary, manifest.binary); assert.deepEqual(launch.reference, artifact(referencePath));
    assert.equal(process.expired, false); assert.notEqual(process.exitCode, 0);
    return value;
  });
  const ordinal = nextAttempt(prior);
  const attempt = join(directory, `attempt-${String(ordinal).padStart(3, '0')}`); mkdirSync(attempt);
  const binary = resolve(ROOT, manifest.binary.path), referencePin = artifact(referencePath);
  const envelope = { schema: 'aethervsr.m13.phase15-launch/1', sourceCommit: manifest.sourceCommit,
    invocationCommit: git(['rev-parse', 'HEAD']), binary: manifest.binary,
    mode, candidate, precision, reference: artifact(referencePath), deadlineMS: 120000, startedAt: new Date().toISOString() };
  write(join(attempt, 'launch.json'), envelope);
  const output = join(attempt, 'native');
  const result = await boundedProcess(binary, [mode, candidate, precision, output, referencePath]);
  writeFileSync(join(attempt, 'process.log'), (result.stdout ?? '') + (result.stderr ?? ''), { flag: 'wx' });
  write(join(attempt, 'process.json'), { exitCode: result.exitCode, signal: result.signal, expired: result.expired,
    finishedAt: new Date().toISOString(), log: artifact(join(attempt, 'process.log')) });
  write(join(attempt, 'pins-after.json'), observePins(manifest, referencePin));
  const common = { candidate, precision, mode, launch: artifact(join(attempt, 'launch.json')), process: artifact(join(attempt, 'process.json')),
    pinObservation: artifact(join(attempt, 'pins-after.json')) };
  let evidence;
  try {
    verifyPins(manifest); verify(referencePin);
    if (result.exitCode !== 0 || result.expired) {
    const failurePath = join(output, 'failure.json');
    const failure = existsSync(failurePath) ? read(failurePath) : null;
    const progressPath = join(output, 'progress.jsonl');
    const events = existsSync(progressPath) ? readFileSync(progressPath, 'utf8').trim().split('\n').filter(Boolean).map(line => JSON.parse(line)) : [];
    evidence = { ...common, ...failure, ...failureClassification(result, failure, events), error: failure?.error ?? 'Process interrupted; inspect retained progress',
      nativeFailure: existsSync(failurePath) ? artifact(failurePath) : null, progress: existsSync(progressPath) ? artifact(progressPath) : null };
    } else {
      evidence = { ...common, ...checkedNativeEvidence(output, manifest, mode, candidate, precision, referencePath), outcome: 'PASS' };
    }
  } catch (error) {
    evidence = { ...common, outcome: 'FAIL', unsafe: true, error: String(error), inspectionFailed: true,
      nativeFailure: existsSync(join(output, 'failure.json')) ? artifact(join(output, 'failure.json')) : null,
      progress: existsSync(join(output, 'progress.jsonl')) ? artifact(join(output, 'progress.jsonl')) : null };
  }
  write(join(attempt, 'result.json'), evidence);
  if (evidence.outcome !== 'NOT_STARTED') write(join(directory, 'resolution.json'), { outcome: evidence.outcome, attempt: artifact(join(attempt, 'result.json')) });
  return evidence;
}

export async function freeze(id) {
  identifier(id); clean(); assert.equal(process.platform, 'darwin'); assert.equal(realpathSync(CACHE), CACHE);
  const directory = join(CACHE, id); assert(!existsSync(directory)); mkdirSync(directory);
  const sourceCommit = git(['rev-parse', 'HEAD']);
  const header = { schema: 'aethervsr.m13.phase15-freeze/1', sourceCommit, sourceTree: git(['rev-parse', 'HEAD^{tree}']), startedAt: new Date().toISOString() };
  write(join(directory, 'attempt.json'), header);
  try {
    const sources = sourceFiles(), env = environment(), scratch = join(CACHE, 'phase15-swift');
    const swiftEnv = { ...process.env, CLANG_MODULE_CACHE_PATH: join(CACHE, 'phase15-clang'),
      SWIFTPM_MODULECACHE_OVERRIDE: join(CACHE, 'phase15-swift-cache'), AETHERVSR_METAL_TESTS: '1' };
    for (const [name, args] of [['physical-tests', ['test']], ['release-build', ['build', '--configuration', 'release']]]) {
      const result = spawnSync('xcrun', ['swift', ...args, '--package-path', 'native/macos', '--scratch-path', scratch, '--jobs', '2'],
        { cwd: ROOT, encoding: 'utf8', env: swiftEnv, maxBuffer: 16 * 1024 * 1024 });
      writeFileSync(join(directory, name + '.log'), (result.stdout ?? '') + (result.stderr ?? ''), { flag: 'wx' });
      assert.equal(result.status, 0, `Freeze ${name} failed`);
    }
    const program = join(directory, 'program'); mkdirSync(program);
    const binary = join(program, 'aether-metal-opt'); copyFileSync(join(scratch, 'release/aether-metal-opt'), binary);
    for (const name of readdirSync(join(scratch, 'release')).filter(name => name.endsWith('.bundle'))) {
      cpSync(join(scratch, 'release', name), join(program, name), { recursive: true, errorOnExist: true, force: false });
    }
    const resourceFiles = resources(program);
    assert.deepEqual(phase1Evidence(), read('results/m13-phase1-metal.json'));
    const original = read('.cache/m13/parity-02/result.json');
    const references = original.runs.map(run => run.webgpu); references.forEach(verify);
    const pins = { ...header, sourceFiles: sources, binary: artifact(binary), resources: resourceFiles, references,
      referenceProvenance: ref('.cache/m13/parity-02/result.json') };
    const qualifications = [];
    for (const candidate of CANDIDATES) {
      for (const precision of ['f32', 'f16']) {
        const reference = references.find(file => file.path.endsWith(`webgpu-${precision}.json`));
        const result = await nativeRun(join(directory, `golden-${candidate}-${precision}`), pins, 'qualify', candidate, precision, resolve(ROOT, reference.path));
        assert.equal(result.outcome, 'PASS', 'Freeze requires all candidate correctness gates');
        qualifications.push(result.qualification);
      }
    }
    clean(); assert.equal(git(['rev-parse', 'HEAD']), sourceCommit); sources.forEach(verify);
    const manifest = { ...pins, schema: 'aethervsr.m13.phase15-manifest/1', baseline: '5b470064aea3cfee4ca95e84696ef8a359478fc9',
      environment: env, plan: ref(PLAN), candidates: CANDIDATES,
      qualifications, physicalTests: artifact(join(directory, 'physical-tests.log')), outcome: 'PASS' };
    write(join(directory, 'manifest.json'), manifest);
    console.log(JSON.stringify({ manifest: artifact(join(directory, 'manifest.json')), sourceCommit, qualifications: qualifications.length, outcome: 'PASS' }));
  } catch (error) { write(join(directory, 'failure.json'), { ...header, error: String(error), outcome: 'FAIL' }); throw error; }
}

export async function sweep(path) {
  const { directory, manifest } = study(path);
  assert(!existsSync(join(directory, 'selection.json')), 'Selection already frozen');
  const rows = [];
  let unsafe = false;
  for (const candidate of CANDIDATES) {
    const reference = manifest.references.find(value => value.path.endsWith('webgpu-f16.json'));
    if (unsafe) { rows.push({ candidate, outcome: 'NOT_RUN', reason: 'Prior unsafe execution' }); continue; }
    const result = await nativeRun(join(directory, `explore-${candidate}`), manifest, 'explore', candidate, 'f16', resolve(ROOT, reference.path));
    if (result.outcome === 'NOT_STARTED') throw new Error('Thermal preflight deferred; no warmup occurred. Resume after ordinary nominal state.');
    unsafe = result.unsafe ?? false;
    const row = result.outcome === 'PASS' ? result.timing : { candidate, outcome: 'FAIL', reason: result.error, evidence: artifact(join(directory, `explore-${candidate}/resolution.json`)) };
    rows.push(row);
    console.log(JSON.stringify({ candidate, outcome: row.outcome, statisticsMS: row.statisticsMS ?? null, bufferBytes: row.configuration?.configuredBufferBytes ?? null }));
  }
  const selected = unsafe ? null : selectCandidate(rows);
  const selection = { schema: 'aethervsr.m13.phase15-selection/1', sourceCommit: manifest.sourceCommit,
    manifest: artifact(join(directory, 'manifest.json')), selected, unsafe, rule: 'p95,p50,bufferBytes,matrixOrder', rows, selectedAt: new Date().toISOString(),
    ...(selected ? {} : { verdict: 'METAL REALTIME BACKEND NOT QUALIFIED' }) };
  write(join(directory, 'selection.json'), selection);
  console.log(JSON.stringify({ selected, selection: artifact(join(directory, 'selection.json')) }));
}

function checkedSelection(directory, manifest) {
  const selection = read(join(directory, 'selection.json'));
  assert.deepEqual(selection.manifest, artifact(join(directory, 'manifest.json'))); assert.equal(selection.sourceCommit, manifest.sourceCommit);
  const reference = manifest.references.find(value => value.path.endsWith('webgpu-f16.json'));
  let unsafe = false;
  const rows = CANDIDATES.map(candidate => {
    if (unsafe) return { candidate, outcome: 'NOT_RUN', reason: 'Prior unsafe execution' };
    const result = inspectRun(join(directory, `explore-${candidate}`), manifest, 'explore', candidate, 'f16', resolve(ROOT, reference.path));
    unsafe = result.unsafe ?? false;
    return result.outcome === 'PASS' ? result.timing : { candidate, outcome: 'FAIL', reason: result.error, evidence: artifact(join(directory, `explore-${candidate}/resolution.json`)) };
  });
  assert.deepEqual(selection.rows, rows); assert.equal(selection.unsafe, unsafe);
  assert.equal(selection.selected, unsafe ? null : selectCandidate(rows));
  return selection;
}

export async function freshReference(path) {
  const { directory, manifest } = study(path);
  const selection = checkedSelection(directory, manifest); assert(selection.selected, 'No selected candidate');
  const output = join(directory, 'selected-reference'); assert(!existsSync(output)); mkdirSync(output);
  const sourceCommit = git(['rev-parse', 'HEAD']); let app;
  write(join(output, 'attempt.json'), { sourceCommit, sourceTree: git(['rev-parse', 'HEAD^{tree}']), manifest: artifact(join(directory, 'manifest.json')),
    selection: artifact(join(directory, 'selection.json')) });
  try {
    const buildDirectory = `.cache/m12/${basename(directory)}-selected-webgpu`;
    assert(!existsSync(buildDirectory));
    const built = await buildDesktop({ diagnostic: true, stageGolden: true, outdir: buildDirectory, renderer: 'src/bench/metal/stage-golden.ts' });
    assert.equal(built.provenance.sourceDirty, false); assert.equal(built.provenance.sourceCommit, sourceCommit);
    const profile = join(output, 'profile'); mkdirSync(profile);
    const { _electron } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
    app = await _electron.launch({ executablePath: (await import('electron')).default, args: [resolve(ROOT, buildDirectory)],
      env: { ...process.env, AETHERVSR_TEST_PROFILE: profile }, timeout: 30000 });
    const versions = await app.evaluate(() => process.versions), page = await app.firstWindow({ timeout: 10000 });
    await page.waitForURL('aethervsr://app/index.html');
    const deadline = Date.now() + 30000; let raw;
    while (Date.now() < deadline) {
      raw = await page.evaluate(() => document.querySelector('#result')?.value);
      if (raw && raw !== 'pending') break;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    assert(raw && raw !== 'pending'); const capture = JSON.parse(raw); write(join(output, 'webgpu.json'), capture);
    await app.close(); app = undefined; rmSync(profile, { recursive: true }); validateWebGPU(capture);
    const references = [], qualifications = [];
    for (const value of capture.runs) {
      const referencePath = join(output, `webgpu-${value.precision}.json`); write(referencePath, value); references.push(artifact(referencePath));
      const result = await nativeRun(join(output, value.precision), manifest, 'qualify', selection.selected, value.precision, referencePath);
      assert.equal(result.outcome, 'PASS', 'Frozen winner failed parity; no runner-up substitution');
      qualifications.push(result.qualification);
    }
    const proof = { schema: 'aethervsr.m13.phase15-selected-parity/1', sourceCommit, implementationCommit: manifest.sourceCommit,
      attempt: artifact(join(output, 'attempt.json')),
      selection: artifact(join(directory, 'selection.json')), build: built.provenance, buildDirectory, versions,
      adapter: capture.adapter, raw: artifact(join(output, 'webgpu.json')), references, qualifications, outcome: 'PASS' };
    verifyPins(manifest); write(join(output, 'result.json'), proof); clean(); console.log(JSON.stringify({ selected: selection.selected, parity: artifact(join(output, 'result.json')), outcome: 'PASS' }));
  } catch (error) {
    let cleanup = 'no app active';
    if (app) { try { await app.close(); cleanup = 'closed'; } catch (closeError) { cleanup = String(closeError); } }
    write(join(output, 'failure.json'), { sourceCommit, error: String(error), cleanup, outcome: 'FAIL' }); throw error;
  }
}

function checkedParity(directory, manifest, selection) {
  const path = join(directory, 'selected-reference/result.json'), parity = read(path);
  assert.equal(parity.outcome, 'PASS'); assert.equal(parity.implementationCommit, manifest.sourceCommit);
  const attempt = JSON.parse(verify(parity.attempt));
  assert.equal(parity.sourceCommit, attempt.sourceCommit); assert.equal(attempt.sourceTree, git(['rev-parse', `${attempt.sourceCommit}^{tree}`]));
  assert.deepEqual(attempt.manifest, artifact(join(directory, 'manifest.json')));
  assert.deepEqual(attempt.selection, artifact(join(directory, 'selection.json')));
  execFileSync('git', ['merge-base', '--is-ancestor', manifest.sourceCommit, attempt.sourceCommit], { cwd: ROOT });
  assert.equal(git(['diff', '--name-only', manifest.sourceCommit, attempt.sourceCommit, '--', ...manifest.sourceFiles.map(value => value.path)]), '');
  assert.deepEqual(parity.selection, artifact(join(directory, 'selection.json')));
  assert.equal(parity.build.sourceCommit, parity.sourceCommit); assert.equal(parity.build.sourceDirty, false);
  assert.deepEqual(read(join(parity.buildDirectory, 'build-provenance.json')), parity.build);
  const sources = new Set(manifest.sourceFiles.map(file => file.path));
  assert(parity.build.inputs.every(file => sources.has(file)), 'Unpinned transitive WebGPU build input');
  for (const [file, value] of Object.entries(parity.build.files)) verify({ path: join(parity.buildDirectory, file), ...value });
  assert.equal(hash(JSON.stringify(parity.build.files)), parity.build.payloadSha256);
  const capture = JSON.parse(verify(parity.raw)); validateWebGPU(capture);
  assert.equal(parity.references.length, 2);
  assert.deepEqual(parity.qualifications.map(value => value.precision), ['f32', 'f16']);
  for (const [index, reference] of parity.references.entries()) {
    assert.deepEqual(JSON.parse(verify(reference)), capture.runs[index]);
    const precision = capture.runs[index].precision;
    const result = inspectRun(join(directory, 'selected-reference', precision), manifest, 'qualify', selection.selected, precision, resolve(ROOT, reference.path));
    assert.equal(result.outcome, 'PASS'); assert.deepEqual(result.qualification, parity.qualifications[index]);
    assert.equal(JSON.parse(verify(result.launch)).invocationCommit, attempt.sourceCommit);
  }
  return parity;
}

export async function binding(path, ordinal) {
  assert([1, 2].includes(ordinal), 'Exactly two preregistered binding ordinals');
  const { directory, manifest } = study(path), selection = checkedSelection(directory, manifest);
  assert(selection.selected, 'No selected candidate');
  const parityPath = join(directory, 'selected-reference/result.json'), parity = checkedParity(directory, manifest, selection);
  const reference = parity.references.find(value => value.path.endsWith('webgpu-f16.json')); verify(reference);
  if (ordinal === 2) {
    const previous = inspectRun(join(directory, 'binding-01'), manifest, 'binding', selection.selected, 'f16', resolve(ROOT, reference.path));
    assert(!previous.unsafe, 'Prior unsafe execution; no second binding');
  }
  const id = `binding-0${ordinal}`;
  const result = await nativeRun(join(directory, id), manifest, 'binding', selection.selected, 'f16', resolve(ROOT, reference.path));
  if (result.outcome === 'NOT_STARTED') throw new Error('Thermal preflight deferred without consuming binding execution');
  console.log(JSON.stringify({ id, outcome: result.outcome, statisticsMS: result.timing?.statisticsMS ?? null }));
  if (ordinal === 1) return result;
  const runs = [1, 2].map(index => {
    const checked = inspectRun(join(directory, `binding-0${index}`), manifest, 'binding', selection.selected, 'f16', resolve(ROOT, reference.path));
    return checked.outcome === 'PASS' ? checked.timing : checked;
  });
  const report = { schema: 'aethervsr.m13.phase15-binding/1', sourceCommit: manifest.sourceCommit, invocationCommit: git(['rev-parse', 'HEAD']),
    manifest: artifact(join(directory, 'manifest.json')), selection: artifact(join(directory, 'selection.json')), parity: artifact(parityPath),
    selected: selection.selected, runs, verdict: classify(runs, true) };
  write(join(directory, 'binding.json'), report); clean(); console.log(JSON.stringify({ verdict: report.verdict, report: artifact(join(directory, 'binding.json')) }));
}

export function report(path) {
  const { directory, manifest } = study(path), selection = checkedSelection(directory, manifest);
  const selected = selection.selected;
  let parity = null, runs = [];
  if (selected) {
    parity = checkedParity(directory, manifest, selection);
    const reference = parity.references.find(value => value.path.endsWith('webgpu-f16.json'));
    runs = [1, 2].map(index => {
      const runPath = join(directory, `binding-0${index}`);
      if (!existsSync(join(runPath, 'resolution.json'))) return { outcome: 'NOT_RUN' };
      const checked = inspectRun(runPath, manifest, 'binding', selected, 'f16', resolve(ROOT, reference.path));
      return checked.outcome === 'PASS' ? checked.timing : checked;
    });
  }
  return { schema: 'aethervsr.m13.phase15-derived/1', baseline: manifest.baseline, sourceCommit: manifest.sourceCommit,
    environment: manifest.environment, binary: manifest.binary, manifest: artifact(join(directory, 'manifest.json')),
    qualifications: manifest.qualifications, exploration: selection.rows, selected, selection: artifact(join(directory, 'selection.json')),
    parity, binding: runs, verdict: classify(runs, selected !== null), phase1: 'METAL INFERENCE QUALIFIED (unchanged)', phase2: 'not implemented' };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [mode, argument, ordinal] = process.argv.slice(2);
  if (mode === 'freeze') await freeze(argument);
  else if (mode === 'sweep') await sweep(argument);
  else if (mode === 'reference') await freshReference(argument);
  else if (mode === 'binding') await binding(argument, Number(ordinal));
  else if (mode === 'report') { const value = report(argument); write(ordinal, value); console.log(JSON.stringify({ verdict: value.verdict, output: ordinal })); }
  else if (mode === 'check') { assert.deepEqual(read(ordinal), report(argument)); console.log('PASS: complete Phase-1.5 evidence reproduced'); }
  else throw new Error('Usage: optimize.mjs freeze ID | sweep|reference|binding STUDY');
}