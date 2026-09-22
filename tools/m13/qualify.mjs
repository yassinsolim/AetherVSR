import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { basename, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildDesktop } from '../../apps/desktop/build.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const command = (name, args) => execFileSync(name, args, { cwd: ROOT, encoding: 'utf8' }).trim();
const git = args => command('git', args);
const write = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
const artifact = path => { const bytes = readFileSync(path); return { path: relative(ROOT, path), bytes: bytes.length, sha256: digest(bytes) }; };

export function recordAttempt(output, header, inspect) {
  try {
    const attempt = { ...header, ...inspect() };
    write(join(output, 'attempt.json'), attempt);
    return attempt;
  } catch (error) {
    write(join(output, 'failure.json'), { ...header, phase: 'initialization', error: String(error),
      cleanup: 'no app launched', finishedAt: new Date().toISOString(), outcome: 'FAIL' });
    throw error;
  }
}

export function validateWebGPU(value) {
  assert.equal(value?.schema, 'aethervsr.m13.webgpu-capture/1');
  assert.equal(value.outcome, 'PASS');
  assert.equal(value.adapter?.fallbackAdapter, false);
  assert.deepEqual(value.errors, []);
  assert.deepEqual(value.runs?.map(run => run.precision), ['f32', 'f16']);
  for (const run of value.runs) {
    assert.equal(run.schema, 'aethervsr.m13.webgpu-golden/1');
    assert.equal(run.outcome, 'PASS');
    assert.equal(run.summary?.passed, true);
    assert.equal(run.finalFloat?.passed, true);
    assert.equal(run.rgbaAgreement?.passed, true);
    assert.deepEqual(run.stages?.map(stage => stage.name), ['stem', 'body.0', 'body.1', 'final']);
  }
}

export async function qualify(id) {
  assert.match(id, /^[a-z0-9][a-z0-9-]{0,63}$/);
  assert.equal(process.platform, 'darwin', 'Physical macOS Metal host required');
  assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '', 'A clean committed source is required');
  const cache = join(ROOT, '.cache/m13');
  mkdirSync(cache, { recursive: true });
  assert.equal(realpathSync(cache), cache, 'Evidence cache must not be a symlink');
  const output = join(cache, id);
  assert(!existsSync(output), 'Attempt IDs are immutable');
  const buildDirectory = join(ROOT, '.cache/m12', `m13-${id}-webgpu-app`);
  assert(!existsSync(buildDirectory), 'Diagnostic build already exists');
  const sourceCommit = git(['rev-parse', 'HEAD']);
  const sourceTree = git(['rev-parse', 'HEAD^{tree}']);
  mkdirSync(output);
  const attempt = recordAttempt(output, { schema: 'aethervsr.m13.attempt/1', id, sourceCommit, sourceTree, startedAt: new Date().toISOString(),
    scope: 'offline tensor inference only; no decode, import, presentation or performance measurement' }, () => ({
    environment: { platform: process.platform, arch: process.arch, node: process.version,
      machine: command('sysctl', ['-n', 'hw.model']), memoryBytes: Number(command('sysctl', ['-n', 'hw.memsize'])),
      os: command('sw_vers', ['-productVersion']), osBuild: command('sw_vers', ['-buildVersion']),
      swift: command('xcrun', ['swift', '--version']), sdk: command('xcrun', ['--show-sdk-version']),
      xcode: command('xcodebuild', ['-version']), display: 'not applicable: no drawable or presentation',
      metalCompiler: { api: 'MTLDevice.makeLibrary(source:options:)', language: 'MSL 3.0', fastMathEnabled: false, fpContract: 'OFF' } },
    model: artifact(join(ROOT, 'public/models/aethersr-c16d2.json')), golden: artifact(join(ROOT, 'public/models/golden-c16d2.json')) }));
  let phase = 'build', app, cleanup = 'no app launched';
  const scratch = join(cache, 'swift');
  const env = { ...process.env, CLANG_MODULE_CACHE_PATH: join(cache, 'clang-cache'), SWIFTPM_MODULECACHE_OVERRIDE: join(cache, 'swift-module-cache') };
  const run = (name, args, log) => {
    const result = spawnSync(name, args, { cwd: ROOT, env, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
    writeFileSync(join(output, log), (result.stdout ?? '') + (result.stderr ?? ''), { flag: 'wx' });
    assert.equal(result.status, 0, `${name} failed: ${result.error?.message ?? result.signal ?? result.status}; see ${log}`);
  };
  try {
    run('xcrun', ['swift', 'build', '--configuration', 'release', '--package-path', 'native/macos', '--scratch-path', scratch, '--jobs', '2'], 'build.log');
    const binary = join(scratch, 'release/aether-metal');
    const resources = readdirSync(join(scratch, 'release'), { recursive: true }).filter(path => path.endsWith('/Shaders.metal') && path.includes('.bundle/'));
    assert.equal(resources.length, 1, 'Exactly one copied native shader resource is required');
    const shader = artifact(join(scratch, 'release', resources[0]));
    assert.equal(shader.sha256, artifact(join(ROOT, 'native/macos/Sources/AetherMetal/Shaders.metal')).sha256);
    const nativeBinary = artifact(binary);
    phase = 'webgpu';
    const built = await buildDesktop({ diagnostic: true, stageGolden: true, outdir: relative(ROOT, buildDirectory), renderer: 'src/bench/metal/stage-golden.ts' });
    assert.equal(built.provenance.sourceDirty, false);
    assert.equal(built.provenance.sourceCommit, sourceCommit);
    const profile = join(output, 'profile');
    mkdirSync(profile);
    const executablePath = (await import('electron')).default;
    const { _electron } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
    app = await _electron.launch({ executablePath, args: [buildDirectory], env: { ...process.env, AETHERVSR_TEST_PROFILE: profile }, timeout: 30000 });
    cleanup = 'app running';
    const browser = await app.evaluate(() => ({ versions: process.versions, arch: process.arch }));
    const page = await app.firstWindow({ timeout: 10000 });
    await page.waitForURL('aethervsr://app/index.html');
    let raw;
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      raw = await page.evaluate(() => document.querySelector('#result')?.value);
      if (raw && raw !== 'pending') break;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    assert(raw && raw !== 'pending', 'WebGPU capture deadline');
    const webgpu = JSON.parse(raw);
    write(join(output, 'webgpu.json'), webgpu);
    await app.close(); app = undefined; cleanup = 'app closed';
    rmSync(profile, { recursive: true });
    validateWebGPU(webgpu);
    const runs = [];
    for (const reference of webgpu.runs) {
      phase = `metal-${reference.precision}`;
      const referencePath = join(output, `webgpu-${reference.precision}.json`);
      write(referencePath, reference);
      const directory = join(output, reference.precision);
      run(binary, ['--model', 'public/models/aethersr-c16d2.json', '--input', 'public/models/golden-c16d2.json',
        '--precision', reference.precision, '--output', directory, '--webgpu', referencePath], `${reference.precision}.log`);
      const resultPath = join(directory, 'result.json');
      const native = JSON.parse(readFileSync(resultPath));
      assert.equal(native.outcome, 'PASS');
      assert.equal(native.precision, reference.precision);
      assert.equal(native.webgpu?.referenceSha256, artifact(referencePath).sha256);
      for (const record of native.artifacts) {
        assert.equal(basename(record.file), record.file);
        const actual = artifact(join(directory, record.file));
        assert.equal(actual.bytes, record.bytes); assert.equal(actual.sha256, record.sha256);
      }
      const { stages: tensors, rgba, ...webgpuSummary } = reference;
      assert(tensors.length === 4 && rgba.length > 0);
      runs.push({ precision: reference.precision, webgpu: artifact(referencePath), webgpuSummary, native: artifact(resultPath), nativeSummary: native });
    }
    assert.equal(git(['rev-parse', 'HEAD']), sourceCommit);
    assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '');
    assert.deepEqual(artifact(binary), nativeBinary);
    assert.deepEqual(artifact(join(ROOT, shader.path)), shader);
    const result = { schema: 'aethervsr.m13.qualification/1', attempt: artifact(join(output, 'attempt.json')), sourceCommit, sourceTree,
      environment: attempt.environment, nativeBinary, shader, browser, webgpuBuild: built.provenance,
      rawWebGPU: artifact(join(output, 'webgpu.json')), adapter: webgpu.adapter, runs, cleanup, finishedAt: new Date().toISOString(), outcome: 'PASS' };
    write(join(output, 'result.json'), result);
    console.log(JSON.stringify({ result: artifact(join(output, 'result.json')), outcome: result.outcome }));
    return result;
  } catch (error) {
    if (app) {
      try { await app.close(); cleanup = 'app closed after failure'; }
      catch (closeError) { cleanup = `app close failed: ${closeError}`; }
    }
    write(join(output, 'failure.json'), { sourceCommit, phase, error: String(error), cleanup, finishedAt: new Date().toISOString(), outcome: 'FAIL' });
    throw error;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await qualify(process.argv[2]);