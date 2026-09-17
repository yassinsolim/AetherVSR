import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { buildDesktop } from '../../apps/desktop/build.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
export async function runSmoke(directory) {
  const output = resolve(ROOT, directory);
  assert(output.startsWith(resolve(ROOT, '.cache/m11') + '/'));
  assert(!existsSync(output), 'Native smoke attempt directory must be new');
  assert.equal(git(['status', '--porcelain']), '', 'Commit and review source before native smoke');
  mkdirSync(output, { recursive: true });
  const built = await buildDesktop({ diagnostic: true, outdir: '.cache/m11/smoke-app', renderer: 'apps/desktop/smoke.ts' });
  const executable = (await import('electron')).default;
  const { _electron } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
  const media = resolve(ROOT, '.cache/m1010r/media-02/replay-60.mp4');
  const result = { sourceCommit: git(['rev-parse', 'HEAD']), startedAt: new Date().toISOString(),
    host: { machine: execFileSync('sysctl', ['-n', 'hw.model'], { encoding: 'utf8' }).trim(),
      chip: execFileSync('sysctl', ['-n', 'machdep.cpu.brand_string'], { encoding: 'utf8' }).trim(),
      memoryBytes: Number(execFileSync('sysctl', ['-n', 'hw.memsize'], { encoding: 'utf8' })),
      os: execFileSync('sw_vers', [], { encoding: 'utf8' }).trim(), physicalRefresh: 'not measured' },
    package: built.provenance, electronBinarySha256: hash(readFileSync(executable)),
    media: { path: '.cache/m1010r/media-02/replay-60.mp4', sha256: hash(readFileSync(media)) },
    runs: [], errors: [], outcome: 'UNRESOLVED' };
  let application;
  try {
    mkdirSync(join(output, 'profile'));
    application = await _electron.launch({ executablePath: executable, args: [resolve(ROOT, built.directory)],
      env: { ...process.env, AETHERVSR_TEST_PROFILE: join(output, 'profile') }, timeout: 30000 });
    result.environment = await application.evaluate(({ app, BrowserWindow }) => ({ versions: process.versions,
      platform: process.platform, arch: process.arch, argv: process.argv, gpu: app.getGPUFeatureStatus(),
      preferences: BrowserWindow.getAllWindows()[0].webContents.getLastWebPreferences() }));
    const page = await application.firstWindow();
    page.on('pageerror', error => result.errors.push(String(error)));
    await page.waitForURL('aethervsr://app/index.html');
    result.security = await page.evaluate(() => ({ secure: isSecureContext, node: typeof window.require,
      process: typeof window.process, electron: typeof window.electron, gpu: typeof navigator.gpu }));
    assert.equal(result.security.node, 'undefined'); assert.equal(result.security.process, 'undefined');
    assert(result.security.secure && result.security.gpu !== 'undefined');
    for (const forceCopy of [false, true]) {
      await page.locator('input[type=file]').setInputFiles(media);
      await page.waitForFunction(() => document.querySelector('video').readyState >= 2);
      await page.locator('video').click();
      const run = await page.evaluate(forceCopy => window.m11Smoke.run(forceCopy), forceCopy);
      result.runs.push(run); assert.equal(run.outcome, 'PASS');
      assert.equal(run.input.width, 1280); assert.equal(run.input.height, 720);
      assert.equal(run.output.width, 2560); assert.equal(run.output.height, 1440);
    }
    assert.equal(result.errors.length, 0); result.outcome = 'PASS';
  } catch (error) { result.errors.push(String(error)); result.outcome = 'FAIL'; }
  finally {
    try { await application?.close(); result.closed = true; } catch (error) { result.errors.push(String(error)); result.closed = false; result.outcome = 'FAIL'; }
    result.endedAt = new Date().toISOString();
    writeFileSync(join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  }
  console.log(JSON.stringify({ directory, outcome: result.outcome, errors: result.errors, runs: result.runs.map(run => ({ forceCopy: run.forceCopy, upscaler: run.upscaler, gpuSamples: run.gpuSamples.length })) }));
  if (result.outcome !== 'PASS') process.exitCode = 1;
  return result;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await runSmoke(process.argv[2] ?? '.cache/m11/smoke-01');