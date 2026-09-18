import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, relative, resolve } from 'node:path';
const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const BUILD = '.cache/m12/stage-golden-app';
const MODEL_BYTES = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
const MODEL_FILE_SHA = '9154d9490b02d8fdf136edd990217cfb6f0e9956ed928da7179d7781832c8e02';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
const waitForObservation = async (observe, timeout = 12000) => {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { const value = await observe(); if (value) return value; await new Promise(resolve => setTimeout(resolve, 50)); }
  throw new Error('Stage golden observation deadline');
};
export const STAGE_NAMES = Object.freeze(['stem', 'body.0', 'body.1', 'head', 'output']);
export function validateGoldenEvidence(value, modelSha = MODEL_FILE_SHA) {
  assert.equal(value?.schema, 'aethervsr.m12.1.stage-golden/1');
  assert.equal(value.outcome, 'PASS'); assert.equal(value.modelSha256, modelSha);
  assert(value.adapter && value.adapter.fallbackAdapter === false);
  assert(Array.isArray(value.runs) && value.runs.length >= 1);
  for (const run of value.runs) {
    assert(['f16', 'f32'].includes(run.precision)); assert.equal(run.result.passed, true);
    assert(run.result.diagnostics.length === 0);
    assert(run.result.stages.length >= 3);
    for (const stage of run.result.stages) {
      assert(STAGE_NAMES.includes(stage.stage)); assert.equal(stage.passed, true);
      assert(Number.isFinite(stage.maxAbsError) && Number.isFinite(stage.meanAbsError));
      assert.equal(stage.nonFinite ?? 0, 0);
    }
    assert.equal(run.result.output.passed, true);
  }
  return true;
}
export async function runGolden(directory = '.cache/m12/stage-golden-01') {
  const output = resolve(ROOT, directory); assert(output.startsWith(resolve(ROOT, '.cache/m12') + '/') || output.startsWith(resolve(ROOT, '.cache/m12.1') + '/')); assert(!existsSync(output));
  assert.equal(git(['status', '--porcelain', '--untracked-files=normal']), '');
  const { buildDesktop } = await import('../../apps/desktop/build.mjs');
  const built = await buildDesktop({ diagnostic: true, stageGolden: true, outdir: BUILD, renderer: 'apps/desktop/stage-golden.ts' });
  const executable = (await import('electron')).default;
  const { _electron } = await import('../../.cache/m9/node_modules/playwright/index.mjs');
  mkdirSync(output, { recursive: true });
  const profile = join(output, 'profile'); mkdirSync(profile);
  let app;
  try {
    app = await _electron.launch({ executablePath: executable, args: [resolve(ROOT, built.directory)], env: { ...process.env, AETHERVSR_TEST_PROFILE: profile }, timeout: 30000 });
    const page = await app.firstWindow({ timeout: 10000 }); await page.waitForURL('aethervsr://app/index.html');
    const raw = await waitForObservation(() => page.evaluate(() => {
      const value = document.querySelector('#result')?.value;
      return value && value !== 'pending' ? value : null;
    }));
    const result = JSON.parse(raw); validateGoldenEvidence(result);
    const envelope = { schema: 'aethervsr.m12.1.stage-golden-envelope/1', sourceCommit: git(['rev-parse', 'HEAD']), modelSha256: result.modelSha256, modelBytesSha256: MODEL_BYTES,
      electron: JSON.parse(readFileSync(join(ROOT, 'node_modules/electron/package.json'))).version,
      media: 'not applicable: stage graph uses committed golden input', packageSha256: built.provenance.payloadSha256,
      result, resultSha256: digest(Buffer.from(JSON.stringify(result))) };
    writeFileSync(join(output, 'result.json'), JSON.stringify(envelope, null, 2) + '\n', { flag: 'wx' });
    console.log(JSON.stringify({ output: relative(ROOT, output), outcome: 'PASS', runs: result.runs.map(run => run.precision) })); return envelope;
  } finally { try { await app?.close(); } finally { rmSync(profile, { recursive: true, force: true }); } }
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await runGolden(process.argv[2] ?? '.cache/m12/stage-golden-01');
