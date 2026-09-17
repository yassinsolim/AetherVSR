import { build } from 'esbuild';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, relative, join } from 'node:path';
import { fileURLToPath, URL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { verifyProductionModel } from '../../tools/build-extension.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
export async function buildDesktop({ diagnostic = true, outdir = '.cache/m11/smoke-app', renderer = 'apps/desktop/smoke.ts' } = {}) {
  const output = resolve(ROOT, outdir);
  assert(output.startsWith(resolve(ROOT, '.cache/m11') + '/') || output === resolve(ROOT, 'dist-desktop'));
  const model = readFileSync(join(ROOT, 'public/models/aethersr-c16d2.json')); verifyProductionModel(model);
  const artifacts = new Map([['index.html', readFileSync(join(ROOT, 'apps/desktop/index.html'))],
    ['player.css', readFileSync(join(ROOT, 'apps/desktop/player.css'))], ['models/production.json', model],
    ['package.json', Buffer.from(JSON.stringify({ name: 'aethervsr-desktop', version: '0.1.0', main: 'main.cjs', private: true }))]]);
  const inputs = new Set();
  for (const [entry, name, platform, format] of [['apps/desktop/main.ts', 'main.cjs', 'node', 'cjs'], [renderer, 'renderer.js', 'browser', 'esm']]) {
    const result = await build({ absWorkingDir: ROOT, entryPoints: [entry], bundle: true, write: false, metafile: true,
      platform, format, target: platform === 'node' ? 'node22' : 'chrome144', outfile: join(output, name),
      external: platform === 'node' ? ['electron'] : [], loader: { '.wgsl': 'text' },
      define: { __DESKTOP_DIAGNOSTIC__: String(diagnostic), 'import.meta.env.DEV': 'false', 'import.meta.env.PROD': 'true' } });
    assert.equal(result.outputFiles.length, 1);
    if (platform === 'browser') assert(Object.values(result.metafile.outputs).every(value => value.imports.length === 0));
    for (const path of Object.keys(result.metafile.inputs)) inputs.add(path);
    artifacts.set(name, Buffer.from(result.outputFiles[0].contents));
  }
  const files = Object.fromEntries([...artifacts].sort(([left], [right]) => left.localeCompare(right)).map(([name, bytes]) => [name, { bytes: bytes.length, sha256: hash(bytes) }]));
  const provenance = { generator: 'aethervsr-desktop', diagnostic, sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim(),
    sourceDirty: execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim() !== '',
    electron: JSON.parse(readFileSync(join(ROOT, 'node_modules/electron/package.json'))).version, modelSha256: hash(model),
    inputs: [...inputs].sort(), files, payloadSha256: hash(JSON.stringify(files)), bytes: [...artifacts.values()].reduce((sum, value) => sum + value.length, 0) };
  artifacts.set('build-provenance.json', Buffer.from(JSON.stringify(provenance, null, 2) + '\n'));
  for (const [name, bytes] of artifacts) { const path = join(output, name); mkdirSync(resolve(path, '..'), { recursive: true }); writeFileSync(path, bytes); }
  return { directory: relative(ROOT, output), provenance };
}