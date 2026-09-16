import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync, cpSync } from 'node:fs';
import { resolve, join, relative } from 'node:path';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
import { ROOT, sha256 } from '../m10-fixtures.mjs';
import { MODEL_SHA256, MODEL_BYTES } from '../build-extension.mjs';

export const manifest = {
  manifest_version: 3, name: 'AetherVSR M10.10 Research', version: '0.0.1', minimum_chrome_version: '116',
  permissions: ['activeTab', 'scripting', 'storage'],
  action: { default_popup: 'launcher.html' }, background: { service_worker: 'service-worker.js', type: 'module' },
  content_security_policy: { extension_pages: "script-src 'self'; object-src 'self'" },
};

export async function buildPlayer(outdir = '.cache/m1010/extension', mediaFixture = {
  path: 'public/media/m9/720p60.mp4', sha256: '8d81acbe164da1d62b7d0d02a3cc66915c96e8aa90d45cac34d818fc33df1d4a',
}) {
  const directory = resolve(ROOT, outdir);
  assert(directory.startsWith(join(ROOT, '.cache/m1010/')), 'Research output must remain ignored');
  if (existsSync(directory)) { const before = JSON.parse(readFileSync(join(directory, 'research-provenance.json'))); assert.equal(before.generator, 'm1010-controlled-player'); rmSync(directory, { recursive: true }); }
  mkdirSync(directory, { recursive: true });
  for (const [name, entry, format] of [['player.js', 'player.ts', 'iife'], ['launcher.js', 'launcher.ts', 'iife'], ['service-worker.js', 'worker.ts', 'esm']]) {
    await build({ entryPoints: [join(ROOT, 'tools/m1010', entry)], bundle: true, write: true, minify: false, sourcemap: false,
      platform: 'browser', target: 'chrome116', format, outfile: join(directory, name), loader: { '.wgsl': 'text' },
      define: { 'import.meta.env.DEV': 'false', 'import.meta.env.PROD': 'true' } });
  }
  for (const name of ['player.html', 'player.css', 'launcher.html']) cpSync(join(ROOT, 'tools/m1010', name), join(directory, name));
  writeFileSync(join(directory, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  mkdirSync(join(directory, 'models')); mkdirSync(join(directory, 'media'));
  const model = readFileSync(join(ROOT, 'public/models/aethersr-c16d2.json'));
  assert.equal(model.length, MODEL_BYTES); assert.equal(sha256(model), MODEL_SHA256);
  writeFileSync(join(directory, 'models/production.json'), model);
  const media = readFileSync(resolve(ROOT, mediaFixture.path));
  assert.equal(sha256(media), mediaFixture.sha256);
  writeFileSync(join(directory, 'media/known.mp4'), media);
  const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
  const names = ['manifest.json', 'player.html', 'player.css', 'player.js', 'launcher.html', 'launcher.js', 'service-worker.js', 'models/production.json', 'media/known.mp4'];
  const files = Object.fromEntries(names.sort().map(name => { const bytes = readFileSync(join(directory, name)); return [name, { bytes: bytes.length, sha256: sha256(bytes) }]; }));
  const provenance = { generator: 'm1010-controlled-player', sourceCommit: git(['rev-parse', 'HEAD']), sourceDirty: git(['status', '--porcelain']) !== '',
    modelSha256: MODEL_SHA256, files, bundleSha256: sha256(JSON.stringify(files)), totalBytes: Object.values(files).reduce((total, file) => total + file.bytes, 0),
    production: false, permissions: manifest.permissions, mediaFixture };
  writeFileSync(join(directory, 'research-provenance.json'), `${JSON.stringify(provenance, null, 2)}\n`);
  return { directory: relative(ROOT, directory), provenance };
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) console.log(JSON.stringify(await buildPlayer(process.argv[2]), null, 2));