import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const MODEL_SHA256 = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
export const MODEL_BYTES = 140467;
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');

export function verifyProductionModel(bytes) {
  if (bytes.byteLength !== MODEL_BYTES || hash(bytes) !== MODEL_SHA256) {
    throw new Error('Production model integrity mismatch: packaging refused.');
  }
}

export async function buildExtension({ contentEntry = 'src/extension/content.ts', outdir, test = false } = {}) {
  if (typeof test !== 'boolean') throw new Error('test must be a boolean.');
  const output = resolve(ROOT, outdir ?? (test ? 'dist-extension-test' : 'dist-extension'));
  if (output === resolve(ROOT) || relative(output, resolve(ROOT)).split('/')[0] !== '..') {
    throw new Error('Output must not contain the source repository.');
  }
  const lock = join(tmpdir(), `aethervsr-extension-${hash(resolve(ROOT)).slice(0, 16)}.lock`);
  try { await mkdir(lock); }
  catch { throw new Error('Another extension build is active; do not run builds concurrently.'); }
  let staging;
  try {
    const model = await readFile(join(ROOT, 'public/models/aethersr-c16d2.json'));
    verifyProductionModel(model);
    const manifestBytes = await readFile(join(ROOT, 'src/extension/manifest.json'));
    const manifest = JSON.parse(manifestBytes);
    const sourceCommit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
    const sourceDirty = execFileSync('git', ['status', '--porcelain', '--untracked-files=normal'], { cwd: ROOT, encoding: 'utf8' }).trim().length > 0;
    staging = await mkdtemp(join(tmpdir(), 'aethervsr-extension-stage-'));
    const options = {
      absWorkingDir: ROOT, bundle: true, platform: 'browser', target: 'chrome106',
      minify: true, sourcemap: false, legalComments: 'none', metafile: true, write: false,
      define: {
        'import.meta.env.DEV': 'false', 'import.meta.env.PROD': 'true',
        'import.meta.env.MODE': '"production"', 'process.env.NODE_ENV': '"production"',
        __AETHERVSR_TEST__: String(test),
      },
    };
    const artifacts = new Map([
      ['manifest.json', manifestBytes], ['models/production.json', model],
      ['popup.html', await readFile(join(ROOT, 'src/extension/popup.html'))],
      ['popup.css', await readFile(join(ROOT, 'src/extension/popup.css'))],
    ]);
    for (const [name, entryPoint, format] of [
      ['service-worker.js', 'src/extension/service-worker.ts', 'esm'],
      ['content.js', contentEntry, 'iife'], ['popup.js', 'src/extension/popup.ts', 'esm'],
    ]) {
      const result = await build({ ...options, entryPoints: [entryPoint], format, outfile: join(staging, name) });
      const inputs = Object.keys(result.metafile.inputs);
      if (inputs.some((input) => input === 'src/main.ts' || input.startsWith('src/bench/'))) {
        throw new Error('Standalone harness or benchmark code entered the extension bundle.');
      }
      if (Object.values(result.metafile.outputs).some((item) => item.imports.length > 0) || result.outputFiles.length !== 1) {
        throw new Error('Extension entry must be fully bundled with no external imports.');
      }
      artifacts.set(name, Buffer.from(result.outputFiles[0].contents));
    }
    const files = Object.fromEntries([...artifacts].sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([name, bytes]) => [name, { sha256: hash(bytes), bytes: bytes.byteLength }]));
    const provenance = {
      schemaVersion: 1, generator: 'aethervsr-m10', buildKind: test ? 'test' : 'production',
      sourceCommit, sourceDirty, manifestVersion: manifest.version,
      modelSha256: MODEL_SHA256, modelBytes: MODEL_BYTES,
      bundleSha256: hash(Object.entries(files).map(([name, info]) => `${name}\0${info.sha256}\n`).join('')),
      totalBytes: Object.values(files).reduce((sum, info) => sum + info.bytes, 0), files,
    };
    artifacts.set('build-provenance.json', Buffer.from(`${JSON.stringify(provenance, null, 2)}\n`));
    let existing = [];
    try { existing = await readdir(output); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (existing.length > 0) {
      let previous;
      try { previous = JSON.parse(await readFile(join(output, 'build-provenance.json'), 'utf8')); } catch { previous = undefined; }
      if (previous?.generator !== 'aethervsr-m10') throw new Error('Refusing to replace a non-extension output directory.');
    }
    await rm(output, { recursive: true, force: true });
    for (const [name, bytes] of artifacts) {
      const destination = join(output, name);
      await mkdir(dirname(destination), { recursive: true });
      await writeFile(destination, bytes);
    }
    return { outdir: output, provenance };
  } finally {
    if (staging) await rm(staging, { recursive: true, force: true });
    await rm(lock, { recursive: true, force: true });
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.some((arg) => arg !== '--test') || args.length > 1) throw new Error('Usage: node tools/build-extension.mjs [--test]');
  const result = await buildExtension({ test: args.includes('--test') });
  console.log(JSON.stringify(result, null, 2));
}