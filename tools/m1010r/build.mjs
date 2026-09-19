import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { MODEL_SHA256, verifyProductionModel } from '../build-extension.mjs';

export const ROOT = resolve(fileURLToPath(new URL('../../', import.meta.url)));
export const DEFAULT_EXTENSION = '.cache/m1010r/extension';
export const DEFAULT_RI_EXTENSION = '.cache/m1010r/ri-extension';
export const DEFAULT_MEDIA = '.cache/m1010r/media-01/media.json';
export const PROVENANCE_FILE = 'research-provenance.json';
export const digest = bytes => createHash('sha256').update(bytes).digest('hex');
export const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
export const manifest = {
  manifest_version: 3, name: 'AetherVSR M10.10R Calibration', version: '0.0.1', minimum_chrome_version: '116',
  background: { service_worker: 'service-worker.js' },
  content_security_policy: { extension_pages: "script-src 'self'; object-src 'self'" },
};
const workerSource = 'chrome.runtime.onInstalled.addListener(()=>{})\n';
export const instrumentManifest = { ...manifest, name: 'AetherVSR M10.10RI Instrument' };
const json = value => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const fileInfo = bytes => ({ bytes: bytes.length, sha256: digest(bytes) });
const buildSources = ['tools/m1010r/build.mjs', 'tools/m1010r/media.mjs', 'tools/m1010/fixtures.mjs',
  'tools/build-extension.mjs', 'tools/m1010r/calibration.html', 'tools/m1010/player.css', 'package.json'];

export function verifyInstrumentInputs(inputs) {
  assert(Array.isArray(inputs) && inputs.length > 0, 'Missing instrument bundle inputs');
  for (const name of inputs) {
    assert(typeof name === 'string' && relative(ROOT, resolve(ROOT, name)) === name, 'Noncanonical instrument input');
    assert(/^(?:src\/core\/(?:pipeline|types|gpu\/device|acquisition\/(?:frame-importer|video-source)|metrics\/(?:gpu-timer|stats)|present\/canvas-target|upscale\/(?:baseline-scaler|baseline\.wgsl))|tools\/m1010r\/(?:calibration|probe|audio-control|render-host|render-control|render-recorder))\.ts$/.test(name),
      `Forbidden instrument input: ${name}`);
  }
}

function canonicalCachePath(path) {
  assert(typeof path === 'string' && !path.includes('\0'), 'Invalid artifact path');
  const absolute = resolve(ROOT, path), cache = join(ROOT, '.cache/m1010r');
  assert(absolute.startsWith(cache + sep), 'Artifacts must stay below ignored .cache/m1010r');
  let current = ROOT;
  for (const part of relative(ROOT, absolute).split(sep)) {
    current = join(current, part);
    if (existsSync(current)) assert(!lstatSync(current).isSymbolicLink(), 'Symlinked artifact paths are forbidden');
  }
  return absolute;
}

export function cachePaths(paths) {
  assert(Array.isArray(paths), 'Artifact paths must be an array');
  const absolute = paths.map(canonicalCachePath);
  const names = [...new Set(absolute.map(path => relative(ROOT, path)))];
  if (!names.length) return absolute;
  const output = execFileSync('git', ['check-ignore', '--stdin', '-z'], {
    cwd: ROOT, encoding: 'utf8', input: names.join('\0') + '\0',
  });
  assert(output.endsWith('\0'), 'Malformed Git ignore response');
  assert.deepEqual(new Set(output.slice(0, -1).split('\0')), new Set(names), 'Every artifact must be ignored');
  return absolute;
}

export function cachePath(path) {
  return cachePaths([path])[0];
}

export function verifyReferences(references) {
  const paths = cachePaths(references.map(reference => {
    assert(reference && typeof reference.path === 'string');
    assert(Number.isSafeInteger(reference.bytes) && reference.bytes >= 0 && /^[a-f0-9]{64}$/.test(reference.sha256));
    return reference.path;
  }));
  return references.map((reference, index) => {
    const bytes = readFileSync(paths[index]);
    assert.deepEqual(fileInfo(bytes), { bytes: reference.bytes, sha256: reference.sha256 }, `Artifact changed: ${reference.path}`);
    return bytes;
  });
}

export function verifyReference(reference) {
  return verifyReferences([reference])[0];
}

export function verifyMediaManifest(input = DEFAULT_MEDIA) {
  const media = typeof input === 'string' ? JSON.parse(readFileSync(cachePath(input), 'utf8')) : structuredClone(input);
  assert.equal(media.schemaVersion, 1); assert.equal(media.study, 'M10.10R');
  assert.equal(media.sampleRate, 48000);
  assert(Number.isInteger(media.seconds) && media.seconds >= 70, 'Calibration requires at least 70 seconds of media');
  assert.equal(media.producerSha256, digest(readFileSync(join(ROOT, 'tools/m1010r/media.mjs'))), 'Media producer changed');
  assert.deepEqual(media.assets.map(asset => asset.fps).sort((left, right) => left - right), [30, 60]);
  verifyReference(media.signal);
  for (const asset of media.assets) {
    verifyReference(asset.media); verifyReference(asset.pcm);
    assert(asset.media.bytes > 0 && asset.pcm.bytes > 0 && asset.pcm.bytes % 4 === 0);
    assert.equal(asset.validation.audio.decodedPcmSha256, asset.pcm.sha256);
    assert.equal(asset.validation.audio.decodedPcmBytes, asset.pcm.bytes);
    assert.equal(asset.validation.video.fps, asset.fps);
    assert.equal(asset.validation.video.allIdentitiesAndPtsExact, true);
  }
  return media;
}

function fileNames(directory, prefix = '') {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    assert(!entry.isSymbolicLink(), 'Symlinked build contents are forbidden');
    const name = prefix + entry.name;
    if (entry.isDirectory()) return fileNames(join(directory, entry.name), `${name}/`);
    assert(entry.isFile(), 'Non-file build contents are forbidden');
    return [name];
  }).sort();
}

export function verifyBuild(outdir = DEFAULT_EXTENSION) {
  const directory = cachePath(outdir);
  const provenance = JSON.parse(readFileSync(join(directory, PROVENANCE_FILE), 'utf8'));
  const instrument = provenance.generator === 'm1010ri-instrument';
  assert.equal(provenance.schemaVersion, 1); assert.equal(provenance.generator, instrument ? 'm1010ri-instrument' : 'm1010r-calibration');
  if (instrument) assert.equal(provenance.instrument, true);
  else assert(provenance.instrument === undefined || provenance.instrument === false, 'Legacy build cannot claim RI');
  assert.equal(provenance.production, false); assert.equal(provenance.neural, false);
  assert.equal(provenance.modelSha256, MODEL_SHA256);
  verifyProductionModel(readFileSync(join(ROOT, 'public/models/aethersr-c16d2.json')));
  assert.deepEqual(provenance.permissions, []);
  const expected = ['manifest.json', 'calibration.js', instrument ? 'render-recorder.js' : 'audio-worklet.js', 'calibration.html', 'player.css', 'service-worker.js',
    ...[30, 60].flatMap(fps => [`media/replay-${fps}.mp4`, `media/decoded-${fps}.f32le`])].sort();
  assert.deepEqual(Object.keys(provenance.files).sort(), expected);
  assert.deepEqual(fileNames(directory), [...expected, PROVENANCE_FILE].sort());
  for (const name of expected) assert.deepEqual(fileInfo(readFileSync(join(directory, name))), provenance.files[name], `Build file changed: ${name}`);
  assert.equal(provenance.bundleSha256, digest(JSON.stringify(provenance.files)));
  assert.equal(provenance.totalBytes, Object.values(provenance.files).reduce((total, file) => total + file.bytes, 0));
  assert.deepEqual(JSON.parse(readFileSync(join(directory, 'manifest.json'), 'utf8')), instrument ? instrumentManifest : manifest);
  assert.equal(readFileSync(join(directory, 'service-worker.js'), 'utf8'), workerSource);
  if (instrument) {
    assert.deepEqual(Object.keys(provenance.bundleInputs).sort(), ['calibration.js', 'render-recorder.js']);
    const expectedSources = new Set(buildSources);
    if (existsSync(join(ROOT, 'package-lock.json'))) expectedSources.add('package-lock.json');
    for (const [output, inputs] of Object.entries(provenance.bundleInputs)) {
      verifyInstrumentInputs(inputs);
      assert(inputs.includes(`tools/m1010r/${output.replace(/\.js$/, '.ts')}`), 'Missing entry input');
      assert.deepEqual(inputs, [...new Set(inputs)].sort(), 'Noncanonical bundle inputs');
      for (const name of inputs) expectedSources.add(name);
    }
    assert.deepEqual(Object.keys(provenance.sourceFiles).sort(), [...expectedSources].sort(), 'Source hash inventory changed');
    assert(/^[a-f0-9]{40}$/.test(provenance.sourceCommit));
    assert.equal(typeof provenance.sourceDirty, 'boolean');
    for (const [name, expectedSource] of Object.entries(provenance.sourceFiles)) {
      const path = resolve(ROOT, name);
      assert(path.startsWith(ROOT + sep) && relative(ROOT, path) === name, 'Build input outside repository');
      assert.deepEqual(fileInfo(readFileSync(path)), expectedSource, `Build input changed: ${name}`);
    }
  }
  const media = verifyMediaManifest(provenance.mediaManifest);
  assert.equal(provenance.mediaManifestSha256, digest(JSON.stringify(media)));
  const mediaAssets = media.assets.map(asset => ({ fps: asset.fps,
    media: { ...provenance.files[`media/replay-${asset.fps}.mp4`], path: `media/replay-${asset.fps}.mp4` },
    pcm: { ...provenance.files[`media/decoded-${asset.fps}.f32le`], path: `media/decoded-${asset.fps}.f32le` } }));
  assert.deepEqual(provenance.mediaAssets, mediaAssets);
  for (const asset of media.assets) {
    assert.deepEqual(provenance.files[`media/replay-${asset.fps}.mp4`], { bytes: asset.media.bytes, sha256: asset.media.sha256 });
    assert.deepEqual(provenance.files[`media/decoded-${asset.fps}.f32le`], { bytes: asset.pcm.bytes, sha256: asset.pcm.sha256 });
  }
  return { directory, provenance };
}

export async function buildCalibration(outdir = DEFAULT_EXTENSION, mediaManifest = DEFAULT_MEDIA, { instrument = false } = {}) {
  assert.equal(typeof instrument, 'boolean');
  const directory = cachePath(outdir), media = verifyMediaManifest(mediaManifest);
  const generator = instrument ? 'm1010ri-instrument' : 'm1010r-calibration';
  const references = [media.signal, ...media.assets.flatMap(asset => [asset.media, asset.pcm])];
  if (typeof mediaManifest === 'string') references.push({ path: mediaManifest });
  for (const reference of references) {
    const path = cachePath(reference.path);
    assert(path !== directory && !path.startsWith(directory + sep), 'Build output must not contain its input media');
  }
  if (existsSync(directory)) {
    const previous = JSON.parse(readFileSync(join(directory, PROVENANCE_FILE), 'utf8'));
    assert.equal(previous.generator, generator, 'Refusing to overwrite foreign output');
  }
  const sourceCommit = git(['rev-parse', 'HEAD']);
  const status = git(['status', '--porcelain=v1', '--untracked-files=all']);
  verifyProductionModel(readFileSync(join(ROOT, 'public/models/aethersr-c16d2.json')));
  const artifacts = new Map([['manifest.json', json(instrument ? instrumentManifest : manifest)], ['service-worker.js', Buffer.from(workerSource)]]);
  const sourceFiles = new Set(buildSources);
  if (instrument && existsSync(join(ROOT, 'package-lock.json'))) sourceFiles.add('package-lock.json');
  const bundleInputs = {};
  for (const entry of ['calibration', instrument ? 'render-recorder' : 'audio-worklet']) {
    const built = await build({ absWorkingDir: ROOT, entryPoints: [`tools/m1010r/${entry}.ts`], bundle: true,
      write: false, minify: false, sourcemap: false, metafile: true, platform: 'browser', target: 'chrome116',
      format: 'iife', outfile: join(directory, `${entry}.js`), loader: { '.wgsl': 'text' },
      define: { 'import.meta.env.DEV': 'false', 'import.meta.env.PROD': 'true', __M1010RI__: String(instrument) } });
    assert.equal(built.outputFiles.length, 1);
    assert(Object.values(built.metafile.outputs).every(output => output.imports.length === 0), 'Bundle must be self-contained');
    const inputs = Object.keys(built.metafile.inputs).map(input => relative(ROOT, resolve(ROOT, input))).sort();
    if (instrument) verifyInstrumentInputs(inputs);
    bundleInputs[`${entry}.js`] = inputs;
    for (const input of inputs) sourceFiles.add(input);
    artifacts.set(`${entry}.js`, Buffer.from(built.outputFiles[0].contents));
  }
  artifacts.set('calibration.html', readFileSync(join(ROOT, 'tools/m1010r/calibration.html')));
  artifacts.set('player.css', readFileSync(join(ROOT, 'tools/m1010/player.css')));
  for (const asset of media.assets) {
    artifacts.set(`media/replay-${asset.fps}.mp4`, verifyReference(asset.media));
    artifacts.set(`media/decoded-${asset.fps}.f32le`, verifyReference(asset.pcm));
  }
  const files = Object.fromEntries([...artifacts].sort(([left], [right]) => left.localeCompare(right))
    .map(([name, bytes]) => [name, fileInfo(bytes)]));
  const sourceFilesHash = Object.fromEntries([...sourceFiles].sort().map(name => [name, fileInfo(readFileSync(join(ROOT, name)))]));
  assert.equal(git(['rev-parse', 'HEAD']), sourceCommit, 'Source moved during build');
  assert.equal(git(['status', '--porcelain=v1', '--untracked-files=all']), status, 'Worktree changed during build');
  const provenance = { schemaVersion: 1, generator, ...(instrument ? { instrument: true, bundleInputs } : {}), sourceCommit, sourceDirty: status !== '',
    modelSha256: MODEL_SHA256, production: false, neural: false, permissions: [], sourceFiles: sourceFilesHash,
    files, bundleSha256: digest(JSON.stringify(files)), totalBytes: Object.values(files).reduce((total, file) => total + file.bytes, 0),
    mediaManifest: media, mediaManifestSha256: digest(JSON.stringify(media)),
    mediaAssets: media.assets.map(asset => ({ fps: asset.fps,
      media: { ...files[`media/replay-${asset.fps}.mp4`], path: `media/replay-${asset.fps}.mp4` },
      pcm: { ...files[`media/decoded-${asset.fps}.f32le`], path: `media/decoded-${asset.fps}.f32le` } })) };
  rmSync(directory, { recursive: true, force: true });
  for (const [name, bytes] of [...artifacts, [PROVENANCE_FILE, json(provenance)]]) {
    mkdirSync(dirname(join(directory, name)), { recursive: true });
    writeFileSync(join(directory, name), bytes, { flag: 'wx' });
  }
  return { directory: relative(ROOT, directory), provenance };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert(process.argv.length <= 5 && (process.argv[4] === undefined || process.argv[4] === '--instrument') &&
    !process.argv.slice(2, 4).some(value => value.startsWith('--')), 'Usage: node tools/m1010r/build.mjs [outdir] [media.json] [--instrument]');
  const instrument = process.argv[4] === '--instrument';
  console.log(JSON.stringify(await buildCalibration(process.argv[2] || (instrument ? DEFAULT_RI_EXTENSION : DEFAULT_EXTENSION),
    process.argv[3] || DEFAULT_MEDIA, { instrument }), null, 2));
}