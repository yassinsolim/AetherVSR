import { chromium } from '../.cache/m9/node_modules/playwright/index.mjs';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { resolve } from 'node:path';

process.env.PLAYWRIGHT_BROWSERS_PATH = resolve('.cache/m9/browsers');
const cases = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const prefix = process.argv[3];
if (!prefix || !Array.isArray(cases)) throw new Error('Usage: node tools/m9-runtime.mjs cases.json output-prefix');
const git = (...args) => execFileSync('git', args, { encoding: 'utf8' }).trim();
const head = git('rev-parse', 'HEAD');
git('diff', '--exit-code');
git('diff', '--cached', '--exit-code');
const digest = (file) => createHash('sha256').update(readFileSync(file)).digest('hex');
const modelFile = 'public/models/aethersr-c16d2.json';
const modelSha256 = digest(modelFile);
if (modelSha256 !== 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a') throw new Error('Production model changed');
if (git('ls-files', '--others', '--exclude-standard', 'src', 'tools/m9-runtime.mjs')) throw new Error('Untracked executed source');
const source = Object.fromEntries(git('ls-files', 'src', 'tools/m9-runtime.mjs').split('\n').map(file => [file, digest(file)]));
const hardware = JSON.parse(execFileSync('system_profiler', ['SPHardwareDataType', 'SPDisplaysDataType', '-json'], { encoding: 'utf8' }));
const machine = hardware.SPHardwareDataType.map(row => ({ name: row.machine_name, chip: row.chip_type, memory: row.physical_memory }));
const displays = hardware.SPDisplaysDataType.map(row => ({ gpu: row.sppci_model,
  displays: row.spdisplays_ndrvs?.map(display => ({ name: display._name, resolution: display._spdisplays_resolution })) }));
const browser = await chromium.launch({ headless: false, args: [
  '--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis',
  '--disable-dawn-features=timestamp_quantization', '--autoplay-policy=no-user-gesture-required',
  '--window-position=0,0', '--window-size=1280,900',
] });
const quantile = (values, fraction) => {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1)))];
};
try {
  for (const config of cases) {
    const out = `${prefix}-${config.name}.json.gz`;
    if (existsSync(out)) throw new Error(`Refusing to overwrite ${out}`);
    const mediaSha256 = digest(`public${config.clip}`);
    const page = await browser.newPage({ viewport: { width: 1200, height: 820 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/bench.html');
    await page.bringToFront();
    console.log(`START ${config.name} at ${new Date().toISOString()}`);
    const result = await page.evaluate(async (settings) => {
      const { startCalibration } = await import('/src/bench/runtime-bench.ts');
      const bench = await startCalibration(settings.clip, settings.neural !== false);
      await new Promise(resolve => setTimeout(resolve, settings.warmupMs ?? 3000));
      const startup = structuredClone(await bench.finish());
      await bench.reset();
      if (settings.passes) bench.load(settings.passes, settings.loadFrames ?? Infinity);
      await new Promise(resolve => setTimeout(resolve, settings.durationMs));
      const measured = structuredClone(await bench.finish());
      bench.stop();
      return { startup, measured };
    }, config);
    const { measured } = result;
    const timings = measured.samples.map(sample => sample.ms);
    const summary = { samples: timings.length, p50: quantile(timings, 0.5), p90: quantile(timings, 0.9),
      p95: quantile(timings, 0.95), max: timings.length ? Math.max(...timings) : null,
      presented: measured.stats.meanSourceFps, rendered: measured.stats.meanRenderFps,
      skips: measured.stats.framesSkipped, drops: measured.stats.quality.droppedVideoFrames,
      callbackP95: quantile(measured.frames.map(frame => frame[3]), 0.95) };
    const valid = timings.length > 0 && measured.environment.focus && measured.environment.visibility === 'visible'
      && measured.events.every(event => !['hidden', 'prerender', 'blur'].includes(event.event))
      && measured.errors.length === 0 && errors.length === 0;
    if (git('rev-parse', 'HEAD') !== head) throw new Error('HEAD changed during measurement');
    if (digest(modelFile) !== modelSha256 || digest(`public${config.clip}`) !== mediaSha256) throw new Error('Model/media changed');
    for (const [file, hash] of Object.entries(source)) if (digest(file) !== hash) throw new Error(`Source changed: ${file}`);
    const artifact = { schema: 'aethervsr.m9-runtime/1', phase: 'calibration', config, head, source,
      mediaSha256, modelSha256, browser: browser.version(),
      os: execFileSync('sw_vers', ['-productVersion'], { encoding: 'utf8' }).trim(),
      osBuild: execFileSync('sw_vers', ['-buildVersion'], { encoding: 'utf8' }).trim(),
      machine, displays,
      measuredAt: new Date().toISOString(), valid, errors, summary, ...result };
    writeFileSync(out, gzipSync(JSON.stringify(artifact)), { flag: 'wx' });
    console.log(JSON.stringify({ name: config.name, valid, ...summary, output: out }));
    await page.screenshot({ path: `.cache/m9/${config.name}.png` });
    await page.close();
    if (!valid) throw new Error(`Invalid measurement ${config.name}; evidence retained`);
  }
} finally { await browser.close(); }