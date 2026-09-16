import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, join, dirname } from 'node:path';
import { execFileSync } from 'node:child_process';
import { createServer } from 'vite';
import { ROOT, sha256 } from '../m10-fixtures.mjs';
import { openNativeChrome } from '../m9-browser.mjs';
import { nativeWindow } from '../m105-accounting.mjs';
import { captureTask, seekPaused, pauseRuntime, compareCaptures } from '../m10-output-parity.mjs';

export function studyIdentity() {
  const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
  assert.equal(git(['status', '--porcelain']), '', 'Freeze clean research source before native measurement');
  git(['diff', '--exit-code', 'aab96fd6255641b590081e27dabd767448632aca', '--', 'src', 'models', 'public/models', 'package.json', 'package-lock.json']);
  const sourceCommit = git(['rev-parse', 'HEAD']);
  const directory = join(ROOT, '.cache/m1010/extension');
  const provenance = JSON.parse(readFileSync(join(directory, 'research-provenance.json')));
  assert.equal(provenance.sourceCommit, sourceCommit); assert.equal(provenance.sourceDirty, false);
  for (const [name, value] of Object.entries(provenance.files)) assert.equal(sha256(readFileSync(join(directory, name))), value.sha256);
  return { sourceCommit, provenance, directory, baseline: 'aab96fd6255641b590081e27dabd767448632aca' };
}

export async function openResearch(identity) {
  const native = await openNativeChrome([`--load-extension=${identity.directory}`, `--disable-extensions-except=${identity.directory}`]);
  try {
    const worker = native.context.serviceWorkers().find(worker => worker.url().endsWith('/service-worker.js')) ??
      await native.context.waitForEvent('serviceworker', { timeout: 10000 });
    const extensionId = await worker.evaluate(() => chrome.runtime.id);
    assert.equal(worker.url(), `chrome-extension://${extensionId}/service-worker.js`);
    return { ...native, worker, extensionId };
  } catch (error) { await native.close(); throw error; }
}

async function ready(page) {
  await page.waitForFunction(() => {
    const driver = globalThis.aethervsrRuntime?.driver;
    return !!driver && driver.pipeline.currentUpscaler.neural && driver.pipeline.framesRendered > 10 &&
      driver.pipeline.configuredSource.width === 1280 && driver.pipeline.configuredSource.height === 720;
  }, undefined, { timeout: 12000 });
}

async function parityFrames(page, forceCopy) {
  await page.evaluate(pauseRuntime, { extension: false });
  const results = [], modelJsonSha256 = sha256(Buffer.from(JSON.stringify(JSON.parse(readFileSync(join(ROOT, 'public/models/aethersr-c16d2.json'))))));
  for (const time of [1, 2, 3]) {
    const seek = await page.evaluate(seekPaused, { time });
    const capture = await page.evaluate(captureTask(), { extension: false, forceCopy, time, sourceURL: seek.source, modelJsonSha256 });
    results.push({ seek, capture });
  }
  return results;
}

export async function surfaceStudy(prefix) {
  prefix = resolve(ROOT, prefix); assert(prefix.startsWith(join(ROOT, '.cache/m1010/'))); mkdirSync(dirname(prefix), { recursive: true });
  const identity = studyIdentity(), report = { schemaVersion: 1, phase: 'P0_CONTROLLED_SURFACE', identity, started: new Date().toISOString(), references: [], results: [] };
  let native, server;
  try {
    server = await createServer({ root: ROOT, mode: 'benchmark', server: { host: '127.0.0.1', port: 5194, strictPort: true } }); await server.listen();
    native = await openResearch(identity);
    report.browser = { version: await native.browser.version(), executableSha256: sha256(readFileSync(native.executable)),
      mediaPolicy: 'default; no autoplay override, no capture permissions or prompts' };
    for (const forceCopy of [false, true]) {
      const harness = await native.context.newPage();
      await nativeWindow(harness, native.context);
      await harness.goto(`http://127.0.0.1:5194/?clip=/media/m9/720p60.mp4&mode=neural${forceCopy ? '&import=copy' : ''}`);
      await harness.bringToFront(); await harness.locator('#source').evaluate(video => { video.muted = true; });
      await harness.locator('#stage').click();
      await harness.evaluate(async () => { await document.querySelector('video').play(); });
      await ready(harness); const reference = await parityFrames(harness, forceCopy);
      report.references.push({ import: forceCopy ? 'copy' : 'external', frames: reference }); await harness.close();
      for (const surface of ['tab', 'window']) {
        const result = { surface, import: forceCopy ? 'copy' : 'external' }; report.results.push(result);
        let page;
        try {
          const url = `chrome-extension://${native.extensionId}/player.html?mode=neural${forceCopy ? '&import=copy' : ''}`;
          if (surface === 'window') {
            const nextPage = native.context.waitForEvent('page', { timeout: 5000 });
            const opened = await native.worker.evaluate(url => chrome.windows.create({ url, type: 'popup', width: 1000, height: 760 }), url);
            result.window = opened;
            page = await nextPage;
            await page.waitForLoadState();
          } else { page = await native.context.newPage(); await page.goto(url); }
          result.placement = await nativeWindow(page, native.context);
          await page.bringToFront(); await page.locator('#play').click(); await ready(page);
          result.before = await page.evaluate(() => globalThis.m1010.snapshot());
          result.frames = await parityFrames(page, forceCopy);
          result.comparisons = result.frames.map((row, index) => compareSurfaceFrame(row, reference[index]));
          assert(result.comparisons.every(row => row.verdict === 'PASS'), 'P0 exact pipeline output parity failed');
          await page.locator('#play').click();
          const session = await native.context.newCDPSession(page);
          try { const { windowId } = await session.send('Browser.getWindowForTarget'); await session.send('Browser.setWindowBounds', { windowId, bounds: { width: 900, height: 650 } }); }
          finally { await session.detach(); }
          result.resized = await page.evaluate(() => ({ width: innerWidth, height: innerHeight, origin: location.origin,
            canvas: { width: document.querySelector('canvas').width, height: document.querySelector('canvas').height } }));
          await page.locator('#fullscreen').click(); await page.waitForFunction(() => !!document.fullscreenElement, undefined, { timeout: 3000 });
          result.fullscreen = await page.evaluate(() => ({ active: !!document.fullscreenElement, origin: location.origin }));
          await page.evaluate(() => document.exitFullscreen());
          result.cleanup = await page.evaluate(() => globalThis.m1010.destroy());
          assert(Object.values(result.cleanup.resources).every(value => value === 0), 'P0 cleanup failed');
          assert.equal(result.cleanup.failure, null); result.outcome = 'SUPPORTED';
        } catch (error) { result.error = String(error); result.outcome = 'UNRESOLVED'; }
        finally { await page?.close(); }
        console.log(JSON.stringify({ surface, import: result.import, outcome: result.outcome, error: result.error ?? null }));
        if (result.outcome !== 'SUPPORTED') return report;
      }
    }
  } catch (error) { report.error = String(error); }
  finally { await native?.close(); await server?.close(); report.finished = new Date().toISOString(); writeFileSync(`${prefix}.json`, JSON.stringify(report, null, 2), { flag: 'wx' }); }
  return report;
}

export function compareSurfaceFrame(player, harness) {
  const result = compareCaptures(player.capture, harness.capture);
  const decodedMediaTimeEqual = player.seek.metadata.mediaTime === harness.seek.metadata.mediaTime;
  return { ...result, decodedMediaTimeEqual, verdict: decodedMediaTimeEqual && result.verdict === 'PASS' ? 'PASS' : 'FAIL' };
}