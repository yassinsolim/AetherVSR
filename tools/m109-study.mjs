import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, resolve, dirname, relative } from 'node:path';
import { gzipSync, gunzipSync } from 'node:zlib';
import { build } from 'esbuild';
import { createServer } from 'vite';
import { ROOT, sha256 } from './m10-fixtures.mjs';
import { openNativeChrome } from './m9-browser.mjs';
import { nativeWindow } from './m105-accounting.mjs';
import { presentationEnvironment } from './m107-presentation.mjs';

export const START = 'c727f0196b47d32eaad16869e6a849d6995f0075';
const semantic = (name, selector, property, value, expected = 'UNSUPPORTED') => ({ name,
  actions: [{ name: 'cssom', payload: { selector, property, value, expected } }, { name: 'reverse' }] });
export const SEMANTICS = [
  semantic('cssom-fit', 'video', 'object-fit', 'cover', 'SUPPORTED'),
  semantic('cssom-position', 'video', 'object-position', '0% 0%', 'SUPPORTED'),
  semantic('clip', '.player', 'clip-path', 'inset(0 80px 0 0)'),
  semantic('radius', 'video', 'border-radius', '24px', 'SUPPORTED'),
  semantic('transform', 'video', 'transform', 'rotate(180deg)'),
  semantic('opacity', 'video', 'opacity', '.5'), semantic('filter', 'video', 'filter', 'blur(1px)'),
  semantic('controls-z-index', '.controls', 'z-index', '0'),
  semantic('motion-path', 'video', 'offset-path', 'path("M 320 200 L 320 200")'),
  semantic('corner-shape', 'video', 'corner-shape', 'bevel'),
  semantic('clip-margin', 'video', 'overflow-clip-margin', '20px'),
  semantic('background', 'video', 'background-color', 'rgb(230,200,30)'),
  semantic('individual-rotate', 'video', 'rotate', '180deg'),
  semantic('individual-scale', 'video', 'scale', '-1 -1'),
  semantic('individual-translate', 'video', 'translate', '0px'),
  semantic('backdrop-filter', 'video', 'backdrop-filter', 'blur(1px)'),
  semantic('mask', 'video', 'mask-image', 'linear-gradient(black,transparent)'),
  semantic('blend', 'video', 'mix-blend-mode', 'multiply'),
  semantic('containment', '.player', 'contain', 'paint'),
  semantic('parent-contents', '.player', 'display', 'contents'),
  semantic('parent-flex', '.player', 'display', 'flex'),
  semantic('parent-grid', '.player', 'display', 'grid'),
  semantic('parent-isolation', '.player', 'isolation', 'auto'),
  semantic('video-content', 'video', 'content', 'url("/m109-media/source.png")'),
  semantic('object-view-box', 'video', 'object-view-box', 'inset(10%)'),
  semantic('box-shadow', 'video', 'box-shadow', 'red 0px 0px 12px 12px'),
  semantic('pseudo-paint', '.player::after', 'content', '"Local caption"', 'SUPPORTED'),
];
const actions = (name, names = [name]) => ({ name, actions: names.map(name => ({ name })) });
export const CASES = [
  actions('initial', ['noop']), actions('page-one'), actions('page-228'), actions('smooth'),
  actions('nested-scroll'), actions('preceding-spacer'), actions('class-movement'),
  ...SEMANTICS.slice(0, 8), actions('css-resize'), actions('viewport-clip'), actions('abr', ['source-high', 'source-low']),
  actions('equivalent-reparent'), actions('fullscreen-scrolled-enter-exit', ['page-228', 'fullscreen', 'exit']),
  actions('source-scroll', ['source-scroll', 'source-low']), actions('open-shadow', ['preceding-spacer']),
  actions('offscreen-return', ['offscreen', 'return']), actions('hidden', ['hide', 'show']),
  actions('paused', ['pause', 'play']), actions('not-ready', ['not-ready', 'ready']),
  { ...SEMANTICS[5], name: 'reverse-semantic-recovery' }, actions('initially-unsupported', ['initial-reverse']),
  actions('native-controls', ['native-controls', 'native-reverse']), actions('showing-track', ['showing-track', 'track-reverse']),
  ...SEMANTICS.slice(8), actions('cover', ['noop']), actions('rounded-video', ['noop']),
  actions('late-render-cssom', ['late-cssom-fit', 'reverse']),
];
export const OWNERSHIP_CASES = {
  O1: ['absent', 'existing', 'important', 'host-append', 'host-replace', 'host-remove-token', 'host-priority',
    'unrelated-style', 'host-remove-existing', 'repeated', 'collision', 'exception', 'explicit-none',
    'host-empty-style', 'unsupported-grammar', 'pending-host', 'same-value', 'style-recreate',
    'synchronous-reaction', 'overflow', 'live-variable', 'live-escape'],
  O2: ['absent', 'collision', 'host-replace', 'host-remove-token', 'same-value', 'style-recreate',
    'pending-host', 'repeated', 'exception', 'observer-visibility', 'style-sensitive', 'generic-data-observer'],
};

export function identity() {
  const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', maxBuffer: 16 * 1024 ** 2 }).trim();
  assert.equal(git(['status', '--porcelain']), '', 'Native capture requires clean committed apparatus');
  git(['diff', '--exit-code', START, '--', 'src', 'models', 'public/models', 'package.json', 'package-lock.json']);
  const frozen = JSON.parse(readFileSync(join(ROOT, '.cache/m109/frozen-controls.json')));
  for (const entry of frozen.packages) for (const [file, info] of Object.entries(entry.provenance.files)) {
    assert.equal(sha256(readFileSync(join(ROOT, entry.path, file))), info.sha256);
  }
  return { sourceCommit: git(['rev-parse', 'HEAD']), baseline: START, frozen,
    pins: Object.fromEntries(['tools/m109-study.mjs', 'tools/m109-contract.ts', 'tools/m109-monitor.ts',
      'tools/m109-submission.ts', 'tools/m109-ownership.ts', 'tools/m109-fixture.html',
      'tools/m109_pixels.py',
      'docs/M10.9-DESIGN-PREREGISTRATION.md'].map(path => [path, sha256(readFileSync(join(ROOT, path)))])) };
}

export function prepareMedia() {
  const directory = join(ROOT, '.cache/m109/media'); mkdirSync(directory, { recursive: true });
  const records = [];
  for (const [name, width, height, fragmented] of [['pattern', 320, 180, false], ['low', 320, 180, true], ['high', 640, 480, true]]) {
    const path = join(directory, `${name}.mp4`);
    const filter = `color=c=0x3050d0:s=${width}x${height}:r=30,drawbox=x=0:y=0:w=iw/3:h=ih:color=0xd04040:t=fill,drawbox=x=iw/3:y=0:w=iw/3:h=ih:color=0x30b060:t=fill`;
    const args = ['-nostdin', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'lavfi', '-i', filter,
      '-t', '4', '-an', '-c:v', 'libx264', '-preset', 'fast', '-profile:v', 'high', '-level:v', '5.2',
      '-pix_fmt', 'yuv420p', '-g', '120', '-bf', '0', '-movflags', fragmented ? 'frag_keyframe+empty_moov+default_base_moof' : '+faststart',
      '-map_metadata', '-1', '-threads', '2', path];
    if (!existsSync(path)) execFileSync('ffmpeg', args, { stdio: 'inherit' });
    const bytes = readFileSync(path), config = bytes.indexOf(Buffer.from('avcC')); assert(config >= 0);
    records.push({ name, width, height, fps: 30, bytes: bytes.length, sha256: sha256(bytes), recipe: args,
      mime: `video/mp4; codecs="avc1.${bytes.subarray(config + 5, config + 8).toString('hex')}"` });
  }
  assert.equal(records[1].mime, records[2].mime);
  const image = join(directory, 'source.png');
  if (!existsSync(image)) execFileSync('ffmpeg', ['-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
    '-i', join(directory, 'pattern.mp4'), '-frames:v', '1', image], { stdio: 'inherit' });
  const imageBytes = readFileSync(image);
  const metadata = { mime: records[1].mime, records, image: { name: 'source.png', sha256: sha256(imageBytes), bytes: imageBytes.length },
    scope: 'Locally generated stationary three-color synthetic video; not a neural quality/performance clip' };
  const path = join(directory, 'config.json');
  if (!existsSync(path)) writeFileSync(path, JSON.stringify(metadata, null, 2), { flag: 'wx' });
  else assert.deepEqual(JSON.parse(readFileSync(path)), metadata);
  return metadata;
}

async function server() {
  const media = prepareMedia(), html = readFileSync(join(ROOT, 'tools/m109-fixture.html'));
  const bundle = await build({ stdin: { contents: 'export * from "./tools/m109-monitor.ts"; export * from "./tools/m109-contract.ts"; export * from "./tools/m109-ownership.ts"; export * from "./tools/m109-submission.ts"; export {VideoAttachment} from "./src/extension/attachment.ts"; export {packModel} from "./src/core/neural/model.ts";',
    resolveDir: ROOT, sourcefile: 'm109-research-entry.ts' }, bundle: true, write: false, format: 'iife', globalName: 'M109', loader: { '.wgsl': 'text' } });
  const instance = await createServer({ root: ROOT, mode: 'benchmark', server: { host: '127.0.0.1', port: 5193, strictPort: true }, plugins: [{
    name: 'm109-local-fixture', configureServer(vite) { vite.middlewares.use((request, response, next) => {
      const path = new URL(request.url, 'http://local').pathname;
      if (path === '/m109') { response.setHeader('Content-Type', 'text/html'); response.end(html); return; }
      const file = { '/m109-media/pattern.mp4': 'pattern.mp4', '/m109-media/low.mp4': 'low.mp4', '/m109-media/high.mp4': 'high.mp4', '/m109-media/config.json': 'config.json', '/m109-media/source.png': 'source.png' }[path];
      if (!file) { next(); return; }
      response.setHeader('Content-Type', file.endsWith('.json') ? 'application/json' : file.endsWith('.png') ? 'image/png' : 'video/mp4');
      response.end(readFileSync(join(ROOT, '.cache/m109/media', file)));
    }); },
  }] });
  await instance.listen();
  return { url: 'http://127.0.0.1:5193/m109', media, source: bundle.outputFiles[0].text,
    bundleSha256: sha256(bundle.outputFiles[0].contents), close: () => instance.close() };
}

async function isolated(native, page, source) {
  const session = await native.context.newCDPSession(page), tree = await session.send('Page.getFrameTree');
  const { executionContextId } = await session.send('Page.createIsolatedWorld', { frameId: tree.frameTree.frame.id, worldName: 'aethervsr-m109-research' });
  const evaluate = async expression => {
    const reply = await session.send('Runtime.evaluate', { contextId: executionContextId, expression, awaitPromise: true, returnByValue: true });
    if (reply.exceptionDetails) throw new Error(JSON.stringify(reply.exceptionDetails));
    return reply.result.value;
  };
  await evaluate(source);
  return { call: (fn, arg) => evaluate(`(${fn.toString()})(${JSON.stringify(arg ?? null)})`), close: () => session.detach() };
}

export function installSampler() {
  const fixture = globalThis.__M109_FIXTURE__, rows = [], interruptions = [];
  let frame, stopped = false, firstFailure = null;
  const rect = element => { const value = element.getBoundingClientRect(); return { left: value.left, top: value.top, width: value.width, height: value.height }; };
  const sample = (boundary, record = null) => {
    const video = fixture.video, canvas = video.parentNode.querySelector?.('canvas[data-aethervsr-m109],canvas[data-aethervsr-m10]') ?? document.querySelector('canvas[data-aethervsr-m109],canvas[data-aethervsr-m10]');
    const original = getComputedStyle(video), output = canvas ? getComputedStyle(canvas) : null;
    const visible = !!canvas && output.visibility === 'visible' && output.display !== 'none';
    const expected = fixture.state.expected;
    const videoRect = rect(video), canvasRect = canvas ? rect(canvas) : null;
    const geometry = canvasRect && Object.keys(videoRect).every(key => Math.abs(videoRect[key] - canvasRect[key]) <= .5);
    const fit = !visible || output.objectFit === original.objectFit && output.objectPosition === original.objectPosition;
    const backing = !!canvas && canvas.width === video.videoWidth * 2 && canvas.height === video.videoHeight * 2;
    const host = video.isConnected && video.parentNode === fixture.state.expectedParent;
    const row = { at: performance.now(), boundary, record, expected, visible, video: videoRect, canvas: canvasRect,
      captureOriginal: fixture.captureOriginal === true,
      videoFit: original.objectFit, canvasFit: output?.objectFit ?? null, videoPosition: original.objectPosition,
      canvasPosition: output?.objectPosition ?? null, videoRadius: original.borderTopLeftRadius, canvasRadius: output?.borderTopLeftRadius ?? null,
      source: [video.videoWidth, video.videoHeight], backing: canvas ? [canvas.width, canvas.height] : null,
      geometry: !!geometry, fit, backingCorrect: backing, host, pointer: output?.pointerEvents ?? null,
      focused: document.hasFocus(), visibility: document.visibilityState, paused: video.paused, ready: video.readyState };
    rows.push(row);
    if (!row.focused || row.visibility !== 'visible') interruptions.push({ at: row.at, focused: row.focused, visibility: row.visibility, boundary });
    if (!firstFailure && ['render', 'submission', 'late-render'].includes(boundary) && visible &&
      (expected === 'UNSUPPORTED' || !geometry || !fit || !backing || !host || output.pointerEvents !== 'none' || original.borderTopLeftRadius !== output.borderTopLeftRadius)) {
      firstFailure = row;
      dispatchEvent(new Event('m109:safety-trip'));
    }
    if (rows.length > 20000) { firstFailure ??= { reason: 'sampler-overflow' }; dispatchEvent(new Event('m109:safety-trip')); }
    return row;
  };
  const submission = event => sample('submission', event.detail);
  const action = () => sample('action');
  const lateRender = () => sample('late-render');
  const interrupted = event => { if (event.type === 'blur' || document.visibilityState !== 'visible') interruptions.push({ at: performance.now(), type: event.type }); };
  addEventListener('m109:submission', submission); addEventListener('m109:action', action);
  addEventListener('m109:late-render', lateRender);
  addEventListener('blur', interrupted); document.addEventListener('visibilitychange', interrupted);
  const render = () => { sample('render'); if (!stopped) frame = requestAnimationFrame(render); };
  globalThis.__M109_SAMPLER__ = { rows, start: () => { frame = requestAnimationFrame(render); }, status: () => ({ firstFailure, interruptions, latest: rows.at(-1) }),
    stop: () => { stopped = true; cancelAnimationFrame(frame); removeEventListener('m109:submission', submission); removeEventListener('m109:action', action);
      removeEventListener('m109:late-render', lateRender); removeEventListener('blur', interrupted); document.removeEventListener('visibilitychange', interrupted); return { rows, firstFailure, interruptions }; } };
}

export function comparePixels(original, replacement, width, height, region, dpr = 2, expectedColor = null, outsideCorner = null) {
  assert.equal(original.length, width * height * 3); assert.equal(replacement.length, original.length);
  const border = Math.ceil(2 * dpr), startX = Math.max(border, Math.ceil(region.left * dpr)), startY = Math.max(border, Math.ceil(region.top * dpr));
  const endX = Math.min(width - border, Math.floor((region.left + region.width) * dpr)), endY = Math.min(height - border, Math.floor((region.top + region.height) * dpr));
  let tested = 0, wrong = 0, wrongOriginal = 0, maxDifference = 0;
  const colors = new Set();
  for (let vertical = startY; vertical < endY; vertical++) for (let horizontal = startX; horizontal < endX; horizontal++) {
    const offset = (vertical * width + horizontal) * 3;
    if (outsideCorner && Math.hypot(horizontal / dpr - outsideCorner.centerX, vertical / dpr - outsideCorner.centerY) <= outsideCorner.radius + 2) continue;
    const neighbors = [offset - border * 3, offset + border * 3, offset - border * width * 3, offset + border * width * 3];
    if (neighbors.some(neighbor => [0, 1, 2].some(channel => Math.abs(original[neighbor + channel] - original[offset + channel]) > 2))) continue;
    tested++; colors.add(original.subarray(offset, offset + 3).toString('hex'));
    if (Array.isArray(expectedColor) && expectedColor.some((value, channel) => Math.abs(value - original[offset + channel]) > 8)) wrongOriginal++;
    if (expectedColor && !Array.isArray(expectedColor) && 'dominantChannel' in expectedColor) {
      const channel = expectedColor.dominantChannel;
      if ([0, 1, 2].filter(index => index !== channel).some(index => original[offset + channel] - original[offset + index] < 64)) wrongOriginal++;
    }
    const difference = Math.max(...[0, 1, 2].map(channel => Math.abs(original[offset + channel] - replacement[offset + channel])));
    maxDifference = Math.max(maxDifference, difference); if (difference > 8) wrong++;
  }
  return { tested, wrong, wrongOriginal, maxDifference, colors: colors.size, edgeExclusionCssPixels: 2, region, expectedColor, outsideCorner,
    verdict: tested < 100 || !expectedColor && colors.size < 2 ? 'UNRESOLVED' : wrong || wrongOriginal ? 'UNSAFE' : 'SUPPORTED_CORRECT' };
}

export function validateReveals(trace, telemetry, actions = []) {
  const failures = []; let reveals = 0, previousVisible = false;
  for (const row of trace.rows) {
    if (!['render', 'submission'].includes(row.boundary)) continue;
    if (row.captureOriginal) continue;
    if (row.visible && !previousVisible) {
      reveals++;
      const records = telemetry.submissions.filter(record => record.observedAt <= row.at);
      const latest = records.at(-1), prior = records.at(-2);
      const keys = ['owner', 'sourceGeneration', 'geometryGeneration', 'frameGeneration', 'backingWidth', 'backingHeight'];
      const proof = latest && telemetry.proofs.find(proof => proof.supported && proof.generation === latest.geometryGeneration && proof.at <= prior?.observedAt);
      const action = actions.filter(action => action.startedAt <= row.at && action.name !== 'noop').at(-1);
      if (!latest?.validForRecovery || !prior?.validForRecovery || latest.sequence <= prior.sequence || !keys.every(key => latest[key] === prior[key]) ||
        action && (prior.observedAt < action.startedAt || !proof || proof.at < action.startedAt) ||
        latest.sourceWidth !== row.source[0] || latest.sourceHeight !== row.source[1] || latest.backingWidth !== row.backing?.[0] || latest.backingHeight !== row.backing?.[1] || !proof) failures.push({ at: row.at, latest, prior, reason: 'reveal-without-two-matching-post-proof-submissions' });
    }
    previousVisible = row.visible;
  }
  return { reveals, failures, verdict: reveals > 0 && failures.length === 0 ? 'SUPPORTED_CORRECT' : failures.length ? 'UNSAFE' : 'UNRESOLVED' };
}

function decodeScreenshot(bytes) {
  assert.equal(bytes.subarray(1, 4).toString(), 'PNG');
  const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20);
  const python = process.env.M109_PYTHON ?? join(ROOT, '.cache/m8-venv/bin/python');
  const pixels = execFileSync(python, [join(ROOT, 'tools/m109_pixels.py')], { input: bytes, maxBuffer: 64 * 1024 ** 2 });
  const colorProfile = JSON.parse(execFileSync(python, [join(ROOT, 'tools/m109_pixels.py'), '--metadata'], { input: bytes, encoding: 'utf8' }));
  assert.equal(pixels.length, width * height * 3);
  return { width, height, pixels, colorProfile };
}

async function capture(page, path) {
  assert(!existsSync(path)); const bytes = await page.screenshot({ path, timeout: 5000 });
  return { path: relative(ROOT, path), bytes: bytes.length, sha256: sha256(bytes), ...decodeScreenshot(bytes) };
}

async function paintedProof(context, prefix) {
  const info = await context.page.evaluate(() => {
    const fixture = globalThis.__M109_FIXTURE__;
    const clip = element => {
      let left = 0, top = 0, right = innerWidth, bottom = innerHeight;
      for (let parent = element.parentElement; parent; parent = parent.parentElement) {
        const style = getComputedStyle(parent), box = parent.getBoundingClientRect();
        if (['hidden', 'clip', 'auto', 'scroll'].includes(style.overflowX)) { left = Math.max(left, box.left + parent.clientLeft); right = Math.min(right, box.left + parent.clientLeft + parent.clientWidth); }
        if (['hidden', 'clip', 'auto', 'scroll'].includes(style.overflowY)) { top = Math.max(top, box.top + parent.clientTop); bottom = Math.min(bottom, box.top + parent.clientTop + parent.clientHeight); }
        if (parent === document.fullscreenElement) break;
      }
      return { left, top, right, bottom };
    };
    return { rect: fixture.rect(fixture.video), source: [fixture.video.videoWidth, fixture.video.videoHeight],
      fit: fixture.state.fit, position: fixture.state.position, radius: fixture.state.radius,
      caption: fixture.rect(fixture.player.querySelector('#caption')), controls: fixture.rect(fixture.player.querySelector('#activate')),
      clips: { video: clip(fixture.video), caption: clip(fixture.player.querySelector('#caption')), control: clip(fixture.player.querySelector('#activate')) },
      viewport: [innerWidth, innerHeight], dpr: devicePixelRatio };
  });
  await context.page.evaluate(() => { globalThis.__M109_FIXTURE__.captureOriginal = true; });
  await context.world.call(() => globalThis.__M109_MONITOR__.setCaptureOriginal(true));
  let original, replacement;
  try { original = await capture(context.page, `${prefix}.original.png`); }
  finally { await context.world.call(() => globalThis.__M109_MONITOR__.setCaptureOriginal(false)); }
  await context.page.waitForFunction(() => { const fixture = globalThis.__M109_FIXTURE__, canvas = fixture.video.parentNode.querySelector('canvas[data-aethervsr-m109]');
    return canvas && getComputedStyle(canvas).visibility === 'visible' && canvas.width === fixture.video.videoWidth * 2 && canvas.height === fixture.video.videoHeight * 2; }, undefined, { timeout: 3000 });
  replacement = await capture(context.page, `${prefix}.replacement.png`);
  const visibleAfter = await context.page.evaluate(() => { const fixture = globalThis.__M109_FIXTURE__, canvas = fixture.video.parentNode.querySelector('canvas[data-aethervsr-m109]');
    return getComputedStyle(canvas).visibility === 'visible' && canvas.width === fixture.video.videoWidth * 2 && canvas.height === fixture.video.videoHeight * 2; });
  assert(visibleAfter, 'Replacement was not proven visible/configured throughout screenshot boundary');
  await context.page.evaluate(() => { globalThis.__M109_FIXTURE__.captureOriginal = false; });
  assert.equal(original.width, info.viewport[0] * info.dpr); assert.equal(original.height, info.viewport[1] * info.dpr);
  assert.equal(replacement.width, original.width); assert.equal(replacement.height, original.height);
  const box = info.rect, ratio = (info.fit === 'cover' ? Math.max : Math.min)(box.width / info.source[0], box.height / info.source[1]);
  const image = { width: info.source[0] * ratio, height: info.source[1] * ratio };
  image.left = box.left + (box.width - image.width) * info.position[0];
  image.top = box.top + (box.height - image.height) * info.position[1];
  const regions = [];
  const add = (name, rect, color, corner = null) => {
    const clip = info.clips[name === 'caption' ? 'caption' : name === 'control' ? 'control' : 'video'];
    const left = Math.max(clip.left, rect.left), top = Math.max(clip.top, rect.top);
    const width = Math.min(clip.right, rect.left + rect.width) - left, height = Math.min(clip.bottom, rect.top + rect.height) - top;
    if (width <= 0 || height <= 0) { regions.push({ name, status: 'NOT VISIBLE', region: rect }); return; }
    regions.push({ name, ...comparePixels(original.pixels, replacement.pixels, original.width, original.height,
      { left, top, width, height }, info.dpr, color, corner) });
  };
  for (const index of [0, 1, 2]) {
    add(`source-color-${index}`, { left: image.left + image.width * (index + .5) / 3 - 12,
      top: Math.max(12, image.top + image.height / 2 - 12), width: 24, height: 24 }, { dominantChannel: index, minimumSeparation: 64 });
  }
  if (info.fit === 'contain' && image.height < box.height - 8) {
    add('contain-top-bar', { left: box.left + box.width / 4, top: box.top + 3,
      width: box.width / 2, height: image.top - box.top - 6 }, [24, 24, 24]);
    add('contain-bottom-bar', { left: box.left + box.width / 4, top: image.top + image.height + 3,
      width: box.width / 2, height: box.top + box.height - image.top - image.height - 6 }, [24, 24, 24]);
  } else if (info.fit === 'contain' && image.width < box.width - 8) {
    add('contain-left-bar', { left: box.left + 3, top: box.top + box.height / 3, width: image.left - box.left - 6, height: box.height / 3 }, [24, 24, 24]);
    add('contain-right-bar', { left: image.left + image.width + 3, top: box.top + box.height / 3,
      width: box.left + box.width - image.left - image.width - 6, height: box.height / 3 }, [24, 24, 24]);
  }
  if (info.radius) add('rounded-outside', { left: box.left, top: box.top, width: info.radius, height: info.radius }, [24, 24, 24],
    { centerX: box.left + info.radius, centerY: box.top + info.radius, radius: info.radius });
  add('caption', { left: info.caption.left + 3, top: info.caption.top + 3, width: info.caption.width - 6, height: info.caption.height - 6 }, [240, 40, 200]);
  add('control', info.controls, { pairedControlPixels: true });
  add('control-background-top', { left: info.controls.left + 6, top: info.controls.top + 3,
    width: info.controls.width - 12, height: 3 }, [220, 220, 220]);
  add('control-background', { left: info.controls.left + 6, top: info.controls.top + info.controls.height - 6,
    width: info.controls.width - 12, height: 3 }, [220, 220, 220]);
  const actual = regions.filter(region => region.status !== 'NOT VISIBLE');
  return { info, image, regions, original: { path: original.path, sha256: original.sha256, bytes: original.bytes, colorProfile: original.colorProfile },
    replacement: { path: replacement.path, sha256: replacement.sha256, bytes: replacement.bytes, colorProfile: replacement.colorProfile },
    verdict: actual.some(region => region.verdict === 'UNSAFE') ? 'UNSAFE' :
      actual.length >= 5 && actual.every(region => region.verdict === 'SUPPORTED_CORRECT') ? 'SUPPORTED_CORRECT' : 'UNRESOLVED',
    scope: 'Browser painted pixels, independently derived colored-source regions; not physical scanout or neural quality' };
}

async function controlProof(context) {
  const before = await context.page.evaluate(() => globalThis.__M109_FIXTURE__.state.activations);
  const point = await context.page.evaluate(() => {
    const control = globalThis.__M109_FIXTURE__.player.querySelector('#activate'), box = control.getBoundingClientRect();
    let left = Math.max(0, box.left), top = Math.max(0, box.top), right = Math.min(innerWidth, box.right), bottom = Math.min(innerHeight, box.bottom);
    for (let parent = control.parentElement; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent), bounds = parent.getBoundingClientRect();
      if (['hidden', 'clip', 'auto', 'scroll'].includes(style.overflowX)) { left = Math.max(left, bounds.left + parent.clientLeft); right = Math.min(right, bounds.left + parent.clientLeft + parent.clientWidth); }
      if (['hidden', 'clip', 'auto', 'scroll'].includes(style.overflowY)) { top = Math.max(top, bounds.top + parent.clientTop); bottom = Math.min(bottom, bounds.top + parent.clientTop + parent.clientHeight); }
      if (parent === document.fullscreenElement) break;
    }
    if (!(right > left && bottom > top)) throw new Error('Control has no inspectable visible hit region');
    return { x: (left + right) / 2, y: (top + bottom) / 2 };
  });
  await context.page.mouse.click(point.x, point.y);
  return context.page.evaluate(({ before, point }) => {
    const fixture = globalThis.__M109_FIXTURE__, control = fixture.player.querySelector('#activate');
    const canvas = fixture.video.parentNode.querySelector('canvas[data-aethervsr-m109],canvas[data-aethervsr-m10]');
    const root = control.getRootNode();
    return { activated: fixture.state.activations === before + 1,
      hit: root.elementsFromPoint(point.x, point.y)[0] === control, point,
      pointerEvents: getComputedStyle(canvas).pointerEvents, focus: fixture.host().focus };
  }, { before, point });
}

export async function startBroadControl() {
  const video = document.querySelector('video'), model = M109.packModel(await fetch('/models/aethersr-c16d2.json').then(response => response.json()));
  const attachment = new M109.VideoAttachment(video, { model, mode: 'baseline', onFailure: (code, message) => { failure = `${code}:${message}`; } });
  let failure = null; const submissions = [];
  await attachment.start();
  if (!attachment.pipeline) throw new Error('Broad control has no pipeline');
  const release = M109.observeSuccessfulSubmissions(attachment.pipeline, () => ({ owner: 'S0-control',
    sourceGeneration: attachment.sourceGeneration, geometryGeneration: attachment.geometryGeneration,
    backingWidth: attachment.canvas.width, backingHeight: attachment.canvas.height, authorized: attachment.snapshot().ready }), record => {
    submissions.push(record); window.dispatchEvent(new CustomEvent('m109:submission', { detail: record }));
  });
  const stop = () => attachment.destroy(); window.addEventListener('m109:safety-trip', stop, { once: true });
  globalThis.__M109_MONITOR__ = { snapshot: () => { const snapshot = attachment.snapshot(); return { visible: snapshot.ready, reason: snapshot.suspendedReason,
    proofCalls: snapshot.infrastructure.geometryCalls, initializes: 1, owner: 'S0-control', error: failure }; },
  dispose: () => { release(); window.removeEventListener('m109:safety-trip', stop); attachment.destroy();
    return { error: failure, resources: attachment.snapshot().resources, initializes: 1, submissions, proofs: [], transitions: [] }; } };
  return globalThis.__M109_MONITOR__.snapshot();
}

function raw(path, data) {
  const bytes = gzipSync(JSON.stringify(data)); writeFileSync(path, bytes, { flag: 'wx' });
  return { path: relative(ROOT, path), bytes: bytes.length, sha256: sha256(bytes) };
}

export async function ownershipCase({ model, name }) {
  const video = document.querySelector('video'), token = `--aethervsr-${crypto.randomUUID()}`;
  const rect = () => { const value = video.getBoundingClientRect(); return [value.left, value.top, value.width, value.height]; };
  const attribute = model === 'O1' ? 'style' : 'data-aethervsr-anchor';
  let hostObserver, sideEffect = 0, observerCount = 0;
  if (['existing', 'host-remove-existing'].includes(name)) video.style.setProperty('anchor-name', '--host-one');
  if (name === 'important') video.style.setProperty('anchor-name', '--host-one', 'important');
  if (name === 'explicit-none') video.style.setProperty('anchor-name', 'none', 'important');
  if (name === 'unsupported-grammar') video.style.setProperty('anchor-name', 'var(--names)');
  if (name === 'collision') { const peer = document.createElement('div'); peer.style.setProperty('anchor-name', token); document.body.append(peer); }
  if (name === 'style-sensitive') document.styleSheets[0].insertRule('video[data-aethervsr-anchor]{width:320px!important}', document.styleSheets[0].cssRules.length);
  if (['generic-data-observer', 'observer-visibility'].includes(name)) {
    hostObserver = new MutationObserver(records => { observerCount += records.length;
      if (name === 'generic-data-observer' && records.some(record => record.attributeName.startsWith('data-'))) { sideEffect++; video.style.width = '320px'; } });
    hostObserver.observe(video, { attributes: true });
  }
  const properties = () => Object.fromEntries([...video.style].map(field => [field, [video.style.getPropertyValue(field), video.style.getPropertyPriority(field)]]));
  const before = { style: video.getAttribute('style'), attribute: video.getAttribute(attribute), rect: rect(), properties: properties() };
  const create = model === 'O1' ? M109.leaseProperty : M109.leaseAttribute;
  if (name === 'repeated') for (let count = 0; count < 3; count++) {
    const temporary = create(video, `--aethervsr-${crypto.randomUUID()}`), active = temporary.check(), released = temporary.release();
    if (!active.active || !['SUPPORTED_CORRECT', 'UNSUPPORTED_SAFE'].includes(released.outcome)) return { model, name, count, acquired: active, released, outcome: 'UNRESOLVED' };
  }
  const lease = create(video, token), acquired = lease.check();
  await Promise.resolve();
  const during = { style: video.getAttribute('style'), attribute: video.getAttribute(attribute), rect: rect() };
  let exception = null;
  try {
    if (name === 'host-append') video.style.setProperty('anchor-name', `${token}, --host-added`, 'important');
    if (name === 'host-replace') model === 'O1' ? video.style.setProperty('anchor-name', '--replacement', 'important') : video.setAttribute(attribute, 'host-replacement');
    if (name === 'host-remove-token') model === 'O1' ? video.style.removeProperty('anchor-name') : video.removeAttribute(attribute);
    if (name === 'host-priority') video.style.setProperty('anchor-name', token, 'important');
    if (name === 'unrelated-style') video.style.setProperty('color', 'blue', 'important');
    if (name === 'host-remove-existing') video.style.setProperty('anchor-name', token);
    if (name === 'host-empty-style') video.setAttribute('style', '');
    if (name === 'same-value') video.setAttribute(attribute, video.getAttribute(attribute));
    if (name === 'style-recreate') { const value = video.getAttribute(attribute); video.removeAttribute(attribute); video.setAttribute(attribute, value); }
    if (name === 'pending-host') model === 'O1' ? video.style.setProperty('color', 'red') : video.setAttribute(attribute, 'pending-host');
    if (name === 'overflow') for (let index = 0; index < 65; index++) video.style.setProperty('color', index % 2 ? 'red' : 'blue');
    if (name === 'live-variable') { video.style.setProperty('--host-names', `${token}, --host-added`); video.style.setProperty('anchor-name', 'var(--host-names)'); }
    if (name === 'live-escape') video.style.setProperty('anchor-name', token.replace('--a', '--\\61 '));
    if (name === 'exception') throw new Error('intentional ownership scope exception');
  } catch (error) { exception = String(error); }
  const host = { style: video.getAttribute('style'), attribute: video.getAttribute(attribute), properties: properties() };
  const checked = lease.check(), released = lease.release();
  const after = { style: video.getAttribute('style'), attribute: video.getAttribute(attribute), rect: rect(), anchor: getComputedStyle(video).getPropertyValue('anchor-name'), properties: properties() };
  if (hostObserver) { const records = hostObserver.takeRecords(); observerCount += records.length;
    if (name === 'generic-data-observer' && records.some(record => record.attributeName.startsWith('data-'))) sideEffect++;
    hostObserver.disconnect(); }
  const stableLayout = before.rect.every((value, index) => value === during.rect[index] && value === after.rect[index]);
  const unrelated = value => Object.fromEntries(Object.entries(value).filter(([field]) => field !== 'anchor-name'));
  let hostPreserved = JSON.stringify(unrelated(host.properties)) === JSON.stringify(unrelated(after.properties));
  if (model === 'O2' && ['host-replace', 'same-value', 'style-recreate', 'pending-host', 'host-remove-token'].includes(name)) hostPreserved &&= after.attribute === host.attribute;
  if (model === 'O1' && name === 'host-append') hostPreserved &&= after.properties['anchor-name']?.[0] === '--host-added' && after.properties['anchor-name']?.[1] === 'important';
  if (model === 'O1' && name === 'host-replace') hostPreserved &&= after.properties['anchor-name']?.[0] === '--replacement';
  if (model === 'O1' && ['host-priority', 'same-value', 'style-recreate', 'host-remove-existing'].includes(name)) hostPreserved &&= after.properties['anchor-name']?.[0] === 'none';
  if (['absent', 'repeated', 'exception'].includes(name)) hostPreserved &&= after.attribute === before.attribute;
  const initialProperty = before.properties['anchor-name'] ?? ['', ''];
  const expectedProperties = {
    absent: ['', ''], existing: ['--host-one', ''], important: ['--host-one', 'important'],
    'host-append': ['--host-added', 'important'], 'host-replace': ['--replacement', 'important'],
    'host-remove-token': ['', ''], 'host-priority': ['none', 'important'], 'unrelated-style': initialProperty,
    'host-remove-existing': ['none', ''], repeated: initialProperty, collision: initialProperty,
    exception: initialProperty, 'explicit-none': ['none', 'important'], 'host-empty-style': ['', ''],
    'unsupported-grammar': initialProperty, 'pending-host': initialProperty, 'same-value': ['none', ''],
    'style-recreate': ['none', ''], 'synchronous-reaction': initialProperty,
    'live-variable': host.properties['anchor-name'], 'live-escape': host.properties['anchor-name'], overflow: ['', ''],
  };
  const hostContainer = ['host-append', 'host-replace', 'host-remove-token', 'host-priority', 'unrelated-style',
    'host-remove-existing', 'host-empty-style', 'pending-host', 'same-value', 'style-recreate', 'synchronous-reaction',
    'overflow', 'live-variable', 'live-escape'].includes(name);
  const expectedStylePresent = before.style !== null || hostContainer;
  const restorationCorrect = model !== 'O1' || JSON.stringify(after.properties['anchor-name'] ?? ['', '']) === JSON.stringify(expectedProperties[name]) &&
    (after.style !== null) === expectedStylePresent;
  const absent = value => {
    if (value === '' || value === 'none') return true;
    const names = value.split(',').map(name => name.trim());
    if (!names.every(name => /^--[A-Za-z0-9_-]+$/.test(name))) return null;
    return !names.includes(token);
  };
  const independentlyAbsent = { inline: absent(after.properties['anchor-name']?.[0] ?? ''), computed: absent(after.anchor) };
  const ownedAttempt = acquired.active || before.style !== during.style || before.attribute !== during.attribute;
  let outcome = released.outcome;
  if (!stableLayout || sideEffect || !hostPreserved || !restorationCorrect) outcome = 'UNSAFE';
  if (model === 'O1' && ownedAttempt && (independentlyAbsent.inline !== true || independentlyAbsent.computed !== true)) outcome = 'UNSAFE';
  if (!acquired.active && !['collision', 'unsupported-grammar'].includes(name) && outcome !== 'UNSAFE') outcome = 'UNRESOLVED';
  return { model, name, token, before, during, host, after, acquired, checked, released, exception,
    observerCount, sideEffect, stableLayout, hostPreserved, restorationCorrect, independentlyAbsent,
    expectedProperty: model === 'O1' ? expectedProperties[name] : null, expectedStylePresent, outcome,
    boundary: 'native property/attribute release; not universal page ownership proof' };
}

async function fresh(native, service, prepare = 'initial') {
  const page = await native.context.newPage();
  const placement = await nativeWindow(page, native.context); await page.goto(service.url); await page.bringToFront();
  const original = await page.evaluate(name => globalThis.__M109_FIXTURE__.prepare(name), prepare);
  const world = await isolated(native, page, service.source);
  return { page, world, placement, original, close: async () => { await world.close(); await page.close(); } };
}

async function observability(native, service) {
  const results = [];
  for (const entry of SEMANTICS) {
    const context = await fresh(native, service);
    try {
      results.push(await context.page.evaluate(async challenge => {
        const fixture = globalThis.__M109_FIXTURE__, target = challenge.selector.includes('::') ? fixture.player : document.querySelector(challenge.selector);
        if (!CSS.supports(challenge.property, challenge.value)) return { ...challenge, outcome: 'UNRESOLVED', reason: 'CSS declaration unsupported' };
        let mutationCount = 0, resizeCount = 0; const events = [], focus = [];
        const recordEvent = event => events.push(event.type), recordFocus = () => focus.push({ at: performance.now(), focused: document.hasFocus(), visibility: document.visibilityState });
        const names = ['resize', 'loadeddata', 'loadstart', 'emptied', 'seeking', 'seeked', 'playing'];
        const observer = new MutationObserver(records => { mutationCount += records.length; });
        for (const node of [fixture.video, fixture.player, fixture.outer]) observer.observe(node, { attributes: true, childList: true });
        const resize = new ResizeObserver(records => { resizeCount += records.length; }); resize.observe(fixture.video); resize.observe(fixture.player);
        for (const name of names) fixture.video.addEventListener(name, recordEvent);
        for (const name of ['focus', 'blur', 'visibilitychange']) addEventListener(name, recordFocus);
        await fixture.settled(); mutationCount = resizeCount = 0; events.length = 0; recordFocus();
        const computed = () => getComputedStyle(target, challenge.selector.includes('::') ? '::after' : null).getPropertyValue(challenge.property);
        const before = { rect: fixture.rect(fixture.video), value: computed() };
        await fixture.action('cssom', challenge); await fixture.settled(); await new Promise(done => setTimeout(done, 120));
        mutationCount += observer.takeRecords().length; recordFocus();
        const after = { rect: fixture.rect(fixture.video), value: computed() };
        observer.disconnect(); resize.disconnect();
        for (const name of names) fixture.video.removeEventListener(name, recordEvent);
        for (const name of ['focus', 'blur', 'visibilitychange']) removeEventListener(name, recordFocus);
        return { ...challenge, before, after, mutationCount, resizeCount, events, focus,
          rectangleUnchanged: JSON.stringify(before.rect) === JSON.stringify(after.rect), valueChanged: before.value !== after.value };
      }, entry.actions[0].payload));
    } finally { await context.close(); }
  }
  return results;
}

async function anchorReuse(native, service) {
  const context = await fresh(native, service, 'existing-anchor');
  try {
    const before = await context.page.evaluate(() => globalThis.__M109_FIXTURE__.host());
    const observations = [];
    await context.world.call(() => {
      const video = document.querySelector('video'), canvas = document.createElement('canvas'); canvas.id = 'm109-anchor-measure';
      for (const [field, value] of Object.entries({ position: 'fixed', 'position-anchor': '--page-existing', left: 'anchor(left)', top: 'anchor(top)', width: 'anchor-size(width)', height: 'anchor-size(height)', 'pointer-events': 'none', visibility: 'hidden' })) canvas.style.setProperty(field, value);
      video.after(canvas);
    });
    for (const name of ['noop', 'preceding-spacer', 'page-228', 'css-resize']) {
      await context.page.evaluate(name => globalThis.__M109_FIXTURE__.action(name), name);
      await context.page.evaluate(() => globalThis.__M109_FIXTURE__.settled());
      observations.push(await context.world.call(() => { const video = document.querySelector('video'), canvas = document.getElementById('m109-anchor-measure');
        const rect = node => { const value = node.getBoundingClientRect(); return [value.left, value.top, value.width, value.height]; };
        return { video: rect(video), canvas: rect(canvas), anchor: getComputedStyle(video).getPropertyValue('anchor-name') }; }));
    }
    await context.world.call(() => document.getElementById('m109-anchor-measure').remove());
    const after = await context.page.evaluate(() => globalThis.__M109_FIXTURE__.host());
    return { before, after, observations, attributesUnchanged: JSON.stringify(before.attributes) === JSON.stringify(after.attributes),
      matches: observations.every(row => row.video.every((value, index) => Math.abs(value - row.canvas[index]) <= .5)), scope: 'Optional zero-host-mutation measuring box, not replacement qualification' };
  } finally { await context.close(); }
}

async function common(native, service, prefix, kind = 'S1') {
  const results = []; let stopped = null;
  const candidate = kind !== 'S0';
  const repeats = kind === 'S0' ? 1 : 3;
  outer: for (let repeat = 1; repeat <= repeats; repeat++) for (const entry of CASES) {
    let context; const result = { name: entry.name, repeat, actions: [], crops: [], controls: [] };
    results.push(result);
    try {
      context = await fresh(native, service, entry.name); result.placement = context.placement; result.original = context.original;
      const challenge = entry.actions.find(action => action.name === 'cssom')?.payload;
      if (challenge && !await context.page.evaluate(challenge => CSS.supports(challenge.property, challenge.value), challenge)) {
        result.notApplicable = true; result.execution = 'NOT RUN: CSS declaration unimplemented';
        result.outcome = 'UNSUPPORTED_SAFE'; result.supportCredit = false;
        if (['cssom-fit', 'cssom-position'].includes(entry.name)) { result.outcome = 'UNRESOLVED'; throw new Error('Required fitted-image capability unavailable'); }
        console.log(JSON.stringify({ phase: 'common', kind, name: entry.name, repeat, execution: result.execution }));
        continue;
      }
      await context.page.evaluate(installSampler);
      if (kind === 'S0') await context.world.call(startBroadControl);
      else await context.world.call(async () => { globalThis.__M109_MONITOR__ = await M109.startMonitor(document.querySelector('video') ?? document.querySelector('#outer').lastElementChild.shadowRoot.querySelector('video')); });
      await context.page.evaluate(() => globalThis.__M109_SAMPLER__.start());
      const wait = async (expected, startedAt) => {
        await context.page.waitForFunction(({ expected, startedAt }) => {
          const status = globalThis.__M109_SAMPLER__.status(), row = status.latest;
          return !!status.firstFailure || status.interruptions.length || row && row.at >= startedAt && ['render', 'submission'].includes(row.boundary) &&
            (expected === 'SUPPORTED' ? row.visible && row.geometry && row.fit && row.backingCorrect && row.host : !row.visible && row.host);
        }, { expected, startedAt }, { timeout: 3000 });
        const status = await context.page.evaluate(() => globalThis.__M109_SAMPLER__.status());
        if (status.firstFailure) { result.outcome = 'UNSAFE'; throw new Error('Independent presentation predicate violated'); }
        if (status.interruptions.length) { result.invalid = 'external-focus-or-visibility'; throw new Error('Interrupted native observation'); }
        assert(status.latest.at >= startedAt && status.latest.host, 'Missing fresh independent boundary evidence');
        return status;
      };
      const admittedAt = await context.page.evaluate(() => performance.now());
      result.admissionOracle = await wait(entry.name === 'initially-unsupported' ? 'UNSUPPORTED' : 'SUPPORTED', admittedAt);
      result.admission = await context.world.call(() => globalThis.__M109_MONITOR__.snapshot());
      result.admittedHost = await context.page.evaluate(() => globalThis.__M109_FIXTURE__.host());
      if (JSON.stringify(result.admittedHost.attributes) !== JSON.stringify(result.original.attributes) ||
        !Object.keys(result.original.rect).every(key => Math.abs(result.original.rect[key] - result.admittedHost.rect[key]) <= .5) || !result.admittedHost.parent) {
        result.outcome = 'UNSAFE'; throw new Error('Owned output insertion changed host state');
      }
      const visual = async label => {
        const proof = await paintedProof(context, `${prefix}.${repeat}-${entry.name}.${label}`); result.crops.push(proof);
        if (proof.verdict !== 'SUPPORTED_CORRECT') { result.outcome = proof.verdict; throw new Error('Required painted proof did not pass'); }
        const controls = await controlProof(context); result.controls.push(controls);
        if (!controls.activated || !controls.hit || controls.pointerEvents !== 'none' || JSON.stringify(controls.focus) !== JSON.stringify(result.original.focus)) {
          result.outcome = 'UNSAFE'; throw new Error('Control/caption/focus preservation failed');
        }
      };
      if (candidate && ['initial', 'cover', 'rounded-video'].includes(entry.name)) await visual('admission');
      for (const action of entry.actions) {
        const actionResult = { ...action, startedAt: await context.page.evaluate(() => performance.now()),
          before: await context.world.call(() => globalThis.__M109_MONITOR__.snapshot()) }; result.actions.push(actionResult);
        if (action.name === 'fullscreen') { await context.page.locator('#fullscreen').click(); await context.page.waitForFunction(() => !!document.fullscreenElement); }
        else if (action.name === 'exit') await context.page.evaluate(() => document.exitFullscreen());
        else await context.page.evaluate(action => globalThis.__M109_FIXTURE__.action(action.name, action.payload), action);
        const expected = await context.page.evaluate(() => globalThis.__M109_FIXTURE__.state.expected);
        actionResult.expected = expected; actionResult.oracle = await wait(expected, actionResult.startedAt);
        actionResult.status = await context.world.call(() => globalThis.__M109_MONITOR__.snapshot());
        actionResult.outcome = expected === 'SUPPORTED' ? 'SUPPORTED_CORRECT' : 'UNSUPPORTED_SAFE';
        if (expected === 'UNSUPPORTED') {
          await context.page.evaluate(() => new Promise(done => setTimeout(done, 120)));
          actionResult.stability = await context.world.call(() => globalThis.__M109_MONITOR__.snapshot());
          assert.equal(actionResult.stability.reason, actionResult.status.reason, 'Rejection reason changed while input was stable');
          assert.equal(actionResult.stability.proofCalls, actionResult.status.proofCalls, 'Stable rejection repeated full proofs');
          assert.equal(actionResult.stability.initializes, 1, 'Repeated GPU initialization');
        }
        if (candidate && expected === 'SUPPORTED' && ['cssom-fit', 'cssom-position', 'radius', 'nested-scroll', 'css-resize', 'viewport-clip', 'abr', 'source-scroll', 'controls-z-index'].includes(entry.name)) await visual(`action-${result.actions.length}`);
        if (entry.name === 'controls-z-index' && expected === 'UNSUPPORTED') {
          const controls = await controlProof(context); result.controls.push(controls); assert(controls.activated && controls.hit);
        }
      }
      result.outcome = 'SUPPORTED_CORRECT';
    } catch (error) {
      result.error = String(error); result.outcome ??= 'UNRESOLVED';
    } finally {
      if (context) {
        let trace;
        try { trace = await context.page.evaluate(() => globalThis.__M109_SAMPLER__?.stop()); } catch (error) { result.traceError = String(error); }
        try { result.cleanup = await context.world.call(() => globalThis.__M109_MONITOR__?.dispose() ?? null); } catch (error) { result.cleanupError = String(error); }
        if (trace) {
          result.firstFailure = trace.firstFailure; result.interruptions = trace.interruptions;
          if (trace.firstFailure) result.outcome = 'UNSAFE';
          if (trace.interruptions.length) { result.invalid = 'external-focus-or-visibility'; if (result.outcome !== 'UNSAFE') result.outcome = 'UNRESOLVED'; }
          if (candidate && result.cleanup) { result.recovery = validateReveals(trace, result.cleanup, result.actions);
            if (result.outcome === 'SUPPORTED_CORRECT' && result.recovery.verdict !== 'SUPPORTED_CORRECT') result.outcome = result.recovery.verdict; }
          result.raw = raw(`${prefix}.${repeat}-${entry.name}.json.gz`, { ...trace, telemetry: result.cleanup });
        } else if (!result.notApplicable) { result.outcome = 'UNRESOLVED'; result.traceError ??= 'Missing final independent trace'; }
        if (!result.notApplicable && (result.cleanup?.error || !result.cleanup || Object.values(result.cleanup.resources).some(value => value !== 0))) result.outcome = 'UNSAFE';
        result.after = await context.page.evaluate(() => globalThis.__M109_FIXTURE__.host());
        if (result.cleanup) { delete result.cleanup.guards; delete result.cleanup.proofs; delete result.cleanup.submissions; }
        await context.close();
      }
    }
    console.log(JSON.stringify({ phase: 'common', kind, name: entry.name, repeat, outcome: result.outcome, reason: result.error ?? null }));
    if (result.outcome !== 'SUPPORTED_CORRECT') { stopped = { name: entry.name, repeat, outcome: result.outcome, invalid: result.invalid ?? null }; break outer; }
  }
  return { contract: kind, ownership: 'O0', cases: CASES, repeats, results, stopped, remaining: CASES.length * repeats - results.length };
}

async function census(native, source) {
  const rows = [];
  for (const [name, url] of [['shaka', 'https://shaka-project.github.io/shaka-player-release/demo/'],
    ['videojs', 'https://videojs.org/'], ['plyr', 'https://plyr.io/']]) {
    const row = { name, url, scope: 'Nonbinding current-page admission only; no replacement or public pixel capture', started: new Date().toISOString() };
    rows.push(row);
    const page = await native.context.newPage(); let world;
    try {
      row.placement = await nativeWindow(page, native.context);
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 10000 }); await page.bringToFront();
      world = await isolated(native, page, source);
      row.result = await world.call(async () => {
        const videos = [...document.querySelectorAll('video')].slice(0, 4), originalPaused = videos.map(video => video.paused);
        const readers = videos.map(video => M109.createContractReader(video));
        const play = [];
        try {
          play.push(...await Promise.all(videos.map(async video => {
            if (video.mediaKeys || !video.currentSrc) return 'not attempted: no clear current source';
            return Promise.race([video.play().then(() => 'playing', () => 'native play rejected'), new Promise(done => setTimeout(() => done('play unresolved within 1s'), 1000))]);
          })));
          const samples = [];
          for (let index = 0; index < 3; index++) {
            samples.push({ at: performance.now(), focused: document.hasFocus(), visibility: document.visibilityState,
              videos: readers.map((read, index) => { const input = read(), admission = M109.assessContract(input);
                return { index, admitted: admission.outcome === 'SUPPORTED', reason: admission.reason, ready: input.ready,
                  playing: input.playing, dimensions: [input.source.width, input.source.height], rect: input.rect }; }),
              ownedCanvases: document.querySelectorAll('canvas[data-aethervsr-m109],canvas[data-aethervsr-m10]').length });
            if (index < 2) await new Promise(done => setTimeout(done, 300));
          }
          return { videoCount: videos.length, inspectedAtMost: 4, play, samples,
            stableReasons: samples.every(sample => JSON.stringify(sample.videos.map(video => [video.admitted, video.reason])) === JSON.stringify(samples[0].videos.map(video => [video.admitted, video.reason]))),
            noOwnedStaleCanvas: samples.every(sample => sample.ownedCanvases === 0),
            limitation: videos.length ? 'Only these current short-lived media states, not loaded-player compatibility or public qualification' : 'No top-document video observed; no structural compatibility conclusion' };
        } finally { videos.forEach((video, index) => { if (originalPaused[index]) video.pause(); }); }
      });
    } catch (error) { row.error = String(error).slice(0, 1000); row.outcome = 'UNRESOLVED'; }
    finally { await world?.close(); await page.close(); row.finished = new Date().toISOString(); }
  }
  return rows;
}

export async function runDefaultPolicyCensus(prefix) {
  prefix = resolve(prefix); assert(prefix.startsWith(join(ROOT, '.cache/m109/')) && !existsSync(`${prefix}.json`));
  const report = { schemaVersion: 1, phase: 'DEFAULT_MEDIA_POLICY_ADMISSION_CENSUS', identity: identity(),
    started: new Date().toISOString(), requestedBrowserArgs: [], createsReplacement: false,
    scope: 'Nonbinding generic current-video admission, no public pixel capture or contract retuning' };
  const bundle = await build({ entryPoints: [join(ROOT, 'tools/m109-contract.ts')], bundle: true, write: false, format: 'iife', globalName: 'M109' });
  let native;
  try {
    native = await openNativeChrome();
    report.browser = { version: await native.browser.version(), executableSha256: sha256(readFileSync(native.executable)) };
    report.results = await census(native, bundle.outputFiles[0].text);
    assert.deepEqual(identity(), report.identity);
  } catch (error) { report.error = String(error); }
  finally { await native?.close(); report.finished = new Date().toISOString(); writeFileSync(`${prefix}.json`, JSON.stringify(report, null, 2), { flag: 'wx' }); }
  return report;
}

export function validateOwnershipResume(prior, current) {
  assert(prior.error?.includes("reading 'define'"), 'Only the recorded host-registry apparatus interruption is resumable here');
  assert.equal(prior.results.O1.results.length, 18);
  assert.deepEqual(prior.results.O1.results.map(row => row.name), OWNERSHIP_CASES.O1.slice(0, 18));
  assert(prior.results.O1.results.every(row => ['SUPPORTED_CORRECT', 'UNSUPPORTED_SAFE'].includes(row.outcome)));
  assert.equal(prior.results.O1.stopped, null);
  assert.equal(prior.results.common, undefined); assert.equal(prior.results.O2, undefined);
  for (const path of ['tools/m109-contract.ts', 'tools/m109-monitor.ts', 'tools/m109-submission.ts', 'tools/m109-ownership.ts']) {
    assert.equal(prior.identity.pins[path], current.pins[path], `Resumption must not retune ${path}`);
  }
  return structuredClone(prior.results);
}

export async function runStudy(prefix, resumePath = null) {
  prefix = resolve(prefix); assert(prefix.startsWith(join(ROOT, '.cache/m109/')) && !existsSync(`${prefix}.json`)); mkdirSync(dirname(prefix), { recursive: true });
  const report = { schemaVersion: 1, identity: identity(), environment: presentationEnvironment(), started: new Date().toISOString(), results: {} };
  if (resumePath) {
    resumePath = resolve(resumePath); assert(resumePath.startsWith(join(ROOT, '.cache/m109/')));
    const bytes = readFileSync(resumePath), prior = JSON.parse(bytes);
    report.results = validateOwnershipResume(prior, report.identity);
    report.retained = { path: relative(ROOT, resumePath), bytes: bytes.length, sha256: sha256(bytes),
      identity: prior.identity, started: prior.started, finished: prior.finished,
      sections: ['observability', 'existingAnchor', 'O1.results[0:18]'], reason: 'Only host custom-element setup moved into the fixture MAIN script; no privileged bridge or candidate retuning' };
  }
  let service, native;
  try {
    service = await server(); report.media = service.media; report.bundleSha256 = service.bundleSha256;
    native = await openNativeChrome(['--autoplay-policy=no-user-gesture-required']);
    report.browser = { version: await native.browser.version(), executableSha256: sha256(readFileSync(native.executable)) };
    report.results.observability ??= await observability(native, service);
    report.results.existingAnchor ??= await anchorReuse(native, service);
    for (const model of ['O1', 'O2']) {
      report.results[model] ??= { results: [], stopped: null, remaining: OWNERSHIP_CASES[model].length };
      const results = report.results[model].results;
      if (model === 'O1' && !report.results.existingAnchor.matches) { report.results[model].reason = 'No demonstrated browser-coupled value'; continue; }
      for (const name of OWNERSHIP_CASES[model].slice(results.length)) {
        const context = await fresh(native, service, name === 'synchronous-reaction' ? name : 'initial');
        try {
          const result = await context.world.call(ownershipCase, { model, name }); results.push(result);
          report.results[model].remaining--;
          console.log(JSON.stringify({ phase: 'ownership', model, name, outcome: result.outcome }));
          if (['UNSAFE', 'UNRESOLVED'].includes(result.outcome)) { report.results[model].stopped = { name, outcome: result.outcome }; break; }
        } finally { await context.close(); }
      }
    }
    report.results.control = await common(native, service, `${prefix}.S0`, 'S0');
    report.results.common = await common(native, service, `${prefix}.S1`);
    report.costs = report.results.common.stopped ? 'NOT RUN: no complete safety survivor' : 'ELIGIBILITY_REVIEW_REQUIRED';
    report.results.census = await census(native, service.source);
    assert.deepEqual(identity(), report.identity);
  } catch (error) { report.error = String(error); }
  finally {
    await native?.close(); await service?.close(); report.finished = new Date().toISOString();
    writeFileSync(`${prefix}.json`, JSON.stringify(report, null, 2), { flag: 'wx' });
  }
  return report;
}

export function validateFiniteRevision(prior, current) {
  assert.equal(prior.error, undefined);
  assert.equal(prior.results.common.results.length, 1);
  assert.equal(prior.results.common.stopped.name, 'initial');
  assert.equal(prior.results.common.stopped.outcome, 'UNRESOLVED');
  assert.equal(prior.results.common.results[0].cleanup.proofCalls, 0);
  assert.equal(prior.results.common.results[0].cleanup.transitions[0].reason, 'unsupported-corner');
  assert.equal(prior.results.common.results[0].firstFailure, null);
  for (const path of ['tools/m109-monitor.ts', 'tools/m109-submission.ts', 'tools/m109-ownership.ts']) {
    assert.equal(prior.identity.pins[path], current.pins[path], `Revision cannot retune ${path}`);
  }
  assert.equal(prior.results.O1.stopped.outcome, 'UNSAFE');
  assert.equal(prior.results.O2.stopped.outcome, 'UNSAFE');
}

export async function runFiniteRevision(prefix, priorPath, apparatusPriorPath = null) {
  prefix = resolve(prefix); priorPath = resolve(priorPath);
  assert(prefix.startsWith(join(ROOT, '.cache/m109/')) && !existsSync(`${prefix}.json`));
  assert(priorPath.startsWith(join(ROOT, '.cache/m109/')));
  const current = identity(), bytes = readFileSync(priorPath), prior = JSON.parse(bytes);
  validateFiniteRevision(prior, current);
  const report = { schemaVersion: 1, phase: 'S1-R1_SINGLE_FINITE_REVISION', identity: current,
    environment: presentationEnvironment(), started: new Date().toISOString(),
    retained: { path: relative(ROOT, priorPath), sha256: sha256(bytes), bytes: bytes.length, identity: prior.identity,
      sections: ['observability', 'existingAnchor', 'O1', 'O2', 'control', 'common(original S1)', 'census(original S1)'],
      policy: 'Original results are retained, never reclassified or pooled with S1-R1' }, results: {} };
  if (apparatusPriorPath) {
    const path = resolve(apparatusPriorPath); assert(path.startsWith(join(ROOT, '.cache/m109/')));
    const bytes = readFileSync(path), previous = JSON.parse(bytes), failed = previous.results.common.results[0];
    const latest = previous.results.common.results.at(-1);
    assert.equal(latest.firstFailure, null);
    if (previous.results.common.results.length === 1) {
      assert.equal(failed.name, 'initial'); assert(failed.crops[0].regions.every(region => region.wrong === 0));
      assert(failed.crops[0].regions.some(region => region.wrongOriginal > 0));
    } else {
      assert.equal(previous.results.common.results.length, 5); assert.equal(latest.name, 'nested-scroll');
      assert(previous.results.common.results.slice(0, 4).every(row => row.outcome === 'SUPPORTED_CORRECT'));
      const regions = latest.crops[0].regions.filter(region => region.status !== 'NOT VISIBLE');
      assert(regions.every(region => region.wrong === 0 && region.wrongOriginal === 0));
      assert.deepEqual(regions.filter(region => region.verdict !== 'SUPPORTED_CORRECT').map(region => region.name), ['control']);
      assert(regions.find(region => region.name === 'control').tested >= 100);
    }
    for (const source of ['tools/m109-contract.ts', 'tools/m109-monitor.ts', 'tools/m109-submission.ts', 'tools/m109-ownership.ts']) {
      assert.equal(previous.identity.pins[source], current.pins[source], 'Oracle correction cannot revise any candidate');
    }
    report.supersededOracle = { path: relative(ROOT, path), bytes: bytes.length, sha256: sha256(bytes), identity: previous.identity,
      reason: previous.results.common.results.length === 1
        ? 'Old screenshot decoder discarded ICC and compared video color-managed pixels/glyphs against inappropriate absolute-color masks; paired differences were <=1. Candidate unchanged; fresh affected evidence required.'
        : 'Nested-scroll retained only a uniform visible control strip; the texture-count guard incorrectly required clipped-away text. All paired/color differences passed; candidate unchanged and fresh affected evidence required.' };
  }
  let service, native;
  try {
    service = await server(); report.media = service.media; report.bundleSha256 = service.bundleSha256;
    native = await openNativeChrome(['--autoplay-policy=no-user-gesture-required']);
    report.browser = { version: await native.browser.version(), executableSha256: sha256(readFileSync(native.executable)) };
    report.results.common = await common(native, service, `${prefix}.S1-R1`, 'S1-R1');
    report.costs = report.results.common.stopped ? 'NOT RUN: no complete safety survivor' : 'ELIGIBILITY_REVIEW_REQUIRED';
    report.results.census = await census(native, service.source);
    assert.deepEqual(identity(), report.identity);
  } catch (error) { report.error = String(error); }
  finally {
    await native?.close(); await service?.close(); report.finished = new Date().toISOString();
    writeFileSync(`${prefix}.json`, JSON.stringify(report, null, 2), { flag: 'wx' });
  }
  return report;
}

export function exportEvidence(destination = null) {
  const names = ['frozen-controls', 'pre-capture-packages', 'study-01', 'study-02', 'default-serialization-01',
    'revision-01', 'oracle-color-audit', 'revision-02', 'revision-03', 'census-default-01'];
  const artifacts = names.map(id => {
    const path = `.cache/m109/${id}.json`, bytes = readFileSync(join(ROOT, path));
    return { id, path, bytes: bytes.length, sha256: sha256(bytes), content: JSON.parse(bytes) };
  });
  const references = new Map();
  const verify = value => {
    if (!value || typeof value !== 'object') return;
    if (typeof value.path === 'string' && typeof value.sha256 === 'string' && Number.isInteger(value.bytes)) {
      const path = resolve(ROOT, value.path); assert(path.startsWith(join(ROOT, '.cache/m109/')));
      const bytes = readFileSync(path); assert.equal(bytes.length, value.bytes); assert.equal(sha256(bytes), value.sha256);
      references.set(relative(ROOT, path), { path: relative(ROOT, path), bytes: bytes.length, sha256: value.sha256 });
    }
    for (const child of Object.values(value)) verify(child);
  };
  for (const artifact of artifacts) verify(artifact.content);
  const content = id => artifacts.find(artifact => artifact.id === id).content;
  const final = content('revision-03').results.common, ownership = content('study-02').results;
  assert.equal(final.results.length, 20); assert.equal(final.remaining, 136);
  assert.equal(final.results.filter(row => row.outcome === 'SUPPORTED_CORRECT').length, 19);
  assert.equal(final.stopped.name, 'fullscreen-scrolled-enter-exit'); assert.equal(final.stopped.outcome, 'UNRESOLVED');
  assert.equal(ownership.O1.stopped.outcome, 'UNSAFE'); assert.equal(ownership.O2.stopped.outcome, 'UNSAFE');
  const nativeAudit = [];
  for (const artifact of artifacts) {
    for (const common of [artifact.content.results?.control, artifact.content.results?.common].filter(Boolean)) {
      for (const row of common.results) {
        if (!row.raw) continue;
        const trace = JSON.parse(gunzipSync(readFileSync(join(ROOT, row.raw.path))));
        assert.deepEqual(trace.firstFailure, row.firstFailure);
        assert.deepEqual(trace.telemetry.resources, row.cleanup.resources);
        if (row.recovery) assert.deepEqual(validateReveals(trace, trace.telemetry, row.actions), row.recovery);
        nativeAudit.push({ artifact: artifact.id, contract: common.contract, name: row.name, repeat: row.repeat,
          rows: trace.rows.length, submissions: trace.telemetry.submissions.length,
          validRecoverySubmissions: trace.telemetry.submissions.filter(record => record.validForRecovery).length,
          interruptions: trace.interruptions.length, resources: trace.telemetry.resources,
          recovery: row.recovery ?? null, firstFailure: row.firstFailure });
      }
    }
  }
  assert.deepEqual(content('census-default-01').requestedBrowserArgs, []);
  const report = { schemaVersion: 1, selection: 'NO USEFUL OBSERVABLE CONTRACT QUALIFIED',
    production: 'UNCHANGED S1', extension: 'PARTIAL', m11: 'GATED',
    costs: { execution: 'NOT RUN', metrics: 'not measured', reason: 'No complete useful safety/ownership survivor' },
    finalContract: { name: 'S1-R1', planned: 156, observed: 20, preliminaryCorrect: 19, unresolvedRequired: 1, notRun: 136,
      reason: 'Required fullscreen remained safely hidden; modal predicate also rejects fullscreen; not proof of unobservability' },
    ownership: { O0: 'Only observed restoration prefixes; not generically qualified',
      O1: { observed: 20, planned: 22, notRun: 2, outcome: 'UNSAFE', stop: 'history-overflow with owned UUID retained' },
      O2: { observed: 11, planned: 12, notRun: 1, outcome: 'UNSAFE', stop: 'host attribute selector changed video width' } },
    limitations: ['Synchronous queue-submit returned is not GPU completion or physical scanout.',
      'S0 uses unchanged VideoAttachment source as an isolated broad control, not a full packaged-extension acceptance run.',
      'Safe rejection is not unsafe exposure or supported compatibility.',
      'The bounded stopped prototype does not prove that a useful browser contract is impossible.',
      'Original S1 and superseded oracle runs remain unpooled; source identities differ explicitly.',
      'Earlier census results used a fixture autoplay flag; only census-default-01 has default media policy.',
      'Unready or playback-denied public states cannot establish brand-level compatibility.',
      'Raw paths/hashes establish local retained evidence identity, not remote availability.'],
    artifacts, references: [...references.values()], nativeAudit };
  const bytes = Buffer.from(`${JSON.stringify(report)}\n`);
  if (destination) {
    const tree = execFileSync('git', ['ls-tree', '-rl', 'HEAD'], { cwd: ROOT, encoding: 'utf8', maxBuffer: 16 * 1024 ** 2 });
    const tracked = tree.trim().split('\n').reduce((sum, line) => sum + Number(line.split('\t')[0].trim().split(/\s+/)[3]), 0);
    assert(tracked + bytes.length + 65000 < 53477376, 'Compact evidence plus documentation reserve exceeds owner-approved cap');
    writeFileSync(resolve(ROOT, destination), bytes, { flag: 'wx' });
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) {
  assert([3, 4].includes(process.argv.length), 'Usage: node tools/m109-study.mjs .cache/m109/study-02 [.cache/m109/study-01.json]');
  const report = await runStudy(process.argv[2], process.argv[3]);
  console.log(JSON.stringify({ saved: `${process.argv[2]}.json`, error: report.error ?? null, common: report.results.common?.stopped, costs: report.costs }));
  if (report.error) process.exitCode = 1;
}