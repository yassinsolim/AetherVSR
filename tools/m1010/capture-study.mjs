import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { ROOT, sha256 } from '../m10-fixtures.mjs';
import { nativeWindow } from '../m105-accounting.mjs';
import { studyIdentity, openResearch } from './native.mjs';
import { startFixtures } from './fixtures.mjs';
import { openCheckpoint } from './checkpoint.mjs';
import { observePlayback, inputFrameEvidence } from './acquisition.mjs';
import { environment, waitForObserved, INPUT_TIMES } from './study.mjs';

export const TAB_FIDELITY_CASES = [
  { id: 'R3-720-native', asset: 'A', scale: 1, offscreen: false },
  { id: 'R3-720-small', asset: 'A', scale: 0.5, offscreen: false },
  { id: 'R3-720-large', asset: 'A', scale: 1.5, offscreen: false },
  { id: 'R3-720-offscreen', asset: 'A', scale: 1, offscreen: true },
  { id: 'R3-1080-small', asset: 'C', scale: 0.5, offscreen: false },
];
export const SELF_CAPTURE_CASES = [
  { id: 'R6-cancel', expected: 'rejection', instruction: 'Click Choose display (manual), then Cancel in Chrome\'s chooser.' },
  { id: 'R6-wrong', expected: 'wrong', instruction: 'Click Choose display (manual). Choose the tab named M10.10 WRONG fixture, then Share. The runner will identify it and stop it.' },
  { id: 'R6-current', expected: 'source', method: 'crop', instruction: 'Click Choose display (manual). Choose this tab, M10.10 INTENDED fixture, then Share.' },
  { id: 'R6-repeat', expected: 'source', method: 'restrict', instruction: 'Repeat: click Choose display (manual), choose this M10.10 INTENDED fixture tab again, then Share.' },
];
export const SELF_SCOPE_CASES = ['baseline', 'movement', 'resize', 'scroll', 'occluder', 'controls-hidden', 'ABR-low', 'ABR-high'];

export function selfChoiceOutcome(expected, observed) {
  if (expected === 'rejection') return observed.captureError ? 'REJECTION_RECORDED' : 'UNRESOLVED';
  return observed.identity === expected ? 'IDENTITY_RECORDED' : 'UNRESOLVED';
}

export function seekDiagnosticFrame({ time, expected }) {
  const video = document.querySelector('video');
  if (!video?.paused || !video.requestVideoFrameCallback || !Number.isFinite(time) ||
    !Array.isArray(expected) || expected.length !== 2 || !expected.every(value => Number.isInteger(value) && value > 0)) throw new Error('Invalid paused diagnostic seek');
  return new Promise((done, reject) => {
    let metadata, sought = false, callback, discarded = 0;
    const finish = error => {
      if (!error && (!metadata || !sought)) return;
      clearTimeout(timer); video.cancelVideoFrameCallback(callback);
      video.removeEventListener('seeked', seeked); video.removeEventListener('error', failed);
      if (error) reject(error);
      else if (!video.paused || video.seeking || video.readyState < 2 || Math.abs(video.currentTime - time) > 1e-6) reject(new Error('Unstable diagnostic seek'));
      else done({ requestedTime: time, metadata, expected, discarded });
    };
    const seeked = () => { sought = true; finish(); }, failed = () => finish(new Error('Diagnostic media error'));
    const timer = setTimeout(() => finish(new Error('Diagnostic seek deadline')), 5000);
    video.addEventListener('seeked', seeked); video.addEventListener('error', failed);
    const sample = (at, value) => {
      if (!Number.isFinite(value.mediaTime) || Math.abs(value.mediaTime - time) > 1 / 60 || value.width !== expected[0] || value.height !== expected[1]) {
        discarded++; callback = video.requestVideoFrameCallback(sample); return;
      }
      metadata = { at, mediaTime: value.mediaTime, width: value.width, height: value.height, presentedFrames: value.presentedFrames }; finish();
    };
    callback = video.requestVideoFrameCallback(sample);
    try { video.currentTime = time; } catch (error) { finish(error); }
  });
}

export function captureConsentResult(events) {
  const event = events.find(value => ['capture-permission', 'permission-error'].includes(value.type));
  if (!event) return null;
  return { granted: event.type === 'capture-permission' && event.value === true, event };
}

export function terminalizeTabSuffix(checkpoint, reason) {
  for (const entry of TAB_FIDELITY_CASES) if (!checkpoint.has(entry.id)) {
    checkpoint.begin(entry.id); checkpoint.complete(entry.id, { outcome: 'UNRESOLVED', executionStatus: 'NOT_RUN', reason });
  }
}

export function stopAfterUnsafeSelf(checkpoint) {
  if (!SELF_CAPTURE_CASES.some(entry => checkpoint.has(entry.id) && checkpoint.read(entry.id).outcome === 'UNSAFE')) return false;
  for (const entry of SELF_CAPTURE_CASES) if (!checkpoint.has(entry.id)) {
    checkpoint.begin(entry.id); checkpoint.complete(entry.id, { outcome: 'UNRESOLVED', executionStatus: 'NOT_RUN', reason: 'Retained unsafe teardown stopped the primitive study' });
  }
  return true;
}

export function diagnosticCrop({ rect, viewport, captured }) {
  const [width, height] = captured, [innerWidth, innerHeight] = viewport;
  assert([width, height, innerWidth, innerHeight, rect.width, rect.height].every(value => Number.isFinite(value) && value > 0));
  assert([rect.x, rect.y].every(Number.isFinite));
  const scale = [width / innerWidth, height / innerHeight];
  const left = Math.max(0, Math.round(rect.x * scale[0]));
  const top = Math.max(0, Math.round(rect.y * scale[1]));
  const right = Math.min(width, Math.round((rect.x + rect.width) * scale[0]));
  const bottom = Math.min(height, Math.round((rect.y + rect.height) * scale[1]));
  assert(right > left && bottom > top, 'Source outside captured viewport');
  return { scale, crop: [left, top, right - left, bottom - top],
    sourceFraction: [(left / scale[0] - rect.x) / rect.width, (top / scale[1] - rect.y) / rect.height,
      (right - left) / scale[0] / rect.width, (bottom - top) / scale[1] / rect.height],
    fullyVisible: rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= innerWidth && rect.y + rect.height <= innerHeight,
    scope: 'Inferred full-viewport mapping for offline diagnosis only; not independently detected geometry or a product crop contract' };
}

export function installCaptureMarker({ identity, title }) {
  if (!/^[0-9a-f]{8}$/.test(identity)) throw new Error('Invalid capture marker');
  document.getElementById('capture-identity')?.remove();
  const canvas = document.createElement('canvas'); canvas.id = 'capture-identity'; canvas.width = 408; canvas.height = 48;
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:408px;height:48px;z-index:2147483647;image-rendering:pixelated';
  const context = canvas.getContext('2d'), value = Number.parseInt(identity, 16);
  for (let cell = 0; cell < 34; cell++) for (let row = 0; row < 2; row++) {
    const white = cell === 0 || (cell >= 2 && (Math.floor(value / 2 ** (cell - 2)) % 2 === 1) !== !!row);
    context.fillStyle = white ? '#ffffff' : '#000000'; context.fillRect(cell * 12, row * 24, 12, 24);
  }
  document.body.append(canvas); document.title = title;
  return { identity, title, viewport: [innerWidth, innerHeight], dpr: devicePixelRatio, marker: [0, 0, 408, 48] };
}

export function decodeCaptureMarker(bytes, width, height, viewport) {
  assert.equal(bytes.length, width * height * 4);
  assert([width, height, ...viewport].every(value => Number.isFinite(value) && value > 0));
  const level = (cell, row) => {
    const horizontal = Math.floor((cell * 12 + 6) * width / viewport[0]);
    const vertical = Math.floor((row * 24 + 12) * height / viewport[1]);
    assert(horizontal < width && vertical < height, 'Marker outside captured image');
    const offset = (vertical * width + horizontal) * 4, rgb = [0, 1, 2].map(channel => bytes[offset + channel]);
    if (rgb.every(value => value > 220)) return 1;
    if (rgb.every(value => value < 35)) return 0;
    throw new Error('Ambiguous capture marker');
  };
  assert(level(0, 0) === 1 && level(0, 1) === 1 && level(1, 0) === 0 && level(1, 1) === 0, 'Capture marker guards missing');
  let identity = 0;
  for (let bit = 0; bit < 32; bit++) {
    const top = level(bit + 2, 0), bottom = level(bit + 2, 1);
    assert(top !== bottom, 'Capture marker complement mismatch'); identity += top * 2 ** bit;
  }
  return identity.toString(16).padStart(8, '0');
}

export async function displayFrameEvidence({ generation }) {
  const api = globalThis.__M1010_SOURCE__, stream = api.displayForGeneration(generation);
  if (!stream?.getVideoTracks().some(track => track.readyState === 'live')) throw new Error('No live display track');
  const receiver = document.createElement('video'); receiver.muted = true; receiver.playsInline = true; receiver.srcObject = stream;
  receiver.style.cssText = 'position:fixed;left:-4096px;width:1px;height:1px'; document.body.append(receiver);
  const canvas = document.createElement('canvas'); let callback, timer;
  try {
    const frames = [];
    await new Promise((done, reject) => {
      timer = setTimeout(() => reject(new Error('Display frame deadline')), 5000);
      const sample = (at, metadata) => {
        frames.push({ at, mediaTime: metadata.mediaTime, width: metadata.width, height: metadata.height });
        done();
      };
      callback = receiver.requestVideoFrameCallback(sample); receiver.play().catch(reject);
    });
    receiver.pause(); api.displayForGeneration(generation);
    canvas.width = receiver.videoWidth; canvas.height = receiver.videoHeight;
    const context = canvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true }); context.drawImage(receiver, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let binary = ''; for (let offset = 0; offset < pixels.length; offset += 16384) binary += String.fromCharCode(...pixels.subarray(offset, offset + 16384));
    const sha256 = [...new Uint8Array(await crypto.subtle.digest('SHA-256', pixels))].map(value => value.toString(16).padStart(2, '0')).join('');
    api.displayForGeneration(generation);
    return { generation, width: canvas.width, height: canvas.height, frames, pixels: { bytes: pixels.length, sha256, base64: btoa(binary) },
      scope: 'First decoded frame of a newly attached muted receiver, paused before diagnostic readback; no track clone, neural, scanout or audio measurement',
      stateTransitionFreshness: 'not measured; one frame is not an acknowledgment of a preceding layout or target change', cadence: 'not measured' };
  } finally {
    clearTimeout(timer); if (callback !== undefined) receiver.cancelVideoFrameCallback(callback);
    receiver.pause(); receiver.srcObject = null; receiver.remove(); canvas.width = canvas.height = 0;
  }
}

export async function manualObservation(checkpoint, page, instruction, read, accept) {
  const action = { instruction, observable: 'Native browser state, never an assumed gesture', requestedAt: new Date().toISOString() };
  checkpoint.manual(action);
  await page.bringToFront();
  await page.evaluate(instruction => {
    let output = document.getElementById('study-instruction');
    if (!output) { output = document.createElement('p'); output.id = 'study-instruction'; document.body.prepend(output); }
    output.textContent = instruction;
  }, instruction);
  console.log(`MANUAL CHECKPOINT: ${instruction}`);
  const value = await waitForObserved(read, accept);
  await page.evaluate(() => document.getElementById('study-instruction')?.remove());
  checkpoint.manual(null);
  return { ...action, observedAt: new Date().toISOString(), value };
}

export function storePixels(root, id, value) {
  for (const name of ['pixels', 'copiedPixels']) if (value[name]?.base64) {
    const { base64, ...metadata } = value[name], bytes = Buffer.from(base64, 'base64');
    assert.equal(bytes.length, metadata.bytes); assert.equal(sha256(bytes), metadata.sha256);
    const path = `${id}-${randomUUID()}-${name}.rgba`;
    writeFileSync(join(root, path), bytes, { flag: 'wx' }); value[name] = { ...metadata, path };
  }
  return value;
}

export function compareCompositedPixels({ source, acquired, mapping }) {
  const decode = value => {
    const bytes = Uint8ClampedArray.from(atob(value.pixels.base64), character => character.charCodeAt(0));
    const canvas = document.createElement('canvas'); canvas.width = value.width; canvas.height = value.height;
    canvas.getContext('2d', { colorSpace: 'srgb' }).putImageData(new ImageData(bytes, value.width, value.height), 0, 0);
    return canvas;
  };
  const original = decode(source), full = decode(acquired), reference = document.createElement('canvas'), crop = document.createElement('canvas');
  try {
    const [left, top, width, height] = mapping.crop;
    reference.width = crop.width = width; reference.height = crop.height = height;
    const referenceContext = reference.getContext('2d', { colorSpace: 'srgb' }), cropContext = crop.getContext('2d', { colorSpace: 'srgb' });
    const [fractionX, fractionY, fractionWidth, fractionHeight] = mapping.sourceFraction;
    referenceContext.imageSmoothingEnabled = true; referenceContext.imageSmoothingQuality = 'high';
    referenceContext.drawImage(original, fractionX * source.width, fractionY * source.height, fractionWidth * source.width, fractionHeight * source.height, 0, 0, width, height);
    cropContext.drawImage(full, left, top, width, height, 0, 0, width, height);
    const expected = referenceContext.getImageData(0, 0, width, height).data, actual = cropContext.getImageData(0, 0, width, height).data;
    let absolute = 0, maximum = 0, changedPixels = 0;
    for (let offset = 0; offset < expected.length; offset += 4) {
      let changed = false;
      for (let channel = 0; channel < 4; channel++) {
        const difference = Math.abs(expected[offset + channel] - actual[offset + channel]);
        absolute += difference; maximum = Math.max(maximum, difference); changed ||= difference !== 0;
      }
      changedPixels += Number(changed);
    }
    return { bytes: expected.length, mae: absolute / expected.length, maximum, changedPixels,
      sourceReference: 'Canvas2D sRGB high-quality smoothing to inferred visible capture extent; not browser compositor kernel parity',
      productFidelity: 'NOT QUALIFIED; whole-tab input, source-region mapping inferred only' };
  } finally { for (const canvas of [original, full, reference, crop]) canvas.width = canvas.height = 0; }
}

export async function runTabFidelity(directory) {
  const identity = studyIdentity(), root = resolve(ROOT, directory), apparatusEnvironment = environment();
  assert(root.startsWith(join(ROOT, '.cache/m1010/')));
  let native, fixtures, source, player, checkpoint;
  try {
    native = await openResearch(identity, { profileDirectory: join(root, 'profile') });
    checkpoint = openCheckpoint(directory, { studyVersion: 'M10.10-tab-fidelity-1', sourceCommit: identity.sourceCommit, browserExecutableSha256: sha256(readFileSync(native.executable)) });
    const interrupted = checkpoint.snapshot().activeExperimentId;
    if (interrupted) checkpoint.complete(interrupted, { outcome: 'UNRESOLVED', executionStatus: 'INTERRUPTED', reason: 'Interrupted attempt retained without another permission or invocation gesture' });
    if (TAB_FIDELITY_CASES.every(entry => checkpoint.has(entry.id))) return checkpoint.snapshot();
    if (checkpoint.has('consent') || checkpoint.has('invocation')) {
      terminalizeTabSuffix(checkpoint, 'Prior session interrupted after consent attempt; reacquisition requires separately authorized evidence');
      return checkpoint.snapshot();
    }
    fixtures = await startFixtures({ extensionOrigins: [`chrome-extension://${native.extensionId}`] });
    const playerUrl = `chrome-extension://${native.extensionId}/acquire.html`;
    const grants = () => native.worker.evaluate(() => chrome.permissions.getAll());
    if (!checkpoint.has('environment')) {
      checkpoint.begin('environment'); checkpoint.complete('environment', { identity, environment: apparatusEnvironment,
        browser: { version: await native.browser.version(), policy: 'default; physical permission and invocation only' }, media: fixtures.media, initialGrants: await grants() });
    }
    checkpoint.begin('consent');
    const consent = { initiallyGranted: (await grants()).permissions?.includes('tabCapture') === true };
    try { if (!consent.initiallyGranted) {
      player = await native.context.newPage(); await nativeWindow(player, native.context); await player.goto(playerUrl);
      const opening = await player.evaluate(() => globalThis.m1010Acquire.snapshot().events.length);
      consent.observation = await manualObservation(checkpoint, player, 'Click Grant tab capture and approve Chrome\'s request. Denial ends this attempt; the runner will otherwise prepare a fresh source tab.',
        async () => captureConsentResult(await player.evaluate(opening => globalThis.m1010Acquire.snapshot().events.slice(opening), opening)), value => value !== null);
      assert(consent.observation.value.granted, 'Tab capture permission denied');
      assert((await grants()).permissions?.includes('tabCapture'), 'Permission completion lacks actual grant');
      await player.close(); player = null;
    } } catch (error) { consent.error = String(error); throw error; }
    finally { checkpoint.complete('consent', consent); }
    source = await native.context.newPage(); await nativeWindow(source, native.context);
    await source.goto(`${fixtures.url}?extension=${native.extensionId}`);
    await source.waitForFunction(() => !!globalThis.__M1010_SOURCE__?.snapshot().selection);
    const nonce = randomUUID(); await source.evaluate(nonce => { document.documentElement.dataset.m1010Study = nonce; }, nonce);
    const invocation = await manualObservation(checkpoint, source,
      'On this fresh source tab, click Extensions > AetherVSR M10.10 Research > Select source and open acquisition. The five fidelity cases then run automatically.',
      async () => {
        for (const page of native.context.pages().filter(page => page.url() === playerUrl)) {
          const info = await page.evaluate(() => globalThis.m1010Acquire?.snapshot().info).catch(() => null);
          if (info?.selection) { player = page; return info; }
        }
        return null;
      }, value => !!value?.selection);
    const info = invocation.value;
    const verified = await native.worker.evaluate(info => chrome.scripting.executeScript({
      target: { tabId: info.sourceTabId, documentIds: [info.selection.documentId] },
      func: () => ({ nonce: document.documentElement.dataset.m1010Study, url: location.href, media: document.querySelector('video')?.currentSrc }),
    }), info);
    assert.equal(verified[0]?.documentId, info.selection.documentId); assert.equal(verified[0]?.result.nonce, nonce);
    assert.equal(verified[0]?.result.url, source.url()); assert.equal(verified[0]?.result.media, info.selection.url);
    if (!checkpoint.has('invocation')) {
      checkpoint.begin('invocation'); checkpoint.complete('invocation', { ...invocation, verified, grants: await grants() });
    }
    await player.bringToFront(); await player.evaluate(() => globalThis.m1010Acquire.tab());
    for (const entry of TAB_FIDELITY_CASES) {
      if (checkpoint.has(entry.id)) { console.log(JSON.stringify({ skippedImmutable: entry.id })); continue; }
      checkpoint.begin(entry.id);
      const result = { case: entry, samples: [], outcome: 'UNRESOLVED', scope: 'Whole-tab compositor characterization, no product crop, neural, latency or A/V qualification' };
      try {
        await source.evaluate(asset => globalThis.__M1010_SOURCE__.prepare({ mode: 'same', asset }), entry.asset);
        await source.evaluate(entry => {
          const video = document.querySelector('video'), width = video.videoWidth * entry.scale / devicePixelRatio, height = video.videoHeight * entry.scale / devicePixelRatio;
          video.style.cssText = `position:fixed;left:${entry.offscreen ? -width / 4 : 100}px;top:160px;width:${width}px;height:${height}px;max-width:none;outline:4px solid #ff00ff;object-fit:fill`;
          video.muted = true;
        }, entry);
        for (const time of INPUT_TIMES) {
          const sample = { time }; result.samples.push(sample);
          await source.bringToFront(); sample.seek = await source.evaluate(seekDiagnosticFrame, { time, expected: entry.asset === 'C' ? [1920, 1080] : [1280, 720] });
          const original = await source.evaluate(inputFrameEvidence);
          sample.source = storePixels(root, entry.id, structuredClone(original));
          sample.geometry = await source.evaluate(() => {
            const video = document.querySelector('video');
            return { rect: video.getBoundingClientRect().toJSON(), viewport: [innerWidth, innerHeight], dpr: devicePixelRatio,
              intrinsic: [video.videoWidth, video.videoHeight], controls: video.controls, objectFit: getComputedStyle(video).objectFit,
              cssPolicy: 'Fixture-controlled intrinsic/DPR sizing, fixed placement, magenta outline; native controls retained' };
          });
          await player.bringToFront(); sample.readiness = await player.evaluate(observePlayback);
          assert(sample.readiness.playable, 'No three acquired frames after paused source update');
          const acquired = await player.evaluate(inputFrameEvidence);
          sample.acquired = storePixels(root, entry.id, structuredClone(acquired));
          sample.tracks = await player.evaluate(() => document.querySelector('video').srcObject?.getTracks().map(track => ({
            kind: track.kind, constructor: track.constructor.name, state: track.readyState, muted: track.muted, settings: track.getSettings(),
          })) ?? []);
          sample.mapping = diagnosticCrop({ ...sample.geometry, captured: [acquired.width, acquired.height] });
          sample.difference = await player.evaluate(compareCompositedPixels, { source: original, acquired, mapping: sample.mapping });
        }
        result.outcome = 'RECORDED_NOT_QUALIFIED';
      } catch (error) { result.error = String(error); }
      checkpoint.complete(entry.id, result); console.log(JSON.stringify({ completed: entry.id, outcome: result.outcome, error: result.error ?? null }));
      if (result.error) break;
    }
  } finally {
    const cleanup = {};
    if (player && !player.isClosed()) {
      try { cleanup.before = await player.evaluate(() => document.querySelector('video').srcObject?.getTracks().map(track => ({ kind: track.kind, state: track.readyState })) ?? []);
        await player.evaluate(() => globalThis.m1010Acquire.stop()); }
      catch (error) { cleanup.error = String(error); }
      cleanup.player = await player.evaluate(() => globalThis.m1010Acquire.snapshot()).catch(error => ({ error: String(error) }));
    }
    if (source && !source.isClosed()) cleanup.source = await source.evaluate(() => globalThis.__M1010_SOURCE__.action({ type: 'dispose' })).catch(error => ({ error: String(error) }));
    await native?.close(); await fixtures?.close();
    if (checkpoint && checkpoint.snapshot().activeExperimentId === null) terminalizeTabSuffix(checkpoint, 'R3 session ended; missing suffix is terminal NOT_RUN, not automatic reacquisition');
    if (checkpoint && !checkpoint.has('cleanup') && checkpoint.snapshot().activeExperimentId === null) {
      checkpoint.begin('cleanup'); checkpoint.complete('cleanup', cleanup);
      checkpoint.candidate('R3', { state: 'CHARACTERIZATION_ONLY', timing: 'UNRESOLVED; prior floor has no absolute instrument error bound',
        cleanup: !cleanup.error && cleanup.player?.pending === false && Object.values(cleanup.player?.resources ?? { unknown: 1 }).every(value => value === 0) ? 'RECORDED' : 'UNRESOLVED' });
    }
  }
  return checkpoint.snapshot();
}

export async function runSelfCapture(directory) {
  const identity = studyIdentity(), root = resolve(ROOT, directory), apparatusEnvironment = environment();
  assert(root.startsWith(join(ROOT, '.cache/m1010/')));
  let native, fixtures;
  try {
    native = await openResearch(identity, { profileDirectory: join(root, 'profile') });
    const checkpoint = openCheckpoint(directory, { studyVersion: 'M10.10-self-capture-1', sourceCommit: identity.sourceCommit, browserExecutableSha256: sha256(readFileSync(native.executable)) });
    const interrupted = checkpoint.snapshot().activeExperimentId;
    if (interrupted) checkpoint.complete(interrupted, { outcome: 'UNRESOLVED', executionStatus: 'INTERRUPTED', reason: 'Interrupted chooser ordinal is not automatically repeated' });
    if (stopAfterUnsafeSelf(checkpoint)) return checkpoint.snapshot();
    if (checkpoint.has('setup-failure')) return checkpoint.snapshot();
    if (SELF_CAPTURE_CASES.every(entry => checkpoint.has(entry.id))) return checkpoint.snapshot();
    fixtures = await startFixtures();
    if (!checkpoint.has('environment')) {
      checkpoint.begin('environment'); checkpoint.complete('environment', { identity, environment: apparatusEnvironment, media: fixtures.media,
        browser: { version: await native.browser.version(), policy: 'default; owner physically starts and operates every chooser; no automatic permission handling' } });
    }
    const source = await native.context.newPage(), wrong = await native.context.newPage();
    await nativeWindow(source, native.context); await nativeWindow(wrong, native.context);
    const markers = {};
    for (const [name, page] of [['source', source], ['wrong', wrong]]) {
      try {
        await page.bringToFront();
        await page.goto(fixtures.url); await page.waitForFunction(() => !!globalThis.__M1010_SOURCE__?.snapshot().selection);
        markers[name] = await page.evaluate(installCaptureMarker, { identity: randomUUID().replaceAll('-', '').slice(0, 8), title: `M10.10 ${name === 'source' ? 'INTENDED' : 'WRONG'} fixture` });
      } catch (error) {
        const diagnostic = await page.evaluate(() => ({ visibility: document.visibilityState, focused: document.hasFocus(),
          status: document.getElementById('status')?.textContent, source: globalThis.__M1010_SOURCE__?.snapshot() })).catch(error => ({ error: String(error) }));
        checkpoint.begin('setup-failure'); checkpoint.complete('setup-failure', { name, error: String(error), diagnostic,
          outcome: 'UNRESOLVED', executionStatus: 'SETUP_FAILED', chooserTrialsStarted: false });
        throw error;
      }
    }
    assert.notEqual(markers.source.identity, markers.wrong.identity);
    for (const entry of SELF_CAPTURE_CASES) {
      if (checkpoint.has(entry.id)) { console.log(JSON.stringify({ skippedImmutable: entry.id })); continue; }
      checkpoint.begin(entry.id);
      const result = { case: entry, markers, scopes: [], outcome: 'UNRESOLVED', limitation: 'Primitive characterization only; timing and neural qualification stopped by unbounded local timing floor' };
      try {
        await source.bringToFront();
        await source.evaluate(mode => globalThis.__M1010_SOURCE__.prepare({ mode, asset: 'A' }), entry.method ? 'ABR' : 'same');
        await source.evaluate(() => {
          const video = document.querySelector('video'); video.style.cssText = ''; video.controls = true; video.muted = true;
          document.getElementById('study-occluder')?.remove(); document.body.style.minHeight = '2000px'; scrollTo(0, 0);
        });
        await source.evaluate(seekDiagnosticFrame, { time: 0.5, expected: entry.method ? [640, 360] : [1280, 720] });
        const before = await source.evaluate(() => globalThis.__M1010_SOURCE__.snapshot());
        result.choice = await manualObservation(checkpoint, source, entry.instruction,
          () => source.evaluate(() => globalThis.__M1010_SOURCE__.snapshot()),
          value => value.captureGeneration > before.captureGeneration && value.captureState !== 'choosing');
        result.captureError = result.choice.value.captureError;
        result.generation = result.choice.value.captureGeneration;
        if (result.choice.value.displayTracks.length) {
          const frame = await source.evaluate(displayFrameEvidence, { generation: result.generation });
          result.identityFrame = storePixels(root, entry.id, structuredClone(frame));
          const bytes = Buffer.from(frame.pixels.base64, 'base64');
          const decoded = [];
          for (const [name, marker] of Object.entries(markers)) {
            try { if (decodeCaptureMarker(bytes, frame.width, frame.height, marker.viewport) === marker.identity) decoded.push(name); }
            catch (error) { (result.markerErrors ??= {})[name] = String(error); }
          }
          result.identity = decoded.length === 1 ? decoded[0] : 'unresolved';
        }
        result.outcome = selfChoiceOutcome(entry.expected, result);
        if (entry.expected === 'rejection') result.rejectionScope = 'Requested cancel; native denial/rejection observed, API alone cannot distinguish chooser Cancel from permission denial';
        if (result.identity === 'wrong') result.wrongChoiceCleanup = await source.evaluate(() => globalThis.__M1010_SOURCE__.action({ type: 'wrong-choice' }));
        if (entry.method && result.outcome === 'IDENTITY_RECORDED') {
          const snapshot = await source.evaluate(generation => {
            globalThis.__M1010_SOURCE__.displayForGeneration(generation); return globalThis.__M1010_SOURCE__.snapshot();
          }, result.generation);
          const track = snapshot.displayTracks.find(track => track.kind === 'video'), method = entry.method === 'crop' ? 'cropTo' : 'restrictTo';
          result.method = { name: method, track, factory: snapshot.targetFactories[entry.method] };
          if (track?.[method] !== 'function' || snapshot.targetFactories[entry.method] !== 'function') result.method.outcome = 'UNSUPPORTED_SAFE';
          else {
            try { await source.evaluate(async ({ kind, generation }) => {
              const api = globalThis.__M1010_SOURCE__; api.displayForGeneration(generation); await api[kind](); api.displayForGeneration(generation);
            }, { kind: entry.method, generation: result.generation }); result.method.outcome = 'FULFILLED'; }
            catch (error) { result.method.outcome = 'REJECTED'; result.method.error = String(error); }
            result.method.native = (await source.evaluate(() => globalThis.__M1010_SOURCE__.snapshot())).targetAttempts.at(-1);
            if (result.method.outcome === 'FULFILLED') for (const scope of SELF_SCOPE_CASES) {
              const observation = { scope, generation: result.generation }; result.scopes.push(observation);
              try {
                observation.geometry = await source.evaluate(({ scope, generation }) => {
                  globalThis.__M1010_SOURCE__.displayForGeneration(generation);
                  const video = document.querySelector('video');
                  document.getElementById('study-occluder')?.remove(); video.controls = scope !== 'controls-hidden';
                  if (scope === 'movement') video.style.marginLeft = '80px';
                  if (scope === 'resize') { video.style.width = '320px'; video.style.height = '180px'; }
                  if (scope === 'scroll') scrollTo(0, 80);
                  if (scope === 'occluder') {
                    const rect = video.getBoundingClientRect(), overlay = document.createElement('div'); overlay.id = 'study-occluder';
                    overlay.style.cssText = `position:fixed;left:${rect.x + rect.width / 4}px;top:${rect.y + rect.height / 4}px;width:${rect.width / 2}px;height:${rect.height / 2}px;background:#ff00ff;z-index:10`;
                    document.body.append(overlay);
                  }
                  const style = getComputedStyle(video);
                  return { rect: video.getBoundingClientRect().toJSON(), viewport: [innerWidth, innerHeight], dpr: devicePixelRatio,
                    scroll: [scrollX, scrollY], intrinsic: [video.videoWidth, video.videoHeight], controls: video.controls,
                    isolation: style.isolation, transformStyle: style.transformStyle, transform: style.transform,
                    scope: 'Deliberately cooperating fixture CSS; not generic-page eligibility' };
                }, { scope, generation: result.generation });
                if (scope.startsWith('ABR-')) {
                  observation.seek = await source.evaluate(seekDiagnosticFrame, { time: scope === 'ABR-low' ? 0.5 : 2.5, expected: scope === 'ABR-low' ? [640, 360] : [1280, 720] });
                  observation.geometry.intrinsic = await source.evaluate(() => [document.querySelector('video').videoWidth, document.querySelector('video').videoHeight]);
                }
                const original = await source.evaluate(inputFrameEvidence);
                observation.source = storePixels(root, entry.id, structuredClone(original));
                const acquired = await source.evaluate(displayFrameEvidence, { generation: result.generation });
                observation.acquired = storePixels(root, entry.id, structuredClone(acquired));
                observation.difference = await source.evaluate(compareCompositedPixels, { source: original, acquired,
                  mapping: { crop: [0, 0, acquired.width, acquired.height], sourceFraction: [0, 0, 1, 1] } });
                observation.snapshot = await source.evaluate(generation => {
                  globalThis.__M1010_SOURCE__.displayForGeneration(generation); return globalThis.__M1010_SOURCE__.snapshot();
                }, result.generation);
              } catch (error) { observation.error = String(error); result.outcome = 'UNRESOLVED'; break; }
            }
          }
        }
      } catch (error) { result.error = String(error); result.outcome = 'UNRESOLVED'; }
      finally {
        result.beforeCleanup = await source.evaluate(() => globalThis.__M1010_SOURCE__.snapshot()).catch(error => ({ error: String(error) }));
        result.cleanup = await source.evaluate(() => globalThis.__M1010_SOURCE__.action({ type: 'cancel' })).catch(error => ({ error: String(error) }));
        if ([result.wrongChoiceCleanup, result.cleanup].filter(Boolean).some(cleanup => cleanup.error || cleanup.displayTracks?.length ||
          cleanup.stoppedTracks?.some(track => track.after.readyState !== 'ended'))) result.outcome = 'UNSAFE';
      }
      checkpoint.complete(entry.id, result); console.log(JSON.stringify({ completed: entry.id, outcome: result.outcome, identity: result.identity, method: result.method?.outcome }));
      if (stopAfterUnsafeSelf(checkpoint)) break;
    }
    checkpoint.candidate('R6', { state: 'PRIMITIVE_CHARACTERIZATION_ONLY', absoluteInstrumentErrorBoundMs: null });
    return checkpoint.snapshot();
  } finally { await native?.close(); await fixtures?.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) {
  assert(['tab', 'self'].includes(process.argv[2]), 'Use tab or self');
  await (process.argv[2] === 'tab' ? runTabFidelity : runSelfCapture)(process.argv[3]);
}