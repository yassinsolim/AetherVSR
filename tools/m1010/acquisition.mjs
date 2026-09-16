import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { ROOT, sha256 } from '../m10-fixtures.mjs';
import { nativeWindow } from '../m105-accounting.mjs';
import { studyIdentity, openResearch } from './native.mjs';
import { startFixtures } from './fixtures.mjs';

export const MEDIA_CASES = [
  { name: 'progressive30', mode: 'same', asset: 'A' },
  { name: 'progressive60', mode: 'same', asset: 'B' },
  { name: 'cors', mode: 'cors', asset: 'A' },
  { name: 'nocors', mode: 'nocors', asset: 'A' },
  { name: 'blob', mode: 'blob', asset: 'A' },
  { name: 'MSE', mode: 'MSE', asset: 'A' },
  { name: 'srcObject', mode: 'srcObject', asset: 'A' },
  { name: 'ABR', mode: 'ABR', asset: 'A' },
  { name: 'credentialed', mode: 'auth', asset: 'A', credentials: 'include' },
];

export async function observePlayback() {
  const video = document.querySelector('video'), frames = [], events = [];
  let handle, timer;
  const start = performance.now();
  const onEvent = event => events.push({ at: performance.now(), type: event.type });
  const names = ['playing', 'pause', 'ended', 'waiting', 'stalled', 'error'];
  names.forEach(name => video.addEventListener(name, onEvent));
  const result = { playable: false, frames, events, start, error: null };
  try {
    await new Promise((resolveReady, reject) => {
      timer = setTimeout(() => reject(new Error(video.error?.message ?? 'No three decoded frames within 5s')), 5000);
      const sample = (at, metadata) => {
        frames.push({ at, mediaTime: metadata.mediaTime, presentedFrames: metadata.presentedFrames, width: metadata.width, height: metadata.height });
        if (frames.length === 3) resolveReady(); else handle = video.requestVideoFrameCallback(sample);
      };
      handle = video.requestVideoFrameCallback(sample);
      video.play().catch(reject);
    });
    result.playable = true;
  } catch (error) { result.error = String(error); }
  finally {
    clearTimeout(timer); if (handle !== undefined) video.cancelVideoFrameCallback(handle);
    video.pause(); names.forEach(name => video.removeEventListener(name, onEvent));
  }
  return { ...result, end: performance.now(), currentSrc: video.currentSrc, width: video.videoWidth, height: video.videoHeight,
    duration: Number.isFinite(video.duration) ? video.duration : null, muted: video.muted, paused: video.paused,
    readyState: video.readyState, mediaError: video.error?.code ?? null,
    cadence: 'not measured; three-frame readiness observation only' };
}

export async function inputFrameEvidence() {
  const video = document.querySelector('video');
  if (!video.paused || video.seeking || video.readyState < 2) throw new Error('Input evidence requires a stable paused decoded frame');
  const width = video.videoWidth, height = video.videoHeight;
  const pack = async bytes => {
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 16384) binary += String.fromCharCode(...bytes.subarray(offset, offset + 16384));
    return { bytes: bytes.length, sha256: [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(value => value.toString(16).padStart(2, '0')).join(''), base64: btoa(binary) };
  };
  const result = { width, height, mediaTime: video.currentTime, origin: location.origin, cssRect: video.getBoundingClientRect().toJSON(),
    dpr: devicePixelRatio, visibility: document.visibilityState, focused: document.hasFocus(),
    normalizedFormat: 'RGBA8 sRGB unpremultiplied', sourceColorSpace: null, originClean: false,
    externalImportable: false, forcedCopyImportable: false, pixels: null, copiedPixels: null, errors: {} };
  let frame, device, texture, buffer;
  try {
    try { frame = new VideoFrame(video); result.sourceColorSpace = frame.colorSpace.toJSON(); }
    catch (error) { result.errors.colorSpace = String(error); }
    finally { frame?.close(); }
    try {
      const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
      const context = canvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true });
      context.drawImage(video, 0, 0); result.pixels = await pack(context.getImageData(0, 0, width, height).data); result.originClean = true;
    } catch (error) { result.errors.pixels = String(error); }
    const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error('No GPU adapter');
    device = await adapter.requestDevice();
    device.pushErrorScope('validation');
    try { device.importExternalTexture({ source: video }); result.externalImportable = true; }
    catch (error) { result.errors.external = String(error); }
    const externalError = await device.popErrorScope();
    if (externalError) { result.externalImportable = false; result.errors.external = externalError.message; }
    device.pushErrorScope('validation');
    try {
      texture = device.createTexture({ size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
      const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
      buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      device.queue.copyExternalImageToTexture({ source: video }, { texture, colorSpace: 'srgb' }, [width, height]);
      const encoder = device.createCommandEncoder(); encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [width, height]);
      device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
      const mapped = new Uint8Array(buffer.getMappedRange()), normalized = new Uint8Array(width * height * 4);
      for (let row = 0; row < height; row++) normalized.set(mapped.subarray(row * bytesPerRow, row * bytesPerRow + width * 4), row * width * 4);
      result.copiedPixels = await pack(normalized); buffer.unmap(); result.forcedCopyImportable = true;
    } catch (error) { result.errors.copy = String(error); }
    const copyError = await device.popErrorScope();
    if (copyError) { result.forcedCopyImportable = false; result.errors.copy = copyError.message; result.copiedPixels = null; }
  } catch (error) { result.errors.gpu = String(error); }
  finally { buffer?.destroy(); texture?.destroy(); device?.destroy(); }
  return result;
}

export function pixelDifference(reference, candidate) {
  assert(reference.length === candidate.length && reference.length > 0 && reference.length % 4 === 0);
  let absolute = 0, maximum = 0, changedPixels = 0;
  for (let offset = 0; offset < reference.length; offset += 4) {
    let changed = false;
    for (let channel = 0; channel < 4; channel++) {
      const difference = Math.abs(reference[offset + channel] - candidate[offset + channel]);
      absolute += difference; maximum = Math.max(maximum, difference); changed ||= difference !== 0;
    }
    changedPixels += Number(changed);
  }
  return { bytes: reference.length, mae: absolute / reference.length, maximum, changedPixels, exact: changedPixels === 0 };
}

export async function inspectMedia({ capture = false, actions = false, adaptive = false } = {}) {
  const video = document.querySelector('video'), started = performance.now(), observations = [], events = [];
  const stop = stream => stream?.getTracks().forEach(track => track.stop());
  let stream, receiver, frame, device, channel;
  const quality = () => { const value = video.getVideoPlaybackQuality?.(); return value ? { total: value.totalVideoFrames, dropped: value.droppedVideoFrames } : null; };
  const opening = quality();
  const sample = (at, value) => { observations.push({ at, mediaTime: value.mediaTime, presented: value.presentedFrames, width: value.width, height: value.height }); frame = video.requestVideoFrameCallback(sample); };
  const named = event => events.push({ at: performance.now(), type: event.type });
  const names = ['playing', 'pause', 'seeking', 'seeked', 'ratechange', 'resize', 'waiting', 'stalled', 'ended', 'error'];
  const boundedPlay = async element => {
    let timer;
    try { await Promise.race([element.play(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Native playback deadline')), 5000); })]); }
    finally { clearTimeout(timer); }
  };
  names.forEach(name => video.addEventListener(name, named));
  const result = { started, origin: location.origin, focused: document.hasFocus(), visibility: document.visibilityState,
    source: video.currentSrc, current: { width: video.videoWidth, height: video.videoHeight, ready: video.readyState },
    playable: false, originClean: false, importable: false, captureStream: null, observations, events, qualityOpening: opening, operations: [] };
  try {
    frame = video.requestVideoFrameCallback(sample);
    try { await boundedPlay(video); result.playable = true; } catch (error) { result.playError = String(error); }
    await new Promise(done => setTimeout(done, 500));
    result.playable &&= observations.length > 0;
    const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    try {
      const context = canvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true }); context.drawImage(video, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      result.originClean = true; result.pixelSha256 = [...new Uint8Array(await crypto.subtle.digest('SHA-256', pixels))].map(value => value.toString(16).padStart(2, '0')).join('');
    } catch (error) { result.pixelError = String(error); }
    try { const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error('No adapter'); device = await adapter.requestDevice(); device.importExternalTexture({ source: video }); result.importable = true; }
    catch (error) { result.importError = String(error); }
    if (capture) {
      try {
        if (video.mediaKeys) throw new Error('Protected media: product rejects before capture');
        if (!video.captureStream) throw new Error('captureStream unavailable');
        stream = video.captureStream();
        const tracks = () => stream.getTracks().map(track => ({ kind: track.kind, state: track.readyState, muted: track.muted, settings: track.getSettings() }));
        result.captureStream = { tracks: tracks(), clones: {}, localDeliveredFrames: 0 };
        try { structuredClone(stream); result.captureStream.clones.structuredClone = 'SUPPORTED'; }
        catch (error) { result.captureStream.clones.structuredClone = String(error); }
        channel = new MessageChannel();
        try { channel.port1.postMessage(stream); result.captureStream.clones.messageChannel = 'SUPPORTED'; }
        catch (error) { result.captureStream.clones.messageChannel = String(error); }
        receiver = document.createElement('video'); receiver.muted = true; receiver.playsInline = true; receiver.srcObject = stream;
        receiver.style.cssText = 'position:fixed;width:1px;height:1px;left:-4px'; document.body.append(receiver);
        let handle;
        const receive = () => { result.captureStream.localDeliveredFrames++; handle = receiver.requestVideoFrameCallback(receive); };
        handle = receiver.requestVideoFrameCallback(receive);
        try { await boundedPlay(receiver); await new Promise(done => setTimeout(done, 500)); }
        finally { receiver.cancelVideoFrameCallback(handle); }
        result.captureStream.received = { width: receiver.videoWidth, height: receiver.videoHeight, ready: receiver.readyState };
        if (adaptive) {
          const before = { source: [video.videoWidth, video.videoHeight], receiver: [receiver.videoWidth, receiver.videoHeight], at: performance.now() };
          video.currentTime = 2.25;
          await new Promise(done => setTimeout(done, 1000));
          const after = { source: [video.videoWidth, video.videoHeight], receiver: [receiver.videoWidth, receiver.videoHeight], at: performance.now() };
          result.captureStream.adaptive = { before, after, outcome: before.source[0] === 640 && after.source[0] === 1280 && after.receiver[0] === 1280 ? 'SUPPORTED' : 'UNRESOLVED' };
        }
        if (actions) for (const type of ['pause', 'play', 'seek', 'rate']) {
          const operation = { type, at: performance.now() }; result.operations.push(operation);
          try {
            if (type === 'pause') video.pause(); else if (type === 'play') await boundedPlay(video);
            else if (type === 'seek') video.currentTime = 1; else video.playbackRate = 1.25;
            await new Promise(done => setTimeout(done, 100)); operation.source = { paused: video.paused, time: video.currentTime, rate: video.playbackRate };
            operation.tracks = tracks();
          } catch (error) { operation.error = String(error); }
        }
        result.captureStream.finalTracks = tracks();
      } catch (error) { result.captureError = String(error); }
    }
    result.finished = performance.now(); result.qualityClosing = quality();
    result.outcome = result.playable && (!capture || result.captureStream?.localDeliveredFrames > 0) ? 'SUPPORTED' : result.playError || result.captureError ? 'UNSUPPORTED_SAFE' : 'UNRESOLVED';
    if (adaptive && result.captureStream?.adaptive?.outcome !== 'SUPPORTED') result.outcome = 'UNRESOLVED';
  } finally {
    if (frame !== undefined) video.cancelVideoFrameCallback(frame);
    names.forEach(name => video.removeEventListener(name, named));
    receiver?.pause(); if (receiver) receiver.srcObject = null; receiver?.remove(); stop(stream);
    channel?.port1.close(); channel?.port2.close(); device?.destroy();
    result.cleanup = { liveTracks: stream?.getTracks().filter(track => track.readyState === 'live').length ?? 0, receiverConnected: receiver?.isConnected ?? false };
  }
  return result;
}

export async function runMediaMatrix(prefix) {
  prefix = resolve(ROOT, prefix); assert(prefix.startsWith(join(ROOT, '.cache/m1010/'))); mkdirSync(dirname(prefix), { recursive: true });
  const report = { schemaVersion: 1, phase: 'UNPRIVILEGED_MEDIA_API_MATRIX', identity: studyIdentity(), started: new Date().toISOString(), sources: [], replay: [] };
  let native, fixtures;
  try {
    native = await openResearch(report.identity);
    fixtures = await startFixtures({ extensionOrigins: [`chrome-extension://${native.extensionId}`] }); report.media = fixtures.media;
    report.browser = { version: await native.browser.version(), executableSha256: sha256(readFileSync(native.executable)),
      policy: 'Default autoplay, no host permission or tab/screen capture request' };
    for (const entry of MEDIA_CASES) {
      const page = await native.context.newPage(), result = { ...entry }; report.sources.push(result);
      try {
        result.placement = await nativeWindow(page, native.context); await page.goto(fixtures.url); await page.bringToFront();
        if (entry.mode === 'auth') await page.evaluate(() => globalThis.__M1010_SOURCE__.prepare('cookieseed'));
        result.selection = await page.evaluate(entry => globalThis.__M1010_SOURCE__.prepare({ mode: entry.mode, asset: entry.asset, credentials: entry.credentials ?? 'omit' }), entry);
        await page.locator('#play').click();
        result.result = await page.evaluate(inspectMedia, { capture: true, actions: ['same', 'cors', 'MSE'].includes(entry.mode), adaptive: entry.mode === 'ABR' });
        result.snapshot = await page.evaluate(() => globalThis.__M1010_SOURCE__.snapshot());
        if (['blob', 'MSE'].includes(entry.mode)) {
          const replay = await native.context.newPage(), candidate = { name: `page-${entry.mode}`, url: result.snapshot.selection.url };
          report.replay.push(candidate);
          try {
            await nativeWindow(replay, native.context); await replay.goto(`chrome-extension://${native.extensionId}/acquire.html`); await replay.bringToFront();
            await replay.evaluate(url => { const video = document.querySelector('video'); video.src = url; video.muted = true; }, candidate.url);
            await replay.locator('h1').click(); candidate.result = await replay.evaluate(inspectMedia);
          } catch (error) { candidate.error = String(error); candidate.outcome = 'UNRESOLVED'; }
          finally { await replay.close(); }
        }
        if (entry.mode === 'same' && entry.asset === 'A') {
          result.protectedState = await page.evaluate(async () => {
            const video = document.querySelector('video');
            try { const access = await navigator.requestMediaKeySystemAccess('org.w3.clearkey', [{ initDataTypes: ['cenc'], videoCapabilities: [{ contentType: 'video/mp4; codecs="avc1.640034"' }] }]);
              const keys = await access.createMediaKeys(); await video.setMediaKeys(keys);
              const rejectedBeforeCapture = video.mediaKeys !== null; await video.setMediaKeys(null);
              return { reproduced: true, rejectedBeforeCapture, scope: 'Only clear-key MediaKeys state; no license, keys or protected pixels accessed' };
            } catch (error) { return { reproduced: false, error: String(error), outcome: 'UNRESOLVED' }; }
          });
        }
      } catch (error) { result.error = String(error); }
      finally { await page.close(); }
      console.log(JSON.stringify({ phase: 'source', name: entry.name, outcome: result.result?.outcome, error: result.error ?? null, captureError: result.result?.captureError ?? null }));
    }
    for (const [name, url, cors] of [
      ['packaged', `chrome-extension://${native.extensionId}/media/known.mp4`, null],
      ['same-page-origin', `${fixtures.origins[0]}/same/A.mp4`, null],
      ['cross-CORS', `${fixtures.origins[1]}/cors/A.mp4`, 'anonymous'],
      ['cross-no-CORS', `${fixtures.origins[1]}/nocors/A.mp4`, null],
      ['authenticated', `${fixtures.origins[1]}/auth/include/A.mp4`, 'use-credentials'],
    ]) {
      const page = await native.context.newPage(), result = { name, url, cors }; report.replay.push(result);
      try {
        await nativeWindow(page, native.context); await page.goto(`chrome-extension://${native.extensionId}/acquire.html`); await page.bringToFront();
        await page.evaluate(({ url, cors }) => { const video = document.querySelector('video'); video.muted = true; if (cors) video.crossOrigin = cors; else video.removeAttribute('crossorigin'); video.src = url; video.load(); }, { url, cors });
        await page.locator('h1').click();
        result.result = await page.evaluate(inspectMedia);
        result.fetch = await page.evaluate(async url => { try { const response = await fetch(url, { redirect: 'error', credentials: 'omit' });
          const bytes = await response.arrayBuffer(); return { fetchable: response.ok, status: response.status, type: response.type, bytes: bytes.byteLength }; }
          catch (error) { return { fetchable: false, error: String(error) }; } }, url);
      } catch (error) { result.error = String(error); }
      finally { await page.close(); }
      console.log(JSON.stringify({ phase: 'replay', name, result: result.result && { playable: result.result.playable, originClean: result.result.originClean, importable: result.result.importable }, fetchable: result.fetch?.fetchable, error: result.error ?? null }));
    }
  } catch (error) { report.error = String(error); }
  finally { await native?.close(); await fixtures?.close(); report.finished = new Date().toISOString(); writeFileSync(`${prefix}.json`, JSON.stringify(report, null, 2), { flag: 'wx' }); }
  return report;
}