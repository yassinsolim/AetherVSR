import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { ROOT, sha256 } from '../m10-fixtures.mjs';
import { nativeWindow } from '../m105-accounting.mjs';
import { studyIdentity, openResearch } from './native.mjs';
import { prepareTimingMedia, prepareMedia, serveFixtures, decodeTimingCounter } from './fixtures.mjs';
import { openCheckpoint } from './checkpoint.mjs';
import { environment } from './study.mjs';

export const FLOOR_MARKERS = Array.from({ length: 30 }, (_, index) => 6 + index * 2);

export async function collectTimingFloor({ fps, markers, workletUrl }) {
  const video = document.querySelector('video');
  const audio = new AudioContext({ sampleRate: 48000 });
  const result = { fps, timeOrigin: performance.timeOrigin, calibration: [], audio: [], anchors: [], frames: [], samples: [], events: [],
    start: null, end: null, error: null, scope: 'Local direct-preview apparatus floor, browser timestamp estimates only; no neural, capture transport, scanout or acoustic measurement' };
  let detector, media, calibrationSource, gain, handle, timer, active = true;
  const preview = document.createElement('canvas'); preview.width = 1280; preview.height = 720;
  preview.style.cssText = 'width:320px;height:180px'; video.after(preview);
  const previewContext = preview.getContext('2d', { colorSpace: 'srgb' });
  const pool = Array.from({ length: markers.length * 7 }, () => ['source', 'preview'].map(() => {
    const canvas = document.createElement('canvas'); canvas.width = 360; canvas.height = 82;
    return { canvas, context: canvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true }) };
  }));
  const captured = [];
  const onEvent = event => result.events.push({ at: performance.now(), type: event.type, mediaTime: video.currentTime });
  window.addEventListener('blur', onEvent); document.addEventListener('visibilitychange', onEvent);
  const makeDetector = generation => {
    const node = new AudioWorkletNode(audio, 'm1010-timing', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], processorOptions: { generation } });
    node.port.onmessage = event => { if (active && event.data.generation === generation) (generation === 0 ? result.calibration : result.audio).push(event.data); };
    return node;
  };
  try {
    await audio.audioWorklet.addModule(workletUrl); await audio.resume();
    if (audio.state !== 'running' || audio.sampleRate !== 48000) throw new Error('Audio clock unavailable at48kHz');
    result.audioDevice = { sampleRate: audio.sampleRate, baseLatency: audio.baseLatency, outputLatency: audio.outputLatency ?? null };
    detector = makeDetector(0); gain = audio.createGain(); gain.gain.value = 0;
    detector.connect(gain).connect(audio.destination);
    const buffer = audio.createBuffer(1, 96000, audio.sampleRate), pcm = buffer.getChannelData(0);
    result.scheduled = [];
    for (let index = 0; index < 3; index++) {
      const startSample = 12000 + index * 24000; let firstThresholdSample = null;
      for (let offset = 0; offset < 4800; offset++) {
        pcm[startSample + offset] = 0.35 * Math.sin(2 * Math.PI * (600 + 20 * index) * offset / audio.sampleRate);
        if (firstThresholdSample === null && Math.abs(pcm[startSample + offset]) > 0.1) firstThresholdSample = startSample + offset;
      }
      result.scheduled.push({ id: index, firstThresholdSample });
    }
    calibrationSource = audio.createBufferSource(); calibrationSource.buffer = buffer; calibrationSource.connect(detector);
    result.calibrationStart = Math.ceil((audio.currentTime + 0.1) * audio.sampleRate) / audio.sampleRate;
    await new Promise((done, reject) => {
      timer = setTimeout(() => reject(new Error('Digital calibration deadline')), 5000);
      calibrationSource.onended = done; calibrationSource.start(result.calibrationStart);
    });
    clearTimeout(timer); calibrationSource.disconnect(); detector.disconnect(); detector.port.close(); gain.disconnect();
    if (result.calibration.length !== 3) throw new Error('Missing digital calibration pulses');
    result.calibrationResidualSamples = result.calibration.map((pulse, index) => {
      if (pulse.id !== index) throw new Error('Digital calibration identity mismatch');
      return pulse.firstSample - Math.round(result.calibrationStart * audio.sampleRate) - result.scheduled[index].firstThresholdSample;
    });
    if (result.calibrationResidualSamples.some(value => Math.abs(value) > 1)) throw new Error('Digital sample calibration differs by more than one sample');
    detector = makeDetector(1); media = audio.createMediaElementSource(video); gain = audio.createGain(); gain.gain.value = 0.25;
    media.connect(detector).connect(gain).connect(audio.destination);
    video.muted = false; video.volume = 1; video.loop = false;
    await new Promise((done, reject) => {
      timer = setTimeout(() => reject(new Error('No initial floor frame')), 5000);
      const sample = (callbackAt, metadata) => {
        try {
          const at = performance.now();
          if (result.start === null) { clearTimeout(timer); result.start = at; timer = setTimeout(done, 65000); }
          const before = performance.now(), timestamp = audio.getOutputTimestamp(), after = performance.now();
          result.anchors.push({ before, after, ...timestamp });
          const row = { at, callbackAt, mediaTime: metadata.mediaTime, expectedDisplayTime: metadata.expectedDisplayTime,
            presentationTime: metadata.presentationTime, presentedFrames: metadata.presentedFrames,
            visible: document.visibilityState === 'visible', focused: document.hasFocus() };
          result.frames.push(row); previewContext.drawImage(video, 0, 0);
          const marker = markers.find(value => metadata.mediaTime >= value - 2 / fps && metadata.mediaTime <= value + 2 / fps);
          if (marker !== undefined) {
            const pair = pool[captured.length]; if (!pair) throw new Error('Finite marker-capture budget exhausted');
            pair[0].context.drawImage(video, 0, 0, 360, 82, 0, 0, 360, 82);
            pair[1].context.drawImage(preview, 0, 0, 360, 82, 0, 0, 360, 82);
            captured.push({ marker, ...row, latchedAt: performance.now(), pair });
          }
          handle = video.requestVideoFrameCallback(sample);
        } catch (error) { reject(error); }
      };
      handle = video.requestVideoFrameCallback(sample); video.play().catch(reject);
    });
  } catch (error) { result.error = String(error); }
  finally {
    result.end = performance.now(); active = false; clearTimeout(timer);
    if (handle !== undefined) video.cancelVideoFrameCallback(handle);
    video.pause(); media?.disconnect(); detector?.disconnect(); detector?.port.close(); gain?.disconnect();
    try { calibrationSource?.stop(); } catch {}
    calibrationSource?.disconnect(); await audio.close();
    window.removeEventListener('blur', onEvent); document.removeEventListener('visibilitychange', onEvent); preview.remove();
    result.cleanup = { audioState: audio.state, previewConnected: preview.isConnected, videoPaused: video.paused };
  }
  for (const { pair, ...entry } of captured) {
    for (const [index, name] of ['source', 'preview'].entries()) {
      const pixels = pair[index].context.getImageData(0, 0, 360, 82).data;
      let binary = ''; for (let offset = 0; offset < pixels.length; offset += 16384) binary += String.fromCharCode(...pixels.subarray(offset, offset + 16384));
      entry[name] = { base64: btoa(binary), width: 360, height: 82, format: 'RGBA8 sRGB' };
    }
    result.samples.push(entry);
  }
  for (const pair of pool) for (const item of pair) { item.canvas.width = 0; item.canvas.height = 0; }
  preview.width = preview.height = 0;
  result.cleanup.diagnosticCanvases = pool.flat().filter(item => item.canvas.width || item.canvas.height).length;
  return result;
}

export function timingPixelIdentity(bytes) {
  assert.equal(bytes.length, 360 * 82 * 4);
  const row = vertical => Uint8Array.from({ length: 360 }, (_, horizontal) => bytes[(vertical * 360 + horizontal) * 4]);
  const frame = decodeTimingCounter(row(24), row(56));
  const offset = (81 * 360 + 100) * 4;
  return { frame, flash: [0, 1, 2].every(channel => bytes[offset + channel] > 220) };
}

export function summarizeTimingFloor(record) {
  const rows = FLOOR_MARKERS.map(id => {
    const samples = record.samples.filter(sample => sample.marker === id);
    const before = samples.filter(sample => sample.source.identity?.frame < id * record.fps && !sample.source.identity.flash).at(-1);
    const after = samples.find(sample => sample.source.identity?.frame >= id * record.fps && sample.source.identity.flash);
    const pulses = record.audio.filter(pulse => pulse.id === id && Math.abs(pulse.hz - (600 + 20 * id)) <= 2);
    if (!before || !after || pulses.length !== 1) return { id, outcome: 'UNRESOLVED', before: !!before, after: !!after, matchingAudioPulses: pulses.length };
    const pulse = pulses[0], contextTime = pulse.firstSample / pulse.sampleRate;
    const preceding = record.anchors.filter(anchor => anchor.contextTime > 0 && anchor.contextTime <= contextTime).at(-1);
    const following = record.anchors.find(anchor => anchor.contextTime >= contextTime);
    if (!preceding || !following) return { id, outcome: 'UNRESOLVED', reason: 'Missing output timestamp bracket' };
    const estimates = [preceding, following].map(anchor => anchor.performanceTime + (contextTime - anchor.contextTime) * 1000);
    const apiBracketMs = Math.max(preceding.after - preceding.before, following.after - following.before);
    const audioEstimate = [Math.min(...estimates) - apiBracketMs, Math.max(...estimates) + apiBracketMs];
    const videoPrediction = [before.expectedDisplayTime, after.expectedDisplayTime];
    return { id, outcome: 'RECORDED', contextTime, audioEstimate, videoPrediction,
      audioMinusVideoEstimateMs: [audioEstimate[0] - videoPrediction[1], audioEstimate[1] - videoPrediction[0]],
      bracketDisagreementMs: Math.abs(estimates[1] - estimates[0]), apiBracketMs };
  });
  const identityMismatch = record.samples.filter(sample => !sample.source.identity || !sample.preview.identity ||
    sample.source.identity.frame !== sample.preview.identity.frame || sample.source.identity.flash !== sample.preview.identity.flash ||
    sample.source.identity.frame !== Math.round(sample.mediaTime * record.fps)).length;
  return { outcome: record.error || identityMismatch || record.events.length || !record.frames.every(frame => frame.visible && frame.focused) || rows.some(row => row.outcome !== 'RECORDED') ? 'UNRESOLVED' : 'RECORDED_NOT_QUALIFIED',
    rows, identityMismatch, durationMs: record.start === null ? null : record.end - record.start,
    absoluteInstrumentErrorBoundMs: null, physicalOutputOffset: 'not measured',
    limitation: 'Pixel identities were latched live, then read offline. rVFC expectedDisplayTime and getOutputTimestamp are browser predictions/estimates; their systematic output error is not bounded by API-call brackets or local residuals. These intervals are not hard A/V uncertainty bounds.' };
}

export async function runTimingFloor(directory) {
  const identity = studyIdentity(), root = resolve(ROOT, directory);
  assert(root.startsWith(join(ROOT, '.cache/m1010/')));
  const media = prepareTimingMedia(), apparatusEnvironment = environment();
  let native, fixtures;
  try {
    native = await openResearch(identity); fixtures = await serveFixtures({ ...media, C: prepareMedia().C }, { extensionOrigins: [`chrome-extension://${native.extensionId}`] });
    const checkpoint = openCheckpoint(directory, { studyVersion: 'M10.10-timing-floor-1', sourceCommit: identity.sourceCommit, browserExecutableSha256: sha256(readFileSync(native.executable)) });
    for (const asset of ['A', 'B']) {
      const id = `floor-${asset}`;
      if (checkpoint.has(id)) { console.log(JSON.stringify({ skippedImmutable: id })); continue; }
      checkpoint.begin(id); console.log(JSON.stringify({ running: id }));
      const page = await native.context.newPage();
      const result = { identity, environment: apparatusEnvironment, media: media[asset], browser: { version: await native.browser.version(), policy: 'default, no capture or host permission requests' } };
      try {
        result.placement = await nativeWindow(page, native.context);
        await page.goto(`chrome-extension://${native.extensionId}/acquire.html`); await page.bringToFront();
        await page.evaluate(url => { const video = document.querySelector('video'); video.crossOrigin = 'anonymous'; video.muted = true; video.src = url; video.load(); }, `${fixtures.origins[1]}/cors/${asset}.mp4`);
        await page.waitForFunction(() => document.querySelector('video').readyState >= 2); await page.locator('h1').click();
        result.record = await page.evaluate(collectTimingFloor, { fps: media[asset].fps, markers: FLOOR_MARKERS, workletUrl: `chrome-extension://${native.extensionId}/timing-worklet.js` });
        for (const sample of result.record.samples) for (const name of ['source', 'preview']) {
          const { base64, ...format } = sample[name], bytes = Buffer.from(base64, 'base64');
          const path = `${id}-${randomUUID()}-${name}.rgba`; writeFileSync(join(root, path), bytes, { flag: 'wx' });
          sample[name] = { ...format, path, bytes: bytes.length, sha256: sha256(bytes) };
          try { sample[name].identity = timingPixelIdentity(bytes); } catch (error) { sample[name].error = String(error); }
        }
        result.summary = summarizeTimingFloor(result.record); result.outcome = result.summary.outcome;
      } catch (error) { result.error = String(error); result.outcome = 'UNRESOLVED'; }
      finally { await page.close(); }
      checkpoint.complete(id, result); console.log(JSON.stringify({ completed: id, outcome: result.outcome, error: result.error ?? result.record?.error ?? null }));
    }
    checkpoint.candidate('TIMING', { state: 'FLOOR_RECORDED_NO_QUALIFICATION', absoluteInstrumentErrorBoundMs: null });
    console.log(JSON.stringify({ checkpoint: relative(ROOT, join(root, 'state.json')) })); return checkpoint.snapshot();
  } finally { await native?.close(); await fixtures?.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) await runTimingFloor(process.argv[2]);