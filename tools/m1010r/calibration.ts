import { acquireGpu, watchDeviceFailures, type GpuContext } from '../../src/core/gpu/device.js';
import { VideoPipeline, type PipelineGpuSample } from '../../src/core/pipeline.js';
import type { FrameTick } from '../../src/core/types.js';
import { BaselineScaler } from '../../src/core/upscale/baseline-scaler.js';
import { IdentityProbe } from './probe.js';
import { runAudioControl, type AudioControlResult } from './audio-control.js';

type State = 'NOT_RUN' | 'RUNNING' | 'RECORDED' | 'UNRESOLVED';
interface RunOptions { fps: 30 | 60; seconds?: 65; mediaUrl: string }
interface Bracket { index: number; submitBefore: number; audioBefore: number }
interface FrameRow extends Bracket {
  sequence: number;
  generation: number;
  currentTime: number;
  mediaTime: number;
  presentedFrames: number | null;
  presentationTime: number;
  expectedDisplayTime: number;
  submitAfter: number;
  audioAfter: number;
  readyAt: number | null;
  neural: false;
  sourceIdentity: number | null;
  identityValid: boolean;
  sourceWidth: number | null;
  sourceHeight: number | null;
}
interface AudioRecording {
  type: 'audio-recording';
  sampleRate: number;
  firstFrame: number | null;
  samples: number;
  blockLengths: number[];
  overflow: boolean;
  discontinuity: boolean;
  pcm: ArrayBuffer;
}
interface CallbackRow {
  now: number;
  callbackTimestamp: number;
  metadata: VideoFrameCallbackMetadata;
  currentTime: number;
  readyState: number;
  visibility: DocumentVisibilityState;
  focused: boolean;
  quality: { totalVideoFrames: number; droppedVideoFrames: number; corruptedVideoFrames: number } | null;
}

function element<T extends HTMLElement>(id: string, constructor: new () => T): T {
  const found = document.getElementById(id);
  if (!(found instanceof constructor)) throw new Error(`Missing calibration element: ${id}`);
  return found;
}

const video = element('source', HTMLVideoElement);
const canvas = element('output', HTMLCanvasElement);
const status = element('status', HTMLOutputElement);
const play = element('play', HTMLButtonElement);
const pause = element('pause', HTMLButtonElement);
const seek = element('seek', HTMLInputElement);
const volume = element('volume', HTMLInputElement);
const mute = element('mute', HTMLInputElement);
const fullscreen = element('fullscreen', HTMLButtonElement);
const close = element('close', HTMLButtonElement);
const stage = element('stage', HTMLElement);
let pcm: ArrayBuffer | null = null;
let controlPcm: ArrayBuffer | null = null;
let active: { abort: AbortController; context: AudioContext; gate: GainNode | null } | null = null;
let running: Promise<CalibrationReport> | null = null;
const controls = new AbortController();

const report = {
  state: 'NOT_RUN' as State,
  options: null as RunOptions | null,
  requestedAt: null as string | null,
  startedAt: null as string | null,
  endedAt: null as string | null,
  timeOrigin: performance.timeOrigin,
  firstSubmissionPerformance: null as number | null,
  startPerformance: null as number | null,
  plannedEndPerformance: null as number | null,
  endPerformance: null as number | null,
  lastSubmissionPerformance: null as number | null,
  audioTailEndPerformance: null as number | null,
  timingScope: {
    submission: 'performance/audio clocks before inner encode and after successful queue.submit; excludes import',
    readiness: 'Queue-completion callback upper endpoint, including queued work and callback delivery; not GPU duration',
    gpuSamples: 'Baseline render-pass timestamp-query elapsed milliseconds; identity compute is outside this pass',
    window: '65 seconds from first post-submit observation: 5 seconds warmup, then 60 seconds observation',
    acceptance: 'Offline analyzer only; RECORDED is not PASS',
  },
  frames: [] as FrameRow[],
  callbacks: [] as CallbackRow[],
  metadataMatchErrors: 0,
  completeness: null as { firstSequence: number; lastSequence: number; count: number; encodedCount: number; observedCallbackCount: number } | null,
  audioControl: null as Omit<AudioControlResult, 'pcm'> | null,
  audio: null as (Omit<AudioRecording, 'pcm'> & { bytes: number; encoding: string; channel: number }) | null,
  graph: null as Record<string, unknown> | null,
  capabilities: null as Record<string, unknown> | null,
  gpuSamples: [] as PipelineGpuSample[],
  events: [] as { type: string; performance: number; detail: unknown }[],
  errors: [] as { stage: string; performance: number; message: string }[],
  cleanup: {
    pipeline: 0, probe: 0, device: 0, audioNodes: 0, observers: 0, timers: 0,
    pendingCompletions: 0, audioContext: 'not created', videoPaused: video.paused,
    completed: false,
  },
};
type CalibrationReport = typeof report;

function recordEvent(type: string, detail: unknown = null): void {
  report.events.push({ type, performance: performance.now(), detail });
}

function recordError(stageName: string, error: unknown): void {
  report.errors.push({ stage: stageName, performance: performance.now(), message: String(error) });
}

function snapshot(): CalibrationReport {
  return structuredClone(report);
}

function audioBase64(control = false): string {
  const data = control ? controlPcm : pcm;
  if (report.state === 'RUNNING' || data === null) throw new Error('Audio recording is not finished');
  const bytes = new Uint8Array(data);
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 0x8000)));
  }
  return btoa(chunks.join(''));
}

function isRecording(value: unknown): value is AudioRecording {
  if (typeof value !== 'object' || value === null) return false;
  const data = value as Partial<AudioRecording>;
  return data.type === 'audio-recording' && data.pcm instanceof ArrayBuffer &&
    typeof data.sampleRate === 'number' && Number.isFinite(data.sampleRate) &&
    (data.firstFrame === null || (typeof data.firstFrame === 'number' && Number.isSafeInteger(data.firstFrame))) &&
    typeof data.samples === 'number' && Number.isSafeInteger(data.samples) && data.samples >= 0 &&
    data.samples * 4 <= data.pcm.byteLength && Array.isArray(data.blockLengths) &&
    data.blockLengths.every((length: unknown) => typeof length === 'number' && Number.isSafeInteger(length) && length > 0) &&
    typeof data.overflow === 'boolean' && typeof data.discontinuity === 'boolean';
}

async function collect(options: RunOptions): Promise<CalibrationReport> {
  const abort = new AbortController();
  const listeners = new AbortController();
  const timers = new Set<number>();
  const pending = new Set<Promise<void>>();
  let context: AudioContext | null = null;
  let gpu: GpuContext | null = null;
  let pipeline: VideoPipeline | null = null;
  let probe: IdentityProbe | null = null;
  let source: MediaElementAudioSourceNode | null = null;
  let gate: GainNode | null = null;
  let worklet: AudioWorkletNode | null = null;
  let unwatch: (() => void) | null = null;
  let callbackId: number | null = null;
  let latest: CallbackRow | null = null;
  let bracket: Bracket | null = null;
  let accepting = false;
  let readyValid = true;
  let stopped = false;
  let completedWindow = false;
  let finishRequested = false;
  let audioStarted = false;
  let observedSubmissions = 0;
  let finishWindow: (() => void) | null = null;

  function schedule(callback: () => void, delay: number): number {
    const id = window.setTimeout(() => {
      timers.delete(id);
      report.cleanup.timers = timers.size;
      callback();
    }, delay);
    timers.add(id);
    report.cleanup.timers = timers.size;
    return id;
  }

  function clearTimer(id: number): void {
    window.clearTimeout(id);
    timers.delete(id);
    report.cleanup.timers = timers.size;
  }

  function bounded<T>(operation: Promise<T>, milliseconds: number, label: string, cancellable = true): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let settled = false;
      const onAbort = () => finish(() => reject(new Error(`${label}: cancelled`)));
      const timeout = schedule(() => finish(() => reject(new Error(`${label}: timeout`))), milliseconds);
      function finish(action: () => void): void {
        if (settled) return;
        settled = true;
        clearTimer(timeout);
        abort.signal.removeEventListener('abort', onAbort);
        action();
      }
      if (cancellable) abort.signal.addEventListener('abort', onAbort, { once: true });
      operation.then((value) => finish(() => resolve(value)), (error: unknown) =>
        finish(() => reject(error instanceof Error ? error : new Error(String(error)))));
      if (cancellable && abort.signal.aborted) onAbort();
    });
  }

  function fail(stageName: string, error: unknown): void {
    recordError(stageName, error);
    abort.abort();
    stopFrames();
    finishWindow?.();
  }

  function stopFrames(): void {
    if (stopped) return;
    stopped = true;
    accepting = false;
    if (pipeline) {
      pipeline.onFrame = null;
      pipeline.stop();
    }
    report.endPerformance = performance.now();
    report.endedAt = new Date().toISOString();
    recordEvent('frame-collection-stopped');
  }

  function stopTap(): void {
    if (callbackId !== null) video.cancelVideoFrameCallback(callbackId);
    callbackId = null;
    latest = null;
  }

  function pauseMedia(): void {
    video.pause();
    report.audioTailEndPerformance ??= performance.now();
    stopTap();
  }

  async function finishAudio(): Promise<void> {
    if (!worklet || !audioStarted || finishRequested) return;
    finishRequested = true;
    const node = worklet;
    try {
      const recording = await bounded(new Promise<AudioRecording>((resolve, reject) => {
        node.port.onmessage = (event: MessageEvent<unknown>) => {
          if (isRecording(event.data)) resolve(event.data);
          else reject(new Error('Malformed audio-worklet response'));
        };
        node.port.onmessageerror = () => reject(new Error('Audio-worklet message decoding failed'));
        node.port.postMessage('finish');
      }), 5000, 'audio finish', false);
      pcm = recording.pcm.slice(0, recording.samples * 4);
      const { pcm: transferred, ...metadata } = recording;
      report.audio = { ...metadata, bytes: pcm.byteLength, encoding: 'Float32 PCM, native byte order', channel: 0 };
      if (transferred.byteLength < pcm.byteLength || recording.samples === 0 || recording.firstFrame === null ||
          recording.sampleRate !== 48000 || recording.overflow || recording.discontinuity) {
        recordError('audio integrity', 'Empty, discontinuous, overflowing or non-48kHz recording');
      }
    } finally {
      node.port.onmessage = null;
      node.port.onmessageerror = null;
    }
  }

  function observe(callbackTimestamp: number, metadata: VideoFrameCallbackMetadata): void {
    callbackId = null;
    const now = performance.now();
    const quality = typeof video.getVideoPlaybackQuality === 'function' ? video.getVideoPlaybackQuality() : null;
    latest = {
      now, callbackTimestamp, metadata: { ...metadata }, currentTime: video.currentTime,
      readyState: video.readyState, visibility: document.visibilityState, focused: document.hasFocus(),
      quality: quality ? {
        totalVideoFrames: quality.totalVideoFrames, droppedVideoFrames: quality.droppedVideoFrames,
        corruptedVideoFrames: quality.corruptedVideoFrames,
      } : null,
    };
    report.callbacks.push(latest);
    callbackId = video.requestVideoFrameCallback(observe);
  }

  async function finishGpu(): Promise<void> {
    if (!gpu || !pipeline || !probe) return;
    await bounded(gpu.device.queue.onSubmittedWorkDone(), 5000, 'GPU drain', false);
    await bounded(Promise.all([...pending]), 5000, 'completion markers', false);
    await bounded(pipeline.drainTimings(), 5000, 'GPU timestamps', false);
    if (probe.encoded === 0) return;
    const identities = await bounded(probe.readAfterPause(), 5000, 'identity readback', false);
    const submitted = pipeline.stats(performance.now()).framesRendered;
    report.completeness = { firstSequence: submitted ? 1 : 0,
      lastSequence: submitted, count: submitted,
      encodedCount: identities[0] ?? 0, observedCallbackCount: observedSubmissions };
    if (identities[0] !== report.frames.length || probe.encoded !== report.frames.length) {
      throw new Error('Identity/submission counts differ');
    }
    for (let index = 0; index < report.frames.length; index++) {
      const row = report.frames[index];
      if (!row || row.index !== index) throw new Error(`Missing probe index ${index}`);
      const offset = 4 + index * 4;
      row.sourceIdentity = identities[offset] ?? null;
      row.identityValid = identities[offset + 1] === 1;
      row.sourceWidth = identities[offset + 2] ?? null;
      row.sourceHeight = identities[offset + 3] ?? null;
      if (!row.identityValid || row.sourceIdentity === null || row.sourceWidth !== 1280 || row.sourceHeight !== 720) {
        recordError('identity', `Invalid identity/extent at probe index ${index}`);
      }
    }
    if (report.frames.some((row) => row.readyAt === null)) recordError('queue completion', 'Missing readiness endpoints');
  }

  try {
    if ((options.fps !== 30 && options.fps !== 60) || (options.seconds !== undefined && options.seconds !== 65)) {
      throw new Error('Calibration requires fps 30 or 60 and exactly 65 seconds');
    }
    const mediaUrl = new URL(options.mediaUrl, document.baseURI);
    if (!['http:', 'https:', 'chrome-extension:'].includes(mediaUrl.protocol)) throw new Error('Progressive media URL required');
    if (typeof video.requestVideoFrameCallback !== 'function') throw new Error('Native rVFC unavailable');
    report.options = { ...options, seconds: 65, mediaUrl: mediaUrl.href };
    context = new AudioContext({ sampleRate: 48000, latencyHint: 'playback' });
    const audio = context;
    active = { abort, context: audio, gate: null };
    report.cleanup.audioContext = audio.state;
    const resumed = bounded(audio.resume(), 5000, 'audio resume');
    const acquisition = acquireGpu().then((acquired) => {
      if (abort.signal.aborted) {
        acquired.device.destroy();
        throw new Error('GPU acquired after cancellation');
      }
      gpu = acquired;
      report.cleanup.device = 1;
      unwatch = watchDeviceFailures(acquired.device, (message) => {
        readyValid = false;
        for (const row of report.frames) row.readyAt = null;
        fail('GPU', message);
      });
      return acquired;
    });
    const [acquired] = await bounded(Promise.all([acquisition, resumed]), 10000, 'GPU/audio setup');
    report.capabilities = {
      ...acquired.capabilities, adapter: acquired.adapterReport, nativeRvfc: true,
      playbackQuality: typeof video.getVideoPlaybackQuality === 'function',
      gpuTiming: acquired.capabilities.timestampQuery ? 'timestamp-query' : 'not measured',
      userAgent: navigator.userAgent, secureContext: window.isSecureContext,
    };
    if (audio.sampleRate !== 48000 || audio.state !== 'running') throw new Error('48kHz running AudioContext required');
    const trackState = (type: string) => recordEvent(type, {
      currentTime: video.currentTime, paused: video.paused, muted: video.muted, volume: video.volume,
      playbackRate: video.playbackRate, readyState: video.readyState,
      visibility: document.visibilityState, focused: document.hasFocus(), audioState: audio.state,
      audioTime: audio.currentTime, gain: gate?.gain.value ?? null,
    });
    document.addEventListener('visibilitychange', () => trackState('visibilitychange'), { signal: listeners.signal });
    for (const type of ['focus', 'blur']) window.addEventListener(type, () => trackState(type), { signal: listeners.signal });
    for (const type of ['playing', 'pause', 'seeking', 'seeked', 'ratechange', 'volumechange', 'waiting', 'stalled', 'ended']) {
      video.addEventListener(type, () => trackState(type), { signal: listeners.signal });
    }
    video.addEventListener('error', () => fail('media', video.error?.message ?? 'Media error'), { signal: listeners.signal });
    video.addEventListener('ended', () => { if (accepting) fail('media', 'Ended before collection endpoint'); }, { signal: listeners.signal });
    video.addEventListener('timeupdate', () => { seek.value = String(video.currentTime); }, { signal: listeners.signal });
    audio.addEventListener('statechange', () => trackState('audio-statechange'), { signal: listeners.signal });
    window.addEventListener('error', (event) => fail('window', event.message), { signal: listeners.signal });
    window.addEventListener('unhandledrejection', (event) => fail('promise', event.reason), { signal: listeners.signal });
    window.addEventListener('pagehide', () => fail('pagehide', 'Page closed during calibration'), { signal: listeners.signal });
    report.cleanup.observers = 1;
    trackState('setup');

    await bounded(audio.audioWorklet.addModule(new URL('audio-worklet.js', document.baseURI).href), 5000, 'worklet load');
    const control = await bounded(runAudioControl(audio, new URL(`media/decoded-${options.fps}.f32le`, document.baseURI).href), 10000, 'scheduled audio control');
    controlPcm = control.pcm;
    const { pcm: controlBytes, ...controlSummary } = control;
    report.audioControl = controlSummary;
    if (!controlBytes.byteLength || control.errors.length || control.maximumSampleError === null || control.maximumSampleError > 12) throw new Error('Scheduled audio-clock calibration failed');
    source = audio.createMediaElementSource(video);
    report.cleanup.audioNodes++;
    gate = audio.createGain();
    report.cleanup.audioNodes++;
    gate.gain.value = 1;
    active.gate = gate;
    const nodeOptions: AudioWorkletNodeOptions = {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2],
      channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'speakers',
    };
    worklet = new AudioWorkletNode(audio, 'm1010r-audio', nodeOptions);
    report.cleanup.audioNodes++;
    worklet.onprocessorerror = () => fail('audio worklet', 'Processor error');
    source.connect(gate).connect(worklet).connect(audio.destination);
    report.graph = {
      path: ['MediaElementAudioSource', 'Gain', 'm1010r-audio', 'destination'],
      sampleRate: audio.sampleRate, requestedSampleRate: 48000, latencyHint: 'playback',
      baseLatency: audio.baseLatency, outputLatency: audio.outputLatency,
      sink: 'default (no setSinkId)', nodeOptions, gain: gate.gain.value,
      recorderChannel: 0, destinationChannelCount: audio.destination.channelCount,
      scope: 'Fixed product-equivalent research graph; candidates must reuse this graph and configuration',
    };
    video.crossOrigin = 'anonymous';
    video.loop = false;
    video.muted = false;
    video.volume = 1;
    video.playbackRate = 1;
    const metadataReady = new Promise<void>((resolve) => {
      video.addEventListener('loadedmetadata', () => resolve(), { once: true, signal: listeners.signal });
    });
    video.src = mediaUrl.href;
    video.load();
    await bounded(metadataReady, 10000, 'media metadata');
    if (!Number.isFinite(video.duration) || video.duration < 70 || video.videoWidth !== 1280 || video.videoHeight !== 720) {
      throw new Error('Expected non-looping 1280x720 progressive media of at least 70 seconds');
    }
    seek.max = String(video.duration);
    recordEvent('metadata', { duration: video.duration, width: video.videoWidth, height: video.videoHeight, currentSrc: video.currentSrc });

    probe = new IdentityProbe(new BaselineScaler('catmull-rom'), (index) => {
      bracket = { index, submitBefore: performance.now(), audioBefore: audio.currentTime };
    });
    report.cleanup.probe = 1;
    pipeline = new VideoPipeline(acquired, video, canvas, probe, { forceCopyImport: false });
    const render = pipeline;
    report.cleanup.pipeline = 1;
    render.onConfiguration = (configuration) => recordEvent('pipeline-configuration', configuration);
    render.onGpuSample = (sample) => report.gpuSamples.push(sample);
    const windowDone = new Promise<void>((resolve) => { finishWindow = resolve; });
    render.onFrame = (tick: FrameTick) => {
      const audioAfter = audio.currentTime;
      const submitAfter = performance.now();
      if (!accepting) return;
      observedSubmissions++;
      const before = bracket;
      bracket = null;
      if (!before) { fail('submission', 'Missing beforeEncode bracket'); return; }
      const native = latest;
      const matched = native !== null && native.metadata.mediaTime === tick.mediaTime &&
        native.metadata.presentationTime === tick.presentationTime;
      latest = null;
      if (!matched) {
        report.metadataMatchErrors++;
        recordError('rVFC match', `No exact native metadata for probe index ${before.index}`);
      }
      const row: FrameRow = {
        ...before, sequence: before.index + 1, generation: render.timingGeneration,
        currentTime: video.currentTime, mediaTime: tick.mediaTime,
        presentedFrames: matched ? native.metadata.presentedFrames : null,
        presentationTime: tick.presentationTime, expectedDisplayTime: tick.expectedDisplayTime,
        submitAfter, audioAfter, readyAt: null, neural: false, sourceIdentity: null, identityValid: false,
        sourceWidth: null, sourceHeight: null,
      };
      const previous = report.frames.at(-1);
      if (row.index !== report.frames.length || row.submitBefore > submitAfter || row.audioBefore > audioAfter ||
          (previous && (row.submitBefore < previous.submitAfter || row.audioBefore < previous.audioAfter))) {
        recordError('clock/index order', `Invalid bracket or index at ${row.index}`);
      }
      report.frames.push(row);
      report.lastSubmissionPerformance = submitAfter;
      if (report.firstSubmissionPerformance === null) {
        report.firstSubmissionPerformance = submitAfter;
        report.startPerformance = submitAfter + 5000;
        report.plannedEndPerformance = submitAfter + 65000;
        report.startedAt = new Date().toISOString();
        recordEvent('first-submission', { performance: submitAfter, generation: row.generation });
        const deadline = report.plannedEndPerformance;
        const finishAtDeadline = () => {
          const remaining = deadline - performance.now();
          if (remaining > 0) { schedule(finishAtDeadline, Math.ceil(remaining)); return; }
          completedWindow = true; stopFrames(); finishWindow?.();
        };
        schedule(finishAtDeadline, Math.max(1, Math.ceil(deadline - performance.now())));
      }
      const completion = acquired.device.queue.onSubmittedWorkDone().then(() => {
        const readyAt = performance.now();
        if (readyValid && row.generation === render.timingGeneration && readyAt >= row.submitAfter) row.readyAt = readyAt;
        else recordError('queue completion', `Invalid generation/clock at probe index ${row.index}`);
      }).catch((error: unknown) => fail('queue completion', error)).finally(() => {
        pending.delete(completion);
        report.cleanup.pendingCompletions = pending.size;
      });
      pending.add(completion);
      report.cleanup.pendingCompletions = pending.size;
    };
    accepting = true;
    const watchdog = schedule(() => fail('collection', '75-second collection watchdog expired'), 75000);
    callbackId = video.requestVideoFrameCallback(observe);
    render.start();
    worklet.port.postMessage('start');
    audioStarted = true;
    for (const control of [pause, seek, volume, mute]) control.disabled = false;
    await bounded(video.play(), 5000, 'video play');
    await bounded(windowDone, 75000, 'collection');
    clearTimer(watchdog);
    if (abort.signal.aborted) throw new Error('Collection interrupted');
    await bounded(new Promise<void>((resolve) => { schedule(resolve, 100); }), 1000, 'audio tail');
    pauseMedia();
    recordEvent('audio-tail-ended', { audioTime: audio.currentTime });
  } catch (error) {
    recordError('run', error);
  } finally {
    abort.abort();
    const cleanup = (name: string, action: () => void) => {
      try { action(); } catch (error) { recordError(name, error); }
    };
    cleanup('stop frames', stopFrames);
    cleanup('pause media', pauseMedia);
    if (pipeline?.error) recordError('pipeline', pipeline.error);
    try { await finishAudio(); } catch (error) { recordError('audio finish', error); }
    try { await finishGpu(); } catch (error) { recordError('GPU finish', error); }
    readyValid = false;
    if (worklet) {
      worklet.onprocessorerror = null;
      worklet.port.onmessage = null;
      worklet.port.onmessageerror = null;
      cleanup('worklet port', () => worklet?.port.close());
    }
    for (const node of [source, gate, worklet]) {
      if (node) cleanup('audio disconnect', () => { node.disconnect(); report.cleanup.audioNodes--; });
    }
    if (context) {
      try { await bounded(context.close(), 5000, 'audio close', false); } catch (error) { recordError('audio close', error); }
      report.cleanup.audioContext = context.state;
    }
    cleanup('pipeline destroy', () => { pipeline?.destroy(); report.cleanup.pipeline = 0; });
    cleanup('probe destroy', () => { probe?.destroy(); report.cleanup.probe = 0; });
    cleanup('device watcher', () => unwatch?.());
    cleanup('device destroy', () => { gpu?.device.destroy(); report.cleanup.device = 0; });
    try { await bounded(Promise.all([...pending]), 5000, 'final completion settlement', false); }
    catch (error) { recordError('final completion settlement', error); }
    listeners.abort();
    controls.abort();
    report.cleanup.observers = 0;
    for (const id of [...timers]) clearTimer(id);
    for (const control of [play, pause, seek, volume, mute, fullscreen, close]) control.disabled = true;
    cleanup('media release', () => { video.removeAttribute('src'); video.load(); });
    report.cleanup.videoPaused = video.paused;
    report.cleanup.completed = report.cleanup.pipeline === 0 && report.cleanup.probe === 0 &&
      report.cleanup.device === 0 && report.cleanup.audioNodes === 0 && report.cleanup.pendingCompletions === 0 &&
      (report.cleanup.audioContext === 'closed' || report.cleanup.audioContext === 'not created') && video.paused;
    if (!report.cleanup.completed) recordError('cleanup', 'Resources did not reach zero/closed state');
    active = null;
    report.state = completedWindow && report.errors.length === 0 ? 'RECORDED' : 'UNRESOLVED';
    status.value = report.state;
  }
  return snapshot();
}

function run(options: RunOptions): Promise<CalibrationReport> {
  if (report.state !== 'NOT_RUN') return Promise.reject(new Error('Calibration page is single-use'));
  report.state = 'RUNNING';
  report.requestedAt = new Date().toISOString();
  report.options = { ...options };
  play.disabled = true;
  status.value = report.state;
  running = collect(options);
  return running;
}

const api = { run, snapshot, audioBase64 };
declare global { interface Window { m1010rCalibration: typeof api } }
window.m1010rCalibration = api;

play.addEventListener('click', () => {
  const params = new URLSearchParams(location.search);
  const fps = Number(params.get('fps') ?? '30') as 30 | 60;
  void run({ fps, mediaUrl: params.get('media') ?? 'media/replay-30.mp4' }).catch((error: unknown) => recordError('play', error));
}, { signal: controls.signal });
pause.addEventListener('click', () => {
  recordEvent('control-pause');
  video.pause();
  recordError('control', 'Calibration paused');
  active?.abort.abort();
}, { signal: controls.signal });
seek.addEventListener('input', () => {
  recordEvent('control-seek', seek.valueAsNumber);
  video.currentTime = seek.valueAsNumber;
}, { signal: controls.signal });
function updateGain(): void {
  if (!active?.gate) return;
  active.gate.gain.setValueAtTime(mute.checked ? 0 : volume.valueAsNumber, active.context.currentTime);
  recordEvent('control-audio', { muted: mute.checked, volume: volume.valueAsNumber, gain: active.gate.gain.value });
}
volume.addEventListener('input', updateGain, { signal: controls.signal });
mute.addEventListener('change', updateGain, { signal: controls.signal });
fullscreen.addEventListener('click', () => {
  const change = document.fullscreenElement ? document.exitFullscreen() : stage.requestFullscreen();
  void change.catch((error: unknown) => recordError('fullscreen', error));
}, { signal: controls.signal });
close.addEventListener('click', () => {
  recordEvent('control-close');
  active?.abort.abort();
  void (running ?? Promise.resolve()).then(() => window.close()).catch((error: unknown) => recordError('close', error));
}, { signal: controls.signal });