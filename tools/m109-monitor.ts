import { acquireGpu, watchDeviceFailures, type GpuContext } from '../src/core/gpu/device.js';
import { VideoPipeline } from '../src/core/pipeline.js';
import { BaselineScaler } from '../src/core/upscale/baseline-scaler.js';
import { inspectGeometry } from '../src/extension/geometry.js';
import { assessContract, createContractReader, RecoveryGate } from './m109-contract.js';
import { observeSuccessfulSubmissions, type SubmissionIdentity, type SuccessfulFrameSubmission } from './m109-submission.js';

export async function startMonitor(video: HTMLVideoElement) {
  const canvas = video.ownerDocument.createElement('canvas');
  canvas.width = 0;
  canvas.height = 0;
  canvas.dataset['aethervsrM109'] = crypto.randomUUID();
  canvas.setAttribute('aria-hidden', 'true');
  const style = (name: string, value: string) => canvas.style.setProperty(name, value, 'important');
  for (const [name, value] of Object.entries({ all: 'initial', position: 'fixed', display: 'block',
    'pointer-events': 'none', visibility: 'hidden', animation: 'none', transition: 'none' })) style(name, value);
  video.after(canvas);
  const read = createContractReader(video, canvas), gate = new RecoveryGate();
  const owner = crypto.randomUUID();
  const submissions: SuccessfulFrameSubmission[] = [];
  const transitions: { at: number; reason: string | null; generation: number; supported: boolean }[] = [];
  const guards: { at: number; ms: number; reason: string }[] = [];
  const proofs: { at: number; ms: number; generation: number; supported: boolean }[] = [];
  const listeners: [EventTarget, string, EventListener][] = [];
  let gpu: GpuContext | null = null, pipeline: VideoPipeline | null = null;
  let release = () => {}, unwatch = () => {};
  let frame: number | null = null, disposed = false, error: string | null = null;
  let sourceGeneration = 0, geometryGeneration = 0, sourceKey = '', fingerprint = '';
  let authorized = false, proofSupported = false, configuredSource = '', provenBacking = '', captureOriginal = false;
  let reason: string | null = 'starting', proofCalls = 0, initializes = 0;
  const hide = () => { style('visibility', 'hidden'); gate.invalidate(); authorized = false; provenBacking = ''; };
  const identity = (): SubmissionIdentity => ({ owner, sourceGeneration, geometryGeneration,
    backingWidth: canvas.width, backingHeight: canvas.height, authorized });
  const fail = (message: string) => { error ??= message; hide(); pipeline?.stop(); reason = 'execution-error'; };
  const guard = (boundary: string) => {
    if (disposed || error) return;
    const at = performance.now();
    try {
      if (pipeline?.error) { fail(String(pipeline.error)); return; }
      const current = read();
      const input = { ...current, output: null };
      const admission = assessContract(input);
      const nextSource = JSON.stringify(current.source);
      if (nextSource !== sourceKey) { sourceKey = nextSource; sourceGeneration++; hide(); }
      const changed = admission.fingerprint !== fingerprint;
      if (changed) {
        fingerprint = admission.fingerprint;
        geometryGeneration++;
        hide();
        proofSupported = false;
        reason = admission.reason;
        if (admission.outcome === 'SUPPORTED') {
          const started = performance.now();
          const geometry = inspectGeometry(video);
          proofCalls++;
          if (geometry.ok) {
            if (canvas.parentNode !== geometry.placement.parent || video.nextSibling !== canvas) video.after(canvas);
            for (const [name, value] of Object.entries(geometry.style)) style(name, value);
            style('visibility', 'hidden');
            style('animation', 'none');
            style('transition', 'none');
            proofSupported = true;
          } else reason = geometry.code;
          proofs.push({ at: started, ms: performance.now() - started, generation: geometryGeneration, supported: geometry.ok });
        }
        transitions.push({ at, reason, generation: geometryGeneration, supported: proofSupported });
      }
      if (admission.outcome === 'UNSUPPORTED') { hide(); pipeline?.stop(); return; }
      if (!proofSupported) { hide(); pipeline?.stop(); return; }
      const output = read();
      if (configuredSource !== nextSource && output.output &&
        (output.output.width !== current.source.width * 2 || output.output.height !== current.source.height * 2)) {
        hide(); reason = 'backing-not-ready'; pipeline?.start(); return;
      }
      if (assessContract(output).outcome !== 'SUPPORTED') {
        hide(); pipeline?.stop(); reason = 'unsupported-output'; return;
      }
      configuredSource = nextSource;
      const backing = `${canvas.width}x${canvas.height}`;
      if (!authorized || backing !== provenBacking) { authorized = true; provenBacking = backing; gate.prove(identity()); }
      reason = null;
      if (pipeline && !pipeline.running) pipeline.start();
    } catch (failure) { fail(String(failure)); }
    finally { guards.push({ at, ms: performance.now() - at, reason: boundary }); }
  };
  const listen = (target: EventTarget, type: string, callback: EventListener) => {
    target.addEventListener(type, callback, { capture: true }); listeners.push([target, type, callback]);
  };
  const invalidate: EventListener = event => {
    if (disposed) return;
    if (event.type === 'loadstart' || event.type === 'emptied') sourceGeneration++;
    hide(); fingerprint = '';
  };
  for (const type of ['scroll', 'resize']) listen(window, type, invalidate);
  for (const type of ['fullscreenchange', 'visibilitychange']) listen(document, type, invalidate);
  for (const type of ['resize', 'loadstart', 'emptied', 'loadeddata', 'pause', 'playing', 'seeking', 'seeked',
    'enterpictureinpicture', 'leavepictureinpicture']) listen(video, type, invalidate);
  for (const type of ['change', 'addtrack', 'removetrack']) listen(video.textTracks, type, invalidate);
  let finalResources: { device: number; pipeline: number; canvas: number; listeners: number; frame: number } | null = null;
  const dispose = () => {
    if (!disposed) {
      disposed = true;
      hide();
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      for (const [target, type, callback] of listeners) target.removeEventListener(type, callback, { capture: true });
      listeners.length = 0;
      for (const cleanup of [release, unwatch,
        () => { pipeline?.destroy(); pipeline = null; },
        () => { gpu?.device.destroy(); gpu = null; },
        () => canvas.remove()]) {
        try { cleanup(); } catch (failure) { error ??= String(failure); }
      }
      finalResources = { device: Number(gpu !== null), pipeline: Number(pipeline !== null), canvas: Number(canvas.isConnected), listeners: 0, frame: 0 };
    }
    return { error, resources: finalResources, initializes, proofCalls, submissions, transitions, guards, proofs };
  };
  listen(window, 'm109:safety-trip', () => fail('independent-oracle-safety-trip'));
  try {
    gpu = await acquireGpu(); initializes++;
    pipeline = new VideoPipeline(gpu, video, canvas, new BaselineScaler('catmull-rom'));
    unwatch = watchDeviceFailures(gpu.device, fail);
    pipeline.onFrame = () => guard('submission');
    release = observeSuccessfulSubmissions(pipeline, identity, record => {
      submissions.push(record);
      if (gate.submit(record) && !captureOriginal) style('visibility', 'visible');
      window.dispatchEvent(new CustomEvent('m109:submission', { detail: record }));
    });
    guard('initial');
    const render = () => {
      frame = null;
      guard('render');
      if (!disposed) frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);
  } catch (failure) { fail(String(failure)); dispose(); throw failure; }
  return { canvas, dispose, setCaptureOriginal: (value: boolean) => { captureOriginal = value;
    style('visibility', !value && gate.visible ? 'visible' : 'hidden'); },
    snapshot: () => ({ error, reason, authorized, visible: gate.visible,
    owner, sourceGeneration, geometryGeneration, proofCalls, initializes, submissions: submissions.length,
    latest: submissions.at(-1) ?? null, transition: transitions.at(-1) ?? null }),
  guard: () => guard('explicit'), invalidate: () => { hide(); fingerprint = ''; } };
}