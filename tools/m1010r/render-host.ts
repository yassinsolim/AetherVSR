export type RenderAbortReason = 'WATCHDOG_ABORT' | 'AUDIO_CONTEXT_SUSPENDED' |
  'PROCESSOR_ERROR' | 'HOST_ABORT' | 'MALFORMED_RESULT';

export type RenderCompletion = 'RENDER_TARGET_REACHED' | RenderAbortReason | 'DISCONTINUITY' | 'OVERFLOW';

export interface RenderRecording {
  type: 'render-recording';
  completionReason: RenderCompletion;
  requestedEndFrame: number;
  actualObservedEndFrame: number | null;
  firstFrame: number | null;
  processedSamples: number;
  processedBlocks: number;
  sampleRate: number;
  blockLengths: ArrayBuffer;
  heartbeatCount: number;
  overflow: boolean;
  discontinuity: boolean;
  pcm: ArrayBuffer;
}

export interface ReceiveObservation {
  hostBefore: number;
  hostAfter: number;
  contextBefore: number;
  contextAfter: number;
}

export interface RenderHeartbeat {
  type: 'render-heartbeat';
  currentFrame: number;
  actualObservedEndFrame: number;
  sampleRate: number;
  processedBlocks: number;
  processedSamples: number;
  blockLength: number;
  state: 'RECORDING';
  heartbeatOrdinal: number;
}

export interface ReceivedHeartbeat extends RenderHeartbeat, ReceiveObservation {}

export interface ContextEvent extends ReceiveObservation {
  state: string;
}

export interface HostRecording {
  terminal: RenderRecording | null;
  terminalObservation: ReceiveObservation | null;
  heartbeats: ReceivedHeartbeat[];
  contextEvents: ContextEvent[];
  watchdogFired: boolean;
  errors: string[];
  timedOut: boolean;
}

const abortReasons: readonly string[] = [
  'WATCHDOG_ABORT', 'AUDIO_CONTEXT_SUSPENDED', 'PROCESSOR_ERROR', 'HOST_ABORT', 'MALFORMED_RESULT',
];
const completions: readonly string[] = ['RENDER_TARGET_REACHED', ...abortReasons, 'DISCONTINUITY', 'OVERFLOW'];
const integer = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
const nullableFrame = (value: unknown): value is number | null => value === null || integer(value);
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

function recording(value: Record<string, unknown>, target: number): value is Record<string, unknown> & RenderRecording {
  if (value.type !== 'render-recording' || typeof value.completionReason !== 'string' || !completions.includes(value.completionReason) ||
    value.requestedEndFrame !== target || !nullableFrame(value.actualObservedEndFrame) || !nullableFrame(value.firstFrame) ||
    !integer(value.processedSamples) || !integer(value.processedBlocks) || !integer(value.heartbeatCount) ||
    value.sampleRate !== 48000 || typeof value.overflow !== 'boolean' || typeof value.discontinuity !== 'boolean' ||
    !(value.pcm instanceof ArrayBuffer) || !(value.blockLengths instanceof ArrayBuffer)) return false;
  try {
    return new Float32Array(value.pcm).length >= value.processedSamples &&
      new Uint32Array(value.blockLengths).length >= value.processedBlocks;
  } catch { return false; }
}

function heartbeat(value: Record<string, unknown>): value is Record<string, unknown> & RenderHeartbeat {
  return value.type === 'render-heartbeat' && value.state === 'RECORDING' && value.sampleRate === 48000 &&
    integer(value.currentFrame) && integer(value.actualObservedEndFrame) && integer(value.processedBlocks) &&
    integer(value.processedSamples) && integer(value.blockLength) && integer(value.heartbeatOrdinal);
}

export function beginRenderRecording(context: AudioContext, targetEndFrame: number, watchdogMs: number): {
  node: AudioWorkletNode;
  finished: Promise<HostRecording>;
  abort(reason: RenderAbortReason): void;
  dispose(): void;
} {
  if (context.state !== 'running') throw new Error('Render recording requires an already resumed AudioContext');
  if (context.sampleRate !== 48000) throw new Error('Render recording requires 48000 Hz');
  if (!integer(targetEndFrame) || targetEndFrame === 0) throw new Error('A positive integer targetEndFrame is required');
  if (!Number.isFinite(watchdogMs) || watchdogMs <= 0 || watchdogMs > 2147483647) throw new Error('Invalid watchdog duration');

  const node = new AudioWorkletNode(context, 'm1010ri-render', { numberOfInputs: 1, numberOfOutputs: 1,
    outputChannelCount: [2], channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'speakers',
    processorOptions: { targetEndFrame } });
  const result: HostRecording = { terminal: null, terminalObservation: null, heartbeats: [], contextEvents: [],
    watchdogFired: false, errors: [], timedOut: false };
  let resolveFinished!: (value: HostRecording) => void;
  const finished = new Promise<HostRecording>(resolve => { resolveFinished = resolve; });
  let watchdog: ReturnType<typeof setTimeout> | null = null;
  let drain: ReturnType<typeof setTimeout> | null = null;
  let settled = false, disposed = false, portClosed = false;
  const requested = new Set<RenderAbortReason>();
  const error = (message: string): void => { if (!result.errors.includes(message)) result.errors.push(message); };
  const observe = (): ReceiveObservation => {
    const hostBefore = performance.now(), contextBefore = context.currentTime;
    const contextAfter = context.currentTime, hostAfter = performance.now();
    return { hostBefore, hostAfter, contextBefore, contextAfter };
  };
  const clearWatchdog = (): void => {
    if (watchdog !== null) clearTimeout(watchdog);
    watchdog = null;
  };
  const detachState = (): void => {
    context.removeEventListener('statechange', onStateChange);
    node.onprocessorerror = null;
  };
  const closePort = (): void => {
    if (portClosed) return;
    portClosed = true;
    try { node.port.close(); } catch (failure) { error(`Port close: ${String(failure)}`); }
  };
  const settle = (): void => {
    if (settled) return;
    settled = true;
    clearWatchdog();
    if (drain !== null) clearTimeout(drain);
    drain = null;
    detachState();
    node.port.onmessage = null;
    node.port.onmessageerror = null;
    if (disposed) closePort();
    resolveFinished(result);
  };
  const abort = (reason: RenderAbortReason): void => {
    if (settled) return;
    if (!abortReasons.includes(reason)) throw new Error('Invalid render abort reason');
    if (reason === 'WATCHDOG_ABORT') result.watchdogFired = true;
    error(reason);
    if (drain === null) drain = setTimeout(() => {
      if (settled) return;
      result.timedOut = true;
      error('No render-recording response within 2000 ms; PCM is missing');
      settle();
    }, 2000);
    if (requested.has(reason)) return;
    requested.add(reason);
    try { node.port.postMessage({ type: 'snapshot-and-abort', reason }); }
    catch (failure) { error(`Snapshot request: ${String(failure)}`); }
  };
  function onStateChange(): void {
    if (settled || disposed) return;
    const observation = observe(), state: string = context.state;
    result.contextEvents.push({ state, ...observation });
    if (state !== 'running') abort('AUDIO_CONTEXT_SUSPENDED');
  }
  node.port.onmessage = (event: MessageEvent<unknown>): void => {
    if (settled) return;
    const observation = observe(), value = event.data;
    if (record(value) && heartbeat(value)) result.heartbeats.push({ ...value, ...observation });
    else if (record(value) && recording(value, targetEndFrame)) {
      result.terminal = value;
      result.terminalObservation = observation;
      settle();
    } else abort('MALFORMED_RESULT');
  };
  node.port.onmessageerror = (): void => { if (!settled) abort('MALFORMED_RESULT'); };
  node.onprocessorerror = (event: Event): void => {
    if (settled || disposed) return;
    const details = ['message', 'filename', 'lineno', 'colno', 'error'].map(key => {
      const value: unknown = Reflect.get(event, key);
      const detail = value instanceof Error ? value.message : typeof value === 'string' || typeof value === 'number' ? String(value) : '';
      return detail ? `${key}=${detail}` : '';
    }).filter(Boolean).join(' ');
    error(`PROCESSOR_ERROR${details ? `: ${details}` : ''}`);
    abort('PROCESSOR_ERROR');
  };
  context.addEventListener('statechange', onStateChange);
  watchdog = setTimeout(() => { if (!settled) abort('WATCHDOG_ABORT'); }, watchdogMs);
  onStateChange();

  return { node, finished, abort, dispose(): void {
    if (disposed) return;
    disposed = true;
    clearWatchdog();
    detachState();
    if (!settled && requested.size === 0) abort('HOST_ABORT');
    try { node.disconnect(); } catch (failure) { error(`Node disconnect: ${String(failure)}`); }
    if (settled) closePort();
  } };
}