import { beginRenderRecording, type HostRecording, type RenderRecording } from './render-host.js';

export interface SignalRegion { startFrame: number; referenceStart: number; samples: number; gain: number }
export interface RenderSummary extends Omit<HostRecording, 'terminal'> {
  terminal: (Omit<RenderRecording, 'pcm' | 'blockLengths'> & { blockLengths: number[] }) | null;
}

export function summarizeRender(record: HostRecording): RenderSummary {
  if (!record.terminal) return { ...record, terminal: null };
  const { pcm, blockLengths, ...terminal } = record.terminal;
  if (pcm.byteLength < terminal.processedSamples * 4 || blockLengths.byteLength < terminal.processedBlocks * 4) throw new Error('Truncated worklet buffers');
  return { ...record, terminal: { ...terminal, blockLengths: [...new Uint32Array(blockLengths, 0, terminal.processedBlocks)] } };
}

export function renderIntegrity(record: RenderSummary, requiredStart: number, requiredEnd: number): string[] {
  const terminal = record.terminal, errors = [...record.errors];
  if (!terminal) return [...errors, 'Missing render terminal'];
  const blocks = terminal.blockLengths, first = terminal.firstFrame, end = terminal.actualObservedEndFrame;
  if (terminal.completionReason !== 'RENDER_TARGET_REACHED' || record.watchdogFired || record.timedOut) errors.push('Render completion not eligible');
  if (terminal.sampleRate !== 48000 || terminal.overflow || terminal.discontinuity) errors.push('Render state/rate invalid');
  if (first === null || end === null || first > requiredStart || end < requiredEnd || terminal.processedSamples !== end - first) errors.push('Required render interval not covered');
  if (!blocks.length || blocks.length !== terminal.processedBlocks || blocks.some(length => !Number.isSafeInteger(length) || length <= 0) ||
    blocks.reduce((sum, length) => sum + length, 0) !== terminal.processedSamples) errors.push('Block count/sum inconsistent');
  const last = blocks.at(-1) ?? 0, maximum = blocks.reduce((largest, length) => Math.max(largest, length), 0);
  if (end === null || end < terminal.requestedEndFrame || end - terminal.requestedEndFrame >= last) errors.push('Terminal target not in last block');
  if (record.heartbeats.length !== terminal.heartbeatCount) errors.push('Missing heartbeats');
  let previousEnd: number | null = null;
  for (const [index, heartbeat] of record.heartbeats.entries()) {
    if (heartbeat.heartbeatOrdinal !== index + 1 || heartbeat.actualObservedEndFrame !== heartbeat.currentFrame + heartbeat.blockLength ||
      first === null || heartbeat.processedSamples !== heartbeat.actualObservedEndFrame - first || heartbeat.processedBlocks > blocks.length ||
      blocks.slice(0, heartbeat.processedBlocks).reduce((sum, length) => sum + length, 0) !== heartbeat.processedSamples ||
      blocks[heartbeat.processedBlocks - 1] !== heartbeat.blockLength || end === null || heartbeat.actualObservedEndFrame > end ||
      (previousEnd !== null && heartbeat.actualObservedEndFrame - previousEnd < 12000)) errors.push('Heartbeat metadata inconsistent');
    previousEnd = heartbeat.actualObservedEndFrame;
  }
  const clockRows = [...record.heartbeats.map(heartbeat => ({ ...heartbeat, endFrame: heartbeat.actualObservedEndFrame })),
    ...(record.terminalObservation && end !== null ? [{ ...record.terminalObservation, endFrame: end }] : [])];
  let previousHost = -Infinity, previousAudio = -Infinity;
  for (const row of clockRows) {
    if (![row.hostBefore, row.hostAfter, row.contextBefore, row.contextAfter].every(Number.isFinite) ||
      row.hostBefore > row.hostAfter || row.contextBefore > row.contextAfter || row.hostBefore < previousHost ||
      row.contextBefore < previousAudio || row.contextBefore * 48000 < row.endFrame - maximum * 2) errors.push('Cross-clock relation invalid');
    previousHost = row.hostAfter; previousAudio = row.contextAfter;
  }
  if (!record.terminalObservation || record.contextEvents.some(event => event.state !== 'running')) errors.push('Missing completion clock or unexpected context state');
  return [...new Set(errors)];
}

export function verifySignalWindows(reference: Float32Array, observed: Float32Array, firstFrame: number, scheduled: SignalRegion[]) {
  let verifiedWindows = 0, expectedWindows = 0, maximumSampleError: number | null = null;
  const errors: string[] = [];
  for (const region of scheduled) for (let offset = 256; offset + 256 < region.samples; offset += 128) {
    expectedWindows++;
    const expected = region.startFrame - firstFrame + offset;
    let matches = 0, matchedShift: number | null = null;
    for (let shift = -24; shift <= 24; shift++) {
      let energy = 0, squared = 0;
      for (let sample = 0; sample < 256; sample++) {
        const target = reference[region.referenceStart + offset + sample]! * region.gain;
        const actual = observed[expected + shift + sample];
        if (actual === undefined || !Number.isFinite(actual)) { squared = Infinity; break; }
        squared += (actual - target) ** 2; energy += target ** 2;
      }
      if (energy > 0 && squared / energy <= 0.000001) { matches++; matchedShift = shift; }
    }
    if (matches !== 1 || matchedShift === null || Math.abs(matchedShift) > 12) errors.push(`Unverified signal window ${region.startFrame + offset}`);
    else { verifiedWindows++; maximumSampleError = Math.max(maximumSampleError ?? 0, Math.abs(matchedShift)); }
  }
  if (!observed.every(Number.isFinite)) errors.push('Nonfinite PCM');
  return { expectedWindows, verifiedWindows, maximumSampleError, errors };
}

export async function runRenderControl(context: AudioContext, referenceUrl: string, signal?: AbortSignal) {
  const deadline = AbortSignal.timeout(5000);
  const response = await fetch(referenceUrl, { credentials: 'omit', redirect: 'error', signal: signal ? AbortSignal.any([signal, deadline]) : deadline });
  if (!response.ok) throw new Error('Missing instrument PCM reference');
  const reference = new Float32Array(await response.arrayBuffer());
  if (reference.length < 256000 || !reference.every(Number.isFinite)) throw new Error('Invalid instrument PCM reference');
  signal?.throwIfAborted();
  const epoch = Math.ceil(context.currentTime * 48000), first = epoch + 12000;
  const targetEndFrame = first + 58592;
  const scheduled: SignalRegion[] = [1, 0.5, 0.75].map((gain, index) => ({ startFrame: first + [0, 26400, 45600][index]!,
    referenceStart: (index + 1) * 48000, samples: 8192, gain }));
  const recorder = beginRenderRecording(context, targetEndFrame, 15000);
  const cancel = () => recorder.abort('HOST_ABORT');
  signal?.addEventListener('abort', cancel, { once: true });
  const sources: AudioBufferSourceNode[] = [], gains: GainNode[] = [], errors: string[] = [];
  let recording: HostRecording;
  try {
    if (signal?.aborted) cancel();
    for (const region of scheduled) {
      const buffer = context.createBuffer(1, region.samples, 48000);
      buffer.copyToChannel(reference.subarray(region.referenceStart, region.referenceStart + region.samples), 0);
      const source = context.createBufferSource(); sources.push(source);
      const gain = context.createGain(); gains.push(gain);
      source.buffer = buffer; gain.gain.value = region.gain;
      source.connect(gain).connect(recorder.node); source.start(region.startFrame / 48000);
    }
    recorder.node.connect(context.destination);
    recording = await recorder.finished;
  } catch (error) {
    errors.push(String(error)); recorder.abort('HOST_ABORT'); recording = await recorder.finished;
  } finally {
    signal?.removeEventListener('abort', cancel);
    for (const source of sources) {
      try { source.stop(); } catch (error) { errors.push(String(error)); }
      try { source.disconnect(); } catch (error) { errors.push(String(error)); }
    }
    for (const gain of gains) { try { gain.disconnect(); } catch (error) { errors.push(String(error)); } }
    recorder.dispose();
  }
  const render = summarizeRender(recording), terminal = recording.terminal;
  const pcm = terminal?.pcm.slice(0, terminal.processedSamples * 4) ?? null;
  const windows = terminal && pcm && terminal.firstFrame !== null ? verifySignalWindows(reference, new Float32Array(pcm), terminal.firstFrame, scheduled)
    : { expectedWindows: 180, verifiedWindows: 0, maximumSampleError: null, errors: ['Missing scheduled PCM'] };
  errors.push(...renderIntegrity(render, first, targetEndFrame), ...windows.errors);
  return { epochFrame: epoch, targetEndFrame, scheduled, postSignalTailFrames: 4800, render,
    sampleRate: 48000, firstFrame: terminal?.firstFrame ?? null, samples: terminal?.processedSamples ?? 0,
    expectedWindows: windows.expectedWindows, verifiedWindows: windows.verifiedWindows,
    maximumSampleError: windows.maximumSampleError, errors, pcm };
}