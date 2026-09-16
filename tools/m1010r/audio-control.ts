interface ControlRecording {
  firstFrame: number | null;
  samples: number;
  sampleRate: number;
  overflow: boolean;
  discontinuity: boolean;
  blockLengths: number[];
  pcm: ArrayBuffer;
}

export interface AudioControlResult {
  sampleRate: number;
  firstFrame: number;
  samples: number;
  blockLengths: number[];
  scheduled: { startFrame: number; referenceStart: number; samples: number; gain: number }[];
  expectedWindows: number;
  verifiedWindows: number;
  maximumSampleError: number | null;
  errors: string[];
  pcm: ArrayBuffer;
}

export async function runAudioControl(context: AudioContext, referenceUrl: string): Promise<AudioControlResult> {
  const response = await fetch(referenceUrl, { credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw new Error('Missing decoded-PCM control');
  const reference = new Float32Array(await response.arrayBuffer());
  if (context.sampleRate !== 48000 || reference.length < 256000) throw new Error('Insufficient 48kHz reference control');
  const recorder = new AudioWorkletNode(context, 'm1010r-audio', { numberOfInputs: 1, numberOfOutputs: 1,
    outputChannelCount: [2], channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'speakers' });
  const sources: AudioBufferSourceNode[] = [];
  const gains: GainNode[] = [];
  const scheduled: AudioControlResult['scheduled'] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  let deadline: ReturnType<typeof setTimeout> | null = null;
  let result: AudioControlResult | null = null;
  let failed = false;
  let failure: unknown;
  const cleanupErrors: unknown[] = [];
  try {
    const first = Math.ceil(context.currentTime * 48000) + 12000;
    const length = 8192;
    const finished = new Promise<ControlRecording>((resolve, reject) => {
      deadline = setTimeout(() => reject(new Error('Scheduled audio control timeout')), 5000);
      recorder.onprocessorerror = () => reject(new Error('Scheduled control processor error'));
      recorder.port.onmessage = (event: MessageEvent<unknown>) => {
        const value = event.data as Partial<ControlRecording> | null;
        if (!value || !(value.pcm instanceof ArrayBuffer) || typeof value.samples !== 'number' ||
          typeof value.sampleRate !== 'number' || typeof value.firstFrame !== 'number' ||
          !Array.isArray(value.blockLengths) || typeof value.overflow !== 'boolean' || typeof value.discontinuity !== 'boolean') {
          reject(new Error('Malformed scheduled control recording')); return;
        }
        resolve(value as ControlRecording);
      };
    });
    recorder.connect(context.destination);
    recorder.port.postMessage('start');
    for (const [ordinal, gainValue] of [1, 0.5, 0.75].entries()) {
      const referenceStart = (ordinal + 1) * 48000;
      const startFrame = first + ordinal * 24000 + [0, 2400, -2400][ordinal]!;
      const buffer = context.createBuffer(1, length, 48000);
      buffer.copyToChannel(reference.subarray(referenceStart, referenceStart + length), 0);
      const source = context.createBufferSource(), gain = context.createGain();
      sources.push(source); gains.push(gain);
      source.buffer = buffer; gain.gain.value = gainValue;
      source.connect(gain).connect(recorder);
      source.start(startFrame / 48000);
      scheduled.push({ startFrame, referenceStart, samples: length, gain: gainValue });
    }
    const last = scheduled.at(-1)!;
    timer = setTimeout(() => recorder.port.postMessage('finish'),
      Math.max(0, (last.startFrame + last.samples + 4800) / 48 - context.currentTime * 1000));
    const recording = await finished;
    if (recording.firstFrame === null || recording.sampleRate !== 48000 || recording.overflow || recording.discontinuity) throw new Error('Invalid scheduled sample-clock recording');
    const pcm = recording.pcm.slice(0, recording.samples * 4), observed = new Float32Array(pcm);
    let verifiedWindows = 0, expectedWindows = 0, maximumSampleError: number | null = null;
    const errors: string[] = [];
    for (const marker of scheduled) for (let offset = 256; offset + 256 < marker.samples; offset += 128) {
      expectedWindows++;
      const expectedStart = marker.startFrame - recording.firstFrame + offset;
      const referenceStart = marker.referenceStart + offset;
      let bestError = Infinity, bestShift: number | null = null, matches = 0;
      for (let shift = -24; shift <= 24; shift++) {
        let squared = 0, energy = 0;
        for (let sample = 0; sample < 256; sample++) {
          const target = reference[referenceStart + sample]! * marker.gain;
          const actual = observed[expectedStart + shift + sample];
          if (actual === undefined) { squared = Infinity; break; }
          squared += (actual - target) ** 2; energy += target ** 2;
        }
        const normalizedError = energy > 0 ? squared / energy : Infinity;
        if (normalizedError < bestError) { bestError = normalizedError; bestShift = shift; }
        if (normalizedError <= 0.000001) matches++;
      }
      if (matches !== 1 || bestShift === null || bestError > 0.000001 || Math.abs(bestShift) > 12) {
        errors.push(`Unverified scheduled waveform at ${marker.startFrame + offset}`);
      } else { verifiedWindows++; maximumSampleError = Math.max(maximumSampleError ?? 0, Math.abs(bestShift)); }
    }
    result = { sampleRate: recording.sampleRate, firstFrame: recording.firstFrame, samples: recording.samples,
      blockLengths: recording.blockLengths, scheduled, expectedWindows, verifiedWindows, maximumSampleError, errors, pcm };
  } catch (error) {
    failed = true; failure = error;
  } finally {
    if (timer !== null) clearTimeout(timer);
    if (deadline !== null) clearTimeout(deadline);
    for (const source of sources) {
      try { source.stop(); } catch (error) { cleanupErrors.push(error); }
      try { source.disconnect(); } catch (error) { cleanupErrors.push(error); }
    }
    for (const gain of gains) { try { gain.disconnect(); } catch (error) { cleanupErrors.push(error); } }
    recorder.onprocessorerror = null; recorder.port.onmessage = null; recorder.port.close(); recorder.disconnect();
  }
  if (failed) throw failure;
  if (cleanupErrors.length === 1) throw cleanupErrors[0];
  if (cleanupErrors.length) throw new AggregateError(cleanupErrors, cleanupErrors.map(String).join('; '));
  if (!result) throw new Error('Missing audio-control result');
  return result;
}