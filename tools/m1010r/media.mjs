import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, relative, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mediaRecipe } from '../m1010/fixtures.mjs';

export const SAMPLE_RATE = 48000;
export const COUNTER_WIDTH = 448;
export const SIGNAL_SEED = 0x6d2b79f5;
export const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const root = fileURLToPath(new URL('../../', import.meta.url));
const ffmpeg = args => execFileSync('ffmpeg', args, { maxBuffer: 256 * 1024 * 1024 });
const probe = (path, selection, entries) => JSON.parse(execFileSync('ffprobe', [
  '-v', 'error', '-select_streams', selection, '-show_entries', entries, '-of', 'json', path,
], { maxBuffer: 32 * 1024 * 1024 }));

export function signalPcm(seconds) {
  if (!Number.isInteger(seconds) || seconds < 1 || seconds > 1000) throw new Error('Invalid signal duration');
  const bytes = Buffer.alloc(seconds * SAMPLE_RATE * 2);
  let state = SIGNAL_SEED;
  for (let sample = 0; sample < seconds * SAMPLE_RATE; sample++) {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    const noise = Math.round(((state >>> 0) / 4294967296 * 2 - 1) * 2048);
    const phase = sample % SAMPLE_RATE;
    bytes.writeInt16LE(phase >= 12000 && phase < 12048 ? 24576 : noise, sample * 2);
  }
  return bytes;
}

export function replayRecipe(fps, output, pcm, seconds) {
  if (![30, 60].includes(fps) || !Number.isInteger(seconds) || seconds < 1 || seconds * fps > 65536) throw new Error('Invalid replay fixture');
  const recipe = mediaRecipe({ width: 1280, height: 720, fps }, output, seconds);
  const videoInput = recipe.indexOf('-i') + 1;
  const cells = [`,drawbox=x=0:y=0:w=${COUNTER_WIDTH}:h=80:color=black:t=fill`];
  for (const row of [16, 48]) {
    cells.push(`,drawbox=x=0:y=${row}:w=16:h=16:color=white:t=fill`);
    for (let bit = 0; bit < 16; bit++) cells.push(`,drawbox=x=${(bit + 2) * 24}:y=${row}:w=16:h=16:color=white:t=fill:enable='${row === 16 ? 'gt' : 'eq'}(bitand(n,${2 ** bit}),0)'`);
  }
  recipe[videoInput] += cells.join('');
  const audioInput = recipe.indexOf('-i', videoInput) - 2;
  recipe.splice(audioInput, 4, '-f', 's16le', '-ar', String(SAMPLE_RATE), '-ac', '1', '-i', pcm);
  return recipe;
}

export function decodeCounter(top, bottom) {
  if (top.length !== COUNTER_WIDTH || bottom.length !== COUNTER_WIDTH) throw new Error('Invalid counter extent');
  const bitAt = (row, column) => {
    const value = row[column + 8];
    if (value <= 64) return 0;
    if (value >= 192) return 1;
    throw new Error('Ambiguous counter cell');
  };
  if (bitAt(top, 0) !== 1 || bitAt(bottom, 0) !== 1 || bitAt(top, 24) || bitAt(bottom, 24)) throw new Error('Counter guard mismatch');
  let identity = 0;
  for (let bit = 0; bit < 16; bit++) {
    const value = bitAt(top, (bit + 2) * 24);
    if (value === bitAt(bottom, (bit + 2) * 24)) throw new Error('Counter complement mismatch');
    identity += value * 2 ** bit;
  }
  return identity;
}

export function audioTimeline(frames, stream, decodedSamples) {
  const [numerator, denominator] = String(stream.time_base).split('/').map(Number);
  const sampleRate = Number(stream.sample_rate);
  if (!frames.length || sampleRate !== SAMPLE_RATE || !Number.isSafeInteger(numerator) || numerator <= 0 ||
    !Number.isSafeInteger(denominator) || denominator <= 0 || !Number.isSafeInteger(decodedSamples)) throw new Error('Invalid decoded audio timeline');
  let counted = 0;
  const firstPts = Number(frames[0].pts);
  if (!Number.isSafeInteger(firstPts)) throw new Error('Missing audio PTS');
  for (const frame of frames) {
    const pts = Number(frame.pts), samples = Number(frame.nb_samples);
    if (!Number.isSafeInteger(pts) || !Number.isSafeInteger(samples) || samples <= 0 ||
      (pts - firstPts) * numerator * sampleRate !== counted * denominator) throw new Error('Discontinuous decoded audio PTS');
    counted += samples;
  }
  if (counted !== decodedSamples) throw new Error('PCM length does not match decoded PTS frames');
  return { sampleRate, decodedSamples, firstPts, timeBase: stream.time_base, firstSampleMediaTime: firstPts * numerator / denominator,
    endExclusiveMediaTime: firstPts * numerator / denominator + counted / sampleRate, decodedFrames: frames.length };
}

export function inspectReplayMedia(path, fps, seconds, { audioShiftSeconds = 0 } = {}) {
  const execute = args => ffmpeg(['-nostdin', '-hide_banner', '-v', 'error', '-i', path, ...args]);
  const row = vertical => execute(['-an', '-vf', `extractplanes=y,crop=${COUNTER_WIDTH}:1:0:${vertical}`, '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1']);
  const top = row(24), bottom = row(56), count = fps * seconds;
  if (top.length !== count * COUNTER_WIDTH || bottom.length !== top.length) throw new Error('Decoded video extent/count mismatch');
  const video = probe(path, 'v:0', 'frame=pts:stream=time_base,width,height,avg_frame_rate');
  const [numerator, denominator] = video.streams[0].time_base.split('/').map(Number);
  if (video.frames.length !== count || video.streams[0].width !== 1280 || video.streams[0].height !== 720) throw new Error('Video probe mismatch');
  for (let index = 0; index < count; index++) {
    const identity = decodeCounter(top.subarray(index * COUNTER_WIDTH, (index + 1) * COUNTER_WIDTH), bottom.subarray(index * COUNTER_WIDTH, (index + 1) * COUNTER_WIDTH));
    if (identity !== index || Number(video.frames[index].pts) * numerator * fps !== index * denominator) throw new Error(`Video identity/PTS mismatch: ${index}`);
  }
  const pcm = execute(['-map', '0:a:0', '-ac', '1', '-ar', String(SAMPLE_RATE), '-c:a', 'pcm_f32le', '-f', 'f32le', 'pipe:1']);
  const audio = probe(path, 'a:0', 'frame=pts,nb_samples:stream=time_base,sample_rate,duration_ts,start_pts');
  const timeline = audioTimeline(audio.frames, audio.streams[0], pcm.length / 4);
  const impulses = [];
  for (let marker = 0; marker < seconds; marker++) {
    const intended = marker + 0.25 + audioShiftSeconds;
    const low = Math.max(0, Math.floor((intended - timeline.firstSampleMediaTime - 0.01) * SAMPLE_RATE));
    const high = Math.min(pcm.length / 4, Math.ceil((intended - timeline.firstSampleMediaTime + 0.01) * SAMPLE_RATE));
    let onset = null;
    for (let sample = low; sample < high; sample++) if (Math.abs(pcm.readFloatLE(sample * 4)) >= 0.25) { onset = sample; break; }
    if (onset === null) throw new Error(`Encoded impulse missing: ${marker}`);
    impulses.push({ id: marker, inputSample: marker * SAMPLE_RATE + 12000, decodedThresholdSample: onset,
      decodedThresholdMediaTime: timeline.firstSampleMediaTime + onset / SAMPLE_RATE });
  }
  return { pcm, report: { video: { fps, frames: count, width: 1280, height: 720, timeBase: video.streams[0].time_base,
    allIdentitiesAndPtsExact: true, rowsSha256: digest(Buffer.concat([top, bottom])) }, audio: { ...timeline, impulses,
    decodedPcmBytes: pcm.length, decodedPcmSha256: digest(pcm), ptsFramesSha256: digest(JSON.stringify(audio.frames)),
    interpretation: 'PCM sample zero is the first decoded frame PTS after demux/edit/skip handling; decoded impulse threshold is not speaker onset.' } } };
}

export function inspectShiftControls(manifest, directory) {
  const output = resolve(directory);
  if (!output.startsWith(resolve(root, '.cache/m1010r') + '/')) throw new Error('Shift controls must stay ignored');
  mkdirSync(output, { recursive: true });
  const controls = [];
  for (const asset of manifest.assets) for (const shift of [-0.05, 0.05]) {
    const path = join(output, `shift-${asset.fps}-${shift}.mp4`);
    const command = ['-nostdin', '-hide_banner', '-v', 'error', '-n', '-copyts', '-i', resolve(root, asset.media.path),
      '-itsoffset', String(shift), '-i', resolve(root, asset.media.path), '-map', '0:v:0', '-map', '1:a:0',
      '-c', 'copy', '-avoid_negative_ts', 'disabled', '-use_editlist', '1', path];
    ffmpeg(command);
    const inspected = inspectReplayMedia(path, asset.fps, manifest.seconds, { audioShiftSeconds: shift });
    const original = readFileSync(resolve(root, asset.pcm.path));
    const referenceFirst = asset.validation.audio.firstSampleMediaTime;
    const shiftedFirst = inspected.report.audio.firstSampleMediaTime;
    const referenceStart = SAMPLE_RATE * 0.75;
    const shiftedStart = Math.round((referenceFirst + referenceStart / SAMPLE_RATE + shift - shiftedFirst) * SAMPLE_RATE);
    const matchedSamples = 8192;
    if (!original.subarray(referenceStart * 4, (referenceStart + matchedSamples) * 4).equals(
      inspected.pcm.subarray(shiftedStart * 4, (shiftedStart + matchedSamples) * 4))) throw new Error('Shifted interior waveform identity differs');
    const observedShift = shiftedFirst + shiftedStart / SAMPLE_RATE - (referenceFirst + referenceStart / SAMPLE_RATE);
    if (Math.abs(observedShift - shift) > 1 / SAMPLE_RATE) throw new Error('Container normalized away the intended shift');
    const packets = probe(path, 'a:0', 'packet=pts,dts,duration,side_data_list:stream=time_base,start_pts,duration_ts');
    const bytes = readFileSync(path);
    controls.push({ fps: asset.fps, requestedAudioShiftSeconds: shift, observedAudioShiftSeconds: observedShift,
      expectedVideoMinusAudioShiftMs: -shift * 1000, videoPtsUnchanged: true, matchedInteriorSamples: matchedSamples,
      referenceStart, shiftedStart, referenceFirstMediaTime: referenceFirst, shiftedFirstMediaTime: shiftedFirst,
      decodedSampleCount: inspected.pcm.length / 4, packetProbeSha256: digest(JSON.stringify(packets)),
      firstPackets: packets.packets.slice(0, 4), impulses: inspected.report.audio.impulses,
      media: { path: relative(root, path), bytes: bytes.length, sha256: digest(bytes) }, command });
  }
  return controls;
}

export function prepareReplayMedia(directory, seconds = 70) {
  const output = resolve(directory);
  if (!output.startsWith(resolve(root, '.cache/m1010r') + '/')) throw new Error('Replay media must be in the ignored R cache');
  mkdirSync(output, { recursive: true });
  const manifestPath = join(output, 'media.json');
  if (existsSync(manifestPath)) {
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    if (manifest.seconds !== seconds || manifest.producerSha256 !== digest(readFileSync(fileURLToPath(import.meta.url)))) throw new Error('Existing media recipe pin changed');
    for (const entry of [manifest.signal, ...manifest.assets.flatMap(asset => [asset.media, asset.pcm])]) {
      const bytes = readFileSync(resolve(root, entry.path));
      if (bytes.length !== entry.bytes || digest(bytes) !== entry.sha256) throw new Error(`Changed media artifact: ${entry.path}`);
    }
    return manifest;
  }
  const evidence = path => { const bytes = readFileSync(path); return { path: relative(root, path), bytes: bytes.length, sha256: digest(bytes) }; };
  const signalPath = join(output, 'signal.s16le');
  writeFileSync(signalPath, signalPcm(seconds), { flag: 'wx' });
  const assets = [];
  for (const fps of [30, 60]) {
    const path = join(output, `replay-${fps}.mp4`), pcmPath = join(output, `decoded-${fps}.f32le`);
    const recipe = replayRecipe(fps, path, signalPath, seconds);
    ffmpeg(recipe);
    const inspected = inspectReplayMedia(path, fps, seconds);
    writeFileSync(pcmPath, inspected.pcm, { flag: 'wx' });
    assets.push({ fps, recipe, media: evidence(path), pcm: evidence(pcmPath), validation: inspected.report });
  }
  const manifest = { schemaVersion: 1, study: 'M10.10R', seconds, sampleRate: SAMPLE_RATE, seed: SIGNAL_SEED,
    producerSha256: digest(readFileSync(fileURLToPath(import.meta.url))),
    ffmpeg: execFileSync('ffmpeg', ['-version'], { encoding: 'utf8' }).split('\n')[0], signal: evidence(signalPath), assets };
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  return manifest;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const manifest = prepareReplayMedia(process.argv[2] ?? '.cache/m1010r/media-01', Number(process.argv[3] ?? 70));
  console.log(JSON.stringify({ seconds: manifest.seconds, assets: manifest.assets.map(({ fps, media, validation }) => ({ fps, media, video: validation.video, audio: {
    firstSampleMediaTime: validation.audio.firstSampleMediaTime, decodedSamples: validation.audio.decodedSamples, impulses: validation.audio.impulses.length,
  } })) }, null, 2));
}