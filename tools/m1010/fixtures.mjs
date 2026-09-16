import http from 'node:http';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = new URL('../../', import.meta.url);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

export function parseRange(header, size) {
  if (!Number.isSafeInteger(size) || size <= 0) return null;
  if (header === undefined) return { start: 0, end: size - 1, partial: false };
  if (typeof header !== 'string') return null;
  const match = /^bytes=(\d+)-(\d*)$/.exec(header);
  if (!match) return null;
  const start = Number(match[1]);
  const requestedEnd = match[2] === '' ? size - 1 : Number(match[2]);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(requestedEnd)
    || start >= size || requestedEnd < start) return null;
  return { start, end: Math.min(requestedEnd, size - 1), partial: true };
}

export function mediaRecipe({ width, height, fps, fragmented = false }, output, duration = 4) {
  const bands = `color=c=0x3050d0:s=${width}x${height}:r=${fps},drawbox=x=0:y=0:w=iw/3:h=ih:color=0xd04040:t=fill,drawbox=x=iw/3:y=0:w=iw/3:h=ih:color=0x30b060:t=fill`;
  const flash = `,drawbox=color=white:t=fill:enable='lt(mod(n,${fps}),${fps / 10})'`;
  return ['-nostdin', '-hide_banner', '-v', 'error', '-n', '-f', 'lavfi', '-i', bands + flash,
    '-f', 'lavfi', '-i', "aevalsrc='0.35*sin(2*PI*880*t)*lt(mod(n,48000),4800)':s=48000",
    '-t', String(duration), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-preset', 'fast',
    '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level:v', '5.2', '-bf', '0', '-g', String(fps),
    '-keyint_min', String(fps), '-sc_threshold', '0', '-c:a', 'aac', '-b:a', '96k', '-ar', '48000',
    '-map_metadata', '-1', '-fflags', '+bitexact', '-movflags',
    fragmented ? 'frag_keyframe+delay_moov+default_base_moof' : '+faststart', output];
}

export function timingRecipe(fps, output, seconds = 70) {
  if (![30, 60].includes(fps) || seconds <= 0 || seconds * fps > 8192) throw new Error('Invalid timing fixture');
  const recipe = mediaRecipe({ width: 1280, height: 720, fps }, output, seconds);
  const firstInput = recipe.indexOf('-i') + 1;
  const cells = [',drawbox=x=0:y=0:w=360:h=80:color=black:t=fill'];
  for (const row of [16, 48]) {
    cells.push(`,drawbox=x=0:y=${row}:w=16:h=16:color=white:t=fill`);
    for (let bit = 0; bit < 13; bit++) cells.push(`,drawbox=x=${(bit + 2) * 24}:y=${row}:w=16:h=16:color=white:t=fill:enable='${row === 16 ? 'gt' : 'eq'}(bitand(n,${2 ** bit}),0)'`);
  }
  recipe[firstInput] += cells.join('');
  recipe[recipe.indexOf('-i', firstInput) + 1] = "aevalsrc='0.35*sin(2*PI*(600+20*floor(t))*t)*lt(mod(n,48000),4800)':s=48000";
  return recipe;
}

export function decodeTimingCounter(top, bottom) {
  if (top.length !== 360 || bottom.length !== 360) throw new Error('Missing counter row');
  const level = value => value > 192 ? 1 : value < 64 ? 0 : null;
  for (const row of [top, bottom]) if (level(row[8]) !== 1 || level(row[32]) !== 0) throw new Error('Counter guard mismatch');
  let value = 0;
  for (let bit = 0; bit < 13; bit++) {
    const upper = level(top[(bit + 2) * 24 + 8]), lower = level(bottom[(bit + 2) * 24 + 8]);
    if (upper === null || lower === null || upper + lower !== 1) throw new Error('Counter complement mismatch');
    value += upper * 2 ** bit;
  }
  return value;
}

export function decodeTimingAudio(bytes, sampleRate = 48000) {
  if (bytes.length % 4 !== 0 || sampleRate !== 48000) throw new Error('Expected mono 48kHz float32 PCM');
  const sample = index => bytes.readFloatLE(index * 4), count = bytes.length / 4, pulses = [];
  let first = null, last = null;
  const finish = () => {
    if (first === null) return;
    const crossings = [];
    for (let index = first + 481; index < last - 480; index++) {
      const before = sample(index - 1), after = sample(index);
      if (before <= 0 && after > 0) crossings.push(index - 1 - before / (after - before));
    }
    const hz = crossings.length > 1 ? (crossings.length - 1) * sampleRate / (crossings.at(-1) - crossings[0]) : null;
    const id = hz === null ? null : Math.round((hz - 600) / 20);
    pulses.push({ firstSample: first, lastSample: last, firstThresholdSeconds: first / sampleRate,
      lastThresholdSeconds: last / sampleRate, hz, id, frequencyErrorHz: hz === null ? null : Math.abs(hz - (600 + 20 * id)) });
    first = last = null;
  };
  for (let index = 0; index < count; index++) {
    if (Math.abs(sample(index)) > 0.1) { if (first === null) first = index; last = index; }
    else if (last !== null && index - last >= 960) finish();
  }
  finish(); return { samples: count, sampleRate, threshold: 0.1, separationSamples: 960, pulses };
}

export function inspectTimingMedia(path, fps, seconds = 70) {
  const execute = args => execFileSync('ffmpeg', ['-nostdin', '-hide_banner', '-v', 'error', '-i', path, ...args], { maxBuffer: 64 * 1024 * 1024, timeout: 120000 });
  const row = vertical => execute(['-an', '-vf', `extractplanes=y,crop=360:1:0:${vertical}`, '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1']);
  const top = row(24), bottom = row(56), count = seconds * fps;
  if (top.length !== count * 360 || bottom.length !== top.length) throw new Error('Timing frame count mismatch');
  const flashes = execute(['-an', '-vf', 'extractplanes=y,crop=1:1:100:200', '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1']);
  if (flashes.length !== count) throw new Error('Missing decoded flash pixels');
  const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', path], { maxBuffer: 8 * 1024 * 1024 }));
  if (probe.frames.length !== count) throw new Error('Missing timing frame PTS');
  const frames = probe.frames.map((frame, index) => {
    const value = decodeTimingCounter(top.subarray(index * 360, (index + 1) * 360), bottom.subarray(index * 360, (index + 1) * 360));
    const pts = Number(frame.best_effort_timestamp_time);
    if (value !== index || Math.abs(pts - index / fps) > 0.000002) throw new Error(`Timing frame identity/PTS mismatch at ${index}`);
    const flash = flashes[index] > 192;
    if (flash !== (index % fps < fps / 10)) throw new Error(`Timing flash mismatch at ${index}`);
    return { frame: value, pts, flash, flashLuma: flashes[index] };
  });
  const pcm = execute(['-vn', '-ac', '1', '-ar', '48000', '-f', 'f32le', 'pipe:1']), audio = decodeTimingAudio(pcm);
  if (audio.pulses.length !== Math.ceil(seconds) || audio.pulses.some((pulse, index) => pulse.id !== index || pulse.frequencyErrorHz > 2 || Math.abs(pulse.firstThresholdSeconds - index) > 0.02)) throw new Error('Encoded audio marker identity mismatch');
  const audioProbe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'a:0', '-show_packets', '-show_frames', '-read_intervals', '%+#4', '-of', 'json', path], { maxBuffer: 1024 * 1024 }));
  return { frames, audio, audioProbe, decodedPcm: { bytes: pcm.length, sha256: hash(pcm) },
    decoding: 'FFmpeg container-timeline decode including edit list/skip metadata; threshold crossings are not physical sound onset.',
    ffmpeg: execFileSync('ffmpeg', ['-version'], { encoding: 'utf8' }).split('\n')[0] };
}

export function prepareTimingMedia() {
  const directory = new URL('.cache/m1010/media/', root); mkdirSync(directory, { recursive: true });
  return Object.fromEntries([30, 60].map(fps => {
    const path = fileURLToPath(new URL(`timing720p${fps}-v1.mp4`, directory)), recipe = timingRecipe(fps, path), recipeSha256 = hash(JSON.stringify(recipe));
    if (!existsSync(path)) execFileSync('ffmpeg', recipe, { timeout: 120000, stdio: ['ignore', 'pipe', 'pipe'] });
    const bytes = readFileSync(path), sha256 = hash(bytes), analysisPath = `${path}.analysis.json`;
    if (!existsSync(analysisPath)) writeFileSync(analysisPath, JSON.stringify({ recipeSha256, sha256, inspection: inspectTimingMedia(path, fps) }), { flag: 'wx' });
    const analysis = JSON.parse(readFileSync(analysisPath));
    if (analysis.recipeSha256 !== recipeSha256 || analysis.sha256 !== sha256) throw new Error('Timing fixture pin mismatch');
    return [fps === 30 ? 'A' : 'B', { path, bytes: bytes.length, sha256, recipe, recipeSha256, fps, width: 1280, height: 720, seconds: 70,
      analysis: { path: analysisPath, bytes: readFileSync(analysisPath).length, sha256: hash(readFileSync(analysisPath)) } }];
  }));
}

export function splitFragments(bytes) {
  const starts = [];
  let offset = 0, tail = bytes.length, moov = false;
  while (offset < bytes.length) {
    if (offset + 8 > bytes.length) throw new Error('Truncated MP4 box');
    const size = bytes.readUInt32BE(offset), type = bytes.toString('ascii', offset + 4, offset + 8);
    if (size < 8 || offset + size > bytes.length) throw new Error('Invalid MP4 box size');
    if (type === 'moov') moov = true;
    if (type === 'moof') { if (!moov) throw new Error('Missing init'); starts.push(offset); }
    if (type === 'mfra') tail = offset;
    offset += size;
  }
  if (!starts.length) throw new Error('Missing media fragments');
  return { init: { start: 0, end: starts[0] - 1 }, fragments: starts.map((start, index) => ({
    start, end: (starts[index + 1] ?? tail) - 1,
  })) };
}

export function prepareMedia() {
  const directory = new URL('.cache/m1010/media/', root);
  mkdirSync(directory, { recursive: true });
  const definitions = { A: [1280, 720, 30], B: [1280, 720, 60], C: [1920, 1080, 30],
    low: [640, 360, 30], high: [1280, 720, 30] };
  return Object.fromEntries(Object.entries(definitions).map(([name, [width, height, fps]]) => {
    const fragmented = name === 'low' || name === 'high';
    const path = fileURLToPath(new URL(fragmented ? `${name}.mp4` : `marked${height}p${fps}.mp4`, directory));
    const recipe = mediaRecipe({ width, height, fps, fragmented }, path);
    const existed = existsSync(path), pinPath = `${path}.json`, recipeSha256 = hash(JSON.stringify(recipe));
    if (!existed) execFileSync('ffmpeg', recipe, { timeout: 120000, stdio: ['ignore', 'pipe', 'pipe'] });
    const bytes = readFileSync(path);
    const sha256 = hash(bytes);
    if (existed && existsSync(pinPath)) {
      const pin = JSON.parse(readFileSync(pinPath, 'utf8'));
      if (pin.sha256 !== sha256 || pin.recipeSha256 !== recipeSha256) throw new Error(`Media pin mismatch: ${path}`);
    }
    const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-show_streams', '-show_format',
      '-show_frames', '-read_intervals', '%+#16', '-of', 'json', path], { encoding: 'utf8', timeout: 15000 }));
    const video = probe.streams.find(stream => stream.codec_type === 'video');
    const audio = probe.streams.find(stream => stream.codec_type === 'audio');
    const config = bytes.indexOf(Buffer.from('avcC'));
    const codec = config < 0 ? '' : bytes.subarray(config + 5, config + 8).toString('hex');
    if (video?.width !== width || video?.height !== height || video?.r_frame_rate !== `${fps}/1`
      || codec !== '640034' || audio?.codec_name !== 'aac' || audio?.sample_rate !== '48000') {
      throw new Error(`Existing/generated media does not match recipe: ${path}`);
    }
    const firstFrames = Object.fromEntries(['video', 'audio'].map(type => {
      const stream = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', type === 'video' ? 'v:0' : 'a:0',
        '-show_frames', '-read_intervals', '%+#16', '-of', 'json', path], { encoding: 'utf8', timeout: 15000 }));
      return [type, stream.frames?.find(frame => frame.media_type === type) ?? null];
    }));
    if (!firstFrames.video || !firstFrames.audio || Math.abs(Number(probe.format.duration) - 4) > 0.08) throw new Error('Incomplete marker media');
    const fragments = fragmented ? splitFragments(bytes) : {};
    if (!existsSync(pinPath)) writeFileSync(pinPath, JSON.stringify({ sha256, recipeSha256 }), { flag: 'wx' });
    return [name, { path, bytes: bytes.length, sha256, recipe, recipeSha256,
      width, height, fps, mime: 'video/mp4; codecs="avc1.640034, mp4a.40.2"', probe, firstFrames,
      marker: { recipeOrigin: 'M10.9 synthetic three bands', seconds: 4, inputFirstPts: 0,
        flashFrames: fps / 10, periodFrames: fps, pulseSamples: 4800, periodSamples: 48000, hz: 880,
        timing: 'Input pulses/flash start at integer seconds. Absolute decoded PTS are in firstFrames; AAC priming/mux shifts are not physical A/V latency measurements.' },
      ...fragments }];
  }));
}

export async function startFixtures(options = {}) {
  const validated = fixtureOptions(options);
  return serveFixtures(prepareMedia(), validated);
}

function fixtureOptions({ ports = [5204, 5205], extensionOrigins = [] } = {}) {
  if (!Array.isArray(ports) || ports.length !== 2 || ports.some(port => !Number.isInteger(port) || port < 0 || port > 65535)
    || (ports[0] !== 0 && ports[0] === ports[1]) || !Array.isArray(extensionOrigins)
    || extensionOrigins.some(origin => !/^chrome-extension:\/\/[a-p]{32}$/.test(origin))) throw new Error('Invalid fixture origins');
  return { ports, extensionOrigins };
}

export async function serveFixtures(media, options = {}) {
  const { ports, extensionOrigins } = fixtureOptions(options);
  const servers = [], origins = [], requests = [], counters = { blocked: 0, redirect: 0, mimeBad: 0, oversize: 0 };
  const close = async () => { await Promise.all(servers.map(server => new Promise(resolve => {
    server.close(resolve); server.closeAllConnections();
  }))); };
  const routes = new Map();
  for (const name of ['A', 'B', 'C']) for (const mode of ['same', 'cors', 'nocors', 'auth/omit', 'auth/include']) {
    routes.set(`/${mode}/${name}.mp4`, { ...media[name], mode });
  }
  for (const name of ['low', 'high']) {
    const entry = media[name];
    if (!entry) continue;
    routes.set(`/mse/${name}/init.mp4`, { ...entry, slice: entry.init });
    entry.fragments.forEach((slice, index) => routes.set(`/mse/${name}/${index}.m4s`, { ...entry, slice }));
  }
  const pages = new Map([['/source.html', ['text/html; charset=utf-8', readFileSync(new URL('./source.html', import.meta.url))]],
    ['/source.js', ['text/javascript; charset=utf-8', readFileSync(new URL('./source.js', import.meta.url))]]]);
  try {
    for (const port of ports) {
      const server = http.createServer((request, response) => {
        const path = new URL(request.url, 'http://local').pathname, origin = request.headers.origin;
        const allowed = origins.includes(origin) || extensionOrigins.includes(origin);
        const cookies = (request.headers.cookie ?? '').split(';').map(value => value.trim());
        const authenticated = cookies.includes('m1010_fixture=allow');
        if (requests.length < 1000) requests.push({ path: routes.has(path) || pages.has(path)
          || ['/config.json', '/requests.json', '/seed', '/redirect.mp4', '/redirect-same.mp4', '/redirect-ungranted.mp4', '/mime-bad.mp4', '/oversize.mp4'].includes(path) ? path : 'blocked',
        originAllowed: allowed, cookiePresent: !!request.headers.cookie, authenticated,
        authority: request.headers.host === `127.0.0.1:${server.address().port}` ? 'numeric-loopback' : 'other',
        includeRequested: path?.startsWith('/auth/include/') ?? false });
        const send = (status, type, body = '', headers = {}) => {
          response.writeHead(status, { 'Content-Type': type, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...headers });
          response.end(request.method === 'HEAD' ? undefined : body);
        };
        const block = status => { counters.blocked++; send(status, 'text/plain', 'Blocked'); };
        if (![ `127.0.0.1:${server.address().port}`, `localhost:${server.address().port}` ].includes(request.headers.host)) return block(421);
        if (!['GET', 'HEAD'].includes(request.method)) return block(405);
        const credentialed = path === '/seed' || path?.startsWith('/auth/');
        const cors = credentialed && allowed ? { 'Access-Control-Allow-Origin': origin,
          'Access-Control-Allow-Credentials': 'true', Vary: 'Origin' } : {};
        if (credentialed && origin && !allowed) return block(403);
        if (path === '/seed') return send(200, 'text/plain', 'Fixture cookie seeded', { ...cors,
          'Set-Cookie': 'm1010_fixture=allow; Path=/; SameSite=Lax; HttpOnly' });
        if (path === '/config.json') return send(200, 'application/json', JSON.stringify({ media, origins }));
        if (path === '/requests.json') return send(200, 'application/json', JSON.stringify({ requests, counters }));
        if (pages.has(path)) return send(200, ...pages.get(path));
        if (path === '/redirect.mp4') { counters.redirect++; return send(302, 'text/plain', '', { Location: `${origins[1]}/cors/A.mp4` }); }
        if (path === '/redirect-same.mp4' || path === '/redirect-ungranted.mp4') {
          counters.redirect++;
          const destination = path === '/redirect-same.mp4' ? `http://${request.headers.host}/cors/A.mp4` : `${origins[1].replace('127.0.0.1', 'localhost')}/cors/A.mp4`;
          return send(302, 'text/plain', '', { Location: destination });
        }
        if (path === '/mime-bad.mp4') { counters.mimeBad++; return send(200, 'text/html', '<p>Not a video</p>'); }
        if (path === '/oversize.mp4') {
          counters.oversize++;
          let remaining = 64 * 1024 * 1024 + 1;
          response.writeHead(200, { 'Content-Type': 'video/mp4', 'Content-Length': remaining });
          if (request.method === 'HEAD') return response.end();
          const chunk = Buffer.alloc(65536);
          const write = () => {
            while (remaining && !response.destroyed) {
              const count = Math.min(remaining, chunk.length); remaining -= count;
              if (!response.write(chunk.subarray(0, count))) { response.once('drain', write); return; }
            }
            if (!response.destroyed) response.end();
          };
          write(); return;
        }
        const entry = routes.get(path);
        if (!entry) return block(404);
        if (credentialed && !authenticated) { counters.blocked++; return send(401, 'text/plain', 'Fixture cookie required', cors); }
        const size = entry.slice ? entry.slice.end - entry.slice.start + 1 : entry.bytes;
        const range = parseRange(request.headers.range, size);
        if (!range) { counters.blocked++; return send(416, 'text/plain', '', { ...cors, 'Content-Range': `bytes */${size}` }); }
        const headers = { ...cors, 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes',
          'Cache-Control': 'no-store', 'Content-Length': range.end - range.start + 1 };
        if (entry.mode === 'cors') headers['Access-Control-Allow-Origin'] = '*';
        if (range.partial) headers['Content-Range'] = `bytes ${range.start}-${range.end}/${size}`;
        response.writeHead(range.partial ? 206 : 200, headers);
        if (request.method === 'HEAD') return response.end();
        const base = entry.slice?.start ?? 0;
        const stream = createReadStream(entry.path, { start: base + range.start, end: base + range.end });
        stream.on('error', error => response.destroy(error)); response.on('close', () => stream.destroy()); stream.pipe(response);
      });
      servers.push(server);
      await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
      origins.push(`http://127.0.0.1:${server.address().port}`);
    }
  } catch (error) { await close(); throw error; }
  return { url: `${origins[0]}/source.html`, media, origins, close };
}