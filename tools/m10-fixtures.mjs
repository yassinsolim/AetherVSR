import http from 'node:http';
import { createReadStream, existsSync, readFileSync, statSync, mkdirSync, mkdtempSync, linkSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { join, relative } from 'node:path';

export const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
export const CASES = ['custom', 'native', 'multiple', 'late', 'spa', 'iframe', 'sameiframe', 'cors', 'nocors',
  'contain', 'cover', 'clipped', 'radius', 'translated', 'controls/captions', 'pip', 'fullscreen', 'mse', 'drm',
  'reinsert', 'open-shadow-late', 'caption-before-auto-passive', 'caption-after-auto-passive', 'caption-before-positive-passive',
  'rounded-ancestor', 'size-container', 'positioned-ancestor', 'rounded-offset'];

export async function startFixtures({ mse = true } = {}) {
  const preferred = join(ROOT, 'public/media/m9/720p60.mp4');
  const media = existsSync(preferred) ? preferred : join(ROOT, 'public/media/aethervsr-testclip-720p30-h264.mp4');
  const mediaHash = sha256(readFileSync(media));
  const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', media], { encoding: 'utf8', timeout: 15000 }));
  const evidence = { path: relative(ROOT, media), sha256: mediaHash, selection: media === preferred ? 'existing M9 CFR fixture' : 'bundled fallback; not the M9 CFR fixture', probe };
  let fragment;
  let mime;
  if (mse) {
    const directory = join(ROOT, '.cache/m10/fixtures');
    mkdirSync(directory, { recursive: true });
    fragment = join(directory, `${mediaHash}-fmp4.mp4`);
    const recipe = ['-nostdin', '-v', 'error', '-n', '-i', media, '-map', '0:v:0', '-t', '4', '-an', '-c:v', 'copy', '-map_metadata', '-1', '-fflags', '+bitexact', '-movflags', 'frag_keyframe+empty_moov+default_base_moof', fragment];
    if (!existsSync(fragment)) {
      const staging = mkdtempSync(join(directory, 'fragment-'));
      const temporary = join(staging, 'fragment.mp4');
      try {
        execFileSync('ffmpeg', [...recipe.slice(0, -1), temporary], { timeout: 30000 });
        try { linkSync(temporary, fragment); } catch (error) { if (error.code !== 'EEXIST') throw error; }
      } finally { rmSync(staging, { recursive: true, force: true }); }
    }
    const bytes = readFileSync(fragment);
    const config = bytes.indexOf(Buffer.from('avcC'));
    if (config < 0) throw new Error('MSE fixture requires H.264 avcC metadata');
    mime = `video/mp4; codecs="avc1.${bytes.subarray(config + 5, config + 8).toString('hex')}"`;
    evidence.mse = { path: relative(ROOT, fragment), sha256: sha256(bytes), mime, recipe, scope: 'Clear same-origin fMP4 SourceBuffer append, not streaming adaptation or encrypted MSE' };
  }
  const html = readFileSync(new URL('./m10-fixtures/index.html', import.meta.url), 'utf8');
  const requests = [];
  const servers = [];
  const close = async () => Promise.all(servers.map(server => new Promise(done => { server.close(done); server.closeAllConnections(); })));
  try {
    for (const port of [5183, 5184]) {
      const server = http.createServer((request, response) => {
        const url = new URL(request.url, `http://127.0.0.1:${port}`);
        if (requests.length < 10000) requests.push({ port, path: url.pathname, origin: request.headers.origin ?? null, range: request.headers.range ?? null });
        const file = url.pathname === '/media/mse.mp4' ? fragment : ['/media/same.mp4', '/media/cors.mp4', '/media/nocors.mp4'].includes(url.pathname) ? media : null;
        if (file) {
          const size = statSync(file).size;
          const match = /^bytes=(\d+)-(\d*)$/.exec(request.headers.range ?? '');
          const start = match ? Number(match[1]) : 0;
          const end = match?.[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
          if ((request.headers.range && !match) || start > end || start >= size) { response.writeHead(416, { 'Content-Range': `bytes */${size}` }); response.end(); return; }
          const headers = { 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1 };
          if (url.pathname === '/media/cors.mp4') headers['Access-Control-Allow-Origin'] = '*';
          if (match) headers['Content-Range'] = `bytes ${start}-${end}/${size}`;
          response.writeHead(match ? 206 : 200, headers);
          if (request.method === 'HEAD') response.end();
          else { const stream = createReadStream(file, { start, end }); stream.on('error', error => response.destroy(error)); response.on('close', () => stream.destroy()); stream.pipe(response); }
          return;
        }
        const name = url.searchParams.get('case') ?? 'custom';
        if (url.pathname !== '/fixture' || !CASES.includes(name) || (name === 'mse' && !fragment)) { response.writeHead(404); response.end(); return; }
        response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        response.end(html.replace('__FIXTURE_CONFIG__', JSON.stringify({ name, mime, media: evidence.selection, background: url.searchParams.get('background') === 'black' ? 'black' : 'white' })));
      });
      servers.push(server);
      await new Promise((done, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', done); });
    }
  } catch (error) { await close(); throw error; }
  return { url: 'http://127.0.0.1:5183/fixture', evidence, requests, close };
}