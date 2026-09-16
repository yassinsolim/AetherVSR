const byId = id => document.getElementById(id);
const video = byId('source');
const config = fetch('/config.json').then(response => response.json());
let generation = 0, controller, selection = null, objectUrl, inputStream, captured, display;
let animation = 0, callback = 0, captureGeneration = 0, captureState = 'idle';
let frames = 0, lastFrame = null, qualityChanges = 0, size = '', rendition = null;
const events = {};
for (const name of ['playing', 'pause', 'seeking', 'seeked', 'ratechange', 'resize', 'waiting', 'stalled', 'ended', 'error']) {
  video.addEventListener(name, () => { events[name] = (events[name] ?? 0) + 1; });
}
video.muted = byId('mute').checked;
video.addEventListener('volumechange', () => { byId('mute').checked = video.muted; });
const stop = stream => stream?.getTracks().forEach(track => track.stop());
const fail = error => { byId('status').textContent = `${error.name}: ${error.message}`; };
function cancelCapture(state = 'cancelled') {
  captureGeneration++; stop(display); stop(captured); display = captured = undefined; captureState = state;
}
function reset() {
  controller?.abort(); cancelCapture('idle'); video.pause();
  cancelAnimationFrame(animation); if (callback) video.cancelVideoFrameCallback(callback);
  animation = callback = 0; stop(inputStream); inputStream = undefined;
  video.srcObject = null; video.removeAttribute('src'); video.load();
  if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = undefined;
  frames = qualityChanges = 0; lastFrame = null; size = ''; rendition = null;
}
function wait(target, event, signal) {
  return new Promise((resolve, reject) => {
    const finish = error => { clearTimeout(timer); target.removeEventListener(event, done);
      target.removeEventListener('error', failed); signal.removeEventListener('abort', aborted);
      if (error) reject(error); else resolve(); };
    const done = () => finish(), failed = () => finish(new Error(`Media ${event} failed`));
    const aborted = () => finish(new DOMException('Preparation cancelled', 'AbortError'));
    const timer = setTimeout(() => finish(new Error(`Timeout: ${event}`)), 10000);
    target.addEventListener(event, done, { once: true }); target.addEventListener('error', failed, { once: true });
    signal.addEventListener('abort', aborted, { once: true }); if (signal.aborted) aborted();
  });
}
function observe() {
  if (!video.requestVideoFrameCallback) return;
  callback = video.requestVideoFrameCallback((now, metadata) => {
    frames++; lastFrame = { now, mediaTime: metadata.mediaTime, presentedFrames: metadata.presentedFrames,
      width: metadata.width, height: metadata.height };
    const next = `${metadata.width}x${metadata.height}`;
    if (size && size !== next) qualityChanges++; size = next;
    rendition = metadata.width === 640 ? 'low' : metadata.width === 1280 ? 'high' : null;
    observe();
  });
}
async function bytes(url, signal, credentials = 'omit') {
  const response = await fetch(url, { signal, credentials });
  if (!response.ok) throw new Error(`Fixture HTTP ${response.status}`);
  return response.arrayBuffer();
}
function generated() {
  const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 360;
  const context = canvas.getContext('2d');
  if (!context || !canvas.captureStream) throw new Error('Canvas captureStream unavailable');
  const start = performance.now();
  const draw = now => {
    const flash = Math.floor((now - start) * 30 / 1000) % 30 < 3;
    ['#d04040', '#30b060', '#3050d0'].forEach((color, index) => {
      context.fillStyle = flash ? '#ffffff' : color;
      context.fillRect(index * canvas.width / 3, 0, canvas.width / 3, canvas.height);
    });
    animation = requestAnimationFrame(draw);
  };
  draw(start); inputStream = canvas.captureStream(30); video.srcObject = inputStream;
}
async function mse(settings, adaptive, signal) {
  const mime = settings.media.low.mime;
  if (adaptive && (mime !== settings.media.high.mime || settings.media.low.fragments.length < 4
    || settings.media.high.fragments.length !== settings.media.low.fragments.length)) throw new Error('Incompatible rendition fragments');
  if (!globalThis.MediaSource?.isTypeSupported(mime)) throw new Error(`Unsupported MSE: ${mime}`);
  const source = new MediaSource();
  objectUrl = URL.createObjectURL(source);
  const opened = wait(source, 'sourceopen', signal); video.src = objectUrl; await opened;
  const buffer = source.addSourceBuffer(mime);
  const append = async path => {
    const data = await bytes(path, signal);
    const updated = wait(buffer, 'updateend', signal);
    try { buffer.appendBuffer(data); } catch (error) { controller.abort(); await updated.catch(() => {}); throw error; }
    await updated;
  };
  await append('/mse/low/init.mp4');
  for (let index = 0; index < settings.media.low.fragments.length; index++) {
    if (adaptive && index === 2) await append('/mse/high/init.mp4');
    await append(`/mse/${adaptive && index >= 2 ? 'high' : 'low'}/${index}.m4s`);
  }
  source.endOfStream();
}
async function prepare(value = 'same') {
  const choice = typeof value === 'string' ? { mode: value } : value;
  if (!choice || Object.keys(choice).some(key => !['mode', 'asset', 'credentials'].includes(key))) throw new Error('Invalid selection');
  const { mode = 'same', asset = 'A', credentials = 'omit' } = choice;
  if (!['same', 'cors', 'nocors', 'auth', 'cookieseed', 'blob', 'MSE', 'srcObject', 'ABR'].includes(mode)
    || !['A', 'B', 'C'].includes(asset) || !['omit', 'include'].includes(credentials)) throw new Error('Invalid case');
  const ticket = ++generation; reset(); selection = null;
  controller = new AbortController(); const signal = controller.signal;
  const settings = await config; signal.throwIfAborted();
  byId('mode').value = mode; byId('asset').value = asset; byId('credentials').value = credentials;
  video.removeAttribute('crossorigin');
  const origin = ['cors', 'nocors', 'auth', 'cookieseed'].includes(mode) ? settings.origins[1] : settings.origins[0];
  let url = `${origin}/${mode}/${asset}.mp4`, sourceClass = 'progressive';
  try {
    if (mode === 'cookieseed') {
      await bytes(`${origin}/seed`, signal, 'include');
      const seeded = { mode, generation: ticket, seeded: true };
      byId('status').textContent = JSON.stringify(seeded); return seeded;
    }
    if (mode === 'auth' || mode === 'blob') {
      url = mode === 'auth' ? `${origin}/auth/${credentials}/${asset}.mp4` : `${origin}/same/${asset}.mp4`;
      objectUrl = URL.createObjectURL(new Blob([await bytes(url, signal, mode === 'auth' ? credentials : 'omit')], { type: 'video/mp4' }));
      video.src = objectUrl; sourceClass = mode === 'auth' ? 'credentialed-fixture' : 'blob';
    } else if (mode === 'MSE' || mode === 'ABR') { await mse(settings, mode === 'ABR', signal); url = objectUrl; sourceClass = 'mse'; }
    else if (mode === 'srcObject') { generated(); url = null; sourceClass = 'stream'; }
    else { if (mode === 'cors') video.crossOrigin = 'anonymous'; video.src = url; }
    if (video.readyState < 1) await wait(video, 'loadedmetadata', signal);
    signal.throwIfAborted(); observe();
    selection = { ownerId: 'source', generation: ticket, mode, asset, credentials: mode === 'auth' ? credentials : 'omit',
      url: mode === 'blob' ? objectUrl : url, sourceClass, protected: false };
    byId('status').textContent = JSON.stringify(selection); return { ...selection };
  } catch (error) { if (ticket === generation) reset(); throw error; }
}
function snapshot() {
  const quality = video.getVideoPlaybackQuality?.();
  return { selection, generation, paused: video.paused, muted: video.muted, volume: video.volume,
    currentTime: video.currentTime, duration: Number.isFinite(video.duration) ? video.duration : null,
    rate: video.playbackRate, width: video.videoWidth, height: video.videoHeight,
    readyState: video.readyState, networkState: video.networkState, error: video.error?.code ?? null,
    frames, lastFrame, qualityChanges, rendition, events: { ...events }, eventsScope: 'page-lifetime; frame counters reset per preparation',
    quality: quality ? { total: quality.totalVideoFrames, dropped: quality.droppedVideoFrames, corrupted: quality.corruptedVideoFrames } : null,
    captureState, displayTracks: display?.getTracks().map(track => ({ kind: track.kind, readyState: track.readyState,
      muted: track.muted, settings: track.getSettings() })) ?? [],
    generatorAudio: false, cssScope: 'fixture-controlled isolation:isolate; transform-style:flat; not generic-page zero-CSS' };
}
async function action(command) {
  const { type } = command ?? {};
  const keys = { play: [], pause: [], seek: ['seconds'], rate: ['rate'], volume: ['volume'], mute: ['muted'],
    size: ['width', 'height'], fullscreen: [], cancel: [], 'wrong-choice': [], dispose: [] };
  if (!Object.hasOwn(keys, type) || Object.keys(command).some(key => key !== 'type' && !keys[type].includes(key))) throw new Error('Invalid source command');
  const numeric = (value, min, max) => typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
  if (type === 'play') await video.play();
  else if (type === 'pause') video.pause();
  else if (type === 'seek' && numeric(command.seconds, 0, 4)) video.currentTime = command.seconds;
  else if (type === 'rate' && numeric(command.rate, 0.25, 4)) video.playbackRate = command.rate;
  else if (type === 'volume' && numeric(command.volume, 0, 1)) video.volume = command.volume;
  else if (type === 'mute' && typeof command.muted === 'boolean') video.muted = command.muted;
  else if (type === 'size' && numeric(command.width, 160, 1920) && numeric(command.height, 90, 1080)) {
    video.style.width = `${command.width}px`; video.style.height = `${command.height}px`;
  } else if (type === 'fullscreen') await video.requestFullscreen();
  else if (type === 'cancel') cancelCapture();
  else if (type === 'wrong-choice') cancelCapture('wrong-choice');
  else if (type === 'dispose') { generation++; reset(); selection = null; }
  else throw new Error('Invalid source command');
  byId('mute').checked = video.muted; return snapshot();
}
function captureDisplay() {
  if (!navigator.userActivation?.isActive) throw new Error('Manual display gesture required');
  if (!navigator.mediaDevices?.getDisplayMedia) throw new Error('Display capture unavailable');
  cancelCapture('choosing'); const ticket = captureGeneration;
  return navigator.mediaDevices.getDisplayMedia({ video: true, audio: true, preferCurrentTab: true }).then(stream => {
    if (ticket !== captureGeneration) { stop(stream); throw new DOMException('Cancelled', 'AbortError'); }
    display = stream; captureState = 'unconfirmed-choice';
    stream.getVideoTracks()[0]?.addEventListener('ended', () => { if (display === stream) cancelCapture('ended'); });
    return stream;
  }, error => { if (ticket === captureGeneration) captureState = error.name; throw error; });
}
function captureStream() {
  if (!video.captureStream) throw new Error('Media captureStream unavailable');
  stop(captured); captured = video.captureStream(); return captured;
}
const cropTarget = () => globalThis.CropTarget.fromElement(video);
const restrictionTarget = () => globalThis.RestrictionTarget.fromElement(video);
async function applyTarget(createTarget, method) {
  const stream = display, ticket = captureGeneration, track = stream?.getVideoTracks()[0];
  if (!track?.[method]) throw new Error(`${method} unavailable`);
  const target = await createTarget();
  if (ticket !== captureGeneration || stream !== display) throw new Error('Capture changed');
  await track[method](target);
}
const crop = () => applyTarget(cropTarget, 'cropTo');
const restrict = () => applyTarget(restrictionTarget, 'restrictTo');
globalThis.__M1010_SOURCE__ = { prepare, action, snapshot, video, captureDisplay, captureStream,
  cropTarget, restrictionTarget, crop, restrict, get stream() { return display ?? captured ?? inputStream; } };
const button = (id, run) => byId(id).addEventListener('click', event => {
  if (!event.isTrusted) return;
  try { Promise.resolve(run()).catch(fail); } catch (error) { fail(error); }
});
button('prepare', () => prepare({ mode: byId('mode').value, asset: byId('asset').value, credentials: byId('credentials').value }));
for (const type of ['play', 'pause', 'fullscreen', 'cancel']) button(type, () => action({ type }));
button('wrong', () => action({ type: 'wrong-choice' }));
button('display', captureDisplay); button('capture', captureStream); button('crop', crop); button('restrict', restrict);
for (const [id, key] of [['seek', 'seconds'], ['rate', 'rate'], ['mute', 'muted']]) byId(id).addEventListener('change', () => {
  action({ type: id, [key]: id === 'mute' ? byId(id).checked : Number(byId(id).value) }).catch(fail);
});
window.addEventListener('pagehide', () => { generation++; reset(); selection = null; });
prepare('same').catch(fail);