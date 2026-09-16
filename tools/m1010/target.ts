export {};

const params = new URLSearchParams(location.search), sourceOrigin = params.get('source'), nonce = params.get('session');
const valid = sourceOrigin === 'http://127.0.0.1:5204' && /^[0-9a-f-]{36}$/.test(nonce ?? '');
const video = document.querySelector<HTMLVideoElement>('video')!;
const evidence: Record<string, unknown> = { origin: location.origin, sourceOrigin, nonce, received: false };
let targets: { crop?: unknown; restriction?: unknown } | null = null, stream: MediaStream | null = null;
const status = () => { document.querySelector('output')!.textContent = JSON.stringify(evidence); };
const stop = () => { stream?.getTracks().forEach(track => track.stop()); stream = null; video.srcObject = null; evidence['stopped'] = true; status(); };
window.addEventListener('message', event => {
  if (!valid || event.origin !== sourceOrigin || event.source !== window.opener || !event.data || typeof event.data !== 'object' || event.data.session !== nonce) return;
  if (event.data.type !== 'targets') return;
  targets = { crop: event.data.crop, restriction: event.data.restriction };
  evidence['received'] = true; evidence['sourceErrors'] = event.data.errors;
  evidence['types'] = { crop: targets.crop?.constructor.name ?? null, restriction: targets.restriction?.constructor.name ?? null }; status();
});
if (valid && window.opener) window.opener.postMessage({ type: 'm1010-target-ready', session: nonce }, sourceOrigin!);
const start = async (id: string) => {
  stop();
  const constraints = { video: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: id } }, audio: false };
  stream = await navigator.mediaDevices.getUserMedia(constraints as unknown as MediaStreamConstraints); video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0]!;
  evidence['track'] = { class: track.constructor.name, settings: track.getSettings() }; evidence['stopped'] = false; status();
  return evidence;
};
const apply = async (kind: 'crop' | 'restriction') => {
  const track = stream?.getVideoTracks()[0] as (MediaStreamTrack & { cropTo?: (target: unknown) => Promise<void>; restrictTo?: (target: unknown) => Promise<void> }) | undefined;
  const target = targets?.[kind], method = kind === 'crop' ? 'cropTo' : 'restrictTo';
  const result: Record<string, unknown> = { kind, targetReceived: !!target, methodPresent: typeof track?.[method] === 'function' };
  try { if (!track?.[method] || !target) throw new Error('Target or method unavailable'); await track[method]!(target);
    result['outcome'] = 'SUPPORTED'; result['settings'] = track.getSettings(); }
  catch (error) { result['outcome'] = 'UNSUPPORTED_SAFE'; result['error'] = String(error); }
  evidence[kind] = result; status(); return result;
};
document.getElementById('stop')!.addEventListener('click', stop);
window.addEventListener('pagehide', stop, { once: true });
(globalThis as unknown as { m1010Targets: unknown }).m1010Targets = { start, apply, stop, snapshot: () => ({ ...evidence, liveTracks: stream?.getTracks().filter(track => track.readyState === 'live').length ?? 0 }) };