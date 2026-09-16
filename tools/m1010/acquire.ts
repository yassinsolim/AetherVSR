import { refetchSelected } from './refetch.js';
import type { SourceSelection } from './policy.js';

type Info = { source: { url: string; protected: boolean; sourceClass: string }; selection: SourceSelection | null;
  sourceTabId: number; playerTabId: number; playerDocumentId: string; permission: string | null };
const video = document.querySelector<HTMLVideoElement>('#input')!, output = document.querySelector('#result')!, status = document.querySelector('#status')!;
const events: { at: number; type: string; value: unknown }[] = [];
type Acquisition = { ticket: number; controller: AbortController; stream: MediaStream | null;
  peer: RTCPeerConnection | null; blobUrl: string | null; releaseNeeded: boolean; listeners: (() => void)[] };
let info: Info | null = null, active: Acquisition | null = null, pending: Promise<void> | null = null;
let owner: 'PAGE_AUTHORITY' | 'PLAYER_AUTHORITY' | null = null;
let epoch = 0, refreshEpoch = 0, disposed = false;
const ensure = (ticket: number) => { if (disposed || ticket !== epoch) throw new Error('Acquisition cancelled or superseded'); };
let callback = 0;
const frames: { at: number; mediaTime: number; presented: number; width: number; height: number }[] = [];
const record = (type: string, value: unknown) => { events.push({ at: performance.now(), type, value }); output.textContent = JSON.stringify(events.slice(-10), null, 2); };
video.muted = true;
const navigationType = (performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined)?.type;
const request = async <T>(message: Record<string, unknown>): Promise<T> => {
  const response: { ok: boolean; value?: T; error?: string } | undefined = await chrome.runtime.sendMessage({ ...message, navigationType });
  if (!response || response.ok !== true) throw new Error(response?.error ?? 'Rejected acquisition request'); return response.value!;
};
const observed = async <Value>(action: () => Promise<Value>): Promise<Value> => {
  try { return await action(); }
  catch (error) { status.textContent = String(error); record('error', String(error)); throw error; }
};
const controls = () => {
  for (const id of ['direct', 'refetch', 'rtc', 'tab']) (document.getElementById(id) as HTMLButtonElement).disabled = disposed || pending !== null;
};
function localCleanup(session: Acquisition | null) {
  if (session) {
    session.controller.abort();
    for (const remove of session.listeners.splice(0)) remove();
    session.stream?.getTracks().forEach(track => track.stop()); session.stream = null;
    session.peer?.close(); session.peer = null;
    if (session.blobUrl) URL.revokeObjectURL(session.blobUrl); session.blobUrl = null;
  }
  if (active === session) { video.pause(); video.srcObject = null; video.removeAttribute('src'); video.load(); owner = null; }
}
async function release(session: Acquisition | null) {
  if (!session?.releaseNeeded) return;
  await request({ type: 'acquire.stop' });
  session.releaseNeeded = false;
}
function runPending(action: () => Promise<void>) {
  const operation = observed(() => Promise.resolve().then(action)).finally(() => {
    if (pending === operation) { pending = null; controls(); }
  });
  pending = operation; controls(); return operation;
}
function stop(): Promise<void> {
  epoch++;
  const session = active, previous = pending;
  localCleanup(session);
  record('stopping', { ticket: session?.ticket ?? null });
  return runPending(async () => {
    await previous?.catch(() => {});
    await release(session);
    record('stopped', { tracks: 0, peers: 0, objectUrls: 0 });
  });
}
function start(kind: string, action: (session: Acquisition) => Promise<void>): Promise<void> {
  if (pending) return observed(() => Promise.reject(new Error('Acquisition cleanup or start pending')));
  if (disposed) return observed(() => Promise.reject(new Error('Acquisition page disposed')));
  const ticket = ++epoch, previous = active;
  localCleanup(previous);
  return runPending(async () => {
    await release(previous); ensure(ticket);
    await readInfo(ticket); ensure(ticket);
    const session: Acquisition = { ticket, controller: new AbortController(), stream: null, peer: null, blobUrl: null, releaseNeeded: false, listeners: [] };
    active = session; record('starting', { kind, ticket });
    try { await action(session); ensure(ticket); record('ready', { kind, ticket }); }
    catch (error) {
      localCleanup(session);
      try { await release(session); }
      catch (cleanupError) { record('cleanup-error', String(cleanupError)); }
      throw error;
    }
  });
}
async function readInfo(ticket: number) {
  const refreshTicket = ++refreshEpoch;
  const selected = await request<Info>({ type: 'acquire.info' }); ensure(ticket);
  if (refreshTicket !== refreshEpoch) throw new Error('Selection refresh superseded');
  info = selected; status.textContent = selected.source.sourceClass; record('selection', selected); return selected;
}
const refresh = () => observed(() => readInfo(epoch));
function waitFor(session: Acquisition, target: EventTarget, event: string, ready: () => boolean, timeout: number, message: string, failure?: () => Error) {
  ensure(session.ticket);
  if (ready()) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    const clean = () => { clearTimeout(timer); target.removeEventListener(event, change); target.removeEventListener('error', fail); session.controller.signal.removeEventListener('abort', cancel); };
    const change = () => { if (ready()) { clean(); resolve(); } };
    const fail = () => { clean(); reject(failure?.() ?? new Error(message)); };
    const cancel = () => { clean(); reject(new Error('Acquisition cancelled or superseded')); };
    const timer = setTimeout(() => { clean(); reject(new Error(message)); }, timeout);
    target.addEventListener(event, change); if (failure) target.addEventListener('error', fail);
    session.controller.signal.addEventListener('abort', cancel, { once: true });
    if (session.controller.signal.aborted) cancel(); else change();
  });
}
async function play(session: Acquisition, authority: typeof owner) {
  await waitFor(session, video, 'loadeddata', () => video.readyState >= 2, 5000, 'Media load timeout', () => new Error(video.error?.message ?? 'Video load failed'));
  ensure(session.ticket); await video.play(); ensure(session.ticket); owner = authority;
}
function setStream(session: Acquisition, media: MediaStream) {
  if (session.controller.signal.aborted || session.ticket !== epoch) { media.getTracks().forEach(track => track.stop()); ensure(session.ticket); }
  session.stream = media;
  const ended = () => { if (active === session && !session.controller.signal.aborted) void stop().catch(() => {}); };
  const watched = new Set<MediaStreamTrack>();
  const watch = () => {
    for (const track of media.getTracks()) {
      if (watched.has(track)) continue;
      watched.add(track); track.addEventListener('ended', ended);
      session.listeners.push(() => track.removeEventListener('ended', ended));
    }
  };
  media.addEventListener('addtrack', watch); media.addEventListener('inactive', ended);
  session.listeners.push(() => media.removeEventListener('addtrack', watch), () => media.removeEventListener('inactive', ended));
  watch(); video.srcObject = media;
}
const direct = (cors: 'anonymous' | null = null) => start('direct', async session => {
  const selected = await readInfo(session.ticket); ensure(session.ticket);
  video.removeAttribute('crossorigin'); if (cors) video.crossOrigin = cors;
  video.src = selected.source.url; await play(session, 'PLAYER_AUTHORITY'); ensure(session.ticket);
  record('direct', { cors, playable: true, sourceClass: selected.source.sourceClass });
});
const refetch = () => start('refetch', async session => {
  const ticket = session.ticket;
  const { selection } = await request<{ selection: SourceSelection }>({ type: 'acquire.refetch' });
  ensure(ticket);
  const blob = await refetchSelected(selection, async () => {
    if (ticket !== epoch || disposed) return false;
    const result = await request<{ current: boolean }>({ type: 'acquire.current' });
    return ticket === epoch && !disposed && result.current;
  }, (input, init) => {
    ensure(ticket);
    return fetch(input, { ...init, signal: AbortSignal.any([session.controller.signal, ...(init?.signal ? [init.signal] : [])]) });
  });
  ensure(ticket); session.blobUrl = URL.createObjectURL(blob); video.removeAttribute('crossorigin'); video.src = session.blobUrl;
  await play(session, 'PLAYER_AUTHORITY'); ensure(ticket); record('refetch', { bytes: blob.size, playable: true });
});
const rtc = () => start('rtc', async session => {
  const ticket = session.ticket;
  session.releaseNeeded = true;
  const offer = await request<RTCSessionDescriptionInit & { tracks: unknown; cloneTests: unknown }>({ type: 'acquire.offer' }); ensure(ticket);
  const receiver = new RTCPeerConnection({ iceServers: [] }); session.peer = receiver;
  const media = new MediaStream(); setStream(session, media);
  const track = (event: RTCTrackEvent) => { if (ticket !== epoch || session.controller.signal.aborted) { event.track.stop(); return; } media.addTrack(event.track); record('track', { kind: event.track.kind, settings: event.track.getSettings() }); };
  receiver.addEventListener('track', track); session.listeners.push(() => receiver.removeEventListener('track', track));
  await receiver.setRemoteDescription({ type: 'offer', sdp: offer.sdp! });
  ensure(ticket);
  const answer = await receiver.createAnswer(); ensure(ticket);
  await receiver.setLocalDescription(answer);
  ensure(ticket);
  await waitFor(session, receiver, 'icegatheringstatechange', () => receiver.iceGatheringState === 'complete', 3000, 'Local ICE gathering timeout');
  ensure(ticket);
  await request({ type: 'acquire.answer', answer: { type: 'answer', sdp: receiver.localDescription!.sdp } });
  ensure(ticket); await play(session, 'PAGE_AUTHORITY'); ensure(ticket);
  record('media-mirror', { sourceTracks: offer.tracks, cloneTests: offer.cloneTests, localDescriptionType: receiver.localDescription?.type });
});
const tab = (consumer = true) => start('tab', async session => {
  const ticket = session.ticket, result = await request<{ id: string }>({ type: 'acquire.tab-id', consumer }); ensure(ticket);
  const constraints = { audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: result.id } },
    video: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: result.id } } };
  const acquired = await navigator.mediaDevices.getUserMedia(constraints as unknown as MediaStreamConstraints);
  if (ticket !== epoch) { acquired.getTracks().forEach(track => track.stop()); ensure(ticket); }
  setStream(session, acquired);
  await play(session, 'PAGE_AUTHORITY'); ensure(ticket); record('tab-mirror', { consumer, tracks: acquired.getTracks().map(track => ({ kind: track.kind, settings: track.getSettings() })) });
});
const probe = () => observed(async () => {
  const ticket = epoch; ensure(ticket);
  const info = { playable: !video.paused && video.readyState >= 2, width: video.videoWidth, height: video.videoHeight, originClean: false,
    pixelError: null as string | null, externalTextureImportable: false, gpuError: null as string | null,
    tracks: active?.stream?.getTracks().map(track => ({ kind: track.kind, state: track.readyState, muted: track.muted, settings: track.getSettings() })) ?? [] };
  const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  try { const context = canvas.getContext('2d', { willReadFrequently: true })!; context.drawImage(video, 0, 0); context.getImageData(0, 0, canvas.width, canvas.height); info.originClean = true; }
  catch (error) { info.pixelError = String(error); }
  let device: GPUDevice | null = null;
  try { const adapter = await navigator.gpu.requestAdapter(); ensure(ticket); if (!adapter) throw new Error('No GPU adapter'); device = await adapter.requestDevice(); ensure(ticket); device.importExternalTexture({ source: video }); info.externalTextureImportable = true; }
  catch (error) { info.gpuError = String(error); } finally { device?.destroy(); }
  ensure(ticket); record('probe', info); return info;
});
const command = (type: 'play' | 'pause' | 'seek') => observed(async () => {
  const ticket = epoch; ensure(ticket);
  if (owner !== 'PAGE_AUTHORITY') throw new Error('Remote command requires PAGE_AUTHORITY');
  const result = await request({ type: 'acquire.command', command: { type, requestId: crypto.randomUUID(), ...(type === 'seek' ? { seconds: 1 } : {}) } }); ensure(ticket); record('command', result); return result;
});
const click = (id: string, action: () => Promise<unknown>) => document.getElementById(id)!.addEventListener('click', () => { void action().catch(() => {}); });
click('info', refresh); click('direct', () => direct()); click('refetch', refetch); click('rtc', rtc); click('tab', () => tab()); click('probe', probe);
click('stop', stop);
for (const type of ['play', 'pause', 'seek'] as const) click(type, () => command(type));
document.getElementById('origin')!.addEventListener('click', () => {
  if (!info?.permission) { record('permission', 'No progressive selected origin'); return; }
  void chrome.permissions.request({ origins: [info.permission] }).then(granted => record('origin-permission', { pattern: info!.permission, granted }), error => record('permission-error', String(error)));
});
document.getElementById('revoke-origin')!.addEventListener('click', () => { if (info?.permission) void chrome.permissions.remove({ origins: [info.permission] }).then(removed => record('origin-revoked', removed)); });
document.getElementById('capture-permission')!.addEventListener('click', () => { void chrome.permissions.request({ permissions: ['tabCapture'] }).then(granted => record('capture-permission', granted), error => record('permission-error', String(error))); });
const frame = (at: number, metadata: VideoFrameCallbackMetadata) => { if (disposed) return; frames.push({ at, mediaTime: metadata.mediaTime, presented: metadata.presentedFrames, width: metadata.width, height: metadata.height }); if (frames.length > 10000) frames.shift(); callback = video.requestVideoFrameCallback(frame); };
callback = video.requestVideoFrameCallback(frame);
const revoked = (message: unknown, sender: chrome.runtime.MessageSender) => {
  if (sender.id !== chrome.runtime.id || sender.url !== chrome.runtime.getURL('service-worker.js') || !message || typeof message !== 'object') return;
  const value = message as Record<string, unknown>;
  if (value['type'] === 'acquire.revoked' && value['playerTabId'] === info?.playerTabId && value['playerDocumentId'] === info?.playerDocumentId) { void stop().catch(() => {}); record('source-revoked', true); }
};
chrome.runtime.onMessage.addListener(revoked);
window.addEventListener('pagehide', () => { disposed = true; video.cancelVideoFrameCallback(callback); chrome.runtime.onMessage.removeListener(revoked); void stop().catch(() => {}); }, { once: true });
(globalThis as unknown as { m1010Acquire: unknown }).m1010Acquire = { refresh, direct, refetch, rtc, tab, probe, command, stop,
  snapshot: () => ({ owner, info, pending: pending !== null, epoch, frames: frames.slice(), events: events.slice(), resources: { tracks: active?.stream?.getTracks().filter(track => track.readyState === 'live').length ?? 0, peers: Number(active?.peer != null), objectUrls: Number(active?.blobUrl != null) } }) };
void refresh().catch(error => record('selection-error', String(error)));