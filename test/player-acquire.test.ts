import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest';

type Message = Record<string, unknown>;
type Snapshot = { owner: string | null; info: unknown; pending: boolean; events: { type: string; value: unknown }[];
  resources: { tracks: number; peers: number; objectUrls: number } };
type Acquire = { refresh(): Promise<unknown>; direct(cors?: 'anonymous' | null): Promise<void>;
  refetch(): Promise<void>; rtc(): Promise<void>; tab(consumer?: boolean): Promise<void>; tabVisible(): Promise<void>;
  stop(): Promise<void>; probe(): Promise<unknown>; snapshot(): Snapshot };

class Target extends EventTarget {
  listeners = new Map<string, Set<EventListenerOrEventListenerObject>>();
  override addEventListener(type: string, listener: EventListenerOrEventListenerObject | null, options?: AddEventListenerOptions | boolean) {
    if (listener) { const listeners = this.listeners.get(type) ?? new Set(); listeners.add(listener); this.listeners.set(type, listeners); }
    super.addEventListener(type, listener, options);
  }
  override removeEventListener(type: string, listener: EventListenerOrEventListenerObject | null, options?: EventListenerOptions | boolean) {
    if (listener) this.listeners.get(type)?.delete(listener);
    super.removeEventListener(type, listener, options);
  }
  get listenerCount() { return [...this.listeners.values()].reduce((count, listeners) => count + listeners.size, 0); }
}
class Element extends Target { textContent = ''; disabled = false; }
class Track extends Target {
  kind = 'video'; readyState = 'live'; muted = false;
  stop = vi.fn(() => { this.readyState = 'ended'; });
  getSettings() { return { width: 320, height: 180 }; }
}
class Stream extends Target {
  constructor(private tracks: Track[] = []) { super(); }
  getTracks() { return this.tracks; }
  addTrack(track: Track) { this.tracks.push(track); this.dispatchEvent(new Event('addtrack')); }
}
class Video extends Element {
  readyState = 2; paused = true; muted = false; videoWidth = 320; videoHeight = 180;
  src = ''; srcObject: Stream | null = null; crossOrigin = ''; error: { message: string } | null = null;
  play = vi.fn(() => { this.paused = false; return Promise.resolve(); });
  pause = vi.fn(() => { this.paused = true; });
  load = vi.fn();
  requestVideoFrameCallback = vi.fn<(callback: VideoFrameRequestCallback) => number>().mockReturnValue(1);
  cancelVideoFrameCallback = vi.fn();
  removeAttribute(name: string) { if (name === 'src') this.src = ''; if (name === 'crossorigin') this.crossOrigin = ''; }
}
class Peer extends Target {
  static instances: Peer[] = [];
  static configure: (peer: Peer) => void = () => {};
  iceGatheringState = 'complete';
  localDescription: RTCSessionDescriptionInit = { type: 'answer', sdp: 'answer' };
  close = vi.fn();
  setRemoteDescription = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
  createAnswer = vi.fn<() => Promise<RTCSessionDescriptionInit>>().mockResolvedValue(this.localDescription);
  setLocalDescription = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
  constructor() { super(); Peer.instances.push(this); Peer.configure(this); }
}
function deferred<Value>() {
  let resolve!: (value: Value) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<Value>((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}
const source = { tabId: 12, documentId: 'source-doc', ownerId: 'video', generation: 1,
  url: 'https://media.example/clip.mp4', protected: false, sourceClass: 'progressive', credentials: 'omit' };
const selectionInfo = { source, selection: source, sourceTabId: 12, playerTabId: 13,
  playerDocumentId: 'player-doc', permission: 'https://media.example/*' };
const reply = (value: unknown) => ({ ok: true, value });
let video: Video, page: Target, buttons: Map<string, Element>, acquire: Acquire;
let sendMessage: ReturnType<typeof vi.fn<(message: Message) => Promise<unknown>>>;
let getUserMedia: ReturnType<typeof vi.fn<() => Promise<Stream>>>;
let removeMessageListener: ReturnType<typeof vi.fn>;
let createObjectURL: MockInstance<typeof URL.createObjectURL>;
let revokeObjectURL: MockInstance<typeof URL.revokeObjectURL>;

async function settle() { for (let index = 0; index < 30; index++) await Promise.resolve(); }
async function setup() {
  await import('../tools/m1010/acquire');
  acquire = (globalThis as unknown as { m1010Acquire: Acquire }).m1010Acquire;
  await settle();
}
beforeEach(() => {
  vi.resetModules(); vi.useFakeTimers();
  Peer.instances = []; Peer.configure = () => {};
  video = new Video(); page = new Target(); buttons = new Map();
  for (const id of ['input', 'result', 'status', 'info', 'direct', 'refetch', 'rtc', 'tab', 'probe', 'stop', 'play', 'pause', 'seek', 'origin', 'revoke-origin', 'capture-permission']) buttons.set(id, id === 'input' ? video : new Element());
  sendMessage = vi.fn((message: Message) => {
    if (message['type'] === 'acquire.info') return Promise.resolve(reply(selectionInfo));
    if (message['type'] === 'acquire.refetch') return Promise.resolve(reply({ selection: source }));
    if (message['type'] === 'acquire.current') return Promise.resolve(reply({ current: true }));
    if (message['type'] === 'acquire.offer') return Promise.resolve(reply({ type: 'offer', sdp: 'offer', tracks: [], cloneTests: [] }));
    return Promise.resolve(reply({ id: 'capture-id', stopped: true }));
  });
  getUserMedia = vi.fn<() => Promise<Stream>>().mockResolvedValue(new Stream());
  removeMessageListener = vi.fn();
  vi.stubGlobal('document', { querySelector: (selector: string) => buttons.get(selector.slice(1)), getElementById: (id: string) => buttons.get(id),
    createElement: () => ({ width: 0, height: 0, getContext: () => ({ drawImage: vi.fn(), getImageData: vi.fn() }) }) });
  vi.stubGlobal('window', page);
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia }, gpu: { requestAdapter: vi.fn().mockResolvedValue(null) } });
  vi.stubGlobal('MediaStream', Stream);
  vi.stubGlobal('RTCPeerConnection', Peer);
  vi.stubGlobal('chrome', { runtime: { id: 'extension', getURL: (path: string) => `chrome-extension://extension/${path}`, sendMessage,
    onMessage: { addListener: vi.fn(), removeListener: removeMessageListener } }, permissions: { request: vi.fn(), remove: vi.fn() },
    tabCapture: { getMediaStreamId: vi.fn((_options: unknown, callback: (id: string) => void) => { callback('visible-id'); }) } });
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Unmocked fetch')));
  createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:owned');
  revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
});
afterEach(async () => {
  page.dispatchEvent(new Event('pagehide'));
  await settle();
  expect(video.listenerCount).toBe(0);
  expect(vi.getTimerCount()).toBe(0);
  vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers();
  Reflect.deleteProperty(globalThis, 'm1010Acquire');
});

describe('acquisition lifecycle with local platform mocks', () => {
  it('issues in the visible player for the authenticated selected tab only', async () => {
    await setup(); await acquire.tabVisible();
    expect(chrome.tabCapture.getMediaStreamId).toHaveBeenCalledWith({ targetTabId: selectionInfo.sourceTabId }, expect.any(Function));
    expect(sendMessage).toHaveBeenCalledWith({ type: 'acquire.current', navigationType: undefined });
    expect(getUserMedia).toHaveBeenCalledWith({ audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: 'visible-id' } },
      video: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: 'visible-id' } } });
  });
  it('rejects a visible-player ID arriving after stop before media redemption', async () => {
    const issued = deferred<(id: string) => void>();
    vi.mocked(chrome.tabCapture.getMediaStreamId).mockImplementation((_options, callback) => { issued.resolve(callback); });
    await setup();
    const pending = acquire.tabVisible().catch((error: unknown) => error);
    const callback = await issued.promise;
    const stopped = acquire.stop(); callback('late-id');
    expect(await pending).toBeInstanceOf(Error); await stopped;
    expect(getUserMedia).not.toHaveBeenCalled();
  });
  it.each(['navigate', 'reload', 'back_forward', undefined])('forwards its own navigation type %s without defaulting', async navigationType => {
    vi.spyOn(performance, 'getEntriesByType').mockReturnValue(navigationType === undefined ? [] : [{ type: navigationType } as unknown as PerformanceEntry]);
    await setup();
    expect(sendMessage).toHaveBeenCalledWith({ type: 'acquire.info', navigationType });
  });
  it.each(['pending', 'active'] as const)('authenticates an early tab start and cleans %s media on revocation', async mediaState => {
    const initialInfo = deferred<unknown>(), startupInfo = deferred<unknown>(), capture = deferred<Stream>();
    const track = new Track(), media = new Stream([track]);
    sendMessage.mockReturnValueOnce(initialInfo.promise).mockReturnValueOnce(startupInfo.promise);
    getUserMedia.mockReturnValueOnce(capture.promise);
    await setup();
    expect(acquire.snapshot().info).toBeNull();
    expect(sendMessage.mock.calls).toEqual([[{ type: 'acquire.info' }]]);
    const acquisition = acquire.tab().then(() => null, (error: unknown) => error);
    await settle();
    expect(sendMessage.mock.calls).toEqual([[{ type: 'acquire.info' }], [{ type: 'acquire.info' }]]);
    expect(getUserMedia).not.toHaveBeenCalled();
    expect(video.play).not.toHaveBeenCalled();
    expect(video.srcObject).toBeNull();
    expect(acquire.snapshot().info).toBeNull();
    startupInfo.resolve(reply(selectionInfo)); await settle();
    expect(acquire.snapshot().info).toEqual(selectionInfo);
    expect(sendMessage).toHaveBeenCalledWith({ type: 'acquire.tab-id', consumer: true });
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    initialInfo.resolve(reply({ ...selectionInfo, playerTabId: 99, playerDocumentId: 'stale-player-doc' })); await settle();
    expect(acquire.snapshot().info).toEqual(selectionInfo);
    if (mediaState === 'active') {
      capture.resolve(media); expect(await acquisition).toBeNull();
      expect(video.srcObject).toBe(media);
      expect(acquire.snapshot().owner).toBe('PAGE_AUTHORITY');
    }
    const revoked = vi.mocked(chrome.runtime.onMessage).addListener.mock.calls[0]![0];
    revoked({ type: 'acquire.revoked', playerTabId: selectionInfo.playerTabId, playerDocumentId: selectionInfo.playerDocumentId },
      { id: chrome.runtime.id, url: chrome.runtime.getURL('service-worker.js') }, vi.fn());
    await settle();
    expect(acquire.snapshot().events.some(event => event.type === 'source-revoked')).toBe(true);
    if (mediaState === 'pending') {
      expect(track.stop).not.toHaveBeenCalled();
      capture.resolve(media);
      const error = await acquisition;
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toMatch(/cancel/i);
      expect(video.play).not.toHaveBeenCalled();
    }
    await settle();
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(media.listenerCount + track.listenerCount).toBe(0);
    expect(video.srcObject).toBeNull();
    expect(acquire.snapshot()).toMatchObject({ owner: null, pending: false, resources: { tracks: 0, peers: 0, objectUrls: 0 } });
  });

  it('cleans a tab stream when playback rejects, including native API errors', async () => {
    const track = new Track();
    getUserMedia.mockResolvedValue(new Stream([track]));
    await setup();
    video.play.mockRejectedValueOnce(new Error('Playback denied'));
    await expect(acquire.tab()).rejects.toThrow('Playback denied');
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(track.listenerCount).toBe(0);
    expect(video.srcObject).toBeNull();
    expect(acquire.snapshot().resources).toEqual({ tracks: 0, peers: 0, objectUrls: 0 });
    expect(acquire.snapshot().events.some(event => event.type === 'error')).toBe(true);
    expect(video.muted).toBe(true);
  });

  it('serializes cancelled RTC offers through source release, keeping Stop usable', async () => {
    await setup();
    const offer = deferred<unknown>(), release = deferred<unknown>();
    sendMessage.mockImplementation(message => message['type'] === 'acquire.offer' ? offer.promise
      : message['type'] === 'acquire.stop' ? release.promise : Promise.resolve(reply(selectionInfo)));
    const acquisition = expect(acquire.rtc()).rejects.toThrow(/cancel/i);
    await settle();
    const stopped = acquire.stop();
    const stoppedAgain = acquire.stop();
    expect(buttons.get('rtc')?.disabled).toBe(true);
    expect(buttons.get('stop')?.disabled).toBe(false);
    await expect(acquire.direct()).rejects.toThrow(/pending/i);
    expect(sendMessage.mock.calls.filter(([message]) => message['type'] === 'acquire.stop')).toHaveLength(0);
    offer.resolve(reply({ type: 'offer', sdp: 'late' })); await settle();
    expect(Peer.instances).toHaveLength(0);
    expect(sendMessage).toHaveBeenCalledWith({ type: 'acquire.stop' });
    await expect(acquire.tab()).rejects.toThrow(/pending/i);
    release.resolve(reply({ stopped: true }));
    await acquisition; await stopped; await stoppedAgain;
    expect(acquire.snapshot().pending).toBe(false);
    sendMessage.mockImplementation(message => Promise.resolve(reply(message['type'] === 'acquire.offer' ? { type: 'offer', sdp: 'new' } : selectionInfo)));
    await acquire.rtc();
    expect(Peer.instances[0]?.close).not.toHaveBeenCalled();
    expect(acquire.snapshot().owner).toBe('PAGE_AUTHORITY');
    expect(sendMessage.mock.calls.filter(([message]) => message['type'] === 'acquire.stop')).toHaveLength(1);
    await acquire.stop();
    expect(Peer.instances[0]?.close).toHaveBeenCalledTimes(1);
  });

  it('closes failed RTC peers and early tracks, and retries failed source cleanup before another start', async () => {
    await setup();
    const track = new Track();
    let lateTrack!: EventListener;
    Peer.configure = receiver => {
      receiver.setRemoteDescription.mockImplementation(() => {
        lateTrack = [...receiver.listeners.get('track')!][0] as EventListener;
        receiver.dispatchEvent(Object.assign(new Event('track'), { track }));
        return Promise.resolve();
      });
      receiver.setLocalDescription.mockRejectedValue(new Error('Signaling failed'));
    };
    const standard = sendMessage.getMockImplementation()!;
    sendMessage.mockImplementation(message => message['type'] === 'acquire.stop'
      ? Promise.resolve({ ok: false, error: 'Release denied' }) : standard(message));
    await expect(acquire.rtc()).rejects.toThrow('Signaling failed');
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(track.listenerCount).toBe(0);
    expect(Peer.instances[0]?.close).toHaveBeenCalledTimes(1);
    expect(Peer.instances[0]?.listenerCount).toBe(0);
    expect(acquire.snapshot().events.some(event => event.type === 'cleanup-error')).toBe(true);
    await expect(acquire.tab()).rejects.toThrow('Release denied');
    expect(getUserMedia).not.toHaveBeenCalled();
    sendMessage.mockImplementation(standard);
    await acquire.direct();
    const stale = new Track(); lateTrack(Object.assign(new Event('track'), { track: stale }));
    expect(stale.stop).toHaveBeenCalledTimes(1);
    expect(acquire.snapshot().owner).toBe('PLAYER_AUTHORITY');
  });

  it('stops late getUserMedia tracks and removes every stream listener on termination', async () => {
    await setup();
    const capture = deferred<Stream>(), late = new Track();
    getUserMedia.mockReturnValueOnce(capture.promise);
    const acquisition = expect(acquire.tab()).rejects.toThrow(/cancel/i);
    await settle(); const stopped = acquire.stop();
    await expect(acquire.rtc()).rejects.toThrow(/pending/i);
    capture.resolve(new Stream([late])); await acquisition; await stopped;
    expect(late.stop).toHaveBeenCalledTimes(1);
    expect(video.srcObject).toBeNull();
    const track = new Track(), added = new Track(), media = new Stream([track]);
    getUserMedia.mockResolvedValueOnce(media); await acquire.tab();
    media.addTrack(added); added.dispatchEvent(new Event('ended')); await settle();
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(added.stop).toHaveBeenCalledTimes(1);
    expect(media.listenerCount + track.listenerCount + added.listenerCount).toBe(0);
    expect(video.srcObject).toBeNull();
    expect(acquire.snapshot().owner).toBeNull();
  });

  it('cleans direct load waits on cancellation, media error and timeout through API and UI', async () => {
    await setup(); video.readyState = 0;
    for (const failure of ['cancel', 'error', 'timeout']) {
      const acquisition = expect(acquire.direct()).rejects.toThrow(failure === 'error' ? 'Decode failed' : failure === 'timeout' ? 'Media load timeout' : /cancel/i);
      await settle();
      if (failure === 'cancel') { buttons.get('stop')!.dispatchEvent(new Event('click')); await settle(); }
      else if (failure === 'error') { video.error = { message: 'Decode failed' }; video.dispatchEvent(new Event('error')); }
      else await vi.advanceTimersByTimeAsync(5000);
      await acquisition; await settle();
      expect(video.listenerCount).toBe(0);
      expect(vi.getTimerCount()).toBe(0);
      expect(video.src).toBe('');
      expect(acquire.snapshot().owner).toBeNull();
    }
    video.readyState = 2; video.muted = false;
    buttons.get('direct')!.dispatchEvent(new Event('click')); await settle();
    expect(acquire.snapshot().owner).toBe('PLAYER_AUTHORITY');
    expect(video.muted).toBe(false);
  });

  it('guards a late RTC answer and cancels or times out ICE waits without leaked listeners', async () => {
    await setup();
    const answer = deferred<RTCSessionDescriptionInit>();
    Peer.configure = receiver => { receiver.createAnswer.mockReturnValue(answer.promise); };
    const acquisition = expect(acquire.rtc()).rejects.toThrow(/cancel/i);
    await settle(); const stopped = acquire.stop();
    answer.resolve({ type: 'answer', sdp: 'late' }); await acquisition; await stopped;
    expect(Peer.instances[0]?.setLocalDescription).not.toHaveBeenCalled();
    expect(Peer.instances[0]?.close).toHaveBeenCalledTimes(1);
    for (const cancel of [true, false]) {
      Peer.configure = receiver => { receiver.iceGatheringState = 'gathering'; };
      const attempt = expect(acquire.rtc()).rejects.toThrow(cancel ? /cancel/i : 'Local ICE gathering timeout');
      await settle();
      if (cancel) await acquire.stop(); else await vi.advanceTimersByTimeAsync(3000);
      await attempt;
      expect(Peer.instances.at(-1)?.listenerCount).toBe(0);
      expect(Peer.instances.at(-1)?.close).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
    }
  });

  it('aborts refetch bodies, rejects stale identity replies and revokes failed playback blobs', async () => {
    await setup();
    let signal: AbortSignal | null | undefined;
    let body!: ReadableStream<Uint8Array>;
    const payload = Uint8Array.of(0, 0, 0, 24, 102, 116, 121, 112, 105, 115, 111, 109, 0, 0, 0, 0, 105, 115, 111, 109, 109, 112, 52, 50);
    const mockFetch = vi.fn<typeof fetch>().mockImplementation((_input, init) => {
      signal = init?.signal;
      body = new ReadableStream<Uint8Array>({ start(controller) {
        signal?.addEventListener('abort', () => controller.error(new DOMException('Cancelled', 'AbortError')), { once: true });
      } });
      const response = new Response(body, { headers: { 'content-type': 'video/mp4' } });
      Object.defineProperty(response, 'url', { value: source.url });
      return Promise.resolve(response);
    });
    vi.stubGlobal('fetch', mockFetch);
    const acquisition = expect(acquire.refetch()).rejects.toThrow('Cancelled');
    await settle(); expect(signal?.aborted).toBe(false);
    await acquire.stop(); await acquisition;
    expect(signal?.aborted).toBe(true);
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
    const current = deferred<unknown>(), standard = sendMessage.getMockImplementation()!;
    sendMessage.mockImplementation(message => message['type'] === 'acquire.current' ? current.promise : standard(message));
    const stale = expect(acquire.refetch()).rejects.toThrow('Stale or invalid');
    await settle(); const stopped = acquire.stop(); current.resolve(reply({ current: true }));
    await stale; await stopped; expect(mockFetch).toHaveBeenCalledTimes(1);
    sendMessage.mockImplementation(standard);
    const response = new Response(payload, { headers: { 'content-type': 'video/mp4' } });
    Object.defineProperty(response, 'url', { value: source.url }); mockFetch.mockResolvedValueOnce(response);
    video.play.mockRejectedValueOnce(new Error('Blob playback failed'));
    await expect(acquire.refetch()).rejects.toThrow('Blob playback failed');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:owned');
    expect(video.src).toBe(''); expect(acquire.snapshot().resources.objectUrls).toBe(0);
  });

  it('handles missing initial responses and stale refresh/GPU completions, cancelling page callbacks', async () => {
    sendMessage.mockResolvedValueOnce(undefined);
    await setup();
    expect(acquire.snapshot().events.some(event => event.type === 'selection-error' && String(event.value).includes('Rejected acquisition request'))).toBe(true);
    const oldInfo = deferred<unknown>(); sendMessage.mockReturnValueOnce(oldInfo.promise);
    const stale = expect(acquire.refresh()).rejects.toThrow(/superseded/i);
    await acquire.refresh(); oldInfo.resolve(reply({ ...selectionInfo, source: { ...source, sourceClass: 'stale' } })); await stale;
    expect(acquire.snapshot().info).toEqual(selectionInfo);
    const device = deferred<GPUDevice>();
    const destroy = vi.fn(), importExternalTexture = vi.fn();
    vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn().mockResolvedValue({ requestDevice: () => device.promise }) } });
    const probe = expect(acquire.probe()).rejects.toThrow(/cancel/i);
    await settle();
    const frame = video.requestVideoFrameCallback.mock.calls[0]![0];
    const json = vi.spyOn(JSON, 'stringify');
    const metadata = { mediaTime: 1, presentedFrames: 1, width: 320, height: 180 } as VideoFrameCallbackMetadata;
    frame(1, metadata); expect(json).not.toHaveBeenCalled();
    page.dispatchEvent(new Event('pagehide'));
    const callbacks = video.requestVideoFrameCallback.mock.calls.length;
    frame(2, metadata); expect(video.requestVideoFrameCallback).toHaveBeenCalledTimes(callbacks);
    device.resolve({ destroy, importExternalTexture } as unknown as GPUDevice); await probe;
    expect(importExternalTexture).not.toHaveBeenCalled(); expect(destroy).toHaveBeenCalledTimes(1);
    expect(removeMessageListener).toHaveBeenCalledTimes(1);
    expect(video.cancelVideoFrameCallback).toHaveBeenCalledTimes(1);
    await expect(acquire.direct()).rejects.toThrow(/disposed|pending/i);
  });
});