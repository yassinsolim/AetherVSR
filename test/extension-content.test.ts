import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { VideoAttachment, type AttachmentOptions } from '../src/extension/attachment.js';
import { discoverVideos, OwnerSelector } from '../src/extension/discovery.js';
import { inspectGeometry, type GeometryResult } from '../src/extension/geometry.js';
import { packModel, type ModelFile, type PackedModel } from '../src/core/neural/model.js';
import { acquireGpu } from '../src/core/gpu/device.js';
import { MODEL_SHA256, type ContentCommand, type ExtensionStatus, type ModelResponse } from '../src/extension/protocol.js';

vi.mock('../src/extension/attachment.js', () => ({ VideoAttachment: vi.fn() }));
vi.mock('../src/extension/discovery.js', () => ({ discoverVideos: vi.fn(), OwnerSelector: vi.fn() }));
vi.mock('../src/extension/geometry.js', () => ({ inspectGeometry: vi.fn() }));
vi.mock('../src/core/neural/model.js', () => ({ packModel: vi.fn() }));
vi.mock('../src/core/gpu/device.js', () => ({ acquireGpu: vi.fn() }));

const modelJson = readFileSync(new URL('../public/models/aethersr-c16d2.json', import.meta.url), 'utf8');
const modelResponse: ModelResponse = { ok: true, modelJson, sha256: MODEL_SHA256 };
const packed: PackedModel = { file: JSON.parse(modelJson) as ModelFile, features: 16, depth: 2,
  stemWeights: new Float32Array(), stemBias: new Float32Array(), bodyWeights: [], bodyBias: [],
  headWeights: new Float32Array(), headBias: new Float32Array() };
const worker = { id: 'test', url: 'chrome-extension://test/service-worker.js', tab: undefined };
type Sender = { id?: string; url?: string; tab?: object | undefined };
type Listener = (message: unknown, sender: Sender, reply: (response: unknown) => void) => boolean;
type Access = { status: () => ExtensionStatus; attachment: () => ReturnType<typeof makeAttachment> | null };
const storage = globalThis as unknown as Record<symbol, unknown> & { __AETHERVSR_EXTENSION_TEST__?: Access };
const CANDIDATES = new Map<TreeNode, GeometryResult>();
const observers: { callback: MutationCallback; targets: Set<Node> }[] = [];
const attachments: ReturnType<typeof makeAttachment>[] = [];
const sendMessage = vi.fn<() => Promise<ModelResponse>>();
const addListener = vi.fn<(listener: Listener) => void>();
const addWindowListener = vi.fn<(type: string, listener: EventListenerOrEventListenerObject | null) => void>();
const digest = vi.fn((algorithm: string, bytes: Uint8Array) =>
  Promise.resolve(Uint8Array.from(createHash(algorithm.replace('-', '').toLowerCase()).update(bytes).digest()).buffer));
let document: TreeNode;
let view: EventTarget;
let startResult: Promise<void>;
const access = () => storage.__AETHERVSR_EXTENSION_TEST__!;
const load = async () => { vi.resetModules(); await import('../src/extension/content.js'); };
const flush = async () => { for (let turn = 0; turn < 16; turn++) await Promise.resolve(); };
const advance = async (milliseconds = 150) => { vi.advanceTimersByTime(milliseconds); await flush(); };
function deferred<Value>() {
  let finish!: (value: Value) => void;
  let resolved = false;
  const promise = new Promise<Value>(resolve => { finish = resolve; });
  return { promise, resolve: (value: Value) => {
    if (resolved) throw new Error('Deferred resolved twice');
    resolved = true; finish(value);
  } };
}
function command(message: ContentCommand = { type: 'm10.start', mode: 'auto' }, sender = worker) {
  const reply = vi.fn();
  expect(addListener.mock.calls[0]![0](message, sender, reply)).toBe(false);
  expect(reply).toHaveBeenCalledWith({ ok: true, status: access().status() });
  return access().status();
}
function mutation(target: TreeNode, addedNodes?: TreeNode[], removedNodes: TreeNode[] = []) {
  const record = { target, type: addedNodes ? 'childList' : 'attributes', addedNodes: addedNodes ?? [], removedNodes };
  for (const observer of observers) if (observer.targets.size && target.isConnected)
    observer.callback([record as unknown as MutationRecord], {} as MutationObserver);
}
class TreeNode extends EventTarget {
  parent: TreeNode | null = null;
  children: TreeNode[] = [];
  paused = false; ended = false; mediaKeys = null; controls = false; src = 'first.mp4'; fullscreenElement = null;
  get isConnected(): boolean { return this === document || (this.parent?.isConnected ?? false); }
  get video(): HTMLVideoElement { return this as unknown as HTMLVideoElement; }
  append(child: TreeNode) { child.parent = this; this.children.push(child); mutation(this, [child]); }
  remove() {
    const parent = this.parent;
    if (!parent) return;
    parent.children = parent.children.filter(child => child !== this); this.parent = null; mutation(parent, [], [this]);
  }
}
function candidate() {
  const video = new TreeNode(); document.append(video);
  const rect = { left: 0, top: 0, width: 640, height: 360 };
  CANDIDATES.set(video, { ok: true, rect, clip: rect, objectFit: 'contain', objectPosition: '50% 50%',
    borderRadius: '0px', placement: { parent: document as unknown as Element, before: null }, style: {} });
  return video;
}
function makeAttachment(video: HTMLVideoElement, options: AttachmentOptions) {
  const canvas = new TreeNode();
  const attachment = { video, canvas, options, disposed: false,
    start: vi.fn(() => { document.append(canvas); return startResult; }),
    refresh: vi.fn(() => mutation(canvas)), setMode: vi.fn(),
    destroy: vi.fn(() => { if (!attachment.disposed) { attachment.disposed = true; canvas.remove(); } }),
    snapshot: () => ({ ready: !attachment.disposed, resources: { canvas: Number(canvas.isConnected) } }) };
  return attachment;
}
beforeEach(() => {
  vi.resetAllMocks(); CANDIDATES.clear(); observers.length = 0; attachments.length = 0;
  vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  vi.setSystemTime(0); document = new TreeNode(); view = new EventTarget(); startResult = Promise.resolve();
  for (const name of ['window', 'self', 'top']) vi.stubGlobal(name, globalThis);
  addWindowListener.mockImplementation((type, listener) => view.addEventListener(type, listener));
  vi.stubGlobal('addEventListener', addWindowListener);
  vi.stubGlobal('removeEventListener', view.removeEventListener.bind(view));
  vi.stubGlobal('dispatchEvent', view.dispatchEvent.bind(view));
  vi.stubGlobal('document', document); vi.stubGlobal('location', { protocol: 'https:' });
  vi.stubGlobal('__AETHERVSR_TEST__', true); vi.stubGlobal('__AETHERVSR_EXTENSION_TEST__', undefined);
  vi.stubGlobal('crypto', { subtle: { digest } });
  vi.stubGlobal('chrome', { runtime: { id: 'test', getURL: (path: string) => `chrome-extension://test/${path}`,
    sendMessage, onMessage: { addListener } } });
  vi.stubGlobal('MutationObserver', vi.fn(function (callback: MutationCallback) {
    const targets = new Set<Node>(); observers.push({ callback, targets });
    return { observe: (target: Node) => { targets.add(target); }, disconnect: () => targets.clear() };
  }));
  digest.mockImplementation((algorithm, bytes) =>
    Promise.resolve(Uint8Array.from(createHash(algorithm.replace('-', '').toLowerCase()).update(bytes).digest()).buffer));
  sendMessage.mockResolvedValue(modelResponse); vi.mocked(packModel).mockReturnValue(packed);
  vi.mocked(VideoAttachment).mockImplementation(function (video, options) {
    const attachment = makeAttachment(video, options); attachments.push(attachment);
    return attachment as unknown as VideoAttachment;
  });
  vi.mocked(discoverVideos).mockImplementation(() => ({ videos: [...CANDIDATES.keys()].filter(video => video.isConnected)
    .map(video => video.video), openRoots: [], embeddedFrames: 0, truncated: false, closedShadowLimitation: true }));
  vi.mocked(inspectGeometry).mockImplementation(video => video.controls
    ? { ok: false, code: 'unsupported-controls', reason: 'Native controls visible.' }
    : CANDIDATES.get(video as unknown as TreeNode)!);
  vi.mocked(OwnerSelector).mockImplementation(function () {
    return { update: vi.fn((candidates: Parameters<OwnerSelector['update']>[0]) =>
      candidates.find(candidate => candidate.eligible && candidate.playing)?.video ??
      candidates.find(candidate => candidate.eligible)?.video ?? null) } as unknown as OwnerSelector;
  });
});
afterEach(async () => {
  view.dispatchEvent(new Event('pagehide')); await flush();
  for (const id of ['test', 'other']) Reflect.deleteProperty(storage, Symbol.for(`aethervsr.m10.document.${id}`));
  vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers();
});

describe('isolated top-document content manager', () => {
  it('is dormant until explicitly enabled', async () => {
    candidate(); await load(); await advance(2000);
    expect(command({ type: 'm10.inspect' })).toMatchObject({ enabled: false, code: 'inactive',
      details: { discoveryActive: false, timerCount: 0 } });
    expect(observers).toHaveLength(0); expect(vi.getTimerCount()).toBe(0);
    expect(discoverVideos).not.toHaveBeenCalled(); expect(sendMessage).not.toHaveBeenCalled();
    expect(VideoAttachment).not.toHaveBeenCalled(); expect(acquireGpu).not.toHaveBeenCalled();
  });
  it('deduplicates reinjection per extension id, not across extension ids', async () => {
    await load(); const original = access(); await load();
    expect(access()).toBe(original); expect(addListener).toHaveBeenCalledTimes(1);
    vi.stubGlobal('chrome', { runtime: { ...chrome.runtime, id: 'other' } }); await load();
    expect(access()).not.toBe(original); expect(addListener).toHaveBeenCalledTimes(2);
    expect(storage[Symbol.for('aethervsr.m10.document.test')]).toBeDefined();
    expect(storage[Symbol.for('aethervsr.m10.document.other')]).toBeDefined();
  });
  it('shares one verified model and attachment across duplicate pending and active starts', async () => {
    candidate(); const pending = deferred<ModelResponse>(); sendMessage.mockReturnValue(pending.promise);
    await load(); command(); command(); expect(sendMessage).toHaveBeenCalledTimes(1);
    pending.resolve(modelResponse); await flush(); command({ type: 'm10.start', mode: 'baseline' });
    expect(createHash('sha256').update(modelJson).digest('hex')).toBe(MODEL_SHA256);
    expect(digest).toHaveBeenCalledWith('SHA-256', new TextEncoder().encode(modelJson));
    expect(packModel).toHaveBeenCalledExactlyOnceWith(JSON.parse(modelJson));
    expect(attachments).toHaveLength(1); expect(attachments[0]!.options.model).toBe(packed);
    expect(attachments[0]!.setMode).toHaveBeenCalledWith('baseline');
    expect(access().status().details).toMatchObject({ infrastructure: { maximumConcurrent: 1, created: 1 } });
  });
  it('disables all owned observers, timers and document listeners, including queued discovery', async () => {
    const add = vi.spyOn(document, 'addEventListener'); const remove = vi.spyOn(document, 'removeEventListener');
    candidate(); await load(); command(); await flush(); document.dispatchEvent(new Event('resize'));
    expect(vi.getTimerCount()).toBe(2); expect(observers[0]!.targets.size).toBe(1);
    expect(command({ type: 'm10.stop' }).details).toMatchObject({ discoveryActive: false, timerCount: 0 });
    expect(remove.mock.calls).toEqual(add.mock.calls); expect(add).toHaveBeenCalledTimes(7);
    expect(observers.every(observer => observer.targets.size === 0)).toBe(true);
    const calls = vi.mocked(discoverVideos).mock.calls.length; document.dispatchEvent(new Event('playing')); await advance(2000);
    expect(discoverVideos).toHaveBeenCalledTimes(calls); expect(vi.getTimerCount()).toBe(0);
    expect(attachments[0]!.destroy).toHaveBeenCalledTimes(1);
  });
  it('does not construct from a model resolving after disable', async () => {
    candidate(); const pending = deferred<ModelResponse>(); sendMessage.mockReturnValue(pending.promise);
    await load(); command(); command({ type: 'm10.stop' }); pending.resolve(modelResponse); await flush();
    expect(VideoAttachment).not.toHaveBeenCalled(); expect(vi.getTimerCount()).toBe(0);
  });
  it('ignores an older model generation after disable and re-enable', async () => {
    const old = candidate(); const first = deferred<ModelResponse>(); const second = deferred<ModelResponse>();
    sendMessage.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    await load(); command(); command({ type: 'm10.stop' }); CANDIDATES.delete(old); const current = candidate(); command();
    second.resolve(modelResponse); await flush(); first.resolve(modelResponse); await flush();
    expect(attachments.map(attachment => attachment.video)).toEqual([current.video]);
    expect(sendMessage).toHaveBeenCalledTimes(2); expect(access().attachment()!.disposed).toBe(false);
  });
  it('keeps only the latest owner through multiple pending attachment starts and stale failures', async () => {
    let owner = candidate(); const pending = deferred<void>(); startResult = pending.promise;
    await load(); command(); await flush();
    for (let change = 0; change < 2; change++) {
      CANDIDATES.delete(owner); owner = candidate(); await advance();
    }
    attachments[0]!.options.onFailure('device-lost', 'obsolete'); pending.resolve(); await flush();
    expect(attachments).toHaveLength(3); expect(attachments.filter(attachment => !attachment.disposed)).toHaveLength(1);
    expect(access().attachment()!.video).toBe(owner.video);
    expect(access().status()).toMatchObject({ code: 'active', details: { infrastructure: { maximumConcurrent: 1 } } });
  });
  it('blocks a failed owner without automatic retry until disable and enable', async () => {
    candidate(); await load(); command(); await flush(); attachments[0]!.options.onFailure('device-lost', 'lost');
    await advance(3000); command(); document.dispatchEvent(new Event('playing')); await advance();
    expect(access().status().code).toBe('device-lost'); expect(attachments).toHaveLength(1);
    expect(attachments[0]!.disposed).toBe(true); expect(vi.getTimerCount()).toBe(1);
    command({ type: 'm10.stop' }); command(); await flush(); expect(attachments).toHaveLength(2);
  });
  it('coalesces page mutations and ignores its own canvas insertion and refresh echoes', async () => {
    candidate(); await load(); command(); await flush(); expect(vi.getTimerCount()).toBe(1);
    mutation(document); mutation(document); expect(vi.getTimerCount()).toBe(2); await advance(1500);
    expect(discoverVideos).toHaveBeenCalledTimes(2); expect(attachments[0]!.refresh).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(1); expect(access().status().details).toMatchObject({ infrastructure: { mutationBatches: 2 } });
  });
  it('refreshes a replaced source on the same video without constructing another runtime', async () => {
    const video = candidate(); await load(); command(); await flush(); video.src = 'replacement.mp4'; mutation(video); await advance();
    expect(attachments).toHaveLength(1); expect(attachments[0]!.refresh).toHaveBeenCalledTimes(1);
    expect(attachments[0]!.destroy).not.toHaveBeenCalled();
  });
  it('destroys a removed owner before constructing its replacement', async () => {
    const old = candidate(); await load(); command(); await flush(); old.remove(); const current = candidate(); await advance();
    expect(attachments.map(attachment => attachment.video)).toEqual([old.video, current.video]);
    expect(attachments[0]!.destroy.mock.invocationCallOrder[0]).toBeLessThan(vi.mocked(VideoAttachment).mock.invocationCallOrder[1]!);
    expect(attachments.filter(attachment => !attachment.disposed)).toHaveLength(1);
  });
  it('rejects native controls both initially and when shown on the current owner', async () => {
    const video = candidate(); video.controls = true; await load(); command(); await flush();
    expect(access().status().code).toBe('unsupported-controls'); expect(sendMessage).not.toHaveBeenCalled();
    video.controls = false; mutation(video); await advance(); expect(attachments).toHaveLength(1);
    video.controls = true; mutation(video); await advance();
    expect(access().attachment()).toBeNull(); expect(attachments[0]!.disposed).toBe(true);
    expect(access().status().code).toBe('unsupported-controls');
  });
  it('creates no adapter in subframes or non-web origins', async () => {
    for (const [top, protocol] of [[{}, 'https:'], [globalThis, 'file:'], [globalThis, 'chrome-extension:']] as const) {
      vi.stubGlobal('top', top); vi.stubGlobal('location', { protocol }); await load();
      expect(access()).toBeUndefined(); expect(addListener).not.toHaveBeenCalled();
    }
    expect(VideoAttachment).not.toHaveBeenCalled(); expect(vi.getTimerCount()).toBe(0);
  });
  it('has no MAIN-world message bridge and rejects every non-worker sender role', async () => {
    candidate(); await load(); window.dispatchEvent(new MessageEvent('message', { data: { type: 'm10.start', mode: 'auto' } }));
    expect(addWindowListener.mock.calls.map(call => call[0])).toEqual(['pagehide']);
    const senders: Sender[] = [{}, { ...worker, id: 'foreign' }, { ...worker, id: '' },
      { ...worker, tab: { id: 1 } }, { ...worker, url: 'https://page.test/' },
      { ...worker, url: 'chrome-extension://test/popup.html' }];
    for (const sender of senders) {
      const reply = vi.fn(); expect(addListener.mock.calls[0]![0]({ type: 'm10.start', mode: 'auto' }, sender, reply)).toBe(false);
      expect(reply).not.toHaveBeenCalled();
    }
    expect(access().status().enabled).toBe(false); expect(sendMessage).not.toHaveBeenCalled();
    command(); await flush(); expect(attachments).toHaveLength(1);
  });
  it('constructs only the latest selection when ownership changes while the model is pending', async () => {
    const original = candidate(); const pending = deferred<ModelResponse>(); sendMessage.mockReturnValue(pending.promise);
    await load(); command(); CANDIDATES.delete(original); const middle = candidate(); await advance();
    CANDIDATES.delete(middle); const latest = candidate(); await advance();
    expect(original.isConnected).toBe(true); expect(attachments).toHaveLength(0);
    expect(discoverVideos).toHaveBeenCalledTimes(3); expect(sendMessage).toHaveBeenCalledTimes(1);
    pending.resolve(modelResponse); await flush();
    expect(attachments.map(attachment => attachment.video)).toEqual([latest.video]);
  });
});