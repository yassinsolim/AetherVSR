/// <reference types="chrome" />
import { packModel, type ModelFile, type PackedModel } from '../core/neural/model.js';
import { VideoAttachment } from './attachment.js';
import { discoverVideos, OwnerSelector } from './discovery.js';
import { inspectGeometry, type GeometryResult } from './geometry.js';
import { MODEL_SHA256, parseContentCommand, parseModelResponse, type ExtensionStatus,
  type RuntimeMode, type StatusCode } from './protocol.js';

declare const __AETHERVSR_TEST__: boolean;

class DocumentAdapter {
  private enabled = false;
  private mode: RuntimeMode = 'auto';
  private generation = 0;
  private attachment: VideoAttachment | null = null;
  private model: Promise<PackedModel> | null = null;
  private selector = new OwnerSelector();
  private observer: MutationObserver | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private watchdog: ReturnType<typeof setInterval> | null = null;
  private listeners: [EventTarget, string, EventListener][] = [];
  private videos: HTMLVideoElement[] = [];
  private ids = new WeakMap<HTMLVideoElement, number>();
  private nextId = 0;
  private blocked = new WeakMap<HTMLVideoElement, { code: StatusCode; message: string }>();
  private fault: { code: StatusCode; message: string } | null = null;
  private embeddedFrames = 0;
  private starting = false;
  private pendingOwner: HTMLVideoElement | null = null;
  private lastResources: ReturnType<VideoAttachment['snapshot']> | null = null;
  private readonly counters = { discoveryCalls: 0, discoveryMs: 0, geometryCalls: 0, geometryMs: 0,
    mutationBatches: 0, ownerChanges: 0, created: 0, destroyed: 0, maximumConcurrent: 0 };
  readonly transitions: { at: number; owner: string | null; reason: string }[] = [];
  private testing: { forceCopy?: boolean; withheldFeatures?: GPUFeatureName[] } = {};

  private id(video: HTMLVideoElement): number {
    let id = this.ids.get(video);
    if (id === undefined) { id = ++this.nextId; this.ids.set(video, id); }
    return id;
  }

  start(mode: RuntimeMode): ExtensionStatus {
    this.mode = mode;
    if (this.enabled) { this.attachment?.setMode(mode); return this.status(); }
    this.enabled = true;
    this.fault = null;
    this.blocked = new WeakMap();
    this.selector = new OwnerSelector();
    this.observer = new MutationObserver(records => {
      if (records.every(record => record.target === this.attachment?.canvas ||
        (record.type === 'childList' && [...record.addedNodes, ...record.removedNodes]
          .every(node => node === this.attachment?.canvas)))) return;
      this.counters.mutationBatches++;
      this.schedule();
    });
    this.listen(document);
    this.watchdog = setInterval(() => {
      if (!chrome.runtime?.id) { this.stop(); return; }
      if (this.attachment && !this.attachment.video.isConnected) {
        this.detach('owner removed');
        this.schedule();
      } else if (this.attachment?.snapshot().ready && !this.attachment.canvas.isConnected) {
        const owner = this.attachment.video;
        this.blocked.set(owner, { code: 'unsupported-geometry', message: 'The page removed the owned output. Disable and enable to retry.' });
        this.detach('overlay removed');
        this.schedule();
      }
    }, 500);
    this.reconcile();
    return this.status();
  }

  setMode(mode: RuntimeMode): ExtensionStatus {
    this.mode = mode;
    this.attachment?.setMode(mode);
    return this.status();
  }

  private schedule(): void {
    if (!this.enabled || this.timer !== null) return;
    this.timer = setTimeout(() => { this.timer = null; this.reconcile(); }, 150);
  }

  private listen(target: EventTarget): void {
    for (const type of ['playing', 'pause', 'loadeddata', 'emptied', 'resize', 'fullscreenchange', 'visibilitychange']) {
      const listener = () => this.schedule();
      target.addEventListener(type, listener, true);
      this.listeners.push([target, type, listener]);
    }
  }

  private reconcile(): void {
    if (!this.enabled) return;
    const started = performance.now();
    const found = discoverVideos(document);
    this.counters.discoveryCalls++;
    this.counters.discoveryMs += performance.now() - started;
    this.videos = found.videos;
    this.embeddedFrames = found.embeddedFrames;
    this.observer?.disconnect();
    const observation: MutationObserverInit = { childList: true, subtree: true, attributes: true,
      attributeFilter: ['src', 'class', 'style', 'controls', 'hidden', 'width', 'height'] };
    this.observer?.observe(document, observation);
    for (const root of found.openRoots) this.observer?.observe(root, observation);
    const roots = new Set<EventTarget>([document, ...found.openRoots]);
    this.listeners = this.listeners.filter(([target, type, listener]) => {
      if (roots.has(target)) return true;
      target.removeEventListener(type, listener, true);
      return false;
    });
    const listening = new Set(this.listeners.map(([target]) => target));
    for (const root of roots) if (!listening.has(root)) this.listen(root);
    const geometry = new Map<HTMLVideoElement, GeometryResult>();
    const candidates = found.videos.map(video => {
      const before = performance.now();
      const result = inspectGeometry(video);
      this.counters.geometryCalls++;
      this.counters.geometryMs += performance.now() - before;
      geometry.set(video, result);
      const keepSuspended = this.attachment?.video === video && video.isConnected && !video.mediaKeys &&
        (!result.ok && (result.code === 'offscreen' || result.code === 'video-not-ready' || document.fullscreenElement === video));
      return { video, id: this.id(video), playing: !video.paused && !video.ended,
        visibleArea: result.ok ? result.clip.width * result.clip.height : keepSuspended ? 1 : 0,
        eligible: !this.blocked.has(video) && !video.mediaKeys && (keepSuspended ||
          result.ok && result.rect.width >= 160 && result.rect.height >= 90) };
    });
    const selected = this.selector.update(candidates, performance.now());
    if (this.starting && selected !== this.pendingOwner) this.detach('pending owner changed');
    if (this.attachment && this.attachment.video !== selected) this.detach('primary owner changed');
    if (selected && !this.attachment && !this.starting) this.attach(selected);
    else if (this.attachment) this.attachment.refresh();
    if (found.truncated) this.fault = { code: 'unsupported', message: 'Discovery reached its bounded node limit; not all video is inspected.' };
    if (!selected && !this.fault) {
      const protectedVideo = found.videos.some(video => video.mediaKeys);
      const blocked = found.videos.map(video => this.blocked.get(video)).find(Boolean);
      const rejected = found.videos.map(video => geometry.get(video)).find(result => result && !result.ok);
      this.fault = protectedVideo ? { code: 'protected-media', message: 'Protected video intentionally remains unsupported.' }
        : blocked ?? (rejected && !rejected.ok ? { code: rejected.code === 'offscreen' || rejected.code === 'video-not-ready'
          ? 'unsupported-media' : rejected.code, message: rejected.reason }
          : { code: found.embeddedFrames ? 'unsupported-frame' : 'no-video', message: found.embeddedFrames
            ? 'Embedded frames are not inspected. This version supports top-document video only.'
            : 'No eligible top-document video found. Inaccessible closed shadow video cannot be inspected.' });
    }
    if (candidates.some(candidate => candidate.video !== selected && candidate.eligible && candidate.playing)) this.schedule();
  }

  private loadModel(): Promise<PackedModel> {
    this.model ??= (async () => {
      const response = parseModelResponse(await chrome.runtime.sendMessage({ type: 'm10.model' }));
      if (!response?.ok) throw new Error(response?.message ?? 'Invalid packaged model response.');
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(response.modelJson));
      const hash = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
      if (hash !== MODEL_SHA256) throw new Error('Packaged model hash mismatch.');
      return packModel(JSON.parse(response.modelJson) as ModelFile);
    })();
    return this.model;
  }

  private attach(video: HTMLVideoElement): void {
    this.starting = true;
    this.pendingOwner = video;
    const generation = ++this.generation;
    this.fault = null;
    void this.loadModel().then(async model => {
      if (!this.enabled || generation !== this.generation || !video.isConnected) return;
      const attachment = new VideoAttachment(video, { mode: this.mode, model, ...this.testing,
        onFailure: (code, message) => {
          if (generation !== this.generation) return;
          this.blocked.set(video, { code, message });
          this.fault = { code, message };
          this.detach('attachment unsupported or failed');
          this.schedule();
        } });
      this.attachment = attachment;
      this.counters.created++;
      this.counters.maximumConcurrent = Math.max(this.counters.maximumConcurrent, this.counters.created - this.counters.destroyed);
      this.counters.ownerChanges++;
      this.log(`video-${this.id(video)}`, 'attached');
      await attachment.start();
      if (!this.enabled || generation !== this.generation) attachment.destroy();
    }).catch(error => {
      if (generation !== this.generation) return;
      this.fault = { code: 'error', message: `Model/runtime initialization failed: ${String(error).slice(0, 500)}` };
      this.blocked.set(video, this.fault);
    }).finally(() => {
      if (generation === this.generation) { this.starting = false; this.pendingOwner = null; }
    });
  }

  private log(owner: string | null, reason: string): void {
    this.transitions.push({ at: performance.now(), owner, reason });
    if (this.transitions.length > 64) this.transitions.shift();
  }

  private detach(reason: string): void {
    this.generation++;
    this.starting = false;
    this.pendingOwner = null;
    const attachment = this.attachment;
    this.attachment = null;
    if (attachment) {
      attachment.destroy();
      this.lastResources = attachment.snapshot();
      this.counters.destroyed++;
      this.log(null, reason);
    }
  }

  stop(): ExtensionStatus {
    this.enabled = false;
    this.detach('disabled');
    this.observer?.disconnect();
    this.observer = null;
    if (this.timer !== null) clearTimeout(this.timer);
    if (this.watchdog !== null) clearInterval(this.watchdog);
    this.timer = null;
    this.watchdog = null;
    for (const [target, type, listener] of this.listeners) target.removeEventListener(type, listener, true);
    this.listeners.length = 0;
    this.model = null;
    this.videos = [];
    this.fault = null;
    return this.status();
  }

  status(): ExtensionStatus {
    const attachment = this.attachment?.snapshot();
    return { schemaVersion: 1, enabled: this.enabled,
      code: !this.enabled ? 'inactive' : this.fault?.code ?? (attachment?.ready ? 'active' : attachment?.suspendedReason ? 'suspended' : 'discovering'),
      message: !this.enabled ? 'Enhancement is disabled; original video is unchanged.' : this.fault?.message ??
        (attachment?.suspendedReason ? `Original video shown: ${attachment.suspendedReason}` :
          attachment?.controllerState === 'unavailable' ? 'GPU timestamps unavailable; Catmull-Rom only.' :
            attachment?.controllerReason ?? 'Finding supported top-document video.'),
      mode: this.mode, current: attachment?.current ?? null, candidates: this.videos.length,
      embeddedFrames: this.embeddedFrames, owner: this.attachment ? `video-${this.id(this.attachment.video)}` : null,
      details: { attachment: attachment ? { ...attachment } : null, lastTeardown: this.lastResources?.resources ?? null,
        infrastructure: { ...this.counters }, discoveryActive: this.observer !== null,
        timerCount: Number(this.timer !== null) + Number(this.watchdog !== null),
        limitations: 'Top-document video only; closed-shadow/iframe video cannot be inspected. Host permission does not override media origin-clean rules.' } };
  }

  readonly testAccess = __AETHERVSR_TEST__ ? () => {
    return { attachment: () => this.attachment, status: () => this.status(),
      configure: (options: typeof this.testing) => { if (this.enabled) throw new Error('Disable before test configuration'); this.testing = options; },
      transitions: () => this.transitions.slice() };
  } : undefined;
}

if (window === window.top && (location.protocol === 'http:' || location.protocol === 'https:')) {
  const key = Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`);
  const storage = globalThis as unknown as Record<symbol, DocumentAdapter>;
  if (!storage[key]) {
    const adapter = new DocumentAdapter();
    storage[key] = adapter;
    chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
      if (sender.id !== chrome.runtime.id || sender.tab !== undefined ||
        sender.url !== chrome.runtime.getURL('service-worker.js')) return false;
      const command = parseContentCommand(message);
      if (!command) return false;
      const status = command.type === 'm10.inspect' ? adapter.status() : command.type === 'm10.stop' ? adapter.stop()
        : 'mode' in command ? command.type === 'm10.start' ? adapter.start(command.mode) : adapter.setMode(command.mode)
          : adapter.status();
      sendResponse({ ok: true, status });
      return false;
    });
    window.addEventListener('pagehide', () => adapter.stop());
    if (__AETHERVSR_TEST__) Object.defineProperty(globalThis, '__AETHERVSR_EXTENSION_TEST__', { value: adapter.testAccess?.() });
  }
}