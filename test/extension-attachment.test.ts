import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { VideoAttachment, type AttachmentOptions } from '../src/extension/attachment.js';
import { geometryProofCurrent, inspectGeometry } from '../src/extension/geometry.js';
import { inactiveStatus, parseExtensionStatus } from '../src/extension/protocol.js';
import { acquireGpu, watchDeviceFailures, type GpuContext } from '../src/core/gpu/device.js';
import { VideoPipeline } from '../src/core/pipeline.js';
import type { PackedModel } from '../src/core/neural/model.js';
import type { Upscaler } from '../src/core/types.js';
import { RuntimeDriver } from '../src/runtime.js';
import { NeuralUpscaler } from '../src/core/upscale/neural-upscaler.js';

vi.mock('../src/extension/geometry.js', () => ({ inspectGeometry: vi.fn(), geometryProofCurrent: vi.fn() }));
vi.mock('../src/core/gpu/device.js', () => ({ acquireGpu: vi.fn(), watchDeviceFailures: vi.fn() }));
vi.mock('../src/core/pipeline.js', () => ({ VideoPipeline: vi.fn() }));
vi.mock('../src/core/upscale/baseline-scaler.js', () => ({
  BaselineScaler: vi.fn(function () { return upscaler(false); }),
}));
vi.mock('../src/core/upscale/neural-upscaler.js', () => ({
  NEURAL_OPTIONAL_FEATURES: ['shader-f16'],
  NeuralUpscaler: vi.fn(function () { return upscaler(true); }),
}));

function upscaler(neural: boolean): Upscaler {
  return { id: neural ? 'neural' : 'catmull-rom', label: 'test', neural, scaleFactor: 2,
    configure: vi.fn(), encode: vi.fn(), destroy: vi.fn() };
}

class Style {
  private values = new Map<string, [string, string]>();
  setProperty = vi.fn((key: string, value: string, priority: string = '') => { this.values.set(key, [value, priority]); });
  getPropertyValue(key: string) { return this.values.get(key)?.[0] ?? ''; }
  getPropertyPriority(key: string) { return this.values.get(key)?.[1] ?? ''; }
  removeProperty(key: string) { this.values.delete(key); }
}

class FakeElement extends EventTarget {
  nodeType = 1;
  style = new Style();
  parentNode: FakeElement | null = null;
  parentElement: FakeElement | null = null;
  nextSibling: FakeElement | null = null;
  children: FakeElement[] = [];
  dataset: Record<string, string> = {};
  width = 0;
  height = 0;
  setAttribute = vi.fn();
  getAttribute = vi.fn((): string | null => null);
  getBoundingClientRect() { return { left: Number.parseFloat(this.style.getPropertyValue('left')) || 10, top: Number.parseFloat(this.style.getPropertyValue('top')) || 20,
    width: Number.parseFloat(this.style.getPropertyValue('width')) || 640, height: Number.parseFloat(this.style.getPropertyValue('height')) || 360 } as DOMRect; }
  insertBefore = vi.fn((child: FakeElement, before: FakeElement | null) => {
    child.remove();
    const index = before === null ? this.children.length : this.children.indexOf(before);
    this.children.splice(index, 0, child);
    child.parentNode = this;
    child.parentElement = this;
    this.link();
    return child;
  });
  contains(element: FakeElement): boolean { return this === element || this.children.some(child => child.contains(element)); }
  remove = vi.fn(() => {
    const parent = this.parentNode;
    if (parent) {
      parent.children.splice(parent.children.indexOf(this), 1);
      parent.link();
    }
    this.parentNode = null;
    this.parentElement = null;
    this.nextSibling = null;
  });
  private link() {
    this.children.forEach((child, index) => { child.nextSibling = this.children[index + 1] ?? null; });
  }
}

class FakeTracks extends EventTarget {
  [index: number]: { mode: TextTrackMode };
  length = 1;
  constructor() {
    super();
    this[0] = { mode: 'disabled' };
  }
}

class FakeVideo extends FakeElement {
  paused = false;
  seeking = false;
  ended = false;
  playbackRate = 1;
  readyState = 4;
  videoWidth = 640;
  videoHeight = 360;
  isConnected = true;
  mediaKeys: object | null = null;
  error: { code: number; message: string } | null = null;
  textTracks = new FakeTracks();
  ownerDocument!: ReturnType<typeof makeDocument>;
  quality = { totalVideoFrames: 0, droppedVideoFrames: 0, corruptedVideoFrames: 0 };
  getVideoPlaybackQuality() { return { ...this.quality }; }
}

function makeDocument() {
  const frames = new Map<number, FrameRequestCallback>();
  let sequence = 0;
  const view = Object.assign(new EventTarget(), {
    requestAnimationFrame: vi.fn((callback: FrameRequestCallback) => { frames.set(++sequence, callback); return sequence; }),
    cancelAnimationFrame: vi.fn((id: number) => { frames.delete(id); }),
  });
  return Object.assign(new EventTarget(), {
    visibilityState: 'visible', fullscreenElement: null as FakeElement | null,
    pictureInPictureElement: null as FakeVideo | null, defaultView: view,
    createElement: vi.fn(() => new FakeElement()), frames,
    flush: () => {
      const pending = [...frames.values()];
      frames.clear();
      for (const callback of pending) callback(performance.now());
    },
  });
}

const attachments: VideoAttachment[] = [];

function harness(options: Partial<AttachmentOptions> = {}) {
  const document = makeDocument();
  vi.stubGlobal('document', document);
  const video = new FakeVideo();
  video.ownerDocument = document;
  const listeners = [video, video.textTracks, document, document.defaultView].map(target => ({
    target, add: vi.spyOn(target, 'addEventListener'), remove: vi.spyOn(target, 'removeEventListener'),
  }));
  const parent = new FakeElement();
  const next = new FakeElement();
  parent.insertBefore(video, null);
  parent.insertBefore(next, null);
  const observer = { observe: vi.fn(), unobserve: vi.fn(), disconnect: vi.fn() };
  let resize: ResizeObserverCallback | null = null;
  vi.stubGlobal('ResizeObserver', vi.fn(function (callback: ResizeObserverCallback) { resize = callback; return observer; }));
  const mutationObserver = { observe: vi.fn(), disconnect: vi.fn() };
  let mutation: MutationCallback | null = null;
  vi.stubGlobal('MutationObserver', vi.fn(function (callback: MutationCallback) { mutation = callback; return mutationObserver; }));
  const device = { destroy: vi.fn(), features: new Set(['timestamp-query', 'shader-f16']) };
  const gpu = { device, capabilities: { timestampQuery: true } } as unknown as GpuContext;
  vi.mocked(acquireGpu).mockResolvedValue(gpu);
  const unwatch = vi.fn();
  vi.mocked(watchDeviceFailures).mockReturnValue(unwatch);
  vi.mocked(geometryProofCurrent).mockReturnValue(true);
  vi.mocked(inspectGeometry).mockImplementation(() => ({
    ok: true, rect: { left: 10, top: 20, width: 640, height: 360 },
    clip: { left: 10, top: 20, width: 640, height: 360 },
    objectFit: 'contain', objectPosition: '50% 50%', borderRadius: '0px',
    placement: { parent: video.parentNode as unknown as Element, before: video.nextSibling as unknown as ChildNode | null },
    style: { all: 'initial', position: 'fixed', left: '10px', top: '20px', width: '640px', height: '360px', 'pointer-events': 'none' },
  }));
  const pipeline = {
    currentUpscaler: upscaler(false), running: false, error: null as unknown, timingGeneration: 0,
    onFrame: null as VideoPipeline['onFrame'], onGpuSample: null as VideoPipeline['onGpuSample'],
    onConfiguration: null as VideoPipeline['onConfiguration'],
    start: vi.fn(() => { pipeline.running = true; }), stop: vi.fn(() => { pipeline.running = false; }),
    invalidateTiming: vi.fn(() => { pipeline.timingGeneration++; }),
    setUpscaler: vi.fn((next: Upscaler) => { pipeline.currentUpscaler.destroy(); pipeline.currentUpscaler = next; }),
    destroy: vi.fn(() => { pipeline.stop(); pipeline.currentUpscaler.destroy(); }),
  };
  vi.mocked(VideoPipeline).mockImplementation(function () { return pipeline as unknown as VideoPipeline; });
  const failure = vi.fn();
  const model = {} as PackedModel;
  const attachment = new VideoAttachment(video as unknown as HTMLVideoElement, { mode: 'auto', model, onFailure: failure, ...options });
  attachments.push(attachment);
  const frame = () => {
    video.quality.totalVideoFrames++;
    attachment.canvas.width = video.videoWidth * 2;
    attachment.canvas.height = video.videoHeight * 2;
    pipeline.onFrame?.({ now: performance.now(), mediaTime: video.quality.totalVideoFrames / 60,
      presentationTime: performance.now() - 1, expectedDisplayTime: performance.now() + 1,
      presentedDelta: 1, size: { width: video.videoWidth, height: video.videoHeight }, decodeLatencyMs: null });
  };
  return { attachment, canvas: attachment.canvas as unknown as FakeElement,
    document, video, parent, next, observer, device, gpu, unwatch, failure, pipeline, frame, listeners,
    resize: () => resize?.([], observer), mutationObserver,
    mutate: (records: MutationRecord[]) => mutation?.(records, mutationObserver as unknown as MutationObserver) };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
  vi.setSystemTime(0);
  vi.spyOn(performance, 'now').mockImplementation(() => Date.now());
});

afterEach(() => {
  for (const attachment of attachments.splice(0)) attachment.destroy();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('VideoAttachment', () => {
  it('invalidates changed video inputs before fractional clipping edges can compound', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, video, canvas, frame } = harness({presentationWatchdog:true});
    await attachment.start(); frame();
    const rect = video.getBoundingClientRect();
    video.getBoundingClientRect = () => ({...rect,left:rect.left+0.32,width:rect.width+0.32});
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
  });

  it('retains the stricter container placement tolerance in the watchdog', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, canvas, frame } = harness({presentationWatchdog:true});
    const originalInspect = vi.mocked(inspectGeometry).getMockImplementation()!;
    vi.mocked(inspectGeometry).mockImplementation(element => ({...originalInspect(element),verifyPlacement:true}));
    await attachment.start(); frame();
    const rect = canvas.getBoundingClientRect();
    canvas.getBoundingClientRect = () => ({...rect,left:rect.left+1/64});
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
  });

  it('hides a stale style proof before submission without performing full inspection in the frame', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, canvas, frame, document } = harness({presentationWatchdog:true});
    await attachment.start(); frame();
    const calls = vi.mocked(inspectGeometry).mock.calls.length;
    vi.mocked(geometryProofCurrent).mockReturnValue(false);
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(inspectGeometry).toHaveBeenCalledTimes(calls);
    vi.mocked(geometryProofCurrent).mockReturnValue(true);
    document.flush(); frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
  });

  it('bounds image-sized canvas error independently of the smaller video box error', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, canvas, video, frame } = harness({presentationWatchdog:true});
    const originalInspect = vi.mocked(inspectGeometry).getMockImplementation()!;
    vi.mocked(inspectGeometry).mockImplementation(element => {
      const geometry = originalInspect(element);
      return geometry.ok ? {...geometry,objectFit:'none',style:{...geometry.style,width:'1280px'}} : geometry;
    });
    await attachment.start(); frame();
    const rect = video.getBoundingClientRect();
    video.getBoundingClientRect = () => ({...rect,width:rect.width + 0.32});
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
  });

  it('observes bounded control dependencies identified by geometry inspection', async () => {
    const { attachment, next, parent, mutationObserver, mutate, frame, canvas } = harness();
    const originalInspect = vi.mocked(inspectGeometry).getMockImplementation()!;
    vi.mocked(inspectGeometry).mockImplementation(element => ({...originalInspect(element),proof:{styles:[{
      element:next as unknown as Element,parent:parent as unknown as Node,style:{} as CSSStyleDeclaration,keys:[],values:[],readers:[],names:[],namedValues:[],
    }],clips:[],viewport:null}}));
    await attachment.start(); frame();
    expect(mutationObserver.observe).toHaveBeenCalledWith(next,expect.objectContaining({attributes:true,childList:true}));
    mutate([{type:'attributes',target:next,attributeName:'style',oldValue:'old'} as unknown as MutationRecord]);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
  });

  it('does not repeat full inspection for an unchanged offscreen layout and notices its return', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, video, frame, document, canvas } = harness({presentationWatchdog:true});
    await attachment.start(); frame();
    const originalInspect = vi.mocked(inspectGeometry).getMockImplementation()!;
    const rect = video.getBoundingClientRect();
    video.getBoundingClientRect = () => ({...rect,top:-1000});
    vi.mocked(inspectGeometry).mockReturnValue({ok:false,code:'offscreen',reason:'outside viewport'});
    attachment.refresh(); document.flush();
    const calls = vi.mocked(inspectGeometry).mock.calls.length;
    for (let iteration = 0; iteration < 3; iteration++) document.flush();
    expect(inspectGeometry).toHaveBeenCalledTimes(calls);
    video.getBoundingClientRect = () => rect;
    vi.mocked(inspectGeometry).mockImplementation(originalInspect);
    document.flush(); document.flush(); frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
  });

  it.each([32, 33])('enforces the %s-node ancestor startup boundary without leaked handles', async count => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, parent, failure, document } = harness({presentationWatchdog:true});
    let outer = parent;
    for (let depth = 2; depth < count; depth++) {
      const ancestor = new FakeElement(); ancestor.insertBefore(outer,null); outer = ancestor;
    }
    await attachment.start();
    if (count === 32) expect(failure).not.toHaveBeenCalled();
    else {
      expect(failure).toHaveBeenCalledWith('unsupported-geometry',expect.stringContaining('bounded limit'));
      expect(acquireGpu).not.toHaveBeenCalled();
    }
    attachment.destroy();
    expect(document.frames.size).toBe(0);
    expect(Object.values(attachment.snapshot().resources).every(value => value === 0)).toBe(true);
  });

  it('reconciles layout changed by canvas insertion before accepting its geometry', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, video, canvas, frame, document, next } = harness({presentationWatchdog:true});
    next.remove();
    const originalRect = video.getBoundingClientRect();
    video.getBoundingClientRect = () => ({...originalRect, top:video.nextSibling === canvas ? 20 : 40});
    const originalInspect = vi.mocked(inspectGeometry).getMockImplementation()!;
    vi.mocked(inspectGeometry).mockImplementation(element => {
      const geometry = originalInspect(element);
      if (!geometry.ok) return geometry;
      const rect = video.getBoundingClientRect();
      return {...geometry,rect,clip:rect,style:{...geometry.style,top:`${rect.top}px`}};
    });
    await attachment.start(); frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    document.flush(); frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
    expect(canvas.style.getPropertyValue('top')).toBe('20px');
  });

  it.each(['seeking', 'paused'] as const)('hides when a submitted frame observes %s before its event', async state => {
    const { attachment, video, canvas, frame } = harness();
    await attachment.start(); frame();
    expect(attachment.snapshot().ready).toBe(true);
    video[state] = true;
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(attachment.snapshot().ready).toBe(false);
  });

  it('hides when a submitted frame observes unavailable decoded pixels before an event', async () => {
    const { attachment, video, canvas, frame } = harness();
    await attachment.start(); frame();
    video.readyState = 1;
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(attachment.snapshot().ready).toBe(false);
  });

  it('hides on the watchdog when media pauses before its event or another submission', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, video, canvas, frame, document } = harness({presentationWatchdog:true});
    await attachment.start(); frame();
    video.paused = true;
    document.flush();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(attachment.snapshot().ready).toBe(false);
  });

  it('watchdog hides unannounced movement before a processed frame can reveal stale placement', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, canvas, video, frame, document } = harness({presentationWatchdog:true});
    await attachment.start();frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
    const rect=video.getBoundingClientRect();
    video.getBoundingClientRect=()=>({...rect,top:57});
    frame();expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    document.flush();
    attachment.destroy();expect(document.frames.size).toBe(0);
    expect(Object.values(attachment.snapshot().resources).every(value=>value===0)).toBe(true);
  });

  it('bounds mutation overload without leaving a watchdog', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const {attachment,mutate,failure,document}=harness({presentationWatchdog:true});
    await attachment.start();
    mutate(Array.from({length:257},()=>({type:'attributes'} as MutationRecord)));
    expect(failure).toHaveBeenCalledWith('unsupported-geometry',expect.stringContaining('bounded limit'));
    expect(document.frames.size).toBe(0);
  });

  it('does not re-show old-size output before a current-source frame and geometry proof', async () => {
    const { attachment, canvas, video, frame, document, pipeline } = harness();
    await attachment.start(); frame();
    video.videoWidth = 960; video.videoHeight = 540;
    video.dispatchEvent(new Event('resize'));
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    document.flush();
    pipeline.onFrame?.({ now: performance.now(), mediaTime: 1, presentedDelta: 1, size: {width:640,height:360},
      presentationTime: 0, expectedDisplayTime: 1, decodeLatencyMs: null });
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    frame(); expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
    expect([canvas.width,canvas.height]).toEqual([1920,1080]);
  });

  it('bounds ancestor observation, invalidates mutations and disconnects on teardown', async () => {
    const { attachment, frame, canvas, parent, mutationObserver, mutate } = harness();
    await attachment.start(); frame();
    expect(mutationObserver.observe.mock.calls.every(call => !(call[1] as MutationObserverInit).subtree)).toBe(true);
    mutate([{type:'attributes',target:parent} as unknown as MutationRecord]);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    attachment.destroy();expect(mutationObserver.disconnect).toHaveBeenCalled();
    expect(attachment.snapshot().resources.mutationObservers).toBe(0);
  });

  it('ignores unchanged ancestor attributes but fails closed for page-owned canvas movement', async () => {
    const {attachment,frame,canvas,parent,mutate,failure}=harness();await attachment.start();frame();
    parent.getAttribute.mockReturnValue('same');
    mutate([{type:'attributes',target:parent,attributeName:'class',oldValue:'same'} as unknown as MutationRecord]);
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
    canvas.remove();
    mutate([{type:'childList',target:parent,addedNodes:[],removedNodes:[canvas]} as unknown as MutationRecord]);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(failure).toHaveBeenCalledWith('unsupported-geometry',expect.stringContaining('moved or removed'));
  });

  it('hides synchronously and cannot reveal before the current geometry reconciliation', async () => {
    const { attachment, canvas, frame, document } = harness();
    await attachment.start(); frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
    attachment.refresh();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    attachment.refresh();
    expect(document.frames.size).toBe(1);
    document.flush();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
  });

  it('keeps diagnostic ownership and geometry live without acquiring a GPU or constructing frame processing', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', true);
    const { attachment, canvas, video, parent, next, observer, document, failure } = harness({ processingDisabled: true });
    await attachment.start();
    expect(acquireGpu).not.toHaveBeenCalled(); expect(VideoPipeline).not.toHaveBeenCalled();
    expect(attachment.driver).toBeNull(); expect(attachment.gpu).toBeNull();
    expect(parent.children).toEqual([video, canvas, next]);
    expect(attachment.snapshot()).toMatchObject({ ready: false, active: false, current: null,
      suspendedReason: 'diagnostic-processing-disabled', resources: {device:0,pipeline:0,canvas:1,resizeObservers:1,frameCallback:0} });
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    const before = attachment.snapshot().infrastructure.geometryCalls;
    attachment.refresh(); document.flush();
    expect(attachment.snapshot().infrastructure.geometryCalls).toBeGreaterThan(before);
    expect(observer.observe).toHaveBeenCalled(); expect(failure).not.toHaveBeenCalled();
    attachment.destroy(); expect(parent.children).toEqual([video,next]);
    expect(Object.values(attachment.snapshot().resources).every(value => value === 0)).toBe(true);
  });

  it('cannot disable processing through diagnostic options in production', async () => {
    vi.stubGlobal('__AETHERVSR_TEST__', false);
    const { attachment } = harness({ processingDisabled: true });
    await attachment.start(); expect(acquireGpu).toHaveBeenCalledTimes(1); expect(VideoPipeline).toHaveBeenCalledTimes(1);
  });

  it.each([false, true])('rejects actual canvas coordinate mismatch after runtime start=%s', async afterStart => {
    const { attachment, canvas, parent, video, next, device, failure, document } = harness();
    const inspect = vi.mocked(inspectGeometry).getMockImplementation()!;
    vi.mocked(inspectGeometry).mockImplementation(source => {
      const geometry = inspect(source);
      return geometry.ok ? { ...geometry, verifyPlacement: true, style: { ...geometry.style, left: '10px', top: '20px' } } : geometry;
    });
    const measure = vi.fn(() => ({ left: afterStart ? 10 : 90, top: 20, width: 640, height: 360 }));
    Object.assign(canvas, { getBoundingClientRect: measure });
    await attachment.start();
    if (afterStart) {
      measure.mockReturnValue({ left: 90, top: 20, width: 640, height: 360 });
      attachment.refresh(); document.flush();
      expect(device.destroy).toHaveBeenCalledTimes(1);
    } else expect(acquireGpu).not.toHaveBeenCalled();
    expect(failure).toHaveBeenCalledWith('unsupported-geometry', expect.stringContaining('placement'));
    expect(parent.children).toEqual([video, next]);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
  });

  it('validates newly admitted placement before output without adding per-frame layout reads', async () => {
    const { attachment, canvas, frame } = harness();
    const inspect = vi.mocked(inspectGeometry).getMockImplementation()!;
    vi.mocked(inspectGeometry).mockImplementation(source => {
      const geometry = inspect(source);
      return geometry.ok ? { ...geometry, verifyPlacement: true, style: { ...geometry.style, left: '10px', top: '20px' } } : geometry;
    });
    const measure = vi.fn(() => ({ left: 10, top: 20, width: 640, height: 360 }));
    Object.assign(canvas, { getBoundingClientRect: measure });
    await attachment.start(); const checks = measure.mock.calls.length;
    expect(checks).toBeGreaterThan(0); expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    frame(); frame(); expect(measure).toHaveBeenCalledTimes(checks);
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
  });

  it('hides immediately on pause with pending geometry and preserves controller policy until fresh output', async () => {
    for (const event of ['pause'] as const) {
      const { attachment, video, document, canvas, pipeline, frame, failure } = harness({ mode: 'neural' });
      await attachment.start();
      for (let index = 0; index < 31; index++) { vi.advanceTimersByTime(1000 / 60); frame(); }
      attachment.driver!.force(true);
      const driver = attachment.driver!;
      const before = driver.snapshot();
      attachment.refresh();
      if (event === 'pause') video.paused = true;
      else video.seeking = true;
      video.dispatchEvent(new Event(event));
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      expect(pipeline.running).toBe(false);
      expect(attachment.snapshot()).toMatchObject({ ready: false, suspendedReason: 'video-paused', mode: 'neural' });
      document.flush();
      vi.advanceTimersByTime(60000);
      expect(driver.snapshot().controller).toMatchObject({ activeMs: before.controller.activeMs,
        backoffMs: before.controller.backoffMs, nextProbeAtMs: before.controller.nextProbeAtMs! + 60000 });
      expect(driver.snapshot().session.activeMs).toBe(before.session.activeMs);
      video.paused = false;
      video.seeking = false;
      video.dispatchEvent(new Event(event === 'pause' ? 'playing' : 'seeked'));
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      document.flush();
      expect(attachment.driver).toBe(driver);
      expect(attachment.pipeline).toBe(pipeline);
      expect(pipeline.running).toBe(true);
      expect(attachment.snapshot().ready).toBe(false);
      frame();
      expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
      expect(failure).not.toHaveBeenCalled();
      attachment.destroy();
    }
  });

  it('keeps playing seek time active while hiding pixels until a fresh completed seek frame', async () => {
    const { attachment, video, document, canvas, pipeline, frame } = harness();
    await attachment.start();
    frame();
    const before = attachment.driver!.snapshot();
    video.seeking = true;
    video.readyState = 1;
    video.dispatchEvent(new Event('seeking'));
    document.flush();
    expect(pipeline.running).toBe(true);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    vi.advanceTimersByTime(100);
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(attachment.driver!.snapshot().session.activeMs).toBe(before.session.activeMs + 100);
    video.seeking = false;
    video.readyState = 4;
    video.dispatchEvent(new Event('loadeddata'));
    video.dispatchEvent(new Event('seeked'));
    document.flush();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
  });

  it('allows paused attachment and paused seeking but keeps the original visible until playing produces a new frame', async () => {
    const { attachment, video, document, canvas, pipeline, frame, failure } = harness();
    video.paused = true;
    await attachment.start();
    expect(acquireGpu).toHaveBeenCalledTimes(1);
    expect(attachment.driver).toBeInstanceOf(RuntimeDriver);
    expect(pipeline.start).not.toHaveBeenCalled();
    expect(attachment.snapshot()).toMatchObject({ active: false, ready: false, suspendedReason: 'video-paused' });
    video.seeking = true;
    video.dispatchEvent(new Event('seeking'));
    document.flush();
    video.seeking = false;
    video.dispatchEvent(new Event('seeked'));
    document.flush();
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(pipeline.start).not.toHaveBeenCalled();
    video.paused = false;
    video.dispatchEvent(new Event('playing'));
    document.flush();
    expect(pipeline.running).toBe(true);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
    video.paused = true;
    video.dispatchEvent(new Event('pause'));
    video.seeking = true;
    video.dispatchEvent(new Event('seeking'));
    video.seeking = false;
    video.dispatchEvent(new Event('seeked'));
    document.flush();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(attachment.snapshot()).toMatchObject({ active: false, ready: false, suspendedReason: 'video-paused' });
    expect(failure).not.toHaveBeenCalled();
  });

  it('terminates on media errors with ready data, precise codes, no retries, and no stale or first-frame reveal', async () => {
    const cases = [
      { code: 1, name: 'MEDIA_ERR_ABORTED', trigger: 'event', output: true },
      { code: 2, name: 'MEDIA_ERR_NETWORK', trigger: 'refresh', output: true },
      { code: 3, name: 'MEDIA_ERR_DECODE', trigger: 'frame', output: false },
      { code: 4, name: 'MEDIA_ERR_SRC_NOT_SUPPORTED', trigger: 'frame', output: true },
      { code: 99, name: 'Media error (code 99)', trigger: 'geometry', output: true },
    ];
    for (const { code, name, trigger, output } of cases) {
      const { attachment, video, document, canvas, parent, next, pipeline, device, frame, failure } = harness();
      await attachment.start();
      if (output) frame();
      attachment.refresh();
      const starts = pipeline.start.mock.calls.length;
      const switches = pipeline.setUpscaler.mock.calls.length;
      video.readyState = 2;
      video.error = { code, message: 'decoder detail' };
      if (trigger === 'event') video.dispatchEvent(new Event('error'));
      else if (trigger === 'frame') frame();
      else if (trigger === 'refresh') attachment.refresh();
      else document.flush();
      expect(failure).toHaveBeenCalledExactlyOnceWith('unsupported-media', `${name}: decoder detail`);
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      expect(parent.children).toEqual([video, next]);
      expect(pipeline.running).toBe(false);
      expect(pipeline.destroy).toHaveBeenCalledTimes(1);
      expect(device.destroy).toHaveBeenCalledTimes(1);
      const stopped = attachment.snapshot();
      video.dispatchEvent(new Event('error'));
      video.dispatchEvent(new Event('playing'));
      attachment.refresh();
      await attachment.start();
      document.flush();
      vi.advanceTimersByTime(60000);
      expect(pipeline.start).toHaveBeenCalledTimes(starts);
      expect(pipeline.setUpscaler).toHaveBeenCalledTimes(switches);
      expect(attachment.snapshot()).toEqual(stopped);
      expect(failure).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
    }
  });

  it('rejects existing or acquisition-time media errors before any GPU pipeline runs, with mediaKeys taking precedence', async () => {
    for (const phase of ['initial', 'acquiring', 'protected'] as const) {
      const { attachment, video, gpu, device, pipeline, failure } = harness();
      video.readyState = 2;
      const acquisitions = vi.mocked(acquireGpu).mock.calls.length;
      const constructions = vi.mocked(VideoPipeline).mock.calls.length;
      let resolve!: (gpu: GpuContext) => void;
      if (phase === 'acquiring') vi.mocked(acquireGpu).mockReturnValueOnce(new Promise(done => { resolve = done; }));
      else video.error = { code: 3, message: '' };
      if (phase === 'protected') video.mediaKeys = {};
      const pending = attachment.start();
      if (phase === 'acquiring') {
        video.error = { code: 3, message: '' };
        resolve(gpu);
      }
      await pending;
      expect(failure).toHaveBeenCalledExactlyOnceWith(phase === 'protected' ? 'protected-media' : 'unsupported-media',
        phase === 'protected' ? 'Encrypted media cannot be enhanced.' : 'MEDIA_ERR_DECODE: The video could not be decoded.');
      expect(acquireGpu).toHaveBeenCalledTimes(acquisitions + (phase === 'acquiring' ? 1 : 0));
      expect(VideoPipeline).toHaveBeenCalledTimes(constructions);
      expect(pipeline.start).not.toHaveBeenCalled();
      expect(device.destroy).toHaveBeenCalledTimes(phase === 'acquiring' ? 1 : 0);
      expect(attachment.snapshot().resources.device).toBe(0);
    }
  });

  it('rechecks native captions on text-track events without DOM changes and removes every track listener on failure', async () => {
    for (const event of ['change', 'addtrack', 'removetrack']) {
      const { attachment, video, document, canvas, parent, next, pipeline, frame, failure, listeners } = harness();
      const geometry = vi.mocked(inspectGeometry).getMockImplementation()!;
      vi.mocked(inspectGeometry).mockImplementation(source => Array.from(source.textTracks).some(track => track.mode === 'showing')
        ? { ok: false, code: 'unsupported-controls', reason: 'native captions' } : geometry(source));
      await attachment.start();
      frame();
      const calls = vi.mocked(inspectGeometry).mock.calls.length;
      video.textTracks[0]!.mode = 'showing';
      video.textTracks.dispatchEvent(new Event(event));
      expect(document.frames.size).toBe(1);
      document.flush();
      expect(inspectGeometry).toHaveBeenCalledTimes(calls + 1);
      expect(failure).toHaveBeenCalledExactlyOnceWith('unsupported-controls', 'native captions');
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      expect(parent.children).toEqual([video, next]);
      expect(pipeline.running).toBe(false);
      const tracks = listeners.find(listener => listener.target === video.textTracks)!;
      expect(tracks.add.mock.calls.map(call => call[0])).toEqual(['change', 'addtrack', 'removetrack']);
      expect(tracks.remove.mock.calls).toEqual(tracks.add.mock.calls);
      video.textTracks.dispatchEvent(new Event(event));
      expect(document.frames.size).toBe(0);
      expect(failure).toHaveBeenCalledTimes(1);
    }
  });

  it('owns one sibling canvas, reveals only after output, and never edits host styles', async () => {
    const { attachment, canvas, video, parent, next, pipeline, frame, document } = harness();
    await Promise.all([attachment.start(), attachment.start()]);
    expect(acquireGpu).toHaveBeenCalledTimes(1);
    expect(attachment.driver).toBeInstanceOf(RuntimeDriver);
    expect(attachment.pipeline).toBe(pipeline);
    expect(parent.children).toEqual([video, attachment.canvas, next]);
    expect(canvas.setAttribute).toHaveBeenCalledWith('aria-hidden', 'true');
    expect(attachment.canvas.dataset.aethervsrM10).toBeTruthy();
    expect(attachment.canvas.style.getPropertyValue('visibility')).toBe('hidden');
    const calls = vi.mocked(inspectGeometry).mock.calls.length;
    vi.advanceTimersByTime(20);
    frame();
    expect(attachment.canvas.style.getPropertyValue('visibility')).toBe('visible');
    const styles = canvas.style.setProperty.mock.calls.length;
    frame();
    expect(canvas.style.setProperty.mock.calls).toHaveLength(styles);
    expect(inspectGeometry).toHaveBeenCalledTimes(calls);
    attachment.refresh();
    document.flush();
    expect(video.style.setProperty).not.toHaveBeenCalled();
    expect(parent.style.setProperty).not.toHaveBeenCalled();
    expect(canvas.style.setProperty.mock.calls.every(call => call[2] === 'important')).toBe(true);
  });

  it('destroys once, cancels pending geometry, and retains a stopped session snapshot', async () => {
    const { attachment, pipeline, frame, document, observer, device, unwatch, failure } = harness();
    await attachment.start();
    vi.advanceTimersByTime(20);
    frame();
    attachment.refresh();
    attachment.destroy();
    const stopped = attachment.snapshot();
    expect(stopped.session.framesRendered).toBe(1);
    expect(stopped.resources).toEqual({ device: 0, pipeline: 0, canvas: 0, resizeObservers: 0,
      listeners: 0, geometryFrame: 0, frameCallback: 0, mutationObservers: 0, presentationFrame: 0 });
    attachment.destroy();
    attachment.refresh();
    attachment.setMode('neural');
    await attachment.start();
    vi.advanceTimersByTime(5000);
    document.flush();
    expect(attachment.snapshot()).toEqual(stopped);
    expect(attachment.driver).toBeNull();
    expect(attachment.pipeline).toBeNull();
    expect(attachment.gpu).toBeNull();
    expect(device.destroy).toHaveBeenCalledTimes(1);
    expect(pipeline.destroy).toHaveBeenCalledTimes(1);
    expect(observer.disconnect).toHaveBeenCalledTimes(1);
    expect(unwatch).toHaveBeenCalledTimes(1);
    expect(failure).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('destroys a GPU delivered after disposal without ever attaching or constructing a pipeline', async () => {
    const { attachment, gpu, device, parent, video, next } = harness();
    let resolve!: (gpu: GpuContext) => void;
    vi.mocked(acquireGpu).mockReturnValue(new Promise(done => { resolve = done; }));
    const pending = attachment.start();
    attachment.destroy();
    resolve(gpu);
    await pending;
    expect(device.destroy).toHaveBeenCalledTimes(1);
    expect(VideoPipeline).not.toHaveBeenCalled();
    expect(watchDeviceFailures).not.toHaveBeenCalled();
    expect(parent.children).toEqual([video, next]);
    expect(attachment.snapshot().resources.device).toBe(0);
  });

  it('coalesces external events into one geometry check and tracks the current parent', async () => {
    const { attachment, document, video, observer, resize, parent, canvas } = harness();
    await attachment.start();
    expect(observer.observe.mock.calls).toEqual([[video], [parent]]);
    const geometryCalls = attachment.snapshot().infrastructure.geometryCalls;
    attachment.refresh();
    resize();
    document.defaultView.dispatchEvent(new Event('scroll'));
    document.defaultView.dispatchEvent(new Event('resize'));
    video.dispatchEvent(new Event('resize'));
    expect(document.frames.size).toBe(1);
    expect(inspectGeometry).toHaveBeenCalledTimes(geometryCalls);
    document.flush();
    expect(inspectGeometry).toHaveBeenCalledTimes(geometryCalls + 1);
    expect(document.frames.size).toBe(0);
    const newParent = new FakeElement();
    newParent.insertBefore(video, null);
    attachment.refresh();
    document.flush();
    expect(observer.unobserve).toHaveBeenCalledWith(parent);
    expect(observer.observe).toHaveBeenLastCalledWith(newParent);
    expect(newParent.children).toEqual([video, canvas]);
    expect(parent.children).not.toContain(canvas);
  });

  it('observes the host when the sibling placement is in a shadow root', async () => {
    const { attachment, video, observer, parent, document } = harness();
    const root = Object.assign(new FakeElement(), { nodeType: 11, host: parent });
    root.insertBefore(video, null);
    video.parentElement = null;
    await attachment.start();
    expect(observer.observe.mock.calls).toEqual([[video], [parent]]);
    expect(attachment.canvas.parentNode).toBe(root);
    attachment.refresh();
    document.flush();
    expect(attachment.snapshot().active).toBe(true);
  });

  it('freezes the same controller and its backoff while geometry is offscreen', async () => {
    const { attachment, document, pipeline, frame, canvas, failure } = harness({ mode: 'neural' });
    await attachment.start();
    for (let index = 0; index < 31; index++) { vi.advanceTimersByTime(1000 / 60); frame(); }
    attachment.driver!.force(true);
    const driver = attachment.driver!;
    vi.mocked(inspectGeometry).mockReturnValueOnce({ ok: false, code: 'offscreen', reason: 'outside clip' });
    attachment.refresh();
    document.flush();
    const stopped = driver.snapshot();
    expect(pipeline.running).toBe(false);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(attachment.snapshot()).toMatchObject({ active: false, ready: false, suspendedReason: 'offscreen', mode: 'neural' });
    vi.advanceTimersByTime(60000);
    expect(driver.snapshot().controller).toMatchObject({
      activeMs: stopped.controller.activeMs, backoffMs: stopped.controller.backoffMs,
      nextProbeAtMs: stopped.controller.nextProbeAtMs! + 60000,
    });
    expect(driver.snapshot().session.activeMs).toBe(stopped.session.activeMs);
    attachment.refresh();
    document.flush();
    expect(attachment.driver).toBe(driver);
    expect(attachment.pipeline).toBe(pipeline);
    expect(pipeline.running).toBe(true);
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    frame();
    expect(canvas.style.getPropertyValue('visibility')).toBe('visible');
    expect(failure).not.toHaveBeenCalled();
  });

  it.each(['picture-in-picture', 'video-fullscreen', 'outside-fullscreen', 'document-hidden'] as const)(
    'suspends immediately for %s and resumes without changing mode', async reason => {
      const { attachment, video, document, pipeline, frame, failure } = harness({ mode: 'baseline' });
      await attachment.start();
      frame();
      const driver = attachment.driver;
      let event: string;
      let target: EventTarget = document;
      if (reason === 'picture-in-picture') {
        document.pictureInPictureElement = video;
        event = 'enterpictureinpicture';
        target = video;
      } else if (reason === 'document-hidden') {
        document.visibilityState = 'hidden';
        event = 'visibilitychange';
      } else {
        document.fullscreenElement = reason === 'video-fullscreen' ? video : new FakeElement();
        event = 'fullscreenchange';
      }
      target.dispatchEvent(new Event(event));
      expect(pipeline.running).toBe(false);
      expect(attachment.snapshot()).toMatchObject({ ready: false, suspendedReason: reason, mode: 'baseline' });
      document.flush();
      attachment.setMode('neural');
      expect(attachment.snapshot().mode).toBe('neural');
      document.pictureInPictureElement = null;
      document.fullscreenElement = null;
      document.visibilityState = 'visible';
      target.dispatchEvent(new Event(reason === 'picture-in-picture' ? 'leavepictureinpicture' : event));
      document.flush();
      expect(attachment.driver).toBe(driver);
      expect(attachment.pipeline).toBe(pipeline);
      expect(attachment.snapshot()).toMatchObject({ active: true, ready: false, mode: 'neural' });
      frame();
      expect(attachment.snapshot().ready).toBe(true);
      expect(failure).not.toHaveBeenCalled();
    });

  it('allows container fullscreen and unrelated PiP without treating them as direct-video fullscreen', async () => {
    const { attachment, document, parent, pipeline } = harness();
    document.fullscreenElement = parent;
    document.pictureInPictureElement = new FakeVideo();
    await attachment.start();
    expect(pipeline.running).toBe(true);
    expect(attachment.canvas.parentNode).toBe(parent);
  });

  it('keeps the source pipeline and invalidation handlers through a source reload and resolution change', async () => {
    const { attachment, document, video, pipeline, frame, failure } = harness();
    await attachment.start();
    frame();
    const driver = attachment.driver;
    const onConfigure = pipeline.onConfiguration;
    const generation = pipeline.timingGeneration;
    video.readyState = 0;
    video.dispatchEvent(new Event('loadstart'));
    expect(pipeline.running).toBe(true);
    expect(pipeline.timingGeneration).toBeGreaterThan(generation);
    expect(attachment.snapshot().ready).toBe(false);
    document.flush();
    expect(attachment.snapshot()).toMatchObject({ active: true, ready: false });
    video.readyState = 4;
    video.videoWidth = 1280;
    video.videoHeight = 720;
    video.dispatchEvent(new Event('loadeddata'));
    document.flush();
    pipeline.onConfiguration?.({ generation: pipeline.timingGeneration, source: { width: 1280, height: 720 },
      target: { width: 2560, height: 1440 }, sourceChanged: true, configureMs: 1 });
    frame();
    expect(attachment.driver).toBe(driver);
    expect(pipeline.onConfiguration).toBe(onConfigure);
    expect(VideoPipeline).toHaveBeenCalledTimes(1);
    expect(attachment.snapshot()).toMatchObject({ active: true, ready: true, source: { w: 1280, h: 720 },
      session: { framesRendered: 2 } });
    expect(failure).not.toHaveBeenCalled();
  });

  it.each(['unsupported-geometry', 'unsupported-controls'] as const)(
    'rejects initial %s before GPU acquisition and never retries', async code => {
      const { attachment, parent, video, next, failure } = harness();
      vi.mocked(inspectGeometry).mockReturnValue({ ok: false, code, reason: 'not safe' });
      await attachment.start();
      attachment.refresh();
      await attachment.start();
      expect(acquireGpu).not.toHaveBeenCalled();
      expect(VideoPipeline).not.toHaveBeenCalled();
      expect(parent.children).toEqual([video, next]);
      expect(failure).toHaveBeenCalledExactlyOnceWith(code, 'not safe');
      expect(attachment.snapshot().resources.canvas).toBe(0);
    });

  it('terminates when controls become unsupported after successful output', async () => {
    const { attachment, document, frame, failure, canvas } = harness();
    await attachment.start();
    frame();
    vi.mocked(inspectGeometry).mockReturnValue({ ok: false, code: 'unsupported-controls', reason: 'native controls' });
    attachment.refresh();
    document.flush();
    expect(failure).toHaveBeenCalledExactlyOnceWith('unsupported-controls', 'native controls');
    expect(canvas.parentNode).toBeNull();
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
  });

  it('rechecks geometry after acquisition and never starts a frame for newly unsupported placement', async () => {
    const { attachment, gpu, pipeline, failure } = harness();
    let resolve!: (gpu: GpuContext) => void;
    vi.mocked(acquireGpu).mockReturnValue(new Promise(done => { resolve = done; }));
    const pending = attachment.start();
    vi.mocked(inspectGeometry).mockReturnValue({ ok: false, code: 'unsupported-geometry', reason: 'changed while pending' });
    resolve(gpu);
    await pending;
    expect(pipeline.start).not.toHaveBeenCalled();
    expect(failure).toHaveBeenCalledExactlyOnceWith('unsupported-geometry', 'changed while pending');
  });

  it('maps an actual pipeline SecurityError to cors-blocked and never attempts copy fallback', async () => {
    const { attachment, pipeline, frame, failure, device } = harness();
    await attachment.start();
    frame();
    pipeline.error = new DOMException('origin is not clean', 'SecurityError');
    pipeline.running = false;
    vi.advanceTimersByTime(100);
    expect(failure).toHaveBeenCalledExactlyOnceWith('cors-blocked', 'origin is not clean');
    expect(device.destroy).toHaveBeenCalledTimes(1);
    expect(acquireGpu).toHaveBeenCalledTimes(1);
    expect(VideoPipeline).toHaveBeenCalledTimes(1);
    expect(attachment.snapshot().suspendedReason).toBe('cors-blocked');
    vi.advanceTimersByTime(5000);
    expect(failure).toHaveBeenCalledTimes(1);
  });

  it.each(['keys-before-start', 'encrypted-event', 'keys-on-error'] as const)(
    'rejects protected media: %s', async cause => {
      const { attachment, video, pipeline, failure } = harness();
      if (cause === 'keys-before-start') video.mediaKeys = {};
      await attachment.start();
      if (cause === 'encrypted-event') video.dispatchEvent(new Event('encrypted'));
      if (cause === 'keys-on-error') {
        video.mediaKeys = {};
        pipeline.error = new DOMException('protected pixels', 'SecurityError');
        vi.advanceTimersByTime(100);
      }
      expect(failure).toHaveBeenCalledTimes(1);
      expect(failure.mock.calls[0]![0]).toBe('protected-media');
      if (cause === 'keys-before-start') expect(acquireGpu).not.toHaveBeenCalled();
      expect(attachment.snapshot().resources.device).toBe(0);
    });

  it('hides immediately on late GPU validation failure and ignores retained callbacks after destroy', async () => {
    const { attachment, pipeline, frame, failure, document, resize, canvas } = harness();
    await attachment.start();
    const onFrame = pipeline.onFrame!;
    const notify = vi.mocked(watchDeviceFailures).mock.lastCall![1];
    frame();
    expect(attachment.snapshot().ready).toBe(true);
    notify('WebGPU validation error');
    expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
    expect(canvas.parentNode).toBeNull();
    const stopped = attachment.snapshot();
    const styleCalls = canvas.style.setProperty.mock.calls.length;
    onFrame({ now: 0, mediaTime: 1, presentedDelta: 1, size: { width: 640, height: 360 },
      presentationTime: 0, expectedDisplayTime: 0, decodeLatencyMs: null });
    notify('device lost again');
    resize();
    document.flush();
    expect(canvas.style.setProperty.mock.calls).toHaveLength(styleCalls);
    expect(attachment.snapshot()).toEqual(stopped);
    expect(failure).toHaveBeenCalledExactlyOnceWith('device-lost', 'WebGPU validation error');
  });

  it('preserves the pipeline exception and cross-realm message before first output or a later device notification', async () => {
    for (const trigger of ['frame', 'device']) {
      const { attachment, pipeline, failure, frame, canvas, device } = harness();
      await attachment.start();
      pipeline.error = { name: 'SecurityError', message: 'cross realm exception' };
      if (trigger === 'frame') frame();
      else vi.mocked(watchDeviceFailures).mock.lastCall![1]('uncaptured error');
      expect(failure).toHaveBeenCalledExactlyOnceWith('cors-blocked', 'cross realm exception');
      expect(canvas.style.getPropertyValue('visibility')).toBe('hidden');
      expect(canvas.parentNode).toBeNull();
      expect(device.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('reports acquisition rejection once, and ignores rejection after disposal', async () => {
    const first = harness();
    vi.mocked(acquireGpu).mockRejectedValue(new Error('no adapter'));
    await first.attachment.start();
    expect(first.failure).toHaveBeenCalledExactlyOnceWith('webgpu-unavailable', 'no adapter');
    const second = harness();
    let reject!: (error: Error) => void;
    vi.mocked(acquireGpu).mockReturnValue(new Promise((_resolve, fail) => { reject = fail; }));
    const pending = second.attachment.start();
    second.attachment.destroy();
    reject(new Error('late failure'));
    await pending;
    expect(second.failure).not.toHaveBeenCalled();
  });

  it('releases event registrations before a manager disables from inside onFailure', async () => {
    const notify = vi.fn(() => {
      expect(attachment.snapshot().resources).toEqual({ device: 0, pipeline: 0, canvas: 0,
        resizeObservers: 0, listeners: 0, geometryFrame: 0, frameCallback: 0, mutationObservers: 0, presentationFrame: 0 });
      attachment.destroy();
      attachment.setMode('neural');
      attachment.refresh();
    });
    const context = harness({ onFailure: notify });
    const attachment = context.attachment;
    await attachment.start();
    const registrations = context.listeners.flatMap(({ target, add }) => add.mock.calls.map(call => ({ target, call })));
    vi.mocked(watchDeviceFailures).mock.lastCall![1]('device lost');
    for (const { add, remove } of context.listeners) {
      expect(remove.mock.calls).toHaveLength(add.mock.calls.length);
      for (const registration of add.mock.calls) expect(remove.mock.calls).toContainEqual(registration);
    }
    for (const { target, call: [type, listener] } of registrations) {
      (listener as EventListener)(new Event(type));
      target.dispatchEvent(new Event(type));
    }
    expect(notify).toHaveBeenCalledTimes(1);
    expect(context.document.frames.size).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
    expect(context.observer.disconnect).toHaveBeenCalledTimes(1);
    expect(context.device.destroy).toHaveBeenCalledTimes(1);
  });

  it('continues teardown after a resource destructor throws and records the cleanup error', async () => {
    const { attachment, pipeline, device, observer, unwatch, failure } = harness();
    await attachment.start();
    pipeline.destroy.mockImplementation(() => { throw new Error('cleanup failed'); });
    vi.mocked(watchDeviceFailures).mock.lastCall![1]('device lost');
    expect(device.destroy).toHaveBeenCalledTimes(1);
    expect(unwatch).toHaveBeenCalledTimes(1);
    expect(observer.disconnect).toHaveBeenCalledTimes(1);
    expect(failure).toHaveBeenCalledTimes(1);
    expect(attachment.snapshot().infrastructure.cleanupErrors).toBe(1);
  });

  it('uses shared baseline behavior without timestamps and exposes the shared reason without failure', async () => {
    const { attachment, gpu, device, frame, failure } = harness();
    vi.mocked(acquireGpu).mockResolvedValue({ ...gpu, capabilities: { ...gpu.capabilities, timestampQuery: false } });
    device.features.delete('timestamp-query');
    await attachment.start();
    for (let index = 0; index < 120; index++) { vi.advanceTimersByTime(1000 / 60); frame(); }
    expect(attachment.snapshot()).toMatchObject({ current: 'baseline', controllerState: 'unavailable',
      controllerReason: attachment.driver!.snapshot().controller.reason, features: { timestampQuery: false },
      gpuMs: { p50: null, p95: null, count: 0 } });
    expect(NeuralUpscaler).not.toHaveBeenCalled();
    expect(failure).not.toHaveBeenCalled();
  });

  it('passes model, import and feature options unchanged to the shared components', async () => {
    const model = {} as PackedModel;
    const { attachment, frame, gpu, video } = harness({ model, forceCopy: true, withheldFeatures: ['shader-f16'] });
    await attachment.start();
    expect(acquireGpu).toHaveBeenCalledExactlyOnceWith({ optionalFeatures: ['shader-f16'], withheldFeatures: ['shader-f16'] });
    expect(VideoPipeline).toHaveBeenCalledWith(gpu, video, attachment.canvas, expect.objectContaining({ id: 'catmull-rom' }),
      { forceCopyImport: true });
    for (let index = 0; index < 31; index++) { vi.advanceTimersByTime(1000 / 60); frame(); }
    expect(NeuralUpscaler).toHaveBeenCalledExactlyOnceWith(model, { passDiagnostics: true });
  });

  it('produces bounded protocol-compatible snapshots without exposing controller history or live objects', async () => {
    const { attachment, pipeline, frame, video } = harness();
    await attachment.start();
    expect(attachment.snapshot().infrastructure.geometryMaxMs).toBe(0);
    for (let index = 0; index < 120; index++) { vi.advanceTimersByTime(1000 / 60); frame(); }
    pipeline.onGpuSample?.({ ms: 6, neural: true, sequence: 1, generation: pipeline.timingGeneration,
      submittedAt: performance.now(), resolvedAt: performance.now(), upscalerId: 'neural', source: { width: 640, height: 360 } });
    const snapshot = attachment.snapshot();
    expect(snapshot.gpuMs).toEqual({ p50: 6, p95: 6, count: 1 });
    expect(snapshot.session.callbackP95).toBe(1);
    const status = { ...inactiveStatus(), details: { attachment: snapshot } };
    expect(parseExtensionStatus(status)).not.toBeNull();
    snapshot.source.w = 1;
    expect(attachment.snapshot().source.w).toBe(video.videoWidth);
    attachment.destroy();
    const stopped = attachment.snapshot();
    stopped.session.framesRendered = 0;
    video.videoWidth = 1;
    expect(attachment.snapshot().session.framesRendered).toBe(120);
    expect(attachment.snapshot().source.w).toBe(640);
    expect(parseExtensionStatus({ ...status, details: { attachment: attachment.snapshot() } })).not.toBeNull();
  });
});