import { afterEach, describe, expect, it, vi } from 'vitest';
import { assessContract, BOX_FIELDS, COMMON_FIELDS, createContractReader, EFFECT_FIELDS, RADIUS_FIELDS, RecoveryGate,
  supportedPosition, type PresentationInput, type SemanticStyle } from '../tools/m109-contract.js';
import type { SuccessfulFrameSubmission } from '../tools/m109-submission.js';

function input(): PresentationInput {
  const base: SemanticStyle = Object.fromEntries(COMMON_FIELDS.map(field => [field, '']));
  Object.assign(base, { display: 'block', visibility: 'visible', position: 'relative', opacity: '1',
    'overflow-x': 'visible', 'overflow-y': 'visible', 'transition-duration': '0s' });
  const video: SemanticStyle = { ...base, 'object-fit': 'contain', 'object-position': '50% 50%', width: '640px',
    height: '360px', 'z-index': '0', 'background-color': 'rgba(0, 0, 0, 0)' };
  for (const field of [...BOX_FIELDS, ...RADIUS_FIELDS]) video[field] = '0px';
  return { connected: true, ready: true, playing: true, seeking: false, protected: false, mediaError: false,
    documentVisible: true, nativeControls: false, showingTracks: false, pip: false, directFullscreen: false,
    outsideFullscreen: false, source: { width: 1280, height: 720, url: 'local-pattern' },
    rect: { left: 40, top: 240, width: 640, height: 360 }, viewport: { width: 1200, height: 760 },
    fullscreen: 0, parent: 1, rootSupported: true, nonvideoTopLayer: false, output: null, video,
    chain: [{ id: 1, parent: 2, style: { ...base, isolation: 'isolate' }, clip: null, slot: false }],
    branches: [{ id: 3, display: 'block', position: 'absolute', zIndex: '3' }], nonemptyText: false, overflow: false };
}

const submission = (sequence: number, geometryGeneration = 1): SuccessfulFrameSubmission => ({
  kind: 'successfulFrameSubmission', boundary: 'queue.submit returned; not GPU completion or scanout',
  owner: 'video-1', sourceGeneration: 1, geometryGeneration, frameGeneration: 2,
  backingWidth: 2560, backingHeight: 1440, sourceWidth: 1280, sourceHeight: 720,
  authorized: true, validForRecovery: true, sequence, mediaTime: sequence / 60, observedAt: sequence * 16,
});

describe('M10.9 explicit simple-chain admission', () => {
  it.each(['0px', 'content-box', 'content-box 0px', 'padding-box', 'border-box'])('S1-R1 accepts native round/zero serialization %s only on undecorated boxes', margin => {
    const state = input(); state.video['corner-shape'] = 'superellipse(1)';
    state.video['overflow-x'] = 'clip'; state.video['overflow-clip-margin'] = margin;
    expect(assessContract(state).outcome).toBe('SUPPORTED');
    state.video['overflow-clip-margin'] = 'content-box 1px';
    expect(assessContract(state).reason).toBe('unsupported-clipping');
    state.video['overflow-clip-margin'] = margin; state.video['corner-shape'] = 'superellipse(0)';
    expect(assessContract(state).reason).toBe('unsupported-corner');
    state.video['corner-shape'] = 'superellipse(1)'; state.video['padding-top'] = '1px';
    expect(assessContract(state).reason).toBe('unsupported-video-box');
  });
  it('admits contain/cover, current movement, size/source change, reparent and container fullscreen', () => {
    for (const fit of ['contain', 'cover']) {
      const state = input(); state.video['object-fit'] = fit;
      expect(assessContract(state).outcome).toBe('SUPPORTED');
      const original = assessContract(state).fingerprint;
      state.rect.top -= 228; state.rect.width = 600; state.video['width'] = '600px';
      state.source.width = 2560; state.source.height = 1440; state.parent = 4; state.fullscreen = 4;
      expect(assessContract(state).outcome).toBe('SUPPORTED');
      expect(assessContract(state).fingerprint).not.toBe(original);
    }
  });

  it.each(EFFECT_FIELDS)('detects CSSOM %s without a rectangle change on video and ancestor', field => {
    for (const target of ['video', 'ancestor']) {
      const state = input(), before = assessContract(state), rect = { ...state.rect };
      const style = target === 'video' ? state.video : state.chain[0]!.style;
      style[field] = 'non-default';
      expect(assessContract(state).fingerprint).not.toBe(before.fingerprint);
      expect(assessContract(state)).toMatchObject({ outcome: 'UNSUPPORTED', reason: 'unsupported-effect' });
      expect(state.rect).toEqual(rect);
    }
  });

  it.each([
    ['opacity', '0.5'], ['mix-blend-mode', 'multiply'], ['clip', 'rect(0px, 1px, 1px, 0px)'], ['zoom', '2'],
    ['visibility', 'hidden'], ['content-visibility', 'hidden'], ['contain', 'paint'], ['will-change', 'transform'],
    ['container-type', 'inline-size'], ['animation-name', 'move'], ['transition-duration', '1s'],
    ['corner-shape', 'bevel'], ['background-image', 'linear-gradient(red, blue)'], ['background-color', 'rgb(1, 2, 3)'],
    ['box-shadow', 'red 0px 0px 20px'], ['outline-style', 'solid'], ['content', 'url(local)'],
    ['object-view-box', 'inset(10%)'], ['appearance', 'auto'], ['display', 'contents'], ['z-index', 'auto'],
  ])('changes the fingerprint and rejects %s=%s', (field, value) => {
    const state = input(), before = assessContract(state).fingerprint;
    state.video[field] = value;
    expect(assessContract(state).fingerprint).not.toBe(before);
    expect(assessContract(state).outcome).toBe('UNSUPPORTED');
  });

  it.each([...BOX_FIELDS, ...RADIUS_FIELDS])('tracks each box/corner field %s', field => {
    const state = input(), before = assessContract(state).fingerprint;
    state.video[field] = '18px';
    expect(assessContract(state).fingerprint).not.toBe(before);
    expect(assessContract(state).outcome).toBe('UNSUPPORTED');
  });

  it('supports only plain clipping and rejects clip margin, rounded ancestors, parent paint loss and control ambiguity', () => {
    const state = input(), parent = state.chain[0]!.style;
    for (const field of [...BOX_FIELDS, ...RADIUS_FIELDS]) parent[field] = '0px';
    parent['overflow-x'] = parent['overflow-y'] = 'clip';
    state.chain[0]!.clip = { ...state.rect };
    expect(assessContract(state).outcome).toBe('SUPPORTED');
    parent['overflow-clip-margin'] = '10px'; expect(assessContract(state).reason).toBe('unsupported-clipping');
    parent['overflow-clip-margin'] = '0px'; parent['border-top-left-radius'] = '12px';
    expect(assessContract(state).reason).toBe('unsupported-clipping');
    for (const display of ['contents', 'flex', 'grid']) {
      const candidate = input(); candidate.chain[0]!.style['display'] = display;
      expect(assessContract(candidate).reason).toBe('unsupported-control-stack');
    }
    const candidate = input(); candidate.branches[0]!.zIndex = '0';
    expect(assessContract(candidate).reason).toBe('unsupported-control-stack');
  });

  it.each(['', 'auto', 'NaNpx', 'Infinitypx', '640px junk', '-1px'])('rejects unproved content dimension %s', value => {
    const state = input(); state.video['width'] = value;
    expect(assessContract(state).outcome).toBe('UNSUPPORTED');
    state.source.width = NaN;
    expect(assessContract(state).reason).toBe('video-not-ready');
  });

  it.each(['left right', 'top bottom', '50%50%', 'left top right', '', 'top 20%'])('rejects invalid position %s', value => {
    expect(supportedPosition(value)).toBe(false);
  });

  it.each(['50% 50%', 'left top', 'top left', 'center', 'bottom', '-20% 130%'])('accepts supported position %s', value => {
    expect(supportedPosition(value)).toBe(true);
  });

  it('includes output placement/backing and rejects video clip margins and untracked top layers', () => {
    const state = input(); state.output = { parent: 1, connected: true, width: 2560, height: 1440, rect: { ...state.rect }, pointerEvents: 'none', followsVideo: true };
    expect(assessContract(state).outcome).toBe('SUPPORTED');
    const fingerprint = assessContract(state).fingerprint;
    state.output.width++;
    expect(assessContract(state).fingerprint).not.toBe(fingerprint);
    expect(assessContract(state).reason).toBe('unsupported-output');
    state.output.width--; state.output.followsVideo = false;
    expect(assessContract(state).reason).toBe('unsupported-output');
    state.output = null; state.video['overflow-x'] = 'clip'; state.video['overflow-clip-margin'] = '8px';
    expect(assessContract(state).reason).toBe('unsupported-clipping');
    state.nonvideoTopLayer = true;
    expect(assessContract(state).reason).toBe('unsupported-semantic-chain');
  });

  it('admits the browser fixed-position container fullscreen only at its matching identity', () => {
    const state = input(); state.chain[0]!.style['position'] = 'fixed';
    expect(assessContract(state).outcome).toBe('UNSUPPORTED');
    state.fullscreen = state.chain[0]!.id;
    expect(assessContract(state).outcome).toBe('SUPPORTED');
  });

  it.each(['ready', 'connected', 'playing', 'documentVisible', 'rootSupported'] as const)('rejects unavailable %s stably', field => {
    const state = input(); state[field] = false;
    expect(assessContract(state)).toEqual(assessContract(state));
    expect(assessContract(state).outcome).toBe('UNSUPPORTED');
  });

  it.each(['seeking', 'protected', 'mediaError', 'nativeControls', 'showingTracks', 'pip', 'directFullscreen', 'outsideFullscreen', 'overflow', 'nonemptyText'] as const)(
    'rejects %s before visibility', field => {
      const state = input(); state[field] = true;
      expect(assessContract(state).outcome).toBe('UNSUPPORTED');
    });
});

describe('M10.9 reverse recovery', () => {
  it('requires two fresh successful submissions and cannot resurrect a stale geometry generation', () => {
    const gate = new RecoveryGate(); gate.prove(submission(1));
    expect(gate.submit(submission(1))).toBe(false);
    expect(gate.submit(submission(1))).toBe(false);
    expect(gate.submit(submission(2))).toBe(true);
    gate.invalidate(); expect(gate.visible).toBe(false);
    gate.prove(submission(3, 2));
    expect(gate.submit(submission(3))).toBe(false);
    expect(gate.submit(submission(4, 2))).toBe(false);
    expect(gate.submit(submission(5, 2))).toBe(true);
  });

  it('resets credits across frame/source/owner changes and failed identity checks', () => {
    for (const change of [{ frameGeneration: 3 }, { sourceGeneration: 2 }, { validForRecovery: false }]) {
      const gate = new RecoveryGate(); gate.prove(submission(1));
      gate.submit(submission(1));
      expect(gate.submit({ ...submission(2), ...change })).toBe(false);
      expect(gate.submit(submission(3))).toBe(false);
      expect(gate.submit(submission(4))).toBe(true);
    }
  });

  it('scopes sequence freshness to the owner and ignores records from retired owners', () => {
    const gate = new RecoveryGate(); gate.prove(submission(100));
    gate.submit(submission(100)); gate.submit(submission(101));
    gate.prove({ ...submission(1), owner: 'video-2' });
    expect(gate.submit({ ...submission(1), owner: 'video-2' })).toBe(false);
    expect(gate.submit(submission(1000))).toBe(false);
    expect(gate.submit({ ...submission(2), owner: 'video-2' })).toBe(true);
  });
});

describe('M10.9 contract reader (fake DOM, not native)', () => {
  class FakeNode {
    static readonly TEXT_NODE = 3;
    parentNode: FakeNode | null = null;
    firstChild: FakeNode | null = null;
    nextSibling: FakeNode | null = null;
    constructor(public nodeType = 1, public textContent = '') {}
    get parentElement(): FakeElement | null { return this.parentNode instanceof FakeElement ? this.parentNode : null; }
    getRootNode(): FakeNode { return this.parentNode?.getRootNode() ?? this; }
    setChildren(...children: FakeNode[]): void {
      for (let child = this.firstChild; child;) {
        const next = child.nextSibling;
        child.parentNode = null; child.nextSibling = null; child = next;
      }
      this.firstChild = children[0] ?? null;
      children.forEach((child, index) => {
        child.parentNode = this; child.nextSibling = children[index + 1] ?? null;
      });
    }
  }

  class FakeElement extends FakeNode {
    assignedSlot: FakeElement | null = null;
    isConnected = true;
    topLayer: ':modal' | ':popover-open' | null = null;
    constructor(public style: SemanticStyle, public rect: PresentationInput['rect']) { super(); }
    getBoundingClientRect(): PresentationInput['rect'] { return this.rect; }
    matches(selector: string): boolean {
      return this.topLayer !== null && selector.split(',').map(value => value.trim()).includes(this.topLayer);
    }
  }

  class FakeShadowRoot extends FakeNode {
    constructor(public host: FakeElement, public mode: 'open' | 'closed') { super(11); }
  }

  function fixture() {
    const state = input();
    const element = (style: SemanticStyle = {}) => new FakeElement({ ...state.chain[0]!.style, ...style }, { ...state.rect });
    const view: { top: object | null } = { top: null }; view.top = view;
    const document = Object.assign(new FakeNode(9), { defaultView: view, visibilityState: 'visible',
      documentElement: { clientWidth: state.viewport.width, clientHeight: state.viewport.height },
      fullscreenElement: null, pictureInPictureElement: null });
    const video = Object.assign(element(state.video), { ownerDocument: document, readyState: 2, paused: false,
      ended: false, seeking: false, mediaKeys: null, error: null, controls: false, textTracks: [],
      videoWidth: state.source.width, videoHeight: state.source.height, currentSrc: state.source.url });
    const canvas = Object.assign(element({ 'pointer-events': 'none' }), {
      width: state.source.width * 2, height: state.source.height * 2 });
    const parent = element(); document.setChildren(parent); parent.setChildren(video, canvas);
    const styleReads = vi.fn((target: FakeElement) => ({
      getPropertyValue: (field: string) => target.style[field] ?? '', isolation: target.style['isolation'],
      display: target.style['display'], position: target.style['position'], zIndex: target.style['z-index'],
      pointerEvents: target.style['pointer-events'],
    }));
    vi.stubGlobal('Node', FakeNode); vi.stubGlobal('Element', FakeElement); vi.stubGlobal('ShadowRoot', FakeShadowRoot);
    vi.stubGlobal('getComputedStyle', styleReads);
    const read = createContractReader(video as unknown as HTMLVideoElement, canvas as unknown as HTMLCanvasElement);
    return { document, video, canvas, parent, element, read, styleReads };
  }

  function supportedSnapshot(read: () => PresentationInput) {
    const snapshot = read(), admission = assessContract(snapshot);
    expect(admission.outcome).toBe('SUPPORTED');
    expect(assessContract(read()).fingerprint).toBe(admission.fingerprint);
    return { snapshot, fingerprint: admission.fingerprint };
  }

  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it('admits a top document and rejects the same video in an iframe document', () => {
    const { document, read } = fixture(), before = supportedSnapshot(read);
    expect(before.snapshot.rootSupported).toBe(true);
    document.defaultView.top = {};
    const current = read();
    expect(current.rootSupported).toBe(false);
    expect(assessContract(current)).toMatchObject({ outcome: 'UNSUPPORTED', reason: 'unsupported-semantic-chain' });
    expect(assessContract(current).fingerprint).not.toBe(before.fingerprint);
  });

  it('rejects an open shadow root nested inside a closed root after admitting the all-open chain', () => {
    const { document, parent, video, element, read } = fixture();
    const outerHost = element(), innerHost = element();
    const outerRoot = new FakeShadowRoot(outerHost, 'open'), innerRoot = new FakeShadowRoot(innerHost, 'open');
    document.setChildren(outerHost); outerRoot.setChildren(innerHost); innerRoot.setChildren(parent);
    const before = supportedSnapshot(read);
    expect(video.getRootNode()).toBe(innerRoot);
    expect(before.snapshot.chain).toHaveLength(3);
    outerRoot.mode = 'closed';
    const current = read();
    expect(innerRoot.mode).toBe('open');
    expect(current.rootSupported).toBe(false);
    expect(assessContract(current).reason).toBe('unsupported-semantic-chain');
    expect(assessContract(current).fingerprint).not.toBe(before.fingerprint);
  });

  it.each([11, 12, 13])('bounds the ancestor read at 12 with %i ancestors', count => {
    const { document, parent, element, read, styleReads } = fixture();
    document.setChildren();
    const ancestors = [parent];
    while (ancestors.length < count) {
      const ancestor = element(); ancestor.setChildren(ancestors.at(-1)!); ancestors.push(ancestor);
    }
    document.setChildren(ancestors.at(-1)!);
    const excluded = ancestors[12];
    if (excluded) {
      Object.defineProperty(excluded, 'style', { get: () => { throw new Error('read thirteenth ancestor style'); } });
      Object.defineProperty(excluded, 'parentElement', { get: () => { throw new Error('traversed thirteenth ancestor'); } });
      vi.spyOn(excluded, 'matches').mockImplementation(() => { throw new Error('matched thirteenth ancestor'); });
    }
    const current = read();
    expect(current.rootSupported).toBe(true);
    expect(current.chain).toHaveLength(Math.min(count, 12));
    expect(current.overflow).toBe(count > 12);
    for (const ancestor of ancestors.slice(0, 12)) expect(styleReads.mock.calls.some(([target]) => target === ancestor)).toBe(true);
    if (excluded) expect(styleReads.mock.calls.some(([target]) => target === excluded)).toBe(false);
    expect(assessContract(current)).toMatchObject(count > 12
      ? { outcome: 'UNSUPPORTED', reason: 'unsupported-observability' }
      : { outcome: 'SUPPORTED', reason: null });
  });

  it.each([
    ['element', 15], ['element', 16], ['element', 17], ['text', 15], ['text', 16], ['text', 17],
  ] as const)('counts %s children before traversal at the %i-child boundary, excluding the output canvas', (kind, count) => {
    const { parent, video, canvas, element, read, styleReads } = fixture();
    const siblings = Array.from({ length: count - 1 }, () => kind === 'text'
      ? new FakeNode(FakeNode.TEXT_NODE, ' \n') : element({ position: 'absolute', 'z-index': '3' }));
    parent.setChildren(video, canvas, ...siblings);
    const excluded = count > 16 ? siblings.at(-1)! : null;
    if (excluded) {
      Object.defineProperty(excluded, 'nextSibling', { get: () => { throw new Error('traversed seventeenth child'); } });
      Object.defineProperty(excluded, 'textContent', { get: () => { throw new Error('read seventeenth child text'); } });
      if (excluded instanceof FakeElement) {
        Object.defineProperty(excluded, 'style', { get: () => { throw new Error('read seventeenth child style'); } });
      }
    }
    const current = read();
    expect(current.overflow).toBe(count > 16);
    expect(current.branches).toHaveLength(kind === 'text' ? 0 : Math.min(count, 16) - 1);
    expect(current.nonemptyText).toBe(false);
    expect(current.output?.followsVideo).toBe(true);
    if (excluded) expect(styleReads.mock.calls.some(([target]) => target === excluded)).toBe(false);
    expect(assessContract(current)).toMatchObject(count > 16
      ? { outcome: 'UNSUPPORTED', reason: 'unsupported-observability' }
      : { outcome: 'SUPPORTED', reason: null });
  });

  it('reads nonempty text at the sixteenth child instead of silently dropping it', () => {
    const { parent, video, canvas, read } = fixture();
    const text = Array.from({ length: 15 }, () => new FakeNode(FakeNode.TEXT_NODE, ' '));
    parent.setChildren(video, canvas, ...text);
    const before = supportedSnapshot(read);
    text.at(-1)!.textContent = 'caption';
    const current = read();
    expect(current).toMatchObject({ overflow: false, nonemptyText: true, branches: [] });
    expect(assessContract(current).reason).toBe('unsupported-control-stack');
    expect(assessContract(current).fingerprint).not.toBe(before.fingerprint);
  });

  it.each([
    ['video', ':modal'], ['video', ':popover-open'], ['ancestor', ':modal'], ['ancestor', ':popover-open'],
  ] as const)('flags a %s matching %s', (target, selector) => {
    const { video, parent, read } = fixture(), before = supportedSnapshot(read);
    (target === 'video' ? video : parent).topLayer = selector;
    const current = read();
    expect(current.nonvideoTopLayer).toBe(true);
    expect(assessContract(current).reason).toBe('unsupported-semantic-chain');
    expect(assessContract(current).fingerprint).not.toBe(before.fingerprint);
  });

  it.each(['width', 'height'] as const)('fingerprints the output backing %s independently of its CSS rectangle', field => {
    const { canvas, read } = fixture(), before = supportedSnapshot(read);
    canvas[field]++;
    const current = read();
    expect(current.output?.[field]).toBe(canvas[field]);
    expect(current.output?.rect).toEqual(before.snapshot.output?.rect);
    expect(assessContract(current).reason).toBe('unsupported-output');
    expect(assessContract(current).fingerprint).not.toBe(before.fingerprint);
  });

  it.each(['left', 'top', 'width', 'height'] as const)('fingerprints output rectangle %s independently of the video', field => {
    const { canvas, read } = fixture(), before = supportedSnapshot(read);
    canvas.rect[field]++;
    const current = read();
    expect(current.output?.rect[field]).toBe(canvas.rect[field]);
    expect(current.rect).toEqual(before.snapshot.rect);
    expect(assessContract(current).reason).toBe('unsupported-output');
    expect(assessContract(current).fingerprint).not.toBe(before.fingerprint);
  });

  it.each(['parent', 'adjacency'] as const)('fingerprints output %s changes even when its rectangle and backing match', change => {
    const { document, parent, video, canvas, element, read } = fixture(), before = supportedSnapshot(read);
    if (change === 'parent') {
      const otherParent = element();
      parent.setChildren(video); otherParent.setChildren(canvas); document.setChildren(parent, otherParent);
    } else parent.setChildren(video, new FakeNode(FakeNode.TEXT_NODE, ' '), canvas);
    const current = read();
    expect(current.parent).toBe(before.snapshot.parent);
    expect(current.output).toMatchObject({ rect: before.snapshot.rect, width: 2560, height: 1440, followsVideo: false });
    expect(current.output?.parent).toBeGreaterThan(0);
    if (change === 'parent') expect(current.output?.parent).not.toBe(current.parent);
    else expect(current.output?.parent).toBe(current.parent);
    expect(assessContract(current).reason).toBe('unsupported-output');
    expect(assessContract(current).fingerprint).not.toBe(before.fingerprint);
  });
});