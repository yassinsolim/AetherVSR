import { describe, expect, it, vi } from 'vitest';
import { calculateImageRect, inspectGeometry } from '../src/extension/geometry.js';
import type { ObjectFit } from '../src/extension/geometry.js';

describe('calculateImageRect', () => {
  const box = { left: 10, top: 20, width: 200, height: 200 };
  const source = { width: 400, height: 200 };

  it.each<[ObjectFit, number, number, number, number]>([
    ['contain', 10, 70, 200, 100],
    ['cover', -90, 20, 400, 200],
    ['fill', 10, 20, 200, 200],
    ['none', -90, 20, 400, 200],
    ['scale-down', 10, 70, 200, 100],
  ])('%s returns the full image, without intersecting its crop', (fit, left, top, width, height) => {
    expect(calculateImageRect(source, box, fit, [0.5, 0.5]))
      .toEqual({ left, top, width, height });
  });

  it('positions relative to remaining space, including negative cover space', () => {
    expect(calculateImageRect(source, box, 'cover', [0.25, 0.75]))
      .toEqual({ left: -40, top: 20, width: 400, height: 200 });
    expect(calculateImageRect(source, box, 'contain', [0.25, 0.75]))
      .toEqual({ left: 10, top: 95, width: 200, height: 100 });
  });

  it('does not enlarge scale-down sources and accepts out-of-box percentage positions', () => {
    expect(calculateImageRect({ width: 100, height: 50 }, box, 'scale-down', [1, 0]))
      .toEqual({ left: 110, top: 20, width: 100, height: 50 });
    expect(calculateImageRect(source, box, 'contain', [0, 1.5]).top).toBe(170);
  });

  it('rejects unavailable source dimensions and nonfinite geometry', () => {
    expect(() => calculateImageRect({ width: 0, height: 1 }, box, 'contain', [0, 0]))
      .toThrow(RangeError);
    expect(() => calculateImageRect(source, { ...box, width: -1 }, 'fill', [0, 0]))
      .toThrow(RangeError);
    expect(() => calculateImageRect(source, box, 'contain', [NaN, 0])).toThrow(RangeError);
  });
});

function fixture() {
  const defaults = {
    objectFit: 'contain', objectPosition: '50% 50%', position: 'static', zIndex: 'auto',
    display: 'block', visibility: 'visible', opacity: '1', width: '320px', height: '180px',
    transform: 'none', translate: 'none', rotate: 'none', scale: 'none',
    filter: 'none', backdropFilter: 'none', perspective: 'none', clipPath: 'none',
    clip: 'auto', maskImage: 'none', mixBlendMode: 'normal', contain: 'none',
    willChange: 'auto', contentVisibility: 'visible', containerType: 'normal',
    overflowX: 'visible', overflowY: 'visible', overflowClipMargin: '0px',
    borderTopWidth: '0px', borderRightWidth: '0px', borderBottomWidth: '0px', borderLeftWidth: '0px',
    paddingTop: '0px', paddingRight: '0px', paddingBottom: '0px', paddingLeft: '0px',
    borderTopLeftRadius: '0px', borderTopRightRadius: '0px',
    borderBottomRightRadius: '0px', borderBottomLeftRadius: '0px',
    getPropertyValue: () => '',
  };
  const videoStyle = { ...defaults };
  const parentStyle = { ...defaults };
  const controlStyle = { ...defaults, position: 'absolute' };
  const view = { getComputedStyle: vi.fn((element: unknown) => {
    if (element === video) return videoStyle;
    if (element === control) return controlStyle;
    return parentStyle;
  }) };
  const document = {
    nodeType: 9, defaultView: view, documentElement: { clientWidth: 1000, clientHeight: 800 },
    fullscreenElement: null as unknown,
    elementsFromPoint: vi.fn((): unknown[] => [video, parent]),
  };
  const parent = {
    nodeType: 1, parentElement: null, parentNode: document, assignedSlot: null,
    getRootNode: () => document, clientLeft: 0, clientTop: 0,
    clientWidth: 320, clientHeight: 180, offsetWidth: 320, offsetHeight: 180,
    getBoundingClientRect: vi.fn(() => ({ left: 10, top: 20, width: 320, height: 180 })),
  };
  const control = { parentNode: parent, parentElement: parent };
  const video = {
    isConnected: true, readyState: 2, videoWidth: 640, videoHeight: 360, controls: false,
    textTracks: [] as { mode: string }[], ownerDocument: document, assignedSlot: null as unknown,
    parentElement: parent as unknown, parentNode: parent as unknown,
    nextSibling: control as unknown, getRootNode: () => document as unknown,
    getBoundingClientRect: vi.fn(() => ({ left: 10, top: 20, width: 320, height: 180 })),
    compareDocumentPosition: vi.fn(() => 4),
  };
  return { video, parent, control, document, videoStyle, parentStyle, controlStyle,
    inspect: () => inspectGeometry(video as unknown as HTMLVideoElement) };
}

describe('inspectGeometry', () => {
  it('returns viewport coordinates, a low stack level and an immediate after-video placement without writes', () => {
    const setup = fixture();
    const result = setup.inspect();
    expect(result).toMatchObject({ ok: true,
      rect: { left: 10, top: 20, width: 320, height: 180 },
      clip: { left: 10, top: 20, width: 320, height: 180 },
      objectFit: 'contain', objectPosition: '50% 50%', borderRadius: '0px',
      placement: { parent: setup.parent, before: setup.control },
      style: { position: 'fixed', left: '10px', top: '20px', width: '320px', height: '180px',
        'pointer-events': 'none', 'z-index': 'auto', 'clip-path': 'inset(0px 0px 0px 0px)' },
    });
    expect(setup.video.getBoundingClientRect).toHaveBeenCalledTimes(1);
    expect(setup.parent.getBoundingClientRect).not.toHaveBeenCalled();
    expect(setup.video.nextSibling).toBe(setup.control);
    expect(setup.videoStyle.position).toBe('static');
    expect(setup.parentStyle.position).toBe('static');
  });

  it.each(['contain', 'cover', 'fill'])('preserves %s on a full-box canvas', (fit) => {
    const setup = fixture();
    setup.videoStyle.objectFit = fit;
    setup.videoStyle.objectPosition = '25% 75%';
    expect(setup.inspect()).toMatchObject({ ok: true, objectFit: fit, objectPosition: '25% 75%',
      style: { 'object-fit': fit, 'object-position': '25% 75%' } });
  });

  it.each([
    ['center', '50% 50%'], ['left', '0% 50%'], ['top', '50% 0%'],
    ['bottom right', '100% 100%'], ['center left', '0% 50%'],
    ['right 25%', '100% 25%'], ['25%', '25% 50%'], ['-20% 120%', '-20% 120%'],
  ])('normalizes keyword/percentage position %s', (position, expected) => {
    const setup = fixture();
    setup.videoStyle.objectPosition = position!;
    expect(setup.inspect()).toMatchObject({ ok: true, objectPosition: expected });
  });

  it.each(['10px 20px', 'calc(50% + 2px) 50%', 'right 10% bottom 20%', 'top 25%', 'left right'])(
    'explicitly refuses unsupported position %s', (position) => {
    const setup = fixture();
    setup.videoStyle.objectPosition = position;
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-geometry' });
  });

  it('sizes none and scale-down independently of a future 2x intrinsic canvas', () => {
    const setup = fixture();
    setup.videoStyle.objectFit = 'none';
    expect(setup.inspect()).toMatchObject({ ok: true, objectFit: 'none',
      style: { left: '-150px', top: '-70px', width: '640px', height: '360px', 'object-fit': 'fill',
        'clip-path': 'inset(90px 160px 90px 160px round 0px)', clip: 'rect(90px, 480px, 270px, 160px)' } });
    setup.videoStyle.objectFit = 'scale-down';
    setup.video.videoWidth = 160;
    setup.video.videoHeight = 90;
    expect(setup.inspect()).toMatchObject({ ok: true,
      style: { left: '90px', top: '65px', width: '160px', height: '90px', 'object-fit': 'fill' } });
  });

  it.each(['paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
    'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth'] as const)(
    'refuses nonzero %s', (property) => {
    const setup = fixture();
    setup.videoStyle[property] = '1px';
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-geometry' });
  });

  it('resolves image-sized percentage corner clips against the video box, not canvas size', () => {
    const setup = fixture();
    Object.assign(setup.videoStyle, { borderTopLeftRadius: '25%', borderTopRightRadius: '25%',
      borderBottomLeftRadius: '25%', borderBottomRightRadius: '25%' });
    expect(setup.inspect()).toMatchObject({ ok: true, borderRadius: '25%',
      style: { 'border-radius': '25%' } });
    setup.videoStyle.objectFit = 'none';
    expect(setup.inspect()).toMatchObject({ ok: true, borderRadius: '25%',
      style: { 'border-radius': '0px', 'clip-path': 'inset(90px 160px 90px 160px round 80px / 45px)' } });
  });

  it('requires connected, decoded video and refuses native controls and showing captions', () => {
    const setup = fixture();
    setup.video.isConnected = false;
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'video-not-ready' });
    setup.video.isConnected = true;
    setup.video.readyState = 1;
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'video-not-ready' });
    setup.video.readyState = 2;
    setup.video.videoWidth = 0;
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'video-not-ready' });
    setup.video.videoWidth = 640;
    setup.video.controls = true;
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-controls' });
    setup.video.controls = false;
    setup.video.textTracks = [{ mode: 'showing' }];
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-controls' });
    setup.video.textTracks[0]!.mode = 'hidden';
    expect(setup.inspect().ok).toBe(true);
  });

  it('uses transformed bounds once and scales pixel radii independently on each axis', () => {
    const setup = fixture();
    setup.videoStyle.objectFit = 'fill';
    setup.videoStyle.transform = 'matrix(2, 0, 0, 1.5, 30, 40)';
    setup.video.getBoundingClientRect.mockReturnValue({ left: 40, top: 60, width: 640, height: 270 });
    Object.assign(setup.videoStyle, { borderTopLeftRadius: '10px', borderTopRightRadius: '10px',
      borderBottomLeftRadius: '10px', borderBottomRightRadius: '10px' });
    expect(setup.inspect()).toMatchObject({ ok: true, borderRadius: '20px / 15px',
      style: { left: '40px', top: '60px', width: '640px', height: '270px', 'border-radius': '20px / 15px' } });
    expect(setup.video.getBoundingClientRect).toHaveBeenCalledTimes(1);
  });

  it.each(['contain', 'cover'])('rejects nonuniform scaling that changes %s image fitting', fit => {
    const setup = fixture();
    setup.videoStyle.objectFit = fit;
    setup.videoStyle.transform = 'matrix(2, 0, 0, 1, 0, 0)';
    setup.video.getBoundingClientRect.mockReturnValue({ left: 10, top: 20, width: 640, height: 180 });
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-geometry' });
  });

  it.each(['matrix(1, 1, 0, 1, 0, 0)', 'matrix(1, 0, 1, 1, 0, 0)',
    'matrix(-1, 0, 0, 1, 0, 0)', 'matrix(1, 0, 0, 0, 0, 0)',
    'matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)'])(
    'refuses unsupported video transform %s', (transform) => {
    const setup = fixture();
    setup.videoStyle.transform = transform;
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-geometry' });
  });

  it.each([
    ['transform', 'matrix(1, 0, 0, 1, 10, 20)'], ['filter', 'brightness(1)'],
    ['perspective', '100px'], ['contain', 'paint'], ['contain', 'layout'],
    ['willChange', 'transform'], ['backdropFilter', 'blur(2px)'], ['maskImage', 'url(mask.png)'],
    ['clipPath', 'circle(50%)'], ['scale', '2'],
  ] as const)('refuses ancestor %s that changes the fixed overlay contract', (property, value) => {
    const setup = fixture();
    setup.parentStyle[property] = value;
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-geometry' });
    expect(setup.parent.getBoundingClientRect).not.toHaveBeenCalled();
  });

  it('intersects viewport and scrolling padding-box clips, with borders and scrollbars excluded', () => {
    const setup = fixture();
    setup.video.getBoundingClientRect.mockReturnValue({ left: -20, top: -10, width: 320, height: 180 });
    setup.parentStyle.overflowX = 'auto';
    setup.parentStyle.overflowY = 'hidden';
    Object.assign(setup.parent, { clientLeft: 2, clientTop: 4, clientWidth: 100, clientHeight: 80,
      offsetWidth: 104, offsetHeight: 84 });
    setup.parent.getBoundingClientRect.mockReturnValue({ left: -12, top: 10, width: 104, height: 84 });
    expect(setup.inspect()).toMatchObject({ ok: true,
      clip: { left: 0, top: 14, width: 90, height: 80 },
      style: { 'clip-path': 'inset(24px 210px 76px 20px)' } });
    expect(setup.parent.getBoundingClientRect).toHaveBeenCalledTimes(1);
    expect(setup.video.getBoundingClientRect).toHaveBeenCalledTimes(1);
  });

  it('clips only the specified axis and incorporates ancestor zoom from measured bounds', () => {
    const setup = fixture();
    setup.parentStyle.overflowX = 'clip';
    setup.parent.clientWidth = 100;
    setup.parent.offsetWidth = 100;
    setup.parent.getBoundingClientRect.mockReturnValue({ left: 0, top: 500, width: 200, height: 360 });
    expect(setup.inspect()).toMatchObject({ ok: true, clip: { left: 10, top: 20, width: 190, height: 180 } });
  });

  it('refuses hidden, zero-sized and fully clipped videos', () => {
    const setup = fixture();
    setup.parentStyle.visibility = 'hidden';
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'offscreen' });
    setup.parentStyle.visibility = 'visible';
    setup.video.getBoundingClientRect.mockReturnValue({ left: 1100, top: 20, width: 320, height: 180 });
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'offscreen' });
    setup.video.getBoundingClientRect.mockReturnValue({ left: 0, top: 0, width: 0, height: 0 });
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'offscreen' });
  });

  it('keeps following positioned captions/controls above, but refuses earlier equal-stack or static controls', () => {
    const setup = fixture();
    setup.document.elementsFromPoint.mockReturnValue([setup.control, setup.video, setup.parent]);
    expect(setup.inspect().ok).toBe(true);
    setup.video.compareDocumentPosition.mockReturnValue(2);
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-controls' });
    setup.controlStyle.zIndex = '1';
    expect(setup.inspect().ok).toBe(true);
    setup.video.compareDocumentPosition.mockReturnValue(4);
    setup.controlStyle.position = 'static';
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-controls' });
  });

  it('checks more than the center and fails closed when the video is absent from hit testing', () => {
    const setup = fixture();
    setup.document.elementsFromPoint.mockImplementation((...args: unknown[]) =>
      Number(args[0]) > 200 ? [setup.control, setup.video] : [setup.video]);
    setup.controlStyle.position = 'static';
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-controls' });
    setup.document.elementsFromPoint.mockReturnValue([setup.parent]);
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-controls' });
  });

  it('supports direct children of open shadow roots and container fullscreen, with host checks', () => {
    const setup = fixture();
    const root = { nodeType: 11, mode: 'open', host: setup.parent, elementsFromPoint: setup.document.elementsFromPoint };
    setup.video.parentElement = null;
    setup.video.parentNode = root;
    setup.video.getRootNode = () => root;
    setup.video.nextSibling = null;
    setup.document.fullscreenElement = setup.parent;
    expect(setup.inspect()).toMatchObject({ ok: true, placement: { parent: root, before: null } });
    setup.parentStyle.transform = 'matrix(1, 0, 0, 1, 0, 0)';
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-geometry' });
    setup.parentStyle.transform = 'none';
    setup.document.fullscreenElement = setup.video;
    expect(setup.inspect()).toMatchObject({ ok: false, code: 'unsupported-geometry' });
  });
});