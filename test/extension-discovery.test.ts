import { describe, expect, it, vi } from 'vitest';
import { discoverVideos, MAX_DISCOVERY_NODES, OwnerSelector } from '../src/extension/discovery.js';
import type { OwnerCandidate } from '../src/extension/discovery.js';

interface FakeRoot {
  children: FakeElement[];
  ownerDocument: unknown;
  mode?: 'open' | 'closed';
}

interface FakeElement {
  localName: string;
  namespaceURI: string;
  children: FakeElement[];
  shadowRoot: FakeRoot | null;
}

function element(localName = 'div', children: FakeElement[] = []): FakeElement {
  return { localName, namespaceURI: 'http://www.w3.org/1999/xhtml', children, shadowRoot: null };
}

function tree(children: FakeElement[]) {
  const next = vi.fn();
  const createTreeWalker = vi.fn((root: FakeRoot, whatToShow: number) => {
    expect(whatToShow).toBe(1);
    const pending = [root.children.values()];
    return { nextNode: () => {
      next();
      while (pending.length) {
        const current = pending[pending.length - 1]!.next();
        if (current.done) {
          pending.pop();
          continue;
        }
        pending.push(current.value.children.values());
        return current.value;
      }
      return null;
    } };
  });
  const document = { children, ownerDocument: null, createTreeWalker };
  const shadow = (contents: FakeElement[], mode: 'open' | 'closed' = 'open'): FakeRoot =>
    ({ children: contents, ownerDocument: document, mode });
  return { document, shadow, next, createTreeWalker,
    discover: () => discoverVideos(document as unknown as Document) };
}

describe('discoverVideos', () => {
  it('discovers light DOM and nested open roots once in deterministic traversal order', () => {
    const first = element('video');
    const nested = element('video');
    const last = element('video');
    const host = element();
    const nestedHost = element();
    const setup = tree([first, host, last]);
    nestedHost.shadowRoot = setup.shadow([nested]);
    host.shadowRoot = setup.shadow([nestedHost]);
    expect(setup.discover()).toEqual({ videos: [first, nested, last],
      openRoots: [host.shadowRoot, nestedHost.shadowRoot], embeddedFrames: 0,
      closedShadowLimitation: true, truncated: false });
    expect(setup.createTreeWalker).toHaveBeenCalledTimes(3);
    expect(setup.discover().videos).toEqual([first, nested, last]);
  });

  it('counts embedded frames without reading any frame document or window', () => {
    const frame = element('iframe');
    const contentDocument = vi.fn(() => { throw new Error('Frame access is forbidden'); });
    const contentWindow = vi.fn(() => { throw new Error('Frame access is forbidden'); });
    Object.defineProperties(frame, { contentDocument: { get: contentDocument }, contentWindow: { get: contentWindow } });
    const setup = tree([frame, element('frame')]);
    expect(setup.discover()).toMatchObject({ videos: [], embeddedFrames: 2, closedShadowLimitation: true });
    expect(contentDocument).not.toHaveBeenCalled();
    expect(contentWindow).not.toHaveBeenCalled();
  });

  it('always reports the closed-shadow limitation without inventing detected contents', () => {
    const host = element();
    const setup = tree([host]);
    expect(setup.discover()).toEqual({ videos: [], openRoots: [], embeddedFrames: 0,
      closedShadowLimitation: true, truncated: false });
    host.shadowRoot = setup.shadow([element('video')], 'closed');
    expect(setup.discover().videos).toEqual([]);
    expect(discoverVideos(host.shadowRoot as unknown as ShadowRoot).videos).toEqual([]);
  });

  it('accepts an open shadow-root entry point and includes it in observable roots', () => {
    const setup = tree([]);
    const video = element('video');
    const root = setup.shadow([video]);
    expect(discoverVideos(root as unknown as ShadowRoot))
      .toMatchObject({ videos: [video], openRoots: [root], truncated: false });
  });

  it('bounds the entire traversal, including shadow trees, and explicitly reports truncation', () => {
    const host = element();
    const setup = tree([host, element('video')]);
    host.shadowRoot = setup.shadow(Array.from({ length: MAX_DISCOVERY_NODES }, () => element()));
    expect(setup.discover()).toMatchObject({ videos: [], truncated: true, openRoots: [host.shadowRoot] });
    expect(setup.next).toHaveBeenCalledTimes(MAX_DISCOVERY_NODES + 1);
  });

  it('does not report truncation at the exact limit or misidentify foreign namespace elements', () => {
    const foreignVideo = element('video');
    foreignVideo.namespaceURI = 'http://www.w3.org/2000/svg';
    const setup = tree([foreignVideo, ...Array.from({ length: MAX_DISCOVERY_NODES - 1 }, () => element())]);
    expect(setup.discover()).toMatchObject({ videos: [], truncated: false });
    expect(setup.next).toHaveBeenCalledTimes(MAX_DISCOVERY_NODES + 1);
  });
});

function candidate(id: number, visibleArea = 20_000, playing = true): OwnerCandidate {
  return { video: { id } as unknown as HTMLVideoElement, id, visibleArea, playing, eligible: true };
}

describe('OwnerSelector', () => {
  it('chooses playing, then largest area, then stable lowest id without sorting its input', () => {
    const paused = candidate(1, 100_000, false);
    const smaller = candidate(2, 19_000);
    const later = candidate(4);
    const earlier = candidate(3);
    const input = Object.freeze([paused, smaller, later, earlier]);
    expect(new OwnerSelector().update(input, 0)).toBe(earlier.video);
    expect(new OwnerSelector().update([earlier, later, smaller, paused], 0)).toBe(earlier.video);
    expect(input).toEqual([paused, smaller, later, earlier]);
  });

  it('retains an eligible owner through one-pixel area noise and id reorderings', () => {
    const selector = new OwnerSelector();
    const owner = candidate(2);
    const other = candidate(1, 20_001);
    expect(selector.update([owner], 0)).toBe(owner.video);
    expect(selector.update([other, owner], 1)).toBe(owner.video);
    expect(selector.update([owner, other], 10_000)).toBe(owner.video);
  });

  it('requires at least 25% more area for a continuous 750 ms before switching', () => {
    const selector = new OwnerSelector();
    const owner = candidate(1);
    const other = candidate(2, 25_000);
    expect(selector.update([owner], 0)).toBe(owner.video);
    expect(selector.update([owner, other], 100)).toBe(owner.video);
    expect(selector.update([owner, other], 849)).toBe(owner.video);
    expect(selector.update([owner, other], 850)).toBe(other.video);
  });

  it('resets dwell when a challenger falls below the threshold or stops playing', () => {
    const selector = new OwnerSelector();
    const owner = candidate(1);
    const other = candidate(2, 25_000);
    selector.update([owner], 0);
    selector.update([owner, other], 100);
    other.visibleArea = 24_999;
    expect(selector.update([owner, other], 700)).toBe(owner.video);
    other.visibleArea = 25_000;
    expect(selector.update([owner, other], 850)).toBe(owner.video);
    other.playing = false;
    expect(selector.update([owner, other], 1500)).toBe(owner.video);
    other.playing = true;
    expect(selector.update([owner, other], 1600)).toBe(owner.video);
    expect(selector.update([owner, other], 2350)).toBe(other.video);
  });

  it('never transfers dwell credit to a different challenger', () => {
    const selector = new OwnerSelector();
    const owner = candidate(1);
    const first = candidate(2, 25_000);
    const second = candidate(3, 30_000);
    selector.update([owner], 0);
    selector.update([owner, first], 100);
    expect(selector.update([owner, first, second], 800)).toBe(owner.video);
    expect(selector.update([owner, first, second], 1549)).toBe(owner.video);
    expect(selector.update([owner, first, second], 1550)).toBe(second.video);
  });

  it('immediately replaces removed/ineligible owners and returns null when none remain', () => {
    const selector = new OwnerSelector();
    const owner = candidate(1);
    const other = candidate(2, 16_000, false);
    selector.update([owner], 0);
    expect(selector.update([other], 1)).toBe(other.video);
    other.eligible = false;
    expect(selector.update([owner, other], 2)).toBe(owner.video);
    expect(selector.update([], 3)).toBeNull();
    expect(selector.update([other], 4)).toBeNull();
  });

  it('lets a playing video challenge a larger paused owner after dwell', () => {
    const selector = new OwnerSelector();
    const owner = candidate(1, 100_000, false);
    const playing = candidate(2);
    selector.update([owner], 0);
    expect(selector.update([owner, playing], 1)).toBe(owner.video);
    expect(selector.update([owner, playing], 750)).toBe(owner.video);
    expect(selector.update([owner, playing], 751)).toBe(playing.video);
  });

  it('never lets a paused challenger displace an eligible owner', () => {
    const selector = new OwnerSelector();
    const owner = candidate(1, 20_000, false);
    const paused = candidate(2, 100_000, false);
    selector.update([owner], 0);
    expect(selector.update([owner, paused], 1)).toBe(owner.video);
    expect(selector.update([owner, paused], 1000)).toBe(owner.video);
  });

  it('ignores invalid areas and ids without reading video geometry or playback', () => {
    const selector = new OwnerSelector();
    const valid = candidate(1);
    Object.defineProperty(valid.video, 'getBoundingClientRect', { get: () => { throw new Error('No DOM reads'); } });
    Object.defineProperty(valid.video, 'paused', { get: () => { throw new Error('No playback reads'); } });
    const invalid = [candidate(2, NaN), candidate(3, Infinity), candidate(4, -1), candidate(5, 0), candidate(NaN)];
    expect(selector.update([...invalid, valid], 0)).toBe(valid.video);
    expect(selector.update(invalid, 1)).toBeNull();
  });

  it('restarts dwell after clock rollback and rejects nonfinite timestamps', () => {
    const selector = new OwnerSelector();
    const owner = candidate(1);
    const other = candidate(2, 30_000);
    selector.update([owner], 900);
    selector.update([owner, other], 1000);
    expect(selector.update([owner, other], 100)).toBe(owner.video);
    expect(selector.update([owner, other], 849)).toBe(owner.video);
    expect(selector.update([owner, other], 850)).toBe(other.video);
    expect(() => selector.update([owner], NaN)).toThrow(RangeError);
  });
});