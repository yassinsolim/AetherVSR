export const MAX_DISCOVERY_NODES = 10_000;

export interface DiscoveryResult {
  videos: HTMLVideoElement[];
  openRoots: ShadowRoot[];
  embeddedFrames: number;
  closedShadowLimitation: true;
  truncated: boolean;
}

export function discoverVideos(root: Document | ShadowRoot): DiscoveryResult {
  const result: DiscoveryResult = {
    videos: [], openRoots: [], embeddedFrames: 0, closedShadowLimitation: true, truncated: false,
  };
  if ('mode' in root) {
    if (root.mode !== 'open') return result;
    result.openRoots.push(root);
  }
  const document = root.ownerDocument ?? root;
  const walkers = [document.createTreeWalker(root, 1)];
  let visited = 0;
  while (walkers.length) {
    const element = walkers[walkers.length - 1]!.nextNode() as Element | null;
    if (!element) {
      walkers.pop();
      continue;
    }
    if (visited === MAX_DISCOVERY_NODES) {
      result.truncated = true;
      break;
    }
    visited++;
    if (element.namespaceURI === 'http://www.w3.org/1999/xhtml') {
      if (element.localName === 'video') result.videos.push(element as HTMLVideoElement);
      if (element.localName === 'iframe' || element.localName === 'frame') result.embeddedFrames++;
    }
    const shadow = element.shadowRoot;
    if (shadow?.mode === 'open') {
      result.openRoots.push(shadow);
      walkers.push(document.createTreeWalker(shadow, 1));
    }
  }
  return result;
}

export interface OwnerCandidate {
  video: HTMLVideoElement;
  visibleArea: number;
  playing: boolean;
  id: number;
  eligible: boolean;
}

function eligible(candidate: OwnerCandidate): boolean {
  return candidate.eligible && Number.isFinite(candidate.visibleArea) && candidate.visibleArea > 0
    && Number.isFinite(candidate.id);
}

function preferred(candidate: OwnerCandidate, incumbent: OwnerCandidate): boolean {
  if (candidate.playing !== incumbent.playing) return candidate.playing;
  if (candidate.visibleArea !== incumbent.visibleArea) return candidate.visibleArea > incumbent.visibleArea;
  return candidate.id < incumbent.id;
}

export class OwnerSelector {
  private owner: HTMLVideoElement | null = null;
  private challenger: HTMLVideoElement | null = null;
  private challengerSince = 0;
  private lastUpdate: number | null = null;

  update(candidates: readonly OwnerCandidate[], now: number): HTMLVideoElement | null {
    if (!Number.isFinite(now)) throw new RangeError('Owner selection requires a finite monotonic time in milliseconds.');
    if (this.lastUpdate !== null && now < this.lastUpdate) this.challenger = null;
    this.lastUpdate = now;
    let current: OwnerCandidate | null = null;
    let best: OwnerCandidate | null = null;
    for (const candidate of candidates) {
      if (!eligible(candidate)) continue;
      if (candidate.video === this.owner) current = candidate;
      if (!best || preferred(candidate, best)) best = candidate;
    }
    if (!current) {
      this.owner = best?.video ?? null;
      this.challenger = null;
      return this.owner;
    }
    if (!best || best.video === current.video || !best.playing
      || (current.playing && best.visibleArea < current.visibleArea * 1.25)) {
      this.challenger = null;
      return this.owner;
    }
    if (this.challenger !== best.video) {
      this.challenger = best.video;
      this.challengerSince = now;
    }
    if (now - this.challengerSince >= 750) {
      this.owner = best.video;
      this.challenger = null;
    }
    return this.owner;
  }
}