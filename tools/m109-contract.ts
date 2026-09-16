import type { SuccessfulFrameSubmission, SubmissionIdentity } from './m109-submission.js';

export const EFFECT_FIELDS = [
  'transform', 'translate', 'rotate', 'scale', 'offset-path', 'filter', 'backdrop-filter',
  'perspective', 'clip-path', 'mask-image', '-webkit-mask-image', 'mask-border-source',
  '-webkit-mask-box-image-source', '-webkit-box-reflect',
] as const;
export const BOX_FIELDS = ['border-top-width', 'border-right-width', 'border-bottom-width',
  'border-left-width', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left'] as const;
export const RADIUS_FIELDS = ['border-top-left-radius', 'border-top-right-radius',
  'border-bottom-right-radius', 'border-bottom-left-radius'] as const;
export const COMMON_FIELDS = [...EFFECT_FIELDS, 'opacity', 'mix-blend-mode', 'clip', 'zoom',
  'display', 'visibility', 'content-visibility', 'contain', 'will-change', 'container-type',
  'position', 'overflow-x', 'overflow-y', 'overflow-clip-margin', 'animation-name',
  'transition-duration'] as const;
export const VIDEO_FIELDS = [...COMMON_FIELDS, ...BOX_FIELDS, ...RADIUS_FIELDS, 'corner-shape',
  'object-fit', 'object-position', 'width', 'height', 'z-index', 'background-image',
  'background-color', 'box-shadow', 'outline-style', 'content', 'object-view-box', 'appearance'] as const;

export interface ContractRect { left: number; top: number; width: number; height: number }
export type SemanticStyle = Record<string, string>;
export interface ChainNode {
  id: number;
  parent: number;
  style: SemanticStyle;
  clip: ContractRect | null;
  slot: boolean;
}
export interface PresentationInput {
  connected: boolean;
  ready: boolean;
  playing: boolean;
  seeking: boolean;
  protected: boolean;
  mediaError: boolean;
  documentVisible: boolean;
  nativeControls: boolean;
  showingTracks: boolean;
  pip: boolean;
  directFullscreen: boolean;
  outsideFullscreen: boolean;
  source: { width: number; height: number; url: string };
  rect: ContractRect;
  viewport: { width: number; height: number };
  fullscreen: number;
  parent: number;
  rootSupported: boolean;
  nonvideoTopLayer: boolean;
  output: { parent: number; connected: boolean; width: number; height: number;
    rect: ContractRect; pointerEvents: string; followsVideo: boolean } | null;
  video: SemanticStyle;
  chain: ChainNode[];
  branches: { id: number; display: string; position: string; zIndex: string }[];
  nonemptyText: boolean;
  overflow: boolean;
}
export interface Admission {
  outcome: 'SUPPORTED' | 'UNSUPPORTED';
  reason: string | null;
  fingerprint: string;
}

const defaulted = (style: SemanticStyle, field: string, value: string) => !style[field] || style[field] === value;
const zero = (value: string | undefined) => value !== undefined && /^0(?:px)?$/.test(value);
const pixels = (value: string | undefined): number => value && /^\d+(?:\.\d+)?px$/.test(value) ? Number.parseFloat(value) : NaN;
export function supportedPosition(value: string): boolean {
  const tokens = value.trim().split(/\s+/);
  const percentage = (token: string) => /^-?\d+(?:\.\d+)?%$/.test(token);
  const horizontal = (token: string) => percentage(token) || ['left', 'right', 'center'].includes(token);
  const vertical = (token: string) => percentage(token) || ['top', 'bottom', 'center'].includes(token);
  if (tokens.length === 1) return horizontal(tokens[0]!) || vertical(tokens[0]!);
  if (tokens.length !== 2) return false;
  const first = tokens[0]!, second = tokens[1]!;
  return horizontal(first) && vertical(second) || !percentage(first) && !percentage(second) && vertical(first) && horizontal(second);
}
const rectIntersects = (first: ContractRect, second: ContractRect) =>
  Math.min(first.left + first.width, second.left + second.width) > Math.max(first.left, second.left) &&
  Math.min(first.top + first.height, second.top + second.height) > Math.max(first.top, second.top);

export function assessContract(input: PresentationInput): Admission {
  const fingerprint = JSON.stringify(input);
  const reject = (reason: string): Admission => ({ outcome: 'UNSUPPORTED', reason, fingerprint });
  if (input.overflow || input.chain.length > 12 || input.branches.length > 16) return reject('unsupported-observability');
  if (!input.connected || !input.ready || ![input.source.width, input.source.height].every(value => Number.isFinite(value) && value > 0)) return reject('video-not-ready');
  if (input.protected || input.mediaError) return reject('unsupported-media');
  if (!input.playing || input.seeking) return reject(input.seeking ? 'video-seeking' : 'video-paused');
  if (!input.documentVisible || input.pip || input.directFullscreen || input.outsideFullscreen) return reject('unsupported-lifecycle');
  if (input.nativeControls || input.showingTracks) return reject('unsupported-native-controls');
  if (!input.rootSupported || input.nonvideoTopLayer || input.chain.some(node => node.slot)) return reject('unsupported-semantic-chain');
  const parent = input.chain[0]?.style;
  if (!parent || !['block', 'flow-root'].includes(parent['display'] ?? '') ||
    !(['relative', 'absolute'].includes(parent['position'] ?? '') || parent['position'] === 'fixed' && input.fullscreen === input.chain[0]?.id) || parent['isolation'] !== 'isolate') {
    return reject('unsupported-control-stack');
  }
  const video = input.video;
  if (video['display'] !== 'block' || !['relative', 'absolute'].includes(video['position'] ?? '') || video['z-index'] !== '0') {
    return reject('unsupported-control-stack');
  }
  if (input.nonemptyText || input.branches.some(branch => branch.display !== 'none' &&
    (!['relative', 'absolute', 'fixed', 'sticky'].includes(branch.position) || !Number.isFinite(Number(branch.zIndex)) || Number(branch.zIndex) <= 0))) {
    return reject('unsupported-control-stack');
  }
  if (!['contain', 'cover'].includes(video['object-fit'] ?? '') || !supportedPosition(video['object-position'] ?? '')) {
    return reject('unsupported-fit');
  }
  if (!BOX_FIELDS.every(field => zero(video[field]))) return reject('unsupported-video-box');
  const width = pixels(video['width']), height = pixels(video['height']);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0 ||
    Math.abs(width - input.rect.width) > 1 / 64 || Math.abs(height - input.rect.height) > 1 / 64) return reject('unsupported-video-box');
  if (!RADIUS_FIELDS.every(field => /^\d+(?:\.\d+)?px$/.test(video[field] ?? '') && video[field] === video[RADIUS_FIELDS[0]]) ||
    !defaulted(video, 'corner-shape', 'round')) return reject('unsupported-corner');
  if (!defaulted(video, 'background-image', 'none') || !['rgba(0, 0, 0, 0)', 'transparent'].includes(video['background-color'] ?? '') ||
    !defaulted(video, 'box-shadow', 'none') || !defaulted(video, 'outline-style', 'none') ||
    !defaulted(video, 'content', 'normal') || !defaulted(video, 'object-view-box', 'none') || !defaulted(video, 'appearance', 'none')) {
    return reject('unsupported-video-paint');
  }
  let clips = 0;
  for (const style of [video, ...input.chain.map(node => node.style)]) {
    if (EFFECT_FIELDS.some(field => !defaulted(style, field, 'none')) || !defaulted(style, 'opacity', '1') ||
      !defaulted(style, 'mix-blend-mode', 'normal') || !defaulted(style, 'clip', 'auto') ||
      !['', '1', 'normal'].includes(style['zoom'] ?? '')) return reject('unsupported-effect');
    if (style['display'] === 'none' || style['visibility'] !== 'visible' || !defaulted(style, 'content-visibility', 'visible')) return reject('offscreen');
    if (!defaulted(style, 'contain', 'none') || !defaulted(style, 'will-change', 'auto') ||
      !defaulted(style, 'container-type', 'normal') || !defaulted(style, 'animation-name', 'none') ||
      !(style['transition-duration'] ?? '0s').split(',').every(value => Number.parseFloat(value) === 0)) return reject('unsupported-semantic-chain');
    if ([style['overflow-x'], style['overflow-y']].some(value => value === 'clip') && !defaulted(style, 'overflow-clip-margin', '0px')) return reject('unsupported-clipping');
    if (style === video) continue;
    if ([style['overflow-x'], style['overflow-y']].some(value => value && value !== 'visible')) {
      clips++;
      if (clips > 2 || !RADIUS_FIELDS.every(field => zero(style[field])) || !BOX_FIELDS.every(field => zero(style[field]))) return reject('unsupported-clipping');
    }
  }
  const viewport = { left: 0, top: 0, ...input.viewport };
  if (![input.rect.left, input.rect.top, input.rect.width, input.rect.height].every(Number.isFinite) ||
    !rectIntersects(input.rect, viewport) || input.chain.some(node => node.clip && !rectIntersects(input.rect, node.clip))) return reject('offscreen');
  if (input.output && (!input.output.connected || !input.output.followsVideo || input.output.parent !== input.parent || input.output.pointerEvents !== 'none' ||
    input.output.width !== input.source.width * 2 || input.output.height !== input.source.height * 2 ||
    !Object.keys(input.rect).every(key => Math.abs(input.rect[key as keyof ContractRect] - input.output!.rect[key as keyof ContractRect]) <= 0.5))) {
    return reject('unsupported-output');
  }
  return { outcome: 'SUPPORTED', reason: null, fingerprint };
}

export function createContractReader(video: HTMLVideoElement, canvas: HTMLCanvasElement | null = null): () => PresentationInput {
  const identifiers = new WeakMap<object, number>();
  let nextId = 1;
  const id = (node: object | null) => {
    if (node === null) return 0;
    let value = identifiers.get(node);
    if (value === undefined) { value = nextId++; identifiers.set(node, value); }
    return value;
  };
  const bounds = (element: Element): ContractRect => {
    const rect = element.getBoundingClientRect();
    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
  };
  const parentOf = (element: Element): Element | null => element.parentElement ??
    (element.parentNode instanceof ShadowRoot ? element.parentNode.host : null);
  const read = (element: Element, fields: readonly string[]) => {
    const computed = getComputedStyle(element);
    return Object.fromEntries(fields.map(field => [field, computed.getPropertyValue(field)]));
  };
  return () => {
    const document = video.ownerDocument;
    const chain: ChainNode[] = [];
    let element = parentOf(video);
    let overflow = false;
    let rootSupported = document.defaultView?.top === document.defaultView && !video.assignedSlot;
    let shadowNode: Element | null = video;
    let shadowDepth = 0;
    while (shadowNode) {
      if (++shadowDepth > 12) { rootSupported = false; break; }
      const root: Node = shadowNode.getRootNode();
      if (root === document) break;
      if (!(root instanceof ShadowRoot) || root.mode !== 'open') { rootSupported = false; break; }
      shadowNode = root.host;
    }
    let nonvideoTopLayer = video.matches(':modal, :popover-open');
    while (element) {
      if (chain.length >= 12) { overflow = true; break; }
      const style = read(element, COMMON_FIELDS);
      nonvideoTopLayer ||= element.matches(':modal, :popover-open');
      if (chain.length === 0) style['isolation'] = getComputedStyle(element).isolation;
      let clip: ContractRect | null = null;
      if ([style['overflow-x'], style['overflow-y']].some(value => value && value !== 'visible')) {
        Object.assign(style, read(element, [...BOX_FIELDS, ...RADIUS_FIELDS]));
        const outer = bounds(element);
        clip = { left: outer.left + element.clientLeft, top: outer.top + element.clientTop,
          width: element.clientWidth, height: element.clientHeight };
      }
      chain.push({ id: id(element), parent: id(element.parentNode), style, clip, slot: element.assignedSlot !== null });
      if (element === document.fullscreenElement) break;
      element = parentOf(element);
    }
    const parent = video.parentNode;
    const children: ChildNode[] = [];
    for (let child = parent?.firstChild ?? null; child; child = child.nextSibling) {
      if (child === canvas) continue;
      if (children.length >= 16) { overflow = true; break; }
      children.push(child);
    }
    const others = children.filter(node => node !== video && node !== canvas && node instanceof Element) as Element[];
    if (others.length > 16) overflow = true;
    return {
      connected: video.isConnected, ready: video.readyState >= 2, playing: !video.paused && !video.ended,
      seeking: video.seeking, protected: video.mediaKeys !== null, mediaError: video.error !== null,
      documentVisible: document.visibilityState === 'visible', nativeControls: video.controls,
      showingTracks: [...video.textTracks].some(track => track.mode === 'showing'),
      pip: document.pictureInPictureElement === video, directFullscreen: document.fullscreenElement === video,
      outsideFullscreen: document.fullscreenElement !== null && !document.fullscreenElement.contains(video),
      source: { width: video.videoWidth, height: video.videoHeight, url: video.currentSrc }, rect: bounds(video),
      viewport: { width: document.documentElement.clientWidth, height: document.documentElement.clientHeight },
      fullscreen: id(document.fullscreenElement), parent: id(parent),
      rootSupported, nonvideoTopLayer,
      output: canvas ? { parent: id(canvas.parentNode), connected: canvas.isConnected, width: canvas.width,
        height: canvas.height, rect: bounds(canvas), pointerEvents: getComputedStyle(canvas).pointerEvents,
        followsVideo: video.nextSibling === canvas } : null,
      video: read(video, VIDEO_FIELDS), chain,
      branches: others.slice(0, 16).map(branch => { const style = getComputedStyle(branch);
        return { id: id(branch), display: style.display, position: style.position, zIndex: style.zIndex }; }),
      nonemptyText: children.some(node => node.nodeType === Node.TEXT_NODE && !!node.textContent?.trim()), overflow,
    };
  };
}

export class RecoveryGate {
  private identity: SubmissionIdentity | null = null;
  private frameGeneration: number | null = null;
  private sequences = new Map<string, number>();
  private credits = 0;
  visible = false;

  invalidate(): void {
    this.identity = null;
    this.frameGeneration = null;
    this.credits = 0;
    this.visible = false;
  }

  prove(identity: SubmissionIdentity): void {
    this.invalidate();
    this.identity = { ...identity };
  }

  submit(record: SuccessfulFrameSubmission): boolean {
    const identity = this.identity;
    if (!identity || record.owner !== identity.owner) return this.visible;
    if (record.sequence <= (this.sequences.get(record.owner) ?? 0)) return this.visible;
    this.sequences.set(record.owner, record.sequence);
    if (!record.validForRecovery || !identity || !identity.authorized || record.owner !== identity.owner ||
      record.sourceGeneration !== identity.sourceGeneration || record.geometryGeneration !== identity.geometryGeneration ||
      record.backingWidth !== identity.backingWidth || record.backingHeight !== identity.backingHeight) {
      this.credits = 0;
      this.visible = false;
      return false;
    }
    if (this.frameGeneration !== record.frameGeneration) { this.credits = 0; this.frameGeneration = record.frameGeneration; }
    this.credits++;
    this.visible = this.credits >= 2;
    return this.visible;
  }
}