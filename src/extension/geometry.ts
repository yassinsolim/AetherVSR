export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export type ObjectFit = 'contain' | 'cover' | 'fill' | 'none' | 'scale-down';

const STYLE_READERS: Partial<Record<string, (style: CSSStyleDeclaration) => unknown>> = {
  objectPosition: style => style.objectPosition, objectFit: style => style.objectFit,
  borderTopWidth: style => style.borderTopWidth, borderRightWidth: style => style.borderRightWidth,
  borderBottomWidth: style => style.borderBottomWidth, borderLeftWidth: style => style.borderLeftWidth,
  paddingTop: style => style.paddingTop, paddingRight: style => style.paddingRight,
  paddingBottom: style => style.paddingBottom, paddingLeft: style => style.paddingLeft,
  transform: style => style.transform, rotate: style => style.rotate, scale: style => style.scale, translate: style => style.translate,
  filter: style => style.filter, backdropFilter: style => style.backdropFilter, perspective: style => style.perspective,
  clipPath: style => style.clipPath, maskImage: style => style.maskImage, clip: style => style.clip,
  mixBlendMode: style => style.mixBlendMode, opacity: style => style.opacity, position: style => style.position, zIndex: style => style.zIndex,
  width: style => style.width, height: style => style.height,
  borderTopLeftRadius: style => style.borderTopLeftRadius, borderTopRightRadius: style => style.borderTopRightRadius,
  borderBottomRightRadius: style => style.borderBottomRightRadius, borderBottomLeftRadius: style => style.borderBottomLeftRadius,
  zoom: style => style.zoom, display: style => style.display, visibility: style => style.visibility,
  contentVisibility: style => style.contentVisibility, contain: style => style.contain, willChange: style => style.willChange,
  containerType: style => style.containerType, overflowX: style => style.overflowX, overflowY: style => style.overflowY,
  overflowClipMargin: style => style.overflowClipMargin, isolation: style => style.isolation,
};

interface GeometryProof {
  styles: { element: Element; parent: Node | null; style: CSSStyleDeclaration;
    keys: PropertyKey[]; values: unknown[]; readers: ((style: CSSStyleDeclaration) => unknown)[]; names: string[]; namedValues: string[] }[];
  clips: { element: HTMLElement; rect: Rect; layout: number[] }[];
  viewport: { element: HTMLElement; width: number; height: number } | null;
}

export type GeometryResult = ({
  ok: true;
  rect: Rect;
  clip: Rect;
  objectFit: string;
  objectPosition: string;
  borderRadius: string;
  verifyPlacement?: boolean;
  placement: { parent: Element | ShadowRoot; before: ChildNode | null };
  style: Record<string, string>;
} | {
  ok: false;
  reason: string;
  code: 'unsupported-geometry' | 'video-not-ready' | 'offscreen' | 'unsupported-controls';
}) & { proof?: GeometryProof };

export function geometryProofCurrent(geometry: GeometryResult): boolean {
  const proof = geometry.proof;
  if (!proof) return false;
  if (proof.viewport && (proof.viewport.element.clientWidth !== proof.viewport.width || proof.viewport.element.clientHeight !== proof.viewport.height)) return false;
  for (const entry of proof.styles) {
    if (entry.element.parentNode !== entry.parent) return false;
    for (let index = 0; index < entry.readers.length; index++) if (entry.readers[index]!(entry.style) !== entry.values[index]) return false;
    for (let index = 0; index < entry.names.length; index++) if (entry.style.getPropertyValue(entry.names[index]!) !== entry.namedValues[index]) return false;
  }
  for (const entry of proof.clips) {
    const bounds = entry.element.getBoundingClientRect();
    if ((['left', 'top', 'width', 'height'] as const).some(key => bounds[key] !== entry.rect[key])) return false;
    const current = [entry.element.clientLeft, entry.element.clientTop, entry.element.clientWidth, entry.element.clientHeight,
      entry.element.offsetWidth, entry.element.offsetHeight];
    if (current.some((value, index) => value !== entry.layout[index])) return false;
  }
  return true;
}

type ReadStyle = (element: Element) => CSSStyleDeclaration;

export function calculateImageRect(
  source: { width: number; height: number },
  box: Rect,
  fit: ObjectFit,
  position: [number, number],
): Rect {
  if (![source.width, source.height, box.left, box.top, box.width, box.height, ...position]
    .every(Number.isFinite) || source.width <= 0 || source.height <= 0
    || box.width < 0 || box.height < 0) {
    throw new RangeError('Image geometry must be finite with positive source dimensions.');
  }
  const contain = Math.min(box.width / source.width, box.height / source.height);
  let scale: number;
  switch (fit) {
    case 'fill': return { ...box };
    case 'contain': scale = contain; break;
    case 'cover': scale = Math.max(box.width / source.width, box.height / source.height); break;
    case 'none': scale = 1; break;
    case 'scale-down': scale = Math.min(1, contain); break;
    default: throw new RangeError('Unsupported object-fit.');
  }
  const width = source.width * scale;
  const height = source.height * scale;
  return {
    left: box.left + (box.width - width) * position[0],
    top: box.top + (box.height - height) * position[1],
    width,
    height,
  };
}

function intersect(first: Rect, second: Rect): Rect {
  const left = Math.max(first.left, second.left);
  const top = Math.max(first.top, second.top);
  return {
    left,
    top,
    width: Math.max(0, Math.min(first.left + first.width, second.left + second.width) - left),
    height: Math.max(0, Math.min(first.top + first.height, second.top + second.height) - top),
  };
}

function shadowRoot(node: Node | null): node is ShadowRoot {
  return node?.nodeType === 11 && 'host' in node && 'mode' in node;
}

function composedParent(element: Element): Element | null {
  if (element.parentElement) return element.parentElement;
  const root = element.getRootNode();
  return shadowRoot(root) ? root.host : null;
}

function parsePosition(value: string): [number, number] | null {
  const tokens = value.trim().toLowerCase().split(/\s+/);
  if (tokens.length > 2) return null;
  const horizontal = (token: string) => token === 'left' || token === 'right';
  const vertical = (token: string) => token === 'top' || token === 'bottom';
  let first = tokens[0] ?? '';
  let second = tokens[1] ?? 'center';
  if (tokens.length === 1 && vertical(first)) [first, second] = ['center', first];
  else if ((vertical(first) && (horizontal(second) || second === 'center'))
    || (first === 'center' && horizontal(second))) [first, second] = [second, first];
  const fraction = (token: string, start: string, end: string): number | null => {
    if (token === start) return 0;
    if (token === 'center') return 0.5;
    if (token === end) return 1;
    if (!/^[+-]?(?:\d+\.?\d*|\.\d+)%$/.test(token)) return null;
    const result = Number.parseFloat(token) / 100;
    return Number.isFinite(result) ? result : null;
  };
  const left = fraction(first, 'left', 'right');
  const top = fraction(second, 'top', 'bottom');
  return left === null || top === null ? null : [left, top];
}

function nonDefault(value: string, fallback = 'none'): boolean {
  return !!value && value !== fallback;
}

function supportedVideoTransform(style: CSSStyleDeclaration): boolean {
  if (nonDefault(style.transform)) {
    const match = /^matrix\(([^)]+)\)$/.exec(style.transform);
    if (!match) return false;
    const values = match[1]!.split(',').map(Number);
    if (values.length !== 6 || !values.every(Number.isFinite)
      || values[0]! <= 0 || values[3]! <= 0 || values[1] !== 0 || values[2] !== 0) return false;
  }
  if (nonDefault(style.rotate) && style.rotate !== '0deg') return false;
  if (nonDefault(style.scale)) {
    const values = style.scale.split(/\s+/);
    if (values.length > 2 || values.some((value) => !/^\d*\.?\d+%?$/.test(value)
      || Number.parseFloat(value) <= 0)) return false;
  }
  if (nonDefault(style.translate)) {
    const values = style.translate.split(/\s+/);
    if (values.length > 2 || values.some((value) => !/^(?:0|[+-]?(?:\d+\.?\d*|\.\d+)(?:px|%))$/.test(value))) return false;
  }
  return true;
}

function unsupportedEffect(style: CSSStyleDeclaration): boolean {
  return [style.filter, style.backdropFilter, style.perspective, style.clipPath,
    style.maskImage, style.getPropertyValue('-webkit-mask-image')].some((value) => nonDefault(value))
    || nonDefault(style.clip, 'auto') || nonDefault(style.mixBlendMode, 'normal');
}

function uniformRadius(
  style: CSSStyleDeclaration,
  scaleX: number,
  scaleY: number,
  percentageBox?: { width: number; height: number },
): string | null {
  const corners = [style.borderTopLeftRadius, style.borderTopRightRadius,
    style.borderBottomRightRadius, style.borderBottomLeftRadius];
  if (!corners.every((value) => value === corners[0])) return null;
  const values = (corners[0] || '0px').split(/\s+/);
  if (values.length > 2) return null;
  const scaled = (value: string, scale: number, basis: number | undefined): string | null => {
    if (!/^(?:0|(?:\d+\.?\d*|\.\d+)(?:px|%))$/.test(value)) return null;
    if (value.endsWith('%')) return basis === undefined ? value : `${Number.parseFloat(value) / 100 * basis}px`;
    return `${Number.parseFloat(value) * scale}px`;
  };
  const horizontal = scaled(values[0]!, scaleX, percentageBox?.width);
  const vertical = scaled(values[1] ?? values[0]!, scaleY, percentageBox?.height);
  if (horizontal === null || vertical === null) return null;
  return horizontal === vertical ? horizontal : `${horizontal} / ${vertical}`;
}

function inset(box: Rect, clip: Rect): string {
  return `${clip.top - box.top}px ${box.left + box.width - clip.left - clip.width}px `
    + `${box.top + box.height - clip.top - clip.height}px ${clip.left - box.left}px`;
}

function paintOrderRisk(video: Element, style: CSSStyleDeclaration, readStyle: ReadStyle): boolean {
  if (style.position !== 'static' || style.zIndex !== 'auto'
    || [style.transform, style.translate, style.rotate, style.scale].some((value) => nonDefault(value))) return false;
  let remaining = 32;
  for (let sibling = video.previousElementSibling; sibling; sibling = sibling.previousElementSibling) {
    if (remaining-- === 0) return true;
    const siblingStyle = readStyle(sibling);
    if (siblingStyle.display === 'none') continue;
    const siblingZ = Number(siblingStyle.zIndex);
    if (siblingStyle.position !== 'static' && Number.isFinite(siblingZ) && siblingZ !== 0) continue;
    return true;
  }
  return false;
}

function controlsRemainAbove(
  video: HTMLVideoElement,
  parent: Element | ShadowRoot,
  clip: Rect,
  style: CSSStyleDeclaration,
  readStyle: ReadStyle,
): boolean {
  const root = video.getRootNode() as Document | ShadowRoot;
  if (typeof root.elementsFromPoint !== 'function') return false;
  const videoZ = style.zIndex === 'auto' ? 0 : Number(style.zIndex);
  let visible = false;
  const checked = new Map<Element, boolean>();
  for (const horizontal of [0.2, 0.5, 0.8]) {
    for (const vertical of [0.2, 0.5, 0.8]) {
      const stack = root.elementsFromPoint(clip.left + clip.width * horizontal, clip.top + clip.height * vertical);
      const videoIndex = stack.indexOf(video);
      if (videoIndex < 0) return false;
      visible = true;
      for (const element of stack.slice(0, videoIndex)) {
        if (checked.has(element)) {
          if (!checked.get(element)) return false;
          continue;
        }
        let branch: Element | null = element;
        while (branch && branch.parentNode !== parent) branch = composedParent(branch);
        if (!branch && parent.nodeType === 1) {
          const container = parent as Element;
          const outer = composedParent(container);
          let outerBranch: Element | null = element;
          while (outerBranch && outerBranch.parentNode !== outer) outerBranch = composedParent(outerBranch);
          if (outer && outerBranch && outerBranch !== container) {
            const containerStyle = readStyle(container);
            const outerStyle = readStyle(outerBranch);
            const containerZ = containerStyle.zIndex === 'auto' ? 0 : Number(containerStyle.zIndex);
            const outerZ = outerStyle.zIndex === 'auto' ? 0 : Number(outerStyle.zIndex);
            if (containerStyle.position !== 'static' && outerStyle.position !== 'static'
              && Number.isFinite(containerZ) && Number.isFinite(outerZ) && outerZ > Math.max(videoZ, containerZ)) {
              checked.set(element, true);
              continue;
            }
          }
        }
        if (!branch) return false;
        const branchStyle = readStyle(branch);
        const branchZ = branchStyle.zIndex === 'auto' ? 0 : Number(branchStyle.zIndex);
        const follows = !!(video.compareDocumentPosition(branch) & 4);
        const above = branchStyle.position !== 'static'
          && (branchZ > videoZ || (branchZ === videoZ && follows));
        checked.set(element, above);
        if (!above) return false;
      }
    }
  }
  return visible;
}

export function inspectGeometry(video: HTMLVideoElement): GeometryResult {
  const proof: GeometryProof = { styles: [], clips: [], viewport: null };
  const reject = (code: Extract<GeometryResult, { ok: false }>['code'], reason: string): GeometryResult =>
    ({ ok: false, code, reason, proof });
  if (!video.isConnected || video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) {
    return reject('video-not-ready', 'Video must be connected with decoded pixels.');
  }
  if (video.controls || Array.from(video.textTracks).some((track) => track.mode === 'showing')) {
    return reject('unsupported-controls', 'Native controls or showing text tracks cannot be covered.');
  }
  const parent = video.parentElement ?? (shadowRoot(video.parentNode) ? video.parentNode : null);
  const view = video.ownerDocument.defaultView;
  if (!parent || !view || (shadowRoot(parent) && parent.mode !== 'open') || video.assignedSlot) {
    return reject('unsupported-geometry', 'A same-parent sibling placement is unavailable.');
  }
  if (video.ownerDocument.fullscreenElement === video) {
    return reject('unsupported-geometry', 'Direct-video fullscreen cannot contain a sibling canvas.');
  }
  const styles = new Map<Element, CSSStyleDeclaration>();
  let overflow = false;
  const readStyle: ReadStyle = element => {
    const cached = styles.get(element);
    if (cached) return cached;
    const style = view.getComputedStyle(element);
    if (styles.size >= 256) { overflow = true; return style; }
    const keys: PropertyKey[] = [], values: unknown[] = [], names: string[] = [], namedValues: string[] = [];
    const readers: ((style: CSSStyleDeclaration) => unknown)[] = [];
    const indices = new Map<PropertyKey, number>(), namedIndices = new Map<string, number>();
    const tracked = new Proxy(style, { get(target, key) {
      if (key === 'getPropertyValue') return (name: string) => {
        const value = target.getPropertyValue(name), index = namedIndices.get(name) ?? names.length;
        if (index === names.length) { namedIndices.set(name, index); names.push(name); }
        namedValues[index] = value;
        return value;
      };
      const value: unknown = Reflect.get(target, key, target), index = indices.get(key) ?? keys.length;
      if (index === keys.length) {
        indices.set(key, index); keys.push(key);
        readers.push(typeof key === 'string' && Object.hasOwn(STYLE_READERS, key) ? STYLE_READERS[key]! : style => Reflect.get(style, key, style) as unknown);
      }
      values[index] = value;
      return value;
    } });
    styles.set(element, tracked);
    proof.styles.push({ element, parent: element.parentNode, style, keys, values, readers, names, namedValues });
    return tracked;
  };
  const computed = readStyle(video);
  const position = parsePosition(computed.objectPosition);
  const fits: readonly string[] = ['contain', 'cover', 'fill', 'none', 'scale-down'];
  if (!position || !fits.includes(computed.objectFit)) {
    return reject('unsupported-geometry', 'Only standard object-fit and one/two keyword or percentage positions are supported.');
  }
  if ([computed.borderTopWidth, computed.borderRightWidth, computed.borderBottomWidth, computed.borderLeftWidth,
    computed.paddingTop, computed.paddingRight, computed.paddingBottom, computed.paddingLeft]
    .some((value) => Number.parseFloat(value) !== 0)) {
    return reject('unsupported-geometry', 'Video borders and padding must be zero.');
  }
  if (!supportedVideoTransform(computed) || unsupportedEffect(computed)
    || Number(computed.opacity) !== 1) {
    return reject('unsupported-geometry', 'Video transforms must be positive axis-aligned 2D scale/translation without effects.');
  }
  if (computed.position === 'static' && computed.zIndex !== 'auto'
    && !/^(?:inline-)?(?:flex|grid)$/.test(readStyle(composedParent(video)!).display)) {
    return reject('unsupported-geometry', 'An inactive video z-index cannot be copied to a fixed canvas.');
  }
  const bounds = video.getBoundingClientRect();
  const rect: Rect = { left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height };
  if (!Object.values(rect).every(Number.isFinite) || rect.width <= 0 || rect.height <= 0) {
    return reject('offscreen', 'Video has no visible layout box.');
  }
  const layoutWidth = Number.parseFloat(computed.width);
  const layoutHeight = Number.parseFloat(computed.height);
  if (!(layoutWidth > 0 && layoutHeight > 0)) {
    return reject('unsupported-geometry', 'Resolved video content dimensions are unavailable.');
  }
  const scaleX = rect.width / layoutWidth;
  const scaleY = rect.height / layoutHeight;
  if ((computed.objectFit === 'contain' || computed.objectFit === 'cover') && Math.abs(scaleX - scaleY) > 0.001) {
    return reject('unsupported-geometry', 'Nonuniform scaling with contain/cover cannot preserve the fitted image.');
  }
  let borderRadius = uniformRadius(computed, scaleX, scaleY);
  if (borderRadius === null) return reject('unsupported-geometry', 'Only uniform pixel or percentage corner radii are supported.');
  let verifyPlacement = false;
  let unresolvedZoom = false;
  let unresolvedOverflow = false;
  let ancestorRadius: string | null = null;
  let containingBlockOutside = computed.position === 'fixed' ? 'fixed' : computed.position === 'absolute' ? 'absolute' : null;
  proof.viewport = { element: video.ownerDocument.documentElement, width: video.ownerDocument.documentElement.clientWidth,
    height: video.ownerDocument.documentElement.clientHeight };
  let clip = intersect(rect, { left: 0, top: 0,
    width: video.ownerDocument.documentElement.clientWidth,
    height: video.ownerDocument.documentElement.clientHeight });
  for (let element: Element | null = video; element; element = composedParent(element)) {
    const style = element === video ? computed : readStyle(element);
    if (style.zoom && !['1', 'normal'].includes(style.zoom)) unresolvedZoom = true;
    if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse'
      || style.contentVisibility === 'hidden') return reject('offscreen', 'Video or ancestor is hidden.');
    if (element === video) continue;
    if (containingBlockOutside === 'absolute' && style.position !== 'static' && style.display !== 'contents') containingBlockOutside = null;
    if ([style.transform, style.translate, style.rotate, style.scale].some((value) => nonDefault(value))
      || unsupportedEffect(style) || /\b(?:layout|paint|strict|content)\b/.test(style.contain)
      || /\b(?:transform|translate|rotate|scale|filter|perspective|contain)\b/.test(style.willChange)
      || nonDefault(style.contentVisibility, 'visible') || (style.containerType && !['normal', 'size', 'inline-size'].includes(style.containerType))
      || element.assignedSlot) {
      return reject('unsupported-geometry', 'An ancestor changes fixed positioning or has unsupported effects/slotting.');
    }
    if (style.containerType === 'size' || style.containerType === 'inline-size') verifyPlacement = true;
    const clipsX = /^(?:hidden|clip|auto|scroll)$/.test(style.overflowX);
    const clipsY = /^(?:hidden|clip|auto|scroll)$/.test(style.overflowY);
    if (!clipsX && !clipsY) {
      if (element === video.ownerDocument.fullscreenElement) break;
      if (containingBlockOutside !== 'fixed' && (style.position === 'absolute' || style.position === 'fixed')) containingBlockOutside = style.position;
      continue;
    }
    if (containingBlockOutside !== null) unresolvedOverflow = true;
    if (nonDefault(style.overflowClipMargin, '0px') && (style.overflowX === 'clip' || style.overflowY === 'clip')) {
      return reject('unsupported-geometry', 'Only rectangular ancestor overflow clipping is supported.');
    }
    const ancestor = element as HTMLElement;
    const ancestorRect = ancestor.getBoundingClientRect();
    proof.clips.push({ element: ancestor, rect: { left: ancestorRect.left, top: ancestorRect.top, width: ancestorRect.width, height: ancestorRect.height },
      layout: [ancestor.clientLeft, ancestor.clientTop, ancestor.clientWidth, ancestor.clientHeight, ancestor.offsetWidth, ancestor.offsetHeight] });
    if (ancestor.offsetWidth <= 0 || ancestor.offsetHeight <= 0) return reject('offscreen', 'Clipping ancestor has no layout box.');
    const radius = uniformRadius(style, 1, 1);
    if (radius !== '0px') {
      const plainBox = [style.borderTopWidth, style.borderRightWidth, style.borderBottomWidth, style.borderLeftWidth,
        style.paddingTop, style.paddingRight, style.paddingBottom, style.paddingLeft].every(value => Number.parseFloat(value) === 0);
      const coincident = Math.abs(ancestorRect.left - rect.left) < 0.01 && Math.abs(ancestorRect.top - rect.top) < 0.01
        && Math.abs(ancestorRect.width - rect.width) < 0.01 && Math.abs(ancestorRect.height - rect.height) < 0.01;
      const matchingRadius = ancestorRadius !== null ? ancestorRadius === radius : borderRadius === '0px'
        || (uniformRadius(computed, 1, 1) === radius && Math.abs(rect.width - layoutWidth) < 1 / 64 && Math.abs(rect.height - layoutHeight) < 1 / 64);
      if (containingBlockOutside !== null || ['HTML', 'BODY'].includes(element.tagName)
        || !plainBox || !coincident || !radius || !/^\d+(?:\.\d+)?px$/.test(radius)
        || !['contain', 'cover', 'fill'].includes(computed.objectFit)
        || !['hidden', 'clip'].includes(style.overflowX) || style.overflowX !== style.overflowY
        || (style.zoom && !['1', 'normal'].includes(style.zoom))
        || !matchingRadius) {
        return reject('unsupported-geometry', 'Rounded ancestor clips require coincident undecorated boxes and matching circular pixel radii.');
      }
      borderRadius = radius;
      ancestorRadius = radius;
      verifyPlacement = true;
    }
    const zoomX = ancestorRect.width / ancestor.offsetWidth;
    const zoomY = ancestorRect.height / ancestor.offsetHeight;
    clip = intersect(clip, {
      left: clipsX ? ancestorRect.left + ancestor.clientLeft * zoomX : clip.left,
      top: clipsY ? ancestorRect.top + ancestor.clientTop * zoomY : clip.top,
      width: clipsX ? ancestor.clientWidth * zoomX : clip.width,
      height: clipsY ? ancestor.clientHeight * zoomY : clip.height,
    });
    if (element === video.ownerDocument.fullscreenElement) break;
    if (containingBlockOutside !== 'fixed' && (style.position === 'absolute' || style.position === 'fixed')) containingBlockOutside = style.position;
  }
  if (clip.width <= 0 || clip.height <= 0) return reject('offscreen', 'Video is outside the viewport or ancestor clip.');
  if (verifyPlacement && unresolvedZoom) return reject('unsupported-geometry', 'New clipping/container cases require an unzoomed ancestor chain.');
  if (verifyPlacement && unresolvedOverflow) return reject('unsupported-geometry', 'Overflow outside the containing-block chain is not supported for new clipping/container cases.');
  if (paintOrderRisk(video, computed, readStyle)) {
    return reject('unsupported-controls', 'Preceding sibling paint order cannot be preserved.');
  }
  if (verifyPlacement) {
    for (let ancestor = composedParent(video); ancestor; ancestor = composedParent(ancestor)) {
      if (ancestor === video.ownerDocument.fullscreenElement) break;
      const style = readStyle(ancestor);
      if (style.position === 'static' && style.zIndex !== 'auto') {
        const parent = composedParent(ancestor);
        if (!parent || !/^(?:inline-)?(?:flex|grid)$/.test(readStyle(parent).display)) {
          return reject('unsupported-controls', 'An inactive ancestor z-index cannot prove caption paint order.');
        }
        break;
      }
      if (style.isolation === 'isolate' || Number(style.opacity) < 1
        || (style.position !== 'static' && style.zIndex !== 'auto')) break;
      if (paintOrderRisk(ancestor, style, readStyle)) {
        return reject('unsupported-controls', 'Preceding ancestor-branch paint order cannot be preserved.');
      }
    }
  }
  if (!controlsRemainAbove(video, parent, clip, computed, readStyle)) {
    return reject('unsupported-controls', 'Sampled video visibility or control stacking cannot be preserved.');
  }
  const objectFit = computed.objectFit as ObjectFit;
  const objectPosition = `${position[0] * 100}% ${position[1] * 100}%`;
  const imageSized = objectFit === 'none' || objectFit === 'scale-down';
  const image = calculateImageRect({ width: video.videoWidth, height: video.videoHeight },
    { left: 0, top: 0, width: layoutWidth, height: layoutHeight }, objectFit, position);
  const canvas = imageSized ? {
    left: rect.left + image.left * scaleX, top: rect.top + image.top * scaleY,
    width: image.width * scaleX, height: image.height * scaleY,
  } : rect;
  const clipRadius = imageSized ? uniformRadius(computed, scaleX, scaleY, rect)! : borderRadius;
  const roundedMask = !imageSized && borderRadius !== '0px';
  const style: Record<string, string> = {
    all: 'initial', position: 'fixed', display: 'block',
    left: `${canvas.left}px`, top: `${canvas.top}px`, width: `${canvas.width}px`, height: `${canvas.height}px`,
    margin: '0', padding: '0', border: '0', 'box-sizing': 'border-box',
    'pointer-events': 'none', 'z-index': computed.zIndex,
    'object-fit': imageSized ? 'fill' : objectFit, 'object-position': objectPosition,
    'border-radius': imageSized ? '0px' : borderRadius,
    'clip-path': imageSized ? `inset(${inset(canvas, rect)} round ${clipRadius})`
      : roundedMask ? `inset(0px round ${borderRadius})` : `inset(${inset(rect, clip)})`,
  };
  if (imageSized || roundedMask) {
    style.clip = `rect(${clip.top - canvas.top}px, ${clip.left + clip.width - canvas.left}px, `
      + `${clip.top + clip.height - canvas.top}px, ${clip.left - canvas.left}px)`;
  }
  if (overflow) return reject('unsupported-geometry', 'Geometry proof exceeds its bounded dependency limit.');
  return { ok: true, rect, clip, objectFit, objectPosition, borderRadius, verifyPlacement,
    placement: { parent, before: video.nextSibling }, style, proof };
}