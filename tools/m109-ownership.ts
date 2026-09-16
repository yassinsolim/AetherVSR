export type OwnershipOutcome = 'SUPPORTED_CORRECT' | 'UNSUPPORTED_SAFE' | 'UNSAFE' | 'UNRESOLVED';

export interface OwnershipRelease {
  outcome: OwnershipOutcome;
  ownedTokenRemains: boolean | null;
  hostPreserved: boolean | null;
  reason: string;
  resources: { observers: number; stylesheets: number };
  hostAdoptedToken?: boolean | null;
}

export interface OwnershipLease {
  check(): { active: boolean; reason: string };
  release(): OwnershipRelease;
}

const MAX_RECORDS = 64;
const MAX_NODES = 256;
const MAX_TEXT = 65536;
const PROPERTY = 'anchor-name';
const ATTRIBUTE = 'data-aethervsr-anchor';

export function parseAnchorNames(value: string): string[] | null {
  if (value.length > MAX_TEXT || /[^\x09\x0a\x0c\x0d\x20-\x7e]/.test(value)) return null;
  const text = value.trim();
  if (text.toLowerCase() === 'none') return [];
  const names = text.split(',').map(name => name.trim());
  return names.length <= MAX_NODES && names.every(name => /^--[A-Za-z0-9_-]+$/.test(name)) ? names : null;
}

export interface AttributeTransition {
  before: string | null;
  after: string | null;
  owner: 'lease' | 'host';
}

export function attributeTransitions(
  oldValues: readonly (string | null)[], current: string | null, previous: string | null,
  own?: { before: string | null; after: string | null },
): { reason: string | null; transitions: AttributeTransition[] } {
  if (oldValues.length > MAX_RECORDS) return { reason: 'history-overflow', transitions: [] };
  if ([current, previous, ...oldValues].some(value => value !== null && value.length > MAX_TEXT)) {
    return { reason: 'history-text-limit', transitions: [] };
  }
  if ((oldValues.length ? oldValues[0] : current) !== previous) {
    return { reason: 'history-gap', transitions: [] };
  }
  const transitions: AttributeTransition[] = oldValues.map((before, index) => ({
    before, after: index + 1 < oldValues.length ? oldValues[index + 1]! : current, owner: 'host',
  }));
  if (own) {
    const matches = transitions.filter(transition => transition.before === own.before && transition.after === own.after);
    if (matches.length !== 1) return { reason: 'own-transition-ambiguous', transitions: [] };
    matches[0]!.owner = 'lease';
  }
  return { reason: null, transitions };
}

export interface PropertyReleaseInput {
  current: string;
  priority: string;
  token: string;
  initial: string;
  initialPriority: string;
  createdStyle: boolean;
  hostStyleWrite: boolean;
  hostAnchorWrite: boolean;
}

export function propertyReleaseDecision(input: PropertyReleaseInput): {
  action: 'keep' | 'remove' | 'set' | 'unsupported'; value: string; priority: string; removeEmptyStyle: boolean;
} {
  const names = input.current === '' ? [] : parseAnchorNames(input.current);
  const base = { value: input.current, priority: input.priority, removeEmptyStyle: false };
  if (names === null) return { ...base, action: 'unsupported' };
  if (!names.includes(input.token)) return { ...base, action: 'keep' };
  const remaining = names.filter(name => name !== input.token);
  if (remaining.length) return { ...base, action: 'set', value: remaining.join(', ') };
  if (input.hostAnchorWrite) return { ...base, action: 'set', value: 'none' };
  if (input.initial.toLowerCase() === 'none') {
    return { ...base, action: 'set', value: 'none', priority: input.initialPriority };
  }
  return { ...base, action: 'remove', value: '', priority: '',
    removeEmptyStyle: input.createdStyle && !input.hostStyleWrite };
}

function styleCopy(document: Document, text: string | null): HTMLElement {
  if (text !== null && text.length > MAX_TEXT) throw new Error('style-text-limit');
  const copy = document.createElement('div');
  if (text !== null) copy.setAttribute('style', text);
  return copy;
}

function propertyState(style: CSSStyleDeclaration): { value: string; priority: string } {
  return { value: style.getPropertyValue(PROPERTY), priority: style.getPropertyPriority(PROPERTY) };
}

function sameNames(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((name, index) => name === right[index]);
}

function inspectRoot(video: HTMLVideoElement): Document | ShadowRoot | null {
  const root = video.getRootNode();
  if (!video.isConnected || !video.ownerDocument.defaultView) return null;
  if (root === video.ownerDocument) return video.ownerDocument;
  if (root.nodeType === 11 && (root as ShadowRoot).mode === 'open') return root as ShadowRoot;
  return null;
}

function collision(video: HTMLVideoElement, token: string, leased: boolean): string | null {
  const root = inspectRoot(video);
  if (!root) return 'uninspectable-root';
  const document = video.ownerDocument;
  const roots: Node[] = [root];
  let visited = 0;
  while (roots.length) {
    const walker = document.createTreeWalker(roots.pop()!, 0xffffffff);
    let node: Node | null = walker.currentNode;
    while (node) {
      if (++visited > MAX_NODES) return 'collision-node-limit';
      if (node.nodeType === 1) {
        const element = node as Element;
        if (element !== video || !leased) {
          if (element.getAttribute(ATTRIBUTE) === token) return 'token-collision';
          const names = parseAnchorNames(document.defaultView!.getComputedStyle(element).getPropertyValue(PROPERTY));
          if (names === null) return 'collision-uninspectable-name';
          if (names.includes(token)) return 'token-collision';
        }
        for (const pseudo of ['::before', '::after']) {
          const style = document.defaultView!.getComputedStyle(element, pseudo);
          if (style.content === 'none' || style.content === 'normal') continue;
          const names = parseAnchorNames(style.getPropertyValue(PROPERTY));
          if (names === null) return 'collision-uninspectable-pseudo-name';
          if (names.includes(token)) return 'token-collision';
        }
        if (element.shadowRoot) roots.push(element.shadowRoot);
      }
      node = walker.nextNode();
    }
  }
  return null;
}

class AttributeJournal {
  failure: string | null = null;
  observing = false;
  private previous: string | null;
  private readonly observer: MutationObserver;

  constructor(private readonly video: HTMLVideoElement, private readonly attribute: string,
    private readonly host: (transition: AttributeTransition) => void) {
    this.previous = video.getAttribute(attribute);
    const Observer = video.ownerDocument.defaultView!.MutationObserver;
    this.observer = new Observer(records => {
      try {
        const pending = this.observer.takeRecords();
        if (records.length + pending.length > MAX_RECORDS) this.fail('history-overflow');
        else this.consume([...records, ...pending]);
      } catch { this.fail('observer-exception'); }
    });
    try {
      this.observer.observe(video, { attributes: true, attributeFilter: [attribute], attributeOldValue: true });
      this.observing = true;
    } catch (error) {
      this.observer.disconnect();
      throw error;
    }
  }

  private fail(reason: string): void {
    this.failure ??= reason;
  }

  private consume(records: MutationRecord[], own?: { before: string | null; after: string | null }): void {
    if (this.failure) return;
    if (records.length > MAX_RECORDS) { this.fail('history-overflow'); return; }
    if (records.some(record => record.target !== this.video || record.type !== 'attributes'
      || record.attributeName !== this.attribute || record.attributeNamespace !== null)) {
      this.fail('history-record-mismatch'); return;
    }
    const current = this.video.getAttribute(this.attribute);
    const history = attributeTransitions(records.map(record => record.oldValue), current, this.previous, own);
    if (history.reason) { this.fail(history.reason); return; }
    this.previous = current;
    try {
      for (const transition of history.transitions) if (transition.owner === 'host') this.host(transition);
    } catch { this.fail('host-transition-uninspectable'); }
  }

  drain(): boolean {
    this.consume(this.observer.takeRecords());
    return this.failure === null;
  }

  write(prepare: (before: string | null) => { after: string | null; apply(): void } | null): void {
    if (!this.drain()) return;
    const before = this.video.getAttribute(this.attribute);
    const operation = prepare(before);
    if (!operation) return;
    try { operation.apply(); }
    finally { this.consume(this.observer.takeRecords(), { before, after: operation.after }); }
  }

  stop(): void {
    this.observer.disconnect();
    this.observing = false;
  }
}

function rejected(reason: string, unresolved = false): OwnershipLease {
  return {
    check: () => ({ active: false, reason }),
    release: () => ({ outcome: unresolved ? 'UNRESOLVED' : 'UNSUPPORTED_SAFE', ownedTokenRemains: false,
      hostPreserved: true, reason, resources: { observers: 0, stylesheets: 0 } }),
  };
}

function validToken(document: Document, token: string): boolean {
  if (token.length > 256 || !/^--[A-Za-z0-9_-]+$/.test(token)) return false;
  const copy = styleCopy(document, null);
  copy.style.setProperty(PROPERTY, token);
  return copy.style.getPropertyValue(PROPERTY) === token;
}

export function anchorTokenPresence(inlineValue: string, computedValue: string, token: string): boolean | null {
  const inline = inlineValue === '' ? [] : parseAnchorNames(inlineValue);
  const computed = parseAnchorNames(computedValue);
  if (inline?.includes(token) || computed?.includes(token)) return true;
  return inline === null || computed === null ? null : false;
}

function tokenPresence(video: HTMLVideoElement, token: string): boolean | null {
  try {
    return anchorTokenPresence(video.style.getPropertyValue(PROPERTY),
      video.ownerDocument.defaultView!.getComputedStyle(video).getPropertyValue(PROPERTY), token);
  } catch { return null; }
}

export function leaseProperty(video: HTMLVideoElement, token: string): OwnershipLease {
  const document = video.ownerDocument;
  let initial: { value: string; priority: string };
  let createdStyle: boolean;
  let root: Document | ShadowRoot;
  try {
    const inspected = inspectRoot(video);
    if (!inspected) return rejected('uninspectable-root');
    root = inspected;
    if (!validToken(document, token)) return rejected('unsupported-token-or-property');
    const text = video.getAttribute('style');
    initial = propertyState(styleCopy(document, text).style);
    createdStyle = text === null;
    const names = initial.value === '' ? [] : parseAnchorNames(initial.value);
    if (names === null) return rejected('unsupported-inline-grammar');
    const computed = parseAnchorNames(document.defaultView!.getComputedStyle(video).getPropertyValue(PROPERTY));
    if (computed === null) return rejected('unsupported-computed-grammar');
    if (!names.length && computed.length) return rejected('stylesheet-only-names');
    if (!sameNames(names, computed)) return rejected('unusable-inline-cascade');
    const conflict = collision(video, token, false);
    if (conflict) return rejected(conflict);
  } catch { return rejected('acquisition-exception', true); }

  let journal: AttributeJournal | undefined;
  let hostStyleWrite = false;
  let hostAnchorWrite = false;
  let attempted = false;
  let admitted = false;
  let stopped: string | null = null;
  let exceptional = false;
  let result: OwnershipRelease | undefined;

  function revoke(reason: string): void { stopped ??= reason; }

  function verify(): { active: boolean; reason: string } {
    if (result) return { active: false, reason: result.reason };
    try {
      journal?.drain();
      if (journal?.failure) revoke(journal.failure);
      if (!stopped && (inspectRoot(video) !== root)) revoke('root-changed');
      if (!stopped) {
        const names = parseAnchorNames(video.style.getPropertyValue(PROPERTY));
        const computed = parseAnchorNames(document.defaultView!.getComputedStyle(video).getPropertyValue(PROPERTY));
        if (names === null || computed === null) revoke('unsupported-live-grammar');
        else if (names.filter(name => name === token).length !== 1) revoke('inline-ownership-lost');
        else if (computed.filter(name => name === token).length !== 1) revoke('computed-ownership-lost');
        else {
          const conflict = collision(video, token, true);
          if (conflict) revoke(conflict);
        }
      }
    } catch { exceptional = true; revoke('check-exception'); }
    return { active: !stopped && !result, reason: stopped ?? (result ? 'released' : 'active') };
  }

  function release(): OwnershipRelease {
    if (result) return result;
    let unsupported = false;
    try {
      journal?.drain();
      if (attempted && journal && !journal.failure) {
        journal.write(before => {
          const copy = styleCopy(document, before);
          const current = propertyState(copy.style);
          const decision = propertyReleaseDecision({ current: current.value, priority: current.priority,
            token, initial: initial.value, initialPriority: initial.priority, createdStyle, hostStyleWrite, hostAnchorWrite });
          if (decision.action === 'unsupported') { unsupported = true; return null; }
          if (decision.action === 'keep') return null;
          if (decision.action === 'set') copy.style.setProperty(PROPERTY, decision.value, decision.priority);
          else copy.style.removeProperty(PROPERTY);
          return { after: copy.getAttribute('style'), apply: () => {
            if (decision.action === 'set') video.style.setProperty(PROPERTY, decision.value, decision.priority);
            else video.style.removeProperty(PROPERTY);
          } };
        });
        journal.write(before => createdStyle && !hostStyleWrite && before === ''
          ? { after: null, apply: () => video.removeAttribute('style') } : null);
      }
    } catch { exceptional = true; revoke('release-exception'); }
    finally {
      try { journal?.stop(); } catch { exceptional = true; }
    }
    const remains = attempted ? tokenPresence(video, token) : false;
    const uncertain = journal?.failure ?? (exceptional ? stopped ?? 'cleanup-exception' : null);
    const outcome: OwnershipOutcome = remains === true ? 'UNSAFE' : uncertain ? 'UNRESOLVED' : remains !== false ? 'UNSAFE'
      : admitted && !stopped && !unsupported ? 'SUPPORTED_CORRECT' : 'UNSUPPORTED_SAFE';
    result = { outcome, ownedTokenRemains: remains, hostPreserved: uncertain ? null : true,
      reason: uncertain ?? (remains !== false ? 'owned-delta-not-proved-absent' : unsupported ? 'unsupported-live-grammar' : stopped ?? 'released'),
      resources: { observers: Number(journal?.observing ?? false), stylesheets: 0 } };
    return result;
  }

  try {
    journal = new AttributeJournal(video, 'style', transition => {
      hostStyleWrite = true;
      const before = propertyState(styleCopy(document, transition.before).style);
      const after = propertyState(styleCopy(document, transition.after).style);
      if (transition.before === transition.after || before.value !== after.value || before.priority !== after.priority) {
        hostAnchorWrite = true;
      }
      if (attempted) {
        const names = after.value === '' ? [] : parseAnchorNames(after.value);
        if (names === null) revoke('unsupported-live-grammar');
        else if (!names.includes(token)) revoke('host-removed-token');
      }
    });
    journal.write(before => {
      const copy = styleCopy(document, before);
      const current = propertyState(copy.style);
      const names = current.value === '' ? [] : parseAnchorNames(current.value);
      if (names === null || names.includes(token)) { revoke('pending-host-conflict'); return null; }
      if (stopped) return null;
      const computed = parseAnchorNames(document.defaultView!.getComputedStyle(video).getPropertyValue(PROPERTY));
      if (computed === null || !sameNames(names, computed)) { revoke('pending-host-cascade'); return null; }
      const conflict = collision(video, token, false);
      if (conflict) { revoke(conflict); return null; }
      const value = [...names, token].join(', ');
      const appended = parseAnchorNames(value);
      if (appended === null) { revoke('lease-value-limit'); return null; }
      copy.style.setProperty(PROPERTY, value, current.priority);
      const serialized = copy.getAttribute('style');
      const parsed = parseAnchorNames(copy.style.getPropertyValue(PROPERTY));
      if (serialized === null || serialized.length > MAX_TEXT || parsed === null || !sameNames(appended, parsed)
        || copy.style.getPropertyPriority(PROPERTY) !== current.priority) {
        revoke('unusable-serialized-property'); return null;
      }
      attempted = true;
      return { after: serialized, apply: () => video.style.setProperty(PROPERTY, value, current.priority) };
    });
    if (!attempted) revoke('acquisition-skipped');
    admitted = verify().active;
  } catch { exceptional = true; revoke('acquisition-exception'); }
  if (stopped) release();
  return { check: () => {
    const state = verify();
    if (!state.active) release();
    return state;
  }, release };
}

export function leaseAttribute(video: HTMLVideoElement, token: string): OwnershipLease {
  const document = video.ownerDocument;
  let root: Document | ShadowRoot;
  let parent: Node;
  try {
    const inspected = inspectRoot(video);
    if (!inspected) return rejected('uninspectable-root');
    root = inspected;
    if (video.hasAttribute(ATTRIBUTE)) return rejected('attribute-collision');
    if (!validToken(document, token)) return rejected('unsupported-token-or-property');
    const conflict = collision(video, token, false);
    if (conflict) return rejected(conflict);
    const container = root === document ? document.head : root;
    if (!container) return rejected('stylesheet-container-unavailable');
    parent = container;
  } catch { return rejected('acquisition-exception', true); }

  let journal: AttributeJournal | undefined;
  let sheet: HTMLStyleElement | undefined;
  let transferred = false;
  let attempted = false;
  let admitted = false;
  let stopped: string | null = null;
  let exceptional = false;
  let result: OwnershipRelease | undefined;

  function revoke(reason: string): void { stopped ??= reason; }

  function verify(): { active: boolean; reason: string } {
    if (result) return { active: false, reason: result.reason };
    try {
      journal?.drain();
      if (journal?.failure) revoke(journal.failure);
      if (!stopped && inspectRoot(video) !== root) revoke('root-changed');
      if (!stopped && video.getAttribute(ATTRIBUTE) !== token) revoke('attribute-ownership-lost');
      if (!stopped && (!sheet?.isConnected || sheet.parentNode !== parent || !sheet.sheet || sheet.sheet.disabled)) {
        revoke('stylesheet-ownership-lost');
      }
      if (!stopped) {
        const names = parseAnchorNames(document.defaultView!.getComputedStyle(video).getPropertyValue(PROPERTY));
        if (!names || !sameNames(names, [token])) revoke('computed-ownership-lost');
        else {
          const conflict = collision(video, token, true);
          if (conflict) revoke(conflict);
        }
      }
    } catch { exceptional = true; revoke('check-exception'); }
    return { active: !stopped && !result, reason: stopped ?? (result ? 'released' : 'active') };
  }

  function release(): OwnershipRelease {
    if (result) return result;
    try {
      journal?.drain();
      if (attempted && journal && !journal.failure) journal.write(before => !transferred && before === token
        ? { after: null, apply: () => video.removeAttribute(ATTRIBUTE) } : null);
    } catch { exceptional = true; revoke('release-exception'); }
    finally {
      try { sheet?.remove(); } catch { exceptional = true; revoke('stylesheet-cleanup-exception'); }
      try { journal?.drain(); } catch { exceptional = true; }
      try { journal?.stop(); } catch { exceptional = true; }
    }
    let remains: boolean | null = null;
    let hostAdoptedToken: boolean | null = null;
    let sheets = 0;
    try {
      sheets = Number(!!sheet?.parentNode);
      remains = !transferred && attempted && video.getAttribute(ATTRIBUTE) === token;
      hostAdoptedToken = transferred && video.getAttribute(ATTRIBUTE) === token;
    } catch { exceptional = true; }
    const uncertain = journal?.failure ?? (exceptional ? stopped ?? 'cleanup-exception' : null);
    result = { outcome: sheets ? 'UNSAFE' : uncertain ? 'UNRESOLVED' : remains !== false ? 'UNSAFE'
      : admitted && !stopped ? 'SUPPORTED_CORRECT' : 'UNSUPPORTED_SAFE',
    ownedTokenRemains: uncertain ? null : remains, hostPreserved: uncertain ? null : true,
    hostAdoptedToken: uncertain ? null : hostAdoptedToken,
    reason: uncertain ?? (sheets ? 'stylesheet-remains' : remains !== false ? 'owned-attribute-remains' : stopped ?? 'released'),
    resources: { observers: Number(journal?.observing ?? false), stylesheets: sheets } };
    return result;
  }

  try {
    journal = new AttributeJournal(video, ATTRIBUTE, () => { transferred = true; revoke('host-attribute-transfer'); });
    sheet = document.createElement('style');
    sheet.textContent = `[${ATTRIBUTE}="${token}"] { ${PROPERTY}: ${token}; }`;
    parent.appendChild(sheet);
    journal.write(() => {
      if (transferred) return null;
      attempted = true;
      return { after: token, apply: () => video.setAttribute(ATTRIBUTE, token) };
    });
    if (!attempted) revoke('acquisition-skipped');
    admitted = verify().active;
  } catch { exceptional = true; revoke('acquisition-exception'); }
  if (stopped) release();
  return { check: () => {
    const state = verify();
    if (!state.active) release();
    return state;
  }, release };
}