import { describe, expect, it } from 'vitest';
import { anchorTokenPresence, attributeTransitions, leaseAttribute, leaseProperty, parseAnchorNames, propertyReleaseDecision,
  type PropertyReleaseInput } from '../tools/m109-ownership.js';

const token = '--aethervsr-12345678-1234-4234-8234-123456789abc';
const releaseInput = (changes: Partial<PropertyReleaseInput> = {}): PropertyReleaseInput => ({
  current: token, priority: '', token, initial: '', initialPriority: '', createdStyle: true,
  hostStyleWrite: false, hostAnchorWrite: false, ...changes,
});

describe('M10.9 ownership pure grammar (not browser validation)', () => {
  it.each(['--host', '--0, --Host_2', '\t--one,\n--two '])('accepts literal ASCII lists: %s', value => {
    expect(parseAnchorNames(value)).toEqual(value.split(',').map(name => name.trim()));
  });
  it.each(['none', 'NONE', ' none '])('accepts %s', value => expect(parseAnchorNames(value)).toEqual([]));
  it.each(['', '--', 'foo', '--a --b', '--a,', 'none, --a', 'var(--host)', '--h\\6fst', '--h\u00f6st',
    '\u00a0--host', '--host/**/', '--a!important', '--a; color:red', 'inherit', 'initial'])('rejects %s', value => {
    expect(parseAnchorNames(value)).toBeNull();
  });
});

describe('M10.9 exact-one attribute transition model', () => {
  it('attributes only the own transition and retains synchronous host reactions', () => {
    expect(attributeTransitions([null, token, token], 'host', null, { before: null, after: token })).toEqual({
      reason: null, transitions: [
        { before: null, after: token, owner: 'lease' },
        { before: token, after: token, owner: 'host' },
        { before: token, after: 'host', owner: 'host' },
      ],
    });
  });
  it('does not infer intent without a record, but fails an unobservable own write', () => {
    expect(attributeTransitions([], token, token)).toEqual({ reason: null, transitions: [] });
    expect(attributeTransitions([], token, null, { before: null, after: token }).reason).toBe('history-gap');
    expect(attributeTransitions([], token, token, { before: token, after: token }).reason).toBe('own-transition-ambiguous');
  });
  it('reconstructs host remove/recreate and same-value ownership transfers', () => {
    expect(attributeTransitions([token, null], token, token).transitions).toEqual([
      { before: token, after: null, owner: 'host' }, { before: null, after: token, owner: 'host' },
    ]);
    expect(attributeTransitions([token], token, token).transitions[0]!.owner).toBe('host');
  });
  it('rejects ambiguous duplicate transitions, gaps and overflow without a partial pass', () => {
    expect(attributeTransitions([null, token, null], token, null, { before: null, after: token }).reason)
      .toBe('own-transition-ambiguous');
    expect(attributeTransitions(['other'], token, null).reason).toBe('history-gap');
    expect(attributeTransitions(Array<string>(65).fill(token), token, token)).toEqual({ reason: 'history-overflow', transitions: [] });
    expect(attributeTransitions(Array<string>(64).fill(token), token, token).transitions).toHaveLength(64);
  });
});

describe('M10.9 O1 semantic release table', () => {
  it('removes the initially absent property and only a lease-created untouched empty style', () => {
    expect(propertyReleaseDecision(releaseInput())).toMatchObject({ action: 'remove', removeEmptyStyle: true });
    expect(propertyReleaseDecision(releaseInput({ hostStyleWrite: true }))).toMatchObject({ action: 'remove', removeEmptyStyle: false });
    expect(propertyReleaseDecision(releaseInput({ createdStyle: false }))).toMatchObject({ action: 'remove', removeEmptyStyle: false });
  });
  it('restores explicit none with its priority after only unrelated host changes', () => {
    expect(propertyReleaseDecision(releaseInput({ initial: 'none', initialPriority: 'important', hostStyleWrite: true })))
      .toMatchObject({ action: 'set', value: 'none', priority: 'important', removeEmptyStyle: false });
  });
  it('preserves current order/priority and removes every literal owned delta, not historical host names', () => {
    expect(propertyReleaseDecision(releaseInput({ current: `--new, ${token}, --second, ${token}`, priority: 'important', initial: '--old' })))
      .toMatchObject({ action: 'set', value: '--new, --second', priority: 'important' });
  });
  it('preserves same-value, priority-only and recreated host property ownership as explicit none', () => {
    expect(propertyReleaseDecision(releaseInput({ hostAnchorWrite: true, priority: 'important' })))
      .toMatchObject({ action: 'set', value: 'none', priority: 'important', removeEmptyStyle: false });
  });
  it('never reasserts a removed delta or rewrites live variables/escapes', () => {
    for (const current of ['', 'none', '--replacement']) {
      expect(propertyReleaseDecision(releaseInput({ current })).action).toBe('keep');
    }
    for (const current of ['var(--names)', '--aethervsr-\\31 2345678-1234-4234-8234-123456789abc']) {
      expect(propertyReleaseDecision(releaseInput({ current }))).toMatchObject({ action: 'unsupported', value: current });
    }
  });
});

class FakeStyle {
  private values = new Map<string, { value: string; priority: string }>();
  constructor(private readonly element: FakeElement) {}
  load(text: string): void {
    this.values.clear();
    for (const declaration of text.split(';')) {
      const separator = declaration.indexOf(':');
      if (separator < 0) continue;
      const field = declaration.slice(0, separator).trim();
      const raw = declaration.slice(separator + 1).trim();
      const important = raw.endsWith('!important');
      this.values.set(field, { value: important ? raw.slice(0, -10).trim() : raw, priority: important ? 'important' : '' });
    }
  }
  getPropertyValue(field: string): string { return this.values.get(field)?.value ?? ''; }
  getPropertyPriority(field: string): string { return this.values.get(field)?.priority ?? ''; }
  setProperty(field: string, value: string, priority = ''): void {
    this.values.set(field, { value, priority });
    this.flush();
  }
  removeProperty(field: string): string {
    const old = this.getPropertyValue(field);
    if (this.values.delete(field)) this.flush();
    return old;
  }
  private flush(): void {
    this.element.setAttribute('style', [...this.values].map(([field, entry]) =>
      `${field}: ${entry.value}${entry.priority ? ' !important' : ''};`).join(' '));
  }
}

class FakeElement {
  readonly nodeType = 1;
  readonly attributes = new Map<string, string>();
  readonly style = new FakeStyle(this);
  readonly shadowRoot = null;
  parentNode: FakeElement | null = null;
  textContent = '';
  sheet: { disabled: boolean } | null = null;
  isConnected = false;
  constructor(readonly ownerDocument: FakeDocument, readonly tag = 'div') {}
  getAttribute(name: string): string | null { return this.attributes.get(name) ?? null; }
  hasAttribute(name: string): boolean { return this.attributes.has(name); }
  setAttribute(name: string, value: string): void {
    const before = this.getAttribute(name);
    this.attributes.set(name, value);
    if (name === 'style') this.style.load(value);
    this.ownerDocument.record(this, name, before);
  }
  removeAttribute(name: string): void {
    const before = this.getAttribute(name);
    if (!this.attributes.delete(name)) return;
    if (name === 'style') this.style.load('');
    this.ownerDocument.record(this, name, before);
  }
  getRootNode(): FakeDocument { return this.ownerDocument; }
  appendChild(child: FakeElement): FakeElement {
    child.parentNode = this;
    child.isConnected = this.isConnected;
    if (child.tag === 'style') child.sheet = { disabled: false };
    this.ownerDocument.nodes.push(child);
    this.ownerDocument.onAppend?.(child);
    return child;
  }
  remove(): void { this.parentNode = null; this.isConnected = false; }
}

class FakeObserver {
  records: MutationRecord[] = [];
  target: FakeElement | null = null;
  attribute = '';
  constructor(readonly callback: MutationCallback) {}
  observe(target: FakeElement, options: MutationObserverInit): void {
    this.target = target;
    this.attribute = options.attributeFilter![0]!;
    target.ownerDocument.observers.push(this);
    target.ownerDocument.onObserve?.();
  }
  takeRecords(): MutationRecord[] { const records = this.records; this.records = []; return records; }
  disconnect(): void { this.target = null; this.records = []; }
}

class FakeDocument {
  readonly nodeType = 9;
  readonly nodes: FakeElement[] = [];
  readonly observers: FakeObserver[] = [];
  readonly head = new FakeElement(this, 'head');
  readonly video = new FakeElement(this, 'video');
  computedOverride: string | null = null;
  peerValue = 'none';
  pseudoValue = 'none';
  onWrite: ((element: FakeElement, name: string) => void) | undefined;
  onAppend: ((element: FakeElement) => void) | undefined;
  onObserve: (() => void) | undefined;
  suppressRecords = false;
  visits = 0;
  readonly defaultView = {
    MutationObserver: FakeObserver,
    getComputedStyle: (element: FakeElement, pseudo?: string) => ({ content: pseudo && this.pseudoValue !== 'none' ? '""' : 'none', getPropertyValue: () => {
      if (pseudo) return this.pseudoValue;
      if (element !== this.video) return this.peerValue;
      if (this.computedOverride !== null) return this.computedOverride;
      const inline = element.style.getPropertyValue('anchor-name');
      if (inline) return inline;
      const attribute = element.getAttribute('data-aethervsr-anchor');
      return attribute && this.nodes.some(node => node.tag === 'style' && node.isConnected) ? attribute : 'none';
    } }),
  };
  constructor() {
    this.head.isConnected = this.video.isConnected = true;
    this.nodes.push(this.head, this.video);
  }
  createElement(tag: string): FakeElement { return new FakeElement(this, tag); }
  createTreeWalker(): { currentNode: FakeDocument; nextNode: () => FakeElement | null } {
    const nodes = this.nodes.filter(node => node.isConnected);
    let index = 0;
    return { currentNode: this, nextNode: () => { this.visits++; return nodes[index++] ?? null; } };
  }
  record(element: FakeElement, name: string, oldValue: string | null): void {
    if (!this.suppressRecords) for (const observer of this.observers) {
      if (observer.target === element && observer.attribute === name) observer.records.push({
        target: element, type: 'attributes', attributeName: name, attributeNamespace: null, oldValue,
      } as unknown as MutationRecord);
    }
    this.onWrite?.(element, name);
  }
  deliver(): void {
    for (const observer of this.observers) {
      if (observer.records.length) observer.callback(observer.takeRecords(), observer as unknown as MutationObserver);
    }
  }
  get domVideo(): HTMLVideoElement { return this.video as unknown as HTMLVideoElement; }
  get resources(): { observers: number; stylesheets: number } {
    return { observers: this.observers.filter(observer => observer.target).length,
      stylesheets: this.nodes.filter(node => node.tag === 'style' && node.parentNode).length };
  }
}

describe('M10.9 lease APIs with deterministic fake DOM, not native CSS/MO proof', () => {
  it('rejects a generated pseudo-anchor collision without mutation', () => {
    for (const create of [leaseProperty, leaseAttribute]) {
      const document = new FakeDocument(); document.pseudoValue = token;
      expect(create(document.domVideo, token).check().active).toBe(false);
      expect(document.video.attributes.size).toBe(0);
      expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
    }
  });

  it('does not claim owned-delta absence when unsupported inline syntax computes to none', () => {
    expect(anchorTokenPresence(`var(--missing), ${token}`, 'none', token)).toBeNull();
    const document = new FakeDocument(), lease = leaseProperty(document.domVideo, token);
    document.video.style.setProperty('anchor-name', `var(--missing), ${token}`);
    document.computedOverride = 'none';
    expect(lease.release()).toMatchObject({ outcome: 'UNSAFE', ownedTokenRemains: null });
    expect(document.video.style.getPropertyValue('anchor-name')).toBe(`var(--missing), ${token}`);
  });

  it('O1 releases repeatedly without leaving a property, container or observer', () => {
    const document = new FakeDocument();
    for (let count = 0; count < 3; count++) {
      const lease = leaseProperty(document.domVideo, token);
      expect(lease.check()).toEqual({ active: true, reason: 'active' });
      const result = lease.release();
      expect(result).toMatchObject({ outcome: 'SUPPORTED_CORRECT', ownedTokenRemains: false, hostPreserved: true });
      expect(document.video.getAttribute('style')).toBeNull();
      expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
      expect(lease.release()).toBe(result);
      expect(lease.check()).toEqual({ active: false, reason: 'released' });
      expect(JSON.parse(JSON.stringify(result)) as unknown).toEqual(result);
    }
  });
  it.each(['--old', '--old !important', 'none', 'none !important'])('O1 preserves initial %s and container', initial => {
    const document = new FakeDocument();
    document.video.setAttribute('style', `anchor-name: ${initial};`);
    const lease = leaseProperty(document.domVideo, token);
    expect(lease.check().active).toBe(true);
    expect(lease.release().outcome).toBe('SUPPORTED_CORRECT');
    expect(document.video.getAttribute('style')).toBe(`anchor-name: ${initial};`);
  });
  it('O1 drains pending host additions/priority and preserves unrelated declarations', () => {
    const document = new FakeDocument();
    const lease = leaseProperty(document.domVideo, token);
    document.video.style.setProperty('anchor-name', `--new, ${token}, --last`, 'important');
    document.video.style.setProperty('color', 'red', 'important');
    expect(lease.release().outcome).toBe('SUPPORTED_CORRECT');
    expect(document.video.style.getPropertyValue('anchor-name')).toBe('--new, --last');
    expect(document.video.style.getPropertyPriority('anchor-name')).toBe('important');
    expect(document.video.style.getPropertyValue('color')).toBe('red');
    expect(document.video.style.getPropertyPriority('color')).toBe('important');
  });
  it.each(['same-value', 'priority', 'recreate'])('O1 preserves host %s ownership as none', operation => {
    const document = new FakeDocument();
    const lease = leaseProperty(document.domVideo, token);
    const text = document.video.getAttribute('style')!;
    if (operation === 'recreate') document.video.removeAttribute('style');
    if (operation === 'priority') document.video.style.setProperty('anchor-name', token, 'important');
    else document.video.setAttribute('style', text);
    expect(lease.release().ownedTokenRemains).toBe(false);
    expect(document.video.style.getPropertyValue('anchor-name')).toBe('none');
    expect(document.video.style.getPropertyPriority('anchor-name')).toBe(operation === 'priority' ? 'important' : '');
    expect(document.video.hasAttribute('style')).toBe(true);
  });
  it('O1 preserves unrelated-only writes without mistaking them for host anchor rewrites', () => {
    const document = new FakeDocument();
    const lease = leaseProperty(document.domVideo, token);
    document.video.style.setProperty('color', 'red');
    document.deliver();
    expect(lease.release().outcome).toBe('SUPPORTED_CORRECT');
    expect(document.video.getAttribute('style')).toBe('color: red;');
  });
  it.each(['remove', 'replace'])('O1 never reasserts after host %s', operation => {
    const document = new FakeDocument();
    const lease = leaseProperty(document.domVideo, token);
    if (operation === 'remove') document.video.style.removeProperty('anchor-name');
    else document.video.style.setProperty('anchor-name', '--replacement');
    expect(lease.check().active).toBe(false);
    expect(lease.release().outcome).toBe('UNSUPPORTED_SAFE');
    expect(document.video.style.getPropertyValue('anchor-name')).toBe(operation === 'remove' ? '' : '--replacement');
  });
  it.each(['var(--names)', '--aethervsr-\\31 2345678-1234-4234-8234-123456789abc'])('O1 leaves live %s untouched and reports retained UUID unsafe', expression => {
    const document = new FakeDocument();
    const lease = leaseProperty(document.domVideo, token);
    document.video.style.setProperty('anchor-name', expression);
    document.video.style.setProperty('--names', token);
    document.computedOverride = token;
    expect(lease.check().active).toBe(false);
    expect(lease.release()).toMatchObject({ outcome: 'UNSAFE', ownedTokenRemains: true, hostPreserved: true });
    expect(document.video.style.getPropertyValue('anchor-name')).toBe(expression);
    expect(document.video.style.getPropertyValue('--names')).toBe(token);
    expect(document.resources.observers).toBe(0);
  });
  it('O1 treats unsupported unprovable computed ownership as unsafe, not absence', () => {
    const document = new FakeDocument();
    const lease = leaseProperty(document.domVideo, token);
    document.video.style.setProperty('anchor-name', 'var(--names)');
    document.computedOverride = '\\31 unresolved';
    expect(lease.release()).toMatchObject({ outcome: 'UNSAFE', ownedTokenRemains: null });
  });
  it('O1 rejects stylesheet-only names and unsupported inline grammar without mutation', () => {
    for (const value of ['--sheet', 'var(--names)']) {
      const document = new FakeDocument();
      if (value === '--sheet') document.computedOverride = value;
      else document.video.style.setProperty('anchor-name', value);
      const before = document.video.getAttribute('style');
      const lease = leaseProperty(document.domVideo, token);
      expect(lease.check().active).toBe(false);
      expect(lease.release().outcome).toBe('UNSUPPORTED_SAFE');
      expect(document.video.getAttribute('style')).toBe(before);
      expect(document.resources.observers).toBe(0);
    }
  });
  it('O2 releases its attribute and own stylesheet, repeatedly', () => {
    const document = new FakeDocument();
    for (let count = 0; count < 3; count++) {
      const lease = leaseAttribute(document.domVideo, token);
      expect(lease.check().active).toBe(true);
      expect(document.resources).toEqual({ observers: 1, stylesheets: 1 });
      expect(lease.release()).toMatchObject({ outcome: 'SUPPORTED_CORRECT', ownedTokenRemains: false });
      expect(document.video.hasAttribute('data-aethervsr-anchor')).toBe(false);
      expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
    }
  });
  it.each(['same-value', 'replace', 'remove', 'recreate'])('O2 transfers host %s writes, including pending records', operation => {
    const document = new FakeDocument();
    const lease = leaseAttribute(document.domVideo, token);
    if (operation === 'remove' || operation === 'recreate') document.video.removeAttribute('data-aethervsr-anchor');
    if (operation !== 'remove') document.video.setAttribute('data-aethervsr-anchor', operation === 'replace' ? 'host' : token);
    const expected = document.video.getAttribute('data-aethervsr-anchor');
    expect(lease.check()).toEqual({ active: false, reason: 'host-attribute-transfer' });
    expect(lease.release()).toMatchObject({ outcome: 'UNSUPPORTED_SAFE', ownedTokenRemains: false, hostPreserved: true });
    expect(lease.release().hostAdoptedToken).toBe(operation === 'same-value' || operation === 'recreate');
    expect(document.video.getAttribute('data-aethervsr-anchor')).toBe(expected);
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it('O2 preserves unrelated attributes and styles', () => {
    const document = new FakeDocument();
    const lease = leaseAttribute(document.domVideo, token);
    document.video.setAttribute('data-host', 'yes');
    document.video.style.setProperty('color', 'blue');
    expect(lease.check().active).toBe(true);
    expect(lease.release().outcome).toBe('SUPPORTED_CORRECT');
    expect(document.video.getAttribute('data-host')).toBe('yes');
    expect(document.video.style.getPropertyValue('color')).toBe('blue');
  });
  it('O2 does not overwrite pending host writes caused by stylesheet insertion', () => {
    const document = new FakeDocument();
    document.onAppend = () => document.video.setAttribute('data-aethervsr-anchor', 'host');
    const lease = leaseAttribute(document.domVideo, token);
    expect(lease.check().active).toBe(false);
    expect(document.video.getAttribute('data-aethervsr-anchor')).toBe('host');
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it.each([leaseProperty, leaseAttribute])('rejects duplicate UUIDs and enforces actual traversal limits', acquire => {
    const document = new FakeDocument();
    document.peerValue = token;
    expect(acquire(document.domVideo, token).check().reason).toBe('token-collision');
    document.peerValue = 'none';
    for (let count = 0; count < 260; count++) {
      const node = document.createElement('div'); node.isConnected = true; document.nodes.push(node);
    }
    document.visits = 0;
    expect(acquire(document.domVideo, token).check().reason).toBe('collision-node-limit');
    expect(document.visits).toBeLessThanOrEqual(256);
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it.each([leaseProperty, leaseAttribute])('skips mutation on overflow and cleans resources without claiming a pass', acquire => {
    const document = new FakeDocument();
    const lease = acquire(document.domVideo, token);
    const name = acquire === leaseProperty ? 'style' : 'data-aethervsr-anchor';
    for (let count = 0; count < 65; count++) document.video.setAttribute(name, document.video.getAttribute(name)!);
    const before = document.video.getAttribute(name);
    expect(lease.release().outcome).toBe(acquire === leaseProperty ? 'UNSAFE' : 'UNRESOLVED');
    expect(document.video.getAttribute(name)).toBe(before);
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it.each([leaseProperty, leaseAttribute])('never silently attributes an unobservable own write', acquire => {
    const document = new FakeDocument();
    document.suppressRecords = true;
    const lease = acquire(document.domVideo, token);
    expect(lease.check().active).toBe(false);
    expect(lease.release().outcome).toBe(acquire === leaseProperty ? 'UNSAFE' : 'UNRESOLVED');
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it.each([leaseProperty, leaseAttribute])('cleans after a mutation throws following its recorded effect', acquire => {
    const document = new FakeDocument();
    document.onWrite = element => {
      if (element !== document.video) return;
      document.onWrite = undefined;
      throw new Error('injected-after-write');
    };
    const lease = acquire(document.domVideo, token);
    expect(lease.release().outcome).toBe('UNRESOLVED');
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
    expect(document.video.getAttribute('data-aethervsr-anchor')).toBeNull();
    expect(document.video.style.getPropertyValue('anchor-name')).toBe('');
  });
  it('O1 refuses an append exceeding the grammar bound before mutating', () => {
    const document = new FakeDocument();
    const names = Array.from({ length: 256 }, (_, index) => `--host${index}`).join(', ');
    document.video.style.setProperty('anchor-name', names);
    const before = document.video.getAttribute('style');
    const lease = leaseProperty(document.domVideo, token);
    expect(lease.check()).toEqual({ active: false, reason: 'lease-value-limit' });
    expect(lease.release().outcome).toBe('UNSUPPORTED_SAFE');
    expect(document.video.getAttribute('style')).toBe(before);
    expect(document.resources.observers).toBe(0);
  });
  it('O1 drains pre-acquisition host empty-style writes and preserves that host container', () => {
    const document = new FakeDocument();
    document.onObserve = () => document.video.setAttribute('style', '');
    const lease = leaseProperty(document.domVideo, token);
    expect(lease.check().active).toBe(true);
    expect(lease.release().outcome).toBe('SUPPORTED_CORRECT');
    expect(document.video.getAttribute('style')).toBe('');
  });
  it.each([leaseProperty, leaseAttribute])('attributes a synchronous same-value host reaction separately', acquire => {
    const document = new FakeDocument();
    const name = acquire === leaseProperty ? 'style' : 'data-aethervsr-anchor';
    document.onWrite = (element, attribute) => {
      if (element !== document.video || attribute !== name) return;
      document.onWrite = undefined;
      element.setAttribute(name, element.getAttribute(name)!);
    };
    const lease = acquire(document.domVideo, token);
    if (acquire === leaseProperty) {
      expect(lease.check().active).toBe(true);
      expect(lease.release().outcome).toBe('SUPPORTED_CORRECT');
      expect(document.video.style.getPropertyValue('anchor-name')).toBe('none');
    } else {
      expect(lease.check().active).toBe(false);
      expect(lease.release()).toMatchObject({ outcome: 'UNSUPPORTED_SAFE', hostAdoptedToken: true, ownedTokenRemains: false });
      expect(document.video.getAttribute(name)).toBe(token);
    }
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it.each([leaseProperty, leaseAttribute])('fails closed on a synchronous ambiguous remove/recreate reaction', acquire => {
    const document = new FakeDocument();
    const name = acquire === leaseProperty ? 'style' : 'data-aethervsr-anchor';
    document.onWrite = (element, attribute) => {
      if (element !== document.video || attribute !== name) return;
      document.onWrite = undefined;
      const value = element.getAttribute(name)!;
      element.removeAttribute(name);
      element.setAttribute(name, value);
    };
    const lease = acquire(document.domVideo, token);
    expect(lease.check().active).toBe(false);
    expect(lease.release()).toMatchObject({ outcome: acquire === leaseProperty ? 'UNSAFE' : 'UNRESOLVED', reason: 'own-transition-ambiguous', hostPreserved: null });
    expect(document.video.getAttribute(name)).not.toBeNull();
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it('O1 detects a temporary host token removal even when the final value matches', () => {
    const document = new FakeDocument();
    const lease = leaseProperty(document.domVideo, token);
    document.video.style.removeProperty('anchor-name');
    document.video.style.setProperty('anchor-name', token, 'important');
    expect(lease.check().active).toBe(false);
    expect(lease.release().outcome).toBe('UNSUPPORTED_SAFE');
    expect(document.video.getAttribute('style')).toBe('anchor-name: none !important;');
  });
  it('O2 rejects even an empty preexisting attribute without overwriting it', () => {
    const document = new FakeDocument();
    document.video.setAttribute('data-aethervsr-anchor', '');
    const lease = leaseAttribute(document.domVideo, token);
    expect(lease.check()).toEqual({ active: false, reason: 'attribute-collision' });
    expect(document.video.getAttribute('data-aethervsr-anchor')).toBe('');
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it('O1 refuses token reassertion when a host reacts during release', () => {
    const document = new FakeDocument();
    const lease = leaseProperty(document.domVideo, token);
    document.onWrite = element => {
      if (element !== document.video) return;
      document.onWrite = undefined;
      element.style.setProperty('anchor-name', token);
    };
    expect(lease.release()).toMatchObject({ outcome: 'UNSAFE', ownedTokenRemains: true });
    expect(document.video.style.getPropertyValue('anchor-name')).toBe(token);
    expect(document.resources.observers).toBe(0);
  });
  it('O2 preserves a host-adopted value written synchronously during release', () => {
    const document = new FakeDocument();
    const lease = leaseAttribute(document.domVideo, token);
    document.onWrite = element => {
      if (element !== document.video) return;
      document.onWrite = undefined;
      element.setAttribute('data-aethervsr-anchor', token);
    };
    expect(lease.release()).toMatchObject({ outcome: 'UNSUPPORTED_SAFE', ownedTokenRemains: false, hostAdoptedToken: true });
    expect(document.video.getAttribute('data-aethervsr-anchor')).toBe(token);
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
  it('O2 removes its sheet after an insertion exception', () => {
    const document = new FakeDocument();
    document.onAppend = () => { throw new Error('injected-append'); };
    expect(leaseAttribute(document.domVideo, token).release().outcome).toBe('UNRESOLVED');
    expect(document.video.getAttribute('data-aethervsr-anchor')).toBeNull();
    expect(document.resources).toEqual({ observers: 0, stylesheets: 0 });
  });
});