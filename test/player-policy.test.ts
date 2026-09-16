import { describe, expect, it, vi } from 'vitest';
import {
  CapabilityRegistry, REFETCH_LIMITS, isRefetchByteCount, parseCommand,
  permissionPatternFor as pattern, resolveRefetch, validateCommand,
  validateRefetchResponse as responseOK, validateSelection as select, validateSourceUrl as validUrl,
} from '../tools/m1010/policy';
import type { SourceSelection } from '../tools/m1010/policy';

const media = 'https://media.example';
const source: SourceSelection = {
  tabId: 12, documentId: 'source-doc', ownerId: 'video', generation: 1,
  url: `${media}:8443/clip.mp4?asset=one`, protected: false,
  sourceClass: 'progressive', credentials: 'omit',
};
const token = '12345678-1234-4123-8123-123456789abc';
const nextToken = '12345678-1234-4123-9123-123456789abd';
const extensionId = 'a'.repeat(32);
const origin = `chrome-extension://${extensionId}`;
const player = { tabId: 42, documentId: 'player-document' };
const sender = {
  id: extensionId, origin, url: `${origin}/player.html`, frameId: 0,
  documentId: player.documentId, tab: { id: player.tabId, active: true },
};
const identity = {
  tabId: source.tabId, documentId: source.documentId, ownerId: source.ownerId,
  generation: source.generation, requestId: token,
};
const response = {
  url: source.url, redirected: false, status: 200, contentType: 'video/mp4',
  contentLength: 100, bytesRead: 100,
};
const registry = () => new CapabilityRegistry(extensionId, '/player.html', () => token);
const denied = (value: unknown) => expect(value).toBeNull();

describe('URL scope', () => {
  it.each([
    source.url, `${media}/clip%20one.mp4`,
    'http://127.0.0.1:8000/x', 'http://localhost:9000/x',
  ])('accepts %s', url => {
    expect(validUrl(url)?.href).toBe(url);
  });

  it.each([
    null, undefined, {}, 123, '', '/x', '//m.test/x',
    `blob:${media}/id`, 'data:video/mp4;base64,AAAA', 'chrome://settings/',
    `${origin}/x`, 'file:///x', 'javascript:alert(1)', 'ftp://m.test/x',
    'https://user:pass@m.test/x', 'https://@m.test/x', ` ${media}/x`,
    ...['/x#t=2', '/x#', '/x\n', '/x\tmp4', '/\u007f', '/%0D%0aX',
      '/%00', '/%1f', '/%7F', '/%GG', '/%', '/../x', '', '/\u202ex',
      `/${'x'.repeat(4096)}`].map(path => `${media}${path}`),
    `${media}\\@evil.example/x`, 'HTTPS://m.test/x', 'https://M.test/x',
    `${media}:99999/x`,
    ...['m.test', 'localhost.evil.example', 'localhost.', '127.0.0.2', '127.1',
      '2130706433', '0x7f000001', '[::1]'].map(host => `http://${host}/x`),
  ])('rejects case %#', value => {
    denied(validUrl(value));
  });

  it('pins origin and URL', () => {
    expect(pattern(new URL(source.url))).toBe(`${media}/*`);
    const fixture = new URL('http://localhost:8000/clip.mp4');
    expect(pattern(fixture)).toBe('http://localhost/*');
    denied(pattern(new URL('data:,x')));
    expect(resolveRefetch(source, `${media}:8443`)?.href).toBe(source.url);
    for (const claim of [media, `${media}:8444`, source.url, `${media}:8443/`, 'https://bad.test', null]) {
      denied(resolveRefetch(source, claim));
    }
    denied(resolveRefetch({ ...source, protected: true }, new URL(source.url).origin));
  });
});

describe('selection', () => {
  it.each([
    { tabId: -1 }, { tabId: 1.5 }, { generation: NaN }, { generation: Infinity },
    { generation: -1 }, { generation: Number.MAX_SAFE_INTEGER + 1 },
    { documentId: '' }, { ownerId: 'a'.repeat(129) }, { protected: 'false' },
    { url: `blob:${media}/id` }, { sourceClass: 'mse' },
    { sourceClass: 'blob' }, { sourceClass: 'stream' }, { arbitrary: true },
    { credentials: 'same-origin' }, { credentials: 'include' },
    { credentials: undefined }, { sourceClass: 'credentialed-fixture' },
  ])('rejects %j', patch => {
    denied(select({ ...source, ...patch }));
  });

  it.each(['omit', 'include'] as const)('local cookies: %s', credentials => {
    const selected = select({
      ...source, url: 'http://localhost:8000/x', sourceClass: 'credentialed-fixture', credentials,
    });
    expect(selected?.credentials).toBe(credentials);
    expect(selected && resolveRefetch(selected, 'http://localhost:8000')?.href)
      .toBe('http://localhost:8000/x');
  });

  it('plain data', () => {
    denied(validUrl({ toString() { throw new Error('coercion'); } }));
    const getter = vi.fn(() => source.url);
    denied(select({ ...source, get url() { return getter(); } }));
    expect(getter).not.toHaveBeenCalled();
    denied(select(Object.create(source)));
    denied(select({ ...source, [Symbol('hidden')]: true }));
    denied(select(new Proxy({}, { getPrototypeOf() { throw new Error('untrusted'); } })));
    denied(select(null));
  });
});

describe('commands', () => {
  it.each([
    { type: 'play' }, { type: 'pause' }, { type: 'seek', seconds: 0 },
    { type: 'seek', seconds: 86400 }, { type: 'rate', rate: 0.25 },
    { type: 'rate', rate: 4 }, { type: 'volume', volume: 0 },
    { type: 'volume', volume: 1 }, { type: 'mute', muted: false }, { type: 'mute', muted: true },
  ])('accepts %j', command => {
    expect(parseCommand({ ...identity, ...command }, source)).toEqual({ ...identity, ...command });
    expect(validateCommand).toBe(parseCommand);
  });

  it.each([
    { type: 'seek', seconds: -1 }, { type: 'seek', seconds: 86401 },
    { type: 'seek', seconds: NaN }, { type: 'seek', seconds: Infinity },
    { type: 'seek', seconds: '10' }, { type: 'rate', rate: 0.24 },
    { type: 'rate', rate: 4.01 }, { type: 'volume', volume: -0.1 },
    { type: 'volume', volume: 1.01 }, { type: 'mute', muted: 1 },
    { type: 'pause', value: 1 }, { type: 'seek' }, { type: 'eval', script: '1' },
    { type: 'play', url: 'https://bad.test/' }, { type: 'play', ownerId: 'other' },
    { type: 'play', generation: 2 }, { type: 'play', documentId: 'other' },
    { type: 'play', tabId: 99 }, { type: 'play', requestId: 'not-a-uuid' },
  ])('rejects %j', command => {
    denied(parseCommand({ ...identity, ...command }, source));
  });

  it('rejects protected commands', () => {
    denied(parseCommand({ ...identity, type: 'play' }, { ...source, protected: true }));
  });
});

describe('responses', () => {
  it('accepts MP4 up to 64 MiB', () => {
    expect(REFETCH_LIMITS).toMatchObject({ redirect: 'error', timeoutMs: 10_000 });
    for (const patch of [{}, { contentLength: null }, {
      contentLength: REFETCH_LIMITS.maxBytes, bytesRead: REFETCH_LIMITS.maxBytes,
    }]) expect(responseOK(source, { ...response, ...patch })).toBe(true);
  });

  it.each([
    null,
    { redirected: true }, { status: 302 }, { status: 206 }, { status: 404 },
    ...[`${media}:8443/other.mp4`, `${media}:8443/clip.mp4?asset=two`,
      `${media}/clip.mp4?asset=one`].map(url => ({ url })),
    { contentType: 'application/octet-stream' }, { contentType: 'text/html' },
    { contentLength: 101 }, { contentLength: '100' }, { bytesRead: 0 },
    { bytesRead: REFETCH_LIMITS.maxBytes + 1, contentLength: null },
    { contentLength: REFETCH_LIMITS.maxBytes + 1 }, { bytesRead: NaN }, { bytesRead: 1.5 },
  ])('rejects %j', patch => {
    expect(responseOK(source, patch && { ...response, ...patch })).toBe(false);
  });

  it('bounds bytes and rechecks protection', () => {
    for (const count of [-1, Infinity, NaN, 0.5, '1', REFETCH_LIMITS.maxBytes + 1]) {
      expect(isRefetchByteCount(count)).toBe(false);
    }
    expect(isRefetchByteCount(0)).toBe(true);
    expect(responseOK({ ...source, protected: true }, response)).toBe(false);
    denied(parseCommand(null, source));
  });
});

describe('capabilities', () => {
  it.each([
    { id: 'b'.repeat(32) }, { origin: 'https://bad.test' },
    { origin: 'null' },
    ...['/other.html', '/player.html?token=x', '/player.html#x', '/folder/../player.html']
      .map(path => ({ url: `${origin}${path}` })), { tab: { id: 43 } },
    { documentId: 'reloaded-player' }, { frameId: 1 }, { tab: undefined },
  ])('rejects sender %j', patch => {
    const store = registry();
    expect(store.register(source, player)).toBe(token);
    denied(store.consume(token, { ...sender, ...patch }));
    expect(store.consume(token, sender)).toEqual(source);
    denied(store.consume(token, sender));
  });

  it.each(['replace', 'revoke', 'second-source'])('revokes %s', mode => {
    const store = new CapabilityRegistry(extensionId, '/player.html', vi.fn()
      .mockReturnValueOnce(token).mockReturnValueOnce(nextToken));
    const raw = { ...source };
    store.register(raw, player);
    raw.url = 'https://bad.test/x';
    const selected = store.consume(token, sender)!;
    expect(selected.url).toBe(source.url);
    expect(selected).toEqual(source);
    expect(Object.isFrozen(selected)).toBe(true);
    expect(store.isCurrent(selected)).toBe(true);
    denied(store.register(source, player));
    if (mode === 'revoke') store.revokeSource(source.tabId);
    const replacement = mode === 'revoke' ? source : {
      ...source, generation: 2, tabId: mode === 'second-source' ? 13 : source.tabId,
    };
    expect(store.register(replacement, player)).toBe(nextToken);
    expect(store.isCurrent(selected)).toBe(false);
    denied(store.consume(token, sender));
    store.revokePlayer(player.tabId);
    denied(store.consume(nextToken, sender));
  });

  it('rejects unsafe registration', () => {
    denied(registry().register({ ...source, protected: true }, player));
    for (const patch of [{ tabId: source.tabId }, { documentId: '' }]) {
      denied(registry().register(source, { ...player, ...patch }));
    }
    for (const factory of [() => 'weak', () => { throw new Error('no crypto'); }]) {
      denied(new CapabilityRegistry(extensionId, '/player.html', factory).register(source, player));
    }
    const store = registry();
    store.register(source, player);
    store.revokeSource(source.tabId);
    denied(store.register(source, player));
    expect(() => new CapabilityRegistry('invalid', '/player.html')).toThrow();
    expect(() => new CapabilityRegistry(extensionId, '/player.html?x=1')).toThrow();
  });
});