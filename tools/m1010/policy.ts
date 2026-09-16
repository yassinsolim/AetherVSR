export const REFETCH_LIMITS = Object.freeze({
  maxBytes: 64 * 1024 * 1024,
  timeoutMs: 10_000,
  redirect: 'error' as const,
  contentType: 'video/mp4' as const,
});

export type SourceSelection = Readonly<{
  tabId: number;
  documentId: string;
  ownerId: string;
  generation: number;
  url: string;
  protected: boolean;
  sourceClass: 'progressive' | 'credentialed-fixture';
  credentials: 'omit' | 'include';
}>;

type CommandIdentity = Readonly<{
  tabId: number;
  documentId: string;
  ownerId: string;
  generation: number;
  requestId: string;
}>;

export type PlayerCommand = CommandIdentity & Readonly<
  | { type: 'play' | 'pause' }
  | { type: 'seek'; seconds: number }
  | { type: 'rate'; rate: number }
  | { type: 'volume'; volume: number }
  | { type: 'mute'; muted: boolean }
>;

export type PlayerBinding = Readonly<{ tabId: number; documentId: string }>;

const selectionKeys = [
  'tabId', 'documentId', 'ownerId', 'generation', 'url', 'protected',
  'sourceClass', 'credentials',
] as const;
const identityKeys = ['tabId', 'documentId', 'ownerId', 'generation'] as const;
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function fields(value: unknown, keys: readonly string[], exact = true): Record<string, unknown> | null {
  try {
    if (value === null || typeof value !== 'object') return null;
    const prototype: unknown = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    if (exact && Reflect.ownKeys(value).length !== keys.length) return null;
    const result: Record<string, unknown> = {};
    for (const key of keys) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (!descriptor?.enumerable || !Object.hasOwn(descriptor, 'value')) return null;
      result[key] = descriptor.value;
    }
    return result;
  } catch {
    return null;
  }
}

function bounded(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
}

function tabId(value: unknown): value is number {
  return bounded(value, 0, 2 ** 31 - 1) && Number.isInteger(value);
}

function identifier(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(value);
}

function loopback(url: URL): boolean {
  return url.hostname === '127.0.0.1' || url.hostname === 'localhost';
}

export function validateSourceUrl(value: unknown): URL | null {
  if (typeof value !== 'string' || value.length > 4096 || !/^[\x21-\x7e]+$/.test(value)
    || /[\\#]/.test(value) || /%(?![\da-f]{2})|%(?:0[\da-f]|1[\da-f]|7f)/i.test(value)) return null;
  try {
    const url = new URL(value);
    if (url.href !== value || url.username || url.password || !url.hostname) return null;
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback(url))) return null;
    return url;
  } catch {
    return null;
  }
}

export function permissionPatternFor(url: URL): string | null {
  const valid = validateSourceUrl(url.href);
  return valid ? `${valid.protocol}//${valid.hostname}/*` : null;
}

export function validateSelection(raw: unknown): SourceSelection | null {
  const value = fields(raw, selectionKeys);
  if (!value || !tabId(value.tabId) || !identifier(value.documentId) || !identifier(value.ownerId)
    || !bounded(value.generation, 0, Number.MAX_SAFE_INTEGER) || !Number.isInteger(value.generation)
    || typeof value.protected !== 'boolean' || typeof value.url !== 'string'
    || (value.sourceClass !== 'progressive' && value.sourceClass !== 'credentialed-fixture')
    || (value.credentials !== 'omit' && value.credentials !== 'include')) return null;
  const url = validateSourceUrl(value.url);
  if (!url || (value.sourceClass === 'credentialed-fixture' && !loopback(url))
    || (value.credentials === 'include' && value.sourceClass !== 'credentialed-fixture')) return null;
  return Object.freeze({
    tabId: value.tabId, documentId: value.documentId, ownerId: value.ownerId,
    generation: value.generation, url: value.url, protected: value.protected,
    sourceClass: value.sourceClass, credentials: value.credentials,
  });
}

export function parseCommand(raw: unknown, selection: SourceSelection): PlayerCommand | null {
  const source = validateSelection(selection);
  const head = fields(raw, ['type'], false);
  if (!source || source.protected || !head) return null;
  const type = head.type;
  let extra: string[];
  switch (type) {
    case 'play': case 'pause': extra = []; break;
    case 'seek': extra = ['seconds']; break;
    case 'rate': extra = ['rate']; break;
    case 'volume': extra = ['volume']; break;
    case 'mute': extra = ['muted']; break;
    default: return null;
  }
  const value = fields(raw, [...identityKeys, 'requestId', 'type', ...extra]);
  if (!value || value.type !== type || !identityKeys.every(key => value[key] === source[key])
    || typeof value.requestId !== 'string' || !uuid.test(value.requestId)) return null;
  const identity = {
    tabId: source.tabId, documentId: source.documentId, ownerId: source.ownerId,
    generation: source.generation, requestId: value.requestId,
  };
  switch (type) {
    case 'play': case 'pause': return Object.freeze({ ...identity, type });
    case 'seek': return bounded(value.seconds, 0, 86400)
      ? Object.freeze({ ...identity, type, seconds: value.seconds }) : null;
    case 'rate': return bounded(value.rate, 0.25, 4)
      ? Object.freeze({ ...identity, type, rate: value.rate }) : null;
    case 'volume': return bounded(value.volume, 0, 1)
      ? Object.freeze({ ...identity, type, volume: value.volume }) : null;
    case 'mute': return typeof value.muted === 'boolean'
      ? Object.freeze({ ...identity, type, muted: value.muted }) : null;
  }
}

export { parseCommand as validateCommand };

export function resolveRefetch(selection: SourceSelection, claimedOrigin: unknown): URL | null {
  const source = validateSelection(selection);
  if (!source || source.protected) return null;
  const url = validateSourceUrl(source.url);
  return url && claimedOrigin === url.origin ? url : null;
}

export function isRefetchByteCount(value: unknown): value is number {
  return bounded(value, 0, REFETCH_LIMITS.maxBytes) && Number.isInteger(value);
}

export function validateRefetchResponse(selection: SourceSelection, raw: unknown): boolean {
  const source = validateSelection(selection);
  const response = fields(raw, ['url', 'redirected', 'status', 'contentType', 'contentLength', 'bytesRead']);
  if (!source || source.protected || !response || response.url !== source.url
    || response.redirected !== false || response.status !== 200
    || response.contentType !== REFETCH_LIMITS.contentType
    || !isRefetchByteCount(response.bytesRead) || response.bytesRead === 0) return false;
  return response.contentLength === null
    || (isRefetchByteCount(response.contentLength) && response.contentLength === response.bytesRead);
}

type Entry = { selection: SourceSelection; player: PlayerBinding; consumed: boolean };

export class CapabilityRegistry {
  readonly #entries = new Map<string, Entry>();
  readonly #issued = new Set<string>();
  readonly #origin: string;
  readonly #playerUrl: string;

  constructor(
    readonly extensionId: string,
    playerPath: string,
    private readonly tokenFactory: () => string = () => crypto.randomUUID(),
  ) {
    if (!/^[a-p]{32}$/.test(extensionId) || !/^\/[A-Za-z0-9/_-]+\.html$/.test(playerPath)
      || playerPath.includes('//')) throw new TypeError('Invalid player');
    this.#origin = `chrome-extension://${extensionId}`;
    this.#playerUrl = `${this.#origin}${playerPath}`;
  }

  register(raw: unknown, binding: unknown): string | null {
    const selection = validateSelection(raw);
    const player = fields(binding, ['tabId', 'documentId']);
    if (!selection || selection.protected || !player || !tabId(player.tabId)
      || !identifier(player.documentId) || player.tabId === selection.tabId
      || this.#issued.size >= 1024) return null;
    for (const entry of this.#entries.values()) {
      if (entry.selection.tabId === selection.tabId
        && entry.selection.documentId === selection.documentId
        && entry.selection.generation >= selection.generation) return null;
    }
    let token: string;
    try { token = this.tokenFactory(); } catch { return null; }
    if (typeof token !== 'string' || !uuid.test(token) || this.#issued.has(token)) return null;
    this.revokeSource(selection.tabId);
    this.revokePlayer(player.tabId);
    this.#issued.add(token);
    this.#entries.set(token, {
      selection, player: Object.freeze({ tabId: player.tabId, documentId: player.documentId }), consumed: false,
    });
    return token;
  }

  consume(token: unknown, raw: unknown): SourceSelection | null {
    if (typeof token !== 'string' || !uuid.test(token)) return null;
    const entry = this.#entries.get(token);
    const sender = fields(raw, ['id', 'tab', 'documentId', 'origin', 'url', 'frameId'], false);
    const tab = sender && fields(sender.tab, ['id'], false);
    if (!entry || entry.consumed || !sender || !tab || sender.id !== this.extensionId
      || sender.origin !== this.#origin || sender.url !== this.#playerUrl || sender.frameId !== 0
      || sender.documentId !== entry.player.documentId || tab.id !== entry.player.tabId) return null;
    entry.consumed = true;
    return entry.selection;
  }

  isCurrent(selection: SourceSelection): boolean {
    return [...this.#entries.values()].some(entry => entry.selection === selection);
  }

  revokeSource(sourceTabId: number): void {
    for (const [token, entry] of this.#entries) {
      if (entry.selection.tabId === sourceTabId) this.#entries.delete(token);
    }
  }

  revokePlayer(playerTabId: number): void {
    for (const [token, entry] of this.#entries) {
      if (entry.player.tabId === playerTabId) this.#entries.delete(token);
    }
  }
}