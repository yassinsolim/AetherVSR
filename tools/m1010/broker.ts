import { parseCommand, permissionPatternFor, validateSelection, validateSourceUrl, type SourceSelection } from './policy.js';

type SourceSnapshot = { ownerId: string; generation: number; url: string; protected: boolean; sourceClass: string;
  paused: boolean; muted: boolean; volume: number; currentTime: number; width: number; height: number };
type Session = { sourceTabId: number; sourceDocumentId: string; ownerId: string; generation: number;
  playerDocumentId: string | null; snapshot: SourceSnapshot; consumedRefetch: boolean; openingNavigation: boolean };
type Sender = chrome.runtime.MessageSender & { documentId?: string; origin?: string };
const sessions = new Map<number, Session>();
const sourceIds = new Set<number>();
const trace: { event: string; tabId: number | null; detail: string }[] = [];
const record = (event: string, tabId: number | null, detail: string) => {
  trace.push({ event, tabId, detail });
  if (trace.length > 128) trace.shift();
};
Object.assign(globalThis, { __M1010_BROKER_TRACE__: () => trace.map(entry => ({ ...entry })) });
const sourceOperation = async (session: Session, operation: 'snapshot' | 'captureOffer' | 'acceptAnswer' | 'command' | 'stop' | 'releaseCapture', value: unknown = null) => {
  const target = { tabId: session.sourceTabId, documentIds: [session.sourceDocumentId] } as chrome.scripting.InjectionTarget;
  const results = await chrome.scripting.executeScript({ target,
    func: async (operation, value, owner, generation) => {
      const agent = (globalThis as unknown as { __M1010_AGENT__?: Record<string, (value?: unknown) => unknown> }).__M1010_AGENT__;
      if (!agent) throw new Error('Selected source agent unavailable');
      if (typeof operation !== 'string' || !['snapshot', 'captureOffer', 'acceptAnswer', 'command', 'stop', 'releaseCapture'].includes(operation)) throw new Error('Rejected source operation');
      const current = agent['read']!() as { ownerId: string; generation: number };
      if (current.ownerId !== owner || current.generation !== generation) throw new Error('Source identity changed before operation');
      return operation === 'snapshot' ? current : await agent[operation]!(value);
    }, args: [operation, value, session.ownerId, session.generation] });
  const entry = results[0] as (chrome.scripting.InjectionResult & { documentId?: string }) | undefined;
  if (!entry || entry.documentId !== session.sourceDocumentId) throw new Error('Source document changed');
  return entry.result as unknown;
};

async function current(session: Session): Promise<boolean> {
  try {
    if (![...sessions.values()].includes(session)) return false;
    const snapshot = await sourceOperation(session, 'snapshot') as SourceSnapshot;
    return [...sessions.values()].includes(session) && snapshot.ownerId === session.ownerId && snapshot.generation === session.generation &&
      snapshot.url === session.snapshot.url && !snapshot.protected;
  } catch { return false; }
}

function fixtureSelection(session: Session): SourceSelection | null {
  const url = validateSourceUrl(session.snapshot.url);
  if (!url || !['http://127.0.0.1:5204', 'http://127.0.0.1:5205'].includes(url.origin) ||
    !/^\/(?:(?:same|cors|nocors|auth\/(?:omit|include))\/[ABC]|redirect-(?:same|ungranted))\.mp4$/.test(url.pathname) || url.search) return null;
  return validateSelection({ tabId: session.sourceTabId, documentId: session.sourceDocumentId,
    ownerId: session.ownerId, generation: session.generation, url: url.href, protected: session.snapshot.protected,
    sourceClass: url.pathname.startsWith('/auth/') ? 'credentialed-fixture' : 'progressive',
    credentials: url.pathname.startsWith('/auth/include/') ? 'include' : 'omit' });
}

function playerSession(sender: Sender, navigationType: unknown): Session | null {
  if (sender.id !== chrome.runtime.id || sender.url !== chrome.runtime.getURL('acquire.html') ||
    sender.origin !== `chrome-extension://${chrome.runtime.id}` || sender.frameId !== 0 ||
    sender.tab?.id === undefined || !sender.documentId) return null;
  const session = sessions.get(sender.tab.id);
  if (!session) { record('rejected', sender.tab.id, 'no-session'); return null; }
  if (session.playerDocumentId && session.playerDocumentId !== sender.documentId) return null;
  if (!session.playerDocumentId) {
    if (navigationType !== 'navigate') { record('rejected', sender.tab.id, 'not-initial-navigation'); return null; }
    record('bound', sender.tab.id, 'first-document');
    session.openingNavigation = false;
  }
  session.playerDocumentId ??= sender.documentId;
  return session;
}

function streamId(targetTabId: number, consumerTabId?: number): Promise<string> {
  return new Promise((resolve, reject) => chrome.tabCapture.getMediaStreamId({ targetTabId, ...(consumerTabId === undefined ? {} : { consumerTabId }) }, id => {
    const error = chrome.runtime.lastError; if (error || !id) reject(new Error(error?.message ?? 'No capture ID')); else resolve(id);
  }));
}

async function selectFixture(tab: chrome.tabs.Tab | undefined) {
  if (tab?.id === undefined || !tab.url?.startsWith('http://127.0.0.1:5204/')) throw new Error('Select the local authoritative fixture first');
  await chrome.scripting.executeScript({ target: { tabId: tab.id, frameIds: [0] }, files: ['source-agent.js'] });
  const selected = await chrome.scripting.executeScript({ target: { tabId: tab.id, frameIds: [0] }, func: () =>
    (globalThis as unknown as { __M1010_AGENT__: { select(): SourceSnapshot } }).__M1010_AGENT__.select() });
  const entry = selected[0] as (chrome.scripting.InjectionResult & { documentId?: string }) | undefined;
  const snapshot = entry?.result as SourceSnapshot | undefined;
  if (!entry?.documentId || !snapshot?.ownerId || snapshot.protected) throw new Error('No usable unprotected source');
  const player = await chrome.tabs.create({ url: 'about:blank', active: false });
  if (player.id === undefined) throw new Error('No acquisition player');
  sessions.set(player.id, { sourceTabId: tab.id, sourceDocumentId: entry.documentId, ownerId: snapshot.ownerId,
    generation: snapshot.generation, playerDocumentId: null, snapshot, consumedRefetch: false, openingNavigation: true });
  record('registered', player.id, 'awaiting-navigation');
  sourceIds.add(tab.id);
  await chrome.tabs.update(player.id, { url: chrome.runtime.getURL('acquire.html'), active: true });
  return { ok: true, playerTabId: player.id };
}

Object.assign(globalThis, { __M1010_RESEARCH_SELECT__: async (tabId: number) => {
  if (!Number.isSafeInteger(tabId) || tabId < 0 || !await chrome.permissions.contains({ origins: ['http://127.0.0.1/*'] })) throw new Error('Existing local host grant required for research reselection');
  return selectFixture(await chrome.tabs.get(tabId));
} });

chrome.runtime.onMessage.addListener((raw: unknown, sender: Sender, respond) => {
  if (!raw || typeof raw !== 'object' || !('type' in raw) || typeof raw.type !== 'string') return false;
  const message = raw as Record<string, unknown>;
  if (message['type'] === 'research.acquire' && sender.id === chrome.runtime.id && sender.url === chrome.runtime.getURL('launcher.html')) {
    void chrome.tabs.query({ active: true, currentWindow: true }).then(tabs => selectFixture(tabs[0]))
      .then(respond, error => respond({ ok: false, error: String(error) }));
    return true;
  }
  const session = playerSession(sender, message['navigationType']); if (!session) return false;
  const playerTabId = sender.tab!.id!;
  void (async () => {
    if (!await current(session)) throw new Error('Source owner or generation changed');
    switch (message['type']) {
      case 'acquire.info': {
        const selection = fixtureSelection(session);
        return { source: session.snapshot, selection, sourceTabId: session.sourceTabId, playerTabId,
          playerDocumentId: session.playerDocumentId, permission: selection ? permissionPatternFor(new URL(selection.url)) : null };
      }
      case 'acquire.current': return { current: true };
      case 'acquire.refetch': {
        if (session.consumedRefetch) throw new Error('Refetch capability already consumed');
        const selection = fixtureSelection(session);
        const origin = selection && permissionPatternFor(new URL(selection.url));
        if (!selection || !origin || !await chrome.permissions.contains({ origins: [origin] }) || !await current(session)) throw new Error('No current selected-origin host grant');
        if (session.consumedRefetch) throw new Error('Refetch capability already consumed');
        session.consumedRefetch = true; return { selection };
      }
      case 'acquire.offer': return sourceOperation(session, 'captureOffer');
      case 'acquire.answer': {
        const answer = message['answer'];
        if (!answer || typeof answer !== 'object' || !('sdp' in answer) || typeof answer.sdp !== 'string' || answer.sdp.length > 262144) throw new Error('Invalid signaling');
        return sourceOperation(session, 'acceptAnswer', { ownerId: session.ownerId, generation: session.generation, type: 'answer', sdp: answer.sdp });
      }
      case 'acquire.command': {
        const selection = fixtureSelection(session);
        if (!selection) throw new Error('Command source requires explicit mirror identity policy');
        const candidate = message['command'];
        if (!candidate || typeof candidate !== 'object') throw new Error('Invalid command');
        const command = parseCommand({ ...candidate, tabId: selection.tabId, documentId: selection.documentId,
          ownerId: selection.ownerId, generation: selection.generation }, selection);
        if (!command) throw new Error('Rejected command');
        const value = command.type === 'seek' ? command.seconds : command.type === 'rate' ? command.rate : command.type === 'volume' ? command.volume : command.type === 'mute' ? command.muted : undefined;
        return sourceOperation(session, 'command', { ownerId: session.ownerId, generation: session.generation,
          requestId: command.requestId, type: command.type, ...(value === undefined ? {} : { value }) });
      }
      case 'acquire.tab-id': {
        if (!await chrome.permissions.contains({ permissions: ['tabCapture'] })) throw new Error('tabCapture not granted');
        if (!await current(session)) throw new Error('Source revoked before capture');
        const id = await streamId(session.sourceTabId, message['consumer'] === false ? undefined : playerTabId);
        if (!await current(session)) throw new Error('Source revoked during capture-ID issuance');
        return { id, sourceTabId: session.sourceTabId };
      }
      case 'acquire.stop': await sourceOperation(session, 'releaseCapture'); return { stopped: true };
      default: throw new Error('Unrecognized acquisition request');
    }
  })().then(async value => {
    if (!await current(session)) { respond({ ok: false, error: 'Source revoked before response' }); return; }
    respond({ ok: true, value });
  }, error => respond({ ok: false, error: String(error) }));
  return true;
});

const revoke = (tabId: number) => {
  for (const [player, session] of sessions) {
    if (session.sourceTabId !== tabId && player !== tabId) continue;
    record('revoked', player, session.playerDocumentId ? 'bound-document' : 'before-handshake');
    sessions.delete(player); sourceIds.delete(session.sourceTabId);
    void sourceOperation(session, 'stop').catch(() => {});
    void chrome.runtime.sendMessage({ type: 'acquire.revoked', playerTabId: player,
      playerDocumentId: session.playerDocumentId, ownerId: session.ownerId }).catch(() => {});
  }
};
chrome.tabs.onRemoved.addListener(revoke);
chrome.tabs.onUpdated.addListener((tabId, changes) => {
  record('updated', tabId, `${changes.status ?? 'no-status'}:${changes.url === undefined ? 'no-url' : changes.url === chrome.runtime.getURL('acquire.html') ? 'acquire' : changes.url === 'about:blank' ? 'blank' : 'other'}`);
  const session = sessions.get(tabId);
  if (session?.openingNavigation) {
    if (changes.url !== undefined && changes.url !== chrome.runtime.getURL('acquire.html') && changes.url !== 'about:blank') { revoke(tabId); return; }
    if (changes.status === 'complete') session.openingNavigation = false;
    if (!sourceIds.has(tabId)) return;
  }
  if (changes.status !== 'loading') return;
  if (sourceIds.has(tabId) || session) revoke(tabId);
});
chrome.permissions.onRemoved.addListener(() => { for (const source of [...sourceIds]) revoke(source); });