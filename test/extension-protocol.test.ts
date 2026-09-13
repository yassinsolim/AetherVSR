import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createExtensionWorker } from '../src/extension/service-worker.js';
import {
  httpOrigin, inactiveStatus, parseContentCommand, parseExtensionResponse, parseExtensionStatus,
  parseModelRequest, parseModelResponse, parsePopupCommand, parsePreferences, parseRegistration,
  MODEL_PATH, MODEL_SHA256, type ContentCommand,
} from '../src/extension/protocol.js';

describe('extension protocol', () => {
  it('accepts only closed popup and content commands', () => {
    for (const type of ['m10.status', 'm10.enable', 'm10.disable']) {
      expect(parsePopupCommand({ type, tabId: 0 })).toEqual({ type, tabId: 0 });
      expect(parsePopupCommand({ type, tabId: 0, origin: 'https://example.com' })).toBeNull();
    }
    for (const mode of ['auto', 'neural', 'baseline']) {
      expect(parsePopupCommand({ type: 'm10.mode', tabId: 1, mode })).not.toBeNull();
      for (const type of ['m10.start', 'm10.set-mode']) {
        expect(parseContentCommand({ type, mode })).toEqual({ type, mode });
      }
    }
    for (const type of ['m10.inspect', 'm10.stop']) expect(parseContentCommand({ type })).toEqual({ type });
  });

  it('rejects malformed messages and caller-selected URLs', () => {
    for (const value of [null, [], 'm10.enable', 1, Object.create(null), Object.create({ type: 'm10.enable', tabId: 1 }),
      { type: 'm10.enable', tabId: -1 }, { type: 'm10.enable', tabId: 1.2 }, { type: 'm10.enable', tabId: Infinity },
      { type: 'm10.enable', tabId: '1' }, { type: 'm10.mode', tabId: 1, mode: 'fast' },
      { type: 'm10.mode', tabId: 1 }, { type: 'm10.fetch', tabId: 1 },
      JSON.parse('{"type":"m10.enable","tabId":1,"__proto__":{}}') as unknown]) {
      expect(parsePopupCommand(value)).toBeNull();
    }
    expect(parseContentCommand({ type: 'm10.start' })).toBeNull();
    expect(parseContentCommand({ type: 'm10.stop', mode: 'auto' })).toBeNull();
    expect(parseContentCommand({ type: 'm10.start', mode: 'auto', token: 'page' })).toBeNull();
    expect(parseModelRequest({ type: 'm10.model' })).toEqual({ type: 'm10.model' });
    expect(parseModelRequest({ type: 'm10.model', url: 'https://example.com/model' })).toBeNull();
    expect(parseModelResponse({ ok: true, modelJson: '{}', sha256: 'bad' })).toBeNull();
  });

  it('does not invoke getters while validating untrusted input', () => {
    const value = { get type() { throw new Error('getter'); }, tabId: 1 };
    expect(parsePopupCommand(value)).toBeNull();
    expect(parseExtensionStatus({ ...inactiveStatus(), details: { get nested() { throw new Error('getter'); } } })).toBeNull();
  });

  it('validates status, envelopes and bounded optional telemetry', () => {
    const status = { ...inactiveStatus(), details: { timings: [null, 1.5], hardware: 'not measured' } };
    expect(parseExtensionStatus(status)).toEqual(status);
    expect(parseExtensionResponse({ ok: true, status })).toEqual({ ok: true, status });
    expect(parseExtensionResponse({ ok: false, code: 'unsupported', message: 'Unavailable' })).not.toBeNull();
    expect(parseExtensionResponse({ ok: true, status, url: 'https://example.com' })).toBeNull();
    for (const patch of [{ schemaVersion: 2 }, { enabled: 1 }, { code: 'unknown' }, { current: 'auto' },
      { mode: 'fast' }, { candidates: -1 }, { embeddedFrames: 0.5 }, { owner: {} }, { arbitrary: true },
      { details: [] }, { details: { value: NaN } }, { details: { value: undefined } },
      { details: { value: new Date() } }, { details: { values: Array(129).fill(0) as unknown[] } },
      { details: { nested: { nested: { nested: { nested: { nested: { nested: { nested: {} } } } } } } } },
      { details: { value: 'x'.repeat(2049) } }, { details: undefined }]) {
      expect(parseExtensionStatus({ ...status, ...patch })).toBeNull();
    }
    const cyclic: Record<string, unknown> = {};
    cyclic.self = cyclic;
    expect(parseExtensionStatus({ ...status, details: cyclic })).toBeNull();
    const values = [1];
    Object.defineProperty(values, 0, { get() { throw new Error('array getter'); } });
    expect(parseExtensionStatus({ ...status, details: { values } })).toBeNull();
  });

  it('keeps only schema-1 origin mode settings, never activation or page URLs', () => {
    const preferences = { schemaVersion: 1, origins: { 'https://example.com': 'baseline' } };
    expect(parsePreferences(preferences)).toEqual(preferences);
    for (const value of [null, { ...preferences, schemaVersion: 2 }, { ...preferences, enabled: true },
      { schemaVersion: 1, origins: { 'https://example.com/video': 'auto' } },
      { schemaVersion: 1, origins: { 'https://example.com': 'fast' } },
      { schemaVersion: 1, origins: { 'file://': 'auto' } }]) {
      expect(parsePreferences(value)).toEqual({ schemaVersion: 1, origins: {} });
    }
    expect(httpOrigin('https://example.com/watch?v=private')).toBe('https://example.com');
    for (const value of ['file:///movie', 'chrome://extensions', 'https://user:pass@example.com', 'bad', null]) {
      expect(httpOrigin(value)).toBeNull();
    }
  });

  it('binds session registration to a tab, top document and canonical origin', () => {
    const registration = { schemaVersion: 1, tabId: 1, documentId: 'test-document', origin: 'https://example.com' };
    expect(parseRegistration(registration)).toEqual(registration);
    for (const patch of [{ schemaVersion: 2 }, { tabId: -1 }, { documentId: '' }, { documentId: {} },
      { origin: 'https://example.com/path' }, { enabled: true }]) {
      expect(parseRegistration({ ...registration, ...patch })).toBeNull();
    }
  });
});

function workerHarness() {
  const local: Record<string, unknown> = {};
  const session: Record<string, unknown> = {};
  const tab = { id: 7, active: true, url: 'https://example.com/video?private=true' } as
    chrome.tabs.Tab & { id: number; url: string };
  let documentId = 'document-1';
  let listenerDocument: string | null = null;
  let status = inactiveStatus();
  const area = (store: Record<string, unknown>) => ({
    setAccessLevel: vi.fn(() => Promise.resolve()),
    get: vi.fn((key: string) => Promise.resolve({ [key]: store[key] })),
    set: vi.fn((values: Record<string, unknown>) => { Object.assign(store, values); return Promise.resolve(); }),
    remove: vi.fn((key: string) => { delete store[key]; return Promise.resolve(); }),
  });
  const api = {
    runtime: { id: 'extension-id', getURL: (path: string) => `chrome-extension://extension-id/${path}` },
    storage: { local: area(local), session: area(session) },
    tabs: {
      get: vi.fn(() => Promise.resolve({ ...tab })), query: vi.fn(() => Promise.resolve([{ ...tab }])),
      sendMessage: vi.fn((_tabId: number, command: ContentCommand, target: { frameId: number; documentId: string }) => {
        if (target.documentId !== documentId || listenerDocument !== documentId) return Promise.reject(new Error('No receiver'));
        if (command.type === 'm10.start') status = { ...status, mode: command.mode, enabled: true, code: 'discovering' };
        if (command.type === 'm10.stop') status = inactiveStatus(status.mode);
        if (command.type === 'm10.set-mode') status = { ...status, mode: command.mode };
        return Promise.resolve({ ok: true, status });
      }),
    },
    scripting: {
      executeScript: vi.fn((injection: { files?: string[]; target: { documentIds?: string[] } }) => {
        if (injection.target.documentIds && !injection.target.documentIds.includes(documentId)) return Promise.reject(new Error('Stale document'));
        if (injection.files) listenerDocument = documentId;
        return Promise.resolve([{ frameId: 0, documentId, result: injection.files ? undefined : new URL(tab.url).origin }]);
      }),
    },
  };
  const worker = createExtensionWorker(api as unknown as typeof chrome);
  const popup = { id: api.runtime.id, url: api.runtime.getURL('popup.html') };
  const content = () => ({ id: api.runtime.id, tab: { ...tab }, frameId: 0, documentId, url: tab.url,
    documentLifecycle: 'active' as chrome.extensionTypes.DocumentLifecycle });
  const command = (type: 'm10.status' | 'm10.enable' | 'm10.disable') => worker.handleMessage({ type, tabId: tab.id }, popup);
  return { api, tab, local, session, popup, content, worker, command,
    navigate: (url: string) => { tab.url = url; documentId = 'document-2'; listenerDocument = null; },
  };
}

afterEach(() => { vi.unstubAllGlobals(); });

describe('extension worker authorization', () => {
  it('restricts both storage areas and never injects on status, mode or disable', async () => {
    const harness = workerHarness();
    expect(await harness.command('m10.status')).toMatchObject({ ok: true, status: { enabled: false, code: 'inactive' } });
    await harness.worker.handleMessage({ type: 'm10.mode', tabId: 7, mode: 'baseline' }, harness.popup);
    await harness.command('m10.disable');
    expect(harness.api.scripting.executeScript).not.toHaveBeenCalled();
    expect(harness.api.storage.local.setAccessLevel).toHaveBeenCalledWith({ accessLevel: 'TRUSTED_CONTEXTS' });
    expect(harness.api.storage.session.setAccessLevel).toHaveBeenCalledWith({ accessLevel: 'TRUSTED_CONTEXTS' });
    expect(harness.local).toEqual({ 'm10.mode-preferences': { schemaVersion: 1, origins: { 'https://example.com': 'baseline' } } });
    expect(JSON.stringify(harness.local)).not.toContain('private');
  });

  it('injects only on enable and records the document before sending start', async () => {
    const harness = workerHarness();
    await harness.command('m10.enable');
    expect(harness.api.scripting.executeScript).toHaveBeenNthCalledWith(1, {
      target: { tabId: 7, frameIds: [0] }, world: 'ISOLATED', func: expect.any(Function) as unknown,
    });
    expect(harness.api.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 7, documentIds: ['document-1'] }, world: 'ISOLATED', files: ['content.js'],
    });
    expect(harness.session['m10.document.7']).toEqual({ schemaVersion: 1, tabId: 7, documentId: 'document-1', origin: 'https://example.com' });
    const startIndex = harness.api.tabs.sendMessage.mock.calls.findIndex((call) => call[1].type === 'm10.start');
    expect(harness.api.storage.session.set.mock.invocationCallOrder[0]).toBeLessThan(
      harness.api.tabs.sendMessage.mock.invocationCallOrder[startIndex] ?? 0);
    expect(harness.api.tabs.sendMessage).toHaveBeenCalledWith(7, { type: 'm10.start', mode: 'auto' }, { frameId: 0, documentId: 'document-1' });
    await harness.command('m10.enable');
    expect(harness.api.scripting.executeScript.mock.calls.filter(call => call[0].files)).toHaveLength(1);
    expect(harness.api.tabs.sendMessage.mock.calls.filter((call) => call[1].type === 'm10.start')).toHaveLength(1);
  });

  it('serializes concurrent enables and recovers status after worker restart without reinjection', async () => {
    const harness = workerHarness();
    await Promise.all([harness.command('m10.enable'), harness.command('m10.enable')]);
    const restarted = createExtensionWorker(harness.api as unknown as typeof chrome);
    expect(await restarted.handleMessage({ type: 'm10.status', tabId: 7 }, harness.popup)).toMatchObject({ ok: true, status: { enabled: true } });
    expect(harness.api.scripting.executeScript.mock.calls.filter(call => call[0].files)).toHaveLength(1);
    expect(await harness.command('m10.disable')).toMatchObject({ ok: true, status: { enabled: false } });
  });

  it('rejects page, other-extension, subframe and non-exact popup senders', async () => {
    const harness = workerHarness();
    for (const sender of [{}, { ...harness.popup, id: 'other' }, { ...harness.popup, tab: harness.tab },
      { ...harness.popup, url: 'https://example.com/popup.html' },
      { ...harness.popup, url: `${harness.popup.url}?x=1` }, { ...harness.popup, url: `${harness.popup.url}#x` },
      { ...harness.popup, url: harness.api.runtime.getURL('nested/popup.html') }]) {
      expect(await harness.worker.handleMessage({ type: 'm10.enable', tabId: 7 }, sender)).toMatchObject({ ok: false });
    }
    expect(harness.api.tabs.get).not.toHaveBeenCalled();
    expect(harness.api.scripting.executeScript).not.toHaveBeenCalled();
  });

  it('requires the current active HTTP(S) tab even with stored preferences', async () => {
    const harness = workerHarness();
    harness.local['m10.mode-preferences'] = { schemaVersion: 1, origins: { 'https://example.com': 'neural' } };
    harness.tab.active = false;
    expect(await harness.command('m10.enable')).toMatchObject({ ok: false });
    harness.tab.active = true;
    harness.api.tabs.query.mockResolvedValueOnce([{ ...harness.tab, id: 99 }]);
    expect(await harness.command('m10.enable')).toMatchObject({ ok: false });
    harness.tab.url = 'chrome://settings';
    expect(await harness.command('m10.enable')).toMatchObject({ ok: false });
    harness.tab.url = 'https://example.com';
    harness.tab.pendingUrl = 'https://other.example';
    expect(await harness.command('m10.enable')).toMatchObject({ ok: false });
    expect(harness.api.scripting.executeScript).not.toHaveBeenCalled();
  });

  it('fails closed on storage access restriction failure and invalid content responses', async () => {
    const harness = workerHarness();
    harness.api.storage.local.setAccessLevel.mockRejectedValueOnce(new Error('denied'));
    const denied = createExtensionWorker(harness.api as unknown as typeof chrome);
    expect(await denied.handleMessage({ type: 'm10.enable', tabId: 7 }, harness.popup)).toMatchObject({ ok: false });
    expect(harness.api.scripting.executeScript).not.toHaveBeenCalled();
    await harness.command('m10.enable');
    harness.api.tabs.sendMessage.mockResolvedValueOnce({ ok: true, status: { ...inactiveStatus(), schemaVersion: 2 as 1 } });
    expect(await harness.command('m10.status')).toMatchObject({ ok: false, code: 'unsupported' });
  });

  it('does not start or authorize an injection after navigation', async () => {
    const harness = workerHarness();
    harness.api.scripting.executeScript.mockImplementationOnce(() => {
      harness.navigate('https://other.example');
      return Promise.resolve([{ frameId: 0, documentId: 'document-1', result: 'https://example.com' }]);
    });
    expect(await harness.command('m10.enable')).toMatchObject({ ok: false });
    expect(harness.session).toEqual({});
    expect(harness.api.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it('rejects same-origin navigation while the Enable preference read is pending', async () => {
    const harness = workerHarness();
    harness.api.storage.local.get.mockImplementationOnce(() => {
      harness.navigate('https://example.com/replacement');
      return Promise.resolve({});
    });
    expect(await harness.command('m10.enable')).toMatchObject({ ok: false, code: 'permission-required' });
    expect(harness.api.scripting.executeScript.mock.calls.filter(call => call[0].files)).toHaveLength(0);
    expect(harness.session).toEqual({});
  });

  it('does not reuse a document registration on same-origin navigation', async () => {
    const harness = workerHarness();
    await harness.command('m10.enable');
    const staleSender = harness.content();
    harness.navigate('https://example.com/other');
    expect(await harness.worker.handleMessage({ type: 'm10.model' }, staleSender)).toMatchObject({ ok: false });
    expect(await harness.command('m10.status')).toMatchObject({ ok: true, status: { enabled: false } });
    await harness.command('m10.enable');
    expect(harness.session['m10.document.7']).toMatchObject({ documentId: 'document-2' });
    expect(await harness.worker.handleMessage({ type: 'm10.model' }, staleSender)).toMatchObject({ ok: false });
  });

  it('rejects unregistered and mismatched model senders without fetching anything', async () => {
    const harness = workerHarness();
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    expect(await harness.worker.handleMessage({ type: 'm10.model' }, harness.content())).toMatchObject({ ok: false });
    await harness.command('m10.enable');
    for (const sender of [harness.popup, { ...harness.content(), frameId: 1 },
      { ...harness.content(), documentId: 'old' }, { ...harness.content(), url: 'https://other.example' },
      { ...harness.content(), documentLifecycle: 'cached' as chrome.extensionTypes.DocumentLifecycle },
      { ...harness.content(), tab: { ...harness.tab, id: 8 } }]) {
      expect(await harness.worker.handleMessage({ type: 'm10.model' }, sender)).toMatchObject({ ok: false });
    }
    expect(await harness.worker.handleMessage({ type: 'm10.model', url: 'https://remote.example' }, harness.content())).toMatchObject({ ok: false });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('delivers only the exact packaged model and caches it within one worker lifetime', async () => {
    const harness = workerHarness();
    const bytes = readFileSync(new URL('../public/models/aethersr-c16d2.json', import.meta.url));
    const fetchMock = vi.fn(() => Promise.resolve(new Response(bytes)));
    vi.stubGlobal('fetch', fetchMock);
    await harness.command('m10.enable');
    const response = await harness.worker.handleMessage({ type: 'm10.model' }, harness.content());
    expect(response).toEqual({ ok: true, modelJson: bytes.toString('utf8'), sha256: MODEL_SHA256 });
    expect(parseModelResponse(response)).toEqual(response);
    await harness.worker.handleMessage({ type: 'm10.model' }, harness.content());
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith(harness.api.runtime.getURL(MODEL_PATH));
  });

  it('rejects corrupted bytes and navigation during model delivery', async () => {
    const harness = workerHarness();
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response('{}'))));
    await harness.command('m10.enable');
    expect(await harness.worker.handleMessage({ type: 'm10.model' }, harness.content())).toMatchObject({ ok: false, code: 'error' });
    const restarted = createExtensionWorker(harness.api as unknown as typeof chrome);
    vi.stubGlobal('fetch', vi.fn(() => {
      harness.navigate('https://example.com/another-document');
      return Promise.resolve(new Response(readFileSync(new URL('../public/models/aethersr-c16d2.json', import.meta.url))));
    }));
    expect(await restarted.handleMessage({ type: 'm10.model' }, harness.content())).toMatchObject({ ok: false });
  });
});