/// <reference types="chrome" />
import {
  MODEL_BYTES, MODEL_PATH, MODEL_SHA256, httpOrigin, inactiveStatus, parseExtensionResponse,
  parseModelRequest, parsePopupCommand, parsePreferences, parseRegistration,
  type ContentCommand, type ExtensionResponse, type ModelResponse, type PopupCommand,
  type RuntimeMode, type TabRegistration,
} from './protocol.js';

const PREFERENCES_KEY = 'm10.mode-preferences';
const registrationKey = (tabId: number) => `m10.document.${tabId}`;
const failure = (message: string, code: 'error' | 'unsupported' | 'permission-required' = 'error'): ExtensionResponse =>
  ({ ok: false, code, message });

export function createExtensionWorker(api: typeof chrome) {
  const ready = Promise.all([
    api.storage.local.setAccessLevel({ accessLevel: 'TRUSTED_CONTEXTS' }),
    api.storage.session.setAccessLevel({ accessLevel: 'TRUSTED_CONTEXTS' }),
  ]).then(() => true, () => false);
  const tabOperations = new Map<number, Promise<unknown>>();
  let preferenceOperation: Promise<unknown> = Promise.resolve();
  let model: Promise<Extract<ModelResponse, { ok: true }>> | undefined;

  async function activeOrigin(tabId: number): Promise<string> {
    const [tab, active] = await Promise.all([api.tabs.get(tabId), api.tabs.query({ active: true, currentWindow: true })]);
    const origin = httpOrigin(tab.url);
    if (!tab.active || active.length !== 1 || active[0]?.id !== tabId || !origin ||
      httpOrigin(active[0].url) !== origin || tab.pendingUrl) throw new Error('Tab is not the current accessible HTTP(S) document.');
    return origin;
  }

  async function registration(tabId: number, origin: string): Promise<TabRegistration | null> {
    const stored = await api.storage.session.get(registrationKey(tabId));
    const record = parseRegistration(stored[registrationKey(tabId)]);
    return record?.tabId === tabId && record.origin === origin ? record : null;
  }

  async function modeFor(origin: string): Promise<RuntimeMode> {
    const stored = await api.storage.local.get(PREFERENCES_KEY);
    return parsePreferences(stored[PREFERENCES_KEY]).origins[origin] ?? 'auto';
  }

  function saveMode(origin: string, mode: RuntimeMode): Promise<void> {
    const operation = preferenceOperation.then(async () => {
      const stored = await api.storage.local.get(PREFERENCES_KEY);
      const preferences = parsePreferences(stored[PREFERENCES_KEY]);
      if (!Object.hasOwn(preferences.origins, origin) && Object.keys(preferences.origins).length >= 512) {
        const oldest = Object.keys(preferences.origins)[0];
        if (oldest) delete preferences.origins[oldest];
      }
      preferences.origins[origin] = mode;
      await api.storage.local.set({ [PREFERENCES_KEY]: preferences });
    });
    preferenceOperation = operation.catch(() => undefined);
    return operation;
  }

  async function send(record: TabRegistration, command: ContentCommand): Promise<ExtensionResponse | null> {
    let response: unknown;
    try {
      response = await api.tabs.sendMessage(record.tabId, command, { frameId: 0, documentId: record.documentId });
    } catch { return null; }
    return parseExtensionResponse(response) ?? failure('The content adapter returned an unsupported response.', 'unsupported');
  }

  async function currentDocument(tabId: number, origin: string): Promise<string> {
    const results = await api.scripting.executeScript({
      target: { tabId, frameIds: [0] }, world: 'ISOLATED', func: () => location.origin,
    });
    const top = results.length === 1 ? results[0] : undefined;
    if (!top || top.frameId !== 0 || !top.documentId || top.result !== origin || await activeOrigin(tabId) !== origin) {
      throw new Error('The target document changed.');
    }
    return top.documentId;
  }

  async function handlePopup(command: PopupCommand): Promise<ExtensionResponse> {
    const origin = await activeOrigin(command.tabId);
    const requestedDocument = command.type === 'm10.enable' ? await currentDocument(command.tabId, origin) : null;
    const mode = await modeFor(origin);
    let record = await registration(command.tabId, origin);
    let inspected = record ? await send(record, { type: 'm10.inspect' }) : null;
    if (inspected && !inspected.ok) return inspected;
    if (await activeOrigin(command.tabId) !== origin) return failure('The page changed. Refresh the popup.', 'permission-required');

    if (command.type === 'm10.status') {
      return inspected ?? { ok: true, status: inactiveStatus(mode, 'inactive', 'This page is not activated or inspected. Use Enable to request activation.') };
    }
    if (command.type === 'm10.enable' && inspected?.ok && inspected.status.enabled) return inspected;
    if (command.type === 'm10.enable' && (!record || !inspected)) {
      const documentId = requestedDocument;
      if (documentId === null || await currentDocument(command.tabId, origin) !== documentId) {
        return failure('The page changed after Enable. Activate the new document explicitly.', 'permission-required');
      }
      const results = await api.scripting.executeScript({
        target: { tabId: command.tabId, documentIds: [documentId] }, world: 'ISOLATED', files: ['content.js'],
      });
      const top = results.length === 1 ? results[0] : undefined;
      if (!top || top.frameId !== 0 || top.documentId !== documentId || await activeOrigin(command.tabId) !== origin) {
        return failure('The page changed during activation.', 'permission-required');
      }
      record = { schemaVersion: 1, tabId: command.tabId, documentId, origin };
      await api.storage.session.set({ [registrationKey(command.tabId)]: record });
      inspected = await send(record, { type: 'm10.inspect' });
      if (!inspected?.ok) return inspected ?? failure('The content adapter is unavailable.', 'unsupported');
    }
    if (command.type === 'm10.mode') {
      await saveMode(origin, command.mode);
    }
    if (!record || !inspected) return { ok: true, status: inactiveStatus(command.type === 'm10.mode' ? command.mode : mode) };
    if (await activeOrigin(command.tabId) !== origin) return failure('The page changed before the command.', 'permission-required');
    const contentCommand: ContentCommand = command.type === 'm10.enable' ? { type: 'm10.start', mode }
      : command.type === 'm10.disable' ? { type: 'm10.stop' } : { type: 'm10.set-mode', mode: command.mode };
    const response = await send(record, contentCommand);
    if (await activeOrigin(command.tabId) !== origin) return failure('The page changed while the command was pending.', 'permission-required');
    return response ?? failure('The document is no longer available. Refresh and enable again.', 'permission-required');
  }

  async function modelAuthorized(sender: chrome.runtime.MessageSender): Promise<boolean> {
    if (sender.id !== api.runtime.id || sender.frameId !== 0 || sender.tab?.id === undefined ||
      !Number.isSafeInteger(sender.tab.id) || sender.tab.id < 0 || !sender.documentId ||
      (sender.documentLifecycle !== undefined && sender.documentLifecycle !== 'active')) return false;
    const origin = httpOrigin(sender.url);
    if (!origin || httpOrigin(sender.tab.url) !== origin) return false;
    const record = await registration(sender.tab.id, origin);
    if (!record || record.documentId !== sender.documentId) return false;
    const tab = await api.tabs.get(sender.tab.id);
    if (tab.pendingUrl || httpOrigin(tab.url) !== origin) return false;
    const results = await api.scripting.executeScript({
      target: { tabId: sender.tab.id, frameIds: [0] }, world: 'ISOLATED', func: () => location.origin,
    });
    const top = results.length === 1 ? results[0] : undefined;
    return top?.frameId === 0 && top.documentId === record.documentId && top.result === origin;
  }

  function loadModel(): Promise<Extract<ModelResponse, { ok: true }>> {
    model ??= (async () => {
      const response = await fetch(api.runtime.getURL(MODEL_PATH));
      if (!response.ok) throw new Error('The packaged model is unavailable.');
      const bytes = await response.arrayBuffer();
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const sha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
      if (bytes.byteLength !== MODEL_BYTES || sha256 !== MODEL_SHA256) throw new Error('The packaged model failed integrity validation.');
      return { ok: true, modelJson: new TextDecoder('utf-8', { fatal: true }).decode(bytes), sha256 };
    })();
    return model;
  }

  async function handleMessage(message: unknown, sender: chrome.runtime.MessageSender): Promise<ExtensionResponse | ModelResponse> {
    if (sender.id !== api.runtime.id) return failure('Untrusted message sender.', 'unsupported');
    if (!await ready) return failure('Trusted extension storage is unavailable.');
    try {
      if (parseModelRequest(message)) {
        if (!await modelAuthorized(sender)) return failure('Unauthorized model request.', 'permission-required');
        let response: Extract<ModelResponse, { ok: true }>;
        try { response = await loadModel(); }
        catch { return failure('The packaged model is unavailable or failed integrity validation.'); }
        return await modelAuthorized(sender) ? response : failure('The model request document changed.', 'permission-required');
      }
      const command = parsePopupCommand(message);
      if (!command || sender.tab !== undefined || sender.url !== api.runtime.getURL('popup.html')) {
        return failure('Unsupported popup command or sender.', 'unsupported');
      }
      const previous = tabOperations.get(command.tabId) ?? Promise.resolve();
      const operation = previous.catch(() => undefined).then(() => handlePopup(command));
      tabOperations.set(command.tabId, operation);
      try { return await operation; }
      finally { if (tabOperations.get(command.tabId) === operation) tabOperations.delete(command.tabId); }
    } catch { return failure('Access is unavailable or the tab changed. Reopen the action and try again.', 'permission-required'); }
  }

  return { handleMessage };
}

if (typeof chrome !== 'undefined' && chrome.runtime?.id) {
  const worker = createExtensionWorker(chrome);
  chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
    void worker.handleMessage(message, sender).then(sendResponse);
    return true;
  });
  chrome.tabs.onRemoved.addListener((tabId) => {
    void chrome.storage.session.remove(registrationKey(tabId)).catch(() => undefined);
  });
}