/// <reference types="chrome" />
import {
  httpOrigin, inactiveStatus, isPlainRecord, isRuntimeMode, parseExtensionResponse,
  type ExtensionStatus, type PopupCommand, type RuntimeMode,
} from './protocol.js';

function element<ElementType extends HTMLElement>(id: string): ElementType {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing popup element: ${id}`);
  return found as ElementType;
}

const modes = element<HTMLFieldSetElement>('modes');
const enable = element<HTMLButtonElement>('enable');
const disable = element<HTMLButtonElement>('disable');
const refresh = element<HTMLButtonElement>('refresh');
const radios = [...document.querySelectorAll<HTMLInputElement>('input[name="mode"]')];
let tabId: number | null = null;
let busy = false;
let status = inactiveStatus();

function controls(): void {
  modes.disabled = busy || tabId === null;
  enable.disabled = busy || tabId === null || status.enabled;
  disable.disabled = busy || tabId === null || !status.enabled;
  refresh.disabled = busy;
  document.body.setAttribute('aria-busy', String(busy));
}

function render(next: ExtensionStatus): void {
  status = next;
  element('state').textContent = next.code.replaceAll('-', ' ');
  element('message').textContent = next.message;
  element('current').textContent = next.current ?? 'Not active';
  const inspected = next.enabled;
  element('candidates').textContent = inspected ? String(next.candidates) : 'Not inspected';
  element('frames').textContent = inspected ? `${next.embeddedFrames} uninspected` : 'Not inspected';
  for (const radio of radios) radio.checked = radio.value === next.mode;
  controls();
}

async function send(command: PopupCommand): Promise<void> {
  const raw: unknown = await chrome.runtime.sendMessage(command);
  const response = parseExtensionResponse(raw);
  if (!response) {
    render(inactiveStatus(status.mode, 'unsupported', 'The worker returned an unsupported response.'));
  } else if (response.ok) render(response.status);
  else render({ ...status, code: response.code, message: response.message });
}

async function perform(type: 'm10.status' | 'm10.enable' | 'm10.disable' | 'm10.mode', mode?: RuntimeMode): Promise<void> {
  if (busy) return;
  busy = true;
  controls();
  try {
    if (type === 'm10.status') {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      const tab = tabs.length === 1 ? tabs[0] : undefined;
      const origin = httpOrigin(tab?.url);
      tabId = origin && tab?.id !== undefined && Number.isSafeInteger(tab.id) && tab.id >= 0 ? tab.id : null;
      element('origin').textContent = origin ?? 'Unsupported page';
      if (tabId === null) {
        render(inactiveStatus('auto', 'unsupported-page', 'Open an HTTP(S) page, then open the AetherVSR action.'));
        return;
      }
    }
    if (tabId === null) return;
    if (type === 'm10.mode') {
      if (mode) await send({ type, tabId, mode });
    } else await send({ type, tabId });
  } catch {
    render({ ...status, code: 'error', message: 'The extension worker is unavailable or access changed. Reopen the action and refresh.' });
  } finally { busy = false; controls(); }
}

async function showBuild(): Promise<void> {
  element('version').textContent = `v${chrome.runtime.getManifest().version}`;
  try {
    const response = await fetch(chrome.runtime.getURL('build-provenance.json'));
    if (!response.ok) return;
    const value: unknown = await response.json();
    if (!isPlainRecord(value) || value.schemaVersion !== 1 || typeof value.sourceCommit !== 'string' ||
      !/^[0-9a-f]{40,64}$/.test(value.sourceCommit) || typeof value.sourceDirty !== 'boolean' ||
      (value.buildKind !== 'production' && value.buildKind !== 'test')) return;
    element('source').textContent = `${value.buildKind === 'test' ? 'TEST BUILD - not production' : 'Production build'} | ` +
      `source ${value.sourceCommit.slice(0, 12)}${value.sourceDirty ? ' (dirty)' : ''}`;
  } catch { return; }
}

enable.addEventListener('click', () => { void perform('m10.enable'); });
disable.addEventListener('click', () => { void perform('m10.disable'); });
refresh.addEventListener('click', () => { void perform('m10.status'); });
for (const radio of radios) {
  radio.addEventListener('change', () => {
    if (radio.checked && isRuntimeMode(radio.value)) void perform('m10.mode', radio.value);
  });
}
void showBuild();
void perform('m10.status');