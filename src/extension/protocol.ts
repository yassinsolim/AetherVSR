import type { RuntimeMode } from '../core/upscale/runtime-controller.js';

export type { RuntimeMode };
export const MODES = ['auto', 'neural', 'baseline'] as const;
export const MODEL_SHA256 = 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a';
export const MODEL_BYTES = 140467;
export const MODEL_PATH = 'models/production.json';
export const STATUS_CODES = [
  'inactive', 'permission-required', 'discovering', 'starting', 'active', 'suspended',
  'no-video', 'unsupported', 'unsupported-page', 'unsupported-geometry',
  'unsupported-media', 'unsupported-controls', 'unsupported-frame', 'protected-media',
  'cors-blocked', 'webgpu-unavailable', 'device-lost', 'error',
] as const;
export type StatusCode = typeof STATUS_CODES[number];

export interface ExtensionStatus {
  schemaVersion: 1;
  enabled: boolean;
  code: StatusCode;
  message: string;
  mode: RuntimeMode;
  current: 'neural' | 'baseline' | null;
  candidates: number;
  embeddedFrames: number;
  owner: string | null;
  details?: Record<string, unknown>;
}

export type PopupCommand =
  | { type: 'm10.status'; tabId: number }
  | { type: 'm10.enable'; tabId: number }
  | { type: 'm10.disable'; tabId: number }
  | { type: 'm10.mode'; tabId: number; mode: RuntimeMode };
export type ContentCommand =
  | { type: 'm10.inspect' | 'm10.stop' }
  | { type: 'm10.start' | 'm10.set-mode'; mode: RuntimeMode };
export type ExtensionResponse =
  | { ok: true; status: ExtensionStatus }
  | { ok: false; code: StatusCode; message: string };
export type ModelRequest = { type: 'm10.model' };
export type ModelResponse =
  | { ok: true; modelJson: string; sha256: string }
  | { ok: false; code: StatusCode; message: string };
export interface ModePreferences { schemaVersion: 1; origins: Record<string, RuntimeMode> }
export interface TabRegistration { schemaVersion: 1; tabId: number; documentId: string; origin: string }

export function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) return false;
  return Reflect.ownKeys(value).every((key) => {
    if (typeof key !== 'string' || ['__proto__', 'constructor', 'prototype'].includes(key)) return false;
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor?.enumerable === true && 'value' in descriptor;
  });
}

function fields(value: Record<string, unknown>, required: string[], optional: string[] = []): boolean {
  return required.every((key) => Object.hasOwn(value, key)) &&
    Object.keys(value).every((key) => required.includes(key) || optional.includes(key));
}

export function isRuntimeMode(value: unknown): value is RuntimeMode {
  return typeof value === 'string' && MODES.some((mode) => mode === value);
}

function isStatusCode(value: unknown): value is StatusCode {
  return typeof value === 'string' && STATUS_CODES.some((code) => code === value);
}

function count(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

export function httpOrigin(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  try {
    const url = new URL(value);
    return (url.protocol === 'http:' || url.protocol === 'https:') && !url.username && !url.password ? url.origin : null;
  } catch { return null; }
}

export function parsePopupCommand(value: unknown): PopupCommand | null {
  if (!isPlainRecord(value) || !count(value.tabId)) return null;
  if (value.type === 'm10.mode') {
    return fields(value, ['type', 'tabId', 'mode']) && isRuntimeMode(value.mode) ? value as PopupCommand : null;
  }
  return typeof value.type === 'string' && ['m10.status', 'm10.enable', 'm10.disable'].includes(value.type) &&
    fields(value, ['type', 'tabId']) ? value as PopupCommand : null;
}

export function parseContentCommand(value: unknown): ContentCommand | null {
  if (!isPlainRecord(value)) return null;
  if (value.type === 'm10.start' || value.type === 'm10.set-mode') {
    return fields(value, ['type', 'mode']) && isRuntimeMode(value.mode) ? value as ContentCommand : null;
  }
  return (value.type === 'm10.inspect' || value.type === 'm10.stop') && fields(value, ['type'])
    ? value as ContentCommand : null;
}

export function parseModelRequest(value: unknown): ModelRequest | null {
  return isPlainRecord(value) && fields(value, ['type']) && value.type === 'm10.model' ? { type: 'm10.model' } : null;
}

function boundedJson(value: unknown, depth: number, budget: { left: number }): boolean {
  if (--budget.left < 0 || depth > 6) return false;
  if (value === null || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value === 'string') return value.length <= 2048;
  if (Array.isArray(value)) {
    if (Object.getPrototypeOf(value) !== Array.prototype || value.length > 128 ||
      Reflect.ownKeys(value).length !== value.length + 1) return false;
    return Array.from({ length: value.length }, (_, index) => {
      const descriptor = Object.getOwnPropertyDescriptor(value, index);
      return descriptor && 'value' in descriptor && boundedJson(descriptor.value, depth + 1, budget);
    }).every(Boolean);
  }
  return isPlainRecord(value) && Object.keys(value).length <= 128 &&
    Object.values(value).every((item) => boundedJson(item, depth + 1, budget));
}

export function parseExtensionStatus(value: unknown): ExtensionStatus | null {
  if (!isPlainRecord(value) || !fields(value,
    ['schemaVersion', 'enabled', 'code', 'message', 'mode', 'current', 'candidates', 'embeddedFrames', 'owner'], ['details'])) return null;
  if (value.schemaVersion !== 1 || typeof value.enabled !== 'boolean' || !isStatusCode(value.code) ||
    typeof value.message !== 'string' || value.message.length > 2048 || !isRuntimeMode(value.mode) ||
    (value.current !== null && value.current !== 'neural' && value.current !== 'baseline') ||
    !count(value.candidates) || !count(value.embeddedFrames) ||
    !(value.owner === null || (typeof value.owner === 'string' && value.owner.length <= 256))) return null;
  if (Object.hasOwn(value, 'details') && (!isPlainRecord(value.details) ||
    !boundedJson(value.details, 0, { left: 512 }) || JSON.stringify(value.details).length > 16384)) return null;
  return value as unknown as ExtensionStatus;
}

function isFailure(value: Record<string, unknown>): value is Extract<ExtensionResponse, { ok: false }> {
  return value.ok === false && fields(value, ['ok', 'code', 'message']) && isStatusCode(value.code) &&
    typeof value.message === 'string' && value.message.length <= 2048;
}

export function parseExtensionResponse(value: unknown): ExtensionResponse | null {
  if (!isPlainRecord(value)) return null;
  if (isFailure(value)) return value;
  if (value.ok !== true || !fields(value, ['ok', 'status'])) return null;
  const status = parseExtensionStatus(value.status);
  return status ? { ok: true, status } : null;
}

export function parseModelResponse(value: unknown): ModelResponse | null {
  if (!isPlainRecord(value)) return null;
  if (isFailure(value)) return value;
  return value.ok === true && fields(value, ['ok', 'modelJson', 'sha256']) &&
    typeof value.modelJson === 'string' && new TextEncoder().encode(value.modelJson).length === MODEL_BYTES &&
    value.sha256 === MODEL_SHA256 ? value as Extract<ModelResponse, { ok: true }> : null;
}

export function parsePreferences(value: unknown): ModePreferences {
  const empty: ModePreferences = { schemaVersion: 1, origins: {} };
  if (!isPlainRecord(value) || !fields(value, ['schemaVersion', 'origins']) || value.schemaVersion !== 1 ||
    !isPlainRecord(value.origins) || Object.keys(value.origins).length > 512) return empty;
  if (!Object.entries(value.origins).every(([origin, mode]) => httpOrigin(origin) === origin && isRuntimeMode(mode))) return empty;
  return { schemaVersion: 1, origins: { ...value.origins } as Record<string, RuntimeMode> };
}

export function parseRegistration(value: unknown): TabRegistration | null {
  return isPlainRecord(value) && fields(value, ['schemaVersion', 'tabId', 'documentId', 'origin']) &&
    value.schemaVersion === 1 && count(value.tabId) && typeof value.documentId === 'string' &&
    /^[a-zA-Z0-9-]{1,128}$/.test(value.documentId) && typeof value.origin === 'string' &&
    httpOrigin(value.origin) === value.origin ? value as unknown as TabRegistration : null;
}

export function inactiveStatus(mode: RuntimeMode = 'auto', code: StatusCode = 'inactive',
  message = 'Enhancement is inactive.'): ExtensionStatus {
  return { schemaVersion: 1, enabled: false, code, message, mode, current: null, candidates: 0, embeddedFrames: 0, owner: null };
}