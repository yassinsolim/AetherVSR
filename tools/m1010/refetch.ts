import { REFETCH_LIMITS, resolveRefetch, validateRefetchResponse, type SourceSelection } from './policy.js';

export async function refetchSelected(selection: SourceSelection, current: () => Promise<boolean>, request: typeof fetch = fetch): Promise<Blob> {
  const url = resolveRefetch(selection, new URL(selection.url).origin);
  if (!url || !await current()) throw new Error('Stale or invalid selected source');
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), REFETCH_LIMITS.timeoutMs);
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  try {
    const response = await request(url.href, { redirect: 'error', credentials: selection.credentials,
      referrerPolicy: 'no-referrer', signal: controller.signal, cache: 'no-store' });
    const header = response.headers.get('content-length');
    const contentLength = header === null ? null : /^\d+$/.test(header) ? Number(header) : NaN;
    if (response.status !== 200 || response.redirected || response.url !== url.href || response.headers.get('content-type')?.split(';')[0] !== 'video/mp4' ||
      contentLength !== null && (!Number.isSafeInteger(contentLength) || contentLength > REFETCH_LIMITS.maxBytes) || !response.body) throw new Error('Unapproved media response');
    reader = response.body.getReader();
    const chunks: ArrayBuffer[] = []; let bytesRead = 0;
    for (;;) {
      const part = await reader.read(); if (part.done) break;
      bytesRead += part.value.byteLength;
      if (bytesRead > REFETCH_LIMITS.maxBytes) throw new Error('Media byte limit');
      chunks.push(new Uint8Array(part.value).buffer);
    }
    if (!validateRefetchResponse(selection, { url: response.url, redirected: response.redirected, status: response.status,
      contentType: response.headers.get('content-type')!.split(';')[0], contentLength, bytesRead }) || !await current()) throw new Error('Media response/source identity changed');
    return new Blob(chunks, { type: 'video/mp4' });
  } finally { clearTimeout(timer); await reader?.cancel().catch(() => {}); controller.abort(); }
}