import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { refetchSelected } from '../tools/m1010/refetch';
import { REFETCH_LIMITS } from '../tools/m1010/policy';
import type { SourceSelection } from '../tools/m1010/policy';

const source: SourceSelection = {
  tabId: 12, documentId: 'source-doc', ownerId: 'video', generation: 1,
  url: 'https://media.example:8443/clip.mp4?asset=one', protected: false,
  sourceClass: 'progressive', credentials: 'omit',
};
const payload = Uint8Array.of(0, 0, 0, 24, 102, 116, 121, 112,
  105, 115, 111, 109, 0, 0, 0, 0, 105, 115, 111, 109, 109, 112, 52, 50);
const current = () => vi.fn<() => Promise<boolean>>().mockResolvedValue(true);

type ResponseOptions = {
  url?: string;
  redirected?: boolean;
  status?: number;
  contentType?: string | null;
  contentLength?: string | null;
  chunks?: Uint8Array[];
  readError?: Error;
  cancelError?: Error;
  open?: boolean;
};

function streamed(options: ResponseOptions = {}) {
  const chunks = options.chunks ?? [payload.subarray(0, 8), payload.subarray(8)];
  let index = 0;
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const pull = vi.fn((sink: ReadableStreamDefaultController<Uint8Array>) => {
    const chunk = chunks[index++];
    if (chunk) sink.enqueue(chunk);
    else if (options.readError) sink.error(options.readError);
    else if (!options.open) sink.close();
  });
  const cancel = vi.fn(() => options.cancelError
    ? Promise.reject(options.cancelError) : Promise.resolve());
  const body = new ReadableStream<Uint8Array>({
    start(sink) { controller = sink; }, pull, cancel,
  }, { highWaterMark: 0 });
  const headers = new Headers();
  if (options.contentType !== null) headers.set('content-type', options.contentType ?? 'video/mp4');
  if (options.contentLength !== null) headers.set('content-length', options.contentLength ?? String(payload.length));
  const response = new Response(body, { status: options.status ?? 200, headers });
  Object.defineProperties(response, {
    url: { value: options.url ?? source.url },
    redirected: { value: options.redirected ?? false },
  });
  const getReader = vi.spyOn(body, 'getReader');
  const readerCancel = vi.spyOn(ReadableStreamDefaultReader.prototype, 'cancel');
  const request = vi.fn<typeof fetch>().mockResolvedValue(response);
  return { response, pull, cancel, getReader, readerCancel, request,
    fail: (error: Error) => controller.error(error) };
}

function cleaned(fixture: ReturnType<typeof streamed>, acquired = true) {
  expect(fixture.request).toHaveBeenCalledTimes(1);
  expect(fixture.request.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
  expect(fixture.readerCancel).toHaveBeenCalledTimes(acquired ? 1 : 0);
  expect(vi.getTimerCount()).toBe(0);
}

beforeEach(() => { vi.useFakeTimers(); });
afterEach(() => {
  try { expect(vi.getTimerCount()).toBe(0); }
  finally { vi.restoreAllMocks(); vi.useRealTimers(); }
});

describe('selected-source refetch with local fetch mocks', () => {
  it('returns a pinned MP4 Blob in stream order', async () => {
    const fixture = streamed({ contentType: 'video/mp4; codecs="avc1"' });
    const identity = current();
    const blob = await refetchSelected(source, identity, fixture.request);
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('video/mp4');
    expect(blob.size).toBe(payload.length);
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(payload);
    expect(fixture.request).toHaveBeenCalledWith(source.url, {
      redirect: 'error', credentials: 'omit', referrerPolicy: 'no-referrer',
      cache: 'no-store', signal: fixture.request.mock.calls[0]?.[1]?.signal,
    });
    expect(identity).toHaveBeenCalledTimes(2);
    expect(fixture.getReader).toHaveBeenCalledTimes(1);
    expect(fixture.pull).toHaveBeenCalledTimes(3);
    cleaned(fixture);
  });

  it.each(['omit', 'include'] as const)('passes explicit fixture cookie mode %s', async credentials => {
    const selection: SourceSelection = { ...source, credentials,
      sourceClass: 'credentialed-fixture', url: 'http://localhost:8000/clip.mp4' };
    const fixture = streamed({ url: selection.url });
    await expect(refetchSelected(selection, current(), fixture.request)).resolves.toBeInstanceOf(Blob);
    expect(fixture.request).toHaveBeenCalledWith(selection.url, expect.objectContaining({
      credentials, redirect: 'error',
    }));
    cleaned(fixture);
  });

  it.each<[string, ResponseOptions]>([
    ['different URL', { url: `${source.url}&changed=1` }],
    ['redirected response', { redirected: true }],
    ['HTML MIME', { contentType: 'text/html' }],
    ['missing MIME', { contentType: null }],
    ['HTTP failure', { status: 404 }],
    ['redirect status', { status: 302 }],
    ['oversized content length', { contentLength: String(REFETCH_LIMITS.maxBytes + 1) }],
    ['negative content length', { contentLength: '-1' }],
    ['fractional content length', { contentLength: '1.5' }],
    ['nonnumeric content length', { contentLength: 'invalid' }],
    ['unsafe integer content length', { contentLength: '9007199254740992' }],
  ])('rejects %s before consumption', async (_name, options) => {
    const fixture = streamed(options);
    const identity = current();
    await expect(refetchSelected(source, identity, fixture.request)).rejects.toThrow('Unapproved media response');
    expect(fixture.getReader).not.toHaveBeenCalled();
    expect(fixture.pull).not.toHaveBeenCalled();
    expect(identity).toHaveBeenCalledTimes(1);
    cleaned(fixture, false);
  });

  it('rejects HTTP 206 before consumption', async () => {
    const fixture = streamed({ status: 206 });
    await expect(refetchSelected(source, current(), fixture.request)).rejects.toThrow();
    expect(fixture.getReader).not.toHaveBeenCalled();
    expect(fixture.pull).not.toHaveBeenCalled();
  });

  it.each([false, true])('cancels an over-limit stream; cancellation rejects: %s', async rejectCancel => {
    const fixture = streamed({ contentLength: null,
      chunks: [new Uint8Array(REFETCH_LIMITS.maxBytes), Uint8Array.of(1), payload],
      ...(rejectCancel ? { cancelError: new Error('cancel failed') } : {}),
    });
    const identity = current();
    await expect(refetchSelected(source, identity, fixture.request)).rejects.toThrow('Media byte limit');
    expect(fixture.pull).toHaveBeenCalledTimes(2);
    expect(fixture.cancel).toHaveBeenCalledTimes(1);
    expect(identity).toHaveBeenCalledTimes(1);
    cleaned(fixture);
  });

  it('accepts a bounded stream without content length', async () => {
    const fixture = streamed({ contentLength: null });
    const blob = await refetchSelected(source, current(), fixture.request);
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(payload);
    cleaned(fixture);
  });

  it.each(['1', '999'])('cleans up mismatched length %s', async contentLength => {
    const fixture = streamed({ contentLength });
    await expect(refetchSelected(source, current(), fixture.request)).rejects.toThrow('Media response/source identity changed');
    expect(fixture.pull).toHaveBeenCalledTimes(3);
    cleaned(fixture);
  });

  it('preserves read errors during cleanup', async () => {
    const readError = new Error('reader failed');
    const fixture = streamed({ readError });
    await expect(refetchSelected(source, current(), fixture.request)).rejects.toBe(readError);
    expect(fixture.pull).toHaveBeenCalledTimes(3);
    cleaned(fixture);
  });

  it('does not fetch a source whose identity is already stale', async () => {
    const fixture = streamed();
    const identity = current().mockResolvedValue(false);
    await expect(refetchSelected(source, identity, fixture.request)).rejects.toThrow('Stale or invalid selected source');
    expect(identity).toHaveBeenCalledTimes(1);
    expect(fixture.request).not.toHaveBeenCalled();
    expect(fixture.getReader).not.toHaveBeenCalled();
    expect(fixture.pull).not.toHaveBeenCalled();
  });

  it('discards bytes when source identity becomes stale during fetch', async () => {
    const fixture = streamed();
    const identity = current();
    fixture.request.mockImplementation(() => {
      identity.mockResolvedValue(false);
      return Promise.resolve(fixture.response);
    });
    await expect(refetchSelected(source, identity, fixture.request)).rejects.toThrow('Media response/source identity changed');
    expect(identity).toHaveBeenCalledTimes(2);
    expect(fixture.pull).toHaveBeenCalledTimes(3);
    cleaned(fixture);
  });

  it.each(['fetch', 'reader'] as const)('aborts a pending %s at the timeout and cleans up', async phase => {
    const fixture = streamed({ chunks: [], contentLength: null, open: true });
    const abortError = new DOMException('Timed out', 'AbortError');
    fixture.request.mockImplementation((_input, init) => {
      expect(init?.signal?.aborted).toBe(false);
      if (phase === 'reader') {
        init?.signal?.addEventListener('abort', () => fixture.fail(abortError), { once: true });
        return Promise.resolve(fixture.response);
      }
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(abortError), { once: true });
      });
    });
    const identity = current();
    const rejection = expect(refetchSelected(source, identity, fixture.request)).rejects.toBe(abortError);
    await vi.advanceTimersByTimeAsync(REFETCH_LIMITS.timeoutMs - 1);
    expect(fixture.request).toHaveBeenCalledTimes(1);
    expect(fixture.request.mock.calls[0]?.[1]?.signal?.aborted).toBe(false);
    expect(vi.getTimerCount()).toBe(1);
    expect(fixture.getReader).toHaveBeenCalledTimes(phase === 'reader' ? 1 : 0);
    await vi.advanceTimersByTimeAsync(1);
    await rejection;
    expect(identity).toHaveBeenCalledTimes(1);
    cleaned(fixture, phase === 'reader');
  });
});