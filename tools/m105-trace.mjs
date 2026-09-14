import assert from 'node:assert/strict';
import { writeFileSync, existsSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { sha256 } from './m10-fixtures.mjs';
import { distribution } from './m10-performance.mjs';

export const TRACE_CONFIG = { transferMode: 'ReturnAsStream', streamFormat: 'json', streamCompression: 'gzip',
  bufferUsageReportingInterval: 1000, traceConfig: { recordMode: 'recordUntilFull', traceBufferSizeInKb: 32768,
    includedCategories: ['media', 'blink'], excludedCategories: ['*'], enableSampling: false, enableSystrace: false, enableArgumentFilter: false } };

export function summarizeTrace(trace) {
  assert(Array.isArray(trace.traceEvents), 'Chrome traceEvents required');
  const names = ['VideoFrameCompositor::SetCurrentFrame', 'VideoRendererImpl::Render', 'VideoFramesDropped',
    'VideoFrameCallbackRequesterImpl::ExecuteVideoFrameCallbacks', 'VideoFrameCallbackRequesterImpl::OnRenderingSteps',
    'VideoFrameCallbackRequesterImpl::ScheduleExecution'];
  const groups = {};
  const stacks = new Map();
  for (const event of trace.traceEvents) {
    const key = `${event.pid}:${event.tid}`;
    if (event.ph === 'B') {
      const stack = stacks.get(key) ?? []; stack.push(event); stacks.set(key, stack);
      continue;
    }
    let value = event;
    if (event.ph === 'E') {
      const begin = stacks.get(key)?.pop();
      if (!begin || (event.name && event.name !== begin.name)) continue;
      value = { ...begin, ph: 'X', dur: event.ts - begin.ts, args: { ...begin.args, ...event.args } };
    }
    if (!names.includes(value.name) || value.ph === 'B') continue;
    const groupKey = `${value.pid}:${value.tid}:${value.name}`;
    const group = groups[groupKey] ??= { name: value.name, pid: value.pid, tid: value.tid, times: [], durations: [], dropBatchCount: 0, playerIds: [] };
    if (Number.isFinite(value.ts)) group.times.push(value.ts);
    if (Number.isFinite(value.dur) && value.dur >= 0) group.durations.push(value.dur / 1000);
    if (value.name === 'VideoFramesDropped') {
      assert(Number.isFinite(value.args?.count) && value.args.count >= 0, 'Unknown drop-batch trace shape');
      group.dropBatchCount += value.args.count;
      if (!group.playerIds.includes(value.args.id)) group.playerIds.push(value.args.id);
    }
  }
  return Object.values(groups).map(group => {
    group.times.sort((left, right) => left - right);
    return { name: group.name, pid: group.pid, tid: group.tid, events: group.times.length,
      firstUs: group.times[0] ?? null, lastUs: group.times.at(-1) ?? null,
      intervalsMs: distribution(group.times.slice(1).map((time, index) => (time - group.times[index]) / 1000)),
      durationMs: distribution(group.durations), dropBatchCount: group.name === 'VideoFramesDropped' ? group.dropBatchCount : null,
      playerIds: group.playerIds };
  });
}

export async function prepareTrace(native, page, path) {
  assert(!existsSync(path), 'Never overwrite trace');
  const browser = await native.browser.newBrowserCDPSession();
  const session = await native.context.newCDPSession(page);
  const record = { path, config: TRACE_CONFIG, requestedDurationMs: 12000, verdict: 'UNVERIFIED', bufferUsage: [],
    scope: 'Separate diagnostic Chrome trace. Event counts/timing correlation, not GPU execution, physical display or exact dropped/callback-missed frame intersection. Drop batches may straddle trace boundaries.' };
  let resolveDone, rejectDone, started = false, timer, deadline;
  const done = new Promise((resolve, reject) => { resolveDone = resolve; rejectDone = reject; });
  done.catch(() => {});
  const finish = async event => {
    clearTimeout(timer); clearTimeout(deadline);
    try {
      record.dataLossOccurred = event.dataLossOccurred ?? null;
      assert(event.stream, 'Trace stream missing');
      const chunks = []; let total = 0;
      try {
        for (;;) {
          const chunk = await browser.send('IO.read', { handle: event.stream, size: 1048576 });
          const bytes = Buffer.from(chunk.data, chunk.base64Encoded ? 'base64' : 'utf8'); total += bytes.length;
          assert(total <= 64 * 1024 * 1024, 'Trace exceeds local byte bound'); chunks.push(bytes);
          if (chunk.eof) break;
        }
      } finally { await browser.send('IO.close', { handle: event.stream }); }
      const bytes = Buffer.concat(chunks); writeFileSync(path, bytes, { flag: 'wx' });
      record.bytes = bytes.length; record.sha256 = sha256(bytes);
      record.events = summarizeTrace(JSON.parse(gunzipSync(bytes)));
      assert.equal(record.dataLossOccurred, false, 'Incomplete trace');
      assert(record.bufferUsage.every(value => value.percentFull === undefined || value.percentFull < 1), 'Trace buffer saturated');
      record.verdict = 'VALID'; resolveDone(record);
    } catch (error) { record.error = String(error); resolveDone(record); }
    finally { await session.detach().catch(() => {}); await browser.detach().catch(() => {}); }
  };
  browser.on('Tracing.bufferUsage', event => record.bufferUsage.push(event));
  browser.once('Tracing.tracingComplete', event => { void finish(event); });
  await session.send('Runtime.enable'); await session.send('Runtime.addBinding', { name: 'm105TraceStart' });
  session.on('Runtime.bindingCalled', event => {
    if (event.name !== 'm105TraceStart' || started) return;
    started = true; record.windowStart = JSON.parse(event.payload);
    void (async () => {
      try {
        record.startRequestedAt = new Date().toISOString();
        await browser.send('Tracing.start', TRACE_CONFIG);
        record.startedAt = new Date().toISOString();
        timer = setTimeout(() => { record.endRequestedAt = new Date().toISOString(); void browser.send('Tracing.end').catch(rejectDone); }, 12000);
      } catch (error) { rejectDone(error); }
    })();
  });
  await page.addInitScript(() => window.addEventListener('aethervsr:m10:performance:start', () => {
    globalThis.m105TraceStart(JSON.stringify({ at: performance.now(), timeOrigin: performance.timeOrigin, visible: document.visibilityState, focused: document.hasFocus() }));
  }, { once: true }));
  deadline = setTimeout(() => rejectDone(new Error('Bounded trace did not complete')), 40000);
  return { done, async close() { clearTimeout(timer); clearTimeout(deadline); if (started) await browser.send('Tracing.end').catch(() => {}); await session.detach().catch(() => {}); await browser.detach().catch(() => {}); } };
}