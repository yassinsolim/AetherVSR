import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runPerformance, parseCases, summarizeCapture } from './m10-performance.mjs';
import { nativeWindow, deliverySummary } from './m105-accounting.mjs';

export function parseFinalCases(value) {
  const cases = parseCases(value);
  assert(cases.length <= 2, 'At most two final trials per invocation');
  for (const item of cases) assert(item.kind === 'extension-auto' && item.durationMs === 600000 && item.warmupMs === 5000, 'Final trial must be Auto600s with5s warmup');
  return cases;
}

export function deliveryGates(summary, accounting) {
  const atLeast = (value, threshold) => typeof value === 'number' && Number.isFinite(value) && value >= threshold;
  const last = summary.windowStats?.last120s;
  return {
    duration: atLeast(summary.activeMs, 600000) && atLeast(summary.durationMs, 600000),
    renderedFps: atLeast(summary.renderedFps, 58), presentedFps: atLeast(summary.runtimePresentedFps, 58),
    combinedLoss: typeof accounting.m10CombinedPercent === 'number' && Number.isFinite(accounting.m10CombinedPercent)
      && accounting.m10CombinedPercent >= 0 && accounting.m10CombinedPercent <= 1,
    lastRenderedFps: atLeast(last?.renderedFps, 58), lastPresentedFps: atLeast(last?.runtimePresentedFps, 58),
    noSubmissionDeficit: accounting.submissionDeficit === 0,
  };
}

export function finalAcceptance(raw) {
  const summary = summarizeCapture(raw), accounting = deliverySummary(raw);
  const last = summary.windowStats?.last120s;
  const states = raw.runtime.states;
  const tierChanges = states.filter((row, index) => index > 0 && row[2] !== states[index - 1][2]).length;
  const before = raw.runtime.opening.status?.details?.infrastructure;
  const after = raw.runtime.closing.status?.details?.infrastructure;
  const ownerChanges = before && after ? after.ownerChanges - before.ownerChanges : null;
  const gates = {
    ...deliveryGates(summary, accounting),
    tierStability: tierChanges === 0 && raw.runtime.frames.length > 0 && raw.runtime.frames.every(row => row[9] === 1),
    ownerStability: ownerChanges === 0,
    noRuntimeError: raw.runtime.error === null,
    noIntegrityEvent: raw.video.events.length === 0,
  };
  return { verdict: Object.values(gates).every(Boolean) ? 'PASS' : 'FAIL', gates,
    metrics: accounting, tierChanges, ownerChanges,
    last120: { renderedFps: last?.renderedFps ?? null, presentedFps: last?.runtimePresentedFps ?? null },
    rule: 'Each of two independent valid final trials must pass every gate. Failed performance is retained, not an invalid-capture retry.' };
}

export async function runFinal(cases, prefix) {
  return runPerformance(cases, prefix, {
    parseCases: parseFinalCases, accounting: true, viewport: () => ({ width: 1200, height: 760 }), rawByteLimit: 12 * 1024 * 1024,
    sources: ['tools/m105-final.mjs', 'tools/m105-accounting.mjs'],
    description: { milestone: 'M10.5 final acceptance', protocol: 'docs/M10.5-PREREGISTRATION.md',
      viewport: 'Native1200x760 DPR2, built-in1512x982 display, desktop40,40; original custom fixture640x360',
      instrumentation: 'M10 raw GPU/core/driver recorder plus opt-in frame metadata/attempts. No Chrome trace, extra rAF/long-task observer or owned-callback wrappers. Not uninstrumented playback.' },
    preparePage: (page, native) => nativeWindow(page, native.context),
    collect: async (native, page) => {
      const session = await native.context.newCDPSession(page);
      try {
        const bounds = await session.send('Browser.getWindowForTarget');
        const screen = await page.evaluate(() => ({ width: screen.width, height: screen.height, dpr: devicePixelRatio, x: screenX, y: screenY }));
        return { closingNative: { bounds: bounds.bounds, screen } };
      } finally { await session.detach(); }
    },
    validate: (raw, item, result) => {
      assert.deepEqual(raw.closingNative.screen, result.preparation.nativeScreen, 'Display changed');
      assert.deepEqual(raw.closingNative.bounds, result.preparation.bounds.bounds, 'Native window changed');
    },
    summarize: finalAcceptance,
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert(process.argv.length === 4, 'Usage: node tools/m105-final.mjs CASES.json .cache/m105/PREFIX');
  const report = await runFinal(resolve(process.argv[2]), resolve(process.argv[3]));
  console.log(JSON.stringify(report.results.map(row => ({ id: row.case.id, capture: row.completion, acceptance: row.diagnostics }))));
}