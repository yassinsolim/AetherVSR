import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { gunzipSync } from 'node:zlib';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const legacyRevision = 'c65f67a';
const legacyPath = 'src/core/upscale/budget-guard.ts';
const currentPath = 'src/core/upscale/runtime-controller.ts';
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

function quantile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lower = sorted[Math.floor(position)];
  return lower + (sorted[Math.ceil(position)] - lower) * (position % 1);
}

export function readCalibration(file) {
  const bytes = readFileSync(file);
  const artifact = JSON.parse(gunzipSync(bytes).toString('utf8'));
  if (artifact.valid === false) return null;
  if (artifact.schema !== 'aethervsr.m9-runtime/1' || artifact.phase !== 'calibration' || artifact.valid !== true) {
    throw new Error(`Not a valid calibration artifact: ${file}`);
  }
  const measured = artifact.measured;
  if (!Array.isArray(measured?.samples) || !measured.samples.length || measured.samples.some(Array.isArray)) {
    throw new Error(`Expected calibration object samples, not final numerical arrays: ${file}`);
  }
  const mediaFile = resolve(root, 'results/m9-media.json');
  const mediaBytes = readFileSync(mediaFile);
  const media = JSON.parse(mediaBytes.toString('utf8'));
  const sourcePath = `public${artifact.config.clip}`;
  const output = media.outputs.find(output => output.path === sourcePath);
  const stream = output?.ffprobe.streams[0];
  const [numerator, denominator] = (stream?.r_frame_rate ?? '60/1').split('/').map(Number);
  const workload = stream ? { width: stream.width, height: stream.height, fps: numerator / denominator }
    : { width: 1280, height: 720, fps: 60 };
  const mediaHash = output?.sha256 ?? (sourcePath === media.parent ? media.parentSha256 : null);
  if (!mediaHash || mediaHash !== artifact.mediaSha256 || ![30, 60].includes(workload.fps) ||
      !artifact.config.name.replace(/^v[23]-/, '').startsWith(`${workload.height}p${workload.fps}-`)) {
    throw new Error(`Unknown or mismatched fixed-cadence workload: ${file}`);
  }
  const generation = measured.samples[0].generation;
  if (measured.samples.some(sample => sample.generation !== generation || !sample.neural ||
      sample.source?.width !== workload.width || sample.source?.height !== workload.height)) {
    throw new Error(`Replay requires a constant-generation, continuously-neural source: ${file}`);
  }
  const frames = measured.frames;
  return {
    started: measured.started, ended: measured.ended, generation, workload, samples: measured.samples,
    input: { file: relative(root, resolve(file)), sha256: sha256(bytes), config: artifact.config,
      captureHead: artifact.head, measuredAt: artifact.measuredAt, browser: artifact.browser,
      machine: artifact.machine, os: artifact.os, osBuild: artifact.osBuild, displays: artifact.displays,
      precision: measured.precision, mediaSha256: artifact.mediaSha256, modelSha256: artifact.modelSha256,
      cadenceBasis: 'Fixed external source metadata; never inferred from callback/render/drop pressure',
      mediaMetadataSha256: sha256(mediaBytes),
      cadenceEvidence: output ? 'results/m9-media.json ffprobe stream' : 'results/m9-media.json 720p60 parent; docs/M9-CALIBRATION.md' },
    pressure: { scope: 'Recorded continuously-neural observations only; never controller inputs',
      frameCount: frames.length,
      presentedSkips: frames.reduce((total, frame) => total + Math.max(0, frame[2] - 1), 0),
      callbackP90Ms: quantile(frames.map(frame => frame[3]), 0.9),
      callbackMaxMs: frames.length ? Math.max(...frames.map(frame => frame[3])) : null,
      recordedStats: measured.stats },
  };
}

function candidates(samples, failMs) {
  const window = [];
  const statistics = Object.fromEntries(['median30', 'p90of30'].map(name => [name,
    { maxMs: null, evaluations: 0, aboveFailureWindows: 0, threeWindowTriggerCount: 0 }]));
  const streaks = { median30: 0, p90of30: 0 };
  for (const sample of samples) {
    if (!sample.neural || !Number.isFinite(sample.ms) || sample.ms <= 0) continue;
    window.push(sample.ms);
    if (window.length > 30) window.shift();
    if (window.length < 30) continue;
    for (const [name, fraction] of [['median30', 0.5], ['p90of30', 0.9]]) {
      const value = quantile(window, fraction);
      const statistic = statistics[name];
      statistic.maxMs = statistic.maxMs === null ? value : Math.max(statistic.maxMs, value);
      statistic.evaluations++;
      if (value > failMs) {
        statistic.aboveFailureWindows++;
        if (++streaks[name] === 3) statistic.threeWindowTriggerCount++;
      } else streaks[name] = 0;
    }
  }
  return { scope: 'Descriptive only, not a policy search or controller input. All raw positive neural samples, including cold exclusions and virtual baseline periods; overlapping arrival-order windows; one trigger per run of >=3 above-threshold windows.',
    failMs, ...statistics };
}

export function replay(trace, { RuntimeController, createReference, referenceMetadata }) {
  const { started, ended, generation, workload } = trace;
  if (!referenceMetadata?.label) throw new Error('An explicit reference label is required');
  if (!Number.isFinite(started) || started < 0 || !Number.isFinite(ended) || ended < started ||
      !Number.isSafeInteger(generation) || generation < 0) throw new Error('Invalid replay bounds/generation');
  const samples = [...trace.samples].sort((left, right) => left.resolvedAt - right.resolvedAt);
  for (const sample of samples) {
    if (!Number.isFinite(sample.submittedAt) || !Number.isFinite(sample.resolvedAt) ||
        sample.submittedAt < 0 || sample.resolvedAt < sample.submittedAt || sample.resolvedAt < started) {
      throw new Error('Samples require original finite submission/arrival timestamps within the replay clock');
    }
  }
  const controller = new RuntimeController({}, started);
  controller.setWorkload(workload.width, workload.height, workload.fps, started);
  let currentView = controller.bindGeneration(generation, started);
  const guard = createReference();
  const makeSide = state => ({ state, tier: 'neural', generation, since: started, fallbackMs: 0,
    fallbackCount: 0, probeCount: 0, transitions: [], routing: { offered: 0, baseline: 0,
      beforeSwitch: 0, wrongGeneration: 0, nonNeural: 0, outsideWindow: 0 },
    bindings: [{ atMs: started, generation, tier: 'neural' }] });
  const current = makeSide(currentView.state);
  const reference = makeSide('stable');
  let clock = started;
  const advance = now => {
    for (const side of [current, reference]) if (side.tier === 'baseline') side.fallbackMs += now - clock;
    clock = now;
  };
  const observe = (side, state, tier, reason) => {
    const switched = side.tier !== tier;
    if (switched) {
      side.generation++;
      side.since = clock;
      side.bindings.push({ atMs: clock, generation: side.generation, tier });
      if (tier === 'baseline') side.fallbackCount++;
      else side.probeCount++;
    }
    if (side.state !== state || switched) {
      side.transitions.push({ atMs: clock, from: side.state, to: state, tier, reason, generation: side.generation });
    }
    side.state = state;
    side.tier = tier;
    return switched;
  };
  const applyCurrent = snapshot => {
    currentView = snapshot;
    if (observe(current, snapshot.state, snapshot.tier, snapshot.reason)) {
      currentView = controller.bindGeneration(current.generation, clock);
    }
  };
  const applyReference = decision => observe(reference,
    decision.state === 'fallback' ? 'fallback' : decision.probing ? 'probing' : 'stable',
    decision.state === 'fallback' ? 'baseline' : 'neural', decision.reason);
  const eligible = (side, sample) => {
    let rejection;
    if (!sample.neural) rejection = 'nonNeural';
    else if (sample.generation !== generation) rejection = 'wrongGeneration';
    else if (sample.submittedAt < started || sample.submittedAt > ended) rejection = 'outsideWindow';
    else if (side.tier !== 'neural') rejection = 'baseline';
    else if (sample.submittedAt < side.since) rejection = 'beforeSwitch';
    if (rejection) { side.routing[rejection]++; return false; }
    side.routing.offered++;
    return true;
  };
  const replayEnd = Math.max(ended, samples.at(-1)?.resolvedAt ?? ended);
  let tickIndex = 1;
  let sampleIndex = 0;
  let tickCount = 0;
  while (sampleIndex < samples.length || started + tickIndex * 100 <= replayEnd) {
    const tickAt = started + tickIndex * 100;
    const sample = samples[sampleIndex];
    if (tickAt <= replayEnd && (!sample || tickAt <= sample.resolvedAt)) {
      advance(tickAt);
      applyCurrent(controller.tick(clock));
      applyReference(guard.tick(clock));
      tickIndex++;
      tickCount++;
    } else {
      advance(sample.resolvedAt);
      if (eligible(current, sample)) applyCurrent(controller.record({ ...sample, generation: current.generation }, clock));
      if (eligible(reference, sample)) applyReference(guard.record(sample.ms, clock));
      sampleIndex++;
    }
  }
  advance(replayEnd);
  currentView = controller.snapshot(replayEnd);
  return {
    input: trace.input ?? null, workload,
    window: { started, measuredEnd: ended, replayEnd, drainMs: replayEnd - ended,
      tickMs: 100, tickCount, rawSamples: samples.length,
      ordering: 'Original absolute times; ticks precede equal-time arrivals, equal-time samples retain input order; trailing timestamp drain included' },
    current: { ...current, final: currentView }, reference: { ...reference, metadata: referenceMetadata },
    candidates: candidates(samples, currentView.failMs), pressure: trace.pressure ?? null,
  };
}

export async function run(argv = process.argv.slice(2)) {
  const { values, positionals } = parseArgs({ args: argv, allowPositionals: true,
    options: { out: { type: 'string' }, help: { type: 'boolean' } } });
  if (values.help) {
    console.log('Usage: node tools/m9-replay.mjs [calibration.json.gz ...] [--out results/m9-replay.json]\nDefault: all valid results/m9-calibration*.json.gz, steady measured windows only. GPU-free; frozen Git history required for CLI only. No overwrite.');
    return;
  }
  const files = positionals.length ? positionals.map(file => resolve(file)) :
    readdirSync(resolve(root, 'results')).filter(name => /^m9-calibration.*\.json\.gz$/.test(name))
      .sort().map(name => resolve(root, 'results', name));
  const traces = [];
  const excluded = [];
  for (const file of files) {
    const trace = readCalibration(file);
    if (trace) traces.push(trace);
    else excluded.push({ file: relative(root, file), sha256: sha256(readFileSync(file)), reason: 'Artifact marked invalid' });
  }
  if (!traces.length) throw new Error('No valid calibration traces');
  const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  let legacySource;
  try {
    legacySource = execFileSync('git', ['show', `${legacyRevision}:${legacyPath}`], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
  } catch {
    throw new Error(`Exact CLI comparison requires ${legacyRevision}:${legacyPath} in local Git history; no substitute or automatic fetch. CI tests inject the separately labeled corrected fixed10/7 guard.`);
  }
  const typescript = (await import('typescript')).default;
  const transpiled = typescript.transpileModule(legacySource.toString('utf8'), {
    compilerOptions: { target: typescript.ScriptTarget.ES2022, module: typescript.ModuleKind.ESNext },
  }).outputText;
  const { BudgetGuard } = await import(`data:text/javascript;base64,${Buffer.from(transpiled).toString('base64')}`);
  const referenceMetadata = { label: 'original BudgetGuard c65f67a (unmodified)', exactLegacy: true,
    revision: git('rev-parse', legacyRevision), blobSha: git('rev-parse', `${legacyRevision}:${legacyPath}`),
    path: legacyPath, sha256: sha256(legacySource), typescript: typescript.version,
    loader: 'git show -> typescript.transpileModule -> data URL', failMs: 10, recoverMs: 7 };
  const provenance = { head: git('rev-parse', 'HEAD'), cleanWorktreeRequired: false,
    current: { path: currentPath, sha256: sha256(readFileSync(resolve(root, currentPath))), loader: 'Vite SSR working-tree source' },
    lockedPolicy: { path: 'docs/M9-CALIBRATION.md', sha256: sha256(readFileSync(resolve(root, 'docs/M9-CALIBRATION.md'))) },
    toolSha256: sha256(readFileSync(fileURLToPath(import.meta.url))), reference: referenceMetadata };
  const { createServer } = await import('vite');
  const server = await createServer({ root, configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null }, optimizeDeps: { noDiscovery: true } });
  let RuntimeController;
  try { ({ RuntimeController } = await server.ssrLoadModule(`/${currentPath}`)); }
  finally { await server.close(); }
  const comparisons = traces.map(trace => replay(trace,
    { RuntimeController, createReference: () => new BudgetGuard(), referenceMetadata }));
  if (sha256(readFileSync(resolve(root, currentPath))) !== provenance.current.sha256) {
    throw new Error('Controller changed during replay');
  }
  const report = { schema: 'aethervsr.m9-replay/1', provenance,
    scope: 'GPU-free counterfactual controller replay of each valid calibration steady window separately, not pooled. Uses all raw tagged arrivals and fixed externally known source FPS for workload configuration. Controller starts in warmup at recorded window start; startup trace is excluded. Each controller gates submissions against its own virtual tier/generation. Sample timestamps and durations are never synthesized; polling uses an explicit virtual 100 ms schedule.',
    limitations: ['Continuously-neural recordings cannot measure actual baseline or recovery performance.',
      'Virtual switches do not model cold starts, destroy/recreate, reconfiguration, decode, scheduling or changed GPU contention; probe outcomes are counterfactual, not runtime acceptance.',
      'Fixed source FPS bypasses live cadence estimation; raw frame pressure is reported but never steers policy.',
      'Descriptive median30/p90of30 comparisons do not retune the already locked policy.'],
    excluded, comparisons };
  for (const result of comparisons) console.error(JSON.stringify({ name: result.input.config.name,
    currentFallbacks: result.current.fallbackCount, legacyFallbacks: result.reference.fallbackCount,
    descriptiveOnly: result.candidates }));
  const serialized = `${JSON.stringify(report, null, 2)}\n`;
  if (values.out) writeFileSync(resolve(values.out), serialized, { flag: 'wx' });
  else process.stdout.write(serialized);
  return report;
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  run().catch(error => { console.error(error.message); process.exitCode = 1; });
}