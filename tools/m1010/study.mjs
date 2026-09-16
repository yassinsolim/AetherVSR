import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, join, relative } from 'node:path';
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { ROOT, sha256 } from '../m10-fixtures.mjs';
import { nativeWindow } from '../m105-accounting.mjs';
import { seekPaused } from '../m10-output-parity.mjs';
import { studyIdentity, openResearch } from './native.mjs';
import { startFixtures } from './fixtures.mjs';
import { openCheckpoint } from './checkpoint.mjs';
import { observePlayback, observeCadence, summarizeCadence, inputFrameEvidence, pixelDifference } from './acquisition.mjs';

export const R1_CASES = [
  { id: 'R1-A', mode: 'same', packaged: true },
  { id: 'R1-B', mode: 'same' },
  { id: 'R1-C', mode: 'cors', cors: 'anonymous' },
  { id: 'R1-D', mode: 'nocors' },
  { id: 'R1-J', mode: 'blob' },
  { id: 'R1-K', mode: 'MSE' },
];
export const INPUT_TIMES = [1.2, 2.2, 3.2];
export const R1_INPUT_CASES = ['A', 'B'].flatMap(asset => ['source', 'replay'].map(surface => ({ asset, surface, id: `R1-input-${asset}-${surface}` })));

export function replayStage(completed, automaticOnly) {
  if (completed.includes('R1-F')) return 'RETAINED';
  return automaticOnly ? 'AUTOMATIC_PREFIX' : 'PRIVILEGED';
}

export function replayOutcome(result) {
  if (result.exception) return 'UNRESOLVED';
  return result.sameBytes === true && result.playback?.playable === true && result.comparisons?.length === 3 &&
    result.comparisons.every(value => value.outcome === 'SUPPORTED') ? 'SUPPORTED' : 'UNRESOLVED';
}

export function redirectOutcome(result) {
  const path = new URL(result.selected.selectedUrl).pathname;
  return result.exception && result.fetch === null && result.requestDelta.some(request => request.path === path) &&
    !result.requestDelta.some(request => request.path === '/cors/A.mp4') ? 'UNSUPPORTED_SAFE' : 'UNRESOLVED';
}

export async function waitForObserved(read, accept, timeoutMs = 900000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await read();
    if (accept(value)) return value;
    if (Date.now() >= deadline) throw new Error('Observable browser-state deadline; no action assumed');
    await new Promise(done => setTimeout(done, Math.min(100, deadline - Date.now())));
  }
}

function environment() {
  const command = (name, args) => execFileSync(name, args, { encoding: 'utf8' }).trim();
  return { machine: { model: command('sysctl', ['-n', 'hw.model']), chip: command('sysctl', ['-n', 'machdep.cpu.brand_string']),
    memoryBytes: Number(command('sysctl', ['-n', 'hw.memsize'])) }, os: command('sw_vers', []),
    timingScope: 'native browser observations; not physical scanout or benchmark results' };
}

async function fetchObservation(page, url, credentials = 'omit') {
  return page.evaluate(async ({ url, credentials }) => {
    try {
      const response = await fetch(url, { credentials, redirect: 'error', cache: 'no-store', signal: AbortSignal.timeout(10000) });
      const bytes = await response.arrayBuffer();
      return { fetchable: response.ok, status: response.status, responseUrl: response.url, bytes: bytes.byteLength,
        sha256: [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(value => value.toString(16).padStart(2, '0')).join(''), credentials };
    } catch (error) { return { fetchable: false, error: String(error), credentials }; }
  }, { url, credentials });
}

export async function runReplayStudy(directory, { automaticOnly = false } = {}) {
  const identity = studyIdentity(), root = resolve(ROOT, directory);
  assert(root.startsWith(join(ROOT, '.cache/m1010/')));
  let native, fixtures, checkpoint, source, player;
  const sourceNonce = randomUUID();
  let sourceTabId;
  try {
    native = await openResearch(identity, { profileDirectory: join(root, 'profile') });
    const executableSha256 = sha256(readFileSync(native.executable));
    checkpoint = openCheckpoint(directory, { studyVersion: 'M10.10-replay-automation-1', sourceCommit: identity.sourceCommit, browserExecutableSha256: executableSha256 });
    if (replayStage(checkpoint.snapshot().completedExperimentIds, automaticOnly) === 'RETAINED') {
      console.log(JSON.stringify({ checkpoint: 'R1_RETAINED_NO_REPEATS', state: `${directory}/state.json` })); return checkpoint.snapshot();
    }
    fixtures = await startFixtures({ extensionOrigins: [`chrome-extension://${native.extensionId}`] });
    assert.equal(identity.provenance.mediaFixture.sha256, fixtures.media.A.sha256, 'Build this R1 package with the fixed marked720p30 fixture');
    const pageUrl = `chrome-extension://${native.extensionId}/acquire.html`;
    const sourceUrl = `${fixtures.url}?extension=${native.extensionId}`;
    const grants = () => native.worker.evaluate(() => chrome.permissions.getAll());
    const hasHost = value => value.origins?.includes('http://127.0.0.1/*') === true;
    const step = async (id, run) => {
      if (checkpoint.has(id)) { console.log(JSON.stringify({ skippedImmutable: id })); return checkpoint.read(id); }
      checkpoint.begin(id); console.log(JSON.stringify({ running: id }));
      try {
        const result = await run(); checkpoint.complete(id, result);
        console.log(JSON.stringify({ completed: id, outcome: result.outcome ?? 'RECORDED' })); return result;
      } catch (error) {
        checkpoint.complete(id, { outcome: 'UNRESOLVED', executionStatus: 'ATTEMPTED', error: String(error) });
        throw error;
      }
    };
    const setInstruction = async (page, instruction) => {
      await page.bringToFront();
      await page.evaluate(instruction => {
        let output = document.getElementById('study-instruction');
        if (!output) { output = document.createElement('p'); output.id = 'study-instruction'; document.body.prepend(output); }
        output.textContent = instruction;
      }, instruction);
    };
    const manual = async (page, instruction, observable, read, accept) => {
      const action = { instruction, observable, requestedAt: new Date().toISOString() };
      checkpoint.manual(action); await setInstruction(page, instruction);
      console.log(`MANUAL CHECKPOINT: ${instruction}`);
      const value = await waitForObserved(read, accept);
      await page.evaluate(() => document.getElementById('study-instruction')?.remove());
      checkpoint.manual(null);
      return { ...action, observedAt: new Date().toISOString(), value };
    };
    const ensureSource = async () => {
      if (source && !source.isClosed()) return;
      source = await native.context.newPage(); await nativeWindow(source, native.context);
      await source.goto(sourceUrl); await source.waitForFunction(() => !!globalThis.__M1010_SOURCE__?.snapshot().selection);
      await source.evaluate(nonce => { document.documentElement.dataset.m1010Study = nonce; }, sourceNonce);
    };
    const prepare = async (mode, asset = 'A', credentials = 'omit') => {
      await ensureSource(); await source.bringToFront();
      if (mode === 'auth-progressive') await source.evaluate(() => globalThis.__M1010_SOURCE__.prepare('cookieseed'));
      const selection = await source.evaluate(choice => globalThis.__M1010_SOURCE__.prepare(choice), { mode, asset, credentials });
      await source.locator('#play').click();
      const playback = await source.evaluate(observePlayback);
      const generic = await source.evaluate(() => {
        const video = document.querySelector('video');
        return { url: video.currentSrc || null, sourceClass: video.srcObject ? 'srcObject' : video.currentSrc.startsWith('blob:') ? 'blob-unknown' : 'progressive',
          intrinsic: [video.videoWidth, video.videoHeight], documentUrl: location.href };
      });
      return { selection, playback, generic };
    };
    const storeFrame = async (page, id, label) => {
      const value = await page.evaluate(inputFrameEvidence);
      for (const key of ['pixels', 'copiedPixels']) if (value[key]) {
        const { base64, ...metadata } = value[key];
        const bytes = Buffer.from(base64, 'base64'); assert.equal(sha256(bytes), metadata.sha256); assert.equal(bytes.length, metadata.bytes);
        const path = `${id}-${label}-${randomUUID()}-${key}.rgba`;
        writeFileSync(join(root, path), bytes, { flag: 'wx' }); value[key] = { ...metadata, path };
      }
      return value;
    };
    const frames = async (page, id) => {
      const values = [];
      await page.evaluate(() => document.querySelector('video').pause());
      for (const time of INPUT_TIMES) {
        const seek = await page.evaluate(seekPaused, { time });
        values.push({ requestedTime: time, seek, frame: await storeFrame(page, id, String(time).replace('.', '-')) });
      }
      return values;
    };
    const compare = (reference, candidate) => candidate.map((entry, index) => {
      const sourceFrame = reference[index].frame, acquired = entry.frame;
      const sameDimensions = sourceFrame.width === acquired.width && sourceFrame.height === acquired.height;
      const result = { time: entry.requestedTime, sourceMediaTime: reference[index].seek.metadata.mediaTime,
        receiverMediaTime: entry.seek.metadata.mediaTime, sameDimensions, rgba: null, forcedCopy: null };
      if (sameDimensions && sourceFrame.pixels && acquired.pixels) result.rgba = pixelDifference(readFileSync(join(root, sourceFrame.pixels.path)), readFileSync(join(root, acquired.pixels.path)));
      if (sameDimensions && sourceFrame.copiedPixels && acquired.copiedPixels) result.forcedCopy = pixelDifference(readFileSync(join(root, sourceFrame.copiedPixels.path)), readFileSync(join(root, acquired.copiedPixels.path)));
      result.outcome = sameDimensions && result.sourceMediaTime === result.receiverMediaTime && result.rgba?.exact && result.forcedCopy?.exact && acquired.externalImportable ? 'SUPPORTED' : 'UNRESOLVED';
      return result;
    });
    const closePlayer = async () => {
      if (!player || player.isClosed()) { player = null; return null; }
      await player.evaluate(() => globalThis.m1010Acquire.stop().catch(() => {}));
      const snapshot = await player.evaluate(() => globalThis.m1010Acquire.snapshot());
      assert(Object.values(snapshot.resources).every(value => value === 0));
      await player.close(); player = null; return { resources: snapshot.resources, pending: snapshot.pending };
    };
    const installedPlayer = async () => {
      const candidates = native.context.pages().filter(page => page.url() === pageUrl);
      for (const page of candidates) {
        const info = await page.evaluate(() => globalThis.m1010Acquire?.snapshot().info).catch(() => null);
        if (info?.sourceTabId !== undefined && info.source?.url) return { page, info };
      }
      return null;
    };
    const verifySelection = async value => {
      const info = value.info;
      const verified = await native.worker.evaluate(async info => chrome.scripting.executeScript({
        target: { tabId: info.sourceTabId, documentIds: [info.selection.documentId] },
        func: () => ({ nonce: document.documentElement.dataset.m1010Study, documentUrl: location.href, url: document.querySelector('video')?.currentSrc }),
      }), info);
      assert.equal(verified[0]?.documentId, info.selection.documentId);
      assert.equal(verified[0]?.result.nonce, sourceNonce); assert.equal(verified[0]?.result.documentUrl, source.url());
      assert.equal(verified[0]?.result.url, info.selection.url); sourceTabId = info.sourceTabId; player = value.page;
      return { sourceTabId, sourceDocumentId: info.selection.documentId, selectedUrl: info.selection.url };
    };
    const selectAfterGrant = async () => {
      await closePlayer();
      assert(hasHost(await grants()));
      if (sourceTabId === undefined) {
        const found = await native.worker.evaluate(url => chrome.tabs.query({ url }), source.url());
        assert.equal(found.length, 1); sourceTabId = found[0].id;
      }
      await native.worker.evaluate(tabId => globalThis.__M1010_RESEARCH_SELECT__(tabId), sourceTabId);
      const selected = await waitForObserved(installedPlayer, value => !!value?.info.selection, 10000);
      return verifySelection(selected);
    };
    const nativeRefetch = async (id, sourceState, reference) => {
      const selected = await selectAfterGrant(), before = await (await fetch(`${fixtures.origins[0]}/requests.json`)).json();
      let exception = null;
      try { await player.evaluate(() => globalThis.m1010Acquire.refetch()); } catch (error) { exception = String(error); }
      const snapshot = await player.evaluate(() => globalThis.m1010Acquire.snapshot());
      const fetched = snapshot.events.findLast(event => event.type === 'refetch')?.value ?? null;
      const result = { source: sourceState, selected, brokerEnforced: true, exception, fetch: fetched,
        sameBytes: fetched ? fetched.sha256 === fixtures.media.A.sha256 : null,
        playback: exception ? null : await player.evaluate(observePlayback), input: [], comparisons: [],
        requestDelta: (await (await fetch(`${fixtures.origins[0]}/requests.json`)).json()).requests.slice(before.requests.length),
        outcome: 'UNRESOLVED' };
      if (!exception && result.playback.playable) { result.input = await frames(player, id); result.comparisons = compare(reference, result.input); }
      result.outcome = replayOutcome(result);
      result.cleanup = await closePlayer(); return result;
    };

    await step('environment', async () => ({ ...environment(), identity, browser: { version: await native.browser.version(), executableSha256,
      policy: 'default; real permission gestures only; native CDP without focus emulation' }, media: fixtures.media, initialGrants: await grants() }));
    await step('retained-prior', async () => {
      const paths = ['surface-01.json', 'media-api-01.json', 'media-api-02.json', 'manual-01.json', 'manual-02.json', 'manual-03.json', 'manual-04.json',
        'sender-context-01.json', 'navigation-01.json', 'navigation-02.json'];
      return { scope: 'Historical immutable evidence; original source revisions retained, not pooled', artifacts: paths.map(name => {
        const path = join(ROOT, '.cache/m1010', name); assert(existsSync(path), `Missing retained raw ${name}`);
        const bytes = readFileSync(path), report = JSON.parse(bytes);
        return { path: relative(ROOT, path), bytes: bytes.length, sha256: sha256(bytes), sourceCommit: report.identity?.sourceCommit ?? report.sourceCommit ?? null };
      }) };
    });
    checkpoint.candidate('R1', { state: 'INCOMPLETE', mandatory: 'progressive720p30/60 plus all preregistered gates' });
    checkpoint.candidate('R2', { state: 'CHARACTERIZATION_ONLY', reason: 'Reduced RTC resolution; no explicit controllable sizing cause identified' });
    checkpoint.candidate('R4-R5', { state: 'UNSUPPORTED_SAFE', reason: 'Absent cropTo/restrictTo on tested extension tabCapture track' });
    const reference = await step('R1-reference-A', async () => {
      const sourceState = await prepare('same'); assert(sourceState.playback.playable);
      return { source: sourceState, input: await frames(source, 'R1-reference-A'), outcome: 'SUPPORTED' };
    });
    for (const entry of R1_CASES) await step(entry.id, async () => {
      assert(!hasHost(await grants()), 'Unprivileged matrix cannot reuse a host grant');
      const sourceState = await prepare(entry.mode);
      player = await native.context.newPage(); await nativeWindow(player, native.context); await player.goto(pageUrl); await player.bringToFront();
      const url = entry.packaged ? `chrome-extension://${native.extensionId}/media/known.mp4` : sourceState.generic.url;
      await player.evaluate(({ url, cors }) => {
        const video = document.querySelector('video'); video.muted = true; video.loop = false;
        if (cors) video.crossOrigin = cors; else video.removeAttribute('crossorigin'); video.src = url; video.load();
      }, { url, cors: entry.cors ?? null });
      const playback = await player.evaluate(observePlayback), fetched = await fetchObservation(player, url);
      const result = { source: sourceState, url, brokerEnforced: false,
        scope: 'Unprivileged browser API matrix in controlled extension media surface; not a synthesized activeTab grant',
        fetch: fetched, playback, sameBytes: fetched.fetchable ? fetched.sha256 === fixtures.media.A.sha256 : null,
        input: [], comparisons: [], outcome: 'UNSUPPORTED_SAFE' };
      if (playback.playable) {
        result.input = await frames(player, entry.id); result.comparisons = compare(reference.input, result.input);
        result.outcome = result.comparisons.every(value => value.outcome === 'SUPPORTED') ? 'SUPPORTED' : result.input.some(value => !value.frame.originClean) ? 'UNSUPPORTED_SAFE' : 'UNRESOLVED';
      }
      if (['blob', 'MSE'].includes(entry.mode)) {
        result.genericReplayContract = 'UNSUPPORTED_SAFE; opaque page-owned blob/MSE is not reconstructed'; result.outcome = 'UNSUPPORTED_SAFE';
      }
      result.cleanup = await closePlayer(); return result;
    });
    if (replayStage(checkpoint.snapshot().completedExperimentIds, automaticOnly) === 'AUTOMATIC_PREFIX') {
      if (!checkpoint.has('R1-host-grant')) checkpoint.manual({ instruction: 'Invoke the research extension on the prepared no-CORS source, then grant its local origin when prompted by the runner.', observable: 'authenticated selected source followed by optional host permission' });
      console.log(JSON.stringify({ checkpoint: 'R1_AUTOMATIC_PREFIX_COMPLETE', next: 'R1 source invocation and one local-origin grant', state: `${directory}/state.json` }));
      return checkpoint.snapshot();
    }
    const selectedSource = await prepare('nocors');
    await step('R1-host-grant', async () => {
      const instructions = [];
      if (!hasHost(await grants())) {
        const invocation = await manual(source,
          'On this source tab, click Extensions > AetherVSR M10.10 Research > Select source and open acquisition.',
          'Authenticated acquisition page for this exact fixture document', installedPlayer, value => !!value?.info.selection);
        const verified = await verifySelection(invocation.value);
        instructions.push({ ...invocation, value: verified });
        instructions.push(await manual(player, 'Click Grant source origin, then approve access to http://127.0.0.1/* in Chrome.',
          'Chrome permissions contains the selected local host grant', grants, hasHost));
      }
      return { instructions, grants: await grants(), scope: 'privileged extension-origin fetch with granted host permission', outcome: 'SUPPORTED' };
    });
    assert(hasHost(await grants()), 'A retained grant checkpoint is not a current browser permission');
    await step('R1-E', () => nativeRefetch('R1-E', selectedSource, reference.input));
    for (const [id, mode] of [['R1-G', 'redirect-same'], ['R1-H', 'redirect-ungranted']]) await step(id, async () => {
      const sourceState = await prepare(mode), result = await nativeRefetch(id, sourceState, reference.input);
      result.outcome = redirectOutcome(result);
      assert.equal(result.outcome, 'UNSUPPORTED_SAFE', 'Selected redirect request/rejection not established');
      return result;
    });
    await step('R1-I', async () => {
      const sourceState = await prepare('auth-progressive', 'A', 'include');
      const result = await nativeRefetch('R1-I', sourceState, reference.input);
      const control = await native.context.newPage();
      try { await control.goto(pageUrl); result.credentialsOmitted = await fetchObservation(control, sourceState.generic.url, 'omit'); }
      finally { await control.close(); }
      return result;
    });
    await step('R1-lifecycle', async () => {
      const cycles = [];
      for (let index = 0; index < 10; index++) {
        const sourceState = await prepare('nocors', index === 5 ? 'B' : 'A');
        const selected = await selectAfterGrant();
        await player.evaluate(() => globalThis.m1010Acquire.refetch());
        const before = await player.evaluate(() => globalThis.m1010Acquire.snapshot());
        assert.equal(before.owner, 'PLAYER_AUTHORITY');
        await player.evaluate(() => document.querySelector('video').pause());
        const seek = await player.evaluate(seekPaused, { time: 1 });
        const resumed = await player.evaluate(observePlayback); assert(resumed.playable);
        const cleanup = await closePlayer(); cycles.push({ index, source: sourceState.generic, selected, seek, resumed, cleanup });
      }
      return { cycles, twoSources: ['A', 'B'], outcome: 'SUPPORTED', scope: 'Replay controls and ten acquisition/close cycles; not neural pipeline lifecycle' };
    });
    await step('R1-F', async () => {
      await prepare('nocors'); await selectAfterGrant(); await player.evaluate(() => globalThis.m1010Acquire.refetch());
      const removed = await manual(player, 'Click Revoke source origin. The runner will verify revocation and cleanup automatically.',
        'Chrome host permission absent and acquisition resources released', async () => ({ grants: await grants(), snapshot: await player.evaluate(() => globalThis.m1010Acquire.snapshot()) }),
        value => !hasHost(value.grants) && Object.values(value.snapshot.resources).every(count => count === 0) && !value.snapshot.pending);
      let denied;
      try { await player.evaluate(() => globalThis.m1010Acquire.refetch()); denied = false; } catch (error) { denied = String(error); }
      const fetchAfter = await fetchObservation(player, `${fixtures.origins[1]}/nocors/A.mp4`);
      assert(denied && !fetchAfter.fetchable, 'Revocation failed to remove future privileged access');
      return { removed, futureRefetchRejected: denied, fetchAfter, cleanup: await closePlayer(), outcome: 'SUPPORTED' };
    });
    checkpoint.candidate('R1', { state: 'ACQUISITION_RECORDED_NOT_QUALIFIED', missing: ['720p60 provenance', '10s cadence', 'latency', 'instrumented A/V', 'neural qualification'] });
    console.log(JSON.stringify({ checkpoint: 'R1_ACQUISITION_COMPLETE_NOT_QUALIFIED', state: `${directory}/state.json` }));
    return checkpoint.snapshot();
  } finally { await player?.close().catch(() => {}); await native?.close(); await fixtures?.close(); }
}

export async function runReplayInputStudy(directory, priorDirectory) {
  const identity = studyIdentity(), root = resolve(ROOT, directory);
  assert(root.startsWith(join(ROOT, '.cache/m1010/')));
  assert.notEqual(root, resolve(ROOT, priorDirectory));
  const priorPath = resolve(ROOT, priorDirectory, 'state.json'), priorBytes = readFileSync(priorPath), priorState = JSON.parse(priorBytes);
  const prior = openCheckpoint(priorDirectory, priorState.pin);
  assert(prior.has('R1-F'), 'Complete the retained acquisition batch first');
  let native, fixtures;
  try {
    native = await openResearch(identity);
    const executableSha256 = sha256(readFileSync(native.executable));
    assert.equal(executableSha256, priorState.pin.browserExecutableSha256, 'Different browser requires a separately registered comparison');
    const checkpoint = openCheckpoint(directory, { studyVersion: 'M10.10-replay-input-1', sourceCommit: identity.sourceCommit, browserExecutableSha256: executableSha256 });
    const step = async (id, run) => {
      if (checkpoint.has(id)) { console.log(JSON.stringify({ skippedImmutable: id })); return checkpoint.read(id); }
      checkpoint.begin(id); console.log(JSON.stringify({ running: id }));
      try {
        const result = await run(); checkpoint.complete(id, result);
        console.log(JSON.stringify({ completed: id, outcome: result.outcome ?? 'RECORDED', cadence: result.cadenceSummary?.callbackFps })); return result;
      } catch (error) { checkpoint.complete(id, { outcome: 'UNRESOLVED', error: String(error) }); throw error; }
    };
    fixtures = await startFixtures({ extensionOrigins: [`chrome-extension://${native.extensionId}`] });
    await step('environment', async () => ({ ...environment(), identity, browser: { version: await native.browser.version(), executableSha256,
      policy: 'default; no permission requests, no focus emulation' }, media: fixtures.media,
      prior: { path: relative(ROOT, priorPath), bytes: priorBytes.length, sha256: sha256(priorBytes), sourceCommit: priorState.pin.sourceCommit },
      scope: 'Independent input and raw media cadence only; no broker session or neural pipeline. Four-second fixtures loop; discontinuities retained.' }));
    for (const entry of R1_INPUT_CASES) await step(entry.id, async () => {
      const page = await native.context.newPage(), input = [], media = fixtures.media[entry.asset];
      const result = { ...entry, mediaSha256: media.sha256, input, outcome: 'UNRESOLVED' };
      try {
        result.placement = await nativeWindow(page, native.context);
        if (entry.surface === 'source') {
          await page.goto(fixtures.url); await page.waitForFunction(() => !!globalThis.__M1010_SOURCE__?.snapshot().selection);
          await page.evaluate(asset => globalThis.__M1010_SOURCE__.prepare({ mode: 'same', asset }), entry.asset);
          await page.bringToFront(); await page.locator('#play').click();
        } else {
          await page.goto(`chrome-extension://${native.extensionId}/acquire.html`); await page.bringToFront();
          await page.evaluate(url => { const video = document.querySelector('video'); video.crossOrigin = 'anonymous'; video.src = url; video.muted = true; video.load(); }, `${fixtures.origins[1]}/cors/${entry.asset}.mp4`);
          await page.locator('h1').click();
        }
        result.playback = await page.evaluate(observePlayback); assert(result.playback.playable);
        result.fetch = await fetchObservation(page, result.playback.currentSrc); assert.equal(result.fetch.sha256, media.sha256);
        await page.evaluate(() => { document.querySelector('video').loop = true; });
        result.cadence = await page.evaluate(observeCadence, { durationMs: 10000 });
        result.cadenceSummary = summarizeCadence(result.cadence);
        for (const time of INPUT_TIMES) {
          const seek = await page.evaluate(seekPaused, { time }), frame = await page.evaluate(inputFrameEvidence);
          for (const key of ['pixels', 'copiedPixels']) if (frame[key]) {
            const { base64, ...metadata } = frame[key], bytes = Buffer.from(base64, 'base64');
            assert.equal(bytes.length, metadata.bytes); assert.equal(sha256(bytes), metadata.sha256);
            const path = `${entry.id}-${time}-${randomUUID()}-${key}.rgba`;
            writeFileSync(join(root, path), bytes, { flag: 'wx' }); frame[key] = { ...metadata, path };
          }
          input.push({ time, seek, frame });
        }
        if (entry.surface === 'replay') {
          const reference = checkpoint.read(`R1-input-${entry.asset}-source`);
          assert.equal(reference.input?.length, INPUT_TIMES.length, 'Independent source reference incomplete');
          result.comparisons = input.map((row, index) => {
            const original = reference.input[index], sameDimensions = row.frame.width === original.frame.width && row.frame.height === original.frame.height;
            const comparison = { time: row.time, sameDimensions, sameMediaTime: row.seek.metadata.mediaTime === original.seek.metadata.mediaTime };
            for (const key of ['pixels', 'copiedPixels']) comparison[key] = sameDimensions && row.frame[key] && original.frame[key] ?
              pixelDifference(readFileSync(join(root, original.frame[key].path)), readFileSync(join(root, row.frame[key].path))) : null;
            comparison.exact = comparison.sameDimensions && comparison.sameMediaTime && comparison.pixels?.exact === true && comparison.copiedPixels?.exact === true;
            return comparison;
          });
        }
        result.outcome = input.every(row => row.frame.width === 1280 && row.frame.height === 720 && row.frame.originClean && row.frame.externalImportable && row.frame.forcedCopyImportable) &&
          (entry.surface === 'source' || result.comparisons.every(row => row.exact)) && result.cadenceSummary.outcome === 'RECORDED' ? 'SUPPORTED' : 'UNRESOLVED';
      } catch (error) { result.error = String(error); result.outcome = 'UNRESOLVED';
      } finally {
        if (!page.isClosed()) result.cleanup = await page.evaluate(() => {
          const video = document.querySelector('video'); video.pause(); video.srcObject = null; video.removeAttribute('src'); video.load();
          return { paused: video.paused, srcObject: video.srcObject, sourceAttribute: video.getAttribute('src') };
        }).catch(error => ({ error: String(error) }));
        await page.close();
      }
      if (result.cleanup?.error || result.cleanup?.paused !== true || result.cleanup?.srcObject !== null || result.cleanup?.sourceAttribute !== null) result.outcome = 'UNRESOLVED';
      return result;
    });
    const supported = R1_INPUT_CASES.every(entry => checkpoint.read(entry.id).outcome === 'SUPPORTED');
    checkpoint.candidate('R1', { state: supported ? 'INPUT_CADENCE_RECORDED_NOT_QUALIFIED' : 'INPUT_CADENCE_UNRESOLVED', missing: ['instrumented latency/A/V', 'pipeline input/output parity for these inputs', 'neural qualification'] });
    return checkpoint.snapshot();
  } finally { await native?.close(); await fixtures?.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) {
  const [phase, directory, option] = process.argv.slice(2);
  assert(directory && (phase === 'r1' && [undefined, '--automatic-only'].includes(option) || phase === 'r1-input' && option),
    'Usage: study.mjs r1 <directory> [--automatic-only] | r1-input <directory> <prior-directory>');
  if (phase === 'r1-input') await runReplayInputStudy(directory, option);
  else await runReplayStudy(directory, { automaticOnly: option === '--automatic-only' });
}