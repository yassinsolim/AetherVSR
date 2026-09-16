import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { join } from 'node:path';
import { ROOT, sha256 } from '../m10-fixtures.mjs';
import { nativeWindow } from '../m105-accounting.mjs';
import { studyIdentity, openResearch } from './native.mjs';
import { startFixtures } from './fixtures.mjs';

const identity = studyIdentity(), native = await openResearch(identity);
const fixtures = await startFixtures({ extensionOrigins: [`chrome-extension://${native.extensionId}`] });
const page = await native.context.newPage();
await nativeWindow(page, native.context);
await page.goto(`${fixtures.url}?extension=${native.extensionId}`);
await page.evaluate(() => globalThis.__M1010_SOURCE__.prepare({ mode: 'nocors', asset: 'A' }));
await page.locator('#play').click(); await page.bringToFront();
const report = { schemaVersion: 1, phase: 'MANUAL_CONSENT_TOPOLOGY', identity, media: fixtures.media,
  started: new Date().toISOString(), browser: { version: await native.browser.version(),
    executableSha256: sha256(readFileSync(native.executable)), policy: 'default; owner operates extension action and permission/capture prompts' }, steps: [] };
const prefix = process.argv[2] ?? '.cache/m1010/manual-01'; assert(prefix.startsWith('.cache/m1010/') && !existsSync(`${prefix}.json`));
const save = () => writeFileSync(`${prefix}.progress.json`, JSON.stringify(report, null, 2));
const player = () => native.context.pages().find(page => page.url() === `chrome-extension://${native.extensionId}/acquire.html`);
const target = () => native.context.pages().find(page => page.url().startsWith(`chrome-extension://${native.extensionId}/target.html`));
console.log(JSON.stringify({ ready: true, sourceUrl: page.url(), extensionId: native.extensionId,
  instruction: 'Owner: use Extensions menu -> AetherVSR M10.10 Research -> Select source and open acquisition. Permission prompts must be operated manually.' }));
const input = createInterface({ input: process.stdin, output: process.stdout });
try {
  for await (const line of input) {
    let command;
    const step = { at: new Date().toISOString(), command: line }; report.steps.push(step);
    try {
      command = JSON.parse(line);
      if (command.type === 'finish') { step.result = 'closing'; break; }
      if (command.type === 'snapshot') {
        step.result = { source: await page.evaluate(() => globalThis.__M1010_SOURCE__.snapshot()),
          player: player() ? await player().evaluate(() => globalThis.m1010Acquire.snapshot()) : null,
          target: target() ? await target().evaluate(() => globalThis.m1010Targets.snapshot()) : null,
          grants: await native.worker.evaluate(() => chrome.permissions.getAll()) };
      } else if (command.type === 'prepare') {
        if (player()) await player().close();
        await page.bringToFront();
        step.result = await page.evaluate(command => globalThis.__M1010_SOURCE__.prepare({ mode: command.mode, asset: command.asset ?? 'A' }), command);
        await page.locator('#play').click();
      } else if (command.type === 'player') {
        const consumer = player(); assert(consumer, 'Owner must first activate the research extension'); await consumer.bringToFront();
        assert(['direct', 'refetch', 'rtc', 'tab', 'probe', 'stop', 'refresh'].includes(command.action));
        step.result = await consumer.evaluate(async command => { const result = await globalThis.m1010Acquire[command.action](command.argument); return { result, snapshot: globalThis.m1010Acquire.snapshot() }; }, command);
      } else if (command.type === 'source') {
        step.result = await page.evaluate(command => globalThis.__M1010_SOURCE__.action(command.action), command);
      } else if (command.type === 'focus') {
        const selected = command.surface === 'source' ? page : command.surface === 'target' ? target() : player(); assert(selected); await selected.bringToFront(); step.result = { url: selected.url() };
      } else if (command.type === 'target-capture') {
        const consumer = target(); assert(consumer, 'Owner must open related extension target');
        const tabs = await native.worker.evaluate(() => chrome.tabs.query({}));
        const source = tabs.find(tab => tab.url?.startsWith(fixtures.url)), destination = tabs.find(tab => tab.url === consumer.url());
        assert(source?.id !== undefined && destination?.id !== undefined);
        const id = await native.worker.evaluate(({ source, destination }) => new Promise((resolve, reject) => chrome.tabCapture.getMediaStreamId({ targetTabId: source, consumerTabId: destination }, value => {
          if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message)); else resolve(value);
        })), { source: source.id, destination: destination.id });
        await consumer.bringToFront(); step.result = await consumer.evaluate(id => globalThis.m1010Targets.start(id), id);
      } else if (command.type === 'target-apply') {
        assert(['crop', 'restriction'].includes(command.kind)); step.result = await target().evaluate(kind => globalThis.m1010Targets.apply(kind), command.kind);
      } else if (command.type === 'self-apply') {
        assert(['crop', 'restrict'].includes(command.kind)); step.result = await page.evaluate(async kind => { await globalThis.__M1010_SOURCE__[kind](); return globalThis.__M1010_SOURCE__.snapshot(); }, command.kind);
      } else if (command.type === 'stop-all') {
        if (player()) await player().evaluate(() => globalThis.m1010Acquire.stop());
        if (target()) await target().evaluate(() => globalThis.m1010Targets.stop());
        step.result = await page.evaluate(() => globalThis.__M1010_SOURCE__.action({ type: 'cancel' }));
      } else throw new Error('Unrecognized manual controller command');
    } catch (error) { step.error = String(error); }
    save(); console.log(JSON.stringify(step)); console.log('Ready for next bounded controller command; browser consent is never automated.');
  }
} finally {
  input.close();
  await native.close(); await fixtures.close(); report.finished = new Date().toISOString();
  writeFileSync(`${prefix}.json`, JSON.stringify(report, null, 2), { flag: 'wx' });
  console.log(JSON.stringify({ finished: true, saved: `${prefix}.json`, steps: report.steps.length }));
}