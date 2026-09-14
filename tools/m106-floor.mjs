import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve, join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { createServer } from 'vite';
import { ROOT, sha256 } from './m10-fixtures.mjs';
import { verifyBuild, openExtension, bounded, until } from './m10-browser.mjs';
import { openNativeChrome } from './m9-browser.mjs';
import { nativeWindow } from './m105-accounting.mjs';
import { installNativeObserver, installRuntimeCounters, summarizeNative, validateNative, COLUMNS } from './m106-counters.mjs';

export const SOURCE_SHA = '8d81acbe164da1d62b7d0d02a3cc66915c96e8aa90d45cac34d818fc33df1d4a';
export const BROWSER_SHA = '8319963f6625accf51c0dd4f55091ceaf9f09ed39e7a52fed4fae12b2a6b668a';
export const PRIMARY_ORDER = [...'ABDCBCADCDBADACB'];
export function parseFloorPlan(value) {
  assert(Array.isArray(value) && value.length > 0 && value.length <= 64, 'Expected fixed cases');
  const ids = new Set();
  const cases = value.map(item => {
    assert(item && Object.keys(item).every(key => ['id','arm','mode','durationMs','phase'].includes(key)), 'Unknown case field');
    assert(typeof item.id === 'string' && /^[\w-]{1,70}$/.test(item.id) && !ids.has(item.id), 'Invalid/duplicate id'); ids.add(item.id);
    assert(['A','B','C','D'].includes(item.arm) && ['none','lean','rich'].includes(item.mode), 'Unknown arm/mode');
    assert(['smoke','observer','primary','proxy'].includes(item.phase), 'Explicit phase required');
    assert(Number.isInteger(item.durationMs) && item.durationMs >= 1000 && item.durationMs <= 600000);
    if (item.phase === 'primary') assert(item.durationMs === 600000 && item.mode === 'lean');
    if (item.phase === 'observer') assert(item.arm === 'A' && item.durationMs === 60000);
    if (item.phase === 'proxy') assert(item.mode === 'rich' && item.durationMs === 12000 && item.arm !== 'B');
    return { ...item, warmupMs: 5000 };
  });
  assert(cases.every(item => item.phase === cases[0].phase), 'Do not mix evidence phases');
  if (cases[0].phase === 'primary') assert.deepEqual(cases.map(item => item.arm), PRIMARY_ORDER, 'Frozen Williams order required');
  if (cases[0].phase === 'observer') assert.deepEqual(cases.map(item => item.mode), ['none','lean','rich','lean','rich','none','rich','none','lean'], 'Frozen observer order required');
  if (cases[0].phase === 'proxy') assert.deepEqual(cases.map(item => item.arm), ['A','C','D'], 'Frozen proxy order required');
  return cases;
}

export async function floorServer() {
  const media = readFileSync(join(ROOT, 'public/media/m9/720p60.mp4')); assert.equal(sha256(media), SOURCE_SHA);
  const html = readFileSync(join(ROOT, 'tools/m105-fixture.html'));
  const server = await createServer({ root: ROOT, mode: 'benchmark', server: { host: '127.0.0.1', port: 5190, strictPort: true },
    plugins: [{ name: 'm106-native-fixture', configureServer(instance) {
      instance.middlewares.use((request, response, next) => {
        if (new URL(request.url, 'http://local').pathname !== '/m106') return next();
        response.setHeader('Content-Type', 'text/html; charset=utf-8'); response.end(html);
      });
    } }] });
  await server.listen();
  try {
    const response = await fetch('http://127.0.0.1:5190/m106'); assert.equal(sha256(Buffer.from(await response.arrayBuffer())), sha256(html));
    return { url: 'http://127.0.0.1:5190/m106?consumer=none', htmlSha256: sha256(html), mediaSha256: SOURCE_SHA, close: () => server.close() };
  } catch (error) { await server.close(); throw error; }
}

export async function activateArm(native, page, arm) {
  const panel = await native.popup(page);
  try {
    if (arm === 'B') {
      await native.workerEval(async tabId => chrome.scripting.executeScript({ target: { tabId, frameIds: [0] }, world: 'ISOLATED', files: ['content.js'] }), panel.tabId);
      await native.isolated(page, () => {
        globalThis.__AETHERVSR_EXTENSION_TEST__.configure({ processingDisabled: true });
        const original = navigator.gpu.requestAdapter;
        globalThis[Symbol.for('aethervsr.m106.adapterRequests')] = 0;
        navigator.gpu.requestAdapter = function(...args) { globalThis[Symbol.for('aethervsr.m106.adapterRequests')]++; return original.apply(this, args); };
      });
    }
    await panel.click('#enable');
    if (arm === 'C') await panel.click('input[value="baseline"]');
    return panel.tabId;
  } finally { await panel.dismiss(); }
}

export function qualifiesDisabledGeometry(opening, moved, restored) {
  return opening?.cssRect?.left===40 && moved?.cssRect?.left===50 && restored?.cssRect?.left===40
    && moved.snapshot.infrastructure.geometryCalls>opening.snapshot.infrastructure.geometryCalls
    && [opening,moved,restored].every(value=>value.connected&&value.visibility==='hidden'&&value.pointerEvents==='none'
      &&value.snapshot.resources.device===0&&value.snapshot.resources.pipeline===0&&value.snapshot.resources.frameCallback===0);
}

export function activeSafety(raw, arm, cleanup, geometry) {
  if (arm === 'A') return { applicable: false, pass: null };
  const before = raw.runtime?.opening, after = raw.runtime?.closing;
  const first = before?.status?.details?.infrastructure, last = after?.status?.details?.infrastructure;
  const owners = first && last && first.ownerChanges === last.ownerChanges && first.created === last.created && last.maximumConcurrent === 1;
  const resources = cleanup?.attachment?.resources;
  const clean = !!resources && Object.values(resources).every(value => value === 0)
    && cleanup.attachment.infrastructure.cleanupErrors === 0 && cleanup.status?.enabled === false
    && cleanup.status.details.timerCount === 0 && cleanup.status.details.discoveryActive === false && cleanup.domPreserved === true;
  if (arm === 'B') {
    const liveResources = after?.status?.details?.attachment?.resources;
    return { applicable: true, pass: !!owners && raw.adapterRequests === 0 && after?.sameAttachment === true
      && before?.status?.enabled === true && after?.status?.enabled === true && after.status.details.discoveryActive === true
      && after.status.details.attachment.suspendedReason === 'diagnostic-processing-disabled'
        && liveResources?.device === 0 && liveResources?.pipeline === 0 && liveResources?.frameCallback === 0 && liveResources?.canvas === 1
        && liveResources?.resizeObservers === 1 && before.runtime === null && after.runtime === null && clean && geometry?.pass === true,
        ownerStable: !!owners, cleanup: clean, geometry: geometry?.pass === true };
  }
  const summary = summarizeNative(raw), target = arm === 'C' ? 'baseline' : 'neural';
  const tier = before?.runtime?.actualTier === target && after?.runtime?.actualTier === target
    && raw.runtime.frames.every(row => row[2] === Number(arm === 'D')) && raw.runtime.states.every(row => row[2] === target);
  const passesRate = value => Number.isFinite(value) && value >= 58;
  const checks = { ownerStable: !!owners, tierStable: tier, noSubmissionDeficit: summary.submissionDeficit === 0,
    noError: after?.pipelineError === null && after?.runtime?.controller.state !== 'failed' && raw.errors.length === 0 && !raw.native.failures?.length,
    averageRendered: passesRate(summary.renderedFps), averagePresented: passesRate(summary.runtimePresentedFps),
    lastRendered: passesRate(summary.last120?.renderedFps), lastPresented: passesRate(summary.last120?.runtimePresentedFps), cleanup: clean };
  return { applicable: true, pass: Object.values(checks).every(Boolean), checks };
}

export async function runFloor(planPath, prefix) {
  prefix = resolve(prefix); assert(relative(join(ROOT,'.cache/m106'),prefix) && !relative(join(ROOT,'.cache/m106'),prefix).startsWith('..'));
  assert(!existsSync(`${prefix}.json`), 'Never overwrite evidence'); mkdirSync(dirname(prefix), { recursive: true });
  const bytes = readFileSync(planPath), cases = parseFloorPlan(JSON.parse(bytes));
  const production = verifyBuild(false), diagnostic = cases.some(item => item.arm === 'B') ? verifyBuild(true) : null;
  const sourcePins = Object.fromEntries(['tools/m106-floor.mjs','tools/m106-counters.mjs','tools/m105-fixture.html','tools/m105-accounting.mjs','tools/m10-browser.mjs','tools/m9-browser.mjs','docs/M10.6-PREREGISTRATION.md']
    .map(path => [path,sha256(readFileSync(join(ROOT,path)))]));
  const report = { schemaVersion: 1, started: new Date().toISOString(), completion: 'RUNNING', cases, casesSha256: sha256(bytes), production, diagnostic, sourcePins,
    machine: { os: execFileSync('sw_vers',['-productVersion'],{encoding:'utf8'}).trim(), chip: 'Apple M5', memory: '24 GB',
      power: execFileSync('pmset',['-g','batt'],{encoding:'utf8'}).trim(),
      displays: JSON.parse(execFileSync('osascript',['-l','JavaScript','-e','ObjC.import("AppKit"); const screens=$.NSScreen.screens;const result=[];for(let index=0;index<screens.count;index++){const display=screens.objectAtIndex(index);result.push({name:display.localizedName.js,frame:display.frame,scale:display.backingScaleFactor});}JSON.stringify(result);'],{encoding:'utf8'})), physicalRefresh: null }, results: [],
    scope: 'Common native-observer counters. No unique physical loss or observer-free callback claim. C/D have a minimal actual attempt/submission hook; B is explicitly diagnostic infrastructure without GPU. All raw local; rates use common native wall window.' };
  assert(report.machine.power.includes('AC Power'), 'AC power required');
  let server, native;
  try {
    server = await floorServer(); report.fixture = { htmlSha256: server.htmlSha256, mediaSha256: server.mediaSha256 };
    for (const item of cases) {
      const result = { case: item, completion: 'UNVERIFIED', errors: [], events: [] }; report.results.push(result);
      const record = (name,data) => result.events.push({name,data});
      native = item.arm === 'A' ? await openNativeChrome(['--autoplay-policy=no-user-gesture-required']) : await openExtension(item.arm === 'B' ? diagnostic : production, record);
      try {
        native.context.setDefaultTimeout(10000); native.context.setDefaultNavigationTimeout(15000);
        const page = await native.context.newPage();
        result.placement = await nativeWindow(page, native.context);
        const cdp = await native.context.newCDPSession(page);
        try { result.browser = { version: await cdp.send('Browser.getVersion'), executableSha256: sha256(readFileSync(native.executable)) }; }
        finally { await cdp.detach(); }
        assert.equal(result.browser.executableSha256, BROWSER_SHA);
        page.on('pageerror', error => result.errors.push(String(error)));
        page.on('console', message => { if (message.type() === 'error') result.errors.push(message.text()); });
        await page.addInitScript(installNativeObserver, { mode: item.mode });
        await page.goto(server.url); await page.bringToFront();
        await page.waitForFunction(mode => { const data=globalThis[Symbol.for('aethervsr.m106.native')],video=document.querySelector('video'); return data?.ready && video.readyState>=2 && !video.paused && (mode==='none'||data.callbacks>=2); }, item.mode);
        const before = await page.evaluate(() => ({html:document.getElementById('player').innerHTML,source:document.querySelector('video').currentSrc}));
        if (item.arm !== 'A') {
          result.tabId = await activateArm(native,page,item.arm);
          result.ready = await until(() => native.inspect(result.tabId), state => item.arm === 'B'
            ? state.details?.attachment?.suspendedReason === 'diagnostic-processing-disabled' && !!state.owner
            : state.code === 'active' && state.details?.attachment?.ready && state.current === (item.arm === 'C'?'baseline':'neural')
              && state.details.attachment.controllerState === (item.arm === 'C'?'manual-baseline':'stable'), 20000);
          result.runtimeSetup = await native.isolated(page,installRuntimeCounters);
        }
        result.geometry = await page.evaluate(() => { const video=document.querySelector('video');return {video:video.getBoundingClientRect().toJSON(), canvases:document.querySelectorAll('canvas').length,source:video.currentSrc,style:getComputedStyle(document.body).backgroundColor}; });
        assert.equal(result.geometry.video.x,40);assert.equal(result.geometry.video.y,120);assert.equal(result.geometry.video.width,640);assert.equal(result.geometry.video.height,360);
        assert.equal(result.geometry.canvases,item.arm==='A'?0:1);
        if (item.phase === 'proxy') {
          await page.evaluate(() => {const marker=document.createElement('div');marker.id='m106-marker';marker.style.cssText='position:fixed;left:40px;top:90px;width:12px;height:16px;background:#cf1020';document.body.append(marker);});
          await page.screenshot({path:`${prefix}.${item.id}.before.png`});
        }
        await page.evaluate(milliseconds=>new Promise(done=>setTimeout(done,milliseconds)),item.warmupMs);
        await page.evaluate(milliseconds=>globalThis[Symbol.for('aethervsr.m106.native')].start(milliseconds),item.durationMs);
        console.log(`${item.id}: recording ${item.durationMs/1000}s ${item.arm}/${item.mode}`);
        await bounded(page.evaluate(()=>globalThis[Symbol.for('aethervsr.m106.native')].done),item.durationMs+15000,'Native observation');
        const observation = await page.evaluate(()=>{const {done,start,...data}=globalThis[Symbol.for('aethervsr.m106.native')];return data;});
        const runtime = item.arm==='A'?null:await native.isolated(page,()=>{
          const {teardownSnapshot,...data}=globalThis[Symbol.for('aethervsr.m106.runtime')];return data;
        });
        const adapterRequests = item.arm==='B'?await native.isolated(page,()=>globalThis[Symbol.for('aethervsr.m106.adapterRequests')]):null;
        const cdpEnd=await native.context.newCDPSession(page);
        const closingPlacement={bounds:(await cdpEnd.send('Browser.getWindowForTarget')).bounds,screen:await page.evaluate(()=>({width:screen.width,height:screen.height,dpr:devicePixelRatio,x:screenX,y:screenY}))};await cdpEnd.detach();
        const raw={case:item,native:observation,runtime,adapterRequests,errors:result.errors,columns:COLUMNS,closingPlacement};
        const packed=gzipSync(JSON.stringify(raw));const rawPath=`${prefix}.${item.id}.json.gz`;writeFileSync(rawPath,packed,{flag:'wx'});result.raw={path:relative(ROOT,rawPath),sha256:sha256(packed),bytes:packed.length};
        try {
          result.summary=validateNative(raw,item.durationMs);
          assert.deepEqual(closingPlacement.screen,result.placement.nativeScreen);assert.deepEqual(closingPlacement.bounds,result.placement.bounds.bounds);
          result.completion='CAPTURED';
        } catch(error) {result.error=String(error);}
        if (item.phase==='proxy')await page.screenshot({path:`${prefix}.${item.id}.after.png`});
        if (item.arm==='B') {
          const geometryState=()=>native.isolated(page,()=>{
            const attachment=globalThis.__AETHERVSR_EXTENSION_TEST__.attachment();
            return {snapshot:attachment.snapshot(),connected:attachment.canvas.isConnected,visibility:getComputedStyle(attachment.canvas).visibility,
              pointerEvents:getComputedStyle(attachment.canvas).pointerEvents,cssRect:attachment.snapshot().cssRect};
          });
          const opening=await geometryState();
          await page.evaluate(()=>document.getElementById('player').style.left='50px');
          let moved, restored;
          try {moved=await until(geometryState,value=>value.cssRect?.left===50,3000);}
          catch(error){result.geometryQualificationError=String(error);}
          finally {await page.evaluate(()=>document.getElementById('player').style.removeProperty('left'));}
          try {restored=await until(geometryState,value=>value.cssRect?.left===40,3000);}
          catch(error){result.geometryQualificationError=String(error);}
          result.geometryQualification={opening,moved,restored,pass:qualifiesDisabledGeometry(opening,moved,restored)};
        }
        if (item.arm!=='A') {
          try {
            const panel=await native.popup(page);try{result.teardown=await panel.click('#disable');}finally{await panel.dismiss();}
            result.teardownAttachment=await native.isolated(page,()=>globalThis[Symbol.for('aethervsr.m106.runtime')].teardownSnapshot());
          } catch(error){result.teardownError=String(error);}
        }
        const after=await page.evaluate(()=>({html:document.getElementById('player').innerHTML,source:document.querySelector('video').currentSrc}));
        result.domPreserved=after.html===before.html&&after.source===before.source;
        result.safety=activeSafety(raw,item.arm,{status:result.teardown,attachment:result.teardownAttachment,domPreserved:result.domPreserved},result.geometryQualification);
        if(item.arm==='A'&&!result.domPreserved){result.completion='UNVERIFIED';result.error='Native source DOM changed';}
        console.log(`${item.id}: ${result.completion}; native L=${result.summary?.nativeCombinedPercent??'not measured'}, callback=${result.summary?.nativeCallbackFps??'not measured'}, submitted=${result.summary?.renderedFps??'not measured'}`);
        if(result.completion!=='CAPTURED')throw new Error(`Invalid case ${item.id}; raw retained; remaining order not run`);
      } finally {await native.close();native=null;}
    }
    assert.deepEqual(verifyBuild(false),production);if(diagnostic)assert.deepEqual(verifyBuild(true),diagnostic);
    for(const [path,digest]of Object.entries(sourcePins))assert.equal(sha256(readFileSync(join(ROOT,path))),digest);
    assert.equal(sha256(readFileSync(planPath)),report.casesSha256);report.completion='CAPTURED';
  } catch(error) {report.error=String(error);report.completion='UNVERIFIED';throw error;}
  finally {await native?.close();await server?.close();report.ended=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  assert(process.argv.length===4,'Usage: node tools/m106-floor.mjs CASES.json .cache/m106/PREFIX');
  await runFloor(resolve(process.argv[2]),resolve(process.argv[3]));
}