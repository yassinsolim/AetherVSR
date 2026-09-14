import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, resolve, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { ROOT, sha256 } from './m10-fixtures.mjs';
import { openNativeChrome } from './m9-browser.mjs';
import { openExtension, verifyBuild, until, bounded, OperationTimeout } from './m10-browser.mjs';
import { nativeWindow } from './m105-accounting.mjs';
import { BROWSER_SHA, activateArm, resumePrefix } from './m106-floor.mjs';
import { SITES, siteSnapshot, geometryCheck, playerButton, visibleButton, consent, safeEvidence, errorObserver, nativeResize } from './m10-sites.mjs';

export const PUBLIC_ORDER = ['PQSR','QRPS','SPRQ'].flatMap(block => [...block]);
export const VIDEOJS_DURATION = 35.963044;
export const SEEK_TARGET = 34.963044;

export function publicCases() {
  return PUBLIC_ORDER.map((arm, index) => ({ id: `videojs-${index + 1}-${arm}`, arm, durationMs: 180000 }));
}

export function advancedForOpening(state, initialTime) {
  return state?.paused === false && Number.isFinite(state.time) && Number.isFinite(initialTime) && state.time - initialTime >= 2;
}

export function loadCommittedReference(path) {
  const name = relative(ROOT,resolve(path));
  assert(name.startsWith('results/') && !name.includes('..'), 'Reference must be a committed results artifact');
  const bytes = readFileSync(path);
  const committed = execFileSync('git',['show',`HEAD:${name}`],{cwd:ROOT,stdio:['ignore','pipe','pipe']});
  assert.deepEqual(bytes,committed,'Reference bytes differ from committed HEAD');
  const record = JSON.parse(bytes);
  assert.equal(record.completion,'PINNED');assert.equal(record.phase,'METADATA_ONLY');
  assert(samePublicIdentity(record.identity,record.identity));
  return {bytes,record};
}

export function retainBindingPublicFailure(item) {
  return item.bindingFailure === true || ['extension','gpu-unattributed','cors-import'].some(category => (item.errors?.counts?.[category]??0)>0);
}

export function publicStateFailure(status, owner) {
  const attachment=status?.details?.attachment,infrastructure=status?.details?.infrastructure;
  return !status || status.enabled!==true || status.owner!==owner || !['active','suspended'].includes(status.code)
    || !attachment || attachment.infrastructure.cleanupErrors!==0 || infrastructure?.maximumConcurrent!==1
    || infrastructure.created-infrastructure.destroyed!==1
    || ['device','pipeline','canvas','resizeObservers'].some(name=>attachment.resources[name]!==1);
}

export async function publicIdentity(page) {
  return page.locator('video').first().evaluate(async video => {
    const source = new URL(video.currentSrc);
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source.href));
    return { origin: source.origin, urlSha256: [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join(''),
      duration: Number.isFinite(video.duration) ? video.duration : null, width: video.videoWidth, height: video.videoHeight };
  });
}

export function installPublicObserver() {
  const key = Symbol.for('aethervsr.m106.public');
  if (globalThis[key]) throw new Error('Duplicate public observer');
  const data = globalThis[key] = { rows: [], events: [], overflow: false, invalid: null, action: null };
  let video, source, active = false, timer;
  const listeners = [];
  const on = (target, type, callback) => { target.addEventListener(type, callback); listeners.push([target,type,callback]); };
  const push = (values, value) => { if (values.length < 10000) values.push(value); else data.overflow = true; };
  const sample = () => {
    const quality = video.getVideoPlaybackQuality?.();
    return { at: performance.now(), currentTime: video.currentTime, qualityTotal: quality?.totalVideoFrames ?? null,
      qualityDrops: quality?.droppedVideoFrames ?? null, readyState: video.readyState, networkState: video.networkState,
      paused: video.paused, ended: video.ended, seeking: video.seeking, rate: video.playbackRate,
      visibility: document.visibilityState, focused: document.hasFocus(), sameVideo: video === document.querySelector('video'),
      sameSource: source === video.currentSrc, error: video.error?.code ?? null, action: data.action };
  };
  data.start = () => {
    video = document.querySelector('video');
    if (!video || active || data.opening) throw new Error('Public observer not startable');
    source = video.currentSrc; data.opening = sample(); active = true;
    for (const type of ['waiting','stalled','playing','seeking','seeked','ended','pause','play','loadstart','error','ratechange']) {
      on(video, type, () => {
        if (!active) return;
        push(data.events, { type, ...sample() });
        const commanded = ['original-pause-resume','original-ended-replay'].includes(data.action);
        if (['pause','play'].includes(type) && !commanded && !video.ended && !video.error) data.invalid ??= `Unexpected media event: ${type}`;
        if (['loadstart','ratechange'].includes(type)) data.invalid ??= `Media identity event: ${type}`;
      });
    }
    for (const [target,type] of [[window,'blur'],[window,'pagehide'],[window,'resize'],[document,'visibilitychange']]) {
      on(target,type,() => {
        if (!active) return;
        push(data.events,{type,...sample()});
        const controlledResize = type === 'resize' && (data.action === 'original-fullscreen' || data.action?.startsWith('resize-'));
        if (!controlledResize) data.invalid ??= `External integrity event: ${type}`;
      });
    }
    timer = setInterval(() => push(data.rows, sample()), 250);
    push(data.rows, data.opening);
    return data.opening;
  };
  data.stop = () => {
    if (active) { data.closing = sample(); push(data.rows,data.closing); }
    active = false; clearInterval(timer);
    for (const [target,type,callback] of listeners.splice(0)) target.removeEventListener(type,callback);
    const {start,stop,...result} = data;
    return result;
  };
}

function localPrefix(prefix) {
  prefix = resolve(prefix);
  const path = relative(join(ROOT,'.cache/m106'),prefix);
  assert(path && !path.startsWith('..'), 'Public raw evidence belongs under .cache/m106');
  assert(!existsSync(`${prefix}.json`), 'Never overwrite public evidence');
  mkdirSync(dirname(prefix),{recursive:true});
  return prefix;
}

async function browserIdentity(native, page) {
  const cdp = await native.context.newCDPSession(page);
  try {
    const identity = { version: await cdp.send('Browser.getVersion'), executableSha256: sha256(readFileSync(native.executable)) };
    assert.equal(identity.executableSha256,BROWSER_SHA); return identity;
  } finally { await cdp.detach(); }
}

export async function pinVideojsIdentity(prefix) {
  prefix = localPrefix(prefix);
  const build = verifyBuild(false);
  const native = await openNativeChrome();
  const record = { schemaVersion: 1, phase: 'METADATA_ONLY', started: new Date().toISOString(), build, origin: new URL(SITES.videojs.url).origin,
    outcome: 'not measured', scope: 'No Play command, callback-loss observation, or journey outcome. Full media URL and frames are not retained.' };
  try {
    const page = await native.context.newPage();
    record.placement = await nativeWindow(page,native.context); record.browser = await browserIdentity(native,page);
    const response = await page.goto(SITES.videojs.url,{waitUntil:'domcontentloaded',timeout:30000});
    assert(response?.ok()); await consent(page,()=>{});
    await page.waitForFunction(()=>{const video=document.querySelector('video');return video?.readyState>=1&&video.videoWidth>0&&video.currentSrc;},undefined,{timeout:45000});
    record.identity = await publicIdentity(page);
    assert(samePublicIdentity(record.identity,record.identity),'Historical asset duration changed; prospective revision required');
    assert.deepEqual(verifyBuild(false),build); record.completion = 'PINNED';
  } catch(error) { record.completion='UNVERIFIED';record.error=safeEvidence(String(error));throw error; }
  finally { await native.close();writeFileSync(`${prefix}.json`,JSON.stringify(record,null,2),{flag:'wx'}); }
  return record;
}

function controlSummary(item) {
  const required = ['prescribed-seek','original-pause-resume','scroll-down','scroll-back','resize-1024','resize-1200','original-fullscreen'];
  const controls = required.every(name => item.actions.some(action => action.name===name && (action.pass===true || action.notApplicable===true)))
    && item.actions.every(action => action.pass===true || action.notApplicable===true);
  const clean = item.cleanup?.sameVideo === true && item.cleanup?.noCanvas === true
    && (!['R','S'].includes(item.case.arm) || item.cleanup?.pass === true);
  const extensionErrors = ['extension','gpu-unattributed','cors-import','unattributed'].some(category => (item.errors.counts[category]??0)>0);
  item.safetyPass = item.comparable === true && controls && clean && !extensionErrors && item.failures.length===0;
  item.journeyPass = item.safetyPass && item.stalls?.length===0;
  item.recoveryComparison = !item.stalls || item.stalls.length ? 'UNRESOLVED' : 'NO_OBSERVED_WORSENING';
  return { controls, clean, extensionErrors };
}

export async function runVideojs(prefix, referencePath, resumePath) {
  prefix=localPrefix(prefix);
  const {bytes:referenceBytes,record:referenceRecord}=loadCommittedReference(referencePath),reference=referenceRecord.identity;
  assert.equal(referenceRecord.completion,'PINNED');assert(samePublicIdentity(reference,reference));
  const build=verifyBuild(false),cases=publicCases();
  const report={schemaVersion:1,started:new Date().toISOString(),completion:'RUNNING',cases,build,
    reference:{path:relative(ROOT,referencePath),sha256:sha256(referenceBytes),identity:reference},results:[],
    scope:'Fixed Video.js P/Q/R/S control study. Original 180s interaction schedule; source stalls are retained. No public frames or full media URLs saved. Right-censored recovery alone cannot satisfy a stall waiver.'};
  if(resumePath){
    const bytes=readFileSync(resumePath),prior=JSON.parse(bytes);assert.deepEqual(prior.build,build);assert.deepEqual(prior.reference,report.reference);
    report.results.push(...resumePrefix(prior,cases));
    for(const result of report.results)assert.equal(sha256(readFileSync(join(ROOT,result.raw.path))),result.raw.sha256);
    report.resume={path:relative(ROOT,resumePath),sha256:sha256(bytes),acceptedOrdinals:report.results.length,interrupted:prior.results.at(-1)};
  }
  let native;
  try {
    for(const item of cases.slice(report.results.length)){
      const result={case:item,completion:'UNVERIFIED',comparable:false,actions:[],checks:[],failures:[],errors:{counts:{},samples:[],omittedSamples:0},events:[],stalls:[]};report.results.push(result);
      const record=(name,data)=>result.events.push({name,data:safeEvidence(data)});
      native=item.arm==='P'?await openNativeChrome():await openExtension(build,record);
      let page,tabId,stopErrors,raw,opening,work;
      const active=['R','S'].includes(item.arm);
      try {
        work=(async()=>{
        page=await native.context.newPage();native.context.setDefaultTimeout(5000);
        result.placement=await nativeWindow(page,native.context);result.browser=await browserIdentity(native,page);
        stopErrors=errorObserver(page,native.extensionId??'not-installed',result);
        await page.addInitScript(installPublicObserver);
        const response=await page.goto(SITES.videojs.url,{waitUntil:'domcontentloaded',timeout:30000});assert(response?.ok());
        await page.bringToFront();await consent(page,record);
        await page.waitForFunction(()=>{const video=document.querySelector('video');return video?.readyState>=2&&video.videoWidth>0;},undefined,{timeout:45000});
        result.identity=await publicIdentity(page);assert(samePublicIdentity(result.identity,reference),'Reference asset changed; no substitute seek');
        await page.evaluate(()=>{globalThis[Symbol.for('aethervsr.m106.original')]=document.querySelector('video');});
        const before=await page.evaluate(()=>({paused:document.querySelector('video').paused,time:document.querySelector('video').currentTime}));
        if(before.paused)await playerButton(page,SITES.videojs,'play');
        else result.playPreparation='Original page was already playing; no scripted play or pause';
        const advancingFrom=before.time;
        if(active){
          try {
            tabId=await activateArm(native,page,item.arm==='R'?'C':'D');
            result.openingStatus=await until(()=>native.inspect(tabId),status=>status.code==='active'&&status.current===(item.arm==='R'?'baseline':'neural')
              &&status.details?.attachment?.controllerState===(item.arm==='R'?'manual-baseline':'stable'),20000);
          } catch(error){result.bindingFailure=true;result.failures.push(`Required active opening unavailable: ${safeEvidence(String(error))}`);}
          try{
            await native.isolated(page,()=>{
              const manager=globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)],attachment=manager?.attachment;
              globalThis[Symbol.for('aethervsr.m106.public-attachment')]=()=>attachment?.snapshot()??null;
            });
          }catch(error){result.bindingFailure=true;result.failures.push(`Attachment inspection failed: ${safeEvidence(String(error))}`);}
        }else if(item.arm==='Q'){
          tabId=await native.tabId(page);
          result.inactive=await native.workerEval(async id=>({registered:!!(await chrome.storage.session.get(`m10.document.${id}`))[`m10.document.${id}`]}),tabId);
          assert.equal(result.inactive.registered,false,'Installed control was activated');
        }
        await until(()=>page.locator('video').first().evaluate(video=>({time:video.currentTime,paused:video.paused})),value=>advancedForOpening(value,advancingFrom),8000);
        const owner=result.openingStatus?.owner;
        const check=async label=>{
          const dom=await bounded(page.evaluate(siteSnapshot,native.extensionId??'not-installed'),5000,'Public geometry snapshot');
          let status=null;
          if(active&&tabId!==undefined){try{status=await native.inspect(tabId);}catch(error){result.bindingFailure=true;result.failures.push(`Inspection rejected: ${safeEvidence(String(error))}`);}}
          if(active&&publicStateFailure(status,owner))result.bindingFailure=true;
          const value={label,dom:safeEvidence(dom),status:safeEvidence(status)};result.checks.push(value);
          try{
            assert.equal(dom.focused,true);assert.equal(dom.visibility,'visible');assert.equal(dom.origin,new URL(SITES.videojs.url).origin);
            assert.equal(dom.mainSingleton,'undefined');assert.equal(dom.mainTestHook,'undefined');
            if(active){
              try{geometryCheck(status,dom,owner);}
              catch(error){result.bindingFailure=true;throw error;}
            }else assert.equal(dom.canvases.length,0);
          }catch(error){result.failures.push(`${label}: ${safeEvidence(String(error))}`);}
          return {dom,status};
        };
        await check('opening');
        opening=await page.evaluate(()=>globalThis[Symbol.for('aethervsr.m106.public')].start());
        const action=async(name,scheduledMs,operation)=>{
          const entry={name,scheduledMs,at:await page.evaluate(()=>performance.now()),pass:false};result.actions.push(entry);
          await page.evaluate(name=>{globalThis[Symbol.for('aethervsr.m106.public')].action=name;},name);
          try{entry.result=safeEvidence(await operation());entry.pass=true;entry.notApplicable=entry.result?.notApplicable===true;}
          catch(error){entry.error=safeEvidence(String(error));}
          finally{entry.finishedAt=await page.evaluate(()=>performance.now());await page.evaluate(()=>{globalThis[Symbol.for('aethervsr.m106.public')].action=null;});}
          await check(`${name}-after`);
        };
        let scrollBack;
        const schedule=[
          [30000,'prescribed-seek',async()=>page.locator('video').first().evaluate((video,target)=>{
            if(Math.abs(video.duration-35.963044)>0.01||!Array.from({length:video.seekable.length},(_,index)=>[video.seekable.start(index),video.seekable.end(index)]).some(([start,end])=>start<=target&&end>=target))throw new Error('Prescribed seek unavailable');
            const from=video.currentTime;video.currentTime=target;return {from,target,method:'Original currentTime within seekable range'};
          },SEEK_TARGET)],
          [50000,'original-pause-resume',async()=>{
            await playerButton(page,SITES.videojs,'pause');await until(()=>page.locator('video').first().evaluate(video=>video.paused),Boolean,4000);
            const paused=await check('settled-original-pause'),frames=paused.status?.details?.attachment?.session.framesRendered;
            if(active)assert.equal(paused.status.details.attachment.active,false);
            await delay(1500);
            const held=active?await native.inspect(tabId):null;
            if(active)assert.equal(held.details.attachment.session.framesRendered,frames,'Processing continued during settled pause');
            await playerButton(page,SITES.videojs,'play');await until(()=>page.locator('video').first().evaluate(video=>!video.paused),Boolean,4000);
            return {holdMs:1500,originalControls:true};
          }],
          [75000,'scroll-down',async()=>{scrollBack=await page.evaluate(()=>({x:scrollX,y:scrollY}));return page.evaluate(()=>{const from=scrollY;scrollBy({top:innerHeight*.3,behavior:'instant'});return {from,to:scrollY,notApplicable:Math.abs(scrollY-from)<1};});}],
          [80000,'scroll-back',async()=>{assert(scrollBack);const result=await page.evaluate(position=>{scrollTo({left:position.x,top:position.y,behavior:'instant'});return {x:scrollX,y:scrollY,restored:Math.abs(scrollX-position.x)<1&&Math.abs(scrollY-position.y)<1};},scrollBack);assert.equal(result.restored,true);return result;}],
          [100000,'resize-1024',()=>nativeResize(native,page,1024)],
          [115000,'resize-1200',()=>nativeResize(native,page,1200)],
          [145000,'original-fullscreen',async()=>{
            const player=page.locator(SITES.videojs.player).filter({has:page.locator('video')}).first();await player.hover();
            const button=await visibleButton(player,/^(Enter )?Full\s*screen(?: mode)?$/i);assert(button,'Original fullscreen control unavailable');
            await button.click();
            try{await page.waitForFunction(()=>!!document.fullscreenElement,undefined,{timeout:4000});await check('original-fullscreen');await delay(2000);}
            finally{if(await page.evaluate(()=>!!document.fullscreenElement)){await page.keyboard.press('Escape');await page.waitForFunction(()=>!document.fullscreenElement,undefined,{timeout:5000});}}
            return {originalControl:true,holdMs:2000,exit:'Escape'};
          }],
        ];
        console.log(`${item.id}: recording 180s public controls`);
        let index=0,endedHandled=false;
        while(true){
          const state=await bounded(page.evaluate(()=>{const data=globalThis[Symbol.for('aethervsr.m106.public')],video=document.querySelector('video');return {at:performance.now(),ended:video.ended,invalid:data.invalid};}),5000,'Public observation tick');
          if(state.invalid)throw new Error(state.invalid);
          const elapsed=state.at-opening.at;if(elapsed>=item.durationMs)break;
          if(schedule[index]&&elapsed>=schedule[index][0]){const [at,name,operation]=schedule[index++];await action(name,at,operation);continue;}
          if(state.ended&&!endedHandled){endedHandled=true;await action('original-ended-replay',null,async()=>{await playerButton(page,SITES.videojs,'play');await until(()=>page.locator('video').first().evaluate(video=>!video.paused&&!video.ended),Boolean,5000);return {genuinelyEnded:true,originalControl:true};});}
          if(!state.ended)endedHandled=false;
          await check('observation');await delay(Math.min(1000,Math.max(1,item.durationMs-elapsed)));
        }
        raw=await page.evaluate(()=>globalThis[Symbol.for('aethervsr.m106.public')].stop());
        result.observedMs=raw.closing.at-raw.opening.at;result.closingIdentity=await publicIdentity(page);
        assert.equal(raw.invalid,null);assert.equal(raw.overflow,false);assert(result.observedMs>=item.durationMs&&result.observedMs<=item.durationMs+5000);
        assert(raw.rows.every(row=>row.sameVideo&&row.sameSource&&row.rate===1),'Original media identity/rate changed');
        assert(samePublicIdentity(result.closingIdentity,reference),'Closing reference identity changed');
        result.comparable=!result.bindingFailure&&result.actions.some(value=>value.name==='prescribed-seek'&&value.pass===true);
        result.stalls=targetStalls(raw.rows,result.actions,raw.closing.at);
        result.completion='CAPTURED';
        })();
        await bounded(work,300000,'Video.js trial deadline');
      }catch(error){
        result.error=safeEvidence(String(error));
        if(retainBindingPublicFailure(result)){result.completion='CAPTURED';result.observation='INCOMPLETE_BINDING_FAILURE';result.comparable=false;result.stalls=null;}
        if(error instanceof OperationTimeout){await native.close();await bounded(work.catch(()=>{}),3000,'Drain failed public trial').catch(()=>{});}
      }
      finally{
        if(page&&!page.isClosed()){
          try{raw??=await bounded(page.evaluate(()=>globalThis[Symbol.for('aethervsr.m106.public')]?.stop()),3000,'Retain public observer');}catch(error){result.rawError=safeEvidence(String(error));}
          try{
            await bounded((async()=>{
            let status,attachment;
            if(active&&tabId!==undefined){const panel=await native.popup(page);try{status=await panel.click('#disable');}finally{await panel.dismiss();}
              attachment=await native.isolated(page,()=>globalThis[Symbol.for('aethervsr.m106.public-attachment')]?.()??null);}
            const dom=await page.evaluate(()=>({sameVideo:globalThis[Symbol.for('aethervsr.m106.original')]===document.querySelector('video'),noCanvas:!document.querySelector('canvas[data-aethervsr-m10]')}));
            result.cleanup={...dom,status:safeEvidence(status),attachment:safeEvidence(attachment),pass:active?!!attachment&&attachment.infrastructure.cleanupErrors===0
              &&Object.values(attachment.resources).every(value=>value===0)&&status.enabled===false&&status.owner===null&&status.details.timerCount===0&&status.details.discoveryActive===false:true};
            })(),15000,'Independent public teardown deadline');
          }catch(error){result.cleanup={pass:false,error:safeEvidence(String(error))};}
        }
        {const bytes=gzipSync(JSON.stringify(safeEvidence({native:raw??null,checks:result.checks,errors:result.errors,actions:result.actions,observation:result.observation??null})));
          const path=`${prefix}.${item.id}.json.gz`;writeFileSync(path,bytes,{flag:'wx'});result.raw={path:relative(ROOT,path),sha256:sha256(bytes),bytes:bytes.length};}
        result.checkCount=result.checks.length;result.firstCheck=result.checks[0];result.lastCheck=result.checks.at(-1);delete result.checks;
        result.gates=controlSummary(result);stopErrors?.();await native.close();native=null;
      }
      console.log(`${item.id}: ${result.completion}; target stalls=${result.stalls?.length??'not measured'}; safety=${result.safetyPass}`);
      if(result.completion!=='CAPTURED')throw new Error(`Interrupted ${item.id}; retained record, remaining order not run`);
    }
    assert.deepEqual(verifyBuild(false),build);assert.equal(sha256(readFileSync(referencePath)),report.reference.sha256);
    report.completion='CAPTURED';report.causality=publicCausality(report.results);
  }catch(error){report.completion='UNVERIFIED';report.error=safeEvidence(String(error));throw error;}
  finally{await native?.close();report.ended=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(safeEvidence(report),null,2),{flag:'wx'});}
  return report;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  if(process.argv[2]==='--pin-identity'){assert.equal(process.argv.length,4);await pinVideojsIdentity(process.argv[3]);}
  else{assert(process.argv[2]==='--run-public'&&[5,6].includes(process.argv.length),'Usage: --pin-identity PREFIX | --run-public PREFIX COMMITTED_REFERENCE [PRIOR_REPORT]');
    await runVideojs(process.argv[3],resolve(process.argv[4]),process.argv[5]?resolve(process.argv[5]):undefined);}
}

export function samePublicIdentity(actual, reference) {
  return typeof actual?.urlSha256 === 'string' && /^[a-f0-9]{64}$/.test(actual.urlSha256)
    && actual.urlSha256 === reference?.urlSha256 && actual.origin === reference.origin
    && actual.width === reference.width && actual.height === reference.height
    && Number.isFinite(actual.duration) && Math.abs(actual.duration - VIDEOJS_DURATION) <= 0.01
    && Number.isFinite(reference.duration) && Math.abs(actual.duration - reference.duration) <= 0.01;
}

export function targetStalls(rows, interventions, closingAt) {
  const episodes = [];
  let candidate = null;
  const commands = interventions.filter(value => Number.isFinite(value.at)).toSorted((left, right) => left.at - right.at);
  const seek = commands.find(value => value.name === 'prescribed-seek' && value.pass === true);
  if (!seek) return episodes;
  const endedAt = rows.find(row => row.at>=seek.at&&row.ended)?.at ?? Infinity;
  const replayAt = commands.find(value=>value.at>=seek.at&&value.name==='original-ended-replay')?.at ?? Infinity;
  const attributionEnd = Math.min(endedAt,replayAt,closingAt);
  const close = (reason, upperAt) => {
    if (candidate && candidate.lastAt - candidate.firstAt >= 15000) episodes.push({ ...candidate, reason,
      durationLowerMs: candidate.lastAt - candidate.firstAt,
      recoveryInterval: reason === 'progress' ? [candidate.lastAt, upperAt] : null,
      censoredAt: reason === 'progress' ? null : upperAt,
      scope: '250ms nominal sampling; duration is a lower bound between observed no-progress samples. Interventions censor spontaneous recovery.' });
    candidate = null;
  };
  for (const row of rows) {
    if (row.at < seek.at) continue;
    if (row.at >= attributionEnd) break;
    const intervened = candidate && commands.find(value => value.at > candidate.lastAt && value.at <= row.at && value.name !== 'prescribed-seek');
    if (intervened) close(`scheduled:${intervened.name}`, intervened.at);
    const eligible = Number.isFinite(row.currentTime) && Number.isFinite(row.qualityTotal) && row.qualityTotal >= 0
      && row.paused === false && row.ended === false && Math.abs(row.currentTime - VIDEOJS_DURATION) <= 0.15;
    if (!eligible) { close(row.paused ? 'paused' : row.ended ? 'ended' : 'progress', row.at); continue; }
    if (candidate && (row.qualityTotal !== candidate.qualityTotal
      || Math.max(row.currentTime, candidate.maximumTime) - Math.min(row.currentTime, candidate.minimumTime) > 0.005)) close('progress', row.at);
    candidate ??= { firstAt: row.at, lastAt: row.at, minimumTime: row.currentTime, maximumTime: row.currentTime,
      qualityTotal: row.qualityTotal, readyStates: [], networkStates: [] };
    candidate.lastAt = row.at;
    candidate.minimumTime = Math.min(candidate.minimumTime, row.currentTime);
    candidate.maximumTime = Math.max(candidate.maximumTime, row.currentTime);
    for (const [values, value] of [[candidate.readyStates, row.readyState], [candidate.networkStates, row.networkState]]) if (!values.includes(value)) values.push(value);
  }
  close(attributionEnd<closingAt?'ended-or-replay':'window-end',attributionEnd);
  return episodes;
}

export function publicCausality(results) {
  const groups = Object.fromEntries(['P','Q','R','S'].map(arm => [arm, results.filter(result => result.case.arm === arm)]));
  const complete = Object.values(groups).every(group => group.length === 3 && group.every(result => result.completion === 'CAPTURED' && result.comparable === true
    &&result.observedMs>=180000&&result.actions?.some(action=>action.name==='prescribed-seek'&&action.pass===true)));
  const counts = Object.fromEntries(Object.entries(groups).map(([arm, group]) => [arm, group.filter(result => result.stalls?.length > 0).length]));
  const native = complete && counts.P >= 2 && counts.Q >= 2;
  const frequency = ['R','S'].every(arm => counts[arm] <= counts.P + 1 && counts[arm] <= counts.Q + 1);
  const associated = complete && counts.P === 0 && counts.Q === 0 && (counts.R >= 2 || counts.S >= 2);
  const active = [...groups.R, ...groups.S];
  const allActivePass = complete && active.every(result => result.journeyPass === true);
  const safety = complete && active.every(result => result.safetyPass === true);
  const recovery = complete && active.every(result => result.recoveryComparison === 'NO_OBSERVED_WORSENING');
  return { verdict: native && frequency ? 'PLAYER/SOURCE REPRODUCED WITHOUT AETHERVSR' : associated ? 'EXTENSION-ASSOCIATED' : 'UNRESOLVED',
    counts, complete, safety, allActivePass, recoveryComparable: recovery,
    videojsScopePass: allActivePass || (native && frequency && safety && recovery),
    scope: 'Three sessions/arm; descriptive reproduction or association, not statistical equivalence, prevalence, or network-cause proof. A matching stall does not waive control, presentation, ownership, cleanup, or recovery failures.' };
}