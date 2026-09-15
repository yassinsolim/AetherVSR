import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, resolve, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { createServer } from 'vite';
import { ROOT, sha256 } from './m10-fixtures.mjs';
import { verifyBuild, openExtension, until } from './m10-browser.mjs';
import { nativeWindow } from './m105-accounting.mjs';

export const TOLERANCE = 0.5;

export function installPresentationObserver({ diagnostic = false } = {}) {
  const key = Symbol.for('aethervsr.m107.observation');
  if (globalThis[key]) throw new Error('Duplicate presentation observer');
  if (diagnostic && !globalThis.__AETHERVSR_EXTENSION_TEST__) throw new Error('Diagnostic build required');
  const manager = globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)];
  const attachment = manager?.attachment;
  if (!attachment) throw new Error('Selected authoritative attachment required');
  const video = attachment.video, canvas = attachment.canvas, pipeline = attachment.pipeline;
  const data = { rows: [], overflow: false, operation: null, generation: 0, wrappers: false };
  let running = true, handle;
  const listeners = [], restores = [];
  const rect = node => { const value = node.getBoundingClientRect(); return { left: value.left, top: value.top, width: value.width, height: value.height }; };
  const category = node => !node ? 'none' : node === video ? 'video' : node === canvas ? 'owned-canvas'
    : node.contains?.(video) ? 'video-ancestor' : 'other';
  const sample = reason => {
    if (!running) return;
    if (data.rows.length >= 8192) { data.overflow = true; return; }
    const snapshot = attachment.snapshot();
    const current = rect(video), actual = rect(canvas);
    const visible = getComputedStyle(canvas).visibility === 'visible' && canvas.isConnected;
    const imageSized = ['none','scale-down'].includes(attachment.appliedGeometry?.objectFit);
    const expected = imageSized && attachment.appliedCanvasRect && attachment.appliedVideoRect ? {
      ...attachment.appliedCanvasRect,
      left:attachment.appliedCanvasRect.left+current.left-attachment.appliedVideoRect.left,
      top:attachment.appliedCanvasRect.top+current.top-attachment.appliedVideoRect.top,
    } : current;
    const mismatch = visible && ['left','top','width','height'].some(name => Math.abs(actual[name] - expected[name]) > 0.5);
    data.rows.push({ at: performance.now(), reason, operation: data.operation, videoRect: current, canvasRect: actual,
      expectedRect: expected, appliedRect: snapshot.cssRect, clipRect: attachment.appliedGeometry?.clip ?? null,
      canvasVisible: visible, videoVisible: video.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}), mismatch,
      intrinsic: {width:video.videoWidth,height:video.videoHeight}, backing: {width:canvas.width,height:canvas.height},
      scroll: {x:scrollX,y:scrollY}, fullscreen: category(document.fullscreenElement), parent: category(canvas.parentNode),
      focused:document.hasFocus(),documentVisibility:document.visibilityState,
      geometryGeneration: attachment.geometryGeneration ?? null, appliedGeometryGeneration: attachment.appliedGeometryGeneration ?? null,
      sourceGeneration: attachment.sourceGeneration ?? null, observedSourceGeneration: data.generation,
      outputGeneration: attachment.outputGeneration ?? null, outputGeometryGeneration: attachment.outputGeometryGeneration ?? null,
      eligible: attachment.eligible, suspendedReason: snapshot.suspendedReason, ready: snapshot.ready,
      refreshScheduled: attachment.geometryFrame !== null, current: snapshot.current, resources: snapshot.resources,
      verifyPlacement:attachment.appliedCanvasRect ? ['left','top','width','height'].every(name=>Math.abs(actual[name]-attachment.appliedCanvasRect[name])<=0.5):null,
      ownerCurrent: manager.attachment === attachment, originalConnected: video.isConnected });
  };
  const on = (target, type, listener, capture = false) => { target.addEventListener(type,listener,capture);listeners.push([target,type,listener,capture]); };
  for (const [target,types] of [[window,['scroll','resize','blur']], [document,['fullscreenchange','visibilitychange']],
    [video,['resize','loadstart','loadeddata','emptied','pause','seeking','seeked','playing']]]) for (const type of types) {
    on(target,type,()=>{ if(type==='loadstart')data.generation++;sample(`event:${type}`); },true);
  }
  if (diagnostic) {
    for (const name of ['refresh','checkGeometry','hide']) {
      const original=attachment[name];
      attachment[name]=function(...args){sample(`${name}:begin`);try{return original.apply(this,args);}finally{sample(`${name}:end`);}};
      restores.push(()=>{attachment[name]=original;});
    }
    data.wrappers=true;
  }
  if(pipeline){const original=pipeline.onFrame;pipeline.onFrame=function(tick){sample('submitted:before');try{return original?.call(this,tick);}finally{sample('submitted:after');}};
    restores.push(()=>{if(attachment.pipeline===pipeline)pipeline.onFrame=original;});}
  const frame=()=>{sample('animation');if(running)handle=requestAnimationFrame(frame);};handle=requestAnimationFrame(frame);
  on(window,'aethervsr:m107:operation',event=>{data.operation=event.detail;sample('operation');});
  globalThis[key]={ sample, stop:()=>{
    sample('stop');running=false;cancelAnimationFrame(handle);
    for(const [target,type,listener,capture]of listeners)target.removeEventListener(type,listener,capture);
    for(const restore of restores)restore();
    return {...data,teardown:()=>attachment.snapshot(),scope:'Diagnostic event/rAF/post-submit samples. Expected rect equals original fixture video box (fixed16:9 contain). Not continuous physical display; null generations mean absent in baseline.'};
  }};
  sample('installed');
  return {diagnostic,owner:manager.status().owner};
}

export function summarizePresentation(trace) {
  assert.equal(trace.overflow,false,'Bounded presentation trace overflow');
  const visible=trace.rows.filter(row=>row.canvasVisible);
  const mismatches=visible.filter(row=>row.mismatch);
  const knownDirty=visible.filter(row=>Number.isInteger(row.geometryGeneration)&&row.geometryGeneration!==row.appliedGeometryGeneration);
  return {samples:trace.rows.length,visibleSamples:visible.length,mismatches:mismatches.length,
    knownDirtyVisible:trace.rows.some(row=>Number.isInteger(row.geometryGeneration))?knownDirty.length:null,
    firstMismatch:mismatches[0]??null,lastMismatch:mismatches.at(-1)??null,
    maximumComponentError:Math.max(0,...mismatches.flatMap(row=>['left','top','width','height'].map(name=>Math.abs(row.videoRect[name]-row.canvasRect[name])))),
    byReason:Object.fromEntries([...new Set(mismatches.map(row=>row.reason))].map(reason=>[reason,mismatches.filter(row=>row.reason===reason).length])),
    scope:'Counts are correlated diagnostic observations, not independent frames. First/last bounds do not prove continuous exposure.'};
}

export async function presentationServer() {
  const html=readFileSync(join(ROOT,'tools/m107-fixture.html'));
  const server=await createServer({root:ROOT,mode:'benchmark',server:{host:'127.0.0.1',port:5191,strictPort:true},plugins:[{
    name:'m107-transition-fixture',configureServer(instance){instance.middlewares.use((request,response,next)=>{
      const path=new URL(request.url,'http://local').pathname;
      const name={'/m107-media/low.mp4':'low.mp4','/m107-media/high.mp4':'high.mp4','/m107-media/config.json':'config.json'}[path];
      if(name){
        const file=join(ROOT,'.cache/m107/media',name);if(!existsSync(file)){response.statusCode=404;response.end();return;}
        response.setHeader('Content-Type',name.endsWith('.json')?'application/json':'video/mp4');response.end(readFileSync(file));return;
      }
      if(path!=='/m107')return next();
      response.setHeader('Content-Type','text/html; charset=utf-8');response.end(html);
    });},
  }]});
  await server.listen();
  return {url:'http://127.0.0.1:5191/m107',htmlSha256:sha256(html),close:()=>server.close()};
}

export function prepareTransitionMedia() {
  const directory=join(ROOT,'.cache/m107/media');mkdirSync(directory,{recursive:true});
  const source=join(ROOT,'public/media/m9/720p60.mp4');
  assert.equal(sha256(readFileSync(source)),'8d81acbe164da1d62b7d0d02a3cc66915c96e8aa90d45cac34d818fc33df1d4a');
  const files=[];
  for(const [name,width,height]of [['low',960,540],['high',2560,1440]]){
    const path=join(directory,`${name}.mp4`);
    const args=['-nostdin','-hide_banner','-loglevel','error','-n','-i',source,'-t','2','-an','-vf',`scale=${width}:${height}:flags=lanczos,fps=30`,
      '-c:v','libx264','-preset','fast','-profile:v','high','-level:v','5.2','-pix_fmt','yuv420p','-g','60','-bf','0',
      '-movflags','frag_keyframe+empty_moov+default_base_moof','-map_metadata','-1','-threads','2',path];
    if(!existsSync(path))execFileSync('ffmpeg',args,{stdio:'inherit'});
    const bytes=readFileSync(path),config=bytes.indexOf(Buffer.from('avcC'));assert(config>=0);
    files.push({name,width,height,sha256:sha256(bytes),bytes:bytes.length,mime:`video/mp4; codecs="avc1.${bytes.subarray(config+5,config+8).toString('hex')}"`,recipe:args});
  }
  assert.equal(files[0].mime,files[1].mime);
  const metadata={mime:files[0].mime,files,scope:'Generated same-codec fMP4 initialization+segments; sequence-appended to one MediaSource without changing video element/blob or CSS box.'};
  const path=join(directory,'config.json');if(!existsSync(path))writeFileSync(path,JSON.stringify(metadata,null,2),{flag:'wx'});
  else assert.deepEqual(JSON.parse(readFileSync(path)),metadata);
  return metadata;
}

export const TRANSITIONS = [
  {name:'scroll-one',actions:['scroll-1']}, {name:'scroll-228',actions:['scroll-228']},
  {name:'scroll-large',actions:['scroll-900','scroll-0']}, {name:'continuous-smooth',actions:['smooth','scroll-0']},
  {name:'nested-scroll',prepare:'nested-scroll',actions:['nested']},
  {name:'size-container',prepare:'size-container',actions:['scroll-228','class-shift']},
  {name:'ancestor-position',actions:['ancestor-shift']}, {name:'class-position',actions:['class-shift']},
  {name:'style-position',actions:['style-shift']}, {name:'equivalent-reparent',actions:['reparent']},
  {name:'fullscreen',actions:['fullscreen','exit']},
  {name:'fullscreen-scrolled',actions:['scroll-228','fullscreen','resize-fullscreen','exit']},
  {name:'source-resize',prepare:'intrinsic-resize',actions:['source-high','source-low']},
  {name:'fullscreen-source',prepare:'intrinsic-resize',actions:['fullscreen','source-high','exit']},
  {name:'scroll-source',prepare:'intrinsic-resize',actions:['scroll-1','source-high','scroll-228','source-low']},
  {name:'warming-scroll',actions:['scroll-228'],warming:true},
  {name:'effect-rejection',actions:['unsupported-effect'],negative:true},
];

export function transitionVerdict(trace, actions, negative=false) {
  assert.equal(trace.overflow,false);
  const rows=trace.rows;
  const known=rows.some(row=>Number.isInteger(row.geometryGeneration));
  const boundary=row=>row.reason==='animation'||row.reason==='submitted:after'||row.reason==='refresh:end'||row.reason.startsWith('event:');
  const dirty=known?rows.filter(row=>boundary(row)&&row.canvasVisible&&row.geometryGeneration!==row.appliedGeometryGeneration):null;
  const processed=rows.filter(row=>row.reason==='submitted:after'&&row.mismatch);
  const outcomes=actions.map(action=>{
    const settled=rows.filter(row=>row.at>=action.finishedAt+500&&row.at<action.finishedAt+1000);
    const detected=rows.filter(row=>row.at>=action.startedAt&&row.at<=action.finishedAt+1000);
    const guardBoundary=detected.find(row=>row.reason==='animation'||row.reason==='submitted:after');
    const persistent=detected.filter(row=>row.mismatch&&guardBoundary&&row.at>guardBoundary.at&&row.reason==='animation');
    const restored=settled.some(row=>row.canvasVisible&&!row.mismatch);
    return {...action,settledSamples:settled.length,visibleRecovery:restored,lateMismatches:persistent.length,
      pass:!action.error&&persistent.length===0&&(negative||action.name==='scroll-900'?settled.every(row=>!row.canvasVisible||!row.mismatch):restored)};
  });
  return {...summarizePresentation(trace),pass:known&&dirty.length===0&&processed.length===0&&outcomes.every(row=>row.pass),knownDirtyVisible:dirty?.length??null,
    postSubmitMismatches:processed.length,actions:outcomes};
}

export async function runTransitions(prefix,{strategy=1,repeats=3,only=null,production=false}={}) {
  prefix=resolve(prefix);assert(!existsSync(`${prefix}.json`));assert(prefix.startsWith(join(ROOT,'.cache/m107/')));mkdirSync(dirname(prefix),{recursive:true});
  const current=verifyBuild(!production);
  let build=current;
  if(strategy===0){
    const directory=join(ROOT,'.cache/m107/frozen-s0');const provenance=JSON.parse(readFileSync(join(directory,'build-provenance.json')));
    for(const [name,value]of Object.entries(provenance.files))assert.equal(sha256(readFileSync(join(directory,name))),value.sha256);
    assert.equal(provenance.bundleSha256,'ef2e17d32009d1c7bb18a1fc0449d64723b7df763ca43d10b63737edb8e977c7');
    build={directory,provenance};
  }
  const cases=only?TRANSITIONS.filter(item=>only.includes(item.name)):TRANSITIONS;
  const report={phase:'TRANSITIONS',strategy,production,repeats,build,apparatus:current.provenance.sourceCommit,started:new Date().toISOString(),results:[],verdict:'UNVERIFIED'};
  let native,server;
  try{
    server=await presentationServer();native=await openExtension(build);report.fixtureSha256=server.htmlSha256;
    for(let repetition=1;repetition<=repeats;repetition++)for(const item of cases){
      const result={name:item.name,repetition,actions:[],verdict:'UNVERIFIED'};report.results.push(result);
      const page=await native.context.newPage();let tabId;
      try{
        await nativeWindow(page,native.context);await page.goto(server.url);await page.bringToFront();
        if(item.prepare)await page.evaluate(name=>globalThis.__M107_FIXTURE__.prepare(name),item.prepare);
        await page.waitForFunction(()=>{const video=document.querySelector('video');return video.readyState>=2&&!video.paused;});
        const panel=await native.popup(page);
        try{
          tabId=panel.tabId;
          if(strategy>0&&!production){await native.workerEval(async tab=>chrome.scripting.executeScript({target:{tabId:tab,frameIds:[0]},world:'ISOLATED',files:['content.js']}),tabId);
            await native.isolated(page,options=>globalThis.__AETHERVSR_EXTENSION_TEST__.configure(options),{presentationWatchdog:strategy===2});}
          await panel.click('input[value="baseline"]');await panel.click('#enable');
        }finally{await panel.dismiss();}
        await until(()=>native.inspect(tabId),state=>!!state.owner&&!!state.details?.attachment,10000);
        if(!item.warming)await until(()=>native.inspect(tabId),state=>state.details?.attachment?.ready,10000);
        await native.isolated(page,installPresentationObserver,{diagnostic:!production});
        if(!item.warming)await page.evaluate(()=>new Promise(done=>setTimeout(done,250)));
        for(const name of item.actions){
          const action={name,startedAt:await page.evaluate(()=>performance.now())};result.actions.push(action);
          try{
            if(name==='fullscreen'){
              await page.evaluate(()=>window.dispatchEvent(new CustomEvent('aethervsr:m107:operation',{detail:'fullscreen:begin'})));
              await page.locator('#fullscreen').click();await page.waitForFunction(()=>!!document.fullscreenElement,undefined,{timeout:3000});
            }else if(name==='exit'){
              await page.evaluate(async()=>{window.dispatchEvent(new CustomEvent('aethervsr:m107:operation',{detail:'exit:begin'}));await document.exitFullscreen();});
              await page.waitForFunction(()=>!document.fullscreenElement,undefined,{timeout:3000});
            }else if(name==='resize-fullscreen')await page.evaluate(()=>{window.dispatchEvent(new CustomEvent('aethervsr:m107:operation',{detail:'resize-fullscreen:begin'}));document.getElementById('player').style.height='85%';});
            else await page.evaluate(action=>globalThis.__M107_FIXTURE__.action(action),name);
            if(name==='smooth')await page.evaluate(()=>new Promise(done=>setTimeout(done,2000)));
            if(name.startsWith('source-'))await page.waitForFunction(width=>document.querySelector('video').videoWidth===width&&document.querySelector('video').readyState>=2,name==='source-high'?2560:960,{timeout:5000});
          }catch(error){action.error=String(error);}
          action.finishedAt=await page.evaluate(()=>performance.now());
          await page.evaluate(()=>new Promise(done=>setTimeout(done,1100)));
        }
        const trace=await native.isolated(page,()=>{const {teardown,...value}=globalThis[Symbol.for('aethervsr.m107.observation')].stop();globalThis[Symbol.for('aethervsr.m107.teardown')]=teardown;return value;});
        const packed=gzipSync(JSON.stringify(trace));const path=`${prefix}.${repetition}-${item.name}.json.gz`;writeFileSync(path,packed,{flag:'wx'});
        result.raw={path:relative(ROOT,path),sha256:sha256(packed),bytes:packed.length};result.summary=transitionVerdict(trace,result.actions,item.negative);
        result.integrity=trace.rows.every(row=>row.focused&&row.documentVisibility==='visible');
        const disablePanel=await native.popup(page);try{result.disabled=await disablePanel.click('#disable');}finally{await disablePanel.dismiss();}
        result.teardown=await native.isolated(page,()=>globalThis[Symbol.for('aethervsr.m107.teardown')]());
        result.cleanup=Object.values(result.teardown.resources).every(value=>value===0)&&result.teardown.infrastructure.cleanupErrors===0;
        result.verdict=result.summary.pass&&result.cleanup&&result.integrity?'PASS':'FAIL';
      }catch(error){result.error=String(error);}
      finally{await page.close();}
      console.log(JSON.stringify({strategy,repetition,case:item.name,verdict:result.verdict,mismatches:result.summary?.mismatches,processed:result.summary?.postSubmitMismatches,recovery:result.summary?.actions.map(action=>[action.name,action.pass])}));
    }
    assert.deepEqual(verifyBuild(!production),current);report.verdict=report.results.every(row=>row.verdict==='PASS')?'PASS':'FAIL';
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

export async function diagnoseScroll(prefix) {
  prefix=resolve(prefix);assert(relative(join(ROOT,'.cache/m107'),prefix)&&!relative(join(ROOT,'.cache/m107'),prefix).startsWith('..'));
  assert(!existsSync(`${prefix}.json`),'Do not overwrite evidence');mkdirSync(dirname(prefix),{recursive:true});
  const build=verifyBuild(true),report={schemaVersion:1,phase:'PRE_FIX_DIAGNOSIS',started:new Date().toISOString(),build,verdict:'UNVERIFIED'};
  let server,native,page;
  try{
    server=await presentationServer();native=await openExtension(build,(name,data)=>{if(name==='browser')report.browser=data;});page=await native.context.newPage();
    report.placement=await nativeWindow(page,native.context);report.fixtureSha256=server.htmlSha256;
    await page.goto(server.url);await page.bringToFront();
    await page.waitForFunction(()=>{const video=document.querySelector('video');return !video.paused&&video.readyState>=2;});
    const popup=await native.popup(page);let tabId;
    try{tabId=popup.tabId;await popup.click('#enable');await popup.click('input[value="baseline"]');}finally{await popup.dismiss();}
    await until(()=>native.inspect(tabId),state=>state.details?.attachment?.ready&&state.current==='baseline',10000);
    await native.isolated(page,installPresentationObserver,{diagnostic:true});
    await page.evaluate(()=>new Promise(done=>setTimeout(done,250)));
    await page.evaluate(()=>{window.dispatchEvent(new CustomEvent('aethervsr:m107:operation',{detail:'page-scroll-228:begin'}));scrollTo({top:228,behavior:'instant'});
      window.dispatchEvent(new CustomEvent('aethervsr:m107:operation',{detail:'page-scroll-228:after'}));});
    await page.evaluate(()=>new Promise(done=>setTimeout(done,1000)));
    const trace=await native.isolated(page,()=>{const result=globalThis[Symbol.for('aethervsr.m107.observation')].stop();const{teardown,...data}=result;globalThis[Symbol.for('aethervsr.m107.teardown')]=teardown;return data;});
    const packed=gzipSync(JSON.stringify(trace));writeFileSync(`${prefix}.json.gz`,packed,{flag:'wx'});
    report.raw={path:relative(ROOT,`${prefix}.json.gz`),sha256:sha256(packed),bytes:packed.length};report.summary=summarizePresentation(trace);
    assert(trace.rows.every(row=>row.focused&&row.documentVisibility==='visible'),'External foreground interruption');
    report.delayed=trace.rows.some(row=>row.reason==='refresh:end'&&row.refreshScheduled&&row.mismatch);
    const disable=await native.popup(page);try{report.disabled=await disable.click('#disable');}finally{await disable.dismiss();}
    report.teardown=await native.isolated(page,()=>globalThis[Symbol.for('aethervsr.m107.teardown')]());
    assert(Object.values(report.teardown.resources).every(value=>value===0));assert.equal(report.teardown.infrastructure.cleanupErrors,0);
    assert.deepEqual(verifyBuild(true),build);
    report.verdict=report.delayed&&report.summary.maximumComponentError===228?'REPRODUCED_DELAYED_INVALIDATION':'NOT_REPRODUCED';
  }catch(error){report.error=String(error);throw error;}
  finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  assert(process.argv[2]==='--diagnose'&&process.argv.length===4,'Usage: --diagnose .cache/m107/PREFIX');
  const report=await diagnoseScroll(process.argv[3]);console.log(JSON.stringify({verdict:report.verdict,summary:report.summary}));
}