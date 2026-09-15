import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, resolve, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { createServer } from 'vite';
import { buildSync } from 'esbuild';
import { ROOT, sha256 } from './m10-fixtures.mjs';
import { verifyBuild, openExtension, until } from './m10-browser.mjs';
import { nativeWindow } from './m105-accounting.mjs';

export const TOLERANCE = 0.5;

export function geometryOracleSource() {
  const result=buildSync({entryPoints:[join(ROOT,'src/extension/geometry.ts')],bundle:true,write:false,format:'iife',globalName:'M107Geometry',platform:'browser',target:'chrome106'});
  const source=result.outputFiles[0].text;
  return {source,sha256:sha256(source)};
}

export async function installGeometryOracle(native,page,oracle=geometryOracleSource()) {
  const install=new Function(`${oracle.source}\nglobalThis[Symbol.for('aethervsr.m107.geometry-oracle')]=M107Geometry.inspectGeometry;return true;`);
  assert.equal(await native.isolated(page,install),true);
  return {sha256:oracle.sha256,scope:'External diagnostic CDP injection in the verified isolated world only. Fresh read-only geometry inspection shares the supported-geometry rules, never the attachment cache. Not an independent algorithm or pixel/scanout proof; absent from cost collection and packaged production.'};
}

export function installPresentationObserver({ diagnostic = false } = {}) {
  const key = Symbol.for('aethervsr.m107.observation');
  if (globalThis[key]) throw new Error('Duplicate presentation observer');
  if (diagnostic && !globalThis.__AETHERVSR_EXTENSION_TEST__) throw new Error('Diagnostic build required');
  const manager = globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)];
  const attachment = manager?.attachment;
  if (!attachment) throw new Error('Selected authoritative attachment required');
  const video = attachment.video, canvas = attachment.canvas, pipeline = attachment.pipeline;
  const inspect = globalThis[Symbol.for('aethervsr.m107.geometry-oracle')];
  if(typeof inspect!=='function')throw new Error('Fresh geometry oracle required');
  const normalizer = document.createElement('div').style;
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
    const geometry = inspect(video);
    const expected = geometry.ok ? Object.fromEntries(['left','top','width','height'].map(name=>[name,Number.parseFloat(geometry.style[name])])) : null;
    const styleMismatches = geometry.ok ? ['object-fit','object-position','border-radius','clip-path','clip','z-index'].filter(name=>{
      normalizer.removeProperty(name);
      if(geometry.style[name]!==undefined)normalizer.setProperty(name,geometry.style[name]);
      return normalizer.getPropertyValue(name)!==canvas.style.getPropertyValue(name);
    }) : [];
    const mismatch = visible && (!expected || styleMismatches.length>0 || ['left','top','width','height'].some(name => Math.abs(actual[name] - expected[name]) > 0.5));
    data.rows.push({ at: performance.now(), reason, operation: data.operation, videoRect: current, canvasRect: actual,
      expectedRect: expected, appliedRect: snapshot.cssRect, clipRect: geometry.ok?geometry.clip:null,appliedClip:attachment.appliedGeometry?.clip??null,
      geometrySupported:geometry.ok,geometryRejection:geometry.ok?null:geometry.code,styleMismatches,
      canvasVisible: visible, videoVisible: video.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}), mismatch,
      intrinsic: {width:video.videoWidth,height:video.videoHeight}, backing: {width:canvas.width,height:canvas.height},
      mediaReadyState:video.readyState,
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
    return {...data,teardown:()=>attachment.snapshot(),scope:'Diagnostic event/rAF/post-submit samples use fresh geometry inspection, actual canvas bounds and canonical owned fit/mask/stack declarations. Shared geometry rules, not an independent implementation. Never performance/physical-display evidence; null generations mean absent in baseline.'};
  }};
  sample('installed');
  return {diagnostic,owner:manager.status().owner};
}

export function summarizePresentation(trace) {
  assert.equal(trace.overflow,false,'Bounded presentation trace overflow');
  const visible=trace.rows.filter(row=>row.canvasVisible);
  const mismatches=visible.filter(row=>row.mismatch);
  const knownDirty=visible.filter(row=>Number.isInteger(row.geometryGeneration)&&row.geometryGeneration!==row.appliedGeometryGeneration);
  const errors=visible.flatMap(row=>{const expected=row.expectedRect===undefined?row.videoRect:row.expectedRect;
    return expected?['left','top','width','height'].map(name=>Math.abs(expected[name]-row.canvasRect[name])):[];});
  return {samples:trace.rows.length,visibleSamples:visible.length,mismatches:mismatches.length,
    knownDirtyVisible:trace.rows.some(row=>Number.isInteger(row.geometryGeneration))?knownDirty.length:null,
    firstMismatch:mismatches[0]??null,lastMismatch:mismatches.at(-1)??null,
    maximumComponentError:errors.length?Math.max(...errors):null,
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

export function presentationEnvironment() {
  const hardware=JSON.parse(execFileSync('system_profiler',['SPHardwareDataType','-json'],{encoding:'utf8',timeout:15000})).SPHardwareDataType[0];
  const files=['tools/m107-presentation.mjs','tools/m107-performance.mjs','tools/m107-fixture.html','tools/m106-counters.mjs','tools/m105-accounting.mjs',
    'tools/m10-performance.mjs','tools/m10-browser.mjs','tools/m9-browser.mjs','src/extension/geometry.ts','public/media/m9/720p60.mp4','public/models/aethersr-c16d2.json'];
  const sourcePins=Object.fromEntries(files.map(path=>[path,sha256(readFileSync(join(ROOT,path)))]));
  assert.equal(sourcePins['public/media/m9/720p60.mp4'],'8d81acbe164da1d62b7d0d02a3cc66915c96e8aa90d45cac34d818fc33df1d4a');
  const power=execFileSync('pmset',['-g','batt'],{encoding:'utf8'}).trim();assert(power.includes('AC Power'));
  return {sourcePins,os:execFileSync('sw_vers',['-productVersion'],{encoding:'utf8'}).trim(),node:process.version,power,
    hardware:{model:hardware.machine_model,chip:hardware.chip_type,memory:hardware.physical_memory},displayRefreshRate:'not measured',
    media:JSON.parse(execFileSync('ffprobe',['-v','error','-show_entries','stream=codec_name,width,height,r_frame_rate,avg_frame_rate:format=duration','-of','json',join(ROOT,'public/media/m9/720p60.mp4')],{encoding:'utf8'}))};
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
  {name:'warming-scroll',prepare:'paused-output',actions:['resume-scroll'],warming:true},
  {name:'effect-rejection',actions:['unsupported-effect'],negative:true},
  {name:'insertion-layout',prepare:'paused-last-child',actions:['resume-scroll'],warming:true},
  {name:'cssom-none-position',prepare:'fit-none',actions:['cssom-position']},
  {name:'cssom-scale-down-position',prepare:'fit-scale-down',actions:['cssom-position']},
  {name:'fractional-image-scale',prepare:'fit-none',actions:['cssom-scale']},
  {name:'caption-mutation',prepare:'passive-caption',actions:['caption-z'],negative:true},
  {name:'caption-cssom',prepare:'passive-caption',actions:['cssom-caption-z'],negative:true},
  {name:'cssom-effect',actions:['cssom-effect'],negative:true},
  {name:'clip-only-resize',prepare:'outer-clip',actions:['cssom-clip']},
  {name:'fullscreen-outer-caption',prepare:'outer-caption',actions:['fullscreen','exit']},
  {name:'overlapping-source-scroll',prepare:'intrinsic-resize',actions:['source-high-scroll','source-low-scroll']},
  {name:'warming-source-scroll',prepare:'intrinsic-resize-paused',actions:['source-high-scroll','source-low'],warming:true},
];

export function transitionVerdict(trace, actions, negative=false) {
  assert.equal(trace.overflow,false);
  const rows=trace.rows;
  const known=rows.some(row=>Number.isInteger(row.geometryGeneration));
  const boundary=row=>row.reason==='animation'||row.reason==='submitted:after'||row.reason==='refresh:end'||row.reason.startsWith('event:');
  const dirty=known?rows.filter(row=>boundary(row)&&row.canvasVisible&&row.geometryGeneration!==row.appliedGeometryGeneration):null;
  const processed=rows.filter(row=>row.reason==='submitted:after'&&row.mismatch);
  const staleOutput=rows.filter(row=>boundary(row)&&row.canvasVisible&&known&&
    (row.outputGeometryGeneration!==row.geometryGeneration||row.outputGeneration!==row.sourceGeneration));
  const outcomes=actions.map(action=>{
    const settled=rows.filter(row=>row.at>=action.finishedAt+500&&row.at<action.finishedAt+1000);
    const detected=rows.filter(row=>row.at>=action.startedAt&&row.at<=action.finishedAt+1000);
    const guardBoundary=detected.find(row=>row.reason==='animation'||row.reason==='submitted:after');
    const persistent=detected.filter(row=>row.mismatch&&guardBoundary&&row.at>guardBoundary.at&&row.reason==='animation');
    const recoveryRows=rows.filter(row=>row.at>=action.finishedAt&&row.at<=action.finishedAt+500);
    const frames=recoveryRows.filter(row=>row.reason==='submitted:after'&&row.canvasVisible&&!row.mismatch);
    const restored=settled.some(row=>row.canvasVisible&&!row.mismatch)&&frames.length>=2;
    const lateSource=rows.filter(row=>row.reason==='submitted:after'&&row.at>=action.startedAt&&row.at<=action.finishedAt+1000&&row.canvasVisible
      &&(row.backing.width!==row.intrinsic.width*2||row.backing.height!==row.intrinsic.height*2||Number.isInteger(row.sourceGeneration)&&row.outputGeneration!==row.sourceGeneration));
    return {...action,settledSamples:settled.length,visibleRecovery:restored,confirmedFramesWithin500ms:frames.length,
      firstVisibleAfterSettledMs:frames.length?frames[0].at-action.finishedAt:null,lateMismatches:persistent.length,staleSourceSamples:lateSource.length,
      pass:!action.error&&persistent.length===0&&lateSource.length===0&&(negative||action.name==='scroll-900'?settled.length>0&&settled.every(row=>!row.canvasVisible):restored)};
  });
  return {...summarizePresentation(trace),pass:known&&dirty.length===0&&processed.length===0&&staleOutput.length===0&&outcomes.every(row=>row.pass),knownDirtyVisible:dirty?.length??null,
    staleOutputSamples:staleOutput.length,
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
  const oracle=geometryOracleSource();
  const report={phase:'TRANSITIONS',strategy,production,repeats,build,apparatus:current.provenance.sourceCommit,environment:presentationEnvironment(),started:new Date().toISOString(),results:[],verdict:'UNVERIFIED'};
  let native,server;
  try{
    server=await presentationServer();native=await openExtension(build,(name,data)=>{if(name==='browser')report.browser=data;});report.fixtureSha256=server.htmlSha256;
    for(let repetition=1;repetition<=repeats;repetition++)for(const item of cases){
      const result={name:item.name,repetition,actions:[],verdict:'UNVERIFIED'};report.results.push(result);
      const page=await native.context.newPage();let tabId;
      try{
        result.placement=await nativeWindow(page,native.context);await page.goto(server.url);await page.bringToFront();
        if(item.prepare)await page.evaluate(name=>globalThis.__M107_FIXTURE__.prepare(name),item.prepare);
        await page.waitForFunction(warming=>{const video=document.querySelector('video');return video.readyState>=2&&(warming||!video.paused);},!!item.warming);
        const panel=await native.popup(page);
        try{
          tabId=panel.tabId;
          if(strategy>0&&!production){await native.workerEval(async tab=>chrome.scripting.executeScript({target:{tabId:tab,frameIds:[0]},world:'ISOLATED',files:['content.js']}),tabId);
            await native.isolated(page,options=>globalThis.__AETHERVSR_EXTENSION_TEST__.configure(options),{presentationWatchdog:strategy===2});}
          await panel.click('input[value="baseline"]');await panel.click('#enable');
        }finally{await panel.dismiss();}
        await until(()=>native.inspect(tabId),state=>!!state.owner&&!!state.details?.attachment,10000);
        if(!item.warming)await until(()=>native.inspect(tabId),state=>state.details?.attachment?.ready,10000);
        else {
          await until(()=>native.inspect(tabId),state=>state.details?.attachment?.resources?.pipeline===1,10000);
          result.warming=await native.isolated(page,()=>{const attachment=globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)].attachment;
            return {ready:attachment.snapshot().ready,paused:attachment.video.paused,framesRendered:attachment.driver.snapshot().session.framesRendered};});
          assert.equal(result.warming.ready,false);assert.equal(result.warming.paused,true);assert.equal(result.warming.framesRendered,0);
        }
        result.oracle=await installGeometryOracle(native,page,oracle);
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
            if(name.startsWith('source-'))await page.waitForFunction(width=>document.querySelector('video').videoWidth===width&&document.querySelector('video').readyState>=2,name.startsWith('source-high')?2560:960,{timeout:5000});
            action.settled=await page.evaluate(action=>globalThis.__M107_FIXTURE__.settle(action),name);
            if(name==='smooth')await page.evaluate(start=>new Promise(done=>setTimeout(done,Math.max(0,start+2000-performance.now()))),action.startedAt);
          }catch(error){action.error=String(error);}
          action.actionCompletedAt=await page.evaluate(()=>performance.now());
          action.finishedAt=action.settled?.at??action.actionCompletedAt;
          await page.evaluate(()=>new Promise(done=>setTimeout(done,1100)));
        }
        const trace=await native.isolated(page,()=>{const {teardown,...value}=globalThis[Symbol.for('aethervsr.m107.observation')].stop();globalThis[Symbol.for('aethervsr.m107.teardown')]=teardown;return value;});
        for(const action of result.actions)if(action.name.startsWith('source-')){
          const width=action.name.startsWith('source-high')?2560:960;
          const decoded=trace.rows.find(row=>row.at>=action.startedAt&&row.intrinsic.width===width&&row.mediaReadyState>=2);
          if(decoded){action.decodedAvailableAt=decoded.at;action.finishedAt=decoded.at;if(decoded.at-action.startedAt>5000)action.error='Decoded source startup exceeded 5s';}
          else action.error??='Decoded source availability was not observed';
        }
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
    report.oracle=await installGeometryOracle(native,page);
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