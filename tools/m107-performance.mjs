import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {join,resolve,relative,dirname} from 'node:path';
import {gzipSync} from 'node:zlib';
import {ROOT,sha256} from './m10-fixtures.mjs';
import {verifyBuild,openExtension,until,bounded} from './m10-browser.mjs';
import {nativeWindow} from './m105-accounting.mjs';
import {installNativeObserver,summarizeNative,validateNative} from './m106-counters.mjs';
import {distribution,counterDelta} from './m10-performance.mjs';
import {presentationServer,presentationEnvironment} from './m107-presentation.mjs';

export const COST_ORDER=['012','120','201','021','210','102'].flatMap((block,index)=>[...block].map(value=>({block:index+1,strategy:Number(value),durationMs:60000})));

export function profileProofReads() {
  const attachment=globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)]?.attachment;
  const entries=attachment?.checkedGeometry?.proof?.styles;
  if(!entries?.length)throw new Error('A retained geometry proof is required');
  const rows=[];
  for(const [ordinal,entry]of entries.entries()){
    const names=entry.keys.map(key=>String(key).replace(/[A-Z]/g,value=>`-${value.toLowerCase()}`));
    const methods={property:()=>entry.keys.every((key,index)=>entry.style[key]===entry.values[index]),
      named:()=>names.every((name,index)=>entry.style.getPropertyValue(name)===entry.values[index])};
    const timing={};
    for(const [method,read]of Object.entries(methods)){
      const samples=[];let equal=true;
      for(let batch=0;batch<12;batch++){
        const started=performance.now();
        for(let repetition=0;repetition<64;repetition++)equal=read()&&equal;
        samples.push(performance.now()-started);
      }
      timing[method]={batchMs:samples,equal,readsPerBatch:entry.keys.length*64};
    }
    rows.push({ordinal,video:entry.element===attachment.video,keys:entry.keys,names:entry.names,timing});
  }
  return {rows,scope:'Diagnostic repeated reads of retained live computed styles, 12 batches of64 whole-entry scans per method. No frame/GPU/performance acceptance; cached-style microbenchmark does not represent one live frame or cold layout work.'};
}

export function profileLiveProof() {
  const attachment=globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)]?.attachment;
  if(!attachment?.checkedGeometry?.proof)throw new Error('A retained geometry proof is required');
  const original=attachment.checkPresentation,rows=[];
  return new Promise(resolve=>{
    const finish=()=>{clearTimeout(timer);attachment.checkPresentation=original;resolve({rows,scope:'Intrusive120-call component attribution only. Alternates original-first/checks-first. Extra reads warm browser caches and are not a qualifying cost window. No thresholds or production behavior changed.'});};
    const timer=setTimeout(finish,3000);
    attachment.checkPresentation=function(...args){
      const first=rows.length%2===0,started=performance.now();let result,originalMs;
      if(first){result=original.apply(this,args);originalMs=performance.now()-started;}
      const begin=performance.now();this.video.getBoundingClientRect();this.canvas.getBoundingClientRect();const rectangles=performance.now();
      const entries=this.checkedGeometry?.proof?.styles??[];let current=true;
      for(const entry of entries)for(let index=0;index<entry.keys.length;index++)if(entry.style[entry.keys[index]]!==entry.values[index])current=false;
      const properties=performance.now();
      for(const entry of entries)for(let index=0;index<entry.names.length;index++)if(entry.style.getPropertyValue(entry.names[index])!==entry.namedValues[index])current=false;
      const aliases=performance.now();
      if(!first){const before=performance.now();result=original.apply(this,args);originalMs=performance.now()-before;}
      rows.push({first,originalMs,rectanglesMs:rectangles-begin,propertiesMs:properties-rectangles,aliasesMs:aliases-properties,current});
      if(rows.length===120)finish();
      return result;
    };
  });
}

export async function runProofProfile(prefix,{trace=false,instrumented=false,observers=false}={}) {
  prefix=resolve(prefix);assert(prefix.startsWith(join(ROOT,'.cache/m107/'))&&!existsSync(`${prefix}.json`));mkdirSync(dirname(prefix),{recursive:true});
  const build=verifyBuild(true),report={phase:'READ_PROFILE_NO_ACCEPTANCE',build,environment:presentationEnvironment(),started:new Date().toISOString()};
  let native,server;
  try{
    server=await presentationServer();native=await openExtension(build,(name,data)=>{if(name==='browser')report.browser=data;});
    const page=await native.context.newPage();report.placement=await nativeWindow(page,native.context);
    if(observers)await page.addInitScript(installNativeObserver,{mode:'lean'});
    await page.goto(server.url);await page.bringToFront();
    const panel=await native.popup(page);let tabId;
    try{tabId=panel.tabId;await native.workerEval(async tab=>chrome.scripting.executeScript({target:{tabId:tab,frameIds:[0]},world:'ISOLATED',files:['content.js']}),tabId);
      if(observers)await native.isolated(page,installMutationCosts);
      await native.isolated(page,()=>globalThis.__AETHERVSR_EXTENSION_TEST__.configure({presentationWatchdog:true}));
      await panel.click(`input[value="${trace?'auto':'baseline'}"]`);await panel.click('#enable');}finally{await panel.dismiss();}
    await until(()=>native.inspect(tabId),state=>state.details?.attachment?.ready&&(!trace||state.current==='neural'&&state.details.attachment.controllerState==='stable'),15000);
    if(trace){
      const session=await native.context.newCDPSession(page);
      try{
        if(instrumented)await native.isolated(page,installPresentationCosts);
        await session.send('Profiler.enable');await session.send('Profiler.setSamplingInterval',{interval:100});await session.send('Profiler.start');
        if(instrumented)await page.evaluate(withObservers=>withObservers?globalThis[Symbol.for('aethervsr.m106.native')].start(12000):window.dispatchEvent(new Event('aethervsr:m106:start')),observers);
        await page.evaluate(()=>new Promise(done=>setTimeout(done,12000)));
        if(instrumented){
          await page.evaluate(withObservers=>withObservers?globalThis[Symbol.for('aethervsr.m106.native')].done:window.dispatchEvent(new Event('aethervsr:m106:end')),observers);
        }
        const {profile}=await session.send('Profiler.stop');
        if(instrumented){
          const data=await native.isolated(page,()=>globalThis[Symbol.for('aethervsr.m107.cost')]);
          const bytes=gzipSync(JSON.stringify(data)),path=`${prefix}.cost.json.gz`;writeFileSync(path,bytes,{flag:'wx'});
          report.cost={path:relative(ROOT,path),sha256:sha256(bytes),bytes:bytes.length,guard:distribution(data.guard.map(row=>row[1])),
            observers,scope:'Same cost wrapper during intrusive CPU profiling; observers flag records the common lean native and mutation wrappers. Diagnostic only, not a registered performance window.'};
        }
        const packed=gzipSync(JSON.stringify(profile)),path=`${prefix}.cpuprofile.gz`;writeFileSync(path,packed,{flag:'wx'});
        report.profile={path:relative(ROOT,path),sha256:sha256(packed),bytes:packed.length,samples:profile.samples?.length??0,
          intervalUs:100,scope:'Separate12s intrusive native V8 CPU sampling of stable-start Auto. Not GPU, qualifying cadence or precise function wall time; overhead and sampling error apply.'};
      }finally{await session.detach();}
    }else{
      report.profile=await native.isolated(page,profileProofReads);
      report.live=await native.isolated(page,profileLiveProof);
    }
    assert.deepEqual(verifyBuild(true),build);
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

export function installMutationCosts() {
  const key=Symbol.for('aethervsr.m107.mutation-cost');
  if(globalThis[key])throw new Error('Duplicate mutation collector');
  const data=globalThis[key]={active:false,rows:[],overflow:false};
  const Original=MutationObserver;let next=0;
  globalThis.MutationObserver=class extends Original{
    constructor(callback){const id=++next;super((records,observer)=>{
      const start=performance.now();try{callback(records,observer);}finally{
        if(data.active){if(data.rows.length<20000)data.rows.push([start,performance.now()-start,id,records.length]);else data.overflow=true;}
      }
    });}
  };
  return true;
}

export function installPresentationCosts() {
  const manager=globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)],attachment=manager?.attachment;
  const driver=attachment?.driver,pipeline=attachment?.pipeline;
  if(!driver||!pipeline)throw new Error('Actual attachment pipeline required');
  const data=globalThis[Symbol.for('aethervsr.m107.cost')]={frames:[],visibility:[],geometry:[],guard:[],states:[],attempts:0,overflow:false};
  const mutation=globalThis[Symbol.for('aethervsr.m107.mutation-cost')];
  let active=false,lastState=null,attachmentMs=0,visibilityHandle;
  const restores=[];
  const push=(values,row)=>{if(values.length<30000)values.push(row);else data.overflow=true;};
  const snapshot=()=>({at:performance.now(),runtime:driver.snapshot(),status:manager.status(),attempts:data.attempts,
    pipelineError:pipeline.error?String(pipeline.error):null,sameOwner:manager.attachment===attachment});
  for(const [name,field]of [['checkGeometry','geometry'],['checkPresentation','guard']]){
    if(typeof attachment[name]!=='function')continue;
    const original=attachment[name];attachment[name]=function(...args){const start=performance.now();try{return original.apply(this,args);}finally{if(active)push(data[field],[start,performance.now()-start]);}};
    restores.push(()=>{attachment[name]=original;});
  }
  const originalAttachmentFrame=driver.onFrame;driver.onFrame=function(tick){const start=performance.now();try{return originalAttachmentFrame?.call(this,tick);}finally{attachmentMs=performance.now()-start;}};
  restores.push(()=>{driver.onFrame=originalAttachmentFrame;});
  const originalFrame=pipeline.onFrame;pipeline.onFrame=function(tick){const start=performance.now(),neural=Number(pipeline.currentUpscaler.neural);
    try{return originalFrame.call(this,tick);}finally{if(active)push(data.frames,[start,performance.now()-start,pipeline.cpuFrame.last(),attachmentMs,
      tick.presentedDelta,neural,Number(attachment.canvas.style.getPropertyValue('visibility')==='visible')]);}};
  restores.push(()=>{pipeline.onFrame=originalFrame;});
  const originalTick=pipeline.onTick;pipeline.onTick=function(tick){if(active)data.attempts++;return originalTick.call(this,tick);};
  restores.push(()=>{pipeline.onTick=originalTick;});
  const originalChange=driver.onChange;driver.onChange=function(state){originalChange?.call(this,state);
    const key=`${state.state}:${state.tier}`;if(active&&key!==lastState){push(data.states,[performance.now(),state.state,state.tier]);lastState=key;}};
  restores.push(()=>{driver.onChange=originalChange;});
  const heartbeat=()=>{if(active){push(data.visibility,[performance.now(),Number(attachment.canvas.style.getPropertyValue('visibility')==='visible')]);visibilityHandle=requestAnimationFrame(heartbeat);}};
  const start=()=>{data.opening=snapshot();active=true;if(mutation)mutation.active=true;lastState=`${data.opening.runtime.controller.state}:${data.opening.runtime.actualTier}`;visibilityHandle=requestAnimationFrame(heartbeat);};
  const end=()=>{data.closing=snapshot();active=false;if(mutation)mutation.active=false;
    cancelAnimationFrame(visibilityHandle);
    data.mutation=mutation?{rows:mutation.rows,overflow:mutation.overflow}:null;
    for(const restore of restores)restore();window.removeEventListener('aethervsr:m106:start',start);window.removeEventListener('aethervsr:m106:end',end);};
  window.addEventListener('aethervsr:m106:start',start);window.addEventListener('aethervsr:m106:end',end);
}

export function summarizeCosts(raw) {
  const native=summarizeNative({native:raw.native,runtime:null}),data=raw.cost;
  const elapsed=native.durationMs,seconds=elapsed/1000;
  const first=data.opening.runtime.session,last=data.closing.runtime.session;
  const sum=rows=>rows.reduce((total,row)=>total+row[1],0);
  const geometryMs=sum(data.geometry),guardMs=sum(data.guard);
  const delta=(owner,key)=>counterDelta(data.opening.status.details[owner].infrastructure?.[key]??data.opening.status.details[owner][key],
    data.closing.status.details[owner].infrastructure?.[key]??data.closing.status.details[owner][key]);
  return {durationMs:elapsed,nativeCallbackFps:native.nativeCallbackFps,presentedFps:(last.framesPresented-first.framesPresented)/seconds,
    renderedFps:(last.framesRendered-first.framesRendered)/seconds,frames:data.frames.length,
    core:distribution(data.frames.map(row=>row[2])),driver:distribution(data.frames.map(row=>row[1])),attachment:distribution(data.frames.map(row=>row[3])),
    geometry:distribution(data.geometry.map(row=>row[1])),guard:distribution(data.guard.map(row=>row[1])),
    geometryCallsPerSecond:data.geometry.length/seconds,guardCallsPerSecond:data.guard.length/seconds,
    geometryGuardMsPerSecond:(geometryMs+guardMs)/seconds,
    refreshCallsPerSecond:delta('attachment','refreshCalls')/seconds,
    mutationCallsPerSecond:data.mutation?data.mutation.rows.length/seconds:null,mutation:distribution(data.mutation?.rows.map(row=>row[1])??[]),
    infrastructureMsPerSecond:(delta('infrastructure','discoveryMs')+delta('infrastructure','geometryMs')+delta('attachment','geometryTotalMs'))/seconds,
    visibleFraction:data.visibility.length?data.visibility.filter(row=>row[1]).length/data.visibility.length:null,
    visibilitySamples:data.visibility.length,
    deficit:data.closing.attempts-data.opening.attempts-(last.framesRendered-first.framesRendered),
    probes:counterDelta(data.opening.runtime.controller.probeCount,data.closing.runtime.controller.probeCount),
    transitions:counterDelta(data.opening.runtime.controller.transitionCount,data.closing.runtime.controller.transitionCount),
    tierStable:data.frames.every(row=>row[5]===1)&&data.states.every(row=>row[2]==='neural')&&data.opening.runtime.actualTier==='neural'&&data.closing.runtime.actualTier==='neural',
    error:data.closing.pipelineError??(raw.errors?.length?raw.errors.join('; '):null),ownerStable:data.closing.sameOwner,
    gpu:last.gpu.neural,layoutStyleMs:null,
    scope:'Frame core brackets import/encode/submit. Driver includes attachment/guard and must not be added to them. Guard+full reconciliation are non-nested standalone calls. CPU raw timer quantization applies; layout/style attribution unmeasured. GPU histogram is cumulative since attachment construction, cold/pre-window included, not a window quantile. Common rAF visibility heartbeat counts all window updates including hidden states, not physical scanout.'};
}

export function costBounds(values) {
  assert(values.length===6&&values.every(Number.isFinite));
  const mean=values.reduce((sum,value)=>sum+value,0)/6,sd=Math.sqrt(values.reduce((sum,value)=>sum+(value-mean)**2,0)/5);
  return {values,mean,sd,lower95:mean-2.0150483733*sd/Math.sqrt(6),upper95:mean+2.0150483733*sd/Math.sqrt(6),
    twoSided95:[mean-2.5705818356*sd/Math.sqrt(6),mean+2.5705818356*sd/Math.sqrt(6)],count:6};
}

export function compareCosts(results) {
  assert.equal(results.length,18);
  const runIntegrity=row=>row.valid&&row.summary.tierStable&&row.summary.probes===0&&row.summary.ownerStable&&row.summary.error===null&&row.summary.deficit===0;
  const controlValid=results.filter(row=>row.case.strategy===0).length===6&&results.filter(row=>row.case.strategy===0).every(runIntegrity);
  const output={};
  for(const strategy of[1,2]){
    const contrasts={native:[],rendered:[],core:[],driver:[],geometry:[]};
    for(let block=1;block<=6;block++){
      const control=results.find(row=>row.case.block===block&&row.case.strategy===0)?.summary,candidate=results.find(row=>row.case.block===block&&row.case.strategy===strategy)?.summary;
      assert(control&&candidate);
      for(const [name,left,right]of [['native',control.nativeCallbackFps,candidate.nativeCallbackFps],['rendered',control.renderedFps,candidate.renderedFps],
        ['core',candidate.core.p95,control.core.p95],['driver',candidate.driver.p95,control.driver.p95],['geometry',candidate.geometryGuardMsPerSecond,control.geometryGuardMsPerSecond]]){
        assert(Number.isFinite(left)&&Number.isFinite(right));contrasts[name].push(left-right);
      }
    }
    const bounds=Object.fromEntries(Object.entries(contrasts).map(([key,values])=>[key,costBounds(values)]));
    const safety=controlValid&&results.filter(row=>row.case.strategy===strategy).every(row=>runIntegrity(row)&&Number.isFinite(row.summary.visibleFraction)&&row.summary.visibleFraction>=.95
      &&(strategy!==2||row.summary.guard.count>0&&Number.isFinite(row.summary.guard.p95)&&Number.isFinite(row.summary.guard.max)&&row.summary.guard.p95<=.2&&row.summary.guard.max<=2));
    output[strategy]={bounds,controlValid,safety,pass:safety&&bounds.native.upper95<=.5&&bounds.rendered.upper95<=.5&&bounds.core.upper95<=.25&&bounds.driver.upper95<=.25&&bounds.geometry.upper95<=3};
  }
  return output;
}

export function costCases({confirmation=false,probe=false}={}) {
  assert(!(confirmation&&probe),'A cost probe is not a production confirmation');
  return probe?[{block:1,strategy:2,durationMs:60000}]:confirmation?[{block:1,strategy:2,durationMs:120000},{block:2,strategy:2,durationMs:120000}]:COST_ORDER;
}

export async function runCostStudy(prefix,{confirmation=false,production=false,probe=false}={}) {
  prefix=resolve(prefix);assert(prefix.startsWith(join(ROOT,'.cache/m107/'))&&!existsSync(`${prefix}.json`));mkdirSync(dirname(prefix),{recursive:true});
  const candidate=verifyBuild(!production),directory=join(ROOT,'.cache/m107/frozen-s0');
  const baseline={directory,provenance:JSON.parse(readFileSync(join(directory,'build-provenance.json')))};
  for(const [name,value]of Object.entries(baseline.provenance.files))assert.equal(sha256(readFileSync(join(directory,name))),value.sha256);
  const cases=costCases({confirmation,probe});
  const report={phase:probe?'COST_PROBE_NO_ACCEPTANCE':confirmation?'FINAL_CONFIRMATION':'STRATEGY_COST',candidate,baseline,started:new Date().toISOString(),cases,results:[],completion:'UNVERIFIED',environment:presentationEnvironment()};
  let server,native;
  try{
    server=await presentationServer();
    for(const [ordinal,item]of cases.entries()){
      const result={case:item,ordinal:ordinal+1,valid:false};report.results.push(result);
      native=await openExtension(item.strategy===0?baseline:candidate,(name,data)=>{if(name==='browser')result.browser=data;});
      try{
        const page=await native.context.newPage();result.placement=await nativeWindow(page,native.context);
        const errors=[];page.on('pageerror',error=>errors.push(String(error)));
        await page.addInitScript(installNativeObserver,{mode:'lean'});await page.goto(server.url);await page.bringToFront();
        await page.waitForFunction(()=>globalThis[Symbol.for('aethervsr.m106.native')]?.callbacks>=2);
        const panel=await native.popup(page);let tabId;
        try{tabId=panel.tabId;await native.workerEval(async tab=>chrome.scripting.executeScript({target:{tabId:tab,frameIds:[0]},world:'ISOLATED',files:['content.js']}),tabId);
          await native.isolated(page,installMutationCosts);
          if(item.strategy>0&&!production)await native.isolated(page,flag=>globalThis.__AETHERVSR_EXTENSION_TEST__.configure({presentationWatchdog:flag}),item.strategy===2);
          await panel.click('#enable');
        }finally{await panel.dismiss();}
        await until(()=>native.inspect(tabId),state=>state.current==='neural'&&state.details?.attachment?.controllerState==='stable'&&state.details.attachment.ready,15000);
        result.adapter=await native.isolated(page,()=>globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)].attachment.gpu.adapterReport);
        await native.isolated(page,installPresentationCosts);
        await page.evaluate(()=>new Promise(done=>setTimeout(done,5000)));
        await page.evaluate(ms=>globalThis[Symbol.for('aethervsr.m106.native')].start(ms),item.durationMs);
        console.log(`cost ${ordinal+1}/${cases.length} S${item.strategy} ${item.durationMs/1000}s`);
        await bounded(page.evaluate(()=>globalThis[Symbol.for('aethervsr.m106.native')].done),item.durationMs+10000);
        const observation=await page.evaluate(()=>{const{done,start,...data}=globalThis[Symbol.for('aethervsr.m106.native')];return data;});
        const cost=await native.isolated(page,()=>globalThis[Symbol.for('aethervsr.m107.cost')]);
        const raw={native:observation,cost,errors},packed=gzipSync(JSON.stringify(raw)),path=`${prefix}.${ordinal+1}.json.gz`;
        writeFileSync(path,packed,{flag:'wx'});result.raw={path:relative(ROOT,path),sha256:sha256(packed),bytes:packed.length};
        result.summary=summarizeCosts(raw);
        try{validateNative({native:observation,runtime:null},item.durationMs);assert.equal(cost.overflow,false);assert.equal(cost.mutation.overflow,false);assert.equal(cost.frames.length,cost.closing.runtime.session.framesRendered-cost.opening.runtime.session.framesRendered);result.valid=true;}
        catch(error){result.error=String(error);}
        const disable=await native.popup(page);try{result.disabled=await disable.click('#disable');}finally{await disable.dismiss();}
        console.log(JSON.stringify({ordinal:ordinal+1,strategy:item.strategy,valid:result.valid,rate:result.summary.renderedFps,guard:result.summary.guard,work:result.summary.geometryGuardMsPerSecond,visible:result.summary.visibleFraction}));
        if(!result.valid)throw new Error('Invalid cost trial; retain raw and stop fixed order');
      }finally{await native.close();native=null;}
    }
    assert.deepEqual(verifyBuild(!production),candidate);report.completion='CAPTURED';if(!confirmation&&!probe)report.comparison=compareCosts(report.results);
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}