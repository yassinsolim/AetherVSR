import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

function check(source: string): void {
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { installNativeObserver, installRuntimeCounters, summarizeNative, validateNative, pairedBounds, floorDecision, observerComparison } from './tools/m106-counters.mjs';
    import { activeSafety, parseFloorPlan, PRIMARY_ORDER, qualifiesDisabledGeometry, resumePrefix, floorCases, bindingNativeFailure } from './tools/m106-floor.mjs';
    import { installPublicObserver, samePublicIdentity, targetStalls, publicCausality, advancedForOpening, loadCommittedReference, retainBindingPublicFailure, publicStateFailure, manifestIdentity, observeManifests, advertisedCatalog, pairedSeekConditions, PUBLIC_ORDER } from './tools/m106-sites.mjs';
    import { runInNewContext } from 'node:vm';
    ${source}
    console.log('checked');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 15000 });
  expect(output.trim()).toBe('checked');
}

const fixture = `
 const boundary={callbacks:10,presented:100,gaps:0,quality:{total:120,dropped:2},currentTime:1,readyState:4,networkState:1,paused:false,rate:1,source:'local',generation:1,visibility:'visible',focused:true};
 const raw={native:{mode:'lean',invalid:null,overflow:false,opening:{...boundary,at:0},closing:{...boundary,at:1000,callbacks:12,presented:104,gaps:2,quality:{total:180,dropped:3}},
 rows:[[100,100,1.1,90,110,102,2,140,2,1.1,4,1,1],[500,500,1.5,490,510,104,2,160,3,1.5,4,1,1]],events:[],raf:[]},runtime:null};
`;

const observerFixture = `
  const callbacks=new Map(),timers=new Map();let nextId=0,clock=0;
  class TrackedTarget extends EventTarget{
    listeners=0;
    addEventListener(...args){this.listeners++;super.addEventListener(...args);}
    removeEventListener(...args){this.listeners--;super.removeEventListener(...args);}
  }
  const video=Object.assign(new TrackedTarget(),{currentTime:1,duration:4,readyState:4,networkState:1,paused:false,ended:false,
    playbackRate:1,currentSrc:'local',getVideoPlaybackQuality:()=>({totalVideoFrames:100+clock,droppedVideoFrames:0,creationTime:clock}),
    requestVideoFrameCallback:callback=>{const id=++nextId;callbacks.set(id,callback);return id;},cancelVideoFrameCallback:id=>callbacks.delete(id)});
  const document=Object.assign(new TrackedTarget(),{visibilityState:'visible',hasFocus:()=>true,querySelector:()=>video});
  const window=new TrackedTarget();
  const context={document,window,Event,performance:{now:()=>clock},setTimeout:(callback,ms)=>{const id=++nextId;timers.set(id,{callback,ms});return id;},clearTimeout:id=>timers.delete(id)};
  const frame=(presentedFrames,mediaTime)=>{const [id,callback]=callbacks.entries().next().value;callbacks.delete(id);callback(clock,{presentedFrames,mediaTime,presentationTime:clock-5,expectedDisplayTime:clock+5});};
  const flush=()=>{const entries=[...timers.values()];timers.clear();for(const entry of entries)entry.callback();};
`;

describe('native browser delivery floor accounting', () => {
  it('keeps native public-stall reproduction distinct from a safe public-scope pass', () => check(`
    const results=PUBLIC_ORDER.map(arm=>({case:{arm},completion:'CAPTURED',observedMs:180000,comparable:true,actions:[{name:'prescribed-seek',pass:true,result:{width:2560,height:1440,duration:35.963044}}],stalls:[{}],safetyPass:true,journeyPass:false,recoveryComparison:'UNRESOLVED'}));
    assert.equal(publicCausality(results).verdict,'PLAYER/SOURCE REPRODUCED WITHOUT AETHERVSR');
    assert.equal(publicCausality(results).videojsScopePass,false);
    for(const result of results)result.recoveryComparison='NO_OBSERVED_WORSENING';
    assert.equal(publicCausality(results).videojsScopePass,true);
    results.find(result=>result.case.arm==='S').safetyPass=false;assert.equal(publicCausality(results).videojsScopePass,false);
    for(const result of results.filter(value=>['P','Q'].includes(value.case.arm)))result.stalls=[];
    assert.equal(publicCausality(results).verdict,'EXTENSION-ASSOCIATED');
    results[0].actions[0].pass=false;assert.equal(publicCausality(results).verdict,'UNRESOLVED');results[0].actions[0].pass=true;
    results[0].comparable=false;assert.equal(publicCausality(results).verdict,'UNRESOLVED');
  `));
  it('requires two seconds from the original opening media time and a committed identity record', () => check(`
    assert.equal(advancedForOpening({time:3.99,paused:false},2),false);assert.equal(advancedForOpening({time:4,paused:false},2),true);
    assert.equal(advancedForOpening({time:4,paused:true},2),false);
    assert.throws(()=>loadCommittedReference('.cache/m106/uncommitted.json'),/committed results/);
    assert.equal(retainBindingPublicFailure({errors:{counts:{extension:1}}}),true);
    assert.equal(retainBindingPublicFailure({bindingFailure:true}),true);
    assert.equal(retainBindingPublicFailure({errors:{counts:{'third-party-page':1}}}),false);
  `));
  it('retains an observed extension failure when an external interruption follows it', () => check(fixture + `
    const status={enabled:true,owner:'owner-1',code:'active',details:{infrastructure:{created:1,destroyed:0,maximumConcurrent:1},
      attachment:{infrastructure:{cleanupErrors:0},resources:{device:1,pipeline:1,canvas:1,resizeObservers:1}}}};
    assert.equal(publicStateFailure(status,'owner-1'),false);status.owner='owner-2';
    const item={bindingFailure:publicStateFailure(status,'owner-1'),error:'External integrity event: blur'};
    assert.equal(retainBindingPublicFailure(item),true);
    raw.errors=[];raw.native.invalid='Integrity event: blur';assert.equal(bindingNativeFailure(raw),false);
    raw.native.failures=['Original media error'];assert.equal(bindingNativeFailure(raw),true);
  `));
  it('does not exempt blur or unexpected playback changes during public actions', () => check(observerFixture + `
    context.setInterval=context.setTimeout;context.clearInterval=context.clearTimeout;
    const data=runInNewContext('('+installPublicObserver.toString()+')();globalThis[Symbol.for("aethervsr.m106.public")]',context);
    data.start();data.action='original-fullscreen';window.dispatchEvent(new Event('resize'));assert.equal(data.invalid,null);
    window.dispatchEvent(new Event('blur'));assert.equal(data.invalid,'External integrity event: blur');
    data.invalid=null;data.action='original-pause-resume';video.dispatchEvent(new Event('pause'));assert.equal(data.invalid,null);
    data.action=null;video.dispatchEvent(new Event('pause'));assert.equal(data.invalid,'Unexpected media event: pause');
    data.stop();assert.equal(timers.size,0);assert.equal(video.listeners+window.listeners+document.listeners,0);
  `));
  it('requires a matched public asset and never invents a substitute seek target', () => check(`
    const reference={urlSha256:'a'.repeat(64),origin:'https://example.test',duration:35.963044,width:1280,height:720};
    assert.equal(samePublicIdentity(reference,reference),true);
    for(const changed of[{duration:36},{width:1920},{urlSha256:'b'.repeat(64)},{duration:null}])assert.equal(samePublicIdentity({...reference,...changed},reference),false);
    assert.equal(samePublicIdentity({...reference,duration:35.925333},reference,'closing'),true);
    assert.equal(samePublicIdentity({...reference,urlSha256:'b'.repeat(64)},reference,'closing'),false);
    assert.throws(()=>samePublicIdentity(reference,reference,'seek'));
  `));
  it('compares stable HLS identities without equating ephemeral blob URLs across sessions', () => check(`
    const master=manifestIdentity('https://stream.example.test/asset.m3u8','application/x-mpegURL',200);
    const rendition=manifestIdentity('https://cdn.example.test/rendition.m3u8?signature=first&expires=1','application/x-mpegURL',200);
    assert.equal(manifestIdentity('https://example.test/script.js','text/javascript',200),null);
    assert.equal(manifestIdentity('http://example.test/asset.m3u8','application/x-mpegURL',200),null);
    const reference={scheme:'blob:',origin:'https://example.test',urlSha256:'a'.repeat(64),duration:35.95354,width:960,height:540,
      manifests:{overflow:false,records:[master,rendition]},hls:{master:{origin:master.origin,urlSha256:master.urlSha256},renditionPathSha256:[rendition.pathSha256]}};
    const actual={...reference,urlSha256:'b'.repeat(64),manifests:{overflow:false,records:[master,
      manifestIdentity('https://cdn.example.test/rendition.m3u8?signature=second&expires=2','application/x-mpegURL',200)]}};
    assert.equal(samePublicIdentity(actual,reference),true);
    actual.manifests.records[1].pathSha256='c'.repeat(64);assert.equal(samePublicIdentity(actual,reference),false);
    actual.manifests.records[1]=rendition;actual.manifests.overflow=true;assert.equal(samePublicIdentity(actual,reference),false);
    assert(!JSON.stringify(reference).includes('signature=first'));
  `));
  it('never erases a failed manifest response when the same URL later succeeds', () => check(`
    let responseHandler,removed=false;
    const page={on(type,callback){assert.equal(type,'response');responseHandler=callback;},off(type,callback){removed=type==='response'&&callback===responseHandler;}};
    const observer=observeManifests(page);
    const response=status=>({url:()=> 'https://stream.example.test/asset.m3u8',headers:()=>({'content-type':'application/x-mpegURL'}),status:()=>status});
    responseHandler(response(503));responseHandler(response(200));
    assert.equal(observer.snapshot().records[0].status,503);
    observer.stop();assert.equal(removed,true);
  `));
  it('pins advertised variants while allowing original ABR selection within that catalog', () => check(`
    const address='https://stream.example.test/master.m3u8';
    const text='#EXTM3U\\n#EXT-X-STREAM-INF:BANDWIDTH=1000,RESOLUTION=960x540,CODECS="avc1.test,mp4a.40.2"\\nlow.m3u8?signature=one\\n#EXT-X-STREAM-INF:BANDWIDTH=3000,RESOLUTION=2560x1440,CODECS="avc1.test,mp4a.40.2"\\nhigh.m3u8?signature=one\\n';
    const catalog=advertisedCatalog(text,address),master={...manifestIdentity(address,'application/x-mpegURL',200),catalog,pendingCatalogReads:0};
    assert.deepEqual(advertisedCatalog(text.replaceAll('signature=one','signature=two'),address),catalog);
    const reference={scheme:'blob:',urlSha256:'a'.repeat(64),origin:'https://example.test',duration:35.95354,width:960,height:540,
      hls:{master:{origin:master.origin,urlSha256:master.urlSha256},catalog},manifests:{overflow:false,records:[master]}};
    const actual={...reference,width:2560,height:1440,urlSha256:'b'.repeat(64),manifests:{overflow:false,records:[master,manifestIdentity('https://cdn.example.test/high.m3u8?signature=two','application/x-mpegURL',200)]}};
    assert.equal(samePublicIdentity(actual,reference),true);actual.width=1920;assert.equal(samePublicIdentity(actual,reference),false);
    actual.width=2560;actual.manifests.records.push(manifestIdentity('https://cdn.example.test/unknown.m3u8?signature=two','application/x-mpegURL',200));assert.equal(samePublicIdentity(actual,reference),false);
    assert.throws(()=>advertisedCatalog('x'.repeat(131073),address));
    const results=PUBLIC_ORDER.map(arm=>({case:{arm},actions:[{name:'prescribed-seek',pass:true,result:{width:2560,height:1440,duration:35.963044}}]}));
    assert.equal(pairedSeekConditions(results).pass,true);results[2].actions[0].result.width=960;assert.equal(pairedSeekConditions(results).pass,false);
    results[2].actions[0].result.width=2560;
    results[0].actions[0].result.duration=35.9531;results[1].actions[0].result.duration=35.9634;
    assert.equal(pairedSeekConditions(results).pass,false);assert.equal(publicCausality(results).verdict,'UNRESOLVED');
  `));
  it('retains catalog drift across repeated successful master responses', () => check(`
    let handle,bodyReads=0;
    const page={on(type,callback){handle=callback;},off(){}};
    const observer=observeManifests(page);
    const response=bandwidth=>({url:()=> 'https://stream.example.test/master.m3u8',headers:()=>({'content-type':'application/x-mpegURL'}),status:()=>200,
      body:async()=>{bodyReads++;return Buffer.from('#EXTM3U\\n#EXT-X-STREAM-INF:BANDWIDTH='+bandwidth+',RESOLUTION=960x540\\nlow.m3u8?signature=1\\n');}});
    handle(response(1000));await observer.settled();assert.equal(observer.snapshot().records[0].catalogError,undefined);
    handle(response(2000));await observer.settled();
    assert.equal(bodyReads,2);assert.equal(observer.snapshot().records[0].catalogError,'Advertised master catalog changed');
    handle(response(1000));await observer.settled();assert.equal(observer.snapshot().records[0].catalogError,'Advertised master catalog changed');
    assert.equal(observer.snapshot().records[0].pendingCatalogReads,0);observer.stop();
  `));
  it('retains qualifying near-end stalls and censors recovery at scheduled actions', () => check(`
    const rows=Array.from({length:81},(_,index)=>({at:30000+index*250,currentTime:35.8852,qualityTotal:844,paused:false,ended:false,readyState:2,networkState:2}));
    const commands=[{name:'prescribed-seek',at:30000,pass:true},{name:'original-pause-resume',at:50000,pass:true}];
    const episodes=targetStalls(rows,commands,51000);assert.equal(episodes.length,1);assert.equal(episodes[0].censoredAt,50000);
    assert.equal(episodes[0].durationLowerMs,19750);assert.equal(episodes[0].recoveryInterval,null);
    assert.equal(targetStalls(rows.slice(0,60),commands,45000).length,0);
    assert.equal(targetStalls(rows.map(row=>({...row,ended:true})),commands,51000).length,0);
    assert.equal(targetStalls(rows,[],51000).length,0);
    assert.equal(targetStalls(rows.map(row=>({...row,qualityTotal:null})),commands,51000).length,0);
    assert.equal(targetStalls(rows,[...commands,{name:'original-ended-replay',at:30250,pass:true}],51000).length,0);
  `));
  it('retains a real submission when the original post-submit callback throws', () => check(`
    let submitted=0;const window=new EventTarget();
    const pipeline={currentUpscaler:{neural:true},error:null,onTick(tick){this.onFrame(tick);},onFrame(){submitted++;throw new Error('post-submit failure');}};
    const driver={onChange:null,snapshot:()=>({session:{framesRendered:submitted},controller:{state:'stable',tier:'neural'}})};
    const attachment={pipeline,driver,snapshot:()=>({})};const manager={attachment,status:()=>({})};
    const context={manager,chrome:{runtime:{id:'test'}},window,Event,performance:{now:()=>10}};
    const originalTick=pipeline.onTick,originalFrame=pipeline.onFrame;
    const data=runInNewContext('globalThis[Symbol.for("aethervsr.m10.document.test")]=manager;('+installRuntimeCounters.toString()+')();globalThis[Symbol.for("aethervsr.m106.runtime")]',context);
    window.dispatchEvent(new Event('aethervsr:m106:start'));
    assert.throws(()=>pipeline.onTick({presentedDelta:2}),/post-submit failure/);
    window.dispatchEvent(new Event('aethervsr:m106:end'));
    assert.equal(data.frames.length,1);assert.equal(data.attempts,1);assert.equal(data.closing.runtime.session.framesRendered,1);
    assert.equal(pipeline.onTick,originalTick);assert.equal(pipeline.onFrame,originalFrame);
  `));
  it('qualifies hidden live geometry using the attachment Rect contract', () => check(`
    const state=(left,geometryCalls)=>({cssRect:{left,top:120,width:640,height:360},connected:true,visibility:'hidden',pointerEvents:'none',
      snapshot:{infrastructure:{geometryCalls},resources:{device:0,pipeline:0,frameCallback:0}}});
    const opening=state(40,1),moved=state(50,2),restored=state(40,3);
    assert.equal(qualifiesDisabledGeometry(opening,moved,restored),true);
    moved.visibility='visible';assert.equal(qualifiesDisabledGeometry(opening,moved,restored),false);moved.visibility='hidden';
    moved.snapshot.infrastructure.geometryCalls=1;assert.equal(qualifiesDisabledGeometry(opening,moved,restored),false);
    assert.equal(qualifiesDisabledGeometry(opening,undefined,restored),false);
  `));
  it('executes native callback bookkeeping and removes observer-owned handles without a GPU', () => check(observerFixture + `
    const data=runInNewContext('('+installNativeObserver.toString()+')({mode:"lean"});globalThis[Symbol.for("aethervsr.m106.native")]',context);
    document.dispatchEvent(new Event('DOMContentLoaded'));
    frame(10,1);clock=16;frame(11,1.016);data.start(1000);
    clock=40;frame(14,1.04);assert.equal(data.rows.length,1);assert.equal(data.gaps,2);
    clock=48;video.dispatchEvent(new Event('seeking'));frame(15,0);
    assert.equal(data.rows[1][12],data.opening.generation);
    clock=1016;flush();flush();await data.done;
    assert.equal(data.closing.presented-data.opening.presented,4);assert.equal(data.invalid,null);
    assert.equal(callbacks.size,0);assert.equal(timers.size,0);assert.equal(video.listeners+window.listeners+document.listeners,0);
    assert.equal(data.tasks.length,0);assert.equal(data.raf.length,0);
  `));
  it('invalidates a transient resize even when final native bounds would match', () => check(observerFixture + `
    const data=runInNewContext('('+installNativeObserver.toString()+')({mode:"lean"});globalThis[Symbol.for("aethervsr.m106.native")]',context);
    document.dispatchEvent(new Event('DOMContentLoaded'));
    frame(10,1);clock=16;frame(11,1.016);data.start(1000);clock=100;window.dispatchEvent(new Event('resize'));flush();await data.done;
    assert.equal(data.invalid,'Integrity event: resize');assert.equal(callbacks.size,0);
    assert.equal(video.listeners+window.listeners+document.listeners,0);
  `));
  it('never requests a video callback in the no-observer comparison', () => check(observerFixture + `
    const data=runInNewContext('('+installNativeObserver.toString()+')({mode:"none"});globalThis[Symbol.for("aethervsr.m106.native")]',context);
    document.dispatchEvent(new Event('DOMContentLoaded'));data.start(1000);
    assert.equal(callbacks.size,0);clock=1000;flush();flush();await data.done;
    assert.equal(data.closing.callbacks,null);assert.equal(data.rows.length,0);
  `));
  it('defines an exact native analogue without a runtime, GPU or invented zero processing rate', () => check(fixture + `
    const summary=validateNative(raw,1000);
    assert.equal(summary.callbacks,2);assert.equal(summary.presented,4);assert.equal(summary.gaps,2);
    assert.equal(summary.nativeCombinedPercent,75);assert.equal(summary.callbackGapPercent,50);
    assert.equal(summary.qualityDropPercent,100/60);assert.equal(summary.nativeCallbackFps,2);
    assert.equal(summary.renderedFps,null);assert.equal(summary.historicalRuntimeCombinedPercent,null);
    assert.equal(summary.submissionDeficit,null);assert.equal(summary.counterOverlap,null);
  `));
  it('rejects source/counter resets, short windows and imperfect callback accounting', () => check(fixture + `
    assert.throws(()=>validateNative(raw,1001));
    raw.native.closing.presented=105;assert.throws(()=>validateNative(raw,1000));raw.native.closing.presented=104;
    raw.native.closing.generation=2;assert.throws(()=>validateNative(raw,1000));raw.native.closing.generation=1;
    raw.native.closing.quality.dropped=0;assert.equal(summarizeNative(raw).nativeCombinedPercent,null);assert.throws(()=>validateNative(raw,1000));
  `));
  it('leaves no-observer callback quantities unavailable while retaining boundary quality metrics', () => check(fixture + `
    raw.native.mode='none';raw.native.rows=[];
    for(const boundary of[raw.native.opening,raw.native.closing]){boundary.callbacks=null;boundary.presented=null;boundary.gaps=null;}
    const summary=validateNative(raw,1000);assert.equal(summary.nativeCombinedPercent,null);assert.equal(summary.callbacks,null);
    assert.equal(summary.qualityDrops,1);assert.equal(summary.qualityTotal,60);
  `));
  it('leaves infrastructure-only runtime slice rates unavailable instead of inventing zero submissions', () => check(fixture + `
    raw.native.closing.at=600000;
    raw.runtime={opening:{runtime:null,attempts:null},closing:{runtime:null,attempts:null},frames:[]};
    const summary=summarizeNative(raw);
    for(const slice of[summary.first120,summary.last120]){
      assert.equal(slice.submitted,null);assert.equal(slice.renderedFps,null);assert.equal(slice.runtimePresentedFps,null);
      assert(Number.isFinite(slice.nativeCallbackFps));
    }
    assert.equal(summary.submitted,null);assert.equal(summary.renderedFps,null);
  `));
  it('reports real runtime counter mismatches separately from the common native analogue', () => check(fixture + `
    raw.runtime={overflow:false,opening:{attempts:0,runtime:{session:{framesRendered:0,framesPresented:0,framesSkipped:0}}},
      closing:{attempts:2,pipelineError:null,sameAttachment:true,runtime:{session:{framesRendered:2,framesPresented:4,framesSkipped:2}}},frames:[[100,2,1],[500,2,1]]};
    const summary=validateNative(raw,1000);assert.equal(summary.historicalRuntimeCombinedPercent,75);assert.equal(summary.submissionDeficit,0);
    assert.deepEqual(summary.runtimeNativeAlignment,{presented:0,skipped:0,submittedMinusCallbacks:0});
    raw.runtime.closing.attempts=3;raw.runtime.closing.pipelineError='device lost';
    assert.equal(validateNative(raw,1000).submissionDeficit,1);
    assert.equal(raw.runtime.closing.pipelineError,'device lost');
  `));
  it('separates exact session quality accumulation from the historical boundary-read capture formula', () => check(fixture + `
    raw.runtime={opening:{attempts:0,runtime:{session:{framesRendered:0,framesPresented:0,framesSkipped:0,decoderDrops:2}}},
      closing:{attempts:2,runtime:{session:{framesRendered:2,framesPresented:4,framesSkipped:2,decoderDrops:4}}},frames:[[100,2,1],[500,2,1]]};
    const summary=summarizeNative(raw);
    assert.equal(summary.nativeCombinedPercent,75);assert.equal(summary.historicalRuntimeCombinedPercent,75);
    assert.equal(summary.runtimeDecoderDrops,2);assert.equal(summary.runtimeSessionCombinedPercent,100);assert.equal(summary.importedFrames,null);
    raw.runtime.closing.runtime.session.decoderDrops=null;assert.equal(summarizeNative(raw).runtimeSessionCombinedPercent,null);
    assert.equal(summarizeNative(raw).historicalRuntimeCombinedPercent,75);
  `));
  it('retains closing loop waits and completely stalled playback as outcomes, not replacement opportunities', () => check(fixture + `
    raw.native.closing.readyState=1;assert.equal(validateNative(raw,1000).callbacks,2);
    raw.native.closing.mediaError=3;raw.native.closing.paused=true;assert.equal(validateNative(raw,1000).callbacks,2);
    raw.native.closing.mediaError=null;assert.throws(()=>validateNative(raw,1000));
    raw.native.rows=[];raw.native.closing={...raw.native.opening,at:1000};
    const summary=validateNative(raw,1000);assert.equal(summary.qualityTotal,0);assert.equal(summary.nativeCombinedPercent,null);
  `));
  it('uses four run-level observations and fixed t bounds, never frame pseudoreplication', () => check(`
    const bounds=pairedBounds([0,0,0,0]);assert.equal(bounds.upper95,0);assert.equal(bounds.sd,0);
    assert.throws(()=>pairedBounds([1,2,3]));assert.throws(()=>pairedBounds([1,2,3,NaN]));
    const spread=pairedBounds([-1,0,0,1]);assert(spread.lower95<0&&spread.upper95>0);
  `));
  it('does not infer uninstrumented callback neutrality from matching quality counters', () => check(`
    const results=floorCases('observer').map(item=>({case:item,completion:'CAPTURED',summary:{qualityTotalFps:60,qualityDropPercent:0,
      nativeCallbackFps:item.mode==='none'?null:59,nativeCombinedPercent:item.mode==='none'?null:2}}));
    const result=observerComparison(results);assert.equal(result.observerCompatible,null);assert.equal(result.callbackDistortionBound,null);
    assert.equal(result.readinessNormalizationJustified,false);assert.equal(result.blocks.length,3);
    assert(result.blocks.every(block=>block.richMinusLeanCallbackFps===0));
  `));
  it('enforces the frozen order before executing an evidence plan', () => check(`
    for(const [phase,count]of[['observer',9],['primary',16],['proxy',3]])assert.equal(parseFloorPlan(floorCases(phase)).length,count);
    const cases=PRIMARY_ORDER.map((arm,index)=>({id:'primary-'+index,arm,phase:'primary',mode:'lean',durationMs:600000}));
    assert.equal(parseFloorPlan(cases).length,16);
    assert.throws(()=>parseFloorPlan(cases.slice(1)));
    [cases[0],cases[1]]=[cases[1],cases[0]];assert.throws(()=>parseFloorPlan(cases));
  `));
  it('resumes only the next interrupted ordinal without replacing captured failures', () => check(`
    const cases=parseFloorPlan(PRIMARY_ORDER.map((arm,index)=>({id:'primary-'+index,arm,phase:'primary',mode:'lean',durationMs:600000})));
    const failed={case:cases[0],completion:'CAPTURED',safety:{pass:false},raw:{sha256:'retained'}};
    const prior={cases,completion:'UNVERIFIED',results:[failed,{case:cases[1],completion:'UNVERIFIED'}]};
    assert.deepEqual(resumePrefix(prior,cases),[failed]);
    prior.results[1].case=cases[2];assert.throws(()=>resumePrefix(prior,cases));
    prior.results=[failed];assert.throws(()=>resumePrefix(prior,cases));
  `));
  it('requires each active run to pass average, final-window and error-aware cleanup gates', () => check(fixture + `
    const counts={ownerChanges:1,created:1,maximumConcurrent:1};
    const status={details:{infrastructure:counts}};
    const controller={state:'stable',tier:'neural'};
    const runtime={controller,actualTier:'neural',session:{framesRendered:0,framesPresented:0,framesSkipped:0}};
    raw.native.closing.at=600000;
    raw.runtime={opening:{attempts:0,status,runtime},closing:{attempts:35400,status,pipelineError:null,sameAttachment:true,
      runtime:{...runtime,session:{framesRendered:35400,framesPresented:35900,framesSkipped:500}}},
      states:[],frames:Array.from({length:7080},(_,index)=>[480001+index*16.7,1,1])};
    raw.errors=[];
    const cleanup={domPreserved:true,status:{enabled:false,details:{timerCount:0,discoveryActive:false}},attachment:{resources:{device:0,canvas:0},infrastructure:{cleanupErrors:0}}};
    assert.equal(activeSafety(raw,'D',cleanup).pass,true);
    cleanup.attachment.infrastructure.cleanupErrors=1;assert.equal(activeSafety(raw,'D',cleanup).pass,false);cleanup.attachment.infrastructure.cleanupErrors=0;
    raw.runtime.frames=raw.runtime.frames.slice(0,6800);assert.equal(activeSafety(raw,'D',cleanup).checks.lastRendered,false);
    raw.runtime.closing.runtime.session.framesRendered=34000;assert.equal(activeSafety(raw,'D',cleanup).checks.averageRendered,false);
    raw.runtime.closing.runtime.session.framesRendered=null;assert.equal(activeSafety(raw,'D',cleanup).checks.averageRendered,false);
    assert.equal(activeSafety(raw,'D',undefined).checks.cleanup,false);
  `));
  it('does not turn uncertainty, a safety failure or absent observer justification into Case C', () => check(`
    const make=()=>Array.from({length:4},()=>({A:{nativeCombinedPercent:3,nativeCallbackFps:59},B:{nativeCombinedPercent:3,nativeCallbackFps:59,safety:true},C:{nativeCombinedPercent:3,renderedFps:59,safety:true},D:{nativeCombinedPercent:3,renderedFps:59,safety:true}}));
    let blocks=make();assert.equal(floorDecision(blocks,true).case,'CASE C');assert.equal(floorDecision(blocks,null).case,'CASE D');
    blocks[0].D.safety=false;assert.equal(floorDecision(blocks,true).case,'CASE D');
    blocks=make();blocks[0].B.safety=false;assert.equal(floorDecision(blocks,true).case,'CASE D');
    blocks=make();for(const block of blocks)block.A.nativeCombinedPercent=.5;assert.equal(floorDecision(blocks,true).case,'CASE A');
    blocks=make();for(const block of blocks)block.D.nativeCombinedPercent=4;assert.equal(floorDecision(blocks,true).case,'CASE B');
    blocks=make();blocks[0].D.nativeCombinedPercent=1;blocks[1].D.nativeCombinedPercent=6;assert.equal(floorDecision(blocks,true).case,'CASE D');
    for(const field of['nativeCombinedPercent','renderedFps'])for(const value of[null,undefined,NaN,Infinity]){
      blocks=make();blocks[0].D[field]=value;const result=floorDecision(blocks,true);
      assert.equal(result.case,'CASE D');assert.equal(result.readinessADRPermitted,false);
    }
  `));
});