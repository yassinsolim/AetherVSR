import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

function check(source: string) {
  const result = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import {readFileSync,rmSync} from 'node:fs';
    import {createHash} from 'node:crypto';
    import {buildPlayer,manifest} from './tools/m1010/build.mjs';
    const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
    ${source}
    console.log('checked');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 60000 });
  expect(result.trim()).toBe('checked');
}

describe('M10.10 research package isolation', () => {
  it('records a static display frame without assuming recurring capture callbacks', () => check(`
    const {displayFrameEvidence}=await import('./tools/m1010/capture-study.mjs'),{runInNewContext}=await import('node:vm');
    for(const scenario of['static','timeout','replaced']){
      let callback,timer,cancelled=0,removed=0,generation=7;
      const stream={getVideoTracks:()=>[{readyState:'live'}]};
      const receiver={style:{},videoWidth:1,videoHeight:1,play:()=>Promise.resolve(),pause(){},remove(){removed++},requestVideoFrameCallback:fn=>{callback=fn;return 1},cancelVideoFrameCallback:()=>cancelled++};
      const canvas={getContext:()=>({drawImage(){},getImageData:()=>({data:new Uint8ClampedArray([1,2,3,255])})})};
      const collect=runInNewContext('('+displayFrameEvidence.toString()+')',{
        __M1010_SOURCE__:{displayForGeneration:ticket=>{assert.equal(ticket,generation,'generation changed');return stream}},
        document:{createElement:name=>name==='video'?receiver:canvas,body:{append(){}}},
        setTimeout:fn=>{timer=fn;return 1},clearTimeout:()=>{timer=null},btoa,
        crypto:{subtle:{digest:async()=>{if(scenario==='replaced')generation++;return new ArrayBuffer(32)}}},
      });
      const pending=collect({generation:7});
      if(scenario==='timeout'){timer();await assert.rejects(()=>pending,/deadline/);}
      else {callback(1,{mediaTime:0,width:1,height:1});
        if(scenario==='replaced')await assert.rejects(()=>pending,/generation changed/);
        else {const result=await pending;assert.equal(result.frames.length,1);assert.equal(result.pixels.bytes,4);assert.equal(result.cadence,'not measured');assert.match(result.stateTransitionFreshness,/not measured/);}
      }
      assert.equal(timer,null);assert.equal(cancelled,1);assert.equal(removed,1);assert.equal(receiver.srcObject,null);assert.equal(canvas.width,0);assert.equal(canvas.height,0);
    }
  `));
  it('uses explicit diagnostic dimensions and terminal permission or prefix outcomes', () => check(`
    const {seekDiagnosticFrame,captureConsentResult,terminalizeTabSuffix,TAB_FIDELITY_CASES,stopAfterUnsafeSelf,SELF_CAPTURE_CASES}=await import('./tools/m1010/capture-study.mjs');
    const {runInNewContext}=await import('node:vm');
    for(const expected of[[640,360],[1280,720],[1920,1080]]){
      let callback,timer,cancelled=0;const listeners=new Map();
      const video={paused:true,seeking:false,readyState:2,currentTime:0,requestVideoFrameCallback:fn=>{callback=fn;return 1},cancelVideoFrameCallback:()=>cancelled++,addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name)};
      const seek=runInNewContext('('+seekDiagnosticFrame.toString()+')',{document:{querySelector:()=>video},setTimeout:fn=>{timer=fn;return 1},clearTimeout:()=>{timer=null}});
      const pending=seek({time:.5,expected});callback(0,{mediaTime:.5,width:1,height:1});assert(timer);
      callback(1,{mediaTime:.5,width:expected[0],height:expected[1]});listeners.get('seeked')();const result=await pending;
      assert.equal(result.discarded,1);assert.equal(result.metadata.width,expected[0]);assert.equal(timer,null);assert.equal(listeners.size,0);assert.equal(cancelled,1);
    }
    assert.equal(captureConsentResult([]),null);
    assert.equal(captureConsentResult([{type:'capture-permission',value:false},{type:'capture-permission',value:true}]).granted,false);
    assert.equal(captureConsentResult([{type:'permission-error',value:'denied'}]).granted,false);
    const {openCheckpoint}=await import('./tools/m1010/checkpoint.mjs'),directory='.cache/m1010/terminal-prefix-'+process.pid;
    try{const store=openCheckpoint(directory,{studyVersion:'prefix',sourceCommit:'a'.repeat(40),browserExecutableSha256:'b'.repeat(64)});
      store.begin(TAB_FIDELITY_CASES[0].id);store.complete(TAB_FIDELITY_CASES[0].id,{outcome:'UNRESOLVED',error:'native failure'});
      terminalizeTabSuffix(store,'stopped');terminalizeTabSuffix(store,'must not overwrite');
      assert.equal(store.read(TAB_FIDELITY_CASES[0].id).error,'native failure');
      for(const entry of TAB_FIDELITY_CASES.slice(1))assert.equal(store.read(entry.id).executionStatus,'NOT_RUN');
      assert.equal(stopAfterUnsafeSelf(store),false);
      store.begin('R6-cancel');store.complete('R6-cancel',{outcome:'UNSAFE',cleanup:{liveTracks:1}});
      const resumed=openCheckpoint(directory,store.snapshot().pin);
      assert.equal(stopAfterUnsafeSelf(resumed),true);assert.equal(stopAfterUnsafeSelf(resumed),true);
      for(const entry of SELF_CAPTURE_CASES.slice(1))assert.equal(resumed.read(entry.id).executionStatus,'NOT_RUN');
      assert.equal(resumed.read('R6-cancel').outcome,'UNSAFE');
    }finally{rmSync(directory,{recursive:true,force:true});}
  `));
  it('decodes distinct display identities and rejects ambiguous guards and complements', () => check(`
    const {installCaptureMarker,decodeCaptureMarker,SELF_CAPTURE_CASES,SELF_SCOPE_CASES,selfChoiceOutcome}=await import('./tools/m1010/capture-study.mjs');
    assert.deepEqual(SELF_CAPTURE_CASES.map(value=>value.id),['R6-cancel','R6-wrong','R6-current','R6-repeat']);
    assert.deepEqual(SELF_SCOPE_CASES,['baseline','movement','resize','scroll','occluder','controls-hidden','ABR-low','ABR-high']);
    assert.equal(selfChoiceOutcome('source',{identity:'source'}),'IDENTITY_RECORDED');
    for(const identity of['wrong','unresolved',undefined])assert.equal(selfChoiceOutcome('source',{identity,displaySurface:'browser'}),'UNRESOLVED');
    assert.equal(selfChoiceOutcome('wrong',{identity:'wrong'}),'IDENTITY_RECORDED');
    assert.equal(selfChoiceOutcome('rejection',{captureError:{name:'NotAllowedError'}}),'REJECTION_RECORDED');
    assert.equal(selfChoiceOutcome('rejection',{}),'UNRESOLVED');
    const runner=readFileSync('tools/m1010/capture-study.mjs','utf8');
    for(const forbidden of['permissions.request(', 'triggerAction', 'grantPermissions(', 'getDisplayMedia(', "locator('#display').click"])assert(!runner.includes(forbidden));
    const {runInNewContext}=await import('node:vm');
    const width=1000,height=500,bytes=Buffer.alloc(width*height*4,127);
    const context={fillStyle:'',fillRect(left,top,columns,rows){for(let vertical=top;vertical<top+rows;vertical++)for(let horizontal=left;horizontal<left+columns;horizontal++){const offset=(vertical*width+horizontal)*4;bytes.fill(this.fillStyle==='#ffffff'?255:0,offset,offset+3);bytes[offset+3]=255;}}};
    const canvas={style:{},getContext:()=>context};
    const install=runInNewContext('('+installCaptureMarker.toString()+')',{document:{getElementById:()=>null,createElement:()=>canvas,body:{append(){}}},innerWidth:width,innerHeight:height,devicePixelRatio:1});
    for(const identity of['00000000','ffffffff','129abcde','f7b53021']){install({identity,title:'local fixture'});assert.equal(decodeCaptureMarker(bytes,width,height,[width,height]),identity);}
    bytes[(12*width+6)*4]=120;assert.throws(()=>decodeCaptureMarker(bytes,width,height,[width,height]),/Ambiguous/);
    install({identity:'12345678',title:'fixture'});bytes.fill(0,(36*width+30)*4,(36*width+30)*4+3);assert.throws(()=>decodeCaptureMarker(bytes,width,height,[width,height]),/complement/);
    assert.throws(()=>install({identity:'not an identity',title:'fixture'}));
    assert.throws(()=>decodeCaptureMarker(Buffer.alloc(4),width,height,[width,height]));
  `));
  it('keeps compositor crop inference distinct from source-faithful acquisition', () => check(`
    const {TAB_FIDELITY_CASES,diagnosticCrop}=await import('./tools/m1010/capture-study.mjs');
    assert.deepEqual(TAB_FIDELITY_CASES.map(value=>value.id),['R3-720-native','R3-720-small','R3-720-large','R3-720-offscreen','R3-1080-small']);
    const geometry={rect:{x:20,y:30,width:640,height:360},viewport:[1512,982],captured:[3024,1964]};
    const full=diagnosticCrop(geometry);assert.deepEqual(full.crop,[40,60,1280,720]);assert.deepEqual(full.sourceFraction,[0,0,1,1]);assert(full.fullyVisible);
    const clipped=diagnosticCrop({...geometry,rect:{x:-320,y:30,width:640,height:360}});
    assert.deepEqual(clipped.crop,[0,60,640,720]);assert.deepEqual(clipped.sourceFraction,[.5,0,.5,1]);assert(!clipped.fullyVisible);
    assert.match(full.scope,/not independently detected/);
    for(const captured of[[0,1964],[NaN,1964]])assert.throws(()=>diagnosticCrop({...geometry,captured}));
    assert.throws(()=>diagnosticCrop({...geometry,rect:{x:-700,y:0,width:640,height:360}}),/outside/);
  `));
  it('measures cadence over actual elapsed time without inventing pixel identity or submissions', () => check(`
    const {summarizeCadence,observeCadence}=await import('./tools/m1010/acquisition.mjs');
    const frames=[0,20,40,160].map((at,index)=>({at,mediaTime:[0,.02,.02,0][index],presentedFrames:[1,2,4,5][index],visibility:'visible',focused:true}));
    const record={frames,start:0,end:200,durationMs:200,error:null};
    const result=summarizeCadence(record);
    assert.equal(result.outcome,'RECORDED');assert.equal(result.callbackFps,15);
    assert.equal(result.presentedCounterDelta,4);assert.equal(result.counterGaps,1);
    assert.equal(result.repeatedMediaTimes,1);assert.equal(result.mediaDiscontinuities,1);
    assert.deepEqual(result.callbackGapMs,{median:20,p95:120,maximum:120,over100ms:1});
    assert.equal(result.pixelDuplicates,'not measured');assert.equal(result.successfulSubmissions,'not measured');
    assert.equal(summarizeCadence({...record,frames:[]}).callbackFps,null);
    assert.equal(result.trailingSilenceMs,40);
    for(const change of[{error:'play denied'},{end:199},{events:[{type:'blur'}]},{frames:[{...frames[0],focused:false},...frames.slice(1)]}])assert.equal(summarizeCadence({...record,...change}).outcome,'UNRESOLVED');
    const {runInNewContext}=await import('node:vm');
    let now=0,callback,timerId=0,paused=0,cancelled=0;const timers=new Map(),listeners=new Set();
    const video={currentTime:0,play:()=>Promise.resolve(),pause:()=>paused++,requestVideoFrameCallback:fn=>{callback=fn;return 1},cancelVideoFrameCallback:()=>{cancelled++;callback=null},addEventListener:name=>listeners.add(name),removeEventListener:name=>listeners.delete(name)};
    const context={document:{querySelector:()=>video,visibilityState:'visible',hasFocus:()=>true,addEventListener:video.addEventListener,removeEventListener:video.removeEventListener},window:{addEventListener:video.addEventListener,removeEventListener:video.removeEventListener},performance:{now:()=>now,timeOrigin:1000},setTimeout:(fn,delay)=>{timers.set(++timerId,{fn,delay});return timerId},clearTimeout:id=>timers.delete(id)};
    const observe=runInNewContext('('+observeCadence.toString()+')',context);
    const pending=observe({durationMs:200});assert.equal(timers.values().next().value.delay,5000);
    callback(0,{mediaTime:0,presentedFrames:1,width:1280,height:720});assert.equal(timers.size,1);assert.equal(timers.values().next().value.delay,200);
    now=20;callback(20,{mediaTime:.02,presentedFrames:2,width:1280,height:720});now=203;timers.values().next().value.fn();
    const measured=await pending;assert.equal(measured.end,203);assert.equal(measured.frames.length,2);assert.equal(measured.error,null);
    assert.equal(paused,1);assert.equal(cancelled,1);assert.equal(listeners.size,0);assert.equal(timers.size,0);
    const denied=observe({durationMs:200});timers.values().next().value.fn();assert.match((await denied).error,/No decoded frame/);assert.equal(timers.size,0);assert.equal(listeners.size,0);
  `));
  it('fixes the R1 automatic prefix and waits for observed state, never an assumed click', () => check(`
    const {R1_CASES,R1_INPUT_CASES,INPUT_TIMES,waitForObserved,replayStage,replayOutcome,redirectOutcome}=await import('./tools/m1010/study.mjs');
    assert.deepEqual(R1_CASES.map(value=>value.id),['R1-A','R1-B','R1-C','R1-D','R1-J','R1-K']);
    assert.deepEqual(R1_INPUT_CASES.map(value=>value.id),['R1-input-A-source','R1-input-A-replay','R1-input-B-source','R1-input-B-replay']);
    assert.deepEqual(INPUT_TIMES,[1.2,2.2,3.2]);
    assert.deepEqual(await waitForObserved(async()=>({granted:true}),value=>value.granted,0),{granted:true});
    await assert.rejects(()=>waitForObserved(async()=>({granted:false}),value=>value.granted,0),/no action assumed/);
    const source=readFileSync('tools/m1010/study.mjs','utf8');
    for(const forbidden of['permissions.request(', 'triggerAction', 'grantPermissions(', 'getDisplayMedia('])assert(!source.includes(forbidden));
    assert(source.includes('verified[0]?.documentId'));assert(source.includes('verified[0]?.result.nonce'));
    for(const label of['Grant source origin','Revoke source origin'])assert(source.includes(label)&&readFileSync('tools/m1010/acquire.html','utf8').includes(label));
    assert.equal(replayStage(['R1-host-grant'],true),'AUTOMATIC_PREFIX');assert.equal(replayStage(['R1-F'],false),'RETAINED');
    const valid={exception:null,sameBytes:true,playback:{playable:true},comparisons:Array.from({length:3},()=>({outcome:'SUPPORTED'}))};
    assert.equal(replayOutcome(valid),'SUPPORTED');
    for(const change of[{exception:'cancelled'},{sameBytes:false},{playback:{playable:false}},{comparisons:[]},{comparisons:[{outcome:'UNRESOLVED'}]}])assert.equal(replayOutcome({...valid,...change}),'UNRESOLVED');
    const redirect={selected:{selectedUrl:'http://127.0.0.1:5204/redirect-same.mp4'},exception:'fetch failed',fetch:null,requestDelta:[{path:'/redirect-same.mp4'}]};
    assert.equal(redirectOutcome(redirect),'UNSUPPORTED_SAFE');
    assert.equal(redirectOutcome({...redirect,requestDelta:[]}), 'UNRESOLVED');
    assert.equal(redirectOutcome({...redirect,requestDelta:[...redirect.requestDelta,{path:'/cors/A.mp4'}]}), 'UNRESOLVED');
  `));
  it('verifies referenced raw pixels as well as checkpoint report JSON', () => check(`
    const {openCheckpoint}=await import('./tools/m1010/checkpoint.mjs');
    const {writeFileSync}=await import('node:fs');
    const directory='.cache/m1010/checkpoint-pixels-'+process.pid;
    const pin={studyVersion:'pixels',sourceCommit:'a'.repeat(40),browserExecutableSha256:'b'.repeat(64)};
    try{
      const study=openCheckpoint(directory,pin),bytes=Buffer.from([1,2,3,255]);
      writeFileSync(directory+'/input.rgba',bytes);study.begin('input');
      study.complete('input',{frames:[{pixels:{path:'input.rgba',bytes:bytes.length,sha256:hash(bytes)}}]});
      assert(openCheckpoint(directory,pin).has('input'));writeFileSync(directory+'/input.rgba',Buffer.from([1,2,4,255]));
      assert.throws(()=>openCheckpoint(directory,pin),/raw artifact changed/);
    }finally{rmSync(directory,{recursive:true,force:true});}
  `));
  it('keeps optional persistent native profiles explicit and research-scoped', () => check(`
    const browser=readFileSync('tools/m9-browser.mjs','utf8'),native=readFileSync('tools/m1010/native.mjs','utf8');
    assert(browser.includes('profileDirectory ? resolve(profileDirectory) : mkdtempSync'));
    assert.equal(browser.split('if (!profileDirectory) rmSync(profile').length-1,2);
    assert(native.includes("resolve(options.profileDirectory).startsWith(join(ROOT, '.cache/m1010/'))"));
  `));
  it('compares every diagnostic input byte independently of neural output parity', () => check(`
    const {pixelDifference}=await import('./tools/m1010/acquisition.mjs');
    const original=new Uint8Array([1,2,3,255,4,5,6,255]);
    assert.deepEqual(pixelDifference(original,original),{bytes:8,mae:0,maximum:0,changedPixels:0,exact:true});
    const changed=original.slice();changed[2]+=4;changed[7]-=8;
    assert.deepEqual(pixelDifference(original,changed),{bytes:8,mae:1.5,maximum:8,changedPixels:2,exact:false});
    assert.throws(()=>pixelDifference(original,new Uint8Array(4)));
  `));
  it('resumes immutable experiments and rejects changed source, browser or raw artifacts', () => check(`
    const {openCheckpoint}=await import('./tools/m1010/checkpoint.mjs');
    const {writeFileSync}=await import('node:fs');
    const directory='.cache/m1010/checkpoint-test-'+process.pid;
    const pin={studyVersion:'test-1',sourceCommit:'a'.repeat(40),browserExecutableSha256:'b'.repeat(64)};
    try{
      const study=openCheckpoint(directory,pin);
      study.begin('R1-A');study.manual({instruction:'Grant exact local origin',observable:'permissions.contains local origin'});
      study.candidate('R1',{state:'INCOMPLETE'});study.complete('R1-A',{outcome:'SUPPORTED'});
      const resumed=openCheckpoint(directory,pin);assert(resumed.has('R1-A'));
      assert.deepEqual(resumed.read('R1-A'),{outcome:'SUPPORTED'});assert.equal(resumed.snapshot().requiredNextManualAction,null);
      assert.deepEqual(resumed.snapshot().candidateState,{R1:{state:'INCOMPLETE'}});
      assert.throws(()=>resumed.begin('R1-A'),/immutable/);
      for(const changed of[{sourceCommit:'c'.repeat(40)},{browserExecutableSha256:'d'.repeat(64)},{studyVersion:'other'}])assert.throws(()=>openCheckpoint(directory,{...pin,...changed}),/identity changed/);
      writeFileSync(directory+'/R1-A.json','{}');assert.throws(()=>openCheckpoint(directory,pin));
      assert.throws(()=>openCheckpoint('results/not-ignored',pin),/ignored/);
    }finally{rmSync(directory,{recursive:true,force:true});}
  `));
  it('recovers a completed artifact if interrupted before checkpoint publication', () => check(`
    const {openCheckpoint}=await import('./tools/m1010/checkpoint.mjs');
    const {writeFileSync}=await import('node:fs');
    const directory='.cache/m1010/checkpoint-recovery-'+process.pid;
    const pin={studyVersion:'test-1',sourceCommit:'a'.repeat(40),browserExecutableSha256:'b'.repeat(64)};
    try{
      const study=openCheckpoint(directory,pin);study.begin('R1-A');
      writeFileSync(directory+'/R1-A.json',JSON.stringify({experimentId:'R1-A',pin,result:{outcome:'SUPPORTED'}}));
      const resumed=openCheckpoint(directory,pin);assert(resumed.has('R1-A'));assert.equal(resumed.snapshot().activeExperimentId,null);
      resumed.begin('R1-B');assert.throws(()=>resumed.begin('R1-C'),/Unfinished/);
      resumed.complete('R1-B',{outcome:'UNRESOLVED'});assert(resumed.has('R1-B'));
    }finally{rmSync(directory,{recursive:true,force:true});}
  `));
  it('activates only visible harness controls and retains reference frames and decoded timestamps', () => check(`
    const source=readFileSync('tools/m1010/native.mjs','utf8');
    assert(source.includes("locator('#stage').click()"));assert(!source.includes("locator('#source').click"));
    assert(source.includes('report.references.push'));assert(source.includes('player.seek.metadata.mediaTime === harness.seek.metadata.mediaTime'));
    assert(source.includes("predicate: worker => worker.url().endsWith('/service-worker.js')"));
    assert(source.includes("installed.manifest.name, 'AetherVSR M10.10 Research'"));
    assert(source.includes('installed.modelSha256, identity.provenance.modelSha256'));
  `));
  it('is not a production permission change or a page-owned output surface', () => check(`
    assert.deepEqual(manifest.permissions,['activeTab','scripting','storage']);
    for(const name of['host_permissions','optional_host_permissions','web_accessible_resources','offscreen','side_panel'])assert.equal(manifest[name],undefined);
    const source=readFileSync('tools/m1010/player.ts','utf8');
    for(const name of['VideoPipeline','NeuralUpscaler','RuntimeDriver'])assert(source.includes(name));
    assert(!source.includes('VideoAttachment'));assert(!source.includes('inspectGeometry'));
    const html=readFileSync('tools/m1010/player.html','utf8');
    for(const id of['play','pause','seek','volume','mute','fullscreen','close','return'])assert(html.includes('id="'+id+'"'));
  `));
  it('limits optional acquisition permissions and web-accessible resources to the local target probe', () => check(`
    const {acquisitionManifest}=await import('./tools/m1010/build.mjs');
    assert.deepEqual(acquisitionManifest.optional_permissions,['tabCapture']);
    assert.deepEqual(acquisitionManifest.optional_host_permissions,['http://127.0.0.1/*']);
    assert.deepEqual(acquisitionManifest.web_accessible_resources,[{resources:['target.html','target.js'],matches:['http://127.0.0.1/*']}]);
    const target=readFileSync('tools/m1010/target.ts','utf8'),manual=readFileSync('tools/m1010/manual.mjs','utf8');
    assert(!target.includes('chrome.runtime.sendMessage'));assert(!target.includes('fetch('));
    assert(target.includes('event.source !== window.opener'));assert(target.includes('event.origin !== sourceOrigin'));
    assert(!manual.includes('permissions.request'));assert(!manual.includes('getDisplayMedia('));
    assert(!manual.includes('triggerAction'));assert(!manual.includes('autoplay-policy'));
  `));
  it('uses authenticated source and consumer self identity without tab URL permission', () => check(`
    const {targetStreamId}=await import('./tools/m1010/native.mjs');
    const extensionId='a'.repeat(32), origin='chrome-extension://'+extensionId;
    let current={sourceTabId:7,playerDocumentId:'document',selection:{generation:1}},destination={id:9},issueCount=0;
    let afterIssue=()=>{},targetUrl=origin+'/target.html?session=fixture';
    globalThis.m1010Acquire={refresh:async()=>structuredClone(current)};
    globalThis.chrome={runtime:{},tabs:{getCurrent:async()=>({...destination})},tabCapture:{getMediaStreamId:(options,callback)=>{
      assert.deepEqual(options,{targetTabId:7,consumerTabId:9});issueCount++;afterIssue();callback('stream-id');
    }}};
    const evaluate=(callback,value)=>callback(value);
    const native={extensionId,worker:{evaluate}},acquisition={url:()=>origin+'/acquire.html',evaluate},consumer={url:()=>targetUrl,evaluate};
    assert.equal(await targetStreamId(native,acquisition,consumer),'stream-id');assert.equal(issueCount,1);
    afterIssue=()=>{current.selection.generation++};
    await assert.rejects(()=>targetStreamId(native,acquisition,consumer));
    afterIssue=()=>{destination={id:10}};
    await assert.rejects(()=>targetStreamId(native,acquisition,consumer));destination={id:9};
    afterIssue=()=>{targetUrl=origin+'/other.html'};
    await assert.rejects(()=>targetStreamId(native,acquisition,consumer));targetUrl=origin+'/target.html?session=fixture';
    afterIssue=()=>{chrome.runtime.lastError={message:'native grant denied'}};
    await assert.rejects(()=>targetStreamId(native,acquisition,consumer),/native grant denied/);delete chrome.runtime.lastError;
    const previous=issueCount;destination={};
    await assert.rejects(()=>targetStreamId(native,acquisition,consumer));assert.equal(issueCount,previous);
  `));
  it('builds deterministically with a tracked fixture, pins every byte and refuses production output paths', () => check(`
    const path='public/media/aethervsr-testclip-720p60-h264.mp4',sha256=hash(readFileSync(path));
    const directory='.cache/m1010/package-test-'+process.pid;
    try{
      const first=await buildPlayer(directory,{path,sha256});
      const original=readFileSync(directory+'/research-provenance.json');
      await buildPlayer(directory,{path,sha256});assert.deepEqual(readFileSync(directory+'/research-provenance.json'),original);
      for(const[name,info]of Object.entries(first.provenance.files)){const bytes=readFileSync(directory+'/'+name);assert.equal(bytes.length,info.bytes);assert.equal(hash(bytes),info.sha256);}
      assert.equal(first.provenance.modelSha256,'d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a');
      assert(!readFileSync(directory+'/player.js','utf8').includes('getImageData'));
      await assert.rejects(()=>buildPlayer('dist-extension',{path,sha256}));
    }finally{rmSync(directory,{recursive:true,force:true});}
  `));
});