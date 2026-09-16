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
  it('fixes the R1 automatic prefix and waits for observed state, never an assumed click', () => check(`
    const {R1_CASES,INPUT_TIMES,waitForObserved,replayStage,replayOutcome,redirectOutcome}=await import('./tools/m1010/study.mjs');
    assert.deepEqual(R1_CASES.map(value=>value.id),['R1-A','R1-B','R1-C','R1-D','R1-J','R1-K']);
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