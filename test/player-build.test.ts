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