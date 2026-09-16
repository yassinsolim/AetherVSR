import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

function check(source: string) {
  const result = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import {readFileSync,mkdirSync,mkdtempSync,rmSync} from 'node:fs';
    import {execFileSync} from 'node:child_process';
    import {runInNewContext} from 'node:vm';
    import {parseRange,mediaRecipe,splitFragments,serveFixtures} from './tools/m1010/fixtures.mjs';
    ${source}
    console.log('ok');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 30000 });
  expect(result.trim()).toBe('ok');
}

describe('M10.10 local sources', () => {
  it('does not treat timestamp estimates or missing live identities as bounded A/V proof', () => check(`
    const {summarizeTimingFloor,FLOOR_MARKERS,timingPixelIdentity}=await import('./tools/m1010/timing.mjs');
    assert.deepEqual(FLOOR_MARKERS,[6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64]);
    const empty=summarizeTimingFloor({samples:[],audio:[],anchors:[],frames:[],events:[],start:null,end:0,fps:60});
    assert.equal(empty.outcome,'UNRESOLVED');assert.equal(empty.absoluteInstrumentErrorBoundMs,null);assert.equal(empty.durationMs,null);
    assert.throws(()=>timingPixelIdentity(Buffer.alloc(4)));
    const record={samples:[],audio:[],anchors:[],frames:[{visible:true,focused:true}],events:[],start:0,end:65000,fps:60};
    for(const id of FLOOR_MARKERS){
      for(const relative of[-1,0]){const identity={frame:id*60+relative,flash:relative===0};record.samples.push({marker:id,mediaTime:identity.frame/60,source:{identity},preview:{identity},expectedDisplayTime:id*1000+relative*1000/60});}
      record.audio.push({id,hz:600+20*id,firstSample:id*48000,sampleRate:48000});
      record.anchors.push({contextTime:id-.01,performanceTime:id*1000-10,before:0,after:.1},{contextTime:id+.01,performanceTime:id*1000+10,before:0,after:.1});
    }
    assert.equal(summarizeTimingFloor(record).outcome,'RECORDED_NOT_QUALIFIED');assert.equal(summarizeTimingFloor(record).absoluteInstrumentErrorBoundMs,null);
    record.samples[0].preview={identity:{frame:99,flash:false}};assert.equal(summarizeTimingFloor(record).identityMismatch,1);assert.equal(summarizeTimingFloor(record).outcome,'UNRESOLVED');
  `));
  it('calibrates sample-addressed audio detection without relying on worklet message arrival', () => check(`
    let Detector;const pulses=[],context={sampleRate:48000,currentFrame:0,registerProcessor:(name,value)=>{Detector=value},AudioWorkletProcessor:class{port={postMessage:value=>pulses.push(value)}}};
    runInNewContext(readFileSync('tools/m1010/timing-worklet.js','utf8'),context);
    const detector=new Detector({processorOptions:{generation:7}});
    for(let start=0;start<6000;start+=64){
      context.currentFrame=start;const input=Float32Array.from({length:64},(_,offset)=>start+offset<4800?.35*Math.sin(2*Math.PI*600*(start+offset)/48000):0),output=new Float32Array(64);
      detector.process([[input]],[[output]]);assert.deepEqual(output,input);
    }
    assert.equal(pulses.length,1);assert.equal(pulses[0].generation,7);assert.equal(pulses[0].firstSample,4);assert.equal(pulses[0].id,0);assert(Math.abs(pulses[0].hz-600)<.01);
    const silence=new Float32Array(256).fill(1);context.currentFrame=6000;detector.process([[]],[[silence]]);assert(silence.every(value=>value===0));
  `));
  it('identifies timing counters with complementary cells and rejects ambiguity', () => check(`
    const {decodeTimingCounter,timingRecipe,decodeTimingAudio}=await import('./tools/m1010/fixtures.mjs');
    const top=Buffer.alloc(360),bottom=Buffer.alloc(360);top[8]=bottom[8]=235;
    for(const value of[0,1,2048,4199,8191]){
      for(let bit=0;bit<13;bit++){top[(bit+2)*24+8]=(value&2**bit)?235:16;bottom[(bit+2)*24+8]=(value&2**bit)?16:235;}
      assert.equal(decodeTimingCounter(top,bottom),value);
    }
    top[56]=120;assert.throws(()=>decodeTimingCounter(top,bottom),/complement/);
    const recipe=timingRecipe(60,'out.mp4');assert(recipe.includes('70'));assert(recipe.some(value=>value.includes('bitand(n,4096)')));
    assert.throws(()=>timingRecipe(24,'out.mp4'));
    const pcm=Buffer.alloc(48000*2*4);
    for(let index=0;index<96000;index++)pcm.writeFloatLE(index%48000<4800?.35*Math.sin(2*Math.PI*(600+20*Math.floor(index/48000))*index/48000):0,index*4);
    const audio=decodeTimingAudio(pcm);assert.deepEqual(audio.pulses.map(pulse=>pulse.id),[0,1]);assert(audio.pulses.every(pulse=>pulse.frequencyErrorHz<.1));
    assert.equal(decodeTimingAudio(Buffer.alloc(48000*4)).pulses.length,0);
  `));
  it('ranges and recipes', () => check(`
    assert.deepEqual(parseRange(undefined,10),{start:0,end:9,partial:false});
    assert.deepEqual(parseRange('bytes=0-99',10),{start:0,end:9,partial:true});
    for(const header of [null,{},'bytes=-1','bytes=10-','bytes=3-2','bytes=0-1,3-4','bytes=0-9007199254740992','bytes=0-1 '])
      assert.equal(parseRange(header,10),null);
    const args=mediaRecipe({width:96,height:54,fps:30},'out.mp4');
    for(const flag of ['-nostdin','-n','5.2','4'])assert(args.includes(flag));assert(!args.includes('-y'));
  `));

  it('HTTP scopes and cleanup', () => check(`
    const path='public/media/aethervsr-testclip-720p60-h264.mp4',data=readFileSync(path);
    const entry={path,bytes:data.length};
    const fixture=await serveFixtures({A:entry,B:entry,C:entry},{ports:[0,0]});
    const [base,other]=fixture.origins,get=(path,options)=>fetch(other+path,options);
    try{
      const part=await get('/same/A.mp4',{headers:{Range:'bytes=2-9'}});
      assert.equal(part.status,206);assert.deepEqual(Buffer.from(await part.arrayBuffer()),data.subarray(2,10));
      for(const [mode,cors]of [['cors','*'],['nocors',null]])assert.equal((await get('/'+mode+'/A.mp4',{method:'HEAD'})).headers.get('access-control-allow-origin'),cors);
      for(const [path,status,options]of [['/unknown',404],['/auth/omit/A.mp4',401],['/seed',403,{headers:{Origin:'https://evil.test'}}],['/same/A.mp4',416,{headers:{Range:'bytes=-1'}}],['/same/A.mp4',405,{method:'POST'}]])assert.equal((await get(path,options)).status,status);
      const auth=await get('/auth/include/A.mp4',{method:'HEAD',headers:{Origin:base,Cookie:'m1010_fixture=allow'}});
      assert.equal(auth.status,200);assert.equal(auth.headers.get('access-control-allow-origin'),base);
      assert((await get('/seed')).headers.get('set-cookie'));
      assert.equal((await get('/redirect.mp4',{redirect:'manual'})).status,302);
      const sameRedirect=await get('/redirect-same.mp4',{redirect:'manual'});
      assert.equal(sameRedirect.status,302);assert.equal(sameRedirect.headers.get('location'),other+'/cors/A.mp4');
      const ungrantedRedirect=await get('/redirect-ungranted.mp4',{redirect:'manual'});
      assert.equal(ungrantedRedirect.status,302);assert.equal(new URL(ungrantedRedirect.headers.get('location')).hostname,'localhost');
      assert.equal((await get('/mime-bad.mp4')).headers.get('content-type'),'text/html');
      assert.equal((await get('/oversize.mp4',{method:'HEAD'})).headers.get('content-length'),'67108865');
      const log=await (await get('/requests.json')).json();assert(log.counters.blocked>=5);
      assert(!JSON.stringify(log).includes('m1010_fixture=allow'));
    }finally{await fixture.close();}
    for(const origin of fixture.origins)await assert.rejects(()=>fetch(origin+'/source.html'));
  `));

  it('manual capture lifecycle', () => check(`
    const nodes={},context={AbortController,DOMException,URL,cancelAnimationFrame(){},window:{addEventListener(){}},fetch:()=>new Promise(()=>{})};
    context.document={getElementById:id=>nodes[id]??=({checked:true,paused:true,addEventListener(){},pause(){},load(){},removeAttribute(){}})};
    let deliver,stops=0;const activation={isActive:false};
    context.navigator={userActivation:activation,mediaDevices:{getDisplayMedia:()=>new Promise(resolve=>deliver=resolve)}};
    runInNewContext(readFileSync('tools/m1010/source.js','utf8'),context);const api=context.__M1010_SOURCE__;
    assert(api.video.paused&&api.video.muted);assert.throws(()=>api.captureDisplay());
    for(const command of [{type:'eval'},{type:'play',script:'x'}])await assert.rejects(()=>api.action(command));
    activation.isActive=true;const pending=api.captureDisplay();await api.action({type:'cancel'});
    deliver({getTracks:()=>[{stop(){stops++}}]});await assert.rejects(()=>pending);assert.equal(stops,1);
    let finishCrop,startedCrop;const cropStarted=new Promise(resolve=>startedCrop=resolve);
    class BrowserCaptureMediaStreamTrack {
      kind='video';readyState='live';muted=false;
      getSettings(){return {displaySurface:'browser'}}
      stop(){this.readyState='ended';stops++}
      addEventListener(){}
      cropTo(){startedCrop();return new Promise(resolve=>finishCrop=resolve)}
      restrictTo(){return Promise.reject(new DOMException('Native restriction rejected','NotAllowedError'))}
    }
    context.CropTarget={fromElement:async()=>({})};context.RestrictionTarget={fromElement:async()=>({})};
    const track=new BrowserCaptureMediaStreamTrack(),audio=new BrowserCaptureMediaStreamTrack();audio.kind='audio';
    const captured=api.captureDisplay();deliver({getTracks:()=>[track,audio],getVideoTracks:()=>[track]});await captured;
    const captureGeneration=api.snapshot().captureGeneration;
    assert.equal(api.displayForGeneration(captureGeneration).getVideoTracks()[0],track);
    assert.throws(()=>api.displayForGeneration(captureGeneration-1),/generation changed/);
    assert.equal(api.snapshot().displayTracks[0].constructor,'BrowserCaptureMediaStreamTrack');
    assert.equal(api.snapshot().displayTracks[0].cropTo,'function');
    await assert.rejects(()=>api.restrict(),/Native restriction rejected/);
    assert.equal(api.snapshot().targetAttempts.at(-1).error.name,'NotAllowedError');
    const pendingCrop=api.crop();await cropStarted;await api.action({type:'cancel'});finishCrop();
    await assert.rejects(()=>pendingCrop,/Capture changed during/);
    const state=api.snapshot();assert.equal(state.targetAttempts.at(-1).outcome,'REJECTED');
    assert.throws(()=>api.displayForGeneration(captureGeneration),/generation changed/);
    assert.equal(state.stoppedTracks.length,2);assert(state.stoppedTracks.every(value=>value.after.readyState==='ended'));
  `));

  it.skipIf(process.env.M1010_FFMPEG !== '1')('tiny mux PTS', () => check(`
    mkdirSync('.cache/m1010/media',{recursive:true});const dir=mkdtempSync('.cache/m1010/media/unit-');
    try{for(const fragmented of [false,true]){
      const path=dir+'/'+fragmented+'.mp4';execFileSync('ffmpeg',mediaRecipe({width:96,height:54,fps:30,fragmented},path,0.2));
      if(fragmented)assert.equal(splitFragments(readFileSync(path)).fragments.length,1);
      const probe=JSON.parse(execFileSync('ffprobe',['-v','error','-show_frames','-of','json',path]));
      const frames=probe.frames.filter(frame=>frame.media_type==='video');assert.equal(frames.length,6);
      frames.forEach((frame,index)=>assert(Math.abs(Number(frame.pts_time)-index/30)<0.000002));
    }}finally{rmSync(dir,{recursive:true,force:true});}
  `));
  it.skipIf(process.env.M1010_FFMPEG !== '1')('encoded timing counter and audio identities', () => check(`
    const {timingRecipe,inspectTimingMedia}=await import('./tools/m1010/fixtures.mjs');
    mkdirSync('.cache/m1010/media',{recursive:true});const dir=mkdtempSync('.cache/m1010/media/timing-unit-');
    try{
      const path=dir+'/timing.mp4';execFileSync('ffmpeg',timingRecipe(30,path,2));
      const inspection=inspectTimingMedia(path,30,2);assert.equal(inspection.frames.length,60);
      assert.deepEqual(inspection.audio.pulses.map(pulse=>pulse.id),[0,1]);
    }finally{rmSync(dir,{recursive:true,force:true});}
  `));
});