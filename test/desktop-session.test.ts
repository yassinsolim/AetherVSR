import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

function check(source: string) {
  const result = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import {build} from 'esbuild';
    import {runInNewContext} from 'node:vm';
    const compiled = await build({entryPoints:['apps/desktop/session.ts'], bundle:true,
      write:false, format:'iife', globalName:'DesktopBundle', plugins:[{name:'platform', setup(build) {
        build.onResolve({filter:/core\\/(gpu\\/device|pipeline|upscale\\/(baseline-scaler|neural-upscaler)|neural\\/model)\\.js$/},
          args => ({path:args.path, namespace:'platform'}));
        build.onLoad({filter:/.*/,namespace:'platform'}, () => ({contents:
          'export const {acquireGpu,watchDeviceFailures,VideoPipeline,BaselineScaler,NeuralUpscaler,NEURAL_OPTIONAL_FEATURES,loadModel}=globalThis.platform;'}));
      }}]});
    const events=[], urls=new Set(), intervals=new Set(), devices=new Set(), watchers=new Map();
    const requests=[], modelRequests=[], pipelines=[], neuralStages=[];
    let nextUrl=0, acquisitions=0, now=0;
    const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no});return {promise,resolve,reject}};
    const flush=async()=>{for(let turn=0;turn<12;turn++)await Promise.resolve()};
    class Target extends EventTarget {
      listeners=new Map();
      addEventListener(name,handler){super.addEventListener(name,handler);if(!this.listeners.has(name))this.listeners.set(name,new Set());this.listeners.get(name).add(handler)}
      removeEventListener(name,handler){super.removeEventListener(name,handler);this.listeners.get(name)?.delete(handler)}
      listenerCount(){return [...this.listeners.values()].reduce((total,handlers)=>total+handlers.size,0)}
    }
    class BaselineScaler {
      neural=false;scaleFactor=2;id='catmull-rom';destroyed=0;
      constructor(filter){assert.equal(filter,'catmull-rom')}
      destroy(){this.destroyed++}
    }
    class NeuralUpscaler extends BaselineScaler {
      neural=true;id='production';
      constructor(model){super('catmull-rom');this.model=model;neuralStages.push(this)}
    }
    class VideoPipeline {
      running=false;error=null;timingGeneration=0;framesRendered=0;destroyed=0;configured=false;sourceConfigured=false;
      onFrame=null;onConfiguration=null;onGpuSample=null;
      constructor(gpu,video,canvas,upscaler,options){Object.assign(this,{gpu,video,canvas,currentUpscaler:upscaler,options});pipelines.push(this)}
      start(){assert.equal(this.destroyed,0);this.running=true}
      stop(){this.running=false}
      invalidateTiming(){this.timingGeneration++}
      resetMeasurements(){this.framesRendered=0;this.invalidateTiming()}
      setUpscaler(upscaler){this.currentUpscaler.destroy();this.currentUpscaler=upscaler;this.framesRendered=0;this.configured=false;this.invalidateTiming()}
      stats(){return {framesRendered:this.framesRendered}}
      frame(){
        if(!this.running)return;
        now+=20;this.video.currentTime+=.02;
        this.video.quality.totalVideoFrames++;
        if(!this.configured){
          this.configured=true;this.canvas.width=this.video.videoWidth*2;this.canvas.height=this.video.videoHeight*2;
          this.configuredUpscaler=this.currentUpscaler;
          const sourceChanged=!this.sourceConfigured;this.sourceConfigured=true;
          this.onConfiguration?.({generation:this.timingGeneration,source:{width:this.video.videoWidth,height:this.video.videoHeight},
            target:{width:this.canvas.width,height:this.canvas.height},configureMs:1,sourceChanged});
        }
          assert.equal(this.currentUpscaler,this.configuredUpscaler,'encode must use the configured stage');
        const tick={now,mediaTime:this.video.currentTime,presentedDelta:1,size:{width:this.video.videoWidth,height:this.video.videoHeight},
          presentationTime:now-1,expectedDisplayTime:now+1,decodeLatencyMs:null};
        this.framesRendered++;this.onFrame?.(tick);return tick;
      }
      destroy(){this.stop();this.destroyed++;this.currentUpscaler.destroy();if(this.throwDestroy)throw Error('pipeline cleanup failed')}
    }
    const platform={VideoPipeline,BaselineScaler,NeuralUpscaler,NEURAL_OPTIONAL_FEATURES:['shader-f16'],
      acquireGpu(options){acquisitions++;const request=deferred();requests.push({...request,options});return request.promise},
      loadModel(url){const request=deferred();modelRequests.push({...request,url});return request.promise},
      watchDeviceFailures(device,callback){watchers.set(device,callback);return ()=>watchers.delete(device)}};
    const makeGpu=(timestampQuery=true)=>{
      const device={destroyed:0,destroy(){this.destroyed++;devices.delete(this)}};devices.add(device);
      return {device,capabilities:{timestampQuery,externalTexture:true,preferredCanvasFormat:'bgra8unorm',maxTextureDimension2D:8192,reportedLimits:{}},adapterReport:{fallbackAdapter:false}};
    };
    const model=Object.freeze({identity:'immutable production'});
    const document=new Target();document.visibilityState='visible';
    const context={platform,document,queueMicrotask,performance:{now:()=>now},
      setInterval:callback=>{intervals.add(callback);return callback},clearInterval:callback=>intervals.delete(callback),
      URL:{createObjectURL:file=>{const url='blob:local/'+(++nextUrl);urls.add(url);events.push(['create',file.name]);return url},
        revokeObjectURL:url=>{assert(urls.delete(url));events.push(['revoke',url])}}};
    runInNewContext(compiled.outputFiles[0].text,context);
    class Video extends Target {
      src='';paused=true;ended=false;seeking=false;currentTime=0;duration=NaN;videoWidth=0;videoHeight=0;
      muted=false;volume=1;playbackRate=1;readyState=0;error=null;
      quality={totalVideoFrames:0,droppedVideoFrames:0,corruptedVideoFrames:0};
      getVideoPlaybackQuality(){return {...this.quality}}
      pause(){this.paused=true;events.push(['pause']);this.dispatchEvent(new Event('pause'))}
      load(){events.push(['load',this.src]);this.readyState=0;this.currentTime=0;this.duration=NaN;this.error=null;this.dispatchEvent(new Event('emptied'))}
      play(){events.push(['play']);this.paused=false;this.dispatchEvent(new Event('playing'));return Promise.resolve()}
      removeAttribute(name){assert.equal(name,'src');this.src=''}
    }
    const video=new Video(),canvas={hidden:false,width:0,height:0};
    const {DesktopSession}=context.DesktopBundle;
    const metadata=()=>{video.readyState=2;video.duration=100;video.videoWidth=1280;video.videoHeight=720;video.dispatchEvent(new Event('loadedmetadata'))};
    const finishSetup=async(session,playing,timestampQuery=true)=>{
      await flush();const gpu=makeGpu(timestampQuery);requests.at(-1).resolve(gpu);await flush();
      modelRequests.at(-1).resolve(model);await playing;return gpu;
    };
    const started=async(options={})=>{
      const session=new DesktopSession(video,canvas,options);session.load({name:'clip.mp4'});metadata();
      const gpu=await finishSetup(session,session.play());return {session,gpu,pipeline:pipelines.at(-1)};
    };
    const assertReleased=()=>{assert.equal(devices.size,0);assert.equal(watchers.size,0);assert.equal(intervals.size,0);assert.equal(urls.size,0);
      assert.equal(video.listenerCount(),0);assert.equal(document.listenerCount(),0)};
    ${source}
    console.log('checked');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 60000 });
  expect(result.trim()).toBe('checked');
}

describe('DesktopSession local media ownership', () => {
  it('selects metadata without autoplay or GPU acquisition and releases replaced URLs', () => check(`
    const session=new DesktopSession(video,canvas);
    assert.equal(session.snapshot().state,'empty');assert.equal(canvas.hidden,true);
    session.load({name:'first.mp4'});const first=session.snapshot();
    assert.equal(first.name,'first.mp4');assert.equal(first.sourceGeneration,1);
    assert.equal(first.url,video.src);assert.equal(urls.size,1);
    session.load({name:'second.mp4'});const second=session.snapshot();
    assert.equal(second.sourceGeneration,2);assert.notEqual(second.url,first.url);
    assert.equal(urls.size,1);assert.equal(acquisitions,0);
    assert.deepEqual(events.map(event=>event[0]),['pause','create','load','pause','revoke','create','load']);
    session.destroy();session.destroy();assert.equal(urls.size,0);
    assert.equal(session.snapshot().state,'disposed');assert.equal(canvas.hidden,true);
  `));

  it('deduplicates play, uses the app model and real driver, and reveals only successful current frames', () => check(`
    const changes=[];let factoryCalls=0;
    const session=new DesktopSession(video,canvas,{forceCopy:true,onChange:value=>changes.push(value),
      createPipeline:(...args)=>{factoryCalls++;assert.equal(args[1],video);assert.equal(args[2],canvas);return new VideoPipeline(...args)}});
    session.load({name:'clip.mp4'});metadata();
    const first=session.play(),second=session.play();await flush();
    assert.equal(acquisitions,1);assert.equal(events.filter(event=>event[0]==='play').length,1);
    assert.equal(session.snapshot().pending,true);
    const gpu=await finishSetup(session,first);await second;
    assert.equal(session.gpu,gpu);assert.equal(factoryCalls,1);assert.equal(session.runtime.pipeline,pipelines[0]);
    assert.deepEqual([...requests[0].options.optionalFeatures],['shader-f16']);
    assert.equal(modelRequests[0].url,'aethervsr://app/models/production.json');
    assert.equal(pipelines[0].options.forceCopyImport,true);assert.equal(canvas.hidden,true);
    pipelines[0].frame();assert.equal(canvas.hidden,false);assert.equal(session.snapshot().ready,true);
    assert.equal(session.snapshot().canvas.width,2560);assert.equal(session.snapshot().canvas.height,1440);
    for(let index=0;index<40;index++)pipelines[0].frame();
    assert.equal(session.runtime.snapshot().actualTier,'neural');assert.equal(neuralStages[0].model,model);
    assert.equal(session.runtime.snapshot().controller.forced,false);assert.equal(session.snapshot().mode,'auto');
    await flush();const notifications=changes.length;for(let index=0;index<10;index++)pipelines[0].frame();await flush();assert.equal(changes.length,notifications);
    assert.equal(session.snapshot().pending,false);session.destroy();assertReleased();
  `));

  it('destroys late acquisitions without attaching them to a replacement file', () => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'old.mp4'});metadata();
    const oldPlay=session.play();await flush();session.load({name:'new.mp4'});metadata();
    const newPlay=session.play();await flush();const current=await finishSetup(session,newPlay);
    const late=makeGpu();requests[0].resolve(late);await oldPlay;
    assert.equal(late.device.destroyed,1);assert.equal(session.gpu,current);assert.equal(pipelines.length,1);
    assert.equal(session.snapshot().name,'new.mp4');session.destroy();assertReleased();
  `));

  it.each(['load', 'destroy'])('releases an allocated device while its model is pending on %s', action => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'old.mp4'});metadata();
    const playing=session.play();await flush();const gpu=makeGpu();requests[0].resolve(gpu);await flush();
    assert.equal(session.snapshot().resources.devices,1);assert.equal(session.snapshot().pending,true);
    ${action === 'load' ? "session.load({name:'new.mp4'});" : 'session.destroy();'}
    assert.equal(gpu.device.destroyed,1);assert.equal(watchers.size,0);assert.equal(canvas.hidden,true);
    modelRequests[0].resolve(model);await playing;assert.equal(pipelines.length,0);
    session.destroy();assert.equal(gpu.device.destroyed,1);assertReleased();
  `));

  it('releases a device acquired after destruction', () => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'clip.mp4'});
    const playing=session.play();await flush();session.destroy();const late=makeGpu();requests[0].resolve(late);await playing;
    assert.equal(late.device.destroyed,1);assert.equal(pipelines.length,0);assert.equal(modelRequests.length,0);
    assert.equal(session.snapshot().pending,false);assertReleased();
  `));

  it('ignores an obsolete native play rejection without stopping the new source', () => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'old.mp4'});metadata();
    const native=deferred();video.play=()=>native.promise;
    const oldPlay=session.play();await flush();session.load({name:'new.mp4'});metadata();video.play=Video.prototype.play;
    const gpu=await finishSetup(session,session.play());pipelines[0].frame();
    native.reject(Error('old play rejected'));const late=makeGpu();requests[0].resolve(late);await oldPlay;
    assert.equal(session.gpu,gpu);assert.equal(video.paused,false);assert.equal(canvas.hidden,false);
    assert.equal(session.snapshot().error,null);session.destroy();assertReleased();
  `));

  it('keeps pause effective during setup and resumes the same shared runtime with fresh warm-up', () => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'clip.mp4'});metadata();
    const playing=session.play();session.pause();await finishSetup(session,playing);
    assert.equal(video.paused,true);assert.equal(pipelines[0].running,false);assert.equal(canvas.hidden,true);
    await session.play();const driver=session.runtime;for(let index=0;index<42;index++)pipelines[0].frame();
    const oldFrame=driver.onFrame,oldTick=pipelines[0].frame();
    const rendered=pipelines[0].framesRendered;session.pause();pipelines[0].frame();
    assert.equal(pipelines[0].framesRendered,rendered);assert.equal(canvas.hidden,false);
    await session.play();assert.equal(session.runtime,driver);assert.equal(acquisitions,1);
    assert.equal(canvas.hidden,true);oldFrame(oldTick);assert.equal(canvas.hidden,true);
    pipelines[0].frame();assert.equal(canvas.hidden,false);
    assert.equal(driver.snapshot().controller.state,'warmup');assert.equal(driver.snapshot().controller.forced,false);
    session.destroy();assertReleased();
  `));

  it('routes finite clamped controls to the same native element and invalidates seek and mode readiness', () => check(`
    const {session,pipeline}=await started();pipeline.frame();const oldFrame=session.runtime.onFrame;
    const before=pipeline.frame();session.seek(200);assert.equal(video.currentTime,100);assert.equal(canvas.hidden,true);
    oldFrame(before);assert.equal(canvas.hidden,true);pipeline.frame();assert.equal(canvas.hidden,false);
    session.seek(-1);assert.equal(video.currentTime,0);
    session.setVolume(2);assert.equal(video.volume,1);session.setVolume(-1);assert.equal(video.volume,0);
    session.setMuted(true);assert.equal(video.muted,true);session.setRate(20);assert.equal(video.playbackRate,4);
    session.setRate(.01);assert.equal(video.playbackRate,.25);
    for(const invalid of [NaN,Infinity,-Infinity])for(const method of ['seek','setVolume','setRate'])assert.throws(()=>session[method](invalid));
    session.setMode('baseline');assert.equal(canvas.hidden,true);assert.equal(session.runtime.snapshot().controller.mode,'baseline');
    session.setMode('neural');assert.equal(session.runtime.snapshot().controller.mode,'neural');assert.equal(session.runtime.snapshot().controller.forced,false);
    assert.throws(()=>session.setMode('forced'));video.duration=NaN;assert.throws(()=>session.seek(1));
    session.destroy();assertReleased();assert.throws(()=>session.load({name:'late.mp4'}));await assert.rejects(session.play());
  `));

  it.each(['pending', 'attached'])('fails closed on device faults while %s without pausing native audio or retrying', phase => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'clip.mp4'});metadata();
    const playing=session.play();await flush();const gpu=makeGpu();requests[0].resolve(gpu);await flush();
    ${phase === 'attached' ? 'modelRequests[0].resolve(model);await playing;pipelines[0].frame();' : ''}
    const fault=watchers.get(gpu.device),pauses=events.filter(event=>event[0]==='pause').length;
    fault('device lost');assert.equal(canvas.hidden,true);assert.equal(video.paused,false);assert.equal(urls.size,1);
    assert.equal(events.filter(event=>event[0]==='pause').length,pauses);assert.equal(gpu.device.destroyed,1);
    assert.equal(session.snapshot().state,'unavailable');assert.match(session.snapshot().error,/device lost/);
    modelRequests[0].resolve(model);await playing;await session.play();session.setMode('neural');assert.equal(acquisitions,1);
    session.load({name:'fresh.mp4'});metadata();await finishSetup(session,session.play());
    fault('obsolete fault');assert.equal(session.snapshot().error,null);assert.equal(acquisitions,2);
    session.destroy();assertReleased();
  `));

  it.each(['acquire', 'model'])('keeps the original playable after a current %s failure', phase => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'clip.mp4'});metadata();const playing=session.play();await flush();
    ${phase === 'acquire' ? "requests[0].reject(Error('acquire failed'));" : "requests[0].resolve(makeGpu());await flush();modelRequests[0].reject(Error('model failed'));"}
    await playing;assert.equal(video.paused,false);assert.equal(canvas.hidden,true);assert.equal(devices.size,0);
    assert.equal(session.snapshot().state,'unavailable');assert.match(session.snapshot().error,/failed/);
    session.pause();await session.play();assert.equal(acquisitions,1);assert.equal(urls.size,1);session.destroy();assertReleased();
  `));

  it('detects runtime frame failures and rejects stale readiness callbacks after file changes', () => check(`
    const {session,pipeline}=await started();const driver=session.runtime;pipeline.frame();const frame=driver.onFrame,config=driver.onConfigure;
    pipeline.error=Error('encode failed');for(const timer of [...intervals])timer();
    assert.equal(session.snapshot().state,'unavailable');assert.equal(canvas.hidden,true);assert.equal(video.paused,false);
    session.load({name:'new.mp4'});frame({now,mediaTime:0});config({});assert.equal(canvas.hidden,true);assert.equal(session.snapshot().error,null);
    session.destroy();assertReleased();
  `));

  it('hides synchronously for native seek and emptied events and releases enhancement on media errors', () => check(`
    const changes=[];const {session,pipeline}=await started({onChange:value=>changes.push(value)});pipeline.frame();
    video.seeking=true;video.dispatchEvent(new Event('seeking'));assert.equal(canvas.hidden,true);assert.equal(changes.at(-1).canvas.visible,false);
    pipeline.frame();assert.equal(canvas.hidden,true);video.seeking=false;video.dispatchEvent(new Event('seeked'));pipeline.frame();assert.equal(canvas.hidden,false);
    video.dispatchEvent(new Event('emptied'));assert.equal(canvas.hidden,true);
    video.error={code:3,message:'decode failure'};video.dispatchEvent(new Event('error'));
    assert.equal(session.snapshot().state,'error');assert.equal(session.runtime,null);assert.equal(devices.size,0);assert.equal(urls.size,1);
    session.destroy();assertReleased();
  `));

  it('cleans all remaining resources despite throwing observers and pipeline teardown', () => check(`
    const {session,pipeline}=await started({onChange:()=>{throw Error('observer failed')}});
    pipeline.throwDestroy=true;session.destroy();session.destroy();assertReleased();
    assert.equal(pipeline.destroyed,1);assert(session.snapshot().cleanupErrors.some(message=>message.includes('pipeline cleanup failed')));
  `));

  it('shares the actual play promise with reentrant observers and releases rejected-play resources', () => check(`
    let second;const native=deferred();video.play=()=>native.promise;
    const session=new DesktopSession(video,canvas,{onChange:value=>{if(value.pending&&!second)second=session.play()}});
    session.load({name:'clip.mp4'});metadata();const first=session.play();assert.equal(first,second);
    const rejected=Promise.all([assert.rejects(first,/denied/),assert.rejects(second,/denied/)]);
    native.reject(Error('denied'));await rejected;
    const late=makeGpu();requests[0].resolve(late);await flush();assert.equal(late.device.destroyed,1);
    assert.equal(session.snapshot().state,'error');assert.equal(session.snapshot().pending,false);assert.equal(canvas.hidden,true);
    session.destroy();assertReleased();
  `));

  it('does not seek a replacement source loaded by the readiness observer', () => check(`
    let replace=false,owner;
    const {session,pipeline}=await started({onChange:value=>{if(replace&&!value.ready){replace=false;owner.load({name:'replacement.mp4'})}}});
    owner=session;pipeline.frame();replace=true;session.seek(20);
    assert.equal(session.snapshot().name,'replacement.mp4');assert.equal(video.currentTime,0);assert.equal(canvas.hidden,true);
    assert.equal(pipeline.destroyed,1);assert.equal(session.runtime,null);session.destroy();assertReleased();
  `));

  it('delivers configuration notifications outside the encode task', () => check(`
    let owner,switched=false;
    const {session,pipeline}=await started({onChange:()=>{
      if(owner?.runtime?.pipeline.currentUpscaler.neural&&owner.runtime.pipeline.configured){switched=true;owner.setMode('baseline')}
    }});owner=session;
    for(let index=0;index<42;index++)pipeline.frame();await flush();
    assert.equal(switched,true);pipeline.frame();assert.equal(session.snapshot().mode,'baseline');
    session.destroy();assertReleased();
  `));

  it('reveals the first successful post-seek frame even after diagnostic measurement resets', () => check(`
    const {session,pipeline}=await started();session.setMode('baseline');for(let index=0;index<6;index++)pipeline.frame();
    session.seek(2);session.runtime.resetMeasurements();assert.equal(canvas.hidden,true);
    pipeline.frame();assert.equal(canvas.hidden,false);session.destroy();assertReleased();
  `));

  it('ignores an old model rejection after a replacement runtime is attached', () => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'old.mp4'});metadata();
    const old=session.play();await flush();const oldGpu=makeGpu();requests[0].resolve(oldGpu);await flush();
    session.load({name:'new.mp4'});metadata();const gpu=await finishSetup(session,session.play());
    modelRequests[0].reject(Error('obsolete model'));await old;
    assert.equal(session.gpu,gpu);assert.equal(oldGpu.device.destroyed,1);assert.equal(session.snapshot().error,null);
    session.destroy();assertReleased();
  `));

  it('releases GPU resources when a diagnostic pipeline factory fails', () => check(`
    const {session,gpu}=await started({createPipeline:()=>{throw Error('pipeline creation failed')}});
    assert.equal(session.snapshot().state,'unavailable');assert.equal(gpu.device.destroyed,1);assert.equal(video.paused,false);
    assert.equal(session.runtime,null);assert.equal(canvas.hidden,true);session.destroy();assertReleased();
  `));

  it('uses the unchanged baseline fallback when timestamps are unavailable', () => check(`
    const session=new DesktopSession(video,canvas);session.load({name:'clip.mp4'});metadata();session.setMode('neural');
    await finishSetup(session,session.play(),false);for(let index=0;index<100;index++)pipelines[0].frame();
    assert.equal(session.snapshot().capabilities.timestampQuery,false);assert.equal(session.runtime.snapshot().actualTier,'baseline');
    assert.equal(session.runtime.snapshot().controller.medianMs,null);assert.equal(neuralStages.length,0);assert.equal(canvas.hidden,false);
    session.destroy();assertReleased();
  `));
});

describe('M11 journey preflight without native execution', () => {
  it('rejects incomplete, failed, dirty and mismatched native parity evidence', () => {
    const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
      import assert from 'node:assert/strict';
      import { verifyJourneyParity } from './tools/m11/journeys.mjs';
      import { PARITY_CASES, MODEL_SHA256 } from './tools/m11/parity.mjs';
      const commit = 'a'.repeat(40);
      const source = { commit };
      const provenance = { diagnostic: true, sourceDirty: false, sourceCommit: commit };
      const valid = { schema: 'aethervsr.m11.paused-parity/1', verdict: 'PASS', parityPrerequisitePassed: true,
        errors: [], sourceBefore: source, sourceAfter: source, packageBefore: provenance, packageAfter: provenance,
        expected: { modelSha256: MODEL_SHA256 }, electronEnvironment: { versions: { electron: '44.4.1' } },
        cleanup: { chrome: true, electron: true, server: true },
        cases: PARITY_CASES.map(spec => ({ ...spec, verdict: 'PASS', comparison: { verdict: 'PASS', checks: { exact: true } } })) };
      assert.equal(verifyJourneyParity(valid, commit), true);
      for (const change of [
        value => { value.schema = 'smoke'; }, value => { value.verdict = 'FAIL'; },
        value => { value.parityPrerequisitePassed = false; }, value => { value.errors.push('native failure'); },
        value => { value.cases.pop(); }, value => { value.cases[0].verdict = 'NOT_RUN'; },
        value => { value.cases[0].comparison.checks.exact = false; },
        value => { value.cases[0].comparison.checks = {}; },
        value => { value.sourceAfter = { commit: 'b'.repeat(40) }; },
        value => { value.packageBefore.sourceDirty = true; }, value => { value.packageBefore.diagnostic = false; },
        value => { value.expected.modelSha256 = '0'.repeat(64); },
        value => { value.electronEnvironment.versions.electron = '44.4.2'; },
        value => { value.cleanup.electron = false; },
      ]) { const invalid = structuredClone(valid); change(invalid); assert.throws(() => verifyJourneyParity(invalid, commit)); }
      assert.throws(() => verifyJourneyParity(valid, 'b'.repeat(40)));
      assert.throws(() => verifyJourneyParity(undefined, commit));
      console.log('PASS');
    `], { cwd: process.cwd(), encoding: 'utf8', timeout: 10000 });
    expect(output.trim()).toBe('PASS');
  });

  it('confines output and rejects build collisions without launching the runner', () => {
    const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
      import assert from 'node:assert/strict';
      import { resolve } from 'node:path';
      import { journeyOutput, runJourneys } from './tools/m11/journeys.mjs';
      assert.equal(typeof runJourneys, 'function');
      assert.equal(journeyOutput(), resolve('.cache/m11/journeys-01'));
      for (const path of ['', '.cache/m11', '.cache/m11/../outside', '.cache/m11/journey-app',
        '.cache/m11/journey-app/nested', 'dist-desktop']) assert.throws(() => journeyOutput(path));
      assert.equal(journeyOutput('.cache/m11/journeys-test'), resolve('.cache/m11/journeys-test'));
      console.log('PASS');
    `], { cwd: process.cwd(), encoding: 'utf8', timeout: 10000 });
    expect(output.trim()).toBe('PASS');
  });
});