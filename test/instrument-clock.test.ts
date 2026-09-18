import { describe, it } from 'vitest';
import { execFileSync } from 'node:child_process';

const check = (body: string, host = false, control = false) => execFileSync(process.execPath, ['--input-type=module', '-e', `
  import assert from 'node:assert/strict';
  import {build} from 'esbuild';
  import {createContext,Script} from 'node:vm';
  const compiled=await build({entryPoints:['tools/m1010r/render-recorder.ts'],bundle:true,write:false,format:'iife'});
  function fixture(targetEndFrame,{initialFrame=0,rate=48000,holdMessages=false,sample=()=>.125}={}){
    let Recorder,hostMilliseconds=0;const messages=[],delivered=[],pending=[];
    const noHostClock=()=>assert.fail('Host clock cannot complete recording');
    const sandbox=createContext({sampleRate:rate,currentFrame:initialFrame,Float32Array,Uint32Array,
      AudioWorkletProcessor:class{constructor(){this.port={onmessage:null,postMessage(message,transfer=[]){
        const cloned=structuredClone(message,{transfer});messages.push(cloned);
        (holdMessages?pending:delivered).push(cloned);
      }};}},
      registerProcessor(name,type){assert.equal(name,'m1010ri-render');Recorder=type;},
      getHostClock:noHostClock,performance:{now:noHostClock},Date:class{constructor(){noHostClock();}static now(){noHostClock();}},
      setTimeout(){assert.fail('No worklet wall timer');},setInterval(){assert.fail('No worklet wall interval');}});
    new Script(compiled.outputFiles[0].text).runInContext(sandbox);
    const recorder=new Recorder({processorOptions:{targetEndFrame}});
    function process(start,inputs,outputs){sandbox.currentFrame=start;
      assert.equal(recorder.process(inputs,outputs),true);}
    return {messages,delivered,process,send:data=>recorder.port.onmessage({data}),
      advanceHost(milliseconds){hostMilliseconds+=milliseconds;return hostMilliseconds;},
      resume(){holdMessages=false;delivered.push(...pending.splice(0));},
      step(start,length=128){const samples=Float32Array.from({length},(_,offset)=>sample(start+offset)),output=new Float32Array(length);
        process(start,[[samples]],[[output]]);assert.deepEqual(output,samples);}};
  }
  function terminal(recorder,reason){
    const recordings=recorder.messages.filter(row=>row.type==='render-recording');
    assert.equal(recordings.length,1);const result=recordings[0];
    assert.equal(result.completionReason,reason);assert.equal(recorder.messages.at(-1),result);
    assert.equal(result.sampleRate,48000);assert.notEqual(result.completionReason,'COMPLETE');
    return result;
  }
  ${host ? `
  const hostCompiled=await build({entryPoints:['tools/m1010r/render-host.ts'],bundle:true,write:false,format:'iife',globalName:'renderHost'});
  function hostFixture({target=70592,watchdogMs=15000,state='running',rate=48000,sample=frame=>(frame%1024-512)/1024}={}){
    let hostMs=0,contextSeconds=0,nextTimer=0,recordingNode,deliveredCount=0,settlements=0,outcome;
    const timers=new Map(),stateListeners=new Set(),commands=[],reads=[];
    const context={state,sampleRate:rate,destination:{},
      get currentTime(){reads.push('context');return contextSeconds;},
      addEventListener(type,listener){assert.equal(type,'statechange');stateListeners.add(listener);},
      removeEventListener(type,listener){assert.equal(type,'statechange');stateListeners.delete(listener);}};
    class MockNode{
      constructor(receivedContext,name,options){
        assert.equal(receivedContext,context);assert.equal(name,'m1010ri-render');this.options=options;
        this.recorder=fixture(options.processorOptions.targetEndFrame,{holdMessages:true,sample});
        this.onprocessorerror=null;this.closed=0;this.disconnected=0;this.connected=0;this.dropCommands=false;
        this.port={onmessage:null,onmessageerror:null,
          postMessage:data=>{commands.push(structuredClone(data));if(!this.dropCommands)this.recorder.send(data);},
          close:()=>{this.closed++;}};
        recordingNode=this;
      }
      connect(destination){assert.equal(destination,context.destination);assert(this.port.onmessage);
        assert(this.port.onmessageerror);assert(this.onprocessorerror);assert.equal(stateListeners.size,1);this.connected++;}
      disconnect(){this.disconnected++;}
    }
    const sandbox=createContext({AudioWorkletNode:MockNode,ArrayBuffer,Float32Array,Uint32Array,
      performance:{now(){reads.push('host');return hostMs;}},
      setTimeout(callback,delay){const id=++nextTimer;timers.set(id,{callback,due:hostMs+delay});return id;},
      clearTimeout(id){timers.delete(id);}});
    new Script(hostCompiled.outputFiles[0].text).runInContext(sandbox);
    const begin=()=>sandbox.renderHost.beginRenderRecording(context,target,watchdogMs);
    const start=()=>{const handle=begin();handle.finished.then(value=>{settlements++;outcome=value;});return handle;};
    const receive=data=>recordingNode.port.onmessage?.({data});
    function serviceTimers(){for(;;){const next=[...timers].filter(([,timer])=>timer.due<=hostMs)
      .sort((left,right)=>left[1].due-right[1].due||left[0]-right[0])[0];
      if(!next)break;timers.delete(next[0]);next[1].callback();}}
    return {context,timers,stateListeners,commands,reads,start,receive,
      get node(){return recordingNode;},get outcome(){return outcome;},get settlements(){return settlements;},
      advanceHost(milliseconds,service=true){hostMs+=milliseconds;if(service)serviceTimers();},serviceTimers,
      setContextTime(seconds){contextSeconds=seconds;},
      render(start,length=128){recordingNode.recorder.step(start,length);},
      serviceMessages(){recordingNode.recorder.resume();const rows=recordingNode.recorder.delivered;
        while(deliveredCount<rows.length)receive(rows[deliveredCount++]);},
      state(value){context.state=value;for(const listener of [...stateListeners])listener();},
      processorError(message='processor failed'){recordingNode.onprocessorerror?.({message,filename:'render-recorder.ts',lineno:12,colno:3});},
      clean(){assert.equal(timers.size,0);assert.equal(stateListeners.size,0);
        assert.equal(recordingNode.port.onmessage,null);assert.equal(recordingNode.port.onmessageerror,null);
        assert.equal(recordingNode.onprocessorerror,null);}
    };
  }
  ` : ''}
  ${control ? `
  const controlCompiled=await build({entryPoints:['tools/m1010r/render-control.ts'],bundle:true,write:false,format:'iife',globalName:'renderControl'});
  const controlSandbox=createContext({Float32Array,Uint32Array});
  new Script(controlCompiled.outputFiles[0].text).runInContext(controlSandbox);
  const {verifySignalWindows,renderIntegrity,summarizeRender}=controlSandbox.renderControl;
  function signalFixture({first=0,shift=0}={}){
    const reference=new Float32Array(256000),observed=new Float32Array(70656-first);
    const scheduled=[1,.5,.75].map((gain,index)=>({startFrame:[12000,38400,57600][index],
      referenceStart:(index+1)*48000,samples:8192,gain}));
    let seed=17;
    for(const region of scheduled)for(let offset=0;offset<region.samples;offset++){
      seed=(Math.imul(seed,1664525)+1013904223)>>>0;
      const value=(seed/4294967296-.5)/2;
      reference[region.referenceStart+offset]=value;
      observed[region.startFrame-first+offset+shift]=reference[region.referenceStart+offset]*region.gain;
    }
    return {reference,observed,scheduled,first};
  }
  function summaryFixture({first=0,target=70592,lengths=[128]}={}){
    const recorder=fixture(target);let end=first,index=0;
    while(end<target||index===0){const length=lengths[index++%lengths.length];recorder.step(end,length);end+=length;}
    const observation=endFrame=>({hostBefore:endFrame/48,hostAfter:endFrame/48,
      contextBefore:endFrame/48000,contextAfter:endFrame/48000});
    const recording={terminal:terminal(recorder,'RENDER_TARGET_REACHED'),terminalObservation:observation(end),
      heartbeats:recorder.messages.filter(row=>row.type==='render-heartbeat').map(row=>({...row,...observation(row.actualObservedEndFrame)})),
      contextEvents:[{state:'running',...observation(0)}],watchdogFired:false,timedOut:false,errors:[]};
    return summarizeRender(recording);
  }
  ` : ''}
  ${body}
`], { encoding: 'utf8', timeout: 15000 });

const integration = (body: string) => check(`
  async function calibrationFixture({deferControl=false,deferMedia=false}={}){
    let hostMs=0,nextTimer=0,settlements=0,result,controlResolve,controlReject,mediaResolve,mediaCollected,gpuFailure;
    const timers=new Map(),targets=[],nodes=[],contexts=[],worklets=[],pipelines=[],probes=[],calls=[];
    const events=[];
    class Events {
      constructor(){this.listeners=new Set();events.push(this);}
      addEventListener(type,callback,options={}){
        if(options.signal?.aborted)return;
        const row={type,callback,once:options.once,detach:()=>{}};const remove=()=>this.removeEventListener(type,callback);
        if(options.signal){options.signal.addEventListener('abort',remove,{once:true});
          row.detach=()=>options.signal.removeEventListener('abort',remove);}
        this.listeners.add(row);
      }
      removeEventListener(type,callback){for(const row of this.listeners)if(row.type===type&&row.callback===callback){row.detach();this.listeners.delete(row);}}
      emit(type){for(const row of [...this.listeners])if(row.type===type){if(row.once)this.removeEventListener(type,row.callback);row.callback({type});}}
    }
    class Element extends Events {removeAttribute(name){delete this[name];}}
    class Video extends Element {
      paused=true;currentTime=0;readyState=4;duration=70;videoWidth=1280;videoHeight=720;
      callbacks=new Map();nextCallback=0;
      get currentSrc(){return this.src??'';}
      load(){if(this.src)this.emit('loadedmetadata');}
      play(){calls.push('play');this.paused=false;this.emit('playing');return Promise.resolve();}
      pause(){this.paused=true;this.emit('pause');}
      requestVideoFrameCallback(callback){const id=++this.nextCallback;this.callbacks.set(id,callback);return id;}
      cancelVideoFrameCallback(id){this.callbacks.delete(id);}
      getVideoPlaybackQuality(){return {totalVideoFrames:1,droppedVideoFrames:0,corruptedVideoFrames:0};}
    }
    const video=new Video(),elements={source:video};
    for(const name of ['output','status','stage','play','pause','fullscreen','close','seek','volume','mute'])elements[name]=new Element();
    const document=Object.assign(new Events(),{baseURI:'https://fixture.test/calibration.html',visibilityState:'visible',
      hasFocus:()=>true,getElementById:id=>elements[id]});
    const window=Object.assign(new Events(),{isSecureContext:true,
      setTimeout(callback,delay){const id=++nextTimer;timers.set(id,{callback,due:hostMs+delay,delay});return id;},
      clearTimeout(id){timers.delete(id);},close(){calls.push('window-close');}});
    class AudioNode {
      constructor(kind){this.kind=kind;this.connections=[];this.disconnected=0;nodes.push(this);}
      connect(destination){this.connections.push(destination);return destination;}
      disconnect(){this.connections=[];this.disconnected++;}
    }
    class Context extends Events {
      state='suspended';sampleRate=48000;currentTime=0;baseLatency=.01;outputLatency=.02;
      destination={channelCount:2};sources=[];
      audioWorklet={addModule:url=>{assert(url.endsWith('/render-recorder.js'));return Promise.resolve();}};
      constructor(){super();contexts.push(this);}
      resume(){this.state='running';this.emit('statechange');return Promise.resolve();}
      close(){this.state='closed';this.emit('statechange');calls.push('audio-close');return Promise.resolve();}
      createBuffer(channels,length){const samples=new Float32Array(length);return {samples,copyToChannel:data=>samples.set(data)};}
      createBufferSource(){const node=new AudioNode('buffer');node.stopped=false;
        node.start=seconds=>{node.startFrame=Math.round(seconds*48000);};node.stop=()=>{node.stopped=true;};
        this.sources.push(node);return node;}
      createGain(){const node=new AudioNode('gain');node.gain={value:1,setValueAtTime(value){this.value=value;}};return node;}
      createMediaElementSource(){calls.push('media-source');return new AudioNode('media');}
    }
    class Worklet extends AudioNode {
      constructor(context,name,options){super('worklet');assert.equal(name,'m1010ri-render');
        this.context=context;this.target=options.processorOptions.targetEndFrame;targets.push(this.target);
        this.delivered=0;this.closed=0;this.onprocessorerror=null;this.commands=[];
        this.recorder=fixture(this.target,{holdMessages:true,sample:frame=>{
          let sum=0;for(const source of context.sources){if(source.stopped)continue;
            const gain=source.connections[0];if(gain?.connections[0]!==this)continue;
            sum+=(source.buffer.samples[frame-source.startFrame]??0)*gain.gain.value;}
          return sum+(!video.paused&&context.sources.every(source=>source.stopped)? .125:0);}});
        this.port={onmessage:null,onmessageerror:null,postMessage:data=>{
          this.commands.push(structuredClone(data));this.recorder.send(data);queueMicrotask(()=>this.deliver());},close:()=>{this.closed++;}};
        worklets.push(this);
      }
      deliver(){this.recorder.resume();const rows=this.recorder.delivered;
        while(this.delivered<rows.length)this.port.onmessage?.({data:rows[this.delivered++]});}
    }
    const gpu={capabilities:{timestampQuery:true},adapterReport:{description:'VM mock GPU'},device:{
      queue:{onSubmittedWorkDone:()=>Promise.resolve()},destroy(){calls.push('device-destroy');}}};
    class Baseline {}
    class Probe {
      encoded=0;constructor(inner,before){this.before=before;probes.push(this);}
      readAfterPause(){assert(video.paused);return Promise.resolve(new Uint32Array([this.encoded,0,0,0,41,1,1280,720]));}
      destroy(){calls.push('probe-destroy');}
    }
    class Pipeline {
      timingGeneration=7;framesRendered=0;onFrame=null;onGpuSample=null;
      constructor(received,input,output,probe){this.probe=probe;pipelines.push(this);}
      start(){this.started=true;calls.push('pipeline-start');}
      stop(){this.started=false;calls.push('pipeline-stop');}
      stats(){return {framesRendered:this.framesRendered};}
      drainTimings(){if(this.framesRendered)this.onGpuSample?.(gpuSample);return Promise.resolve();}
      destroy(){calls.push('pipeline-destroy');}
    }
    const gpuSample={ms:.25,generation:7,sequence:1,submittedAt:0,resolvedAt:65000,neural:false,upscalerId:'catmull-rom',source:{width:1280,height:720}};
    const doubles={acquireGpu:()=>Promise.resolve(gpu),watchDeviceFailures:(device,callback)=>{gpuFailure=callback;calls.push('watch');return ()=>calls.push('unwatch');},
      VideoPipeline:Pipeline,BaselineScaler:Baseline,IdentityProbe:Probe,
      runRenderControl:()=>{calls.push('deferred-control');return new Promise((resolve,reject)=>{controlResolve=resolve;controlReject=reject;});},
      beginRenderRecording(begin,...args){const handle=begin(...args);handle.finished.then(value=>{mediaCollected=value;});
        return {...handle,finished:new Promise(resolve=>{mediaResolve=resolve;})};}};
    const mappings={'device.js':['acquireGpu','watchDeviceFailures'],'pipeline.js':['VideoPipeline'],
      'baseline-scaler.js':['BaselineScaler'],'probe.js':['IdentityProbe']};
    if(deferControl)mappings['render-control.js']=['runRenderControl'];
    if(deferMedia)mappings['render-host.js']=['beginRenderRecording'];
    const bundled=await build({entryPoints:['tools/m1010r/calibration.ts'],bundle:true,write:false,format:'iife',
      define:{__M1010RI__:'true'},plugins:[{name:'ri-integration',setup(plugin){
        plugin.onResolve({filter:/./},args=>{const names=mappings[args.path.split('/').at(-1)];
          if(names&&args.importer.endsWith('/calibration.ts'))return {path:names.join(','),namespace:'ri-mock'};});
        plugin.onLoad({filter:/.*/,namespace:'ri-mock'},args=>({loader:'js',resolveDir:process.cwd(),contents:
          (args.path==='beginRenderRecording'?'import {beginRenderRecording as begin} from "./tools/m1010r/render-host.ts";'+
            'export const beginRenderRecording=(...args)=>globalThis.doubles.beginRenderRecording(begin,...args);':
            (args.path==='runRenderControl'?'export {summarizeRender,renderIntegrity} from "./tools/m1010r/render-control.ts";':'')+
            args.path.split(',').map(name=>'export const '+name+' = globalThis.doubles.'+name+';').join(''))}));}}]});
    let seed=17;const reference=Float32Array.from({length:256000},()=>{
      seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed/4294967296-.5)/2;});
    const sandbox=createContext({window,document,doubles,HTMLVideoElement:Video,HTMLCanvasElement:Element,
      HTMLOutputElement:Element,HTMLButtonElement:Element,HTMLInputElement:Element,HTMLElement:Element,
      AudioContext:Context,AudioWorkletNode:Worklet,AbortController,AbortSignal,ArrayBuffer,Float32Array,Uint32Array,
      URL,URLSearchParams,structuredClone,location:{search:''},navigator:{userAgent:'VM only'},
      performance:{timeOrigin:0,now:()=>hostMs},setTimeout:window.setTimeout,clearTimeout:window.clearTimeout,
      btoa:value=>Buffer.from(value,'binary').toString('base64'),fetch:async(url,options)=>{
        assert(url.endsWith('/media/decoded-30.f32le'));assert.equal(options.credentials,'omit');
        return {ok:true,arrayBuffer:()=>Promise.resolve(reference.buffer.slice(0))};}});
    new Script(bundled.outputFiles[0].text).runInContext(sandbox);
    const api=window.m1010rCalibration;
    const flush=async()=>{for(let turn=0;turn<80;turn++)await Promise.resolve();};
    return {api,video,elements,contexts,worklets,pipelines,targets,calls,flush,gpuSample,
      get result(){return result;},get settlements(){return settlements;},
      start(){const task=api.run({fps:30,mediaUrl:'media/replay-30.mp4'});task.then(value=>{result=value;settlements++;});return task;},
      elapse(milliseconds){hostMs+=milliseconds;for(;;){const next=[...timers].filter(([,row])=>row.due<=hostMs)
        .sort((left,right)=>left[1].due-right[1].due||left[0]-right[0])[0];if(!next)break;
        timers.delete(next[0]);next[1].callback();}},
      render(node,first,end){for(let frame=first;frame<end;frame+=128){node.context.currentTime=(frame+128)/48000;
        node.recorder.step(frame);node.deliver();}},
      submit(){const pipeline=pipelines[0];assert(pipeline.started);assert(!video.paused);
        const metadata={mediaTime:1/30,presentationTime:hostMs,expectedDisplayTime:hostMs+16,presentedFrames:1,width:1280,height:720};
        video.currentTime=metadata.mediaTime;for(const [id,callback] of [...video.callbacks]){video.callbacks.delete(id);callback(hostMs,metadata);}
        pipeline.probe.before(pipeline.probe.encoded++);pipeline.framesRendered++;
        pipeline.onFrame({now:hostMs,mediaTime:metadata.mediaTime,size:{width:1280,height:720},presentedDelta:1,
          presentationTime:metadata.presentationTime,expectedDisplayTime:metadata.expectedDisplayTime,decodeLatencyMs:null});},
      resolveControl:value=>controlResolve(value),rejectControl:error=>controlReject(error),
      failGpu:()=>gpuFailure('Injected device loss'),
      resolveMedia(eligible){assert(mediaCollected);mediaResolve({...mediaCollected,watchdogFired:!eligible,
        errors:eligible?[]:['WATCHDOG_ABORT']});},
      clean(){assert.equal(timers.size,0);assert.equal(video.callbacks.size,0);assert.equal(video.paused,true);assert.equal(video.src,undefined);
        assert(events.every(target=>target.listeners.size===0));assert(contexts.every(context=>context.state==='closed'));
        assert(nodes.every(node=>node.disconnected>0&&node.connections.length===0));
        assert(worklets.every(node=>node.closed>0&&node.port.onmessage===null&&node.port.onmessageerror===null&&node.onprocessorerror===null));
        assert(contexts.flatMap(context=>context.sources).every(source=>source.stopped));
        assert.equal(calls.filter(value=>value==='device-destroy').length,1);assert.equal(calls.filter(value=>value==='unwatch').length,1);
        const cleanup=api.snapshot().cleanup;for(const name of ['pipeline','probe','device','audioNodes','observers','timers','pendingCompletions'])assert.equal(cleanup[name],0,name);
        assert.equal(cleanup.completed,true);assert.equal(cleanup.audioContext,'closed');}
    };
  }
  ${body}
`);

describe('RI integration', () => {
  it('runs actual control then media targets, maps one submission, and never finishes from wall time', () => integration(`
    for(const outcome of ['target','suspend','abort','deferred-valid','deferred-invalid']){
      const deferred=outcome.startsWith('deferred'),test=await calibrationFixture({deferMedia:deferred});
      const task=test.start();await test.flush();assert.equal(test.worklets.length,1);
      test.render(test.worklets[0],0,70528);await test.flush();assert(!test.calls.includes('media-source'));
      test.render(test.worklets[0],70528,70656);await test.flush();
      assert.deepEqual(test.targets,[70592,3286656]);assert.equal(test.api.snapshot().audioControl.verifiedWindows,180);
      const bounds=test.api.snapshot().renderWindow;
      assert.deepEqual(bounds,{epochFrame:70656,startFrame:310656,endFrame:3190656,targetEndFrame:3286656,sampleRate:48000});
      assert.equal(test.api.snapshot().frames.length,0);test.submit();await test.flush();
      const media=test.worklets[1];test.render(media,bounds.epochFrame,bounds.epochFrame+128);
      test.elapse(65000);await test.flush();assert.equal(test.settlements,0);assert.equal(test.api.snapshot().state,'RUNNING');
      assert.equal(test.api.snapshot().plannedEndPerformance,null);assert.equal(test.api.snapshot().startPerformance,null);
      assert.throws(()=>test.api.audioBase64());assert.equal(media.commands.length,0);
      if(outcome==='suspend'){test.contexts[0].state='suspended';test.contexts[0].emit('statechange');}
      else if(outcome==='abort')test.elements.close.emit('click');
      else {test.render(media,bounds.epochFrame+128,bounds.targetEndFrame-128);await test.flush();assert.equal(test.settlements,0);
        test.render(media,bounds.targetEndFrame-128,bounds.targetEndFrame);await test.flush();
        if(deferred){assert.equal(test.settlements,0);assert.equal(test.api.snapshot().renderAudio,null);test.resolveMedia(outcome==='deferred-valid');}}
      await test.flush();const result=await task,success=['target','deferred-valid'].includes(outcome);
      assert.equal(result.state,success?'RECORDED':'UNRESOLVED');
      const partial=['suspend','abort'].includes(outcome),samples=partial?128:3216000;
      assert.equal(result.audio.samples,samples);assert.equal(Buffer.from(test.api.audioBase64(),'base64').length,samples*4);
      assert.equal(result.renderAudio.terminal.completionReason,partial?(outcome==='abort'?'HOST_ABORT':'AUDIO_CONTEXT_SUSPENDED'):'RENDER_TARGET_REACHED');
      assert.equal(result.errors.length===0,success);assert.deepEqual(result.gpuSamples,[test.gpuSample]);
      assert.deepEqual(result.completeness,{firstSequence:1,lastSequence:1,count:1,encodedCount:1,observedCallbackCount:1});
      const row=result.frames[0];assert.equal(result.metadataMatchErrors,0);assert.equal(row.readyAt,0);
      assert.deepEqual([row.index,row.sequence,row.generation,row.sourceIdentity,row.identityValid,row.sourceWidth,row.sourceHeight],
        [0,1,7,41,true,1280,720]);assert.equal(result.callbacks.length,1);
      assert.equal(test.settlements,1);test.clean();
      assert.equal(test.calls.filter(value=>value==='pipeline-destroy').length,1);assert.equal(test.calls.filter(value=>value==='probe-destroy').length,1);
    }
  `), 15000);

  it('retains partial and queued-complete short-control PCM after watchdog without starting media', () => integration(`
    for(const samples of [7168,70656]){const test=await calibrationFixture(),task=test.start();await test.flush();
      assert.deepEqual(test.targets,[70592]);assert.equal(test.api.snapshot().instrument,'M10.10RI');
      const node=test.worklets[0],deliver=node.deliver;node.deliver=()=>{};test.render(node,0,samples);test.elapse(15000);
      node.deliver=deliver;node.deliver();await test.flush();const result=await task,control=result.audioControl;
      assert.equal(result.state,'UNRESOLVED');assert.equal(control.render.watchdogFired,true);
      assert.equal(control.render.terminal.completionReason,samples===7168?'WATCHDOG_ABORT':'RENDER_TARGET_REACHED');
      assert.equal(control.samples,samples);assert.equal(control.verifiedWindows,samples===7168?0:180);
      assert(control.errors.includes('Render completion not eligible'));
      assert.equal(Buffer.from(test.api.audioBase64(true),'base64').length,samples*4);
      assert.deepEqual(result.frames,[]);assert.equal(result.renderWindow,null);assert.equal(result.audio,null);
      assert(!test.calls.includes('media-source'));assert(!test.calls.includes('play'));assert.equal(test.pipelines.length,0);
      assert.equal(test.settlements,1);test.clean();}
  `));

  it('rejects short-control setup and ignores a late settlement after rejection', () => integration(`
    const donor=await calibrationFixture();const donorTask=donor.start();await donor.flush();donor.render(donor.worklets[0],0,70656);
    await donor.flush();donor.elements.close.emit('click');await donor.flush();const donorResult=await donorTask;donor.clean();
    const control={...donorResult.audioControl,pcm:Uint8Array.from(Buffer.from(donor.api.audioBase64(true),'base64')).buffer};
    assert.equal(control.errors.length,0);assert.equal(control.verifiedWindows,180);
    for(const mode of ['reject']){const test=await calibrationFixture({deferControl:true}),task=test.start();await test.flush();
      assert(test.calls.includes('deferred-control'));assert.equal(test.worklets.length,0);
      test.rejectControl(new Error('short-control setup rejected'));
      await test.flush();const result=await task;assert.equal(result.state,'UNRESOLVED');
      assert(result.errors.some(row=>row.message.includes('short-control setup rejected')));
      assert.equal(result.audioControl,null);assert.equal(result.renderWindow,null);test.clean();
      test.resolveControl(control);await test.flush();test.elapse(100000);await test.flush();
      assert.equal(test.settlements,1);assert.equal(test.worklets.length,0);assert.equal(test.pipelines.length,0);
      assert(!test.calls.includes('media-source'));assert(!test.calls.includes('play'));assert.deepEqual(test.api.snapshot(),result);test.clean();
    }
  `));

  it('drains actual partial control before closing its context on host or GPU cancellation', () => integration(`
    for(const failure of ['close','gpu']){
      const test=await calibrationFixture(),task=test.start();await test.flush();
      const worklet=test.worklets[0];test.render(worklet,0,7168);
      const deliver=worklet.deliver;worklet.deliver=()=>{};
      if(failure==='close')test.elements.close.emit('click');else test.failGpu();
      await test.flush();assert.equal(test.contexts[0].state,'running');assert.equal(test.settlements,0);
      assert(worklet.commands.some(command=>command.reason==='HOST_ABORT'));
      worklet.deliver=deliver;worklet.deliver();await test.flush();
      const result=await task;assert.equal(result.state,'UNRESOLVED');
      assert.equal(result.audioControl.render.terminal.completionReason,'HOST_ABORT');
      assert.equal(result.audioControl.samples,7168);assert.equal(Buffer.from(test.api.audioBase64(true),'base64').length,28672);
      assert.equal(result.renderWindow,null);assert(!test.calls.includes('media-source'));assert(!test.calls.includes('play'));
      test.clean();assert.equal(test.settlements,1);
    }
  `));
});

describe('M10.10RI render-control verification', () => {
  it('locates all 180 windows in three known regions with their actual gains and a nonzero first frame', () => check(`
    const {reference,observed,scheduled,first}=signalFixture({first:4096});
    assert.deepEqual(scheduled.map(row=>[row.startFrame,row.gain]),[[12000,1],[38400,.5],[57600,.75]]);
    const result=verifySignalWindows(reference,observed,first,scheduled);
    assert.deepEqual(structuredClone(result),{expectedWindows:180,verifiedWindows:180,maximumSampleError:0,errors:[]});
    for(const region of scheduled){const single=verifySignalWindows(reference,observed,first,[region]);
      assert.equal(single.expectedWindows,60);assert.equal(single.verifiedWindows,60);assert.equal(single.errors.length,0);
      assert.equal(observed[region.startFrame-first+1000],Math.fround(reference[region.referenceStart+1000]*region.gain));}
    assert(observed.subarray(0,12000-first).every(value=>value===0));
    assert(observed.subarray(65792-first).every(value=>value===0));
    const summary=summaryFixture();assert.deepEqual(Array.from(renderIntegrity(summary,12000,70592)),[]);
  `, false, true));

  it('bounds located shifts and rejects wrong gains instead of accepting a nearby unrelated window', () => check(`
    for(const shift of [-25,-13,-12,12,13,25]){
      const {reference,observed,scheduled}=signalFixture({shift});
      const result=verifySignalWindows(reference,observed,0,scheduled),valid=Math.abs(shift)<=12;
      assert.deepEqual([result.expectedWindows,result.verifiedWindows,result.maximumSampleError,result.errors.length],
        [180,valid?180:0,valid?Math.abs(shift):null,valid?0:180]);
    }
    const {reference,observed,scheduled}=signalFixture();
    for(const [index,region] of scheduled.entries()){
      const wrong=scheduled.map((row,position)=>({...row,gain:position===index?row.gain*2:row.gain}));
      const result=verifySignalWindows(reference,observed,0,wrong);
      assert.equal(result.verifiedWindows,120);assert.equal(result.errors.length,60);
      assert.equal(result.errors[0],'Unverified signal window '+(region.startFrame+256));
    }
  `, false, true), 15000);

  it('rejects the old 7168-sample PCM and a recording missing only the first signal', () => check(`
    const {reference,observed,scheduled}=signalFixture();
    const old=verifySignalWindows(reference,observed.slice(0,7168),0,scheduled);
    assert.deepEqual([old.expectedWindows,old.verifiedWindows,old.maximumSampleError,old.errors.length],[180,0,null,180]);
    observed.fill(0,12000,20192);
    const missing=verifySignalWindows(reference,observed,0,scheduled);
    assert.deepEqual([missing.expectedWindows,missing.verifiedWindows,missing.errors.length],[180,120,60]);
    assert.equal(missing.errors[0],'Unverified signal window 12256');
    assert.equal(missing.errors.at(-1),'Unverified signal window 19808');
    assert(renderIntegrity(summaryFixture({target:7168}),12000,70592).includes('Required render interval not covered'));
  `, false, true));

  it('rejects nonfinite PCM inside signal windows and in otherwise unchecked silence', () => check(`
    const {reference,observed,scheduled}=signalFixture();
    for(const position of [0,12512,70655])for(const value of [NaN,Infinity,-Infinity]){
      const corrupt=observed.slice();corrupt[position]=value;
      const result=verifySignalWindows(reference,corrupt,0,scheduled);
      assert(result.errors.includes('Nonfinite PCM'));assert.equal(result.expectedWindows,180);
      if(position===12512)assert(result.verifiedWindows<180);else assert.equal(result.verifiedWindows,180);
    }
  `, false, true), 15000);

  it('requires the complete tail and keeps the whole final block for variable quanta', () => check(`
    for(const lengths of [[64],[96],[128],[192],[256],[512],[64,96,192,128,256,512]]){
      const summary=summaryFixture({first:4096,lengths}),row=summary.terminal;
      assert.deepEqual(Array.from(renderIntegrity(summary,12000,70592)),[]);
      assert(row.actualObservedEndFrame-65792>=4800);
      assert(row.actualObservedEndFrame-70592>=0);
      assert(row.actualObservedEndFrame-70592<row.blockLengths.at(-1));
      assert.equal(row.blockLengths.reduce((sum,length)=>sum+length,0),row.processedSamples);
      assert.equal(row.blockLengths.length,row.processedBlocks);
    }
    const exact=summaryFixture({first:64});assert.equal(exact.terminal.actualObservedEndFrame-65792,4800);
    assert.deepEqual(Array.from(renderIntegrity(exact,12000,70592)),[]);
    const short=summaryFixture({first:63,target:70591});assert.equal(short.terminal.actualObservedEndFrame-65792,4799);
    assert(renderIntegrity(short,12000,70592).includes('Required render interval not covered'));
    const spanning=summaryFixture({target:300,lengths:[96,192,64]});
    assert.equal(spanning.terminal.actualObservedEndFrame,352);assert.equal(spanning.terminal.processedSamples,352);
    assert.deepEqual(Array.from(renderIntegrity(spanning,0,300)),[]);
    for(const requestedEndFrame of [288,353]){const corrupt=structuredClone(spanning);
      corrupt.terminal.requestedEndFrame=requestedEndFrame;
      assert(renderIntegrity(corrupt,0,300).includes('Terminal target not in last block'));}
  `, false, true));

  it('rejects missing and late first frames even when the real worklet reports render success', () => check(`
    const exact=summaryFixture({first:12000});assert.deepEqual(Array.from(renderIntegrity(exact,12000,70592)),[]);
    for(const first of [12001,20192,70600]){
      const summary=summaryFixture({first});assert.equal(summary.terminal.completionReason,'RENDER_TARGET_REACHED');
      assert.equal(summary.terminal.firstFrame,first);
      assert(renderIntegrity(summary,12000,70592).includes('Required render interval not covered'));
    }
    for(const key of ['firstFrame','actualObservedEndFrame']){const summary=summaryFixture();summary.terminal[key]=null;
      assert(renderIntegrity(summary,12000,70592).includes('Required render interval not covered'));}
    const missing=summaryFixture();missing.terminal=null;
    assert(renderIntegrity(missing,12000,70592).includes('Missing render terminal'));
  `, false, true));

  it('rejects inconsistent block counts, sums and heartbeat sequences without mutating evidence', () => check(`
    const good=summaryFixture({lengths:[64,96,192,128,256,512]});
    const cases=[
      [row=>row.terminal.processedBlocks--,'Block count/sum inconsistent'],
      [row=>row.terminal.processedSamples--,'Block count/sum inconsistent'],
      [row=>row.terminal.blockLengths.pop(),'Block count/sum inconsistent'],
      ...[0,-1,1.5].map(value=>[row=>row.terminal.blockLengths[0]=value,'Block count/sum inconsistent']),
      [row=>row.terminal.heartbeatCount++,'Missing heartbeats'],
      [row=>row.heartbeats.shift(),'Missing heartbeats'],
      [row=>row.heartbeats.reverse(),'Heartbeat metadata inconsistent'],
      [row=>row.heartbeats[1]={...row.heartbeats[0],heartbeatOrdinal:2},'Heartbeat metadata inconsistent'],
      ...['heartbeatOrdinal','currentFrame','actualObservedEndFrame','processedSamples','processedBlocks','blockLength']
        .map(key=>[row=>row.heartbeats[1][key]++,'Heartbeat metadata inconsistent']),
      [row=>row.heartbeats[1].processedBlocks=row.terminal.processedBlocks+1,'Heartbeat metadata inconsistent'],
    ];
    for(const [corrupt,message] of cases){const row=structuredClone(good);corrupt(row);const before=structuredClone(row);
      assert(renderIntegrity(row,12000,70592).includes(message),message);assert.deepEqual(row,before);}
  `, false, true));

  it('never clears overflow, watchdog or timeout failures on an otherwise intact late success', () => check(`
    const good=summaryFixture();assert.equal(good.terminal.completionReason,'RENDER_TARGET_REACHED');
    for(const key of ['watchdogFired','timedOut']){const row=structuredClone(good);row[key]=true;
      assert(renderIntegrity(row,12000,70592).includes('Render completion not eligible'));}
    for(const key of ['overflow','discontinuity']){const row=structuredClone(good);row.terminal[key]=true;
      assert(renderIntegrity(row,12000,70592).includes('Render state/rate invalid'));}
  `, false, true));

  it('allows monotonic asynchronous clock delay but rejects negative lag beyond two measured quanta', () => check(`
    const good=summaryFixture({lengths:[64,96,192,128,256,512]});
    const delayed=structuredClone(good);
    for(const row of [...delayed.heartbeats,delayed.terminalObservation]){
      row.hostBefore=14000;row.hostAfter=14000;row.contextBefore=60;row.contextAfter=60;
    }
    assert.deepEqual(Array.from(renderIntegrity(delayed,12000,70592)),[]);
    const maximum=Math.max(...good.terminal.blockLengths),end=good.terminal.actualObservedEndFrame;
    for(const deficit of [2*maximum-.01,2*maximum+1]){
      const row=structuredClone(good);row.terminalObservation.contextBefore=(end-deficit)/48000;
      row.terminalObservation.contextAfter=row.terminalObservation.contextBefore;
      assert.equal(renderIntegrity(row,12000,70592).includes('Cross-clock relation invalid'),deficit>2*maximum);
    }
    const corruptions=[row=>row.heartbeats[1].hostBefore=row.heartbeats[0].hostAfter-1,
      row=>row.heartbeats[1].contextBefore=row.heartbeats[0].contextAfter-1,
      row=>row.terminalObservation.hostAfter=row.terminalObservation.hostBefore-1,
      row=>row.terminalObservation.contextAfter=row.terminalObservation.contextBefore-1,
      row=>delete row.heartbeats[0].contextBefore,
      ...['hostBefore','hostAfter','contextBefore','contextAfter'].flatMap(key=>
        [NaN,Infinity,-Infinity].map(value=>row=>row.terminalObservation[key]=value))];
    for(const corrupt of corruptions){const row=structuredClone(good);corrupt(row);
      assert(renderIntegrity(row,12000,70592).includes('Cross-clock relation invalid'));}
    good.terminalObservation=null;
    assert(renderIntegrity(good,12000,70592).includes('Missing completion clock or unexpected context state'));
  `, false, true));
});

describe('M10.10RI host recording', () => {
  it('keeps a host-clock-ahead watchdog snapshot partial despite advanced context time', () => check(`
    const host=hostFixture(),handle=host.start();assert.equal(host.node.connected,0);
    assert.deepEqual(structuredClone(host.node.options),{numberOfInputs:1,numberOfOutputs:1,
      outputChannelCount:[2],channelCount:2,channelCountMode:'explicit',channelInterpretation:'speakers',
      processorOptions:{targetEndFrame:70592}});
    assert.deepEqual(host.commands,[]);handle.node.connect(host.context.destination);
    for(let frame=0;frame<7168;frame+=128)host.render(frame);
    host.serviceMessages();host.setContextTime(600);host.advanceHost(15000);
    assert.equal(host.outcome,undefined);
    assert.deepEqual(host.commands,[{type:'snapshot-and-abort',reason:'WATCHDOG_ABORT'}]);
    host.serviceMessages();const result=await handle.finished;
    assert.equal(result.terminal.completionReason,'WATCHDOG_ABORT');assert.equal(result.watchdogFired,true);
    assert.equal(result.timedOut,false);assert.equal(result.terminal.processedSamples,7168);
    assert.equal(result.terminal.actualObservedEndFrame,7168);assert.equal(result.heartbeats.length,1);
    assert.equal(result.terminalObservation.contextBefore,600);assert.equal(result.terminalObservation.hostAfter,15000);
    assert.deepEqual(Array.from(result.errors),['WATCHDOG_ABORT']);host.clean();
    assert.equal(host.node.disconnected,0);assert.equal(host.node.closed,0);
    handle.dispose();assert.equal(host.node.disconnected,1);assert.equal(host.node.closed,1);
  `, true));

  it('keeps intact render-completed PCM while host servicing is delayed below the watchdog', () => check(`
    const host=hostFixture({target:300}),handle=host.start();
    host.render(0,96);host.render(96,192);host.render(288,64);
    assert.equal(host.outcome,undefined);assert.deepEqual(host.commands,[]);
    host.advanceHost(14000);host.setContextTime(18);host.reads.length=0;host.serviceMessages();
    const result=await handle.finished;assert.equal(result.terminal.completionReason,'RENDER_TARGET_REACHED');
    assert.equal(result.watchdogFired,false);assert.equal(result.timedOut,false);assert.equal(result.errors.length,0);
    assert.equal(result.terminal.actualObservedEndFrame,352);assert.equal(result.terminal.processedSamples,352);
    assert.deepEqual([...new Uint32Array(result.terminal.blockLengths).subarray(0,3)],[96,192,64]);
    assert.deepEqual(new Float32Array(result.terminal.pcm).subarray(0,352),
      Float32Array.from({length:352},(_,frame)=>(frame%1024-512)/1024));
    assert.deepEqual(host.reads,['host','context','context','host','host','context','context','host']);
    assert.equal(result.heartbeats[0].hostBefore,14000);assert.equal(result.heartbeats[0].contextAfter,18);
    assert.equal(result.terminalObservation.hostAfter,14000);host.clean();handle.dispose();
  `, true));

  it('rejects unresumed contexts and invalid configuration before allocating a node or timer', () => check(`
    const options=[...['suspended','interrupted','closed','unknown'].map(state=>({state})),
      ...[44100,96000,NaN,Infinity].map(rate=>({rate})),
      ...[0,-1,1.5,NaN,Infinity,Number.MAX_SAFE_INTEGER+1,'70592',null].map(target=>({target})),
      ...[0,-1,NaN,Infinity,2147483648,'15000',null].map(watchdogMs=>({watchdogMs}))];
    for(const option of options){const host=hostFixture(option);
      assert.throws(()=>host.start());assert.equal(host.node,undefined);
      assert.equal(host.timers.size,0);assert.equal(host.stateListeners.size,0);assert.equal(host.commands.length,0);}
  `, true));

  it('uses the caller watchdog duration and never declares completion from clocks alone', () => check(`
    for(const watchdogMs of [15000,75000]){
      const host=hostFixture({watchdogMs}),handle=host.start();host.setContextTime(10000);
      host.advanceHost(watchdogMs-1);await Promise.resolve();assert.equal(host.outcome,undefined);
      assert.deepEqual(host.commands,[]);assert.deepEqual(host.node.recorder.messages,[]);
      host.advanceHost(1);assert.deepEqual(host.commands,[{type:'snapshot-and-abort',reason:'WATCHDOG_ABORT'}]);
      host.serviceMessages();const result=await handle.finished;
      assert.equal(result.terminal.completionReason,'WATCHDOG_ABORT');assert.equal(result.watchdogFired,true);
      assert.equal(result.terminal.firstFrame,null);assert.equal(result.terminal.actualObservedEndFrame,null);
      assert.equal(result.terminal.processedSamples,0);assert.equal(result.terminal.processedBlocks,0);
      assert.equal(result.timedOut,false);assert.equal(result.heartbeats.length,0);host.clean();handle.dispose();
    }
  `, true));

  it('retains an already queued render success after the watchdog without clearing its failure flag', () => check(`
    const host=hostFixture({target:300}),handle=host.start();host.render(0,192);host.render(192,192);
    assert.equal(host.node.recorder.messages.at(-1).completionReason,'RENDER_TARGET_REACHED');
    host.advanceHost(15000);assert.equal(host.commands[0].reason,'WATCHDOG_ABORT');
    host.advanceHost(1999);host.serviceMessages();const result=await handle.finished;
    assert.equal(result.terminal.completionReason,'RENDER_TARGET_REACHED');assert.equal(result.terminal.processedSamples,384);
    assert.equal(result.watchdogFired,true);assert.equal(result.timedOut,false);assert(result.errors.includes('WATCHDOG_ABORT'));
    assert.deepEqual(new Float32Array(result.terminal.pcm).subarray(0,384),
      Float32Array.from({length:384},(_,frame)=>(frame%1024-512)/1024));
    host.clean();assert.equal(host.settlements,1);handle.dispose();
  `, true));

  it('keeps rendering and all heartbeat metadata during an independently advanced bounded host stall', () => check(`
    const starts=[12000,38400,57600],sample=frame=>{const pulse=starts.findIndex(start=>frame>=start&&frame<start+8192);
      return pulse<0?0:(pulse+1)*(frame-starts[pulse]+1)/32768;};
    const host=hostFixture({sample}),handle=host.start(),lengths=[];let frame=0;
    for(;frame<7168;frame+=128){host.render(frame);lengths.push(128);}host.serviceMessages();
    assert.equal(host.node.recorder.messages.filter(row=>row.type==='render-recording').length,0);
    host.advanceHost(3000,false);
    while(frame<40000){const length=[96,192,256][lengths.length%3];host.render(frame,length);lengths.push(length);frame+=length;}
    host.advanceHost(8000,false);
    while(frame<70592){const length=[64,128,512][lengths.length%3];host.render(frame,length);lengths.push(length);frame+=length;}
    assert.equal(host.outcome,undefined);assert.equal(host.commands.length,0);
    host.setContextTime(frame/48000);host.serviceMessages();host.serviceTimers();const result=await handle.finished;
    assert.equal(result.terminal.completionReason,'RENDER_TARGET_REACHED');assert.equal(result.watchdogFired,false);
    assert.equal(result.timedOut,false);assert.equal(result.errors.length,0);assert.equal(result.terminal.processedSamples,frame);
    assert.deepEqual([...new Uint32Array(result.terminal.blockLengths).subarray(0,lengths.length)],lengths);
    assert.deepEqual(new Float32Array(result.terminal.pcm).subarray(0,frame),Float32Array.from({length:frame},(_,offset)=>sample(offset)));
    const emitted=host.node.recorder.messages.filter(row=>row.type==='render-heartbeat');
    assert.equal(result.heartbeats.length,emitted.length);assert.equal(result.terminal.heartbeatCount,emitted.length);
    for(const [index,beat] of result.heartbeats.entries()){
      for(const key of Object.keys(emitted[index]))assert.equal(beat[key],emitted[index][key]);
      assert.equal(beat.hostBefore,index===0?0:11000);assert.equal(beat.hostAfter,beat.hostBefore);
      assert.equal(beat.contextBefore,index===0?0:frame/48000);assert.equal(beat.contextAfter,beat.contextBefore);
    }
    assert.deepEqual(Object.keys(result.terminal).sort(),Object.keys(host.node.recorder.messages.at(-1)).sort());
    host.clean();handle.dispose();
  `, true));

  it('records actual context transitions without treating running as render progress', () => check(`
    for(const firstState of ['suspended','interrupted','closed']){
      const host=hostFixture(),handle=host.start();host.render(0,96);host.serviceMessages();
      host.advanceHost(20);host.setContextTime(.002);host.state(firstState);
      host.state('running');host.advanceHost(400);host.state('interrupted');host.state('closed');
      assert.equal(host.commands.length,1);assert.equal(host.commands[0].reason,'AUDIO_CONTEXT_SUSPENDED');
      host.serviceMessages();const result=await handle.finished;
      assert.deepEqual(Array.from(result.contextEvents,row=>row.state),['running',firstState,'running','interrupted','closed']);
      assert.deepEqual(Array.from(result.contextEvents,row=>row.hostBefore),[0,20,20,420,420]);
      assert.deepEqual(Array.from(result.contextEvents,row=>row.contextAfter),[0,.002,.002,.002,.002]);
      assert.equal(result.terminal.completionReason,'AUDIO_CONTEXT_SUSPENDED');assert.equal(result.terminal.processedSamples,96);
      assert.equal(result.watchdogFired,false);assert.equal(result.timedOut,false);host.clean();handle.dispose();
    }
  `, true));

  it('drains processor failure without a response and keeps heartbeats with explicit missing PCM', () => check(`
    const host=hostFixture(),handle=host.start();host.render(0,128);host.serviceMessages();
    host.node.dropCommands=true;host.processorError('native render exception');
    host.advanceHost(1000);host.processorError('native render exception');host.state('interrupted');
    host.advanceHost(999);await Promise.resolve();assert.equal(host.outcome,undefined);
    host.advanceHost(1);const result=await handle.finished;
    assert.equal(result.terminal,null);assert.equal(result.terminalObservation,null);assert.equal(result.timedOut,true);
    assert.equal(result.watchdogFired,false);assert.equal(result.heartbeats.length,1);
    assert.equal(result.heartbeats[0].actualObservedEndFrame,128);
    assert(result.errors.some(message=>message.includes('native render exception')&&message.includes('lineno=12')));
    assert(result.errors.includes('PROCESSOR_ERROR'));assert(result.errors.some(message=>message.includes('PCM is missing')));
    assert.deepEqual(host.commands.map(row=>row.reason),['PROCESSOR_ERROR','AUDIO_CONTEXT_SUSPENDED']);
    host.clean();assert.equal(host.settlements,1);handle.dispose();
  `, true));

  it('preserves actual partial buffers from processor failure and manual abort', () => check(`
    for(const reason of ['PROCESSOR_ERROR','HOST_ABORT']){
      const host=hostFixture(),handle=host.start();host.render(0,192);host.serviceMessages();
      if(reason==='PROCESSOR_ERROR')host.processorError();else handle.abort('HOST_ABORT');
      host.advanceHost(100);host.serviceMessages();const result=await handle.finished;
      assert.equal(result.terminal.completionReason,reason);assert.equal(result.terminal.processedSamples,192);
      assert.equal(result.terminal.firstFrame,0);assert.equal(result.terminal.actualObservedEndFrame,192);
      assert.equal(new Uint32Array(result.terminal.blockLengths)[0],192);
      assert.equal(new Float32Array(result.terminal.pcm)[191],(191-512)/1024);
      assert.equal(result.timedOut,false);assert.equal(result.watchdogFired,false);host.clean();handle.dispose();
    }
  `, true));

  it('bounds all no-response failures and retains the watchdog flag even during an earlier failure drain', () => check(`
    for(const reason of ['WATCHDOG_ABORT','AUDIO_CONTEXT_SUSPENDED','HOST_ABORT','MALFORMED_RESULT']){
      const host=hostFixture(),handle=host.start();host.render(0,64);host.serviceMessages();host.node.dropCommands=true;
      if(reason==='WATCHDOG_ABORT')host.advanceHost(15000);
      else if(reason==='AUDIO_CONTEXT_SUSPENDED')host.state('suspended');
      else if(reason==='MALFORMED_RESULT')host.receive(null);else handle.abort(reason);
      host.advanceHost(1999);await Promise.resolve();assert.equal(host.outcome,undefined);
      host.advanceHost(1);const result=await handle.finished;
      assert.equal(result.terminal,null);assert.equal(result.timedOut,true);assert.equal(result.heartbeats.length,1);
      assert.equal(result.watchdogFired,reason==='WATCHDOG_ABORT');host.clean();handle.dispose();
    }
    const host=hostFixture(),handle=host.start();host.advanceHost(14999);host.node.dropCommands=true;
    handle.abort('HOST_ABORT');host.advanceHost(1);host.advanceHost(1999);const result=await handle.finished;
    assert.equal(result.watchdogFired,true);assert.equal(result.timedOut,true);
    assert.deepEqual(host.commands.map(row=>row.reason),['HOST_ABORT','WATCHDOG_ABORT']);host.clean();handle.dispose();
  `, true));

  it('rejects malformed terminal fields and buffers without overwriting the valid drained snapshot', () => check(`
    const corruptions=[
      row=>null,row=>[],row=>'{}',row=>({...row,type:'unknown'}),row=>({...row,completionReason:'COMPLETE'}),
      row=>({...row,requestedEndFrame:70593}),row=>({...row,firstFrame:undefined}),
      row=>({...row,overflow:1}),row=>({...row,discontinuity:'false'}),row=>({...row,sampleRate:NaN}),
      row=>({...row,sampleRate:Infinity}),row=>({...row,sampleRate:44100}),
      row=>({...row,pcm:new Float32Array(96)}),row=>({...row,blockLengths:[96]}),
      row=>({...row,pcm:new ArrayBuffer(3)}),row=>({...row,blockLengths:new ArrayBuffer(5)}),
      row=>({...row,pcm:new ArrayBuffer(4)}),row=>({...row,blockLengths:new ArrayBuffer(0)}),
      row=>{const pcm=new ArrayBuffer(384);structuredClone(pcm,{transfer:[pcm]});return {...row,pcm};},
      ...['firstFrame','actualObservedEndFrame','processedSamples','processedBlocks','heartbeatCount'].flatMap(key=>
        [-1,1.5,NaN,Infinity,Number.MAX_SAFE_INTEGER+1,'96'].map(value=>row=>({...row,[key]:value}))),
    ];
    const reference=fixture(70592);reference.step(0,96);reference.send('finish');const valid=reference.messages.at(-1);
    for(const corrupt of corruptions){const host=hostFixture(),handle=host.start();host.render(0,96);host.serviceMessages();
      host.receive(corrupt(valid));assert.equal(host.outcome,undefined);
      assert.deepEqual(host.commands,[{type:'snapshot-and-abort',reason:'MALFORMED_RESULT'}]);
      host.serviceMessages();const result=await handle.finished;
      assert(result.errors.includes('MALFORMED_RESULT'));assert.equal(result.timedOut,false);
      assert.equal(result.terminal.completionReason,'HOST_ABORT');assert.equal(result.terminal.processedSamples,96);
      assert.equal(result.terminal.pcm,host.node.recorder.messages.at(-1).pcm);
      assert.equal(result.heartbeats.length,1);host.clean();handle.dispose();
    }
  `, true));

  it('rejects malformed heartbeat metadata and decode errors without recursive abort storms or renewed deadlines', () => check(`
    const beat={type:'render-heartbeat',currentFrame:0,actualObservedEndFrame:128,sampleRate:48000,
      processedBlocks:1,processedSamples:128,blockLength:128,state:'RECORDING',heartbeatOrdinal:1};
    const corruptions=[{...beat,state:'COMPLETE'},{...beat,sampleRate:NaN},
      ...['currentFrame','actualObservedEndFrame','processedBlocks','processedSamples','blockLength','heartbeatOrdinal']
        .flatMap(key=>[-1,NaN,Infinity,1.5,'1',null,undefined].map(value=>({...beat,[key]:value})))];
    for(const corrupted of corruptions){const host=hostFixture(),handle=host.start();host.receive(corrupted);
      host.serviceMessages();const result=await handle.finished;
      assert.equal(result.heartbeats.length,0);assert(result.errors.includes('MALFORMED_RESULT'));host.clean();handle.dispose();}
    const host=hostFixture(),handle=host.start();host.render(0);host.serviceMessages();
    host.node.port.postMessage=data=>{host.commands.push(data);host.receive({type:'broken'});};
    host.node.port.onmessageerror({});assert.equal(host.commands.length,1);
    for(let count=0;count<19;count++){host.advanceHost(100);host.receive(null);host.node.port.onmessageerror({});}
    host.advanceHost(100);const result=await handle.finished;
    assert.equal(result.terminal,null);assert.equal(result.timedOut,true);assert.equal(result.heartbeats.length,1);
    assert.equal(host.commands.length,1);host.clean();handle.dispose();
  `, true));

  it('retains flagged partial evidence and leaves full continuity and waveform validation to offline analysis', () => check(`
    for(const kind of ['DISCONTINUITY','OVERFLOW','MALFORMED_RESULT']){
      const host=hostFixture(),handle=host.start();host.render(0,64);
      if(kind==='DISCONTINUITY')host.render(256,128);
      else if(kind==='OVERFLOW')host.render(64,48000*80);
      else host.node.recorder.process(64,[[new Float32Array(32)]],[[new Float32Array(128)]]);
      host.serviceMessages();const result=await handle.finished;
      assert.equal(result.terminal.completionReason,kind);assert.equal(result.terminal.processedSamples,64);
      assert.equal(result.terminal.discontinuity,kind==='DISCONTINUITY');assert.equal(result.terminal.overflow,kind==='OVERFLOW');
      assert.equal(new Uint32Array(result.terminal.blockLengths)[0],64);assert.equal(result.timedOut,false);
      host.clean();handle.dispose();
    }
    const host=hostFixture({target:128}),handle=host.start();host.render(0);
    const raw=host.node.recorder.messages.at(-1);raw.processedBlocks=0;raw.overflow=true;
    new Float32Array(raw.pcm)[0]=NaN;host.serviceMessages();const result=await handle.finished;
    assert.equal(result.terminal,raw);assert(Number.isNaN(new Float32Array(result.terminal.pcm)[0]));
    assert.equal(result.terminal.completionReason,'RENDER_TARGET_REACHED');assert.equal(result.terminal.overflow,true);
    assert.equal(result.terminal.processedBlocks,0);assert.equal('qualified' in result,false);
    host.clean();handle.dispose();
  `, true));

  it('settles once and ignores saved late callbacks without stopping caller-owned replay', () => check(`
    for(const timeout of [false,true]){
      const host=hostFixture({target:128}),handle=host.start();handle.node.connect(host.context.destination);
      const message=host.node.port.onmessage,messageerror=host.node.port.onmessageerror;
      const processorerror=host.node.onprocessorerror,statechange=[...host.stateListeners][0];
      const watchdog=[...host.timers.values()][0].callback;
      if(timeout)host.node.dropCommands=true;else host.render(0);
      handle.abort('HOST_ABORT');const drain=[...host.timers.values()].find(timer=>timer.due===2000).callback;
      if(timeout)host.advanceHost(2000);else host.serviceMessages();
      const result=await handle.finished,before=JSON.stringify(result),commands=host.commands.length;
      host.render(128);message({data:host.node.recorder.messages.at(-1)});message({data:null});messageerror({});
      processorerror({message:'late error'});host.context.state='suspended';statechange();watchdog();drain();handle.abort('WATCHDOG_ABORT');
      host.advanceHost(100000);await Promise.resolve();
      assert.equal(host.settlements,1);assert.equal(JSON.stringify(result),before);assert.equal(host.commands.length,commands);
      assert.equal(host.node.disconnected,0);assert.equal(host.node.closed,0);host.clean();
      handle.dispose();handle.dispose();host.clean();assert.equal(host.node.disconnected,1);assert.equal(host.node.closed,1);
    }
  `, true));

  it('disposes early with bounded drainage and preserves an abort before source connection', () => check(`
    for(const mode of ['dispose','source-failed','queued-success','no-response']){
      const host=hostFixture({target:128}),handle=host.start();assert.equal(host.node.connected,0);
      if(mode==='source-failed')handle.abort('PROCESSOR_ERROR');
      if(mode==='queued-success')host.render(0);
      if(mode==='no-response')host.node.dropCommands=true;
      handle.dispose();handle.dispose();assert.equal(host.node.disconnected,1);
      assert.equal(host.stateListeners.size,0);assert.equal(host.node.onprocessorerror,null);
      assert.equal(host.timers.size,1);assert.equal(host.node.closed,0);
      assert.equal(host.commands.length,1);assert.equal(host.commands[0].reason,mode==='source-failed'?'PROCESSOR_ERROR':'HOST_ABORT');
      if(mode==='no-response'){host.advanceHost(1999);await Promise.resolve();assert.equal(host.outcome,undefined);host.advanceHost(1);}
      else host.serviceMessages();
      const result=await handle.finished;assert.equal(result.timedOut,mode==='no-response');
      if(mode==='no-response')assert.equal(result.terminal,null);
      else assert.equal(result.terminal.completionReason,mode==='queued-success'?'RENDER_TARGET_REACHED':mode==='source-failed'?'PROCESSOR_ERROR':'HOST_ABORT');
      if(mode==='source-failed')assert.equal(result.terminal.firstFrame,null);
      assert.equal(host.settlements,1);assert.equal(host.node.closed,1);host.clean();handle.dispose();assert.equal(host.node.closed,1);
    }
  `, true));

  it('bounds snapshot-send exceptions and rejects attempts to send a successful host abort reason', () => check(`
    const host=hostFixture(),handle=host.start();
    for(const reason of ['RENDER_TARGET_REACHED','COMPLETE','finish',null,undefined])assert.throws(()=>handle.abort(reason));
    assert.equal(host.commands.length,0);assert.equal(host.timers.size,1);
    host.node.port.postMessage=()=>{throw new Error('port unavailable');};
    handle.abort('HOST_ABORT');host.advanceHost(2000);const result=await handle.finished;
    assert.equal(result.terminal,null);assert.equal(result.timedOut,true);
    assert(result.errors.some(message=>message.includes('port unavailable')));host.clean();handle.dispose();
  `, true));
});

describe('M10.10RI render-clock completion authority', () => {
  it('completes only at an actual block end for every supported quantum', () => check(`
    for(const quantum of [1,64,96,128,192,256,512]){
      for(const offset of new Set([quantum,Math.max(1,Math.floor(quantum/2))])){
        const first=37,target=first+2*quantum+offset,sample=frame=>(frame-first+1)/4096;
        const recorder=fixture(target,{initialFrame:first,sample});
        recorder.step(first,quantum);recorder.step(first+quantum,quantum);
        assert(!recorder.messages.some(row=>row.type==='render-recording'));
        recorder.step(first+2*quantum,quantum);
        const result=terminal(recorder,'RENDER_TARGET_REACHED');
        assert.equal(result.firstFrame,first);
        assert.equal(result.requestedEndFrame,target);
        assert.equal(result.actualObservedEndFrame,first+3*quantum);
        assert(result.actualObservedEndFrame-target>=0);
        assert(result.actualObservedEndFrame-target<quantum);
        assert.equal(result.processedSamples,3*quantum);assert.equal(result.processedBlocks,3);
        assert.deepEqual([...new Uint32Array(result.blockLengths).subarray(0,3)],Array(3).fill(quantum));
        const pcm=new Float32Array(result.pcm);
        assert.deepEqual(pcm.subarray(0,result.processedSamples),Float32Array.from({length:3*quantum},(_,offset)=>sample(first+offset)));
        assert.equal(pcm[result.processedSamples-1],sample(result.actualObservedEndFrame-1));
        assert.equal(pcm[result.processedSamples],0);
        assert.equal(result.overflow,false);assert.equal(result.discontinuity,false);
      }
    }
  `));

  it('cannot complete at the old 7168-frame point and returns an explicit partial abort', () => check(`
    const recorder=fixture(70592);for(let frame=0;frame<7168;frame+=128)recorder.step(frame);
    assert(!recorder.messages.some(row=>row.type==='render-recording'));
    recorder.send({type:'snapshot-and-abort',reason:'WATCHDOG_ABORT'});
    const result=recorder.messages.at(-1);assert.equal(result.completionReason,'WATCHDOG_ABORT');
    assert.equal(result.firstFrame,0);assert.equal(result.actualObservedEndFrame,7168);
    assert.equal(result.processedSamples,7168);assert.equal(result.processedBlocks,56);
    assert.equal(result.requestedEndFrame,70592);
  `));

  it('covers all scheduled signal PCM after the old 7168-frame point through 70592', () => check(`
    const starts=[12000,38400,57600];
    const sample=frame=>{const pulse=starts.findIndex(start=>frame>=start&&frame<start+8192);
      return pulse<0?0:(pulse+1)*(frame-starts[pulse]+1)/32768;};
    const recorder=fixture(70592,{sample});
    for(let frame=0;frame<7168;frame+=128)recorder.step(frame);
    assert(!recorder.messages.some(row=>row.type==='render-recording'));
    assert.equal(recorder.advanceHost(120000),120000);
    assert(!recorder.messages.some(row=>row.type==='render-recording'));
    for(let frame=7168;frame<70592;frame+=128){
      assert(!recorder.messages.some(row=>row.type==='render-recording'));recorder.step(frame,Math.min(128,70592-frame));
    }
    const result=terminal(recorder,'RENDER_TARGET_REACHED');
    assert.equal(result.firstFrame,0);assert.equal(result.actualObservedEndFrame,70592);
    assert.equal(result.requestedEndFrame,70592);assert.equal(result.processedSamples,70592);
    assert.equal(result.processedBlocks,552);
    assert.deepEqual([...new Uint32Array(result.blockLengths).subarray(0,552)],[...Array(551).fill(128),64]);
    const pcm=new Float32Array(result.pcm).subarray(0,result.processedSamples);
    assert.deepEqual(pcm,Float32Array.from({length:70592},(_,frame)=>sample(frame)));
    assert(pcm.subarray(0,7168).every(value=>value===0));
    for(const start of starts)assert.equal(pcm.subarray(start,start+8192).filter(value=>value!==0).length,8192);
    assert(pcm.subarray(65792).every(value=>value===0));
  `));

  it('cannot advance heartbeats or completion when only the host clock advances', () => check(`
    const recorder=fixture(70592);
    assert.equal(recorder.advanceHost(80000),80000);assert.deepEqual(recorder.messages,[]);
    recorder.step(0);
    const before=structuredClone(recorder.messages);
    for(let tick=0;tick<10;tick++)recorder.advanceHost(60000);
    assert.deepEqual(recorder.messages,before);
    recorder.send({type:'snapshot-and-abort',reason:'WATCHDOG_ABORT'});
    const result=terminal(recorder,'WATCHDOG_ABORT');
    assert.equal(result.actualObservedEndFrame,128);assert.equal(result.processedSamples,128);
    assert.equal(result.processedBlocks,1);assert.equal(result.heartbeatCount,1);
  `));

  it('retains the whole spanning quantum and completes without host message servicing', () => check(`
    const recorder=fixture(300);recorder.step(0,96);recorder.step(96,192);recorder.step(288,64);
    const result=recorder.messages.at(-1);assert.equal(result.completionReason,'RENDER_TARGET_REACHED');
    assert.equal(result.actualObservedEndFrame,352);assert.equal(result.requestedEndFrame,300);
    assert.deepEqual([...new Uint32Array(result.blockLengths).subarray(0,3)],[96,192,64]);
    recorder.send({type:'snapshot-and-abort',reason:'WATCHDOG_ABORT'});recorder.step(352,128);
    assert.equal(recorder.messages.filter(row=>row.type==='render-recording').length,1);
  `));

  it('emits monotone heartbeats at least 12000 frames apart and no more than four per audio second', () => check(`
    for(const quantum of [1,64,96,128,192,256,512]){
      const target=48000+2*quantum,recorder=fixture(target);
      for(let frame=0;frame<target;frame+=quantum)recorder.step(frame,quantum);
      const result=terminal(recorder,'RENDER_TARGET_REACHED');
      const heartbeats=recorder.messages.filter(row=>row.type==='render-heartbeat');
      const stride=Math.ceil(12000/quantum)*quantum;
      assert.deepEqual(heartbeats.map(row=>row.actualObservedEndFrame),
        Array.from({length:Math.ceil((target-quantum)/stride)},(_,index)=>quantum+index*stride));
      for(const [index,beat] of heartbeats.entries()){
        assert.equal(beat.heartbeatOrdinal,index+1);assert.equal(beat.state,'RECORDING');
        assert.equal(beat.sampleRate,48000);assert.equal(beat.blockLength,quantum);
        assert.equal(beat.currentFrame+beat.blockLength,beat.actualObservedEndFrame);
        assert.equal(beat.processedSamples,beat.actualObservedEndFrame);
        assert.equal(beat.processedBlocks*quantum,beat.processedSamples);
        if(index){const previous=heartbeats[index-1];
          assert(beat.actualObservedEndFrame-previous.actualObservedEndFrame>=12000);
          assert(beat.processedBlocks>previous.processedBlocks);
          assert(beat.processedSamples>previous.processedSamples);
        }
        assert(heartbeats.filter(row=>row.actualObservedEndFrame>=beat.actualObservedEndFrame&&
          row.actualObservedEndFrame<beat.actualObservedEndFrame+48000).length<=4);
      }
      assert.equal(result.heartbeatCount,heartbeats.length);
      assert.equal(recorder.messages.length,heartbeats.length+1);
      assert(heartbeats.length<result.processedBlocks/4);
    }
  `));

  it('delivers queued heartbeats and intact terminal PCM when the host resumes after audio completion', () => check(`
    const sample=frame=>(frame%1024-512)/1024;
    const recorder=fixture(48001,{holdMessages:true,sample}),lengths=[];
    const quantums=[1,64,96,128,192,256,512];let end=0;
    while(end<48001){const length=quantums[lengths.length%quantums.length];
      recorder.step(end,length);lengths.push(length);end+=length;
    }
    assert.deepEqual(recorder.delivered,[]);
    const result=terminal(recorder,'RENDER_TARGET_REACHED');
    assert.equal(result.actualObservedEndFrame,end);assert.equal(result.processedSamples,end);
    assert.equal(result.processedBlocks,lengths.length);assert(end-48001<lengths.at(-1));
    assert.deepEqual([...new Uint32Array(result.blockLengths).subarray(0,lengths.length)],lengths);
    assert.deepEqual(new Float32Array(result.pcm).subarray(0,end),Float32Array.from({length:end},(_,frame)=>sample(frame)));
    assert(result.heartbeatCount>=4);assert.equal(result.heartbeatCount,recorder.messages.length-1);
    const before=structuredClone(recorder.messages);
    recorder.advanceHost(90000);recorder.send({type:'snapshot-and-abort',reason:'WATCHDOG_ABORT'});
    recorder.send('finish');recorder.send({type:'invalid'});recorder.step(end);
    assert.deepEqual(recorder.messages,before);assert.deepEqual(recorder.delivered,[]);
    recorder.resume();assert.deepEqual(recorder.delivered,before);
    recorder.resume();assert.deepEqual(recorder.delivered,before);
  `));

  it('keeps abort and malformed terminals immutable, including distinct context and processor failure labels', () => check(`
    const cases=[
      [{type:'snapshot-and-abort',reason:'WATCHDOG_ABORT'},'WATCHDOG_ABORT'],
      ['finish','HOST_ABORT'],[{type:'finish'},'HOST_ABORT'],
      [null,'MALFORMED_RESULT'],[{type:'unknown'},'MALFORMED_RESULT'],
      [{type:'snapshot-and-abort',reason:'COMPLETE'},'HOST_ABORT'],
      [{type:'snapshot-and-abort',reason:'AUDIO_CONTEXT_SUSPENDED'},'AUDIO_CONTEXT_SUSPENDED'],
      [{type:'snapshot-and-abort',reason:'PROCESSOR_ERROR'},'PROCESSOR_ERROR'],
    ];
    for(const [message,reason] of cases){
      const recorder=fixture(300);recorder.step(0,96);recorder.send(message);
      const result=terminal(recorder,reason);
      assert.equal(result.firstFrame,0);assert.equal(result.actualObservedEndFrame,96);
      assert.equal(result.requestedEndFrame,300);assert.equal(result.processedSamples,96);
      assert.equal(result.processedBlocks,1);assert.equal(new Uint32Array(result.blockLengths)[0],96);
      const before=structuredClone(recorder.messages);
      for(const [later] of cases)recorder.send(later);
      recorder.step(96,256);recorder.advanceHost(120000);
      assert.deepEqual(recorder.messages,before);
    }
  `));

  it('rejects non-future or invalid targets and sample rates other than 48000', () => check(`
    for(const target of [undefined,null,'513',NaN,Infinity,Number.MAX_SAFE_INTEGER+1,513.5,-1,0,511,512])
      assert.throws(()=>fixture(target,{initialFrame:512}),/future integer render-frame target at 48000 Hz/);
    for(const rate of [44100,96000])
      assert.throws(()=>fixture(1000,{rate}),/future integer render-frame target at 48000 Hz/);
    const recorder=fixture(513,{initialFrame:512});recorder.step(512,1);
    assert.equal(terminal(recorder,'RENDER_TARGET_REACHED').actualObservedEndFrame,513);
  `));

  it('records zero-input outputs as silence with real clock progress and every block length', () => check(`
    const recorder=fixture(200),lengths=[64,96,128];let end=0;
    recorder.process(0,[[]],[[]]);assert.deepEqual(recorder.messages,[]);
    for(const length of lengths){
      const outputs=[new Float32Array(length).fill(NaN),new Float32Array(length).fill(NaN)];
      recorder.process(end,end===0?[]:[[]],[outputs]);end+=length;
      assert(outputs.every(channel=>channel.every(value=>value===0)));
      if(end<200)assert(!recorder.messages.some(row=>row.type==='render-recording'));
    }
    const result=terminal(recorder,'RENDER_TARGET_REACHED');
    assert.equal(result.firstFrame,0);assert.equal(result.actualObservedEndFrame,288);
    assert.equal(result.processedSamples,288);assert.equal(result.processedBlocks,3);
    assert.deepEqual([...new Uint32Array(result.blockLengths).subarray(0,3)],lengths);
    assert(new Float32Array(result.pcm).subarray(0,288).every(value=>value===0));
  `));

  it('rejects early manual finish and discontinuous render progress', () => check(`
    const manual=fixture(1000);manual.step(0);manual.send('finish');
    terminal(manual,'HOST_ABORT');
    for(const start of [256,64,0]){
      const recorder=fixture(200);recorder.step(0);recorder.step(start);
      const result=terminal(recorder,'DISCONTINUITY');
      assert.equal(result.actualObservedEndFrame,128);assert.equal(result.firstFrame,0);
      assert.equal(result.processedSamples,128);assert.equal(result.processedBlocks,1);
      assert.equal(result.discontinuity,true);assert.equal(result.overflow,false);
      assert.deepEqual([...new Uint32Array(result.blockLengths).subarray(0,2)],[128,0]);
      const pcm=new Float32Array(result.pcm);assert(pcm.subarray(0,128).every(value=>value===.125));
      assert.equal(pcm[128],0);
    }
  `));

  it('terminates unqualified on capacity overflow, including a single oversized block', () => check(`
    const capacity=48000*80;
    for(const huge of [false,true]){
      const recorder=fixture(capacity+1);
      if(!huge){recorder.process(0,[[]],[[new Float32Array(capacity)]]);
        assert(!recorder.messages.some(row=>row.type==='render-recording'));}
      recorder.process(huge?0:capacity,[[]],[[new Float32Array(huge?capacity+1:1)]]);
      const result=terminal(recorder,'OVERFLOW');
      assert.equal(result.overflow,true);assert.equal(result.discontinuity,false);
      assert.equal(result.requestedEndFrame,capacity+1);
      assert.equal(result.actualObservedEndFrame,huge?null:capacity);
      assert.equal(result.firstFrame,huge?null:0);
      assert.equal(result.processedSamples,huge?0:capacity);assert.equal(result.processedBlocks,huge?0:1);
      assert.equal(result.pcm.byteLength,capacity*4);
      assert.deepEqual([...new Uint32Array(result.blockLengths).subarray(0,2)],[huge?0:capacity,0]);
      const before=structuredClone(recorder.messages);recorder.send('finish');recorder.step(capacity+1);
      assert.deepEqual(recorder.messages,before);
    }
  `));

  it.each([
    { label: 'short mono input', inputs: [64], outputs: [128] },
    { label: 'oversized mono input', inputs: [128], outputs: [64] },
    { label: 'nonempty input with zero-length output', inputs: [128], outputs: [0] },
    { label: 'inconsistent input channel lengths', inputs: [64, 32], outputs: [64, 64] },
    { label: 'inconsistent output channel lengths', inputs: [], outputs: [64, 32] },
  ])('returns MALFORMED_RESULT for $label without throwing or counting the bad block', ({ inputs, outputs }) => { check(`
    const recorder=fixture(192);recorder.step(0);
    const inputs=${JSON.stringify(inputs)}.map(length=>new Float32Array(length).fill(.25));
    const outputs=${JSON.stringify(outputs)}.map(length=>new Float32Array(length));
    assert.doesNotThrow(()=>recorder.process(128,[inputs],[outputs]));
    const result=terminal(recorder,'MALFORMED_RESULT');
    assert.equal(result.actualObservedEndFrame,128);assert.equal(result.processedSamples,128);
    assert.equal(result.processedBlocks,1);assert.equal(result.discontinuity,false);assert.equal(result.overflow,false);
    assert.deepEqual([...new Uint32Array(result.blockLengths).subarray(0,2)],[128,0]);
    assert.equal(new Float32Array(result.pcm)[128],0);
    const before=structuredClone(recorder.messages);recorder.send('finish');recorder.step(128);
    assert.deepEqual(recorder.messages,before);
  `); });
});