import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

describe('M11 diagnostic pipeline ownership', () => {
  it('resets frame identity on replacement and retains partial evidence on device loss', () => {
    execFileSync(process.execPath, ['--input-type=module', '-e', `
      import assert from 'node:assert/strict';import {build} from 'esbuild';import{runInNewContext}from'node:vm';
      let now=1000,nextFrame=0;const raf=new Map(),timers=new Set(),probes=[];
      class Pipeline{constructor(gpu,video,canvas,stage){this.currentUpscaler=stage;this.timingGeneration=1;}setUpscaler(next){this.currentUpscaler=next;}async drainTimings(){}}
      class Probe{encoded=0;constructor(inner,before){this.inner=inner;this.before=before;probes.push(this);}async readAfterPause(){const data=new Uint32Array(4+this.encoded*4);data[0]=this.encoded;for(let index=0;index<this.encoded;index++){data[4+index*4]=index+30;data[5+index*4]=1;}return data;}}
      const compiled=await build({entryPoints:['tools/m11/renderer-diagnostics.ts'],bundle:true,write:false,format:'iife',globalName:'module',plugins:[{name:'mock',setup(plugin){
        plugin.onResolve({filter:/core\\/pipeline\\.js|m1010r\\/probe\\.js/},args=>({path:args.path,namespace:'mock'}));
        plugin.onLoad({filter:/.*/,namespace:'mock'},()=>({contents:'export const VideoPipeline=globalThis.Pipeline;export const IdentityProbe=globalThis.Probe;'}));}}]});
      const sandbox={Pipeline,Probe,performance:{now:()=>now},document:{visibilityState:'visible'},
        requestAnimationFrame:callback=>{raf.set(++nextFrame,callback);return nextFrame;},cancelAnimationFrame:handle=>raf.delete(handle),
        setInterval:callback=>{timers.add(callback);return callback;},clearInterval:callback=>timers.delete(callback)};
      runInNewContext(compiled.outputFiles[0].text,sandbox);
      const video={currentTime:1},canvas={hidden:false};const diagnostics=new sandbox.module.DesktopDiagnostics(video,canvas);
      let driver;const session={get runtime(){return driver;},pause(){},snapshot(){return{state:driver?'paused':'unavailable'};}};diagnostics.bind(()=>session);
      let popped=0;const gpu={device:{pushErrorScope(){},async popErrorScope(){popped++;return null;},queue:{async onSubmittedWorkDone(){}}}};
      function start(){const options=diagnostics.options(true),pipeline=options.createPipeline(gpu,video,canvas,{neural:true},{});driver={pipeline,onSample:null,snapshot(){return{actualTier:'neural',controller:{state:'stable',reason:'test'}};}};return options;}
      function frame(options){const probe=probes.at(-1);probe.before(probe.encoded++);now++;options.onFrame({mediaTime:1,presentationTime:now-2,expectedDisplayTime:now+1},driver);}
      let options=start();frame(options);const staleRaf=[...raf.values()][0];diagnostics.begin();frame(options);const staleSample=driver.onSample;
      now+=10;options=start();assert.equal(timers.size,0);assert.equal(raf.size,0);staleRaf();staleSample({sequence:99});
      diagnostics.begin();frame(options);driver.onSample({sequence:1,generation:1,neural:true,ms:1});for(const [handle,callback] of [...raf]){raf.delete(handle);callback();}
      let record=await diagnostics.finish();assert.equal(record.rows.length,1);assert.equal(record.rows[0].sequence,1);assert.equal(record.rows[0].probe,0);
      assert.equal(record.rows[0].sourceIdentity,30);assert.equal(record.samples.length,1);assert.equal(record.samples[0].sequence,1);assert.equal(record.errors.length,0);
      assert.equal(timers.size,0);assert.equal(raf.size,0);
      now+=10;options=start();diagnostics.begin();frame(options);driver=null;record=await diagnostics.finish();
      assert.equal(record.rows.length,1);assert(record.errors.some(error=>error.includes('partial evidence')));assert.equal(record.snapshot.state,'unavailable');
      assert.equal(record.rows[0].identityValid,false);assert.equal(timers.size,0);assert.equal(raf.size,0);assert.equal(popped,9);
    `], { encoding: 'utf8', timeout: 15000 });
  });
});

function check(source: string) {
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { analyzePlayback } from './tools/m11/metrics.mjs';
    const options = {fps:60, requireNeural:true, minDurationMs:60000};
    const fixture = ({fps=60, seconds=60, startAt=1000, neural=true}={}) => {
      const rows = Array.from({length:fps*seconds}, (_,index) => {
        const submittedAt=startAt+index*1000/fps;
        return {sequence:index+1, probe:0, index, neural, generation:1, before:submittedAt-1,
          submittedAt, opportunityAt:submittedAt+2, mediaTime:index/fps,
          presentationTime:submittedAt-2, expectedDisplayTime:submittedAt+1, currentTime:index/fps,
          sourceIdentity:index, identityValid:true, superseded:false, canvasVisible:true, visibility:'visible'};
      });
      return {startAt, stopAt:startAt+seconds*1000, durationMs:seconds*1000, rows,
        samples:rows.map(row=>({sequence:row.sequence, generation:1, neural, upscalerId:neural?'production':'catmull-rom',
          submittedAt:row.submittedAt-1.5, resolvedAt:row.submittedAt+3, ms:4})),
        states:Array.from({length:seconds*10},(_,index)=>({at:startAt+index*100, generation:1,
          actualTier:neural?'neural':'baseline', state:neural?'stable':'manual-baseline', reason:'synthetic'})),
        errors:[], snapshot:{state:'paused', error:null, observerError:null, cleanupErrors:[],
          video:{error:null}, runtime:{controller:{state:'suspended'}, session:{decoderDrops:7, framesSkipped:3}},
          resources:{devices:1, pipelines:1}}};
    };
    ${source}
    console.log('checked');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 60000 });
  expect(output.trim()).toBe('checked');
}

describe('M11 software playback metrics (synthetic observations, not native measurements)', () => {
  it('counts the opening event without an anchor subtraction and preserves actual window duration', () => check(`
    const record=fixture(), original=structuredClone(record), result=analyzePlayback(record,options);
    assert.equal(result.outcome,'PASS',result.reason);
    assert.equal(result.metrics.counts.usefulSubmissions,3600);
    assert.equal(result.metrics.windows.overall.submittedFps,60);
    assert.equal(result.metrics.windows.final20s.renderedFps,60);
    assert.equal(result.metrics.windows.first20s.usefulSubmissions,1200);
    assert.equal(result.metrics.windows.overall.gpuMs.count,3599);
    assert.equal(result.metrics.decodeDropped,null);
    assert.equal(result.metrics.snapshotCounters.decoderDrops,7);
    assert.deepEqual(record,original);
    record.durationMs=90000;
    assert.equal(analyzePlayback(record,options).metrics.durationMs,60000);
    record.stopAt+=1000;
    const longer=analyzePlayback(record,options);
    assert.equal(longer.metrics.windows.overall.submittedFps,3600/61);
    assert.equal(longer.criteria['final20s.submittedFps'].pass,false);
  `));

  it('qualifies 30fps and an explicitly instrumented baseline without pretending it is native raw video', () => check(`
    for(const neural of [true,false]) {
      const result=analyzePlayback(fixture({fps:30,neural}),{fps:30,requireNeural:neural});
      assert.equal(result.outcome,'PASS',result.reason);
      assert.equal(result.metrics.minimumFps,29);
      assert.equal(result.metrics.windows.overall.renderedFps,30);
    }
    assert.equal(analyzePlayback(fixture({neural:false}),options).criteria.controller.pass,false);
  `));

  it('reports missing distributions as null, never as zero latency or a passing empty run', () => check(`
    const result=analyzePlayback({startAt:0,stopAt:60000,durationMs:60000,rows:[],samples:[],states:[],errors:[]},options);
    assert.equal(result.outcome,'FAIL');
    assert.equal(result.metrics.invalidFraction,null);
    for(const metric of ['sourceCallbackAgeMs','encodeSubmissionMs','opportunityDelayMs','opportunityVisualAgeMs','gpuMs','callbackLatencyMs']) {
      assert.equal(result.metrics.windows.overall[metric].p50,null);
      assert.equal(result.metrics.windows.overall[metric].p95,null);
      assert.equal(result.metrics.windows.overall[metric].max,null);
    }
    assert.equal(result.criteria.controllerCoverage.pass,false);
    assert.equal(result.criteria.gpuEvidence.pass,false);
    assert.equal(result.metrics.resources,null);
  `));

  it('uses half-open submission boundaries, includes zero identities and censors terminal pending opportunities', () => check(`
    const record=fixture({startAt:0}); record.rows[0].before=0;
    record.rows.unshift({...record.rows[0],submittedAt:-1,identityValid:false,neural:false});
    record.rows.push({...record.rows.at(-1),submittedAt:record.stopAt,identityValid:false,neural:false});
    record.states.unshift({at:-1,actualTier:'baseline',state:'fallback',generation:0});
    record.states.push({at:record.stopAt,actualTier:'baseline',state:'probing',generation:2});
    let result=analyzePlayback(record,options);
    assert.equal(result.outcome,'PASS',result.reason);
    assert.equal(result.metrics.counts.outsideWindow,2);
    assert.equal(result.metrics.counts.observations,3600);
    const last=record.rows.at(-2); last.opportunityAt=record.stopAt;
    record.states.at(-1).generation=1;
    result=analyzePlayback(record,options);
    assert.equal(result.metrics.counts.renderedOpportunities,3600);
    record.states.at(-1).generation=2;
    result=analyzePlayback(record,options);
    assert.equal(result.criteria.freshness.pass,false);
    assert.equal(result.metrics.counts.renderedOpportunities,3599);
    record.states.at(-1).generation=1;
    record.rows.at(-1).sequence=3601;
    result=analyzePlayback(record,options);
    assert.equal(result.criteria.freshness.pass,false);
    assert.equal(result.metrics.counts.usefulSubmissions,3600);
    last.opportunityAt=record.stopAt+0.01;
    result=analyzePlayback(record,options);
    assert.equal(result.metrics.counts.usefulSubmissions,3600);
    assert.equal(result.metrics.counts.renderedOpportunities,3599);
    assert.equal(result.metrics.counts.afterStopOpportunities,1);
    assert.equal(result.metrics.observations.at(-1).opportunityDelayMs,null);
    assert.ok(result.metrics.observations.at(-1).rawOpportunityDelayMs>0);
    assert.equal(result.outcome,'PASS',result.reason);
    last.opportunityAt=null;
    assert.equal(analyzePlayback(record,options).metrics.counts.missingOpportunities,1);
  `));

  it('assigns subwindow rendered cadence to opportunity timestamps, not submission timestamps', () => check(`
    const record=fixture();
    record.rows[1199].opportunityAt=record.startAt+20001;
    Object.assign(record.rows[1200],{submittedAt:record.startAt+20002,before:record.startAt+20001,opportunityAt:record.startAt+20004});
    const result=analyzePlayback(record,options);
    assert.equal(result.metrics.windows.first20s.usefulSubmissions,1200);
    assert.equal(result.metrics.windows.first20s.renderedOpportunities,1199);
    assert.equal(result.metrics.windows.middle20s.renderedOpportunities,1201);
    assert.equal(result.metrics.counts.stalePresentations,0);
  `));

  it('fails the final20s independently even when overall throughput passes', () => check(`
    const record=fixture();
    record.rows=record.rows.filter((row,index)=>index<2400 || index%20!==0);
    const result=analyzePlayback(record,options);
    assert.equal(result.metrics.windows.overall.submittedFps,59);
    assert.equal(result.metrics.windows.final20s.submittedFps,57);
    assert.equal(result.criteria['overall.submittedFps'].pass,true);
    assert.equal(result.criteria['overall.renderedFps'].pass,true);
    assert.equal(result.criteria['final20s.submittedFps'].pass,false);
    assert.equal(result.criteria['final20s.renderedFps'].pass,false);
    assert.equal(result.outcome,'FAIL');
    assert.ok(result.metrics.counts.submissionSequenceGaps>0);
    assert.ok(result.metrics.counts.mediaPtsGapFrames>0);
    assert.ok(result.metrics.counts.sourceIdentityGapFrames>0);
  `));

  it('never substitutes currentTime for missing native metadata or tolerates ambiguous identities', () => check(`
    for(const change of [{identityValid:false},{sourceIdentity:null},{sourceIdentity:0.5},{sourceIdentity:-1},
      {mediaTime:null},{mediaTime:NaN},{mediaTime:Infinity},{mediaTime:undefined}]) {
      const record=fixture(); Object.assign(record.rows[10],change);
      const result=analyzePlayback(record,options);
      assert.equal(result.criteria.identity.pass,false,JSON.stringify(change));
      assert.equal(result.metrics.counts.usefulSubmissions,3599);
      assert.equal(result.metrics.observations[10].sourceCallbackAgeMs,null);
      assert.equal(result.outcome,'FAIL');
    }
    const record=fixture();record.rows.at(-1).mediaTime=Number.MAX_VALUE;
    const result=analyzePlayback(record,options);
    assert.equal(result.criteria.sourceAge.pass,false);
    assert.equal(result.metrics.observations.at(-1).sourceCallbackAgeMs,null);
  `));

  it('uses exact numbered PTS and only the 0.001ms source timestamp quantization tolerance in both directions', () => check(`
    for(const fps of [30,60]) for(const direction of [-1,1]) {
      const record=fixture({fps});
      for(const row of record.rows){row.sourceIdentity+=100;row.mediaTime=row.sourceIdentity/fps+direction*(1000/fps+0.0005)/1000;}
      const result=analyzePlayback(record,{fps,requireNeural:true});
      assert.equal(result.criteria.sourceAge.pass,true);
      assert.equal(result.metrics.counts.usefulSubmissions,record.rows.length);
      for(const row of record.rows) row.mediaTime=row.sourceIdentity/fps+direction*(1000/fps+0.002)/1000;
      assert.equal(analyzePlayback(record,{fps,requireNeural:true}).criteria.sourceAge.pass,false);
    }
  `));

  it('counts exactly 1% invalid opportunities without rounding and fails above it', () => check(`
    const record=fixture();
    for(const row of record.rows.slice(0,36)) row.opportunityAt=null;
    let result=analyzePlayback(record,options);
    assert.equal(result.metrics.invalidFraction,0.01);
    assert.equal(result.criteria.invalidFraction.pass,true);
    assert.equal(result.metrics.windows.overall.opportunityDelayMs.missing,36);
    record.rows[36].opportunityAt=null;
    result=analyzePlayback(record,options);
    assert.equal(result.metrics.counts.invalidObservations,37);
    assert.equal(result.criteria.invalidFraction.pass,false);
    for(const row of record.rows)row.opportunityAt=null;
    result=analyzePlayback(record,options);
    assert.equal(result.metrics.windows.overall.opportunityDelayMs.p95,null);
    assert.equal(result.metrics.windows.overall.opportunityVisualAgeMs.p50,null);
    assert.equal(result.criteria.ageIncrease.pass,false);
  `));

  it('does not count duplicate sequence or numbered texture identity as useful new output', () => check(`
    const record=fixture();record.rows.splice(10,0,{...record.rows[10]});
    let result=analyzePlayback(record,options);
    assert.equal(result.metrics.counts.duplicateSequences,1);
    assert.equal(result.metrics.counts.usefulSubmissions,3600);
    assert.equal(result.metrics.counts.renderedOpportunities,3600);
    assert.equal(result.metrics.counts.stalePresentations,0);
    assert.equal(result.outcome,'PASS',result.reason);
    const repeat=fixture(); repeat.rows[10].sourceIdentity=repeat.rows[9].sourceIdentity;
    result=analyzePlayback(repeat,options);
    assert.equal(result.metrics.counts.duplicateIdentities,1);
    assert.equal(result.metrics.counts.usefulSubmissions,3599);
    assert.equal(result.metrics.counts.renderedOpportunities,3599);
    assert.equal(result.metrics.observations[10].usefulSubmission,false);
    assert.match(result.metrics.scope.controls,/texture bit-repeat/);
  `));

  it('rejects superseded, hidden, old-generation and falsely latest presentation claims', () => check(`
    for(const patch of [{superseded:true},{canvasVisible:false},{visibility:'hidden'}]) {
      const record=fixture();Object.assign(record.rows[10],patch);
      const result=analyzePlayback(record,options);
      assert.equal(result.metrics.counts.usefulSubmissions,3600);
      assert.equal(result.metrics.counts.renderedOpportunities,3599);
    }
    const jumped=fixture();jumped.rows[10].opportunityAt=jumped.rows[11].submittedAt+1;
    let result=analyzePlayback(jumped,options);
    assert.equal(result.criteria.freshness.pass,false);
    assert.equal(result.metrics.counts.stalePresentations,1);
    jumped.rows[10].superseded=true;
    result=analyzePlayback(jumped,options);
    assert.equal(result.metrics.counts.stalePresentations,0);
    assert.equal(result.metrics.counts.renderedOpportunities,3599);
    const stale=fixture();stale.rows[10].generation=0;
    assert.equal(analyzePlayback(stale,options).criteria.freshness.pass,false);
    const reset=fixture();reset.rows[10].sequence=1;
    assert.equal(analyzePlayback(reset,options).criteria.freshness.pass,false);
    const reverse=fixture();reverse.rows[10].sourceIdentity=8;
    assert.equal(analyzePlayback(reverse,options).metrics.counts.identityRegressions,1);
  `));

  it('allows forward generations between state polls but rejects a generation invalidated before opportunity', () => check(`
    const record=fixture();
    for(const row of record.rows.slice(3))row.generation=2;
    for(const sample of record.samples.slice(3))sample.generation=2;
    for(const state of record.states.slice(1))state.generation=2;
    let result=analyzePlayback(record,options);
    assert.equal(result.outcome,'PASS',result.reason);
    assert.equal(result.metrics.windows.overall.gpuMs.count,3599);
    const changed=fixture(), row=changed.rows[10];
    changed.states.splice(2,0,{...changed.states[0],at:row.submittedAt+1,generation:2});
    result=analyzePlayback(changed,options);
    assert.equal(result.criteria.freshness.pass,false);
    assert.ok(result.metrics.counts.stalePresentations>0);
    assert.equal(result.criteria.controllerCoverage.pass,false);
  `));

  it('keeps GPU timestamps separate from software brackets, next-rAF delay and callback latency', () => check(`
    const record=fixture();
    for(const row of record.rows){row.before=row.submittedAt-2;row.opportunityAt=row.submittedAt+3;row.presentationTime=row.before-7;row.mediaTime+=0.004;}
    for(const sample of record.samples){sample.ms=123;sample.resolvedAt=record.stopAt+1000;}
    const result=analyzePlayback(record,options), overall=result.metrics.windows.overall;
    assert.equal(result.outcome,'PASS',result.reason);
    assert.equal(overall.gpuMs.p50,123);
    assert.equal(overall.gpuMs.p95,123);
    assert.equal(overall.encodeSubmissionMs.p95,2);
    assert.equal(overall.opportunityDelayMs.p95,3);
    assert.equal(overall.elapsedToOpportunityMs.p50,5);
    assert.ok(Math.abs(overall.sourceCallbackAgeMs.p50-4)<1e-8);
    assert.ok(Math.abs(overall.opportunityVisualAgeMs.p50-9)<1e-8);
    assert.equal(overall.callbackLatencyMs.p50,7);
    record.rows[10].presentationTime=null;
    assert.equal(analyzePlayback(record,options).metrics.observations[10].callbackLatencyMs,null);
  `));

  it('bounds GPU membership by sample pre-encode time, verifies sequence/generation, and deduplicates samples', () => check(`
    const record=fixture(), last=record.rows.at(-1);
    record.rows.push({...last,sequence:3601,submittedAt:record.stopAt,sourceIdentity:3600});
    const good={...record.samples[10],sequence:3601,submittedAt:record.stopAt-1,resolvedAt:record.stopAt+100,ms:9};
    record.samples=[good,{...good},{...good,sequence:9999},{...good,generation:0},
      {...good,neural:false},{...good,ms:NaN},{...good,ms:-1},{...good,resolvedAt:null},
      {...good,submittedAt:record.startAt-1},{...good,submittedAt:record.stopAt}];
    let result=analyzePlayback(record,options);
    assert.equal(result.metrics.windows.overall.gpuMs.count,1);
    assert.equal(result.metrics.windows.overall.gpuMs.p95,9);
    assert.equal(result.metrics.gpuSamples.inWindow,8);
    assert.equal(result.criteria.gpuEvidence.pass,true);
    record.samples=[{...good,neural:false}];
    result=analyzePlayback(record,options);
    assert.equal(result.metrics.windows.overall.gpuMs.p95,null);
    assert.equal(result.criteria.gpuEvidence.pass,false);
  `));

  it('checks encode p95, opportunity p95 and maximum against the registered limits', () => check(`
    const encode=fixture();for(const row of encode.rows)row.before=row.submittedAt-17;
    assert.equal(analyzePlayback(encode,options).criteria.encodeSubmission.pass,false);
    const slow=fixture();for(const row of slow.rows)row.opportunityAt=row.submittedAt+34;
    let result=analyzePlayback(slow,options);
    assert.equal(result.criteria.opportunityP95.pass,false);
    const spike=fixture();spike.rows[10].opportunityAt=spike.rows[10].submittedAt+250;
    result=analyzePlayback(spike,options);
    assert.equal(result.criteria.opportunityMax.pass,true);
    spike.rows[10].opportunityAt+=0.001;
    assert.equal(analyzePlayback(spike,options).criteria.opportunityMax.pass,false);
  `));

  it('fails three increasing 5s medians even if first20s/final20s pooled medians conceal the growth', () => check(`
    const record=fixture();
    for(const row of record.rows){
      const offset=row.submittedAt-record.startAt;
      const level=offset>=25000&&offset<30000?1:offset>=30000&&offset<35000?2:0;
      row.before=row.submittedAt-1-level*20;
    }
    const result=analyzePlayback(record,options);
    assert.equal(result.criteria.ageIncrease.pass,true);
    assert.equal(result.criteria.ageGrowth.pass,false);
    assert.deepEqual(result.metrics.growingWindows[0],[record.startAt+20000,record.startAt+25000,record.startAt+30000]);
    const two=fixture();for(const row of two.rows)if(row.submittedAt-two.startAt>=25000&&row.submittedAt-two.startAt<30000)row.before-=20;
    assert.equal(analyzePlayback(two,options).criteria.ageGrowth.pass,true);
  `));

  it('uses signed final-minus-first age increase and does not turn decreasing age into a failure', () => check(`
    for(const increasing of [true,false]) {
      const record=fixture();
      for(const row of record.rows){
        const offset=row.submittedAt-record.startAt;
        const fraction=Math.min(1,Math.max(0,(offset-20000)/20000));
        row.before-=40*(increasing?fraction:1-fraction);
      }
      const result=analyzePlayback(record,options);
      assert.ok(Math.abs(result.metrics.ageIncreaseMs-(increasing?40:-40))<1e-8);
      assert.equal(result.criteria.ageIncrease.pass,!increasing);
      assert.equal(result.criteria.ageGrowth.pass,true);
    }
  `));

  it('checks actual controller state names, all neural rows, errors and final enhancement failure', () => check(`
    for(const state of ['fallback','probing','failed','unavailable','manual-baseline','probe']) {
      const record=fixture();record.states[100].state=state;
      assert.equal(analyzePlayback(record,options).criteria.controller.pass,false,state);
    }
    const baseline=fixture();baseline.states[100].actualTier='baseline';
    assert.equal(analyzePlayback(baseline,options).criteria.controller.pass,false);
    const mixed=fixture();mixed.rows[10].neural=false;
    const mixedResult=analyzePlayback(mixed,options);
    assert.equal(mixedResult.criteria.controller.pass,false);
    assert.equal(mixedResult.metrics.counts.usefulSubmissions,3599);
    for(const mutate of [record=>record.errors.push('GPU error'),record=>record.errors.push(NaN),
      record=>record.snapshot.error='failure',record=>record.snapshot.observerError='observer failed',
      record=>record.snapshot.video.error={code:3},record=>record.snapshot.cleanupErrors.push('cleanup failed'),
      record=>record.snapshot.enhancementFailed=true,record=>record.snapshot.state='unavailable',
      record=>record.snapshot.runtime.controller.state='failed']) {
      const record=fixture();mutate(record);
      assert.equal(analyzePlayback(record,options).criteria.errors.pass,false);
    }
  `));

  it('retains state sampling holes without an unregistered interval threshold, but fails missing or unordered coverage', () => check(`
    const record=fixture();record.states=[record.states[0],record.states.at(-1)];
    const result=analyzePlayback(record,options);
    assert.equal(result.outcome,'PASS',result.reason);
    assert.equal(result.metrics.stateCoverage.maxGapMs,59900);
    record.states=[];
    assert.equal(analyzePlayback(record,options).criteria.controllerCoverage.pass,false);
    const reversed=fixture();reversed.states.reverse();
    assert.equal(analyzePlayback(reversed,options).criteria.controllerCoverage.pass,false);
    const missing=fixture();missing.states[0].at=null;
    assert.equal(analyzePlayback(missing,options).criteria.controllerCoverage.pass,false);
  `));

  it('rejects missing/nonfinite/regressing mandatory clocks and malformed options without coercion', () => check(`
    for(const before of [null,undefined,NaN,Infinity,-1]) {
      const record=fixture();record.rows[10].before=before;
      assert.equal(analyzePlayback(record,options).criteria.clocks.pass,false);
    }
    const backwards=fixture();backwards.rows[10].before=backwards.rows[9].before-1;
    assert.equal(analyzePlayback(backwards,options).criteria.clocks.pass,false);
    const badOpportunity=fixture();badOpportunity.rows[10].opportunityAt=badOpportunity.rows[10].submittedAt-1;
    assert.equal(analyzePlayback(badOpportunity,options).criteria.clocks.pass,false);
    const unordered=fixture();[unordered.rows[9],unordered.rows[10]]=[unordered.rows[10],unordered.rows[9]];
    assert.equal(analyzePlayback(unordered,options).criteria.clocks.pass,false);
    const absent=fixture();absent.rows[10].submittedAt=null;
    assert.equal(analyzePlayback(absent,options).criteria.rowCoverage.pass,false);
    for(const stopAt of [null,NaN,Infinity,1000,999]) {
      assert.equal(analyzePlayback({...fixture(),stopAt},options).criteria.duration.pass,false);
    }
    assert.equal(analyzePlayback(null,options).outcome,'FAIL');
    for(const fps of [0,24,'60',NaN])assert.throws(()=>analyzePlayback(fixture(),{...options,fps}),RangeError);
    assert.throws(()=>analyzePlayback(fixture(),{...options,requireNeural:undefined}),TypeError);
    assert.throws(()=>analyzePlayback(fixture(),{...options,minDurationMs:59999}),RangeError);
  `));

  it('supports the 600s soak with independent first/middle/final20s evidence and unchanged gates', () => check(`
    const record=fixture({seconds:600}), soak={...options,minDurationMs:600000};
    let result=analyzePlayback(record,soak);
    assert.equal(result.outcome,'PASS',result.reason);
    assert.equal(result.metrics.durationMs,600000);
    assert.equal(result.metrics.counts.usefulSubmissions,36000);
    assert.equal(result.metrics.fiveSecondWindows.length,120);
    for(const name of ['first20s','middle20s','final20s']) {
      assert.equal(result.metrics.windows[name].durationMs,20000);
      assert.equal(result.metrics.windows[name].usefulSubmissions,1200);
      assert.equal(result.metrics.windows[name].gpuMs.p95,4);
    }
    assert.equal(result.metrics.windows.middle20s.startAt,record.startAt+290000);
    assert.equal(analyzePlayback(fixture(),soak).criteria.duration.pass,false);
    record.rows=record.rows.filter((row,index)=>index<34800 || index%20!==0);
    result=analyzePlayback(record,soak);
    assert.equal(result.criteria['overall.submittedFps'].pass,true);
    assert.equal(result.criteria['final20s.submittedFps'].pass,false);
    assert.equal(result.outcome,'FAIL');
  `));
});