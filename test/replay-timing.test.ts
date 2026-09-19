import { describe, it } from 'vitest';
import { execFileSync } from 'node:child_process';

const check = (body: string): void => { execFileSync(process.execPath, ['--input-type=module', '-e', `
  import assert from 'node:assert/strict';
  import {readFileSync,mkdtempSync,mkdirSync,rmSync} from 'node:fs';
  import {signalPcm,replayRecipe,decodeCounter,audioTimeline,prepareReplayMedia,inspectShiftControls,COUNTER_WIDTH} from './tools/m1010r/media.mjs';
  ${body}
`], { encoding: 'utf8' }); };

describe('RI batched artifact guard', () => {
  const guardCheck = (body: string) => check(`
    import {writeFileSync,symlinkSync} from 'node:fs';
    import {tmpdir} from 'node:os';
    import {join,resolve} from 'node:path';
    import {pathToFileURL} from 'node:url';
    import {execFileSync} from 'node:child_process';
    import {createRequire} from 'node:module';
    import {transformSync} from 'esbuild';
    const root=mkdtempSync(join(tmpdir(),'ri-guard-'));
    const calls=[];let failGit=false,gitResponse=null;
    try {
      const environment={...process.env,GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:join(root,'empty-config')};
      for(const key of Object.keys(environment))if(key.startsWith('GIT_')&&!['GIT_CONFIG_NOSYSTEM','GIT_CONFIG_GLOBAL'].includes(key))delete environment[key];
      writeFileSync(environment.GIT_CONFIG_GLOBAL,'');
      execFileSync('git',['init','--quiet',root],{env:environment});
      mkdirSync(join(root,'.cache/m1010r'),{recursive:true});
      writeFileSync(join(root,'.gitignore'),'.cache/m1010r/*\\n!.cache/m1010r/unignored.json\\n');
      const source=readFileSync('tools/m1010r/build.mjs','utf8');
      const compiled=transformSync(source.slice(0,source.indexOf('\\nif (process.argv[1]')),{format:'cjs',define:{'import.meta.url':JSON.stringify(pathToFileURL(join(root,'tools/m1010r/build.mjs')).href)}}).code;
      const require=createRequire(process.cwd()+'/package.json'),module={exports:{}};
      new Function('require','module','exports','process',compiled)(name=>{
        if(name==='../build-extension.mjs')return {MODEL_SHA256:'synthetic',verifyProductionModel(){}};
        if(name==='node:child_process')return {execFileSync(file,args,options){
          calls.push({file,args,input:options?.input});
          if(failGit)throw Object.assign(Error('Git check failed'),{status:128});
          if(gitResponse!==null)return gitResponse;
          return execFileSync(file,args,{...options,env:environment});
        }};
        return require(name);
      },module,module.exports,{argv:[]});
      const {cachePath,cachePaths,verifyReference,digest}=module.exports;
      const ignored='.cache/m1010r/ignored.json',other='.cache/m1010r/other.json',unignored='.cache/m1010r/unignored.json';
      ${body}
    } finally {rmSync(root,{recursive:true,force:true});}
  `);

  it('batches every distinct artifact without accepting a partially ignored set', () => guardCheck(`
    const paths=[ignored,other,'.cache/m1010r/nested/evidence.json'];
    for(const path of paths)execFileSync('git',['check-ignore','--quiet','--',path],{cwd:root,env:environment});
    const before=calls.length;
    assert.deepEqual(cachePaths(paths),paths.map(path=>resolve(root,path)));
    assert.equal(calls.length-before,1,'One Git process for a known artifact set');
    assert.equal(calls.at(-1).input,paths.join('\\0')+'\\0');
    assert.throws(()=>cachePaths([ignored,unignored]),/ignored/i);
    assert.equal(cachePath(ignored),resolve(root,ignored));
    assert.throws(()=>cachePath(unignored));
  `));

  it('rechecks policy and Git failures without reusing another path result', () => guardCheck(`
    assert.equal(cachePath(ignored),resolve(root,ignored));
    writeFileSync(join(root,'.gitignore'),'.cache/m1010r/*\\n!.cache/m1010r/ignored.json\\n');
    assert.throws(()=>cachePath(ignored));
    assert.equal(cachePath(other),resolve(root,other));
    failGit=true;
    assert.throws(()=>cachePath(other),/Git check failed/);
    assert.throws(()=>cachePaths([other,unignored]),/Git check failed/);
    failGit=false;
    assert.equal(cachePath(other),resolve(root,other));
  `));

  it('preserves path and provenance rejection before accepting batched artifacts', () => guardCheck(`
    const before=calls.length;
    for(const path of ['../escape',root,'.cache/m1010r','.cache/m1010r/../../outside',ignored+'\\0'+other,null,42])assert.throws(()=>cachePaths([ignored,path]));
    assert.equal(calls.length,before,'Unsafe sets must not reach Git');
    mkdirSync(join(root,'.cache/m1010r/target'));
    symlinkSync(join(root,'.cache/m1010r/target'),join(root,'.cache/m1010r/alias'),'dir');
    assert.throws(()=>cachePaths([ignored,'.cache/m1010r/alias/evidence.json']),/Symlinked/);
    writeFileSync(join(root,ignored),'evidence');
    const reference={path:ignored,bytes:8,sha256:digest(Buffer.from('evidence'))};
    assert.equal(verifyReference(reference).toString(),'evidence');
    for(const change of [{bytes:7},{sha256:'0'.repeat(64)},{path:unignored},{path:'../outside'},{bytes:-1}])assert.throws(()=>verifyReference({...reference,...change}));
  `));

  it('preserves tracked-file rules and rejects incomplete or malformed Git responses', () => guardCheck(`
    const before=calls.length;assert.deepEqual(cachePaths([]),[]);assert.equal(calls.length,before);
    assert.deepEqual(cachePaths([ignored,resolve(root,ignored)]),[resolve(root,ignored),resolve(root,ignored)]);
    assert.equal(calls.at(-1).input,ignored+'\\0');
    for(const output of ['',ignored,other+'\\0',ignored+'\\0'+other+'\\0']){
      gitResponse=output;assert.throws(()=>cachePath(ignored));
    }
    gitResponse=null;
    writeFileSync(join(root,ignored),'tracked');
    execFileSync('git',['add','-f','--',ignored],{cwd:root,env:environment});
    assert.throws(()=>cachePaths([other,ignored]),/ignored/i);
    const spaced='.cache/m1010r/space and\\nnewline.json';
    assert.equal(cachePath(spaced),resolve(root,spaced));
  `));
});

describe('M10.10R digital timing ground truth', () => {
  it('keeps RI partial render coverage distinct from target overshoot and discontinuity magnitude', () => check(`
    const {renderEvidence,INSTRUMENT_VERDICT}=await import('./tools/m1010r/instrument_report.mjs');
    const render={terminal:{completionReason:'DISCONTINUITY',sampleRate:48000,firstFrame:256,actualObservedEndFrame:512,
      requestedEndFrame:2000,processedSamples:256,processedBlocks:2,blockLengths:[128,128],heartbeatCount:1,
      overflow:false,discontinuity:true},heartbeats:[{}],watchdogFired:false,timedOut:false,errors:[],contextEvents:[{state:'running'}]};
    const result=renderEvidence(render,Buffer.alloc(1024));
    assert.equal(result.completionOvershootFrames,null);assert.equal(result.discontinuityMagnitudeFrames,null);
    assert.equal(result.offendingNextFrame,null);assert.equal(result.processedSamples,256);assert.equal(result.nonzeroSamples,0);
    assert.equal(INSTRUMENT_VERDICT,'DIGITAL TIMING INSTRUMENT NOT QUALIFIED');
    assert.throws(()=>renderEvidence(render,Buffer.alloc(4)));
    const reached=renderEvidence({...render,terminal:{...render.terminal,completionReason:'RENDER_TARGET_REACHED',requestedEndFrame:500,discontinuity:false}},Buffer.alloc(1024));
    assert.equal(reached.completionOvershootFrames,12);
  `));

  it('distinguishes retained silent sample coverage from scheduled timing observations', () => check(`
    const {controlDiagnosis,VERDICT}=await import('./tools/m1010r/report.mjs');
    const record={sampleRate:48000,firstFrame:0,samples:7168,
      scheduled:[{startFrame:12000,samples:8192,referenceStart:48000,gain:1}],
      expectedWindows:180,verifiedWindows:0,maximumSampleError:null,errors:Array(180).fill('unverified')};
    const result=controlDiagnosis(record,Buffer.alloc(7168*4));
    assert(result.endsBeforeFirstScheduledSignal);assert(!result.overlapWithScheduledSignals);
    assert.equal(result.nonzeroSamples,0);assert.equal(result.maximumSampleError,null);
    assert.equal(result.sampleCoverageMs,7168/48000*1000);
    assert.equal(VERDICT,'CONTROLLED REPLAY PATH NOT QUALIFIED');
    assert.throws(()=>controlDiagnosis(record,Buffer.alloc(4)));
    const later=controlDiagnosis({...record,firstFrame:12000},Buffer.alloc(7168*4));
    assert(later.overlapWithScheduledSignals);assert(!later.endsBeforeFirstScheduledSignal);
  `));

  it('preserves signed audio origins and rejects discontinuity or unaccounted decoded samples', () => check(`
    const stream={time_base:'1/48000',sample_rate:'48000'};
    for(const first of [-960,0,960]) {
      const result=audioTimeline([{pts:first,nb_samples:1024},{pts:first+1024,nb_samples:1024}],stream,2048);
      assert.equal(result.firstSampleMediaTime,first/48000);
      assert.equal(result.endExclusiveMediaTime,first/48000+2048/48000);
    }
    assert.throws(()=>audioTimeline([{pts:0,nb_samples:1024},{pts:1025,nb_samples:1024}],stream,2048),/Discontinuous/);
    assert.throws(()=>audioTimeline([{pts:0,nb_samples:1024}],stream,1023),/PCM length/);
    assert.throws(()=>audioTimeline([{nb_samples:1024}],stream,1024),/Missing audio PTS/);
  `));

  it('decodes all sixteen bits with independent guards and complements', () => check(`
    const rows=identity=>{
      const top=Buffer.alloc(COUNTER_WIDTH,16),bottom=Buffer.alloc(COUNTER_WIDTH,16);
      top[8]=bottom[8]=235;
      for(let bit=0;bit<16;bit++){const value=(identity>>bit)&1;top[(bit+2)*24+8]=value?235:16;bottom[(bit+2)*24+8]=value?16:235;}
      return [top,bottom];
    };
    for(const identity of [0,1,8191,8192,35999,65535]) assert.equal(decodeCounter(...rows(identity)),identity);
    const [top,bottom]=rows(12);bottom[8]=16;assert.throws(()=>decodeCounter(top,bottom),/guard/);
    bottom[8]=235;bottom[56]=top[56];assert.throws(()=>decodeCounter(top,bottom),/complement/);
    top[56]=120;assert.throws(()=>decodeCounter(top,bottom),/Ambiguous/);
  `));

  it('generates repeatable sample-addressed impulses and nonrepeating identity noise', () => check(`
    const first=signalPcm(2),second=signalPcm(2);assert.deepEqual(first,second);
    assert.equal(first.length,192000);
    for(const index of [12000,12047,60000,60047]) assert.equal(first.readInt16LE(index*2),24576);
    assert(Math.abs(first.readInt16LE(11999*2))<=2048);
    assert.notDeepEqual(first.subarray(0,256),first.subarray(96000,96256));
    assert.throws(()=>signalPcm(0),/duration/);
    const recipe=replayRecipe(60,'output.mp4','signal.s16le',70);
    assert(recipe.includes('s16le'));assert(recipe.includes('signal.s16le'));
    assert(recipe.some(value=>value.includes('bitand(n,32768)')));
    assert.throws(()=>replayRecipe(60,'output','signal',1100),/fixture/);
  `));

  it.skipIf(process.env['M1010R_FFMPEG'] !== '1')('validates actual muxed PTS, every frame and decoded audio for both rates', () => check(`
    mkdirSync('.cache/m1010r',{recursive:true});const directory=mkdtempSync('.cache/m1010r/unit-media-');
    try {
      const manifest=prepareReplayMedia(directory,2);
      assert.equal(manifest.assets.length,2);
      for(const asset of manifest.assets){assert.equal(asset.validation.video.frames,asset.fps*2);assert.equal(asset.validation.audio.impulses.length,2);assert.equal(asset.validation.audio.firstSampleMediaTime,0);}
      const shifts=inspectShiftControls(manifest,directory+'/shifts');assert.equal(shifts.length,4);
      for(const shift of shifts){assert(Math.abs(shift.observedAudioShiftSeconds-shift.requestedAudioShiftSeconds)<1/48000);assert.equal(shift.expectedVideoMinusAudioShiftMs,-shift.requestedAudioShiftSeconds*1000);assert.equal(shift.matchedInteriorSamples,8192);assert(shift.videoPtsUnchanged);}
      assert.deepEqual(prepareReplayMedia(directory,2),manifest);
      assert.throws(()=>prepareReplayMedia(directory,3),/pin changed/);
    } finally {rmSync(directory,{recursive:true,force:true});}
  `));
});

describe('M10.10R immutable calibration checkpoints', () => {
  const checkpointCheck = (body: string) => { check(`
    import {writeFileSync,readdirSync,existsSync,symlinkSync} from 'node:fs';
    import {openCalibrationCheckpoint,analyzeCalibration,CALIBRATION_CASES} from './tools/m1010r/study.mjs';
    import {digest} from './tools/m1010r/build.mjs';
    const directory='.cache/m1010r/unit-checkpoint-'+process.pid;
    const pin={studyVersion:'unit',sourceCommit:'source',buildSha256:'build',browserExecutableSha256:'browser',analyzerSha256:'analyzer'};
    const ids=CALIBRATION_CASES.map(entry=>entry.id);
    const artifact=path=>{const bytes=readFileSync(path);return {path,bytes:bytes.length,sha256:digest(bytes)};};
    const readJson=path=>JSON.parse(readFileSync(path,'utf8'));
    mkdirSync('.cache/m1010r',{recursive:true});mkdirSync(directory);
    try { ${body} } finally {rmSync(directory,{recursive:true,force:true});}
  `); };

  it('batches fresh checkpoint reads and materially reduces Git subprocesses', () => checkpointCheck(`
    const {default:childProcess}=await import('node:child_process');
    const {syncBuiltinESMExports}=await import('node:module');
    const {withRIRepeatability,RI_CASES,RI_STUDY_VERSION}=await import('./tools/m1010r/study.mjs');
    const execute=childProcess.execFileSync,calls=[];let failGit=false;
    childProcess.execFileSync=function(file,args,...options){
      if(file==='git'&&args[0]==='check-ignore'){
        calls.push(args);if(failGit)throw Error('Injected Git failure');
      }
      return execute.call(this,file,args,...options);
    };
    syncBuiltinESMExports();
    try {
      const riPin={...pin,studyVersion:RI_STUDY_VERSION},start=performance.now();
      const checkpoint=openCalibrationCheckpoint(directory,riPin);
      for(const entry of RI_CASES){checkpoint.begin(entry.id);checkpoint.raw(entry.id,{outcome:'RECORDED'});
        checkpoint.complete(entry.id,withRIRepeatability(entry.id,{outcome:'PASS',summary:{medianDigitalPhaseMs:0}},checkpoint.analyses()));}
      const reopenStart=calls.length;
      const reopened=openCalibrationCheckpoint(directory,riPin);
      assert.equal(calls.length-reopenStart,3,'One root, one envelope batch, one reference batch');
      assert(calls.length<172/2,'Diagnostic baseline: 172 launches for four checkpoints and reopen');
      console.log(JSON.stringify({probe:'checkpoint-ignore-launches',gitCalls:calls.length,milliseconds:performance.now()-start}));
      const readStart=calls.length;
      assert.equal(reopened.analyses().length,4);
      assert.equal(calls.length-readStart,2,'Every analysis read rechecks both batches');
      failGit=true;assert.throws(()=>reopened.analyses(),/Injected Git failure/);failGit=false;
      const file=directory+'/ri-60-2.json',before=readFileSync(file);writeFileSync(file,Buffer.concat([before,Buffer.from(' ')]));
      assert.throws(()=>reopened.analyses(),/Artifact changed|Immutable raw JSON changed/);
    } finally {childProcess.execFileSync=execute;syncBuiltinESMExports();}
  `));

  const runnerCheck = (body: string) => checkpointCheck(`
    const {transformSync}=await import('esbuild'),{createRequire}=await import('node:module');
    const fs=await import('node:fs'),path=await import('node:path'),buildModule=await import('./tools/m1010r/build.mjs');
    const realRequire=createRequire(process.cwd()+'/package.json');
    const source=readFileSync('tools/m1010r/study.mjs','utf8');
    const compiled=transformSync(source.slice(0,source.indexOf('\\nif (process.argv[1]'))
      .replace('cases.slice(checkpoint.snapshot().completedExperimentIds.length)',
        'cases.slice(checkpoint.snapshot().completedExperimentIds.length, process.caseLimit)'),{format:'cjs',supported:{'dynamic-import':false}}).code;
    const outdir=directory+'/package',studyDir=directory+'/study',executable=path.resolve(directory,'unit-chrome');
    const analyzerPath=path.resolve('tools/m1010r/instrument_analysis.py'),pythonPath=path.resolve('.cache/m8-venv/bin/python');
    mkdirSync(outdir);writeFileSync(outdir+'/research-provenance.json','{}');
    const browserHash='8319963f6625accf51c0dd4f55091ceaf9f09ed39e7a52fed4fae12b2a6b668a';
    const mode={scenario:'pass',verdict:'PASS',phases:[-20,8,-15,12],dirty:false,version:'153.0.8010.12',cdpVersion:'153.0.8010.12',browserHash};
    const calls={opens:0,pages:0,versions:0,analyses:[],flags:[],timers:[],git:[],closes:0};
    const syntheticFiles=new Map([[executable,Buffer.from('unit browser')],[analyzerPath,Buffer.from('unit analyzer')],[pythonPath,Buffer.from('unit python')]]);
    const localFs={...fs,existsSync:name=>syntheticFiles.has(String(name))||fs.existsSync(name),
      readFileSync:(name,...args)=>syntheticFiles.get(String(name))??fs.readFileSync(name,...args)};
    const buildIdentity={directory:path.resolve(outdir),provenance:{generator:'m1010ri-instrument',instrument:true,
      sourceCommit:'a'.repeat(40),sourceDirty:false,modelSha256:'unit model',mediaManifestSha256:'unit media',
      sourceFiles:{'tools/m1010r/calibration.html':{bytes:readFileSync('tools/m1010r/calibration.html').length,sha256:digest(readFileSync('tools/m1010r/calibration.html'))}},
      mediaManifest:{assets:[{fps:30,media:{sha256:'e006f3d5381d73b1ca5739f5e74216f4bf0c312f28621634d258a770822a8f9b'}},
        {fps:60,media:{sha256:'5171a8b6da7303c8c409167f4191b7330f41120fcd89d7e3aa0df9483b237b29'}}]}}};
    const fakeBuild={...buildModule,verifyBuild:()=>structuredClone(buildIdentity),
      digest:bytes=>bytes.toString()==='unit browser'?mode.browserHash:digest(bytes),
      cachePath:name=>buildModule.cachePath(name==='.cache/m1010r/calibration.lock'?directory+'/runner.lock':name),
      git:args=>{calls.git.push(args);if(args[0]==='status')return mode.dirty?' M dirty':'';
        if(args[0]==='show')return readFileSync('package.json','utf8');if(args[0]==='rev-parse')return 'a'.repeat(40);return '';}};
    const recording=()=>({terminal:{completionReason:'RENDER_TARGET_REACHED',requestedEndFrame:70592,actualObservedEndFrame:70656},
      watchdogFired:mode.scenario==='watchdog',timedOut:false,errors:[],heartbeats:[{actualObservedEndFrame:7168}]});
    const makePage=()=>{
      calls.pages++;let snapshots=0;
      return {bringToFront:async()=>{},goto:async()=>{},locator:selector=>{assert.equal(selector,'#play');return {click:async()=>{}};},
        waitForFunction:async(fn,unused,options)=>{if(options.timeout===125000&&mode.scenario==='timeout')await new Promise(()=>{});},
        evaluate:async(fn,control)=>{
          if(String(fn).includes('audioBase64')){if(mode.scenario==='missing'&&control)throw Error('No control PCM');return Buffer.alloc(4).toString('base64');}
          snapshots++;
          const report={instrument:mode.scenario==='wrong-instrument'?'M10.10R':'M10.10RI',state:'RECORDED',
            audio:{samples:1},audioControl:{samples:1,render:recording()},renderAudio:recording(),renderWindow:{targetEndFrame:3216000},
            frames:[{sequence:1}],callbacks:[{metadata:{mediaTime:0}}],gpuSamples:[{sequence:1}],
            completeness:{count:snapshots},events:[{type:'unit context transition'}],errors:[],cleanup:{completed:true}};
          if(snapshots>1){report.frames.push({sequence:2});report.renderAudio.heartbeats.push({actualObservedEndFrame:19200});}
          return report;
        },close:async()=>{if(mode.scenario==='close')throw Error('unit close failure');}};
    };
    const worker={url:()=> 'chrome-extension://'+'a'.repeat(32)+'/service-worker.js',evaluate:async()=>buildModule.instrumentManifest};
    const native={executable,context:{setDefaultTimeout:()=>{},serviceWorkers:()=>[worker],newPage:async()=>makePage()},
      browser:{version:async()=>mode.cdpVersion,newBrowserCDPSession:async()=>({
        send:async method=>method==='Browser.getVersion'?{product:'Chrome/'+mode.cdpVersion}:{processInfo:[{type:'browser',id:123}]},detach:async()=>{}})},
      close:async()=>{calls.closes++;}};
    const childProcess={execFileSync:(file,args)=>{
      if(file===executable){calls.versions++;assert.deepEqual(args,['--version']);return 'Google Chrome '+mode.version;}
      if(file==='ps')return 'unit native command';
      assert.equal(file,pythonPath);assert.equal(args[0],analyzerPath);calls.analyses.push(args);
      const index=runner.RI_CASES.findIndex(entry=>entry.id===readJson(args[1]).id);
      return JSON.stringify({outcome:mode.scenario==='second-fail'&&calls.analyses.length===2?'FAIL':mode.verdict,
        summary:{reason:'unit analyzer only',failedCriteria:[],missingEvidence:[],controlVerifiedWindows:3,controlMaximumSampleError:0,
          medianDigitalPhaseMs:mode.phases[index],uncertaintyBoundMs:1}});
    }};
    const mockRequire=name=>{
      if(name==='./build.mjs')return fakeBuild;
      if(name==='node:fs')return localFs;
      if(name==='node:child_process')return childProcess;
      if(name==='../m9-browser.mjs')return {openNativeChrome:async flags=>{calls.opens++;calls.flags.push(flags);return native;}};
      if(name==='../m105-accounting.mjs')return {nativeWindow:async()=>({mock:true})};
      if(name==='../m1010/study.mjs')return {environment:()=>({machine:'mock, not a native observation'})};
      if(name.includes('playwright'))return {chromium:{executablePath:()=>executable}};
      return realRequire(name);
    };
    const module={exports:{}};
    const mockedProcess={argv:[],env:{M9_CHROME_EXECUTABLE_PATH:executable},cwd:()=>process.cwd(),pid:process.pid,once:()=>{},removeListener:()=>{}};
    new Function('require','module','exports','process','console','setTimeout','clearTimeout',compiled)(
      mockRequire,module,module.exports,mockedProcess,{log:()=>{}},(callback,ms)=>{
        calls.timers.push(ms);if(ms===125000&&mode.scenario==='timeout')queueMicrotask(callback);return calls.timers.length;
      },()=>{});
    const runner=module.exports;
    ${body}
  `);

  it.each(['pass', 'analyzer-fail', 'second-fail', 'third-outlier', 'fourth-outlier', 'first-missing', 'second-missing', 'timeout', 'watchdog', 'missing', 'wrong-instrument', 'close'])('runs RI %s with immutable evidence and no terminal reacquisition', scenario => runnerCheck(`
    mode.scenario=${JSON.stringify(scenario)};if(mode.scenario==='analyzer-fail')mode.verdict='FAIL';
    if(mode.scenario==='third-outlier')mode.phases[2]=20;
    if(mode.scenario==='fourth-outlier')mode.phases[3]=-12;
    if(mode.scenario==='first-missing')mode.phases[0]=undefined;
    if(mode.scenario==='second-missing')mode.phases[1]=null;
    const state=await runner.runCalibrationStudy(studyDir,{outdir,instrument:true}),passed=mode.scenario==='pass';
    const expected=['ri-30-1','ri-60-1','ri-30-2','ri-60-2'];
    assert.equal(state.status,passed?'COMPLETE':'STOPPED');assert.deepEqual(state.order,expected);
    const attempted=passed||mode.scenario==='fourth-outlier'?4:mode.scenario==='third-outlier'?3:['second-fail','second-missing'].includes(mode.scenario)?2:1;
    assert.equal(calls.pages,attempted);assert.equal(calls.opens,1);assert.equal(calls.closes,1);
    assert.equal(state.pin.baseline,runner.RI_FROZEN_BASELINE);assert.equal(state.pin.analyzerScript,runner.RI_ANALYZER);
    assert.equal(state.pin.analyzerSha256,digest(Buffer.from('unit analyzer')));
    assert.deepEqual(state.pin.repeatability,runner.RI_REPEATABILITY_POLICY);
    assert(calls.git.some(args=>args[0]==='diff'&&args[2]==='f0c3c49fd6e01746ebc232850455da14ac347c63'));
    assert.deepEqual(calls.flags,[[ '--load-extension='+path.resolve(outdir) ]]);assert(calls.timers.includes(125000));
    if(!passed)for(const id of expected.slice(attempted))assert.equal(readJson(studyDir+'/'+id+'.json').result.outcome,'NOT_RUN');
    if(mode.scenario==='second-fail')assert.equal(state.stopReason.id,'ri-60-1');
    const analyses=expected.slice(0,attempted).map(id=>readJson(studyDir+'/'+id+'-analysis.json').result);
    if(passed){
      assert.equal(calls.analyses.length,4);assert.equal(Object.keys(state.analysisArtifacts).length,4);
      assert.deepEqual(analyses.map(result=>result.individualOutcome),['PASS','PASS','PASS','PASS']);
      assert.deepEqual(analyses.map(result=>result.repeatability.validatedPairCount),[0,0,1,2]);
      assert.deepEqual(analyses.map(result=>result.repeatability.outcome),['PENDING','PENDING','PENDING','PASS']);
      assert.deepEqual(analyses[3].repeatability.pairs.map(pair=>[pair.fps,pair.outcome,pair.spreadMs]),[[30,'PASS',5],[60,'PASS',4]]);
      for(const [index,result] of analyses.entries())for(const pair of result.repeatability.pairs){
        assert.equal(pair.thresholdMs,1000/pair.fps);
        assert.deepEqual(pair.runs.map(run=>run.id),expected.slice(0,index+1).filter(id=>id.startsWith('ri-'+pair.fps+'-')));
        for(const run of pair.runs)assert.deepEqual(run.analysis,run.id===expected[index]?null:artifact(studyDir+'/'+run.id+'-analysis.json'));
      }
    }
    if(mode.scenario.endsWith('outlier')){
      const last=analyses.at(-1);assert.equal(calls.analyses.length,attempted);
      assert.equal(last.individualOutcome,'PASS');assert.equal(last.outcome,'FAIL');assert.equal(last.reason,'repeatspread');
      assert.equal(state.stopReason.reason,'repeatspread');assert.equal(last.repeatability.validatedPairCount,attempted===3?0:1);
      assert.equal(last.summary.medianDigitalPhaseMs,mode.phases[attempted-1]);
    }
    if(['first-missing','second-missing'].includes(mode.scenario)){
      const last=analyses.at(-1);assert.equal(calls.analyses.length,attempted);
      assert.equal(last.individualOutcome,'PASS');assert.equal(last.outcome,'UNRESOLVED');assert.equal(last.reason,'missing_phase_median');
      assert.equal(last.repeatability.pairs[attempted-1].runs[0].medianDigitalPhaseMs,null);
      assert.equal(last.repeatability.pairs[attempted-1].spreadMs,null);
    }
    const raw=readJson(studyDir+'/'+expected[0]+'.json').result;
    assert.deepEqual(raw.launchPin,artifact(raw.launchPin.path));assert.deepEqual(readJson(raw.launchPin.path).pin,state.pin);
    assert.equal(readJson(raw.rawSnapshot.path).frames.length,1);
    assert.equal(readJson(raw.finalRawSnapshot.path).frames.length,2);
    assert.deepEqual(raw.report.frames,[{sequence:1},{sequence:2}]);assert.equal(raw.report.renderAudio.heartbeats.length,2);
    if(mode.scenario==='missing'){assert.equal(raw.controlAudioPcm,null);assert(raw.errors.some(error=>error.stage==='control-audio'));}
    if(mode.scenario==='timeout')assert(raw.errors.some(error=>error.stage==='collection'&&error.message.includes('timeout')));
    const before=Object.fromEntries(readdirSync(studyDir).map(name=>[name,readFileSync(studyDir+'/'+name)]));
    const count=JSON.stringify(calls);
    assert.deepEqual(await runner.runCalibrationStudy(studyDir,{outdir,instrument:true}),state);
    const oldCalls=JSON.parse(count);
    for(const key of ['opens','pages','versions','analyses','flags','closes'])assert.deepEqual(calls[key],oldCalls[key]);
    assert.deepEqual(Object.fromEntries(readdirSync(studyDir).map(name=>[name,readFileSync(studyDir+'/'+name)])),before);
  `));

  it.each(['pass', 'outlier'])('resumes an RI two-PASS prefix with %s repeatability and no new IDs', scenario => runnerCheck(`
    mockedProcess.caseLimit=2;
    const prefix=await runner.runCalibrationStudy(studyDir,{outdir,instrument:true});
    assert.equal(prefix.status,'READY');assert.equal(calls.pages,2);
    assert.deepEqual(prefix.completedExperimentIds,['ri-30-1','ri-60-1']);
    const before=Object.fromEntries(readdirSync(studyDir).filter(name=>name!=='state.json').map(name=>[name,readFileSync(studyDir+'/'+name)]));
    delete mockedProcess.caseLimit;
    const failed=${JSON.stringify(scenario)}==='outlier';if(failed)mode.phases[2]=20;
    const state=await runner.runCalibrationStudy(studyDir,{outdir,instrument:true});
    assert.equal(state.status,failed?'STOPPED':'COMPLETE');assert.equal(calls.pages,failed?3:4);
    assert.deepEqual(calls.analyses.map(args=>readJson(args[1]).id),state.order.slice(0,failed?3:4));
    if(failed){assert.equal(state.stopReason.id,'ri-30-2');assert.equal(readJson(studyDir+'/ri-60-2.json').result.outcome,'NOT_RUN');}
    const last=readJson(studyDir+'/'+(failed?'ri-30-2':'ri-60-2')+'-analysis.json').result;
    assert.equal(last.repeatability.validatedPairCount,failed?0:2);
    assert.deepEqual(last.repeatability.pairs[0].runs[0].analysis,prefix.analysisArtifacts['ri-30-1']);
    for(const [name,bytes] of Object.entries(before))assert.deepEqual(readFileSync(studyDir+'/'+name),bytes);
  `));

  it('rejects missing/nonfinite RI medians without coercion and accepts exact one-frame boundaries', () => checkpointCheck(`
    const {withRIRepeatability,RI_CASES}=await import('./tools/m1010r/study.mjs');
    for(const median of [undefined,null,NaN,Infinity,-Infinity,'0',false]){
      const result=withRIRepeatability('ri-30-1',{outcome:'PASS',summary:{medianDigitalPhaseMs:median}},[]);
      assert.equal(result.individualOutcome,'PASS');assert.equal(result.outcome,'UNRESOLVED');
      assert.equal(result.repeatability.pairs[0].spreadMs,null);assert.equal(result.repeatability.pairs[0].runs[0].medianDigitalPhaseMs,null);
    }
    assert.equal(withRIRepeatability('ri-30-1',{outcome:'PASS'},[]).outcome,'UNRESOLVED');
    const previous=[];
    for(const entry of RI_CASES){
      const result=withRIRepeatability(entry.id,{outcome:'PASS',summary:{medianDigitalPhaseMs:entry.repeat===1?0:1000/entry.fps}},previous);
      assert.equal(result.outcome,'PASS');previous.push({id:entry.id,result,reference:null});
    }
    assert.equal(previous[3].result.repeatability.validatedPairCount,2);
    assert(previous[3].result.repeatability.pairs.every(pair=>pair.outcome==='PASS'&&pair.runs.length===2));
    assert.throws(()=>withRIRepeatability('ri-60-2',{outcome:'PASS'},previous.slice(0,2)),/prefix/);
  `));

  it.each(['hash', 'raw', 'pin', 'prior-reference', 'pair-count', 'missing-pair', 'prefix'])('rejects RI %s corruption on read and resume without mutation', target => checkpointCheck(`
    const {withRIRepeatability,RI_CASES,RI_STUDY_VERSION}=await import('./tools/m1010r/study.mjs');
    const riPin={...pin,studyVersion:RI_STUDY_VERSION},checkpoint=openCalibrationCheckpoint(directory,riPin);
    for(const entry of RI_CASES){checkpoint.begin(entry.id);checkpoint.raw(entry.id,{outcome:'RECORDED'});
      checkpoint.complete(entry.id,withRIRepeatability(entry.id,{outcome:'PASS',summary:{medianDigitalPhaseMs:0}},checkpoint.analyses()));}
    const original=checkpoint.analyses();original[0].result.summary.medianDigitalPhaseMs=999;
    assert.equal(checkpoint.analyses()[0].result.summary.medianDigitalPhaseMs,0);
    const target=${JSON.stringify(target)},id='ri-60-2',file=directory+'/'+id+'-analysis.json',envelope=readJson(file),state=readJson(directory+'/state.json');
    if(target==='hash')writeFileSync(file,readFileSync(file)+' ');
    else {
      if(target==='raw')envelope.result.raw=state.rawArtifacts['ri-30-2'];
      if(target==='pin')envelope.pin.sourceCommit='changed';
      if(target==='prior-reference')envelope.result.repeatability.pairs[0].runs[0].analysis=state.analysisArtifacts['ri-60-1'];
      if(target==='pair-count')envelope.result.repeatability.validatedPairCount=1;
      if(target==='missing-pair')envelope.result.repeatability.pairs.pop();
      if(target==='prefix')state.completedExperimentIds.reverse();
      writeFileSync(file,JSON.stringify(envelope));state.analysisArtifacts[id]=artifact(file);writeFileSync(directory+'/state.json',JSON.stringify(state));
    }
    const before=Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)]));
    if(target!=='prefix')assert.throws(()=>checkpoint.analyses());
    assert.throws(()=>openCalibrationCheckpoint(directory,riPin));
    assert.deepEqual(Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)])),before);
  `));

  it('cannot complete RI from bare PASS or forged aggregate counts', () => checkpointCheck(`
    const {withRIRepeatability,RI_CASES,RI_STUDY_VERSION}=await import('./tools/m1010r/study.mjs');
    const checkpoint=openCalibrationCheckpoint(directory,{...pin,studyVersion:RI_STUDY_VERSION});
    for(const entry of RI_CASES){
      checkpoint.begin(entry.id);checkpoint.raw(entry.id,{outcome:'RECORDED'});
      const individual={outcome:'PASS',summary:{medianDigitalPhaseMs:0}};
      const valid=withRIRepeatability(entry.id,individual,checkpoint.analyses()),forged=structuredClone(valid);
      forged.repeatability.validatedPairCount=entry.repeat===1?2:0;
      const before=readFileSync(directory+'/state.json');
      assert.throws(()=>checkpoint.complete(entry.id,individual));assert.throws(()=>checkpoint.complete(entry.id,forged));
      assert.deepEqual(readFileSync(directory+'/state.json'),before);assert(!existsSync(directory+'/'+entry.id+'-analysis.json'));
      checkpoint.complete(entry.id,valid);
    }
    assert.equal(checkpoint.snapshot().status,'COMPLETE');assert.equal(checkpoint.analyses()[3].result.repeatability.validatedPairCount,2);
  `));

  it.each(['dirty', 'hash', 'version', 'cdp'])('rejects RI %s pin failure before page collection', failure => runnerCheck(`
    const failure=${JSON.stringify(failure)};
    if(failure==='dirty')mode.dirty=true;
    if(failure==='hash')mode.browserHash='0'.repeat(64);
    if(failure==='version')mode.version='153.0.8010.13';
    if(failure==='cdp')mode.cdpVersion='153.0.8010.13';
    if(['dirty','hash'].includes(failure))await assert.rejects(runner.runCalibrationStudy(studyDir,{outdir,instrument:true}));
    else {const state=await runner.runCalibrationStudy(studyDir,{outdir,instrument:true});assert.equal(state.status,'STOPPED');
      for(const id of state.order.slice(1))assert.equal(readJson(studyDir+'/'+id+'.json').result.outcome,'NOT_RUN');}
    assert.equal(calls.pages,0);assert.equal(calls.opens,failure==='cdp'?1:0);
  `));

  it.each(['build', 'study'])('keeps legacy %s CLI arguments and accepts RI only in the fixed final position', file => check(`
    const {resolve,join}=await import('node:path'),{fileURLToPath}=await import('node:url');
    const buildModule=await import('./tools/m1010r/build.mjs'),studyModule=await import('./tools/m1010r/study.mjs');
    const file=${JSON.stringify(file)},filename=resolve('tools/m1010r/'+file+'.mjs');
    const source=readFileSync(filename,'utf8');
    const cli=source.slice(source.indexOf('\\nif (process.argv[1]')).replaceAll('import.meta.url',JSON.stringify('file://'+filename));
    const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
    for(const args of [[],['old-output','old-input'],['ri-output','ri-input','--instrument'],['','','--instrument'],
      ['--instrument'],['out','--instrument'],['out','input','--invalid'],['out','input','--instrument','extra']]) {
      const calls=[],mockProcess={argv:[process.execPath,filename,...args],exitCode:0};
      const invoke=async(...values)=>{calls.push(values);return {status:'COMPLETE'};};
      const valid=args.length===0||args.length===2&&!args[1].startsWith('--')||args.length===3&&args[2]==='--instrument';
      const action=new AsyncFunction('process','assert','resolve','join','fileURLToPath','console','buildCalibration','runCalibrationStudy',
        'DEFAULT_EXTENSION','DEFAULT_RI_EXTENSION','DEFAULT_MEDIA','DEFAULT_CALIBRATION','DEFAULT_RI_CALIBRATION','errorInfo',cli)(
          mockProcess,assert,resolve,join,fileURLToPath,{log:()=>{},error:()=>{}},invoke,invoke,
          buildModule.DEFAULT_EXTENSION,buildModule.DEFAULT_RI_EXTENSION,buildModule.DEFAULT_MEDIA,
          studyModule.DEFAULT_CALIBRATION,studyModule.DEFAULT_RI_CALIBRATION,error=>({message:String(error)}));
      if(!valid&&file==='build')await assert.rejects(action);else await action;
      assert.equal(calls.length,valid?1:0);
      if(!valid)continue;
      const instrument=args[2]==='--instrument';
      if(file==='build')assert.deepEqual(calls[0],[args[0]||(instrument?buildModule.DEFAULT_RI_EXTENSION:buildModule.DEFAULT_EXTENSION),
        args[1]||buildModule.DEFAULT_MEDIA,{instrument}]);
      else assert.deepEqual(calls[0],[args[0]||(instrument?studyModule.DEFAULT_RI_CALIBRATION:studyModule.DEFAULT_CALIBRATION),
        {outdir:args[1]||(instrument?buildModule.DEFAULT_RI_EXTENSION:buildModule.DEFAULT_EXTENSION),instrument}]);
    }
  `));

  it('builds permission-free RI and legacy inventories from hash-valid dummy media', () => checkpointCheck(`
    const {buildCalibration,verifyBuild,verifyInstrumentInputs,PROVENANCE_FILE}=await import('./tools/m1010r/build.mjs');
    const mediaFile=directory+'/dummy.mp4',pcmFile=directory+'/dummy.f32le';
    writeFileSync(mediaFile,'unit media, not native evidence');writeFileSync(pcmFile,Buffer.alloc(4));
    const media={schemaVersion:1,study:'M10.10R',sampleRate:48000,seconds:70,
      producerSha256:digest(readFileSync('tools/m1010r/media.mjs')),signal:artifact(pcmFile),
      assets:[30,60].map(fps=>({fps,media:artifact(mediaFile),pcm:artifact(pcmFile),validation:{
        audio:{decodedPcmSha256:artifact(pcmFile).sha256,decodedPcmBytes:4},video:{fps,allIdentitiesAndPtsExact:true}}}))};
    for(const instrument of [false,true]) {
      const outdir=directory+(instrument?'/ri':'/legacy');
      const built=await buildCalibration(outdir,media,{instrument});
      const verified=verifyBuild(outdir).provenance;
      assert.deepEqual(verified,built.provenance);
      const manifest=readJson(outdir+'/manifest.json');
      assert.equal(manifest.name,instrument?'AetherVSR M10.10RI Instrument':'AetherVSR M10.10R Calibration');
      assert.equal(manifest.version,'0.0.1');
      for(const key of ['permissions','optional_permissions','host_permissions','optional_host_permissions','web_accessible_resources']) assert(!(key in manifest));
      assert.deepEqual(Object.keys(verified.files).sort(),['manifest.json','calibration.js',instrument?'render-recorder.js':'audio-worklet.js',
        'calibration.html','player.css','service-worker.js',... [30,60].flatMap(fps=>['media/replay-'+fps+'.mp4','media/decoded-'+fps+'.f32le'])].sort());
      if(instrument) {
        assert.equal(verified.instrument,true);assert.equal(verified.generator,'m1010ri-instrument');
        assert.match(readFileSync(outdir+'/calibration.js','utf8'),/renderClockMode = true/);
        for(const inputs of Object.values(verified.bundleInputs)) {
          verifyInstrumentInputs(inputs);
          for(const name of inputs) assert.equal(verified.sourceFiles[name].sha256,digest(readFileSync(name)));
        }
        const path=outdir+'/'+PROVENANCE_FILE,before=readFileSync(path);
        for(const mutate of [value=>{value.instrument=false;},value=>{value.bundleInputs['calibration.js'].push('src/core/neural/model.ts');},
          value=>{value.sourceFiles['tools/m1010r/calibration.ts'].sha256='0'.repeat(64);},value=>{delete value.sourceFiles['tools/m1010r/calibration.ts'];},
          value=>{delete value.sourceFiles['tools/m1010r/build.mjs'];}]) {
          const changed=structuredClone(verified);mutate(changed);writeFileSync(path,JSON.stringify(changed));assert.throws(()=>verifyBuild(outdir));
        }
        writeFileSync(path,before);
        await assert.rejects(buildCalibration(outdir,media),/foreign output/);
      } else {
        assert.match(readFileSync(outdir+'/calibration.js','utf8'),/renderClockMode = false/);
        assert(!('instrument' in verified));
        const path=outdir+'/'+PROVENANCE_FILE;
        const historical={...verified,sourceFiles:{'historic-source.ts':{bytes:1,sha256:'0'.repeat(64)}}};
        writeFileSync(path,JSON.stringify(historical));assert.deepEqual(verifyBuild(outdir).provenance,historical);
        await assert.rejects(buildCalibration(outdir,media,{instrument:true}),/foreign output/);
      }
    }
    for(const input of ['src/core/neural/model.ts','src/core/neural/conv.wgsl.ts','src/core/upscale/neural-upscaler.ts',
      'src/runtime.ts','src/runtime/controller.ts','src/runtime-controller.ts','tools/m1010/acquire.ts',
      'tools/m1010/authority.ts','src/extension/authority.ts','public/models/aethersr-c16d2.json','models/model.json'])
      assert.throws(()=>verifyInstrumentInputs([input]),/Forbidden instrument input/);
    const esbuild=await import('esbuild'),{createRequire}=await import('node:module'),{dirname}=await import('node:path');
    const productionBuild=await import('./tools/build-extension.mjs');
    const require=createRequire(process.cwd()+'/tools/m1010r/build.mjs');
    const source=readFileSync('tools/m1010r/build.mjs','utf8');
    const compiled=esbuild.transformSync(source.slice(0,source.indexOf('\\nif (process.argv[1]'))
      .replaceAll('import.meta.url',JSON.stringify('file://'+process.cwd()+'/tools/m1010r/build.mjs')),{format:'cjs'}).code;
    const contaminated={exports:{}};
    new Function('require','module','exports',compiled)(name=>name==='esbuild'?{...esbuild,build:options=>esbuild.build({...options,plugins:[{
      name:'unit-transitive-neural-import',setup:builder=>builder.onLoad({filter:/[/]m1010r[/]probe[.]ts$/},args=>({
        contents:"import '../../src/core/neural/model.js'; export class IdentityProbe {}",loader:'ts',resolveDir:dirname(args.path)}))}]})}
        :name==='../build-extension.mjs'?productionBuild:require(name),
      contaminated,contaminated.exports);
    await assert.rejects(contaminated.exports.buildCalibration(directory+'/forbidden',media,{instrument:true}),/Forbidden instrument input: src\\/core\\/neural\\//);
    assert(!existsSync(directory+'/forbidden'));
  `));

  it.each(['PASS', 'FAIL', 'UNRESOLVED', 'INTERRUPTED'])('keeps RI order and terminal %s evidence separate from R', outcome => checkpointCheck(`
    const {RI_CASES,withRIRepeatability}=await import('./tools/m1010r/study.mjs');
    const expected=['ri-30-1','ri-60-1','ri-30-2','ri-60-2'];
    assert.deepEqual(RI_CASES.map(entry=>entry.id),expected);
    assert.deepEqual(RI_CASES.map(entry=>entry.fps),[30,60,30,60]);
    assert(Object.isFrozen(RI_CASES));assert(RI_CASES.every(Object.isFrozen));
    const riPin={...pin,studyVersion:'M10.10RI-instrument-1'},outcome=${JSON.stringify(outcome)};
    let checkpoint=openCalibrationCheckpoint(directory,riPin,RI_CASES);
    assert.deepEqual(checkpoint.snapshot().order,expected);
    assert.throws(()=>checkpoint.begin(ids[0]));assert.throws(()=>checkpoint.begin(expected[1]));
    for(const id of outcome==='PASS'?expected:expected.slice(0,1)){
      checkpoint.begin(id);checkpoint.raw(id,{outcome:'RECORDED',report:{instrument:'M10.10RI',frames:[{sequence:9}],heartbeats:[{endFrame:7168}]}});
      if(outcome!=='INTERRUPTED') checkpoint.complete(id,withRIRepeatability(id,{outcome,summary:{medianDigitalPhaseMs:0}},checkpoint.analyses()));
    }
    if(outcome==='INTERRUPTED') checkpoint=openCalibrationCheckpoint(directory,riPin);
    const stopped=checkpoint.snapshot();
    assert.equal(stopped.status,outcome==='PASS'?'COMPLETE':'STOPPED');
    assert.deepEqual(stopped.completedExperimentIds,expected);
    if(outcome!=='PASS') for(const id of expected.slice(1)) assert.equal(readJson(directory+'/'+id+'.json').result.outcome,'NOT_RUN');
    assert.deepEqual(readJson(directory+'/'+expected[0]+'.json').result.report.heartbeats,[{endFrame:7168}]);
    const before=Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)]));
    checkpoint=openCalibrationCheckpoint(directory,riPin);
    assert.deepEqual(checkpoint.snapshot(),stopped);
    for(const id of [undefined,...ids,...expected]) assert.throws(()=>checkpoint.begin(id));
    assert.throws(()=>openCalibrationCheckpoint(directory,riPin,CALIBRATION_CASES));
    assert.deepEqual(Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)])),before);
  `));

  it('rejects completion without raw evidence without mutating the active attempt', () => checkpointCheck(`
    const checkpoint=openCalibrationCheckpoint(directory,pin);checkpoint.begin(ids[0]);
    const before=readFileSync(directory+'/state.json');
    assert.throws(()=>checkpoint.complete(ids[0],{outcome:'PASS'}));
    assert.deepEqual(readFileSync(directory+'/state.json'),before);
    assert(!existsSync(directory+'/'+ids[0]+'-analysis.json'));
    assert.equal(checkpoint.snapshot().activeExperimentId,ids[0]);
  `));

  it('keeps an all-PASS terminal resume immutable and rejects every extra acquisition', () => checkpointCheck(`
    let checkpoint=openCalibrationCheckpoint(directory,pin);
    for(const id of ids){checkpoint.begin(id);checkpoint.raw(id,{outcome:'RECORDED'});checkpoint.complete(id,{outcome:'PASS'});}
    const before=Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)]));
    checkpoint=openCalibrationCheckpoint(directory,pin);
    assert.equal(checkpoint.snapshot().status,'COMPLETE');
    assert.deepEqual(checkpoint.snapshot().completedExperimentIds,ids);
    for(const id of [undefined,null,'unknown',...ids]) assert.throws(()=>checkpoint.begin(id));
    assert.deepEqual(Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)])),before);
  `));

  it('persists a fresh pin and fixed order and isolates snapshot mutations', () => checkpointCheck(`
    const checkpoint=openCalibrationCheckpoint(directory,pin),fresh=checkpoint.snapshot();
    assert.deepEqual(fresh,readJson(directory+'/state.json'));
    assert.deepEqual(fresh.pin,pin);assert.deepEqual(fresh.order,ids);
    assert.equal(fresh.status,'READY');assert.equal(fresh.activeExperimentId,null);
    assert.equal(fresh.requiredNextManualAction,null);assert.equal(fresh.stopReason,null);
    assert.deepEqual(fresh.completedExperimentIds,[]);assert.deepEqual(fresh.rawArtifacts,{});assert.deepEqual(fresh.analysisArtifacts,{});
    fresh.pin.sourceCommit='changed';fresh.order.reverse();fresh.completedExperimentIds.push(ids[0]);
    assert.deepEqual(checkpoint.snapshot(),readJson(directory+'/state.json'));
    const before=readFileSync(directory+'/state.json');
    assert.deepEqual(openCalibrationCheckpoint(directory,structuredClone(pin)).snapshot(),checkpoint.snapshot());
    assert.deepEqual(readFileSync(directory+'/state.json'),before);
  `));

  it('resumes a PASS prefix without rewriting evidence or rerunning completed attempts', () => checkpointCheck(`
    let checkpoint=openCalibrationCheckpoint(directory,pin);
    const pcmPath=directory+'/'+ids[0]+'-audio.f32le';writeFileSync(pcmPath,Buffer.from([0,0,128,63]));
    for(const id of ids.slice(0,2)){
      checkpoint.begin(id);
      const raw=checkpoint.raw(id,{outcome:'RECORDED',report:{audio:{pcm:artifact(pcmPath)}}});
      assert.deepEqual(raw,artifact(directory+'/'+id+'.json'));
      const rawBytes=readFileSync(raw.path);
      assert.throws(()=>checkpoint.raw(id,{outcome:'REPLACED'}),/EEXIST/);
      assert.deepEqual(readFileSync(raw.path),rawBytes);
      checkpoint.complete(id,{outcome:'PASS'});
      assert.deepEqual(readJson(directory+'/'+id+'-analysis.json').result.raw,raw);
    }
    const before=Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)]));
    checkpoint=openCalibrationCheckpoint(directory,pin);
    assert.equal(checkpoint.snapshot().status,'READY');assert.deepEqual(checkpoint.snapshot().completedExperimentIds,ids.slice(0,2));
    for(const id of ids.slice(0,2)){assert.throws(()=>checkpoint.begin(id));assert.throws(()=>checkpoint.raw(id,{}));assert.throws(()=>checkpoint.complete(id,{outcome:'PASS'}));}
    assert.deepEqual(Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)])),before);
    checkpoint.begin(ids[2]);assert.equal(checkpoint.snapshot().activeExperimentId,ids[2]);
    for(const [name,bytes] of Object.entries(before)) if(name!=='state.json') assert.deepEqual(readFileSync(directory+'/'+name),bytes);
  `));

  it.each(['raw', 'analysis', 'PCM'])('rejects %s hash tampering on resume without rewriting state', target => checkpointCheck(`
    const checkpoint=openCalibrationCheckpoint(directory,pin),id=ids[0],pcmPath=directory+'/'+id+'-audio.f32le';
    checkpoint.begin(id);writeFileSync(pcmPath,Buffer.from([0,0,128,63]));
    checkpoint.raw(id,{outcome:'RECORDED',report:{audio:{pcm:artifact(pcmPath)}}});checkpoint.complete(id,{outcome:'PASS'});
    const target=${JSON.stringify(target)},path=target==='PCM'?pcmPath:directory+'/'+id+(target==='analysis'?'-analysis':'')+'.json';
    const original=readFileSync(path),changed=Buffer.from(original);
    if(target==='PCM') changed[0]^=1;else changed[changed.length-1]=32;
    writeFileSync(path,changed);const stateBytes=readFileSync(directory+'/state.json');
    assert.throws(()=>openCalibrationCheckpoint(directory,pin),/Artifact changed|Immutable raw JSON changed|Immutable analysis changed/);
    assert.deepEqual(readFileSync(path),changed);assert.deepEqual(readFileSync(directory+'/state.json'),stateBytes);
  `));

  it.each(['FAIL', 'UNRESOLVED'])('closes a %s attempt with an immutable NOT_RUN suffix', outcome => checkpointCheck(`
    let checkpoint=openCalibrationCheckpoint(directory,pin);
    checkpoint.begin(ids[0]);checkpoint.raw(ids[0],{outcome:'RECORDED'});checkpoint.complete(ids[0],{outcome:'PASS'});
    checkpoint.begin(ids[1]);checkpoint.raw(ids[1],{outcome:'RECORDED'});checkpoint.complete(ids[1],{outcome:${JSON.stringify(outcome)},summary:'unit stop',reason:'RI-style reason'});
    const stopped=checkpoint.snapshot();assert.equal(stopped.status,'STOPPED');assert.deepEqual(stopped.completedExperimentIds,ids);
    assert.equal(stopped.stopReason.outcome,${JSON.stringify(outcome)});assert.equal(stopped.activeExperimentId,null);assert.equal(stopped.requiredNextManualAction,null);
    assert.equal(stopped.stopReason.reason,'unit stop');
    assert.deepEqual(Object.keys(stopped.analysisArtifacts),ids.slice(0,2));
    for(const id of ids.slice(2)){
      assert.deepEqual(readJson(directory+'/'+id+'.json').result,{outcome:'NOT_RUN',reason:stopped.stopReason});
      assert.deepEqual(stopped.rawArtifacts[id],artifact(directory+'/'+id+'.json'));
    }
    const before=Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)]));
    checkpoint=openCalibrationCheckpoint(directory,pin);assert.deepEqual(checkpoint.snapshot(),stopped);
    for(const id of [undefined,...ids]) assert.throws(()=>checkpoint.begin(id));
    assert.throws(()=>checkpoint.raw(ids[1],{}));assert.throws(()=>checkpoint.complete(ids[1],{outcome:'PASS'}));
    assert.deepEqual(Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)])),before);
  `));

  it.each([false, true])('closes an interrupted active attempt without rerun (raw present: %s)', withRaw => checkpointCheck(`
    let checkpoint=openCalibrationCheckpoint(directory,pin);
    checkpoint.begin(ids[0]);checkpoint.raw(ids[0],{outcome:'RECORDED'});checkpoint.complete(ids[0],{outcome:'PASS'});
    const id=ids[1],pcmPath=directory+'/'+id+'-audio.f32le';
    checkpoint.begin(id);writeFileSync(pcmPath,Buffer.from([0,0,128,63]));
    if(${withRaw}) checkpoint.raw(id,{outcome:'RECORDED',audioPcm:artifact(pcmPath)});
    const retained=Object.fromEntries(readdirSync(directory).filter(name=>name!=='state.json').map(name=>[name,readFileSync(directory+'/'+name)]));
    checkpoint=openCalibrationCheckpoint(directory,pin);const stopped=checkpoint.snapshot();
    assert.equal(stopped.status,'STOPPED');assert.deepEqual(stopped.completedExperimentIds,ids);
    assert.equal(stopped.activeExperimentId,null);assert.equal(stopped.requiredNextManualAction,null);assert.equal(stopped.stopReason.outcome,'INTERRUPTED');
    const result=readJson(directory+'/'+id+'.json').result;
    assert.equal(result.outcome,${withRaw}?'RECORDED':'INTERRUPTED');
    if(!${withRaw}) assert.deepEqual(result.rawReferences,[artifact(pcmPath)]);
    assert.deepEqual(stopped.interruption,artifact(directory+'/'+id+'-interrupted.json'));
    const interrupted=readJson(stopped.interruption.path);
    assert.equal(interrupted.id,id);assert.deepEqual(interrupted.pin,pin);assert.equal(interrupted.result.outcome,'INTERRUPTED');
    assert.deepEqual(interrupted.result.raw,stopped.rawArtifacts[id]);
    for(const pending of ids.slice(2)) assert.equal(readJson(directory+'/'+pending+'.json').result.outcome,'NOT_RUN');
    for(const [name,bytes] of Object.entries(retained)) assert.deepEqual(readFileSync(directory+'/'+name),bytes);
    const before=Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)]));
    checkpoint=openCalibrationCheckpoint(directory,pin);assert.throws(()=>checkpoint.begin(id));
    assert.deepEqual(Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)])),before);
  `));

  it.each(['setup', 'active', 'recorded', 'closed'])('retains %s runner errors as terminal UNRESOLVED', stage => checkpointCheck(`
    let checkpoint=openCalibrationCheckpoint(directory,pin);const stage=${JSON.stringify(stage)};
    if(stage==='closed') for(const id of ids){checkpoint.begin(id);checkpoint.raw(id,{outcome:'RECORDED'});checkpoint.complete(id,{outcome:'PASS'});}
    else if(stage!=='setup'){
      checkpoint.begin(ids[0]);if(stage==='recorded') checkpoint.raw(ids[0],{outcome:'RECORDED'});
    }
    const retained=Object.fromEntries(readdirSync(directory).filter(name=>name!=='state.json').map(name=>[name,readFileSync(directory+'/'+name)]));
    checkpoint.fail(new Error('unit '+stage+' failure'));const stopped=checkpoint.snapshot();
    assert.equal(stopped.status,'STOPPED');assert.deepEqual(stopped.completedExperimentIds,ids);
    assert.equal(stopped.activeExperimentId,null);assert.equal(stopped.requiredNextManualAction,null);assert.equal(stopped.stopReason.outcome,'UNRESOLVED');
    assert.match(stopped.stopReason.message,/unit .* failure/);
    assert.deepEqual(stopped.runnerError,artifact(stopped.runnerError.path));
    const error=readJson(stopped.runnerError.path);assert.equal(error.id,'runner-error');assert.deepEqual(error.pin,pin);
    assert.equal(error.result.outcome,'UNRESOLVED');assert.match(error.result.stack,/unit .* failure/);
    for(const [name,bytes] of Object.entries(retained)) assert.deepEqual(readFileSync(directory+'/'+name),bytes);
    if(stage==='active') assert.equal(readJson(directory+'/'+ids[0]+'.json').result.outcome,'UNRESOLVED');
    if(stage!=='closed') for(const id of ids.slice(stage==='setup'?0:1)) assert.equal(readJson(directory+'/'+id+'.json').result.outcome,'NOT_RUN');
    const before=Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)]));
    checkpoint=openCalibrationCheckpoint(directory,pin);assert.deepEqual(checkpoint.snapshot(),stopped);assert.throws(()=>checkpoint.begin(ids[0]));
    assert.deepEqual(Object.fromEntries(readdirSync(directory).map(name=>[name,readFileSync(directory+'/'+name)])),before);
  `));

  it.each(['not JSON', 'null', '[]', '42', '{}', '{"outcome":"UNKNOWN"}', 'nonzero'])('turns analyzer output %s into terminal UNRESOLVED with errors', output => checkpointCheck(`
    const checkpoint=openCalibrationCheckpoint(directory,pin),id=ids[0],script=directory+'/analyzer.mjs',output=${JSON.stringify(output)};
    checkpoint.begin(id);const raw=checkpoint.raw(id,{outcome:'RECORDED'});
    writeFileSync(script,output==='nonzero'?'process.stdout.write(JSON.stringify({outcome:"PASS"}));process.stderr.write("unit analyzer failed");process.exitCode=7;':'process.stdout.write('+JSON.stringify(output)+');');
    const analysis=analyzeCalibration(raw,{executable:process.execPath,script});
    assert.equal(analysis.outcome,'UNRESOLVED');assert.match(analysis.summary,/valid verdict/);assert(analysis.error.message);assert(analysis.error.stack);
    if(output==='nonzero'){assert.equal(analysis.stdout,'{"outcome":"PASS"}');assert.equal(analysis.stderr,'unit analyzer failed');}
    else assert.equal(analysis.stdout,output);
    checkpoint.complete(id,analysis);
    assert.equal(openCalibrationCheckpoint(directory,pin).snapshot().status,'STOPPED');
    assert.equal(readJson(directory+'/'+id+'-analysis.json').result.outcome,'UNRESOLVED');
    for(const pending of ids.slice(1)) assert.equal(readJson(directory+'/'+pending+'.json').result.outcome,'NOT_RUN');
  `));

  it.each(['PASS', 'FAIL', 'UNRESOLVED'])('preserves a valid analyzer %s verdict and raw bytes', outcome => checkpointCheck(`
    const checkpoint=openCalibrationCheckpoint(directory,pin),script=directory+'/analyzer.mjs';
    checkpoint.begin(ids[0]);const raw=checkpoint.raw(ids[0],{outcome:'RECORDED'}),before=readFileSync(raw.path);
    const expected={outcome:${JSON.stringify(outcome)},summary:'unit verdict'};
    writeFileSync(script,'process.stdout.write('+JSON.stringify(JSON.stringify(expected))+');');
    assert.deepEqual(analyzeCalibration(raw,{executable:process.execPath,script}),expected);
    assert.deepEqual(readFileSync(raw.path),before);
  `));

  it('rejects changed pins, unknown or out-of-order IDs, and paths outside the cache', () => checkpointCheck(`
    const checkpoint=openCalibrationCheckpoint(directory,pin),before=readFileSync(directory+'/state.json');
    for(const key of Object.keys(pin)) assert.throws(()=>openCalibrationCheckpoint(directory,{...pin,[key]:'changed'}),/identity changed/);
    for(const id of [undefined,null,'unknown',ids[1]]) assert.throws(()=>checkpoint.begin(id));
    assert.deepEqual(readFileSync(directory+'/state.json'),before);
    for(const path of ['.','.cache/m1010r',directory+'/../../../unit-checkpoint-outside']) assert.throws(()=>openCalibrationCheckpoint(path,pin),/Artifacts must stay below/);
    for(const change of [{completedExperimentIds:['unknown']},{activeExperimentId:'unknown'},{rawArtifacts:{unknown:{}}},{analysisArtifacts:{unknown:{}}}]){
      writeFileSync(directory+'/state.json',JSON.stringify({...JSON.parse(before.toString()),...change}));
      const invalid=readFileSync(directory+'/state.json');assert.throws(()=>openCalibrationCheckpoint(directory,pin));
      assert.deepEqual(readFileSync(directory+'/state.json'),invalid);
    }
  `));

  it('refuses nonempty fresh directories, unregistered attempts and symlinked paths', () => checkpointCheck(`
    const occupied=directory+'/occupied';mkdirSync(occupied);writeFileSync(occupied+'/foreign.json','{}');
    assert.throws(()=>openCalibrationCheckpoint(occupied,pin),/Fresh calibration directory must be empty/);
    assert.equal(readFileSync(occupied+'/foreign.json','utf8'),'{}');assert(!existsSync(occupied+'/state.json'));
    mkdirSync(directory+'/target');symlinkSync('target',directory+'/alias','dir');
    assert.throws(()=>openCalibrationCheckpoint(directory+'/alias',pin),/Symlinked artifact paths/);
    const attempts=directory+'/attempts';const checkpoint=openCalibrationCheckpoint(attempts,pin);
    writeFileSync(attempts+'/'+ids[1]+'-audio.f32le',Buffer.alloc(4));const before=readFileSync(attempts+'/state.json');
    assert.throws(()=>openCalibrationCheckpoint(attempts,pin),/Unregistered attempt artifacts/);
    assert.deepEqual(readFileSync(attempts+'/state.json'),before);assert.equal(checkpoint.snapshot().status,'READY');
  `));
});