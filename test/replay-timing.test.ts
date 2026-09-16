import { describe, it } from 'vitest';
import { execFileSync } from 'node:child_process';

const check = (body: string) => execFileSync(process.execPath, ['--input-type=module', '-e', `
  import assert from 'node:assert/strict';
  import {readFileSync,mkdtempSync,mkdirSync,rmSync} from 'node:fs';
  import {signalPcm,replayRecipe,decodeCounter,audioTimeline,prepareReplayMedia,inspectShiftControls,COUNTER_WIDTH} from './tools/m1010r/media.mjs';
  ${body}
`], { encoding: 'utf8' });

describe('M10.10R digital timing ground truth', () => {
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
    checkpoint.begin(ids[1]);checkpoint.raw(ids[1],{outcome:'RECORDED'});checkpoint.complete(ids[1],{outcome:${JSON.stringify(outcome)},summary:'unit stop'});
    const stopped=checkpoint.snapshot();assert.equal(stopped.status,'STOPPED');assert.deepEqual(stopped.completedExperimentIds,ids);
    assert.equal(stopped.stopReason.outcome,${JSON.stringify(outcome)});assert.equal(stopped.activeExperimentId,null);assert.equal(stopped.requiredNextManualAction,null);
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