import {describe,expect,it} from 'vitest';
import {execFileSync} from 'node:child_process';

function check(source: string) {
  const output=execFileSync(process.execPath,['--input-type=module','-e',`
    import assert from 'node:assert/strict';
    import {CASES,fixtureOracle,anchorIdentifier,summarizeCase} from './tools/m108-architecture.mjs';
    ${source}
    console.log('checked');
  `],{cwd:new URL('../',import.meta.url),encoding:'utf8'});
  expect(output.trim()).toBe('checked');
}

describe('M10.9 independent evidence guards',()=>{
  it('never passes blank images or wrong expected colors as crop proof',()=>check(`
    const {comparePixels}=await import('./tools/m109-study.mjs');
    const image=Buffer.alloc(40*40*3,100),region={left:0,top:0,width:40,height:40};
    assert.equal(comparePixels(image,image,40,40,region,1).verdict,'UNRESOLVED');
    assert.equal(comparePixels(image,image,40,40,region,1,[100,100,100]).verdict,'SUPPORTED_CORRECT');
    assert.equal(comparePixels(image,image,40,40,region,1,[200,200,200]).verdict,'UNSAFE');
    assert.equal(comparePixels(image,Buffer.alloc(image.length,120),40,40,region,1,[100,100,100]).verdict,'UNSAFE');
    assert.equal(comparePixels(image,image,40,40,{left:0,top:0,width:2,height:2},1,[100,100,100]).verdict,'UNRESOLVED');
  `));
  it('requires two ordered matching post-proof submissions before observed reveal',()=>check(`
    const {validateReveals}=await import('./tools/m109-study.mjs');
    const record={owner:'one',sourceGeneration:1,geometryGeneration:1,frameGeneration:1,backingWidth:2,backingHeight:2,sourceWidth:1,sourceHeight:1,validForRecovery:true};
    const trace={rows:[{at:3,boundary:'render',visible:true,source:[1,1],backing:[2,2]}]};
    const make=()=>({submissions:[{...record,sequence:1,observedAt:1},{...record,sequence:2,observedAt:2}],proofs:[{at:0,generation:1,supported:true}]});
    assert.equal(validateReveals(trace,make()).verdict,'SUPPORTED_CORRECT');
    for(const field of['owner','sourceGeneration','geometryGeneration','frameGeneration','backingWidth']){
      const evidence=make();evidence.submissions[1][field]='changed';assert.equal(validateReveals(trace,evidence).verdict,'UNSAFE');
    }
    const missing=make();missing.submissions.shift();assert.equal(validateReveals(trace,missing).verdict,'UNSAFE');
    const duplicate=make();duplicate.submissions[1].sequence=1;assert.equal(validateReveals(trace,duplicate).verdict,'UNSAFE');
    const late=make();late.proofs[0].at=2;assert.equal(validateReveals(trace,late).verdict,'UNSAFE');
    assert.equal(validateReveals(trace,make(),[{name:'reverse',startedAt:1.5}]).verdict,'UNSAFE');
    const comparison={rows:[{...trace.rows[0],captureOriginal:true}]};
    assert.equal(validateReveals(comparison,make()).reveals,0);
  `));
  it('freezes common order and forward/reverse semantic cases independently of candidate choice',()=>check(`
    const {CASES:common,SEMANTICS,OWNERSHIP_CASES}=await import('./tools/m109-study.mjs');
    assert.equal(common.length,52);assert.equal(new Set(common.map(row=>row.name)).size,52);
    for(const entry of SEMANTICS)assert.deepEqual(entry.actions.map(action=>action.name),['cssom','reverse']);
    assert.equal(common.at(-1).name,'late-render-cssom');
    assert.equal(OWNERSHIP_CASES.O1.length,22);assert.equal(OWNERSHIP_CASES.O2.length,12);
  `));
  it('rejects native-token operands even when a lease falsely claims safe live-variable cleanup',()=>check(`
    const {ownershipCase}=await import('./tools/m109-study.mjs');
    const values=new Map();let created=false;
    const style={getPropertyValue:name=>values.get(name)??'',getPropertyPriority:()=>'',setProperty:(name,value)=>{created=true;values.set(name,value);},[Symbol.iterator]:()=>values.keys()};
    const video={style,getBoundingClientRect:()=>({left:0,top:0,width:640,height:400}),getAttribute:name=>name==='style'&&created?[...values].map(([key,value])=>key+':'+value).join(';'):null};
    globalThis.document={querySelector:()=>video};
    globalThis.getComputedStyle=()=>({getPropertyValue:()=>values.get('--host-names')??'none'});
    globalThis.M109={leaseProperty:(_video,token)=>{style.setProperty('anchor-name',token);return{check:()=>({active:true,reason:'active'}),release:()=>({outcome:'SUPPORTED_CORRECT',hostPreserved:true,ownedTokenRemains:false,resources:{observers:0,stylesheets:0}})};}};
    const result=await ownershipCase({model:'O1',name:'live-variable'});
    assert.equal(result.restorationCorrect,true);assert.equal(result.independentlyAbsent.inline,null);
    assert.equal(result.independentlyAbsent.computed,false);assert.equal(result.outcome,'UNSAFE');
  `));
});

describe('M10.8 independent architecture helpers',()=>{
  it('retains the frozen failure denominators and nonqualifying controls',()=>check(`
    const {readFileSync}=await import('node:fs');
    const report=JSON.parse(readFileSync('results/m108-feasibility.json','utf8'));
    assert.equal(report.selection,'NO ARCHITECTURE QUALIFIED');
    assert.equal(report.artifacts.length,13);
    for(const name of ['A','D5','D10','D15']) {
      const candidate=report.candidates.find(row=>row.name===name);
      assert.deepEqual([candidate.outcome,candidate.observed,candidate.planned,candidate.notRun,candidate.cost],['REJECTED',25,96,71,'NOT RUN']);
      const artifact=report.artifacts.find(row=>row.path.endsWith('/common-'+name+'-01.json'));
      assert.equal(artifact.results.length,25);
      assert.equal(artifact.results.at(-1).actions[0].challenge.valid,true);
      const failure=artifact.results.at(-1).summary.firstFailure;
      assert.equal(failure.visible,true);assert.equal(failure.tetherError,0);assert.deepEqual(failure.semantic,['object-fit']);
      assert.equal(failure.focused,true);assert.equal(failure.hostIntact,true);
    }
    const s2=report.artifacts.find(row=>row.path.endsWith('/common-s2-01.json'));
    assert.equal(s2.repeats,1);assert.equal(s2.results.length,32);
    assert.equal(report.candidates.some(row=>row.name==='s2'),false);
    assert(report.limitations.some(text=>text.includes('not independently verified successful submission')));
    const tether=report.artifacts.find(row=>row.path.endsWith('/anchor-tether-01.json'));
    assert.equal(tether.results[0].cleanupRelease.exactOriginal,true);assert.equal(tether.results[0].cleanupRelease.ownedTokenRemains,false);
  `));

  it('never promotes stopped stale cases or a nonreplacement capability matrix',()=>check(`
    const good={at:100,reason:'render',boundary:true,stale:false,visible:true,tetherError:0,semantic:[],focused:true,visibility:'visible',expected:{visible:true}};
    const trace={overflow:false,rows:[good,{...good,at:200}]},actions=[{name:'move',settledAt:0}];
    assert.equal(summarizeCase(trace,actions).verdict,'SUPPORTED_CORRECT');
    trace.rows[1].stale=true;assert.equal(summarizeCase(trace,actions).verdict,'SUPPORTED_STALE');
    assert.equal(summarizeCase(trace,actions,{tetherOnly:true}).verdict,'CHARACTERIZED');
    trace.rows[1].focused=false;assert.equal(summarizeCase(trace,actions).verdict,'INVALID');
    const hidden={...good,visible:false,expected:{visible:false,originalLayoutVisible:true}};
    assert.equal(summarizeCase({overflow:false,rows:[hidden,{...hidden,at:200}]},actions).verdict,'UNSUPPORTED_SAFE');
    hidden.hostIntact=false;assert.notEqual(summarizeCase({overflow:false,rows:[hidden,{...hidden,at:200}]},actions).verdict,'UNSUPPORTED_SAFE');
    assert.equal(summarizeCase({overflow:false,rows:[good,{...good,at:200}]},actions,{realPipeline:true}).verdict,'UNVERIFIED_RECOVERY');
    const submitted={...good,reason:'submitted'};assert.equal(summarizeCase({overflow:false,rows:[submitted,{...submitted,at:200}]},actions,{realPipeline:true}).verdict,'SUPPORTED_CORRECT');
  `));

  it('uses one ordered fixture set with notification-free semantic cases',()=>check(`
    assert.equal(CASES.length,32);assert.equal(new Set(CASES.map(row=>row.name)).size,32);
    assert.deepEqual(CASES.filter(row=>row.semantic).map(row=>row.name),['fit','position','clip','radius','controls','transform','opacity','filter'].map(name=>'cssom-'+name));
    for(const name of['preceding-spacer','open-shadow','existing-anchor','source-scroll','offscreen-return'])assert(CASES.some(row=>row.name===name));
  `));
  it('derives fit and viewport clipping independently of production geometry',()=>check(`
    const input={video:{left:40,top:-20,width:640,height:400},viewport:{width:1200,height:760},clip:null,
      state:{fit:'contain',position:[.5,.5],radius:0,expectedVisible:true,unsupported:false},source:{width:1280,height:720},media:{readyState:4,paused:false}};
    const contain=fixtureOracle(input);assert.equal(contain.visible,true);assert.deepEqual(contain.clip,{left:40,top:0,width:640,height:380});
    assert.deepEqual(contain.image,{width:640,height:360,left:40,top:0});
    input.state.fit='cover';const cover=fixtureOracle(input);assert.equal(cover.image.height,400);assert(cover.image.width>640);
    input.state.unsupported=true;assert.equal(fixtureOracle(input).visible,false);
    input.state.unsupported=false;input.media.readyState=1;assert.equal(fixtureOracle(input).visible,false);
    input.source={width:0,height:0};assert.equal(fixtureOracle(input).image,null);
    input.media.readyState=4;input.video.top=-500;assert.equal(fixtureOracle(input).visible,false);
  `));
  it('keeps ancestor clipping, radius and noncentered fit independent from replacement visibility',()=>check(`
    const input={video:{left:40,top:20,width:640,height:400},viewport:{width:1200,height:760},clip:{left:50,top:30,width:200,height:250},
      state:{fit:'contain',position:[0,1],radius:12,expectedVisible:true,unsupported:false,caption:true,captionZ:3},source:{width:1280,height:720},media:{readyState:4,paused:false}};
    let result=fixtureOracle(input);assert.deepEqual(result.clip,{left:50,top:30,width:200,height:250});assert.equal(result.radius,'12px');assert.equal(result.image.top,60);
    input.state.fit='cover';input.state.position=[1,0];result=fixtureOracle(input);assert.equal(result.image.left+result.image.width,680);
    input.state.radius=18;assert.equal(fixtureOracle(input).radius,'18px');
    input.state.captionZ=0;input.state.unsupported=true;result=fixtureOracle(input);assert.equal(result.visible,false);assert.equal(result.originalLayoutVisible,true);assert.equal(result.captionAboveReplacement,false);
    input.clip.left=900;result=fixtureOracle(input);assert.equal(result.clip.width,0);assert.equal(result.originalLayoutVisible,false);
  `));
  it('uses only valid UUID-derived dashed identifiers and rejects CSS injection',()=>check(`
    const first=anchorIdentifier('12345678-1234-4123-8123-123456789abc');
    const second=anchorIdentifier('12345678-1234-4123-8123-123456789abd');assert.notEqual(first,second);
    assert.match(first,/^--aethervsr-[a-f0-9-]+$/);assert.throws(()=>anchorIdentifier('x; color:red'));
  `));
});