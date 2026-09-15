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

describe('M10.8 independent architecture helpers',()=>{
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