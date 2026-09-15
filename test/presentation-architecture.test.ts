import {describe,expect,it} from 'vitest';
import {execFileSync} from 'node:child_process';

function check(source: string) {
  const output=execFileSync(process.execPath,['--input-type=module','-e',`
    import assert from 'node:assert/strict';
    import {CASES,fixtureOracle,anchorIdentifier} from './tools/m108-architecture.mjs';
    ${source}
    console.log('checked');
  `],{cwd:new URL('../',import.meta.url),encoding:'utf8'});
  expect(output.trim()).toBe('checked');
}

describe('M10.8 independent architecture helpers',()=>{
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
    input.media.readyState=4;input.video.top=-500;assert.equal(fixtureOracle(input).visible,false);
  `));
  it('uses only valid UUID-derived dashed identifiers and rejects CSS injection',()=>check(`
    const first=anchorIdentifier('12345678-1234-4123-8123-123456789abc');
    const second=anchorIdentifier('12345678-1234-4123-8123-123456789abd');assert.notEqual(first,second);
    assert.match(first,/^--aethervsr-[a-f0-9-]+$/);assert.throws(()=>anchorIdentifier('x; color:red'));
  `));
});