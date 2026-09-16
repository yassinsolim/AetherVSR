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
});