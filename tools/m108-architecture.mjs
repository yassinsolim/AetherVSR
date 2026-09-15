import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {join,resolve,relative,dirname} from 'node:path';
import {gzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {createServer} from 'vite';
import {ROOT,sha256} from './m10-fixtures.mjs';
import {openNativeChrome} from './m9-browser.mjs';
import {nativeWindow} from './m105-accounting.mjs';
import {prepareTransitionMedia,presentationEnvironment,geometryOracleSource} from './m107-presentation.mjs';

export const BASELINE='b1a87a1ec96062d551994b5257cb48e586b3ece6';

export function studyIdentity() {
  const git=args=>execFileSync('git',args,{cwd:ROOT,encoding:'utf8'}).trim();
  assert.equal(git(['status','--porcelain']),'','Freeze a clean prototype commit before native evidence');
  git(['diff','--exit-code',BASELINE,'--','src','public/models/aethersr-c16d2.json','package.json','package-lock.json']);
  const controls=JSON.parse(readFileSync(join(ROOT,'.cache/m108/frozen-controls.json')));
  for(const control of controls){
    const provenance=JSON.parse(readFileSync(join(ROOT,control.directory,'build-provenance.json')));
    assert.equal(provenance.sourceCommit,BASELINE);assert.equal(provenance.sourceDirty,false);assert.equal(provenance.bundleSha256,control.bundleSha256);
    for(const[name,value]of Object.entries(provenance.files))assert.equal(sha256(readFileSync(join(ROOT,control.directory,name))),value.sha256);
  }
  return {sourceCommit:git(['rev-parse','HEAD']),controls,sourcePins:Object.fromEntries([
    'tools/m108-architecture.mjs','tools/m108-fixture.html','tools/m10-browser.mjs','tools/m9-browser.mjs','src/extension/geometry.ts',
    'docs/M10.8-DESIGN-PREREGISTRATION.md',
  ].map(path=>[path,sha256(readFileSync(join(ROOT,path)))]))};
}

export const CASES=[
  {name:'page-one',actions:['scroll-one']},{name:'page-228',actions:['scroll-228']},{name:'smooth',actions:['smooth']},
  {name:'nested-scroll',actions:['nested']},{name:'preceding-spacer',actions:['spacer']},{name:'ancestor-class',actions:['class-move']},
  {name:'ancestor-style',actions:['style-move']},{name:'container-query',actions:['query-move']},{name:'viewport-resize',actions:['viewport']},
  {name:'video-css-resize',actions:['video-resize']},{name:'abr',actions:['source-high','source-low']},{name:'reparent',actions:['reparent']},
  {name:'fullscreen',actions:['fullscreen','exit']},{name:'fullscreen-scrolled',actions:['scroll-228','fullscreen','exit']},
  {name:'rounded-clip',actions:['scroll-228']},{name:'contain',actions:['scroll-228']},{name:'cover',actions:['scroll-228']},
  {name:'open-shadow',actions:['spacer']},{name:'existing-anchor',actions:['spacer']},{name:'source-scroll',actions:['source-high-scroll','source-low']},
  {name:'offscreen-return',actions:['offscreen','scroll-zero']},{name:'hidden-media',actions:['hide','show']},
  {name:'paused-media',actions:['pause','play']},{name:'not-ready-media',actions:['not-ready','ready']},
  ...['fit','position','clip','radius','controls','transform','opacity','filter'].map(name=>({name:`cssom-${name}`,actions:[`cssom-${name}`],semantic:true})),
];

export function fixtureOracle({video,viewport,clip,state,source,media}) {
  const left=Math.max(video.left,0,clip?.left??-Infinity),top=Math.max(video.top,0,clip?.top??-Infinity);
  const right=Math.min(video.left+video.width,viewport.width,clip?clip.left+clip.width:Infinity);
  const bottom=Math.min(video.top+video.height,viewport.height,clip?clip.top+clip.height:Infinity);
  const visible=state.expectedVisible&&!state.unsupported&&media.readyState>=2&&!media.paused&&video.width>0&&video.height>0&&right>left&&bottom>top;
  const factor=state.fit==='cover'?Math.max(video.width/source.width,video.height/source.height):Math.min(video.width/source.width,video.height/source.height);
  const image={width:source.width*factor,height:source.height*factor};
  image.left=video.left+(video.width-image.width)*state.position[0];image.top=video.top+(video.height-image.height)*state.position[1];
  return {visible,rect:{...video},clip:{left,top,width:Math.max(0,right-left),height:Math.max(0,bottom-top)},image,
    fit:state.fit,position:`${state.position[0]*100}% ${state.position[1]*100}%`,radius:`${state.radius}px`};
}

export function anchorIdentifier(uuid) {
  if(!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(uuid))throw new Error('A canonical UUIDv4 is required');
  return `--aethervsr-${uuid.toLowerCase()}`;
}

export function leaseAnchor(video,token) {
  if(!/^--aethervsr-[0-9a-f-]{36}$/.test(token)||CSS.escape(token)!==token)throw new Error('Invalid anchor identifier');
  const root=video.getRootNode(),elements=[...root.querySelectorAll('*')];
  if(elements.length>256)throw new Error('Anchor collision scan exceeds bound');
  if(elements.some(element=>getComputedStyle(element).getPropertyValue('anchor-name').includes(token)))throw new Error('Anchor identifier collision');
  const before=video.getAttribute('style'),value=video.style.getPropertyValue('anchor-name'),priority=video.style.getPropertyPriority('anchor-name');
  const resolved=getComputedStyle(video).getPropertyValue('anchor-name');
  const appended=!resolved||resolved==='none'?token:`${resolved}, ${token}`;
  let appliedPriority=priority;
  video.style.setProperty('anchor-name',appended,appliedPriority);
  if(!getComputedStyle(video).getPropertyValue('anchor-name').includes(token)){
    appliedPriority='important';video.style.setProperty('anchor-name',appended,appliedPriority);
  }
  const applied=video.getAttribute('style');
  if(!getComputedStyle(video).getPropertyValue('anchor-name').includes(token)){
    if(before===null)video.removeAttribute('style');else video.setAttribute('style',before);
    throw new Error('Anchor declaration did not resolve');
  }
  let released=false;
  return {token,before,value,priority,resolved,appended,appliedPriority,applied,
    release(){
      if(released)return {repeated:true};released=true;
      const current=video.getAttribute('style'),unchanged=current===applied;
      if(unchanged){if(before===null)video.removeAttribute('style');else video.setAttribute('style',before);}
      else if(video.style.getPropertyValue('anchor-name')===appended&&video.style.getPropertyPriority('anchor-name')===appliedPriority){
        if(value)video.style.setProperty('anchor-name',value,priority);else video.style.removeProperty('anchor-name');
      }
      return {unchanged,exactOriginal:video.getAttribute('style')===before,current:video.getAttribute('style'),ownedTokenRemains:getComputedStyle(video).getPropertyValue('anchor-name').includes(token)};
    }};
}

export async function studyServer() {
  prepareTransitionMedia();
  const html=readFileSync(join(ROOT,'tools/m108-fixture.html'));
  const server=await createServer({root:ROOT,mode:'benchmark',server:{host:'127.0.0.1',port:5192,strictPort:true},plugins:[{
    name:'m108-architecture-fixture',configureServer(instance){instance.middlewares.use((request,response,next)=>{
      const path=new URL(request.url,'http://local').pathname;
      if(path==='/m108'){response.setHeader('Content-Type','text/html; charset=utf-8');response.end(html);return;}
      const name={'/m108-media/low.mp4':'low.mp4','/m108-media/high.mp4':'high.mp4','/m108-media/config.json':'config.json'}[path];
      if(!name)return next();response.setHeader('Content-Type',name.endsWith('.json')?'application/json':'video/mp4');response.end(readFileSync(join(ROOT,'.cache/m107/media',name)));
    });},
  }]});await server.listen();
  return {url:'http://127.0.0.1:5192/m108',htmlSha256:sha256(html),close:()=>server.close()};
}

export async function isolatedWorld(native,page) {
  const session=await native.context.newCDPSession(page);
  const frame=(await session.send('Page.getFrameTree')).frameTree.frame.id;
  const {executionContextId}=await session.send('Page.createIsolatedWorld',{frameId:frame,worldName:'aethervsr-m108-research'});
  const evaluate=async expression=>{
    const result=await session.send('Runtime.evaluate',{contextId:executionContextId,expression,returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;
  };
  await evaluate(`${geometryOracleSource().source};globalThis.__M108_PROOF__=M107Geometry.inspectGeometry;globalThis.__M108_LEASE__=${leaseAnchor.toString()};`);
  return {evaluate,call:(fn,arg)=>evaluate(`(${fn.toString()})(${JSON.stringify(arg??null)})`),close:()=>session.detach()};
}

export async function runElementary(prefix) {
  prefix=resolve(prefix);assert(prefix.startsWith(join(ROOT,'.cache/m108/'))&&!existsSync(`${prefix}.json`));mkdirSync(dirname(prefix),{recursive:true});
  const report={phase:'ELEMENTARY_ANCHOR_RESEARCH',identity:studyIdentity(),environment:presentationEnvironment(),started:new Date().toISOString()};
  let server,native;
  try{
    server=await studyServer();report.fixtureSha256=server.htmlSha256;native=await openNativeChrome(['--autoplay-policy=no-user-gesture-required']);
    report.browser={version:await native.browser.version(),executableSha256:sha256(readFileSync(native.executable))};
    const page=await native.context.newPage();report.placement=await nativeWindow(page,native.context);await page.goto(server.url);await page.bringToFront();
    const world=await isolatedWorld(native,page);
    try{report.result=await world.call(()=>{
      const declarations=[['anchor-name','--aethervsr-test'],['anchor-name','--page-one, --page-two'],['position-anchor','--aethervsr-test'],
        ['left','anchor(left)'],['top','anchor(top)'],['width','anchor-size(width)'],['height','anchor-size(height)']];
      const support=declarations.map(([property,value])=>({property,value,supported:CSS.supports(property,value)}));
      if(support.some(row=>!row.supported))return {support,outcome:'UNSUPPORTED_SAFE',mutated:false};
      const video=document.querySelector('video'),token=`--aethervsr-${crypto.randomUUID()}`,before=video.getAttribute('style');
      const lease=globalThis.__M108_LEASE__(video,token),canvas=document.createElement('canvas');
      for(const[property,value]of Object.entries({all:'initial',position:'fixed',display:'block','pointer-events':'none','position-anchor':token,left:'anchor(left)',top:'anchor(top)',width:'anchor-size(width)',height:'anchor-size(height)'}))canvas.style.setProperty(property,value,'important');
      video.after(canvas);const rect=node=>{const value=node.getBoundingClientRect();return {left:value.left,top:value.top,width:value.width,height:value.height};};
      const actual=rect(canvas),expected=rect(video);canvas.remove();const restored=lease.release();
      return {support,token,validIdentifier:CSS.escape(token)===token,before,actual,expected,restored,mainPrivilegedBridge:false,
        outcome:Object.keys(expected).every(key=>Math.abs(expected[key]-actual[key])<=.5)&&restored.exactOriginal?'ELEMENTARY_PASS':'ELEMENTARY_FAIL'};
    });}finally{await world.close();}
    assert.deepEqual(studyIdentity(),report.identity);
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

export function writeTrace(path,trace) {
  const bytes=gzipSync(JSON.stringify(trace));writeFileSync(path,bytes,{flag:'wx'});return {path:relative(ROOT,path),sha256:sha256(bytes),bytes:bytes.length};
}