import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {join,resolve,relative,dirname} from 'node:path';
import {gzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {createServer} from 'vite';
import {ROOT,sha256} from './m10-fixtures.mjs';
import {openNativeChrome} from './m9-browser.mjs';
import {openExtension,until,bounded} from './m10-browser.mjs';
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
  let image=null;
  if(source.width>0&&source.height>0){
    const factor=state.fit==='cover'?Math.max(video.width/source.width,video.height/source.height):Math.min(video.width/source.width,video.height/source.height);
    image={width:source.width*factor,height:source.height*factor};
    image.left=video.left+(video.width-image.width)*state.position[0];image.top=video.top+(video.height-image.height)*state.position[1];
  }
  return {visible,rect:{...video},clip:{left,top,width:Math.max(0,right-left),height:Math.max(0,bottom-top)},image,
    originalLayoutVisible:!state.hidden&&video.width>0&&video.height>0&&right>left&&bottom>top,
    captionAboveReplacement:!state.caption||state.captionZ>0,
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
  const normalizer=document.createElement('div').style;normalizer.setProperty('anchor-name',appended);
  const canonical=normalizer.getPropertyValue('anchor-name');
  if(!canonical)throw new Error('Anchor name list did not parse');
  let appliedPriority=priority;
  video.style.setProperty('anchor-name',appended,appliedPriority);
  if(getComputedStyle(video).getPropertyValue('anchor-name')!==canonical){
    appliedPriority='important';video.style.setProperty('anchor-name',appended,appliedPriority);
  }
  const applied=video.getAttribute('style');
  if(getComputedStyle(video).getPropertyValue('anchor-name')!==canonical){
    if(before===null)video.removeAttribute('style');else video.setAttribute('style',before);
    throw new Error('Anchor declaration did not resolve');
  }
  let released=false;
  return {token,before,value,priority,resolved,appended,appliedPriority,applied,
    release(){
      if(released)return {repeated:true};released=true;
      const current=video.getAttribute('style'),unchanged=current===applied;
      if(unchanged){if(before===null)video.removeAttribute('style');else video.setAttribute('style',before);}
      else if(video.style.getPropertyValue('anchor-name')===canonical&&video.style.getPropertyPriority('anchor-name')===appliedPriority){
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
      const rect=node=>{const value=node.getBoundingClientRect();return {left:value.left,top:value.top,width:value.width,height:value.height};};
      const beforeRect=rect(video),result={support,token,validIdentifier:CSS.escape(token)===token,before,beforeRect,mainPrivilegedBridge:false};
      let lease,canvas;
      try{
        lease=globalThis.__M108_LEASE__(video,token);canvas=document.createElement('canvas');
        for(const[property,value]of Object.entries({all:'initial',position:'fixed',display:'block','pointer-events':'none','position-anchor':token,left:'anchor(left)',top:'anchor(top)',width:'anchor-size(width)',height:'anchor-size(height)'}))canvas.style.setProperty(property,value,'important');
        video.after(canvas);result.actual=rect(canvas);result.expected=rect(video);
      }catch(error){result.error=String(error);}
      finally{canvas?.remove();result.restored=lease?.release()??null;result.afterRect=rect(video);}
      result.outcome=!result.error&&result.restored?.exactOriginal&&Object.keys(beforeRect).every(key=>Math.abs(result.expected[key]-result.actual[key])<=.5&&result.expected[key]===beforeRect[key]&&result.afterRect[key]===beforeRect[key])?'ELEMENTARY_PASS':'ELEMENTARY_FAIL';
      return result;
    });}finally{await world.close();}
    assert.deepEqual(studyIdentity(),report.identity);
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

export function writeTrace(path,trace) {
  const bytes=gzipSync(JSON.stringify(trace));writeFileSync(path,bytes,{flag:'wx'});return {path:relative(ROOT,path),sha256:sha256(bytes),bytes:bytes.length};
}

export function installCandidate({kind,frequency=0,tetherOnly=false}) {
  if(!['A','B','D'].includes(kind)||kind==='D'&&![5,10,15].includes(frequency))throw new Error('Unregistered architecture');
  const video=document.querySelector('video')??document.getElementById('shadow-host')?.shadowRoot.querySelector('video');
  if(!video)throw new Error('Fixture video missing');
  const canvas=document.createElement('canvas');canvas.dataset.m108Owned='true';canvas.setAttribute('aria-hidden','true');
  canvas.style.setProperty('visibility','hidden','important');
  const data={kind,frequency,tetherOnly,guard:[],proof:[],events:[],overflow:false,epoch:0,applied:-1,sourceEpoch:0,outputSource:-1,outputGeometry:-1};
  let active=true,frame=null,pending=null,timer=null,geometry=null,source=null,fullscreen=null,parent=null,readyFrames=0,lease=null;
  let watched=[];const listeners=[];
  const push=(array,value)=>{if(array.length>=8192){data.overflow=true;throw new Error('Prototype telemetry overflow');}array.push(value);};
  const style=(name,value)=>{if(canvas.style.getPropertyValue(name)!==value)canvas.style.setProperty(name,value,'important');};
  const hide=()=>{style('visibility','hidden');readyFrames=0;data.outputSource=-1;data.outputGeometry=-1;};
  const identity=()=>[video.currentSrc,video.videoWidth,video.videoHeight].join('|');
  const originalStyle=video.getAttribute('style');
  if(kind==='B')lease=globalThis.__M108_LEASE__(video,`--aethervsr-${crypto.randomUUID()}`);
  const on=(target,type,callback,capture=false)=>{target.addEventListener(type,callback,capture);listeners.push([target,type,callback,capture]);};
  const observer=new MutationObserver(records=>{
    if(!active||tetherOnly)return;
    if(records.length>256){stop();data.error='Mutation bound exceeded';return;}
    const relevant=records.some(record=>record.target!==canvas&&(record.type!=='childList'||[...record.addedNodes,...record.removedNodes].some(node=>node!==canvas)));
    if(relevant)invalidate('mutation');
  });
  const resize=new ResizeObserver(()=>{if(active&&!tetherOnly)invalidate('resize-observer');});
  function observe() {
    const nodes=[];let node=video;
    while(node&&nodes.length<=32){nodes.push(node);node=node.parentNode??node.host??null;}
    if(node||nodes.length>32)throw new Error('Ancestor bound exceeded');
    for(const entry of geometry?.proof?.styles??[])if(!nodes.includes(entry.element))nodes.push(entry.element);
    if(nodes.length>256)throw new Error('Dependency bound exceeded');
    if(nodes.length===watched.length&&nodes.every((entry,index)=>entry===watched[index]))return;
    observer.disconnect();watched=nodes;
    for(const entry of nodes)observer.observe(entry,{attributes:true,childList:true,attributeFilter:['style','class','hidden','width','height','controls']});
  }
  function invalidate(reason) {
    if(!active)return;
    data.epoch++;hide();push(data.events,{at:performance.now(),reason,epoch:data.epoch});
    window.dispatchEvent(new Event('m108:invalidated'));
    if(pending===null)pending=requestAnimationFrame(()=>{pending=null;prove();});
  }
  function prove() {
    if(!active)return;
    const start=performance.now();hide();
    try{
      if(identity()!==source){source=identity();data.sourceEpoch++;}
      geometry=globalThis.__M108_PROOF__(video);fullscreen=document.fullscreenElement;parent=video.parentNode;observe();
      data.applied=-1;
      if(!geometry.ok||video.paused||video.hidden||document.visibilityState!=='visible')return;
      canvas.style.cssText='';
      for(const[name,value]of Object.entries(geometry.style))canvas.style.setProperty(name,value,'important');
      for(const[name,value]of Object.entries({visibility:'hidden',animation:'none',transition:'none',background:tetherOnly?'transparent':'rgb(32, 130, 110)'}))style(name,value);
      if(kind==='B')for(const[name,value]of Object.entries({'position-anchor':lease.token,left:'anchor(left)',top:'anchor(top)',width:'anchor-size(width)',height:'anchor-size(height)'}))style(name,value);
      canvas.width=video.videoWidth*2;canvas.height=video.videoHeight*2;
      if(video.nextSibling!==canvas)video.after(canvas);
      data.applied=data.epoch;readyFrames=0;
    }finally{push(data.proof,{at:start,ms:performance.now()-start,epoch:data.epoch});}
  }
  function guard() {
    if(!active)return;
    const start=performance.now();
    try{
      if(identity()!==source||document.fullscreenElement!==fullscreen||video.parentNode!==parent||!video.isConnected){invalidate('identity');return;}
      if(video.paused||video.readyState<2||video.hidden||document.visibilityState!=='visible'){hide();return;}
      if(data.applied!==data.epoch||pending!==null||!geometry?.ok)return;
      const current=video.getBoundingClientRect(),actual=canvas.getBoundingClientRect();
      const expected=kind==='B'?current:geometry.rect;
      const changed=video.nextSibling!==canvas||['left','top','width','height'].some(key=>Math.abs(actual[key]-expected[key])>.5||kind!=='B'&&current[key]!==geometry.rect[key]);
      if(changed){invalidate('rect');return;}
      if(++readyFrames>=2){style('visibility','visible');data.outputSource=data.sourceEpoch;data.outputGeometry=data.epoch;}
    }finally{push(data.guard,{at:start,ms:performance.now()-start});}
  }
  function stop() {
    active=false;hide();if(frame!==null)cancelAnimationFrame(frame);if(pending!==null)cancelAnimationFrame(pending);if(timer!==null)clearInterval(timer);
    frame=null;pending=null;timer=null;observer.disconnect();resize.disconnect();
    for(const[target,type,callback,capture]of listeners)target.removeEventListener(type,callback,capture);listeners.length=0;watched=[];
  }
  const periodic=()=>{if(active){data.epoch++;prove();}};
  try{
    if(!tetherOnly){
    on(window,'scroll',()=>invalidate('scroll'),true);on(window,'resize',()=>invalidate('resize'));
    for(const type of['fullscreenchange','visibilitychange'])on(document,type,()=>invalidate(type));
    for(const type of['resize','loadeddata','loadstart','emptied','pause','play','seeking','seeked'])on(video,type,()=>invalidate(type));
    on(window,'m108:safety-trip',stop);
    resize.observe(video);if(video.parentElement)resize.observe(video.parentElement);
  }
    prove();
    if(tetherOnly){style('visibility','visible');observer.disconnect();watched=[];}
    else{
      const tick=()=>{frame=null;guard();if(active)frame=requestAnimationFrame(tick);};frame=requestAnimationFrame(tick);
      if(kind==='D')timer=setInterval(periodic,1000/frequency);
    }
  }catch(error){stop();canvas.remove();lease?.release();throw error;}
  globalThis.__M108_CANDIDATE__={
    forceProof(){if(pending!==null){cancelAnimationFrame(pending);pending=null;}data.epoch++;prove();
      if(kind==='D'){clearInterval(timer);timer=setInterval(periodic,1000/frequency);}return performance.now();},
    snapshot:()=>({...data,active,resources:{frame:Number(frame!==null),proofFrame:Number(pending!==null),timer:Number(timer!==null),mutationObserver:Number(active&&!tetherOnly),resizeObserver:Number(active&&!tetherOnly),listeners:listeners.length}}),
    dispose(){stop();canvas.remove();const restored=lease?.release()??{exactOriginal:video.getAttribute('style')===originalStyle,ownedTokenRemains:false};
      return {restored,resources:{frame:Number(frame!==null),proofFrame:Number(pending!==null),timer:Number(timer!==null),canvas:Number(canvas.isConnected),listeners:listeners.length},telemetry:data};},
  };
  return {kind,frequency,tetherOnly,lease:lease?{token:lease.token,before:lease.before,resolved:lease.resolved,priority:lease.priority,appliedPriority:lease.appliedPriority}:null};
}

export function installFixtureSampler(oracle) {
  const fixture=globalThis.__M108_FIXTURE__,video=fixture.video,rows=[],events=[];
  const canvas=[...document.querySelectorAll('canvas'),...(document.getElementById('shadow-host')?.shadowRoot.querySelectorAll('canvas')??[])][0]??null;
  const allowedStyle=video.getAttribute('style');
  const canonical=document.createElement('div').style;let active=true,handle=null,tripped=false;const listeners=[];
  let trip;const done=new Promise(resolve=>{trip=resolve;});
  const rect=node=>{const value=node.getBoundingClientRect();return {left:value.left,top:value.top,width:value.width,height:value.height};};
  const sample=(reason,boundary=true)=>{
    if(!active)return;
    if(rows.length>=8192){active=false;trip();return;}
    const current=rect(video),fullscreen=document.fullscreenElement;
    const clip=fixture.state.clip&&!fullscreen?rect(fixture.outer):null;
    const expected=oracle({video:current,viewport:{width:document.documentElement.clientWidth,height:document.documentElement.clientHeight},clip,state:fixture.state,
      source:{width:video.videoWidth,height:video.videoHeight},media:{readyState:video.readyState,paused:video.paused}});
    const actual=canvas?rect(canvas):null,computed=canvas?getComputedStyle(canvas):null;
    const visible=!!canvas&&canvas.isConnected&&computed.visibility==='visible'&&computed.display!=='none';
    const tetherError=actual?Math.max(...['left','top','width','height'].map(key=>Math.abs(actual[key]-expected.rect[key]))):null;
    const inset=[expected.clip.top-current.top,current.left+current.width-expected.clip.left-expected.clip.width,current.top+current.height-expected.clip.top-expected.clip.height,expected.clip.left-current.left];
    const expectedMask=fixture.state.radius?`inset(0px round ${expected.radius})`:`inset(${inset.map(value=>`${value}px`).join(' ')})`;
    canonical.cssText='all: initial !important';
    for(const[name,value]of Object.entries({'object-fit':expected.fit,'object-position':expected.position,'border-radius':expected.radius,'clip-path':expectedMask}))canonical.setProperty(name,value,'important');
    if(fixture.state.radius)canonical.setProperty('clip',`rect(${inset[0]}px, ${current.width-inset[1]}px, ${current.height-inset[2]}px, ${inset[3]}px)`,'important');
    const semantic=visible?['object-fit','object-position','border-radius','clip-path','clip'].filter(name=>canvas.style.getPropertyValue(name)!==canonical.getPropertyValue(name)):[];
    const originalVisible=video.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
    const controls=[...fixture.player.querySelectorAll('button')].map(button=>{
      const bounds=button.getBoundingClientRect(),root=button.getRootNode();
      const onscreen=bounds.width>0&&bounds.height>0&&bounds.left>=0&&bounds.top>=0&&bounds.right<=innerWidth&&bounds.bottom<=innerHeight;
      return {onscreen,reachable:!onscreen||root.elementFromPoint(bounds.left+bounds.width/2,bounds.top+bounds.height/2)===button};
    });
    const hostIntact=video.isConnected&&video.parentNode===fixture.expectedParent&&video.getAttribute('style')===allowedStyle&&
      (!expected.originalLayoutVisible||originalVisible)&&controls.every(control=>control.reachable);
    const outputIntact=!visible||computed.pointerEvents==='none'&&(computed.zIndex==='auto'||computed.zIndex==='0')&&video.nextSibling===canvas&&canvas.parentNode===video.parentNode;
    const row={at:performance.now(),reason,boundary,operation:fixture.state.operation,video:current,canvas:actual,expected,visible,tetherError,semantic,
      focused:document.hasFocus(),visibility:document.visibilityState,originalCheckVisibility:originalVisible,hostIntact,outputIntact,controls,
      intrinsic:{width:video.videoWidth,height:video.videoHeight},backing:canvas?{width:canvas.width,height:canvas.height}:null,
      originalStyle:video.getAttribute('style'),originalParent:video.parentNode===fixture.player?'player':'other',fullscreen:fullscreen===fixture.player?'container':fullscreen?'other':'none'};
    row.stale=!hostIntact||!outputIntact||visible&&(!expected.visible||tetherError>.5||semantic.length>0||canvas.width!==video.videoWidth*2||canvas.height!==video.videoHeight*2);
    rows.push(row);
    if(boundary&&row.stale&&!fixture.tetherOnly){tripped=true;active=false;canvas?.style.setProperty('visibility','hidden','important');window.dispatchEvent(new Event('m108:safety-trip'));trip();}
  };
  const on=(target,type,callback)=>{target.addEventListener(type,callback);listeners.push([target,type,callback]);};
  on(window,'m108:operation',event=>{events.push({at:performance.now(),...event.detail});sample('action',false);});
  on(window,'m108:invalidated',()=>sample('invalidated',true));
  on(window,'m108:submitted',()=>sample('submitted',true));
  const frame=()=>{sample('render',true);if(active)handle=requestAnimationFrame(frame);};handle=requestAnimationFrame(frame);
  globalThis.__M108_SAMPLER__={done,sample,get stopFlag(){return tripped||!active;},stop:()=>{sample('stop',false);active=false;cancelAnimationFrame(handle);for(const[target,type,callback]of listeners)target.removeEventListener(type,callback);
    return {rows,events,tripped,overflow:rows.length>=8192,scope:'Fresh fixture-state oracle and DOM rectangles; no inspectGeometry call/cache in oracle. Same-task observations, not pixels/scanout or uninstrumented cost.'};}};
}

export function summarizeCase(trace,actions,{tetherOnly=false,realPipeline=false}={}) {
  assert.equal(trace.overflow,false);assert(trace.rows.length>0);
  const invalid=trace.rows.some(row=>!row.focused||row.visibility!=='visible');
  const failures=trace.rows.filter(row=>row.boundary&&row.stale);
  const outcomes=actions.map(action=>{
    const rows=trace.rows.filter(row=>row.at>=action.settledAt&&row.at<=action.settledAt+500);
    const expected=rows[0]?.expected.visible;
    const correct=rows.filter(row=>row.hostIntact!==false&&(expected?row.reason===(realPipeline?'submitted':'render')&&row.visible&&!row.stale:row.reason==='render'&&!row.visible));
    return {...action,recoveryFrames:correct.length,recovered:correct.length>=2,expectedVisible:expected??null,
      outcome:correct.length<2?'UNVERIFIED':expected?'SUPPORTED_CORRECT':'UNSUPPORTED_SAFE'};
  });
  return {invalid,tetherOnly,samples:trace.rows.length,staleBoundaries:failures.length,firstFailure:failures[0]??null,
    maximumTetherError:trace.rows.some(row=>Number.isFinite(row.tetherError))?Math.max(...trace.rows.filter(row=>Number.isFinite(row.tetherError)).map(row=>row.tetherError)):null,
    semanticMismatchSamples:trace.rows.filter(row=>row.semantic.length).length,actions:outcomes,
    verdict:invalid?'INVALID':tetherOnly?'CHARACTERIZED':failures.length?'SUPPORTED_STALE':outcomes.every(row=>row.recovered)?outcomes.every(row=>row.outcome==='UNSUPPORTED_SAFE')?'UNSUPPORTED_SAFE':'SUPPORTED_CORRECT':'UNVERIFIED_RECOVERY'};
}

export async function runMatrix(prefix,{kind='A',frequency=0,tetherOnly=false,repeats=3}={}) {
  prefix=resolve(prefix);assert(prefix.startsWith(join(ROOT,'.cache/m108/'))&&!existsSync(`${prefix}.json`));mkdirSync(dirname(prefix),{recursive:true});
  const report={phase:tetherOnly?'NONREPLACEMENT_TETHER_CHARACTERIZATION':'BINDING_ARCHITECTURE',kind,frequency,tetherOnly,repeats,
    identity:studyIdentity(),environment:presentationEnvironment(),cases:CASES,results:[],started:new Date().toISOString(),stopped:null};
  let native,server;
  try{
    server=await studyServer();report.fixtureSha256=server.htmlSha256;
    const control=report.identity.controls.find(row=>row.kind===kind);
    native=control?await openExtension({directory:join(ROOT,control.directory),provenance:JSON.parse(readFileSync(join(ROOT,control.directory,'build-provenance.json')))},(name,data)=>{if(name==='browser')report.browser=data;}):await openNativeChrome(['--autoplay-policy=no-user-gesture-required']);
    report.browser??={version:await native.browser.version(),executableSha256:sha256(readFileSync(native.executable))};
    outer:for(let repeat=1;repeat<=repeats;repeat++)for(const entry of CASES){
      const result={name:entry.name,repeat,actions:[]};report.results.push(result);const page=await native.context.newPage();let world;
      try{
        result.placement=await nativeWindow(page,native.context);await page.goto(server.url);await page.bringToFront();
        await bounded(page.evaluate(name=>globalThis.__M108_FIXTURE__.prepare(name),entry.name),6000);
        result.original=await page.evaluate(()=>({style:globalThis.__M108_FIXTURE__.video.getAttribute('style'),html:globalThis.__M108_FIXTURE__.video.outerHTML}));
        if(control){
          const panel=await native.popup(page);let tabId;
          try{tabId=panel.tabId;if(kind==='s2'){
            await native.workerEval(async tab=>chrome.scripting.executeScript({target:{tabId:tab,frameIds:[0]},world:'ISOLATED',files:['content.js']}),tabId);
            await native.isolated(page,()=>globalThis.__AETHERVSR_EXTENSION_TEST__.configure({presentationWatchdog:true}));
          }await panel.click('input[value="baseline"]');await panel.click('#enable');}finally{await panel.dismiss();}
          await until(()=>native.inspect(tabId),state=>state.details?.attachment?.ready,10000);
          await native.isolated(page,()=>{const attachment=globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)].attachment,original=attachment.pipeline.onFrame;
            attachment.pipeline.onFrame=function(tick){try{return original.call(this,tick);}finally{window.dispatchEvent(new Event('m108:submitted'));}};
            window.addEventListener('m108:safety-trip',()=>attachment.destroy(),{once:true});});
          world={call:(fn,arg)=>native.isolated(page,fn,arg),close:async()=>{},control:true};
        }else{world=await isolatedWorld(native,page);result.admission=await world.call(installCandidate,{kind,frequency,tetherOnly});}
        await page.evaluate(value=>{globalThis.__M108_FIXTURE__.tetherOnly=value;},tetherOnly);
        await page.evaluate(new Function(`return (${installFixtureSampler.toString()})(${fixtureOracle.toString()});`));
        await page.evaluate(()=>new Promise(done=>setTimeout(done,250)));
        for(const name of entry.actions){
          if(await page.evaluate(()=>globalThis.__M108_SAMPLER__.stopFlag??false))break;
          const action={name,startedAt:await page.evaluate(()=>performance.now())};result.actions.push(action);
          if(entry.semantic&&!control)action.afterProofAt=await world.call(()=>globalThis.__M108_CANDIDATE__.forceProof());
          if(name==='fullscreen'){await page.locator('#fullscreen').click();await page.waitForFunction(()=>!!document.fullscreenElement,undefined,{timeout:3000});}
          else if(name==='exit')await page.evaluate(()=>document.exitFullscreen());
          else if(name==='viewport'){
            const session=await native.context.newCDPSession(page);try{const current=await session.send('Browser.getWindowForTarget');await session.send('Browser.setWindowBounds',{windowId:current.windowId,bounds:{width:1100,height:903}});}finally{await session.detach();}
          }else action.applied=await bounded(page.evaluate(value=>globalThis.__M108_FIXTURE__.action(value),name),6000);
          action.settledAt=await page.evaluate(()=>globalThis.__M108_FIXTURE__.settle());
          await page.evaluate(()=>Promise.race([globalThis.__M108_SAMPLER__.done,new Promise(done=>setTimeout(done,1100))]));
          if(await page.evaluate(()=>globalThis.__M108_SAMPLER__.stopFlag??false))break;
        }
        const trace=await page.evaluate(()=>globalThis.__M108_SAMPLER__.stop());result.raw=writeTrace(`${prefix}.${repeat}-${entry.name}.json.gz`,trace);
        result.summary=summarizeCase(trace,result.actions,{tetherOnly,realPipeline:!!control});
        if(!control)result.cleanup=await world.call(()=>globalThis.__M108_CANDIDATE__.dispose());
        else result.cleanup=await world.call(()=>{const manager=globalThis[Symbol.for(`aethervsr.m10.document.${chrome.runtime.id}`)],attachment=manager.attachment;attachment?.destroy();return {status:manager.status(),resources:attachment?.snapshot().resources??null};});
        result.restored=await page.evaluate(()=>globalThis.__M108_FIXTURE__.video.getAttribute('style'))===result.original.style;
        if(entry.semantic&&!control&&!tetherOnly){
          for(const action of result.actions){
            const next=result.cleanup.telemetry.proof.find(row=>row.at>action.afterProofAt);
            const observations=trace.rows.filter(row=>row.reason==='render'&&row.at>=action.applied?.at&&row.at<(next?.at??Infinity));
            action.challenge={afterProofAt:action.afterProofAt,appliedAt:action.applied?.at??null,nextProofAt:next?.at??null,renderObservations:observations.length,
              valid:action.applied?.at>=action.afterProofAt&&observations.length>=2};
          }
          if(!result.summary.staleBoundaries&&result.actions.some(action=>!action.challenge.valid))result.summary.verdict='UNVERIFIED_CHALLENGE_ORDER';
        }
        if(!tetherOnly&&(!['SUPPORTED_CORRECT','UNSUPPORTED_SAFE'].includes(result.summary.verdict)||!result.restored)){report.stopped={name:entry.name,repeat,reason:result.summary.verdict,restored:result.restored};break outer;}
      }catch(error){
        result.error=String(error);
        if(!result.raw)try{const trace=await page.evaluate(()=>globalThis.__M108_SAMPLER__?.stop());if(trace?.rows.length){result.raw=writeTrace(`${prefix}.${repeat}-${entry.name}.json.gz`,trace);result.summary=summarizeCase(trace,result.actions,{tetherOnly,realPipeline:!!control});}}catch(failure){result.traceError=String(failure);}
        report.stopped={name:entry.name,repeat,reason:result.summary?.staleBoundaries?'SUPPORTED_STALE':'APPARATUS_OR_ADMISSION_ERROR'};break outer;
      }
      finally{
        if(world&&!world.control&&!result.cleanup)try{result.cleanup=await world.call(()=>globalThis.__M108_CANDIDATE__?.dispose()??{unavailable:true});}catch(error){result.cleanupError=String(error);}
        await world?.close().catch(()=>{});await page.close();
      }
      console.log(JSON.stringify({kind,frequency,tetherOnly,repeat,name:entry.name,verdict:result.summary?.verdict}));
    }
    report.remaining=CASES.length*repeats-report.results.length;assert.deepEqual(studyIdentity(),report.identity);
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

export async function runObservability(prefix) {
  prefix=resolve(prefix);assert(prefix.startsWith(join(ROOT,'.cache/m108/'))&&!existsSync(`${prefix}.json`));mkdirSync(dirname(prefix),{recursive:true});
  const report={phase:'NO_REPLACEMENT_OBSERVABILITY',identity:studyIdentity(),environment:presentationEnvironment(),started:new Date().toISOString(),results:[]};
  let native,server;
  try{
    server=await studyServer();native=await openNativeChrome(['--autoplay-policy=no-user-gesture-required']);
    report.browser={version:await native.browser.version(),executableSha256:sha256(readFileSync(native.executable))};
    for(const entry of CASES.filter(row=>row.semantic)){
      const page=await native.context.newPage();await nativeWindow(page,native.context);await page.goto(server.url);await page.bringToFront();
      try{
        await bounded(page.evaluate(name=>globalThis.__M108_FIXTURE__.prepare(name),entry.name),6000);
        const result=await page.evaluate(async name=>{
          const fixture=globalThis.__M108_FIXTURE__,video=fixture.video;
          const target=name==='cssom-clip'?fixture.player:name==='cssom-controls'?document.getElementById('caption'):video;
          const properties={'cssom-fit':'objectFit','cssom-position':'objectPosition','cssom-clip':'clipPath','cssom-radius':'borderRadius',
            'cssom-controls':'zIndex','cssom-transform':'transform','cssom-opacity':'opacity','cssom-filter':'filter'};
          const property=properties[name],retained=getComputedStyle(target),rect=()=>{const bounds=video.getBoundingClientRect();return [bounds.left,bounds.top,bounds.width,bounds.height];};
          const notifications={mutations:0,resize:0,events:[]},nodes=[video,fixture.player,fixture.outer,document.body,document.documentElement,target];
          const mutation=new MutationObserver(records=>{notifications.mutations+=records.length;});
          for(const node of new Set(nodes))mutation.observe(node,{attributes:true,childList:true,attributeFilter:['style','class','hidden','width','height','controls']});
          const resize=new ResizeObserver(()=>{notifications.resize++;});resize.observe(video);resize.observe(fixture.player);
          const events=['resize','seeking','seeked','pause','play','loadstart','loadeddata'],listener=event=>notifications.events.push(event.type);
          for(const type of events)video.addEventListener(type,listener);
          await new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)));
          notifications.mutations=0;notifications.resize=0;notifications.events=[];
          const before={rect:rect(),value:retained[property]};
          try{
            const action=await fixture.action(name);await new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)));
            return {name,action,before,after:{rect:rect(),retainedValue:retained[property],freshValue:getComputedStyle(target)[property]},notifications,
              focused:document.hasFocus(),visibility:document.visibilityState,canvasCount:document.querySelectorAll('canvas').length};
          }finally{mutation.disconnect();resize.disconnect();for(const type of events)video.removeEventListener(type,listener);}
        },entry.name);
        report.results.push(result);
      }finally{await page.close();}
    }
    assert.deepEqual(studyIdentity(),report.identity);
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

export async function runOwnership(prefix) {
  prefix=resolve(prefix);assert(prefix.startsWith(join(ROOT,'.cache/m108/'))&&!existsSync(`${prefix}.json`));mkdirSync(dirname(prefix),{recursive:true});
  const cases=['absent','empty','noncanonical','inline-names','stylesheet-important','escaped-names','two-videos','forced-collision','exception-cleanup','host-append'];
  const report={phase:'ANCHOR_OWNERSHIP_BINDING',identity:studyIdentity(),environment:presentationEnvironment(),cases,results:[],started:new Date().toISOString(),stopped:null};
  let native,server;
  try{
    server=await studyServer();native=await openNativeChrome(['--autoplay-policy=no-user-gesture-required']);
    report.browser={version:await native.browser.version(),executableSha256:sha256(readFileSync(native.executable))};
    for(const name of cases){
      const page=await native.context.newPage();await nativeWindow(page,native.context);await page.goto(server.url);await page.bringToFront();const world=await isolatedWorld(native,page);
      try{
        const result=await world.call(name=>{
          const video=document.querySelector('video'),sheet=document.getElementById('fixture-style').sheet;
          const box=node=>{const rect=node.getBoundingClientRect();return [rect.left,rect.top,rect.width,rect.height];};
          const declaration=node=>[...node.style].map(key=>[key,node.style.getPropertyValue(key),node.style.getPropertyPriority(key)]);
          if(name==='empty')video.setAttribute('style','');
          if(name==='noncanonical')video.setAttribute('style',' color : red ;   --page-token: 7; anchor-name: --page-one !important;');
          if(name==='inline-names')video.style.setProperty('anchor-name','--page-one, --page-two','important');
          if(name==='stylesheet-important')sheet.insertRule('video { anchor-name: --page-one, --page-two!important; }');
          if(name==='escaped-names')video.style.setProperty('anchor-name',`${CSS.escape('--page,comma')}, ${CSS.escape('--page space')}, ${CSS.escape('--page\\slash')}`);
          const before={attribute:video.getAttribute('style'),declarations:declaration(video),resolved:getComputedStyle(video).getPropertyValue('anchor-name'),rect:box(video)};
          const token=`--aethervsr-${crypto.randomUUID()}`,owned=[],leases=[];let hostState=null,error=null;
          const consumer=document.createElement('div');owned.push(consumer);
          consumer.style.cssText='position:fixed;width:5px;height:5px;pointer-events:none';
          if(!['none',''].includes(before.resolved)){
            consumer.style.setProperty('position-anchor',name==='escaped-names'?CSS.escape('--page,comma'):'--page-one');
            consumer.style.setProperty('left','anchor(right)');consumer.style.setProperty('top','anchor(top)');video.after(consumer);
          }
          const consumerBefore=consumer.isConnected?box(consumer):null;let lease,restored,second=null,during=null;
          try{
            lease=globalThis.__M108_LEASE__(video,token);leases.push(lease);
            during={rect:box(video),resolved:getComputedStyle(video).getPropertyValue('anchor-name'),consumer:consumer.isConnected?box(consumer):null};
            if(name==='forced-collision'){
              let rejected=false;try{globalThis.__M108_LEASE__(video,token);}catch{rejected=true;}second={collisionRejected:rejected};
            }
            if(name==='two-videos'){
              const other=video.cloneNode(false);other.removeAttribute('src');other.style.cssText='position:absolute;left:710px;top:0;width:100px;height:100px';video.parentNode.append(other);owned.push(other);
              const otherBefore=other.getAttribute('style'),otherToken=`--aethervsr-${crypto.randomUUID()}`,otherLease=globalThis.__M108_LEASE__(other,otherToken);leases.push(otherLease);
              const otherRelease=otherLease.release();leases.pop();second={different:otherToken!==token,exact:other.getAttribute('style')===otherBefore,otherRelease,firstStillOwned:getComputedStyle(video).getPropertyValue('anchor-name').includes(token)};
            }
            if(name==='host-append'){
              video.style.setProperty('anchor-name',`${video.style.getPropertyValue('anchor-name')}, --host-added`,'important');
              video.style.setProperty('color','blue');hostState={declarations:declaration(video),resolved:getComputedStyle(video).getPropertyValue('anchor-name')};
            }
            if(name==='exception-cleanup')throw new Error('Injected post-lease failure');
          }catch(failure){error=String(failure);}
          finally{for(const entry of leases.reverse()){const release=entry.release();if(entry===lease)restored=release;}}
          const after={attribute:video.getAttribute('style'),declarations:declaration(video),resolved:getComputedStyle(video).getPropertyValue('anchor-name'),rect:box(video)};
          const consumerAfter=consumer.isConnected?box(consumer):null;for(const node of owned)node.remove();
          const noHostChange=!hostState;
          const secondPass=!second||(name==='forced-collision'?second.collisionRejected:second.different&&second.exact&&second.firstStillOwned);
          const pass=!!restored&&(!error||name==='exception-cleanup')&&(noHostChange?restored.exactOriginal:!restored.ownedTokenRemains&&after.resolved.includes('--host-added')&&video.style.color==='blue')
            &&JSON.stringify(before.rect)===JSON.stringify(during?.rect)&&JSON.stringify(before.rect)===JSON.stringify(after.rect)
            &&JSON.stringify(consumerBefore)===JSON.stringify(during?.consumer)&&JSON.stringify(consumerBefore)===JSON.stringify(consumerAfter)&&secondPass;
          return {name,token,before,during,after,consumerBefore,consumerAfter,hostState,restored,second,error,pass};
        },name);
        report.results.push(result);
        if(!result.pass){report.stopped={name,reason:'RESTORATION_OR_HOST_STATE_NOT_PRESERVED'};break;}
      }finally{await world.close();await page.close();}
    }
    report.remaining=cases.length-report.results.length;assert.deepEqual(studyIdentity(),report.identity);
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}

export async function runWrapperProbe(prefix) {
  prefix=resolve(prefix);assert(prefix.startsWith(join(ROOT,'.cache/m108/'))&&!existsSync(`${prefix}.json`));mkdirSync(dirname(prefix),{recursive:true});
  const report={phase:'C_INTRUSIVENESS_PROBE',identity:studyIdentity(),environment:presentationEnvironment(),started:new Date().toISOString()};let native,server;
  try{
    server=await studyServer();native=await openNativeChrome(['--autoplay-policy=no-user-gesture-required']);
    report.browser={version:await native.browser.version(),executableSha256:sha256(readFileSync(native.executable))};
    const page=await native.context.newPage();await nativeWindow(page,native.context);await page.goto(server.url);await page.bringToFront();
    report.result=await page.evaluate(()=>{
      const fixture=globalThis.__M108_FIXTURE__,video=fixture.video,parent=video.parentNode,next=video.nextSibling,attribute=video.getAttribute('style');
      const sheet=document.getElementById('fixture-style').sheet;sheet.insertRule('.player > video { width:640px;height:360px; }');sheet.insertRule('.player > div > video { width:320px;height:180px; }');
      const before={width:video.getBoundingClientRect().width,selector:parent.querySelector(':scope > video')===video,parentSame:video.parentNode===parent};
      const wrapper=document.createElement('div');wrapper.style.cssText='position:relative;width:640px;height:360px';parent.insertBefore(wrapper,video);wrapper.append(video);
      const during={width:video.getBoundingClientRect().width,selector:parent.querySelector(':scope > video')===video,parentSame:video.parentNode===parent,sameVideo:fixture.video===video};
      parent.insertBefore(video,next);wrapper.remove();
      const after={width:video.getBoundingClientRect().width,selector:parent.querySelector(':scope > video')===video,parentSame:video.parentNode===parent,styleRestored:video.getAttribute('style')===attribute};
      return {before,during,after,requiredInvasiveReparenting:true,restored:after.width===before.width&&after.selector&&after.parentSame&&after.styleRestored};
    });assert.deepEqual(studyIdentity(),report.identity);
  }finally{await native?.close();await server?.close();report.finished=new Date().toISOString();writeFileSync(`${prefix}.json`,JSON.stringify(report,null,2),{flag:'wx'});}
  return report;
}