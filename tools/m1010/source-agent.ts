export type Snapshot={
	ownerId: string|null; generation: number;
	sourceClass: 'progressive'|'blob-unknown'|'srcObject'|'none';
	url: string|null; width: number; height: number; protected: boolean;
	currentTime: number|null; duration: number|null;
	paused: boolean; ended: boolean; rate: number; volume: number; muted: boolean;
};
type Binding={ownerId: string; generation: number};
export type AgentCommand=Binding&{requestId: string}&(
	{type: 'play'|'pause'}|{type: 'seek'|'rate'|'volume'; value: number}
	|{type: 'mute'; value: boolean}
);
type Stamp=ReturnType<typeof stamp>;
type CaptureVideo=HTMLVideoElement&{captureStream?: () => MediaStream};
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const maxSdp=256*1024;

function fields(raw: unknown,keys: string[],exact=true) {
	try {
		if(!raw||typeof raw!=='object') return null;
		const prototype: unknown=Object.getPrototypeOf(raw);
		if(prototype!==Object.prototype&&prototype!==null) return null;
		if(exact&&Reflect.ownKeys(raw).length!==keys.length) return null;
		const data: Record<string,unknown>={};
		for(const key of keys) {
			const field=Object.getOwnPropertyDescriptor(raw,key);
			if(!field?.enumerable||!Object.hasOwn(field,'value')) return null;
			data[key]=field.value;
		}
		return data;
	} catch {return null;}
}

function bounded(value: unknown,min: number,max: number): value is number {
	return typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max;
}

export function parseAgentCommand(raw: unknown,snap: Snapshot) {
	const type=fields(raw,['type'],false)?.type;
	if(typeof type!=='string'||!['play','pause','seek','rate','volume','mute'].includes(type)) return null;
	const data=fields(raw,['ownerId','generation','requestId','type',
		...(type==='play'||type==='pause'? []:['value'])]);
	if(!data||!snap.ownerId||snap.protected||data.type!==type
		||data.ownerId!==snap.ownerId||data.generation!==snap.generation
		||!uuid.test(snap.ownerId)||!Number.isSafeInteger(snap.generation)
		||snap.generation<0||typeof data.requestId!=='string'||!uuid.test(data.requestId)) return null;
	const valid=type==='play'||type==='pause'
		||(type==='seek'&&bounded(snap.duration,0,Number.MAX_VALUE)&&bounded(data.value,0,snap.duration))
		||(type==='rate'&&bounded(data.value,0.25,4))
		||(type==='volume'&&bounded(data.value,0,1))||(type==='mute'&&typeof data.value==='boolean');
	return valid? data as unknown as AgentCommand:null;
}

export function sameSource(left: Stamp,right: Stamp) {
	return left.currentSrc===right.currentSrc&&left.srcObject===right.srcObject
		&&left.videoWidth===right.videoWidth&&left.videoHeight===right.videoHeight;
}

function connected(video: HTMLVideoElement,doc: Document) {
	return video.isConnected&&video.getRootNode()===doc;
}

export function selectLargestVideo(doc: Document) {
	const videos=doc.getElementsByTagName('video');
	let best: HTMLVideoElement|null=null;
	let largest=0;
	for(let index=0;index<Math.min(videos.length,16);index++) {
		const video=videos[index];
		if(!video||!connected(video,doc)) continue;
		const rect=video.getBoundingClientRect();
		const area=rect.width*rect.height;
		if(rect.width>0&&rect.height>0&&Number.isFinite(area)&&area>largest) {
			best=video;
			largest=area;
		}
	}
	return best;
}

function stamp(video: HTMLVideoElement) {
	return {
		currentSrc: video.currentSrc,srcObject: video.srcObject,
		videoWidth: video.videoWidth,videoHeight: video.videoHeight
	};
}
function finite(value: number) {return Number.isFinite(value)? value:null;}
function errorInfo(error: unknown) {
	return error instanceof Error? {name: error.name.slice(0,128),message: error.message.slice(0,512)}
		:{name: 'Error',message: 'Non-Error rejection'};
}
function cloneTest(action: () => void) {
	try {action(); return {ok: true as const};} catch(error) {return {ok: false as const,error: errorInfo(error)};}
}
function testClones(stream: MediaStream) {
	return {
		structuredClone: cloneTest(() => {
			const clone=structuredClone(stream);
			if(clone!==stream) for(const track of clone.getTracks()) {
				if(!stream.getTracks().includes(track)) track.stop();
			}
		}),
		messageChannel: cloneTest(() => {
			const channel=new MessageChannel();
			try {channel.port1.postMessage(stream);}
			finally {channel.port1.close(); channel.port2.close();}
		}),
	};
}
function sdpValid(sdp: unknown): sdp is string {
	return typeof sdp==='string'&&!!sdp&&sdp.length<=maxSdp
		&&new TextEncoder().encode(sdp).byteLength<=maxSdp;
}

type Session=Binding&{
	video: HTMLVideoElement; stream: MediaStream; peer: RTCPeerConnection;
	offered: boolean;
	cancel: (reason: string) => void; unlisten: () => void; unwatch: () => void;
};

export function createSourceAgent(doc: Document,invalidated: (binding: Binding) => void=() => {},
	watchCapture: (binding: Binding,lost: () => void) => () => void=() => () => {}) {
	const view=doc.defaultView;
	let video: CaptureVideo|null=null;
	let source: Stamp|null=null;
	let owner: string|null=null;
	let gen=0,retired=false,encrypted=false,unavailable=false;
	let session: Session|null=null;
	let observer: MutationObserver|null=null;
	const seen=new Set<string>();
	const events=['loadstart','resize','loadedmetadata','emptied','encrypted','error'];
	function release(reason: string) {
		const owned=session;
		session=null;
		if(!owned) return;
		owned.cancel(reason);
		owned.unlisten();
		owned.unwatch();
		owned.peer.close();
		for(const track of owned.stream.getTracks()) track.stop();
	}
	function invalidate(reason: string) {
		const binding=owner? {ownerId: owner,generation: gen}:null;
		gen++;
		release(reason);
		if(binding) {try {invalidated(binding);} catch {return;}}
	}
	function refresh(event?: Event) {
		if(!video) return;
		if(!connected(video,doc)) {stop(); return;}
		const next=stamp(video);
		if(event?.type==='encrypted') encrypted=true;
		const blocked=Boolean(video.mediaKeys||video.error||encrypted);
		if(!source||!sameSource(source,next)||(blocked&&!unavailable)||['loadstart','emptied','encrypted','error'].includes(event?.type??'')) {
			source=next;
			invalidate('source-changed');
		}
		unavailable=blocked;
	}
	function read(): Snapshot {
		refresh();
		const url=video?.currentSrc??'';
		const progressive=/^https?:\/\//.test(url);
		return {
			ownerId: owner,generation: gen,
			sourceClass: video?.srcObject? 'srcObject':url.startsWith('blob:')? 'blob-unknown'
				:progressive? 'progressive':'none',
			url: progressive&&url.length<=4096? url:null,
			width: video?.videoWidth??0,height: video?.videoHeight??0,
			protected: Boolean(video?.mediaKeys||encrypted),
			currentTime: video? finite(video.currentTime):null,duration: video? finite(video.duration):null,
			paused: video?.paused??true,ended: video?.ended??false,rate: video?.playbackRate??1,
			volume: video?.volume??1,muted: video?.muted??false
		};
	}
	function stop() {
		if(video) invalidate('stopped');
		observer?.disconnect();
		observer=null;
		for(const name of events) video?.removeEventListener(name,refresh);
		video=null; source=null; owner=null; encrypted=false; unavailable=false;
		return read();
	}
	function select() {
		stop();
		if(retired||!view||view.top!==view) throw new Error('inactive-document');
		video=selectLargestVideo(doc);
		if(video) {
			owner=crypto.randomUUID(); gen++; source=stamp(video);
			for(const name of events) video.addEventListener(name,refresh);
			observer=new MutationObserver(() => {refresh();});
			observer.observe(doc,{childList: true,subtree: true});
		}
		return read();
	}
	function current(binding: Binding,target: HTMLVideoElement) {
		refresh();
		return video===target&&owner===binding.ownerId&&gen===binding.generation
			&&!retired&&!target.mediaKeys&&!target.error&&!encrypted;
	}
	async function command(raw: unknown) {
		const cmd=parseAgentCommand(raw,read());
		const id=fields(raw,['requestId'],false)?.requestId;
		const ack=(ok: boolean,reason: string|null) => ({
			requestId: typeof id==='string'&&uuid.test(id)? id:null,ok,reason,snapshot: read()
		});
		if(!cmd||!video) return ack(false,'invalid-command');
		const target=video;
		if(!current(cmd,target)) return ack(false,'stale-source');
		const nonce=cmd.requestId.toLowerCase();
		if(seen.has(nonce)) return ack(false,'duplicate-request');
		if(seen.size>=128) return ack(false,'request-capacity');
		seen.add(nonce);
		try {
			if(cmd.type==='play') await target.play();
			else if(cmd.type==='pause') target.pause();
			else if('value' in cmd) Object.assign(target,{
				[{seek: 'currentTime',rate: 'playbackRate',volume: 'volume',mute: 'muted'}[cmd.type]]: cmd.value,
			});
			return current(cmd,target)? ack(true,null):ack(false,'stale-source');
		} catch(error) {
			return ack(false,errorInfo(error).name);
		}
	}
	function check(owned: Session) {
		if(!current(owned,owned.video)||session!==owned) throw new Error('stale-source');
	}
	async function deadline<Result>(owned: Session,work: () => Promise<Result>) {
		let timer: ReturnType<typeof setTimeout>|undefined;
		try {
			return await new Promise<Result>((resolve,reject) => {
				owned.cancel=reason => {reject(new Error(reason));};
				timer=setTimeout(() => {if(session===owned) release('rtc-timeout');},3000);
				void work().then(resolve,reject);
			});
		} catch(error) {
			if(session===owned) release('rtc-failed');
			throw error;
		} finally {
			clearTimeout(timer);
			owned.cancel=() => {};
			owned.unlisten();
			owned.unlisten=() => {};
		}
	}
	async function captureOffer() {
		const snap=read();
		if(!video||!snap.ownerId||snap.protected||video.error) throw new Error('media-unavailable');
		if(session) throw new Error('capture-active');
		if(!video.captureStream) throw new Error('captureStream-unavailable');
		const target=video;
		const stream=video.captureStream();
		let peer: RTCPeerConnection;
		try {peer=new RTCPeerConnection({iceServers: []});}
		catch(error) {for(const track of stream.getTracks()) track.stop(); throw error;}
		const owned: Session={
			ownerId: snap.ownerId,generation: snap.generation,video: target,
			stream,peer,offered: false,cancel: () => {},unlisten: () => {},unwatch: () => {}
		};
		session=owned;
		return deadline(owned,async () => {
			owned.unwatch=watchCapture(owned,() => {if(session===owned) release('broker-lost');});
			check(owned);
			const tracks=stream.getTracks();
			if(!tracks.some(track => track.kind==='video'&&track.readyState==='live')) {
				throw new Error('no-live-video-track');
			}
			const cloneTests=testClones(stream);
			for(const track of tracks) peer.addTrack(track,stream);
			const offer=await peer.createOffer();
			check(owned);
			await peer.setLocalDescription(offer);
			check(owned);
			await new Promise<void>(resolve => {
				const check=() => {if(peer.iceGatheringState==='complete') resolve();};
				peer.addEventListener('icegatheringstatechange',check);
				owned.unlisten=() => {peer.removeEventListener('icegatheringstatechange',check); resolve();};
				check();
			});
			check(owned);
			const local=peer.localDescription;
			if(local?.type!=='offer'||!sdpValid(local.sdp)) throw new Error('invalid-local-sdp');
			owned.offered=true;
			return {
				ownerId: owned.ownerId,generation: owned.generation,type: 'offer' as const,sdp: local.sdp,
				tracks: tracks.map(track => ({kind: track.kind,readyState: track.readyState})),cloneTests
			};
		});
	}
	async function acceptAnswer(raw: unknown) {
		refresh();
		const answer=fields(raw,['ownerId','generation','sdp','type']);
		const owned=session;
		if(!answer||!owned||!owned.offered||answer.ownerId!==owned.ownerId
			||answer.generation!==owned.generation||answer.type!=='answer'||!sdpValid(answer.sdp)) {
			throw new Error('invalid-answer');
		}
		check(owned);
		owned.offered=false;
		const description: RTCSessionDescriptionInit={type: 'answer',sdp: answer.sdp};
		await deadline(owned,async () => {
			await owned.peer.setRemoteDescription(description);
			check(owned);
		});
	}
	view?.addEventListener('pagehide',() => {retired=true; stop();},{once: true});
	return Object.freeze({select,read,command,captureOffer,acceptAnswer,stop,
		releaseCapture: () => {release('capture-stopped'); return read();}});
}

export type SourceAgent=ReturnType<typeof createSourceAgent>;
export type Ack=Awaited<ReturnType<SourceAgent['command']>>;
export type CaptureOffer=Awaited<ReturnType<SourceAgent['captureOffer']>>;
declare global {var __M1010_AGENT__: SourceAgent|undefined;}
if(typeof window!=='undefined'&&window.top===window&&!globalThis.__M1010_AGENT__) {
	globalThis.__M1010_AGENT__=createSourceAgent(document,binding => {
		void chrome.runtime.sendMessage({type: 'acquire.source-invalid',...binding}).catch(() => {});
	},(binding,lost) => {
		const port=chrome.runtime.connect({name: `m1010-source:${binding.ownerId}:${binding.generation}`});
		port.onDisconnect.addListener(lost);
		return () => {port.onDisconnect.removeListener(lost); port.disconnect();};
	});
}