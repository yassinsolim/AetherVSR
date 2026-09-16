import {afterEach,describe,expect,it,vi} from 'vitest';
import {createSourceAgent,parseAgentCommand as parse,sameSource,selectLargestVideo} from '../tools/m1010/source-agent';

const ownerId='12345678-1234-4123-8123-123456789abc';
const requestId='12345678-1234-4123-9123-123456789abc';
const selected={
	ownerId,generation: 1,sourceClass: 'blob-unknown' as const,url: null,
	width: 640,height: 360,protected: false,currentTime: 0,duration: 10,
	paused: true,ended: false,rate: 1,volume: 1,muted: false
};
const cmd=(type='pause',extra={}) => ({ownerId,generation: 1,requestId,type,...extra});
afterEach(() => {vi.unstubAllGlobals(); vi.useRealTimers();});

describe('commands',() => {
	it.each([['play'],['pause'],['seek',0],['seek',10],['rate',0.25],['rate',4],
	['volume',0],['volume',1],['mute',false]] as const)('accepts %s',(type,value?) => {
		const raw=cmd(type,value===undefined? {}:{value});
		expect(parse(raw,selected)).toEqual(raw);
	});
	it.each([{type: 'eval'},{value: 1},{ownerId: 'other'},{generation: 2},
	{requestId: 'bad'},{type: 'seek',value: -1},{type: 'seek',value: 11},
	{type: 'seek',value: NaN},{type: 'rate',value: Infinity},{type: 'rate',value: 0.2},
	{type: 'rate',value: 4.1},{type: 'volume',value: 1.1},{type: 'volume',value: '1'},
	{type: 'mute',value: 1},{url: 'https://bad.test'},{[Symbol('extra')]: 1}])('rejects %j',extra => {
		expect(parse(cmd('pause',extra),selected)).toBeNull();
	});
	it('rejects unsafe inputs',() => {
		const getter=vi.fn(() => 'pause');
		expect(parse(cmd('pause',{type: {toString: getter}}),selected)).toBeNull();
		expect(parse({...cmd(),get type() {return getter();} },selected)).toBeNull();
		expect(getter).not.toHaveBeenCalled();
		expect(parse(Object.create(cmd()),selected)).toBeNull();
		expect(parse(null,selected)).toBeNull();
		expect(parse(cmd('seek',{value: 1}),{...selected,duration: null})).toBeNull();
		expect(parse(cmd(),{...selected,protected: true})).toBeNull();
	});
});

function deferred() {
	let resolve!: () => void;
	const promise=new Promise<void>(done => {resolve=done;});
	return {promise,resolve};
}

function fixture() {
	let mutation: () => void=() => {};
	vi.stubGlobal('MutationObserver',class {
		constructor(callback: () => void) {mutation=callback;}
		observe() {}
		disconnect() {}
	});
	const view: EventTarget&{top: unknown}=Object.assign(new EventTarget(),{top: null});
	view.top=view;
	const doc={defaultView: view,getElementsByTagName: () => videos} as unknown as Document;
	const track={kind: 'video',readyState: 'live',stop: vi.fn()};
	const audio={kind: 'audio',readyState: 'live',stop: vi.fn()};
	const stream={getTracks: () => [track,audio]} as unknown as MediaStream;
	const target=Object.assign(new EventTarget(),{
		isConnected: true,ownerDocument: doc,getRootNode: () => doc,
		getBoundingClientRect: () => ({width: 640,height: 360}),
		currentSrc: 'blob:x',srcObject: null,videoWidth: 640,videoHeight: 360,
		currentTime: 0,duration: 10,playbackRate: 1,volume: 1,muted: false,
		paused: false,ended: false,error: null,mediaKeys: null,
		play: vi.fn(() => {target.paused=false; return Promise.resolve();}),pause: vi.fn(() => {target.paused=true;}),
		captureStream: vi.fn(() => stream),
	});
	const videos=[target as unknown as HTMLVideoElement];
	const peer=Object.assign(new EventTarget(),{
		iceGatheringState: 'complete',localDescription: {type: 'offer',sdp: 'v=0\r\n'},
		createOffer: vi.fn(() => Promise.resolve({type: 'offer',sdp: 'v=0\r\n'})),
		setLocalDescription: vi.fn(() => Promise.resolve()),setRemoteDescription: vi.fn(() => Promise.resolve()),
		close: vi.fn(),addTrack: vi.fn(),
	});
	const rtc=vi.fn(function() {return peer;});
	vi.stubGlobal('RTCPeerConnection',rtc);
	const closePort=vi.fn();
	vi.stubGlobal('MessageChannel',class {
		port1={postMessage() {throw new DOMException('stream','DataCloneError');},close: closePort};
		port2={close: closePort};
	});
	const invalidated=vi.fn();
	let lost=() => {};
	const unwatch=vi.fn();
	const watchCapture=vi.fn((_binding: unknown,listener: () => void) => {lost=listener; return unwatch;});
	const agent=createSourceAgent(doc,invalidated,watchCapture);
	const binding=agent.select();
	const send=(type='pause',extra={}) => agent.command({
		ownerId: binding.ownerId,
		generation: binding.generation,requestId,type,...extra
	});
	return {
		agent,binding,send,target,videos,doc,peer,rtc,track,audio,view,closePort,invalidated,unwatch,
		loseBroker: () => lost(),
		mutation: () => {mutation();}
	};
}

describe('ownership',() => {
	it.each(['replace','remove','resize','encrypted'] as const)('notifies the retired binding on %s without a read or active source RTC',change => {
		const {agent,binding,target,mutation,invalidated}=fixture();
		if(change==='replace'||change==='remove') {target.isConnected=false; mutation();}
		else if(change==='resize') {target.videoWidth=1280; target.dispatchEvent(new Event('resize'));}
		else target.dispatchEvent(new Event('encrypted'));
		expect(invalidated).toHaveBeenCalledWith({ownerId: binding.ownerId,generation: binding.generation});
		expect(agent.read().generation).toBeGreaterThan(binding.generation);
		agent.stop();
	});
	it('bounds top-document selection to sixteen',() => {
		const {agent,videos,target,doc}=fixture();
		const beyondLimit=vi.fn(() => ({width: 10000,height: 1000}));
		for(let index=1;index<17;index++) videos.push({
			...target,
			getBoundingClientRect: index===16? beyondLimit:() => ({width: index*100,height: 1000}),
		} as unknown as HTMLVideoElement);
		expect(selectLargestVideo(doc)).toBe(videos[15]);
		expect(beyondLimit).not.toHaveBeenCalled();
		Object.assign(videos[15]!,{isConnected: false});
		Object.assign(videos[14]!,{getRootNode: () => ({})});
		expect(selectLargestVideo(doc)).toBe(videos[13]);
		agent.stop();
	});
	it('tracks source generations',async () => {
		const {agent,binding,target,send}=fixture();
		expect(sameSource(target,{...target})).toBe(true);
		for(const patch of [{currentSrc: 'blob:y'},{srcObject: {} as MediaStream},{videoHeight: 720}]) {
			expect(sameSource(target,{...target,...patch})).toBe(false);
		}
		target.videoWidth=1280;
		const ack=await send();
		expect(ack.ok).toBe(false);
		expect(ack.snapshot.ownerId).toBe(binding.ownerId);
		expect(ack.snapshot.generation).toBeGreaterThan(binding.generation);
		expect(target.pause).not.toHaveBeenCalled();
		target.dispatchEvent(new Event('loadstart'));
		expect((await agent.command(null)).snapshot.generation).toBeGreaterThan(ack.snapshot.generation);
		expect(agent.select().ownerId).not.toBe(binding.ownerId);
		agent.stop();
	});
	it('controls host; retains replay IDs',async () => {
		const {agent,send,target}=fixture();
		expect((await send()).snapshot.paused).toBe(true);
		expect((await send('pause',{requestId: requestId.toUpperCase()})).reason).toBe('duplicate-request');
		for(let index=1;index<128;index++) {
			expect((await send('volume',{requestId: crypto.randomUUID(),value: 0.5})).ok).toBe(true);
		}
		expect(target.volume).toBe(0.5);
		expect((await send('play',{requestId: crypto.randomUUID()})).reason).toBe('request-capacity');
		const next=agent.select();
		expect((await agent.command({...cmd(),ownerId: next.ownerId,generation: next.generation})).reason)
			.toBe('duplicate-request');
		agent.stop();
	});
	it('reports failure, redacts data, stops lifecycle',async () => {
		const {agent,send,target,mutation,view}=fixture();
		target.play.mockRejectedValue(new DOMException('not activated','NotAllowedError'));
		expect((await send('play')).reason).toContain('NotAllowedError');
		target.currentSrc='data:,private';
		expect((await agent.command(null)).snapshot.url).toBeNull();
		target.isConnected=false;
		mutation();
		expect((await agent.command(null)).snapshot.ownerId).toBeNull();
		view.dispatchEvent(new Event('pagehide'));
		expect(() => agent.select()).toThrow('inactive-document');
		expect(target.pause).not.toHaveBeenCalled();
	});
});

describe('RTC lifecycle',() => {
	it.each(['pending','active'] as const)('releases %s source RTC resources on broker loss',async stage => {
		const {agent,peer,track,audio,loseBroker,unwatch}=fixture(),pending=deferred();
		if(stage==='pending') peer.createOffer.mockImplementation(() => pending.promise.then(() => ({type: 'offer',sdp: 'v=0\r\n'})));
		const offer=agent.captureOffer().catch((error: unknown) => error);
		if(stage==='active') await offer;
		loseBroker(); pending.resolve();
		if(stage==='pending') expect(await offer).toBeInstanceOf(Error);
		expect(peer.close).toHaveBeenCalledOnce(); expect(track.stop).toHaveBeenCalledOnce(); expect(audio.stop).toHaveBeenCalledOnce();
		expect(unwatch).toHaveBeenCalledOnce();
		agent.stop();
	});
	it('offers, answers, cleans up',async () => {
		const {agent,binding,peer,rtc,track,audio,target,closePort}=fixture();
		const offer=await agent.captureOffer();
		expect(rtc).toHaveBeenCalledWith({iceServers: []});
		expect(offer.ownerId).toBe(binding.ownerId);
		for(const result of Object.values(offer.cloneTests)) {
			expect(result).toMatchObject({ok: false,error: {name: 'DataCloneError'}});
		}
		expect(closePort).toHaveBeenCalledTimes(2);
		expect(peer.addTrack).toHaveBeenCalledTimes(2);
		const answer={ownerId: offer.ownerId,generation: offer.generation,type: 'answer',sdp: 'v=0\r\n'};
		for(const extra of [{generation: 99},{type: 'offer'},{sdp: ''},{iceServers: []},
		{sdp: 'x'.repeat(262145)},{sdp: '\u00e9'.repeat(131073)}]) {
			await expect(agent.acceptAnswer({...answer,...extra})).rejects.toThrow('invalid-answer');
		}
		expect(peer.setRemoteDescription).not.toHaveBeenCalled();
		await agent.acceptAnswer(answer);
		await expect(agent.acceptAnswer(answer)).rejects.toThrow('invalid-answer');
		agent.stop();
		expect(peer.close).toHaveBeenCalledOnce();
		expect(track.stop).toHaveBeenCalledOnce();
		expect(audio.stop).toHaveBeenCalledOnce();
		expect(target.pause).not.toHaveBeenCalled();
	});
	it('bounds ICE gathering to 3000ms',async () => {
		vi.useFakeTimers();
		const {agent,peer,track}=fixture();
		peer.iceGatheringState='gathering';
		const offer=agent.captureOffer();
		const rejected=expect(offer).rejects.toThrow('rtc-timeout');
		await vi.advanceTimersByTimeAsync(3000);
		await rejected;
		expect(peer.close).toHaveBeenCalledOnce();
		expect(track.stop).toHaveBeenCalledOnce();
		expect(vi.getTimerCount()).toBe(0);
		agent.stop();
	});
	it.each(['offer','answer'] as const)('rejects stale %s completion',async stage => {
		const {agent,peer,target,track,binding,invalidated}=fixture();
		const pending=deferred();
		let result: Promise<unknown>;
		if(stage==='offer') {
			peer.createOffer.mockImplementation(() => pending.promise.then(() => ({type: 'offer',sdp: 'v=0\r\n'})));
			result=agent.captureOffer();
		} else {
			const offer=await agent.captureOffer();
			peer.setRemoteDescription.mockReturnValue(pending.promise);
			result=agent.acceptAnswer({ownerId: offer.ownerId,generation: offer.generation,type: 'answer',sdp: 'v=0\r\n'});
		}
		const rejected=expect(result).rejects.toThrow('source-changed');
		target.currentSrc='blob:replacement';
		target.dispatchEvent(new Event('loadstart'));
		expect(invalidated).toHaveBeenCalledWith({ownerId: binding.ownerId,generation: binding.generation});
		pending.resolve();
		await rejected;
		if(stage==='offer') expect(peer.setLocalDescription).not.toHaveBeenCalled();
		expect(track.stop).toHaveBeenCalledOnce();
		agent.stop();
	});
	it.each(['mediaKeys','error'] as const)('rejects %s',async property => {
		const {agent,target}=fixture();
		Object.assign(target,{[property]: {}});
		await expect(agent.captureOffer()).rejects.toThrow('media-unavailable');
		expect(target.captureStream).not.toHaveBeenCalled();
		agent.stop();
	});
});