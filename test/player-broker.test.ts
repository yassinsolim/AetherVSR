import {afterEach,describe,expect,it,vi} from 'vitest';

const ownerId='12345678-1234-4123-8123-123456789abc';
const origin=`chrome-extension://${'a'.repeat(32)}`;
const sourceUrl='http://127.0.0.1:5204/same/A.mp4';
type Sender=chrome.runtime.MessageSender;
type Injection={target: chrome.scripting.InjectionTarget; files?: string[];
	func?: (...args: unknown[]) => unknown; args?: unknown[]};
function event<Args extends unknown[]>() {
	const listeners: ((...args: Args) => unknown)[]=[];
	return {addListener: (listener: (...args: Args) => unknown) => {listeners.push(listener);},
		emit: (...args: Args) => listeners.map(listener => listener(...args))};
}
function deferred<Value=void>() {
	let resolve!: (value: Value) => void;
	const promise=new Promise<Value>(done => {resolve=done;});
	return {promise,resolve};
}
async function denied(result: Promise<unknown>) {expect(await result).toMatchObject({ok: false});}
async function fixture(url=sourceUrl,register=true) {
	vi.resetModules();
	const state={ownerId,generation: 1,url,protected: false};
	const snapshot=() => ({...state});
	const peer={close: vi.fn()};
	const agent={select: vi.fn(snapshot),read: vi.fn(snapshot),
		captureOffer: vi.fn(() => ({ownerId: state.ownerId,sdp: 'offer'})),
		stop: vi.fn(() => {peer.close();}),command: vi.fn()};
	const messages=event<[unknown,Sender,(value: unknown) => void]>();
	const updated=event<[number,{status?: string;url?: string}]>();
	const revoked=event<[chrome.permissions.Permissions]>();
	const inject=async (details: Injection) => {
		if(details.files) return [];
		const result: unknown=await details.func!(...(details.args??[]));
		return [{frameId: 0,documentId: 'selected',result}];
	};
	const api={runtime: {id: 'a'.repeat(32),getURL: (path: string) => `${origin}/${path}`,
		onMessage: messages,sendMessage: vi.fn(() => Promise.resolve())},
		tabs: {query: vi.fn(() => Promise.resolve([{id: 11,url: 'http://127.0.0.1:5204/fixture'}])),
			get: vi.fn((tabId: number) => Promise.resolve({id: tabId,url: 'http://127.0.0.1:5204/fixture'})),
			create: vi.fn(() => Promise.resolve({id: 22})),
			update: vi.fn((tabId: number,properties: {url: string}) => {updated.emit(tabId,{status: 'loading',url: properties.url});return Promise.resolve({id: tabId});}),
			onUpdated: updated,onRemoved: event<[number]>(),sendMessage: vi.fn()},
		scripting: {executeScript: vi.fn(inject)},permissions: {
			contains: vi.fn((grant: unknown) => Promise.resolve(Boolean(grant))),onRemoved: revoked},
		tabCapture: {getMediaStreamId: vi.fn((_options: unknown,
			callback: (id: string) => void) => {callback('stream-id');})}};
	vi.stubGlobal('chrome',api);
	vi.stubGlobal('__M1010_AGENT__',agent);
	await import('../tools/m1010/broker');
	const player: Sender={id: api.runtime.id,url: `${origin}/acquire.html`,origin,frameId: 0,
		documentId: 'player',tab: {id: 22} as chrome.tabs.Tab};
	const send=(raw: string | Record<string, unknown>,sender=player) => new Promise<unknown>(resolve => {
		if(!messages.emit({navigationType: 'navigate',...(typeof raw==='string'? {type: raw}:raw)},sender,resolve).includes(true)) resolve(false);
	});
	const launch=() => send('research.acquire',{id: api.runtime.id,url: `${origin}/launcher.html`});
	if(register) await launch();
	return {state,agent,peer,api,player,send,launch,updated,revoked,inject};
}
afterEach(() => {vi.unstubAllGlobals(); Reflect.deleteProperty(globalThis,'__M1010_BROKER_TRACE__'); Reflect.deleteProperty(globalThis,'__M1010_RESEARCH_SELECT__');});

describe('broker: mocked callbacks, not native grants',() => {
	it('allows worker-private reselection only with an existing local host grant',async () => {
		const {api,send}=await fixture(sourceUrl,false);
		const select=(globalThis as unknown as {__M1010_RESEARCH_SELECT__: (tabId: number) => Promise<unknown>}).__M1010_RESEARCH_SELECT__;
		api.permissions.contains.mockResolvedValueOnce(false);
		await expect(select(11)).rejects.toThrow('Existing local host grant required');
		expect(api.scripting.executeScript).not.toHaveBeenCalled();
		await expect(select(-1)).rejects.toThrow('Existing local host grant required');
		api.tabs.get.mockResolvedValueOnce({id: 11,url: 'https://ungranted.test/'});
		await expect(select(11)).rejects.toThrow('local authoritative fixture');
		expect(await select(11)).toEqual({ok: true,playerTabId: 22});
		expect(await send('acquire.info')).toMatchObject({ok: true,value: {sourceTabId: 11}});
		await denied(send({type: 'research.select',tabId: 11}));
	});
	it('bounds private navigation diagnostics without retaining URLs',async () => {
		const {updated}=await fixture();
		for(let index=0;index<140;index++) updated.emit(99,{status: 'loading',url: 'https://private.test/?token=secret'});
		const read=(globalThis as unknown as {__M1010_BROKER_TRACE__: () => unknown[]}).__M1010_BROKER_TRACE__;
		expect(read()).toHaveLength(128);
		expect(read()).toEqual(Array.from({length: 128},() => ({event: 'updated',tabId: 99,detail: 'loading:other'})));
		read().pop(); expect(read()).toHaveLength(128);
	});
	it('registers; reads retain target',async () => {
		const {send,launch,api,agent}=await fixture(sourceUrl,false);
		expect(await send('acquire.info')).toBe(false);
		api.tabs.query.mockResolvedValueOnce([{id: 11,url: 'https://evil.test/'}]);
		await denied(launch());
		expect(await launch()).toEqual({ok: true,playerTabId: 22});
		expect(await send('acquire.info')).toMatchObject({ok: true,value: {
			selection: {documentId: 'selected',ownerId,url: sourceUrl}}});
		expect(await send('acquire.offer')).toMatchObject({ok: true,value: {ownerId,sdp: 'offer'}});
		expect(agent.select).toHaveBeenCalledOnce();
		const calls=api.scripting.executeScript.mock.calls.map(([details]) => details);
		for(const details of calls.slice(2)) expect(details.target).toEqual({tabId: 11,documentIds: ['selected']});
	});
	it.each([{id: 'foreign'},{origin: 'https://evil.test'},{url: `${origin}/other.html`},
		{frameId: 1},{tab: {id: 99} as chrome.tabs.Tab},{documentId: ''}])('sender %j',async patch => {
		const {send,player,agent}=await fixture();
		expect(await send('acquire.offer',{...player,...patch})).toBe(false);
		expect(agent.read).not.toHaveBeenCalled();
	});
	it('stale document/command',async () => {
		const {send,player,agent}=await fixture(); await send('acquire.info');
		expect(await send('acquire.info',{...player,documentId: 'replacement'})).toBe(false);
		for(const raw of [{type: 'eval',url: sourceUrl},{type: 'acquire.command',command: {
			type: 'pause',requestId: ownerId,url: 'https://evil.test/a.mp4'}}]) await denied(send(raw));
		expect(agent.command).not.toHaveBeenCalled();
	});
	it.each(['ownerId','generation'] as const)('binds %s before action',async key => {
		const {send,agent,state}=await fixture();
		agent.read.mockImplementationOnce(() => {
			const old={...state}; Object.assign(state,{[key]: key==='ownerId'? crypto.randomUUID():2}); return old;
		});
		await denied(send('acquire.offer'));
		expect(agent.captureOffer).not.toHaveBeenCalled();
	});
	it('post-await registration',async () => {
		const {send,api,inject,revoked}=await fixture();
		const entered=deferred(),gate=deferred();
		api.scripting.executeScript.mockImplementationOnce(async details => {
			const result=await inject(details); entered.resolve(); await gate.promise; return result;
		});
		const pending=send('acquire.tab-id'); await entered.promise;
		revoked.emit({permissions: ['tabCapture']}); gate.resolve();
		await denied(pending);
		expect(api.permissions.contains).not.toHaveBeenCalled();
	});
	it.each(['permission','id'])('late %s revoked',async stage => {
		const {send,api,revoked}=await fixture();
		const entered=deferred(),grant=deferred<boolean>(),issued=deferred<(id: string) => void>();
		if(stage==='permission') api.permissions.contains.mockImplementationOnce(() => {entered.resolve(); return grant.promise;});
		else api.tabCapture.getMediaStreamId.mockImplementationOnce((_options,callback) => {issued.resolve(callback); entered.resolve();});
		const pending=send('acquire.tab-id'); await entered.promise;
		revoked.emit({permissions: ['tabCapture']});
		if(stage==='permission') grant.resolve(true); else (await issued.promise)('late-id');
		await denied(pending);
		if(stage==='permission') expect(api.tabCapture.getMediaStreamId).not.toHaveBeenCalled();
	});
	it.each([11,22])('navigation %s revokes',async tabId => {
		const {send,updated,peer,api}=await fixture(); await send('acquire.offer');
		updated.emit(tabId,{status: 'loading'});
		expect(peer.close).toHaveBeenCalledOnce();
		expect(api.scripting.executeScript.mock.lastCall?.[0].args).toEqual(['stop',null,ownerId,1]);
		expect(api.runtime.sendMessage).toHaveBeenCalledWith({type: 'acquire.revoked',playerTabId: 22,playerDocumentId: 'player',ownerId});
		expect(api.tabs.sendMessage).not.toHaveBeenCalled();
		expect(await send('acquire.current')).toBe(false);
	});
	it('navigation before handshake',async () => {
		const {send,updated,player}=await fixture();
		updated.emit(22,{status: 'complete'}); updated.emit(22,{status: 'loading'});
		expect(await send('acquire.info',{...player,documentId: 'replacement'})).toBe(false);
	});
	it('native bootstrap loading events omit URLs',async () => {
		const {send,updated,agent}=await fixture();
		updated.emit(22,{status: 'loading'}); updated.emit(22,{status: 'loading'});
		expect(agent.stop).not.toHaveBeenCalled();
		expect(await send('acquire.info')).toMatchObject({ok: true});
		updated.emit(22,{status: 'complete'}); updated.emit(22,{status: 'loading'});
		expect(agent.stop).toHaveBeenCalledOnce();
		expect(await send('acquire.current')).toBe(false);
	});
	it.each(['reload','back_forward',undefined])('rejects initial document with navigation %s',async navigationType => {
		const {send,updated,agent}=await fixture();
		updated.emit(22,{status: 'loading'});
		expect(await send({type: 'acquire.info',navigationType})).toBe(false);
		expect(agent.read).not.toHaveBeenCalled();
	});
	it('rejects a known departure during bootstrap',async () => {
		const {send,updated}=await fixture();
		updated.emit(22,{url: 'https://other.test/'});
		expect(await send('acquire.info')).toBe(false);
	});
	it('refetch grant/URL',async () => {
		const {send,api}=await fixture(); api.permissions.contains.mockResolvedValueOnce(false);
		await denied(send('acquire.refetch'));
		expect(await send({type: 'acquire.refetch',url: 'https://evil.test/a.mp4'})).toMatchObject({ok: true,value: {selection: {url: sourceUrl}}});
		expect(api.permissions.contains).toHaveBeenCalledWith({origins: ['http://127.0.0.1/*']});
	});
	it('consumes only one concurrent refetch after permission checks',async () => {
		const {send,api}=await fixture(),entered=deferred(),grant=deferred<boolean>();
		let waiting=0;
		api.permissions.contains.mockImplementation(() => {if(++waiting===2) entered.resolve(); return grant.promise;});
		const requests=[send('acquire.refetch'),send('acquire.refetch')];
		await entered.promise;grant.resolve(true);
		const results=await Promise.all(requests);
		expect(results.filter(result => (result as {ok: boolean}).ok)).toHaveLength(1);
		expect(results.filter(result => !(result as {ok: boolean}).ok)).toEqual([{ok: false,error: 'Error: Refetch capability already consumed'}]);
	});
	it.each(['https://evil.test/A.mp4','http://127.0.0.1:5204/other/A.mp4',`${sourceUrl}?token=x`])('invalid %s',async url => {
		const {send}=await fixture(url); await denied(send('acquire.refetch'));
	});
	it('changed path',async () => {
		const {send,state}=await fixture(); state.url='http://127.0.0.1:5204/same/B.mp4';
		await denied(send('acquire.refetch'));
	});
	it.each(['redirect-same','redirect-ungranted'])('authorizes only the selected %s URL, never its redirect destination',async route => {
		const url=`http://127.0.0.1:5204/${route}.mp4`,{send}=await fixture(url);
		expect(await send('acquire.refetch')).toMatchObject({ok: true,value: {selection: {url}}});
	});
});