import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';

function check(body: string): void {
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { build } from 'esbuild';
    import { createContext, Script } from 'node:vm';
    async function load(entry, globals) {
      const result = await build({entryPoints: [entry], bundle: true, write: false,
        format: 'iife', globalName: 'observer', platform: 'browser'});
      const sandbox = createContext(globals);
      new Script(result.outputFiles[0].text, {filename: entry}).runInContext(sandbox);
      return sandbox;
    }
    ${body}
    console.log('checked without browser');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8', timeout: 15000 });
  expect(output.trim()).toBe('checked without browser');
}

const audioFixture = `
  async function audioFixture(rate = 48000) {
    const allocations = [], messages = [];
    let Recorder;
    const port = {onmessage: null, postMessage(message, transfer) {
      assert.equal(transfer.length, 1);
      assert.equal(transfer[0], message.pcm);
      assert.equal(message.pcm, allocations[0].buffer);
      messages.push(structuredClone(message, {transfer}));
    }};
    const sandbox = await load('tools/m1010r/audio-worklet.ts', {
      sampleRate: rate, currentFrame: 0,
      Float32Array: new Proxy(Float32Array, {construct(target, args) {
        const allocation = new target(...args); allocations.push(allocation); return allocation;
      }}),
      AudioWorkletProcessor: class {constructor() {this.port = port;}},
      registerProcessor(name, processor) {assert.equal(name, 'm1010r-audio'); Recorder = processor;},
      performance: {now() {assert.fail('No message-arrival clock');}},
      Date: class {static now() {assert.fail('No wall clock');}},
    });
    const recorder = new Recorder({});
    const send = data => port.onmessage({data, get timeStamp() {assert.fail('No message timestamp');}});
    const process = (frame, inputs, outputs = []) => {
      sandbox.currentFrame = frame;
      assert.equal(recorder.process(inputs, outputs), true);
    };
    return {recorder, sandbox, send, process, allocations, messages};
  }
`;

const controlFixture = `${audioFixture}
  async function controlFixture(options = {}) {
    const worklet = await audioFixture();
    const timers = new Map(), timerLog = [], events = [], sources = [], gains = [], buffers = [], nodes = [];
    const requests = [], aborts = [], commands = [];
    let now = 73000, nextTimer = 0, rendered = null, recording = null;
    const window = {
      setTimeout(callback, delay) {
        const id = ++nextTimer;
        timers.set(id, {callback, at: now + delay}); timerLog.push({id, delay}); return id;
      },
      clearTimeout(id) {events.push(['clear', id]); timers.delete(id);},
      performance: {now() {return now;}},
    };
    function advance(milliseconds) {
      const target = now + milliseconds;
      for (let count = 0; count < 10; count++) {
        const entry = [...timers].sort((left, right) => left[1].at - right[1].at)[0];
        if (!entry || entry[1].at > target) {now = target; return;}
        const [id, timer] = entry; now = timer.at; timers.delete(id); timer.callback();
      }
      assert.fail('Unexpected timer loop');
    }
    let seed = 173;
    const reference = Float32Array.from({length: options.referenceLength ?? 256000}, () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return options.constant ?? ((seed >>> 8) / 16777216 - .5);
    });
    function connectable(kind) {
      return {kind, target: null, disconnected: 0,
        connect(target) {this.target = target; events.push(['connect', this, target]); return target;},
        disconnect() {this.disconnected++; events.push(['disconnect', this]);},
      };
    }
    const context = {
      sampleRate: options.sampleRate ?? 48000, currentTime: options.currentTime ?? 0,
      destination: {kind: 'destination'},
      createBuffer(channels, length, rate) {
        const buffer = {channels, length, sampleRate: rate, data: new Float32Array(length),
          copyToChannel(data, channel) {assert.equal(channel, 0); this.data.set(data);}};
        buffers.push(buffer); return buffer;
      },
      createBufferSource() {
        const source = Object.assign(connectable('source'), {buffer: null, starts: [], stops: 0,
          start(time) {this.starts.push(time); events.push(['start', this]);},
          stop() {this.stops++; events.push(['stop', this]);},
        });
        sources.push(source); return source;
      },
      createGain() {
        const gain = Object.assign(connectable('gain'), {gain: {value: 1}});
        gains.push(gain); return gain;
      },
    };
    const firstFrame = options.firstFrame ?? Math.floor(context.currentTime * 48000) + 1024;
    function render() {
      const end = Math.ceil(Math.max(...sources.map(source =>
        source.starts[0] * 48000 + source.buffer.length)) + 4800);
      rendered = new Float32Array(end - firstFrame);
      sources.forEach((source, ordinal) => {
        const start = Math.round(source.starts[0] * 48000) - firstFrame + (options.shifts?.[ordinal] ?? 0);
        source.buffer.data.forEach((sample, offset) => {
          rendered[start + offset] += sample * source.target.gain.value;
        });
      });
      options.mutate?.(rendered, {sources, firstFrame});
      if (options.recordedSamples !== undefined) rendered = rendered.subarray(0, options.recordedSamples);
      for (let offset = 0; offset < rendered.length; offset += 128) {
        const left = rendered.subarray(offset, offset + 128), right = left.slice();
        const outputs = [new Float32Array(left.length), new Float32Array(right.length)];
        worklet.process(firstFrame + offset, [[left, right]], [outputs]);
        assert.deepEqual(outputs, [left, right], 'Scheduled stereo pass-through');
      }
    }
    class WorkletNode {
      constructor(receivedContext, name, descriptor) {
        assert.equal(receivedContext, context); assert.equal(name, 'm1010r-audio');
        Object.assign(this, connectable('recorder'));
        this.descriptor = descriptor; this.onprocessorerror = null;
        this.port = {onmessage: null, closed: 0,
          close() {this.closed++;},
          postMessage: command => {
            commands.push(command);
            if (command === 'finish') {
              if (options.dropRecording) return;
              render(); worklet.send(command); recording = worklet.messages[0];
              Object.assign(recording, options.metadata);
              this.port.onmessage({data: recording,
                get timeStamp() {assert.fail('No arrival timestamp as timing evidence');}});
            } else worklet.send(command);
          },
        };
        nodes.push(this);
      }
    }
    const sandbox = await load('tools/m1010r/audio-control.ts', {
      window, setTimeout: window.setTimeout, clearTimeout: window.clearTimeout,
      performance: window.performance, ArrayBuffer, Float32Array,
      AbortSignal: {timeout(milliseconds) {const signal = {milliseconds}; aborts.push(signal); return signal;}},
      fetch: async (url, attributes) => {
        requests.push({url, attributes});
        if (options.fetchError) throw options.fetchError;
        return {ok: options.ok ?? true, arrayBuffer: async () => reference.buffer};
      },
      AudioWorkletNode: WorkletNode,
    });
    async function start() {
      const pending = sandbox.observer.runAudioControl(context, '/controls/independent-48k.f32');
      void pending.catch(() => {});
      for (let turn = 0; turn < 8; turn++) await Promise.resolve();
      return {pending};
    }
    function cleaned() {
      assert.equal(timers.size, 0);
      for (const source of sources) {assert.equal(source.stops, 1); assert.equal(source.disconnected, 1);}
      for (const gain of gains) assert.equal(gain.disconnected, 1);
      for (const node of nodes) {
        assert.equal(node.disconnected, 1); assert.equal(node.port.closed, 1);
        assert.equal(node.port.onmessage, null); assert.equal(node.onprocessorerror, null);
      }
      assert.deepEqual(events.filter(event => event[0] === 'clear').map(event => event[1]).sort(),
        timerLog.map(timer => timer.id).sort());
    }
    return {start, advance, cleaned, context, firstFrame, reference, timers, timerLog, events,
      sources, gains, buffers, nodes, requests, aborts, commands, window,
      get rendered() {return rendered;}, get recording() {return recording;}};
  }
`;

const probeFixture = `
  const events = [], buffers = [], groups = [], allocations = {};
  let running = false, releaseMap, mapFailure = null;
  const mapReady = new Promise(resolve => {releaseMap = resolve;});
  function resource(kind, descriptor) {
    assert(!running || kind === 'bindGroup', 'Unexpected per-frame allocation: ' + kind);
    allocations[kind] = (allocations[kind] ?? 0) + 1;
    events.push(['allocate', kind]);
    return {kind, descriptor};
  }
  function encoder() {
    const copies = [];
    return {
      beginComputePass(descriptor) {
        events.push(['begin', descriptor]);
        return {setPipeline(value) {events.push(['pipeline', value]);},
          setBindGroup(index, value) {events.push(['group', index, value]);},
          dispatchWorkgroups(...size) {events.push(['dispatch', ...size]);},
          end() {events.push(['end']);}};
      },
      copyBufferToBuffer(...copy) {assert(!running); events.push(['copy', ...copy]); copies.push(copy);},
      finish() {events.push(['finish']); return {copies};},
    };
  }
  const device = {
    createBuffer(descriptor) {
      const buffer = Object.assign(resource('buffer', descriptor), {
        data: new ArrayBuffer(descriptor.size), destroyed: false, mapped: false,
        async mapAsync(mode) {
          assert(!running, 'No readback in the frame loop'); events.push(['map', mode]);
          await mapReady; if (mapFailure) throw mapFailure; this.mapped = true;
        },
        getMappedRange() {assert(this.mapped); events.push(['range']); return this.data;},
        unmap() {assert(this.mapped); events.push(['unmap']); this.mapped = false;},
        destroy() {this.destroyed = true; events.push(['destroy', this]);},
      });
      buffers.push(buffer); return buffer;
    },
    createSampler(descriptor) {return resource('sampler', descriptor);},
    createBindGroupLayout(descriptor) {return resource('bindGroupLayout', descriptor);},
    createPipelineLayout(descriptor) {return resource('pipelineLayout', descriptor);},
    createShaderModule(descriptor) {return resource('shaderModule', descriptor);},
    createComputePipeline(descriptor) {return resource('computePipeline', descriptor);},
    createBindGroup(descriptor) {const group = resource('bindGroup', descriptor); groups.push(group); return group;},
    createTexture() {assert.fail('Probe must not allocate textures');},
    importExternalTexture() {assert.fail('Probe must use the supplied external texture');},
    createCommandEncoder() {events.push(['encoder']); return encoder();},
    queue: {
      writeBuffer(buffer, offset, data) {
        assert(!running); events.push(['write', buffer, offset]);
        new Uint8Array(buffer.data).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
      },
      submit(commands) {
        assert(!running); events.push(['submit', commands]);
        for (const command of commands) for (const [source, sourceOffset, target, targetOffset, size] of command.copies)
          new Uint8Array(target.data, targetOffset, size).set(new Uint8Array(source.data, sourceOffset, size));
      },
    },
  };
  const sandbox = await load('tools/m1010r/probe.ts', {
    GPUBufferUsage: {STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, MAP_READ: 1, UNIFORM: 64},
    GPUShaderStage: {COMPUTE: 4}, GPUMapMode: {READ: 1},
  });
  const {IdentityProbe, PROBE_CAPACITY, probeShader} = sandbox.observer;
  const inner = {id: 'inner', label: 'Inner upscaler', scaleFactor: 2, neural: true,
    configure(value) {assert.equal(this, inner); events.push(['configure', value]);},
    encode(value) {assert.equal(this, inner); events.push(['inner', value]);},
    destroy() {assert.equal(this, inner); events.push(['inner-destroy']);},
  };
  const probe = new IdentityProbe(inner, index => events.push(['before', index]));
  const config = {device, source: {width: 1280, height: 720}, target: {width: 2560, height: 1440},
    targetFormat: 'rgba8unorm', sourceKind: 'external'};
  const context = frame => ({frame, encoder: encoder(), target: {id: 'target'},
    timing: {querySet: {id: 'timestamps'}, beginIndex: 0, endIndex: 1}});
`;

describe('M10.10R bundled audio observer (browser-free)', () => {
  it('passes every channel through exactly while idle, recording and finished', () => check(`${audioFixture}
    const fixture = await audioFixture();
    const inputs = [new Float32Array([0, -0, .25, -.75]), new Float32Array([.5, -.5, 1, -1]),
      new Float32Array([.125, -.125, .875, -.875])];
    const outputs = Array.from({length: 4}, () => new Float32Array(4).fill(9));
    for (const [index, state] of ['idle', 'start', 'finish'].entries()) {
      if (state !== 'idle') fixture.send(state);
      outputs.forEach(channel => channel.fill(9));
      fixture.process(100 + index * 4, [inputs], [outputs]);
      inputs.forEach((channel, channelIndex) => assert.deepEqual(outputs[channelIndex], channel));
      assert.deepEqual([...outputs[3]], [0, 0, 0, 0]);
    }
    assert.equal(fixture.messages[0].samples, 4);
    assert.equal(fixture.allocations.length, 1);
  `));

  it('records firstFrame plus exact sample offsets and transfers actual PCM and unique block lengths', () => check(`${audioFixture}
    const fixture = await audioFixture();
    fixture.sandbox.currentFrame = 17;
    fixture.send('start');
    const blocks = [[.25, -.5, .75], [-1, 0], [1, .125, -.125]];
    let offset = 0;
    for (const block of blocks) {
      const input = new Float32Array(block);
      fixture.process(8192 + offset, [[input, new Float32Array(input.length).fill(.5)]]);
      offset += input.length;
      input.fill(99);
    }
    fixture.sandbox.currentFrame = 999999;
    assert.equal(fixture.messages.length, 0);
    fixture.send('finish');
    const recording = fixture.messages[0];
    const {pcm, ...metadata} = recording;
    assert.deepEqual(metadata, {type: 'audio-recording', sampleRate: 48000, firstFrame: 8192,
      samples: 8, overflow: false, discontinuity: false, blockLengths: [3, 2]});
    assert.equal(pcm.byteLength, 48000 * 80 * 4);
    assert.deepEqual([...new Float32Array(pcm, 0, recording.samples)], blocks.flat());
    assert.equal(new Float32Array(pcm)[recording.samples], 0);
    assert.equal(fixture.allocations[0].byteLength, 0, 'Sender buffer was really transferred');
    assert.equal(fixture.allocations.length, 1);
  `));

  it('detects skipped and repeated sample-clock positions without padding the PCM', () => check(`${audioFixture}
    for (const nextFrame of [1001, 1003]) {
      const fixture = await audioFixture();
      fixture.send('start');
      fixture.process(1000, [[new Float32Array([.25, .5])]]);
      fixture.process(nextFrame, [[new Float32Array([.75])]]);
      fixture.send('finish');
      const result = fixture.messages[0];
      assert.equal(result.firstFrame, 1000); assert.equal(result.samples, 3);
      assert.equal(result.discontinuity, true); assert.equal(result.overflow, false);
      assert.deepEqual([...new Float32Array(result.pcm, 0, 3)], [.25, .5, .75]);
    }
  `));

  it('silences absent inputs, waits for the first input and flags a later missing-input gap on resume', () => check(`${audioFixture}
    for (const gap of [false, true]) {
      const fixture = await audioFixture();
      fixture.send('start');
      const outputs = [new Float32Array(2).fill(9), new Float32Array(2).fill(9)];
      fixture.process(10, [], [outputs]);
      outputs.forEach(channel => assert.deepEqual([...channel], [0, 0]));
      fixture.process(20, [[]]);
      fixture.process(40, [[new Float32Array([.25, .5])]]);
      if (gap) fixture.process(42, []);
      fixture.process(gap ? 44 : 42, [[new Float32Array([.75, 1])]]);
      fixture.send('finish');
      const result = fixture.messages[0];
      assert.equal(result.firstFrame, 40); assert.equal(result.samples, 4);
      assert.equal(result.discontinuity, gap);
      assert.deepEqual(result.blockLengths, [2]);
      assert.deepEqual([...new Float32Array(result.pcm, 0, 4)], [.25, .5, .75, 1]);
    }
  `));

  it('stops at capacity without partial writes, allocation or reset while preserving pass-through', () => check(`${audioFixture}
    for (const exactFit of [false, true]) {
      const fixture = await audioFixture(1);
      assert.equal(fixture.allocations[0].length, 80);
      fixture.send('start');
      fixture.process(500, [[new Float32Array(64).fill(.25)]]);
      if (exactFit) fixture.process(564, [[new Float32Array(16).fill(.5)]]);
      const overflowBlock = new Float32Array(exactFit ? 1 : 17).fill(.75);
      const output = new Float32Array(overflowBlock.length);
      fixture.process(exactFit ? 580 : 564, [[overflowBlock]], [[output]]);
      assert.deepEqual(output, overflowBlock);
      fixture.process(600, [[new Float32Array([1])]]);
      assert.throws(() => fixture.send('start'), /single-use/);
      fixture.send('finish');
      const result = fixture.messages[0], pcm = new Float32Array(result.pcm);
      assert.equal(result.firstFrame, 500); assert.equal(result.samples, exactFit ? 80 : 64);
      assert.equal(result.overflow, true); assert.equal(result.discontinuity, false);
      assert.deepEqual([...pcm.subarray(0, 64)], Array(64).fill(.25));
      assert.deepEqual([...pcm.subarray(64)], Array(16).fill(exactFit ? .5 : 0));
      assert.deepEqual(result.blockLengths, exactFit ? [64, 16, 1] : [64, 17]);
      assert.equal(fixture.allocations.length, 1);
    }
  `));

  it('rejects duplicate starts while active and after a nonempty recording', () => check(`${audioFixture}
    const fixture = await audioFixture();
    fixture.send('start');
    assert.throws(() => fixture.send('start'), /single-use/);
    fixture.process(0, [[new Float32Array([.5])]]);
    fixture.send('finish');
    assert.throws(() => fixture.send('start'), /single-use/);
    assert.equal(fixture.allocations.length, 1);
  `));

  it('rejects restart after finishing an empty recording', () => check(`${audioFixture}
    const fixture = await audioFixture();
    fixture.send('start'); fixture.process(20, []); fixture.send('finish');
    assert.equal(fixture.messages[0].firstFrame, null);
    assert.equal(fixture.messages[0].samples, 0);
    assert.deepEqual(fixture.messages[0].blockLengths, []);
    assert.throws(() => fixture.send('start'), /single-use/);
  `));

  it('rejects restart when the first block alone exceeds capacity', () => check(`${audioFixture}
    const fixture = await audioFixture(1);
    fixture.send('start'); fixture.process(0, [[new Float32Array(81)]]);
    assert.equal(fixture.allocations.length, 1);
    assert.throws(() => fixture.send('start'), /single-use/);
  `));
});

describe('M10.10R bundled scheduled audio control (virtual sample clock)', () => {
  it('fails closed when host finish precedes the scheduled audio sample frames', () => check(`${controlFixture}
    const fixture = await controlFixture({firstFrame: 0, recordedSamples: 7168});
    const {pending} = await fixture.start();
    fixture.advance(2000);
    const result = await pending;
    assert.equal(result.firstFrame, 0); assert.equal(result.samples, 7168);
    assert.equal(result.scheduled[0].startFrame, 12000);
    assert.equal(result.expectedWindows, 180); assert.equal(result.verifiedWindows, 0);
    assert.equal(result.maximumSampleError, null); assert.equal(result.errors.length, 180);
    assert(new Float32Array(result.pcm).every(sample => sample === 0));
    fixture.cleaned();
  `));

  it('fetches the known reference and schedules three exact segments through a stereo pass-through graph', () => check(`${controlFixture}
    const fixture = await controlFixture();
    const {pending} = await fixture.start();
    assert.equal(fixture.requests.length, 1);
    const request = fixture.requests[0];
    assert.equal(request.url, '/controls/independent-48k.f32');
    assert.deepEqual({...request.attributes}, {credentials: 'omit', redirect: 'error', signal: fixture.aborts[0]});
    assert.deepEqual(fixture.aborts, [{milliseconds: 5000}]);
    assert.equal(fixture.nodes.length, 1);
    const node = fixture.nodes[0];
    assert.deepEqual(JSON.parse(JSON.stringify(node.descriptor)), {numberOfInputs: 1, numberOfOutputs: 1,
      outputChannelCount: [2], channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'speakers'});
    assert.equal(node.target, fixture.context.destination);
    assert.deepEqual(fixture.commands, ['start']);
    assert.equal(fixture.sources.length, 3); assert.equal(fixture.gains.length, 3);
    assert.equal(fixture.buffers.length, 3);
    for (const [ordinal, time] of [.25, .8, 1.2].entries()) {
      const source = fixture.sources[ordinal], gain = fixture.gains[ordinal], buffer = fixture.buffers[ordinal];
      assert.deepEqual(source.starts, [time]); assert.equal(source.buffer, buffer);
      assert.equal(source.target, gain); assert.equal(gain.target, node);
      assert.equal(gain.gain.value, [1, .5, .75][ordinal]);
      assert.equal(buffer.channels, 1); assert.equal(buffer.length, 8192); assert.equal(buffer.sampleRate, 48000);
      const referenceStart = [48000, 96000, 144000][ordinal];
      assert.deepEqual(buffer.data, fixture.reference.slice(referenceStart, referenceStart + 8192));
    }
    assert.deepEqual(fixture.timerLog.map(timer => timer.delay), [5000, 70592 / 48]);
    fixture.advance(70592 / 48 - .001); assert.deepEqual(fixture.commands, ['start']);
    fixture.advance(.002);
    const result = await pending;
    assert.deepEqual(fixture.commands, ['start', 'finish']);
    assert.deepEqual(JSON.parse(JSON.stringify(result.scheduled)), [
      {startFrame: 12000, referenceStart: 48000, samples: 8192, gain: 1},
      {startFrame: 38400, referenceStart: 96000, samples: 8192, gain: .5},
      {startFrame: 57600, referenceStart: 144000, samples: 8192, gain: .75},
    ]);
    assert.equal(result.sampleRate, 48000); assert.equal(result.firstFrame, 1024);
    assert.equal(result.samples, 70592 - 1024); assert.deepEqual([...result.blockLengths], [128, 64]);
    assert.equal(result.expectedWindows, 180); assert.equal(result.verifiedWindows, 180);
    assert.equal(result.maximumSampleError, 0); assert.deepEqual([...result.errors], []);
    assert.equal(result.pcm.byteLength, result.samples * 4);
    assert.notEqual(result.pcm, fixture.recording.pcm);
    assert.deepEqual(new Float32Array(result.pcm), fixture.rendered);
    assert.equal(fixture.recording.pcm.byteLength, 48000 * 80 * 4);
    assert.deepEqual(new Float32Array(fixture.recording.pcm, 0, result.samples), fixture.rendered);
    new Float32Array(result.pcm).fill(99);
    assert.deepEqual(new Float32Array(fixture.recording.pcm, 0, result.samples), fixture.rendered);
    fixture.cleaned(); fixture.advance(10000); assert.deepEqual(fixture.commands, ['start', 'finish']);
  `));

  it('rounds a fractional context sample upward independently of the wall-clock timer origin', () => check(`${controlFixture}
    const fixture = await controlFixture({currentTime: 7 + 1 / 96000});
    fixture.advance(987654321);
    const {pending} = await fixture.start();
    assert.equal(fixture.window.performance.now(), 987727321);
    assert.deepEqual(fixture.sources.map(source => source.starts[0]), [348001 / 48000, 374401 / 48000, 393601 / 48000]);
    assert(Math.abs(fixture.timerLog[1].delay - (406593 / 48 - (7 + 1 / 96000) * 1000)) < 1e-6);
    fixture.advance(2000);
    const result = await pending;
    assert.deepEqual(Array.from(result.scheduled, marker => marker.startFrame), [348001, 374401, 393601]);
    assert.equal(result.firstFrame, 337024); assert.equal(result.expectedWindows, 180);
    assert.equal(result.verifiedWindows, 180); assert.equal(result.maximumSampleError, 0);
    assert.deepEqual([...result.errors], []); fixture.cleaned();
  `));

  it.each([
    {shifts: [-12, 0, 12], maximum: 12},
    {shifts: [-1, 1, 7], maximum: 7},
    {shifts: [12, -12, -3], maximum: 12},
  ])('recovers independently shifted PCM segments $shifts within the inclusive sample-error limit', ({ shifts, maximum }) => check(`${controlFixture}
    const fixture = await controlFixture({shifts: ${JSON.stringify(shifts)}});
    const {pending} = await fixture.start(); fixture.advance(2000);
    const result = await pending;
    assert.equal(result.expectedWindows, 180); assert.equal(result.verifiedWindows, 180);
    assert.equal(result.maximumSampleError, ${maximum}); assert.deepEqual([...result.errors], []);
    assert.deepEqual(new Float32Array(result.pcm), fixture.rendered);
    assert.deepEqual(new Float32Array(fixture.recording.pcm, 0, result.samples), fixture.rendered);
    fixture.cleaned();
  `));

  it.each([-25, -24, -13, 13, 24, 25])('does not verify a segment shifted by %s samples, but still checks later segments', shift => check(`${controlFixture}
    const fixture = await controlFixture({shifts: [${shift}, 0, 0]});
    const {pending} = await fixture.start(); fixture.advance(2000);
    const result = await pending;
    assert.equal(result.expectedWindows, 180); assert.equal(result.verifiedWindows, 120);
    assert.equal(result.maximumSampleError, 0);
    assert.deepEqual([...result.errors], Array.from({length: 60}, (_, index) =>
      'Unverified scheduled waveform at ' + (12256 + 128 * index)));
    assert.deepEqual(new Float32Array(result.pcm), fixture.rendered);
    assert.deepEqual(new Float32Array(fixture.recording.pcm, 0, result.samples), fixture.rendered);
    fixture.cleaned();
  `));

  it.each(['nonunique', 'noise', 'silence'])('rejects every fixed window for %s instead of claiming a zero timing error', kind => check(`${controlFixture}
    const kind = ${JSON.stringify(kind)};
    const options = kind === 'noise' ? {mutate(pcm) {
      let seed = 917;
      for (let index = 0; index < pcm.length; index++) {
        seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
        pcm[index] = (seed >>> 8) / 16777216 - .5;
      }
    }} : {constant: kind === 'silence' ? 0 : .25};
    const fixture = await controlFixture(options);
    const {pending} = await fixture.start(); fixture.advance(2000);
    const result = await pending;
    assert.equal(result.expectedWindows, 180); assert.equal(result.verifiedWindows, 0);
    assert.equal(result.maximumSampleError, null);
    assert.deepEqual([...result.errors], [12000, 38400, 57600].flatMap(start =>
      Array.from({length: 60}, (_, index) => 'Unverified scheduled waveform at ' + (start + 256 + 128 * index))));
    assert.deepEqual(new Float32Array(result.pcm), fixture.rendered);
    assert.deepEqual(new Float32Array(fixture.recording.pcm, 0, result.samples), fixture.rendered);
    fixture.cleaned();
  `));

  it('checks the last fixed window of every segment and preserves the offending PCM samples', () => check(`${controlFixture}
    const fixture = await controlFixture({mutate(pcm, {firstFrame}) {
      for (const start of [12000, 38400, 57600]) pcm[start - firstFrame + 8063] = 99;
    }});
    const {pending} = await fixture.start(); fixture.advance(2000);
    const result = await pending;
    assert.equal(result.expectedWindows, 180); assert.equal(result.verifiedWindows, 177);
    assert.equal(result.maximumSampleError, 0);
    assert.deepEqual([...result.errors], [19808, 46208, 65408].map(frame => 'Unverified scheduled waveform at ' + frame));
    assert.deepEqual(new Float32Array(result.pcm), fixture.rendered);
    assert.deepEqual(new Float32Array(fixture.recording.pcm, 0, result.samples), fixture.rendered);
    fixture.cleaned();
  `));

  it.each([
    {metadata: {overflow: true}, message: 'Invalid scheduled sample-clock recording'},
    {metadata: {discontinuity: true}, message: 'Invalid scheduled sample-clock recording'},
    {metadata: {sampleRate: 44100}, message: 'Invalid scheduled sample-clock recording'},
    {metadata: {firstFrame: null}, message: 'Malformed scheduled control recording'},
  ])('rejects invalid sample-clock metadata $metadata and cleans every resource', ({ metadata, message }) => check(`${controlFixture}
    const fixture = await controlFixture({metadata: ${JSON.stringify(metadata)}});
    const {pending} = await fixture.start(); fixture.advance(2000);
    await assert.rejects(pending, {message: ${JSON.stringify(message)}});
    assert.deepEqual(new Float32Array(fixture.recording.pcm, 0, fixture.recording.samples), fixture.rendered);
    fixture.cleaned();
  `));

  it.each([
    {options: {ok: false}, message: 'Missing decoded-PCM control'},
    {options: {referenceLength: 0}, message: 'Insufficient 48kHz reference control'},
    {options: {referenceLength: 255999}, message: 'Insufficient 48kHz reference control'},
    {options: {sampleRate: 44100}, message: 'Insufficient 48kHz reference control'},
    {options: {sampleRate: 96000}, message: 'Insufficient 48kHz reference control'},
  ])('rejects missing or incompatible references $options before allocating a graph or timers', ({ options, message }) => check(`${controlFixture}
    const fixture = await controlFixture(${JSON.stringify(options)});
    const {pending} = await fixture.start();
    await assert.rejects(pending, {message: ${JSON.stringify(message)}});
    assert.deepEqual(fixture.nodes, []); assert.deepEqual(fixture.sources, []);
    assert.deepEqual(fixture.buffers, []); assert.deepEqual(fixture.gains, []);
    assert.deepEqual(fixture.timerLog, []); assert.deepEqual(fixture.commands, []);
    fixture.cleaned();
  `));

  it('propagates reference fetch failure without creating timers, sources or a recorder', () => check(`${controlFixture}
    const failure = new Error('reference fetch aborted');
    const fixture = await controlFixture({fetchError: failure});
    const {pending} = await fixture.start();
    await assert.rejects(pending, error => error === failure);
    assert.deepEqual(fixture.nodes, []); assert.deepEqual(fixture.sources, []);
    assert.deepEqual(fixture.buffers, []); assert.deepEqual(fixture.gains, []);
    assert.deepEqual(fixture.timerLog, []); assert.deepEqual(fixture.commands, []);
    fixture.cleaned();
  `));

  it.each(['null', '{}', '{pcm: new Float32Array(128)}', '{samples: "128"}',
    '{sampleRate: "48000"}', '{firstFrame: "1024"}', '{blockLengths: null}',
    '{overflow: 0}', '{discontinuity: 0}'])('rejects malformed worklet messages %s and removes active callbacks', malformed => check(`${controlFixture}
    const fixture = await controlFixture();
    const {pending} = await fixture.start();
    const patch = ${malformed};
    const valid = {pcm: new ArrayBuffer(512), samples: 128, sampleRate: 48000,
      firstFrame: 1024, blockLengths: [128], overflow: false, discontinuity: false};
    fixture.nodes[0].port.onmessage({data: patch === null || Object.keys(patch).length === 0 ? patch : {...valid, ...patch}});
    await assert.rejects(pending, /Malformed scheduled control recording/);
    fixture.cleaned(); fixture.advance(10000); assert.deepEqual(fixture.commands, ['start']);
  `));

  it('rejects processor failure and clears both timers before further worklet commands', () => check(`${controlFixture}
    const fixture = await controlFixture();
    const {pending} = await fixture.start();
    assert.equal(fixture.timers.size, 2); fixture.nodes[0].onprocessorerror();
    await assert.rejects(pending, /Scheduled control processor error/);
    fixture.cleaned(); fixture.advance(10000);
    assert.deepEqual(fixture.commands, ['start']); assert.equal(fixture.recording, null);
  `));

  it('rejects a missing recording exactly at the virtual deadline and stops all scheduled sources', () => check(`${controlFixture}
    const fixture = await controlFixture({dropRecording: true});
    const {pending} = await fixture.start();
    let settled = false; void pending.then(() => {settled = true;}, () => {settled = true;});
    fixture.advance(4999); await Promise.resolve(); assert.equal(settled, false);
    assert.deepEqual(fixture.commands, ['start', 'finish']); assert.equal(fixture.timers.size, 1);
    fixture.advance(1); await assert.rejects(pending, /Scheduled audio control timeout/);
    fixture.cleaned(); fixture.advance(10000); assert.equal(settled, true);
    assert.deepEqual(fixture.commands, ['start', 'finish']); assert.equal(fixture.recording, null);
  `));

  it('releases a partially scheduled source and gain when the second source start throws', () => check(`${controlFixture}
    const fixture = await controlFixture(), failure = new Error('source start failed');
    const createSource = fixture.context.createBufferSource;
    fixture.context.createBufferSource = () => {
      const source = createSource();
      if (fixture.sources.length === 2) source.start = () => {throw failure;};
      return source;
    };
    const {pending} = await fixture.start();
    await assert.rejects(pending, error => error === failure);
    assert.equal(fixture.sources.length, 2); assert.equal(fixture.gains.length, 2);
    fixture.cleaned(); fixture.advance(10000); assert.deepEqual(fixture.commands, ['start']);
  `));

  it('still disconnects every node and clears callbacks when stopping one source throws', () => check(`${controlFixture}
    const fixture = await controlFixture(), failure = new Error('source stop failed');
    const {pending} = await fixture.start();
    const source = fixture.sources[0], stop = source.stop.bind(source);
    source.stop = () => {stop(); throw failure;};
    fixture.advance(2000); await assert.rejects(pending, error => error === failure);
    fixture.cleaned(); fixture.advance(10000); assert.deepEqual(fixture.commands, ['start', 'finish']);
  `));
});

describe('M10.10R bundled identity probe (mock GPU, no hardware evidence)', () => {
  it('delegates the interface and configuration, using source dimensions and preallocated buffers', () => check(`${probeFixture}
    assert.equal(probe.inner, inner);
    for (const key of ['id', 'label', 'scaleFactor', 'neural']) assert.equal(probe[key], inner[key]);
    Object.assign(inner, {id: 'changed', label: 'Changed', scaleFactor: 3, neural: false});
    for (const key of ['id', 'label', 'scaleFactor', 'neural']) assert.equal(probe[key], inner[key]);
    probe.configure(config);
    assert.deepEqual(events[0], ['configure', config]);
    assert.equal(events[0][1], config);
    assert.deepEqual(buffers.map(buffer => ({...buffer.descriptor})), [
      {size: 16 + PROBE_CAPACITY * 16, usage: 132},
      {size: 16 + PROBE_CAPACITY * 16, usage: 9}, {size: 16, usage: 72}]);
    assert.deepEqual([...new Float32Array(buffers[2].data)], [1280, 720, 0, 0]);
    assert.deepEqual(allocations, {buffer: 3, sampler: 1, bindGroupLayout: 1,
      pipelineLayout: 1, shaderModule: 1, computePipeline: 1});
  `));

  it('uses each supplied external texture and encoder, after the inner encode, with only one bind allocation per frame', () => check(`${probeFixture}
    probe.configure(config);
    const configured = {...allocations}; events.length = 0; running = true;
    for (let index = 0; index < 3; index++) {
      const texture = {id: index}, input = context({kind: 'external', texture});
      assert.equal(probe.encode(input), undefined);
      assert.deepEqual(events.map(event => event[0]), ['before', 'inner', 'allocate', 'begin',
        'pipeline', 'group', 'dispatch', 'end']);
      assert.deepEqual(events[0], ['before', index]); assert.equal(events[1][1], input);
      assert.equal(events[3][1].label, 'm1010r:identity');
      const group = groups[index], pipeline = events[4][1];
      assert.equal(group.descriptor.entries[0].resource, texture);
      assert.equal(group.descriptor.entries[2].resource.buffer, buffers[0]);
      assert.equal(group.descriptor.entries[3].resource.buffer, buffers[2]);
      assert.deepEqual({...group.descriptor.entries[1].resource.descriptor}, {minFilter: 'nearest', magFilter: 'nearest'});
      assert.equal(pipeline.descriptor.layout.descriptor.bindGroupLayouts[0], group.descriptor.layout);
      assert('externalTexture' in group.descriptor.layout.descriptor.entries[0]);
      assert.deepEqual(events[5], ['group', 0, group]); assert.deepEqual(events[6], ['dispatch', 1]);
      events.length = 0;
    }
    assert.equal(probe.encoded, 3); assert.equal(new Set(groups).size, 3);
    assert.deepEqual(allocations, {...configured, bindGroup: 3});
    running = false; probe.destroy();
  `));

  it('reuses the sampled bind group per frame and replaces it only at reconfiguration', () => check(`${probeFixture}
    const view = {id: 'sampled'};
    Object.assign(config, {sourceKind: 'sampled', sampledSourceView: view}); probe.configure(config);
    const configured = {...allocations}, firstGroup = groups[0];
    assert.equal(firstGroup.descriptor.entries[0].resource, view);
    assert.equal(firstGroup.descriptor.layout.descriptor.entries[0].texture.sampleType, 'float');
    events.length = 0; running = true;
    for (let index = 0; index < 3; index++) probe.encode(context({kind: 'sampled', view}));
    assert.deepEqual(allocations, configured);
    assert.deepEqual(events.filter(event => event[0] === 'group'), Array(3).fill(['group', 0, firstGroup]));
    assert.equal(events.filter(event => ['map', 'encoder', 'copy', 'submit', 'write'].includes(event[0])).length, 0);
    running = false;
    const replacement = {id: 'replacement'};
    probe.configure({...config, source: {width: 640, height: 360}, sampledSourceView: replacement});
    assert.equal(buffers.length, 3); assert.equal(allocations.sampler, 1);
    assert.deepEqual([...new Float32Array(buffers[2].data)], [640, 360, 0, 0]);
    assert.equal(groups[1].descriptor.entries[0].resource, replacement);
    events.length = 0; running = true;
    probe.encode(context({kind: 'sampled', view: replacement}));
    assert.equal(events.find(event => event[0] === 'group')[2], groups[1]);
    running = false; probe.destroy();
  `));

  it('copies and maps only after pause, returns detached row data and unmaps before count validation', () => check(`${probeFixture}
    probe.configure(config); running = true;
    probe.encode(context({kind: 'external', texture: {}}));
    probe.encode(context({kind: 'external', texture: {}}));
    running = false;
    const words = new Uint32Array(buffers[0].data);
    words.set([2, 0, 0, 0, 8192, 1, 1280, 720, 8193, 0, 1280, 720]);
    events.length = 0;
    let settled = false;
    const pending = probe.readAfterPause().then(rows => {settled = true; return rows;});
    await Promise.resolve(); assert.equal(settled, false);
    assert.deepEqual(events.map(event => event[0]), ['encoder', 'copy', 'finish', 'submit', 'map']);
    assert.deepEqual(events[1], ['copy', buffers[0], 0, buffers[1], 0, 16 + PROBE_CAPACITY * 16]);
    assert.deepEqual(events[4], ['map', 1]);
    releaseMap(); const rows = await pending;
    assert.deepEqual([...rows], [...words]);
    assert.notEqual(rows.buffer, buffers[1].data); assert.equal(buffers[1].mapped, false);
    assert.deepEqual(events.slice(-2).map(event => event[0]), ['range', 'unmap']);
    new Uint32Array(buffers[1].data).fill(99); assert.equal(rows[4], 8192);
    probe.destroy();
  `));

  it('rejects GPU count mismatches and mapping failures, allowing all owned buffers to be destroyed', () => check(`${probeFixture}
    probe.configure(config); probe.encode(context({kind: 'external', texture: {}}));
    releaseMap();
    for (const count of [0, 2]) {
      new Uint32Array(buffers[0].data)[0] = count;
      await assert.rejects(probe.readAfterPause(), /GPU and CPU identity record counts differ/);
      assert.equal(buffers[1].mapped, false);
      assert.equal(events.at(-1)[0], 'unmap');
    }
    mapFailure = new Error('map failed');
    await assert.rejects(probe.readAfterPause(), error => error === mapFailure);
    assert.equal(buffers[1].mapped, false);
    events.length = 0; probe.destroy();
    assert.deepEqual(events.map(event => event[0]), ['inner-destroy', 'destroy', 'destroy', 'destroy']);
    assert(buffers.every(buffer => buffer.destroyed));
    await assert.rejects(probe.readAfterPause(), /No identity records/);
  `));

  it('rejects unconfigured operations and missing sampled input without dispatching a probe', () => check(`${probeFixture}
    assert.throws(() => probe.encode(context({kind: 'external', texture: {}})), /Identity diagnostic capacity/);
    await assert.rejects(probe.readAfterPause(), /No identity records/);
    assert.equal(probe.encoded, 0); assert.deepEqual(events, []);
    probe.configure({...config, sourceKind: 'sampled'}); events.length = 0;
    assert.throws(() => probe.encode(context({kind: 'sampled', view: {}})), /Missing sampled identity input/);
    assert.equal(events.some(event => event[0] === 'begin'), false);
    probe.destroy();
  `));

  it('accepts exactly the bounded number of encodes and rejects the next before touching the inner stage', () => check(`${probeFixture}
    const view = {};
    probe.configure({...config, sourceKind: 'sampled', sampledSourceView: view});
    const configured = {...allocations}, input = context({kind: 'sampled', view}); running = true;
    for (let index = 0; index < PROBE_CAPACITY; index++) {
      events.length = 0; probe.encode(input); assert.deepEqual(events[0], ['before', index]);
    }
    assert.equal(probe.encoded, PROBE_CAPACITY); events.length = 0;
    assert.throws(() => probe.encode(input), /Identity diagnostic capacity/);
    assert.deepEqual(events, []); assert.deepEqual(allocations, configured);
    running = false; probe.destroy();
  `));

  it.each(['external', 'sampled'])('rejects %s encode after destroy before callbacks or inner work', kind => check(`${probeFixture}
    const view = {}, kind = ${JSON.stringify(kind)};
    probe.configure({...config, sourceKind: kind, ...(kind === 'sampled' ? {sampledSourceView: view} : {})});
    probe.destroy(); events.length = 0;
    assert.throws(() => probe.encode(context(kind === 'external' ? {kind, texture: {}} : {kind, view})));
    assert.equal(probe.encoded, 0); assert.deepEqual(events, []);
  `));

  it('retains static shader identity guards, complements, record bounds and import-specific sampling', () => check(`${probeFixture}
    for (const external of [true, false]) {
      const shader = probeShader(external);
      assert(shader.includes(external ? 'texture_external' : 'texture_2d<f32>'));
      assert(shader.includes(external ? 'textureSampleBaseClampToEdge(source, filtering, coordinates)' :
        'textureSampleLevel(source, filtering, coordinates, 0.0)'));
      assert(!shader.includes(external ? 'textureSampleLevel(' : 'textureSampleBaseClampToEdge('));
      assert(shader.includes('horizontal + 0.5, vertical + 0.5'));
      assert(shader.includes('vec3<f32>(0.2126, 0.7152, 0.0722)'));
      assert(shader.includes('luma <= 0.25')); assert(shader.includes('luma >= 0.75'));
      assert(shader.includes('cell(8.0,24.0)==1u && cell(8.0,56.0)==1u && cell(32.0,24.0)==0u && cell(32.0,56.0)==0u'));
      assert(shader.includes('bit < 16u')); assert(shader.includes('(bit+2u)*24u+8u'));
      assert(shader.includes('top < 2u && bottom < 2u && top != bottom'));
      assert(shader.includes('identity = identity | ((top & 1u) << bit)'));
      const increment = shader.indexOf('atomicAdd(&records.count, 1u)');
      const bound = shader.indexOf('if (slot >= ' + PROBE_CAPACITY + 'u) { return; }');
      assert(increment >= 0 && bound > increment && shader.indexOf('records.rows[slot]') > bound);
      assert(shader.includes('vec4<u32>(identity,select(0u,1u,valid),u32(extent.x),u32(extent.y))'));
    }
  `));
});