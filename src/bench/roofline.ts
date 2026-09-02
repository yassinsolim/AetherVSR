/**
 * Two microbenchmarks that bracket the convolution kernel.
 *
 * A GMAC/s figure on its own says nothing about *why* a kernel runs at that
 * speed. To claim "bandwidth bound" or "ALU bound" you need the device's own
 * achievable numbers for each, measured the same way, on the same machine, in
 * the same browser. Vendor peak figures would not do: they are not what a
 * WebGPU compute shader can reach through Dawn and Metal.
 *
 * Neither of these is a hardware ceiling. They are the best *this harness* can
 * get out of the device with a kernel designed to do nothing else, which is
 * the honest reference point for a kernel that has real work to do.
 */

const NS_PER_MS = 1_000_000;

export interface RooflineResult {
  readonly label: string;
  readonly medianMs: number;
  readonly iterations: number;
  /** Streaming bandwidth in GB/s, for the bandwidth probe. */
  readonly gbPerSecond: number | null;
  /** Fused multiply-add throughput in GFLOP/s, for the ALU probe. */
  readonly gflopPerSecond: number | null;
  readonly diagnostics: readonly string[];
}

/**
 * Streaming read+write with no reuse and no arithmetic.
 *
 * `vec4` accesses, one element per invocation, fully coalesced. The sum guards
 * against the compiler eliminating the loads.
 */
function bandwidthShader(useF16: boolean): string {
  const T = useF16 ? 'f16' : 'f32';
  return `${useF16 ? 'enable f16;\n' : ''}
@group(0) @binding(0) var<storage, read> src: array<vec4<${T}>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<${T}>>;
@group(0) @binding(2) var<uniform> count: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= count) { return; }
  dst[i] = src[i];
}
`;
}

/**
 * Dependent-free FMA chain with negligible memory traffic.
 *
 * Sixteen independent accumulators so the pipeline is not stalled on a
 * dependency chain, and the result is written out so nothing is dead code.
 * `UNROLL` fused multiply-adds per loop iteration, on `vec4`.
 */
function aluShader(useF16: boolean, unroll: number): string {
  const T = useF16 ? 'f16' : 'f32';
  const V = `vec4<${T}>`;
  const lanes = 16;
  const decl = Array.from({ length: lanes }, (_, i) => `  var a${i}: ${V} = seed + ${V}(${T}(${i}));`).join('\n');
  const body = Array.from({ length: lanes }, (_, i) => `      a${i} = fma(a${i}, k, seed);`).join('\n');
  const reduce = Array.from({ length: lanes }, (_, i) => `a${i}`).join(' + ');
  return `${useF16 ? 'enable f16;\n' : ''}
@group(0) @binding(0) var<storage, read> src: array<${V}>;
@group(0) @binding(1) var<storage, read_write> dst: array<${V}>;
@group(0) @binding(2) var<uniform> iters: u32;

const UNROLL: u32 = ${unroll}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let seed = src[gid.x];
  let k = seed + ${V}(${T}(1.0));
${decl}
  for (var t: u32 = 0u; t < iters; t = t + 1u) {
    for (var u: u32 = 0u; u < UNROLL; u = u + 1u) {
${body}
    }
  }
  dst[gid.x] = ${reduce};
}
`;
}

/** Runs one compute pass repeatedly and returns the median GPU timestamp span. */
async function timePass(
  device: GPUDevice,
  label: string,
  code: string,
  buffers: readonly GPUBuffer[],
  dispatchX: number,
  iterations: number,
): Promise<{ medianMs: number; samples: number; diagnostics: string[] }> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label, code });
  const pipeline = device.createComputePipeline({ label, layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolve = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const dispatch = (timed: boolean): GPUCommandBuffer => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(
      timed ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {},
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchX, 1, 1);
    pass.end();
    if (timed) {
      encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, staging, 0, 16);
    }
    return encoder.finish();
  };

  // Same duration-based warm-up as ConvBench, for the same clock-ramp reason.
  const start = performance.now();
  for (let batch = 0; batch < 256 && performance.now() - start < 60; batch++) {
    for (let i = 0; i < 8; i++) device.queue.submit([dispatch(false)]);
    await device.queue.onSubmittedWorkDone();
  }

  const diagnostics: string[] = [];
  const validation = await device.popErrorScope();
  if (validation) diagnostics.push(`validation ${validation.message}`);
  const info = await module.getCompilationInfo();
  for (const m of info.messages) if (m.type !== 'info') diagnostics.push(`${m.type} ${m.message}`);

  const samples: number[] = [];
  for (let i = 0; i < iterations; i++) {
    device.queue.submit([dispatch(true)]);
    await staging.mapAsync(GPUMapMode.READ);
    const [begin, end] = new BigInt64Array(staging.getMappedRange().slice(0));
    staging.unmap();
    if (begin !== undefined && end !== undefined) {
      const ms = Number(end - begin) / NS_PER_MS;
      if (ms > 0) samples.push(ms);
    }
  }

  for (const b of [resolve, staging]) b.destroy();
  querySet.destroy();

  samples.sort((a, b) => a - b);
  return {
    medianMs: samples.length > 0 ? (samples[Math.floor(samples.length / 2)] as number) : NaN,
    samples: samples.length,
    diagnostics,
  };
}

/**
 * Measures achievable streaming bandwidth and FMA throughput.
 *
 * `useF16` selects the element type for both probes so the comparison against
 * an f16 convolution is like-for-like.
 */
export async function measureRoofline(device: GPUDevice, useF16: boolean): Promise<RooflineResult[]> {
  const bytesPerElement = useF16 ? 2 : 4;
  const results: RooflineResult[] = [];

  // --- bandwidth -----------------------------------------------------------
  // 64 MiB each way: large enough that no cache holds it, small enough to fit
  // the default 128 MiB storage binding limit.
  const vec4Count = (64 * 1024 * 1024) / (bytesPerElement * 4);
  const bufBytes = vec4Count * bytesPerElement * 4;
  const src = device.createBuffer({ size: bufBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const dst = device.createBuffer({ size: bufBytes, usage: GPUBufferUsage.STORAGE });
  const countBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(countBuf, 0, new Uint32Array([vec4Count]));

  const bw = await timePass(
    device,
    'roofline:bandwidth',
    bandwidthShader(useF16),
    [src, dst, countBuf],
    Math.ceil(vec4Count / 256),
    30,
  );
  results.push({
    label: `bandwidth ${useF16 ? 'f16' : 'f32'}`,
    medianMs: bw.medianMs,
    iterations: bw.samples,
    // Read once, written once.
    gbPerSecond: (bufBytes * 2) / (bw.medianMs / 1000) / 1e9,
    gflopPerSecond: null,
    diagnostics: bw.diagnostics,
  });

  // --- arithmetic ----------------------------------------------------------
  const aluThreads = 256 * 1024;
  const aluSrc = device.createBuffer({
    size: aluThreads * bytesPerElement * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const aluDst = device.createBuffer({ size: aluThreads * bytesPerElement * 4, usage: GPUBufferUsage.STORAGE });
  const iters = 64;
  const unroll = 8;
  const itersBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(itersBuf, 0, new Uint32Array([iters]));

  const alu = await timePass(
    device,
    'roofline:alu',
    aluShader(useF16, unroll),
    [aluSrc, aluDst, itersBuf],
    aluThreads / 256,
    30,
  );
  // 16 accumulators x 4 lanes x 2 flops per fma, per unrolled step.
  const flops = aluThreads * iters * unroll * 16 * 4 * 2;
  results.push({
    label: `fma ${useF16 ? 'f16' : 'f32'}`,
    medianMs: alu.medianMs,
    iterations: alu.samples,
    gbPerSecond: null,
    gflopPerSecond: flops / (alu.medianMs / 1000) / 1e9,
    diagnostics: alu.diagnostics,
  });

  for (const b of [src, dst, countBuf, aluSrc, aluDst, itersBuf]) b.destroy();
  return results;
}
