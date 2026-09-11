// Browser half of the Milestone 7 equivalence chain, as executable code.
//
// The CPU half (tools/m7-equivalence.py --export-dir) writes fused models, their
// golden vectors, and the SHA-256 of both files. This script is what turns that
// into a WebGPU measurement: it re-hashes the bytes the browser actually fetched
// and refuses to dispatch if they differ from the CPU record, then runs the
// stage-by-stage comparison in both precisions.
//
// Recorded as a file rather than as prose in a protocol document, because a
// verification step that exists only as an unrepeatable manual action cannot be
// re-run, cannot be regression-tested, and cannot be honestly repeated for a
// future trained candidate.
//
// Run inside the bench page:
//   browser open  http://127.0.0.1:5173/bench.html
//   browser run   <this file's contents>
//
// It writes results/m7-linked-webgpu.json and throws on any mismatch.

const report = await tab.evaluate(async () => {
  const cpu = await (await fetch('/results/m7-linked-equivalence.json')).json();
  const adapter = await navigator.gpu.requestAdapter();
  const info = adapter.info;
  const out = {
    schema: 'aethervsr.m7-linked-webgpu/1',
    measuredAt: new Date().toISOString(),
    scope: 'Randomized R1/R2/R3 fusion proof; not trained quality candidates',
    userAgent: navigator.userAgent,
    adapter: {
      vendor: info.vendor, architecture: info.architecture,
      device: info.device, description: info.description,
    },
    cpuReport: '/results/m7-linked-equivalence.json',
    runs: [],
  };

  async function hash(bytes) {
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
  }

  for (const boundary of cpu.boundaries) {
    const a = boundary.artifacts;
    const modelUrl = '/' + a.model;
    const goldenUrl = '/' + a.golden;
    const modelResponse = await fetch(modelUrl);
    const goldenResponse = await fetch(goldenUrl);
    if (!modelResponse.ok || !goldenResponse.ok) throw new Error('Artifact fetch failed');
    const modelBytes = await modelResponse.arrayBuffer();
    const goldenBytes = await goldenResponse.arrayBuffer();
    const modelFileSha256 = await hash(modelBytes);
    const goldenFileSha256 = await hash(goldenBytes);
    // The bytes, not the self-declared label fields inside them. A model that
    // names the right SHA in its own metadata proves nothing about its contents.
    if (modelFileSha256 !== a.modelFileSha256 || goldenFileSha256 !== a.goldenFileSha256) {
      throw new Error(`Artifact bytes differ from the CPU proof: ${modelUrl}`);
    }

    const precisions = {};
    for (const useF16 of [false, true]) {
      if (useF16 && !adapter.features.has('shader-f16')) {
        precisions.f16 = { supported: false };
        continue;
      }
      const result = await window.aethervsrGolden(modelUrl, goldenUrl, useF16);
      precisions[useF16 ? 'f16' : 'f32'] = {
        supported: true,
        ...result,
        // The output stage is quantised to 8 bits by the storage texture, so it
        // carries a 1.5/255 floor; a tighter bound would measure the texture
        // format rather than the graph.
        outputTolerance: Math.max(result.tolerance, 1.5 / 255),
      };
    }
    out.runs.push({
      rung: boundary.rung, seed: boundary.seed, modelUrl, goldenUrl,
      modelSha256: a.modelSha256, modelFileSha256, goldenFileSha256, precisions,
    });
  }

  out.passed = out.runs.every((run) =>
    Object.values(run.precisions).every((p) => !p.supported || p.passed));
  return out;
});

const fs = await import('node:fs/promises');
await fs.writeFile(
  '/Users/ysoli/Projects/AetherVSR/results/m7-linked-webgpu.json',
  JSON.stringify(report, null, 1) + '\n',
);
if (!report.passed) throw new Error('WebGPU parity failed; see results/m7-linked-webgpu.json');
display({
  passed: report.passed,
  models: report.runs.length,
  adapter: report.adapter,
  artifact: 'results/m7-linked-webgpu.json',
});
