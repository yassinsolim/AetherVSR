import { acquireGpu } from '../../src/core/gpu/device.js';
import type { ModelFile } from '../../src/core/neural/model.js';
import { verifyGolden, type GoldenVectors, type GoldenResult } from '../../src/bench/golden-verify.js';
import golden from '../../public/models/golden-c16d2.json';

declare const __DESKTOP_STAGE_GOLDEN__: boolean;
const output = document.querySelector<HTMLOutputElement>('#result')!;
const MODEL_URL = 'aethervsr://app/models/production.json';

function serialise(result: GoldenResult): GoldenResult {
  return JSON.parse(JSON.stringify(result)) as GoldenResult;
}

async function runGolden(): Promise<void> {
  if (!__DESKTOP_STAGE_GOLDEN__) throw new Error('Stage golden renderer requires the diagnostic stage build');
  const gpu = await acquireGpu({ optionalFeatures: ['shader-f16', 'timestamp-query'] });
  if (gpu.adapterReport.fallbackAdapter) throw new Error('Software WebGPU adapter rejected');
  const response = await fetch(MODEL_URL); if (!response.ok) throw new Error(`Model request failed: ${response.status}`);
  const model = await response.json() as ModelFile;
  const vectors = golden as GoldenVectors;
  if (vectors.modelSha256 !== model.sha256) throw new Error('Golden/model SHA mismatch');
  const useF16 = gpu.device.features.has('shader-f16');
  const runs: Array<{ precision: 'f16' | 'f32'; result: GoldenResult }> = [];
  runs.push({ precision: 'f32', result: serialise(await verifyGolden(gpu.device, model, vectors, false)) });
  if (useF16) runs.push({ precision: 'f16', result: serialise(await verifyGolden(gpu.device, model, vectors, true)) });
  output.value = JSON.stringify({ schema: 'aethervsr.m12.1.stage-golden/1', source: 'shared verifyGolden', modelSha256: model.sha256,
    adapter: gpu.adapterReport, features: [...gpu.device.features], limits: gpu.capabilities.reportedLimits, externalTexture: gpu.capabilities.externalTexture,
    timestampQuery: gpu.capabilities.timestampQuery, runs, outcome: runs.every(run => run.result.passed) ? 'PASS' : 'FAIL' });
  gpu.device.destroy();
}

void runGolden().catch(error => { output.value = JSON.stringify({ schema: 'aethervsr.m12.1.stage-golden/1', outcome: 'FAIL', error: String(error) }); });
