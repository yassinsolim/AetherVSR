import { acquireGpu } from '../../core/gpu/device.js';
import type { ModelFile } from '../../core/neural/model.js';
import { verifyGolden, type GoldenVectors } from '../golden-verify.js';

declare const __DESKTOP_STAGE_GOLDEN__: boolean;
const output = document.querySelector<HTMLOutputElement>('#result')!;
const digest = async (bytes: ArrayBuffer): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');

async function run(): Promise<void> {
  if (!__DESKTOP_STAGE_GOLDEN__) throw new Error('M13 export requires the diagnostic stage build');
  const modelResponse = await fetch('aethervsr://app/models/production.json');
  const goldenResponse = await fetch('aethervsr://app/models/golden-c16d2.json');
  if (!modelResponse.ok || !goldenResponse.ok) throw new Error('Model/golden request failed');
  const modelBytes = await modelResponse.arrayBuffer();
  const goldenBytes = await goldenResponse.arrayBuffer();
  const model = JSON.parse(new TextDecoder().decode(modelBytes)) as ModelFile;
  const golden = JSON.parse(new TextDecoder().decode(goldenBytes)) as GoldenVectors;
  const modelBytesSha256 = await digest(modelBytes);
  const goldenBytesSha256 = await digest(goldenBytes);
  if (modelBytesSha256 !== 'd76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a' ||
      goldenBytesSha256 !== '7ffe8d5c26ef02f605c04e9057dd5ef9767dc83e84695211eff39f04a1e3a759') throw new Error('Fixture byte identity mismatch');
  const input = Float32Array.from(golden.input);
  const inputBytes = new ArrayBuffer(input.length * 4);
  const view = new DataView(inputBytes);
  for (let index = 0; index < input.length; index++) view.setFloat32(index * 4, input[index] as number, true);
  const inputFloat32Sha256 = await digest(inputBytes);
  const gpu = await acquireGpu({ optionalFeatures: ['shader-f16'] });
  const errors: string[] = [];
  gpu.device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const runs = [];
  try {
    if (gpu.adapterReport.fallbackAdapter || !gpu.device.features.has('shader-f16')) throw new Error('Physical adapter with shader-f16 required');
    for (const precision of ['f32', 'f16'] as const) {
      const { capture, ...summary } = await verifyGolden(gpu.device, model, golden, precision === 'f16', precision === 'f16' ? 0.05 : 0.001, true);
      if (!capture) throw new Error('GPU tensor capture missing');
      runs.push({ schema: 'aethervsr.m13.webgpu-golden/1', modelBytesSha256, modelIdentity: model.sha256,
        goldenBytesSha256, inputFloat32Sha256, precision, stages: capture.stages, rgba: capture.rgba,
        summary, finalFloat: capture.finalFloat, rgbaAgreement: capture.rgbaAgreement,
        outcome: summary.passed ? 'PASS' : 'FAIL' });
    }
    await gpu.device.queue.onSubmittedWorkDone();
    output.value = JSON.stringify({ schema: 'aethervsr.m13.webgpu-capture/1', adapter: gpu.adapterReport,
      features: [...gpu.device.features], limits: gpu.capabilities.reportedLimits, errors, runs,
      outcome: errors.length === 0 && runs.every(value => value.outcome === 'PASS') ? 'PASS' : 'FAIL' });
  } finally { gpu.device.destroy(); }
}

void run().catch(error => { output.value = JSON.stringify({ schema: 'aethervsr.m13.webgpu-capture/1', outcome: 'FAIL', error: String(error) }); });