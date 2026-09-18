import { describe, it } from 'vitest';
import { execFileSync } from 'node:child_process';

describe('M12 cross-vendor envelope policy', () => {
  it('keeps the four binding classes and does not fabricate missing hardware', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { buildUnavailableEnvelope, VENDORS } from './tools/m12/envelope.mjs';
    assert.deepEqual(VENDORS, ['apple', 'nvidia', 'amd', 'intel']);
    for (const vendor of VENDORS.slice(1)) assert.deepEqual(buildUnavailableEnvelope(vendor), {
      vendor, machineVerdict: 'NOT_RUN_HARDWARE_UNAVAILABLE',
      reason: 'No qualifying physical local machine was available; software or CI adapters are not substitutes.'
    });
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));
});
