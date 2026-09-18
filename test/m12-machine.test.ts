import { describe, it } from 'vitest';
import { execFileSync } from 'node:child_process';

describe('M12.1 portable machine workflow', () => {
  it('records explicit staged states and rejects noncanonical output roots', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';import{machineOutput,machineEnvironment}from'./tools/m12/validate-machine.mjs';
    assert(machineOutput('.cache/m12.1/test-machine').endsWith('/.cache/m12.1/test-machine'));
    for(const value of ['results/machine','../escape','.cache/m12/test'])assert.throws(()=>machineOutput(value));
    const env=machineEnvironment();assert.equal(typeof env.sourceCommit,'string');assert.equal(typeof env.modelSha256,'string');assert.equal(typeof env.electron,'string');
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));
});
