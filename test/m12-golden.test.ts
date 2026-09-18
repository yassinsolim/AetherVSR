import { describe, it } from 'vitest';
import { execFileSync } from 'node:child_process';

describe('M12.1 native golden evidence contract', () => {
  it('rejects corrupted model identity, shape, stage, nonfinite and unsupported-adapter evidence', () => execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';import{validateGoldenEvidence}from'./tools/m12/golden.mjs';
    const stage=stageName=>({stage:stageName,maxAbsError:.001,meanAbsError:.0001,referenceRange:[0,1],elements:10,passed:true});
    const value={schema:'aethervsr.m12.1.stage-golden/1',outcome:'PASS',modelSha256:'${'d'.repeat(64)}',adapter:{fallbackAdapter:false},runs:[{precision:'f32',result:{passed:true,diagnostics:[],stages:[stage('stem'),stage('body.0'),stage('body.1')],output:{...stage('output'),passed:true}}}]};
    assert(validateGoldenEvidence(value,'${'d'.repeat(64)}'));
    for(const mutate of [v=>v.modelSha256='bad',v=>v.adapter.fallbackAdapter=true,v=>v.runs[0].result.stages[0].stage='missing',v=>v.runs[0].result.stages[0].nonFinite=1,v=>v.runs[0].result.stages[0].passed=false,v=>v.runs[0].result.output.passed=false]){const bad=structuredClone(value);mutate(bad);assert.throws(()=>validateGoldenEvidence(bad,'${'d'.repeat(64)}'));}
  `], { cwd: new URL('../', import.meta.url), encoding: 'utf8' }));
});