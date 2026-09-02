import { describe, expect, it } from 'vitest';
import { buildBaselineShader } from '../src/core/upscale/baseline.wgsl.js';

/**
 * These assertions guard the one shader property that cannot be checked by the
 * type system and that silently produces a black canvas when wrong: external
 * textures and 2D textures are different WGSL types with different sampling
 * builtins, and mixing them fails only at pipeline-creation time in a browser.
 */
describe('buildBaselineShader', () => {
  it('binds an external texture and clamps to edge on the zero-copy path', () => {
    const src = buildBaselineShader('external', 'bilinear');
    expect(src).toContain('var srcTex: texture_external;');
    expect(src).toContain('textureSampleBaseClampToEdge(srcTex, srcSampler, uv)');
    expect(src).not.toContain('textureSampleLevel');
  });

  it('binds a 2D texture and samples level 0 on the copy fallback path', () => {
    const src = buildBaselineShader('sampled', 'bilinear');
    expect(src).toContain('var srcTex: texture_2d<f32>;');
    expect(src).toContain('textureSampleLevel(srcTex, srcSampler, uv, 0.0)');
    expect(src).not.toContain('texture_external');
  });

  it('emits exactly one source fetch for bilinear', () => {
    const src = buildBaselineShader('external', 'bilinear');
    expect(src).not.toContain('catmullRom');
    expect(countSampleCalls(src)).toBe(0);
  });

  it('emits the nine bilinear-fused taps for Catmull-Rom, not the naive sixteen', () => {
    // Sixteen taps would be correct but ~1.7x the texture traffic; nine is the
    // performance claim this project makes, so it is worth pinning.
    const src = buildBaselineShader('external', 'catmull-rom');
    expect(countSampleCalls(src)).toBe(9);
  });

  it('produces the same kernel regardless of source binding kind', () => {
    const external = buildBaselineShader('external', 'catmull-rom');
    const sampled = buildBaselineShader('sampled', 'catmull-rom');
    expect(countSampleCalls(external)).toBe(countSampleCalls(sampled));
  });

  it('clamps the negative lobes of the cubic kernel', () => {
    const src = buildBaselineShader('external', 'catmull-rom');
    expect(src).toContain('clamp(catmullRom(in.uv).rgb');
  });
});

/** Counts `sampleSource(...)` call sites inside the kernel (not its definition). */
function countSampleCalls(src: string): number {
  return src.split('\n').filter((line) => line.includes('acc += sampleSource(')).length;
}
