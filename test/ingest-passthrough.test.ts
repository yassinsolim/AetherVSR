import { describe, expect, it } from 'vitest';
import { ExternalTextureIngest } from '../src/core/ingest/external-texture-ingest.js';
import type { FrameTexture, PassTiming } from '../src/core/types.js';

/**
 * The ingest stage records a render pass only for an external texture. A frame
 * that already arrived as an ordinary texture — the `copyExternalImageToTexture`
 * fallback, used on any browser without external textures — is passed straight
 * through.
 *
 * That distinction is load-bearing for measurement, not just efficiency: a
 * caller that claims a GPU timestamp slot for a pass that is never recorded
 * resolves a query nothing wrote, and publishes a fabricated duration.
 */
describe('ExternalTextureIngest.writesPass', () => {
  const external: FrameTexture = {
    kind: 'external',
    texture: {} as unknown as GPUExternalTexture,
  };
  const sampled: FrameTexture = {
    kind: 'sampled',
    view: {} as unknown as GPUTextureView,
  };

  it('reports that an external frame needs a conversion pass', () => {
    expect(ExternalTextureIngest.writesPass(external)).toBe(true);
  });

  it('reports that an already-sampleable frame does not', () => {
    expect(ExternalTextureIngest.writesPass(sampled)).toBe(false);
  });

  it('refuses timestamps for a passthrough rather than silently dropping them', () => {
    // Encoding a sampled frame records nothing, so timestamp slots handed to it
    // would never be written. Failing here is what stops a zero-length span
    // from being published as an ingest cost.
    const ingest = new ExternalTextureIngest();
    const timing: PassTiming = {
      querySet: {} as unknown as GPUQuerySet,
      beginIndex: 0,
      endIndex: 1,
    };
    expect(() =>
      ingest.encode({} as unknown as GPUCommandEncoder, sampled, timing),
    ).toThrow(/records no pass/);
  });

  it('passes a sampled frame straight through when no timing is claimed', () => {
    const ingest = new ExternalTextureIngest();
    const view = ingest.encode({} as unknown as GPUCommandEncoder, sampled, null);
    expect(view).toBe(sampled.view);
  });
});
