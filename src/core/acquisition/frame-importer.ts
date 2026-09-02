import type { FrameTexture, FrameTextureKind, Size } from '../types.js';

/**
 * Format used by the copy fallback. `rgba8unorm` is the only 8-bit format that
 * `copyExternalImageToTexture` is guaranteed to accept as a destination across
 * implementations.
 */
const FALLBACK_FORMAT: GPUTextureFormat = 'rgba8unorm';

/**
 * Moves the currently displayed video frame into GPU-visible form.
 *
 * Two strategies, chosen once at construction:
 *
 * - `external` — `GPUDevice.importExternalTexture()`. We hand the browser the
 *   video element and get a sampleable handle back. Chromium *can* wrap the
 *   decoder's IOSurface with no copy on Metal, but only when several internal
 *   conditions hold, and whether it did is exposed only behind developer
 *   features. Treat this as "the import API", not as a guaranteed no-copy
 *   path. This is the path Milestone 1 targets.
 * - `sampled` — `GPUQueue.copyExternalImageToTexture()` into a texture we own
 *   and reuse. One explicit GPU-side copy per frame, still no CPU readback.
 *   Used when the UA does not offer external textures, and forceable for
 *   testing.
 *
 * Neither path ever reads pixels back to JavaScript.
 */
export class FrameImporter {
  readonly kind: FrameTextureKind;

  private fallbackTexture: GPUTexture | null = null;
  private fallbackFrame: FrameTexture | null = null;
  private fallbackSize: Size = { width: 0, height: 0 };

  constructor(
    private readonly device: GPUDevice,
    private readonly video: HTMLVideoElement,
    supportsExternalTexture: boolean,
  ) {
    this.kind = supportsExternalTexture ? 'external' : 'sampled';
  }

  /**
   * Produces the frame handle for the current tick.
   *
   * The returned value is valid only for the current task: a
   * `GPUExternalTexture` expires when control returns to the event loop, so it
   * must be bound and submitted before this function is called again.
   */
  acquire(size: Size): FrameTexture {
    if (this.kind === 'external') {
      // A small handle object per frame; whether the browser copies the image
      // data behind it is implementation-defined and not observable here.
      return { kind: 'external', texture: this.device.importExternalTexture({ source: this.video }) };
    }
    return this.copyIntoOwnedTexture(size);
  }

  private copyIntoOwnedTexture(size: Size): FrameTexture {
    let texture = this.fallbackTexture;
    let frame = this.fallbackFrame;
    if (
      texture === null ||
      frame === null ||
      this.fallbackSize.width !== size.width ||
      this.fallbackSize.height !== size.height
    ) {
      texture?.destroy();
      texture = this.device.createTexture({
        label: 'aethervsr:frame-fallback',
        size: { width: size.width, height: size.height },
        format: FALLBACK_FORMAT,
        usage:
          GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      // The view is created once and reused; only the texture contents change.
      frame = { kind: 'sampled', view: texture.createView() };
      this.fallbackTexture = texture;
      this.fallbackFrame = frame;
      this.fallbackSize = size;
    }

    this.device.queue.copyExternalImageToTexture(
      { source: this.video },
      { texture },
      { width: size.width, height: size.height },
    );
    return frame;
  }

  destroy(): void {
    this.fallbackTexture?.destroy();
    this.fallbackTexture = null;
    this.fallbackFrame = null;
    this.fallbackSize = { width: 0, height: 0 };
  }
}
