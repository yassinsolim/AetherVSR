import type { Size } from '../types.js';

/**
 * Owns the output canvas and its WebGPU swap chain.
 *
 * The backing store is sized to exactly `source * scaleFactor` — never to
 * `devicePixelRatio` — because the whole point of the project is a known,
 * exact upscale ratio. CSS then letterboxes that fixed buffer into whatever
 * space the page gives it. Conflating the two is the classic way to end up
 * silently benchmarking a 1.5x scaler on a Retina display.
 */
export class CanvasTarget {
  readonly context: GPUCanvasContext;
  private configured: { format: GPUTextureFormat; size: Size } | null = null;

  constructor(readonly canvas: HTMLCanvasElement) {
    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new Error("canvas.getContext('webgpu') returned null; WebGPU canvas support is missing");
    }
    this.context = context;
  }

  get size(): Size {
    return { width: this.canvas.width, height: this.canvas.height };
  }

  /** (Re)configures the swap chain. Cheap and idempotent when nothing changed. */
  configure(device: GPUDevice, size: Size, format: GPUTextureFormat): void {
    const unchanged =
      this.configured !== null &&
      this.configured.format === format &&
      this.configured.size.width === size.width &&
      this.configured.size.height === size.height;
    if (unchanged) return;

    // Exact upscaled resolution, set before configuring the swap chain.
    this.canvas.width = size.width;
    this.canvas.height = size.height;
    this.context.configure({
      device,
      format,
      // The upscaler writes fully opaque pixels; `opaque` lets the compositor
      // skip blending the canvas against the page.
      alphaMode: 'opaque',
      // Match the destination colour space that `importExternalTexture()`
      // converts video into by default, so no implicit re-encode happens
      // between the upscale pass and the compositor.
      colorSpace: 'srgb',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.configured = { format, size };
  }

  /**
   * The swap-chain texture view for this frame.
   *
   * `getCurrentTexture()` must be called once per frame: the texture it
   * returns is only valid until the frame is presented.
   */
  currentView(): GPUTextureView {
    return this.context.getCurrentTexture().createView();
  }

  unconfigure(): void {
    this.context.unconfigure();
    this.configured = null;
  }
}
