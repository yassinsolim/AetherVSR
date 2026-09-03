/**
 * Cross-stage contracts for the AetherVSR pipeline.
 *
 * The stage boundaries defined here are the project's load-bearing
 * architectural commitment: frame acquisition, upscaling and presentation must
 * remain replaceable in isolation. A neural super-resolution backend is
 * expected to arrive as nothing more than a new {@link Upscaler}.
 */

/** Integer pixel dimensions. */
export interface Size {
  readonly width: number;
  readonly height: number;
}

/**
 * A single video frame already resident in GPU memory.
 *
 * `external` is the import path: the browser hands us a `GPUExternalTexture`
 * wrapping the decoder's output (typically still YUV planes, with conversion
 * folded into the sample instruction). Whether the browser avoided a copy
 * internally is implementation-defined and not observable from the page, so
 * this variant means "we called `importExternalTexture()`", not "no copy
 * occurred". `sampled` is the path where the frame was explicitly copied into
 * a regular `GPUTexture` by us.
 *
 * Consumers must handle both. The variant is stable for the lifetime of a
 * session, so pipelines can be specialised once in {@link Upscaler.configure}
 * rather than branching per frame.
 */
export type FrameTexture =
  | { readonly kind: 'external'; readonly texture: GPUExternalTexture }
  | { readonly kind: 'sampled'; readonly view: GPUTextureView };

/** The `kind` discriminant of {@link FrameTexture}, usable before a frame exists. */
export type FrameTextureKind = FrameTexture['kind'];

/** Everything an {@link Upscaler} needs to build its pipelines exactly once. */
export interface UpscalerConfig {
  readonly device: GPUDevice;
  /** Decoded video dimensions. */
  readonly source: Size;
  /** Canvas backing-store dimensions. */
  readonly target: Size;
  /** Format of the texture view passed to {@link Upscaler.encode}. */
  readonly targetFormat: GPUTextureFormat;
  /** Which {@link FrameTexture} variant this session will deliver. */
  readonly sourceKind: FrameTextureKind;
}

/**
 * GPU timestamp slots the stage must attach to its passes so the harness can
 * measure real GPU time rather than command-recording time.
 *
 * A stage writes `beginIndex` at the start of its *first* pass and `endIndex`
 * at the end of its *last* pass, so a multi-pass neural backend reports its
 * full cost rather than one convolution.
 */
export interface PassTiming {
  readonly querySet: GPUQuerySet;
  /** Omit to leave the opening timestamp to an earlier pass. */
  readonly beginIndex?: number;
  /** Omit to leave the closing timestamp to a later pass. */
  readonly endIndex?: number;
}

/** Per-frame inputs to {@link Upscaler.encode}. */
export interface EncodeContext {
  readonly encoder: GPUCommandEncoder;
  readonly frame: FrameTexture;
  readonly target: GPUTextureView;
  /** Null when the device does not support `timestamp-query`. */
  readonly timing: PassTiming | null;
}

/**
 * The replaceable image-processing stage.
 *
 * Contract:
 * - {@link configure} is called on start and whenever geometry or format
 *   changes. All pipelines, layouts, samplers and intermediate textures must be
 *   created here, never in {@link encode}.
 * - {@link encode} runs inside the per-frame hot path. It must not allocate GPU
 *   resources, must not read pixels back to the CPU, and must not await.
 *   Creating one bind group per frame is permitted only because
 *   `GPUExternalTexture` is single-frame-valid by specification.
 * - Implementations declare {@link scaleFactor} so the harness can size the
 *   output canvas without knowing the algorithm.
 */
export interface Upscaler {
  /** Stable machine-readable identifier, used in benchmark records. */
  readonly id: string;
  /** Short label for the diagnostic overlay. */
  readonly label: string;
  /** Output-to-input linear scale, e.g. 2 for 720p -> 1440p. */
  readonly scaleFactor: number;
  /** True when this stage runs a learned model. Always false in Milestone 1. */
  readonly neural: boolean;

  configure(config: UpscalerConfig): void;
  encode(ctx: EncodeContext): void;
  destroy(): void;
}

/** One rVFC delivery, normalised for the pipeline. */
export interface FrameTick {
  /** `performance.now()` at callback entry. */
  readonly now: number;
  /** Video timeline position of the frame being presented, in seconds. */
  readonly mediaTime: number;
  /** Decoded frame dimensions reported by the UA for this frame. */
  readonly size: Size;
  /**
   * Frames the compositor presented since the previous tick. `1` is the ideal
   * case; `>1` means the page did not get a callback for every presented frame
   * and we skipped some.
   */
  readonly presentedDelta: number;
  /** UA-reported presentation timestamp, in the `performance.now()` timebase. */
  readonly presentationTime: number;
  /** UA-estimated next display time, in the `performance.now()` timebase. */
  readonly expectedDisplayTime: number;
  /**
   * UA-reported time from submitting the encoded packet to the frame being
   * ready for presentation, in milliseconds, or null when unreported.
   *
   * This is a *latency*, not a per-frame cost: the decoder runs ahead of
   * presentation, so the interval includes queueing and routinely exceeds the
   * frame interval by an order of magnitude on a healthy pipeline. It is
   * useful for spotting a decoder in trouble; it must never be added to the
   * upscale time as though the two were consecutive costs. rVFC declares the
   * field optional and expresses it in seconds.
   */
  readonly decodeLatencyMs: number | null;
}
