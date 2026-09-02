import type { FrameTick } from '../types.js';

/**
 * Which clock is driving the pipeline.
 *
 * `rvfc` is the only mode that is synchronised to *presented video frames*;
 * `raf` ticks at display refresh rate regardless of video cadence and is a
 * degraded fallback for browsers without `requestVideoFrameCallback`
 * chosen by feature detection. Benchmarks taken under `raf` are not
 * comparable to `rvfc` benchmarks and must be labelled as such.
 */
export type FrameClockKind = 'rvfc' | 'raf';

export const FRAME_CLOCK_SUPPORTED: FrameClockKind =
  typeof HTMLVideoElement !== 'undefined' && 'requestVideoFrameCallback' in HTMLVideoElement.prototype
    ? 'rvfc'
    : 'raf';

/** Decoder-level counters, distinct from the compositor-level skip count. */
export interface PlaybackQuality {
  readonly totalVideoFrames: number;
  readonly droppedVideoFrames: number;
  readonly corruptedVideoFrames: number;
}

const EMPTY_QUALITY: PlaybackQuality = {
  totalVideoFrames: 0,
  droppedVideoFrames: 0,
  corruptedVideoFrames: 0,
};

export type FrameTickHandler = (tick: FrameTick) => void;

/**
 * Turns an `HTMLVideoElement` into a stream of {@link FrameTick}s.
 *
 * This is the only place in the codebase that knows about video elements or
 * frame callbacks. It performs no GPU work and holds no GPU resources, which
 * keeps the acquisition boundary independent of the upscaler and presenter.
 */
export class VideoFrameSource {
  readonly clock: FrameClockKind;

  private active = false;
  private handle: number | null = null;
  private handler: FrameTickHandler | null = null;
  private lastPresentedFrames = -1;
  private generation = 0;

  constructor(
    private readonly video: HTMLVideoElement,
    clock: FrameClockKind = FRAME_CLOCK_SUPPORTED,
  ) {
    this.clock = clock;
    // The media load algorithm zeroes the playback-quality counters. Consumers
    // that report deltas need a deterministic signal that this happened;
    // inferring it from a counter going backwards misses the case where the
    // new resource overtakes the old total between two polls.
    video.addEventListener('loadstart', () => this.generation++);
  }

  /**
   * Increments whenever the element starts loading a new resource, and
   * therefore whenever {@link quality} restarts from zero.
   */
  get loadGeneration(): number {
    return this.generation;
  }

  get running(): boolean {
    return this.active;
  }

  start(handler: FrameTickHandler): void {
    if (this.active) throw new Error('VideoFrameSource is already running');
    this.active = true;
    this.handler = handler;
    this.lastPresentedFrames = -1;
    this.schedule();
  }

  /**
   * Safe to call from inside a tick handler: `handle` is already null at that
   * point, so the running state is tracked separately from the callback
   * registration rather than inferred from it.
   */
  stop(): void {
    if (!this.active) return;
    this.active = false;
    this.handler = null;
    if (this.handle === null) return;
    if (this.clock === 'rvfc') {
      this.video.cancelVideoFrameCallback(this.handle);
    } else {
      cancelAnimationFrame(this.handle);
    }
    this.handle = null;
  }

  quality(): PlaybackQuality {
    if (typeof this.video.getVideoPlaybackQuality !== 'function') return EMPTY_QUALITY;
    const q = this.video.getVideoPlaybackQuality();
    return {
      totalVideoFrames: q.totalVideoFrames,
      droppedVideoFrames: q.droppedVideoFrames,
      corruptedVideoFrames: q.corruptedVideoFrames,
    };
  }

  private schedule(): void {
    // Both callbacks are one-shot: re-registration every frame is required by
    // spec, not an oversight.
    this.handle =
      this.clock === 'rvfc'
        ? this.video.requestVideoFrameCallback((now, metadata) => this.onRvfc(now, metadata))
        : requestAnimationFrame((now) => this.onRaf(now));
  }

  private onRvfc(now: number, metadata: VideoFrameCallbackMetadata): void {
    const handler = this.handler;
    if (handler === null) return;
    this.handle = null;

    const presentedDelta =
      this.lastPresentedFrames < 0 ? 1 : Math.max(0, metadata.presentedFrames - this.lastPresentedFrames);
    this.lastPresentedFrames = metadata.presentedFrames;

    handler({
      now,
      mediaTime: metadata.mediaTime,
      size: { width: metadata.width, height: metadata.height },
      presentedDelta,
      presentationTime: metadata.presentationTime,
      expectedDisplayTime: metadata.expectedDisplayTime,
      // rVFC reports seconds and marks the field optional; absent must stay
      // distinguishable from zero.
      decodeLatencyMs:
        metadata.processingDuration === undefined ? null : metadata.processingDuration * 1000,
    });

    // Only reschedule if this callback is still the live one. A handler that
    // calls stop() then start() has already registered a fresh callback, and
    // rescheduling here too would double the callback rate.
    if (this.active && this.handle === null) this.schedule();
  }

  private onRaf(now: number): void {
    const handler = this.handler;
    if (handler === null) return;
    this.handle = null;

    // Without rVFC there is no per-frame metadata. Approximate the presented
    // count from the decoder's total, and take geometry from the element.
    const total = this.quality().totalVideoFrames;
    const presentedDelta = this.lastPresentedFrames < 0 ? 1 : Math.max(0, total - this.lastPresentedFrames);
    this.lastPresentedFrames = total;

    handler({
      now,
      mediaTime: this.video.currentTime,
      size: { width: this.video.videoWidth, height: this.video.videoHeight },
      presentedDelta,
      presentationTime: now,
      expectedDisplayTime: now,
      decodeLatencyMs: null,
    });

    // Only reschedule if this callback is still the live one. A handler that
    // calls stop() then start() has already registered a fresh callback, and
    // rescheduling here too would double the callback rate.
    if (this.active && this.handle === null) this.schedule();
  }
}
