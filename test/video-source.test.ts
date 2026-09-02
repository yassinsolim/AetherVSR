import { afterEach, describe, expect, it, vi } from 'vitest';
import { VideoFrameSource } from '../src/core/acquisition/video-source.js';
import type { FrameTick } from '../src/core/types.js';

/**
 * Minimal stand-in for the parts of `HTMLVideoElement` the acquisition stage
 * touches. This is not a mock of WebGPU or of decoding: it is a controllable
 * clock, which is the only way to test skipped-frame accounting deterministically.
 */
class FakeVideo {
  currentTime = 0;
  videoWidth = 1280;
  videoHeight = 720;
  totalVideoFrames = 0;
  droppedVideoFrames = 0;
  corruptedVideoFrames = 0;

  private pending = new Map<number, VideoFrameCallback>();
  private nextHandle = 1;
  private listeners: Record<string, (() => void)[]> = {};
  cancelled: number[] = [];

  addEventListener(type: string, handler: () => void): void {
    (this.listeners[type] ??= []).push(handler);
  }

  /** Simulates the media load algorithm starting on a new resource. */
  emitLoadStart(): void {
    for (const handler of this.listeners['loadstart'] ?? []) handler();
  }

  requestVideoFrameCallback(cb: VideoFrameCallback): number {
    const handle = this.nextHandle++;
    this.pending.set(handle, cb);
    return handle;
  }

  cancelVideoFrameCallback(handle: number): void {
    this.cancelled.push(handle);
    this.pending.delete(handle);
  }

  getVideoPlaybackQuality(): VideoPlaybackQuality {
    return {
      creationTime: 0,
      totalVideoFrames: this.totalVideoFrames,
      droppedVideoFrames: this.droppedVideoFrames,
      corruptedVideoFrames: this.corruptedVideoFrames,
    };
  }

  /** Number of callbacks currently registered. Should never exceed one. */
  get registered(): number {
    return this.pending.size;
  }

  /** Fires every registered callback with the given metadata. */
  present(now: number, presentedFrames: number, overrides: Partial<VideoFrameCallbackMetadata> = {}): void {
    const callbacks = [...this.pending.values()];
    this.pending.clear();
    for (const cb of callbacks) {
      cb(now, {
        presentationTime: now - 2,
        expectedDisplayTime: now + 8,
        width: this.videoWidth,
        height: this.videoHeight,
        mediaTime: now / 1000,
        presentedFrames,
        ...overrides,
      });
    }
  }
}

type VideoFrameCallback = (now: number, metadata: VideoFrameCallbackMetadata) => void;

/** The fake implements exactly the surface `VideoFrameSource` consumes. */
function asVideo(fake: FakeVideo): HTMLVideoElement {
  return fake as unknown as HTMLVideoElement;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('VideoFrameSource (rVFC)', () => {
  it('re-registers exactly one callback per delivered frame', () => {
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    source.start(() => {});

    expect(fake.registered).toBe(1);
    fake.present(16, 1);
    expect(fake.registered).toBe(1);
    fake.present(33, 2);
    expect(fake.registered).toBe(1);
  });

  it('reports a delta of 1 for the first frame whatever the counter starts at', () => {
    // Playback may already be underway when the pipeline attaches; the first
    // tick must not be reported as thousands of skipped frames.
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    const ticks: FrameTick[] = [];
    source.start((t) => ticks.push(t));

    fake.present(1000, 5000);
    expect(ticks[0]?.presentedDelta).toBe(1);
  });

  it('derives skipped frames from the presentedFrames counter', () => {
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    const ticks: FrameTick[] = [];
    source.start((t) => ticks.push(t));

    fake.present(16, 1);
    fake.present(33, 2); // no skip
    fake.present(50, 5); // three presented, one callback -> delta 3
    expect(ticks.map((t) => t.presentedDelta)).toEqual([1, 1, 3]);
  });

  it('never reports a negative delta if the counter resets', () => {
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    const ticks: FrameTick[] = [];
    source.start((t) => ticks.push(t));

    fake.present(16, 90);
    fake.present(33, 3); // seek/loop rewound the counter
    expect(ticks[1]?.presentedDelta).toBe(0);
  });

  it('passes through per-frame geometry and timing from the metadata', () => {
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    const ticks: FrameTick[] = [];
    source.start((t) => ticks.push(t));

    fake.present(100, 1, { width: 640, height: 360, mediaTime: 3.5, presentationTime: 97 });
    expect(ticks[0]).toMatchObject({
      now: 100,
      mediaTime: 3.5,
      size: { width: 640, height: 360 },
      presentationTime: 97,
    });
  });

  it('converts the reported decode latency from seconds to milliseconds', () => {
    // rVFC reports seconds; every other timing in the pipeline is in ms.
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    const ticks: FrameTick[] = [];
    source.start((t) => ticks.push(t));

    fake.present(100, 1, { processingDuration: 0.0042 });
    expect(ticks[0]?.decodeLatencyMs).toBeCloseTo(4.2, 9);
  });

  it('reports a missing decode latency as null rather than zero', () => {
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    const ticks: FrameTick[] = [];
    source.start((t) => ticks.push(t));

    fake.present(100, 1);
    expect(ticks[0]?.decodeLatencyMs).toBeNull();
  });

  it('stops delivering and cancels the outstanding callback', () => {
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    let count = 0;
    source.start(() => count++);

    fake.present(16, 1);
    expect(source.running).toBe(true);
    source.stop();

    expect(source.running).toBe(false);
    expect(fake.cancelled).toHaveLength(1);
    fake.present(33, 2);
    expect(count).toBe(1);
  });

  it('advances the load generation when the element loads a new resource', () => {
    // Consumers report decoder counters as deltas; the element zeroes those
    // counters on load, so they need a deterministic signal that it happened
    // rather than having to notice a counter going backwards, which a new clip
    // can hide by overtaking the old total between two polls.
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    const before = source.loadGeneration;

    fake.emitLoadStart();
    expect(source.loadGeneration).toBe(before + 1);

    fake.emitLoadStart();
    expect(source.loadGeneration).toBe(before + 2);
  });

  it('does not double-schedule when the handler restarts the source', () => {
    // stop() inside the handler clears the live callback; start() registers a
    // fresh one. If the returning callback also rescheduled, the source would
    // run at twice the frame rate from then on.
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    let restarted = false;
    const handler = (): void => {
      if (restarted) return;
      restarted = true;
      source.stop();
      source.start(handler);
    };
    source.start(handler);

    fake.present(16, 1);
    expect(fake.registered).toBe(1);
  });

  it('stops cleanly when the handler itself calls stop', () => {
    // The pipeline does exactly this when a frame throws.
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    source.start(() => source.stop());

    fake.present(16, 1);
    expect(source.running).toBe(false);
    expect(fake.registered).toBe(0);
  });

  it('refuses to start twice', () => {
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    source.start(() => {});
    expect(() => source.start(() => {})).toThrow(/already running/);
  });

  it('restarts with fresh skip accounting', () => {
    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'rvfc');
    const ticks: FrameTick[] = [];
    source.start((t) => ticks.push(t));
    fake.present(16, 10);
    source.stop();

    source.start((t) => ticks.push(t));
    fake.present(33, 400);
    expect(ticks[1]?.presentedDelta).toBe(1);
  });
});

describe('VideoFrameSource (rAF fallback)', () => {
  it('derives frame deltas from decoder totals when rVFC is unavailable', () => {
    const callbacks: FrameRequestCallback[] = [];
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      callbacks.push(cb);
      return callbacks.length;
    });
    vi.stubGlobal('cancelAnimationFrame', () => {});

    const fake = new FakeVideo();
    const source = new VideoFrameSource(asVideo(fake), 'raf');
    const ticks: FrameTick[] = [];
    source.start((t) => ticks.push(t));

    fake.totalVideoFrames = 10;
    callbacks.pop()?.(16);
    fake.totalVideoFrames = 13;
    callbacks.pop()?.(33);

    expect(ticks.map((t) => t.presentedDelta)).toEqual([1, 3]);
    expect(ticks[0]?.size).toEqual({ width: 1280, height: 720 });
  });
});
