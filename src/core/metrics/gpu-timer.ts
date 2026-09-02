import type { PassTiming } from '../types.js';

const NS_PER_MS = 1_000_000;

interface Slot {
  readonly resolve: GPUBuffer;
  readonly staging: GPUBuffer;
  readonly beginIndex: number;
  readonly endIndex: number;
  busy: boolean;
  /** Measurement epoch this slot was claimed in; see {@link GpuTimer.newEpoch}. */
  epoch: number;
}

/**
 * Measures actual GPU execution time of the upscale pass using
 * `timestamp-query`.
 *
 * Why this exists: `performance.now()` around `encode()` measures command
 * *recording*, which on a healthy pipeline is a few tens of microseconds and
 * says nothing about GPU cost. AGENTS.md forbids reporting guessed numbers, so
 * the only honest per-frame processing figure comes from the device itself.
 *
 * Resource discipline: a fixed pool of slots is allocated up front. Each frame
 * takes a free slot, or gets `null` and simply goes unmeasured — the frame loop
 * never blocks and never allocates. Reading back 16 bytes of timestamps (two
 * 64-bit values) is not a pixel readback and does not stall the pipeline.
 *
 * Caveat for benchmark reports: Chromium quantises WebGPU timestamps for
 * fingerprinting resistance, so individual samples are coarse. Aggregate over
 * many frames, and see BENCHMARKS.md for the flag that disables quantisation.
 */
export class GpuTimer {
  private readonly querySet: GPUQuerySet;
  private readonly slots: Slot[] = [];
  private active: Slot | null = null;
  private lastMs = Number.NaN;
  private epoch = 0;

  constructor(
    device: GPUDevice,
    private readonly onSample: (ms: number) => void,
    poolSize = 4,
  ) {
    this.querySet = device.createQuerySet({
      label: 'aethervsr:timestamps',
      type: 'timestamp',
      count: poolSize * 2,
    });
    for (let i = 0; i < poolSize; i++) {
      this.slots.push({
        resolve: device.createBuffer({
          label: `aethervsr:timestamp-resolve:${i}`,
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
        staging: device.createBuffer({
          label: `aethervsr:timestamp-staging:${i}`,
          size: 16,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        beginIndex: i * 2,
        endIndex: i * 2 + 1,
        busy: false,
        epoch: 0,
      });
    }
  }

  /**
   * Starts a new measurement epoch, discarding readbacks still in flight.
   *
   * Up to `poolSize` frames can be awaiting `mapAsync()` at any moment. Without
   * this, samples measured before a stats reset — or worse, under the previous
   * upscaler after a runtime swap — would land in the freshly cleared window
   * and be attributed to the new configuration.
   */
  newEpoch(): void {
    this.epoch++;
    this.lastMs = Number.NaN;
  }

  /** Most recent measured pass duration in milliseconds, or NaN. */
  get last(): number {
    return this.lastMs;
  }

  /**
   * Claims a slot for the frame about to be encoded. Returns `null` when every
   * slot is still awaiting readback, in which case the frame is not measured.
   */
  begin(): PassTiming | null {
    const slot = this.slots.find((s) => !s.busy);
    if (!slot) return null;
    slot.busy = true;
    slot.epoch = this.epoch;
    this.active = slot;
    return { querySet: this.querySet, beginIndex: slot.beginIndex, endIndex: slot.endIndex };
  }

  /** Appends the resolve/copy commands. Must be called before submit. */
  end(encoder: GPUCommandEncoder): void {
    const slot = this.active;
    if (!slot) return;
    encoder.resolveQuerySet(this.querySet, slot.beginIndex, 2, slot.resolve, 0);
    encoder.copyBufferToBuffer(slot.resolve, 0, slot.staging, 0, 16);
  }

  /**
   * Releases a claimed slot without reading it back.
   *
   * Required when a frame throws between `begin()` and `afterSubmit()`: the
   * slot would otherwise stay `busy` forever, and after a few failures the
   * pool would be exhausted and GPU timing silently dead for the session.
   */
  abort(): void {
    const slot = this.active;
    if (!slot) return;
    this.active = null;
    slot.busy = false;
  }

  /** Starts the asynchronous readback. Must be called after submit. */
  afterSubmit(): void {
    const slot = this.active;
    if (!slot) return;
    this.active = null;
    void slot.staging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const [begin, end] = new BigInt64Array(slot.staging.getMappedRange());
        slot.staging.unmap();
        if (begin === undefined || end === undefined) return;
        // Dropped rather than published: this sample belongs to a measurement
        // window the caller has already discarded.
        if (slot.epoch !== this.epoch) return;
        const deltaNs = Number(end - begin);
        if (deltaNs >= 0) {
          this.lastMs = deltaNs / NS_PER_MS;
          this.onSample(this.lastMs);
        }
      })
      .catch(() => {
        // Device lost or buffer destroyed mid-flight; drop the sample.
      })
      .finally(() => {
        slot.busy = false;
      });
  }

  destroy(): void {
    this.active = null;
    for (const slot of this.slots) {
      slot.resolve.destroy();
      slot.staging.destroy();
    }
    this.slots.length = 0;
    this.querySet.destroy();
  }
}
