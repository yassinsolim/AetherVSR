import type { GpuContext } from '../core/gpu/device.js';
import { describeAdapter } from '../core/gpu/device.js';
import { FRAME_BUDGET_60HZ_MS } from '../core/metrics/stats.js';
import type { Aggregate, PipelineStats } from '../core/pipeline.js';

/** Refresh rate of the overlay. Deliberately far below frame rate. */
export const OVERLAY_INTERVAL_MS = 250;

const ROWS = [
  'source',
  'output',
  'sourceFps',
  'renderFps',
  'skipped',
  'decoder',
  'decode',
  'gpuPass',
  'cpuFrame',
  'latency',
  'budget',
  'upscaler',
  'importPath',
  'clock',
  'adapter',
  'device',
] as const;

type RowKey = (typeof ROWS)[number];

const ROW_LABELS: Record<RowKey, string> = {
  source: 'source',
  output: 'output',
  sourceFps: 'presented fps',
  renderFps: 'rendered fps',
  skipped: 'skipped frames',
  decoder: 'decoder drops',
  decode: 'decode latency',
  gpuPass: 'gpu upscale',
  cpuFrame: 'cpu per frame',
  latency: 'callback lag',
  budget: '60hz budget',
  upscaler: 'upscaler',
  importPath: 'frame import',
  clock: 'frame clock',
  adapter: 'adapter',
  device: 'device',
};

/** `12.3` / `n/a`. Keeps "not measured" visually distinct from "zero". */
function ms(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function formatAggregate(agg: Aggregate | null, unmeasuredReason: string): string {
  if (agg === null) return unmeasuredReason;
  if (agg.samples === 0) return 'awaiting samples';
  return `${ms(agg.mean)} ms avg · p50 ${ms(agg.p50)} · p95 ${ms(agg.p95)} · max ${ms(agg.max)} (n=${agg.samples})`;
}

/**
 * Diagnostic overlay.
 *
 * Plain DOM, no framework, no virtual DOM diffing: sixteen text nodes updated
 * four times a second. It is driven by a timer rather than the frame callback
 * so that measuring the pipeline cannot perturb the pipeline — string
 * formatting and layout in the hot path would show up in our own numbers.
 */
export class DiagnosticOverlay {
  readonly element: HTMLElement;
  private readonly values: Partial<Record<RowKey, HTMLElement>> = {};
  private readonly adapterText: string;
  private readonly deviceText: string;

  constructor(gpu: GpuContext) {
    this.element = document.createElement('div');
    this.element.className = 'overlay';

    const table = document.createElement('dl');
    table.className = 'overlay-grid';
    for (const key of ROWS) {
      const label = document.createElement('dt');
      label.textContent = ROW_LABELS[key];
      const value = document.createElement('dd');
      value.textContent = '—';
      table.append(label, value);
      this.values[key] = value;
    }
    this.element.append(table);

    this.adapterText = describeAdapter(gpu.adapterReport);
    const caps = gpu.capabilities;
    this.deviceText = `${caps.preferredCanvasFormat} · maxTex2D ${caps.maxTextureDimension2D} · ${
      caps.timestampQuery ? 'timestamp-query' : 'no timestamp-query'
    }`;
  }

  update(stats: PipelineStats): void {
    const gpuAgg = stats.gpuPassMs;
    const budgetSource = gpuAgg && gpuAgg.samples > 0 ? gpuAgg.mean : Number.NaN;

    this.set('source', `${stats.sourceSize.width}x${stats.sourceSize.height}`);
    this.set(
      'output',
      `${stats.targetSize.width}x${stats.targetSize.height} (${stats.scaleFactor.toFixed(2)}x)`,
    );
    this.set('sourceFps', `${stats.meanSourceFps.toFixed(1)} mean · ${stats.sourceFps.toFixed(1)} now`);
    this.set(
      'renderFps',
      `${stats.meanRenderFps.toFixed(1)} mean · ${stats.renderFps.toFixed(1)} now · ` +
        `${stats.framesRendered} frames in ${(stats.elapsedMs / 1000).toFixed(1)} s`,
    );
    this.set('skipped', `${stats.framesSkipped}`);
    this.set(
      'decoder',
      `${stats.quality.droppedVideoFrames} dropped / ${stats.quality.totalVideoFrames} decoded` +
        (stats.quality.corruptedVideoFrames > 0
          ? ` · ${stats.quality.corruptedVideoFrames} corrupted`
          : ''),
    );
    this.set('decode', formatAggregate(stats.decodeLatencyMs, 'not reported by browser'));
    this.set(
      'gpuPass',
      formatAggregate(gpuAgg, 'not measured (timestamp-query unavailable)') +
        (stats.importPath === 'sampled' && gpuAgg && gpuAgg.samples > 0
          ? ' — render pass only; excludes the import copy'
          : ''),
    );
    this.set('cpuFrame', formatAggregate(stats.cpuFrameMs, ''));
    this.set('latency', formatAggregate(stats.callbackLatencyMs, ''));
    this.set(
      'budget',
      Number.isFinite(budgetSource)
        ? `${((budgetSource / FRAME_BUDGET_60HZ_MS) * 100).toFixed(1)}% of ${FRAME_BUDGET_60HZ_MS.toFixed(2)} ms · ${
            stats.importPath === 'external'
              ? 'upscale render pass (whole stage)'
              : 'upscale render pass only, EXCLUDES import copy'
          }`
        : 'not measured',
    );
    this.set('upscaler', `${stats.upscalerLabel}${stats.neural ? '' : ' · non-neural'}`);
    this.set(
      'importPath',
      // Deliberately not labelled "zero-copy": whether Chromium takes its
      // internal no-copy branch is only observable behind developer features,
      // and we did not enable them. What we can state is which API we called.
      stats.importPath === 'external'
        ? 'importExternalTexture (no JS-side copy; internal path not observable)'
        : 'copyExternalImageToTexture (explicit GPU copy)',
    );
    this.set(
      'clock',
      stats.clock === 'rvfc' ? 'requestVideoFrameCallback' : 'requestAnimationFrame (degraded)',
    );
    this.set('adapter', this.adapterText);
    this.set('device', this.deviceText);
  }

  private set(key: RowKey, text: string): void {
    const node = this.values[key];
    if (node && node.textContent !== text) node.textContent = text;
  }
}
