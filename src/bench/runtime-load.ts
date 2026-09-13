import type { EncodeContext, Upscaler, UpscalerConfig } from '../core/types.js';

export interface RuntimeLoad {
  passes: number;
  frames: number;
  every?: number;
}

export class LoadedUpscaler implements Upscaler {
  readonly neural = true;
  private frameIndex = 0;

  constructor(private readonly inner: Upscaler, private readonly load: RuntimeLoad) {
    if (!import.meta.env.DEV) throw new Error('Runtime load is development-only');
  }

  get id(): string { return this.inner.id; }
  get label(): string { return this.inner.label; }
  get scaleFactor(): number { return this.inner.scaleFactor; }

  configure(config: UpscalerConfig): void {
    this.inner.configure(config);
  }

  encode(context: EncodeContext): void {
    const passes = this.load.frames > 0 && this.frameIndex++ % (this.load.every ?? 1) === 0 ? this.load.passes : 0;
    if (passes > 0) this.load.frames--;
    for (let index = 0; index <= passes; index++) {
      this.inner.encode({ ...context, timing: context.timing ? {
        querySet: context.timing.querySet,
        ...(index === 0 && context.timing.beginIndex !== undefined ? { beginIndex: context.timing.beginIndex } : {}),
        ...(index === passes && context.timing.endIndex !== undefined ? { endIndex: context.timing.endIndex } : {}),
      } : null });
    }
  }

  destroy(): void {
    this.inner.destroy();
  }
}