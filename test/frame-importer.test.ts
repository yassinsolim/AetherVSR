import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FrameImporter } from '../src/core/acquisition/frame-importer.js';

function makeHarness(external = false) {
  const textures: { createView: ReturnType<typeof vi.fn>; destroy: ReturnType<typeof vi.fn> }[] = [];
  const device = {
    createTexture: vi.fn(() => {
      const texture = { createView: vi.fn(() => ({})), destroy: vi.fn() };
      textures.push(texture);
      return texture;
    }),
    importExternalTexture: vi.fn(() => ({})),
    queue: { copyExternalImageToTexture: vi.fn() },
  };
  const video = {} as HTMLVideoElement;
  const importer = new FrameImporter(device as unknown as GPUDevice, video, external);
  return { importer, device, textures, video };
}

beforeEach(() => {
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 0x04, COPY_DST: 0x02, RENDER_ATTACHMENT: 0x10 });
});

afterEach(() => vi.unstubAllGlobals());

describe('FrameImporter configuration', () => {
  it('rejects unconfigured copy acquisition without allocating or copying', () => {
    const { importer, device } = makeHarness();
    expect(importer.sampledView).toBeNull();
    expect(() => importer.acquire({ width: 320, height: 180 })).toThrow(/configure\(\)/);
    expect(device.createTexture).not.toHaveBeenCalled();
    expect(device.queue.copyExternalImageToTexture).not.toHaveBeenCalled();
  });

  it('preallocates the copy texture and view, reusing both from the first frame', () => {
    const { importer, device, textures, video } = makeHarness();
    const size = { width: 320, height: 180 };
    importer.configure(size);
    const view = importer.sampledView;
    expect(view).not.toBeNull();
    expect(device.createTexture).toHaveBeenCalledExactlyOnceWith({
      label: 'aethervsr:frame-fallback', size, format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    expect(textures[0]!.createView).toHaveBeenCalledTimes(1);
    expect(device.queue.copyExternalImageToTexture).not.toHaveBeenCalled();

    const first = importer.acquire(size);
    expect(first).toEqual({ kind: 'sampled', view });
    for (let index = 0; index < 3; index++) expect(importer.acquire(size)).toBe(first);
    importer.configure({ ...size });
    expect(importer.sampledView).toBe(view);
    expect(device.createTexture).toHaveBeenCalledTimes(1);
    expect(textures[0]!.createView).toHaveBeenCalledTimes(1);
    expect(textures[0]!.destroy).not.toHaveBeenCalled();
    expect(device.queue.copyExternalImageToTexture).toHaveBeenCalledTimes(4);
    for (const args of device.queue.copyExternalImageToTexture.mock.calls) {
      expect(args).toEqual([{ source: video }, { texture: textures[0] }, size]);
    }
  });

  it.each([{ width: 640, height: 180 }, { width: 320, height: 360 }])(
    'requires reconfiguration for changed geometry %j and replaces only then', (resized) => {
      const { importer, device, textures } = makeHarness();
      const size = { width: 320, height: 180 };
      importer.configure(size);
      const previous = importer.sampledView;
      Object.assign(size, resized);
      expect(() => importer.acquire(size)).toThrow(/current size/);
      expect(device.createTexture).toHaveBeenCalledTimes(1);
      expect(device.queue.copyExternalImageToTexture).not.toHaveBeenCalled();
      expect(textures[0]!.destroy).not.toHaveBeenCalled();

      importer.configure(size);
      expect(textures[0]!.destroy).toHaveBeenCalledTimes(1);
      expect(importer.sampledView).not.toBe(previous);
      expect(importer.acquire(size)).toEqual({ kind: 'sampled', view: importer.sampledView });
      expect(device.createTexture).toHaveBeenCalledTimes(2);
      expect(textures[1]!.createView).toHaveBeenCalledTimes(1);
    },
  );

  it('clears prepared resources on destroy and can configure again', () => {
    const { importer, device, textures } = makeHarness();
    const size = { width: 320, height: 180 };
    importer.configure(size);
    const previous = importer.sampledView;
    importer.destroy();
    importer.destroy();
    expect(textures[0]!.destroy).toHaveBeenCalledTimes(1);
    expect(importer.sampledView).toBeNull();
    expect(() => importer.acquire(size)).toThrow(/configure\(\)/);
    expect(device.createTexture).toHaveBeenCalledTimes(1);
    importer.configure(size);
    expect(importer.sampledView).not.toBe(previous);
    expect(device.createTexture).toHaveBeenCalledTimes(2);
    importer.destroy();
    expect(textures[1]!.destroy).toHaveBeenCalledTimes(1);
  });

  it('leaves external configuration a no-op and imports each handle synchronously', () => {
    const { importer, device, video } = makeHarness(true);
    const size = { width: 320, height: 180 };
    expect(importer.acquire(size).kind).toBe('external');
    importer.configure(size);
    importer.configure({ width: 640, height: 360 });
    expect(importer.sampledView).toBeNull();
    expect(importer.acquire(size).kind).toBe('external');
    expect(device.importExternalTexture).toHaveBeenCalledTimes(2);
    expect(device.importExternalTexture).toHaveBeenLastCalledWith({ source: video });
    expect(device.createTexture).not.toHaveBeenCalled();
    expect(device.queue.copyExternalImageToTexture).not.toHaveBeenCalled();
  });
});