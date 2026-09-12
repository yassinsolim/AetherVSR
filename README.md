# AetherVSR

Real-time 2x neural video super-resolution in the browser, with measured gains
on selected captured-content and degradation classes. Broad web-video
improvement remains under validation.

AetherVSR upscales video in the browser using WebGPU, entirely on your machine —
no uploads, no server. Apple Silicon is a first-class target; the architecture
is cross-platform through WebGPU.

**Status: Milestone 8 measured; final publication checks pending.** A
6,291-parameter neural upscaler runs in the production pipeline at **5.7 ms
p50, ~35% of a 60 Hz frame budget**, 2560×1440 output from a 1280×720 source,
with automatic fallback to a conventional scaler when it cannot hold the budget.

Milestone 8 tested R0/R3 at 81,180 updates with matched effective initialization
and paired training streams. Fixed-final R3 minus R0 averaged -15.7896 dB across
three seeds; original-best checkpoints also lost (-0.2167 dB). The registered
decision is **NO SELECTED ADVANTAGE; production is unchanged**. Exact paired
sign-flip p=0.25 cannot establish conventional significance with three pairs.
The initial MPS scoring pass was withdrawn and all captured scores regenerated
with the verified CPU reference. No candidate or confirmation scoring followed.
See [BENCHMARKS.md](BENCHMARKS.md) for the CPU evidence, unresolved deterioration
and the separate historical M7 result; the runtime figure above predates M8.

## What the evidence supports

Measured on **17 independently sourced captured clips** that share no creator
with the training corpus, under controlled 720p H.264, against the production
Catmull-Rom baseline this project ships. The set was frozen before the model was
trained and read after model selection closed.

| Input quality | Gain over Catmull-Rom | Clips won | vs previous model |
| --- | ---: | ---: | ---: |
| high (CRF 18) | **+1.39 dB** | 17/17 | +0.42 dB |
| typical (CRF 26) | **+0.66 dB** | 17/17 | +0.19 dB |
| poor (CRF 34) | **+0.12 dB** | 13/17 | +0.04 dB |

Retraining on a corpus of 151 clips from 108 creators — against 12 clips from 8
creators — improved every compression tier at identical runtime. The inference
graph is unchanged; only the weights differ.

**What it does not support.** The gain is mostly *training length*, not corpus
size: at equal optimizer updates, twelve times the data is worth only +0.10 dB
and that is not statistically established. Two content classes are negative at
heavy compression (motion −0.09 dB, texture −0.01 dB, both on two clips). The
model remains **less temporally stable than Catmull-Rom**. Faces are measured
only on a narrow eight-clip set, of which three survive a creator-disjoint
filter. Most per-class results rest on two clips and are indicative only.

So: a real, reproducible improvement across the compression range on
independently sourced footage, with the caveat that more data is no longer the
lever it looked like. `BENCHMARKS.md` gives the full evidence, including every
withdrawn and corrected claim.

## Evaluation methodology

Quality claims are separated into three categories that are never averaged
together: **regression** artefacts (the synthetic reference and the eight-image
natural set, both overlapping the training corpus), **source-disjoint**
validation and test splits, and a **held-out independent** corpus from a
different institution. Only the last supports a generalisation claim.

The dataset is split by source image, not by patch, and CI enforces that no
photograph appears in two splits. Training never reads the test split. See
`DECISIONS.md` ADR-0028 and ADR-0029, and the Milestone 4.5 section of
`BENCHMARKS.md`, which also documents two defects in the Milestone 4 evaluation
and what was rerun because of them.


## What works today

- Local video decoded into an `HTMLVideoElement`, never displayed directly.
- `requestVideoFrameCallback()` as the frame clock, so work is synchronised to
  presented video frames rather than to display refresh.
- `GPUDevice.importExternalTexture()` for the frame import, with a
  `copyExternalImageToTexture()` fallback that can be forced for testing.
- Exact 2x output: 1280x720 in, 2560x1440 out, always.
- Two non-neural GPU scalers: hardware bilinear, and a 9-tap bilinear-fused
  Catmull-Rom bicubic.
- A diagnostic overlay with real GPU timings where `timestamp-query` is
  available, and an explicit "not measured" where it is not.
- No CPU pixel readback anywhere in the frame loop.

Measured on a MacBook Pro (Apple M5, 24 GB), Chrome for Testing 152, 720p60
H.264 → 1440p:

| Upscaler | Presented fps | Rendered fps | GPU upscale | 60 Hz budget |
|---|---:|---:|---|---:|
| Catmull-Rom 9-tap | 59.7 | 59.3 | 3.90 ms avg (p95 4.52) | 23.4% |
| Bilinear | 59.7 | 59.5 | 0.86 ms avg (p95 1.96) | 5.1% |

30 s run; FPS are means over the run, GPU timings aggregate the trailing 240
samples (~4 s at 60 fps). 1779 and 1786 frames upscaled respectively, with
13 and 6 presented frames skipped. Full environment, method, caveats and the things that were **not**
measured are in `BENCHMARKS.md`. Numbers here are never estimates — see
`AGENTS.md` §2.

## Requirements

- A Chromium-based browser with WebGPU (Chrome/Edge/Brave 113+). Safari 26+
  should work but has not been tested.
- Any browser without `requestVideoFrameCallback` falls back to a labelled
  degraded rAF clock. Firefox has supported rVFC since 132, so the clock is not
  the blocker there; its WebGPU availability is, and Firefox was not tested.
- Node.js 20+.

## Quick start

```bash
npm install
npm run dev
```

Open the printed URL. A bundled test clip loads automatically; use the **clip**
picker to load any local video instead (it never leaves your machine).

**Keep the window frontmost while measuring.** A backgrounded tab suspends
`requestVideoFrameCallback` and clamps timers, and every number becomes
meaningless.

### Commands

Each of these is a separate command; run them individually.

```bash
npm install
npm run dev
npm run typecheck
npm run lint
npm run test
npm run build
```

| Command | Purpose |
|---|---|
| `npm install` | Install dev dependencies |
| `npm run dev` | Vite dev server with the harness |
| `npm run typecheck` | `tsc --noEmit`, strict |
| `npm run lint` | ESLint (type-aware) |
| `npm run test` | Vitest unit tests (`npm test` is equivalent) |
| `npm run build` | Typecheck and production build |

### Reproducible runs

| Parameter | Values | Effect |
|---|---|---|
| `clip` | a path under `/media/...` | Selects the clip |
| `filter` | `catmull-rom`, `bilinear` | Selects the upscaler |
| `import` | `copy` | Forces the copy fallback import path |

```
http://127.0.0.1:5173/?clip=/media/aethervsr-testclip-720p60-h264.mp4&filter=catmull-rom
```

## Milestone 2 feasibility bench

`http://localhost:5173/bench.html` is a separate entry point holding the
Milestone 2 experiments. It is not part of the baseline harness — the ONNX
Runtime probe alone fetches a ~26 MB WASM artefact, which is why the two are
separate bundles.

Everything is driven from the console so a run is reproducible:

```js
await window.aethervsrEnvironment();
await window.aethervsrIngestBench({ mode: 'ingest', filter: 'catmull-rom', ingestFormat: 'rgba8unorm' }, 3000, 15000);
await window.aethervsrConvVerify([{ width: 17, height: 13, inChannels: 3, outChannels: 4, blockX: 2, activation: 'none', residual: false }]);
await window.aethervsrConvBench([{ label: '16->16', width: 1280, height: 720, inChannels: 16, outChannels: 16, tileX: 8, tileY: 8, blockX: 4, activation: 'relu', useF16: true, residual: false }]);
await window.aethervsrQualityBench(['bilinear', 'catmull-rom']);
await window.aethervsrOrtProbe({ modelUrl: '/models/conv16-720p.onnx', channels: 16, height: 720, width: 1280, iterations: 20, providers: ['webgpu'], outputLocation: 'cpu' });
```

Keep the window frontmost for `aethervsrIngestBench`: it is driven by
`requestVideoFrameCallback`, which a backgrounded tab suspends. The harness
throws rather than returning an empty result if that happens.

Regenerate the ONNX benchmark fixture with
`node tools/make-onnx-conv.mjs 16 public/models/conv16-720p.onnx 720 1280`.

## Test clips

`public/media/` holds four clips (1280x720, at 30 and 60 fps, in VP9/WebM and
H.264/MP4). The drawing is a pure function of frame index: a 1-to-16 px
frequency wedge, a radial zone plate, thin diagonals, small text, hard edges
and continuous sub-pixel motion — chosen to make resampling errors visible.

All four clips are original content generated by this repository's own tool
from a procedural drawing — no third-party video, imagery or audio is included
— so they are covered by the project's Apache-2.0 licence and are safe to
redistribute.

The **committed binaries are the benchmark reference**, not the recipe.
Regenerating on another machine will not be bit-identical: Canvas2D text
rasterisation depends on the font stack, and encoding depends on the browser
build and refresh rate.

Regenerate with `tools/make-test-clip.html`, served by the dev server:

```
http://127.0.0.1:5173/tools/make-test-clip.html?fps=60&seconds=4&codec=h264
```

Press **record**, then save the download into `public/media/`. The generator is
paced on `requestAnimationFrame` for a reason — see `DECISIONS.md` ADR-0009.

## Testing

`npm test` covers the logic that can be tested without a GPU: the statistics
and rate meters that produce every published number, and the frame-clock state
machine including skipped-frame accounting and stop/start reentrancy.

One test asserts on generated WGSL text — that the external-texture variant
uses `textureSampleBaseClampToEdge` and the 2D variant `textureSampleLevel`,
and that the bicubic kernel emits nine taps. That is a source-text assertion,
not a rendering test: it pins a contract whose violation fails at pipeline
creation inside a browser, where no unit test can reach it. Whether pixels are
correct is established by running the harness and looking.

GPU and video behaviour is verified by running the harness and reading the
overlay; the procedure is in `BENCHMARKS.md`. There are deliberately no mocked
WebGPU tests — a mock would assert that our mock works.

## Repository layout

```
src/
  main.ts                      harness bootstrap and wiring
  core/
    types.ts                   Upscaler, FrameTexture, FrameTick contracts
    pipeline.ts                stage orchestration, the hot path
    acquisition/
      video-source.ts          rVFC frame clock (no GPU)
      frame-importer.ts        external texture / copy fallback
    gpu/
      device.ts                adapter/device acquisition, capability probing
    upscale/
      baseline-scaler.ts       Milestone 1 Upscaler implementation
      baseline.wgsl.ts         WGSL, specialised per source kind and filter
    present/
      canvas-target.ts         swap chain, exact-2x backing store
    metrics/
      stats.ts                 SampleWindow, RateMeter (pure)
      gpu-timer.ts             timestamp-query pool
  ui/
    overlay.ts                 diagnostic overlay
    harness.css
index.html                     harness entry point
test/                          unit tests for the non-GPU logic
tools/make-test-clip.html      deterministic clip generator
public/media/                  committed test clips (30/60 fps, VP9 and H.264)
```

## Documentation

| File | Contents |
|---|---|
| `AGENTS.md` | Binding engineering rules for contributors |
| `ARCHITECTURE.md` | Data flow, stage contracts, where the neural stage goes |
| `DECISIONS.md` | Architecture decision records |
| `BENCHMARKS.md` | Benchmark format, environment, measured results, non-results |
| `ROADMAP.md` | Milestones and acceptance criteria |

## Prior art and licensing

AetherVSR is independently engineered. Other projects were read for
architectural ideas and are cited; no third-party source was copied. Licences
were established by reading each repository rather than assumed — the verdict
table is in `DECISIONS.md` ADR-0010.

Of note: **no `LICENSE` file was present in the `fishy-ops/WebVSR` snapshot we
reviewed, and the GitHub API reported `license: null`.** We treated it as
unlicensed for this project and copied nothing from it; anyone reusing it
should verify the current state themselves.

Selected references:

- `requestVideoFrameCallback` — https://wicg.github.io/video-rvfc/
- `GPUExternalTexture` — https://gpuweb.github.io/gpuweb/#gpuexternaltexture
- Media Playback Quality — https://w3c.github.io/media-playback-quality/
- WebVSR (WebGPU/WGSL SPAN-Lite browser SR) — https://github.com/fishy-ops/WebVSR
- StreamSR / EfRLFN, ICLR 2026 — https://arxiv.org/abs/2602.11339
- SPAN — https://github.com/hongyuanyu/SPAN (Apache-2.0)
- RLFN — https://github.com/bytedance/RLFN (Apache-2.0)
- ONNX Runtime Web, WebGPU EP — https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html

## Licence

Apache-2.0.
