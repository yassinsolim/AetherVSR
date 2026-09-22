# AetherVSR

**Primary product direction: AetherVSR Desktop.** M11 is a local-file Electron
player for macOS / Apple Silicon, reusing the production WebGPU engine, and is
published as **DESKTOP PLAYER MVP READY**. The browser extension and its results
below are preserved, not the primary product surface. M12 closed as
**CROSS-VENDOR DESKTOP PARTIAL** and M12.1 as **CROSS-VENDOR COMPLETION PARTIAL**;
unavailable NVIDIA/AMD/Intel hardware remains unqualified.
[Roadmap](ROADMAP.md), [M12 report](docs/M12-REPORT.md),
[M12.1 report](docs/M12.1-REPORT.md).

**M13 Phase 1: METAL INFERENCE QUALIFIED.** The exact production C16D2 model
passes native M5 f32/f16 golden, falsification and actual WebGPU parity checks.
This isolated offline engine is not a real-time player: unfused 720p whole-graph
GPU p50 is 83.98 ms (f32) / 65.85 ms (f16), 60 samples each after ten warmups.
Production remains WebGPU; no video-ingest, capture or native GUI was added.
[Full scope, timing windows and publication status](docs/M13-PHASE1-METAL.md).

**M13 Phase 1.5: METAL REALTIME BACKEND QUALIFIED.** Optimized candidate F retains
the exact model and passes f32/f16 golden, falsification and fresh WebGPU parity.
Two fixed M5 f16 binding runs measured whole-graph GPU p50/p95 of
3.922/4.458 ms and 3.933/4.452 ms, each 60 samples after ten warmups.
This qualifies isolated 720p-to-1440p neural timing, not video playback or a
production rollout. Phase 1 and its simple backend remain unchanged; Phase 2
has not started. [Optimization report and scope](docs/M13-PHASE1.5-METAL-OPTIMIZATION.md).

Real-time 2x neural video super-resolution in the browser, with measured gains
on selected captured-content and degradation classes. Broad web-video
improvement remains under validation.

AetherVSR upscales video in the browser using WebGPU, entirely on your machine —
no uploads, no server. Apple Silicon is a first-class target; the architecture
is cross-platform through WebGPU.

**Status: M10.7 presentation synchronization FAIL; extension MVP PARTIAL.**
Known scroll/source invalidations now hide stale output before reconciliation,
but production S1 still misses unannounced layout changes. Stronger diagnostic
proof guards passed local pilots but did not qualify under the fixed cost gates;
S2 remains research-only. Candidate-dependent public/final acceptance is gated,
not passed. Independent lifecycle and six exact parity cases are retained with
their initial failures and scoped rechecks. The former cross-vendor M11 gate is
historical; desktop cross-vendor results remain partial. The unpacked extension
does not guarantee synchronized replacement of page video.
[Full30-section report](docs/M10.7-REPORT.md).

**M10.8 design study: NO ARCHITECTURE QUALIFIED.** Cheap sentinel and periodic
proof prototypes missed CSSOM semantics; the tested anchor lease failed restoration
and the wrapper violated host layout. Costs were not run after safety/ownership
stops. Production and historical verdicts are unchanged. This is research, not a
presentation rollout. [Architecture feasibility report](docs/M10.8-REPORT.md).

**M10.9: NO USEFUL OBSERVABLE CONTRACT QUALIFIED.** The simple-chain research
contract passed19 preliminary local cases but did not satisfy required fullscreen
recovery; two host-mutation ownership models stopped on unsafe restoration/layout.
Costs were NOT RUN. This is not proof that all browser contracts are impossible.
The next recommendation is a separately authorized controlled-player/window study,
not more generic replacement implementation. Production S1 and all historical
verdicts remain unchanged. [Observable contract report](docs/M10.9-REPORT.md).

**M10.10: NO CONTROLLED WEB PRODUCT PATH QUALIFIED.** Incomplete research
checkpoint: controlled-player parity and progressive acquisition were recorded,
but timing uncertainty remains unbounded and capture/authority/audio coverage is
incomplete. Candidate neural performance and public census were NOT RUN. This is
not browser impossibility or authorization for M10.11. Production is unchanged.
[Controlled-player study](docs/M10.10-PRODUCT-REPORT.md).

**M10.10R: CONTROLLED REPLAY PATH NOT QUALIFIED.** The first replay timing
calibration ended before its audio render clock reached the scheduled controls.
No candidate handover, A/V, neural playback or soak ran. This is an instrument
blocker, not proof that progressive replay is impossible. M10.10 history and
production remain unchanged; M10.11 is not authorized.
[Replay qualification report](docs/M10.10R-REPORT.md).

**M10.10RI: DIGITAL TIMING INSTRUMENT NOT QUALIFIED.** The render-clock-driven
short control verified all180 waveform windows, but the following baseline media
recorder stopped on discontinuity before its timing window. Three remaining runs
were NOT_RUN; no digital A/V bound or repeatability was earned. The finite instrument
investigation is exhausted. No replay qualification or M10.11 follows automatically.
[Instrument report](docs/M10.10RI-REPORT.md).

**M10.6 history remains PARTIAL, Case D.** Four ten-minute native-video
controls exceeded the1% combined callback/quality-loss reference, averaging
4.262931%. This is a lean-instrumented browser result, not physical display loss
or justification for normalization. Paired uncertainty and one Baseline late-window
failure prevent practical noninferiority; model/controller thresholds are unchanged.

Video.js reproduced the near-end stall in3/3 native and3/3 installed-inactive
controls, but all six active journeys failed binding presentation checks. Formal
public causality/scope remains unresolved, not "no added harm." Final native
lifecycle matrices pass45/45 and49/49; six output-parity cases pass exactly.
Shaka passed one of two checks; the other rejected offscreen video. Plyr remains
unsupported. Full results and limits: [docs/M10.6-REPORT.md](docs/M10.6-REPORT.md).

**M10.5 history remains PARTIAL.** Both preregistered ten-minute M5
trials failed the unchanged 1% delivery-loss gate: **3.48% and 2.90%**, at
58.74 and 58.91 rendered fps. Both final 120-second rate gates passed. The
counter audit found possible overlap, not a measurable unique-loss correction;
the original metric and controller thresholds remain unchanged.

The M10.5 MV3 candidate passed 45 production and 49 diagnostic native journeys,
including 20 cleanup cycles and same-document reactivation after Chrome's
extensions-page Reload control. Six output-parity cases match exactly. Generic
coincident rounded clips and verified size-query containers are supported.
Shaka passed repeatedly; Plyr retains a dynamic control-stack rejection. Video.js
passed an earlier candidate but stalled near the source's end in both final
sessions. Runtime-API reload automation remains unverified. Full evidence, limits
and reproduction details are in [docs/M10.5-REPORT.md](docs/M10.5-REPORT.md).

**M10 history remains PARTIAL.** Its ten-minute run retained neural at 58.3515888645 rendered fps and
59.811586918 presented fps, but **4.8011814863% combined callback/decoder loss
failed the absolute 1% gate**. Passing relative GPU overhead does not waive
that failure. See the separate M10 environment, results and measurement scopes
in [BENCHMARKS.md](BENCHMARKS.md) and the full
[M10 report](docs/M10-REPORT.md).

**M9 history: complete; runtime controller replaced, weights retained.** The
production 6,291-parameter C16D2 weights remain unchanged through M10.7. Auto performance selects
neural or Catmull-Rom from runtime evidence, with hysteresis and real neural
recovery probes; Baseline mode never probes. This does not predict scene quality.

In M9 on the M5, a ten-minute constant-frame-rate 720p60 run retained neural with
59.61 rendered fps, 0.876% combined callback/decoder loss, and no fallbacks or
errors. Neural GPU p50/p95 were 6.31/9.89 ms. Some short normal rows exceeded
the 1% loss reference, including the original irregular-cadence clip; this is
not a universal stutter-free claim. Exact environment, scopes, failed checks
and overload/recovery results are in [docs/M9-REPORT.md](docs/M9-REPORT.md).

Milestone 8 tested R0/R3 at 81,180 updates with matched effective initialization
and paired training streams. Fixed-final R3 minus R0 averaged -15.7896 dB across
three seeds; original-best checkpoints also lost (-0.2167 dB). The registered
decision is **NO SELECTED ADVANTAGE; production is unchanged**. Exact paired
sign-flip p=0.25 cannot establish conventional significance with three pairs.
The initial MPS scoring pass was withdrawn and all captured scores regenerated
with the verified CPU reference. No candidate or confirmation scoring followed.
See [BENCHMARKS.md](BENCHMARKS.md) for the CPU evidence, unresolved deterioration
and the separate historical M7 result. M9 performs no new model training.

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

- Standalone harness: local video decoded into an `HTMLVideoElement`, never
  displayed directly. The extension instead keeps the original page video
  under an owned canvas and reveals it whenever enhancement is unsupported.
- Both consumers reuse `RuntimeDriver`, `RuntimeController` and `VideoPipeline`;
  the extension does not introduce a second inference or runtime policy path.
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

### Install the unpacked extension

Run these commands separately from the repository root:

```bash
npm ci
npm run build:extension
```

The build generates `dist-extension`. In Chrome, open `chrome://extensions`,
turn on **Developer mode**, select **Load unpacked**, and choose that directory.
On an HTTP(S) page, open the native extension action and select **Enable current
page**. Mode is saved per origin; activation is per document, not a persistent
site grant. Navigation to a new document requires explicit activation again.

M10 was tested with Chrome for Testing 153.0.8010.12 on an Apple M5. Its three
permissions are `activeTab`, `scripting` and `storage`: no permanent host access,
web-accessible resources (WAR), or MAIN-world execution. Model delivery is
extension-owned; page scripts receive no privileged bridge.

**Supported scope:** the top document and accessible open shadow roots, with
supported visible geometry. Iframes and closed shadow roots are outside scope.
Native video controls and showing native text tracks are unsupported. Native
PiP and direct-video fullscreen reveal the original video; supported container
fullscreen can retain enhancement. Unsupported geometry or media security
also leaves original playback available. The page remains authoritative for
playback, audio, source, seeking and controls: no page-owned node, style or media
attribute is modified, replaced or reparented; only extension-owned DOM is added
and removed.

The measured M10.5 production payload is **265,958 bytes**. Its SHA256 is
`a796a9ec1b0a1f050f992d3d712bf1164799eeb073ca23d5035a61199ee228c3`.
The unchanged 6,291-parameter C16D2 model SHA256 is
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.
This is a load-unpacked MVP, not store publication or broad player compatibility.

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
| `npm run build:extension` | Generate the load-unpacked MV3 extension in `dist-extension` |

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

Resource-lifetime and control contracts use mocked WebGPU objects in unit tests.
Those tests check allocation, ownership and control flow, not hardware pixel
correctness or performance. Real browser/GPU evidence is separate: M10 passed
3/3 production external-import and 3/3 test sampled-import parity cases with
exact input and output RGBA hashes against the same harness within each route.
These were paused, instrumented replays, not SR-quality or real-time performance
measurements. Native extension journeys and live timing runs supply different
evidence; their scopes and unresolved gates are in [BENCHMARKS.md](BENCHMARKS.md).

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

M10.6 public-test tooling uses `m3u8-parser` 7.2.0 (Apache-2.0), verified from
the installed package's LICENSE. It parses already observed HLS master responses
for catalog identity only. It is a development dependency, excluded from the
production extension and shared frame-processing path; upstream notices remain
in the installed package. No third-party implementation source was copied.

M10.9 offline screenshot analysis uses the existing Python environment's Pillow
12.3.0 and ImageCms (MIT-CMU), verified from its installed license metadata and
LICENSE. It interprets embedded ICC profiles outside the frame loop. The package
and its notices remain installed separately; no Pillow source is copied, bundled
in the extension, or newly added to the JavaScript dependencies.

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
