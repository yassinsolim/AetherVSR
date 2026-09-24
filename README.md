# AetherVSR

Local GPU video super-resolution, with an Electron/WebGPU desktop player and
a native Metal/AVFoundation research player. The production C16D2 model performs
2x upscaling on your machine; local video is not uploaded for processing.

## Current Status

**The primary product is AetherVSR Desktop.** Its local 720p30/60 H.264 MVP is
qualified on the measured macOS / Apple Silicon setup. The native Metal path
and browser extension have separate, narrower qualification limits.

| Surface | Current qualification | Evidence |
|---|---|---|
| Electron desktop / WebGPU | **DESKTOP PLAYER MVP READY** on measured Apple Silicon; cross-vendor completion remains **PARTIAL** | [M11](docs/M11-REPORT.md), [M12](docs/M12-REPORT.md), [M12.1](docs/M12.1-REPORT.md) |
| Native Metal inference | **METAL REALTIME BACKEND QUALIFIED** for isolated M5 candidate-F inference, not playback | [M13 Phase 1](docs/M13-PHASE1-METAL.md), [Phase 1.5](docs/M13-PHASE1.5-METAL-OPTIMIZATION.md) |
| Native AVFoundation playback | **NATIVE LOCAL PLAYBACK PARTIAL**; decoded-frame parity and lifecycle pass, real-time playback remains unqualified | [M13 Phase 2](docs/M13-PHASE2-NATIVE-PLAYBACK.md) |
| Browser extension | **PARTIAL**; page-video presentation synchronization remains unqualified | [M10.7](docs/M10.7-REPORT.md) |

The latest native binding baseline30 trial ran with an occluded window on M5:
110 useful presentation opportunities in 60.033667 seconds, **1.832 FPS** overall,
**1.75 FPS** in the final 20 seconds, and software-age p95 **981.263 ms**. It
failed the registered gates. Foreground performance, neural30, all 60 FPS timing
and the ten-minute soak remain unqualified or `NOT_RUN`; no favorable rerun was
performed. Both native test clips lack audio tracks, so audible playback and A/V
synchronization are not qualified by those tests. No Phase 3 or capture
integration has started. NVIDIA/AMD/Intel hardware remains unqualified.

[Benchmarks and measurement scopes](BENCHMARKS.md) | [Roadmap](ROADMAP.md) |
[Architecture](ARCHITECTURE.md) | [Engineering rules](AGENTS.md)

## Quick Start

Commands run from the repository root. These are source-build workflows;
there are no published release binaries or installers.

### Desktop Player

Requires **Node.js 22.12.0 or newer**. The qualified setup is macOS on Apple
Silicon; other desktop platforms are not hardware-qualified.

```sh
npm ci
npm run desktop:dev
```

The command builds and launches the Electron player. Select a local H.264/MP4;
the committed [30 FPS](public/media/aethervsr-testclip-720p30-h264.mp4) and
[60 FPS](public/media/aethervsr-testclip-720p60-h264.mp4) clips match the tested
720p scope. [Desktop behavior and limits](docs/M11-REPORT.md).

### Native Metal Player

Requires macOS 26 or newer and an Xcode toolchain with the macOS 26 SDK or newer.
The measured hardware is Apple M5. Run the separate Swift package:

```sh
swift run --package-path native -c release aether-player
```

Use **Open Video** to select a local 720p H.264/MP4. This is the **partial native
research path**, not a real-time-qualified replacement for the desktop player.
It needs no Screen Recording or Accessibility permission.
[Native playback scope](docs/M13-PHASE2-NATIVE-PLAYBACK.md).

### Web Harness

Requires Node.js 22.12.0+ and a Chromium browser with WebGPU.

```sh
npm ci
npm run dev
```

Open the printed URL. A bundled clip loads automatically; the clip picker
accepts local video. The harness is a development and measurement surface,
not the primary product. [Unpacked extension setup](#install-the-unpacked-extension)
is documented below with its compatibility limits.

## Milestone History

<details>
<summary>Earlier results, retained failures and qualification boundaries</summary>

The summaries below describe their original milestone scopes, not the current
status of every player. Historical measurements and verdicts remain unchanged.

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
production rollout. Phase 1 and its simple backend remain unchanged.
[Optimization report and scope](docs/M13-PHASE1.5-METAL-OPTIMIZATION.md).

**M13 Phase 2: NATIVE LOCAL PLAYBACK PARTIAL.** The macOS26 AVFoundation/Core
Video player passes decoded-frame f32/f16 parity, lifecycle and native resize
checks with unchanged F. The first binding baseline30 trial, recorded with an
occluded window, fails cadence and software age:1.832fps overall,1.75fps final20,
age p95 981.263ms. Foreground performance is unqualified; neural30, all60fps
timing and the ten-minute soak are NOT_RUN. No favorable rerun, capture API,
new permission or Phase3 followed. Audio fixtures contain no audio track.
[Full scope, reproduction and retained failures](docs/M13-PHASE2-NATIVE-PLAYBACK.md).

**M10.7: presentation synchronization FAIL; extension MVP PARTIAL.**
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

</details>

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


## WebGPU Pipeline

- The Electron desktop player, standalone harness and browser extension reuse
  the production WebGPU pipeline and C16D2 model. The native Metal player is
  a separate implementation with independently checked numerical parity.
- Desktop and harness video is decoded through an `HTMLVideoElement`. The
  extension preserves page playback and reveals the original video when
  enhancement is unsupported.
- `requestVideoFrameCallback()` as the frame clock, so work is synchronised to
  presented video frames rather than to display refresh.
- `GPUDevice.importExternalTexture()` for the frame import, with a
  `copyExternalImageToTexture()` fallback that can be forced for testing.
- Exact 2x output: 1280x720 in, 2560x1440 out, always.
- Production neural upscaling uses the 6,291-parameter C16D2 model, with runtime
  performance policy and explicit baseline controls.
- Two non-neural GPU scalers: hardware bilinear, and a 9-tap bilinear-fused
  Catmull-Rom bicubic.
- A diagnostic overlay with real GPU timings where `timestamp-query` is
  available, and an explicit "not measured" where it is not.
- No CPU pixel readback anywhere in the frame loop.

### Historical Non-Neural Baseline

These earlier harness measurements are not M13 native-playback results.
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

### Browser Support

The web harness and extension require WebGPU. Tested Chromium builds and
hardware are recorded in [BENCHMARKS.md](BENCHMARKS.md); Safari and Firefox
are not qualified. Without `requestVideoFrameCallback`, the harness uses an
explicitly degraded animation-frame clock, not an equivalent timing source.

**Record focus and visibility while measuring.** Backgrounded or occluded
windows can throttle frame callbacks. A failed or backgrounded run is not
discarded or relabeled as a foreground performance result.

## Install the unpacked extension

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

The production extension payload remains byte-identical to the M13 Phase 2
freeze: SHA256 `46cebd53b7665ce48772792d2ec2eb73057e915d93ef76374900f30a159ac551`.
The unchanged 6,291-parameter C16D2 model SHA256 is
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.
[Payload verification](results/m13-phase2-verification.json). This is a
load-unpacked MVP, not store publication or broad player compatibility.

## Development Commands

Each of these is a separate command; run them individually.

```bash
npm ci
npm run dev
npm run typecheck
npm run lint
npm run test
npm run build
```

| Command | Purpose |
|---|---|
| `npm ci` | Install exact locked dependencies |
| `npm run desktop:dev` | Build and launch the Electron desktop player |
| `npm run desktop:build` | Build the desktop app without launching it |
| `npm run desktop:check` | Desktop typecheck and lint |
| `npm run dev` | Vite dev server with the harness |
| `npm run typecheck` | `tsc --noEmit`, strict |
| `npm run lint` | ESLint (type-aware) |
| `npm run test` | Vitest unit tests (`npm test` is equivalent) |
| `npm run build` | Typecheck and production build |
| `npm run build:extension` | Generate the load-unpacked MV3 extension in `dist-extension` |
| `swift test --package-path native/macos --jobs 2` | Original Metal package CPU contracts; GPU tests opt-in |
| `swift test --package-path native --jobs 2` | Native bridge/playback CPU contracts; GPU tests opt-in |

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

`npm test` covers metrics, frame clocks, model and shader contracts, runtime
policy, desktop/extension ownership and the evidence checkers. CI also runs
Python experiment-integrity tests and both Swift packages' build/CPU contracts.
CPU and source-text tests do not establish GPU pixel correctness or performance.

At the M13 Phase 2 closure: **1,504 Vitest tests passed**, with three explicit
skips; **612 Python tests**, **80 opt-in fixture tests**, and **71 desktop tests**
passed. The desktop and opt-in groups overlap the main suite and are not additive.
Physical native suites passed **18 original Metal tests** and **7 playback tests**.
Hosted native CI skips GPU-only tests; set `AETHERVSR_METAL_TESTS=1` only for an
intentional physical-device run. [Verification and scopes](results/m13-phase2-verification.json).

Hardware correctness uses golden tensors, cross-backend comparisons and
same-decoded-frame CPU/GPU parity with retained numerical artifacts. Native
performance, lifecycle and long-run stability are separate acceptance gates.

Resource-lifetime and control contracts use mocked WebGPU objects in unit tests.
Those tests check allocation, ownership and control flow, not hardware pixel
correctness or performance. Real browser/GPU evidence is separate: M10 passed
3/3 production external-import and 3/3 test sampled-import parity cases with
exact input and output RGBA hashes against the same harness within each route.
These were paused, instrumented replays, not SR-quality or real-time performance
measurements. Native extension journeys and live timing runs supply different
evidence; their scopes and unresolved gates are in [BENCHMARKS.md](BENCHMARKS.md).

## Repository layout

```text
apps/desktop/                  Electron desktop player and build tooling
native/
  Package.swift                macOS 26 playback package
  macos/                       frozen original/optimized Metal engines and tests
  playback/                    Core Video bridge, F adapter, native player and tests
src/
  core/                        shared WebGPU pipeline, upscalers and runtime policy
  extension/                   preserved MV3 browser extension
  main.ts                      standalone harness entry
  ui/                          harness controls and metrics
public/models/                 production model and golden reference tensors
public/media/                  committed test clips
test/                          TypeScript contracts and evidence-checker tests
tools/                         experiment, training, fixture and validation tooling
tools/m13/                     Metal and native-playback evidence runners/checkers
docs/                          phase protocols, reports and dataset documentation
results/                       compact, source-bound evidence indexes
index.html                     web harness
bench.html                     isolated WebGPU feasibility experiments
```

## Documentation

| File | Contents |
|---|---|
| [AGENTS.md](AGENTS.md) | Binding engineering rules for contributors |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Data flow and shared WebGPU stage contracts |
| [DECISIONS.md](DECISIONS.md) | Architecture decision records, including the native player |
| [BENCHMARKS.md](BENCHMARKS.md) | Measurement environments, results and non-results |
| [ROADMAP.md](ROADMAP.md) | Milestones and acceptance criteria |
| [M11 report](docs/M11-REPORT.md) | Qualified desktop MVP scope |
| [M13 Phase 1.5 report](docs/M13-PHASE1.5-METAL-OPTIMIZATION.md) | Optimized native Metal inference |
| [M13 Phase 2 report](docs/M13-PHASE2-NATIVE-PLAYBACK.md) | Native local playback results and remaining blockers |

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
