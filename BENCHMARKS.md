# BENCHMARKS.md

Every number in this file was measured on the machine described below. Nothing
here is estimated, extrapolated, or carried over from another project. Where a
measurement could not be taken, the row says so.

See `AGENTS.md` §2 and §3 for the rules this file exists to enforce.

## Benchmark record format

A benchmark result is only meaningful with its environment. Record all of:

| Field | Why it matters |
|---|---|
| Machine, SoC, memory | GPU architecture dominates every number here |
| OS and build | Media stack and Metal version |
| Browser and exact build | WebGPU and decoder behaviour changes between builds |
| Display refresh rate | Caps presented FPS; measure it, do not assume 60 |
| Window state | A backgrounded tab throttles timers and suspends rVFC entirely |
| Clip: codec, container, resolution, frame rate, source | Decode cost and cadence are codec-dependent |
| Import path | `importExternalTexture` vs `copyExternalImageToTexture` |
| Upscaler id | e.g. `baseline-catmull-rom` |
| GPU timestamp availability | Without `timestamp-query` there is no honest processing time |
| Measurement scope | Exactly what the timestamps bracket, and what they exclude |
| Sample count and window | An average over 12 frames is not a benchmark |

The **copy benchmark JSON** button emits everything in that list that is
observable from a web page: adapter identity and features, limits, canvas
format, DPR, document visibility and focus, clip path and decoded size,
playback rate, import path, upscaler, run duration, aggregation windows, and
all measurements. Machine, SoC, OS build, browser build and display refresh
rate are **not** web-exposed; the record lists them under `manualFields` and
they must be filled in by hand. See `benchmarkRecord()` in `src/main.ts`; the
schema id is `aethervsr.benchmark/1`.

### Metric definitions

| Metric | Definition | Trap it avoids |
|---|---|---|
| `presented fps` | Frames the compositor presented per second, from rVFC `presentedFrames` deltas | Not the same as decoded frames |
| `rendered fps` | Frames this pipeline actually upscaled per second | Diverges from presented when we miss callbacks |
| `skipped frames` | Presented frames that never reached a callback: `max(0, presentedDelta - 1)` summed, so a counter reset on seek or loop contributes zero rather than a negative | Distinct from decoder drops |
| `decoder drops` | `getVideoPlaybackQuality().droppedVideoFrames`, reported as a **delta since the last stats reset** (the underlying counter is cumulative from media load and cannot be zeroed) | A *different counter* from `skipped frames`, with a different definition; never subtract one from the other |
| `gpu upscale` | GPU execution time of the upscale render pass, from `timestamp-query` | Not CPU time, not frame time |
| `cpu per frame` | Main-thread wall time for the frame: the import call, command recording, and submit | Not GPU time. It excludes decode, compositing and present, so it is not "the frame's cost" either |
| `decode latency` | rVFC `processingDuration`: packet-submit to frame-ready | A **latency that includes decoder queueing**, not a per-frame cost. The decoder runs ahead of presentation, so this can exceed the frame interval by a wide margin while nothing is wrong. Never add it to upscale time. This document publishes no decode-latency figures |
| `callback lag` | `now - presentationTime` at callback entry | Negative values are normal (callback can precede the recorded presentation instant) |

### Measurement scope, stated explicitly

- **External import path:** the timestamps bracket the upscale render pass.
  AetherVSR records no separate import pass, so this is the full cost of the
  work *we* encode — but any GPU work the browser performs inside
  `importExternalTexture()` is outside the bracket and was not observed.
- **Copy fallback path:** the timestamps bracket the render pass **only**.
  `copyExternalImageToTexture()` is a queue operation outside our command
  encoder and cannot be bracketed by our pass timestamps, so its cost is
  **excluded**. Copy-path and external-path numbers therefore measure different
  things and must not be ranked against each other as total stage cost. The
  overlay and the JSON export both carry this caveat inline.

### Known measurement caveats

- Chromium quantises WebGPU timestamp values for fingerprinting resistance.
  Individual samples are coarse; aggregate over hundreds of frames. Runs here
  use n=240 samples.
- rVFC caps callback rate at the lesser of the video frame rate and the
  browser's paint rate. A 30 fps clip cannot demonstrate 60 fps operation.
- **A backgrounded or occluded window invalidates the run.** rVFC stops firing
  and `setTimeout` is clamped to 1 s. The harness must be frontmost.

## Environment for the runs below

| Field | Value |
|---|---|
| Machine | MacBook Pro, Apple M5, 24 GB unified memory |
| Display | 3024x1964 Retina, `devicePixelRatio` 2 |
| Measured paint rate | 59.2 Hz (`requestAnimationFrame` over 5 s, in the benchmarked window) |
| OS | macOS 26.6.2 (build 25G83) |
| Browser | Google Chrome for Testing 152.0.7977.42 |
| Launch flags | `--remote-debugging-port`, `--no-first-run`, `--enable-unsafe-webgpu`, `--autoplay-policy=no-user-gesture-required` |
| Window state | Headful, foregrounded via CDP `Page.bringToFront` before each run |
| WebGPU adapter | `apple` / `metal-3` (`device` and `description` masked by Chrome) |
| Preferred canvas format | `bgra8unorm` |
| `timestamp-query` | Available and used |
| Output | 2560x1440 in every run (exact 2x of 1280x720) |
| Warm-up | 5 s of playback, then counters reset |
| Run duration | 30 s after reset |
| Rate aggregation | `presented fps` / `rendered fps` are trailing 1 s sliding windows, sampled at the end of the run |
| Timing aggregation | Each timing statistic is over the trailing **240 samples**, not the whole run. At ~60 fps that is roughly the last 4 s; at 30 fps roughly the last 8 s. `framesRendered`, `skipped` and `decoder drops` are cumulative over the full 30 s |
| Date | 2026-09-01 |

Clips are the committed deterministic test patterns in `public/media/`,
generated by `tools/make-test-clip.html` (see README).

## Results

**Scope for every table below:** each `GPU upscale` figure brackets the
upscale render pass only. On external rows AetherVSR encodes no other GPU
work, but browser-internal import work is outside the bracket and unmeasured.
On the copy-fallback row the `copyExternalImageToTexture` upload is also
outside the bracket. Rows are therefore not directly rankable across import
paths. FPS columns are means over the full run; timing columns aggregate the
trailing 240 samples.

### 720p60 → 1440p, H.264 (the 60 fps target case)

| Upscaler | Import | Presented fps | Rendered fps | Frames | Skipped | Decoder drops | GPU upscale (ms) | 60 Hz budget | Scope |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Catmull-Rom 9-tap | external | 59.7 | 59.3 | 1779 | 13 | 13 / 1792 | 3.90 avg · p50 3.93 · p95 4.52 · max 5.49 | 23.4% | render pass; no other AetherVSR GPU work |
| Bilinear | external | 59.7 | 59.5 | 1786 | 6 | 6 / 1792 | 0.86 avg · p50 0.81 · p95 1.96 · max 4.29 | 5.1% | render pass; no other AetherVSR GPU work |
| Catmull-Rom 9-tap | copy fallback | 59.7 | 59.3 | 1779 | 13 | 12 / 1792 | 1.55 avg · p50 1.47 · p95 3.11 · max 4.48 | 9.3% | **render pass only, excludes import copy** |

`cpu per frame` was 0.11–0.13 ms average in all three runs.

The pipeline tracks a 60 fps source essentially frame for frame at the measured
59.2 Hz paint rate: 1779–1786 frames upscaled in 30 s, with 6–13 presented
frames skipped over the whole run (0.3–0.7%).

### 720p60 → 1440p, VP9

| Upscaler | Import | Presented fps | Rendered fps | Frames | Skipped | Decoder drops | GPU upscale (ms) | 60 Hz budget |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Catmull-Rom 9-tap | external | 59.9 | 59.5 | 1785 | 12 | 10 / 1797 | 2.43 avg · p50 2.32 · p95 3.59 · max 4.46 | 14.6% |

### 720p30 → 1440p

| Clip | Upscaler | Import | Presented fps | Rendered fps | Frames | Skipped | Decoder drops | GPU upscale (ms) | 60 Hz budget |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| VP9 | Catmull-Rom 9-tap | external | 29.9 | 29.9 | 899 | 0 | 0 / 898 | 2.55 avg · p50 2.37 · p95 4.09 · max 5.82 | 15.3% |
| VP9 | Bilinear | external | 29.9 | 29.9 | 898 | 0 | 0 / 899 | 0.48 avg · p50 0.36 · p95 1.38 · max 4.04 | 2.9% |
| H.264 | Catmull-Rom 9-tap | external | 30.1 | 30.1 | 903 | 0 | 0 / 903 | 4.64 avg · p50 4.42 · p95 6.50 · max 8.25 | 27.8% |

At 30 fps the pipeline is frame-perfect: zero skipped frames and zero decoder
drops over 30 s in all three runs.

### Re-verification run — 2026-09-02

Independent re-run of the two H.264 60 fps rows above on the same machine and
browser, after the Milestone 1 audit. Recorded separately rather than replacing
the originals: the run-to-run spread is itself information.

| Upscaler | Import | Presented fps | Rendered fps | Frames | Skipped | Decoder drops | GPU upscale (ms) | 60 Hz budget | Duration |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| Catmull-Rom 9-tap | external | 59.7 | 59.3 | 1778 | 13 | 13 / 1791 | 3.79 avg · p50 3.54 · p95 5.57 · max 8.13 | 22.7% | 30 s |
| Bilinear | external | 59.7 | 59.3 | 1778 | 13 | 13 / 1792 | 0.91 avg · p50 0.59 · p95 2.23 · max 5.90 | 5.4% | 30 s |

`cpu per frame` 0.14–0.15 ms avg; `decode latency` ~101 ms avg (queueing, not
cost). Input 1280x720 H.264/MP4 at 60 fps, output 2560x1440, exact 2x.

Averages agree with the originals to within 0.11 ms (Catmull-Rom 3.79 vs 3.90)
and 0.05 ms (bilinear 0.91 vs 0.86); p95 and max are noisier, as expected from
a 240-sample trailing window on a quantised timer. Frame delivery is
indistinguishable. **The original rows above remain the reference figures.**

## Observations

**The upscaler is not the bottleneck.** Bilinear costs 0.86 ms and Catmull-Rom
3.90 ms on the same 60 fps clip — a 4.5x difference in stage cost — yet both
upscale within seven frames of each other over 30 s (1786 vs 1779). A stage
consuming 5–23% of the frame interval leaves the frame rate unchanged, which is
the evidence needed before attributing any frame drop to the upscaler.

**Sampling an external texture repeatedly is expensive.** On the same H.264
clip and the same kernel, nine taps cost 3.90 ms through `texture_external` but
1.55 ms through an ordinary `texture_2d<f32>`. The likely mechanism is that
each `textureSampleBaseClampToEdge` on a multi-planar external texture samples
the planes and performs colour conversion, so a 9-tap kernel pays for that nine
times — but that mechanism is inferred from the specification, not measured
here. **This comparison is also not a total-cost ranking**: the copy path's own
upload is not in its number. It is evidence about per-tap sampling cost only,
and it is the reason `ARCHITECTURE.md` expects a multi-tap neural stage to want
a one-time ingest pass. Quantifying the copy itself is Milestone 2 work.

**Codec changes upscale cost, and the cause is not established.** The identical
Catmull-Rom kernel at identical resolution measured 4.64 ms on H.264 and
2.55 ms on VP9 at 30 fps. The plausible explanation is a different decoded
pixel format reaching the external texture, but this was not verified and is
recorded as an open question, not a conclusion.

## Superseded measurements — not results

The figures in this section are **not** AetherVSR benchmark results. They were
measured on the recorded hardware but against a test asset that has since been
replaced, so they are kept only as the record of a diagnosis. Nothing here may
be quoted as a result. See `AGENTS.md` §3.

**A generated clip's frame cadence is part of the experiment.** The first
60 fps test clips were captured with a `setTimeout`-paced loop. Because
`MediaRecorder` timestamps frames by wall clock, their presentation timestamps
scattered between 15 and 21 ms with frequent 31–33 ms gaps (measured from rVFC
`mediaTime` deltas). Chrome could not align them to vsync and presented only
46–50 fps *regardless of which upscaler ran* — which reads exactly like a
pipeline that cannot keep up, and would have been reported as a hard 60 fps
limitation. Re-pacing the generator on `requestAnimationFrame` put the
timestamps on the display grid; the same configurations then reached
58.7–59.5 fps with the committed clips, as reported above.

The lesson, and the reason this is recorded: when a frame rate ceiling does not
move as the cost of the stage under test changes by 5x, the ceiling is
somewhere else — and the test asset is part of the apparatus.

## Not measured

These were not obtained and must not be inferred:

- **Image quality.** No PSNR/SSIM/LPIPS/VMAF comparison was run. Bilinear and
  Catmull-Rom are compared on cost only. Quality metrics need a reference
  pipeline that does not exist yet.
- **Cost of `copyExternalImageToTexture`.** Not bracketable by our render-pass
  timestamps; see above.
- **Whether Chromium took its internal zero-copy branch.** `isZeroCopy` is only
  exposed under `WebGPUDeveloperFeatures`, which was not enabled. "external
  import path" here means we called `importExternalTexture()`, not that no copy
  occurred inside the browser.
- **Hardware vs software decode.** Not observable from the page; would need
  `chrome://media-internals`. No claim is made either way.
- **Other browsers.** Only Chrome for Testing 152 was run. Safari and Firefox
  were not tested at all; no claim is made about either.
- **Other GPUs, and power draw.** Single machine, no power instrumentation.
- **Sustained thermal behaviour.** Longest measurement window was 30 s.

## Milestone 2 — feasibility spike measurements

Same machine and browser as above (MacBook Pro, Apple M5, 24 GB; macOS 26.6.2
build 25G83; Chrome for Testing 152.0.7977.42; adapter `apple`/`metal-3`).
Date 2026-09-02. Run from `bench.html`, which is a separate Vite entry point so
none of this code ships in the Milestone 1 harness.

Device limits recorded for these runs: `maxStorageBufferBindingSize`
134,217,728 (128 MiB), `maxBufferSize` 268,435,456, `maxComputeInvocationsPerWorkgroup`
256, `maxComputeWorkgroupStorageSize` 16,384. `timestamp-query` and
`shader-f16` both available.

### External-texture ingest

720p60 H.264 clip, live rVFC loop, both passes timed with independent
`timestamp-query` sets in the same command encoder, n=240 trailing samples.

| Mode | Consumer | ingest (ms) | upscale (ms) | total GPU (ms) |
|---|---|---:|---:|---:|
| direct | bilinear — 1 external tap | — | 0.882 | 0.882 |
| direct | Catmull-Rom — 9 external taps | — | 3.765 | 3.765 |
| ingest `rgba8unorm` | bilinear — 1 2D tap | 0.357 | 0.510 | 0.867 |
| ingest `rgba8unorm` | Catmull-Rom — 9 2D taps | 0.391 | 1.312 | **1.703** |
| ingest `rgba16float` | bilinear — 1 2D tap | 0.447 | 0.643 | 1.090 |
| ingest `rgba16float` | Catmull-Rom — 9 2D taps | 0.359 | 1.128 | 1.487 |

Both totals now cover all GPU work AetherVSR encodes, so unlike the Milestone 1
comparison these rows *are* rankable against each other.

Differencing the two filters within a mode isolates the marginal cost of eight
extra taps:

| Sampling domain | 8 extra taps cost | per tap |
|---|---:|---:|
| `texture_external` | 2.883 ms | **≈0.360 ms** |
| `texture_2d<f32>` from `rgba8unorm` | 0.802 ms | **≈0.100 ms** |
| `texture_2d<f32>` from `rgba16float` | 0.485 ms | ≈0.061 ms |

**Caveat on that derivation:** bilinear and Catmull-Rom differ by more than tap
count — the cubic also evaluates weights and clamps — so the per-tap figures
are upper bounds that attribute all of the difference to sampling. The
direction and rough magnitude are robust; the exact numbers are not.

The ingest pass costs ~0.36–0.45 ms and saves ~0.26 ms per subsequent tap, so
it pays for itself at roughly two taps. A 3x3 convolution reads nine times per
output pixel per input channel, which is far past that break-even.

### 3x3 convolution throughput

1280x720, planar storage buffers, median of 40 timed dispatches after 8
warm-up dispatches. Every kernel was first verified against a CPU reference. The fp32 variant
passes 9 cases at ≤1.5e-7 absolute error. The fp16 variant is verified
separately against the same fp32 reference at a tolerance appropriate to half
precision: 4 cases, absolute error 0.0006–0.006, growing with accumulation
depth (144 MACs at 16→16) exactly as half precision predicts.

Channel sweep, `blockX=4`, workgroup 8x8, ReLU:

| Case | Precision | GMAC/dispatch | median (ms) | GMAC/s |
|---|---|---:|---:|---:|
| 3 → 16 | fp32 | 0.40 | 1.966 | 202.5 |
| 16 → 16 | fp32 | 2.12 | 11.207 | 189.5 |
| 28 → 28 | fp32 | 6.50 | 34.800 | 186.9 |
| 48 → 48 | fp32 | 19.11 | **rejected** | — |
| 16 → 16 | fp16 | 2.12 | 8.585 | 247.3 |
| 28 → 28 | fp16 | 6.50 | 26.411 | 246.2 |
| 48 → 48 | fp16 | 19.11 | 77.070 | 248.0 |
| 52 → 52 | fp16 | 22.43 | 90.964 | 246.6 |

**48→48 fp32 at 720p is not slow, it is impossible on this device as
configured.** One activation tensor is 176,947,200 bytes against a 128 MiB
`maxStorageBufferBindingSize`, so the bind group is rejected. Dawn's message
notes the *adapter* supports up to 4,294,967,292 bytes and the higher limit can
be requested in `requiredLimits` at device creation — this is a Chrome default,
not a hardware ceiling. fp16 halves the tensor and fits.

Tuning sweep, 16→16 at 720p:

| Configuration | Precision | median (ms) | GMAC/s |
|---|---|---:|---:|
| `blockX=1`, 8x8 | fp32 | 13.763 | 154.3 |
| `blockX=2`, 8x8 | fp32 | 12.321 | 172.3 |
| `blockX=4`, 8x8 | fp32 | 11.403 | 186.2 |
| `blockX=8`, 8x8 | fp32 | 12.911 | 164.5 |
| `blockX=4`, 16x16 | fp32 | 10.945 | 194.0 |
| `blockX=4`, 32x2 | fp32 | **10.224** | **207.7** |
| `blockX=8`, 8x8 | fp16 | 10.027 | 211.8 |
| `blockX=4`, 16x16 | fp16 | **8.585** | **247.3** |

Best measured: **207.7 GMAC/s fp32, 247.3 GMAC/s fp16** — fp16 is worth about
1.3x. Tuning within this kernel family moves the result by only ~1.3x from
worst to best, so the family itself is the limit, not the parameters.

**Why it is slow.** The reported `GB/s` column in the raw results counts each
activation element once and is therefore a floor. The traffic the kernel
actually issues is 14.7M outputs x 144 taps x 4 B ≈ 8.5 GB per dispatch, which
at 11.4 ms is ≈745 GB/s of load requests. The kernel is bound on redundant
global loads: it has no shared-memory tiling and no cooperative loading, and
relies entirely on cache for reuse.

### ONNX Runtime Web, native WebGPU execution provider

`onnxruntime-web` 1.29.0, `/webgpu` entry point (the native EP, not JSEP).
Model is a single `Conv` 16→16 3x3 pad 1 with bias plus `Relu` at 1280x720 —
deliberately the same arithmetic as the 16→16 row above. 20 iterations,
ORT-owned output tensors disposed each iteration.

| Output location | session create (ms) | first inference (ms) | steady median (ms) | mean | min | max | What the timing covers |
|---|---:|---:|---:|---:|---:|---:|---|
| `cpu` | 205.2 | 74.4 | 35.30 | 37.50 | 32.5 | 59.4 | End-to-end, GPU completion forced by the output download |
| `gpu-buffer` | 5.5 (warm) | 14.0 | 13.20 | 13.26 | 12.2 | 15.2 | **Submission latency only — not GPU completion** |

Three scoping facts, all of which must travel with these numbers:

1. **Neither row is comparable to the WGSL figures above.** Those are GPU pass
   time from `timestamp-query`. These are wall clock around `session.run()`.
2. **Both rows include a CPU→GPU upload of the 56.3 MB input tensor on every
   iteration.** `preferredOutputLocation` changes only the output side. A fair
   comparison needs `Tensor.fromGpuBuffer` on the input too, which this spike
   did not implement.
3. **The `gpu-buffer` row does not wait for the GPU.** ORT's native EP ends a
   run by flushing — submitting to the queue — and nothing forces
   synchronisation when no output is downloaded. So 13.2 ms is a lower bound on
   completion time, not a measurement of it. The `cpu` row is the only
   completion-synchronised figure here, and it is dominated by transfers.

**Fully GPU-resident ORT inference cost on this machine is therefore
unmeasured.** See DECISIONS.md ADR-0015.

What the run does establish: the native WebGPU EP loads and executes correctly
on this device; session creation is ~205 ms cold and ~5 ms warm; and the first
inference costs roughly 5x a steady one, so shader compilation and allocation
must be warmed before any frame-rate claim.

### Image quality

Generated 2560x1440 reference, exact integer 2x box downsample to 1280x720,
upscaled by the real GPU scalers, compared against the reference.

| Upscaler | PSNR-Y (dB) | PSNR-RGB (dB) | SSIM (luma) |
|---|---:|---:|---:|
| Nearest neighbour (control) | 16.924 | 17.084 | 0.87956 |
| Bilinear | 17.507 | 17.667 | 0.86200 |
| Catmull-Rom 9-tap | **19.448** | **19.607** | **0.92522** |

Catmull-Rom leads on every metric, by 1.94 dB PSNR-Y over bilinear.

Two honest caveats. First, absolute values are low because the reference is
deliberately adversarial — a zone plate sweeping to Nyquist, which no 2x scaler
can reconstruct. These numbers rank scalers on one fixed image; they are not
comparable with PSNR figures from the SR literature, which use natural-image
sets. Second, **nearest neighbour scores higher SSIM than bilinear** (0.880 vs
0.862) while scoring lower PSNR. That is expected rather than a bug: SSIM
rewards preserved local variance, and on near-Nyquist content blurring
suppresses variance more than blocking does. It is a useful warning that SSIM
alone would mis-rank scalers on this kind of content.

## Milestone 2 budget implication

The published arithmetic for a SPAN-Lite C16-class model (four SPAB blocks,
16 channels) is ≈30.5 GMAC per 720p frame. At the best measured convolution
throughput on this machine (247 GMAC/s, fp16) that is **≈123 ms per frame** —
against a 16.67 ms total budget of which the upscale stage should use a
fraction. Reaching ~8 ms would require roughly **15x** the measured throughput.

That gap, not the runtime choice and not the ingest cost, is the finding that
governs Milestone 3.

## Reproducing

```bash
npm install
npm run dev
```

Open the printed URL in a Chromium-based browser with WebGPU, **keep the window
frontmost**, let it warm up, press **reset stats**, wait, then press **copy
benchmark JSON**.

Query parameters make a run reproducible from a URL:

| Parameter | Values | Effect |
|---|---|---|
| `clip` | absolute path under `/media/...` | Selects the clip |
| `filter` | `catmull-rom`, `bilinear` | Selects the upscaler |
| `import` | `copy` | Forces the copy fallback path |

Example — the first row of the 60 fps table:

```
http://127.0.0.1:5173/?clip=/media/aethervsr-testclip-720p60-h264.mp4&filter=catmull-rom
```
