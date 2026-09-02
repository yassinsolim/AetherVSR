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
| Catmull-Rom 9-tap | external | 59.8 | 59.4 | 1780 | 12 | 12 / 1792 | 4.03 avg · p50 4.07 · p95 4.86 · max 5.82 | 24.2% | render pass; no other AetherVSR GPU work |
| Bilinear | external | 59.8 | 59.5 | 1784 | 8 | 8 / 1792 | 0.86 avg · p50 0.81 · p95 2.22 · max 3.99 | 5.1% | render pass; no other AetherVSR GPU work |
| Catmull-Rom 9-tap | copy fallback | 59.8 | 59.5 | 1783 | 9 | 9 / 1792 | 1.46 avg · p50 1.44 · p95 2.98 · max 4.53 | 8.7% | **render pass only, excludes import copy** |

`cpu per frame` was 0.11–0.13 ms average in all three runs.

The pipeline tracks a 60 fps source essentially frame for frame at the measured
59.2 Hz paint rate: 1780–1784 frames upscaled in 30 s, with 8–12 presented
frames skipped over the whole run (0.5–0.7%).

### 720p60 → 1440p, VP9

| Upscaler | Import | Presented fps | Rendered fps | Frames | Skipped | Decoder drops | GPU upscale (ms) | 60 Hz budget |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Catmull-Rom 9-tap | external | 59.9 | 59.3 | 1779 | 18 | 14 / 1797 | 2.40 avg · p50 2.28 · p95 3.36 · max 4.52 | 14.4% |

### 720p30 → 1440p

| Clip | Upscaler | Import | Presented fps | Rendered fps | Frames | Skipped | Decoder drops | GPU upscale (ms) | 60 Hz budget |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| VP9 | Catmull-Rom 9-tap | external | 30.0 | 30.0 | 898 | 0 | 0 / 899 | 2.41 avg · p50 2.60 · p95 3.36 · max 5.82 | 14.5% |
| VP9 | Bilinear | external | 30.0 | 30.0 | 898 | 0 | 0 / 898 | 0.50 avg · p50 0.36 · p95 1.68 · max 3.83 | 3.0% |
| H.264 | Catmull-Rom 9-tap | external | 30.1 | 30.1 | 902 | 0 | 0 / 903 | 4.70 avg · p50 4.64 · p95 6.26 · max 7.59 | 28.2% |

At 30 fps the pipeline is frame-perfect: zero skipped frames and zero decoder
drops over 30 s in all three runs.

## Observations

**The upscaler is not the bottleneck.** Bilinear costs 0.86 ms and Catmull-Rom
4.03 ms on the same 60 fps clip — a 4.7x difference in stage cost — yet both
upscale within four frames of each other over 30 s (1784 vs 1780). A stage
consuming 5–24% of the frame interval leaves the frame rate unchanged, which is
the evidence needed before attributing any frame drop to the upscaler.

**Sampling an external texture repeatedly is expensive.** On the same H.264
clip and the same kernel, nine taps cost 4.03 ms through `texture_external` but
1.46 ms through an ordinary `texture_2d<f32>`. The likely mechanism is that
each `textureSampleBaseClampToEdge` on a multi-planar external texture samples
the planes and performs colour conversion, so a 9-tap kernel pays for that nine
times — but that mechanism is inferred from the specification, not measured
here. **This comparison is also not a total-cost ranking**: the copy path's own
upload is not in its number. It is evidence about per-tap sampling cost only,
and it is the reason `ARCHITECTURE.md` expects a multi-tap neural stage to want
a one-time ingest pass. Quantifying the copy itself is Milestone 2 work.

**Codec changes upscale cost, and the cause is not established.** The identical
Catmull-Rom kernel at identical resolution measured 4.70 ms on H.264 and
2.41 ms on VP9 at 30 fps. The plausible explanation is a different decoded
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
