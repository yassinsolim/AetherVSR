# ROADMAP.md

AetherVSR's goal: high-quality, real-time, local video super-resolution for web
video, GPU-accelerated, Apple Silicon first-class, cross-platform through
WebGPU.

Milestones are sequential. A milestone is complete only when its acceptance
criteria are demonstrably met on real hardware, with measurements recorded in
`BENCHMARKS.md`.

---

## Milestone 0 — Repository foundation ✅ complete

Governance and architecture documented before code: `README.md`, `AGENTS.md`,
`ARCHITECTURE.md`, `ROADMAP.md`, `BENCHMARKS.md`, `DECISIONS.md`.

---

## Milestone 1 — WebGPU video baseline ✅ complete

The smallest browser harness that proves the whole non-neural path.

| # | Criterion | Status |
|---|---|---|
| 1 | Loads a local video into an `HTMLVideoElement` | ✅ bundled clips and a file picker |
| 2 | `requestVideoFrameCallback()` synchronises work to presented frames | ✅ rVFC clock, rAF only as a labelled degraded fallback |
| 3 | Initialises WebGPU | ✅ adapter/device with capability probing |
| 4 | Imports the frame via external video texture where supported | ✅ `importExternalTexture()`, with a forceable copy fallback |
| 5 | Renders to a canvas | ✅ WebGPU swap chain |
| 6 | Exact 2x output path (1280x720 → 2560x1440) | ✅ verified: `canvas.width/height` = 2560x1440 in every run |
| 7 | Conventional non-neural scaling, no inference | ✅ bilinear and 9-tap Catmull-Rom; `neural: false` throughout |
| 8 | Diagnostic overlay | ✅ 16 rows: source/output resolution, mean and instantaneous presented and rendered FPS, skipped frames, decoder drops, decode latency, measured GPU upscale time, CPU per-frame time, callback lag, 60 Hz budget, upscaler, import path, frame clock, adapter and device info |
| 9 | Clean seam for a future neural upscaler | ⚠️ designed and lifecycle-exercised, sufficiency unproven. `Upscaler` isolates the stage and runtime swapping exercises configure/destroy, but both variants are the same `BaselineScaler` class; no second implementation exists yet. See DECISIONS ADR-0011 |

Measured: 720p60 H.264 → 1440p at **59.7 presented / 59.3–59.5 rendered fps**
over 30 s, upscale stage consuming 5.1–23.4% of a 60 Hz frame interval
depending on filter. Full results and caveats in `BENCHMARKS.md`.

---

## Milestone 2 — Neural inference feasibility spike ✅ complete

**Question asked:** what neural super-resolution workload realistically fits
inside AetherVSR's frame budget on a base Apple M5?

**Answer: not the one workload we costed — a C16-class network at full 720p —
with the convolution kernel as written, by roughly an order of magnitude.**
That is a statement about one model size, one resolution and one naive kernel,
not about every lightweight architecture. Full numbers and scoping in
`BENCHMARKS.md`.

| Experiment | Result |
|---|---|
| External-texture ingest | Built and measured. An external tap costs ≈0.360 ms vs ≈0.100 ms for an ordinary texture tap; the ingest pass costs ≈0.36–0.45 ms and repays itself at ~2 taps. Nine-tap total 3.765 ms → 1.703 ms |
| Convolution throughput | Best measured 207.7 GMAC/s fp32, 247.3 GMAC/s fp16. fp16 worth ~1.3x. Kernel verified against a CPU reference in both precisions |
| Memory limits | 48-channel fp32 activations at 720p exceed Chrome's default 128 MiB `maxStorageBufferBindingSize`. The adapter allows 4 GiB via `requiredLimits`; fp16 also fits |
| ONNX Runtime Web | Native WebGPU EP runs. Cold session 205 ms, warm 5 ms, first inference ~5x steady. Fully GPU-resident cost **unmeasured** — see ADR-0015 |
| Image quality | Deterministic harness established. Catmull-Rom 19.45 dB PSNR-Y / 0.925 SSIM, bilinear 17.51 / 0.862, nearest control 16.92 / 0.880 |

**The budget arithmetic.** A SPAN-Lite C16-class model is ≈30.5 GMAC per 720p
frame — a third-party architecture figure derived from the published SPAN
design, not measured by us. At our measured 247 GMAC/s that is ≈123 ms, against
a 16.67 ms total frame budget of which the upscale stage should use a fraction.
Reaching ~8 ms needs ≈15x the throughput of the current kernel.

That gap — not the runtime choice, not the ingest cost, not the model choice —
is the finding that governs Milestone 3.

---

## Milestone 3 — Close the convolution throughput gap ✅ complete

**Goal:** establish, by measurement, the maximum 3x3 convolution throughput
*reachable across the configurations we sweep* on a base Apple M5 through
WebGPU, and decide from that whether a neural stage is viable at 720p, at
reduced internal resolution, or not at all. Deliberately not "the maximum the
hardware can do" — see the bottleneck note below.

**Outcome: the gap is largely closed, and 720p C16 is viable.** Convolution
throughput went from 240 to **1990 GMAC/s** in fp16 — **8.3x** — putting a
1280x720 16->16 layer at **1.067 ms**. Full tables in `BENCHMARKS.md`.

| Step | GPU ms | GMAC/s | vs original |
| --- | ---: | ---: | ---: |
| M2 baseline (naive) | 8.846 | 240 | 1.00x |
| + workgroup tiling with halo | 6.697 | 317 | 1.32x |
| + vec4 input-channel packing | 3.936 | 539 | 2.25x |
| + 8 output channels per invocation | 1.213 | 1751 | 7.29x |
| + 2D spatial blocking | 1.141 | 1861 | 7.75x |
| + tap-major weight layout | 1.086 | 1955 | 8.15x |
| + occupancy tuning | **1.067** | **1990** | **8.29x** |

Against acceptance criteria:

- ✅ A measured GMAC/s figure per optimisation, each verified against the CPU
  reference before timing.
- ✅ Maximum throughput measured across the configurations swept, and the
  configuration that produced it: 1990 GMAC/s, fp16,
  `blocked` variant, 8x4 workgroup, blockX 2, blockY 2, outBlock 16, tap-major
  weights, inside the guaranteed workgroup-storage floor.
- ✅ A documented reason the GPU-resident ORT figure could not be obtained: ORT
  1.29.0 rejects a caller-supplied `GPUDevice` at session creation across all
  four configurations tried (ADR-0018).
- ✅ An explicit ≤8 ms verdict — see below.
- ✅ Temporal baseline numbers for bilinear and Catmull-Rom, with an exactly-zero
  static control and an integer-pan control showing both filters are
  shift-invariant.
- ✅ No production model, no extension, no temporal VSR implementation.

**The ≤8 ms verdict.** At 1280x720 with 16 channels, **7 convolution layers**
fit in 8 ms (1.078 ms each). At 960x540, 12 fit; at 640x360, 28. This is an
upper bound on convolution alone: a real network also has activations, pixel
shuffle, format conversion, residual adds and per-layer dispatch overhead, so
layer-count arithmetic is not a model prediction. It does mean a C16-class
network at full 720p is now a question of how many layers, not whether.

**Bottleneck.** Not resolved into a single cause, and deliberately not claimed
as hardware saturation. The optimum sits in a trough between two measured walls:
below outBlock 8 the kernel is at or above the device's measured streaming
bandwidth, and above it register pressure dominates. The shipped configuration
sits at roughly **54% of measured FMA throughput and ~53% of measured streaming
bandwidth** — about half of each endpoint, which is why it is faster than the
outBlock 8 row that reaches 73% of bandwidth.

**Rejected:** `chromium-experimental-subgroup-matrix`, 8.1x slower than the
portable kernel and unavailable without `--enable-unsafe-webgpu` (ADR-0017).
**Not needed:** raised workgroup-storage limits — every tile above the
guaranteed 16 KiB floor was slower.

---

## Milestone 4 — First neural upscaler [DONE]

A real 2x neural model runs in the production video pipeline behind the existing
`Upscaler` interface: **5.48 ms p50 whole stage, 59.7 fps presented at 2560x1440,
33% of the 60 Hz budget**, quality **+2.20 dB / +0.037 SSIM over Catmull-Rom** on
the deterministic reference and +0.53 dB on natural images.

Architecture is ADR-0022: a low-resolution 3x3 convolution trunk at C16 with a
resize-convolution reconstruction head, weights trained in-house on a 500-image
CC0 corpus. Sub-pixel convolution was rejected during independent review over an
active European patent, EP3259916B1, in force to 2036.

Delivered: chainable packed activation pipeline; fused external-texture-to-
activation ingest; 5x5 texture-native stem; trusted PyTorch reference and
stage-by-stage golden vectors; model format and loader; budget fallback with
hysteresis, exercised on live video; whole-stage and per-pass instrumentation;
still-image, natural-image and temporal evaluation.

## Milestone 5 — Robustness and content coverage

Non-square pixel aspect ratios, resolution changes mid-stream, seeking, HDR and
10-bit sources, cross-origin and DRM behaviour (both expected to be
unsupported — the task is to fail clearly and prove where), tiling for large
inputs, and memory ceilings on constrained GPUs.

---

## Milestone 6 — Dynamic quality selection

Measure the per-frame budget continuously and choose the most expensive model
that fits, degrading to the baseline under pressure. Milestone 1's metrics are
the input; the control loop must be damped enough not to oscillate.

---

## Milestone 7 — Chrome/Chromium extension

Package the pipeline as an MV3 extension that attaches to video elements on
third-party pages: injection, lifecycle, per-site controls, and the security
and performance implications of running on pages we do not control.

---

## Milestone 8 — Cross-vendor validation

Verify on NVIDIA, AMD and Intel GPUs across Windows and Linux. Everything
measured so far is one Apple Silicon machine; nothing here should be assumed to
transfer.

---

## Beyond

Explicitly out of scope until the above are done, listed so they are not
mistaken for near-term plans: compression-artefact removal, temporal
super-resolution using previous frames, frame interpolation, and an
Apple-native Core ML / Metal / ANE backend.
