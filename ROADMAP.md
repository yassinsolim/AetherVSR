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

**Answer: none of the published lightweight architectures, by roughly an order
of magnitude, with the convolution kernel as written.** Full numbers and
scoping in `BENCHMARKS.md`.

| Experiment | Result |
|---|---|
| External-texture ingest | Built and measured. An external tap costs ≈0.360 ms vs ≈0.100 ms for an ordinary texture tap; the ingest pass costs ≈0.36–0.45 ms and repays itself at ~2 taps. Nine-tap total 3.765 ms → 1.703 ms |
| Convolution throughput | Best measured 207.7 GMAC/s fp32, 247.3 GMAC/s fp16. fp16 worth ~1.3x. Kernel verified against a CPU reference in both precisions |
| Memory limits | 48-channel fp32 activations at 720p exceed Chrome's default 128 MiB `maxStorageBufferBindingSize`. The adapter allows 4 GiB via `requiredLimits`; fp16 also fits |
| ONNX Runtime Web | Native WebGPU EP runs. Cold session 205 ms, warm 5 ms, first inference ~5x steady. Fully GPU-resident cost **unmeasured** — see ADR-0015 |
| Image quality | Deterministic harness established. Catmull-Rom 19.45 dB PSNR-Y / 0.925 SSIM, bilinear 17.51 / 0.862, nearest control 16.92 / 0.880 |

**The budget arithmetic.** A SPAN-Lite C16-class model is ≈30.5 GMAC per 720p
frame. At 247 GMAC/s that is ≈123 ms, against a 16.67 ms total frame budget of
which the upscale stage should use a fraction. Reaching ~8 ms needs ≈15x the
measured throughput.

That gap — not the runtime choice, not the ingest cost, not the model choice —
is the finding that governs Milestone 3.

---

## Milestone 3 — Close the convolution throughput gap

**Goal:** establish, by measurement, the maximum 3x3 convolution throughput
achievable on a base Apple M5 through WebGPU, and decide from that whether a
neural stage is viable at 720p, at reduced internal resolution, or not at all.

This is deliberately still not "ship a model". Milestone 2 showed the
bottleneck is arithmetic throughput; shipping a model before that is resolved
would just produce a slow model.

1. **Optimise the kernel.** The current one is naive: no shared-memory tiling,
   no cooperative loading, scalar loads. Issued load traffic is ≈8.5 GB per
   16→16 dispatch (14.7M outputs x 144 taps x 4 B), ≈745 GB/s at the measured
   time, so it is bound on redundant global reads. Implement and measure, in
   order: workgroup-shared input tiles; `vec4` channel packing; accumulating
   several output channels per invocation; and `chromium-experimental-subgroup-matrix`,
   which this adapter advertises and which is the one lever that could plausibly
   deliver a large multiple.
2. **Re-measure the ceiling.** Report GMAC/s for each optimisation
   independently, so the contribution of each is attributable.
3. **Take the deferred ORT measurement.** Same model, `Tensor.fromGpuBuffer`
   input *and* `gpu-buffer` output on AetherVSR's own `GPUDevice`, with an
   explicit completion fence. This is the experiment ADR-0015 defers, and it
   may show ORT's mature kernels beat ours.
4. **Establish the resolution/channel trade.** If 720p cannot be reached,
   measure what can: internal processing at 960x540 upscaled 2x then resampled,
   or a narrower network.
5. **Add temporal measurement.** Frame-to-frame difference on static shots and
   motion-compensated difference on moving shots, on the *baseline* scalers
   first, to establish what "no flicker" looks like before any model exists.

**Acceptance criteria:**

- A measured GMAC/s figure for each optimisation step, each verified against
  the CPU reference before timing.
- A stated maximum achievable throughput with the configuration that produced
  it.
- A completion-synchronised, GPU-resident ORT figure, or a documented reason it
  could not be obtained.
- An explicit verdict: at what (channels, resolution, layer count) a neural
  stage fits in ≤8 ms, or a statement that none does on this hardware.
- Temporal baseline numbers for bilinear and Catmull-Rom.
- No production model, no extension, no temporal VSR implementation.

---

## Milestone 4 — First neural upscaler

A real 2x model behind the existing `Upscaler` interface: weight loading and
packing, the full inference graph, quality comparison against the baseline, and
automatic fallback to the baseline when the frame budget is exceeded. Adding it
must not modify acquisition, import or presentation.

---

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
