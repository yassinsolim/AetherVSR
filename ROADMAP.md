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

Measured: 720p60 H.264 → 1440p at **59.8 presented / 59.4–59.5 rendered fps**
over 30 s, upscale stage consuming 5.1–24.2% of a 60 Hz frame interval
depending on filter. Full results and caveats in `BENCHMARKS.md`.

---

## Milestone 2 — Neural inference spike (recommended next)

**Goal:** determine whether a compact SR model can run inside the measured
frame budget on this hardware, before committing to a model or a runtime.

Not "ship a neural upscaler" — answer the question that decides how.

1. **Ingest pass.** Convert the external texture to a regular RGBA texture once
   per frame in a timestamped pass. Milestone 1 measured nine
   `texture_external` taps at 4.03 ms versus nine `texture_2d` taps at 1.46 ms
   on the same clip and kernel; quantify the conversion so multi-tap consumers
   have a real number to plan against. This also closes the measurement gap in
   the copy-fallback path, whose upload cost is currently unmeasured.
2. **Cost model.** Implement one representative convolution block in WGSL and
   measure GMAC/s achieved on the M5. Published parameter counts are worthless
   without a measured throughput for this device.
3. **Runtime comparison, on evidence.** Hand-written WGSL versus ONNX Runtime
   Web's WebGPU EP. Per ORT's own documentation, `Tensor.fromGpuBuffer` and
   `preferredOutputLocation: 'gpu-buffer'` can avoid a per-frame CPU readback,
   but the buffers must belong to ORT's own `GPUDevice`, the WebGPU EP is
   documented as experimental, and an unsupported node falls back. The
   `ort-wasm-simd-threaded.jsep.wasm` artefact is ~27.8 MB per the npm/unpkg
   metadata for `onnxruntime-web` 1.29.0. **None of these are AetherVSR
   measurements** — they are third-party claims to be verified by measurement
   before the ADR is written.
4. **Model selection.** Candidates, with verified licences, in `DECISIONS.md`
   ADR-0010. As orders of magnitude only, derived by us from published
   architectures and **not measured on this hardware**: a C16 SPAN-Lite-class
   model is ~33k parameters and ~30 GMAC at 720p input; a full C48 SPAN is
   ~378 GMAC. The gap between those two numbers is the design space.
5. **Quality methodology.** Establish PSNR/SSIM/LPIPS against the baseline
   before any model lands, so "better" is measurable. Milestone 1 deliberately
   measured cost only.

**Acceptance:** a written, measured answer to "what fits in the budget", one
new ADR, and a GPU-time table for the ingest pass and a representative
convolution. **No user-facing neural upscaling required.**

---

## Milestone 3 — First neural upscaler

A real 2x model behind the existing `Upscaler` interface: weight loading and
packing, the full inference graph, quality comparison against the baseline, and
automatic fallback to the baseline when the frame budget is exceeded. Adding it
must not modify acquisition, import or presentation.

---

## Milestone 4 — Robustness and content coverage

Non-square pixel aspect ratios, resolution changes mid-stream, seeking, HDR and
10-bit sources, cross-origin and DRM behaviour (both expected to be
unsupported — the task is to fail clearly and prove where), tiling for large
inputs, and memory ceilings on constrained GPUs.

---

## Milestone 5 — Dynamic quality selection

Measure the per-frame budget continuously and choose the most expensive model
that fits, degrading to the baseline under pressure. Milestone 1's metrics are
the input; the control loop must be damped enough not to oscillate.

---

## Milestone 6 — Chrome/Chromium extension

Package the pipeline as an MV3 extension that attaches to video elements on
third-party pages: injection, lifecycle, per-site controls, and the security
and performance implications of running on pages we do not control.

---

## Milestone 7 — Cross-vendor validation

Verify on NVIDIA, AMD and Intel GPUs across Windows and Linux. Everything
measured so far is one Apple Silicon machine; nothing here should be assumed to
transfer.

---

## Beyond

Explicitly out of scope until the above are done, listed so they are not
mistaken for near-term plans: compression-artefact removal, temporal
super-resolution using previous frames, frame interpolation, and an
Apple-native Core ML / Metal / ANE backend.
