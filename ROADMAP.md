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
`Upscaler` interface: **5.34 ms p50 whole stage, 59.7 fps presented at 2560x1440,
32.7% of the 60 Hz budget**, quality **+1.11 dB / +0.027 SSIM over Catmull-Rom**
on the deterministic reference and +0.35 dB on natural images.

It is *worse* than the baselines on sub-pixel temporal response - 3.25x
Catmull-Rom's motion-compensated variance - and BENCHMARKS.md records that,
along with the fact that only one training seed was run, so the variance behind
these figures is unquantified.

Architecture is ADR-0022: a low-resolution 3x3 convolution trunk at C16 with a
resize-convolution reconstruction head, weights trained in-house on a 495-image
CC0 corpus. Sub-pixel convolution was rejected during independent review over an
active European patent, EP3259916B1, in force to 2036.

Published. Distribution was held while the freedom-to-operate question raised in
ADR-0022 was resolved, and released by the project owner on accepted risk rather
than on legal advice — ADR-0027 records the reasoning and what would reopen it.

Delivered: chainable packed activation pipeline; fused external-texture-to-
activation ingest; 5x5 texture-native stem; trusted PyTorch reference and
stage-by-stage golden vectors; model format and loader; budget fallback with
hysteresis, exercised on live video; whole-stage and per-pass instrumentation;
still-image, natural-image and temporal evaluation.

## Milestone 4.5 — Generalization and real-world video validation [DONE]

Milestone 4 proved engineering feasibility. Whether the quality result
generalizes is a separate question, which is why this milestone sits between
them. The answer is **conditional**.

**Established.** Against the production Catmull-Rom, on genuinely unseen
photographs from a different institution, the model gains **+0.387 ± 0.034 dB**
across five seeds. Training on realistic web-video degradation roughly doubles
the advantage on compressed input (+0.149 → +0.343 dB at typical CRF, exact
permutation p = 0.0079, the floor at this sample size) while giving up most of
the clean-condition gain. Same architecture throughout — **the limitation was
the training data, not the network**.

**Not established.** On the ground-truth video benchmark the apparent win rests
entirely on one category that independent review identified as adversarial by
construction. Excluding it both conditions are neutral; excluding synthetic text
as well, both are negative. The corpus contains no captured footage, so it
cannot settle the question either way.

Ten methodology defects were found and fixed: two carried in from Milestone 4's
external review, two I found myself during 4.5, and six from this milestone's
independent review. They include a patch-level split that leaked 100% of
validation sources, a baseline that was the wrong filter, perceptual duplicates
straddling splits under two different hashes, and an MPS backend that returned
different answers for identical model-free computations. Every affected result was
regenerated; nothing measured before the fixes is carried forward.

Delivered: cluster-level split with CI-enforced disjointness; independent CC0
test corpus with three overlap checks; reproducible degradation pipeline with
measured CRF tiers; five-seed variance for two conditions; cross-degradation
matrix with exact permutation statistics; ground-truth video benchmark with
alignment proof and scope-sensitivity analysis; deterministic visual crops;
GitHub Actions CI.

Closed after three independent audits and a confirmation round: no P0 or P1
outstanding, gate green, CI green on the published commit. What remains open is
recorded as the next milestone's first item, not as a caveat here — this
benchmark contains no captured footage, so it cannot settle whether the model
helps on real video either way.

## Milestone 5 — Captured-footage validation [DONE]

Milestone 4.5 could not answer the web-video question: its benchmark was
procedurally generated and its only positive result came from an adversarial
zone plate. This milestone answers it with genuinely captured camera footage.

**Verdict: PARTIALLY.** On 10 independently sourced captured clips under
controlled 720p H.264, the shipped model beats production Catmull-Rom by
**+0.72 dB at high quality** (9/10 clips, p=0.004) and **+0.36 dB at typical
quality** (9/10, p=0.006), with both confidence intervals excluding zero. At
poor quality it adds nothing measurable (+0.05, p=0.22). Every category gains at
high and typical.

Bounded by what the corpus is: predominantly aerial, web-sourced 4K footage,
**no faces**, lowlight n=1, and a consistent temporal cost of 1.24x
Catmull-Rom's motion-compensated residual on 10/10 clips.

Three rounds of independent review found the Milestone 4.5 failure mode
recurring - a clip with a burned-in telemetry HUD and rendered map inset, where
77-99% of its advantage came from synthetic pixels - plus an inverted optical-flow
warp, a provenance pin that was silently rewritten on mismatch, duplicated
reference frames that disabled an alignment check, and a metric ordering VMAF
fails on captured footage exactly as it did on synthetic. All were fixed and
every affected result regenerated.

Delivered: captured corpus with API-verified licences and enforced hash pinning;
aligned preparation pipeline with a four-check alignment proof including a
shuffle control; metric sanity gate run before any model number; clip-level
paired statistics with bootstrap CIs and exact permutation tests; temporal
analysis with a sharpness confound control; deterministic visual comparisons;
production runtime validation; repository size guard.

## Milestone 5.5 — Heavy compression and GOP-aware degradation [DONE]

Milestone 5 left the model with no measurable advantage at heavily compressed
input: +0.055 dB at CRF 34, 7/10 clips, not significant. The rule for this
milestone was to test whether the *training* compression process was teaching
the wrong problem before making the network larger.

The hypothesis was that single-frame all-I training degradation mismatched real
GOP-structured video. **Refuted** — the model does better on GOP input at every
CRF, so the all-intra approximation was the harder condition. A 2x2 with the
architecture fixed at 6,291 parameters then showed GOP-aware degradation is a
large win at high and typical quality (+0.62 / +0.27 dB) and worth nothing at
CRF 34, while what actually moved CRF 34 was the training corpus: video frames
instead of still photographs.

The shipped model changed weights and not the graph. On the frozen ten-clip
captured test, read once after selection closed: **+0.91 dB at CRF 18, +0.60 at
CRF 26, +0.23 at CRF 34, 10/10 clips at every level.** Runtime unchanged at
6.3 ms p50, 39% of a 60 Hz budget.

Corrections made after independent review, recorded because they matter more
than the headline: the temporal "improvement" was the metric responding to a
7.5% softer image and is retracted; the corpus attribution was five-way
confounded and is now isolated at half its original size; the quality-signal
null result was Simpson's paradox and inverts on a within-clip analysis; the
matched-quality comparison had clamped two extrapolated points; and the visual
comparison sheets had been rendered against the wrong model's scores.

## Milestone 6 — Training-data scale and diversity [DONE]

Milestone 5.5 concluded data was the bottleneck. This milestone tested how far
that goes, holding the 6,291-parameter architecture fixed throughout.

The corpus grew from 12 clips / 8 creators / no category labels to **151 clips
from 108 creators across 146 shoots and 8 content categories**, with validation
and confirmation corpora that are creator-disjoint from training by
construction. Nothing was downloaded: clips are streamed and pinned by Commons
hash plus a prefix digest.

The shipped model improves on independently sourced footage by **+1.39 / +0.66 /
+0.12 dB** at CRF 18 / 26 / 34, 17/17 clips at the first two tiers, at unchanged
runtime.

**But data scale is not what did it.** At equal optimizer updates, twelve times
the corpus is worth +0.103 dB, the curve is non-monotonic, and nothing clears
significance at three seeds. At fixed epochs the same contrast reads +0.358 dB —
about 71% of it is extra gradient updates. A milestone reporting only the second
number would have drawn the opposite conclusion.

Corrections made after independent review, recorded because they matter more
than the headline: the learning-rate schedule was stepping per epoch and gave
larger corpora a 7.8% cumulative advantage, so every experiment was rerun; the
first final seed set predated its own selection rule by 42 seconds and was
discarded and retrained; a confirmation clip labelled "texture" was a rendered
title card; per-class results were pooled across compression tiers and hid two
negative cells; and the creator-diversity ablation was withdrawn as unmatched.

**Verdict: diminishing returns.** Another 150 clips is not indicated.

## Milestone 7 — Architecture / structural reparameterization [DONE]

Tested whether linear training-only branches before the existing `tanh` learn
better fused weights for the 6,291-parameter C16D2 inference graph. Four rungs
× three seeds on one frozen corpus at an equal 16,200-update budget, then five
fresh seeds for the winner.

**Verdict: `R3` wins the ladder, production is retained.** `R3` (3×3 + 1×1 +
identity + 1×3 + 3×1) scored −0.1373 dB against the frozen production model
versus the `R0` control's −0.1620 — **+0.0247 dB**, outside the 0.01 dB tie
band. The registered seed-level test returned p = 0.100, which is its floor at
three seeds, so the effect is **selected under a pre-committed rule, not
established**. Post-hoc and descriptive only: `R3` leads at every compression
tier and beats `R0` on 14 of 16 clips, but is best of all four rungs in only
15 of 24 category×CRF cells, and all of those re-pool the same clips and seeds.

No candidate shipped. The best final seed reached −0.1259 dB against
production, missing the pre-registered +0.10 dB replacement threshold by
0.23 dB. Every arm trained 16,200 updates against production's 81,180 and no
matched-budget arm was run, so that asymmetry is a confound large enough to
explain the gap rather than a measured cause of it.

The zero-cost property held: 9,971 training parameters deploy as 6,291, with
identical tensor names, layers and normalisation, and a runtime p50 difference
smaller than the production model's own run-to-run spread.

Corrections recorded rather than buried: the first nine-model screen was
withdrawn for an RNG confound that desynchronised initialisation between arms;
a creator overlap between training and validation was repaired into a versioned
split; and one source's absolute container timestamps were corrected against a
verified relative duration.

## Milestone 8 — Budget-matched architecture confirmation [NEXT]

Answer the question Milestone 7 could not: does `R3`'s advantage survive at the
production budget? Train `R0` and `R3` at 81,180 updates on the same frozen
corpus, three seeds each, declared before running and uniformly applied. That
is the only comparison that can produce a deployable candidate, and it also
tests whether the +0.0247 dB at a fifth of the schedule was a head start that
the control catches up on.

Prerequisites already in place: frozen corpus and hashes, a repaired
creator-disjoint validation split, a registered scoring and reporting
invocation, and 106 corpus/report/fusion tests in CI.

Still outstanding: `data/captured-confirm-m7/proposal.json` is a 16-clip
**proposal**, not frozen. Its schema, creator/shoot/source disjointness against
all seven existing manifests, licences, resolutions and pin coverage were
verified here; its poster-frame observations and reviewer-trust claims were
not. It also covers only seven of eight categories — there are no `text`
clips — so it must be completed, independently verified and frozen before any
candidate scores against it.

---

## Milestone 8 — Dynamic quality selection

Measure the per-frame budget continuously and choose the most expensive model
that fits, degrading to the baseline under pressure. Milestone 1's metrics are
the input; the control loop must be damped enough not to oscillate.

---

## Milestone 9 — Chrome/Chromium extension

Package the pipeline as an MV3 extension that attaches to video elements on
third-party pages: injection, lifecycle, per-site controls, and the security
and performance implications of running on pages we do not control.

---

## Milestone 10 — Cross-vendor validation

Verify on NVIDIA, AMD and Intel GPUs across Windows and Linux. Everything
measured so far is one Apple Silicon machine; nothing here should be assumed to
transfer.

---

## Beyond

Explicitly out of scope until the above are done, listed so they are not
mistaken for near-term plans: compression-artefact removal, temporal
super-resolution using previous frames, frame interpolation, and an
Apple-native Core ML / Metal / ANE backend.
