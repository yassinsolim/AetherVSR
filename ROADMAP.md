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

## Milestone 8 — Budget-matched architecture confirmation [DONE]

**Verdict: NO SELECTED ADVANTAGE; RETAIN PRODUCTION.** Result-commit local gates,
fresh-clone validation and CI passed; evidence is in `docs/M8-STATUS.md`.
Six paired runs completed exactly
81,180 updates, matched initial effective functions and streams, and 60 eligible
checkpoint draws. CPU captured validation gives fixed-final R3-R0 **-15.789624 dB**
across seeds 8101/8102/8103: -0.333521, -25.072357, -21.962996. Original best
checkpoints are secondary and also negative, mean -0.216718 dB. Exact paired
sign-flip p=0.25 is the three-pair minimum, not conventional significance.

The early mean lead at update 5,412 (+0.023539 dB) was not retained. Two R3 seeds
deteriorated severely; independent training review found no violated invariant,
and the mechanism remains unresolved. No tuning, dropped seed or rescue run was
used. An interrupted R3-8101 was archived and restarted with its original seed
because no resumable state survived the battery failure.

The initial captured MPS scoring pass was wholly withdrawn after identical
weights produced unequal scores. The original CPU reference was restored,
repeated forward/reverse parity committed, and every model/cell rescored. The
full CPU duplicate checks and independent frame/statistical audits passed.
All tables and horizons in `BENCHMARKS.md` use CPU scores, not withdrawn values.
Registration and original training source remain immutable.

M7's historical +0.0247 dB remains separate: its seeds, identity initialization
and fully annealed short schedule differ. M8's 16,200 snapshot is a partial long
schedule. Neither comparison establishes convergence or a universal result about
reparameterization; the literature does not explain the deterioration.

Fresh R3 seeds 8201-8205 and all candidate confirmation/temporal/runtime gates
were **not reached**, not passed. Confirmation is still an unfrozen 16-clip,
seven-category proposal: native-source/interval proof, alias/shared-shoot
clearance and qualifying text remain unresolved. No confirmation scoring took
place. Graph/resource structure is verified; actual device memory and new
runtime performance are not measured. Production weights and runtime are unchanged.
M9 remains separate work and does not depend on replacing the production model.

---

## Milestone 9 — Adaptive runtime quality controller [DONE]

Measured decision: **REPLACE BudgetGuard** with a pure two-tier runtime
controller and separate session observation. Only production C16D2 and
Catmull-Rom are real tiers; no model training or content-quality routing.
Cadence-aware median budgets, genuine neural probes, bounded backoff and
explicit manual/lifecycle/fatal states are independently tested.

The amended-CFR ten-minute M5 720p60 gate passed with no tier changes or probes,
59.61 rendered fps and 0.876% combined loss. Controlled overload fell back in
0.406 s and recovered through a real probe 16.356 s after removal. Twelve native
browser lifecycle journeys passed. The original 720p60 and short CFR Auto loss
failures remain visible; this is bounded feasibility, not a universal guarantee.
Verification evidence and scope limits are in [docs/M9-STATUS.md](docs/M9-STATUS.md).

---

## Milestone 10 - Chrome/Chromium extension [PARTIAL]

**Verdict: EXTENSION MVP PARTIAL. Not complete or READY.** The load-unpacked MV3
adapter and standalone harness reuse `RuntimeDriver`, `RuntimeController` and
`VideoPipeline`. Production remains the same 6,291-parameter C16D2 model; no new
model, training or inference policy was introduced.

| Acceptance surface | Closure evidence |
|---|---|
| Native production journeys | 40 PASS |
| Diagnostic journeys | 44 PASS; separate from native production acceptance |
| Whole extension reload | UNVERIFIED: native action timeout |
| Public Shaka clear-content player | PASS over 180.026923 s: seek, pause, resizes and container fullscreen; scroll down/back N/A because there was no scroll extent |
| Public Plyr / Video.js | NEGATIVE geometry; only one successful public player, so PARTIAL ceiling |
| Harness/extension pixel parity | 3/3 production external + 3/3 test sampled cases: exact input/output RGBA hashes within each route; paused instrumented replay, not quality or real-time timing |
| Long active duration / rates | PASS: 600001.5 active ms; 58.3515888645 rendered / 59.811586918 presented fps |
| Long absolute combined loss | FAIL: 876 skips + 847 decoder drops / 35,887 presented frames = 4.8011814863%, above 1% |
| Long runtime stability | All neural; zero tier changes, owner changes or errors; 299 rewarm state records are not fallbacks |
| Relative GPU overhead | PASS against frozen 0.6 ms p50 / 1.3 ms p95 allowances; does not waive the absolute loss gate |

Activation is per document via the native action's **Enable current page**;
mode persists per origin. Permissions are only `activeTab`, `scripting` and
`storage`, with no permanent hosts, web-accessible resources or MAIN-world
execution. Ownership covers the top document and accessible open shadow roots,
not iframes or closed shadow roots. Native controls and showing native tracks
are unsupported; PiP/direct-video fullscreen show the original. Page-owned
video, controls, DOM and styles remain authoritative and unmodified beneath
an extension-owned canvas.

The measured production payload is 263,008 bytes. Installation is documented in
[README.md](README.md). The separate M10 environment, measurement scopes and
failed gates are in [BENCHMARKS.md](BENCHMARKS.md), governed by
[docs/M10-PREREGISTRATION.md](docs/M10-PREREGISTRATION.md) and the frozen
[docs/M10-CALIBRATION-PLAN.md](docs/M10-CALIBRATION-PLAN.md). M9's historical
passing long window and failed short windows are retained, not substituted for
M10 acceptance.

---

## Milestone 10.5 - Extension stabilization and delivery-loss attribution [PARTIAL]

Counter semantics audited against Chromium and native probes; unique overlap is
not measurable, so the historical combined-loss gate remains binding. Fresh M9/M10
worktrees, four balanced repetitions per primary arm, instrumentation comparisons,
four initial long replications and bounded browser traces did not justify any
model/controller change. No thermal/GC/GPU-saturation attribution is supported.

Generic rounded clipping and size-query container support now have native corner,
placement, control and cleanup evidence. Chrome extensions-page Reload supports
observed same-document reactivation; runtime-API reload remains unverified. Final
production45/45, diagnostic49/49 and output parity6/6 pass.

Both final preregistered 600-second trials failed <=1% combined loss:
**3.477970% / 2.897662%**, despite passing average and final-120s >=58-fps gates.
Shaka passed; Plyr retains control-stack rejection; Video.js had earlier success
but two final source stalls after seeking. **EXTENSION MVP PARTIAL** remains the
verdict. See [docs/M10.5-REPORT.md](docs/M10.5-REPORT.md). Owner explicitly approved
a 50 MiB tracked-tree cap for source/tests/docs and compact evidence; raw traces
stay local and the 8 MiB per-file restriction remains.

---

## Milestone 10.6 - Browser delivery floor and public-media causality [PARTIAL]

The reviewed four-arm, sixteen600s study completed, retaining an interrupted B
attempt separately. Native controls all exceeded1% combined loss; mean4.262931%
with one-sided95% lower bound3.895832%. This is a lean-observer apparatus result,
not observer-free physical loss. Four noninferiority bounds failed, one Baseline
final120s rate was57.941667fps, and observer distortion remains unbounded.
**Case D: no normalization or readiness ADR.** Model/controller unchanged.

The target Video.js stall reproduced in3/3 sessions for each native, inactive,
Baseline and Auto arm. All six active journeys retained binding presentation
failures; recovery was censored. Formal public causality/scope remains unresolved.
Shaka passed1/2 checks; Plyr safely rejected. Final production45/45, diagnostic49/49,
UI reload and six exact output-parity cases pass, without waiving those failures.
[Full28-section report](docs/M10.6-REPORT.md). Public identity corrections were
owner-approved, prospectively recorded and retain every superseded attempt.

---

## Milestone 10.7 - Presentation synchronization [FAIL]

Owner-directed failed closure, not completed acceptance. The original228px stale
scroll canvas was reproduced as delayed invalidation. Geometry/source/output
epochs and synchronous hiding improve known transitions; production S1 still
misses unannounced position changes. S2 remains diagnostic-only.

Initial S2 cadence-penalty upper95 was0.927326fps, above the0.5fps allowance.
Stronger current-style/clip proofs passed two separate28-case pilots but their
nonqualifying60s cost variants measured11.54-12.22ms/s total geometry/guard work.
No limit, controller, model, supported scope or historical verdict was weakened.
Final strategy/public qualification is gated/not run. Independent lifecycle,
UI-reload recheck and exact parity remain separately evidenced, not a waiver.
Overall extension PARTIAL; [30-section report](docs/M10.7-REPORT.md). Its bounded
presentation-design follow-up is recorded below; this is not permission for M11.

---

## Milestone 10.8 - Presentation architecture [NO ARCHITECTURE QUALIFIED]

Prospectively registered local feasibility study, no production integration.
A and fixed D5/D10/D15 each stopped on the25th/96 case with stale CSSOM object-fit;
B stopped on concurrent anchor ownership restoration; C violated host layout by
reparenting. No E was registered. Cost studies NOT RUN because no safety survivor
qualified to enter them. S1 production and strongest S2 diagnostic are unchanged.
[30-section report](docs/M10.8-REPORT.md), ADR-0048. Its separately authorized
observable-support/ownership follow-up is recorded below; no M10.8 candidate was
promoted or integrated into production.

---

## Milestone 10.9 - Observable contract and ownership [NO USEFUL OBSERVABLE CONTRACT QUALIFIED]

Bounded design/feasibility complete. S1-R1/O0 passed19 preliminary common cases,
then required fullscreen recovery remained unresolved with output safely hidden.
The implementation's modal predicate is too restrictive; this is not proof that
fullscreen is unobservable. O1 left an owned token after history overflow; O2 changed
host layout through an attribute selector. No complete useful safety survivor,
so COSTS NOT RUN. Production S1/model/controller/permissions remain unchanged.
[30-section report](docs/M10.9-REPORT.md), ADR-0050. The product-pivot gate now
recommends a separately authorized extension-owned controlled-player/window study,
not another generic in-page synchronization implementation or M11. No next milestone
is implemented by this result.

---

## Milestone 10.10 - Controlled player study [INCOMPLETE RESEARCH CHECKPOINT]

M10.10 also remains an **INCOMPLETE RESEARCH CHECKPOINT / NO CONTROLLED WEB PRODUCT
PATH QUALIFIED**. P0 controlled tab/window parity and R1 progressive acquisition
are useful primitives, not completed product gates. Unbounded timing and partial
capture/control coverage stop candidate neural performance and public census.
No M10.11 implementation or additional native retry is authorized by this result.
See [the30-section report](docs/M10.10-PRODUCT-REPORT.md), ADR-0052/0053.

---

## Milestone 11 - Cross-vendor validation [GATED]

Not the default next step while M10.7 presentation synchronization fails and
M10.6 delivery/public-media robustness remains
unresolved and has not been proved vendor-specific. No M11 implementation is
included. After READY or a justified vendor-specific research gate, verify on NVIDIA, AMD and Intel
GPUs across Windows and Linux. Everything measured so far is one Apple Silicon
machine; nothing here should be assumed to transfer.

---

## Beyond

Explicitly out of scope until the above are done, listed so they are not
mistaken for near-term plans: compression-artefact removal, temporal
super-resolution using previous frames, frame interpolation, and an
Apple-native Core ML / Metal / ANE backend.
