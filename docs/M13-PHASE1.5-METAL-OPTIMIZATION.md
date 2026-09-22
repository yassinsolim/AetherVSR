# M13 Phase 1.5 - Native Metal Optimization and Real-Time Qualification

## 1. Starting state

Started from clean `main`, with HEAD and origin/main both exactly
`5b470064aea3cfee4ca95e84696ef8a359478fc9`. Exact publication CI
[35767539893](https://github.com/yassinsolim/AetherVSR/actions/runs/35767539893)
had gate, fusion and native-metal success. The fresh baseline passed 1499
Vitest tests with three existing skips, 612 Python tests, 80 opt-in tests,
71 desktop tests, and 13 physical M5 native tests. Install, typecheck, lint,
research/desktop checks, all builds and Phase-1 evidence reproduction passed.

This phase optimizes GPU kernels and backend execution only. It introduces
no AVFoundation, Core Video, CVPixelBuffer, CVMetalTextureCache, IOSurface,
ScreenCaptureKit, GUI, playback, browser product integration, permissions,
Core ML or MPSGraph. The deployed WebGPU core and product behavior are unchanged.

## 2. Immutable Phase-1 result

Phase 1 remains **METAL INFERENCE QUALIFIED**, permanently. Its simple
`MetalEngine`, bundled `Shaders.metal`, source-model loader, validator, report,
golden vectors and raw measurements are unchanged. The original executable
and raw tensor/timing artifacts remain retained. The new path is a separate
`OptimizedEngine`; it does not replace or delete the reference implementation.

Historical Phase-1 1280x720 to 2560x1440 whole-graph GPU measurements:

| Precision | p50 ms | p95 ms | max ms | Warmups / measured samples |
|---|---:|---:|---:|---:|
| f32 | 83.980375 | 84.548329 | 84.898750 | 10 / 60 |
| f16 | 65.852229 | 66.653373 | 67.086625 | 10 / 60 |

These are the original unfused results, not measurements of the optimized
implementation. Their original windows and scope remain in the unchanged
[Phase-1 report](M13-PHASE1-METAL.md). M11/M12/M12.1 reports and outcomes remain
unchanged as well.

## 3. Performance target

The owner-set definitions were recorded before any optimized timing:

- QUALIFIED requires numerical/evidence PASS and both binding runs to contain
  60 valid samples, p50 <= 8.0 ms and p95 <= 10.0 ms, with no invalid or unbounded
  execution and no model/tolerance change.
- PARTIAL requires numerical/evidence PASS and both valid binding runs to have
  p95 < 16.67 ms, while missing the 8/10 ms headroom target.
- NOT QUALIFIED covers selected-path correctness failure, any binding p95
  >= 16.67 ms, or incomplete/untrustworthy evidence.

The headroom target reserves part of the nominal 16.67 ms interval for future
ingest, scheduling and presentation. No threshold was changed after results.
f16 is the intended performance path. Shared f32 kernels are correctness
controls; no optimized f32 performance result was measured.

## 4. Initial bottleneck

Retained Phase-1 f16 isolated p50 identified convolution as the priority:
stem 10.522021 ms, body.0 17.980208 ms, body.1 17.986646 ms and head
13.583271 ms. Smaller activation/nearest/composition dispatches were not treated
as the first optimization target. No extra profiler pass was needed.

The simple kernel assigned one output element to each thread, repeatedly
loading overlapping inputs for different output channels. It also materialized
the high-resolution 16-channel feature map. The hypothesis was to reuse those
values through packing, blocking and tiling, then remove unnecessary intermediates.

## 5. Optimization plan

The [prospective plan](M13-PHASE1.5-OPTIMIZATION-PLAN.md) and ADR-0061 were
written before optimized measurements. Production WebGPU stem/body/head builders
were inspected first. Portable algorithms were adopted: vector channel groups,
tap-major weights, cooperative halo loading, output-channel and spatial reuse,
post-convolution activation stores, and direct nearest-coordinate head evaluation.
MSL was generated for Metal's buffer/pipeline/dispatch model, not mechanically
transliterated from WGSL. No external implementation or dependency was imported.

Observed M5 limits: threadExecutionWidth 32, maximum pipeline threadgroup 1024
threads, 32768 bytes threadgroup memory and maxBufferLength 14302248960 bytes.
Each actual pipeline was checked against its own thread and shared-memory limits.
Compilation uses MSL 3.0, fast math disabled and `FP_CONTRACT OFF`.

All six candidate sources, generated configurations and packed-weight hashes
were frozen before exploration. There was no configuration added after seeing
timings, no Metal-specific second study and no post-selection kernel change.

## 6. Candidate matrix

All measured candidates used the exact production model, f16 performance and
the same input/output sizes. All six share implementation source
`523217d336e92f60efae7bcd89bc4489c5012b6d`. Geometry below is workgroup threads;
spatial block is output pixels per invocation. `OB` is output channels per
invocation. Head OB is three, stem OB sixteen. All new paths use C4 activations
and tap-major weights. Only body kernels use shared-memory tiles.

| ID | Family | Body group / spatial / OB | Stem group / spatial | Head group / spatial | Fusion |
|---|---|---|---|---|---|
| A | Frozen scalar reference | 64x1 linear / 1x1 / 1 | Phase 1 | Phase 1 | Unfused; historical timing only |
| B | Packed/vectorized | 8x4 / 1x1 / 4 | 8x4 / 1x1 | 8x4 / 1x1 | Separate tanh; materialized nearest/head; final composition |
| C | Packed + tiled body | 8x4 / 1x1 / 4 | 8x4 / 1x1 | 8x4 / 1x1 | As B |
| D | Tiled + output blocked | 8x4 / 1x1 / 16 | 8x4 / 1x1 | 8x4 / 1x1 | Stem/body tanh fused |
| E | Tiled + spatial blocked | 8x4 / 2x2 / 16 | 8x8 / 2x2 | 8x4 / 2x2 | As D |
| F | Direct reconstruction | 8x4 / 2x2 / 16 | 8x8 / 2x2 | 8x4 / 2x4 | Tanh + direct LR head/residual/clamp/RGBA |
| G | Larger body workgroup | 8x8 / 2x2 / 16 | 8x8 / 2x2 | 8x4 / 2x4 | As F |

B/C to D changes both output blocking and activation fusion, so it is not an
isolated causal estimate for either feature. F also changes head spatial block.
The complete candidate table includes those confounds rather than attributing
every gain to one operation.

## 7. Packed layout

Activations use `[channel/4][row][column][channel%4]`: four groups for C16,
dense vectors, eight-byte half4 or sixteen-byte float4 alignment. Plane stride
is width*height vectors, row stride width vectors. No edge padding is stored;
out-of-image accesses explicitly return zero. Input RGB is packed into float4
with a zero fourth lane; the f16 stem explicitly narrows its RGB reads.

Packing and inverse-layout tests use distinct channels and asymmetric extents.
Tiny kernels use 19x11 inputs that span full and partial workgroups; both
channel and pixel permutations are demonstrated to change the CPU oracle.
The residual retains original float32 RGB, including values not representable
in half. There is no image-based assumption that could hide a lane permutation.

## 8. Weight layout

Production JSON byte SHA256 remains
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.
Embedded identity remains
`9154d9490b02d8fdf136edd990217cfb6f0e9956ed928da7179d7781832c8e02`.
There are the same 6291 float32 source parameters and the same exact graph.

Configuration deterministically converts OIHW into
`[inputGroup][kernelTap][outputChannel][lane]`, with a zero fourth RGB weight
lane in the stem. Head output channels remain three. Every converted buffer is
hashed, uploaded once, then checked against its expected bytes before dispatch.
Wrong source identity, missing buffers, bit-corrupted representations, invalid
shapes, non-finite weights and half overflow fail. Derived representation hashes
are not new model identities. No hand-maintained or copied Metal model exists.

## 9. Tiled convolution

C-G body kernels stage one C4 input group at a time, including the one-pixel
halo for the 3x3 convolution. The tile is
`(groupX*blockX+2) * (groupY*blockY+2)` vectors. Threads cooperatively load
the tile, zeroing out-of-bounds positions. Uniform barriers occur before tile
overwrite and after load; edge threads never return before those barriers.

For selected F the tile is 18x10 half4, 1440 bytes. For G it is 18x18 half4,
2592 bytes. Bounds checks occur on output stores. Stem and head use direct
global reads with register reuse, not threadgroup tiling. Tiny asymmetric CPU
oracles cover halo bounds and partial workgroup stores in both precisions.

## 10. Output-channel blocking

B/C accumulate four body output channels per invocation. D-G accumulate all
sixteen, allowing each staged input vector to feed multiple output dot products.
Weights for one tap/output block are contiguous in the packed representation.
The stem always accumulates sixteen channels; the head three RGB channels.

The generator emits statically named accumulators rather than dynamically
indexed accumulator arrays. Output-channel blocks own complete vectors and
write whole C4 values, avoiding component-write races. The CPU oracle uses
output/channel/tap-distinct weights and checks the exact packing mapping.

## 11. Spatial blocking

E-G body and stem invocations compute 2x2 neighboring output pixels. F/G head
invocations compute 2x4 high-resolution pixels. These reuse weights across
neighboring pixels while preserving separate accumulators and store guards.
The selected body workgroup covers a 16x8 output region, plus its halo.

The test input varies independently across row, column and channel; reversing
pixels or channels cannot match its expected output. High-resolution head
tests include all four odd/even coordinate parities and partial tail groups.
No untested geometry was selected after timing.

## 12. Fused activation

D-G apply tanh to convolution accumulators before writing the packed output.
B/C retain separate activation dispatches. The stage outputs observable to
the validator are still exactly stem, body.0 and body.1 after tanh.

f16 retains half4 dot groups, half accumulation and half tanh; f32 retains
float equivalents. Reduction traversal is input group, kernel row, kernel
column. Rounding-sensitive tests distinguish grouped dots from scalar-half
accumulation and half accumulation from retained float32 accumulation through
body, stem and head paths. No fast approximate activation or tolerance change.

## 13. Fused reconstruction/head

F/G read body.1 at low resolution and evaluate the head at high resolution
without materializing nearest-upsampled C16 features. Each tap first checks
its high-resolution coordinates against the 2x image bounds, then divides
valid coordinates by two. Zero padding is therefore the same as convolution
over a materialized nearest-upsampled tensor, not low-resolution padded convolution.

Independent CPU-oracle tests compare materialized and direct reconstruction
at odd/even coordinates and boundaries. The head remains a linear 3x3
16-to-3 convolution; only its input addressing and execution are fused.

## 14. Fused residual/output

F/G convert half head accumulators to float, add nearest original float RGB,
clamp [0,1], and write opaque RGBA8 in the same kernel. B-E materialize a float
head result and run a separate composition kernel. No color transfer or sRGB
conversion is introduced.

Diagnostic mode additionally writes final float RGB before quantization and
copies all three activated stage outputs. Fast mode does neither; its small
dummy final buffer is guarded by the capture flag. Every candidate and precision
matches diagnostic and fast RGBA exactly on the golden input. Full-size post-run
audits also match those modes byte-for-byte. Non-half-exact RGB controls reject
accidental residual narrowing or half addition.

## 15. Memory changes

Observed configured buffer bytes, excluding the private output texture and
driver/pipeline overhead:

| Implementation | Precision | Configured buffer bytes |
|---|---|---:|
| Phase-1 reference | f32 | 792,601,268 |
| Phase-1 reference | f16 | 481,087,886 |
| B, C, D, E fast paths | f16 | 265,434,210 |
| F, G fast paths | f16 | 88,487,010 |

Selected F's private 2560x1440 RGBA8 output accounts for another 14,745,600
logical bytes, giving 103,232,610 explicitly accounted buffer/texture bytes.
This is not measured peak RSS or driver residency. The configured buffers
include an offline RGBA readback buffer that is unused during timing.

Two reusable low-resolution packed activation buffers replace the individual
stage intermediates. F/G eliminate the high-resolution C16 nearest tensor and
separate float head buffer. Larger diagnostic stage/final buffers are allocated
only in separately configured diagnostic engines outside measured windows.

## 16. Command scheduling

One reusable MTLCommandQueue owns the workload. Selected F encodes four compute
stages into one command buffer per inference: stem+tanh, body.0+tanh,
body.1+tanh, and fused head/residual/clamp/RGBA. There are no interstage CPU
waits or readbacks. Metal tracked hazards and encoder order preserve dependencies.

Weights, constants, input/activation buffers, textures and pipelines are
configured once. Recurring allocations are the command buffer and compute
encoders required by Metal, plus small CPU evidence objects. No model, large
activation, argument buffer or pipeline allocation occurs per inference.

Offline measurement waits once after each complete command buffer to inspect
status and timestamps. Progressive evidence is flushed afterward, outside the
GPU interval; that overhead remains inside the reported wall observation window.
Load/measure/capture/destroy are lock-serialized. A GPU execution error latches
terminal failure; destroy is idempotent. This is not a live asynchronous player.

## 17. Golden results

All six candidates passed trusted f32/f16 stage and final-output golden checks
before exploratory timing. Tiny layout, convolution, rounding and head oracles
also passed before the full model gate. The trusted fixture byte hash remains
`7ffe8d5c26ef02f605c04e9057dd5ef9767dc83e84695211eff39f04a1e3a759`.

Selected F versus trusted golden, maximum / mean absolute errors:

| Stage | f32 max / mean | f16 max / mean |
|---|---:|---:|
| stem | 6.910414e-7 / 7.577895e-8 | 0.005311131 / 0.000496277 |
| body.0 | 1.579523e-6 / 1.117129e-7 | 0.012225986 / 0.000603176 |
| body.1 | 1.549721e-6 / 1.356082e-7 | 0.012660682 / 0.000675364 |
| final float | 4.470348e-7 / 5.685939e-8 | 0.001717180 / 0.000274581 |

Tolerance remains f32 0.001, f16 0.05, relative zero. Every stage has zero
failing and non-finite elements. Normalized RGBA uses the unchanged
`max(precision tolerance, 1.5/255)` rule with opaque alpha. No stage omission,
shape relaxation, model change or different precision declaration was accepted.

## 18. WebGPU parity

After selecting F, a fresh actual common-input WebGPU diagnostic capture ran
at selection commit `a6c4ea73d2eee5921dd6ad853a96cdd2147205af`. Electron 44.4.1 /
Chromium 152.0.7977.78 used the non-fallback Apple adapter. Production WGSL and
packing code are unchanged. The existing labeled float32 diagnostic head
storage adaptation and original RGBA8 path both remain checked.

Selected F versus that fresh WebGPU capture:

| Stage | f32 max / mean | f16 max / mean |
|---|---:|---:|
| stem | 1.788139e-7 / 2.396503e-8 | 0.007568359 / 0.000554267 |
| body.0 | 1.043081e-6 / 6.086863e-8 | 0.010986328 / 0.000704501 |
| body.1 | 1.192093e-6 / 8.000536e-8 | 0.008911133 / 0.000824438 |
| final float | 2.384186e-7 / 3.685374e-8 | 0.001831055 / 0.000357714 |

Every stage has zero failing/non-finite elements and zero edge/interior
failures at the unchanged limits. f32 RGBA8 is fully byte-identical; f16 is
not, with normalized maximum error 0.003921598 and mean 0.000356584, within
0.05. No float-stage bit identity is asserted. Both precisions and fast/diagnostic
agreement passed before binding-01; independent golden passes alone were not
treated as sufficient backend parity.

## 19. Falsification

PASS. The original source-hash/model, wrong-shape, checkpoint-order, layout,
NaN/Inf, perturbed-reference and precision-metadata controls remain unchanged.
Every optimized candidate's actual GPU checkpoint is also deliberately corrupted
with NaN, positive/negative infinity and a large finite error; qualification fails.
New derived-weight bit flips, missing buffers and bad source identity are rejected.

Packing tests inspect exact C4/tap-major positions and zero RGB padding.
Asymmetric CPU oracles reject channel and spatial permutations. Half rounding
and float residual controls execute the real new kernels, including fused paths.
Inputs/capture-after-destroy guards and terminal execution-state tests remain.

Evidence controls reject incomplete candidate inventories, wrong candidate
configuration/memory counts, modified tensors, invalid or fabricated warmup
timestamps, altered percentiles, incomplete sample sets, non-nominal starts,
replaced consumed attempts, unsafe failure flags, and changed source/binary pins.
Owned process deadlines and ordinary thermal deferrals are tested. These controls
do not claim an intentionally induced physical GPU hang or thermal event.

## 20. Candidate timings

All new rows measured on Apple M5, Mac17,2, 24 GiB, macOS 26.6.2 build 25G83,
Swift 6.4 (`swiftlang-6.4.0.34.1`), Xcode 27.0 build 27A266a, SDK 27.0.
All used the frozen binary SHA256
`b4364cec9173ff0e2fc109cbb4bf7707b6f98eefd2e55234f86b7e026d0971ea`.

Each exploratory row: f16, five warmups, twenty valid completed-command-buffer
GPU samples, fixed 1280x720 input to 2560x1440 RGBA8. Input is coordinate-modulo
tiled trusted RGB with float32 byte SHA256
`e43b54d00e708a81b1e537e5345e72195300b5ab3dde75f2e05204d273953c19`.
Codec, source frame rate, import path, display refresh and presentation window
are not applicable. No browser participates in native execution.

| ID | Golden f32/f16 | GPU p50 ms | GPU p95 ms | GPU max ms | Wall observation ms | Buffer bytes | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| B | PASS/PASS | 10.073625 | 11.029383 | 11.191042 | 211.105917 | 265,434,210 | Higher p95 |
| C | PASS/PASS | 10.149187 | 10.772052 | 11.205292 | 211.497375 | 265,434,210 | Higher p95 |
| D | PASS/PASS | 7.093396 | 7.926704 | 7.970958 | 150.583750 | 265,434,210 | Higher p95 |
| E | PASS/PASS | 6.301354 | 7.088048 | 7.135667 | 133.218000 | 265,434,210 | Higher p95 |
| F | PASS/PASS | 3.972354 | 4.242588 | 4.652750 | 84.473792 | 88,487,010 | Selected lowest p95 |
| G | PASS/PASS | 3.913063 | 4.673531 | 4.745375 | 85.201292 | 88,487,010 | Higher p95 despite lower p50 |

A was not rerun: its historical Phase-1 f16 p50/p95/max remain
65.852229/66.653373/67.086625 ms with the different original 10/60 sample scope.
No row was dropped, no new configuration added, and no numerical failure was
timed. There were no thermal preflight deferrals, rejected exploration attempts
or experiment replacements in the measured study.

Every GPU bracket includes complete neural execution and final RGBA conversion.
Setup, compilation, upload, CPU readback, decode and presentation are excluded.
Wall windows include submission gaps and post-completion evidence logging.
No CPU wall time is presented as GPU work, no isolated medians are summed,
and no speedup ratio against the differently implemented historical WebGPU
pass is claimed. Raw timestamps, hashes, configurations and exact values are in
[the compact evidence](../results/m13-phase15-metal.json).

## 21. Selected candidate

Exactly **F** was selected, by the predeclared ordering: numerical PASS, lower
p95, then p50, then configured buffer bytes, then simpler matrix order. F wins
on p95; G's lower p50 did not override that rule. Selection was committed at
`a6c4ea73d2eee5921dd6ad853a96cdd2147205af` before either binding experiment.
[The selection record](../results/m13-phase15-selection.json) contains every row,
the source pin, both selected golden results and fixed binding thresholds.

The implementation and immutable copied binary still come from
`523217d336e92f60efae7bcd89bc4489c5012b6d`. Metadata-only commits do not change
the frozen source inventory. Fresh selected WebGPU qualification and independent
pre-binding review passed. There was no runner-up substitution or tuning after
selection, and optimized f32 was not performance-ranked.

## 22. Binding run 1

Independent process and engine, candidate F f16, ten whole-graph warmups and
exactly sixty measured samples. Started at 2026-09-22T20:10:57Z; wall observation
window 256.835292 ms. All timestamps are valid and ordered.

| p50 ms | p95 ms | max ms | Valid samples | Thermal before / after |
|---:|---:|---:|---:|---|
| 3.921708 | 4.458015 | 4.757042 | 60/60 | nominal / nominal |

Both p50 <= 8.0 and p95 <= 10.0 pass. The full-size post-window diagnostic
audit has zero non-finite/out-of-range values, opaque alpha, and exact RGBA
agreement with the fast path. Readback and that extra diagnostic execution
are outside the measured series. This audit is not a full-size independent
numerical golden; the numerical oracle remains the trusted small fixture.

## 23. Binding run 2

Second independent process and engine, same frozen candidate and input, ten
warmups and exactly sixty samples. Started at 2026-09-22T20:11:05Z; wall observation
window 255.409125 ms. All timestamps are valid and ordered.

| p50 ms | p95 ms | max ms | Valid samples | Thermal before / after |
|---:|---:|---:|---:|---|
| 3.932604 | 4.451823 | 4.554542 | 60/60 | nominal / nominal |

Both thresholds pass again, as do all post-window output checks. This was the
second preregistered run, not the best of additional attempts. The two runs
are reported separately, not pooled. There was no third run, favorable retry,
stopped-run replacement or new configuration after binding.

## 24. Thermal state

Both binding runs and all exploratory runs started nominal. Every recorded
warmup/sample thermal observation and every after-state was nominal (raw enum 0).
Thermal observations are status readings, not temperature or continuous DVFS
measurements. The two binding sample windows are about 0.256 seconds each;
they do not establish sustained thermal behavior, a long soak or energy use.

The preregistered controller refuses elevated preflight and records a separate
non-started attempt. Once warmups begin, interruption/error/deadline consumes
the attempt and cannot be replaced. A 120-second owned-process-group deadline
bounds each experiment. No fan-control, artificial cooling, power-limit or
clock changes were used. None of those stop/defer paths occurred in this study.

## 25. Reviewer findings

Independent read-only reviewers covered all fifteen requested areas:

| Areas | Review result |
|---|---|
| 1 packed layout, 2 weight conversion | PASS |
| 3 threadgroup bounds, 4 halo indexing | PASS |
| 5 output-channel blocking, 6 spatial blocking | PASS |
| 7 nearest/head fusion, 8 residual/output fusion | PASS |
| 9 fp16 reduction semantics | PASS after stronger actual-kernel controls |
| 10 buffer lifetime, 11 command synchronization | PASS |
| 12 benchmark methodology, 13 selection rule | PASS after evidence-controller repairs |
| 14 selected golden/parity evidence | PASS before binding |
| 15 final performance verdict | PASS after both binding runs |

Pre-timing findings strengthened spatially discriminating inputs, stem/head
rounding fixtures and non-half-exact residual tests. The evidence controller
was repaired to rederive summaries, pin all transitive sources/resources,
require all twelve golden records, validate original/fresh WebGPU provenance,
check every warmup timestamp, preserve rejected/unsafe outcomes and distinguish
thermal deferral from a consumed run. Post-run pin observations make failures
replayable even after a changed worktree is restored.

All known scoped P0/P1 findings were resolved, tested and re-reviewed before
the first exploratory timing. Development-only MSL constructor and synthetic
Swift/type-escaping errors were fixed before affected tests passed; no timing
was taken from them. Reviewers did not execute the GPU workloads. Main-runner
execution and independently recomputed raw evidence support the observations.
Final deliverable review accepted this 28-section report and the scoped verdict
with no remaining findings; exact closing-HEAD CI is the publication prerequisite.

## 26. Phase-1.5 verdict

**METAL REALTIME BACKEND QUALIFIED**

The exact production model passes original f32/f16 numerical and falsification
contracts, actual selected WebGPU parity, fast/diagnostic agreement, resource
review and both fixed binding timing gates. Both runs have 60/60 valid samples,
p50 below 8 ms and p95 below 10 ms. The conclusion follows the prospective
definition, not an expectation about native Metal speed.

The claim is **real-time-capable isolated neural backend timing on the measured
Apple M5 at 720p input / 1440p output**. It is not 60 FPS video playback,
display latency, A/V synchronization, decoder/ingest performance, ScreenCaptureKit
performance, cross-machine portability or a production rollout. Phase-1
qualification and deployed WebGPU behavior remain unchanged.

## 27. Repository/CI/size

Measured code: `523217d336e92f60efae7bcd89bc4489c5012b6d`; exact code CI
[35777639851](https://github.com/yassinsolim/AetherVSR/actions/runs/35777639851)
passed gate, fusion and native-metal before timing. Selection commit:
`a6c4ea73d2eee5921dd6ad853a96cdd2147205af`. Measured-evidence and full clean-clone
verification commit: `bcb63d4b709ce6a36fca04140ebc7fd32a7a8596`.

| Gate | Clean local | Clean clone |
|---|---|---|
| Locked npm install, typecheck, lint, research checks | PASS | PASS |
| Historical Vitest | 1501 PASS, 3 existing skips | 1501 PASS, 3 existing skips |
| Python | 612 PASS | 612 PASS |
| FFmpeg opt-in | 80 PASS | 80 PASS |
| Desktop static/tests | 71 PASS | 71 PASS |
| Web, extension, desktop builds | PASS | PASS |
| Native release build | PASS | PASS |
| Native CPU mode | 10 PASS, 8 physical-GPU skips | 10 PASS, 8 physical-GPU skips |
| Native physical M5 tests | 18 PASS | 18 PASS |
| Optimized f32/f16 golden, parity, falsification | PASS | PASS |
| Complete performance evidence replay | PASS | PASS |
| Immutable Phase-1 evidence replay | PASS | PASS |

The clone uses its own npm install and native build, while reusing the available
Python environment. Retained raw data was copied to ignored paths only for
evidence replay, not for a second timing study. All four clean package builds
per product (twice in each workspace) match byte-for-byte including provenance.
Production extension payload remains `46cebd53b7665ce48772792d2ec2eb73057e915d93ef76374900f30a159ac551`,
diagnostic extension `7334c5db0f476048a10b63a2e0d9e4b240436fa3a669925e113cedc93cb1003f`,
desktop `d9e1f8e9bdc544084ae1ce5a7857562e4ce047b7046478b6936413fed09246b7`.
The two pre-existing moderate npm advisories remain; no dependency was changed.

[Verification index](../results/m13-phase15-verification.json) pins all gate
logs and package checks. The verified code/evidence tree is 55,287,032 bytes,
below the unchanged 58,720,256-byte cap. Raw tensors, binaries, GPU captures,
DerivedData and caches remain untracked; only compact derived evidence is added.
The native hosted CI job explicitly skips physical execution and does not stand
in for measured M5 qualification. Closing documentation-only publication must
also pass exact-final-HEAD gate/fusion/native-metal CI with clean HEAD equal
origin/main; the exact closing run is recorded in the publication handoff.

Closing tracked tree: 55,328,779 bytes, leaving 3,391,477 bytes below
the unchanged 58,720,256-byte cap.

With the retained local evidence, reproduce the complete result without timing:

```sh
node tools/m13/optimize.mjs check .cache/m13/phase15-study-01 results/m13-phase15-metal.json
node tools/m13/report.mjs --check results/m13-phase1-metal.json
```

The isolated CLI/library can be built from the package without Electron. The
fresh WebGPU comparison uses the existing diagnostic shell and cached Playwright
installation; these are evidence tooling, not native inference dependencies.

## 28. Phase-2 recommendation

After reviewed publication, the normal next milestone can be separately
authorized **M13 Phase 2 - Native Core Video / AVFoundation Playback**. That
would evaluate AVPlayer to AVPlayerItemVideoOutput, CVPixelBuffer and
CVMetalTextureCache, then texture input adaptation to the qualified optimized
graph and native presentation. These interfaces and their end-to-end behavior
are not implemented or qualified here.

The remaining work is genuine ingest/scheduling/presentation and playback
validation, not attaching video to the old 65 ms graph. The current result
provides measured neural-stage headroom on one M5; it does not spend or prove
the rest of the frame budget. No Phase-2 implementation began automatically.