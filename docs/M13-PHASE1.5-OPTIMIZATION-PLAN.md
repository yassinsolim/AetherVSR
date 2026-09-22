# M13 Phase 1.5 Optimization Plan

## Scope and Frozen Baseline

Preregistered before optimized kernels or timings, 2026-09-22. Starting clean
main equals origin/main at `5b470064aea3cfee4ca95e84696ef8a359478fc9`.
Exact CI 35767539893: gate, fusion, native-metal success. Fresh baseline:
1499 Vitest PASS, three existing skips; 612 Python; 80 opt-in; 71 desktop;
13 physical native tests; CPU mode nine PASS/four explicit GPU skips. Install,
static checks, all builds and Phase-1 evidence reproduction pass.

Phase 1 remains METAL INFERENCE QUALIFIED. Its simple MetalEngine, Shaders.metal,
golden contract, report and raw timing are retained unchanged. Production model
bytes SHA256 is `d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`;
embedded identity is `9154d9490b02d8fdf136edd990217cfb6f0e9956ed928da7179d7781832c8e02`.
No new model identity, weights, activation approximation, padding/resize change,
quantization change, production WebGPU change or historical evidence rewrite.
No video, Core Video, AVFoundation, capture, GUI, Core ML or MPSGraph work.

## Hypothesis and Portable Algorithms

Retained f16 isolated p50: stem convolution 10.5220 ms; body.0 17.9802 ms;
body.1 17.9866 ms; head 13.5833 ms. Convolution dominates; no new profiler run
is necessary. Historical whole graph p50/p95: f16 65.852229/66.653373 ms,
f32 83.980375/84.548329 ms, ten warmups and sixty samples each. These remain
Phase-1 measurements, not new candidate measurements.

The production WGSL builders were inspected before design. Portable ideas:
group four channels per vector; store tap-major output-channel weights together;
stage a spatial tile plus zero halo; reuse one sample across output channels;
retain multiple neighboring output accumulators in registers; apply tanh at
the convolution store; evaluate nearest-expanded head taps directly from LR
features. These reduce redundant loads or materialization without changing the
graph. WGSL binding syntax, browser limits, and compiler lowering are not assumed
portable. MSL is generated with statically named accumulators, public Metal
pipelines and explicit threadgroup synchronization, not mechanically translated.

Observed public limits on Apple M5: threadExecutionWidth 32, pipeline maximum
1024 threads, device maximum threadgroup memory 32768 bytes, maximum buffer
14302248960 bytes. These are capabilities, not performance predictions. Every
actual candidate pipeline must also validate its own limits and shared storage.

## Finite Candidate Matrix

IDs and values are fixed below. f16 is the performance target. f32 uses the
same generated family only for correctness and shared-code regression. Workgroups
are XxY threads; spatial blocks are output pixels per invocation. All use C4
grouped activations and tap-major weights unless labeled reference.

| ID | Family | Body group | Body spatial | Body output block | Shared halo | Conv+tanh | Stem group/spatial/output | Head group/spatial/output | Reconstruction |
|---|---|---|---|---:|---|---|---|---|---|
| A | Frozen Phase-1 scalar reference | 64x1 linear | 1x1 | 1 | No | Separate | Reference | Reference | Materialized; historical timing only |
| B | Packed vector | 8x4 | 1x1 | 4 | No | Separate | 8x4 / 1x1 / 16 | 8x4 / 1x1 / 3 | Materialized nearest, separate final composition |
| C | Packed tiled | 8x4 | 1x1 | 4 | Yes | Separate | 8x4 / 1x1 / 16 | 8x4 / 1x1 / 3 | As B |
| D | Output blocked | 8x4 | 1x1 | 16 | Yes | Fused | 8x4 / 1x1 / 16 | 8x4 / 1x1 / 3 | As B |
| E | Spatial blocked | 8x4 | 2x2 | 16 | Yes | Fused | 8x8 / 2x2 / 16 | 8x4 / 2x2 / 3 | As B |
| F | Fused reconstruction | 8x4 | 2x2 | 16 | Yes | Fused | 8x8 / 2x2 / 16 | 8x4 / 2x4 / 3 | Direct LR head + original RGB + clamp + RGBA |
| G | Larger body group | 8x8 | 2x2 | 16 | Yes | Fused | 8x8 / 2x2 / 16 | 8x4 / 2x4 / 3 | As F |

This is six new configurations, not an open parameter sweep. B/C versus D
changes both output blocking and activation fusion; it is a candidate comparison,
not an isolated causal estimate of either change. No configuration is added
after an unfavorable measurement without a new prospective hypothesis and finite
matrix. The optional SIMD-group second study is not preregistered or authorized
by this matrix; only consider it prospectively if the portable family misses.

Only body convolutions use threadgroup staging; stem and head use direct global
reads with register reuse in every candidate. Conv+tanh covers stem, body.0 and
body.1 together, not the linear head. Reduction traversal is input group, kernel
row, kernel column, then one half4 dot/add per output accumulator. All six source
manifests and generated shader hashes freeze before the first exploratory timing.
No timing-informed source revision under an existing candidate ID is permitted.

## Layout, Precision and Fusion Contract

Activations: `[channel/4][row][column][channel%4]`, dense group-major vectors,
four groups for C16. Vector alignment is eight bytes for half4, sixteen for
float4. No image-edge padding is stored; out-of-bounds taps explicitly read zero.
Input is configured RGB in float4 (fourth lane zero), narrowed on stem reads in
f16. Original float RGB remains available for residual addition.

Weights: deterministic source OIHW to `[inputGroup][tap][outputChannel][lane]`.
RGB stem's fourth lane is zero; head retains exactly three output channels.
Packing validates shape/finiteness/source identity and is performed only during
configuration. Derivative buffer hashes record representation, not a new model.
Half weights and accumulators, half tanh and float residual/clamp match Phase 1.
MSL 3.0, fast math disabled, contraction disabled. No fast approximate tanh.

Body tile size is `(groupX*blockX+2) * (groupY*blockY+2)` vectors, reused for
each input group with uniform barriers before overwrite and after load. No
edge thread exits before a barrier. Output bounds are checked only at stores.
The fused head bounds-checks high-resolution coordinates before integer divide
by two. Odd/even output coordinates and border zeros must be independently tested.

Diagnostic mode copies the three post-tanh stages and emits final float RGB
before quantization. It must also match fast-mode RGBA. Fast mode has no stage
readbacks or diagnostic float writes. Two preallocated packed activation buffers
may be reused; all diagnostic allocations occur during configuration, outside
timing. B-E retain materialized nearest features for the comparison; F/G remove
that large tensor and separate head/residual intermediates. Report actual
configured buffer bytes and texture allocation separately, not peak RSS.

## Correctness and Failure Controls

Order for each candidate: packing tests, tiny asymmetric CPU-oracle tests,
full trusted golden in f16 and shared f32, then exploration timing. Tiny tests
cover non-square and partial workgroups, distinct channel/output/tap values,
zero halo, all four output parities, activated checkpoints and final composition.
They must detect deliberate channel/weight permutations, not just natural images.
Every new convolution family also executes discriminating half4-versus-scalar
and half-versus-wide accumulation fixtures, including when activation is fused.
Wrong source hash, corrupt derived weights, wrong shapes/stage order, bad layout,
NaN/Inf and perturbed expected tensors must fail. Existing Phase-1 tests remain.

Unchanged acceptance: absolute f16 0.05, f32 0.001, relative zero; final float
uses the same precision tolerance, normalized RGBA uses max(tolerance, 1.5/255),
opaque alpha, zero failing and non-finite elements. No numerical failure is timed.
Selected candidate also passes actual WebGPU common-input stage/final/RGBA
comparison, not merely separate golden acceptance. Reference arrays are retained
GPU captures, byte-hashed and independently checked against their source pins.
Both precisions and diagnostic/fast-path RGBA agreement must pass before binding-01.
A frozen winner that fails parity is NOT QUALIFIED; no runner-up substitution.

## Exploration and Selection

Fixed order B, C, D, E, F, G. Each eligible candidate gets one exploratory f16
whole-graph series: five warmups, twenty measured samples, fixed 1280x720 tiled
trusted input and 2560x1440 output. These are explicitly non-binding rankings.
No exploratory f32 performance runs. A stays as the historical timing reference.
Exploration also requires nominal thermal state at start. Any non-nominal state
observed during an experiment is recorded but never used to discard samples or
restart. Thermal state is recorded after every whole-graph completion; only
command failure, invalid timestamps or the fixed process deadline stops execution.
Record all candidates, even rejected/not measured ones, with source, binary,
generated shader/packed weight hashes, geometry, fusion, memory, numerical status,
raw timestamps, p50/p95/max and reason. Compilation/configuration/input upload,
readback and CPU waits are outside the GPU bracket. One queue and one command
buffer per graph, no interstage CPU waits.

Selection after the complete finite sweep: only numerical PASS and twenty valid
samples; lexicographically lowest p95, then p50, then configured buffer bytes,
then simpler portable configuration in matrix order. Publish exactly one ID
and its immutable source/configuration before binding. No selection from binding
outcomes. If all candidates fail correctness/evidence, stop with NOT QUALIFIED.

## Two Binding Runs and Thermal Policy

Exactly two independent process/engine runs of the selected f16 candidate:
`binding-01`, then `binding-02`. Each: ten whole-graph warmups and sixty measured
samples. No third run, favorable replacement, best-two selection or new tuning.
Before each run, record actual thermal state and require nominal where supported.
If elevated, do not begin or consume a binding run; resume only after ordinary
nominal state. No fan, power, clock or cooling modifications. Record thermal
state again after each run; do not discard unfavorable thermal outcomes.

Each experiment has a 120-second process deadline. Reserve an immutable attempt
before launch; a successful nominal preflight is required before warmups. A thermal
preflight rejection records NOT_STARTED and does not consume binding execution.
Once warmups begin, interruption, timeout, error or incomplete samples consumes
that binding run, retains its partial evidence, and cannot be replaced. The
second predeclared run may still execute unless an unsafe GPU failure requires
stopping. An incomplete binding pair cannot qualify.

Read gpuStartTime/gpuEndTime only after completed status; retain all sixty raw
pairs and durations. Invalid/unavailable timestamps are null, not zero, and
prevent qualification. Command errors stop further timing and are retained.
Use Phase-1 linear-interpolated percentiles at `(n-1)*p`. Report wall observation
windows separately, p50/p95/max, sample count, device/toolchain/OS, hashes and
working-set bytes. No CPU pixel copies/readbacks or resource allocation in the
recurring graph. A post-window capture checks finite outputs and quantization.

## Immutable Verdict Rules

METAL REALTIME BACKEND QUALIFIED: numerical qualification and evidence PASS,
no model/tolerance change, no invalid/unbounded execution, and BOTH binding runs
have sixty valid samples, p50 <= 8.0 ms and p95 <= 10.0 ms.

METAL REALTIME BACKEND PARTIAL: numerical/evidence PASS with sixty valid samples
in BOTH binding runs and both p95 < 16.67 ms, but the <=8/<=10 headroom target
is not achieved.

METAL REALTIME BACKEND NOT QUALIFIED: correctness failure of the selected path,
any binding p95 >= 16.67 ms, or untrustworthy/incomplete binding evidence. Invalid
exploratory candidates are rejected rather than erasing a correct surviving one.

This qualifies only isolated neural GPU timing on the measured M5, never video
FPS, decode, display latency, A/V synchronization or capture performance. Only
QUALIFIED normally permits recommending a separately scoped Phase 2; PARTIAL
returns to the owner, NOT QUALIFIED stops. No automatic integration or runtime pivot.

## Review and Publication Gates

Independent reviews must cover packing/conversion, threadgroup bounds/halo,
output/spatial blocking, nearest/head and residual/output fusion, fp16 reduction,
buffer lifetime, synchronization, timing, selection, final parity and verdict.
Any P0/P1 correctness finding blocks binding until repaired and revalidated.
Historical/native full gates, clean clone, exact-final-HEAD three-job CI, clean
main equal origin and the unchanged 58,720,256-byte cap are required. Raw tensors,
binaries, GPU captures and caches remain ignored. Publish compact derived
evidence and the required 28-section report; keep Phase-1 evidence immutable.