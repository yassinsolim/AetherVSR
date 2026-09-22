# M13 Phase 1 - Native Metal Inference and Golden Qualification

## 1. Starting state

Started from clean, published `main` at
`1fc5953496fd029262c51dd1266a6b8c3b4923a5`, with exact CI
[35417235937](https://github.com/yassinsolim/AetherVSR/actions/runs/35417235937)
passing gate and fusion. This is the separately accepted checkpoint-infrastructure
repair, not the older M12.1 baseline. That repair was not changed again.

M11 remains **DESKTOP PLAYER MVP READY**. M12 remains **CROSS-VENDOR DESKTOP
PARTIAL** and M12.1 remains **CROSS-VENDOR COMPLETION PARTIAL**. Their reports,
measurements, hardware-unavailable rows and Windows-scaffold limitations are
not rewritten by this native result. WebGPU remains the deployed engine.

The baseline passed 1497 default tests with three existing skips, 612 Python
tests, 80 opt-in tests and 71 desktop tests, plus install, static checks and
builds. Its tracked tree was 54,844,578 bytes against the unchanged 58,720,256-byte
(56 MiB) cap.

## 2. Native architecture

An isolated Swift library owns Metal resources and offline tensor execution;
a CLI owns fixtures, validation readback and evidence. There is no native
player or presentation integration. The implementation does not touch the
existing acquisition/import/upscale/presentation boundaries.

No AVFoundation, Core Video, CVPixelBuffer, IOSurface, ScreenCaptureKit, capture
permission, GUI, overlay, Stremio, Core ML or MPSGraph work was performed.
The existing Electron diagnostic shell supplies only the WebGPU reference.
No Electron security setting or production shader was changed.

[Prospective architecture and timing policy](M13-NATIVE-METAL-ARCHITECTURE.md)
and ADR-0060 in [DECISIONS.md](../DECISIONS.md) preceded native implementation.
Apple's public APIs and MSL specification informed independently written code;
no third-party implementation or new core dependency was imported.

## 3. Build target

[native/macos/Package.swift](../native/macos/Package.swift) provides the
`AetherMetal` library, `aether-metal` executable and XCTest targets, with a
macOS 13 deployment minimum. Qualification was on Apple M5, Mac17,2, 24 GiB,
macOS 26.6.2 build 25G83, Xcode 27.0 build 27A266a, SDK 27.0 and Apple Swift 6.4
(`swiftlang-6.4.0.34.1`, clang `2100.3.34.1`). Other native machines are not qualified.

The optional offline Metal Toolchain was absent. Bundled MSL was successfully
compiled using public `MTLDevice.makeLibrary(source:options:)`, once per engine
configuration: MSL 3.0, fast math disabled, `FP_CONTRACT OFF`. No installation,
signing or permission workaround was needed.

```sh
swift build -c release --package-path native/macos --scratch-path .cache/m13/build
AETHERVSR_METAL_TESTS=1 swift test --package-path native/macos --scratch-path .cache/m13/build
.cache/m13/build/release/aether-metal --model public/models/aethersr-c16d2.json --input public/models/golden-c16d2.json --precision f16 --output .cache/m13/new-golden-id
```

Output IDs must be new. The existing cached Playwright installation and Electron
are additionally required by the paired-reference runner, not by native inference.

## 4. Model loading

The loader reads the existing production JSON directly. It verifies the full
file hash before decoding, then the embedded identity, architecture version 2,
resize-convolution architecture, scale 2, C16D2, RGB input/output, normalization,
ordered layer definitions, tensor key set, shapes and finite values.

The exact 6291 source float32 parameters are retained: stem weight/bias 1200/16,
two body weight/bias pairs of 2304/16, and head weight/bias 432/3. No retraining,
replacement model, alternate activation, reduced channel count or silent fallback
was used. Invalid input or model metadata fails closed.

## 5. Weight provenance

These are separate identities, not interchangeable:

| Object | SHA256 |
|---|---|
| Production JSON bytes | `d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a` |
| Embedded model identity | `9154d9490b02d8fdf136edd990217cfb6f0e9956ed928da7179d7781832c8e02` |
| Trusted golden JSON bytes | `7ffe8d5c26ef02f605c04e9057dd5ef9767dc83e84695211eff39f04a1e3a759` |
| Measured native release executable | `eac26bd0f31a855704c32076861d56a160969d86edb6ee29b9f3f11682afa017` |
| Bundled MSL bytes | `0a0cb45167ca08fa36036b7c661e67d076b9226e4a601a181d719770e653309d` |

Binding native/parity/timing source:
`f370d6ccac90779aa1028589f08904a5e59dcaf2`. The paired and timing wrappers required
the same clean source and binary. The independent checker rehashes the executable,
MSL, archived WebGPU payload, raw reference tensors and native tensor files,
and checks source attribution across their envelopes.

## 6. Tensor layout

Native tensors are dense planar CHW, batch one, indexed as
`channel * width * height + row * width + column`. Source weights are OIHW.
There are no packed storage lanes or implicit row padding in activation buffers.
The validation-only RGBA readback row pitch is aligned to 256 bytes and stripped
explicitly before comparison.

Golden input is 3x16x24. Hidden checkpoints are 16x16x24, each 6144 elements.
Final RGB is 3x32x48, 4608 elements. The golden's interleaved final RGB and the
WebGPU hidden C4 grouping are explicitly converted to CHW. Dumps use little-endian
float32; f16 checkpoint values are widened for inspection, with precision recorded
separately. RGBA8 is normalized linear data with opaque alpha, not an sRGB conversion.

## 7. MSL graph implementation

The complete graph is materialized without fusion:

1. 3-to-16 convolution, 5x5, zero padding 2; tanh.
2. 16-to-16 convolution, 3x3, zero padding 1; tanh.
3. 16-to-16 convolution, 3x3, zero padding 1; tanh.
4. Nearest-neighbor 2x feature expansion.
5. 16-to-3 convolution, 3x3, at high resolution with zero padding 1.
6. Add nearest-upsampled original RGB, clamp to [0,1], write RGBA8.

High-resolution head bounds are applied before mapping to source features.
Each operation has a separate encoder. Dispatch is one output element per
thread with 64-thread groups and explicit bounds. There is no tiling,
threadgroup storage, spatial blocking or output-channel blocking.

The f32 correctness slice was committed at
`45f93b6c72e2fcd00c6e697231264347309f6ddb`; f16 and paired capture followed at
`8f952b68e847d72c4a81225bfb0eb12715fb049d`. Before timing, dispatch constants
were moved from inline arguments to preallocated buffers and both precisions
passed a fresh paired capture. Shader arithmetic was unchanged by that step.

## 8. f32 golden

PASS at the original absolute tolerance 0.001, relative tolerance zero.
Every stage had zero failing elements and zero NaN/Inf values.

| Checkpoint | Maximum absolute error | Mean absolute error |
|---|---:|---:|
| stem | 3.576279e-7 | 5.167706e-8 |
| body.0 | 8.940697e-7 | 8.854951e-8 |
| body.1 | 1.229346e-6 | 1.092454e-7 |
| final float | 2.384186e-7 | 4.927925e-8 |

The initial immutable f32 CLI result and later paired results are retained.
Stage float values are not bit-identical to the golden. The compact evidence
records exact errors, worst indices and edge/interior failure counts.

## 9. f32 falsification

PASS. Tests reject wrong whole-model hash, changed weights, wrong architecture
and convolution metadata, malformed dimensions/counts, missing or reordered
checkpoints, wrong layout, wrong precision declaration, non-opaque/truncated
RGBA, and perturbed expected vectors. Actual captured GPU checkpoints are
also corrupted with NaN, positive/negative infinity, a large finite error
and reversed layout; those comparisons fail.

A separate asymmetric 3x2, two-input/two-output-channel convolution matches
an independent scalar CPU oracle exactly. Channel-coded layout conversion
tests and extreme dimension checks cover indexing and multiplication guards.
These are failure controls, not acceptance based on a supplied PASS flag.

## 10. f16 golden

PASS at the original absolute tolerance 0.05, relative tolerance zero.
All four checkpoints have zero failing and zero non-finite elements.

| Checkpoint | Maximum absolute error | Mean absolute error |
|---|---:|---:|
| stem | 0.005311131 | 0.000496277 |
| body.0 | 0.012225986 | 0.000603176 |
| body.1 | 0.012660682 | 0.000675364 |
| final float | 0.001717180 | 0.000274581 |

Weights/input features narrow from float32 to Float16 using nearest-even.
Convolution uses half4 dot groups and half accumulation, with half tanh and
feature storage. The head is half; conversion to float precedes addition of
the original float RGB residual and float clamp. These precision boundaries
match the intended production path, without promising bit-identical reductions.

## 11. f16 falsification

PASS. The same loader, metadata, expected-vector and actual-output corruption
controls run for f16. Tiny GPU/CPU fixtures cover RGB padding to four lanes,
multiple channel groups and channel/output/tap-distinct weights; deliberate
channel and weight permutations do not match the oracle.

Rounding-sensitive controls distinguish grouped dot accumulation from serial
scalar-half accumulation, and half accumulation from accumulation retained in
float32. Separate conversion cases verify nearest-even ties. A reviewer found
the first small fixture insufficiently discriminating; it was strengthened and
passed before the binding paired capture and performance run.

## 12. WebGPU/Metal stage comparison

PASS using the identical committed float32 input and production model. The
reference ran on the non-fallback Apple WebGPU adapter in Electron 44.4.1,
Chromium 152.0.7977.78, embedded Node 24.21.0. The host CLI used Node 26.5.0.

The existing production WGSL builders and packer are unchanged. An opt-in
diagnostic verifier exports actual GPU hidden-stage arrays. For final float,
one guarded storage declaration changes rgba8unorm to rgba32float; the original
RGBA8 head also runs and is checked against that diagnostic output. This is
labeled diagnostic output, not an unchanged production storage format or a
CPU reconstruction.

| Checkpoint | f32 max | f32 mean | f16 max | f16 mean |
|---|---:|---:|---:|---:|
| stem | 6.854534e-7 | 8.263126e-8 | 0.007568359 | 0.000554267 |
| body.0 | 1.549721e-6 | 1.203814e-7 | 0.010986328 | 0.000704501 |
| body.1 | 1.728535e-6 | 1.465003e-7 | 0.008911133 | 0.000824438 |
| final float | 3.278255e-7 | 6.106347e-8 | 0.001831055 | 0.000357714 |

Tolerances remain 0.001/0.05. Every comparison has zero failing/non-finite
elements, zero edge failures and zero interior failures. No float checkpoint
is claimed bit-identical. WebGPU arrays independently pass the trusted golden
check before becoming a native comparison reference.

## 13. Final RGBA comparison

PASS after explicit alpha and row-layout checks. Normalized RGBA8 uses the
registered `max(precision tolerance, 1.5/255)` threshold.

| Comparison | Maximum error | Mean error | Byte identity |
|---|---:|---:|---|
| Native f32 vs trusted float golden | 0.001960665 | 0.000925975 | Not applicable: quantized vs float |
| Native f16 vs trusted float golden | 0.002953649 | 0.000959689 | Not applicable: quantized vs float |
| Native vs WebGPU, f32 | 0 | 0 | Yes, complete RGBA8 arrays |
| Native vs WebGPU, f16 | 0.003921598 | 0.000356584 | No |

The WebGPU float-head diagnostic agrees with its original normalized RGBA8
output: maximum errors 0.001960725 (f32) and 0.001960069 (f16), both below
1.5/255. All alpha values are opaque. This is offline tensor/output parity,
not decode, color-management, presentation or video-ingest parity.

## 14. Resource/lifetime review

PASS for this synchronous offline backend. One device/queue owns reusable
pipelines, weight/constant/input/intermediate buffers and output texture.
Dimensions are bounded before arithmetic, buffer sizes checked against the
device limit, and outputs use separate resources with tracked hazards.
No shared threadgroup memory or nonuniform barrier is involved.

Load, capture, benchmark and destroy are serialized by one lock. CPU access
follows successful command completion. Unsuccessful completion permanently
latches failure; a later success cannot revive the engine. Destroy is
idempotent and subsequent work is rejected. Actual lifecycle tests exercise
load-before-capture and destroy behavior; the failure latch is tested with a
controlled state input, not a claim of physically induced device loss.

No explicit GPU resource is allocated per inference except command/encoder
objects required for submission. Pixel readback occurs only in offline
validation outside measured windows. Configured buffer totals were 792,601,268
bytes (f32) and 481,087,886 bytes (f16); these exclude the private output texture,
pipelines, driver allocations and CPU captures. Peak process memory was not measured.

## 15. GPU timing methodology

The timing policy was written before the first timing run. The fixed input is
1280x720 planar RGB, made by coordinate-modulo tiling of each trusted 24x16
golden plane; output is 2560x1440. Input float32-byte SHA256 is
`e43b54d00e708a81b1e537e5345e72195300b5ab3dde75f2e05204d273953c19`.
This is a tensor workload: codec, frame rate, import path, drawable and display
refresh are not applicable. There is no presentation window.

For each precision: ten whole-graph warmups, sixty measured whole-graph
iterations, ten complete stage-isolated warmup cycles, then sixty stage cycles
in graph order. f32 preceded f16, once each. A whole iteration uses one command
buffer; each isolated operation uses its own command buffer. Dependencies and
input remain on the same configured GPU resources.

`gpuEndTime - gpuStartTime` is read only after successful completion. Every
raw start/end/duration is retained. Non-positive, non-finite or unavailable
timestamps become null and prevent complete timing qualification; none were
unavailable in this run. Percentiles use linear interpolation at
`(sampleCount - 1) * percentile`. Wall observation windows include host gaps;
they are reported as aggregation scope, not as GPU processing time.

Whole timing includes every unfused conv/tanh/nearest/residual/clamp/RGBA
operation and dependencies. Compilation, configuration, uploads, CPU readback,
decode, ingest and presentation are excluded. A small correctness check
precedes each precision; full-size finite/range/RGBA checks follow timing.
That full-size audit is not an independent full-size numerical golden.

## 16. Metal performance

Measured on the M5 apparatus in section 3, 2026-09-22T17:34:31Z through
17:34:53Z. Thermal state was nominal at both endpoints of each precision;
no continuous thermal or DVFS trace was collected. GPU timestamps were available
for all sixty samples of every series. Times below are milliseconds.

| Whole graph | p50 | p95 | Maximum | Samples | Observation window |
|---|---:|---:|---:|---:|---:|
| f32 | 83.980375 | 84.548329 | 84.898750 | 60 | 5058.778042 ms |
| f16 | 65.852229 | 66.653373 | 67.086625 | 60 | 3970.640292 ms |

Each isolated row below has sixty samples following ten warmups. All f32
isolated rows share the 5220.793750 ms interleaved observation window; all f16
rows share 4135.191500 ms. They are not whole-graph timings or additive medians.

| Isolated operation | f32 p50 / p95 / max | f16 p50 / p95 / max |
|---|---:|---:|
| stem convolution | 11.775896 / 11.954460 / 12.538167 | 10.522021 / 10.748213 / 11.236417 |
| stem tanh | 0.874854 / 0.968979 / 1.067042 | 0.440125 / 0.566065 / 0.625833 |
| body.0 convolution | 23.503521 / 23.557229 / 24.122458 | 17.980208 / 18.169108 / 18.815000 |
| body.0 tanh | 0.866021 / 1.051169 / 1.074833 | 0.419500 / 0.535185 / 0.630833 |
| body.1 convolution | 23.502813 / 23.554681 / 23.685750 | 17.986646 / 18.191356 / 18.774125 |
| body.1 tanh | 0.866646 / 1.079054 / 1.091250 | 0.419958 / 0.488463 / 0.579500 |
| nearest features | 2.323625 / 2.641238 / 3.217208 | 1.975146 / 2.209940 / 2.216292 |
| head convolution | 17.581021 / 17.725833 / 17.816875 | 13.583271 / 13.728540 / 14.276375 |
| nearest RGB | 0.404458 / 0.539742 / 0.616125 | 0.408167 / 0.504587 / 0.577542 |
| residual addition | 1.028271 / 1.274306 / 1.356833 | 0.864271 / 1.065573 / 1.083375 |
| clamp | 0.652396 / 0.837927 / 0.894125 | 0.653625 / 0.851915 / 0.880375 |
| RGBA write | 0.422187 / 0.503077 / 0.570500 | 0.423833 / 0.491610 / 0.617708 |

The full-size post-run audit found zero non-finite/out-of-range hidden or final
values and zero RGBA normalization failures. This simple implementation is
**not a real-time 720p path**: inference alone exceeds a 16.67 ms frame interval.
No playback FPS, latency, quality improvement, energy saving or native speedup
was measured or inferred.

## 17. WebGPU timing comparison scope

The historical M11 M5 WebGPU neural soak recorded p50 6.225920 ms and p95
7.0156288 ms across 36,000 upscale samples in 600001.3 ms. Its timestamps bracket
the optimized production upscale pass, excluding decode/import/presentation.
See the unchanged [M11 report](M11-REPORT.md).

The new Metal workload is an unfused offline tensor graph with explicitly
materialized activations, nearest tensors and final conversion. Different
implementation, representation, scheduling and sample windows prevent an
equivalent-scope speedup or slowdown ratio. The values are contextual, not an
API comparison. No new WebGPU performance run was conducted in Phase 1.

## 18. Optimization sweep

Not performed, as preregistered. One simple implementation and one fixed timing
run per precision were measured. No slow result triggered tuning or a favorable
rerun. Correctness development and dispatch-constant preallocation preceded
timing and are recorded separately from performance optimization.

Retained evidence includes initial f32, the first paired capture, the fresh
binding paired capture after preallocation, and the single performance attempt.
The initial MSL compile error (`kernel` used as a reserved field name) was fixed
before any successful golden result; it was not concealed as a passing run.

## 19. Reviewer findings

Independent, read-only reviews covered Apple API choices, f32 semantics and
lifetime, f16 precision/layout, paired capture, timing, and post-run evidence.
All requested repairs were validated before the affected gate proceeded:

- A failed command could leave the context reusable: added a terminal failure latch.
- Synthesized extent decoding could bypass bounds: removed unchecked decoding.
- A missing-stage test retained a prior shape mutation: restored its pristine fixture.
- RGBA metadata arithmetic preceded validation: bounded dimensions before multiplication.
- A half fixture missed channel/rounding mistakes: added distinct taps and discriminating reductions.
- Early setup failures lacked an envelope: added tested immutable failure recording.
- The evidence checker trusted some descriptors/source copies: rehashed actual native/MSL/WebGPU/timing artifacts and rejected contradictory pins.
- The first publication CI compiler could not type-check the nested CPU tiling expression: replaced it with explicit bounded loops and checked the entire 720p tensor against its measured byte hash.
- Hosted Swift 6.1.2 then inferred a test's expected arithmetic as Int: made the Float16 operand types explicit without changing its expected value or assertion.

Re-reviews passed with no remaining actionable finding in those scopes.
Reviewers did not claim to execute the GPU tests. The main runner executed the
tests, measurements and independent numeric recomputation. Final documentation
review also accepted the scoped qualification verdict, and a scoped re-review
accepted the hosted-compiler compatibility fixes. Exact code-publication CI
passed all three jobs as recorded below.

## 20. Phase-1 verdict

**METAL INFERENCE QUALIFIED**

The exact production C16D2 model executes natively on the Apple M5 through the
complete Metal graph and reproduces the trusted semantics within the original
f32/f16 tolerances. Golden, falsification, actual common-input WebGPU parity,
resource review, native timing and historical nonregression gates pass.

This verdict qualifies offline inference correctness on the recorded M5, not
native player readiness, real-time performance, cross-device portability or
Phase-2 integration. Production remains WebGPU. Publication status is separate
and recorded in section 21.

## 21. Repository/CI/size state

Measured implementation: `f370d6ccac90779aa1028589f08904a5e59dcaf2`.
Independently checked evidence and clean-clone source:
`124e98f8e94897cd85e7bfc89edf08057b1a868c`.

| Gate | Local | Clean clone |
|---|---|---|
| Locked npm install, typecheck, lint, research checks | PASS | PASS |
| Default Vitest | 1499 PASS, 3 existing skips | 1499 PASS, 3 existing skips |
| Python | 612 PASS | 612 PASS |
| FFmpeg opt-in | 80 PASS | 80 PASS |
| Desktop tests and static checks | 71 PASS | 71 PASS |
| Web, production/diagnostic extension, desktop builds | PASS | PASS |
| Native release build and physical M5 tests | 13 tests PASS | 13 tests PASS |
| Native CI mode | Physical tests explicitly opt-in | 9 CPU PASS, 4 GPU skips |
| Independent evidence recomputation | PASS | PASS |

The clone installed its own npm dependencies and built its own native binary.
It reused the available Python environment; retained raw measurements were
copied only to ignored paths for checker replay. This does not claim a second
performance study. Repeated clean package builds in both workspaces were
byte-identical, including provenance, at the same commit.

Frozen payloads remain unchanged: production extension
`46cebd53b7665ce48772792d2ec2eb73057e915d93ef76374900f30a159ac551`, diagnostic
extension `7334c5db0f476048a10b63a2e0d9e4b240436fa3a669925e113cedc93cb1003f`,
desktop `d9e1f8e9bdc544084ae1ce5a7857562e4ce047b7046478b6936413fed09246b7`.
The model, production graph, desktop shell, historical results and accepted
checkpoint repair are unchanged. Two existing moderate npm advisories remain.

[Compact measured evidence](../results/m13-phase1-metal.json) contains exact
statistics and raw hashes; [verification evidence](../results/m13-phase1-verification.json)
records gate logs and package checks. Reproduce the measured index with:

```sh
node tools/m13/report.mjs --check results/m13-phase1-metal.json
```

Raw tensors, binaries, build products and logs are ignored, not added to Git.
The 56 MiB cap and 8 MiB unapproved-file limit are unchanged. A macOS CI job
now builds the native target and runs CPU contracts, explicitly without claiming
physical GPU qualification. The verified code/evidence commit tracks 54,989,043
bytes, leaving 3,731,213 bytes below the cap. Initial publication
`c5b3a9ad32160227b71ca16f45075bc9383eb6bd` tracked 55,023,734 bytes. Its CI
[35765017443](https://github.com/yassinsolim/AetherVSR/actions/runs/35765017443)
passed Linux gate/fusion but failed native compilation of the CPU fixture
builder. That failed run is retained. The scoped repair changes only fixture
construction, adds its full input-hash invariant, and logs the hosted toolchain;
it does not change Metal kernels, model semantics, acceptance limits or timing.
Local release/native tests and unchanged raw-evidence reproduction pass after
the repair. Its clean source `e1b6fa306d18532b87f48f24d54d7147a07a7500` also
reproduced both precision golden/parity reports and all tensor/RGBA artifacts
byte-for-byte against the measured executable. Clean-clone release, CPU and
physical GPU checks passed.

CI [35766108065](https://github.com/yassinsolim/AetherVSR/actions/runs/35766108065)
at that repair passed Linux gate/fusion and native release compilation, then
failed test compilation on an implicitly typed expected value. The runner used
Swift 6.1.2, Xcode 16.4 build 16F6, target arm64 macOS 15. Explicit Float16
operands repair that test-only issue; all 13 local native tests pass afterward.
Both failed publication runs remain recorded. Replacement code commit
`4d8d8e219e15a23c8fa636651e80225b1e7a1fb9` passed exact CI
[35766779995](https://github.com/yassinsolim/AetherVSR/actions/runs/35766779995):
gate, fusion and native-metal all succeeded. Hosted Swift 6.1.2 compiled the
release target and reported nine CPU tests passed with four explicit physical
GPU skips. The clean-clone M5 rerun passed all 13 tests. That code-ready tree
contains 55,026,132 bytes; final documentation/evidence metadata does not alter
the verified code. No performance rerun or new performance result is claimed.

The closing documentation-only commit is also subject to exact-HEAD CI before
handoff. Its run identity and clean HEAD/origin equality are recorded in the
publication handoff; the versioned CI record above identifies the tested code
without claiming a self-referential commit hash.

Closing tracked tree: 55,029,635 bytes, with 3,690,621 bytes of headroom
under the unchanged 58,720,256-byte cap.

## 22. Recommendation for Phase 2

After this verdict is reviewed and published, Phase 2 may be separately scoped
around the now-qualified tensor engine. The measured performance is not ready
for a 60 Hz 720p player; any optimization needs its own bounded, prospective
plan and repeated numerical qualification. Neither a faster native backend
nor end-to-end ingest/presentation behavior follows from this result.

No Phase-2 implementation was started. AVFoundation/Core Video, screen capture,
permissions, GUI and playback integration remain outside this completed
inference phase.