# M13 Native Metal Architecture

## Phase-1 Scope and Baseline

Owner-authorized Phase 1 only: a non-interactive Swift/Metal inference tool.
No Core Video, AVFoundation, ScreenCaptureKit, player, presentation surface,
permissions, UI, Core ML, or MPSGraph. WebGPU, Electron, historical evidence,
the model and the accepted checkpoint infrastructure are frozen.

Starting source: `1fc5953496fd029262c51dd1266a6b8c3b4923a5`, clean main equal
to origin/main. Exact CI `35417235937`: gate and fusion success. Fresh baseline
on 2026-09-22: 1497 Vitest passed, 3 existing skips, 54 files; Python 612;
desktop 71; opt-ins 80. Install, typecheck, lint, research checks, desktop
checks, web/extension/desktop builds passed. Two existing moderate npm
advisories remain. Tracked tree 54,844,578 bytes; cap 58,720,256 bytes.

## Official API Research

Primary Apple documentation checked 2026-09-22:

- [Compute lifecycle](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu): MTLDevice, compute pipeline state, command queue/buffer, encoders, resources, and completion.
- [MTLDevice](https://developer.apple.com/documentation/metal/mtldevice), [MTLComputePipelineState](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate): one device owns all resources and compiled pipeline limits constrain dispatch.
- [MTLBuffer](https://developer.apple.com/documentation/metal/mtlbuffer) and [MTLTexture](https://developer.apple.com/documentation/metal/mtltexture): explicit storage/usage and byte layouts; shared memory does not remove CPU/GPU synchronization requirements.
- [Runtime source compilation](https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)): compile bundled MSL once during configuration. This supported API does not require an offline metallib, signing, or capture permission.
- [MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), revision 2026-06-04: binary32 float, binary16 half; float narrowing normally round-to-nearest/ties-even; fast math can assume finite values and reassociate; contraction is independently controllable; threadgroup barriers must be uniform. Sections 1.6.3, 2, 4, 6, and 8 govern the implementation.
- [GPU start/end times](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime): read only after completion; initially zero; command-buffer duration is not CPU submission time or physical presentation latency.
- [SwiftPM resources](https://developer.apple.com/documentation/xcode/bundling-resources-with-a-swift-package): `.copy` preserves MSL bytes, accessed with `Bundle.module`; no generated Xcode user state required.

Local toolchain: Apple Swift 6.4 / Xcode SDK 27.0. The optional offline Metal
Toolchain is not installed. The initial build uses a copied MSL source resource
and public runtime compilation. If that compiler is unavailable, stop and
report the prerequisite; do not silently install tools or bypass permissions.

## Native Target and Backend Contract

`native/macos` is an isolated Swift package: an `AetherMetal` library, a CLI
validation executable, and native tests. The library owns model buffers,
pipelines, command queue and configured tensors. The CLI owns fixture loading,
validation-only readback, evidence output and benchmarking, not a future GUI.

Contract: load/configure immutable model and extent; process an explicit tensor;
wait for completion for offline validation; release/destroy resources. Reuse
configured allocations. A failed or destroyed context cannot process again.
No JavaScript participates in native dispatch. WebGPU remains the reference
and cross-platform backend; this does not change its Upscaler abstraction.

## Model and Tensor Contract

Read the existing production JSON directly, with CryptoKit SHA256 over the
entire file before decoding:
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.
Embedded identity is separately
`9154d9490b02d8fdf136edd990217cfb6f0e9956ed928da7179d7781832c8e02`.
There are 6291 source Float32 parameters; validate exact tensor names, sizes,
layer metadata, finite values, RGB mean zero/scale one, scale two and C16D2.
No generated, copied or retrained weights are committed.

Initial layout is dense planar CHW (batch one): element index
`channel * width * height + y * width + x`; row stride is width, plane stride
is width*height. Weights are OIHW, indexed
`((out * inChannels + in) * kernel + ky) * kernel + kx`;
bias index is output channel. Zero padding is explicit, never clamp-to-edge.
The first implementation has no packed tensor lanes, tiling, spatial blocking,
shared threadgroup storage or operation fusion. Tiny asymmetric/channel-coded
fixtures must expose transposition and stride errors.

Exact graph, each operation materialized separately:
stem conv3->16 k5 p2 -> tanh -> body.0 conv16->16 k3 p1 -> tanh ->
body.1 conv16->16 k3 p1 -> tanh -> nearest2x features ->
head conv16->3 k3 p1 -> add nearest2x original RGB -> clamp[0,1].
Head zero padding is in high-resolution coordinates, before any downsampling.

## Precision and Numeric Acceptance (Before Metal Results)

First implement f32. Compile safe/precise math, MSL 3.0, contraction disabled.
After f32 golden and falsification pass, implement half weights/activations
using Float32->Float16 nearest-even, not integer quantization. Half convolution
accumulation and half tanh are intentional; residual addition/clamp use float
as in the production head. A group-of-four dot accumulation may reproduce
the WGSL reduction boundary; its layout must be tested explicitly.

Trusted fixture: `public/models/golden-c16d2.json`, byte SHA256
`7ffe8d5c26ef02f605c04e9057dd5ef9767dc83e84695211eff39f04a1e3a759`.
Input is planar float32 3x16x24. Stem/body checkpoints are planar
16x16x24. Golden final is interleaved RGB 32x48x3, converted explicitly for
comparison. Required ordered checkpoints: stem, body.0, body.1, final.

- Absolute tolerance: f32 0.001; f16 0.05. Relative tolerance: zero.
- Final float: the same precision-specific absolute tolerance.
- Normalized RGBA8: max(precision tolerance, 1.5/255), matching verifyGolden.
- Alpha is one/255-byte opaque; no sRGB transfer is applied to normalized data.
- Maximum failing elements: zero. Any NaN/Inf, missing/reordered stage, wrong
  shape/layout/model/precision, or mismatched evidence identity fails.
- Report max/mean error, failing/nonfinite counts and worst index/edge-region
  information. Preserve full small tensor dumps outside Git for reproduction.
- Do not loosen tolerances after results. Exact identity is reported only when
  all compared float bits/normalized bytes actually match.

Falsification covers wrong model SHA, altered weight, wrong shape, wrong
checkpoint order/name, nonfinite output, perturbed expected vector, bad packing
and incorrect precision metadata. Exercise actual comparison code and loader,
not only a returned PASS Boolean. GPU absence in CI is not a hardware PASS.

## Resource Ownership and Synchronization

Check dimensions and overflow before multiplying byte counts; bound allocations
to supported device limits and Phase-1 maximum1280x720. Use buffers with explicit
Float/Float16 representation, separate outputs and complete dispatch bounds.
Use a single serial command queue and tracked resource hazards. End each encoder
before dependent work. Strong references remain owned by the configured backend
until all submitted commands finish. No CPU read/write or allocation reuse while
GPU access is pending. Validation readbacks occur only after successful completed
status; errors fail closed. No per-frame capture or pixel readback path exists.

## WebGPU Comparison

Use unchanged production WGSL builders, packModel and committed vectors in a
diagnostic-only export. Retain actual stage arrays and final float/normalized
outputs from both backends. Never reconstruct WebGPU output on the CPU and call
it a GPU comparison. The production head exposes rgba8unorm/rgba16float storage;
any diagnostic float32 output adaptation must be confined to an identified
storage-format substitution and compared against its original quantized path.
Do not change production WGSL or the Electron security shell.

## Timing and Search Policy (Before Timing)

No performance measurements before both precision golden gates and parity pass.
One initial implementation, no configuration sweep planned. Fixed1280x720
deterministic tiled trusted input; separately 10 warmup and 60 measured iterations
per precision. Pipelines, model, input and activation buffers are configured
outside the window. Collect gpuEndTime-gpuStartTime after completed status.
Measure stage-isolated command buffers and a separate whole-graph command buffer;
report them separately, not sum-of-medians. Whole graph includes materialized
conv/activation/upsample/residual/clamp/RGBA conversion, excludes input upload,
CPU readback, compile, decode, ingest and presentation. Record sample counts,
observation window, p50/p95/max and unavailable timestamps as null/not measured.

The M11 M5 soak (600001.3ms, 36000 samples) observed WebGPU upscale p50=6.22592ms,
p95=7.0156288ms. That production pass scope differs from this unfused offline
tensor graph. Report both scopes; no speedup percentage or real-time-player
claim. No optimization follows a slow result in this initial qualification.

## Phase Gates

Review loader/layout/bounds -> f32 golden and corruption -> f16 golden and
corruption -> actual WebGPU/Metal common-input comparison -> fixed performance
measurement -> full historical/native gates and independent review -> published
Phase-1 verdict. Only METAL INFERENCE QUALIFIED permits recommending Phase 2;
Phase 2 is not implemented here. All raw attempts, failures and binary outputs
remain ignored; the56MiB cap is unchanged.

## Timing Implementation Detail (Before the First Timing Run)

The first common-input paired capture at source
`8f952b68e847d72c4a81225bfb0eb12715fb049d` passed both precisions. Its raw tensors
remain in `.cache/m13/parity-01`. Before timing, inline Metal dispatch constants
are replaced with configured reusable buffers; shaders and arithmetic remain
unchanged. The resulting source must pass a new paired capture before timing.

For each precision, run ten whole-graph warmups then sixty whole-graph samples;
next run ten complete stage-isolated warmup cycles then sixty stage-isolated
cycles in graph order. Each isolated operation uses its own completed command
buffer and the same preallocated tensors. Stage samples share the recorded
wall-clock observation window for those sixty interleaved cycles. Whole-graph
and isolated-stage series are not interchangeable or additive. Retain each
command buffer's GPU start/end seconds and derived milliseconds. Percentiles
use linear interpolation at `(sampleCount - 1) * percentile`; missing GPU
timestamps are null, counted, and prevent complete timing qualification.

The CLI performs a small golden/parity check before the fixed large workload
and a separate full-size finite-output/readback check after it. Neither is
inside a timing window. Tile each RGB golden plane by coordinate modulo its
24x16 extent to construct the 1280x720 input. Run f32 then f16 serially, once;
no sweep, workgroup changes, kernel fusion, retries to improve timing, or
same-scope speedup claim is planned.