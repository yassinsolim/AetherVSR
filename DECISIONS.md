# DECISIONS.md

Architecture decision records. One per meaningful technical choice. Records are
append-only: to reverse a decision, add a new record that supersedes the old
one rather than editing history.

Status values: `accepted`, `superseded by ADR-nnnn`.

---

## ADR-0001 — WebGPU as the only GPU abstraction

**Status:** accepted (Milestone 1)

**Context.** The baseline needs GPU-accelerated resampling in a browser. WebGL2
is more widely available today; WebGPU is the newer API.

**Decision.** Target WebGPU exclusively. No WebGL2 fallback.

**Why.** The baseline scaler is the easy case; every later stage is not. Neural
inference wants compute shaders, storage buffers, workgroup memory and
ideally `shader-f16` — WebGL2 has none of these. A WebGL2 fallback would be a
second rendering path that could never host the actual product, maintained
forever for a browser population that is shrinking. The measured adapter on the
target machine reports `shader-f16`, `subgroups` and `timestamp-query`, all of
which the neural stage will want.

**Consequences.** No support in engines without WebGPU. The harness must
fail with a clear, actionable message rather than a blank canvas.

---

## ADR-0002 — `requestVideoFrameCallback` as the frame clock

**Status:** accepted (Milestone 1)

**Context.** Work can be driven by `requestAnimationFrame`, by a timer, or by
rVFC.

**Decision.** rVFC drives the pipeline. rAF exists only as an explicitly
degraded fallback where rVFC is absent, and the overlay labels it as such.

**Why.** rVFC fires when a frame is sent to the compositor and carries
per-frame metadata: decoded dimensions, media time, and `presentedFrames`,
which is the only way to detect frames that were presented but never upscaled.
An rAF loop ticks at display rate and must guess whether the frame changed —
typically by polling `currentTime`, which cannot distinguish "no new frame"
from "we were late". Notably, WebVSR uses rAF with a `currentTime` gate and
uses rVFC only to estimate source FPS; we consider that the weaker choice.

**Consequences.** Engines without rVFC get only the degraded clock. Benchmarks
taken under the two clocks are not comparable and are labelled accordingly.

---

## ADR-0003 — `importExternalTexture()` primary, `copyExternalImageToTexture()` fallback

**Status:** accepted (Milestone 1)

**Context.** A decoded video frame must reach the GPU without a CPU readback.

**Decision.** Use `importExternalTexture()` where available. Fall back to
`copyExternalImageToTexture()` into a persistent texture we own and reuse. The
fallback is forceable at runtime with `?import=copy`.

**Why.** The import path can reach a genuine no-copy branch. The Chromium
implementation only does so when several internal conditions hold — shared
image, NV12, multi-planar format support, WebGPU-compatible backing, supported
colour space — so zero-copy is an optimisation we opt into, never a guarantee
we may claim. Chrome exposes `isZeroCopy` only behind developer features, so we
report "we called the zero-copy API", not "no copy occurred".

Making the fallback forceable matters because it is the path *least* likely to
run on a developer's machine, and therefore the one most likely to rot.

**Consequences.** Two shader variants (`texture_external` vs `texture_2d<f32>`),
selected at configuration time. GPU timestamps around the render pass do not
capture the fallback's upload, so the two paths' timings measure different
things and are labelled inline everywhere they appear.

---

## ADR-0004 — Bind groups are rebuilt per frame on the external path

**Status:** accepted (Milestone 1)

**Context.** A `GPUExternalTexture` imported from an `HTMLVideoElement` expires
in an automatic-expiry task shortly after import. The specification does permit
reuse: re-importing an unchanged frame returns the *same* object, un-expired,
and bind groups referencing it stay valid.

**Decision.** Create the bind group inside each frame's `encode()` on the
external path. Do not cache by object identity. The copy fallback binds a view
we own for the whole session and caches one bind group.

**Why.** An identity-keyed cache was implemented and then removed. Chromium
replaces the external texture object whenever the decoded frame advances, and
under rVFC every callback *is* a new frame, so the cache could only ever hit
while playback was stalled. That is no measurable gain in exchange for
per-frame reasoning about resource expiry in the hottest code in the project.
One small bind group per frame is the price the API charges.

**Consequences.** One documented, unavoidable per-frame allocation. Everything
else in the frame loop is preallocated.

---

## ADR-0005 — GPU timestamp queries are the primary performance metric

**Status:** accepted (Milestone 1)

**Context.** `AGENTS.md` forbids guessed performance numbers, so the project
needs a real per-frame processing cost.

**Decision.** Measure the upscale pass with `timestamp-query` via a fixed pool
of resolve/staging buffer pairs. A frame with no free slot goes unmeasured.
Main-thread frame time is recorded separately as `cpu per frame` and labelled
with its exact scope (import call, command recording, submit). When
`timestamp-query` is unavailable the overlay prints "not measured".

**Why.** Wall-clock timing on the main thread does not observe GPU work: it
measured 0.11–0.14 ms in every run here while the GPU pass ranged from 0.48 to
4.64 ms depending on filter and codec. Publishing the CPU figure as "processing
time" would be misleading. Reading back 16 bytes of timestamps (two 64-bit
values) is not a pixel readback and never blocks the frame.

A sampled `onSubmittedWorkDone()` fallback was considered and rejected: it
measures submit-to-completion latency including queue wait, which is not the
same quantity, and a plausible-looking wrong number is worse than an honest
"not measured".

**Consequences.** Chromium quantises timestamps for fingerprinting resistance,
so aggregates over hundreds of frames are required. Documented in
`BENCHMARKS.md`.

---

## ADR-0006 — Filtering happens in the video's encoded (non-linear) domain

**Status:** accepted (Milestone 1)

**Context.** Correct image resampling weights samples in linear light.
`importExternalTexture()` yields values in the requested colour space —
sRGB-encoded by default, not linearised.

**Decision.** Apply reconstruction weights directly to the encoded values, and
present to a `bgra8unorm` (non-`-srgb`) canvas so no implicit conversion occurs
on write.

**Why.** This matches what the browser's own `<video>` compositing does, which
makes the baseline a fair reference for "what the user sees today". Inserting a
linearise/re-encode pair would make the baseline differ visibly from native
playback and would compare a future model against something no user has.

This is a deliberate deviation from strict correctness, recorded so it is not
mistaken for an oversight.

**Consequences.** Revisit for the neural stage: a model trained on linear-light
data will need the transfer function handled explicitly. That will be a new ADR
with measured quality evidence, not a silent change.

---

## ADR-0007 — Canvas backing store is exactly `source x scaleFactor`

**Status:** accepted (Milestone 1)

**Context.** On a Retina display `devicePixelRatio` is 2, and the conventional
canvas recipe multiplies CSS size by DPR.

**Decision.** Set `canvas.width/height` to exactly the upscaled resolution.
Never involve `devicePixelRatio`. CSS letterboxes the fixed buffer.

**Why.** The project's contract is an exact, known scale factor. Deriving the
backing store from CSS size and DPR would silently make the measured scale
factor depend on window size and display, so "2x upscale" would mean something
different in every run.

**Consequences.** The canvas is usually displayed smaller than its backing
store. Runtime letterboxing is done entirely in CSS (`max-width: 100%;
max-height: 100%` in `src/ui/harness.css`), which preserves aspect ratio
without any JavaScript touching the backing store. An earlier `fitContain()`
helper computed the same box in TypeScript; it was never called by the
application and has been deleted rather than left as tested dead code.

---

## ADR-0008 — Plain DOM for the harness, no UI framework

**Status:** accepted (Milestone 1)

**Decision.** Vite plus TypeScript. No React, Vue, Svelte or component library.
The overlay is sixteen text nodes updated on a 250 ms timer.

**Why.** The UI is a diagnostic surface for a GPU pipeline. A framework would
add a dependency, a build step and a render loop competing with the frame loop,
to update sixteen strings. Driving the overlay from a timer rather than the
frame callback also keeps string formatting and layout out of the measurements.

**Consequences.** Any future extension UI is a separate concern and must not
pull a framework into the core pipeline.

---

## ADR-0009 — Test clips are generated deterministically and committed

**Status:** accepted (Milestone 1)

**Context.** Benchmarks need a fixed input. No `ffmpeg` build capable of
producing suitable clips was available on the development machine.

**Decision.** Generate clips in-browser from a procedural pattern with
`tools/make-test-clip.html` and commit the results (~1.2–1.9 MB each). Frame
content is a pure function of frame index. Ship 30 and 60 fps variants in both
VP9/WebM and H.264/MP4.

**Why.** Frame content is chosen to stress an upscaler: a 1-to-16 px frequency
wedge, a radial zone plate, thin diagonals, small text, hard edges and
continuous sub-pixel motion. Committing the artefacts means a benchmark row can
be reproduced exactly rather than approximately. Both codecs are shipped
because the codec measurably changes both decode behaviour and upscale cost.

**Why the generator is paced on `requestAnimationFrame`:** `MediaRecorder`
timestamps captured frames by wall clock, so the encoded presentation
timestamps inherit the jitter of whatever drives capture. A `setTimeout`-paced
capture produced clips with visibly irregular frame intervals; Chrome could not
align them to vsync and dropped frames on playback, which looked exactly like a
pipeline failing to keep up. Pacing on rAF puts timestamps on the display grid
and the problem disappeared. The measured interval distributions from that
diagnosis are recorded in `BENCHMARKS.md` under *Superseded measurements*,
which is the only place figures from a replaced apparatus belong. This cost a
real debugging cycle and is recorded so it is not repeated.

**Consequences.** ~6 MB of binary test assets in the repository. Regenerating
on a machine with a different refresh rate will produce a slightly different
clip; the committed artefacts are the reference.

---

## ADR-0010 — Independently engineered; no third-party code copied

**Status:** accepted (Milestone 1)

**Context.** Several projects solve adjacent problems, and prior-art research
was performed before any architectural choice.

**Decision.** Read for ideas, cite in documentation, copy nothing. Licences
were verified by reading each repository, not assumed.

| Project | Licence observed | Copy code? |
|---|---|---|
| `fishy-ops/WebVSR` | **No LICENSE file**; GitHub API reports `license: null` | **No** — all rights reserved |
| `EvgeneyBogatyrev/EfRLFN` (StreamSR) | MIT | Yes, with notice; weight terms unverified |
| `bytedance/RLFN` | Apache-2.0 | Yes, with conditions |
| `hongyuanyu/SPAN` | Apache-2.0 | Yes, with conditions |
| `onnxruntime-web` | MIT | Yes, with notice |
| FSRCNN / ESPCN papers | No code licence established | Implement the equations independently |

**Consequences.** WebVSR informed our thinking on scheduling and on external
texture use, and we reached the opposite conclusion on the frame clock
(ADR-0002). No line of its code is present. Project licence is Apache-2.0.

---

## ADR-0011 — `Upscaler` is the single seam for future backends

**Status:** accepted (Milestone 1)

**Decision.** One interface with `configure()` / `encode()` / `destroy()`,
plus a declared `scaleFactor` and `neural` flag. `encode()` receives a command
encoder and optional `PassTiming` with separate begin and end query indices.

**Why.** Splitting configuration from encoding is what makes the "no allocation
in the hot path" rule enforceable rather than aspirational. Passing the encoder
rather than a render pass lets a multi-pass neural model record whatever it
needs. Separate begin/end timestamp indices let such a model report its full
cost instead of one convolution. Declaring `scaleFactor` lets the harness size
the canvas without knowing the algorithm, so a future 4x model needs no harness
change.

The seam's *lifecycle* is exercised at runtime: the upscaler selector destroys
the running instance and configures a fresh one while acquisition and
presentation keep going, so `configure`/`destroy`/re-`configure` cannot rot.

**What this does not yet prove.** Both current implementations are the same
class, `BaselineScaler`, differing only in filter. The interface has never been
implemented by a second class, so its sufficiency for a multi-pass neural
backend is a design argument, not a demonstrated fact. Milestone 2's ingest
pass will be the first genuine second implementation and is the real test.

**Consequences.** Implementations must tolerate both `FrameTexture` variants,
or declare which they support.

---

## ADR-0012 — A dedicated ingest pass converts the external texture once

**Status:** accepted (Milestone 2)

**Context.** Milestone 1 observed that a nine-tap kernel was much more
expensive reading `texture_external` than reading an ordinary texture, but the
two figures bracketed different work so the comparison was suggestive only.
Milestone 2 measured it properly, timing an ingest pass and its consumer
separately in the same command encoder.

**Decision.** `ExternalTextureIngest` converts the imported video frame into a
regular `GPUTexture` in one pass. Any consumer that samples the frame more than
about twice should run behind it.

**Why.** Measured: a `texture_external` tap costs ≈0.360 ms per 720p frame
against ≈0.100 ms for a `texture_2d<f32>` tap. The ingest pass costs
0.357–0.447 ms and saves ≈0.26 ms per subsequent tap, so it repays itself at
roughly two taps. Nine-tap Catmull-Rom total drops from 3.765 ms to 1.703 ms —
2.2x — and that comparison covers all GPU work we encode, so it is a real
ranking rather than two differently-scoped numbers.

A convolution reads nine times per output pixel *per input channel*. For any
neural stage the question is not whether to ingest but that ingesting is
obviously correct.

**Consequences.** One extra pass and one extra full-resolution texture. The
output is `TEXTURE_BINDING | RENDER_ATTACHMENT` so a ping-pong graph can write
back into it. `rgba16float` costs marginally more to produce than `rgba8unorm`
but is slightly cheaper to sample; the choice is deferred to whichever the
neural stage needs.

---

## ADR-0013 — Benchmarks must refuse to report unmeasured work

**Status:** accepted (Milestone 2)

**Context.** Three separate times during this milestone, a harness produced a
number that looked like a result and was not one: a throttled tab yielded
all-`NaN` aggregates that became `null` when marshalled out of the page as
JSON; a shader that failed to compile left its output buffer zeroed, which
scored as a plausible image; and a bind group rejected for exceeding a device
limit produced identical begin/end timestamps, i.e. 0 ms and infinite
throughput.

**Decision.** Every measurement path fails loudly instead of returning a
figure. `IngestBench` throws when it collected no frames or no GPU samples.
`ConvBench` marks a case invalid when a dispatch produced a zero-length
timestamp span and attaches the Dawn validation message. `verifyConv` fails on
any non-info shader compilation diagnostic.

**Why.** `AGENTS.md` §3 forbids reporting a benchmark that was not measured.
That rule is only enforceable if the tooling can tell the difference. A silent
zero is worse than a crash because it survives into a table.

**Consequences.** Slightly more code in every harness. Each of the three
failures above was caught by the mechanism rather than by luck.

---

## ADR-0014 — Convolution kernels are verified against a CPU reference before timing

**Status:** accepted (Milestone 2)

**Decision.** `conv-verify.ts` runs the same shader the benchmark uses on small
tensors and compares every output element against a straightforward CPU
implementation, including padding, register blocking, activation and residual
variants.

**Why.** It immediately found a real bug: the residual path referenced a
binding scoped inside the accumulation loop, so pipeline creation failed
silently and the output stayed zero. Without the check that configuration would
have been recorded as unusually fast. A throughput number from an incorrect
kernel is not merely useless, it is fast *because* it is skipping work.

**Consequences.** Verification reads pixels back to the CPU, which is forbidden
in the frame hot path and entirely appropriate offline. All nine cases now pass
at ≤1.5e-7 absolute error.

---

## ADR-0015 — Runtime choice deferred: hand-written WGSL first, ORT re-evaluated with GPU IO binding

**Status:** accepted (Milestone 2)

**Context.** Milestone 2 was to decide between hand-written WGSL and ONNX
Runtime Web on the WebGPU EP, by measurement.

**Decision.** Continue with hand-written WGSL for Milestone 3, and keep ORT
open pending one specific experiment: the same model with `Tensor.fromGpuBuffer`
input *and* `gpu-buffer` output on AetherVSR's own device.

**Why.** ORT's native WebGPU EP ran the identical convolution at 34.7 ms with
CPU output and 20.1 ms with GPU-resident output and an explicit queue
completion fence. Both still include a 56.3 MB CPU→GPU input upload per
iteration, so neither is a clean comparison against our GPU-pass-only 8.585 ms.

**No ranking is drawn from those numbers.** They are a different scope from our
8.585 ms WGSL pass time, and backing out an estimated upload to compare them
would be inventing the very number in question. ORT's compute-only cost on this
device remains unmeasured.

The decision to continue with WGSL for Milestone 3 therefore rests on the
non-performance evidence below, and on the fact that Milestone 2's throughput
gap must be closed in kernels regardless of who wrote them:

What *does* inform the decision now, from source review rather than taste:

- ORT's execution-provider placement is not programmatically queryable. An
  unsupported node falls back to CPU, inserting GPU→CPU→GPU transfers
  mid-graph, and the only way to observe it is parsing verbose console output.
  We confirmed that mechanism works — with `logSeverityLevel: 0` the probe
  captured `All nodes placed on [WebGpuExecutionProvider]. Number of nodes: 3`
  — so the risk is manageable, but only by asserting on a log line in CI.
- The native `/webgpu` entry point fetches a 25.7 MB WASM artefact.
- ORT has no WebGPU *texture* tensor — input and output are `GPUBuffer` only —
  so AetherVSR would need conversion passes on both ends, which is exactly the
  work ADR-0012 already does in one direction.
- ORT can adopt an externally created `GPUDevice` via `{name:'webgpu', device}`,
  so same-device interop is possible; that is the enabling fact for the
  deferred experiment.
- `js/web` still labels WebGPU experimental at 1.29.0.

Against that, ORT's kernels are mature and ours are naive, so it may well win
once transfers are removed. The decision is deferred on evidence, not
preference.

**Consequences.** Milestone 3 owns the GPU-IO-bound ORT measurement. Until it
is taken, no performance claim may be made in either direction.

---

## ADR-0016 — Quality is evaluated against a generated reference, not a compressed one

**Status:** accepted (Milestone 2)

**Decision.** The quality harness generates a 2560x1440 reference, downsamples
it by an exact integer 2x box filter implemented in our own code, upscales the
result with the real GPU scaler, and compares against the reference. PSNR over
luma and RGB, plus mean SSIM over 8x8 luma windows, with a nearest-neighbour
control.

**Why.** Using a compressed source would measure compression-artefact
restoration and super-resolution simultaneously with no way to separate them,
and those are different problems on the roadmap. Delegating the downsample to
`drawImage` would make the reference browser-defined. Averaging is done in the
encoded domain to match ADR-0006, so the metric does not penalise the scaler
for a colour-space convention the pipeline deliberately chose.

**Consequences.** Absolute values are low and not comparable with the SR
literature, which uses natural-image sets; these numbers rank scalers on one
fixed adversarial image. The SSIM window is the fast 8x8 uniform
approximation, not the canonical 11x11 Gaussian.

The harness immediately produced a useful warning: nearest neighbour scores
*higher* SSIM than bilinear (0.880 vs 0.862) while scoring lower PSNR, because
SSIM rewards preserved local variance and blurring suppresses variance more
than blocking does. Ranking on SSIM alone would have been wrong. Any future
model must be judged on both, and on temporal behaviour that neither captures.

## ADR-0017 — Subgroup matrix rejected: measured slower, and unshippable anyway

**Status:** accepted (Milestone 3)

**Context.** Milestone 2 named `chromium-experimental-subgroup-matrix` as the
one lever that could plausibly deliver a large multiple, because this adapter
advertises it. Milestone 3 had to either use it or say why not.

It was implemented as an implicit GEMM — not im2col, whose 9x activation
expansion is 265 MB in f16 at C16/720p, past the 128 MiB default storage
binding limit. K was ordered `tap * inChannels + ic` so each 8-wide slice stays
inside one 3x3 tap, making the staged operand eight channels of eight adjacent
pixels. It is numerically correct: <=4.5e-7 f32, 2.8e-3 f16.

**Decision.** Do not pursue it. Keep the portable `blocked` kernel.

**Why.** Measured at 1280x720 C16, it runs at 8.664 ms against the portable
kernel's 1.067 ms — **8.1x slower**, mean of four repeats each — and
`rowsPerGroup` from 1 to 16 barely moves it. Dawn on Metal exposes exactly two configurations, both 8x8x8. That
tile does 512 MACs against 64 staged activations, 8 MACs per staged value,
plus two barriers per K-slice; the `blocked` kernel reaches 64 per staged
value. The matrix path is bound on staging before its arithmetic units matter.

Three further reasons hold regardless of speed:

1. It requires `--enable-unsafe-webgpu`. Measured: the feature is absent from
   `adapter.features` without it, so no shipped page can reach it.
2. It is absent from Chrome 152's release notes and marked Experimental in Dawn.
3. The W3C draft has already renamed the extension to `subgroup_matrix` and
   replaced the boolean `col_major` argument with a template parameter, so the
   shader is written against a moving target.

**What this is not.** It is not evidence that the M5's matrix hardware is slow.
It is evidence that an 8x8 tile cannot amortise a gather for a 3x3 convolution
at 16 channels. A much wider layer could plausibly change the answer, and the
experiment is retained so it can be re-run.

## ADR-0018 — ORT Web is not a candidate until it accepts a caller's GPUDevice

**Status:** accepted (Milestone 3), supersedes the deferral in ADR-0015

**Context.** ADR-0015 deferred the runtime choice pending a fair comparison:
same workload, GPU-resident input and output, completion-fenced. Milestone 2's
figure was not comparable because a 59 MB tensor was re-uploaded every run.

**Decision.** Milestone 4 uses hand-written WGSL. ORT Web is not a candidate
for the video path, and this does not rest on a timing comparison.

**Why.** The fair comparison could not be constructed. Sharing AetherVSR's
`GPUDevice` with ORT's native WebGPU EP — the documented mechanism, and the
prerequisite for `Tensor.fromGpuBuffer` on a buffer we own — fails at session
creation with `Failed to wait for the operation:3`. Four configurations were
tried: with `env.webgpu.adapter` supplied alongside the device; with the plain
and `.jsep` artefacts, neither of which exports the `webgpuInit` the entry
point needs; and with `.jspi`, whose native stack switching is exactly the
mechanism a blocking Dawn future wait would require.

The consequence is architectural rather than numerical. A runtime that cannot
accept our device cannot consume a decoded video frame without a round trip
through host memory — the one thing this project's data flow exists to avoid.
Even a hypothetically faster ORT would have to pay that crossing every frame.

For the record, and **not as a ranking**, the scope-mismatched figures are ORT
20.1 ms (CPU input, GPU output, fenced, all nodes on WebGPU) against 1.067 ms
for our kernel. The 59 MB upload inside ORT's figure measures 8.3 ms standalone
and is reported alongside, never subtracted. Crediting it back entirely still
leaves ORT around 11x slower.

**Revisit when** ORT Web supports an externally supplied device on the native
WebGPU EP. The probe and both wasm variants are retained so this is a re-run,
not a rewrite.

## ADR-0019 — Warm-up is measured in time, not iterations

**Status:** accepted (Milestone 3)

**Context.** The feasibility map produced an impossible row: C4 at 854x480
measured faster than C4 at 640x360 with 1.8x the pixels. Repeating each cell
four times showed only the first pass of a session was wrong, and only for
small workloads — 0.135 ms against a steady 0.031 ms, a 4.3x error.

Eight warm-up dispatches of a 0.03 ms kernel is 0.25 ms of work, nowhere near
enough to bring the GPU off its idle clock. Milestone 2 never saw this because
every kernel it measured took milliseconds and ramped the clock itself.

**Decision.** Warm-up runs until at least 60 ms of wall time has elapsed,
batched 16 dispatches per fence and bounded against a hung case. Fixed
iteration counts are not used.

**Why not tell callers to discard a pass.** That leaves the trap in place for
whoever forgets, and the failure is silent: a plausible number, in the right
units, wrong by 4x. After the fix the first measurement of a session is already
correct, with <=1% spread across four repeats.

**Consequence.** Milestone 2's published figures stand: they are all
millisecond-scale workloads in the regime where both strategies agree, and the
re-measured baseline reproduces them within run-to-run variance.

No sub-millisecond figure *measured through the convolution harness* was
published before this fix. BENCHMARKS.md does publish sub-millisecond ingest
figures (0.357-0.510 ms), and those are unaffected: `IngestBench` warms up
through four seconds of continuous rVFC-driven rendering, so it was never in
the regime this defect occupied.

## ADR-0020 — The portable floor is the target; raised limits are not required

**Status:** accepted (Milestone 3)

**Context.** 11 of 72 spatial configurations were rejected by the WebGPU
guaranteed `maxComputeWorkgroupStorageSize` of 16 384 bytes, while Dawn reported
this adapter would allow 32 768 through `requiredLimits`.

**Decision.** `acquireGpu` may raise limits opportunistically — clamped to what
the adapter reports, and falling back to a plain request if that is refused —
but no AetherVSR configuration may depend on a raised limit.

**Why.** The experiment came back negative, which settles it. With 32 768 bytes
granted, the best newly-legal configuration is 1.168 ms against 1.067 ms for a
configuration that fits inside the floor; every larger tile is slower. The rule
is now machine-checked: every benchmark result carries `sharedBytes` and a
`requiresRaisedLimit` flag, so a non-portable configuration cannot be published
as portable by oversight. The
guaranteed floor is not binding for this kernel, so there is no speed-versus-
portability trade to make here. That is worth recording precisely because the
result could have gone the other way and forced one.

## ADR-0021 — Throughput denominators come from dispatch geometry, not block size

**Status:** accepted (Milestone 3)

**Context.** `convMacCount` is the denominator of every GMAC/s figure this
project publishes, so an error in it mis-scales a whole milestone quietly
instead of failing. Through Milestone 2 it inferred the issued extent from
`blockX` and used height unrounded. That was correct for the only kernel that
existed: the naive one returns *before* accumulating when an invocation falls
outside the image.

Every Milestone 3 kernel guards at the **store** and accumulates its whole
spatial block regardless, so an overhanging dispatch really does pay for the
overhang — and once `blockY` existed, vertically too, which the function had no
parameter for.

**Decision.** `convMacCount` takes the issued width and height directly. The
caller derives them from the same numbers it passes to `dispatchWorkgroups`,
per variant, because how far a dispatch overhangs is a property of the kernel's
guard placement and not something a shared helper can infer.

**Why not just add a `blockY` argument.** Because the next kernel would break it
again. The failure mode is a helper quietly guessing at something only the call
site knows; adding one more guessed parameter preserves that. Guard placement
is a per-kernel decision, so extent must be a per-kernel calculation.

**Impact on published figures.** Small, and in the conservative direction —
understating issued work understates throughput. Only cells whose dimensions do
not divide the dispatch grid were affected: 854x480 C16 moved 1894 -> 1917
GMAC/s, 640x360 C12 moved 1724 -> 1757. The headline 1280x720 configuration
divides exactly and did not move. Every *throughput* figure in the Milestone 3
section was re-measured after the fix regardless of whether it was expected to
change; millisecond timings were unaffected either way, since only the
denominator moved.

## ADR-0022 — First neural upscaler: low-resolution convolution trunk with a resize-convolution head, C16, weights trained in-house on a CC0 corpus

**Status:** accepted (Milestone 4), superseding a rejected first draft

**Context.** Milestone 4 needs one real 2x neural upscaler that beats Catmull-Rom
(19.45 dB / 0.925 SSIM) inside a <=8 ms whole-stage budget on a base Apple M5.
A research pass covered SPAN/SPAN-F, RLFN/RLFN-S/RLFN-NTIRE, RFDN/E-RFDN,
IMDN/IMDN-RTC, EFDN, FMEN, RepRFN, ESPCN and FSRCNN, with code and weight
licences established separately from primary sources.

An earlier version of this ADR selected a canonical ESPCN with a pixel-shuffle
head. An independent gate rejected it and found four errors of fact in the
reasoning. This version records the corrections rather than quietly dropping
them, because three of them cut against conclusions that felt obvious.

### What the candidates actually cost

Costed against our measured per-width throughput at 1280x720. Figures marked
*checked* were independently re-derived from the published source by the gate.

| Architecture | MACs | Est. convolution time |
| --- | ---: | ---: |
| FSRCNN-S (d32,s5,m1) | 9.88 GMAC *checked* | ~5.0 ms |
| IMDN-RTC (nf12) | 18.34 GMAC *checked* | ~9.2 ms |
| ESPCN (64/32 canonical) | 24.60 GMAC *checked* | ~12.4 ms |
| SPAN-F (C32) | ~125.2 GMAC *checked* | ~63 ms |
| RLFN (C52) | ~461.7 GMAC *checked, corrected from 511.4* | ~230 ms |
| SPAN (C48) | 377.6 GMAC *checked* | ~190 ms |

The modern efficiency-challenge architectures are out of budget by a wide
margin - though "an order of magnitude" was an overstatement for IMDN-RTC
(9.2 ms against an 8 ms budget) and SPAN-F, and that wording has been removed.
Only the 2016-era architectures are in range.

### Corrections to the first draft

1. **The FSRCNN-S width argument was wrong.** The draft rejected it partly
   because its `s=5` bottleneck sits where our kernels are least efficient.
   That path is 5.1% of its MACs; the C32->RGB deconvolution is 72.5%. A
   packed `s=8` variant costs 10.38 GMAC and about 5.34 ms - competitive, not
   disqualified. The width argument is withdrawn.
2. **Pixel shuffle is not checkerboard-safe.** Aitken et al. (arXiv:1707.02937)
   documents checkerboard artefacts from ordinary sub-pixel convolution with
   independently initialised phase kernels. Conversely a stride-2 transposed
   convolution *is* exactly shift-equivariant away from boundaries. Both of the
   draft's temporal arguments pointed the wrong way and are withdrawn.
3. **A transposed convolution is not a mandatory primitive for FSRCNN-S.** Its
   9x9 stride-2 deconvolution polyphase-decomposes into four low-resolution
   convolutions plus an interleave - which is sub-pixel convolution, so that
   route does not sidestep the issue below.
4. **The licence inventory was too categorical.** Saying no candidate has usable
   x2 weights overstated it: ByteDance/RLFN ships `rlfn_s_x2.pth` and
   `rlfn_x2.pth` under an unqualified root Apache-2.0, and fannymonori/TF-ESPCN
   ships `ESPCN_x2.pb` likewise. Whether a repository-wide grant reaches the
   binary weights is a question for counsel, not something to record as NOT
   FOUND. Neither rescues this target - RLFN is hundreds of GMAC and TF-ESPCN is
   a luminance 64/32 model - but the inventory must be accurate.
   Flickr2K is likewise "no dataset-wide grant found; rights remain
   image-specific", not "academic only" as the draft claimed. DIV2K and LSDIR
   *do* state academic research use only.

### The blocking constraint: sub-pixel convolution is patented and active

**EP3259916B1, "Visual processing using sub-pixel convolutions"** (Magic Pony
Technology Ltd; inventors Wang, Bishop, Shi, Caballero, Aitken, Totz; priority
2015-02-19; granted 2021-05-26). Verified from the primary record: status
**Active**, anticipated expiry **2036-02-19**. Its inventors are the ESPCN
authors and its subject is the low-resolution feature trunk followed by an
`r^2 * C` convolution and periodic shuffle.

**This ADR does not assert that any particular implementation infringes.**
Patent scope is claim-, jurisdiction- and use-specific and that determination
needs counsel, not an engineering document. What is recorded here is that an
active patent exists squarely over the technique the draft selected, that no
licence or non-assertion pledge was found, and that freedom to operate is
therefore *unresolved*.

Given that it is unresolved and that an alternative exists at comparable cost,
the engineering decision is to choose the alternative. That is risk reduction,
not a legal conclusion, and it is cheap.

**The alternative is not clearance either.** A resize-convolution is
mathematically expressible as a constrained polyphase sub-pixel convolution -
one whose phase kernels are tied rather than independent. Not materialising an
`r^2 * C` tensor and not performing a shuffle reduces literal-match risk; it
does not establish freedom to operate. FTO remains **unresolved** and is a
matter for counsel before any distribution.

**Decision.** Implement:

```
ingest texture
  -> 5x5 texture-native stem, 3 -> C16, tanh
  -> D x (3x3 convolution, C16 -> C16, tanh)
  -> resize-convolution head: nearest 2x upsample + 3x3 convolution, C16 -> RGB
  -> output texture
```

The trunk is a plain stack of low-resolution 3x3 convolutions - the generic
part of the ESPCN/FSRCNN lineage, and not what the patent is about. The
reconstruction is a **resize-convolution**: a fixed nearest-neighbour upsample
followed by an ordinary convolution, evaluated directly at high resolution.
There is no `r^2 * C` convolution and no periodic shuffle anywhere in the graph.
Because the four phases share one kernel rather than being independently
parameterised, the head is also structurally immune to the uneven-overlap and
independent-phase mechanisms that produce checkerboard artefacts, and it is
exactly shift-equivariant away from boundaries.

**Why this head rather than a direct transposed convolution**, which the gate
recommended:

| | resize-convolution | direct transposed conv (FSRCNN-S) |
| --- | --- | --- |
| Head cost | **1.726 ms, measured** | ~3.6 ms, estimated, unmeasured |
| New operators | none beyond a 5x5 stem | 5x5 stem, 1x1 conv, PReLU, HR-gather deconvolution |
| Largest unknown | the 5x5 stem | the deconvolution, estimated at 3-5 days with checkerboard risk |
| Whole stage, D=2 | **5.04 ms** | ~5.59 ms |

Everything in the left column except the stem is measured. The right column's
dominant term is not.

**Budget.** Every term marked measured or estimated.

| Term | Cost | Basis |
| --- | ---: | --- |
| ingest | 0.245 ms | **measured**, in-pipeline p50 |
| stem 5x5 3->16 | 0.86 ms | *estimated* — 3x3 measures 0.310 ms and 5x5 is 25/9 the taps. Scaled linearly: an earlier draft applied a sublinear discount that nothing justified |
| body, per layer | 1.102 ms | **measured**, mean incremental over 1..8 chained layers |
| resize-conv head | 1.726 ms | **measured**, 1.593 GMAC at 923 GMAC/s |

| Body depth | Estimated whole stage |
| ---: | ---: |
| 2 | 5.04 ms |
| 3 | 6.14 ms |

**These figures are optimistic and the milestone does not pass on them.** They
are assembled from tight-loop measurements, and the clock-ramp experiment shows
2.48 ms of tight-loop work taking 4.817 ms in-pipeline at 60 fps. A stage of
this size does more work per frame than that experiment's smallest points, so it
should sit closer to tight-loop efficiency — but "should" is exactly why the
whole stage gets its own instrumented measurement on real video before any depth
is fixed, and why the phrase "genuine headroom" has been removed.

**Training corpus, decided before training rather than recorded after.**
Wikimedia Commons `Category:CC-Zero`, fetched by a reproducible script that
records per-image URL, dimensions, SHA-256 and the licence string returned by
the API, and that **rejects any image whose licence metadata is not CC0**. CC0
is an explicit public-domain dedication permitting commercial use and
derivatives. DIV2K, LSDIR and Flickr2K are not used. No images are committed;
the manifest is.

**Training-to-inference contract, frozen before training.** The weights are only
correct with respect to a specific degradation and colour handling, so these are
fixed and recorded with the model: HR sourced at native resolution, LR produced
by a box downsample by exactly 2 matching `boxDownsample2` in `quality.ts`; RGB
in [0,1], no dataset mean subtracted (the stem's mean/scale are identity);
zero padding sized to keep every layer same-resolution, which is
`padding=2` for the 5x5 stem and `padding=1` for the 3x3 body and head - not a
uniform 1, as an earlier draft said; tanh
activations in the trunk; linear head with the output clamped to [0,1] at write
time, matching the shader; channel order RGB throughout. A golden export
compares the PyTorch forward pass against the WebGPU graph on identical input
before any quality claim is made.

**Weight provenance.** Trained in-house from this architecture. No third-party
checkpoint is downloaded, converted or shipped, and the weights are never
described as upstream.

**Biggest risk.** A C16 trunk may not beat Catmull-Rom by a worthwhile margin.
That is a quality question no architecture reasoning settles, and Milestone 4
fails honestly if it comes out that way.

## ADR-0023 — Request `shader-f16` at device creation, unconditionally

**Status:** accepted (Milestone 4)

WebGPU devices expose only the features named when the device is created, and
device creation happens during startup — before any upscaler exists to ask for
anything. The neural stage silently ran in fp32 for its first production
measurement: correct output, 59 MB of activations instead of 29.5, 9.30 ms
instead of 5.48, 25 fps instead of 59.7, and no error at any layer. Those three
figures are from the session that exposed the bug, on the since-withdrawn
500-image model; they are kept because they are what the incident looked like,
and the shipped model's re-measured figure is 5.34 ms.

The harness now requests `shader-f16` as an optional feature at acquisition and
the shaders branch on whether it was granted. Optional features are free when
absent, so there is no cost to asking early, and the alternative — recreating
the device when a stage wants a feature — would invalidate every resource.

**Recorded because the failure mode is invisible.** A missing optional feature
is not an error; it is a slower correct answer. Any future stage wanting a
feature must add it to the acquisition list, not request it at use.

## ADR-0024 — Fall back on measured stage time, not on a frame-rate symptom

**Status:** accepted (Milestone 4)

`BudgetGuard` watches the median whole-stage GPU time over a 30-frame window and
switches to the baseline scaler above 10 ms, recovering below 7 after 3
consecutive qualifying evaluations.

Dropped frames were rejected as the trigger: they are caused by decode stalls,
compositor scheduling and background throttling as readily as by the upscaler,
so the signal does not identify what to fix. Median rather than mean, because
one 40 ms hitch must not trip a switch. Separated thresholds, because a single
one dithers when the stage sits near it.

Milestone 6 owns dynamic quality selection. This is deliberately a mechanism
with a forced-failure entry point, not a controller, and it is verified by
forcing an over-budget condition on live video and observing recovery.

## ADR-0025 — Publish per-pass timings only alongside a whole-stage span

**Status:** accepted (Milestone 4)

The neural stage writes one timestamp span covering ingest through presentation,
plus separate per-pass spans. The whole-stage figure is the published number;
per-pass figures are diagnostics and are never summed to produce it.

Summing passes was measurably wrong. Tight-loop per-pass measurements
underestimate in-pipeline cost by up to 4.2x on small passes, because a GPU that
has just idled through a frame interval runs at a lower clock. Encoding a pass
n times per frame drives its per-dispatch cost monotonically from 1.303 ms to
0.346 ms, and the *preceding* ingest pass speeds up too — which only a raised
clock explains. Two correct measurements added together produced a number that
described no real configuration.

## ADR-0026 — CC0 only, and publication is blocked pending counsel

**Status:** accepted (Milestone 4). The corpus decision stands. **The
publication block is closed — superseded by ADR-0027**, in which the project
owner accepted the residual risk. The block's reasoning is preserved below as
written, because it is why the question was asked at all.

### Corpus

ADR-0022 described the training corpus as CC0. It was not: `tools/fetch-corpus.py`
accepted `public domain` and `pd` alongside `cc0`, and five of the 500 images
carried Commons' generic "Public domain" tag. Independent review found this.

A bare public-domain tag is a claim about *some* jurisdiction — usually that a
term expired, sometimes that it is a government work — and it is not a
dedication the uploader made. It is not equivalent to CC0 and it is not
uniformly true worldwide.

The filter now accepts `cc0`, `cc0 1.0` and `cc-zero` only. The five files were
removed and **the model was retrained on the remaining 495**, rather than
restating the claim to match the corpus. Deleting a legal question is cheaper
than deferring one, and 1% of the corpus is not worth an ambiguity.

The retrained model is the one in the repository and every Milestone 4 figure
was re-measured against it. It is *worse* on the sub-pixel temporal metric than
the withdrawn model, and BENCHMARKS.md says so.

`tools/train.py` now streams and hashes every corpus file and compares it to the
committed manifest before sampling. Previously it checked only that a
`sha256` string was non-empty and the path existed, while the recorded
`corpusDigest` hashed the manifest's strings — so edited image bytes would have
trained a different model under an unchanged digest. The manifest is written by
the fetcher and committed; the trainer only reads it, so the comparison is
against an independent record rather than self-consistent.

### Publication

**AetherVSR Milestone 4 must not be distributed until counsel clears the
resize-convolution reconstruction head.**

ADR-0022 selected that head after rejecting sub-pixel convolution over
EP3259916B1, and stated plainly that the alternative "is not clearance either":
a resize convolution is mathematically expressible as a constrained polyphase
sub-pixel convolution, so not materialising an `r^2 * C` tensor reduces
literal-match risk without establishing freedom to operate. That ADR made
counsel a precondition of distribution.

No such review has happened. The engineering is complete and independently
reviewed; the legal precondition the project set for itself is not satisfied,
and the fact that milestones 0-3 are already public does not satisfy it, because
none of them shipped a model or a reconstruction head.

Concretely, publication is held on:

1. Whether the shipped resize-convolution head infringes EP3259916B1 or its
   family in any jurisdiction AetherVSR distributes to.
2. Whether distributing trained weights, as distinct from source, changes that
   answer.

Until both are answered, Milestone 4 stays on the local branch. Weakening this
gate by editing the wording of ADR-0022 would be exactly the "silently change
architecture" failure AGENTS.md prohibits; superseding it requires the answers
above, not a rewrite.

## ADR-0027 — Publication approved by the project owner on accepted risk

**Status:** accepted (Milestone 4). Supersedes the publication block in
ADR-0026, which is now **closed**. The corpus half of ADR-0026 stands unchanged.

ADR-0022 made counsel a precondition of distributing the resize-convolution
head, and ADR-0026 held Milestone 4 on the local branch because no such review
existed. The project owner has reviewed the question and authorised publication
on accepted risk.

**This is not a legal clearance and does not claim to be one.** No attorney
reviewed this repository. What changed is who is accepting the residual risk,
not what is known about it.

### The reasoning placed before the owner

- **The patented technique is not the one implemented.** EP3259916B1's claims
  are directed at sub-pixel convolution: producing an `r^2 * C` tensor and
  rearranging it by periodic shuffle. AetherVSR's head is a nearest-neighbour
  upsample followed by an ordinary 3x3 convolution. It never materialises an
  `r^2 * C` tensor and performs no shuffle.
- **Resize convolution has abundant independent prior art.** It is standard
  practice and was the explicit recommendation of Odena, Dumoulin and Olah,
  "Deconvolution and Checkerboard Artifacts" (Distill, 2016), among others. It
  did not originate with, and does not depend on, the patented method.
- **ADR-0022's own caution proves less than it appears to.** That a resize
  convolution is *expressible* as a constrained polyphase sub-pixel convolution
  is a statement about linear algebra: almost any upsampler can be written as a
  constrained linear operator. Expressibility is not a reading of the claims.
- **The rejected operator is quarantined.** `pixel-shuffle.wgsl.ts` lives under
  `src/bench/`, carries a header explaining why it was rejected, and is imported
  by nothing under `src/core/`. Machine-checked:
  `grep -rn "pixel-shuffle" --include=*.ts src/core/ | wc -l` returns 0.
- **Milestones 0-3 were already public.** What publication adds is the head and
  12 kB of weights.

### What would reopen this

New information about the claims, a change to the reconstruction head that
reintroduces an `r^2 * C` tensor or a periodic shuffle, or any commercial
distribution — which is a materially different posture from an open research
repository and was not what was weighed here.

### Process note

The block was raised by an independent review agent, not by the engineer who
wrote the code, and it stopped a design that had already been implemented and
benchmarked. It was worth the delay: the first architecture was withdrawn on the
strength of it. Recording that here so the cost of the gate is not mistaken for
wasted effort the next time one fires.

## ADR-0028 — Split by source image; validation selects, test judges once

**Status:** accepted (Milestone 4.5)

Milestone 4 pooled patches from every photograph and split the pool, so 46.7% of
the corpus contributed to both training and validation and **100% of validation
patches came from a photograph the model had trained on**. That score measured
memorisation, so it could not answer the only question worth asking of it.

The split unit is now the source image, assigned deterministically from the
SHA-256 of file content plus a recorded salt. Content hash rather than filename
or URL, because the fetcher assigns filenames and one photograph can have
several URLs. Hashing rather than seeding an RNG over a list, because the
assignment must not depend on directory order and must not reshuffle when the
corpus grows.

Roles are fixed and asymmetric:

- **train** — gradient updates only.
- **val** — checkpoint selection and hyperparameter development.
- **test** — evaluated once, on a frozen model, and never read during training.

`train.py` does not load the test split at all. Not to log a number, not to draw
a curve. A test metric that is available every epoch will eventually be looked
at, and the cheapest way to guarantee it cannot influence a decision is for the
data not to be in the process.

`test/dataset-split.test.ts` enforces disjointness in CI against the committed
manifest, with no Python and no GPU.

## ADR-0029 — Evaluate against the filter the product ships

**Status:** accepted (Milestone 4.5)

Every quality claim in this project is *neural minus Catmull-Rom*, which makes
the baseline's identity load-bearing. The Python evaluators used
`F.interpolate(mode="bicubic")` — Keys cubic with a = -0.75 — while the shipped
WGSL is B = 0, C = 0.5, i.e. Keys a = -0.5. Different filters, and the wrong one
was sharper, so the neural stage was being scored against an opponent that
exists nowhere in the product.

Baselines must now be verified against the production implementation, not
against a library function that resembles it. The check is a fixed fixture
pushed through the real `BaselineScaler` render pass and read back:

    python a = -0.5     max 0.557/255    below one 8-bit quantisation step
    pytorch a = -0.75   max 17.588/255   thirty times that

Recorded in `results/catmull-rom-parity.json` with its tolerance and scope. The
lesson generalises past this one kernel: a reference implementation is only a
reference if something checks it against the thing being shipped.

## ADR-0030 — Ship the realistic-degradation model for compressed input

**Status:** accepted (Milestone 4.5)

Two models, identical architecture, differing only in training degradation. The
realistic-trained model is selected as the default because the product's input
is compressed web video.

It is a targeted trade and not a uniform improvement:

    better   compressed stills   +0.397 vs +0.180 dB at typical CRF
             ground-truth video  +0.131 vs -0.141 dB overall
             synthetic reference 21.065 vs 20.542
             sub-pixel temporal  2.938x vs 3.067x Catmull-Rom
    worse    clean box stills    +0.091 vs +0.416 dB
             video natural       -0.174 vs -0.153
             video motion        -0.132 vs -0.068

Anyone upscaling pristine imagery should prefer the clean-trained weights, which
remain reproducible from `tools/run-seeds.sh`. The seed was chosen on validation
alone, before any test or video figure was consulted, and every seed's results
are published rather than only the selected one.

The finding behind this decision matters more than the decision: the network was
never too small to help on compressed video. It had been trained to invert a
degradation that compressed video does not have. Architecture was held fixed
precisely so that could be established.

## ADR-0031 — Public claims are bounded by the captured-footage corpus

**Status:** accepted (Milestone 5)

The README previously read "Real-time, local, GPU-accelerated video
super-resolution for web video" and, three milestones out of date, "Milestone 2
complete — there is still no AI inference in this codebase". Both were wrong in
opposite directions: the headline over-claimed, the status under-claimed.

Public claims now state the measured scope and its limits together:

- **+0.72 dB at high quality, +0.36 dB at typical**, 9/10 captured clips each,
  against the Catmull-Rom the product actually ships.
- **No improvement at poor quality**, and no faces in the corpus at all.
- **1.24x Catmull-Rom's motion-compensated residual** on 10/10 clips.

The rule this fixes into the project: a claim about "web video" requires
captured-footage evidence. Synthetic proxies and unseen photographs are not
sufficient, because Milestone 4.5 demonstrated a synthetic benchmark ranking two
models in the opposite order from real photographs, and Milestone 5 found a
single clip with a rendered HUD supplying 77-99% of its own measured advantage.

## ADR-0032 — Metric sanity controls run before model numbers

**Status:** accepted (Milestone 5)

Milestone 4.5 published VMAF for 25 cells and then discovered it ranked
nearest-neighbour above Lanczos in 20 of them. Milestone 5 ran the control
first, on captured footage, and VMAF failed it on 10 of 10 clips — the same
pathology on an entirely different corpus.

Every metric used for a conclusion must first reproduce an ordering that is not
in dispute: on real footage, `nearest <= bilinear <= Catmull-Rom <= Lanczos`.

    PSNR-Y   10/10   usable, carries the conclusions
    SSIM      6/10   diagnostic only
    VMAF      0/10   diagnostic only

A metric that cannot rank four conventional filters correctly cannot be trusted
to rank a neural model against them, however sophisticated it is.

The same principle applied to the temporal metric and caught two defects: an
inverted warp that made "motion compensation" score worse than no compensation
at all, and an unstated sharpness confound — blurring the reference, which
cannot change the footage's real stability, buys a 14.9% residual reduction.
Without that control the 24.3% neural excess would have been reported as pure
temporal instability.
