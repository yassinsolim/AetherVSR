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
