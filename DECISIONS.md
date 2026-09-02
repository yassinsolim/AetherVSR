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
measured 0.11–0.14 ms in every run here while the GPU pass ranged from 0.50 to
4.70 ms depending on filter and codec. Publishing the CPU figure as "processing
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
