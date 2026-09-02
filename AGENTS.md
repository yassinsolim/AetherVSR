# AGENTS.md — engineering rules for AetherVSR

These rules bind every contributor, human or agent. They are not aspirations;
a change that violates one of them is not ready to merge. When a rule and a
deadline conflict, the rule wins and the scope shrinks.

## 1. Correctness before feature count

A small pipeline that is provably right beats a large one that is probably
right. Do not add a second capability while the first is unverified. "It should
work" is not a status; "I ran it and here is the output" is.

## 2. Measured performance, never guessed performance

- Never state a number you did not observe on a machine you can name.
- Never extrapolate a figure from another GPU, another resolution, or another
  browser and present it as an AetherVSR result. Cite the source and mark it
  as somebody else's measurement.
- Prefer GPU timestamps to main-thread wall-clock timing. Main-thread time
  measures the import call, command recording and submit; it barely moves when
  the queued GPU work changes by an order of magnitude. Compare the
  `cpu per frame` and `gpu upscale` columns in `BENCHMARKS.md` before
  publishing either as "processing time".
- State the scope of every timing: what the measurement brackets, what it
  excludes, and over how many samples and how long a window it aggregates. A
  number whose boundaries are unstated is not a measurement.
- If a metric is unavailable, print "not measured". Never print `0`.

## 3. Never report a benchmark that was not actually measured

Numbers in `BENCHMARKS.md` come from a run on the recorded hardware, with the
recorded browser build, using the committed test clips. If a run could not be
performed, the table says so explicitly. Deleting a row is acceptable;
inventing one is misconduct.

Measurements taken against an apparatus that has since changed — a superseded
test clip, a fixed harness bug — may be kept, but only in a section explicitly
marked as superseded and never in a results table. Recording why a number was
discarded is worth more than the number was.

## 4. No CPU readbacks in the per-frame hot path

Forbidden inside the frame callback:

- `getImageData()`, `readPixels`-style reads, `GPUBuffer.mapAsync` of pixel
  data, `toDataURL`, `createImageBitmap` of a frame.
- Any `await` that lets the task end mid-frame. A `GPUExternalTexture` imported
  from a video element expires in an automatic-expiry task shortly after
  import: import, bind, encode and submit within one task.

Reading back a few bytes of GPU *timestamps* is not a pixel readback and is
allowed, provided it never blocks the frame.

## 5. No per-frame GPU-resource allocation that a preallocated resource could serve

This rule is about **GPU resources**, which are expensive and which we control.
Pipelines, bind group layouts, shader modules, samplers, uniform buffers and
intermediate textures are created in a configuration step, never in the frame
loop.

The per-frame GPU allocations WebGPU mandates, and which are therefore allowed:
a command encoder and its command buffer, the render pass encoder, the
swap-chain texture and view, the external-texture handle, and a bind group on
the external-texture path (an external texture expires each task).

Small JavaScript objects — the frame tick, the encode context, the timing
handle, descriptor literals, the `submit()` array, the timestamp `mapAsync`
promise — are allocated per frame and are **not** covered by this rule. They
are short-lived nursery allocations, they have never appeared in a measurement,
and pooling them would trade real readability for imagined performance. If a
profile ever shows GC pressure in the frame loop, revisit with the profile in
hand and record an ADR. Do not pre-optimise this on principle.

## 6. Maintain the modular boundaries

Four stages, each replaceable without touching the others:

```
acquisition  ->  import  ->  upscale  ->  presentation
```

- Acquisition knows about `HTMLVideoElement` and frame callbacks. It owns no
  GPU resources.
- The upscale stage implements `Upscaler`. It knows nothing about video
  elements, canvases or the page.
- Presentation owns the canvas and the swap chain. It knows nothing about how
  pixels were produced.
- Metrics observe; they never steer the pipeline.

A neural backend must be able to arrive as one new `Upscaler` implementation.
If a change would require editing acquisition or presentation to add one, the
change is wrong.

## 7. Target 60 FPS where practical, and know which stage is the ceiling

60 Hz gives a 16.67 ms frame interval for decode, upscale, composite and
present combined. The upscale stage should aim for a small fraction of that,
not all of it. Before claiming the upscaler is the bottleneck, prove it: run
the cheapest upscaler and see whether the frame rate moves. It usually does
not.

## 8. Document the benchmark environment

Every benchmark records: machine, OS version, browser and exact build, display
refresh rate, clip (codec, resolution, frame rate), import path, upscaler,
window state, and whether GPU timestamps were available. A number without an
environment is not reproducible and therefore is not evidence.

## 9. Do not silently change architecture

Any change to a stage boundary, a data flow, or a core dependency needs an ADR
in `DECISIONS.md` in the same commit. Reversing an existing ADR requires a new
ADR that supersedes it, not an edit to the old one.

## 10. Keep commits small and working

Every commit typechecks, lints and passes tests. No commit leaves the harness
unable to start. No commit mixes a refactor with a behaviour change.

## 11. Third-party code and licensing

AetherVSR is independently engineered. Read other projects for ideas; do not
copy their source unless their licence permits it and the notice is preserved.
A repository with no `LICENSE` file grants no rights — study it, cite it, do
not copy it. Record every licence verdict in `DECISIONS.md` or `README.md`
before relying on it.

## 12. No unfinished work behind confident language

No stubs, placeholder implementations, silent fallbacks or `TODO: implement`
presented as complete. If a prerequisite is missing, say precisely what is
missing and finish everything that does not depend on it.
