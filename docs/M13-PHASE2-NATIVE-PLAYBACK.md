# M13 Phase 2 - Native Local Playback

**NATIVE LOCAL PLAYBACK PARTIAL.** The supported native acquisition, color,
unchanged-F inference and presentation path works and passes decoded-frame
parity and lifecycle checks. The first binding baseline30 trial fails cadence,
software-age and useful-frame warmup gates. Its window was occluded. Foreground
performance is not qualified; causation is not established. Neural30, all short60
trials and the ten-minute soak remain NOT_RUN. No favorable rerun was performed.

Protocol: [prospective plan](M13-PHASE2-PLAN.md). Reproducible evidence:
[study index](../results/m13-phase2-playback.json). Validation and publication
provenance: [verification index](../results/m13-phase2-verification.json).

## 1. Starting State

Started from clean main/origin
`e381bd9366803fe91c048a97bbe5088844627f10`, exact CI35781116341 successful in
gate, fusion and native-metal. Fresh baseline: 17 checks, 1501 Vitest passes
with three existing skips, 612 Python, 80 opt-in fixture and 71 desktop tests,
18 physical native tests, builds and both historical evidence replays.
Baseline Git-object tree: 55,328,779 bytes; unchanged cap: 58,720,256 bytes.

M11 DESKTOP PLAYER MVP READY, M12 CROSS-VENDOR DESKTOP PARTIAL, M12.1
CROSS-VENDOR COMPLETION PARTIAL, the accepted checkpoint repair, and the
closed browser acquisition studies remain unchanged. NVIDIA/AMD/Intel remain
unavailable, not qualified by this Apple-native work.

## 2. Frozen Phase 1 And 1.5

Phase 1 remains METAL INFERENCE QUALIFIED at
`5b470064aea3cfee4ca95e84696ef8a359478fc9`, CI35767539893. Its unfused
720p GPU p50 measurements, 83.980375 ms f32 and 65.852229 ms f16, are isolated
60-sample measurements after ten warmups, not playback.

Phase 1.5 remains METAL REALTIME BACKEND QUALIFIED at the starting commit.
Its two fixed F runs measured GPU p50/p95 3.921708/4.458015 ms and
3.932604/4.451823 ms, 60 samples each after ten warmups. Those short isolated
windows do not qualify video, thermal endurance or this integrated path.
Original native package, generators, packing, model, golden data, reports and
evidence are byte-preserved. No new convolution candidate or search was added.

## 3. AVFoundation API Research

Current Apple documentation and installed SDK declarations were checked for
AVPlayer, AVPlayerVideoOutput, AVVideoOutputSpecification, current sample and
pixel-buffer APIs, Core Video Metal texture mapping and lifetime, image-buffer
attachments, MTKView, CAMetalLayer and CAMetalDisplayLink. Official links and
availability are retained in the prospective plan. Implementation is independent;
no Apple example source or external unlicensed code was copied.

Local environment: Apple M5, Mac17,2, 24 GiB, macOS26.6.2 build25G83;
Xcode27.0 build27A266a, SDK27.0, Swift6.4. Runtime MSL compilation is used;
the optional offline Metal Toolchain was not installed. No browser participates
in native playback. Hosted CPU/build checks use macOS26 with Xcode26.6; they
are not physical-GPU qualification.

## 4. Selected Video-Output API

Use AVPlayerVideoOutput attached to AVQueuePlayer, its current
`sample(forHostTime:)` API, and AVVideoOutputSpecification.defaultOutputSettings.
The player requires macOS26; there is no deprecated or silent fallback.
AVPlayerItemVideoOutput was researched but not selected.

A sample supplies tagged CVReadOnlyPixelBuffer content, an item-timebase PTS,
and active configuration. The configuration's weak sourcePlayerItem is promoted
to a strong frame owner before use. `withUnsafeBuffer` bridges the existing
buffer; it does not introduce a CPU pixel copy. Only one video buffer and the
qualified identity transform are accepted. An unavailable sample is not counted
as a decoded or presented frame.

## 5. Native Player Architecture

The separate top-level Swift package compiles the unchanged original AetherMetal
sources plus the command-encoding adapter. Acquisition/authority is VideoSource;
Core Video mapping and GPU conversion are NativeVideo; F is FrameNetwork;
native presentation and bounded scheduling are PlaybackController. Offline
capture is isolated in PlaybackDiagnostics. ADR-0062 records this boundary.

AppKit supplies local MP4 selection, play/pause, seek, mute, volume, baseline/
neural selection and fullscreen controls. This is a local research player, not
an installed distribution or a replacement for the qualified Electron product.

```sh
swift run --package-path native -c release aether-player
```

Run from the repository root, where the pinned production model is located.
Use Open Video for the committed local720p H.264 files. Arbitrary codecs,
rotation, HDR, network sources and non720p playback are not qualified.

## 6. Core Video Bridge

One CVMetalTextureCache is created per renderer/device. BGRA maps to BGRA8Unorm;
NV12 maps luma to R8Unorm and chroma to RG8Unorm. Plane dimensions, format,
device and destination capacity are checked. The native decoded fixtures use
420v, 1280x720 luma and640x360 chroma. Mapping is shader-read only.

This is a no-CPU-pixel-copy hot path, not a proven system-wide zero-copy claim.
Internal decoder, IOSurface and driver behavior was not measured. Mapping failure
is explicit; no CPU conversion fallback or unrelated capture API is used.

## 7. Pixel Formats

Synthetic tests exercise BGRA, 420v and420f, including asymmetric rows,
channel ordering, padded strides and Left/Center chroma placement. Actual
AVFoundation qualification uses requested Metal-compatible420v. Other pixel
formats are rejected. Supporting BGRA/420f in the bridge does not establish
that the committed decoder produced those formats.

CVMetalTextureIsFlipped was true for the observed NV12 views. Initial blanket
rejection was corrected after the independent asymmetric-row oracle established
the raw texel-coordinate interpretation. No unconditional image flip is inferred
from that flag alone.

## 8. Color Metadata

Actual buffers explicitly report ITU_R_709_2 matrix, primaries and transfer,
video range and Left chroma placement. The bridge requires supported metadata;
it does not silently assume every video is BT.709. NV12 requires an explicit
supported matrix and chroma location. Supported RGB transfer handling is encoded
SDR, not linear-light neural input.

Malformed/non-full clean aperture, nonsquare or nonpositive aspect, unsupported
field layout, conflicting chroma attachments and unsupported color are rejected.
Some Metal-compatible IOSurface allocations discard malformed attachment types;
negative metadata tests use plain buffers and verify the attachment survives
before checking rejection. GPU mapping tests still use real Metal-compatible
buffers. HDR and display colorimetry are not qualified.

## 9. YUV/RGB Validation

420v uses `(Y-16)/219`, `(Cb-128)/224`, `(Cr-128)/224`; 420f uses255
denominators. The GPU uses BT.709 conversion, registered chroma interpolation,
clamping and encoded RGB. The CPU oracle independently derives coefficients
from luma weights in Double and interpolates retained chroma planes.

Synthetic CPU/GPU tolerance is1e-5, with zero nonfinite/failing elements.
All six actual decoded frames, each checked in f32 and f16, pass; maximum
observed ingest error is9.897857888852002e-8. Raw planes, stride metadata,
inputs, checkpoints and RGBA are hash-pinned and independently replayed by
the JavaScript checker. Shared labels or stored PASS strings are insufficient.

## 10. CVMetalTexture Lifetime

Immutable leases strongly retain the CVPixelBuffer, CVMetalTexture wrappers and
MTLTexture planes. DecodedFrame also owns the AVFoundation sample and originating
item. Completion handlers keep these owners alive through GPU use; mutable
slots remain on MainActor. A physical weak-lifetime test verifies release after
command completion and autorelease drainage. Actual owner deinit counters,
not merely slot booleans, appear in snapshots. Cache flushing occurs after drain.

## 11. GPU Ingest

One GPU pass writes preallocated float4 input, preserving F's expected encoded
RGB and zero padding lane. No CPU pixel lock, base-address copy, readback or
mid-frame await occurs in the production callback. CPU plane access and large
readbacks exist only in offline diagnostics/tests, outside measurement windows.

Ingest fusion was not performed. Separate ingest and network command buffers
are submitted on one queue without a CPU interstage wait. No per-frame pipeline,
buffer, intermediate texture or sampler creation was added.

## 12. Candidate-F Integration

The exact6,291-parameter C16D2 model is unchanged: file SHA256
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.
Stem/body/head kernels, C4 packing, tap-major weights, thread geometry, f32/f16
tolerances and graph semantics remain original F. The adapter's both-precision
golden stages match the original optimized engine exactly on identical input.
Explicit safe math and precise floating-point functions preserve that contract.

Playback enables F's existing float-output path plus preallocated GPU finite/
range audits of each hidden stage and final float output. Only a four-byte flag
is read after completion; poisoned-input tests prove it rejects invalid output.
These extra writes/audits are inside integrated processing timing and were not
in Phase1.5's isolated measurement. No kernel retuning followed playback results.

## 13. Native Presentation

The fixed2560x1440 private RGBA output is rendered as an aspect-fit fullscreen
triangle into a BGRA8Unorm MTKView/CAMetalLayer drawable, with BT.709 color space.
Drawable acquisition/presentation uses CAMetalDisplayLink updates; ownership
extends through completion. An opaque black layer immediately covers invalid
output without hiding the Metal view and stopping its scheduler.

Native command completion and drawable geometry are observed, not physical
scanout, display colorimetry or camera-verified visual quality. No screen-recording
or Accessibility permission was requested for verification.

## 14. Display Scheduler

CAMetalDisplayLink runs on the main common run loop, requesting60Hz on a display
reporting maximum120Hz and backing scale2. Sampling targets the supplied
presentation host timestamp. Timers are diagnostic deadlines/action delays,
not playback authority. Completion promotes only current-generation/item output.

The binding run reports windowVisible=true, windowKey=false and
occlusionVisible=false throughout recorded heartbeats. Some ready frames wait
about one second between GPU completion and presentation opportunity. These
observations are consistent with a scheduling/environment issue but do not prove
its cause. Foreground cadence was not measured. Synchronous JSONL logging and
heartbeat callbacks share MainActor; their cost was not isolated.

## 15. Frame Queue Policy

At most two configured frame slots hold processing, ready, current or presenting
work. Promote/retire precedes acquisition so occupied old output does not
artificially halve throughput. PTS is deduplicated within item/generation;
seek, replacement, loop and terminal changes invalidate old work. Item changes
are detected even when no new sample is available. Busy input is dropped,
not queued without bound. An in-use slot is not overwritten.

Raw evidence records acquired/submitted/completed/presented, duplicate/busy,
approximate PTS gaps and loop totals. These are not decoder-internal counts;
separate no-new and stale-drop totals are not retained. The checker validates
source identity, ordered PTS, unique sequence, generation attribution and age.

## 16. AVPlayer And Audio Authority

AVQueuePlayer/AVPlayerLooper retain media rate, item time, seeking, mute, volume
and looping authority. No custom audio pipeline or separately advancing media
clock exists. Lifecycle controls confirm AVPlayer state.

Both committed fixtures are video-only. Audible playback, speaker timing and
A/V synchronization are **not measured**. Controls succeeding is not audio
qualification. Complete AVFoundation asset loops, including boundary time, were
retained: durations6.177/4.177seconds, nominal average rates29.138315/57.450169.
Different FFmpeg duration/rate metadata was documented before binding; no trim,
replacement clip or relaxed29/58 floor was used.

## 17. Baseline 30 FPS

The only binding performance attempt is baseline30, unchanged acquisition,
ingest, scheduler and presentation with nearest2x compute replacing F.
Source `c20e066971a3c7c4c3f0d81bb561f624e485e3c8`, exact successful
CI35804039192. Frozen executable:1,275,616bytes, SHA256
`3bb2f9e49adc6d3fb6afb938091323e17141a16a3a0a8c58f31609339ebdf70f`.
Machine/toolchain are in section3; native app, no browser, GPU timestamps available.
Window state is explicitly occluded, not a valid foreground qualification claim.

| Metric | Observed | Gate |
|---|---:|---:|
| Full observation |60.033666750seconds|60 to60.5seconds|
| Useful completed presentation opportunities |110|unique source frames|
| Overall cadence |1.832305fps|>=29fps|
| Final20seconds |35frames,1.75fps|>=29fps|
| Useful pre-window frames |11|>=116 plus five elapsed seconds|
| Full-window loop transitions |10|all boundary time included|

**FAIL:** warmup useful-frame count, overall/final cadence and age. No row is
discarded, normalized by active time, or replaced with a favorable run. Runtime
completion PASS only means orderly process completion; it is not acceptance.

## 18. Neural 30 FPS

**NOT_RUN:** blocked by baseline30. There is no integrated neural30 cadence,
headroom or age qualification. Lifecycle and paused parity demonstrate execution
but cannot substitute for the registered5-second warmup plus60-second trial.

## 19. Baseline 60 FPS

All three registered baseline60 cases are **NOT_RUN**, blocked by the30 gate.
The B/N, N/B, B/N order was frozen before measurements. No full-loop60 cadence
measurement or comparison against the58fps floor exists.

## 20. Neural 60 FPS

All three registered neural60 cases are **NOT_RUN**. The60fixture passes paused
decoded-frame parity, not sustained60fps playback. Phase1.5's isolated GPU timing
is not imported as a replacement product result.

## 21. GPU Frame-Path Timing

Descriptive timings below are from the110 measured baseline frames on the M5
over the entire60.033666750-second failed, occluded run. No neural performance
is measured. The raw field `gpuNeuralMS` means baseline processing in this case.

| Completed-command scope |p50 ms|p95 ms|max ms|
|---|---:|---:|---:|
| Ingest |0.369104|3.043792|3.226208|
| Baseline nearest compute |0.515625|2.189831|3.000625|
| Presentation render |0.198583|2.961279|3.329375|
| Ingest start through compute end |1.021833|3.588592|3.776458|
| Per-frame sum of active GPU durations |3.532000|3.839542|4.159625|

Each interval derives from retained GPU start/end endpoints after completion.
The active sum excludes inter-command/display gaps; processing span includes
the ingest-to-compute gap. Medians are never added. Initialization, decode,
CPU logging, display wait and physical scanout are excluded from GPU scopes.
Preferred10/12ms active-path headroom passes, but cannot override cadence or age.
The invalid warmup and occlusion limit its interpretation as a benchmark.

## 22. Software Frame Age

Age is originating AVPlayerItem time at software presentation opportunity minus
that exact sample's PTS. Signed lead is retained; the checker recomputes it from
raw timestamps. All110 in-window observations are finite: p50 -16.842771ms,
p95 981.263103ms, maximum985.343416ms, minimum-31.525584ms.
P95 exceeds66.666667ms and maximum exceeds250ms: **FAIL**.

First20-to-last20 median growth is1.091709ms, below one30fps interval; the
five-second median trend test passes. Neither result conceals the failed tails.
Loop time is not removed. This is not end-to-end physical display latency.

## 23. Pause And Resume

The frozen17-check lifecycle proves pause stops additional neural submissions,
AVPlayer rate becomes zero, and resume advances unique output. Valid paused
stills may remain; they are not counted as new current-generation frames.
Play intent, rather than transient preroll state, drives the UI control.

## 24. Seek

Forward/backward, pause-during-seek and successive seeks pass. Request/generation
tokens prevent superseded completion from reviving stale state. A five-second
one-shot completion deadline fails explicitly. Continued display-driven sample
pulls permit AVFoundation seek preroll, but no seek-pending GPU work is presented.
The first post-seek sample is checked against the paused target before resuming.

Earlier developmental seek timeouts and the overly weak intermediate assertion
remain retained and superseded, not removed. Final checks require no terminal
error and the intended authority state.

## 25. Source Replacement

Replacement30->60 hides the old surface immediately, drains ownership and accepts
only the new item. A superseded asynchronous open cannot poison the current
source. Real output after replacement and a software terminal fault that prevents
revival pass. Forced hardware removal, decoder corruption and GPU crash recovery
were not exercised; a terminal context requires reopening the application.

## 26. Resize And Fullscreen

The initial frozen lifecycle established liveness but incorrectly accepted a
constant1960x1136 drawable. Review caught this before any timing. The corrected
external scheduler explicitly synchronizes drawable size from backing bounds.

Final raw frames show1960x1136 windowed,1240x656 smaller,2984x1754 fullscreen,
and restored1960x1136. Backing scale is2; network output stays2560x1440.
Independent replay checks these transitions and rejects fixed-size evidence.
Other monitors/backing scales, screenshots and physical presentation are not
qualified by this result.

## 27. Parity

For each clip, paused targets1/2/3seconds retain the same exact AVFoundation
pixel buffer for CPU-reference ingest/original F and GPU-ingest/integrated F.
Actual PTS30:0.979266667,1.979566667,2.9788; PTS60:0.996166667,1.995066667,
2.994366667. Both precisions pass every checkpoint, range/finite audit and
opaque RGBA comparison at unchanged tolerances.

Across these samples, f32 hidden maximum error is4.231929779052734e-6 and final
2.980232238769531e-7. f16 hidden checkpoints are identical; final maximum is
1.1920928955078125e-7. Maximum normalized RGBA-to-reference-RGBA difference is
0.003921572118997574, within the existing quantization-aware rule. All failing
and nonfinite counts are zero. These are observed input/conversion differences,
not a claim that independently decoded browser/native frames are identical.

Full plane/input/checkpoint/RGBA bytes remain ignored raw artifacts; tracked
summaries pin hashes, dimensions, layouts, reference and actual metrics. Replay
recomputes conversion, all comparisons and prerequisite provenance, and rejects
substituted snapshots or altered timestamps. No large tensor is tracked.

## 28. Ten-Minute Soak

**NOT_RUN.** The single registered600-second neural soak requires all short60
prerequisites. Baseline30 failure blocks those. No ten-minute stability,
sustained thermal or resource-growth claim is made, and no optional replacement
soak or favorable rerun was started.

## 29. Resource Stability

All measured frame/heartbeat resource snapshots satisfy the registered two-slot,
one-output, one-cache and one-display-link limits. No unbounded retained frame
queue was observed in the failed short trial. Neural audits remain active in
neural lifecycle/parity; baseline uses a finite, format-validated unorm path.

All recorded thermal states are nominal. These short-run observations do not
establish thermal endurance, process RSS stability or driver-internal resource
release. RSS was not measured. No soak-derived conclusion is available.

## 30. Cleanup

Stop detaches output/looper/items and display link, invalidates generations,
drains processing and presentation, releases slots and flushes the texture
cache. Disposal releases configured slots and cache. Every lifecycle, parity
and baseline cleanup snapshot has every declared counter zero, including
decoded sample owners, leased buffers and texture wrappers.

No native player, GPU measurement or development server remains needed at
closure. Raw study directories, frozen binaries and unsuccessful developmental
attempts are retained. Only the explicitly owned disposable clean clone/build
caches may be removed after verification; historical evidence is preserved.

## 31. Reviewer Findings

Independent read-only reviews covered all requested areas. Static review does
not itself rerun hardware; executable/hash/tensor replay was performed separately.

| Area | Disposition |
|---|---|
| Current API choice |supported current API, macOS26 explicit|
| Output timing |item-timebase and host target recorded; not scanout|
| Pixel-buffer ownership |strong sample/item/buffer retention reviewed|
| CVMetalTexture lifetime |retained through completion; physical release test|
| YUV mapping |actual planes plus independent synthetic/decoded oracle|
| Color interpretation |missing/unsupported metadata fail closed|
| Fused ingest semantics |fusion not performed; intermediate qualified|
| Candidate F integration |unchanged source/packing and exact adapter test|
| Drawable ownership |completion lifetime reviewed; explicit Void CI fix|
| Display scheduler |implemented; failed cadence/occlusion unresolved|
| Stale invalidation |generation/item and terminal checks reviewed|
| Seek/replacement |request ordering and final native lifecycle pass|
| Software age |raw recomputation passes; acceptance tails FAIL|
| 30fps evidence |binding baseline FAIL, neural gated|
| 60fps evidence |NOT_RUN, cannot qualify|
| Soak evidence |NOT_RUN, cannot qualify|
| Verdict |PARTIAL, no automatic Phase3|

Repaired before binding: strict metadata validation, actual owner counters,
input math exactness, async request/seek races, slot retirement order, invalid
surface handling, item-timebase attribution, raw tensor/endpoints retention,
case order/mode/warmup, evidence substitution, timeout replay and real drawable
resizing. Two hosted compiler failures are retained: ambiguous Task inference,
then Task-versus-Void closure return. Explicit discarded nonthrowing task handles
fixed both; final source CI passes. No timing was taken on the failed CI sources.

Open product findings are cadence, age, warmup and unmeasured foreground/60/soak/
audio. They are not repaired or reclassified into a pass after measurement.
Observer cost and precise stall attribution remain unmeasured.

## 32. Phase 2 Verdict

**NATIVE LOCAL PLAYBACK PARTIAL.** Fundamental decoded-frame correctness and a
working supported native path are established, so this is not an unimplemented
bridge or a failed model integration. Mandatory product cadence and age are not
qualified, and downstream gates remain unmeasured. READY is not justified.
The failed occluded trial is not proof that foreground native playback or F is
intrinsically too slow. There is no causal GPU-bottleneck claim or favorable retry.

## 33. Repository, CI And Size

Binding source is `c20e066971a3c7c4c3f0d81bb561f624e485e3c8`;
[exact-source CI35804039192](https://github.com/yassinsolim/AetherVSR/actions/runs/35804039192)
passes gate, fusion and native-metal. New player CPU contracts explicitly skip
three physical tests on hosted CI; local physical suite passes7/7. Original
native suite passes18/18 locally, hosted10 CPU passes/eight GPU skips.

Clean clone uses its own npm/native builds and the explicitly shared baseline
Python interpreter:18 successful checks,1504 Vitest passes/three existing skips,
612 Python,80 opt-in,71 desktop, original18 and new7 physical native tests.
Original Phase1/1.5 evidence remains replayable. Two existing moderate npm
advisories are unchanged, not silently remediated. Production model/core,
Electron/extension implementations and frozen reports are not edited.

Source tree before compact publication:55,482,498 Git-object bytes,3,237,758
bytes below58,720,256. The8MiB per-file guard is unchanged. Publication checks
use Git object sizes, not symlink-following filesystem sizes. Final documentation/
evidence commit size and exact-final-HEAD CI are checked after publication and
reported in the closeout; the verification index records source-bound gates and
unchanged payload hashes. Published documentation does not retime the source.

```sh
node tools/m13/playback.mjs check .cache/m13/phase2-study-03 results/m13-phase2-playback.json
node tools/m13/report.mjs --check results/m13-phase1-metal.json
node tools/m13/optimize.mjs check .cache/m13/phase15-study-01 results/m13-phase15-metal.json
```

Replay requires the retained raw artifacts and clean committed source. A fresh
clone without them can run build/CPU tests but cannot reproduce private raw
evidence from summary hashes alone. The frozen runner rejects consumed case IDs.

## 34. Phase 3 Recommendation

Do not start Phase3, ScreenCaptureKit, browser/Stremio integration or permission
experiments. A separately authorized continuation should first investigate the
recorded occlusion/scheduler and observer behavior, then prospectively define
any new foreground measurement and audio/soak evidence. This report authorizes
no rerun, kernel search, model change, capture capability or new permission.