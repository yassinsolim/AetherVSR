# M13 Phase 2 - Native Playback Protocol

## Baseline and Scope

Prospective registration on 2026-09-22, before bridge/playback measurements.
Clean main/origin `e381bd9366803fe91c048a97bbe5088844627f10`, exact CI35781116341
gate/fusion/native-metal success. Fresh 17 gates pass: 1501 Vitest/3 existing
skips, 612 Python, 80 opt-in, 71 desktop, 18 physical native tests. Both historical
evidence checkers pass. Phase-1 and Phase-1.5 reports/results remain immutable.
Candidate F's generator, packing, geometry, weights and numerical limits remain
unchanged. No ScreenCaptureKit, browser/Stremio integration, HDR, permissions,
network video, custom audio or Phase 3.

## Current Apple APIs

Official documentation checked before implementation:

- [AVPlayer](https://developer.apple.com/documentation/avfoundation/avplayer): media timeline, seeks, rate, mute and volume authority.
- [AVPlayerVideoOutput](https://developer.apple.com/documentation/avfoundation/avplayervideooutput): player-attached output, macOS14.2+.
- [sample(forHostTime:)](https://developer.apple.com/documentation/avfoundation/avplayervideooutput/sample(forhosttime:)): current sample API, macOS26+, returns tagged buffers, item-timebase PTS and configuration.
- [AVVideoOutputSpecification](https://developer.apple.com/documentation/avfoundation/avvideooutputspecification): current defaultOutputSettings macOS15+, old pixel-buffer settings deprecated.
- [AVPlayerItemVideoOutput](https://developer.apple.com/documentation/avfoundation/avplayeritemvideooutput): legacy copyPixelBuffer deprecated; current pixelBufferAndDisplayTime returns CVReadOnlyPixelBuffer on macOS26.
- [CVReadOnlyPixelBuffer](https://developer.apple.com/documentation/corevideo/cvreadonlypixelbuffer): immutable wrapper, macOS26+; unsafe-buffer interoperation does not itself copy pixels.
- [CVMetalTextureCache mapping](https://developer.apple.com/documentation/corevideo/cvmetaltexturecachecreatetexturefromimage(_:_:_:_:_:_:_:_:_:)): BGRA8, R8 luma and RG8 chroma mappings; retain CVMetalTexture explicitly until GPU completion.
- [CVImageBuffer](https://developer.apple.com/documentation/corevideo/cvimagebuffer): matrix, primaries, transfer, clean aperture and chroma attachments must be considered.
- [MTKView](https://developer.apple.com/documentation/metalkit/mtkview) and [CAMetalLayer](https://developer.apple.com/documentation/quartzcore/cametallayer): explicit drawing, drawable size in pixels, render-only drawable, acquire late and release promptly.
- [CAMetalDisplayLink](https://developer.apple.com/documentation/quartzcore/cametaldisplaylink): current display-driven host scheduling, macOS14+; no Timer/sleep authority.

Choose a macOS26 player target, current AVPlayerVideoOutput.sample and settings,
not a deprecated fallback. Older Macs are explicitly unsupported by this player;
the original native tensor package keeps its existing deployment target. A new
top-level native package compiles the same original AetherMetal source files plus
an integration adapter; no duplicate kernels or independently maintained weights.

## Bridge and Color Gate

One CVMetalTextureCache per device/renderer, never per frame. Frame leases own
the source pixel buffer and every CVMetalTexture/MTLTexture until GPU completion.
Synthetic BGRA and 420v/420f fixtures precede any playback: verify plane count,
dimensions, formats, orientation, byte-stride handling and retained lifetime.
GPU conversion versus independent CPU coefficients: absolute tolerance 0.00001,
zero non-finite values, zero failing elements. BGRA channels and alpha are tested;
NV12 vectors include 0/16/128/235/240/255, neutral values, saturated chroma,
odd/even coordinates and spatial transitions. Wrong UV/range/matrix/orientation
must be distinguished by the fixture.

Mandatory committed H.264 fixtures report 1280x720, yuv420p video range and
BT.709 matrix/primaries/transfer. BGRA is decoded RGB; NV12 uses recorded range
and matrix. Missing/unsupported color metadata, HDR, non-full clean aperture,
non-square pixels or unsupported chroma placement fail safely unless a separately
verified explicit interpretation exists. No universal BT.709 assumption.
Keep encoded SDR RGB values for the network, not linear-light RGB. Display
color management is separate. Initial ingest is one GPU conversion pass into
F's float4 input buffer; no fused ingest or convolution retuning is planned.

## Pipeline and Ownership

AVPlayer/AVQueuePlayer with current output -> CVMetalTexture planes -> GPU
conversion -> unchanged F -> private RGBA -> one GPU presentation render into
MTKView drawable. Neural output always 2560x1440 for720p source, independently
of points, backing scale or drawable resize. Baseline uses identical acquisition,
bridge and scheduler but bypasses F. Use current display-link callbacks, not
polling/timers, and deduplicate sample PTS within source/loop generation.

The paused MTKView is externally driven, so drawable size is explicitly updated
from backing-pixel bounds after layout, fullscreen and backing-scale changes.
Lifecycle evidence must observe smaller, restored and fullscreen drawable sizes
while the network output stays2560x1440; playback liveness alone is insufficient.

Maximum two owned frame slots, no unbounded pending queue. Skip stale/unprocessed
input when slots are busy. Retain input leases until completion, never reuse a
slot while GPU work or presentation uses it. Generation invalidation on pause,
resume, seek, mode, loop/source change and terminal failure rejects stale results.
Pause retains the last valid still (not labeled newly current); seek/replacement
hide it immediately. AVPlayer alone owns audio and timeline. No pixel base-address
locking, CPU copies, NSImage/CGImage or pixel readback in production callbacks.
Offline tests and paused diagnostic captures are allowed and separately labeled.

## Ordered Evidence

1. Synthetic bridge/color/lifetime tests and independent review.
2. Small non-binding720p30 acquisition/presentation smoke and lifecycle development.
3. Same-exact-decoded-frame paused parity at1,2,3seconds: reference CPU ingest
   plus original qualified F versus integrated GPU input/F. Input tolerance1e-5;
   final f16/f32 limits0.05/0.001 and existing normalized-output rule. No claim
   of identical independent decoders. Tiny and full golden regression remains.
4. Native lifecycle: open/ready/play/pause/resume/forward/backward seek/stop,
   A->B replacement, baseline/neural, resize/fullscreen enter/exit, error and
   cleanup. Real callbacks and software fault tests, no forced GPU crash.
5. Binding720p30: baseline then neural, each5seconds warmup plus60seconds.
6. Only after30passes,720p60: three baseline/neural pairs B/N, N/B, B/N,
   each5seconds warmup plus60seconds. Each run stands alone.
7. Only after all short60neural gates pass: exactly one600-second neural soak,
   after5seconds warmup. No favorable rerun. No automatic Phase3.

Use the committed local H.264 clips, native file selection in the product,
automated explicit file URLs only in diagnostic runs. Short fixtures require
recorded loop boundaries. Prefer AVPlayerLooper/AVQueuePlayer, invalidate
frame identity on item/configuration/PTS wrap and report all loop transitions.
Do not exclude boundary time from observation denominators. No generated long
media or replacement fixture without prospective attribution.

## Pre-binding Fixture Characterization

AVFoundation on this OS exposes complete asset durations6.177/4.177seconds and
average nominalFrameRate29.138315/57.450169, although FFmpeg reports video
durations5.978500/4.013267 and nominal29.75/60. Both committed files contain
video only, no audio track. Keep complete AVFoundation loops and all boundary
time; no trimming or fixture substitution to improve cadence. Mute/volume/timeline
authority can be tested, but these fixtures do not prove audible playback or
speaker synchronization. Cadence gates remain29/58 independently of reported
average metadata. Signed early-frame age is also recorded; absolute lead must
not exceed two registered source intervals.

## Registered Acceptance and Metrics

Historical floor unchanged:30fps >=29 useful unique completed AND presented
generations/sec overall and final20seconds;60fps >=58 overall and final20seconds.
Use actual wall durations, not nominal sleep durations or repeated presentation
of one frame. Record acquired, duplicate/no-new, submitted, GPU-completed,
unique-rendered opportunities, busy/stale drops, PTS gaps, loop boundaries,
AVPlayer current time/status/rate and errors. No physical scanout or decoder
internal count claim: acquisition is output availability, not all decoded frames.

Age is AVPlayer item time at software presentation opportunity minus the exact
sample PTS, with signed values and host acquisition/submission/completion/opportunity
times retained. P95 <=two source intervals, max<=250ms, at most1% missing/invalid
observations. First20 versus last20median increase<=one source interval; no three
consecutive5second medians increasing by>one interval each. Reset only across
explicit generations, never to hide drift; display age is not physical latency.

Preferred GPU frame-path p50<=10ms/p95<=12ms is separately reported headroom,
not a replacement for cadence. Preserve neural8/10Phase1.5 limits as historical.
Use GPU timestamps for ingest, neural, presentation render, and combined active
GPU processing where supported. Report exact bracket, gaps, counts and unavailable
metrics as null/not measured. No adding medians; combine per-frame scopes only
with documented boundaries. Timestamp readback is not pixel readback.

Playback additionally enables F's existing float-output buffer and GPU-only
finite/range audits of each hidden stage and the prequantized final output.
All buffers/pipelines are preallocated; only a four-byte status word is read
after command completion. Invalid output is terminal. These audits and the
float-output write are included in the processing command's GPU interval;
they were not part of Phase1.5 isolated timings. F's kernels remain unchanged.

Resource snapshots count retained source buffers, wrappers, in-flight commands,
active outputs/schedulers and configured slots. Require bounded counts<=registered
capacity with zero owned resources after teardown; RSS descriptive only if measured.
No monotonically growing ownership. Terminal decode/format/GPU failure hides
enhancement and stops new work. API lifetime/scheduler/color/generation P0/P1 blocks
binding. All17requested review areas, full historical/new tests, clean clone,
unchanged58,720,256-byte cap and exact-final-HEAD CI precede READY/PARTIAL/NOT READY.