# M11 - Desktop-Owned Player MVP Qualification

Prospective protocol; no desktop native data at registration. ADR-0058 records
the owner-directed Electron/macOS/Apple Silicon pivot and56MiB cap. M10 through
M10.10RI remain frozen historical results, not evidence that the neural engine
failed. The finite browser instrument investigation stays closed. M12 replaces
the old cross-vendor M11 and is not executed here.

## Baseline

Required separate starting commands verified clean main==origin/main==
`9e33e9fd1c23f073b827967bbac817ebc3691dae` at the requested repository. Fresh separate
npm ci/typecheck/lint/check:m1010r/test/build/build:extension passed. Vitest48 files,
1,418 passed,3 opt-in skips; Python612 passed; relevant FFmpeg/runner opt-in75 passed.
Exact starting CI35208378849 passed gate/fusion and research static checks.
Model140,467bytes SHA256
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`;
production274,684bytes bundle
`46cebd53b7665ce48772792d2ec2eb73057e915d93ef76374900f30a159ac551`;
diagnostic276,414bytes bundle
`7334c5db0f476048a10b63a2e0d9e4b240436fa3a669925e113cedc93cb1003f`.
Starting tracked54,461,963bytes; owner-approved cap58,720,256. Preserve8MiB per-file
rules, history and generated-data exclusions. Desktop builds, profiles, media,
readbacks and screenshots are ignored. Only source, lockfile, tests, compact hashes
and documentation are tracked. No increase beyond56MiB without owner approval.

Electron44.4.1 is pinned (MIT); Node>=22.12.0. Ten resolved packages were added,
no existing dependency versions changed. A dependency-advisory scan of the new
set reported no known CVEs; the two existing npm moderate advisories remain.
This does not guarantee absence of vulnerabilities or replace Electron review.

## Security and Scope

Sandboxed BrowserWindow: nodeIntegration=false, contextIsolation=true,
sandbox=true, webSecurity=true, webviewTag=false, no preload/IPC. No experimental
features or WebGPU-enabling flags. Local custom scheme serves an exact static
asset map and immutable model, never arbitrary filesystem paths. Deny remote
documents/network, popups, unexpected frame navigation, redirects, permissions,
downloads and webviews. No frame/pixel transfer through Electron IPC.

Prospective control implementation note, after the capability smoke and before
product qualification: fullscreen is the sole permission exception. Both Electron
permission handlers require `fullscreen`, the exact local app document, a main
frame, and the app origin/document as requester. Automatic fullscreen, media,
capture, filesystem and all other permissions remain denied. The original smoke
used blanket denial; it did not claim to test the later fullscreen UI.

Mandatory input: user-selected local progressive H.264/MP4,720p30/60, seekable with
audio. File -> object URL -> one app-owned video. The same video owns all playback,
audio, rate, seek, mute and volume; canvas processing is subordinate. No AudioContext,
source-page handover, Return, remote extraction, cookies, DRM, MSE/HLS/DASH or live
media. Optional URL/VP9/1080 characterization is not required and is not substituted
for mandatory local-file gates. Original video is visible whenever output is not
ready, invalidated or unavailable. No hostile-page geometry code is required.

## Ordered Native Gates

1. Before substantial UI: normal Electron WebGPU hard smoke. Require navigator.gpu,
   real adapter/device, external import, timestamp-query, production shader/model
   execution on external AND forced-copy paths, actual neural submissions/timestamp
   samples, nonblank normalized readback and no GPU errors. Record exact Electron/
   Chromium/OS/hardware/binary hash/security settings. f16 is requested when available;
   actual selected precision is recorded. No flag workaround if the gate fails.
2. Six deterministic paused parity cases against standalone harness, at1/2/3s,
   external and forced copy. Reuse full normalized RGBA equality methodology;
   exact input dimensions/PTS/bytes and production output bytes, same model/options.
   A genuine pixel mismatch stops performance qualification pending diagnosis.
3. Native local-file/audio/controls and lifecycle: pause/resume, forward/backward
   seek, small/source/fullscreen/exit/repeated resize, A->B replacement, diagnostic
   device loss and close during paused/baseline/neural/seek. Require one media
   authority, one pipeline, old URL revoked, no stale generation, playable original
   fallback, cleanup. No packaged production fault-injection API.
4. Preregistered720p30 characterization and primary720p60 qualification below.
5. Only if short720p60 gates pass: one600s desktop soak. No M12 or browser trials.

Tests/builds run outside serialized native windows. Raw attempts have unique IDs,
clean source pins, binary/media hashes and immutable artifacts. No completed attempt
is overwritten. Apparatus/implementation defects may be fixed before candidate
qualification with a new pin and explicit retained-failure attribution; a failed
binding performance run is not replaced by selective retries. No claiming fixes as
measured before new evidence. Native platform capability failure stops M11.

## Visual Timing, Not A/V Qualification

No sample-accurate digital A/V, speaker or physical scanout claim. Product audio
is the audible HTMLVideoElement that supplies visual callbacks. The visual metric
uses rVFC media identity/time, successful submission and next canvas presentation
opportunity (requestAnimationFrame after submission), not physical display time.

Use existing deterministic numbered70s non-looping720p assets for short controls.
Generate ignored long numbered media for soak if needed. GPU diagnostic identity
must come from the same imported texture used by the core, with preallocated
storage mapped after pause; no per-frame CPU pixel readback. Production builds
exclude diagnostic APIs/probes. Measure the instrumented candidate explicitly;
do not present callback wall time as GPU time or submission as scanout.

At steady state, identified texture PTS must not be older than the associated
rVFC mediaTime by more than one source frame (33.334ms at30fps,16.667ms at60fps),
and cannot be ahead by more than one frame. Unrecognized/ambiguous identities fail
the identity gate. The same-task encode/submission bracket must have p95<=one source
frame; next-rAF opportunity delay p95<=two source frames and max<=250ms. Report
these separately; they are not summed into a physical latency claim.

No-unbounded-lag gate: first20s versus last20s median opportunity-relative visual
age increase<=one source frame; no stale sequence/generation presented after seek,
pause/resume, mode or source invalidation. Age is callback-authoritative mediaTime
minus actual texture PTS plus elapsed software time to the opportunity, with all
components and raw unsubtracted distributions retained. Three consecutive5s windows
whose median age increases by>one frame each fail growth acceptance. Missing
opportunities/identities are not0ms; more than1% invalid observations fails. These
are prospectively chosen usability/software bounds, not thresholds from candidates.

## Cadence and Soak

720p60 ->2560x1440 production C16D2, unchanged RuntimeController. Three raw/neural
pairs in R/N, N/R, R/N order, each5s warmup plus60s actual observation. Neural warmup
must be actual neural work. Main comparison is original video/raw vs production
neural; Catmull has one separately labeled control where helpful. Every neural run
requires>=58 useful successful submissions and rendered output generations/second
overall AND final20s, valid visual-lag gates, no fallback/probe/error after warmup.
Each run stands alone; no pooling failed runs. Record GPU p50/p95 from timestamps,
callback latency, quality drops/gaps, controller transitions, actual duration and
resource counts. Modes are Auto performance, Prefer neural and Baseline; qualification
uses Auto after true neural stabilization, with other modes in lifecycle checks.

720p30 has one5+60s raw and one neural characterization, same visual/lifecycle
criteria and>=29 useful FPS. At1.25x playback report effective media rate separately;
the60fps throughput gate is at1.0x only. Audio remains native-element playback.

Conditional600s soak at720p60: same58FPS overall/final20s and visual limits, first/
middle/final20s evidence, timestamp GPU statistics, decoder drops/callback gaps,
controller states, errors and available app process/resource metrics. No temperature,
power, physical GPU-memory or throttling claim without separate measured sensors.

## Review and Closure

Independent review areas: pivot/roadmap, storage ADR, Electron security, core reuse,
file input, renderer lifecycle, WebGPU gate, parity, visual-lag metric,720p60 method,
seek/replacement, device loss, cleanup, long-run result and final verdict. Any P0/P1
affecting a product result invalidates it; repair requires new attributed evidence.

Exactly one verdict: DESKTOP PLAYER MVP READY / DESKTOP PLAYER MVP PARTIAL /
DESKTOP PLAYER MVP NOT READY. READY requires every mandatory secure/local/audio/
parity/control/lifecycle/performance/lag/soak gate; it does not require arbitrary
web support. A failed later product gate may leave a usable but honestly PARTIAL
local build, never a READY claim. No M12 implementation.

Close with all baseline gates plus desktop checks/build, frozen extension byte
comparison, raw evidence reproduction, size guard, independent clean clone, clean
HEAD==origin/main and exact-final-HEAD CI. Provide31 required report sections.
`npm run desktop:dev` and reproducible local build output are required; signing,
notarization, distribution/updaters/analytics/accounts are not part of M11.