# M9 Runtime Controller Report

Evidence date: 2026-09-13. **RECOMMENDED: REPLACE BudgetGuard**, subject to the pending closure gates in section 22. This is bounded measured success, not an all-pass verdict or completed-milestone announcement. Production neural weights are unchanged; R3 remains closed to production; M10 is not implemented.

## 1. Starting

M9 started from clean main and origin/main at `c65f67a45519bbbc5617a325f1e96ee4b4192f34`. The recorded baseline passed npm ci, typecheck, lint, build, 168 Vitest tests and 308 Python tests. Exact-source GitHub CI run `34694344764` succeeded. These are baseline results, not evidence for the final M9 HEAD.

The binding scope is [M9-PREREGISTRATION.md](M9-PREREGISTRATION.md), followed by the committed candidate in [M9-CALIBRATION.md](M9-CALIBRATION.md). [M9-STATUS.md](M9-STATUS.md) records the apparatus amendment, reviews and closure gates. [AGENTS.md](../AGENTS.md) requires measured, scoped claims.

Only production C16D2 neural x2 and Catmull-Rom are tiers. Production SHA256 remains `d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`. No training, R3 recovery, quality routing, temporal model, extension or NTIRE claim belongs to M9.

## 2. Audit

The original guard already consumed raw neural GPU durations, used a true 30-sample median, and separated stage epochs. It did not consume a trailing median as though it were a raw sample. Baseline work could not certify neural recovery.

Starting defects were shared failure/recovery dwell, rejected samples advancing dwell, stale source evidence, delayed-model intent overrides, redundant reconstruction after successful probes, and timers/callbacks surviving lifecycle or fatal boundaries. The first two were reproduced and corrected before redesign. Starting review found no P0; this is not a claim that later software has no defects.

The original fixed policy was failure >10 ms, recovery <=7 ms, dwell three, cumulative probe patience 60, and backoff 2/4/8/.../30 seconds. Its useful hysteresis and genuine-probe principles are retained.

## 3. Design

[src/core/upscale/runtime-controller.ts](../src/core/upscale/runtime-controller.ts) owns pure timestamp-driven policy. [src/runtime.ts](../src/runtime.ts) applies actual tier changes, binds generations and handles video intent/lifecycle. [src/core/metrics/runtime-session.ts](../src/core/metrics/runtime-session.ts) observes without selecting a tier.

States are warmup, stable, fallback, probing, manual-baseline, suspended, unavailable and failed. Intent is independent of state; fatal failure is terminal until reload. Successful probe confirmation retains the already-running neural stage.

| Edge | Evidence / dwell | Reset and clock behavior |
|---|---|---|
| Baseline acquisition -> warmup | Model, timestamps and clean cadence window or 1.5-s conservative default | New neural generation; cold exclusion; 2-s qualification deadline |
| Warmup -> stable | Full window; three non-failing medians | Keep resources; reset backoff to 2 s |
| Warmup/stable -> fallback | Three failing medians, or 2-s missing evidence | Clear evidence; wait 2 s initially |
| Fallback -> probing | Active backoff deadline, eligible intent/capabilities | New neural generation; 2-s probe deadline |
| Probing -> stable | Three recovery-qualified medians | Keep stage; reset backoff to 2 s |
| Probing -> fallback | Three failing medians or deadline | Clear evidence; double wait to 30-s cap, anchored at probe start |
| Seek/loop -> rewarm | Timeline discontinuity | Preserve tier, intent and backoff; interrupted probe stays provisional |
| Pause/hidden -> suspended -> resume | Explicit lifecycle | Freeze active clocks; flush evidence on resume; preserve remaining wait |
| Any -> manual-baseline/unavailable/failed | User intent, absent capability, fatal error | No probes; failure is terminal; manual intent survives later model arrival |

Acquisition, import, upscale and presentation remain replaceable boundaries. ADR-0041/0043 in [DECISIONS.md](../DECISIONS.md) document provenance hooks, separate session observation and prepared copy-import views. No video/page dependency enters the neural Upscaler.

## 4. Modes

Auto performance and Prefer neural intentionally share performance safeguards with only two real tiers. Prefer neural is not a force-through-overload switch. Baseline remains Catmull-Rom and never probes.

Missing usable GPU timestamps keeps Auto/Prefer neural on baseline with an explicit unavailable reason; CPU timing is not substituted. Delayed model availability cannot override baseline intent, forced fallback or fatal failure. Development load controls are not production modes.

## 5. Budget

For source interval T in milliseconds, failure is `min(24, max(6, 0.90*T))`; recovery is `min(22, max(5, 0.78*T))`. This gives 15/13 ms at 60 fps and 24/22 ms at 30 fps. These are calibrated policy boundaries, not universal GPU deadlines or a proof that the remaining frame stages fit.

Unknown cadence starts conservatively at 60 fps. The estimator uses the median of 30 positive media-time intervals divided by presented delta and playback rate; it snaps to 24/25/30/50/60/120 within 8%, otherwise rounds measured FPS.

Relaxation requires 30 consecutive clean baseline intervals with exposed monotonic decoder counters and no drop increase. Neural observations may tighten, never relax, the inferred budget. Source/resolution/rate changes clear evidence; delivered cadence is not guaranteed encoded-file FPS. Other rates remain unvalidated workloads.

## 6. Statistic

Decisions use 30 fresh positive finite neural samples and three consecutive evaluations of the same predicate. Invalid, duplicate, out-of-order, stale-generation, future or >500-ms-old submissions cannot advance dwell. Startup confirms non-failure; probes require recovery qualification.

Exclude the first three resolved samples and submissions within the first 150 active ms of a generation from decisions, but retain them in telemetry. Warmup/probe deadlines and stable-evidence expiry are 2000 active ms; missing evidence is bounded rather than treated as success.

Median was retained over an unvalidated tail trigger: calibration's every-fifth-frame burst had median 5.536 ms and p95 26.423 ms while rendering 59.064 fps. p90/p95 remain observations. Decoder loss and callback pressure corroborate performance but do not establish neural causality or steer policy.

## 7. Trace

[results/m9-replay-final.json](../results/m9-replay-final.json) identifies 15 valid calibration traces and explicitly excludes the retained invalid shader/device-loss run. Original diagnostic-enabled normal traces and later diagnostic-disabled repeated-graph traces are separate, not pooled.

Submission-bound generation, sequence, tier, source size, submittedAt and resolvedAt accompany each raw duration. Final compressed artifacts retain samples, frames, actions, transitions, configurations, boundary snapshots, errors and source/model/media provenance. Missing timing slots are explicit gaps, never invented samples.

Calibration used Playwright's default focus emulation: focus=true there is not independent native foreground evidence. Native final captures remove focus/visibility emulation. Earlier final artifacts remain retained, but the matrix below uses corrected final-v2/control/CFR/long summaries only; superseded accounting is not mixed into these numbers.

## 8. Real

All 13 matrix rows were captured on MacBook Pro, Apple M5, 24 GB, macOS 26.6.2/25G83, Chromium 153.0.8010.12, hardware Metal WebGPU, shader-f16 and GPU timestamps, external-texture import, visible/focused native foreground. Main-page viewport was 1200x820. The artifacts contain adapter features and exact execution flags.

System display inventory lists the internal panel at 120 Hz and attached VG248 at 60 Hz; full-run physical refresh is **not measured**. Startup rAF cadence is a dispatch observation, not a whole-run monitor measurement. These are synthetic H.264 runtime workloads, not native-1080 quality references or physical-mobile measurements.

The original 720p60 parent has PTS intervals 1.233..28.167 ms and average rate 7200000/121313. The post-observation CFR amendment pins its parent hash, ffmpeg 9.0.1 recipe, exact 60/1 output and output hash in [results/m9-media-cfr.json](../results/m9-media-cfr.json); other variants are pinned in [results/m9-media.json](../results/m9-media.json).

The parent and all original results remain retained permanently. CFR is an apparatus comparison, not retroactive repair of failing observations. Neither policy nor acceptance thresholds changed. Re-encoding can change workload as well as timestamps; it does not isolate every cause of loss.

## 9. Changes

Policy integration replaces live BudgetGuard routing with RuntimeController/RuntimeDriver and separate session accounting. Workload and lifecycle boundaries invalidate pending timing evidence; successful confirmation no longer rebuilds neural. Production model bytes and default neural graph are unchanged.

Reusable bind groups/views, samplers, buffers and textures are prepared in configuration, including copy import. The frame path adds no pixel readback or mid-frame await. Required external-texture binding and command/swap-chain allocations remain; deterministic resource tests are not a quantified live allocation-rate measurement.

Destroy/recreate is retained. Corrected known requested graph storage is 77,427,856 bytes at 720p and 174,195,856 bytes at 1080p, excluding timers and browser/driver overhead. The raw payload-style report is 10 bytes below aligned allocation requests. API wording such as "persistent GPU allocation" means owned requested resources, not measured device residency or proof of no leak.

## 10. Probes

Initial fallback waits two active seconds. Failed probes increase backoff to 4/8/16/30 seconds; confirmation resets it to two. Scheduling is measured from probe start and never earlier than probe end. Force release does not reset evidence or create a fast recovery route.

A probe must qualify within two active seconds, stricter than the registered four-second ceiling. Interrupted probes retain their provisional status and cumulative active deadline. Pause/hidden intervals freeze remaining wait. The calibrated adversarial recovery bound is <=32 seconds plus <=250-ms polling under restored usable evidence; the driver polls nominally every 100 ms, without a browser scheduling guarantee.

Configuration wall time in calibration was 1.5-1.9 ms, excluding first-submit/driver work. First GPU frames were 16.440 ms at 720p60 and 18.277 ms at 1080p30; first-three means were 9.074 and 14.306 ms, respectively. A cold-start anomaly spanned 681 ms across the first three 720p60 submissions; these are distinct scopes, not isolated shader compilation measurements. Live failed-probe dwell stayed below 0.878 seconds. Energy, watts and actual GPU memory are **not measured**.

## 11. Load

Development-only stress repeats the existing neural graph inside one outer GPU timestamp span: seven extra executions on one frame, or three extra executions during sustained load. No fabricated timing values or force-fallback proxy enter live evidence. Inner diagnostics are disabled for these rows; normal rows retain production diagnostics.

The original shader stimulus suffered device loss and is invalid, retained and excluded. The replacement apparatus has verified single-span timing and configuration-time reuse. Load removal changes only the stimulus, never controller evidence, state or backoff.

The one-shot rows recorded slow neural maxima of 45.692428 ms (original) and 48.172890 ms (CFR), exceeding the >=25-ms stimulus target, with zero fallback/probes. They are controlled stress results, not ordinary neural processing costs.

## 12. Long

[results/m9-long-720p60-cfr-long.json.gz](../results/m9-long-720p60-cfr-long.json.gz) records **600.0011 active seconds**: 59.611557 rendered fps, 59.914890 presented fps, 182 callback skips and 133 decoder drops over 35,949 presented frames. Combined loss is `315/35949 = 0.876241%`.

Neural raw GPU p50/p95 are 6.314195/9.8921986 ms over 35,617 samples; callback-latency p95 is 9.8 ms. Zero tier changes, probes, failed probes, fallback duration and errors were observed. The literal ten-minute >=58-fps, <=1%-loss and no-spontaneous-fallback targets pass on this amended source and machine.

There were 298 in-window state changes from 149 loop rewarm/stable cycles, not tier oscillation. The final bounded log contains 128 entries, reports 175 discarded entries and 303 lifetime transitions; raw capture retains all 303, including pre-window history.

First 120-second neural p50/p95: 9.0857775/10.1558031 ms, 7,118 samples. Last 120 seconds: 5.635247/5.75806205 ms, 7,138 samples. This is observed drift, with no established temperature or power cause. Subwindows select raw submissions/callback observations; decoder brackets are not interpolated into fictitious exact endpoints.

## 13. Background

[results/m9-lifecycle-final.json.gz](../results/m9-lifecycle-final.json.gz) records the final 12/12 passed native journeys at execution source `818b82a`, preserved in evidence commit `0f9c1d8`, with unchanged source during capture. They cover manual baseline, baseline seek/loop, Prefer neural desktop, neural seek/loop, source-picker round trip, neural pause/resume, backoff pause/resume, hidden/frozen resume, mobile viewport, fatal device loss, missing timestamps and delayed-model/manual races. The earlier passing report at `467e5b4` remains historical evidence.

Visibility was genuinely hidden and CDP freeze was exercised without Playwright focus emulation. Chrome may pause media after forced freeze; the controller remains suspended until explicit play and never overrides pause intent. Isolated native Chrome uses mock Keychain storage; benchmark HTML omits the reload client so freeze does not reload the document.

Default model-delay behavior never overrides baseline selection; fatal remains terminal. Source round trip is 720->1080->720 without reload. Owned-resource counters and deterministic stale-generation tests complement browser checks; actual memory leakage is not measured. Mobile here means desktop-Chromium viewport resizing, not mobile GPU validation.

## 14. Telemetry

Whole-session counters survive tier/source changes; explicit Reset restarts session measurements. Active time includes stalls, configuration and callback silence, stopping only for explicit inactive lifecycle state. Legacy stage-local pipeline statistics retain their reset semantics.

Report rates divide initial-to-boundary frame deltas by active wall time, excluding drain. Decoder totals include the final callback-to-boundary endpoint. Loss is `(callback skips + decoder drops) / presented frames`, with unavailable counters reported as not measured, not zero. Callback latency is max(0, callback metadata time - presentationTime), not GPU time.

GPU quantiles below are exact linear interpolation on raw neural durations selected by submission time, including cold samples and in-window submissions resolved during drain. The external-import span includes neural ingest through final blit, excluding the import call, decode, pre-span queue wait, CPU encode/configuration and browser presentation. Copy-path upload is outside its span; copy performance is not measured here.

Session UI quantiles instead use bounded 0.05-ms bins through 200 ms plus overflow; count/mean/max use raw values, and empty/overflow quantiles are null. Controller windows are 30 samples; transition history is bounded to 128 with lifetime/discard counts. Neither a UI trailing statistic nor a loaded-row aggregate is relabeled steady-state cost.

## 15. Oldnew

Replay loads the exact original `c65f67a` BudgetGuard git blob `70573ac611d4f70d829ab70dd7dfd0751d826c47`, not a rewritten approximation. It compares identical recorded arrivals with the pinned candidate and a virtual 100-ms polling schedule, separately for each of 15 valid traces.

On each normal 1080p30 and 1080p60 trace, legacy replay enters fallback four times and probes three times; the new controller has zero fallback and zero probes. This is evidence against the old fixed 10/7-ms policy for these captured workloads.

Replay is GPU-free and counterfactual. It uses externally known source FPS and virtual tiers/generations; it does not measure actual reconstruction, baseline performance, changed contention, cold-start cost or browser cadence inference. Live acceptance comes from the independent native artifacts, not replay counters.

## 16. Replacement

**RECOMMENDED: REPLACE BudgetGuard** in live routing with the bounded controller, while retaining exact legacy replay evidence. Its benefit is explicit lifecycle/intent handling, independent dwell, bounded missing-evidence behavior and avoiding unnecessary rejection of the measured normal 1080 workloads, while retaining real overload fallback and recovery.

This is not "all tests and workloads pass". The original and short CFR clean-run loss failures remain limitations. Closure still requires final source review, software/replay checks and exact-HEAD CI. No stronger model-quality, universal 60-fps, mobile, energy or memory claim follows.

## 17. Normal

The 13-row matrix is recomputed from each linked raw artifact's summary; `valid=true` means valid capture, not target success. All rows have zero recorded errors. Neural n is the in-window timing count; p50/p95 and callback p95 are milliseconds. Rates are whole-window means. Dash means **not measured**. Durations rounded below do not override exact boundaries in sections 12/18.

| Native input / scenario / artifact | Active s | Neural n | GPU p50 / p95 | Render fps | Present fps | Skips / drops | Loss % | Callback p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p30 Auto [original](../results/m9-final-v2-720p30-normal.json.gz) | 30.002300 | 891 | 7.136516 / 7.967087 | 29.931039 | 29.931039 | 0 / 0 | 0.000000 | 8.300000 |
| 720p60 Auto [original](../results/m9-final-v2-720p60-normal.json.gz) | 29.999300 | 1762 | 5.992096 / 7.124539 | 58.968043 | 59.734727 | 23 / 23 | 2.566964 | 9.900000 |
| 720p60 Baseline [original](../results/m9-control-720p60-baseline.json.gz) | 29.999600 | 0 | - / - | 57.467433 | 59.700796 | 67 / 66 | 7.426019 | 9.900000 |
| 1080p30 Auto [normal](../results/m9-final-v2-1080p30-normal.json.gz) | 30.000300 | 892 | 10.791710 / 10.980810 | 29.966367 | 29.966367 | 0 / 0 | 0.000000 | 8.300000 |
| 1080p60 Auto [normal](../results/m9-final-v2-1080p60-normal.json.gz) | 29.999900 | 1786 | 10.789315 / 10.956124 | 59.766866 | 59.933533 | 5 / 4 | 0.500556 | 9.700000 |
| 540p60 Auto [normal](../results/m9-final-v2-540p60-normal.json.gz) | 30.000300 | 1784 | 6.049179 / 7.248152 | 59.699403 | 59.932734 | 7 / 7 | 0.778643 | 9.700000 |
| 720p60 Auto [original spike](../results/m9-final-v2-720p60-single-load7.json.gz) | 30.000000 | 1777 | 7.338722 / 8.244524 | 59.466667 | 59.733333 | 8 / 8 | 0.892857 | 9.500000 |
| 720p60 Auto [original load/recovery](../results/m9-final-v2-720p60-sustained-load3.json.gz) | 99.999300 | 2488 | 7.226848 / 19.404035 | 58.610410 | 59.520417 | 91 / 107 | 3.326613 | 9.700000 |
| 720p60 Baseline [CFR](../results/m9-cfr-720p60-cfr-baseline.json.gz) | 29.999500 | 0 | - / - | 59.634327 | 59.867664 | 7 / 6 | 0.723831 | 9.700000 |
| 720p60 Auto [CFR](../results/m9-cfr-720p60-cfr-auto.json.gz) | 30.000800 | 1780 | 8.341393 / 9.033962 | 59.565078 | 59.865070 | 9 / 10 | 1.057906 | 9.800000 |
| 720p60 Auto [CFR spike](../results/m9-cfr-720p60-cfr-spike.json.gz) | 29.999900 | 1783 | 8.391705 / 9.159103 | 59.700199 | 59.933533 | 7 / 5 | 0.667408 | 9.800000 |
| 720p60 Auto [CFR load/recovery](../results/m9-cfr-720p60-cfr-load-recovery.json.gz) | 100.000600 | 2512 | 8.730700 / 19.376600 | 59.209645 | 59.779641 | 57 / 65 | 2.040816 | 9.800000 |
| 720p60 Auto [CFR long](../results/m9-long-720p60-cfr-long.json.gz) | 600.001100 | 35617 | 6.314195 / 9.892199 | 59.611557 | 59.914890 | 182 / 133 | 0.876241 | 9.800000 |

Dimensions are 960x540->1920x1080, 1280x720->2560x1440 and 1920x1080->3840x2160, always x2; 30/60 denotes registered source FPS, not measured render rate. "Native" denotes browser execution, not native-quality reference material. Baseline rows have zero neural samples, not zero neural processing time.

**Short-row FAIL against the 1% clean-loss limit:** original Auto 2.566964%, original baseline 7.426019%, CFR Auto 1.057906%. The original baseline also misses 58 rendered fps. Baseline being worse does not prove Auto loss is unrelated to neural. The literal preregistered 1% acceptance gate is the ten-minute run, which passes; neither statement erases the short failures. Other normal rows demonstrate bounded feasibility, not ten-minute certification at every resolution/rate.

## 18. Overload

Elapsed fallback is measured from the actual load action, including configuration and delivery, not from the first high sample: original 0.3898 seconds; CFR 0.4057 seconds. Both meet <=3 seconds. Four failed probes occur in each load run, with backoff 4/8/16/30 seconds and each failed dwell <0.878 seconds. No baseline observation confirms recovery.

Original load duration is **44.9998 seconds**, 0.2 ms below the literal >=45-second condition; it is near-target evidence, not a strict duration PASS. CFR load lasts **45.0000 seconds** and meets that condition. Whole loaded-window loss is 3.326613% original and 2.040816% CFR; those are not clean-run loss passes or steady-neural timing estimates.

Total fallback residence is 56.8403 seconds original and 56.8041 seconds CFR over the approximately 100-second rows. Each has five probe entries, four failures and one recovery. The observed schedule reaches the ceiling; untested hardware/scheduling regimes are not certified.

## 19. Recovery

Confirmed stable neural occurs 16.3668 seconds after original load removal and 16.3560 seconds after CFR removal, both below 35 seconds. Probe entry alone is not counted as recovery. Remaining observation is 33.6333/33.6440 seconds respectively, with no further fallback, exceeding the 15-second stability target subject to expected loop rewarming.

Original final state is stable; CFR final state is warmup at a loop boundary, not failed recovery. Both preserve neural tier after confirmation. Normal/spike short rows have 14 loop-driven state changes; original/CFR load rows have 31/30, respectively. A state-change count is not a tier-change count.

## 20. Reviewfindings

Reviews corrected mixed dwell/invalid evidence, source generations, async intent and fatal handling; bounded interrupted probes and clocks; then timestamp provenance, diagnostic teardown and reusable configuration resources. Calibration review prevented neural-induced slowing from relaxing cadence. Integration review corrected queued-rVFC metadata versus observation clocks and source changes erasing session totals.

Native apparatus review removed emulated visibility, Keychain startup blocking, HMR reload after freeze and assumed media auto-resume. Capability withholding now affects the actual device feature request. Final accounting review corrected boundary denominators, decoder endpoints, transition provenance and bounded cleanup; canceled media ranges are separate from playback errors.

The final reviewer P2 fix refreshes the controller clock after an optional frame observer reads a later snapshot; its regression passes in [test/runtime-driver.test.ts](../test/runtime-driver.test.ts). That callback behavior was not the measurement path, so this finding does not invalidate the recorded matrix. The legacy resource API's payload-versus-alignment scope is now explicit without changing historical values. All twelve native journeys passed again after the fix at `818b82a`.

[GPU output parity](../results/m9-output-parity.json) compared the actual old c65f67a and new NeuralUpscaler: fp32/f16 on odd sampled inputs, reconfiguration and a paused 720p external video frame. All six cases were byte-identical, maximum byte difference zero. This checks binding-lifetime output continuity, not new image-quality gains or performance. Independent final evidence review reconstructed all 13 rows, 63,574 callbacks, 63,279 GPU samples and 236 evidence-driven decisions without a count/quantile/provenance mismatch.

## 21. Commits

Actual commits after `c65f67a`, oldest first; these are existing history, not commits created for this report:

| Commit | Change |
| --- | --- |
| `2683399` | Preregister M9; close R3 runtime path |
| `dc0c028` | Isolate guard dwell; reject invalid evidence |
| `e38acee` | Provenance-bound GPU traces and lifecycle costs |
| `ef45590` | Preserve calibration; repair diagnostic load |
| `33fb503` | Verified load traces and tail stimulus |
| `afdc05a` | Freeze calibrated policy before stress tests |
| `10be3bd` | Two-tier controller and session telemetry |
| `8ff4924` | Separate metadata and observation clocks |
| `d281e11` | Genuine native visibility validation |
| `ba3affb` | Correct boundaries and native lifecycle accounting |
| `60d2de3` | Disable hot reload during suspension validation |
| `99cfc9c` | Omit benchmark-document reload client |
| `37ab201` | Distinguish frozen media pause from suspension |
| `467e5b4` | Honor timestamp capability in device setup |
| `7242a43` | Record passing native lifecycle verification |
| `508cfa6` | Separate media cancellation from playback failure |
| `88ecc36` | Preserve final matrix; register CFR comparison |
| `818b82a` | Audited long-run/result, observer fix and old/new output parity |
| `0f9c1d8` | Final native lifecycle, replay and scoped acceptance records |

CFR and long captures pin source `88ecc36856fc89730127137610d0965386eab0de`; original corrected matrix/control rows pin `508cfa6`. Their recorded RuntimeController and RuntimeDriver hashes match across all 13 rows. HTML/apparatus history is not silently labeled identical; the P2 observer fix is later than these captures. Closing documentation commits follow this evidence ledger; `git log --oneline c65f67a..HEAD` gives the complete published list including metadata-only closure.

## 22. Finalrepo

**Publication gates: PENDING push/exact-HEAD CI.** Result/evidence commit
`0f9c1d8e475846a0bde1d77a3a4c3a834127c6ab` passed final npm ci, typecheck, lint,
build, **421 Vitest tests and 308 Python tests**, plus final frozen-legacy trace
replay and independent review. A fresh local clone of that exact commit passed
the same install/code/Python gates without generated media or browser binaries.
The clone reused the installed Python environment; it did not rerun GPU tests.
The same two moderate npm advisories seen at registration remain unchanged.

At that evidence commit, the tracked tree is **46,631,312 bytes (44.471 MiB)**,
below the owner-approved 48 MiB cap in ADR-0042. Per-file and generated-media
restrictions remain. The fresh local clone's `.git` consumed **437,784 KiB** on
disk, including available object history; this is not network transfer size and
excludes installed dependencies/caches. No history rewrite is implied.

[results/m9-acceptance.json](../results/m9-acceptance.json) pins all 13 native
rows and keeps the three short loss-reference failures separate from the passed
ten-minute gate. Final native lifecycle is 12/12 and output parity is 6/6.
The production build excludes the development load/control hooks; production
weights are unchanged. Publication still requires clean main == origin/main
and successful exact-HEAD GitHub Actions; baseline CI is not M9 final CI.

## 23. NextM10

M10 remains the **Chrome/Chromium MV3 extension**, not further architecture training.
Preregister least-privilege site activation, content-script/video discovery and
SPA navigation, multiple-video ownership, attach/detach cleanup, fullscreen and
picture-in-picture behavior, cross-origin/DRM constraints, user mode persistence,
device-loss teardown, and performance against the unmodified page. Keep the
controller policy independent of page injection and reuse its lifecycle tests.

The initial extension gate should cover opt-in supported non-DRM video, source
replacement and fullscreen without duplicate loops/resources, clear permission
and unavailable-feature states, and measured overhead using the same M5 clips
plus real supported pages. Preserve the original/CFR loss limitations rather
than promising stutter-free behavior on all pages. No R3, extra model tier,
temporal processing, temperature/energy claim or M10 implementation is included.