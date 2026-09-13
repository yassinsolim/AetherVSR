# M9 status and review ledger

M9 is closed: REPLACE BudgetGuard, with explicit performance limits.
Result publication gates passed; no all-scenarios pass is claimed.
Baseline: c65f67a, 168 Vitest/308 Python, typecheck/lint/build/npm ci and exact
GitHub CI success. Scope/protocol committed at 2683399; calibrated candidate
at afdc05a. Production model bytes remain unchanged. R3 is research-only.

## Apparatus amendments before remaining final runs

The first native final 720p60 normal row exceeded the 1% loss target. A
same-source manual-baseline control was worse: 67 callback skips and 66 decoder
drops over 1,791 presented frames (7.426%), versus Auto 23/23 over 1,792 (2.567%).
Neither is a passing clean-run result, and baseline loss does not prove that
all Auto loss was unrelated to neural. Both original artifacts are retained.

ffprobe found the bundled 60-fps H.264 clip has average rate 7200000/121313,
240 frames over 4.013267 s and PTS intervals from 1.233 to 28.167 ms, including
two departures greater than 2 ms from 16.667. The M9-derived 540p/1080p exact-CFR
clips did not show comparable loss in their short normal rows. This motivates
a controlled source-timing comparison; it does not retroactively pass the old
row or identify every decoder loss mechanism.

Before the long run, create an exact 1280x720/60-fps CFR variant with the same
registered ffmpeg recipe used for other M9 sizes. Pin parent, command/version,
output hash and ffprobe properties in a separate media record. Use it for new
baseline, clean Auto, transient, sustained-load and 600-s rows, all under new
artifact names. Keep the original clip as a separate irregular-cadence stress
case. Acceptance targets (including 1% loss) and calibrated controller policy
do not change. Both source variants and their results must remain visible in
the final interpretation; this is a post-observation apparatus amendment.

Preliminary calibration contexts used Playwright's default focus emulation.
Their focus:true fields are not independent native foreground verification.
Their source/timing traces motivated a candidate, not final acceptance. Native
Chrome/CDP noDefaults in the final runner removes focus/visibility emulation.
Its isolated profile uses mock Keychain storage to avoid an observed native
Keychain IPC wait; it never connects to a personal browser. Benchmark-mode
HTML excludes the Vite reload client, which otherwise reloads after freeze.

## Independent review and corrections

1. Starting audit: raw neural-only samples and stage-swap epochs were correct.
   Found mixed failure/recovery dwell, invalid samples advancing dwell, stale
   source evidence, delayed-model intent override, redundant probe-confirmation
   reconstruction and nonterminal lifecycle timers. Deterministic regressions
   reproduced the first two; bounded fixes committed before redesign.
2. Design review: clarified provisional probes across interruptions, missing
   evidence deadlines, complete recovery latency bounds, active-session clocks
   and measured load boundaries before calibration.
3. Apparatus review: added per-submission identity, epoch invalidation, async
   draining, model/media hashes, real focus logging and safe neural diagnostic
   teardown. Reusable bindings/views moved into configuration, including copy
   import. Initial diagnostic shader suffered device loss; that row is invalid
   and preserved, never used to select policy. Repeated existing neural graph
   work replaced it with verified single-span timestamp tests.
4. Calibration review: relaxed budgets cannot be inferred from neural-induced
   slowing. Lower cadence requires 30 clean baseline intervals with exposed
   decoder counters and no loss. Median retained; tail-only trigger lacks a
   demonstrated benefit in the captured burst trace. No content-quality signal.
5. Integration review: fixed backward-frame monotonic clock mismatch, Reset
   cadence disagreement, and file selection erasing session totals. Browser
   pause/resume found a queued rVFC timestamp preceding resume; observation and
   metadata clocks are now distinct and permanently tested.
6. Native lifecycle testing: corrected emulated visibility, Keychain startup,
   HMR document reload and Chrome-paused media handling. A forced freeze may
   pause the video; the controller stays suspended until explicit play, never
   overrides pause intent. Timestamp withholding now applies to the device's
   actual feature request. All twelve native browser journeys passed at 467e5b4.
7. Final accounting review: FPS uses initial-to-boundary active time, not drain;
   decoder endpoints include last-callback-to-boundary drops; compact controller
   snapshots retain backoff/threshold/generation transitions. Native cleanup is
   bounded. Media range cancellations are recorded separately from media errors.
   No test is passed merely because a row was captured successfully.
8. Final independent statistics audit reconstructed all 13 native rows, 63,574
    callbacks, 63,279 GPU samples, 26 early/late subwindows and 236 evidence-driven
    decisions. Counts, exact neural percentiles, decoder endpoints, model/media
    identity and source hashes matched. It recommended REPLACE while explicitly
    preserving original and short-CFR loss failures, not an all-pass closure.
9. Final source review found no reproduced P0/P1 in the measured paths. A P2
    read-only observer could advance the clock before a subsequent old-time
    snapshot; this was repaired and permanently tested. The actual measurement
    hooks did not exercise it, so the recorded matrix is not invalidated. API
    comments now label legacy payload accounting and its excluded padding.
10. Actual old/new NeuralUpscaler GPU output comparison passed all six cases:
    fp32/f16, odd sampled sizes and reconfiguration, plus paused 720p external
    video. Maximum byte difference zero. This verifies binding-lifetime output
   continuity, not new quality gains or performance. All twelve native lifecycle
   journeys passed again on final code at 818b82a, saved in
   results/m9-lifecycle-final.json.gz at evidence commit 0f9c1d8.
11. Final report/closure review rechecked artifact hashes, scoped acceptance,
   all 23 sections, the R3 prohibition and M10 extension recommendation. No
   additional P0/P1 or missing technical acceptance evidence was found. The
   stale closure ledger was updated; push and exact-HEAD CI remained required.

## Measured decision

The amended-CFR ten-minute gate passed: 600.0011 active seconds, 59.611557
rendered and 59.914890 presented fps, 315 combined skips/drops over 35,949
presented frames (0.876241%), zero errors/fallback/probes/tier changes. There
were 298 loop rewarm/confirmation state changes, not zero state transitions.
The short original Auto loss was 2.566964%, original baseline 7.426019%, and
short CFR Auto 1.057906%; those do not pass the 1% reference and are retained.
The literal preregistered clean-loss acceptance gate is the ten-minute interval,
not a claim that every shorter window or source passes it.

CFR sustained load fell back 0.4057 s after injection, had four failed probes
with 4/8/16/30-s backoff, and confirmed genuine neural recovery 16.3560 s after
removal. It remained neural for another 33.6440 s with expected loop rewarmups.
The 48.172890-ms single-frame stimulus caused no fallback. Normal 1080p30/60
were feasible in the measured short runs; no cross-vendor or universal 4K claim.
Full scopes, every matrix row and limitations are in M9-REPORT.md.

## Closure verification

Result/evidence commit 0f9c1d8 passed npm ci, typecheck, lint, build, 421 Vitest,
308 Python, and final trace replay. A fresh local clone passed the same code
gates without generated runtime media. Tracked bytes: 46,631,312 under the
owner-approved 48 MiB cap. Local clone .git disk usage: 437,784 KiB, not network
transfer size. Production-build diagnostic hook exclusion was checked.

Result documentation was pushed at 0f86f3e1f6c3a04f1eaf90bb2bad7f185eec2a6b.
Exact-SHA GitHub Actions run 34745933686 completed/success for both gate and
fusion; main was clean and equal to origin/main. The closing metadata commit
requires its own final fetch/clean/equal-HEAD/successful-CI check, returned in
the completion message rather than self-pinned here.

M10 is the recommended Chrome/Chromium extension, not implemented. Closure
never means the short-window loss-reference failures disappeared or that every
web stream is stutter-free.