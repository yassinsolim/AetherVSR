# M9 status and review ledger

M9 is in progress; no replacement verdict or completed milestone is claimed.
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

## Remaining gates

- Exact-CFR paired source comparison, full clean/transient/load matrix and
  ten-minute stability, including first/last two-minute windows.
- Independent final source/trace/statistics/acceptance review and explicit
  REPLACE / EVOLVE IN PLACE / RETAIN verdict.
- Final documentation, all software/replay gates, clean pushed main and
  exact-HEAD GitHub CI. M10 is not implemented.