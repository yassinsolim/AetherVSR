# M9: adaptive runtime performance control

## Starting evidence

Starting clean main and origin/main:
`c65f67a45519bbbc5617a325f1e96ee4b4192f34`. Separately executed npm ci,
typecheck, lint, test (168 Vitest), build and all tools/test_*.py (308 Python)
passed. Exact-SHA CI run 34694344764 completed/success. Node 26.5.0; Python
3.12.13/PyTorch 2.14.0. The two moderate npm advisories predate M9. Docker's
daemon is unavailable; no container services are involved in this browser app.

Only current production C16D2 neural x2 and Catmull-Rom are adaptive tiers.
Production SHA256:
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.
No training, R3 investigation, temporal processing, content-quality routing or
extension work. R3 remains research-only; its unresolved long-horizon failure
does not block runtime control. An explicit new architecture experiment is
required to reopen it, not a runtime dependency or hidden model override.

## Starting guard audit

Independent source review at the starting SHA found no P0. Raw outer GPU
timestamps feed the guard after asynchronous readback, not trailing medians.
The span covers ingest through final neural blit on external import; the copy
path excludes its preceding upload. Decode, queue waiting before the span,
configuration, CPU encoding and browser presentation are outside the span.
Four readback slots can skip observations. Readback arrival is not submission
time. Baseline timings are filtered in main; stage swaps advance the timer
epoch, preventing old-stage readbacks from certifying neural.

The policy uses a true median of 30 positive raw samples, fail >10 ms,
recover <=7 ms, dwell 3, cumulative probe patience 60 and 2/4/8/.../30 s backoff.
Force requests fallback, and release preserves its existing probe deadline.
Baseline does not prove recovery. These controls are worth preserving.

Concrete defects to reproduce with deterministic tests:
- Opposite failure/recovery predicates share a dwell counter: 18/12/6 ms
  medians can confirm recovery with just one recovery-qualified evaluation.
- Nonpositive/nonfinite samples do not enter the window but still evaluate it,
  allowing rejected evidence to complete dwell.
- Source resolution reconfiguration does not invalidate outer timestamps or
  guard history. File selection resets pipeline measurements, not the guard.
- Delayed model loading can overwrite a later baseline selection or force.
- Successful probe confirmation rebuilds an already-active neural stage.
- Probe timers and callbacks are not terminal on device loss and are not
  gated on pause/visibility; missing timing evidence can leave a probe forever.

setUpscaler destroys the old stage and resets pipeline measurements; the next
frame allocates/configures the replacement. Configuration is outside CPU-frame
timing. Measurement reset clears timing/rate/counter windows and advances the
outer GPU epoch, not guard history or inner stage diagnostics. loadstart only
advances decoder-counter generation. Seek/loop have no explicit guard reset;
pause/visibility have no explicit suspension. Device failure stops acquisition
and protects fatal status, but leaves controller timers active. These facts
are source audit findings, not measured runtime performance.

## Intent and execution contract

- Auto performance: choose between the two real tiers using runtime evidence.
- Prefer neural: explicitly request neural; performance fallback remains ON,
  with the same safety thresholds and genuine recovery probes as Auto. With
  only two tiers these modes intentionally share policy, not different models.
- Baseline: stay on Catmull-Rom; never probe or consume neural evidence.
- Fatal device/frame-execution failure outranks all intent and is terminal
  until reload. No new stages/probes or healthy-status messages after failure.
- Missing usable GPU timestamps cannot certify neural; Auto/Prefer neural use
  baseline with an explicit unavailable-evidence reason. No CPU-time proxy.
- Unsupported timestamps disable neural trials. Temporarily missing samples
  allow only bounded warmup/probe trials and bounded-backoff retries; stable
  neural has an evidence-age timeout. Calibration fixes deadlines before final
  tests. Timer checks run even when no new sample arrives.
- Diagnostic overrides are development/benchmark-only, not production modes.
  No model other than the frozen production weights enters adaptive results.

Policy accepts explicit monotonic timestamps, workload identity, neural raw
samples and lifecycle events. It returns tier, state and transition reason;
it owns no video, GPU, model or canvas. Execution applies only an actual tier
change, never reconstructs neural merely to confirm a successful probe.

States: neural warmup, stable neural, waiting in fallback, neural probe,
manual baseline, suspended, unavailable evidence, terminal failure. Forced
overload is an explicit fallback condition. Intent is independent of state.
Every reset invalidates asynchronous sample generations before accepting new
evidence. Each raw observation carries submission-bound tier, generation,
sequence and timestamp plus arrival timestamp; reject stale, duplicate and
out-of-order observations. Future tier metadata may be representable; only two
tiers exist.

Lifecycle rules fixed before implementation:
- Resolution/source/playback-rate or established cadence change: discard old
  evidence and pending readbacks, preserve intent, restart conservative neural
  warmup if eligible, reset probe backoff for the new workload.
- Seek/loop: discard evidence/readbacks, preserve intent and backoff; rewarm
  neural if already selected, otherwise retain fallback/probe schedule.
- Rewarming an interrupted probe never converts it to ordinary startup:
  recovery qualification and cumulative active-time deadline still apply.
  Eligible always excludes manual baseline, forced fallback, unsupported
  timestamps and terminal failure. Suspended fallback resumes on baseline;
  suspended probes remain probes, with their unexpired active-time budget.
- Pause/hidden/suspended: no samples/probes/transitions due to elapsed silence;
  freeze remaining backoff. Foreground active playback resumes with fresh
  evidence and warmup, preserving intent and remaining wait.
- Manual changes discard evidence; baseline cancels probes. Measurement reset
  does not change intent or policy. Source replacement cannot be undone by an
  older asynchronous model-load callback.

## Measurement stages

First capture calibration traces with the existing fixed thresholds and an
unguarded neural diagnostic mode, clearly labeled calibration, not final tests.
Use committed synthetic test clips, or deterministic variants with exact
ffmpeg commands and hashes. Investigate 540p, 720p and 1080p x2 at 30/60 fps;
unsupported limits are recorded, never silently reduced to a smaller output.
Do not infer source FPS from rendered FPS. Examine rVFC media-time intervals,
presented deltas, callback cadence and rAF refresh observations separately.
Refresh is an observation, not an unverified monitor specification.

Compare candidate budget fractions/caps and median, p90, and median-plus-tail
rules using recorded raw observations. Fix the selected formula, thresholds,
window, cold-start exclusion and backoff in a NEW committed calibration decision
before final stress tests. Do not overwrite this registration. Retaining the
10/7-ms fixed budget is acceptable if cadence-dependent relaxation lacks support;
30 fps is never assumed to permit doubled thresholds. Decoder drops, callback
lateness and frame skips are corroborating safety observations, not proof that
neural caused pressure. No content/CRF/quality predictor enters selection.

Measure configuration wall time and the first neural samples per generation
separately from steady-state samples. Compare destroy/recreate against the
calculated inactive-resource cost before considering retention. No resource
retention without quantified bytes; actual driver memory is not measured.
Probe energy is not measured without a suitable instrument; report submitted
neural work/sample count and probe dwell as proxies, never joules or watts.

## Final acceptance targets

These targets apply after a separately committed calibration decision. A failed
target is reported and diagnosed; changing it requires a new protocol and
rerunning all affected evidence, not relabeling the same run.

1. Clean 720p60 M5 Auto: ten active foreground minutes, zero spontaneous
   fallback, no errors, mean presented/rendered rates >=58 fps, combined
   callback skips and decoder drops <=1% of observed presented frames.
2. Single transient: one injected slow GPU frame of >=25 ms must not cause
   fallback. Synthetic equivalent must also pass.
3. Sustained injected neural overload: fallback <=3 active seconds after
  injection begins, including configuration and sample delivery latency,
  no non-probe return to neural. Do not move onset to the first sample.
4. Continued load for >=45 s: failed probes back off monotonically to the
   30-s ceiling; no confirmation from baseline samples, invalid samples or
   opposite dwell predicates. Probes are bounded to <=4 active seconds.
5. Load removal: genuine neural recovery <=35 active seconds after removal,
   then >=15 s stable; failed probe increases backoff. No skipped cold-start
   sample may be relabeled as steady-state evidence.
  Recovery means confirmed stable neural under restored recovery-eligible
  performance with usable timing, not just probe entry. Calibration must
  budget interrupted-probe remainder, 30-s backoff, new configuration and
  qualification/readback within 35 s, including adversarial removal phases.
6. Manual baseline: zero probes; pause/background: no timer-driven probes.
   User intent survives seek/loop/source replacement. Device loss terminates
   controller activity and cannot be masked by UI or pending promises.
7. 720->1080->720 without reload: configure correct output, reject old-generation
   evidence; bounded owned GPU resource counts return to the same-size baseline.
   Actual device-memory leakage remains unmeasured without an external profiler.
8. Whole-session frame/drop/transition counters survive every tier switch;
   legacy stage-local benchmark fields retain their original reset semantics.

Final matrix: 720p30 normal; 720p60 normal, transient, sustained overload and
overload/removal; 1080p30/60 if supported; 540p feasibility. Each row records
source/target/FPS, exact browser/OS/adapter/features/import, visibility/focus,
neural-only full-window p50/p95, callback latency, mean presented/rendered FPS,
skips, decoder drops, state/tier timeline, probe counts and fallback time.
Use explicit start/end windows, no trailing-window percentiles labeled whole
run. Background runs are lifecycle checks, not comparable GPU benchmarks.
Long-run first/last two-minute windows report observable drift, not temperature.
The final acceptance interval begins on an explicit active foreground reset,
not each stage switch. Active wall time advances through stalls/configuration
even without callbacks and excludes explicit pause/hidden intervals only.
Opening ticks bound the interval rather than add a fictitious extra interval.
Accumulate decoder counter deltas across load generations; count resets once.
Loss ratio is (callback skips + decoder drops) / observed presented frames;
report its components and unavailable counters separately. Calibration freezes
full-window quantiles and raw-trace retention/truncation semantics.

## Testing strategy and evidence

App: plain-DOM WebGPU SPA, no DB/auth/backend. Existing Vitest guard/source/
timer tests are reused and expanded under npm test; Python suite remains intact.
New tests use deterministic timestamps, no sleeps, raw sample traces including
absence, lifecycle/user changes, and bounded transition logs. Browser tests use
Playwright on real Chromium/Metal, not software-rendered GPU benchmarks.
No auth/persistence service tests apply to this local video processor.
Browser journeys: normal playback/manual intent; overload/probe/recovery;
source/seek/pause/visibility changes; fatal device failure. Unknown browser
capabilities are tested and reported; no GPU availability inferred from Node.

Development-only deterministic GPU load must alter measured neural execution,
not inject fabricated timing values into live results. Record shader/work
configuration, one-shot/sustained scope and whether its cost is inside the
timing span. Load hooks must be absent or inert in production builds.
The binding load is inside the neural timestamp span; outside-span contention
is a separate experiment. Freeze workload size and onset/removal timestamps.
Force fallback is not load evidence, and removal never directly resets policy.

Bound transition history and whole-session telemetry; export user intent,
state/tier, thresholds/statistic/window/count, generation, reasons, transition
count, fallback duration, probe/failure counts, backoff and source cadence/size.
Retain compact real traces for identical old/new policy replay. No nested
timing windows. Lost/truncated trace rows must be explicit.

Independent reviews: starting audit; state/budget/statistic design;
oscillation/probe/lifecycle/telemetry/load implementation; final matrix and
replacement verdict. Any result-affecting P0/P1 invalidates affected runs.
Choose REPLACE, EVOLVE IN PLACE or RETAIN BudgetGuard based on measured benefit,
not milestone pressure. All local gates, trace replay, live evidence, clean
pushed main and exact-HEAD CI are required for closure. Missing hardware data
is not measured, never a passing result. M10 may be recommended, not implemented.