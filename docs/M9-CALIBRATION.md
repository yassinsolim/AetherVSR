# M9 controller calibration decision

This decision precedes controller implementation and final stress testing.
Calibration is not final acceptance. All valid source-pinned compressed traces
remain under results/m9-calibration*.json.gz; the device-loss row is explicitly
invalid and excluded. The original diagnostic-enabled normal traces are not
pooled with the later diagnostic-disabled repeated-graph traces.

## Measured evidence

MacBook Pro Apple M5 24 GB, macOS 26.6.2/25G83, Chromium 153.0.8010.12, external
texture import, fp16, hardware GPU timestamps, foreground. Internal display
reported 120 Hz; startup rAF median 8.3 ms corroborates that, not full-run refresh.
Sources are synthetic runtime workloads, not native-1080 quality references.

Original normal 30-s traces: 720p60 p50/p95 6.475/7.423 ms, rendered 59.246 fps,
13 skips/0 drops; 720p30 7.389/8.239 ms, 29.932 fps, no loss; 1080p30
11.585/11.793 ms, 29.899 fps, no loss; 1080p60 11.743/11.951 ms, 59.710 fps,
5 skips/0 drops. 540p60 15-s retry: 6.141/7.330 ms, 59.724 fps, 3 skips/0 drops.
Legacy 10-ms failure rejects both normal 1080 cases despite observed sustained
short-run playback. Recovery at 7 ms also cannot certify normal 720p30 at its
observed warm-window cost. This motivates bounded evolution, not a universal
claim about 1080p suitability.

Corrected repeated-graph load, 15-20-s windows: two extra passes at 720p60
p50 15.369 ms still rendered 59.417 fps with 4 skips/0 drops; three extra
passes p50 20.464 ms rendered 46.617 fps with 144 skips/231 drops. The paired
30-fps three-extra-pass trace p50 20.317 ms rendered 29.950 fps with no loss.
Five extra passes at 30 fps cost 33.120 ms, rendered 29.234 fps with 12 drops.
Thus 30 fps can tolerate more measured stage work here, but not a simple
doubling of arbitrary thresholds. Browser overlap means a GPU timestamp span
is not the sum of decode/upscale/composition deadlines.

Tail-only threshold is rejected for this iteration: every fifth frame with four
extra passes produced median 5.536/p95 26.423 ms yet rendered 59.064 fps, 13
skips/0 drops. That does not demonstrate the benefit of a p90/p95 trigger over
median. Single seven-extra-pass stimulus reached 41.375 ms; steady median
6.040 ms. Max/p95 triggers would react to a transient the spec says to tolerate.
Keep raw p90/p95 in telemetry, not as unvalidated control inputs. Decoder drops
and callback pressure remain corroborating observations, not causal triggers.

## Frozen candidate policy

- Window: 30 fresh positive finite raw neural GPU durations; exact even median.
- Failure median > min(24, max(6, 0.90 * frameIntervalMs)); recovery median <=
  min(22, max(5, 0.78 * frameIntervalMs)). At 60 fps: 15/13 ms; at 30: 24/22 ms.
  Caps/floors are milliseconds. The 60-fps 15-ms failure boundary deliberately
  retains a margin below the observed 15.369-ms fit point; it is not a claimed
  maximum or universal safe fraction. Final matrix must test this candidate.
- Unknown cadence starts at 60 fps. Estimate source intervals from positive
  rVFC media-time deltas / presentedDelta / playbackRate over 30 observations;
  reject discontinuities and snap to the nearest 24/25/30/50/60/120 fps only
  within 8%; otherwise round measured FPS to an integer. Never use rendered FPS
  or decoder drops as source cadence. Other rates remain unvalidated workloads.
  Use the true median, not a mean. Relaxing below the assumed 60-fps cadence
  requires 30 consecutive clean BASELINE-only intervals, presentedDelta=1,
  exposed decoder counters with no drop increase, visible active playback and
  no media discontinuity. Source load/resolution changes clear this estimator.
  While neural executes, observations may only increase FPS (tighten budget),
  never infer slower source and relax it. A slower in-stream cadence remains
  conservative until a genuine baseline window establishes it; unavailable
  counters/rAF never relax the default. This is delivered source cadence under
  clean baseline conditions, not guaranteed encoded-file FPS; deterministic
  benchmark source FPS is separately known from ffprobe. Presentation cadence
  alone cannot distinguish all decoder/compositor pathologies.
- Dwell: three consecutive evaluations of the SAME predicate. Rejected samples
  do not advance dwell. Baseline, stale generation/order, >500-ms delivery age
  or future samples are ineligible. Keep sequence identity across timing resets.
- New neural generation: exclude the first three resolved samples and the
  first 150 active ms from decisions, but retain all in telemetry. Startup
  first frames reached 16-18 ms while normal steady values were much lower.
  No evidence-based claim that three frames is universally sufficient.
- Warmup/probe must finish within 2000 active ms; missing evidence or unresolved
  hysteresis enters fallback. Stable evidence expires after 2000 active ms.
  A probe requires recovery, not merely failure-threshold compliance.
- Backoff: 2/4/8/16/30 s after failed probes, reset to 2 s on confirmation.
  Schedule from probe start, no sooner than its end. This bounds adversarial
  removal latency to <=32 s plus <=250-ms polling (including any previous probe
  remainder), below the preregistered 35-s target when performance recovers.
  An initial fallback waits 2 s. No faster force-release route.
- States: warmup, stable, fallback, probing, manual-baseline, suspended,
  unavailable, failed. Suspended holds the previous state and active clock;
  terminal failed cannot transition again. Intent auto/prefer-neural/baseline
  is separate. Auto and Prefer neural retain identical performance safeguards.
- Every source/resolution/cadence change invalidates evidence and resets backoff;
  intent and forced state survive. Seek/loop preserve tier and backoff; probing
  remains provisional and retains its active deadline. Pause/hidden freezes
  active time and remaining wait; resume flushes evidence and rewams without
  granting recovery. Baseline intent never probes. Unsupported timestamps stay
  baseline. Timer and async-model paths respect fatal/lifecycle eligibility.

## Resource and session decisions

Keep destroy/recreate for now. Corrected known requested graph resources at
720p are 77,427,856 bytes, at 1080p 174,195,856, excluding optional timers and
driver/browser overhead. Retaining an inactive neural graph doubles neither
quality nor evidence and costs that residency request; measured configuration
wall time was 1.5-1.9 ms, not including all first-submit/driver work. The first
three 720p60 submissions spanned a 681-ms startup anomaly, so cold-start probe
behavior remains a live acceptance question. No energy/watt/actual-memory claim.

Session observer is separate from legacy pipeline statistics: explicit reset
starts wall-clock accounting; pauses/hidden intervals alone stop active time.
Tier switches never clear counts. Full-session neural quantiles use a bounded
0.05-ms histogram (0-200 ms plus overflow), with counts/exact mean/max;
raw traces are captured by benchmark tooling for exact independent quantiles.
Transition log is bounded to 128, reporting dropped entries and lifetime totals.
Probe count, failed probes, fallback active duration, current backoff and source
cadence/generation are exported. Stage-local legacy reset semantics remain.

## Final stimuli, separate from calibration

Normal matrix rows use the actual application controller and frozen model.
Development-only loaded runs use inner diagnostics disabled, explicitly labeled.
Sustained stimulus: three extra graph executions at 720p60, 45 s continuous
load, then removal and >=35 s observation. Transient: seven extra executions
on one neural frame. Load removal changes only the stimulus, never controller
evidence/state. Source transition: 720p60->1080p60->720p60 without page reload.
Foreground long run: 600 active seconds; record exact first/last 120-s windows.
Verify no spontaneous fallbacks, >=58 fps and <=1% specified loss ratio. These
remain targets, not already passed measurements. All P0/P1 fixes require rerun.