# M10 runtime overhead calibration plan

Commit this plan before recording overhead calibration. It does not change
the shared M9 controller, media-security scope or model. Native production
extension bundles use ordinary browser WebGPU features/timestamp quantization;
no unsafe WebGPU, Dawn override or web-security flags. Unpacked-extension
action automation is separately identified, not an extension permission.

## Fixed sampling and boundaries

Exact M9 CFR 1280x720 H.264, 60/1 fps, source SHA256
`8d81acbe164da1d62b7d0d02a3cc66915c96e8aa90d45cac34d818fc33df1d4a`.
Production C16D2 hash remains fixed. Headful native Chrome on the base M5,
1200x820 viewport, active foreground, AC power, no simultaneous GPU experiment.
Record exact installed build hashes, browser/OS, source and model bytes.

Five-second warmup begins after actual runtime readiness. Each calibration
window is 30 seconds, selected before viewing its outcome. Three adjacent
harness/extension Auto pairs run in order H1/E1, E2/H2, H3/E3. Controls are
no extension loaded, installed/unactivated, and activated Baseline, one window
each. An invalid capture is retained and repeated with a new name; no valid
unfavorable run is discarded. Harness CSS/overlay and extension fixture CSS
are reported as different composition workloads, not assumed identical.

Native source callback observations are common to every arm. Additional raw
runtime hooks exist only for active pipelines; this observer asymmetry is a
measurement limitation. Raw GPU neural samples select submittedAt within the
window; pending readbacks drain afterward and never extend the FPS denominator.
Non-runtime arms have callback/presented rates, not invented processing FPS.
CPU records separate import/encode/submit, shared driver callback, nested
attachment callback, discovery and geometry. Do not sum nested measurements or
call these a whole-renderer CPU profile. No frame-rate DOM scanning or messages.

## Tolerance rule fixed before calibration

For each GPU quantile q (p50, p95), compute the three harness values Hq, the
paired extension values Eq, and differences Dq=Eq-Hq. Define noise envelope
Nq=max(range(Hq), range(Eq), range(Dq)). Final GPU allowances are N50+0.5 ms
and N95+1.0 ms, rounded upward to 0.1 ms. These three pairs provide a bounded
observed noise envelope, not statistical confidence or cross-device proof.
Report every paired difference; a large/noisy allowance must remain visible.

Calibrate CPU tolerances from the same six runs before final measurement:
shared-driver p95 allowance is max paired/calibration range plus 0.1 ms;
geometry/discovery summed work limit is max(1 ms/s, twice the highest extension
calibration work per second). No layout/discovery call may be a persistent
>=50-ms long task. Final extension callback/presented FPS must be within
max(1 fps, the observed three-pair rate-difference range) of a paired harness
run. A 600-active-second extension window still independently requires >=58
rendered/presented fps, <=1% combined loss, no false fallback/owner changes/
GPU errors or geometry failures. Noise allowances do not waive that gate.

Commit numerical allowances, actual calibration results and interpretation
before the final overhead and 600-second extension runs. Record first/last
120-second neural and callback windows separately, with decoder cut brackets.
Temperature, energy, actual device memory and SR quality are not measured by
these tests. Native-controls, frame, fullscreen/PiP and real-site limitations
remain separate from runtime acceptance.

## Frozen numerical allowances

Calibration-03 completed all nine fixed windows at source
`005e4e0165ab3deb09e123796cce67711a4ce308`, with production bundle
`85b4aa298146278524c7fdfc7888e6783ee996feb7be56180bbec5f3aa5d3dec`.
Machine: Apple M5 MacBook Pro (Mac17,2), 24 GB, macOS 26.6.2,
Chrome for Testing 153.0.8010.12, external import, 1200x820 foreground
window, default GPU timestamp quantization. AC power was checked immediately
after calibration; display refresh was not independently measured in this run.
The exact CFR clip and model hashes above remain unchanged.

| Pair | GPU p50 E-H (ms) | GPU p95 E-H (ms) | Native callback E-H (fps) | Native presented E-H (fps) |
|---|---:|---:|---:|---:|
| E1-H1 | 0.589824 | 0.196608 | 0.235248227 | -0.131331560 |
| E2-H2 | 0.524288 | 0.1507328 | 0.399769928 | -0.066860186 |
| E3-H3 | 0.589824 | 0.393216 | -0.801314538 | -0.101392310 |

The p50 noise envelope is 0.065536 ms; p95 is 0.2424832 ms. Applying
the fixed rule gives a **0.6 ms GPU p50 allowance** and **1.3 ms GPU p95
allowance**. All shared-driver CPU p95 paired differences and ranges are
zero at the observed approximately 0.1-ms clock granularity, giving a
**0.1 ms driver p95 allowance**. This is not a claim of zero CPU overhead.

Auto extension discovery plus geometry work was 0.213329067, 0.196652246
and 0.219988268 ms per playback second. The fixed aggregate allowance is
therefore **1 ms/s**, with the existing no-persistent-50-ms-call rule.
These brackets exclude other adapter/browser work and are not total CPU.
Nested attachment callback time must not be added to driver callback time.

The frozen relative native callback FPS allowance is
**1.2010844655309683 fps**; the native presented FPS allowance is **1 fps**.
These are observed three-pair noise envelopes, not confidence intervals.

| Window | Rendered fps | Native callback fps | Decoder drops | Skipped frames |
|---|---:|---:|---:|---:|
| H1 | 57.729485 | 57.729100 | 58 | 65 |
| E1 | 57.965507 | 57.964348 | 52 | 54 |
| E2 | 58.229063 | 58.227899 | 44 | 49 |
| H2 | 57.828322 | 57.828129 | 59 | 63 |
| H3 | 58.398443 | 58.397859 | 44 | 46 |
| E3 | 57.596928 | 57.596544 | 64 | 67 |
| No extension | not measured | 58.730201 | 0 | not measured |
| Installed, inactive | not measured | 58.429633 | 1 | not measured |
| Extension Baseline | 58.364527 | 58.363359 | 43 | 41 |

All active windows exceed 30000 ms without rounding up. Calibration Auto
combined loss, using the existing sum of skips and decoder drops divided
by runtime presented frames, was 5.18-7.30%; harness was 5.01-6.84%.
These unfavorable observations remain included. No-extension and inactive
controls have no measured runtime loss or processing rate. Their callback
rates do not establish the cause of the loss in active arms.

Calibration-01 and calibration-02 remain historical, nonbinding evidence:
their extension active windows were only 29904.8 and 29895.6 ms because
the adapter suspended at loop readiness events. The repairs count requested
playing seeks/buffering in the unchanged shared runtime clock while hiding
stale output. The earlier valid harness observations remain retained but
are not pooled with the corrected nine-window apparatus. An independent
raw-data reconstruction matched all 774 calibration-03 summary values and
the source, bundle, model, media and case hashes.

Final sampling is one 30-second window each for no extension, inactive
installed extension, Baseline, standalone harness Auto and installed Auto,
followed by one 600-active-second installed Auto window. Every window has
the same five-second readiness warmup. Only invalid captures may be repeated,
with the original preserved; a valid failed performance gate is a result.
The literal >=600 active seconds, >=58 rendered/presented fps and <=1%
combined-loss gates are unchanged. Temperature, energy, whole-renderer CPU
and actual GPU residency remain not measured.