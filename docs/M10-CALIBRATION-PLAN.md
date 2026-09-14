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