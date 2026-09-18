# M11 Desktop Player MVP Report

## 1. Starting State
M11 began from the frozen M10.10RI closure at `9e33e9f`. The browser extension remained frozen; M11 changed the product surface to a macOS/Apple Silicon Electron player.

## 2. Pivot ADR
ADR-0058 records the desktop-owned player pivot, Electron 44.4.1, M12 renumbering, and the 56 MiB cap.

## 3. Storage ADR
The approved tracked-tree cap is 58,720,256 bytes with existing 8 MiB per-file exceptions. Generated media, profiles, caches, and Electron outputs remain ignored.

## 4. Architecture
The pipeline preserves acquisition -> import -> upscale -> presentation. Desktop acquisition owns one HTMLVideoElement; presentation owns one canvas; the core upscaler and RuntimeController are reused unchanged.

## 5. Security
The shell uses a sandboxed, context-isolated, node-disabled BrowserWindow, exact `aethervsr://app` assets, restrictive CSP, denied navigation/popups/downloads/webviews, and no preload or IPC. Trusted app-main-frame fullscreen is the sole permission exception.

## 6. WebGPU
Native smoke PASS on Electron 44.4.1: external and forced-copy import, timestamp queries, production model/shader execution, neural GPU samples, nonblank readback, Apple adapter, and no renderer Node access.

## 7. Core Reuse
No production core, model, WGSL, or RuntimeController behavior was changed. The production model digest remains `d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.

## 8. Local Input
The user selects a local progressive H.264/MP4 through the native file input. The file becomes one app-owned object URL and is revoked on replacement and cleanup.

## 9. Authority
The same HTMLVideoElement owns playback, audio, seek, mute, volume, rate, and frame timing. The canvas is subordinate and hides stale output during invalidation.

## 10. Presentation
The player has one responsive stage and one output canvas. Original video remains available when enhancement is unavailable or invalidated.

## 11. Controls
Native journeys passed play, pause, resume, forward/backward seek, audio volume/mute, rate, baseline, neural, Auto, resize, fullscreen, replacement, and recovery.

## 12. Parity
Six paused cases passed exact normalized RGBA parity against the standalone harness: 1/2/3 seconds, external and forced-copy import paths.

## 13. Audio
The native element remained the audio authority. Journey evidence verified digital Electron audibility transitions for play, mute, and unmute. Physical speaker synchronization was not measured.

## 14. Visual-Lag Method
The diagnostic observer records same-imported-texture identity, media timing, same-task submission brackets, next-rAF opportunities, generations, and GPU timestamps. It does not claim physical scanout or A/V synchronization.

## 15. 30 FPS
Short raw30 and neural30 arms passed. Raw callback cadence was 29.9993 FPS; neural rendered cadence was 29.9822 FPS with valid observations.

## 16. 60 FPS
All six short 60 FPS arms passed. Raw arms measured approximately 59.9987-59.9993 FPS; neural rendered arms measured approximately 59.9968-60.0152 FPS, including final-20-second gates.

## 17. GPU Timing
Short neural GPU p50 was 6.226-6.291 ms and p95 was 7.078-7.340 ms. Binding neural soak GPU p50 was 6.226 ms and p95 was 7.016 ms. These are GPU upscale timestamps, not end-to-end display latency.

## 18. Controller
Neural qualification used unchanged Auto behavior after actual neural stabilization. No fallback, probing, failed, or unavailable state was accepted in the binding neural observation.

## 19. Seek
Forward and backward seek passed with output invalidation and re-readiness at the requested media positions.

## 20. Pause/Resume
Pause stopped the enhancement pipeline and hid output; resume restarted it without stale-frame presentation. Native journey coverage passed.

## 21. Resize/Fullscreen
Repeated large/small resize passed. Fullscreen enter and exit passed after fencing the runner on native Electron fullscreen completion events.

## 22. Replacement
A->B replacement passed. The old object URL was rejected after revocation, the old pipeline stopped, and the new generation became authoritative.

## 23. Device Loss
Diagnostic device destruction passed. Playback continued, enhancement resources were released, output fell back safely, and no automatic retry occurred until a new source.

## 24. Window/Process Cleanup
All 22 journey cases passed with natural Electron exit code 0 and removed owned profiles. Short and soak runs also ended with zero resources and clean profiles.

## 25. 10-Minute Soak
Binding neural soak passed 600,001.3 ms, 36,000 useful/rendered frames, zero invalid observations, stable visual age, and clean teardown. The original raw soak remains an immutable FAIL because its closing native focus was false despite approximately 60 FPS cadence. Exactly one owner-authorized replacement raw soak then passed 600,000.4 ms with 59.99996 FPS, 601 one-second integrity witnesses, focus/visibility true at the final live snapshot, zero drops/gaps, and clean teardown.

Long-run table:

| Run | Duration | Cadence | Focus/visibility | Invalid | GPU p50/p95 | Visual age | Cleanup | Result |
|---|---:|---:|---|---:|---:|---|---|---|
| Raw original, retained failed experiment | 600000.7 ms | 59.99993 callback FPS | Final native focus false | n/a | n/a | n/a | PASS | FAIL |
| Raw replacement, owner-authorized | 600000.4 ms | 59.99996 callback FPS | 601 witnesses and final snapshot true | n/a | n/a | n/a | PASS | PASS |
| Neural, original binding pass | 600001.3 ms | 59.99987 submitted/rendered FPS | Focus/visibility true | 0 | 6.226/7.016 ms | -0.099 ms | PASS | PASS |

## 26. Optional URL
Not implemented or characterized. Local-file playback is the registered mandatory scope; URL, VP9, 1080p, MSE, DRM, and live media are outside M11.

## 27. Security Review
Independent review found no P0/P1 issue. Secure shell, permission exception, renderer isolation, navigation denial, CSP, and diagnostic-build boundary passed focused and native checks.

## 28. Product Verdict
**DESKTOP PLAYER MVP READY**

All registered mandatory secure/local/audio/parity/control/lifecycle/performance/lag/soak gates passed after the single authorized raw replacement. The original raw failure remains disclosed and immutable.

## 29. Public Claim
AetherVSR Desktop MVP is ready for local 720p30/60 H.264 playback on the measured macOS Apple Silicon apparatus, with native audio authority and production neural enhancement. This does not claim physical A/V sync, physical scanout latency, arbitrary web support, or cross-vendor support.

## 30. Repository, CI, and Size
The final source passed the complete repository gate set, clean-clone reproducibility, frozen extension byte comparison, production/diagnostic package checks, and the 56 MiB tree guard. The closure is published on `main`; exact final-HEAD CI run `35312989947` passed both `gate` and `fusion`.

## 31. M12 Recommendation
Keep M12 Cross-Vendor Desktop Validation unstarted. Publish M11 only after final gates, clean clone, push, and exact final-HEAD CI verification. No signing, notarization, updater, store, analytics, or account work is part of M11.
