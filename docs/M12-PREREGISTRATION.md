# M12 Cross-Vendor Desktop Validation Preregistration

**Status:** owner-authorized, validation in progress
**Binding product scope:** local 720p30/60 H.264 playback with production 2x neural enhancement
**Common Electron:** 44.4.1
**M11 source:** `bbe127e193961909bc98d421aee47bf0047811a9`
**M11 verdict:** `DESKTOP PLAYER MVP READY`

## 1. Purpose
M12 determines whether the unchanged M11 desktop player and production WebGPU neural path remain correct, secure, stable and real-time on representative real GPU implementations.

## 2. Non-goals
No model optimization, retraining, architecture redesign, extension work, UI redesign, codec expansion, 1080p product claim, distribution, signing, notarization, store or updater work.

## 3. Frozen M11
M11 evidence remains immutable: original raw soak FAIL from closing focus false, one authorized raw replacement PASS, neural soak PASS, 22 journeys PASS, six parity cases PASS, secure Electron shell and Apple M5 measurements.

## 4. Starting baseline
Required starting state is clean `main`, `HEAD == origin/main == bbe127e193961909bc98d421aee47bf0047811a9`, exact CI `35313331062` with `gate` and `fusion` success, and all M11 repository gates green.

## 5. Source pin
The M12 binding source is declared only after apparatus review and common-source checks. All binding machines use the exact source, Electron, model and media pins. A runtime fix invalidates dependent evidence and requires a new common source plus reruns.

## 6. Hardware classes
Classes are Apple macOS/Apple Silicon, NVIDIA Windows x64 RTX-class, AMD Windows x64 Radeon, and modern Intel Windows x64 Xe/Arc-class hardware. Missing hardware is not substituted or fabricated.

## 7. Hardware availability
A class without qualifying physical hardware is `NOT_RUN_HARDWARE_UNAVAILABLE`. It is neither PASS nor FAIL and makes the overall M12 result PARTIAL.

## 8. Local graphics session
Binding runs require a normal local graphical session on the intended physical GPU. RDP, altered VNC, headless mode, CI virtual graphics and translated virtual adapters are excluded.

## 9. Environment identity
Record CPU, RAM, OS/version, GPU adapter/vendor/device strings, driver version where exposed, display resolution/refresh, devicePixelRatio, Electron/Chromium/Node versions, source commit and model SHA. Do not record serial numbers or account identifiers.

## 10. Electron pin
All machines use Electron 44.4.1. An Electron change stops M12, requires prospective documentation, a new Apple control, a new common source and reruns.

## 11. Security contract
Every platform retains nodeIntegration=false, contextIsolation=true, sandbox=true, no preload/IPC bridge, no arbitrary renderer Node access, no remote documents, no webviews, denied navigation/popups/downloads, and only the trusted app-main-frame fullscreen exception.

## 12. Adapter verification
Before every native run, record the actual WebGPU adapter and reject software or unintended adapters. OS GPU preference may be set before launch on hybrid systems; unsafe Chromium flags are forbidden.

## 13. Capability record
Record navigator.gpu, adapter features and limits, device acquisition, shader-f16, timestamp-query, external-texture import, forced-copy import, production shader compilation, model loading and device-loss behavior where testable.

## 14. Import paths
The preferred external video texture path and existing generic forced-copy path are both tested where supported. A vendor may qualify through either existing path only if parity and all product gates pass; the actual route is reported.

## 15. Shared shader/model policy
Production weights, WGSL and graph remain shared. No vendor-specific shader/model branches, threshold changes, fallback policy changes or performance tuning are allowed.

## 16. Media pins
Every machine uses the exact deterministic 720p30 and 720p60 H.264 fixture hashes already pinned by M11. Re-encoded or per-machine substitutions are invalid.

## 17. Golden vectors
Existing stage-by-stage golden vectors are run before performance: ingest, stem, convolution body, activation, reconstruction/head and final output as applicable. A stage failure stops that vendor before long performance runs.

## 18. Same-machine parity
Each machine compares the desktop player with the standalone harness using the existing normalized RGBA8 contract: at least three external/preferred-path and three forced-copy paused cases where supported.

## 19. Cross-vendor numeric conformance
Cross-vendor output uses the trusted golden/reference tolerances. Bit-identical fp16 output is not required. Report max error, mean error, failing pixels/elements and tolerance; do not loosen tolerances after observing a vendor.

## 20. Validation funnel
Per vendor, in order: security launch, WebGPU capability, shader/model, golden vectors, parity, raw/neural 30 FPS, raw/neural 60 FPS, lifecycle, raw 600-second soak, neural 600-second soak. A binding failure stops that vendor.

## 21. Performance contract
Reuse M11 definitions unchanged: useful submitted/rendered cadence, final sustained window, zero invalid observations, bounded and non-growing visual age, valid Auto controller state, GPU/device error absence, focus/visibility integrity and clean resource/process teardown.

## 22. 30 FPS gate
Raw and neural 30 FPS require complete duration, cadence, callback gaps, decoder drops, valid observations, controller state, errors and clean teardown. Neural records GPU timestamps and visual age.

## 23. 60 FPS gate
Raw and neural 60 FPS require the unchanged approximately >=58 useful FPS overall and final window, zero binding invalid observations, bounded visual age, valid controller state, no device errors and clean focus/visibility evidence.

## 24. GPU timing
Report neural GPU timestamp p50/p95/max under the existing core-upscale scope. GPU milliseconds are descriptive and are not vendor rankings or physical display latency.

## 25. Controller
Auto behavior remains unchanged. Neural binding windows reject fallback, probing, unavailable and failed states unless an existing M11 contract explicitly permits them.

## 26. Lifecycle
Run all platform-neutral 22 journeys where possible: file open, play, pause/resume, seeks, audio controls, resize, fullscreen, quality modes, replacement, device loss/recovery, security and close states. OS-specific cases are classified prospectively.

## 27. Raw soak
After short gates, run one M11-style 600-second raw soak per vendor with continuous low-rate renderer/native focus and visibility witnesses, no in-run focus repair, full cadence/coverage/error/cleanup criteria and no retry absent a prospectively defined external invalidity.

## 28. Neural soak
After raw soak, run one 600-second neural soak per vendor with unchanged cadence, visual-age, controller, GPU/device, focus/visibility and cleanup requirements. No retry until favorable.

## 29. Windows discipline
Windows binding requires native Windows x64 builds, clean clone, `npm ci`, exact source, exact Electron and verified adapter/driver. Do not cross-compile from macOS and call it native validation.

## 30. Driver discipline
Record and freeze driver identity once binding begins. A driver update creates a new environment identity and invalidates comparison with the prior run.

## 31. Attribution
Apple versus Windows results cannot automatically isolate GPU vendor from OS, Dawn backend or media stack. A common Windows failure is reported as an OS/backend hypothesis, not three independent vendor failures.

## 32. Evidence envelope
Each machine produces a compact envelope with environment, source/model/media pins, adapter capabilities, golden/parity summaries, short/long performance, lifecycle, device loss, cleanup and machine verdict. Raw logs stay ignored.

## 33. Machine verdicts
Exactly one per hardware class: `PASS`, `FAIL`, `UNSUPPORTED`, `UNRESOLVED`, or `NOT_RUN_HARDWARE_UNAVAILABLE`.

## 34. Overall verdict
Exactly one overall result: `CROSS-VENDOR DESKTOP VALIDATED`, `CROSS-VENDOR DESKTOP PARTIAL`, or `CROSS-VENDOR DESKTOP NOT VALIDATED`. Full validation requires Apple, NVIDIA, AMD and Intel PASS.

## 35. Public claim and next milestone
If partial, claim only measured hardware/classes. If fully validated, name the measured systems; never claim every GPU. Recommend M13 Desktop Distribution & Release Engineering only after M12 closure, and do not begin it in M12.
