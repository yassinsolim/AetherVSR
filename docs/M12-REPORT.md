# M12 Cross-Vendor Desktop Validation Report

## 1. Starting State
M12 started from clean `main` at M11 publication `bbe127e193961909bc98d421aee47bf0047811a9`, synchronized with origin and exact CI `35313331062` passing both `gate` and `fusion`.

## 2. Frozen M11 Result
M11 remains `DESKTOP PLAYER MVP READY`. Its original raw focus failure, one replacement raw PASS, neural soak PASS, 22 journeys, six parity cases, security shell and Apple M5 evidence remain immutable.

## 3. M12 Preregistration
The owner-authorized contract is recorded in [M12-PREREGISTRATION.md](M12-PREREGISTRATION.md) and ADR-0059. It preserves the M11 acceptance thresholds and forbids vendor-specific tuning.

## 4. Binding Source
The M12 evidence envelope is pinned to source `8523fda`. Runtime, model, desktop player, package and M11 validation files are unchanged from the M11 publication; only M12 documentation/apparatus was added.

## 5. Electron and Platform Versions
Electron is 44.4.1. The Apple control environment is macOS 26.6.2 build 25G83, Chromium 152.0.7977.78, Apple M5, arm64. Windows environments were unavailable.

## 6. Hardware Matrix

| Class | Hardware | Status |
|---|---|---|
| Apple | Mac17,2, Apple M5, 10-core GPU | UNRESOLVED |
| NVIDIA | No qualifying local Windows RTX machine | NOT_RUN_HARDWARE_UNAVAILABLE |
| AMD | No qualifying local Windows Radeon machine | NOT_RUN_HARDWARE_UNAVAILABLE |
| Intel | No qualifying local Windows Xe/Arc machine | NOT_RUN_HARDWARE_UNAVAILABLE |

## 7. Driver Matrix
The Apple control records the native macOS graphics environment; no Windows driver identity exists because no Windows binding machine was available. No software or translated adapter was substituted.

## 8. Security Parity
M11 security evidence remains valid because runtime/security files are unchanged: sandbox, context isolation, node disabled, exact app scheme, restrictive CSP, denied navigation/popups/downloads/webviews and narrow fullscreen exception.

## 9. WebGPU Capability Matrix
The retained Apple smoke passed navigator.gpu, device acquisition, external and forced-copy paths, timestamp-query, production shader/model execution and nonblank output. NVIDIA/AMD/Intel capability rows are NOT_RUN_HARDWARE_UNAVAILABLE.

## 10. Adapter Selection
The Apple smoke ran on the physical Apple adapter in a local graphical session. No remote, software or CI adapter is counted. Windows adapter selection was not measured.

## 11. Golden-Vector Conformance
Model identity and committed golden-vector architecture checks pass. Native stage-by-stage GPU golden execution has no committed runner in M11 and was not fabricated; this gate is UNRESOLVED and prevents an Apple machine PASS.

## 12. Same-Machine Output Parity
The retained Apple exact parity matrix passed six cases: three external/preferred-path and three forced-copy paused cases with normalized RGBA equality.

## 13. Cross-Vendor Numeric Conformance
No cross-vendor outputs exist. No fp16 bit-identity assumption or tolerance relaxation was made.

## 14. Import-Path Matrix
Apple external and forced-copy paths passed the retained smoke and parity evidence. Other vendor import paths are NOT_RUN_HARDWARE_UNAVAILABLE.

## 15. Apple Short Result
Retained Apple M11 short matrix passed all eight raw/neural 30/60 FPS arms, including final sustained windows, zero invalid neural observations and bounded visual age. It is supporting evidence under unchanged runtime bytes, not a replacement for the unresolved native golden gate.

## 16. Apple Long Result
Retained Apple neural soak passed. The original raw soak remains failed only for closing native focus; the one owner-authorized raw replacement passed continuous integrity, cadence and cleanup. The M12 envelope references both without overwriting either.

## 17. NVIDIA Short Result
NOT_RUN_HARDWARE_UNAVAILABLE. No physical NVIDIA RTX-class Windows system was available.

## 18. NVIDIA Long Result
NOT_RUN_HARDWARE_UNAVAILABLE. No long run was attempted or fabricated.

## 19. AMD Short Result
NOT_RUN_HARDWARE_UNAVAILABLE. No physical AMD Radeon Windows system was available.

## 20. AMD Long Result
NOT_RUN_HARDWARE_UNAVAILABLE. No long run was attempted or fabricated.

## 21. Intel Short Result
NOT_RUN_HARDWARE_UNAVAILABLE. No physical modern Intel Xe/Arc Windows system was available.

## 22. Intel Long Result
NOT_RUN_HARDWARE_UNAVAILABLE. No long run was attempted or fabricated.

## 23. Lifecycle Matrix
All 22 retained M11 journeys passed on the Apple M5 control, including file input, controls, seeks, audio, resize, fullscreen, replacement, device loss/recovery, security and close states. Windows lifecycle is not measured.

## 24. Device-Loss Matrix
The Apple diagnostic device-loss journey passed with playback fallback, resource release and recovery through a new source. Other classes are NOT_RUN.

## 25. GPU Timing
Apple short neural GPU p50 was approximately 6.23-6.29 ms and p95 approximately 7.08-7.34 ms. Binding neural soak p50 was 6.226 ms and p95 7.016 ms. These are descriptive GPU upscale timestamps, not vendor rankings.

## 26. Visual Age and Cadence
Apple short and long neural observations passed unchanged M11 cadence and visual-age rules. The raw replacement passed 59.99996 callback FPS with zero frame drops/gaps. No cross-vendor comparison is possible.

## 27. OS-versus-Vendor Interpretation
No Windows observations exist, so no GPU-vendor attribution or OS/backend comparison is made.

## 28. Unsupported and Unresolved States
NVIDIA, AMD and Intel are `NOT_RUN_HARDWARE_UNAVAILABLE`. Apple is `UNRESOLVED` for M12 because native stage-by-stage golden execution was not available in the retained apparatus. This is not an M11 failure.

## 29. Optional 1080p Characterization
Not run. It is outside the mandatory 720p30/60 product claim and M12 scope until all mandatory gates are resolved.

## 30. Security and Reviewer Findings
The M12 preregistration and envelope policy were reviewed locally. No security weakening, vendor-specific shader path, threshold change, or fabricated hardware result was introduced. A final independent M12 evidence review remains a closure gate.

## 31. Per-Machine Verdicts
Apple: `UNRESOLVED`. NVIDIA: `NOT_RUN_HARDWARE_UNAVAILABLE`. AMD: `NOT_RUN_HARDWARE_UNAVAILABLE`. Intel: `NOT_RUN_HARDWARE_UNAVAILABLE`.

## 32. Overall M12 Verdict
**CROSS-VENDOR DESKTOP PARTIAL**

M11 Apple evidence remains valid; M12 does not yet establish cross-vendor validation because three required physical classes are unavailable and the Apple native golden-stage gate is unresolved.

## 33. Public Claim Scope
The public claim remains the measured local Apple M5 desktop claim from M11. No NVIDIA, AMD or Intel support claim is made. M12 does not support “works on every GPU.”

## 34. Repository, CI, and Size State
The M12 envelope is `results/m12-envelope.json`. The tree remains under the approved 58,720,256-byte cap. M11 exact CI passed; M12 changes require their own final gates and exact-HEAD CI before publication.

## 35. Recommended Next Milestone
Do not begin M13. The bounded next M12 action is either to obtain qualifying physical NVIDIA/AMD/Intel machines and a reviewed native stage-golden runner, or close M12 as partial with the current explicit limitations. No optimization loop is authorized.
