# M10: Chrome/Chromium MV3 extension

## 1. Starting state

Started on clean, synchronized `main` at
`2c62a2cf27decc6c88c8636ee85be1e76544a63b`. Separate npm install, typecheck,
lint, build, 421 Vitest and 308 Python tests passed. Exact-SHA CI
[34746007298](https://github.com/yassinsolim/AetherVSR/actions/runs/34746007298)
passed. Two moderate npm advisories predated M10. This is baseline evidence,
not closing CI. Architecture and acceptance were committed before implementation
in [M10-PREREGISTRATION.md](M10-PREREGISTRATION.md).

## 2. Extension architecture

MV3 service worker: validated activation, document registration, fixed packaged
model delivery and origin-keyed mode settings. ISOLATED content: discovery,
one owner, geometry, lifecycle and status. Shared core: the existing
`RuntimeDriver`, `RuntimeController`, `VideoPipeline`, C16D2 and Catmull-Rom.
The harness remains a separate consumer. No second inference engine or controller
policy was introduced. Initial shared attachment teardown was made terminal and
complete; M10 did not retune M9 thresholds. See
[ADR-0045](../DECISIONS.md#adr-0045---mv3-isolated-adapter-temporary-access-and-shared-runtime).

The worker owns no GPU, frame loop or runtime lease. The page retains playback,
audio, source, seeking and original DOM/style authority. The extension owns only
its canvas, listeners, observers and processing resources. No training, model
replacement, R3, temporal processing, interpolation, cloud or native backend.

## 3. Manifest and permissions

The [manifest](../src/extension/manifest.json) declares version 0.1.0 and minimum
Chrome 106. That minimum is an API floor, not a tested-browser compatibility claim.

| Permission | Justification |
|---|---|
| `activeTab` | Temporary access following an actual user action on the active tab |
| `scripting` | Explicit top-frame ISOLATED injection and content status commands |
| `storage` | Versioned, origin-keyed user mode, not browsing history or video URLs |

No permanent/optional host permissions, declarative content scripts, WAR,
`externally_connectable`, tabs permission, arbitrary fetch command or persistent
auto-activation. Permission research is retained in
[zero-WAR evidence](../results/m10-permission-zero-war-06.json.gz) and
[model-delivery research](../results/m10-permission-model-war-07.json.gz).
Optional-origin native approval/revocation was not verified and is not shipped.

## 4. Execution world

Production runs only in ISOLATED. Actual installed-extension activation proved
video DOM access, WebGPU creation and fixed model loading there. No MAIN bridge,
page messages or DOM marker can grant privileged authority. The singleton is
private to the extension's isolated global; its key includes the extension ID.
MAIN-world code in offline runners is a read-only media observer, not shipped
extension code. Native automation uses the real extension action, not a page
module import masquerading as installation.

## 5. Build and CSP

`npm ci` then `npm run build:extension` produces ignored `dist-extension`.
Load unpacked through Chrome's extensions page in Developer mode. Bundled worker,
content and popup have local code/WGSL and the fixed JSON; no remote imports,
eval, WASM exception or source maps. CSP is `script-src 'self'; object-src 'none'`.
esbuild 0.28.2 and erased @types/chrome 0.2.9 are MIT, with verdicts in ADR-0045.

Measured production payload: **263,008 bytes**, digest
`724507ec7d3e7a8c9a4ce7bf0fc772f2d2de9f56c1218693db52d6cba195258a`.
This digest covers sorted payload filename/hash pairs; it is not a ZIP digest.
Payload size excludes the generated provenance JSON. Provenance separately pins
source commit/dirty state, manifest, individual artifact bytes and model hash.
Later tool/documentation commits preserve this runtime payload, but are not
misrepresented as fresh native executions. The diagnostic build is separate,
263,295 payload bytes, and is not the shipped acceptance artifact.

## 6. Activation model

Open the native action on an HTTP(S) page and choose **Enable current page**.
Unactivated installation injects no content observer, timer or GPU runtime.
Activation is temporary for that document; navigation does not auto-enable the
next document. **Disable** tears down owned resources and restores original
visuals. Stored mode is a preference, not injection authority. A new document
requires another explicit activation. No store publication was attempted.

## 7. Video discovery

Top-document and accessible open-shadow discovery uses batched mutation/media
events, not a full-DOM frame-rate scan. Eligibility requires decoded readiness,
connection, supported media/geometry and at least 160x90 CSS pixels of useful
visible area. Late insertion and late-ready open-shadow media passed native
journeys. Non-composed media events have root-local listeners, removed when a
root detaches or activation ends. A low-frequency safety reconciliation remains;
its measured discovery/geometry work is reported separately below.

## 8. Multiple-video ownership

Playing, visible, largest-area candidates lead, with stable insertion order for
ties. An eligible owner is retained until another playing candidate has at least
25% more visible area for 750 ms; invalid ownership detaches immediately.
Generation fences dispose stale asynchronous model/device completions before
replacement. Native size/playback switching, reparenting and two enabled tabs
passed. At most one expensive runtime was live per tab. An unenabled second tab
remained untouched; enabled tabs had independent modes and visibility suspension.

## 9. Frame/iframe scope

| Surface | Shipped behavior / evidence |
|---|---|
| Top document | Supported eligible video ownership |
| Open shadow root | Accessible roots supported; late media and listener cleanup passed |
| Same-origin iframe | Not owned; explicit embedded-frame scope limitation tested |
| Cross-origin iframe | Not owned; no permission escalation; scope limitation tested |
| Closed shadow root | Not inspected or pierced; contents cannot be claimed detected |

Research injection capabilities do not imply shipped iframe support. The popup
reports uninspected embedded frames rather than asserting that no video exists.

## 10. Media security matrix

| Media case | Actual result |
|---|---|
| Same-origin clear MP4 | Production external route and diagnostic sampled route passed |
| Valid cross-origin CORS | Both import routes passed without changing the page's media attributes |
| Non-CORS cross-origin | Explicit rejection on both routes; original playback preserved |
| Clear MSE fixture | Supported native journey passed |
| Attached ClearKey MediaKeys | Detection and rejection passed on clear media |
| Actual encrypted stream | **UNVERIFIED**; the MediaKeys fixture is not encrypted-stream evidence |

Host permission is not decoded-pixel permission. No crossorigin mutation, source
replacement, proxy, alternate-API retry after SecurityError, DRM bypass or
web-security disabling. Native automation flags include unpacked-extension
debugging, autoplay and window controls; these are apparatus flags, not extension
permissions. No unsafe WebGPU/Dawn timing override was used for M10 performance.

## 11. Model/WebGPU loading

Production C16D2 remains 6,291 parameters, 140,467 JSON bytes, SHA256
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.
The build refuses changed model bytes. The worker serves only this fixed package
resource; content cannot ask it to fetch a URL. Live model/packed-weight checks
also accompany output parity. WebGPU feature negotiation is shared with the
harness. Diagnostic native runs exercise withheld optional features, forced-copy
import and actual `GPUDevice.destroy()`. Device loss is terminal: original
restored, resources released, no automatic reacquisition loop.

Pipelines, shaders, samplers, uniforms and reusable textures/bindings are created
in configuration. The frame task imports, binds, encodes and submits synchronously.
Only mandated transient WebGPU objects are allocated there. Timestamp readback
does not block frames; there is no production pixel readback.

## 12. Overlay geometry

The owned sibling canvas is aria-hidden and pointer-transparent. Intrinsic x2
backing dimensions are independent of CSS display size. Supported local fixtures
passed contain/cover, object positioning, rectangular clipping, video radius,
translation, scrolling, native resize and owner changes. Geometry mutations
revalidate eligibility; unsafe layout reveals original rather than guessing.
The adapter neither replaces nor reparents page nodes and does not mutate their
styles. DOM/style snapshots exclude only reference-owned extension nodes.

Unsupported rotation/skew/perspective, unhandled ancestor effects/fixed containing
blocks, uncertain stacking and rounded ancestor overflow clipping remain explicit
limits. Generic hit testing alone cannot establish passive-caption visibility.

## 13. Page controls/captions

Native video controls and showing native text tracks are explicitly unsupported:
a sibling canvas cannot generically reproduce their paint order. Supported custom
controls were activated by trusted native clicks and remained reachable/visible.
Passive captions after an auto-stacked video and preceding positive-stacked
captions passed. Uncertain preceding auto/zero-stacked passive captions reject
with `unsupported-controls`; their original remains visible. Live style, z-index
and caption mutations also passed rejection/restoration checks.

Local browser raster screenshots were visually inspected for nonblank output,
alignment, captions, controls, scrolling, ownership, fullscreen and device-loss
restoration. OS-native screen capture was unavailable due capture permission;
browser raster evidence does not prove every native compositor/top-layer detail.
No public copyrighted frames are redistributed.

The manifest pins seven original PNG hashes and their reduced JPEG derivatives
(maximum dimension 600 px, quality 45). These lossy illustrations establish
layout context, not pixel equality: [normal](visual/m10-22-normal.jpg),
[scroll](visual/m10-22-scroll.jpg), [caption](visual/m10-12-caption-after-auto-passive-white.jpg),
[unsupported caption](visual/m10-18-unsupported-caption-before-auto-passive.jpg),
[owner switch](visual/m10-25-owner-size-switch.jpg),
[container fullscreen](visual/m10-33-fullscreen.jpg),
[device-loss original](visual/m10-44-device-lost-original.jpg).

## 14. Fullscreen

Page-owned **container fullscreen** passed with eligible sibling canvas/control
subtree retained and geometry recomputed. Shaka's original container-fullscreen
control also passed. **Direct-video fullscreen** suspends the enhancement and
shows the original; it is not enhanced fullscreen. No page API interception,
fullscreen hijack or original style change. Local fixture fullscreen preserves
the fixture's original sizing, not a newly imposed full-bleed layout.

## 15. Picture-in-picture

Native PiP entry/exit passed the original-video fallback journey. Enhancement is
suspended in PiP and resumes only when original eligibility permits. No PiP
capture/replacement API, video clone or enhanced-PiP claim.

## 16. SPA/source lifecycle

Native production: **40 PASS, 1 UNVERIFIED**. Diagnostic build: **44 PASS,
1 UNVERIFIED**. These are journey counts, not 84 distinct production scenarios.
Archives: [production](../results/m10-journeys-production-04.json.gz),
[diagnostic](../results/m10-journeys-test-07.json.gz),
[reload retry](../results/m10-journeys-reload-05.json.gz).

Late insertion, SPA/source replacement/back, identity-preserving removal/reinsert,
reparent then old-parent removal, bfcache handling, origin navigation, worker
stop/restart and repeated enable/disable passed. Twenty cycles returned owned
device/pipeline/canvas/listener/observer/frame-callback counts to zero and preserved
page DOM/styles. Hostile owned-canvas removal is bounded terminal rejection, not
a recreation fight. Playing seeks/buffering hide stale output but remain active
in the shared runtime clock; pause/hidden/offscreen freeze active time.

**Whole extension reload remains UNVERIFIED.** Old-worker closure is observed,
but Chrome's native `Extensions.triggerAction` wakeup times out after reload.
This is not a proven runtime defect, nor a pass inferred from successful worker
stop/restart or duplicate enable. A diagnostic startup CDP timeout produced zero
journeys and is retained separately, not counted as an application pass.

## 17. User modes/settings

Auto, Prefer neural and Baseline retain the shared M9 behavior. Auto uses measured
runtime evidence and hysteretic probes, not scene-quality prediction. Prefer
neural is not a security override; fatal/media/geometry limits still stop output.
Baseline never probes. Versioned local storage saves mode per origin only;
unknown settings schemas default safely. Worker restarts query content state;
two tabs do not share a GPU owner. No persistent per-site activation feature is
advertised without a verified permission workflow.

## 18. Security review

Closed message schemas validate extension sender ID, popup URL/role, active tab,
HTTP(S) origin, top frame and exact registered document ID. Async completions
remain document/origin/generation-bound across navigation. Content can request
status or the fixed model, not arbitrary fetch/injection. Page DOM markers are
not ownership credentials; only actual retained references authorize removal.

Production excludes `__AETHERVSR_EXTENSION_TEST__`, benchmark entry points,
debug flags and public privileged hooks. Offline parity serialization uses
diagnostic evaluation only outside shipped code. Final independent static review
found no new P0/P1 security/ownership issue. This scoped review is not a claim of
exhaustive exploit resistance; native reload and encrypted-stream gaps remain.

## 19. Extension overhead

Environment: Mac17,2 Apple M5 MacBook Pro, 24 GB; macOS 26.6.2, Darwin 25.6.0;
Chrome for Testing **153.0.8010.12**, executable SHA256
`8319963f6625accf51c0dd4f55091ceaf9f09ed39e7a52fed4fae12b2a6b668a`.
1200x820 viewport, genuine headful foreground, AC power, default timestamp
quantization. Display refresh, energy, temperature and actual GPU residency:
**not measured**. Local CFR H.264/MP4 is 1280x720, 60/1 fps, SHA256
`8d81acbe164da1d62b7d0d02a3cc66915c96e8aa90d45cac34d818fc33df1d4a`;
external import, x2 2560x1440 output, fixed production neural stage.

Five-second readiness warmup precedes each independently fixed 30-second window.
Pipeline rates divide opening/closing session counter deltas by the wall
observation interval; native observer rates use its own synchronous boundaries.
The active-clock delta independently gates duration. Decode stalls remain in active time. GPU
quantiles select raw submitted-in-window upscale timestamps, including cold
samples; late readbacks drain without extending rate denominators. This brackets
the upscale stage, not decode/import, queue wait or browser presentation. Default
quantization means these are not directly comparable to M9's Dawn-override times.

| Arm | Wall / active ms | Native callback fps | Rendered / runtime presented fps | Skips / decoder drops | Combined loss |
|---|---:|---:|---:|---:|---:|
| No extension | 30002.2 / not measured | 53.328534 | not measured | not measured / 1 | not measured |
| Installed, inactive | 30000.8 / not measured | 54.665027 | not measured | not measured / 1 | not measured |
| Extension Baseline | 30000.5 / 30001.0 | 56.465161 | 56.465726 / 59.899002 | 103 / 97 | 11.129661% |
| Harness Auto | 30001.6 / 30002.0 | 58.863331 | 58.863527 / 59.963469 | 33 / 29 | 3.446359% |
| Extension Auto | 30001.3 / 30001.6 | 58.163565 | 58.164146 / 59.730745 | 47 / 45 | 5.133929% |

| Active arm | GPU samples / rendered frames | GPU p50 / p95 ms | Core CPU p50 / p95 ms | Driver p95 ms | Discovery + geometry ms/s |
|---|---:|---:|---:|---:|---:|
| Extension Baseline | 1687 / 1694 | 3.407872 / 4.325376 | 0.1 / 0.3 | 0.1 | 0.236663 |
| Harness Auto | 1758 / 1766 | 5.373952 / 6.291456 | 0.1 / 0.2 | 0.1 | not measured |
| Extension Auto | 1737 / 1745 | 5.898240 / 7.012352 | 0.1 / 0.2 | 0.1 | 0.189992 |

Core CPU measures import/encode/submit, not GPU execution. Driver callback CPU
measures session/cadence/controller plus its nested attachment callback after
submission. Do not add that nested callback again. Discovery and geometry are
measured synchronous function brackets; MutationObserver handler work and other
adapter/browser work are excluded. These are not total renderer CPU. Timer values
rounded to zero mean below resolution, not zero cost. Model initialization alone
is not measured; readiness includes navigation, popup and readiness polling.

The [frozen calibration](M10-CALIBRATION-PLAN.md), committed at `c6c6c25` before
final observations, yields these final relative checks:

| Extension Auto minus harness | Observed | Frozen allowance | Verdict |
|---|---:|---:|---|
| GPU p50 | +0.524288 ms | +0.6 ms | PASS |
| GPU p95 | +0.720896 ms | +1.3 ms | PASS |
| Driver p95 | No resolved difference | +0.1 ms | PASS |
| Discovery + geometry | 0.189992 ms/s | 1 ms/s | PASS |
| Native callback rate deficit | 0.699766 fps | 1.201084 fps | PASS |
| Native presented rate deficit | 0.233121 fps | 1 fps | PASS |

No persistent 50-ms discovery/layout call was observed in the recorded brackets.
These are bounded observed-noise allowances, not confidence intervals or an
absolute-loss waiver. All three active short arms fail the 1% loss reference.
The cheaper baseline did not improve delivered FPS here; this does not identify
decode, GPU or compositing as the cause. Harness CSS/UI and extension's 640x360
fixture composition differ. Active arms also have extra runtime instrumentation;
inactive arms do not establish zero extension CPU. Full raw short traces and
all sample scopes are in [the report archive](../results/m10-final-performance-02.json.gz).

## 20. Harness parity

Actual installed production content and the actual standalone harness replayed
the same paused source at 1, 2 and 3 seconds. Same-source decoded input hashes,
live model/packed weights, precision, import route, stage/options and output
extent matched. Exact normalized RGBA8 output hashes matched **3/3 production
external** and **3/3 diagnostic forced-copy** cases. Output was nonuniform and
opaque. Archives: [external](../results/m10-output-parity-production-01.json.gz),
[sampled](../results/m10-output-parity-copy-01.json.gz).

This is instrumented, offline one-frame replay through each existing configured
pipeline. The tool temporarily adds swap-chain COPY_SRC, submits exactly one
frame in one task, copies the actual output, then maps staging outside the frame
callback and restores/destroys temporary resources. BGRA is normalized explicitly.
No full public frames are captured. Byte-delta statistics are **not measured**;
full-hash equality, not invented delta values, is the comparison criterion.
This is within-route harness/extension equivalence, not cross-import equality,
uninstrumented live-output capture, PSNR, new SR quality or performance evidence.
Unit mocks verify apparatus contracts, not hardware pixel correctness.

## 21. Long-run extension result

Final valid capture at source `b75ded5c08e30735239dbee2424acf9ce5b04552`,
production payload above. Same environment/clip as section 19, five-second warmup.
Raw [600-second trace](../results/m10-final-performance-05.long.json.gz) and
[summary](../results/m10-final-performance-05.json.gz) independently validate and
recompute exactly. Active duration is 600001.5 ms, wall duration 600000.8 ms;
small boundary quantization/placement differences are retained, not rounded up.

| Gate or observation | Measured result |
|---|---|
| >=600 active seconds | 600.0015 s: PASS |
| >=58 rendered / presented fps | 58.351589 / 59.811587: PASS |
| Submitted / runtime presented | 35011 / 35887 frames |
| Callback skips / decoder drops | 876 / 847 |
| <=1% combined loss | **4.8011814863%: FAIL** |
| Neural / baseline frames | 35011 / 0 |
| Tier changes / owner changes / errors | 0 / 0 / 0 |
| Controller state records | 299, including normal loop rewarming; not 299 fallbacks |
| GPU samples / p50 / p95 | 34862 / 6.488064 / 7.667712 ms |
| GPU mean / maximum | 6.468955 / 9.568256 ms over those available samples |
| Core CPU p50 / p95 | 0.1 / 0.2 ms over 35011 frames |
| Driver CPU p50 / p95 | Below timer resolution / 0.1 ms over 35011 callbacks |
| Discovery + geometry | 597 calls, 125.6 ms total, 0.209333 ms/playback-second |
| Runtime callback latency p50 / p95 | 7.4 / 9.2 ms over 35008 available samples |
| Integrity events | None; all required boundaries visible/focused |

Registered combined loss is `100 * (skips + decoder drops) / presented frames`.
The counters may overlap; this is not an estimate of mutually exclusive lost
frames and neither counter is subtracted from the other. Runtime rendered counts
successful submissions, not an independent compositor display count.

| Slice | Rendered / presented fps | Skips | GPU sample count | GPU p50 / p95 ms |
|---|---:|---:|---:|---:|
| First 120 s | 58.641667 / 59.800000 | 139 | 7007 | 6.881280 / 7.864320 |
| Last 120 s | 57.858333 / 59.808333 | 234 | 6914 | 6.029312 / 6.946816 |

The last slice falls below 58 fps although the aggregate passes. Decoder slice
counts use outward native observations, with exact coverage endpoints in raw
data, not interpolated 120-second counts. GPU times improved while delivered
rate fell; no thermal/bottleneck attribution follows. No further valid run was
selected to improve this unfavorable result.

Earlier attempts remain visible: final-01 closed before measurement; final-02's
short rows are valid but its long row lost focus at about 16.1 s; final-03 lost
focus at about 149.8 s; final-04 aborted at about 51.8 s and identified VS Code
as the frontmost app. User-confirmed foreground retries did not relax integrity.
Final-05 used a separate process and completed. Full failed-run reports are
archived; large invalid raw sidecars remain locally retained and hash-indexed in
[m10-evidence.json](../results/m10-evidence.json), not represented as distributable
raw evidence or valid performance. Calibration-01/02 active-time defects and
their later repair are separately documented in the calibration plan.

## 22. Third-party page results

Actual public pages, original controls and unmodified source/CORS/CSS:

| Page/player | Result | Observed scope |
|---|---|---|
| https://plyr.io/ | NEGATIVE | Rounded ancestor overflow: only rectangular ancestor clipping supported |
| https://videojs.org/ | NEGATIVE | Ancestor changes fixed positioning or has unsupported effects/slotting |
| https://shaka-project.github.io/shaka-player-release/demo/ | PASS | Existing clear asset "Big Buck Bunny: the Dark Truths"; 180.026923 s supported observation |

[Plyr/Video.js archive](../results/m10-sites-public-01.json.gz) and
[Shaka archive](../results/m10-sites-shaka-04.json.gz) pin source commits, exact
bundles/browser, real RuntimeSession snapshots and errors. Shaka seek, 1.5-second
pause hold/resume, 1024/1200-width resize and original container-fullscreen passed.
Scroll-down **and scroll-back are N/A** because the document had no scroll extent.
Historical raw incorrectly marks the no-op return as PASS; the manifest records
the correction, the runner now propagates N/A, and no historical raw was rewritten
or new public run claimed. Source replacement on this public player was not tested.

Earlier Shaka attempts are retained: metadata-only readiness did not establish
usable controls; a later approximately 75-second run stopped on unscrollable
layout. The final runner waits for decoded readiness and records no-op actions
as limitations without forcing page layout. Public records redact full media URLs
and contain no redistributed player frames. These are compatibility observations,
not calibrated performance, image-quality or sustained-overhead claims. **One
successful public player imposes a PARTIAL ceiling.**

## 23. Unsupported cases

Explicit limits: non-HTTP(S)/restricted pages; unactivated documents; subframe or
closed-shadow ownership; missing WebGPU or required capabilities; protected/media
security failures; native controls/showing native tracks; unsupported transforms,
effects, clipping and stacking; insufficient/offscreen video; direct-video
fullscreen or PiP enhancement; hostile canvas mutation. Recoverable visibility
suspension and terminal rejection are distinct states. Original playback/audio
remain available; unsupported is never a reason to weaken browser security.

Whole-extension reload, actual encrypted streams and native OS screenshots are
**unverified**, not silently categorized as implemented successes. Cross-vendor,
other browser builds, HDR/general color fidelity, whole-renderer CPU, temperature,
energy and new real-page SR quality are **not measured**.

## 24. Reviewer findings

Independent reviews covered permissions/world, message authority, lifecycle,
geometry, native evidence, parity and acceptance. Earlier corrections included
terminal shared teardown, bfcache/source restoration, import error classification,
playing-seek/readiness clock accounting and native capture limitations. Affected
initial runs remain historical, not promoted into final acceptance.

Final P2 geometry/ownership findings were fixed before renewed native coverage:
passive-caption paint order, same-video reinsertion mistaken for hostile canvas
removal, and missing non-composed late media events inside open shadow roots.
Focused units and the final production/diagnostic journeys passed afterward.
Actual extension reload still times out at native action wakeup and remains open.

Final independent security/ownership review reported no new P0/P1 and validated
those P2 repairs. Independent parity/performance/public review requested one
evidence-label correction: propagate N/A to scroll-back. That is fixed with the
historical interpretation explicitly corrected in section 22. No valid performance
trace was invalidated by that reporting-only change. Apparatus fail-fast focus
capture has its own unit regression; it does not force focus or forgive loss.

Independent final report and packaged-evidence review returned PASS without
additional findings. Closing executable and publication gates remain separate;
no review substitutes for an unperformed gate.

A final direct arithmetic audit corrected documentation that had described
pipeline rate denominators as active time. The actual summarizer uses the wall
observation interval and independently checks active duration. Every active short
and long rate was rechecked against raw counts; numbers, traces and verdicts did
not change. This reporting correction supersedes the wording in the publication
checkpoint, not its recorded measurements.

## 25. Extension verdict

**EXTENSION MVP PARTIAL**

Real supported local enhancement, bounded ownership/teardown, same-pipeline
output parity and one actual public player establish functional integration.
The valid ten-minute loss gate **fails**, the last 120-second rate is below
58 fps, two public players reject geometry, and whole-extension reload is
unverified. Relative GPU overhead passes cannot turn this into READY. The
verdict is not softened by implementation effort or passing unit tests.

## 26. Public claim scope

AetherVSR has an opt-in unpacked MV3 prototype using the unchanged local C16D2
pipeline on supported non-DRM top-document video, including accessible open
shadow roots. Tested M5/Chrome behavior preserves original playback and restores
original visuals at unsupported boundaries. One tested public player integrated
successfully. On the recorded ten-minute M5 clip it delivered 58.35 rendered fps
but failed the registered combined-loss target at 4.80%.

No universal 60-fps/stutter-free claim, broad player compatibility, new quality
advantage, cross-vendor success, enhanced PiP/native controls, store release,
zero CPU overhead or guaranteed zero-copy claim is supported.

## 27. Commits

Actual M10 history after the starting SHA, oldest first:

| Commit | Change |
|---|---|
| `c76ba19` | Architecture/permissions preregistration |
| `8fd7708` | Terminal, complete shared attachment teardown |
| `4183ee4` | ISOLATED MV3 activation/shared-runtime adapter |
| `df8bc4d` | Source visual restoration across media/geometry changes |
| `c7f0731` | bfcache and native capture limitations |
| `2b82253` | Native lifecycle evidence and both CORS routes |
| `9a4f1ea` | Paired calibration plan and native runners |
| `d47e504` | Playing seek active accounting |
| `005e4e0` | Transient media readiness active accounting |
| `c6c6c25` | Frozen measured overhead allowances |
| `e7cb51b` | Shadow readiness, reinsertion and caption ownership fixes |
| `504a797` | Extended native lifecycles and actual output parity |
| `760cc3e` | Native action wakeup for reload attempt |
| `97108a7` | Shaka public controls/fullscreen runner |
| `67cee27` | Decoded public media readiness |
| `f0b8b57` | Non-scrollable public action limitation |
| `b75ded5` | Fail-fast invalid foreground recording and focus diagnosis |
| `1bebcef` | Public scroll N/A propagation and explicit MV3 CI build gate |
| `f5e5af0` | Measured PARTIAL report and native/visual evidence archive |
| `776e02b` | Fresh-clone gates and deterministic production/diagnostic artifacts |

Production journeys pin `504a797`; diagnostic journeys pin `760cc3e`; Shaka
pins `f0b8b57`; final long performance pins `b75ded5`. Their payload identity
is checked separately from source/provenance identity. Closing evidence and
documentation commits follow this ledger; `git log --oneline
2c62a2c..HEAD` gives all publication/closure commits without requiring this file
to self-pin a commit hash that changes when the file changes.

## 28. Final repository state

Evidence packaging includes raw valid short/long traces, lossless native/parity/
public archives, hashes and explicit invalid-attempt retention limits. Every
archived JSON hash and all six valid performance summaries were independently
recomputed. No generated media, browser binary, model replacement or installable
bundle is tracked. The 48 MiB tree / 8 MiB per-file restrictions remain binding.

Separate final `npm ci`, typecheck, lint, production harness build, **629 Vitest
tests / 33 files**, **308 Python tests**, production extension build and diagnostic
extension build passed locally (Node 26.5.0, established Python environment).
The existing two moderate npm advisories remain; no force-upgrade was attempted.
Tool `.mjs` files are outside ESLint's configured coverage: syntax, unit execution
and focused actual scroll-action behavior checks provide their relevant validation.

Fresh local clone of evidence commit
`f5e5af0ee8cf541e221218d4ea46f28910d0ede8` passed separate install, typecheck,
lint, 629 Vitest tests, build and 308 Python tests. It installed its own npm
dependencies and reused the existing Python environment; no ignored generated
media/browser files or GPU measurements were imported into clone validation.
Two production builds matched every file including provenance and reproduced the
measured payload. Two diagnostic builds also matched. Manifest/CSP, exact model
and production debug stripping passed; clone worktree remained clean. The
[reproducibility record](../results/m10-reproducibility.json) pins these artifacts.

At that evidence commit, the tracked tree is **50,265,261 bytes**, below
50,331,648 (48 MiB), with existing per-file exceptions unchanged. The local fresh
clone's `.git` occupied **442,172 KiB**, including history: this is not network
transfer size or installed-dependency size. No history rewrite was performed.

Publication commit `776e02b2a410af49b6595ea9edd83c3e86a86481` was pushed to
`main`; origin/main matched that SHA. Exact-SHA GitHub Actions
[34813673799](https://github.com/yassinsolim/AetherVSR/actions/runs/34813673799)
completed **success**, both `gate` and `fusion`, including the explicit MV3
build. The current documentation-only closure corrects the rate-denominator
wording without modifying runtime or raw results. It must independently meet
clean `main == origin/main` and exact-HEAD CI after publication; its final SHA,
tracked size and CI result are returned in the completion message rather than
self-pinned in a file that would change its own commit hash. Earlier CI success
is never substituted for that closing check.

## 29. Recommended Milestone 11

The shared runtime/ISOLATED ownership foundation is useful, but a broad validation
rollout should first close or explicitly disposition M10's loss and native-reload
gaps and establish a second supported public player. Then preregister measured
cross-vendor/browser validation with the same model, source hashes, absolute
gates, media-security boundaries and negative-result retention. Do not extrapolate
M5 timings to other hardware. No M11 implementation was performed.