# M10 MV3 extension architecture and acceptance

## Starting evidence

Clean main and origin/main both equal
`2c62a2cf27decc6c88c8636ee85be1e76544a63b`. Separately executed npm ci,
typecheck, lint, 421 Vitest tests, build and 308 Python tests passed. Exact-SHA
CI 34746007298 completed/success. Two moderate npm advisories predate M10.
The standalone harness remains a separate consumer of shared runtime code.
Production model SHA256 is immutable:
`d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a`.

No training, R3, interpolation, temporal processing, native backend, cloud,
DRM circumvention, media-security weakening, store publication or M11 work.
M9's short-window loss failures remain limitations, not erased by integration.
M9's final copy-import contract is prepared sampled views in configure, not
the lazy binding described in the M10 request. Preserve the actual tested
contract; no extra hot-path allocation or pixel readback is authorized.

## Architecture before implementation

MV3 0.1.0: service worker handles validated popup commands, origin-keyed mode
settings and injection. It owns no video, GPU or frame timer. ISOLATED content
code owns discovery, one attachment, geometry and status; it imports the same
RuntimeDriver, RuntimeController, VideoPipeline and production upscalers as
the harness. No MAIN-world bridge, page messages or privileged page API.

Start with activeTab, scripting and storage, no permanent host permissions.
Before freezing the manifest, use a real load-unpacked research extension to
test action/activeTab, top/same/cross-origin frame injection, isolated WebGPU,
packaged model access and optional-origin grants. Optional persistent activation
is added only if those APIs are exercised and tested, not just advertised.
No arbitrary fetch, URL opening or executable code commands exist in messages.
No externally_connectable, eval, remote JS, remote weights, WASM exception or
MAIN-world execution is permitted. A web-accessible resource may be considered
only after demonstrating necessity; no other security exception is implied.
Permission research proved fixed worker model delivery without WAR. M10 uses
one-time activeTab activation and origin-keyed mode storage only; persistent
optional grants are deferred because native approval/revocation was not verified.

M10 ownership scope is top document, including accessible open shadow
roots if safely observable. Subframes are not owned: permission research does
not authorize iframe runtime support in this milestone. Popup reports uninspected embedded
frames rather than asserting no videos. Closed-shadow contents are not observable;
report the discovery limitation, never falsely claim detection of their video.
No DOM monkey-patching to pierce those boundaries.

Only top-frame injection can create an owner. Idempotent reinjection communicates
with the existing isolated singleton; stale marked roots are removed only after
verifying actual ownership through retained isolated state, not a spoofable
DOM marker alone. Unknown marked DOM is never removed as though page-owned
attributes conferred authority. Worker state is not a GPU
lease dependency: content ownership survives worker unload; popup queries it.
All asynchronous attach/model/device completions are generation-fenced and
disposed before a newer owner may acquire resources. Disable stops discovery,
timers, callbacks, GPU resources and removes only owned DOM.

## Discovery and geometry

MutationObserver handles additions/removals/source/attribute changes, batched
outside the video frame callback. Candidate eligibility requires connection,
ready decoded pixels, meaningful visible area (at least 160x90 CSS pixels),
supported geometry/media and no explicit exclusion. Choose playing, visible,
largest area, then stable insertion order. Retain an eligible current owner
unless another playing candidate has at least 25% more visible area for 750 ms;
invalid/removed owner detaches immediately. No full-DOM frame-rate polling.

The original video remains authoritative for playback, audio, source, seek and
page controls. Never replace, reload, clone or alter its crossorigin attribute.
An owned aria-hidden canvas, pointer-events:none, overlays its displayed image.
No style mutation, replacement or reparenting of page-owned nodes. Event-driven geometry observes
resize, scroll, video dimensions, fullscreen and attributes, with a bounded
low-frequency safety reconciliation only if measured necessary. Intrinsic
source x2 backing dimensions are distinct from CSS display dimensions.

Support will be claimed only for tested axis-aligned layouts and object-fit
contain/cover/position, translated/scaled video, rectangular clipping and radius.
Reject perspective/3D, rotation/skew, unhandled filters/masks or stacking that
would hide page controls/captions, including controls/captions appearing or
changing stacking during playback. Unproved visibility means no overlay;
pointer passthrough alone does not prove visual accessibility. Native video controls and showing native
track cues may require explicit unsupported status if a generic sibling overlay
cannot keep them visible. Container fullscreen may retain the sibling subtree;
direct-video fullscreen and native PiP suspend enhancement and reveal original
video. No fullscreen/PiP capture or global API interception.

## Lifecycle mapping fixed before implementation

| Event | Action | Preserved |
|---|---|---|
| Explicit enable | Discover and attach one eligible owner | Origin mode |
| Source/resolution change | Existing shared workload reset; verify media/geometry again | User mode |
| Seek/loop | Shared evidence invalidation/rewarm | Mode, tier/backoff rules |
| Pause/hidden/offscreen | Adapter visibility suspension; show original | Mode and remaining backoff |
| Visible/play | Shared rewarm when media/geometry are supported | Mode |
| Container fullscreen | Recompute within same subtree or suspend | Video identity |
| Direct-video fullscreen/PiP | Suspend visual replacement, original remains | Mode/backoff |
| Owner removed/replaced | Destroy old runtime before new attachment | Origin mode |
| Overlay removed/hostile mutation | Detach; bounded recovery, no recreate loop | Explicit unsupported reason |
| Device/media SecurityError/EME | Terminal attachment rejection and restoration | Failure reason, no automatic retry |
| Disable/navigation | Destroy callbacks/observers/resources and owned canvas | Persisted origin mode only |
| Worker restart | Content continues; popup queries current content | Active runtime and settings |

Media-security boundaries are final. Same origin and valid CORS are tested on
both import routes; non-CORS and mediaKeys/protected cases reject without trying
another API to bypass failure. Host access permits injection, not forbidden
decoded-pixel access. Unsupported state preserves original playback.

## Build and messages

Reproducible build emits dist-extension, ignored by Git, with bundled local
worker/content/popup, locally packaged WGSL and exact production JSON copied
mechanically from its single repository source. Source maps omitted deliberately
from the installable bundle. Build twice and compare file hashes; source commit,
manifest version, bundle digest and model digest pin every reported test.
Strict MV3 script-src self and object-src none; no eval/WASM/remote exception.

Messages are closed discriminated schemas: status, enable, disable, set-mode
and only explicitly supported site-setting operations. Validate sender extension
ID, popup URL, tab/frame and HTTP(S) origin. Popup targets are checked against
the browser's current active tab; content messages carry only status or the
fixed model request and must come from the injected top-frame document. Bind
async completions to tab/document/origin and reject navigation-stale messages.
Stored preferences never grant injection authority; unknown settings schemas
default inactive. Content cannot command arbitrary
injection or fetch. No per-frame worker traffic. Settings schema version 1 stores
origin-keyed mode/explicit enable permission only; no video/full-page URLs,
history, secrets or page localStorage. Unknown schema safely defaults.

## Validation and acceptance before final measurements

Use canonical Vitest plus complete Python suite; add built manifest/CSP/model
hash/debug exclusion tests. Actual Chrome load-unpacked extension, not page
module imports, validates integration. ISOLATED WebGPU must be proved; failure
means unsupported/NOT READY, not an alternate execution world. Headful native context with genuine
visibility, exact browser and build hashes; no disabled web security.

Local fixtures cover native controls, custom controls/captions, multiple owners,
late insertion, SPA/source replacement, iframe scope, CORS/non-CORS, container
and direct fullscreen, PiP if reliable, object-fit/position, clipping/radius,
scroll, removal/reinsertion and hostile mutation. Before/after page DOM/styles
must match excluding owned nodes and changing media playback counters. Repeated
attach/detach (20 cycles) must return live owned resource/callback/observer counts
to zero; no simultaneous runtimes. Native-controls limitation is explicit if
unsupported, not hidden by clicks passing through an opaque overlay.

Security/permission gates: unapproved pages have no injected runtime/observer/
GPU/timer. An explicit activation obtains access; non-CORS/EME failures restore
original pixels, report precise states, and never retry through a different
import route. Revocation, if optional site access is implemented, unregisters
scripts/settings and stops the affected attachment. Worker restart and two-tab
tests prove content continuity and tab-local state.

Before final performance tests, calibrate repeated paired harness/extension
normal runs to set p50/p95 overhead tolerances above observed run noise. Commit
those tolerances and CPU/discovery/geometry thresholds separately, not amend
this document after seeing final results. Compare no extension, installed idle,
activated Baseline and activated Auto on the same exact-CFR source. Record
frames, decoder loss, callback skips/latency, raw neural timings, owned CPU and
geometry/discovery costs, owner/controller transitions and errors. No per-frame
messages/layout queries; no unmeasured zero-cost claim.

Final supported extension run: >=600 active foreground seconds on base M5
720p60, one owner, no false fallback/GPU/geometry errors. Target >=58 rendered
and presented FPS and <=1% combined loss, with M9 source-specific caveats and
paired harness comparator retained. Pixel comparison uses identical source
frame/model/precision/import with exact RGBA8 equality expected, or a justified
unchanged numerical tolerance preregistered before that comparison.

Test at least two actual public non-DRM third-party pages, conventional and
dynamic/custom where available, for several minutes if supported. Include
seeking/controls/scroll/resize and feasible fullscreen/source changes. Negative
site results are valid findings. No copyrighted frame redistribution; local
fixture screenshots are the committed visual evidence. No objective SR-quality
claim on unknown real-page ground truth.

Independent review covers architecture/permissions/world/security, ownership/
teardown, geometry/media restrictions, build provenance, calibrated performance,
real pages and final claims. A result-affecting P0/P1 invalidates affected runs.
Verdict is EXTENSION MVP READY, PARTIAL or NOT READY based on evidence, not effort.
READY requires all supported-scope/security/lifecycle/performance gates and
successful integration on both real pages. Functional PARTIAL requires real
enhancement on supported fixtures and at least one third-party page; zero
successful public-page attachments means NOT READY. Security/ownership failures
cannot be excused as unsupported scope. Native-controls/frame/PiP limitations
must be user-visible. The 600-s >=58-fps and <=1%-loss targets are literal
gates; any failed gate remains failed even if a different window passes.
All 29 requested report sections, clean-clone builds, separate final gates,
clean pushed main == origin/main and exact-closing-HEAD CI are required.
48 MiB tracked cap and 8 MiB/file remain; no bundle/media/cache commits or
history rewrite. M11 remains a recommendation, never implementation.