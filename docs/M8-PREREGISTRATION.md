# Milestone 8: Budget-matched architecture confirmation

This document and `results/m8-plan.json` must be committed before any M8
production-budget training starts. The runner must record that commit, require
it to be an ancestor of the execution commit, hash its inputs and source, and
refuse to overwrite a run. Neither a date string nor a model's self-declared
training metadata alone proves chronology. No M8 model existed when this plan
was written. Smoke models are explicitly non-candidates and cannot enter a
binding result.

## Question and scope

Does the body-only R3 advantage over R0 survive **81,180 optimizer updates**
when both arms share the initial effective function and training stream?

Only R0 (3x3) and R3 (3x3 + 1x1 + identity + 1x3 + 3x1) are trained. Both deploy
as the existing C16D2, scale-2, 6,291-parameter graph. No R1/R2 retraining,
stem/head branches, BatchNorm, new losses, distillation, deeper network,
additional branches, temporal SR, quality selection, or extension work belongs
to this experiment. M7 remains closed; its rules and historical evidence are
not edited by M8.

The 16,200-update M7 result is not evidence about 81,180 updates. Literature
motivates measurement, not a predicted direction or effect size.

## Starting evidence

- Required and observed starting branch: clean `main`.
- Starting HEAD and tracked `origin/main`:
  `be1e8cbc88feb51854334337fe09c8ff1dba1941`.
- Separately run: `npm ci`, typecheck, lint, **168 Vitest tests**, and build:
  all passed. Installation reported two moderate advisories; dependencies were
  not upgraded as part of the architecture experiment.
- All five M7 Python test modules: **106 passed**, with Python 3.12.13,
  PyTorch 2.14.0, pytest 9.1.1, NumPy 2.5.3, Pillow 12.3.0. The initial shell
  lacked `python` and system Python lacked pytest; an isolated environment was
  installed at `.cache/m8-venv`, without changing project dependencies.
- Starting GitHub Actions run `34652035253`: `completed`, `success`, for the
  exact starting SHA.
- Local training environment: macOS 26.6.2, MPS available. Hardware identity,
  execution versions and elapsed times are recorded by the runner, not inferred.

## Immutable data and production baseline

The production budget is established by
`results/m7-baseline-freeze.json::trainingConfig` and the M6 selection record:
60 epochs, 1,353 updates per epoch, **81,180 actual optimizer updates**.

Use the N151 recovery cache frozen in `results/m7-pairs-freeze.json`, not a
reconstructed or silently substituted cache. Its files were rehashed before
this registration:

| Input | SHA256 |
|---|---|
| Training manifest | `6d3d91058755eaf59ba7c8912a90e2aeecdb1d4ce9d433791169b9884ba2b428` |
| Repaired validation manifest | `e7a4447fc88c38873826aa616a017aeb2eb8cc6b40257337957f2ea28cb403f1` |
| Frozen pairs index | `1ad3716618e645dc2160a95c0046191e32ec8577947891dc118b8270112f0e55` |
| Frozen train.pt | `4c5cdea7bf38fed1f02fc637194ea783bf3e25a62d386705acf4a57f5bc0bf98` |
| Frozen val.pt | `96169de3ae931f6efffce2fef202e5a43c1aa93acba0337684588ac6c846eb17` |
| Frozen production JSON | `d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a` |

The trainer corpus digest is
`3f73af7f48ffee2b8efe5e7505dd370944e5614efb9ae4d3c5bff64d3fc563e6`:
SHA256 of the index's sorted-key JSON, followed by raw train.pt and val.pt.
The cache contains 151 training clips, 453 training sequences, 43,488 training
patches, and 16 validation clips, 32 validation sequences, 3,072 validation
patches. Validation uses `data/captured-val-m7/manifest.json` and its existing
creator-disjoint freeze proof. It does not revert to the exposed, overlapping
historical validation manifest.

This cache is not claimed byte-identical to the lost M6 cache. Membership,
degradation protocol and update budget match production; both M8 arms use the
same recovered bytes. Any production comparison retains this qualification.
The captured validation PNG cache is inventoried and hashed before training;
all scoring must use that inventory, with exactly eight corresponding frames
in each of 16 clips and CRFs 18, 26 and 34. Missing or changed bytes stop the
experiment, rather than triggering an automatic cache rebuild.

## Training constants

The machine-readable values in `results/m8-plan.json` are part of this
registration. A disagreement with this document is a pre-training blocker.

- Binding paired seeds: **8101, 8102, 8103**, each used once for R0 and once for
  R3. Fixed serial order: 8101 R0/R3, 8102 R3/R0, 8103 R0/R3. No replacement
  seed may be chosen from its score.
- Exactly **81,180 optimizer updates** per run, with batch 32 and drop-last
  epoch permutations of the same 43,488 patches. Sixty epochs are a sufficient
  upper bound, not the stopping rule. Failure to reach the exact budget is an
  invalid run, not an eligible shorter run.
- Adam: lr 0.002, betas (0.9, 0.999), eps 1e-8, weight decay 0, amsgrad false;
  float32, L1 loss, existing preprocessing and clamped network output.
- LR before zero-based optimizer update `u` is
  `0.002 * 0.5 * (1 + cos(pi * min(1, u / 81180)))`.
  Report equality at 0, 20,295, 40,590, 60,885 and 81,180. The last index is
  the completed-budget boundary (LR zero), not an 81,181st optimizer update.
- Frozen uint8 RGB patches are converted to float32 / 255; HR 128x128 and LR
  64x64, from 2560x1440 references and 1280x720 inputs. No online recropping.
- Existing per-batch transforms: independent Bernoulli(0.5) horizontal flip,
  vertical flip, and one 90-degree turn, applied identically to LR and HR.
- Frozen GOP degradation: libx264 medium, keyint=48, min-keyint=24,
  scenecut=40, bframes=3, ref=3, 24 fps, 24-frame sequences, CRF uniform integer
  [18,36] per sequence, preparation seed 20260906. There is no fresh online CRF
  draw: fixed patch identities also fix the realized codec degradation.
- Model construction, shuffling and augmentation must not share mutable RNG
  streams. Use a CPU torch.Generator and random.Random, each initialized with
  the paired seed, for shuffling and augmentation respectively. Construct
  models under an isolated/restored torch RNG. No augmentation is enabled in
  validation. Record the actual sample and transform stream, not only seeds.
- MPS runs are serialized. Do not run another Metal training or evaluation
  process concurrently. Record full process wall time, training-loop duration
  (including scheduled validation/checkpoint work), and completed updates per
  that duration. These are not pure optimizer kernel timings.

## Effective initialization gate

Keep M7's legacy construction unchanged by default. M8 explicitly opts into
matched effective initialization. Draw the same R0 kernel, stem and head for
each pair. R3's added learnable branches start at zero; subtract the centered
identity delta from its trainable 3x3 kernel. Thus the sum of all R3 branches,
including the fixed identity, equals the paired R0 kernel up to float32
rounding. If a learnable branch is ever nonzero at initialization its embedded
weight and bias contribution must also be compensated; M8 does not introduce
such a nonzero branch initialization.

Permanent tests and the pre-update smoke gate must compare fused body weights
and biases, body pre-tanh output, post-tanh output, and full-network output,
including borders and several input shapes. **Absolute tolerance 1e-5, rtol 0**
for every comparison. Shared stem and head tensors must be bit-identical.
Changing the tolerance after observing failure is not permitted. No mathematical
impossibility is assumed merely because a branch is fixed: its delta can be
compensated in another linear branch.

The diagnostic prefix covers at least **eight complete batches (256 sample
IDs)** and records patch row, source/sequence/frame identity where available,
the cached crop bounds, codec settings, and each flip/rotation. The original
cache index does not store source-frame crop coordinates. Therefore prove
crop identity with identical frozen LR/HR patch bytes and full cached-patch
bounds; do not invent missing original coordinates or describe the training
loop as drawing new crops. If original coordinates can be reconstructed from
preparation records, record them additionally and validate against patch bytes.

## Checkpoints and validation

The existing trainer's checkpoint metric is **RGB PSNR from the global MSE of
the entire fixed validation patch tensor**, not luma PSNR or a mean of batch
PSNRs. SSIM is secondary and uses the existing luma convention. A memory-batched
implementation must combine squared errors and pixel counts before PSNR, and
pass a numerical comparison with the existing full-tensor definition.

There are exactly **60 eligible checkpoint draws**, after updates
`1353 * k` for k=1..60. Both arms use the same list. Choose best by strictly
greater patch PSNR; exact ties retain the earliest update. Record actual
evaluations and chosen update. Saving extra diagnostics cannot create extra
eligible draws.

Always preserve and export the **fixed-final update 81,180** before restoring
best weights. Export best-validation weights separately, even when identical.
Save diagnostic snapshots at **5,412; 10,824; 16,200; 29,766; 50,061; 64,944;
81,180**. Update 16,200 is a snapshot only, not a 61st checkpoint-selection
opportunity. Save patch-validation curves at all 60 scheduled updates.

Both exports must fuse to the unchanged deployed graph, reload successfully,
and meet the 1e-5 float32 training-versus-fused/export bound on fixed diagnostic
inputs. Corpus, registration, code, initial state, model bytes, step count,
chosen checkpoint, and stream hashes belong in per-run evidence.

## Smoke gate before six long runs

Use paired seed 8099 for a short integration run, with the same batch, actual
frozen data and MPS device. This is an apparatus check, never a candidate.
Check initial function equality, eight-batch stream equality, LR equality,
finite losses, exact stopping, checkpoint saving, validation, fusion and JSON
reload. Tests separately exercise the registered 81,180-update LR and checkpoint
lists without doing 81,180 test updates. Commit the passing smoke report and
implementation before any binding long run starts.

## Binding scoring and statistics

After all six valid runs exist, score all fixed-final and best-validation
models on the frozen captured validation cache using `tools/m6-validate.py`'s
RGB PSNR definition and existing preprocessing. The registered invocation is
the M7 one: repaired manifest, CRFs 18/26/34, eight frames, 24 fps, requested
start 12.0 seconds with the already-frozen per-clip extraction. Use exactly the
same 48 clip/CRF cells for production, R0, R3 and Catmull-Rom. Do not silently
change the evaluator's precision, geometry, clamp, frame pairing or aggregation.
An execution optimization requires parity testing before any binding scores.

Average frames within a clip/CRF, tiers within each clip, then clips equally.
For each paired seed, primary difference is R3 minus R0 at fixed final. Report
every paired difference, mean, median, min/max and sample SD (ddof=1), each CRF,
all 24 category-by-CRF cells, and clip tables. Best-validation differences are
a separate secondary comparison and cannot override the fixed-final verdict.

The declared seed-level test is an **exact paired two-sided sign-flip test**
on the three seed differences: enumerate all eight sign assignments, and count
absolute mean differences at least as large as observed. There is no unpaired
seed permutation test, frame-level pseudo-replication, or post-hoc choice of a
more favorable test. Its exact calibration assumes sign exchangeability
(symmetry under the null), not merely a zero population mean. With three pairs
its minimum attainable two-sided p is
0.25: this design cannot establish an effect at alpha 0.05. Report that limitation
next to the verdict. Category and horizon analyses are descriptive, not extra
opportunities for hypothesis rejection.

The following are finite-seed **selection labels**, not claims of population
significance or optimizer equivalence:

- **PERSISTENT ADVANTAGE**: fixed-final mean R3-R0 is strictly greater than
  0.01 dB and every paired seed difference is positive. This retains M7's
  simplicity tie band and adds a pre-training consistency requirement.
- **SMALL / UNCERTAIN**: fixed-final mean is positive but the preceding rule
  fails. Do not proceed to fresh production seeds.
- **NO SELECTED ADVANTAGE**: fixed-final mean is zero or negative. Stop the R3
  deployment path. This is not evidence that reparameterization never works,
  or that the optimizers are equivalent.

Only PERSISTENT ADVANTAGE permits the subsequent candidate stage. Even that
label must state that statistical resolution is insufficient for a conventional
significance claim. No new seeds are added to rescue a non-selected result.

## Horizon interpretation

Score the seven diagnostic snapshots with the same captured evaluator. Report
each seed and aggregate R3-R0 against update count, retaining crossings.
Compare exact 16,200 and 81,180 snapshots. These snapshots share the **81,180-
update cosine schedule**, so the first is a partial long schedule, not a repeat
of M7's fully annealed 16,200-update schedule. Their final-minus-short difference
is descriptive; absolute change within 0.01 dB is called similar, otherwise
increased/decreased, with sign changes explicitly retained.

Also report M7's corrected short-budget R0/R3 results using its equivalent
scoring protocol and frozen cache. M7 used different seed IDs and uncompensated
R3 identity initialization. Keep the historical +0.0247 dB separate from the
new matched-initialization horizon curve. Neither comparison establishes
convergence, an infinite-horizon solution, or monotonic behavior between two
points. Distinguish an early lead from a lead retained at this finite horizon.

## Conditional production stage and confirmation

The authoritative replacement conditions remain
**`docs/M7-PREREGISTRATION.md`, Replacement criterion**, including validation
candidate-minus-production >= +0.10 dB, positive confirmation mean, no overall
tier regression, no new severe class-by-tier regression, and runtime parity.
They are referenced, not replaced by candidate-versus-Catmull gains. M8 adds
the stricter pre-training class-cell rule that no confirmation category-by-CRF
mean candidate-minus-production may be below **-0.05 dB**. Report every negative
cell even when it is within that bound.

Only after the matched-budget decision is committed may five fresh R3 seeds
**8201, 8202, 8203, 8204, 8205** train for 81,180 updates with these same controls.
Choose their best checkpoints on validation only. Across seeds, highest captured
validation mean wins, with candidates within 0.01 dB of that maximum tied to
the lowest seed number, as in M7. Record the selected model's SHA in a separate
commit before any architecture-confirmation scoring. A selected model below
the existing +0.10 dB validation gate never opens confirmation.

`data/captured-confirm-m7/proposal.json` is not a freeze. Before scoring any
candidate, independently verify each retained or replacement source's licence,
creator, shoot/event, source, dimensions, active image geometry, original
reference resolution, overlays, category and pins from source evidence and
representative frames. Do not rely solely on previous manifest prose. Cover
all eight existing categories, including naturally captured text/signage, not
subtitles, title cards, HUD or screen recordings. Check creator, shoot and
source disjointness against all historical training/evaluation manifests.
Commit the final manifest, digest, source list, category counts, audit and
review signoff before the one permitted confirmation opening. Ambiguity is
grounds for rejection. No confirmation scoring for a non-candidate.

Confirmation compares selected R3, frozen production and Catmull-Rom at CRFs
18/26/34. Primary contrast remains candidate-minus-production. Report mean,
median, wins/losses/ties, all clip and class-by-tier cells and paired 95%
percentile bootstrap intervals from 20,000 resamples of **clips**, preserving
each clip's tiers, seed 20260911. No frame or cell pseudo-replication. Explicitly
inspect motion/texture at CRF34, and faces, text and lowlight, without using
these cells to reselect the model.

Any deployment candidate also needs static determinism, integer-shift,
synthetic subpixel, captured-motion-residual and sharpness diagnostics with
matched-sharpness interpretation: no unsupported temporal-improvement claim
and no clear new temporal regression. Runtime requires alternating current /
candidate runs on the actual base-M5 24-GB MacBook Pro at 720p60 -> 1440p,
recording environment, windows, p50/p95/max, presented/rendered FPS, callback
skips, decoder drops and fallback events, under the M7 thresholds. Requested
allocation bytes and tensor shapes may establish structural parity; actual
device/driver memory remains **not measured** unless externally profiled.

Production replacement is atomic and weights-only, only if every gate passes.
Preserve the previous hash and evidence. No shader or runtime change is implicit
in this milestone. Otherwise the decision is RETAIN PRODUCTION.

## Stop conditions, review and audit trail

Stop before training on failed baseline, uncommitted registration, hash/split
failure, initialization or stream mismatch, LR/checkpoint inequality, missing
provenance, nonfinite loss, or unavailable/contended MPS. A truncated run is
invalid. No early stopping, extension beyond budget, hidden retries or best-of-
restarts. An interrupted run may resume only with complete model, Adam, RNG,
step and selection state; otherwise preserve it as invalid and restart the
same paired seeds with the repair and reason recorded. A validity-changing
bug invalidates and regenerates affected results, not just the prose.

The first complete eligible paired result is binding. Do not change input
membership, scoring, checkpoint eligibility, seeds, tie band, statistical test
or continuation rule after seeing scores. Any necessary pre-result amendment
is a new committed section, never an overwrite of this registration.

Independent review must cover chronology, initialization, RNG, LR, checkpoint
counts, exact updates, paired statistics, horizon interpretation, confirmation
licences/disjointness/categories/freeze, temporal interpretation, runtime,
replacement decision and literature wording. Review must challenge source
claims, not merely repeat them. Literature discussion after measurement states
task, data, scale, degradation, control and verified schedule differences;
unverified schedules and negative dataset cells remain visible.

Fresh M8 status is tracked in `docs/M8-STATUS.md` against committed evidence.
No blocked M7 task is mechanically declared an M8 requirement. Final local
gates, hygiene tests, a fresh clone, pushed clean main equal to origin/main,
and completed/success GitHub Actions are required before claiming completion.