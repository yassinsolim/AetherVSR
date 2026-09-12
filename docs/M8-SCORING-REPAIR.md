# M8 scoring apparatus correction

This is a post-result apparatus correction, not a new selection rule. Commit
this document and the repair before producing replacement captured scores.
The registration at c95574336246ea5114afbec4d2a48e538afd3f7f remains unchanged.
No new model is trained, no seed is dropped, and no saved checkpoint is reselected.

## Defect and withdrawal

After all six runs were committed at d1e50c674a9d0301daba1d72a33aa5a206f78889,
the accelerated MPS scorer evaluated their 49 production/fixed/best/snapshot
exports. Independent review found materially different scores for identical
inference weights, despite matching source, input and model hashes. The three
duplicate pairs were R0-8101 best/final, R0-8102 best/final, and R3-8101 best/
update 10,824. Largest observed absolute cell differences were 0.563399792,
0.542372465 and 0.521115363 dB, respectively. These exceed the existing 1e-5
parity tolerance and invalidate the entire captured scoring pass in either
direction. Its raw and derived files are archived in results/m8-withdrawn-mps,
with hashes in withdrawal.json. No conclusion or replacement decision rests
on those numbers. They must never appear in a current results table.

The existing tools/evaluate.py explicitly documents intermittent MPS scoring
corruption and defaults to CPU. M8's production/synthetic one-frame CPU/MPS
parity check passed but did not cover repeated, multi-model execution. Root
missed this existing warning. That was an apparatus error, not evidence that
identical weights genuinely perform differently. The low-level backend cause
is not established. The original MPS parity artifact remains historical only.

## Unchanged experiment

Keep all six completed training runs, original Adam states, fixed-final and
best exports, full streams, update counts, 60-draw checkpoint selections and
seven snapshot updates unchanged. Training remains the measured MPS procedure.
The existing independent review found no validity-changing training-code bug.
This correction does not certify every historical MPS patch-validation value;
best-checkpoint results remain secondary, based on the originally selected
exports. There is no post-hoc CPU reselection of those checkpoints.

The N151/repaired validation inputs, frozen PNG inventory, 49 explicit model
paths, all CRFs, RGB PSNR, SSIM, frame/cell/clip/seed weighting, sign-flip test,
0.01-dB band, consistency requirement and production thresholds do not change.
The primary question remains fixed-final R3-R0 at exactly 81,180 updates.

## Repair and gates, fixed before replacement scores

1. Restore the original M6 evaluator's CPU reference. CPU float32, four intra-op
   threads, deterministic algorithms enabled, no MPS fallback or fast-math.
   Preserve the RGB HWC-to-NCHW conversion, pixel geometry and all border pixels.
   Do not replace mean frame dB by pooled-frame MSE or substitute a luma metric.
2. Keep original training-source hashes authoritative, verified against the
   original smoke and all execution commits. Only tools/m8_score.py and
   tools/test_m8_score.py may differ from that source inventory. Shared model,
   training, metric and statistical-report code must remain unchanged. Record
   repaired scorer hashes separately; do not relabel them as the training code.
3. Group exports by actual loaded float32 tensor identity, not file hashes:
   checkpoint metadata can differ with identical weights. Independently run
   every copy. On every frame require duplicate output max absolute error,
   PSNR difference and SSIM difference <= 1e-5, rtol 0. Do not copy/reuse scores
   to manufacture equality. Retain frame metrics for independent reaggregation.
4. Before binding CPU scores, run parity in two separate processes, forward and
   reversed model order, on all frozen text cells at CRFs 18/26/34 plus motion
   at CRF34. Use production and the three duplicate pairs above (seven neural
   exports), plus Catmull-Rom. Compare complete eight-frame cell metrics against
   the original tools/m6-validate.py score function. Both metrics must differ
   by <= 1e-5; all repeated per-frame metric digests must match exactly. Commit
   both passing reports. Synthetic tests additionally cover actual C16D2 on
   odd, non-square RGB tensors, metadata-different exports and failed guards.
5. Recompute every one of the 49 neural entries and Catmull-Rom over all 48
   frozen cells/eight frames, not only the cells where corruption was noticed.
   Check input/model/source bytes before and after, repeated-weight invariants,
   and the original registration/chronology. Stop on any discrepancy.
6. Regenerate all aggregate, seed, tier, category, clip and horizon analyses
   exclusively from the CPU scores. Publication must pin CPU scores, repair,
   both parity reports, model inventory and completed runs. The generic
   supplied-score arithmetic helper is not by itself an experimental signoff.
7. Independently audit raw frame-to-cell aggregation and all duplicate groups,
   then apply the unchanged fixed-final continuation rule. No new R3 seeds or
   confirmation access is authorized by this correction.

This follows the original stop rule requiring regeneration of affected results
after a validity-changing defect. It restores the registered reference metric;
it does not tune the architecture or rescue a poor training outcome.