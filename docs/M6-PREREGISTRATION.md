# Milestone 6 — pre-registered final decision rule

Written **before** the final seed set was trained and **before** the frozen
regression benchmark or the confirmation corpus were read with any Milestone 6
model. Committed as its own commit so the ordering is checkable from git alone —
Milestone 5.5 recorded its rules in a commit that also carried the results, and
a reviewer correctly pointed out that this proves nothing.

## What is already decided, and on what evidence

The scaling curve is complete (`results/m6-validation.json`,
`results/m6-validation-arms.json`, commit `3c1d667`). It used validation only.

* **Architecture:** `AetherSR C16D2`, 6,291 parameters, unchanged. The
  milestone's rule was not to touch it unless data scaling saturated, and the
  measurements below do not establish saturation.
* **Corpus:** `captured-train-v2`, all 151 clips (`N151`), digest
  `9bf33e3ea9f8…`. It is the best scale on validation and the compute analysis
  shows why: it is the only corpus large enough that a long training run is not
  mostly repetition.
* **Degradation:** unchanged from Milestone 5.5 — real 720p H.264, GOP
  `keyint=48:bframes=3`, CRF uniform on [18, 36] per sequence.
* **Training length:** 60 epochs, the project's standard, which on N151 is
  81,180 optimizer updates. Chosen because the compute control showed more
  updates help at every scale, not tuned per candidate.
* **Loss, optimizer, schedule:** L1, Adam, cosine. Unchanged.

## Seed protocol and selection rule

Five seeds, 1–5. For each: validation mean delta, per-CRF curve, per-category
breakdown, weight SHA-256.

> **Selection.** The shipped candidate is the seed with the highest mean
> clip-level validation delta against Catmull-Rom, pooled across all CRFs and
> categories. If two seeds fall within 0.01 dB — below the seed standard
> deviation observed across this milestone — the lower seed number wins, so the
> tie-break cannot be steered by inspecting scores.

Seeds are indistinguishable in this project's experience; the rule exists to
stop an argmax over noise being presented as a finding.

## Replacement threshold

The candidate replaces the shipped `aethersr-c16d2.json` only if **all** hold:

1. **Validation improvement exceeds seed variance.** Mean delta at least
   0.05 dB above the current production model — roughly four times the observed
   seed sd of ~0.013 dB.
2. **No category collapses.** No content class regresses by more than 0.05 dB
   against production, and the worst-performing class does not get worse.
3. **No runtime regression.** The inference graph is identical, so p50 ≤ 8 ms,
   p95 < 10 ms, ~60 fps maintained, no budget fallback.
4. **No new temporal regression**, judged with the sharpness control. A residual
   ratio that moves only in proportion to output sharpness is not a temporal
   change, in either direction.

If the improvement is smaller or uncertain, the current default is retained. A
null result is an acceptable outcome and will be reported as one.

## Confirmation-set acceptance criteria

Fixed now, before the set is opened. On `captured-confirm` (17 clips, 12
creators, all 8 categories, creator-disjoint from training):

* positive mean neural-minus-Catmull-Rom at CRF 18, 26 and 34;
* 95% paired bootstrap CI excluding zero at typical (26) and poor (34);
* wins in a clear majority of clips at each tier;
* no content class in outright collapse.

Reported per tier and per class, never as a single pooled number. The
statistical unit is the clip. The set is evaluated **once**, after the model is
frozen and this document is committed. If a methodology defect is found
afterwards it will be fixed and regenerated, and the exposure documented rather
than hidden.

## What this milestone will not claim

* Not a scaling law. "Observed AetherVSR C16D2 data-scaling behaviour over the
  measured range" is the most that 12 to 151 clips supports.
* No temporal improvement unless it survives the sharpness control.
* No general faces claim. The faces benchmark is eight clips from largely one
  institution; the confirmation set adds a small number from unrelated
  creators, and both are reported separately.
* No routing or model-selection work. Milestone 5.5 found the candidate quality
  signals carry no cross-stream calibration.
