# Milestone 5.5 — pre-registered experiment plan

Written **before** any Milestone 5.5 model was trained, and before any candidate
touched the frozen captured test set. Recorded so that the analysis cannot be
retrofitted to the results, which is the specific failure Part Z asks reviewers
to look for.

## Question

The shipped C16D2 model gains +0.72 dB at CRF 18 and +0.36 dB at CRF 26 over
production Catmull-Rom on captured video, but only +0.055 dB (not significant)
at CRF 34. Why, and can it be fixed without enlarging the network?

## What is already known

The milestone's leading hypothesis — that single-frame all-I training
degradation taught the model the wrong problem — is **refuted**
(`results/gop-diagnostic.json`, commit `3156dd5`). Varying only GOP structure,
the model does *better* on GOP input at every CRF from 18 to 38, and still
better at 6/6 matched-input-quality points. The gain instead tracks delivered
input quality monotonically and reaches zero near 26 dB input PSNR under both
structures.

That diagnostic measured how the **existing model responds to** GOP input. It
does not answer whether **training on** GOP context produces a better model.
Those are different questions: one is test-time difficulty, the other is
train-time distribution match. The GOP training arm is therefore retained.

## Design

Two factors, fully crossed. The architecture, loss, optimizer, schedule,
corpus, split and patch sampling are identical in every cell; only the
degradation generator changes.

| Factor | Levels |
|---|---|
| Compression context | `all_i` (keyint=1, no B-frames) · `gop` (keyint=48, bframes=3) |
| CRF distribution | `uniform` U[18,36] · `poor` two-thirds of draws in [28,36] |

4 cells × 3 seeds = 12 runs.

`all_i` × `uniform` is the control: it reproduces the shipped model's
degradation on the new video corpus, so the video-corpus change is isolated
from the degradation change rather than confounded with it.

## Held fixed

* Architecture `AetherSR C16D2` — 5×5 stem, 2× 3×3 body, resize-convolution
  head, 6,291 parameters. Unchanged from production. No capacity change in
  this experiment.
* Loss L1. Optimizer Adam, cosine schedule.
* Corpus: the 12-clip captured-video training corpus
  (`data/captured-train/manifest.json`), disjoint from the frozen test set by
  content hash, clip id and location/series name.
* Split by **source video**, decided before any frame is written.

## Selection rules

* Checkpoints and seeds are selected on **validation PSNR only**.
* The frozen captured test set is evaluated **once per frozen candidate**, after
  selection is complete.
* No clip is added to or removed from any evaluation set on the basis of a
  model result.
* CRF distributions are not retuned after reading captured test scores. If
  exploratory test access becomes unavoidable it is recorded and a fresh
  confirmation set is required before the final claim.

## Pre-registered acceptance target (Part S)

For a candidate to count as a real heavy-compression improvement, on the frozen
captured CRF 34 set:

* positive paired mean neural-minus-Catmull-Rom PSNR;
* 95% paired bootstrap CI excluding zero;
* preferably ≥ 8/10 clip wins;
* approximately **+0.20 dB or better**, so the result is not statistical-only.

For a **general replacement** additionally: no material regression at CRF 26,
where a loss beyond about 0.10 dB requires explicit justification. CRF 18 is
reported either way.

Runtime for any default candidate: p50 whole neural stage ≤ 8 ms, p95 < 10 ms,
720p60 presented rate maintained, no persistent decoder drops, no repeated
budget fallback.

## Outcomes that are acceptable

A negative result is a successful experiment. If no candidate meets the target
the milestone reports that the current default is retained and says why. The
project does not weaken the criterion after seeing results.

---

# Amendment 1 — tie-break between two qualifying candidates

**Timing, stated plainly.** This amendment was written at 23:20, after
`results/captured-gopvideo-poor.json` was produced at 22:50, and before its
contents were inspected. It therefore does **not** carry pre-registration
weight and must not be read as one. It is recorded here, in the same document,
because a rule invented after a result is still worth writing down explicitly
rather than applying silently.

## Why an amendment was needed

The original document fixed an acceptance target but specified no tie-break.
Independent review (`M55Stats`) established that the 2×2 has a large
interaction the original analysis averaged over: at CRF 34 the interaction is
−0.0613 (t = −23), roughly nine times either main effect. Within the GOP arm,
the heavy-CRF distribution is worth **+0.0344 dB** at CRF 34 on validation,
making `gop_poor` the best CRF-34 cell in the experiment and `gop_uniform` —
the candidate selected first — the worst of that arm.

So two candidates can satisfy the general-replacement criteria at once, and
the original document does not say which ships.

## The rule

> Where more than one candidate satisfies the general-replacement criteria, the
> default is the candidate selected first on validation. It is displaced only by
> a candidate that beats it on the primary endpoint — frozen CRF 34 — by at
> least the pre-registered materiality margin of **0.20 dB**, measured as a
> paired clip-level difference between the two models on the same ten clips.
> An advantage below that margin does not displace a default that is
> simultaneously behind at CRF 18 and CRF 26.

## Why this rule is not outcome-shopping

The rule is decided entirely by numbers that were available **before any test
read**, and it inherits its threshold rather than inventing one:

* The 0.20 dB margin is not new. The original target already declared that a
  CRF 34 movement should be "approximately +0.20 dB or better, so the result is
  not statistical-only" — that is a materiality floor for this endpoint, fixed
  in advance.
* The validation tradeoff was fully characterised before the freeze was read:
  `gop_poor` costs 0.1589 dB at CRF 18 and 0.0213 at CRF 26 to buy 0.0344 at
  CRF 34. That 0.0344 dB is a factor of six below the materiality floor at the
  endpoint the floor applies to.

The outcome is therefore determined by the validation gap, not by the frozen
test. If the test reproduces the validation pattern it adds nothing already
unknown; if it contradicts it at the 0.03 dB scale, the honest reading is that
one read is noisy at that scale, not that the ranking reversed.

## Consequences for reporting

* CRF 34 is the primary endpoint for the second read. CRF 18 and CRF 26 are
  descriptive. Two candidates across three conditions is six cells, and the
  freeze only protects a decision that hangs on one named cell.
* Any candidate-versus-candidate comparison is computed as a **paired per-clip
  difference between the two models**, never by comparing two intervals against
  a shared baseline. The Catmull-Rom baseline is bit-identical across runs, so
  it cancels exactly and the paired interval is far tighter.
