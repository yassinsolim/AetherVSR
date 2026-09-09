# Milestone 7 — pre-registered architecture experiment plan

Committed **before** any reparameterized block exists in the repository, and
before any Milestone 7 model is trained. Milestone 6 shipped a rule 42 seconds
after two of its final seeds already existed; a reviewer proved it from git and
the seed set had to be discarded and retrained. This document is committed on
its own so the ordering is checkable from `git log` alone.

## The question

Milestone 6 found dataset scale exhausted: 12 → 151 clips bought +0.103 dB at
equal optimizer updates, not statistically established. The remaining lever is
the *training* graph.

> Can a more expressive training-time parameterization learn better weights that
> collapse **exactly** back into the current 6,291-parameter inference graph?

## What is and is not being claimed

A sum of linear branches before a nonlinearity is exactly one convolution. So
the fused **function class is unchanged** — R1 and R2 cannot represent anything
`R0` could not already represent. The hypothesis is strictly about
**optimization**: whether an over-parameterized but algebraically equivalent
training graph reaches a better point in the *same* weight space.

That is a real and testable claim, and it is the only one this milestone makes.
Any statement that reparameterization adds inference capacity would be false.

## Safety rule

A branch is eligible only if it is **linear and applied before the existing
`tanh`**. Branches with their own nonlinearity cannot be fused and are excluded.
Every claimed fusion must be exact up to floating-point rounding, verified
numerically, not visually.

## Experiment ladder

Simplest first. Each rung is measured before the next is written.

| Rung | Training-time body block | Fused inference block |
|---|---|---|
| `R0` | single 3×3 (current) | 3×3 |
| `R1` | 3×3 + 1×1 | 3×3 |
| `R2` | 3×3 + 1×1 + identity | 3×3 |
| `R3` | R2 + 1×3 + 3×1 | 3×3 |
| `R4` | richer block, only if R1–R3 justify it | 3×3 |

`R0` is retrained under identical conditions rather than reusing the shipped
model, so the comparison isolates the block and not the training run.

Stem (Part N) and head (Part O) reparameterization are attempted **only** if a
body rung wins. BatchNorm (Part K), sequential 1×1→3×3 branches (Part M),
frequency loss (Part W) and self-distillation (Part X) are explicitly **not** in
the first experiment and each requires its own isolated comparison.

## Held constant across the ladder

Corpus `captured-train-v2` (N151), the Milestone 6 split, GOP degradation with
CRF uniform [18,36], Adam, L1, batch 32, `lr` 2e-3, cosine **over optimizer
updates**, the same step budget, the same augmentation RNG, and best-validation
checkpoint selection. Only the training-time block structure changes.

## Fusion correctness gate

Before any rung is trained, for randomized weights and randomized inputs across
multiple seeds and several shapes, comparing the multi-branch block against its
fused `Conv2d`:

* **pre-`tanh` max absolute error ≤ 1e-5** (float32)
* post-`tanh` max absolute error ≤ 1e-5
* full-network fused-vs-training max absolute error ≤ 1e-5

Border pixels are included, not cropped. A rung that cannot meet this is not
trained. These become permanent tests.

## Screening protocol

3 seeds per rung on the Milestone 6 validation corpus (16 clips, 8 categories,
creator-disjoint from training), reporting mean Δ vs Catmull-Rom, Δ vs the
frozen production model, CRF 18/26/34, per-category results and seed sd.

## Selection rule

> The winning rung is the one with the highest mean validation Δ **against the
> frozen production model**, pooled across CRFs and categories. If two rungs are
> within **0.01 dB**, the simpler rung wins — fewer branches, and `R0` beats
> everything on ties. Final seeds are then trained for the winner only.

Ties resolve toward simplicity rather than toward the argmax, because at this
project's seed variance (~0.006–0.02 dB) an argmax over rungs is noise.

## Final seed protocol

**5 fresh seeds**, all trained after this document is committed, selected on
validation by the same rule as Milestone 6: highest mean, ties within 0.01 dB
to the lowest seed number.

## Replacement criterion

The candidate replaces the shipped model only if **all** hold:

1. **Validation** Δ vs the frozen production model ≥ **+0.10 dB**, which is
   well above the observed seed sd.
2. **Confirmation** mean candidate-minus-current is positive on the new
   architecture confirmation corpus.
3. No overall regression at any compression tier.
4. No new severe per-category regression, judged **per class and per tier** —
   Milestone 6 pooled these and hid a negative motion cell.
5. Runtime parity: identical tensor names, shapes and parameter count, p50
   ≤ 8 ms, p95 < 10 ms, ~60 fps, no budget fallback.

If the gain is smaller or uncertain, the current default is retained. **A null
result is an acceptable and expected outcome** and will be reported as one; it
would be evidence that the existing optimizer already finds equally good fused
kernels.

## Confirmation corpus

The Milestone 6 confirmation set has been opened twice and is now a regression
benchmark. A **new** architecture confirmation corpus is built, creator- and
shoot-disjoint from all training data and from every previous evaluation set,
frozen with its digest before any candidate scores it. If sourcing enough valid
material proves infeasible, that is reported and the Milestone 6 set is used
explicitly labelled as exposed.

## Weak cells

Milestone 6 left `motion@CRF34` (−0.0907) and `texture@CRF34` (−0.0069)
negative. These are reported explicitly for every rung. They are not optimized
for directly, and no rung is selected on them.
