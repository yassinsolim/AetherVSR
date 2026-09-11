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

### One asymmetry, recorded rather than hidden

Learnable added branches (1×1, 1×3, 3×1) are zero-initialized, so `R1` begins
numerically identical to `R0`. The identity branch cannot: it is a fixed `+x`
with no parameters, so `R2` and `R3` begin at `k3(x) + x` — their fused kernel
starts with a diagonal `+1` that `R0` does not have.

This is inherent to having an identity branch at all, not an implementation
choice, and it means an `R2`-vs-`R0` difference could in principle be an
initialization effect rather than an optimization one. Two consequences, fixed
now rather than after seeing results:

* `R1` is the clean test of the hypothesis, because it isolates
  parameterization with initialization held identical.
* If `R2` wins and `R1` does not, that is reported as *possibly an
  initialization effect*, not as evidence for reparameterization.

`tools/test_reparam.py::test_added_branches_start_as_intended` asserts both
behaviours so the distinction stays visible in the test suite.

## Fusion correctness gate

Before any rung is trained, for randomized weights and randomized inputs across
multiple seeds and several shapes, comparing the multi-branch block against its
fused `Conv2d`:

* **pre-`tanh` max absolute error ≤ 1e-5** (float32)
* post-`tanh` max absolute error ≤ 1e-5
* full-network fused-vs-training max absolute error ≤ 1e-5

Border pixels are included, not cropped. A rung that cannot meet this is not
trained. These become permanent tests.

### Artifact-linked structural verification

`results/m7-linked-equivalence.json` and `results/m7-linked-webgpu.json` bind
the CPU and browser runs to the same model bytes, golden-vector bytes and
24×16 input. Nine randomized R1/R2/R3 models (seeds 0–2) passed both f32 and
f16 browser verification on Apple M5, macOS 26.6.2, Chrome 152.0.7977.42
(`apple / metal-3`). CPU branch→fused max error was 1.1921e-6; JSON reload
error was zero. Browser intermediate maxima were 1.4119e-6 (f32) and
0.005666 (f16). Output uses `max(precision tolerance, 1.5/255)` because the
storage texture quantizes to eight bits; both paths passed that bound.

Reproduce the CPU/export side with:

```sh
python tools/m7-equivalence.py \
  --export-dir data/captured-train-v2/sources/m7-recovery/parity \
  --out results/m7-linked-equivalence.json
```

Then run `tools/m7-browser-parity.js` inside `bench.html`. That script, not a
manual step, is what re-hashes the fetched model and golden bytes, refuses to
dispatch when they differ from the CPU record, and calls
`window.aethervsrGolden` in both precisions. It is committed as code because a
verification that exists only as prose cannot be re-run or regression-tested,
and the same sentence would otherwise be repeated for a future candidate whose
files nobody hashed. Note the distinction it enforces: the runtime guard in
`golden-verify.ts` compares self-declared `modelSha256` label fields, which
catches vectors exported from a different model but proves nothing about file
contents; the script hashes the bytes themselves.

These are synthetic structural proofs, not quality candidates; a selected
trained model still requires its own export and browser verification before
deployment.

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

---

## Amendment 1 — the first screen was confounded and is withdrawn

*Written after the first nine models were trained and scored, before any
replacement model exists. The results that prompted it are stated here so the
amendment can be judged against them rather than around them.*

### What the first screen found

Nine models, R0/R1/R2 × 3 seeds, equal 16,200-step budget, per-clip mean Δ
against the frozen production model on the Milestone 6 validation corpus:

| rung | mean dB | seed sd |
|---|---|---|
| `R0` | −0.1557 | 0.0165 |
| `R1` | −0.1505 | 0.0302 |
| `R2` | −0.1677 | 0.0079 |

`R1 − R0 = +0.0052 dB`, exact permutation `p = 0.800`. Under the selection rule
that is a tie, and a tie selects `R0`: a null result.

### Why it is withdrawn anyway

Independent review found the ladder was not varying one thing. `nn.Conv2d`
draws its default initialisation from the global generator, so constructing an
optional branch consumed draws even though the branch is zeroed immediately
afterwards. Measured at seed 1: `body.1.k3.weight` and `head.weight` differed
between `R0` and `R1`, and the next three draws were `[0.60637, 0.65406,
−0.54377]` for `R0` against `[−0.80873, 0.13418, −0.09037]` for `R1`.

So each arm differed by block structure **plus** a different initialisation of
the layers built after the first block **plus** a different data order. The
claim in *One asymmetry* above — that `R1` begins numerically identical to `R0`
— was intended but false, which removes the one arm designated as the clean
test.

The effect under measurement is ~0.005 dB against a seed sd of ~0.017. The
confound is the same size as the signal, so it cannot be argued away as small.

### An honest statement of what this does and does not change

An earlier draft of this amendment argued that because the confound adds
variance, and added variance makes a true effect harder to see, the null was
unlikely to be an artefact of it. That reasoning is backwards and is retracted.
Added variance raises the **false-negative** risk, and the conclusion here *is*
a null — so the confound is precisely the kind of defect that could have
produced it. It cannot manufacture a spurious *win*, but a spurious null is
exactly what it can manufacture.

What can be said is narrower, and it is the whole of it: **the pilot cannot
establish a null.** Added variance can mask a real effect, so a non-detection
under the confound is not evidence of absence. That is why the screen is
withdrawn rather than annotated and kept.

A prediction, recorded before the replacement data exists and labelled as
nothing more: the corrected screen is expected to reach the same verdict. If it
does, that verdict will rest on the corrected run alone, never on the pilot
agreeing with it — two runs sharing a defect-driven answer would not be
corroboration.

### What changes

* `RepBody` builds optional branches inside `torch.random.fork_rng(devices=[])`.
  All shared tensors and the post-construction RNG stream are now bit-identical
  across `R0`–`R3`, asserted by
  `test_rung_choice_does_not_disturb_initialisation`.
* The nine models move to `models/m7-pilot-confounded/` and their scores to
  `results/pilot-confounded/`. They are preserved as pilot evidence, are
  excluded from selection, and no claim rests on them.
* The screen is rerun on fresh models. **`R3` is included**: it was in the
  ladder from the start, and dropping a registered rung after seeing a null
  would be the same outcome-dependent editing this amendment exists to correct.
* Nothing else moves. Same corpus, same split, same budget, same selection rule,
  same replacement criterion, same tie-break toward simplicity.

### Cache recovery and eligibility gates

The original N151 patch cache was lost from `/tmp`. Its replacement cannot be
claimed byte-identical: extraction failures affect the retained sequences and
the subsequent CRF draws. All corrected R0–R3 arms must therefore use one newly
frozen cache, with its exact post-subset tensor digest recorded in every model.
No old pilot arm may be pooled with a replacement arm.

The first recovery attempt retained 136/151 training clips but only 294/453
requested sequences, and 14/16 validation clips with 22/32 sequences. Clip
presence alone is insufficient. Preparation and merging now reject partial
sequence coverage; failed preparation writes diagnostics but no `all.pt`.
The required recovery coverage is all three sequence slots per training clip
and both slots per validation clip. The 16,200-update budget and selection rule
remain unchanged. No corrected training has started.

`tools/m7-screen-report.py` requires declared `inputStatus` (or `--input-status`
when absent). Withdrawn inputs produce arithmetic-only reports with no winner,
argmax or null verdict. Corrected reports require all four rungs, three seeds
each, and complete matched validation cells. Dataset identity, frozen baseline
identity and model training records remain separate prerequisites for accepting
that ranking as an eligible selection.

## Amendment 2 — repair the validation creator overlap

Before any corrected model was trained, a provenance audit found a real
creator overlap: training contains `Foto: PantheraLeo1359531` (the closed
butcher shop), while validation contains `PantheraLeo1359531` (Knollenteich).
These are the same operator under the project's existing normalization rules.
The original validation manifest's creator-disjoint flag is therefore not a
valid independence claim. Recovery was stopped when this was identified.

The 151 training sources remain unchanged. A versioned validation manifest,
`data/captured-val-m7/manifest.json` (`captured-val-m7-v1`), retains 15 original
entries verbatim and replaces only the conflicting nature clip with Veljo
Runnel's own-work katydid recording (Commons page 142193278, CC BY 4.0).
The copyright-holder name is Veljo Runnel; the uploader account is Veljorunnel.
Both identities are checked against training. The 3840×2160, 22.061-second
source is explicitly slowed fourfold; its Commons SHA1, byte length and 2-MiB
prefix SHA256 are pinned in the new manifest. The inspected source poster
shows a real insect on vegetation, not a rendered scene or title card.

The replacement preserves 16 clips and all category counts. The disjointness
audit covers both the current 151-clip training manifest and the legacy
12-clip training manifest: 163 is the audit union, not a changed training size.
It checks normalized creator aliases, stored and derived shoot IDs, URLs,
titles, page IDs and available source hashes. The complete manifest SHA256 and
audit result are recorded in `results/m7-validation-freeze.json`.

This amendment supersedes earlier instructions to use the unchanged M6
validation split. All corrected arms and the frozen production model must be
evaluated on the new manifest; no old validation score may be pooled with
that result. The training membership, 16,200 updates, rung ladder, seed counts,
loss, selection metric, 0.01-dB tie rule and replacement threshold do not change.
The historical validation manifest is retained for historical evidence and is
not silently rewritten into the new split. No neural score was consulted in
choosing the replacement. This new validation source is also excluded from
the future architecture-confirmation corpus.

### Verified relative timing for the squirrels source

Recovery found one training source with missing duration metadata:
`nature-squirrels-at-point-pleasant-park`. The previous 60-second fallback
requested a window outside the clip and yielded only two of three sequences.
The pinned 29,063,069-byte file was downloaded and its Commons SHA1 matched.
`ffprobe -count_frames` found 2,274 video frames, nominally 60 fps, with format
start 84,721.155 and reported end/duration 84,759.415. Those absolute timestamps
give a usable relative duration of 38.26 seconds, not 84,759 seconds.

`results/m7-squirrels-timing.json` records the probe, full-file hash and explicit
duration correction used by the new whole-clip request. The failed request is
preserved separately. Source identity and membership, seed 20260906, three
24-frame sequences, 24-fps preparation and crop settings remain unchanged.
This is a declared timing-metadata repair before corrected training, not a
silent resampling or padding of the failed output. The preparation tool now
rejects missing, zero or non-finite durations instead of inventing 60 seconds.
