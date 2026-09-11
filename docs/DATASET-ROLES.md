# Dataset roles

Every corpus in this repository has exactly one role. The roles are written
down because the failure they prevent is silent: an evaluation set that quietly
becomes training data does not throw an error, it just returns a better number.

Roles are enforced in CI by `test/dataset-roles.test.ts`, which checks the
declared role of every manifest and refuses overlap between anything trained on
and anything evaluated on.

| Corpus | Role | Status |
|---|---|---|
| `data/corpus` | TRAINING (stills) | Milestone 4 still-photograph corpus, CC0 only per ADR-0026 |
| `data/captured-train` | TRAINING (video) | **v1**, 12 clips — superseded by v2, retained for the scaling curve |
| `data/captured-train-v2` | TRAINING (video) | Milestone 6 expanded corpus |
| `data/captured-val` | VALIDATION (superseded) | Milestone 6 split. Retained as history: one clip shared a creator with training |
| `data/captured-val-m7` | VALIDATION | Milestone 7 onward. Same 16 clips with that one replaced; creator-disjoint from both training corpora |
| `data/captured` | REGRESSION BENCHMARK | The ten-clip Milestone 5 set |
| `data/captured-faces` | FACES BENCHMARK | Eight clips, narrow scope |
| `data/captured-confirm` | CONFIRMATION TEST | Opened once, after every decision is frozen |

## The ten-clip set is no longer a pristine test set

It was frozen for Milestone 5 and read once. It has since been read again for
Milestone 5.5's first candidate, again for its second candidate, again for the
dense CRF curve, again for the capacity probe, and again for the creator
sensitivity analysis.

Each of those reads was legitimate and declared. The cumulative effect is not:
a set read six times across two milestones has informed enough decisions that
calling it untouched would be false. It is demoted to **regression benchmark**
— its job now is to catch a model that gets worse, not to certify that one is
good.

This is why Milestone 6 builds a new confirmation set. A project that keeps
quoting the same "held-out" numbers after repeatedly consulting them is
measuring its own memory.

## Rules

1. **Nothing evaluated may be trained on.** Enforced by content hash, source
   URL, media identifier, creator and shoot id — not by filename, because a
   re-upload under a new name is the same footage.
2. **Selection happens on validation only.** Not on the regression benchmark,
   not on the faces benchmark, never on the confirmation set.
3. **The confirmation set stays closed** until the corpus, the methodology, the
   model configuration and the seed-selection rule are all frozen and
   committed. It is then evaluated exactly once.
4. **A benchmark is never promoted to training data.** If a benchmark is
   exhausted, the answer is a new benchmark, not recycling the old one.
5. **Roles are declared in the manifest**, in a `role` field, so a corpus
   cannot be used for something it was not built for without that being visible
   in a diff.
