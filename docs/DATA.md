# Data and training documentation

How AetherVSR's corpora are assembled, split and used, and which numbers each
one is allowed to support.

## Corpora

| Corpus | Path | Source | Licence | Role |
| --- | --- | --- | --- | --- |
| Training corpus | `data/corpus/` | Wikimedia Commons, CC-Zero category | CC0 only | train / val / test splits |
| Independent test | `data/eval-independent/` | Metropolitan Museum of Art Open Access | CC0 | held-out test, never trained on |
| Natural regression set | `public/eval/` | crops of the training corpus | CC0 | regression only — **overlaps training** |
| Video benchmark | `data/video/` | procedurally generated | n/a, synthetic | ground-truth video test |

Corpus images are gitignored; manifests recording provenance, licence and
SHA-256 are committed. The natural regression crops and the video corpus are
committed so those benchmarks run from a clean checkout.

**The video corpus is synthetic.** It is deterministic, licence-clean and
deliberately high-frequency, but it is not captured footage, and results on it
should not be read as results on real camera content.

### Licence policy

CC0 only. Wikimedia's generic "public domain" tag is a claim about some
jurisdiction — usually that a term expired — and is not a dedication the
uploader made. Five such files were found in the original 500-image corpus and
removed; the model was retrained on the remaining 495 rather than restating the
claim to match the corpus. Both `tools/fetch-corpus.py` and `tools/train.py`
enforce the same three accepted spellings, and the trainer streams and hashes
every file against the committed manifest before sampling.

## Splitting

The split unit is the **source image**, never the extracted patch.

```
all source images
        ↓  deterministic, content-addressed
 ┌───────────┬────────────┬───────────┐
 │ train 326 │ val 88     │ test 81   │
 └───────────┴────────────┴───────────┘
        ↓  patches extracted independently within each split
```

Assignment is `sha256(salt + ':' + sha256(file_bytes))` mapped to `[0,1)` and
bucketed. Consequences that matter:

- independent of filesystem or manifest ordering
- stable across machines and repeated runs
- unchanged by patch size, patch count or extraction logic
- stable under corpus growth — an image's bucket depends only on its own bytes
- a duplicate photograph under two filenames lands in one bucket, not two

Nominal 70/15/15, realised 65.9/17.8/16.4. Hash bucketing gives approximate
counts and **the salt is not tuned** to produce rounder ones; searching salts
for a target count is selecting a split.

Rebuild with:

```bash
python3 tools/dataset.py --manifest data/corpus/manifest.json --out data/splits/corpus-v1.json
```

`test/dataset-split.test.ts` enforces pairwise disjointness at content hash and
filename, in CI, with no Python and no GPU.

## Split roles

| Split | May be used for | May not be used for |
| --- | --- | --- |
| train | gradient updates | anything else |
| val | checkpoint selection, hyperparameter and degradation tuning | final reporting as generalisation evidence |
| test | one evaluation of a frozen model | any selection decision |

`tools/train.py` does not load the test split at all — not to log a number, not
to draw a curve. A test metric available every epoch will eventually be looked
at, and the cheapest guarantee is for the data to be absent from the process.
`tools/evaluate.py` scores a frozen model on a named set, once.

## Independence of the test corpus

Evaluating on more Wikimedia CC-Zero images would test a different sample, not a
different source. The independent corpus comes from a different institution with
its own imaging pipeline, and every candidate must pass three checks:

1. **exact** — SHA-256 of the downloaded bytes against every training hash
2. **source identity** — source URL against every training URL
3. **perceptual** — 64-bit dHash within Hamming distance 10 of any training
   image, which catches the same photograph resized or re-encoded

On the shipped fetch: 0 exact, 0 URL, 3 perceptual rejections. The perceptual
check firing is evidence it works rather than a claim that it would.

## Degradation

`tools/degrade.py`. `box` is the clean condition and is `torch.equal` to
`box_downsample2`. Compression profiles perform a real libx264 encode and
decode; CRF tiers were measured on real 720p crops rather than chosen. Chroma
and range are explicit throughout (full-range RGB ↔ yuv420p BT.709 limited), and
encodes pin threads so results reproduce.

Training-time throughput at batch 32 is ~607 patches/s, so the real encode is
used during training rather than an approximation of it.

## Reproducing

```bash
# corpora
python3 tools/fetch-corpus.py --out data/corpus --count 500 --min-side 640
python3 tools/fetch-independent-eval.py --count 60
python3 tools/dataset.py

# five seeds of each condition
tools/run-seeds.sh box models/box
tools/run-seeds.sh realistic models/realistic

# evaluation
python3 tools/aggregate-seeds.py --models 'models/box/seed*.json' --sets val,test,independent
python3 tools/cross-degradation.py --set independent --limit 30
python3 tools/videobench.py --prepare && python3 tools/videobench.py --score-baselines --json
python3 tools/videobench-neural.py --model public/models/aethersr-c16d2-realistic.json
```

Corpus fetches depend on live upstream APIs, so exact image sets may drift.
Manifests pin what was actually used, by hash.
