# Milestone 8 status

M8 is in progress. Production remains unchanged. No matched-budget result,
candidate qualification, confirmation score or deployment result is claimed.
Completed rows require a committed artifact; local progress alone is not done.

| Work | State | Evidence or gate |
|---|---|---|
| Starting Git state, local gates and CI | Complete in this registration commit | Recorded in M8 preregistration: 168 Vitest, 106 Python, build pass, CI success |
| Correct M8-M11 roadmap numbering | Complete in this registration commit | Future headings checked; completed milestone text unchanged |
| Commit preregistration and seed/config plan | Complete in this registration commit | docs/M8-PREREGISTRATION.md; results/m8-plan.json; independent design approval |
| Permanent paired initialization tests | Complete | f806f0c; results/m8-paired-initialization.json proves all three seeds on CPU/MPS before training |
| RNG, patch/crop identity and augmentation proof | Complete | results/m8-smoke.json; identical eight-batch/256-sample prefix and patch-byte hashes; source crop coordinates unavailable |
| LR, exact budget and checkpoint fairness tests | Complete | f806f0c; permanent tests cover 81,180 updates, 60 eligible draws and 16,200 snapshot exclusion |
| Paired integration smoke and fusion/export | Complete in this evidence commit | results/m8-smoke.json and results/m8-scoring-parity.json; real serialized MPS smoke passes |
| Six serialized paired training runs | Not started | Seeds 8101-8103, R0/R3, fixed order |
| Fixed-final and best-validation scoring | Not started | Complete frozen 48-cell validation only |
| Per-seed statistics and all category/tier cells | Not started | Registered paired test; no favorable test switching |
| Horizon curves and historical M7 comparison | Not started | Distinguish partial long schedule from M7 short schedule |
| Matched-budget continuation decision | Not started | Primary fixed-final rule only |
| Confirmation source and category audit | Blocked, not frozen | Independent source review below; camera-native/interval evidence and qualifying text additions unresolved |
| Confirmation disjointness and committed freeze | Blocked, not run | No unconditional alias/shoot signoff or category-complete native-source proof; no confirmation scores |
| Five fresh R3 production seeds | Conditional, not run | Only after committed PERSISTENT ADVANTAGE decision |
| Validation +0.10 dB production gate | Conditional, not run | Candidate selected on validation only |
| One-time confirmation, weak cells and bootstrap | Conditional, not run | Committed freeze and candidate SHA first |
| Temporal and runtime/resource parity | Conditional, not run | Actual memory is not measured by WebGPU |
| Independent evidence and literature review | In progress | First configuration review received; unsupported predictions rejected |
| Production decision and recommendation | Not started | No automatic promotion; no next-milestone implementation |
| Final tests, hygiene, fresh clone, push and CI | Not started | Clean HEAD == origin/main and completed/success CI |

M7 remains closed at 42/46 done, zero open, four blocked. Its blocked
stem/head work and actual-memory measurement are not silently added to M8.

## Independent review ledger

1. Initial configuration reviewer (read-only): confirmed source constants and
	frozen-cache availability. Root rejected its unsupported claims that identity
	compensation is inherent/impossible, that no trainer changes are needed, and
	that the literature predicts decay. Direct code inspection also corrected its
	description of checkpoint PSNR as luma: the trainer uses RGB global MSE.
2. Independent preregistration reviewer (read-only, before registration commit):
	approved the design with no P0/P1 blockers. Confirmed exact budget versus the
	recovered cache's 1,359 batches per epoch, 60 equal eligible draws, separate
	fixed-final/best exports, conditional production stages and three-pair test
	resolution. Requested the sign-exchangeability assumption, now explicit.
	Initialization, RNG, cache cardinality/hash enforcement and smoke are still
	implementation gates, not certified by this design approval.
3. Implementation reviewer found two P1 defects before any real smoke: CPU-only
	RNG forking paired with `torch.manual_seed` also reseeded MPS, and final weights
	were only in RAM until after best-checkpoint postprocessing. Both were repaired
	in f806f0c. Permanent tests check accelerator RNG preservation and final JSON/
	state persistence even when best-checkpoint SSIM processing throws. No binding
	run needed regeneration because none existed.
4. Final pre-training reviewer independently rehashed all 520 frozen files and
	smoke artifacts, verified the current source hashes against both reports,
	confirmed chronology and all training/report controls, and approved exactly
	six registered serial runs after this evidence commit. This is not approval
	to open confirmation or replace production. Full local gates at apparatus
	commit: 298 Python, 168 Vitest, typecheck/lint/build passed.

## Measured apparatus evidence

Registration: c95574336246ea5114afbec4d2a48e538afd3f7f. Apparatus and real-smoke
execution commit: f806f0cbca103a1d72b421d4a6981a14115b3552. Neither registration
file was amended. `results/m8-input-freeze.json` inventories the training inputs
and 512 captured validation PNGs (48 cells, eight paired frames each).

Real seed-8099 smoke, Apple M5 / 24 GiB / macOS 26.6.2 / PyTorch 2.14.0:
both arms completed eight updates, with the same stream SHA256
`d19edfe6f69afb9761dc76daa3e423d1c0b19e93df5070ec9d5432d3aa0c7995`.
Maximum initial errors: CPU 2.4437904357910156e-6, MPS 8.642673492431641e-7,
against the fixed 1e-5 bound. Both JSON/state reload checks passed on CPU/MPS.
Training-loop durations including eight full patch-validation passes and final
snapshot copying were R0 4.8863 s, R3 7.1771 s. These short-run timings are not
optimizer-only throughput and are not extrapolated into long-run measurements.

Scoring parity on one frozen full-resolution production frame and seeded odd-
shaped synthetic data: maximum pixel error 3.5763e-7, maximum PSNR difference
5.9605e-7 dB, maximum SSIM difference 1.5497e-6, all below 1e-5. No M8 candidate
was scored. Exact measurements and scope are in the parity JSON.

## Confirmation audit limitations

An independent source-only reviewer checked current Commons metadata and raw
pages for all 16 proposal clips and viewed their official posters. Licences,
encoded 3840x2160 dimensions, provider byte counts and SHA1 pins were corroborated;
nine first-2-MiB hashes were independently recomputed. Seven remaining downloads
were not retried after Commons returned HTTP 429 with Retry-After 600. Camera-
native resolution, absence of prior synthetic upscaling, active geometry and
overlays over each intended interval remain unresolved for every clip. Encoded
dimensions are not independent proof of native capture.

Specific issues remain: Alexey M. also uses Oleksiy Muzalyev / Alex-7; the lake
source jointly credits Spekking and Elke Wetzig; the interview has a PCM watermark;
the eclipse source has inconsistent place names; several sources are time-lapse
or accelerated; Mexico's metadata reports 1000 fps despite prefix timing near
30 fps. The previous seven-manifest audit omitted historical evaluation paths.
The review checked 25 revisions across 14 manifest/split paths without matching
the recorded identifiers, but unknown aliases or shared shoots are not cleared.

Two unaccepted text leads were investigated: Commons page 114955471 (Araisyohei,
CC BY 4.0, own-work Canon EOS R6 declaration) has sparse physical signage; page
47907747 (YeshuaAgapao / GreatInca, CC BY 3.0, Sony RX10-II declaration) had no
verified readable signage in inspected early frames. Neither has sufficient
independently verified native/usable-interval evidence for acceptance. The rally
proposal's physical wayfinding signs are visible but its native provenance is
also unresolved. No licence restriction to CC0 was invented, no ambiguous clip
was accepted, no source was scored, and no freeze is claimed.