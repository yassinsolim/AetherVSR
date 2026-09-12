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
| Six serialized paired training runs | Complete in the run-evidence commit | results/m8-runs.json; all six complete 81,180 updates, 60 draws, full paired stream equality |
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
5. Recovery reviewer approved retaining the independently verified completed
	R0-8101 and restarting only the interrupted R3-8101 at step zero, with the
	same seed and unchanged source. The incomplete attempt is excluded, not an
	additional checkpoint draw or a best-of-restarts option.
6. First post-training reviewer observed R3 deterioration but incorrectly
	called it a proven P0 code defect without identifying a violated invariant.
	Its identity-gradient explanation, assertion of historical gradient
	finiteness from finite loss, and suggestions to tune LR or add clipping/BN
	were rejected. Poor outcomes alone do not invalidate the registered test.
7. A second independent, read-only review verified all 78 run artifacts, 520
	frozen input files, unchanged source/registration, saved Adam counters and
	settings, all 60 validation draws, and complete replay of 2,597,760 patch
	presentations per run. Paired stream hashes match for every seed. CPU export
	checks and independent forward/input/parameter-gradient checks of the
	analytically embedded branches passed. No validity-changing defect was found.
	This review permits frozen captured scoring after the run evidence commit;
	it does not approve confirmation or deployment. R3's deterioration is
	measured; its mechanism remains unresolved and no run is removed for it.

## Power-loss recovery and completed training

Training executed at e60b11303f4adf337692877eea32ec0307acfd39. All six runs are
complete, with 81,180 optimizer updates, 60 eligible validation draws and all
seven diagnostic snapshots. Production, registration and training code did not
change. Detailed commands, hashes, toolchains, curves and paired streams are in
`results/m8-runs.json`.

The user reported battery exhaustion after R0-8101 completed. R3-8101's last
durable log was validation update 79,827; actual interrupted update count and
duration are not measured. No resumable model/Adam/RNG/selection state existed.
Its three files are preserved byte-for-byte under
`models/m8-interrupted/power-loss-01`, pinned by `results/m8-recovery.json`.
The recovery record was written before restarting, but was kept untracked to
preserve the frozen execution HEAD; no pre-restart Git commit is claimed.
R0 was independently reverified and retained. R3 restarted from zero with
seed 8101, and the remaining four runs followed in registered serial order.
All incomplete-attempt scores are excluded from selection and statistics.

The Mac was on AC power for recovery. Wrapping the trainer in `caffeinate`
caused a pre-launch concurrency-guard refusal because macOS leaves a child
whose arguments repeat the training command. No optimizer update occurred in
that refused launch. A separate `caffeinate -is` process prevented sleep
without changing the concurrency guard or training code.

All completed runs used Apple M5, 24 GiB, macOS 26.6.2 and PyTorch 2.14.0/MPS.
Measured subprocess wall seconds (Popen through wait and log drain, excluding
runner verification): R0-8101 1704.4349; R3-8101 2252.1507; R3-8102 2179.7717;
R0-8102 1450.9740; R0-8103 1454.7601; R3-8103 2150.4861. Per-run training-loop
durations and updates/second in the JSON include scheduled validation and
snapshot work; none of these are optimizer-kernel timings or inference results.

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