# Milestone 8 status

M8 is in progress. Production remains unchanged. No matched-budget result,
candidate qualification, confirmation score or deployment result is claimed.
Completed rows require a committed artifact; local progress alone is not done.

| Work | State | Evidence or gate |
|---|---|---|
| Starting Git state, local gates and CI | Complete in this registration commit | Recorded in M8 preregistration: 168 Vitest, 106 Python, build pass, CI success |
| Correct M8-M11 roadmap numbering | Complete in this registration commit | Future headings checked; completed milestone text unchanged |
| Commit preregistration and seed/config plan | Complete in this registration commit | docs/M8-PREREGISTRATION.md; results/m8-plan.json; independent design approval |
| Permanent paired initialization tests | Not started | No long training before proof |
| RNG, patch/crop identity and augmentation proof | Not started | At least eight complete batches |
| LR, exact budget and checkpoint fairness tests | Not started | 81,180 updates; 60 eligible draws |
| Paired integration smoke and fusion/export | Not started | Commit passing evidence before binding runs |
| Six serialized paired training runs | Not started | Seeds 8101-8103, R0/R3, fixed order |
| Fixed-final and best-validation scoring | Not started | Complete frozen 48-cell validation only |
| Per-seed statistics and all category/tier cells | Not started | Registered paired test; no favorable test switching |
| Horizon curves and historical M7 comparison | Not started | Distinguish partial long schedule from M7 short schedule |
| Matched-budget continuation decision | Not started | Primary fixed-final rule only |
| Confirmation source and category audit | Not started | Existing proposal is unfrozen and lacks text |
| Confirmation disjointness and committed freeze | Not started | Before any candidate scores |
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