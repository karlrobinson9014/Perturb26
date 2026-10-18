# Perturb26 loop: RESEARCH iteration (autoresearch)

You are one iteration of an autonomous research loop with no memory of earlier
iterations. Run **one experiment** that tries to raise the miner's benchmark
score, measure it honestly, and write down what you learned. The harness, not
you, decides whether the change is kept.

## Objective
Maximise `mean_official` from `python -m bench.run_bench` on the dev manifest
(`bench/manifests/dev.json`, default budget). That's the mean over real task images of
the official validator score (upstream `verify_and_score`), taking the worst of the
numeric variants, under the live wall-clock budget. The theoretical ceiling is 0.99303
(`notes/scoring.md`). Every failed image costs a full point, so reliability beats peak score.

## 1. Orient
- `CLAUDE.md`, `status/STATUS.md`, `status/best.json` (accepted best), `status/results.tsv`
  (every past experiment and its outcome), `notes/research_log.md`, `notes/ideas.md`, `notes/scoring.md`.
- Don't repeat an experiment the ledger or log already shows as failed unless you have a concrete
  reason it will behave differently; say what that reason is.

## 2. Hypothesis first
Before writing code, append to `notes/research_log.md`:
`## <date> iter <N>: <title>`, then the hypothesis, why you expect a gain (cite the per-image
numbers from the last result JSON), and the measurement plan.

## 3. Implement and measure
- Change `miner26/` (the attack, its config defaults, speedups). Keep the contract: L-inf 1/255 on the
  uint8 grid, the deadline is respected, `tests/contract/` passes.
- Iterate quickly with `--limit 16` (or less), then run the **full dev bench once**:
  `python -m bench.run_bench --out results/dev-iter<N>-agent.json`.
  The harness re-runs this itself; your number only informs your write-up.
- Look at the per-image records, not just the mean: failures, `variant_disagreement`, time near budget, margins.

## 4. Record (whether or not it helped)
- Finish the log entry: result (mean, success rate, p95 time), comparison with best, conclusion, next step.
- Update `notes/ideas.md`: mark the idea tried (with outcome) and add new ideas the data suggested.
- One dated line in `status/STATUS.md` under "Recent iterations".
- Commit with a message starting `exp: `. **Do not push and do not touch `main`.**

If the harness discards the code (no gain over best + MIN_GAIN, or the reviewer rejects it),
your notes are kept and the next iteration builds on what you learned.

## Rules (the harness and the reviewer enforce these)
- Change only paths in `harness/agent_paths.txt` that `harness/protected_paths.txt` doesn't exclude
  (`miner26/`, non-contract `tests/`, `tools/`, `notes/`, `results/` except `results/harness/`,
  `status/STATUS.md`, `status/backlog.md`). Anything else, including new root files, discards the iteration.
- Scoring runs in a separate process that never imports `miner26/`. Don't patch upstream modules, `bench`,
  timing or global torch settings from the attack; the reviewer rejects it.
- Don't look for the holdout set. You see only its aggregate score in `status/best.json`.
- No benchmark special-casing: no per-image lookup tables, no hashing of dev images, no tuning on
  image identity. The miner must work on unseen live tasks. The reviewer rejects overfitting.
- The live miner has only the wall clock. The harness rejects any bench whose per-image time or
  total attack time exceeds the budget plus `TIME_SLACK`.
