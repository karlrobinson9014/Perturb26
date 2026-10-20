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
- If the harness context below contains an **Operator task for THIS iteration**, that task is your
  experiment; the operator directives override the defaults in this prompt.
- `CLAUDE.md`, `status/STATUS.md`, `status/best.json` (accepted best), `status/results.tsv`
  (every past experiment and its outcome), `notes/research_log.md`, `notes/ideas.md`, `notes/scoring.md`.
- Don't repeat an experiment the ledger or log already shows as failed unless you have a concrete
  reason it will behave differently; say what that reason is.

## 1b. Literature (when the idea is new or you are stuck)
You have two research connectors: Consensus (`mcp__claude_ai_Consensus__search`) and SciSpace
(`mcp__claude_ai_SciSpace__search-papers`); load them with ToolSearch if they are not listed yet.
Use them to find methods for the problem at hand (sparse / L0 adversarial attacks, target-class
selection, quantized perturbations, attack speed), at most ~6 searches per iteration. Record what you
found in `notes/literature.md`: one bullet per paper with title, year, link, the one idea that applies
here and whether you tried it. Read that file first so you don't repeat searches. A paper is a source
of hypotheses, not evidence: only the bench decides.

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
- Long benches: a full dev bench takes ~25 min, more than one foreground Bash call allows. Start it in the
  background and wait on its **PID** (`python -m bench.run_bench ... > log 2>&1 & pid=$!`, then
  `while kill -0 $pid 2>/dev/null; do sleep 30; done`). Never wait with `pgrep -f "<pattern>"` or
  `pkill -f`: the pattern also matches the waiting shell's own command line, so the wait never ends (iter 204
  idled ~60 min that way until the harness killed it). The harness kills the session at 90 min; plan for <= 70.
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
