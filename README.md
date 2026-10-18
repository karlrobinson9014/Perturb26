# Perturb26

A competitive miner for the [Perturb](https://github.com/0xsigurd/Perturb) Bittensor subnet, plus an
autonomous improvement loop (Ralph-style build iterations and autoresearch experiments, driven by the
`claude` CLI under pm2) with an independent reviewer that has veto power.

## Where things stand
| | score (mean official, dev, 64 real task images) |
|---|---|
| theoretical ceiling | 0.99303 |
| network: best response per image (proxy for rank 1) | 0.9784 (max single response 0.9858) |
| upstream baseline miner (PGD, 2/255) | ~0.6-0.7 (to be measured on GPU) |
| Perturb26 v0 | not yet measured on GPU |

Details: `notes/scoring.md` (why 1/255 and sparsity win), `notes/competition.md`.

## Layout
```
miner26/      product: attack, score model, live neuron
bench/        benchmark with the upstream verifier (protected)
tests/        contract tests (protected) + unit tests
tools/        analysis scripts (competitor scores, ...)
harness/      loop.sh, review.sh, gate.sh, lib.sh, ctl.sh, ecosystem.config.js (pm2), setup_box.sh
prompts/      build.md, research.md, review.md
notes/        knowledge base the agents read and extend
status/       STATUS.md, backlog.md, best.json, results.tsv (experiment ledger), iterations.tsv, reviews/
results/      bench result JSONs
logs/         raw agent transcripts (stream-json) and harness logs, not committed
upstream/     pinned upstream Perturb (submodule, 0dad954)
```

## How the loop works
Each iteration is a fresh headless session: `claude -p --output-format stream-json` with
`prompts/<mode>.md` and a context block (best score, recent ledger rows).

1. **sync**: fetch, rebase the `loop` branch, merge `main` (human edits to the harness and prompts land there).
2. **agent**: build mode does the first open `[build]` backlog item; research mode runs one experiment.
   The agent commits on `loop` and updates notes and status.
3. **guards**: protected-path check, then `harness/gate.sh` (pytest, import, smoke bench). The harness runs them, not the agent.
4. **measure**: if `miner26/` changed, the harness runs the full dev bench itself.
5. **review**: on a gain above `MIN_GAIN`, a separate `claude -p` reviewer session (`prompts/review.md`)
   checks the diff for overfitting, re-runs the tests, runs the **holdout** (128 images, outside the repo)
   and writes a verdict. The harness cross-checks the verdict against the raw holdout file.
6. **decide**: accept means `status/best.json` is updated, `loop` is merged into `main` with tag
   `best-iterN`, and both are pushed (with `PUSH=1`). Discard means the code is reset and the notes are kept.
   Every outcome goes to `status/results.tsv`.

The loop stops on `STOP`, `MAX_ITERS`, `MAX_COST_USD`, or when dev and holdout both reach `TARGET_SCORE`.

## Setup (GPU box, Linux + CUDA)
```bash
git clone --recurse-submodules https://github.com/karlrobinson9014/Perturb26.git && cd Perturb26
bash harness/setup_box.sh          # venv, deps, weights, pm2, claude CLI, dev + holdout data, gates
claude setup-token                 # once, so headless sessions are authenticated
$EDITOR harness/loop.env           # model, budgets, PUSH=1 once git push works on the box
bash harness/ctl.sh start          # pm2 start perturb26-loop
bash harness/ctl.sh status         # best, ledger, spend
bash harness/ctl.sh tail           # harness log
bash harness/ctl.sh stop           # finish the current iteration, then exit
```

## Deploying the miner
The loop never deploys. To mine with an accepted version (on a different GPU from the loop):
```bash
git checkout best-iterN
cp miner.env.example miner.env && $EDITOR miner.env     # wallet, NETUID, storage keys
pm2 start harness/ecosystem.config.js --only perturb26-miner
```
