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
3. **guards** (run by the harness, not the agent):
   - changed paths must be on the allowlist (`harness/agent_paths.txt`, minus `protected_paths.txt`), renames included;
   - the venv fingerprint must be unchanged;
   - `harness/gate.sh`: contract tests under the harness's pytest config, other tests, import, smoke bench.
4. **measure**: if `miner26/` changed, the harness runs the full dev bench. Scoring happens in a separate
   process (`bench/score.py`) that never imports agent code. A bench that overruns the time budget is rejected.
5. **holdout**: on a dev gain above `MIN_GAIN`, the harness benchmarks the holdout (128 images, outside the
   repo, secret seed). It must stay within `HOLDOUT_TOL` of the best holdout ever accepted.
6. **review**: a separate `claude -p` reviewer session (`prompts/review.md`) audits **everything not yet on
   `main`**, re-runs the gates, reproduces 16 dev images, and reads the holdout digest. It has veto power.
7. **decide**: accept means `status/best.json` is updated, `loop` is merged into `main` with tag
   `best-iterN-<sha>`, and both are pushed (with `PUSH=1`). Discard means the code is reset and the notes are kept.
   Every outcome goes to `status/results.tsv`.

### Isolation
Agent-written code runs in three places: the agent session, its tests in the gate, and the attack
phase of every bench. After each of these the harness runs `integrity_check`: no stray or ignored
files in the tree, `upstream/` on its pin and clean, and no file in `.venv/` or `data/` with a newer
ctime than the loop's start time, which only the harness holds. Then the scorer runs as `python -I`
with a private pycache and imports nothing from `miner26/`. A violation discards the iteration and
everything it produced. If `.venv`, `upstream` or `data` changed, the loop stops for a human.

None of this stops code that runs **as the harness user** from reading the holdout or writing outside the repo.
For real isolation run agent sessions and attack phases as a second user that can write the repo but can't
read `$HOLDOUT_DIR` or write `.venv`:
```bash
sudo useradd -m perturb26-agent && sudo usermod -aG "$(id -gn)" perturb26-agent
chmod -R g+rwX . && find . -type d -exec chmod g+s {} +   # shared group, new files inherit it
git config core.sharedRepository group
sudo -u perturb26-agent git config --global --add safe.directory "$PWD"
chmod -R g-w .venv && chmod 700 "$HOLDOUT_DIR"
# harness/loop.env (both run under umask 002):
AGENT_RUN_PREFIX="sudo -u perturb26-agent -H --preserve-env=PYTHONPATH,ANTHROPIC_API_KEY"
BENCH_RUN_PREFIX="sudo -u perturb26-agent -H --preserve-env=PYTHONPATH"
```
The holdout images still pass through the attack phase (the attack has to see them), so with
`BENCH_RUN_PREFIX` empty an attack could copy them outside the repo, where `integrity_check` doesn't look.

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
