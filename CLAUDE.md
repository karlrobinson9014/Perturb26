# Perturb26: rules for every Claude session in this repo

The product is a miner for the Perturb Bittensor subnet: given a task image, return an
imperceptible perturbation that flips torchvision EfficientNetV2-L, scored by the validator
formula in `notes/scoring.md`. The repo also contains the autonomous loop that improves it.

## Layout
| path | what | who edits |
|---|---|---|
| `miner26/` | product: `attack.py` (the attack), `scoring.py` (score model), `neuron.py` (live miner), `imaging.py` | loop agents |
| `bench/` | `run_bench.py` (attack phase), `score.py` (authoritative scorer, separate process), `official.py` (upstream verifier), `fetch_data.py`, `manifests/` | humans only |
| `tests/contract/` | invariants: score parity, admissibility | humans only |
| `tests/` (other) | unit tests for new code | loop agents |
| `tools/` | analysis scripts (competitor scores, probes) | loop agents |
| `notes/` | knowledge base: scoring, timing, competition, ideas, research log, harness proposals | loop agents |
| `status/STATUS.md`, `status/backlog.md` | current state, prioritised work | loop agents |
| `results/` | bench outputs; `results/harness/` is the harness's own | agents (except `harness/`) |
| `status/best.json`, `results.tsv`, `iterations.tsv`, `reviews/` | accepted best, ledgers, verdicts | harness only |
| `harness/`, `prompts/` | the loop (pm2 + claude CLI), prompts | humans only |
| `upstream/` | pinned Perturb checkout (git submodule); the source of truth for scoring | never |

Agents may change only what `harness/agent_paths.txt` allows minus `harness/protected_paths.txt`;
anything else (renames and new files included) discards the iteration.

## Commands (GPU box, repo root, `source .venv/bin/activate`, `export PYTHONPATH=$PWD`)
- tests: `python -m pytest -q tests`
- gates (what the harness runs): `bash harness/gate.sh`
- quick bench: `python -m bench.run_bench --limit 16`
- full dev bench: `python -m bench.run_bench --out results/dev-<tag>.json`
- try attack settings without code changes: `--attack-config '{"kappa": 6, "safety_margin": 1.5}'`
- live check, no submission: `python -m miner26.neuron --dry-run`

## Non-negotiables
- Output L-inf exactly 1/255 on the uint8 grid (2/255 costs 0.18 points). The attack is anytime and must
  return before its deadline.
- The validator's true class is `scoring.validator_true_index(clean_pred)` (label-alias round trip), not the argmax.
- Never special-case benchmark images (no lookup tables or hashes of dev images). Never look for the holdout.
- Commit your work; **never push and never touch `main`**. The harness benchmarks, gets an independent
  review, merges to `main` and pushes.
- Write down numbers with the command that produced them. A claim without a result file isn't a result.
