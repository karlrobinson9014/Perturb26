# Research log (append-only, newest at the bottom)

Each entry: `## <date> iter <N>: <title>`, then hypothesis, plan, result (numbers + command), conclusion.

## 2026-10-19 iter 0: bootstrap (human-supervised setup session)
- Scorer: `bench/official.py` calls upstream `PerturbValidator.verify_and_score` on a stub. Parity with
  `miner26/scoring.estimate_score` is within 2e-5 (float32 delta rounding), see `tests/contract/test_scoring_parity.py`.
- Alias collision confirmed in torchvision categories: 264 "Cardigan" -> 474 "cardigan" (the only one).
- v0 attack on TinyNet (CPU, 6 s): sign attack reaches margin 0.6 at 1/255; v0 with safety 0.1 flips at step 4
  and prunes to 17k of 360k values (est 0.946). With safety 1.0 it falls back to the max-margin flip (est 0.902).
  The fallback was added after v0 first returned an unflipped seed when nothing cleared the safety margin.
- Dev set: 64 shards from `data/validation` of perturb-ai/efficientnet-v2-l-adv-dataset @164079ef, with
  336 competitor adversarials (about 5 per image) for `tools/competitor_scores.py`.
- Not yet measured on the real model on GPU; that is the first build item.
- Real model, CPU (`python -m bench.run_bench --limit 2 --budget 300 --variants default`):
  images 20260922T053148Z-0003 / 20260923T033553Z-0002 scored 0.963 / 0.942 (competitor best 0.981 / 0.975).
  The second image: margin 1.00 (exactly the safety floor), 80,964 nonzero values, 159 steps, first flip at step 9.
  Reading: v0 flips easily at 1/255; the gap to the leaders is margin (+0.03 available) and sparsity.
  Raw: results/cpu-sanity-v0.json.

## 2026-10-19: GPU baselines (RTX 4090, 22 s budget, all 3 variants)
- `python -m bench.run_bench --out results/dev-v0.json`: mean 0.9536, success 64/64, p95 20.8 s, max 22.3 s,
  0 variant disagreements. Margins mostly 1-4 logits, nonzero 10k-120k values: far from the competitor
  envelope (best responses: margin ~10.0, median 3.1 % nonzero; `results/competitors-dev.json`).
- `python -m bench.run_bench --attack baseline --out results/dev-baseline.json`: mean 0.3371, 33/64 success.
- Throughput (eager, B=1, 480x480): fwd+bwd ~125 ms in fp32, fp16 or bf16 alike; channels_last is slower.
  Looks launch-bound (many small kernels), so batching and CUDA graphs / torch.compile should pay off.

## 2026-10-19: what limits the score (human-supervised session; prototypes in a scratchpad, numbers on dev images 1-8)
Est = `miner26.scoring.estimate_score` on fp32-verified candidates (parity with official < 2e-5).

**Speed.** EfficientNetV2-L at B=1 is launch-bound: eager fwd+bwd ~125 ms in fp32, fp16 or bf16 alike.
CUDA-graph capture of the fp16 fwd+bwd: 18 ms (B=1), 44.5 ms (B=4, 11 ms/img), 75 ms (B=8). fp32 graph 24 ms.
torch.compile: 79 ms after a 162 s compile. Outputs of graph replay equal eager exactly.

**Margin.** v0 lands at margins 1-4; the score pays 0.003/logit up to 10. Every new method targets 10.05.

**Search methods tried (untargeted margin objective, 20 s, images 1-4 unless noted):**
| method | mean est | note |
|---|---|---|
| top-k momentum projection (k shrinks on success) | 0.9716 (8 img) | k schedule stalls; slower schedules never reach margin 10 |
| ISTA: d in [-1,1], STE round, L1 soft-threshold, adaptive lambda | 0.9708-0.9736 | best plain method; insensitive to lr (0.005-0.05), lam0, up/down, lam from step 0 |
| ISTA 60 s instead of 20 s | 0.9731 | plateaus by ~300 steps (~6 s at 18 ms) |
| overshoot (margin 12/14) then first-order prune | 0.9684-0.9709 | worse: removals are too nonlinear at that margin |
| sparse-group lasso (4/8/16 px blocks) | 0.9715-0.9730 | clusters like competitors, no fewer values |
| greedy accretion (add top-B |grad| per step) | <= 0.963 | often never reaches target |
| random init noise, 5 seeds | spread < 0.1 % nz | no diversity to exploit |

**The real lever: the class you flip to.** Competitors' best responses (results/competitors-dev.json, top-2 per image,
128 responses): the adversarial class is ranked >= 20th in the clean logits for 78/128, often 400th+, with the
target logit pushed to 12-17. On the same image, the far class beats the runner-up (image 3: schipperke rank 408
0.9807 vs cheetah rank 1 0.9773). ISTA **targeted** at the competitor's class (`l_j - l_true`) gives
0.9806 / 0.9760 / 0.9805 / 0.9762 = mean 0.9783 vs competitors 0.9782 on images 1-4: the search is as good as theirs;
choosing the class is the gap. Warm-starting ISTA from the competitor's perturbation keeps or improves it
(2.46 -> 2.46 %, 2.50 -> 2.34 %, 4.73 -> 4.66 % nonzero): better optima exist, ISTA from zero lands elsewhere.

**Finding the class (open problem):**
- linear screen at the clean image (logit gap + top-2 % |grad| of every class, 30-45 s for 1000 classes): the
  competitor classes rank 405 / 605 / 602 / 129: useless. The cheap classes are nonlinear (feature creation).
- dense 1/255 sign attack, rank classes by logit rise: not predictive either. Dense +-1/255 everywhere only
  reaches margin 0-10, i.e. concentrated sparse changes are far stronger than dense ones.
- objectives that only push the true class down let far classes emerge: `-l_true` ended in the competitor's
  class on 463 (img 1), 680 (img 2), 293 (img 8); `tau*LSE(others) - l_true` (tau=1) found 463 at 2.62 % on img 1.
  But those lanes end with many values (they don't optimise the margin).
- v1 (miner26/attack.py): 4 lanes in one batch (margin, LSE tau=1, -l_true, LSE tau=0.3), weakest 2 re-target
  the surfaced classes at 35 % of the budget. Prototype: 0.9718 on images 1-8 (competitors 0.9761, v0 ~0.954).

**Geometry bug to avoid:** torchvision CenterCrop offset is `int(round((W - 480) / 2))`, half to even:
width 591 -> 56, not `(W - 480) // 2 = 55`. v1 verifies the offset numerically per shape.
- **v1 bench** (`python -m bench.run_bench --limit 16 --out results/dev16-v1.json`, RTX 4090, 22 s, 3 variants):
  mean 0.9738, 16/16 success, 0 variant disagreements, time max 21.9 s, |est - official| <= 3e-6.
  Same 16 images: v0 0.9553, competitors' best per image 0.9771 (v1 beats it on 0/16). Next: class discovery.

## 2026-10-19 iter 1: target-class race (discover, then target every lane, replace losers)
Hypothesis: v1 spends ~35 % of the budget with 4 untargeted lanes and only re-targets 2 of them once, so most images
end in near classes (classes match competitors' best on 3/16; mean nonzero 6.9 % vs their ~3 %). A targeted lane's
ranking at 60-100 steps already predicts its final ranking (probe, image 1: 463 leads from step 60, final 0.9803;
image 3: 223/224 lead from step 100, final 0.9800 vs v1 0.9735 / 0.9793), so a race should pay: short untargeted
discovery, then all lanes targeted at the top candidate classes from the lanes' fp32 top-k logits, then replace the
worst lanes with the next candidates; retired lanes' candidates stay eligible.
Literature (notes/literature.md): dominant-class UAP papers suggest a global prior; dev competitors show none (best
class 151 on 4/64 images), so no prior. MALT-style linear target choice already failed here (rank 129-605).
Plan: `tools/class_eval.py` (est only, same 16 images as results/dev16-v1.json) for configs, then the full dev bench.

## 2026-10-19 iter 2 (build): competitor proxy on GPU under all numeric variants
`python -m tools.competitor_scores --variants default,tf32_off,tf32_on --out results/competitor-variants-dev.json`
(RTX 4090, 64 dev images, 336 dataset submissions; the tool now keeps per-variant score/margin per response).
- Same aggregate as the CPU run: mean best-per-image 0.97837, median 0.97917, max 0.98584.
- **0 variant disagreements** on 336 responses; largest score spread between variants 2.8e-5. Even responses
  with margin 0.009 pass under all three variants on this GPU (selection bias: the dataset keeps responses that
  passed at submission). TF32 numerics do not separate competitors here.
- Best response per image: min-over-variants margin median 9.994, p10 9.98, max 10.03; only 13/64 >= 10 and 1 below 2
  (1.57). The leaders stop exactly at the saturation point and spend nothing beyond it.
- vs our seed bench (`results/harness/dev-seed-1223aac.json`, 0.97484): our margin median 10.02 (54/64 >= 10),
  so margin is not the gap. Best competitor beats us on **64/64** images by mean 0.0035; their best responses'
  RMSE median 0.00069 vs ours 0.00086 (their nonzero fraction median 3.1 %). The gap is RMSE / sparsity at the
  same margin (and, per iter 1, the class that makes low RMSE possible).

## 2026-10-19 iter 3 (build): fast task fetch in the live miner (`miner26/fetch.py`)
Upstream `_fetch_new_task_at_boundary` lets one `GET /task` timeout (10 s) raise out of the loop, which skips the
round; its retries sleep 2 s. Now: 3 s per request (`PERTURB26_FETCH_TIMEOUT_SECONDS`), 0.5 s between retries,
errors and timeouts retried until boundary + 34 - upload reserve - 5 s (`PERTURB26_MIN_ATTACK_SECONDS`). The
pre-boundary snapshot uses the same loop until boundary - 1 s. A task is "new" only if its id differs from the last
one *and* its epoch prefix is not older than the current boundary, so a failed snapshot can no longer make the
miner attack the previous round's task. `_process_task` takes the boundary from the id epoch, records the task as
processed, and downloads the image with 4 s per attempt plus retries. Tests: `tests/test_fetch.py` (local fake
HTTP server that hangs / returns 500 before serving; upstream `get_current_task` / `image_url_to_b64` unchanged).
Live from the GPU box (inline script calling `fetch_new_task` exactly as the miner does, 2 boundaries):
new task seen at boundary + 7.3 s (3 attempts) and + 2.2 s (3 attempts), image downloaded 0.6 s later.
`python -m miner26.neuron --dry-run`: task 1792417440-hf-..., official 0.97051, margin 10.035, 317 steps, 22 s.
Also fixed: dry run printed nothing because bittensor's root handlers made `logging.basicConfig` a no-op (`force=True`).

## 2026-10-19 iter 4 (build): timing probe from the GPU box
`python -m tools.task_probe --boundaries 10 --out results/timing-probe-gpu-2026-10-19.json`: 11 task changes,
0 poll errors; new task visible at boundary + 2.0 s median (1.3-2.7 s), GET /task 0.31 s median (one 6.3 s spike),
download 0.50 s median (max 0.72 s), clock +0.4 s vs server. The worst case leaves ~25.6 s for the attack (submit-by 34 s,
5 s reserve), more than BENCH_BUDGET 22. The iter-3 + 7.3 s observation did not recur. No harness proposal. notes/timing.md.

## 2026-10-19 iter 5 (build): live dry runs on 5 consecutive tasks
New: `python -m miner26.neuron --dry-run --rounds N --out <json>` waits for the next N boundaries and handles each
new task like the live miner (fast fetch loop, boundary from the id epoch, attack deadline boundary + 34 - 5 s),
scores it with `bench.official.OfficialScorer`, submits nothing. Plain `--dry-run` (current task, `--budget`) unchanged.
`python -m miner26.neuron --dry-run --rounds 5 --out results/dry-run-live-2026-10-19.json` (RTX 4090, default config):

| task | official | margin | nonzero | steps | seen | attack start | done |
|---|---|---|---|---|---|---|---|
| 1792421040-...-1235189 | 0.96875 | 10.002 | 75502 | 398 | +2.9 s | +3.8 s | +28.7 s |
| 1792421160-...-0856873 | 0.98047 | 10.015 | 26903 | 370 | +2.7 s | +3.5 s | +28.7 s |
| 1792421280-...-0529088 | 0.98328 | 10.001 | 14474 | 356 | +3.0 s | +3.9 s | +28.7 s |
| 1792421400-...-1006678 | 0.97102 | 10.039 | 68495 | 333 | +2.8 s | +3.5 s | +28.7 s |
| 1792421520-...-0508541 | 0.98073 | 10.040 | 21243 | 365 | +2.0 s | +2.8 s | +28.7 s |

Mean 0.97685, 5/5 success, 0 variant disagreements, est == official to 3e-6. Live the attack gets ~25 s (more than
BENCH_BUDGET 22) because it runs to boundary + 29 s; with the 5 s upload reserve the submission lands by ~+34 s,
6 s before the validator's first `/submits` poll at + 40 s. Seen-at-boundary + 2.0-3.0 s matches iter 4's probe.

Bug found on the way: the 5-round log file was empty. bittensor's default logging state (`before_enable_default`)
sets every non-bittensor logger to CRITICAL when bittensor is imported/configured, so all `miner26` logs (dry run,
fetch retries, and the live miner's `Submitted task=...` lines) were silently dropped; iter 3's `basicConfig(force=True)`
did not help because the logger level, not the handler, was the problem. Fix: `_setup_miner_logging()` gives the
`miner26` logger its own stderr handler at INFO with propagate off, called in the dry run and after
`PerturbMiner.__init__`. Test: `tests/test_neuron_dry_run.py`. Verified: `--dry-run --budget 5` prints again
(task 1792421640-..., official 0.97952). The live-miner side still needs a check in the pm2 log (backlog).
