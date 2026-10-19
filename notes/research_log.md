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
