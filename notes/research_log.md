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

## 2026-10-19 iter 6: CUDA-graph lanes vs eager lanes (regression test)
`python -m pytest -q tests/test_graph_lanes_gpu.py` (RTX 4090, torch 2.12.0+cu130): EfficientNetV2-L, first crop-path dev
image, B=4 lanes with the default spec modes and the last lane targeted at class clean+1, sparse +-1 q (density 0-5 %
per lane). fp32 `check`: margins and argmax identical (max diff 0.0). fp16 `grad`: margins identical, gradient cosine
1.0, relative L2 error 0.0. So capturing with `cudnn.benchmark = True` did not change the numerics here; the graph path
is a pure speed-up. Test tolerances keep slack (margin 1e-3 fp32 / 0.05 fp16, cosine > 0.999) for other GPUs.

## 2026-10-19 iter 201 (build): early exit at the per-image ceiling
- `Attacker.run` now stops as soon as a kept (fp32, >= safety_margin) candidate scores the per-image ceiling:
  `_estimate(seed nonzero, total, margin 10, seed pixels)`. The novelty seed is always kept, so no candidate can
  beat that. In practice this fires on label-alias images (clean 264 -> validator true 474): the seed-only
  candidate at step 1 already has a huge margin. `info["early_exit"]` records it.
- Ordinary images never reach the ceiling (they need thousands of changed values), so their behaviour is unchanged:
  `bash harness/gate.sh` smoke 3/3, mean official 0.96927 at budget 8 (previous gate run 0.96880, noise).
- Test: `python -m pytest -q tests/test_attack_early_exit.py` (alias TinyNet exits after 1 step with the ceiling
  estimate; plain TinyNet runs to its deadline).

## 2026-10-19 iter 202: greedy prune phase after the search
Hypothesis: the ISTA lanes stop at margin ~10.0 with an L1-balanced support, not a minimal one (seed bench
`results/harness/dev-seed-1223aac.json`: margin median 10.02, 54/64 >= 10; RMSE median 0.00086 vs the leaders'
0.00069). ISTA plateaus by ~300 steps (research_log), so the last ~20 % of the budget is better spent on a
first-order greedy prune of the best fp32 candidate: gradient of (l_cls - l_true) at the candidate, drop the
nonzero values with the smallest contribution g*q, test 4 drop sizes in one batched fp32 check, keep the best
estimate (which already trades margin below 10 against RMSE), adapt the drop sizes. It only accepts improvements,
so it cannot lower the estimate; the risk is the lost search time. The iter-0 "overshoot then prune" failure was
pruning from margin 12-14 with large first-order steps; this prunes from ~10 in small, verified steps.
Plan: `--limit 16` with prune at 0.8 / 0.7 of the budget vs off, then the full dev bench.
Result (negative, code reverted; nothing in miner26/ changed):
- `python -m bench.run_bench --limit 16 --out results/dev16-iter202-p08.json` (prune from 0.8 of the budget):
  mean 0.97300, 16/16 success, 0 variant disagreements, time mean 18.9 s (prune stops early), vs the same 16 images
  in the seed bench 0.97365. Worse on 14/16 images (lost search time; image 3 landed in a worse class: 80k vs 41k values).
- Trace (scratch script, images 1 and 3, 22 s): the best candidate is already at margin 9.99-10.0 and *every*
  greedy removal by smallest g*q lowers the fp32 margin: dropping 0.5 % of the values costs 0.37 logit (img 1) and
  dropping 0.06 % (16 values) is break-even. Score per removed value ~2e-7 vs 0.003 per logit, so a removal must cost
  < ~7e-5 logit; first-order picks don't achieve that. Image 1: 3 accepts, +4e-6 total; image 3 (from margin 10.15):
  15 accepts, ~+4e-4, i.e. only the margin overshoot above 10 is recoverable.
- Conclusion: the ISTA lanes' L1 schedule (lam_up/lam_down around target_margin 10.05) already leaves a locally tight
  support at margin 10. Pure removal is redundant with ISTA; only swap moves (= ISTA itself) or a different basin
  (class / support) can lower RMSE. Confirms the iter-0 view that the gap is the class/basin, not the final polish.
- Side observation: image 3 ended at 0.97759 (41.5k values) in the seed bench but 0.9717 / 0.9709 (80-86k, classes
  293/256) in two runs here: the class a run lands in varies a lot between runs at the same config, so per-image
  variance from class discovery is large (~0.006 on that image). More/diverse discovery (restarts, more lanes) is
  still the lever; small polish ideas are below MIN_GAIN (0.002).

## 2026-10-19 iter 203: deferred fp32 checks (more search steps per budget)
Hypothesis: a v1 step is ~73 ms (seed bench image 1: 294 steps in 21.5 s), but the B=4 CUDA-graph fp16 grad is only
44.3 ms; the fp32 check graph costs 32.5 ms and runs on almost every step, because during ISTA the projected
estimate creeps up nearly every step (scratch profile, RTX 4090, 50 replays each). Keeping, per lane, the best
*projected* (fp16-margin) candidate since the last check and verifying all lanes' pending candidates in one
batched fp32 check every K steps (and before re-target / at the end) should give ~40 % more ISTA steps (K=4:
~52 ms/step) at nearly the same candidate quality. ISTA lanes were seen to plateau around ~300 steps (research_log
2026-10-19), and the re-targeted lanes get only ~190 of the 294 steps now, so they should gain most.
Plan: `check_every` in AttackConfig; `--limit 16` with K=1 (current behaviour) vs 4 vs 8, then the full dev bench.
Result (code kept in this commit, `check_every = 8` default):
- 16 images (`python -m bench.run_bench --limit 16 --attack-config '{"check_every": K}' --out results/dev16-iter203-kK.json`),
  vs the same images in the seed bench (0.97365):
  | K | mean | steps (mean) | success | disagreements | t max |
  |---|---|---|---|---|---|
  | 1 (old behaviour) | 0.97341 | 302 | 16/16 | 0 | 21.80 s |
  | 4 | 0.97393 | 408 | 16/16 | 0 | 21.82 s |
  | 8 | 0.97404 | 435 | 16/16 | 0 | 21.83 s |
- Full dev, K=8 (`python -m bench.run_bench --out results/dev-iter203-agent.json`): **mean 0.97529** vs best 0.97484
  (+0.00045), 64/64 success, 0 variant disagreements, time p95 21.82 s / max 21.85 s, steps mean 437 (seed ~300),
  higher than the seed on 60/64 images, nonzero median 44.2k.
- Conclusion: +44 % steps gives a consistent but small gain (+0.0005, probably below MIN_GAIN 0.002). It confirms
  that the lanes are near their plateau and the remaining gap is the class. The freed time is the useful part: it
  is the budget for more lanes (B=8 ~75 ms/grad) or more re-target rounds, which were too costly with per-step checks.
- Next: B=8 lanes (4 discovery + 4 targeted) with check_every=8; or a second re-target round at ~60 % of the budget.

## 2026-10-19 iter 204: wider target-class discovery (top-k runner-ups, more re-target rounds)
Hypothesis: the class is the lever, and the re-target step sees too few classes. Lane diagnostic on 16 dev images
with iter-203's `check_every=8` re-applied (scratch script, est scores, mean 0.97447): a re-targeted lane wins 4/16
images, by a lot when it does (img 3: 0.98050 targeted on 224 vs 0.97452 best untargeted; img 8: 0.97676 vs 0.97270;
img 10: 0.97603 vs 0.97467), while the untargeted lanes often collapse onto one class (img 6: all four on 148;
img 7: three on 311). `_retarget` only offers each lane's best class and current argmax, i.e. 1-3 distinct classes.
Plan: rank target classes also from every lane's top-k fp32 logits at the switch (`switch_topk`), and allow
extra re-target rounds (`extra_switches`, freeing the weakest non-leader, non-young lanes), plus a B=8 variant with
four more LSE-tau discovery lanes. Re-apply iter 203's `check_every=8` (discarded only as below MIN_GAIN; it frees
the time these rounds need). Compare on `--limit 16`, full dev bench with the best.
Result (code kept in this commit: `check_every=8`, `switch_topk=3`, `extra_switches="0.6"`, `n_switch_extra=1`):
- 16 images, est score (scratch lane script, same images and budget 22 s):
  | config | mean est | steps |
  |---|---|---|
  | k8 (iter 203 re-applied) | 0.97447 | ~437 |
  | + switch_topk 3 | 0.97452 | ~437 |
  | + switch_topk 3 + extra round at 0.6 | **0.97474** | ~437 |
  | B=8 (4 more LSE-tau lanes), n_switch 4, topk 3 | 0.97373 | ~256 |
  B=8 loses: the steps drop by 40 % and the extra discovery lanes mostly land in the same classes.
  The per-image max over all four 16-img runs is only 0.97507: among the classes these lanes surface, the
  best one is worth only ~+0.0006 over a single run. Run-to-run class luck is large on single images
  (img 3: 0.9805 / 0.9745 / 0.9766 / 0.9773 across configs) but small on the mean.
- Full dev (`python -m bench.run_bench --out results/dev-iter204-agent.json`): **mean 0.97561** vs best 0.97484
  (+0.00077) and iter 203 0.97529 (+0.0003); 64/64 success, 0 variant disagreements, time p95 21.84 s / max 21.86 s.
  Against iter 203 per image: 25 up / 21 down / 18 equal; the gain comes from 4 images up 0.0027-0.0045
  (a better re-target class), offset by many small losses (max -0.0019) from lanes freed for the extra round.
- Conclusion: wider discovery among *the classes our lanes surface* gives about +0.0003 to +0.0008, below MIN_GAIN.
  The 0.003-0.004 gap to the leaders is not reachable by re-ranking the same few classes; it needs either
  classes the lanes never surface (prior over adversarial classes, from a non-dev source) or lower RMSE inside a class.
- Next: (a) class prior learned from the public train split (ideas.md); (b) per-class RMSE: compare our support with
  the leaders' on images where we already hit their class (iter 2 data) to see if it is the search, not the class.

## 2026-10-20 iter 205: reweighted L1 (L0 surrogate) in the ISTA lanes
Hypothesis: the gap to the leaders is RMSE at the same margin (their median 3.1 % nonzero vs our ~4.8 %, iter 2 build;
warm-starting ISTA from a competitor's perturbation keeps or improves it, so sparser optima exist that ISTA from zero
misses). The score counts values (L0 on the rounded d), but the soft threshold shrinks every element by the same
lr*lam: saturated elements (|d| = 1, already "paid for") must out-pull lam to stay, while weak elements below the 0.5
rounding point (free) linger. Iteratively reweighted L1 (Candes, Wakin, Boyd 2008) with per-element threshold
lr*lam*(0.5+eps)/(|d|+eps) approximates L0: weak entries die fast, strong ones are barely shrunk, and the adaptive lam
re-scales the overall level. Expect fewer nonzeros at margin 10 on the same class (iter 204 per-image nonzero median ~44k).
Base: iter 204's code re-applied (check_every=8, switch_topk=3, extra round at 0.6; discarded only as below MIN_GAIN,
dev 0.97561). Plan: `rw_eps` in AttackConfig (0 = off); `--limit 16` with rw_eps 0 / 0.1 / 0.3, then full dev with the best.
Result (code kept in this commit: iter 204 base + `rw_power=2.0`, `rw_eps=0.3`; the per-element ISTA threshold is
lr*lam*((|d|+0.3)/0.8)^p, weights from the pre-step iterate):
- 16 images, `python -m bench.run_bench --limit 16 --attack-config '<cfg>' --out results/dev16-iter205-<tag>.json`,
  compared to the same images in `results/dev-iter204-agent.json` (0.97476 on those 16, nonzero median 52.4k):
  | config | mean | delta | images up | nonzero median |
  |---|---|---|---|---|
  | rw_power -1, eps 0.1 (reweighted L1, the hypothesis) | 0.97245 | -0.00231 | 1/16 | 58.4k |
  | rw_power -1, eps 0.3 | 0.97341 | -0.00135 | 3/16 | 56.5k |
  | rw_power +1, eps 0.3 | 0.97536 | +0.00060 | 14/16 | 50.2k |
  | rw_power +1, eps 0.05 | 0.97530 | +0.00054 | 13/16 | 50.8k |
  | **rw_power +2, eps 0.3** | **0.97563** | **+0.00087** | 15/16 | 49.2k |
  | rw_power +4, eps 0.3 | 0.97543 | +0.00067 | 12/16 | 49.6k |
- The hypothesis was wrong in sign. Reweighted L1 keeps saturated entries almost unshrunk, so the support that is
  already paid for never gets pruned and nonzeros grow. The **inverse** helps: shrink saturated entries (|d| near 1)
  ~2.6x harder and entries below the 0.5 rounding point ~7x softer. A saturated entry then survives only while its
  gradient keeps paying for it, and sub-threshold entries (free in L0) can explore and grow where the gradient is
  consistent. This puts the penalty where the score's L0 actually is.
- Full dev (`python -m bench.run_bench --out results/dev-iter205-agent.json`): **mean 0.97628** vs best 0.97484
  (+0.00144) and iter 204 0.97561 (+0.00067); 64/64 success, 0 variant disagreements, time p95 21.89 s / max 21.99 s,
  steps mean 430. Up on 58/64 images vs iter 204 (61/64 vs seed); nonzero median 39.3k vs 44.2k (iter 204).
  Tests 29/29 pass and `bash harness/gate.sh` passes (an earlier contract-test failure happened only while the bench
  held the CPU: TinyNet then got 1-2 steps in 6 s with any rw_power).
- Conclusion: a real sparsity gain inside the same search (-11 % nonzeros) on top of iter 203/204's changes, but
  +0.0014 is still under MIN_GAIN (0.002 -> needs 0.97684). Nonzero median is still ~39k vs the leaders' ~28k.
- Next: the shrink profile is now a lever. Try a profile aimed straight at the rounding point (e.g. threshold that
  is small below 0.5 and large above it, a smoothed step) and retune lam_up/lam_down/lr with it, since the adaptive
  lam was tuned for plain L1.

## 2026-10-20 iter 206: stack iter 203 + 204 (+ 205), then B=8 lanes
Operator task: stack the two near-misses (iter 203 `check_every=8`, iter 204 `switch_topk=3` + extra re-target
round at 0.6), then try B=8 lanes (4 discovery + 4 targeted) on --limit 16.
Note: iter 204's own full dev bench already *was* 203 + 204 stacked (0.97561, `results/dev-iter204-agent.json`), and
iter 205's patch (`logs/rejected-iter-205.patch`) is 203 + 204 + the shrink profile `rw_power=2, rw_eps=0.3`
(dev 0.97628, 58/64 up vs iter 204). With MIN_GAIN now 0.0005 the stack that adds most is the full 203+204+205
code, so this iteration re-applies `miner26/attack.py` from that patch unchanged and re-measures it on full dev
(a second sample of its run-to-run noise), instead of re-running the 203+204-only bench already on record.
Hypothesis for step 3: the freed steps (~430/image) can pay for 8 lanes; iter 204's B=8 (4 more LSE lanes) lost
0.001 on 16 images because steps fell 40 % while the extra lanes reached the same classes. With the shrink
profile, lanes converge faster in-class, so more targeted lanes (n_switch 4) might now pay off.
Plan: full dev on the stack (background); then `--limit 16` B=8 (`n_switch` 4) and B=6 against the stack's
same 16 images in `results/dev-iter206-agent.json`.
Result (code in this commit: `miner26/attack.py` from `logs/rejected-iter-205.patch`, i.e. check_every=8,
switch_topk=3, extra_switches="0.6", rw_power=2, rw_eps=0.3; no other change):
- Full dev (`python -m bench.run_bench --out results/dev-iter206-agent.json`): **mean 0.97622** vs best 0.97484
  (+0.00138, above MIN_GAIN 0.0005), 64/64 success, 0 variant disagreements, time p95 21.86 s / max 22.00 s,
  ~435 steps/image. Iter 205's run of the same code gave 0.97624: run-to-run noise on the full mean is ~2e-5.
- Lane count on 16 images (`python -m bench.run_bench --limit 16 --attack-config '<cfg>' --out results/dev16-iter206-<tag>.json`),
  compared to the same 16 images in the full run (0.97573):
  | config | mean | images up | steps | nonzero median |
  |---|---|---|---|---|
  | B=4 (stack, from the full run) | 0.97573 | - | 436 | 45.1k |
  | B=8, lanes x2, n_switch 4, n_switch_extra 2 (4 discovery + 4 targeted) | 0.97502 | 2/16 | 254 | 45.6k |
  | B=6 (+ LSE tau 3 and 0.1), n_switch 3 | 0.97575 | 5/16 | 313 | 45.9k |
- Conclusion: the stack holds (0.9762, twice). More lanes still do not pay. B=8 costs 42 % of the steps and loses
  0.0007, the same result as iter 204's B=8 even with the shrink profile. B=6 ties. The freed steps are worth more as
  in-class search depth than as extra class guesses from the same basins.
- Next: the class lever needs classes from outside our lanes (class prior from the public train split); within a
  class, tune the shrink profile (smoothed step at 0.5) and lr/lam_up/lam_down under rw_power=2.

## 2026-10-20 iter 207: shrink profile as a smoothed step at the rounding point; lr / lam retune
Base: iter 206's code re-applied unchanged (`miner26/attack.py` from `logs/rejected-iter-206.patch`: check_every=8,
switch_topk=3, extra round at 0.6, rw_power=2, rw_eps=0.3; dev 0.97622 / 0.97624 in two runs, 64/64).
Hypothesis: the score's L0 counts q = round(d), i.e. |d| >= 0.5; the gradient is taken at q. Iter 205 found that
shrinking saturated entries harder (power profile, weight 0.14 at |d|=0, 1 at 0.5, 2.64 at 1) beats plain and
reweighted L1. Liu & Barber (2018, "Between hard and soft thresholding") show that thresholding operators between
soft and hard are optimal for iterative thresholding, and the L0 surrogates in sigma-zero (Cina et al. 2024) and
proximal IHT (Zhang et al. 2018) put the penalty where the count changes. Here the count changes at 0.5, so a
threshold profile that is a smoothed *step* at 0.5 (rw_lo below, rw_hi above, width rw_step) should prune entries
just above 0.5 that the gradient does not hold (the power profile shrinks them at only ~1x) while still letting
sub-threshold entries explore. Iter 206's 16 dev images: mean 0.97573, nonzero median 45.1k; the leaders' ~28k.
Second lever: lr / lam_up / lam_down were tuned for plain L1 and never retuned under rw_power=2.
Plan: `rw_step`/`rw_lo`/`rw_hi` in AttackConfig (0 = off, default unchanged); `--limit 16` sweep of the step
profile and a few lr/lam settings against iter 206's same 16 images; full dev bench of the best if it gains.
Result (code in this commit: iter 206's stack re-applied unchanged as the default, plus the `rw_step` option, off):
- 16 images (`python -m bench.run_bench --limit 16 --attack-config '<cfg>' --out results/dev16-iter207-<tag>.json`),
  vs the same 16 images in `results/dev-iter206-agent.json` (0.97573, nonzero median 45.1k):
  | tag | config | mean | delta | up | nonzero median |
  |---|---|---|---|---|---|
  | s1 | rw_step 0.1 (lo 0.14, hi 2.64) | 0.97569 | -0.00004 | 7/16 | 47.1k |
  | s2 | rw_step 0.1, rw_hi 4 | 0.97583 | +0.00010 | 8/16 | 46.7k |
  | s3 | rw_step 0.05, lo 0.1, hi 3 | 0.97511 | -0.00061 | 4/16 | 50.9k |
  | lr15 | lr 0.015 (power profile) | 0.97581 | +0.00008 | 7/16 | 48.8k |
  | lam | lam_up 1.08, lam_down 0.95 | 0.97559 | -0.00014 | 9/16 | 48.7k |
  All 16/16 success, 0 variant disagreements, t max <= 21.84 s.
- Full dev of s2 (`python -m bench.run_bench --attack-config '{"rw_step": 0.1, "rw_hi": 4.0}' --out results/dev-iter207-agent-s2.json`):
  **0.97632** vs the stack's 0.97622 (+0.00009), 64/64, 0 disagreements, p95 21.85 s / max 21.88 s; per image 27 up /
  37 down, nonzero median 38.6k vs 39.9k. Flat.
- Conclusion: the step profile at the rounding point does not beat the power profile (it is
  no better when soft, and worse when sharp: s3 sends entries across 0.5 back and forth, so the support churns and
  nonzeros grow). lr and lam_up/lam_down are flat around the current values. The in-class ISTA search is at a local
  optimum in these knobs; per-image noise (+-0.002) dominates. Left the default at the iter 206 stack so the harness
  re-measures it (0.9762 in two runs; above best + MIN_GAIN 0.0005 if the running loop has picked up the new value; iter 206
  was discarded as no_gain at 0.97618, which suggests the loop still applied 0.002).
- Next: more progress per step is the only in-class lever that has paid (iter 203). Try momentum / extrapolation on d
  (accelerated proximal IHT, Zhang 2018 / Wu 2019), or l0 change control (Zhu 2021). For the class lever, try a class prior
  from the public train split.

## 2026-10-20 iter 208: momentum / extrapolation on the ISTA iterate
Base: iter 207's accepted code (dev 0.97633 harness, 0.9762-0.9763 in three agent runs).
Hypothesis: more progress per step is the only in-class lever that has paid (iter 203: 302 -> 437 steps, +0.0005).
Momentum does the same without more steps. The gradient is taken at q = round(d) (straight-through), so between two
steps only the few entries that cross 0.5 change q and the gradient direction is noisy step to step; a gradient EMA
(heavy ball, as in MI-FGSM, Dong 2018; NMI-FGSM, Xu 2020; EAD's FISTA, Chen 2017) averages that noise and moves entries
with a consistent gradient across 0.5 sooner, while entries whose gradient flips sign get shrunk away. FISTA-style
extrapolation on d (Zhang 2018, Wu 2019; APG-restart, Zhou 2020; adaptive restart, O'Donoghue & Candes 2015) is the
other form; restart is needed when lam changes direction. Iter 207 per-image: nonzero median 38.6k vs leaders ~28k,
~435 steps/image, so a faster in-class descent should lower nonzeros at margin 10.
Plan: `momentum` (gradient EMA beta, 0 = off) and `extrap` (FISTA beta on d, 0 = off) in AttackConfig, reset on
re-target. `--limit 16` with momentum 0.5 / 0.8 and extrap 0.3 vs iter 207's same 16 images, full dev of the best.
Result (code in this commit: `momentum` = 0.5 default, `extrap` option off; momentum state reset for re-targeted lanes):
- 16 images (`python -m bench.run_bench --limit 16 --attack-config '<cfg>' --out results/dev16-iter208-<tag>.json`),
  vs the same 16 images in the accepted run `results/harness/dev-iter207-4fc2437.json` (0.97555, nonzero median 50.6k):
  | tag | config | mean | delta | up | nonzero median |
  |---|---|---|---|---|---|
  | m5 | momentum 0.5 | 0.97604 | +0.00049 | 12/16 | 45.0k |
  | m8 | momentum 0.8 | 0.97575 | +0.00020 | 9/16 | 48.4k |
  | x3 | extrap 0.3 | 0.97582 | +0.00028 | 13/16 | 49.3k |
  All 16/16 success, 0 variant disagreements, ~435 steps, t max <= 21.84 s.
- Full dev with momentum 0.5 (`python -m bench.run_bench --out results/dev-iter208-agent.json`): **0.97649** vs best
  0.97633 (+0.00017, below MIN_GAIN 0.0005), 64/64, 0 disagreements, p95 21.85 s / max 21.88 s; per image **47 up / 17
  down** vs the accepted run, nonzero median 37.3k vs 39.2k. `bash harness/gate.sh` passes.
- Conclusion: momentum helps consistently (47/64 images, -5 % nonzeros) but by little; the 16-image gain (+0.0005)
  halved on the full set. Both forms of acceleration help, the gradient EMA more than extrapolation on d; a long
  memory (0.8) is worse than 0.5, consistent with STE gradients that go stale once entries cross 0.5.
- Next: momentum 0.3 and momentum + larger lr (smoother direction may tolerate lr 0.015-0.02, which was flat without
  momentum in iter 207); Nesterov look-ahead; the class lever (prior from the public train split) remains the big gap.

## 2026-10-20 iter 209: momentum (iter 208) + decaying per-lane step size
Base: iter 207's accepted code (dev 0.97633) plus iter 208's momentum 0.5 re-applied (`logs/rejected-iter-208.patch`,
full dev 0.97649, +0.00017, 47/64 images up: consistent but below MIN_GAIN 0.0005).
Hypothesis: the lane step lr is constant (0.01 levels per unit normalised gradient) for the whole budget. Late in a
lane's life the support is nearly settled and a constant step keeps entries hopping across the 0.5 rounding point
(nonzero median 37-39k vs leaders ~28k). Tao et al. 2026 ("The Power of Decaying Steps", MDCS) show that
non-decaying step sizes cause non-convergence/instability in iterative attacks and that monotonically decreasing steps
(with momentum, MDCS-MI) converge and are more stable. A linear per-lane decay from lr to lr*lr_final over the lane's
remaining budget (re-targeted lanes restart their schedule) should let late steps refine the support instead of churning
it, and a higher start lr (flat alone in iter 207: lr 0.015 +0.00008) may now pay because the late steps shrink.
The ISTA threshold is lr*lam, so it decays too; lam's multiplicative schedule compensates.
Plan: `lr_final` in AttackConfig (1.0 = off). `--limit 16` with lr_final 0.3 / 0.5 and lr 0.015 + lr_final 0.3, all with
momentum 0.5, vs momentum alone (iter 208's 16-img run `results/dev16-iter208-m5.json`, 0.97604); full dev of the best.
Result (code in this commit: momentum 0.5 (iter 208) + `retarget_warm` = True as defaults; `lr_final` option, off):
- 16 images (`python -m bench.run_bench --limit 16 --attack-config '<cfg>' --out results/dev16-iter209-<tag>.json`),
  vs momentum alone on the same 16 images (`results/dev16-iter208-m5.json`, 0.97604, nonzero median 45.0k):
  | tag | config | mean | delta | up | nonzero median |
  |---|---|---|---|---|---|
  | f3 | lr_final 0.3 | 0.97562 | -0.00042 | 4/16 | 48.0k |
  | f5 | lr_final 0.5 | 0.97582 | -0.00022 | 4/16 | 47.5k |
  | l15f3 | lr 0.015, lr_final 0.3 | 0.97594 | -0.00010 | 4/16 | 48.8k |
  | warm | retarget_warm (re-targeted lanes start from the leader's d) | 0.97625 | +0.00021 | 8/16 | 45.1k |
  All 16/16 success, 0 variant disagreements, ~435 steps, t max <= 21.84 s.
- The decay hypothesis was wrong: the more lr decays, the worse (more nonzeros). Late steps still need full size; lam
  keeps moving the boundary until the end, and a smaller step just stalls the prune. A higher start lr only partly
  compensates. Instead, the side test (warm start, cf. the guided-init idea of Zhou et al. 2024) gained: a re-targeted
  lane at 35 % / 60 % of the budget no longer re-discovers a support from d = 0; it starts from the leader's sparse
  iterate, which already sits near several class boundaries, and only has to move it into the target class.
- Full dev with momentum + warm start (`python -m bench.run_bench --out results/dev-iter209-agent.json`): **0.97683**
  vs best 0.97633 (+0.00050) and momentum alone 0.97649 (+0.00033); 64/64, 0 disagreements, p95 21.83 s / max 21.85 s,
  438 steps; per image 45/64 up vs the accepted run (35/64 vs iter 208), nonzero median 36.0k vs 39.2k (accepted) / 37.3k.
  `bash harness/gate.sh` passes.
- Next: warm-start variants (copy the leader's momentum; start from only its |d| > 0.5 entries; start from the lane that
  surfaced the class); late-budget support shrinking (Lin et al. 2023); class prior from the public train split.
