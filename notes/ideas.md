# Experiment ideas (research mode picks from here; mark outcome when tried)

Format: `- [ ] idea (why)`; when tried: `- [x] idea: iter N, result, kept/discarded`.

## Target class (highest value, see research_log 2026-10-19)
- [ ] Measure run-to-run class variance (same image, several seeds/timings): iter 202 saw image 3 at 0.9776 vs 0.9709 between runs; if large, more independent discovery lanes / restarts pay directly.
- [x] B=8 lanes: iter 204, 16-img est 0.97373 vs 0.97447 (B=4); 40 % fewer steps, extra lanes find the same classes. Discarded.
- [x] Re-target several times: iter 204, one extra round at 0.6 (1 lane) part of the +0.0003 above; small.
- [ ] Discovery lanes without L1 at higher lr (they only need to reveal a class, not a sparse solution).
- [x] Candidate classes from the top-k of each discovery lane's logits, not only its argmax: iter 204, with an extra re-target round at 0.6, dev 0.97561 (+0.0003 over iter 203); per-image oracle over 4 configs on 16 imgs only +0.0006, so the surfaced classes are exhausted.
- [ ] Class prior: from the public dataset's train split (never dev/test), which adversarial classes win for
      which clean classes / globally; seed targeted lanes with the prior's top classes.
- [x] Warm-start re-targeted lanes from the leading lane's iterate d (`retarget_warm`): iter 209, 16-img +0.00021 vs momentum alone (8/16 up, nonzero median 45.1k vs 45.0k); made default with momentum 0.5, full dev in research_log.
- [ ] Warm-start variants: copy the leader's momentum too; start from a pruned leader (only |d|>0.5 entries); warm-start from the lane that surfaced the target class instead of the leader.

- [ ] Diagnose RMSE within a class: on images where our final class equals the leaders', compare nonzero counts (is the gap search quality, not class?).

## Objective / trade-off
- [ ] Sweep `kappa` (search hinge target) and `safety_margin`; the score rewards margin up to 10 logits at 0.003/logit.
- [ ] Choose the margin target per image from the marginal score: stop pruning when d(score)/d(nnz) turns negative.
- [ ] Calibrate `safety_margin` from the bench's `variant_disagreement` rate (how small can it get without failures?).

## Search
- [ ] L1 schedule: `l1_init`, `l1_up`, `l1_down`; or a fixed sparsity budget (top-k projection) instead of L1.
- [ ] Initialise from one FGSM sign step restricted to the top-k |grad| values instead of w = 0.
- [ ] Untargeted vs targeted at the runner-up class (often a far smaller step).
- [ ] Per-pixel moves (all 3 channels together) vs single values: RMSE counts values, novelty counts pixels.
- [ ] Multi-restart in one batch (different seeds or targets), keep the best.
- [ ] `search_fraction`: time split between search and prune.

- [x] Reweighted L1 (Candes 2008, threshold ~ 1/(|d|+eps)): iter 205, 16-img -0.0014..-0.0023, more nonzeros. Discarded.
- [x] Inverse reweighting (threshold * ((|d|+0.3)/0.8)^p, saturated entries shrink harder): iter 205, p=2 best,
      dev 0.97628 (+0.00144 vs best, +0.00067 vs iter 204), nonzero median 39.3k vs 44.2k.
- [x] Shrink profile as a smoothed step at the 0.5 rounding point (`rw_step`): iter 207, 16-img width 0.1 -0.00004 (lo 0.14/hi 2.64) / +0.0001 (hi 4); sharp step (0.05, lo 0.1, hi 3) -0.0006. No gain over the power profile; option kept, off.
- [x] Re-tune lr / lam_up / lam_down under rw_power=2: iter 207, lr 0.015 +0.00008, lam_up 1.08/lam_down 0.95 -0.00014 (16 img). Flat: the in-class search is at its optimum for these knobs.
- [x] Momentum / extrapolation on the ISTA iterate d (accelerated proximal IHT, Zhang 2018, Wu 2019; EAD's FISTA): iter 208, 16-img vs iter 207: gradient EMA beta 0.5 +0.0005 (12/16 up, nonzero median 45.0k vs 50.6k), beta 0.8 +0.0002, FISTA-style extrapolation 0.3 +0.0003. Momentum 0.5 made default (full dev: see research_log).
- [ ] Momentum follow-ups: beta 0.3; momentum + lr retune (the EMA smooths, so a larger lr may now be stable); Nesterov look-ahead (gradient at round(d + beta*m)); oscillation-triggered restart (O'Donoghue & Candes 2015) when a lane's lam flips direction.
- [x] Decaying per-lane step size (`lr_final`, MDCS, Tao 2026), with momentum 0.5: iter 209, 16-img lr_final 0.3 -0.00042, 0.5 -0.00022, lr 0.015 + lr_final 0.3 -0.00010 (4/16 up each). Worse the more it decays: late steps still need full size (lam keeps moving the boundary). Option kept, off.
- [ ] The opposite of decay: lr warm-up / larger late lr, or lr tied to lam (lam rising = support still too large).
- [ ] Support shrinking late in the budget (Non-Lipschitz Attack, Lin 2023): freeze entries at 0 once they have left the support, so the last phase only prunes.
- [ ] l0 change control (Zhu 2021 homotopy attack): cap how many entries cross 0.5 per step, to stop the support churning.
- [ ] B>4 only pays if lanes get cheaper: e.g. B=6 with check_every=16 to win back steps.

## Prune
- [x] Greedy first-order prune of the best candidate in the last 20 % of the budget (4 drop sizes per batched fp32 check): iter 202, 16-img 0.97300 vs 0.97365, discarded. Candidates are already tight at margin 10; only the overshoot above 10 is recoverable (~1e-5..4e-4/image).
- [ ] Clip margin overshoot only: if the final best has margin > 10.1, a few cheap prune rounds (no search time lost) recover up to ~4e-4 on such images.
- [ ] Batched prune: evaluate several drop fractions in one forward batch.
- [ ] Re-gradient after each accepted prune (already done) vs swap moves (drop k, add k elsewhere).
- [ ] Final polish: try flipping signs of the weakest entries instead of dropping them.

## Speed (more steps per budget)
- [ ] channels_last + fp16/bf16 autocast for the search; keep fp32 checks.
- [ ] `torch.compile` / CUDA graphs on the fixed-size forward+backward (shapes vary per image; cache per shape).
- [x] Defer fp32 checks: per-lane best projected candidate, verified in one batch every K steps: iter 203, K=8 +44 % steps (302 -> 437), dev 0.97529 vs 0.97484 (+0.00045, 60/64 images up).
- [x] With the freed time: B=8 lanes or a second re-target round: iter 204 (round) / iter 206 (lanes). Stack 203+204+205 = dev 0.97622;
      B=8 (4 discovery + 4 targeted) 16-img -0.0007 (254 vs 436 steps), B=6 +0.00002. Lanes discarded; steps beat lanes.
- [ ] Early exit once the estimate is at the ceiling (alias images); saves GPU, no score effect.

## Image geometry
- [x] Resize-influence ideas: moot. Task images are always 480 on the short side, so the preprocessing is a
  pure crop (notes/scoring.md, 2026-10-19).
- [ ] Attack the 480x480 crop directly (skip the no-op resize in PREPROCESS) for a slightly faster step;
  map back with the crop offset.
