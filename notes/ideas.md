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
- [ ] Warm-start: the best untargeted solution's support as the init of a targeted lane on its class.

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
- [ ] Shrink profile as a smoothed step at the 0.5 rounding point (sigmoid((|d|-0.5)/w)), with retuned lam_up/lam_down/lr.
- [ ] Re-tune lr / lam_up / lam_down under rw_power=2 (the adaptive lam was tuned for plain L1). Next after iter 206.
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
