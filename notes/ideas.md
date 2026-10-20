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
- [x] Warm-start + inherited L1 weight (`retarget_lam`, homotopy, Xiao & Zhang 2012): iter 210, 16-img 1.0 -0.00015, 0.5 -0.00002; a class switch is not a nearby problem, off.
- [ ] Warm-start variants: copy the leader's momentum too; start from a pruned leader (only |d|>0.5 entries); warm-start from the lane that surfaced the target class instead of the leader.

- [ ] Diagnose RMSE within a class: on images where our final class equals the leaders', compare nonzero counts (is the gap search quality, not class?).

## Objective / trade-off
- [x] Sweep `kappa` (search hinge target) and `safety_margin`; the score rewards margin up to 10 logits at 0.003/logit. Iter 211 (operator): safety 10 / target 10.5 / kappa 11.5-12 all worse (16-img -0.00013..-0.00034, full dev -0.00025); a hard margin-10 floor hurts the one image no lane can hold at 10 (0.9346 -> 0.9307).
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
- [x] Momentum beta 0.3: iter 210, 16-img -0.0001 vs 0.5 (6/16 up); 0.5 stays.
- [ ] Momentum follow-ups: momentum + lr retune (the EMA smooths, so a larger lr may now be stable); Nesterov look-ahead (gradient at round(d + beta*m)); oscillation-triggered restart (O'Donoghue & Candes 2015) when a lane's lam flips direction.
- [x] Decaying per-lane step size (`lr_final`, MDCS, Tao 2026), with momentum 0.5: iter 209, 16-img lr_final 0.3 -0.00042, 0.5 -0.00022, lr 0.015 + lr_final 0.3 -0.00010 (4/16 up each). Worse the more it decays: late steps still need full size (lam keeps moving the boundary). Option kept, off.
- [ ] The opposite of decay: lr warm-up / larger late lr, or lr tied to lam (lam rising = support still too large).
- [x] Support shrinking late in the budget (Lin 2023) + exact drops fed back into the frozen support (operator directive, fifth time): iter 216, freeze 0.75 -0.00040 (1/16 up), +drops -0.00063, freeze 0.9 tau 0.25 +drops +0.00007 (noise). Late ISTA swaps entries in and out; freezing removes those swaps. Closed (`freeze_at`, off).
- [ ] Flaky: `test_ordinary_image_runs_to_the_deadline` failed once on a cold GPU (iter 216); consider a warmup in that test (tests/ is agent-editable).
- [ ] l0 change control (Zhu 2021 homotopy attack): cap how many entries cross 0.5 per step, to stop the support churning.
- [ ] B>4 only pays if lanes get cheaper: e.g. B=6 with check_every=16 to win back steps.

## Prune
- [x] Greedy first-order prune of the best candidate in the last 20 % of the budget (4 drop sizes per batched fp32 check): iter 202, 16-img 0.97300 vs 0.97365, discarded. Candidates are already tight at margin 10; only the overshoot above 10 is recoverable (~1e-5..4e-4/image).
- [x] Clip margin overshoot only (operator directive, seventh time): `clip_time` reserve when the leader's margin > 10.05, then exact drop-ladder / swap rounds on the chosen candidate (`_clip`, off): iter 218, 16-img 0.6 s -0.00015, 1.0 s -0.000004; per-image gains on overshoot images 1e-5..5.4e-4, mean ~3e-5, below class noise. First-order k sizing (cumulative g*q vs slack) fails: the cumulative cost stays negative for thousands of entries. Reserving time on mid-run overshoot costs ISTA steps when the final pick has no slack.
- [x] Batched exact local search (operator directive): 16-32 drop / drop-2k-add-k / random-subset moves per fp32 batch, exact estimate, re-gradient per round (`polish_fraction`, off): iter 212, 16-img p0.85 -0.00036, p0.9 batch 16 -0.00017. It accepts ~90 % of rounds (+1e-4..1e-3/image) but loses to the ISTA steps it replaces; ISTA is still pruning at the deadline.
- [x] Free polish: exact drop moves of the leader in idle check-batch slots (`prune_slots`, ~50 slots/run), with/without zeroing them in d, plus check-only final rounds (`prune_final` 0.4 s): iter 213, 16-img -0.00008 / -0.00003 (noise). Accepts are superseded by later ISTA candidates; feedback hurts. Options kept, off. The prune/swap direction is closed (iters 202, 212, 213).
- [ ] Random drop masks: in the probe, random subsets of the cheapest pool won as often as first-order top-k. Try them in ISTA's own checks.
- [x] Exact local search with 64 candidates per fp16 screen forward, top 4 scored in fp32 (`polish_screen`, operator directive, third time): iter 214, 16-img -0.00003 (5/16 up), neutral. Probe bound: only the ~10-200 cheapest entries per image are free; dropping 1000 together costs ~1 logit, so a local move is worth <= ~5e-5/image. fp16 forward is only 1.9x cheaper than fp32 (4.3 vs 8.2 ms/image). The first fp16 batch-64 call warms up ~1 s (22.28 s overrun on image 1): pre-warm it before enabling. Closed: "1 % -> 0.3 %" needs a different basin, not local moves.
- [x] Rounding-threshold moves (operator directive, fourth time): exact-scored round(d) at t in 0.4..0.7 plus drop-k-smallest-|d| / add-k/2 swaps on every lane's best, last 0.45 s (`sweep_final`, off): iter 215, probe +1e-5..5e-5/image, 16-img -0.00002 (9/16 up). Neutral, like 202/212/213/214.
- [ ] Seed-ensemble variance: per-image noise from which class wins is +-5e-4; quantify with 3 repeated runs on 16 images before chasing gains < 3e-4.
- [x] Final polish: try flipping signs of the weakest entries instead of dropping them. Iter 217: sign flips of entries opposing the margin (alone and flip k + drop 2k) were never the best exact move on 8 images, 0 accepts (`tools/localsearch_ceiling.py`).
- [x] Time-unlimited ceiling of exact batched prune/swap/flip (operator directive, sixth time): iter 217, 8 dev images, 22 s attack + up to 60 s of exact local search (32 candidates/fp32 forward): mean +1.6e-4/image, stalls after 8-15 s; 7/8 images +2.5e-5..1.4e-4, one image with final margin 10.2 +8.9e-4. The move set is exhausted at the ISTA point, whatever the time budget (`results/localsearch-ceiling-iter217.json`). Only the margin-overshoot case below has anything to recover.
- [x] (iter 219: zeroing g*q<0 entries with |g|>=t*mean inside ISTA: 16-img t=4 -0.00112, 8 -0.00055, 16 -0.00019; knocks lanes off their trajectory, support grows 2-3x on 2 images. Closed.) Entries working against the margin (iter 218 evidence: on dev image ...063626Z-0002, dropping the 64-256 most negative g*q entries of the final point raises the margin 10.06 -> 10.57; on most images it does not): at the final point, 1-3 % of nonzeros have g*q < 0 (dropping the 10-200 cheapest *raises* the margin by up to +0.2 on a 22k-nonzero image). An ISTA-side fix (zero entries whose gradient opposes their sign at round time) might keep the support cleaner throughout, rather than polishing at the end. Expected small.

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

- [ ] Learned sparse universal init (Meta Adversarial Perturbations, Yuan 2021) for EfficientNetV2-L from public train images (never dev): seed discovery lanes so they surface far classes (leaders' best class clean rank median ~90).
- [ ] The hard image (clean class held to margin ~1-5 only, 0.93-0.935): it alone costs ~0.0007 of the mean. Find out why no lane reaches margin 10 there (target class? budget?) and give such images more lanes on the strongest class (iter 211).

- [tried iter 220, neutral] Snapshot-intersection drops: final q AND earlier verified bests of the same lane (`consensus_time`):
  1.2e-4..3.7e-4 logit/value, 3-10x cheaper than random drops but 2-4x above break-even; 16-img -0.00017 (basin noise).
  First-order "add k" swaps fail: top-|g| zero adds barely raise the margin and lower it in bulk (exact fp32, 1000 adds -0.48).
