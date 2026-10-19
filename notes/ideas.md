# Experiment ideas (research mode picks from here; mark outcome when tried)

Format: `- [ ] idea (why)`; when tried: `- [x] idea: iter N, result, kept/discarded`.

## Target class (highest value, see research_log 2026-10-19)
- [ ] B=8 lanes (9.4 ms/img-step): 4 discovery objectives + 4 targeted lanes fed from them.
- [ ] Re-target several times (e.g. at 25 %, 45 %, 65 % of the budget), each time replacing the worst lanes.
- [ ] Discovery lanes without L1 at higher lr (they only need to reveal a class, not a sparse solution).
- [ ] Candidate classes from the top-k of each discovery lane's logits, not only its argmax.
- [ ] Class prior: from the public dataset's train split (never dev/test), which adversarial classes win for
      which clean classes / globally; seed targeted lanes with the prior's top classes.
- [ ] Warm-start: the best untargeted solution's support as the init of a targeted lane on its class.

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

## Prune
- [ ] Batched prune: evaluate several drop fractions in one forward batch.
- [ ] Re-gradient after each accepted prune (already done) vs swap moves (drop k, add k elsewhere).
- [ ] Final polish: try flipping signs of the weakest entries instead of dropping them.

## Speed (more steps per budget)
- [ ] channels_last + fp16/bf16 autocast for the search; keep fp32 checks.
- [ ] `torch.compile` / CUDA graphs on the fixed-size forward+backward (shapes vary per image; cache per shape).
- [ ] Skip the fp32 check when the amp margin is far above `safety_margin` (check only the final few).
- [ ] Early exit once the estimate is at the ceiling (alias images); saves GPU, no score effect.

## Image geometry
- [x] Resize-influence ideas: moot. Task images are always 480 on the short side, so the preprocessing is a
  pure crop (notes/scoring.md, 2026-10-19).
- [ ] Attack the 480x480 crop directly (skip the no-op resize in PREPROCESS) for a slightly faster step;
  map back with the crop offset.
