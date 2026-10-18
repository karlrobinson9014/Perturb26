# Experiment ideas (research mode picks from here; mark outcome when tried)

Format: `- [ ] idea (why)`; when tried: `- [x] idea: iter N, result, kept/discarded`.

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
- [ ] Weight values by their bicubic resize influence (some inputs feed more output pixels).
- [ ] Check images whose shorter side is < 480 (upscaling spreads each input value over several outputs).
