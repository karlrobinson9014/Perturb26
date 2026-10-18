# How the validator scores a response (upstream 0dad954)

Source: `upstream/neurons/validator.py` `verify_and_score` (lines 388-537), constants in
`upstream/perturbnet/constants.py`. The bench calls that exact function (`bench/official.py`).

## Gates (any failure = 0.0)
- decode OK, same shape as clean, values in [0,1]
- `0.003 <= Linf <= 0.03` (both images quantized to the uint8 grid first, so Linf is k/255)
- predicted class index != validator's "true" index (see the alias note below)
- SSIM >= 0.98, PSNR >= 38 dB (trivially met at 1/255)
- image hash must match the submitted pixel hash; byte-identical responses from two miners both get 0

## Formula
```
linf_ratio = (Linf - 0.003) / (0.03 - 0.003)          linf_score = (1 - linf_ratio)^2
rmse_ratio = RMSE / 0.03                               rmse_score = (1 - rmse_ratio)^2
perturbation = 0.7*linf_score + 0.3*rmse_score
margin = best_non_true_logit - true_logit              margin_score = clamp(margin/10, 0, 1)
novelty = clamp(changed_pixels / 8, 0, 1)              (pixels = HxW positions, any channel)
final = perturbation + 0.03*margin_score + 0.01*novelty
```

## Consequences
| choice | effect |
|---|---|
| Linf = 1/255 | linf_score 0.93291; the best possible (1/255 is the smallest admissible step) |
| Linf = 2/255 | linf_score 0.67340: **-0.18** final. Never exceed 1/255 |
| all values changed at 1/255 | rmse_score 0.7557: -0.073 |
| 5% of values changed | -0.017 |
| 1% of values changed | -0.008 |
| margin | +0.003 per logit, saturating at 10 logits (+0.03) |
| >= 8 changed pixels | +0.01 (pad with pixels the model can't see) |

**Theoretical ceiling: 0.99303** (Linf 1/255, RMSE -> 0, margin >= 10, novelty full).
Realistically, margin and sparsity pull against each other: every value removed lowers RMSE and
usually lowers the margin. Each logit of margin is worth about as much as an RMSE improvement of
`0.003 / (0.3 * d rmse_score)`. The attack ranks candidates with `miner26/scoring.estimate_score`,
which equals the official score within ~3e-6 (float32 deltas, see the parity test).

## Free wins
- **Outside the crop**: the model sees resize(shorter side -> 480, bicubic, antialias) + center crop 480.
  Values outside the crop never affect the prediction but count toward RMSE. `Attacker.influence_mask`
  finds them. Use them only for the 8 novelty pixels.
- **Label-alias collision**: the validator stores the clean prediction as a label string and maps it back
  with `resolve_target_index`. torchvision has "Cardigan" (corgi, 264) and "cardigan" (sweater, 474), so
  a clean prediction of 264 becomes true index 474. The unchanged prediction 264 then counts as a flip with
  a large margin; 8 unseen pixels score ~0.993. `miner26/scoring.validator_true_index` models this. Rare.

## Numeric risk
The validator runs fp32 on its own GPU, with whatever TF32/cuDNN algorithms it picks. A flip with a
0.05-logit margin can fail there. The bench scores every response under `default`, `tf32_off` and
`tf32_on` and takes the minimum; the attack keeps `safety_margin` logits (default 1.0) on its own fp32 check.

## Ranking and emissions (why consistency matters)
- `avgScore` = mean over the last 300 rounds (about 10 h). A miss or failure is 0 for that round.
  One zero costs ~0.0032 of average, about the size of the gaps between top miners.
- Only the top 3 by stake-weighted consensus rank get scanning emissions: 75 / 20 / 5 % of 80 %.
