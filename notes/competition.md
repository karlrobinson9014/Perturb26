# What the network achieves

## Live consensus (2026-10-19, `python -m tools.leaderboard`, read-only)
5 validators; uid 0 holds ~80 % of validator stake (1.55M of ~1.94M), so its ranking decides.
| consensus rank | uid | avg score (validator uid 0, 300 rounds) |
|---|---|---|
| 1 | 36 | 0.9801 |
| 2 | 161 | 0.9799 |
| 3 | 162 | 0.9799 |
| 4-9 | 45, 136, 141, ... | 0.9792-0.9794 |
Rank 1 needs a 300-round average above ~0.980 with no misses: about 0.981-0.982 per image. The dataset proxy
below (0.9784) is older than the live averages and underestimates today's leaders.

# Competitor proxy from the dataset

`python -m tools.competitor_scores` on the 64 dev images: the official scorer applied to the
336 verified adversarial versions other miners submitted for those images (dataset rows, about 5 per
image). Run 2026-10-19 on CPU, `default` variant only. Raw: `results/competitors-dev.json`.

| statistic | value |
|---|---|
| mean of best-per-image | **0.9784** |
| median of best-per-image | 0.9792 |
| best single response | 0.9858 |
| worst best-per-image | 0.9398 |
| mean of all responses | 0.9439 |

What the best responses look like:
- **All** best-per-image responses are L-inf = 1/255 (`linf_levels = 1`).
- Nonzero fraction of the best ones: 0.8% - 13%, typically 1-5%. The ~0.985 responses change about 1% of
  values *and* saturate the margin (score ~0.985 = 0.653 + 0.3*0.974 + 0.03 + 0.01).
- A second cluster at ~0.72-0.73 is 2/255 (linf_score 0.673); ~0.42 is a few levels more.
  Some miners are still on PGD-style attacks.
- The top 2-3 responses per image are usually within 0.001-0.005 of each other: the race is close.

Implications:
- To take rank 1 we need roughly **>= 0.98 on average with ~100% success**, i.e. beat the per-image
  envelope of everyone else, consistently. 0.985+ looks like the frontier.
- 0.993 (the ceiling) would need RMSE ~ 0 *and* a 10-logit margin on every image; nothing in the data
  comes near. Treat it as the asymptote, not a milestone.
- Lever ranking from the data: (1) never fail, (2) margin >= 10 cheaply, (3) sparsity to ~1% or below.
