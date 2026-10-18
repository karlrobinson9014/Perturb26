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
