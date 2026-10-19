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
