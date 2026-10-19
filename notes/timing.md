# Live timing (from upstream code; verify with a live probe, backlog item)

- Tasks change on a 120 s wall-clock cadence (`TASK_CADENCE_SECONDS`), boundaries at multiples of 120 s.
- Upstream miner: snapshots the task 10 s before the boundary, sleeps to the boundary, then polls
  `GET /task` every 2 s until the task_id changes.
- Validator: sleeps to the boundary, waits **40 s** (`VALIDATOR_EVALUATION_DELAY_SECONDS`), fetches the task,
  downloads and labels the clean image, then polls `GET /submits` every 5 s and **takes the first non-empty
  list**. Anything submitted after that first successful poll is not scored for that round.
- So the response must be submitted before about boundary + 40 s (plus the validator's own download and
  inference). The live miner targets `PERTURB26_SUBMIT_BY_SECONDS=34` and reserves an EMA of measured
  upload + submit time (initially 5 s) for the attack deadline.
- Budget accounting: about 1-2 s to see the new task and download it, ~22 s of attack, ~5 s to upload and
  submit, leaving ~5 s of slack. `BENCH_BUDGET=22` mirrors this. If the probe shows the task appears
  earlier or later, propose a new budget in `notes/harness_proposals.md` (BENCH_BUDGET is harness-owned).

## Measured 2026-10-19 (`python -m tools.task_probe --boundaries 10`, from the Windows dev machine)
Raw: `results/timing-probe-2026-10-19.json`. 11 task changes, 2 poll timeouts.

| quantity | value |
|---|---|
| new task visible after the boundary | median **3.0 s** (1.8 - 3.7 s; once 6.5 s after timeouts) |
| GET /task latency | median 2.0 s, max 8.1 s, 2 timeouts at 10 s |
| task image download | median 0.33 s, max 1.55 s |
| task_id prefix | the boundary epoch (`1792372440-hf-...` = boundary 1792372440) |
| local clock vs server Date | local ~0.5 s fast (1 s resolution) |

Consequences:
- The attack can start at about boundary + 3.5 s; with submit-by 34 s and ~5 s upload reserve it gets ~25 s,
  so `BENCH_BUDGET=22` is safe with a few seconds to spare.
- API latency is spiky. Upstream's 10 s fetch timeout and 2 s retry sleep can cost a whole round; a shorter
  per-request timeout with quick retries is a cheap reliability win (backlog).
- Keep the miner box on NTP: boundaries are wall clock on both sides.
- Re-run the probe from the GPU box; latency there may differ.

Open question: Hippius vs R2 upload latency from the miner box.

## 2026-10-19 from the GPU box (iter 3, `miner26.fetch.fetch_new_task`, 3 s timeout / 0.5 s retries)
2 boundaries: new task seen at + 7.3 s and + 2.2 s (3 attempts each), image download ~0.6 s. The 7.3 s case
leaves ~21 s of attack under submit-by 34 s and a 5 s reserve: still >= BENCH_BUDGET-ish, but tight. Worth a
longer probe (`tools/task_probe.py`) from this box.
