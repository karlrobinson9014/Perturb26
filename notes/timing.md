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

Open questions for the probe:
1. When does `GET /task` actually switch relative to the boundary?
2. Task image download latency (R2 presigned URL).
3. Hippius vs R2 upload latency from the miner box.
