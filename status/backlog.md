# Backlog

Top = highest priority. `[build]` items run in build mode (auto mode does them first);
`[research]` items are experiment directions for research mode (also see `notes/ideas.md`).
Format: `- [ ] [build] title: details / acceptance`. Tick with `[x]` and add the date and result.

## Build
- [ ] [build] GPU baselines: on the GPU box run the full dev bench for `--attack baseline` (upstream miner)
  and for v0 with `--out results/dev-baseline.json` / `results/dev-v0.json`; record mean, success rate,
  p95 time, failures, variant disagreements in STATUS.md and research_log. Explain every failed image (reason).
- [ ] [build] Competitor proxy on GPU with all variants: `python -m tools.competitor_scores --variants default,tf32_off,tf32_on`;
  update notes/competition.md (does any top response fail under some variant? how thin are their margins?).
- [ ] [build] Throughput profile: steps/s of the search step at the dev images' sizes for fp32, fp16 autocast,
  bf16, channels_last, torch.compile; write notes/throughput.md and adopt the fastest admissible default in AttackConfig.
- [ ] [build] Live timing probe `tools/task_probe.py`: poll GET /task every 0.5 s across >= 10 boundaries; log when
  task_id changes relative to the boundary and the task image download time. Update notes/timing.md; if the
  usable budget differs from 22 s, write the proposal in notes/harness_proposals.md.
- [ ] [build] Live dry runs: `python -m miner26.neuron --dry-run` on 5 consecutive live tasks; record official
  scores and wall-clock since boundary in research_log.
- [ ] [build] Early exit in the attack when the estimate cannot improve (e.g. alias images already at the
  ceiling); the attack must still respect the deadline. Unit test it on TinyNet.
- [ ] [build] `tools/leaderboard.py` (needs bittensor + NETUID): list validators (permit, stake > 10k), GET
  /leaderboard/<hotkey>, write the top-10 avgScore per validator plus a stake-weighted top list to
  status/leaderboard.json. NETUID is 26 (finney).

## Research directions (most promising first)
- [ ] [research] Margin-sparsity trade-off: top competitors reach ~1% nonzero with margin >= 10. Tune kappa,
  safety_margin, L1 schedule to land there.
- [ ] [research] Reliability: drive failures and variant disagreements to zero before chasing score.
- [ ] [research] Speed: more steps per budget (see throughput notes).
