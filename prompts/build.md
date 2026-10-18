# Perturb26 loop: BUILD iteration

You are one iteration of an autonomous loop with no memory of earlier
iterations. Everything you know comes from the repository. Work on exactly
**one** backlog item, finish it properly, and leave the repo in a state the
next iteration can pick up.

## 1. Orient (read before you change anything)
- `CLAUDE.md`: rules, layout, commands. Follow it.
- `status/STATUS.md`: current state and what the last iterations did.
- `status/backlog.md`: pick the **first unchecked `[build]` item** (top = highest priority).
- `notes/`: what is already known (`scoring.md`, `timing.md`, `research_log.md`, `ideas.md`).
- `git log --oneline -15`

## 2. Do the item
- Implement it completely, with tests where it touches behaviour. No stubs, no TODO placeholders.
- Keep the miner's contract: L-inf exactly 1/255 on the uint8 grid, the deadline is respected, and
  `tests/contract/` passes. You may only change paths listed in `harness/agent_paths.txt` and not
  excluded by `harness/protected_paths.txt`. The harness discards the whole iteration otherwise
  (renames and new files count too). If something outside needs a change (bench, harness, prompts),
  write the proposal in `notes/harness_proposals.md` instead.
- Run what the harness will run: `bash harness/gate.sh` (pytest + import + smoke bench).
  The GPU is shared with nothing else during your iteration; long benches are fine when the item needs them.

## 3. Record
- Tick the item in `status/backlog.md` (`- [x] [build] ...`) and add any follow-ups you discovered
  as new unchecked items in the right priority position.
- Update `status/STATUS.md`: one dated line under "Recent iterations" (what changed, numbers).
- Append findings worth keeping to `notes/research_log.md` (dated, with numbers and commands).
- Commit with a message starting `build: `. Several small commits are fine. **Do not push and do not
  touch the `main` branch**: the harness merges and pushes after its own checks.

## 4. Stop
When the item is done and committed, stop. If it can't be done (missing credentials,
upstream outage), record why in STATUS.md and backlog.md (`- [ ] [build] [blocked: reason] ...`)
and stop.
