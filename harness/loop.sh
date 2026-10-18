#!/usr/bin/env bash
# Perturb26 autonomous loop: Ralph-style build iterations + autoresearch experiments,
# each one a fresh headless Claude CLI session. Run under pm2 (harness/ecosystem.config.js).
#
# One iteration:
#   sync git -> run agent (prompts/<mode>.md) -> commit leftovers -> protected-path check
#   -> gates (harness/gate.sh) -> if miner code changed: harness runs the dev bench itself
#   -> on a gain, an independent reviewer session re-checks on the holdout with veto
#   -> accept (status/best.json, merge loop->main, tag, push) or discard code, keep notes.
#
# Control: touch STOP (exit after the current iteration), touch PAUSE (idle), see harness/ctl.sh.
set -uo pipefail
# shellcheck source=harness/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
cd "$ROOT" || exit 1

ITERS_TSV="$STATUS/iterations.tsv"
BEST_JSON="$STATUS/best.json"

init_ledgers() {  # harness-owned files exist and are committed before any agent runs
  [[ -f "$ITERS_TSV" ]] || printf 'ts\titer\tmode\trc\tseconds\tcost_usd\tturns\tresult\n' > "$ITERS_TSV"
  [[ -f "$STATUS/results.tsv" ]] || printf 'ts\titer\tmode\tcommit\tdev_score\tholdout_score\tdecision\treason\n' > "$STATUS/results.tsv"
  commit_all "loop: init ledgers"
}

next_iter() {
  local last
  last="$(tail -n 1 "$ITERS_TSV" | cut -f2)"
  [[ "$last" =~ ^[0-9]+$ ]] && echo $((last + 1)) || echo 1
}

total_cost() {
  [[ -f "$ITERS_TSV" ]] || { echo 0; return; }
  awk -F'\t' 'NR>1 && $6 ~ /^[0-9.]+$/ {s+=$6} END {printf "%.2f", s+0}' "$ITERS_TSV"
}

pick_mode() {
  case "$LOOP_MODE" in build | research) echo "$LOOP_MODE"; return ;; esac
  if grep -qE '^- \[ \] \[build\]' "$STATUS/backlog.md" 2>/dev/null; then echo build; else echo research; fi
}

render_prompt() {  # render_prompt mode iter
  cat "$ROOT/prompts/$1.md"
  cat <<EOF

---
## Loop context (written by the harness for this iteration)
- iteration: $2
- mode: $1
- branch: $WORK_BRANCH (never touch $MAIN_BRANCH; never push; the harness does both)
- bench budget per image: ${BENCH_BUDGET}s
- target score: $TARGET_SCORE (theoretical ceiling 0.99303)
- accepted best (status/best.json):
\`\`\`json
$(cat "$BEST_JSON" 2>/dev/null || echo '{}')
\`\`\`
- last ledger rows (status/results.tsv):
\`\`\`
$(tail -n 15 "$STATUS/results.tsv" 2>/dev/null)
\`\`\`
EOF
}

record_iteration() {  # record_iteration iter mode rc seconds jsonl
  local stats
  stats="$("$PY" - "$5" <<'EOF'
import json, sys
cost, turns, kind = "", "", "no_result"
try:
    for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("type") == "result":
            cost = event.get("total_cost_usd", "")
            turns = event.get("num_turns", "")
            kind = event.get("subtype", "result")
except Exception:
    pass
print(f"{cost}\t{turns}\t{kind}")
EOF
)"
  # Staged outside the tree so the protected-path check sees only the agent's changes;
  # flushed by flush_iter_row (or discard_iteration) once the iteration is decided.
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$1" "$2" "$3" "$4" "$stats" > "$LOGS/.last_iter_row"
}

flush_iter_row() {
  [[ -f "$LOGS/.last_iter_row" ]] || return 0
  cat "$LOGS/.last_iter_row" >> "$ITERS_TSV"
  rm -f "$LOGS/.last_iter_row"
}

run_agent() {  # run_agent mode iter
  local tag jsonl started rc
  tag="$(printf '%04d' "$2")"
  jsonl="$LOGS/iter-$tag-$1.jsonl"
  started=$(date +%s)
  log "iter $2: agent start mode=$1 model=$CLAUDE_MODEL log=$jsonl"
  # shellcheck disable=SC2086
  render_prompt "$1" "$2" | timeout --kill-after=60 "$ITER_TIMEOUT" \
    "$CLAUDE_BIN" -p --model "$CLAUDE_MODEL" --max-turns "$MAX_TURNS" \
    --output-format stream-json --verbose $CLAUDE_FLAGS > "$jsonl" 2> "$jsonl.err"
  rc=$?
  record_iteration "$2" "$1" "$rc" $(($(date +%s) - started)) "$jsonl"
  log "iter $2: agent exit rc=$rc after $(($(date +%s) - started))s"
}

write_best() {  # write_best iter commit dev holdout dev_json holdout_json
  "$PY" - "$BEST_JSON" "$@" <<'EOF'
import json, sys, datetime
path, iteration, commit, dev, holdout, dev_json, holdout_json = sys.argv[1:8]
json.dump({
    "iter": int(iteration), "commit": commit, "dev_score": float(dev), "holdout_score": float(holdout),
    "dev_result": dev_json, "holdout_result": holdout_json,
    "accepted_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, open(path, "w", encoding="utf-8"), indent=1)
EOF
}

holdout_score_of() { json_get "$1" 'd["summary"]["mean_official"]'; }

seed_best() {
  [[ -n "$(json_get "$BEST_JSON" 'd["dev_score"]')" ]] && return 0
  log "no status/best.json: benchmarking current HEAD to seed the baseline"
  local sha dev holdout hold_json
  sha="$(git rev-parse --short HEAD)"
  dev="$(bench_score "$DEV_MANIFEST" "$ROOT/results/dev-seed-$sha.json")" || { log "ERROR seed dev bench failed"; return 1; }
  mkdir -p "$HOLDOUT_DIR/results"
  hold_json="$HOLDOUT_DIR/results/holdout-seed-$sha.json"
  holdout="$(bench_score "$HOLDOUT_DIR/holdout.json" "$hold_json")" || { log "ERROR seed holdout bench failed"; return 1; }
  write_best 0 "$sha" "$dev" "$holdout" "results/dev-seed-$sha.json" "$hold_json"
  ledger 0 seed "$sha" "$dev" "$holdout" seed "baseline of HEAD"
  commit_all "loop: seed baseline dev=$dev holdout=$holdout"
  push_branch "$WORK_BRANCH"
}

target_reached() {
  local dev holdout
  dev="$(json_get "$BEST_JSON" 'd["dev_score"]')"
  holdout="$(json_get "$BEST_JSON" 'd["holdout_score"]')"
  [[ -n "$dev" && -n "$holdout" ]] && ! float_gt "$TARGET_SCORE" "$dev" && ! float_gt "$TARGET_SCORE" "$holdout"
}

evaluate_and_decide() {  # evaluate_and_decide iter mode start_sha
  local iter="$1" mode="$2" start="$3" sha dev best_dev best_hold dev_json verdict holdout
  sha="$(git rev-parse --short HEAD)"
  dev_json="results/dev-iter$iter.json"
  log "iter $iter: harness dev bench on $sha"
  dev="$(bench_score "$DEV_MANIFEST" "$ROOT/$dev_json")"
  if [[ -z "$dev" ]]; then
    discard_iteration "$start" bench_failed "$iter"; ledger "$iter" "$mode" "$sha" "" "" discard bench_failed; return
  fi
  best_dev="$(json_get "$BEST_JSON" 'd["dev_score"]')"; best_dev="${best_dev:-0}"
  best_hold="$(json_get "$BEST_JSON" 'd["holdout_score"]')"; best_hold="${best_hold:-0}"
  local threshold
  threshold="$("$PY" -c "print(float('$best_dev') + float('$MIN_GAIN'))")"
  if ! float_gt "$dev" "$threshold"; then
    local floor
    floor="$("$PY" -c "print(float('$best_dev') - float('$MIN_GAIN'))")"
    if [[ "$mode" == build ]] && float_gt "$dev" "$floor"; then
      ledger "$iter" "$mode" "$sha" "$dev" "" keep "build change, no regression"
      commit_all "loop: iter $iter build kept (dev=$dev)"; push_branch "$WORK_BRANCH"
    else
      discard_iteration "$start" "no_gain dev=$dev best=$best_dev" "$iter"
      ledger "$iter" "$mode" "$sha" "$dev" "" discard "no_gain (best $best_dev)"
    fi
    return
  fi
  log "iter $iter: dev $dev > best $best_dev + $MIN_GAIN; requesting independent review"
  commit_all "loop: iter $iter dev bench $dev"
  read -r verdict holdout < <("$HARNESS/review.sh" "$iter" "$start" "$(git rev-parse HEAD)" "$dev_json")
  local hold_floor
  hold_floor="$("$PY" -c "print(float('$best_hold') - float('$HOLDOUT_TOL'))")"
  if [[ "$verdict" == APPROVE && -n "$holdout" ]] && float_gt "$holdout" "$hold_floor"; then
    write_best "$iter" "$sha" "$dev" "$holdout" "$dev_json" "$HOLDOUT_DIR/results/holdout-iter$iter.json"
    ledger "$iter" "$mode" "$sha" "$dev" "$holdout" accept "reviewer APPROVE"
    commit_all "loop: iter $iter accepted dev=$dev holdout=$holdout"
    push_branch "$WORK_BRANCH"
    promote_to_main "promote: iter $iter dev=$dev holdout=$holdout" "best-iter$iter" || log "WARN promote failed"
  else
    discard_iteration "$start" "review $verdict holdout=$holdout" "$iter"
    ledger "$iter" "$mode" "$sha" "$dev" "$holdout" discard "review $verdict (holdout floor $hold_floor)"
  fi
}

# ---------------------------------------------------------------- main
log "loop start (pid $$, mode=$LOOP_MODE, push=$PUSH, max_iters=$MAX_ITERS)"
git_sync || { log "ERROR initial git sync failed"; exit 1; }
init_ledgers
seed_best || { log "ERROR could not seed baseline; fix setup (harness/setup_box.sh)"; exit 1; }

while true; do
  [[ -f "$ROOT/STOP" ]] && { log "STOP file present; exiting"; exit 0; }
  while [[ -f "$ROOT/PAUSE" ]]; do sleep 30; done
  iter="$(next_iter)"
  if ((iter > MAX_ITERS)); then log "MAX_ITERS=$MAX_ITERS reached"; touch "$ROOT/STOP"; exit 0; fi
  if [[ "$MAX_COST_USD" != 0 ]] && float_gt "$(total_cost)" "$MAX_COST_USD"; then
    log "cost cap reached ($(total_cost) > $MAX_COST_USD USD)"; touch "$ROOT/STOP"; exit 0
  fi
  if target_reached; then log "target $TARGET_SCORE reached on dev and holdout"; touch "$ROOT/STOP"; exit 0; fi
  if ! git_sync; then sleep 300; continue; fi

  start_sha="$(git rev-parse HEAD)"
  mode="$(pick_mode)"
  run_agent "$mode" "$iter"
  commit_all "loop: iter $iter $mode leftovers (uncommitted by agent)"
  head_sha="$(git rev-parse HEAD)"

  if [[ "$head_sha" == "$start_sha" ]]; then
    ledger "$iter" "$mode" "$(git rev-parse --short HEAD)" "" "" noop "agent made no commits"
    flush_iter_row; commit_all "loop: iter $iter noop"; push_branch "$WORK_BRANCH"; sleep "$ITER_SLEEP"; continue
  fi
  violations="$(protected_violations "$start_sha" "$head_sha" | tr '\n' ' ')"
  if [[ -n "$violations" ]]; then
    log "iter $iter: protected paths touched: $violations"
    discard_iteration "$start_sha" "protected: $violations" "$iter"
    ledger "$iter" "$mode" "$(git rev-parse --short "$head_sha")" "" "" discard "protected: $violations"
    sleep "$ITER_SLEEP"; continue
  fi
  if ! "$HARNESS/gate.sh" > "$LOGS/gate-iter$iter.log" 2>&1; then
    log "iter $iter: gates failed (logs/gate-iter$iter.log)"
    discard_iteration "$start_sha" gate_failed "$iter"
    ledger "$iter" "$mode" "$(git rev-parse --short "$head_sha")" "" "" discard gate_failed
    sleep "$ITER_SLEEP"; continue
  fi
  if git diff --quiet "$start_sha" "$head_sha" -- miner26; then
    ledger "$iter" "$mode" "$(git rev-parse --short HEAD)" "" "" keep "no miner26 change; gates pass"
    commit_all "loop: iter $iter kept"; push_branch "$WORK_BRANCH"
  else
    evaluate_and_decide "$iter" "$mode" "$start_sha"
  fi
  flush_iter_row
  commit_all "loop: iter $iter status"
  push_branch "$WORK_BRANCH"
  sleep "$ITER_SLEEP"
done
