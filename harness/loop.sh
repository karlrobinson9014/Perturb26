#!/usr/bin/env bash
# Perturb26 autonomous loop: Ralph-style build iterations + autoresearch experiments,
# each one a fresh headless Claude CLI session. Run under pm2 (harness/ecosystem.config.js).
#
# One iteration:
#   sync git -> run agent (prompts/<mode>.md) -> commit leftovers -> path allowlist check
#   -> venv unchanged -> gates (harness/gate.sh) -> if miner26/ changed: the harness runs the
#   dev bench itself -> on a gain: the harness runs the holdout bench, then an independent
#   reviewer session reviews everything not yet on main, with veto
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
HARNESS_RESULTS="results/harness"
mkdir -p "$ROOT/$HARNESS_RESULTS" "$HOLDOUT_DIR/results"

init_ledgers() {  # harness-owned files exist and are committed before any agent runs
  [[ -f "$ITERS_TSV" ]] || printf 'ts\titer\tmode\trc\tseconds\tcost_usd\tturns\tresult\n' > "$ITERS_TSV"
  [[ -f "$STATUS/results.tsv" ]] || printf 'ts\titer\tmode\tcommit\tdev_score\tholdout_score\tdecision\treason\n' > "$STATUS/results.tsv"
  flush_iter_row  # a row staged by a run that was killed mid-iteration
  commit_all "loop: init ledgers"
}

next_iter() {  # one past the highest iteration either ledger has seen
  awk -F'\t' 'FNR>1 && $2 ~ /^[0-9]+$/ && $2+0 > m {m = $2+0} END {print m+1}' "$ITERS_TSV" "$STATUS/results.tsv"
}

total_cost() {
  awk -F'\t' 'NR>1 && $6 ~ /^[0-9.]+$/ {s+=$6} END {printf "%.2f", s+0}' "$ITERS_TSV"
}

pick_mode() {  # pick_mode [operator_item]
  if [[ -n "${1:-}" ]]; then operator_item_mode "$1"; return; fi
  case "$LOOP_MODE" in build | research) echo "$LOOP_MODE"; return ;; esac
  # blocked items ("- [ ] [build] [blocked: ...]") must not pin the loop in build mode
  if grep -E '^- \[ \] \[build\]' "$STATUS/backlog.md" 2>/dev/null | grep -qv '\[blocked'; then echo build; else echo research; fi
}

render_prompt() {  # render_prompt mode iter [operator_item]
  cat "$ROOT/prompts/$1.md"
  printf '\n---\n## Operator directives (from the human operator; they override your defaults and the backlog)\n'
  cat "$OPERATOR_DIR/directives.md" 2>/dev/null
  if [[ -n "${3:-}" && -f "$3" ]]; then
    printf '\n## Operator task for THIS iteration (do this instead of picking from the backlog)\n'
    sed '1{/^mode:/d}' "$3"
  fi
  cat <<EOF

---
## Loop context (written by the harness for this iteration)
- iteration: $2
- mode: $1
- branch: $WORK_BRANCH (never touch $MAIN_BRANCH; never push; the harness does both)
- bench budget per image: ${BENCH_BUDGET}s (an attack that overruns it is rejected)
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
  # Staged outside the tree so the path check sees only the agent's changes;
  # flushed by flush_iter_row (or discard_iteration) once the iteration is decided.
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$1" "$2" "$3" "$4" "$stats" > "$LOGS/.last_iter_row"
}

flush_iter_row() {
  [[ -f "$LOGS/.last_iter_row" ]] || return 0
  cat "$LOGS/.last_iter_row" >> "$ITERS_TSV"
  rm -f "$LOGS/.last_iter_row"
}

run_agent() {  # run_agent mode iter [operator_item]
  local tag jsonl prompt_file started rc
  tag="$(printf '%04d' "$2")"
  jsonl="$LOGS/iter-$tag-$1.jsonl"
  prompt_file="$(mktemp)"
  render_prompt "$1" "$2" "${3:-}" > "$prompt_file"
  cp "$prompt_file" "$LOGS/iter-$tag-$1.prompt.md"  # the dashboard shows what the agent was told
  started=$(date +%s)
  log "iter $2: agent start mode=$1 model=$CLAUDE_MODEL log=$jsonl"
  # shellcheck disable=SC2086
  run_isolated "$ITER_TIMEOUT" "$prompt_file" "$jsonl" "$jsonl.err" \
    $AGENT_RUN_PREFIX "$CLAUDE_BIN" -p --model "$CLAUDE_MODEL" --max-turns "$MAX_TURNS" \
    --output-format stream-json --verbose $CLAUDE_FLAGS --disallowedTools "$CLAUDE_DENY_TOOLS"
  rc=$?
  rm -f "$prompt_file"
  record_iteration "$2" "$1" "$rc" $(($(date +%s) - started)) "$jsonl"
  log "iter $2: agent exit rc=$rc after $(($(date +%s) - started))s"
}

# best.json holds scores only: no holdout paths, since it is pasted into every agent prompt.
write_best() {  # write_best iter commit dev holdout dev_json
  "$PY" - "$BEST_JSON" "$@" <<'EOF'
import json, sys, datetime
path, iteration, commit, dev, holdout, dev_json = sys.argv[1:7]
try:
    peak = float(json.load(open(path, encoding="utf-8")).get("holdout_peak", float(holdout)))
except Exception:
    peak = float(holdout)
json.dump({
    "iter": int(iteration), "commit": commit, "dev_score": float(dev), "holdout_score": float(holdout),
    "holdout_peak": max(peak, float(holdout)), "dev_result": dev_json,
    "accepted_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, open(path, "w", encoding="utf-8"), indent=1)
EOF
}

seed_best() {
  [[ -n "$(json_get "$BEST_JSON" 'd["dev_score"]')" ]] && return 0
  log "no status/best.json: benchmarking current HEAD to seed the baseline"
  local sha dev holdout
  sha="$(git rev-parse --short HEAD)"
  dev="$(bench_score "$DEV_MANIFEST" "$ROOT/$HARNESS_RESULTS/dev-seed-$sha.json" "$SINCE")" || { log "ERROR seed dev bench failed"; return 1; }
  commit_all "loop: seed dev bench $dev"
  holdout="$(bench_score "$HOLDOUT_DIR/holdout.json" "$HOLDOUT_DIR/results/holdout-seed-$sha.json" "$SINCE")" || { log "ERROR seed holdout bench failed"; return 1; }
  write_best 0 "$sha" "$dev" "$holdout" "$HARNESS_RESULTS/dev-seed-$sha.json"
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

bench_failure() {  # bench_failure rc iter mode start sha dev label
  if [[ "$1" == 2 ]]; then
    discard_iteration "$4" "integrity violation in $7 bench" "$2" strict
    ledger "$2" "$3" "$5" "$6" "" discard "integrity violation during $7 bench (strict discard)"
  else
    discard_iteration "$4" "$7 bench failed or overran" "$2"
    ledger "$2" "$3" "$5" "$6" "" discard "$7 bench failed or overran budget"
  fi
}

evaluate_and_decide() {  # evaluate_and_decide iter mode start_sha
  local iter="$1" mode="$2" start="$3" sha dev best_dev peak dev_json hold_json holdout verdict rc problems
  sha="$(git rev-parse --short HEAD)"
  dev_json="$HARNESS_RESULTS/dev-iter$iter-$sha.json"
  log "iter $iter: harness dev bench on $sha"
  dev="$(bench_score "$DEV_MANIFEST" "$ROOT/$dev_json" "$SINCE")"; rc=$?
  if ((rc != 0)) || [[ -z "$dev" ]]; then bench_failure "$rc" "$iter" "$mode" "$start" "$sha" "" dev; return; fi
  commit_all "loop: iter $iter dev bench $dev"
  best_dev="$(json_get "$BEST_JSON" 'd["dev_score"]')"; best_dev="${best_dev:-0}"
  if ! float_gt "$dev" "$(calc "$best_dev + $MIN_GAIN")"; then
    if [[ "$mode" == build ]] && float_gt "$dev" "$(calc "$best_dev - $MIN_GAIN")"; then
      ledger "$iter" "$mode" "$sha" "$dev" "" keep "build change, no regression (reviewed at next promotion)"
    else
      discard_iteration "$start" "no_gain dev=$dev best=$best_dev" "$iter"
      ledger "$iter" "$mode" "$sha" "$dev" "" discard "no_gain (best $best_dev)"
    fi
    return
  fi

  log "iter $iter: dev $dev > best $best_dev + $MIN_GAIN; harness holdout bench"
  hold_json="$HOLDOUT_DIR/results/holdout-iter$iter-$sha.json"
  holdout="$(bench_score "$HOLDOUT_DIR/holdout.json" "$hold_json" "$SINCE")"; rc=$?
  if ((rc != 0)) || [[ -z "$holdout" ]]; then bench_failure "$rc" "$iter" "$mode" "$start" "$sha" "$dev" holdout; return; fi
  peak="$(json_get "$BEST_JSON" 'd.get("holdout_peak", d["holdout_score"])')"; peak="${peak:-0}"
  local hold_floor
  hold_floor="$(calc "$peak - $HOLDOUT_TOL")"
  if ! float_gt "$holdout" "$hold_floor"; then
    discard_iteration "$start" "holdout $holdout <= floor $hold_floor" "$iter"
    ledger "$iter" "$mode" "$sha" "$dev" "$holdout" discard "holdout below peak-tol ($hold_floor)"; return
  fi

  verdict="$("$HARNESS/review.sh" "$iter" "$(reviewed_base)" "$(git rev-parse HEAD)" "$dev_json" "$hold_json")"
  problems="$(integrity_check "$SINCE" "status/reviews/iter-$iter.json")"
  if [[ -n "$problems" ]]; then
    log "iter $iter: integrity violation after review: $(echo "$problems" | tr '\n' ' ')"
    verdict=REJECT
  fi
  if [[ "$verdict" == APPROVE ]]; then
    write_best "$iter" "$sha" "$dev" "$holdout" "$dev_json"
    ledger "$iter" "$mode" "$sha" "$dev" "$holdout" accept "reviewer APPROVE"
    commit_all "loop: iter $iter accepted dev=$dev holdout=$holdout"
    push_branch "$WORK_BRANCH"
    promote_to_main "promote: iter $iter dev=$dev holdout=$holdout" "best-iter$iter-$sha" || log "WARN promote failed"
  else
    discard_iteration "$start" "review $verdict" "$iter"
    ledger "$iter" "$mode" "$sha" "$dev" "$holdout" discard "review $verdict"
  fi
}

finish_iteration() {  # finish_iteration iter
  operator_finish "$1" "$(awk -F'\t' -v i="$1" '$2 == i {o = $7 ": " $8} END {print o}' "$STATUS/results.tsv" 2>/dev/null)"
  flush_iter_row
  commit_all "loop: iter $1 status"
  push_branch "$WORK_BRANCH"
  sleep "$ITER_SLEEP"
}

# ---------------------------------------------------------------- main
log "loop start (pid $$, mode=$LOOP_MODE, push=$PUSH, max_iters=$MAX_ITERS)"
git_sync || { log "ERROR initial git sync failed"; exit 1; }
init_ledgers
# Integrity reference time, held in this process only (agents can't move it). Anything in
# .venv, data/, upstream/ or the work tree that changes after this is caught by integrity_check.
SINCE="$(date +%s)"
sleep 1
seed_best || { log "ERROR could not seed baseline; fix setup (harness/setup_box.sh)"; exit 1; }

# Integrity problems right after an agent-code step. venv/upstream/data problems stop the
# loop for a human; stray tree files only discard the iteration (strictly).
check_after() {  # check_after step iter mode start short -> 0 clean, 1 handled (caller continues)
  local problems
  problems="$(integrity_check "$SINCE")"
  [[ -z "$problems" ]] && return 0
  log "iter $2: integrity violation after $1: $(echo "$problems" | tr '\n' ' ')"
  discard_iteration "$4" "integrity violation after $1" "$2" strict
  ledger "$2" "$3" "$5" "" "" discard "integrity violation after $1: $(echo "$problems" | head -3 | tr '\n' ' ')"
  if grep -qE '^(venv|upstream|data):' <<<"$problems"; then
    log "environment outside git was modified; stopping for a human"
    finish_iteration "$2"; touch "$ROOT/STOP"; exit 0
  fi
  finish_iteration "$2"
  return 1
}

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
  op_item="$(operator_claim)"
  mode="$(pick_mode "$op_item")"
  [[ -n "$op_item" ]] && log "iter $iter: operator task $(basename "$op_item") (mode=$mode)"
  run_agent "$mode" "$iter" "$op_item"
  commit_all "loop: iter $iter $mode leftovers (uncommitted by agent)"
  head_sha="$(git rev-parse HEAD)"
  short="$(git rev-parse --short HEAD)"

  if [[ "$head_sha" == "$start_sha" ]]; then
    ledger "$iter" "$mode" "$short" "" "" noop "agent made no commits"
    finish_iteration "$iter"; continue
  fi
  check_after "agent session" "$iter" "$mode" "$start_sha" "$short" || continue
  violations="$(path_violations "$start_sha" "$head_sha" | tr '\n' ' ')"
  if [[ -n "$violations" ]]; then
    log "iter $iter: paths outside the agent allowlist: $violations"
    discard_iteration "$start_sha" "paths: $violations" "$iter"
    ledger "$iter" "$mode" "$short" "" "" discard "paths: $violations"
    finish_iteration "$iter"; continue
  fi
  if ! "$HARNESS/gate.sh" > "$LOGS/gate-iter$iter.log" 2>&1; then
    log "iter $iter: gates failed (logs/gate-iter$iter.log)"
    discard_iteration "$start_sha" gate_failed "$iter"
    ledger "$iter" "$mode" "$short" "" "" discard gate_failed
    finish_iteration "$iter"; continue
  fi
  check_after gates "$iter" "$mode" "$start_sha" "$short" || continue
  if git diff --quiet "$start_sha" "$head_sha" -- miner26; then
    ledger "$iter" "$mode" "$short" "" "" keep "no miner26 change; gates pass"
  else
    evaluate_and_decide "$iter" "$mode" "$start_sha"
  fi
  finish_iteration "$iter"
done
