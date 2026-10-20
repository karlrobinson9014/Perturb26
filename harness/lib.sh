#!/usr/bin/env bash
# Shared helpers for the Perturb26 loop. Sourced by loop.sh / review.sh / gate.sh / ctl.sh.
# shellcheck disable=SC2034

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="$ROOT/harness"
STATUS="$ROOT/status"
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$STATUS/reviews" "$ROOT/results"

load_env() {
  if [[ -f "$HARNESS/loop.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$HARNESS/loop.env"
    set +a
  fi
  WORK_BRANCH="${WORK_BRANCH:-loop}"
  MAIN_BRANCH="${MAIN_BRANCH:-main}"
  REMOTE="${REMOTE:-origin}"
  PUSH="${PUSH:-0}"
  CLAUDE_BIN="${CLAUDE_BIN:-claude}"
  CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-5-5}"
  REVIEW_MODEL="${REVIEW_MODEL:-$CLAUDE_MODEL}"
  CLAUDE_FLAGS="${CLAUDE_FLAGS:---dangerously-skip-permissions}"
  # claude.ai connectors loop sessions must never use (mail, files, calendar, ...); the research
  # connectors (Consensus, SciSpace) stay available.
  CLAUDE_DENY_TOOLS="${CLAUDE_DENY_TOOLS:-mcp__claude_ai_Gmail,mcp__claude_ai_Google_Drive,mcp__claude_ai_Google_Calendar,mcp__claude_ai_Claude_Docs,mcp__claude_ai_Slack,mcp__claude_ai_Notion,mcp__claude_ai_Asana,mcp__claude_ai_Atlassian,mcp__claude_ai_Box,mcp__claude_ai_Canva,mcp__claude_ai_Figma,mcp__claude_ai_HubSpot,mcp__claude_ai_Intercom,mcp__claude_ai_Linear,mcp__claude_ai_monday_com}"
  AGENT_RUN_PREFIX="${AGENT_RUN_PREFIX:-}"
  MAX_TURNS="${MAX_TURNS:-150}"
  ITER_TIMEOUT="${ITER_TIMEOUT:-5400}"
  REVIEW_TIMEOUT="${REVIEW_TIMEOUT:-3600}"
  MAX_ITERS="${MAX_ITERS:-200}"
  MAX_COST_USD="${MAX_COST_USD:-0}"
  ITER_SLEEP="${ITER_SLEEP:-30}"
  LOOP_MODE="${LOOP_MODE:-auto}"
  BENCH_BUDGET="${BENCH_BUDGET:-22}"
  TIME_SLACK="${TIME_SLACK:-2}"
  MIN_GAIN="${MIN_GAIN:-0.0005}"
  HOLDOUT_TOL="${HOLDOUT_TOL:-0.003}"
  TARGET_SCORE="${TARGET_SCORE:-0.993}"
  HOLDOUT_DIR="${HOLDOUT_DIR:-$HOME/perturb26-holdout}"
  OPERATOR_DIR="${OPERATOR_DIR:-$HOME/perturb26-operator}"
  DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/bench/manifests/dev.json}"
  PY="${PY:-$ROOT/.venv/bin/python}"
  BENCH_RUN_PREFIX="${BENCH_RUN_PREFIX:-}"
  export PYTHONPATH="$ROOT"
  # The claude CLI is a node script installed with nvm; pm2 and cron do not load nvm.
  if ! command -v node >/dev/null 2>&1; then
    local nvm_bin
    nvm_bin="$(ls -d /opt/nvm/versions/node/*/bin "$HOME"/.nvm/versions/node/*/bin 2>/dev/null | sort -V | tail -1)"
    [[ -n "$nvm_bin" ]] && export PATH="$nvm_bin:$PATH"
  fi
  # Claude Code refuses --dangerously-skip-permissions as root unless told it is in a sandbox.
  # This box is a disposable container; for real isolation run agents as another user
  # (AGENT_RUN_PREFIX, README "Isolation") and set IS_SANDBOX=0 in loop.env.
  if [[ "$(id -u)" == 0 ]]; then export IS_SANDBOX="${IS_SANDBOX:-1}"; fi
  # loop.env is sourced with set -a; the holdout location must not reach agent environments.
  export -n HOLDOUT_DIR OPERATOR_DIR 2>/dev/null || true
  mkdir -p "$OPERATOR_DIR/queue" "$OPERATOR_DIR/active" "$OPERATOR_DIR/done"
  [[ -f "$OPERATOR_DIR/directives.md" ]] || printf '(none)\n' > "$OPERATOR_DIR/directives.md"
}

# ---------------------------------------------------------------- operator directives
# The operator (a human, via harness/dashboard.py or by hand) steers the loop with:
#   $OPERATOR_DIR/directives.md   standing directives, pasted into every agent prompt
#   $OPERATOR_DIR/queue/*.md      one-shot ideas or plans; the oldest is handed to the next
#                                 iteration (first line `mode: build|research`, default research)
#                                 and moved to done/ with the iteration's outcome.
# The directory lives outside the repo (dashboard edits never dirty the tree) and is not
# exported to agent environments.

operator_claim() {  # operator_claim -> prints the claimed item (now in active/) or nothing
  local item
  item="$(ls -1 "$OPERATOR_DIR"/active/*.md 2>/dev/null | head -1)"  # left over by a killed run
  if [[ -z "$item" ]]; then
    item="$(ls -1 "$OPERATOR_DIR"/queue/*.md 2>/dev/null | sort | head -1)"
    [[ -n "$item" ]] || return 0
    mv "$item" "$OPERATOR_DIR/active/" || return 0
    item="$OPERATOR_DIR/active/$(basename "$item")"
  fi
  echo "$item"
}

operator_item_mode() {  # operator_item_mode file -> build|research
  local mode
  mode="$(head -1 "$1" 2>/dev/null | sed -n 's/^mode:[[:space:]]*\(build\|research\)[[:space:]]*$/\1/p')"
  echo "${mode:-research}"
}

operator_finish() {  # operator_finish iter outcome
  local item
  for item in "$OPERATOR_DIR"/active/*.md; do
    [[ -f "$item" ]] || continue
    printf '\n---\ndone: iter %s, %s, %s\n' "$1" "$(ts)" "$2" >> "$item"
    mv "$item" "$OPERATOR_DIR/done/iter-$1-$(basename "$item")"
  done
}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# stderr, never stdout: review.sh's stdout is its machine-read verdict line.
log() { echo "[$(ts)] $*" | tee -a "$LOGS/harness.log" >&2; }

retry() {  # retry N cmd...
  local n="$1"; shift
  local i
  for ((i = 1; i <= n; i++)); do
    "$@" && return 0
    sleep $((i * 3))
  done
  return 1
}

json_get() {  # json_get file python-expr-on-d   (prints "" when missing)
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    v = eval(sys.argv[2], {}, {"d": d})
    print("" if v is None else v)
except Exception:
    print("")
EOF
}

float_gt() { "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$1" "$2"; }
calc() { "$PY" -c "print($1)"; }

# Run a command in its own session and kill whatever it leaves behind (background
# benches, servers), so nothing an agent started competes with the harness for the GPU.
run_isolated() {  # run_isolated timeout_s stdin_file stdout_file stderr_file cmd...
  local limit="$1" input="$2" out="$3" err="$4"; shift 4
  local pid rc
  if ! command -v setsid >/dev/null; then  # non-Linux dev machines: no session isolation
    timeout --kill-after=60 "$limit" "$@" < "$input" > "$out" 2> "$err"
    return $?
  fi
  setsid bash -c 'exec timeout --kill-after=60 "$@"' _ "$limit" "$@" < "$input" > "$out" 2> "$err" &
  pid=$!
  wait "$pid"; rc=$?
  pkill -KILL -s "$pid" 2>/dev/null || true
  return "$rc"
}

# What agent-written code (the agent session, its tests in the gate, the attack phase of a
# bench) may have left behind since `since` (epoch seconds, held only in harness memory).
# Uses ctime, which a user cannot set back. Prints one line per problem; empty = clean.
#   tree:<path>      uncommitted or untracked file in the work tree (ignored files included)
#   upstream:<what>  submodule moved off its pin or modified
#   venv:<path>      a file in .venv changed (site-packages, .pth, sitecustomize, ...)
#   data:<path>      a benchmark image changed
integrity_check() {  # integrity_check since_epoch [allowed_glob...]
  local since="$1"; shift
  local line path pattern hit
  (
    cd "$ROOT" || exit 1
    # -z: paths are never quoted (spaces, unicode); pathspec excludes keep the scan cheap.
    git status --porcelain=v1 -z --no-renames --ignored --untracked-files=all -- . \
      ':(exclude).venv' ':(exclude)data' ':(exclude)logs' | while IFS= read -r -d '' line; do
      path="${line:3}"
      case "$path" in
        data/* | .venv/* | logs/* | harness/loop.env | miner.env | STOP | PAUSE) continue ;;
        __pycache__/* | */__pycache__/* | .pytest_cache/* | */.pytest_cache/*) continue ;;
      esac
      for pattern in "$@"; do
        # shellcheck disable=SC2053
        [[ "$path" == $pattern ]] && continue 2
      done
      echo "tree:$path"
    done
    [[ "$(git -C upstream rev-parse HEAD 2>/dev/null)" == "$(git ls-tree HEAD upstream | awk '{print $3}')" ]] || echo "upstream:moved"
    [[ -z "$(git -C upstream status --porcelain --untracked-files=all 2>/dev/null | grep -v __pycache__)" ]] || echo "upstream:dirty"
    hit="$(find "$ROOT/.venv" -newerct "@$since" -type f -not -path '*/__pycache__/*' -print -quit 2>/dev/null)"
    [[ -z "$hit" ]] || echo "venv:$hit"
    hit="$(find "$ROOT/data" -newerct "@$since" -type f -print -quit 2>/dev/null)"
    [[ -z "$hit" ]] || echo "data:$hit"
  )
}

# ---------------------------------------------------------------- git

has_remote_branch() { git -C "$ROOT" ls-remote --exit-code --heads "$REMOTE" "$1" >/dev/null 2>&1; }

git_sync() {
  cd "$ROOT" || return 1
  retry 3 git fetch --prune "$REMOTE" || log "WARN fetch failed; continuing offline"
  if ! git rev-parse --verify -q "$WORK_BRANCH" >/dev/null; then
    git checkout -q -b "$WORK_BRANCH" "$MAIN_BRANCH"
  else
    git checkout -q "$WORK_BRANCH"
  fi
  if has_remote_branch "$WORK_BRANCH"; then
    git pull -q --rebase "$REMOTE" "$WORK_BRANCH" || { git rebase --abort 2>/dev/null; log "WARN rebase on $REMOTE/$WORK_BRANCH failed; kept local"; }
  fi
  # Human edits land on main (prompts, harness); bring them into the work branch.
  if has_remote_branch "$MAIN_BRANCH"; then
    git merge -q --no-edit "$REMOTE/$MAIN_BRANCH" || { git merge --abort 2>/dev/null; log "ERROR merge of $MAIN_BRANCH into $WORK_BRANCH conflicts; fix by hand"; return 1; }
  fi
  git submodule update --init -q
}

commit_all() {  # commit_all "message"
  cd "$ROOT" || return 1
  git add -A
  git diff --cached --quiet && return 0
  git commit -q -m "$1"
}

push_branch() {  # push_branch branch
  [[ "$PUSH" == "1" ]] || { log "PUSH=0: skip push of $1"; return 0; }
  retry 3 git -C "$ROOT" push -q "$REMOTE" "$1" --follow-tags || log "WARN push of $1 failed"
}

# Last work-branch commit already merged into main: everything after it is unreviewed.
reviewed_base() { git -C "$ROOT" merge-base "$MAIN_BRANCH" "$WORK_BRANCH"; }

promote_to_main() {  # promote_to_main "message" tag
  cd "$ROOT" || return 1
  # A failed checkout (e.g. main checked out in another git worktree) used to fall through to a
  # no-op merge on the work branch, so accepted versions silently never reached main.
  if ! git checkout -q "$MAIN_BRANCH" || [[ "$(git branch --show-current)" != "$MAIN_BRANCH" ]]; then
    git tag -a "$2" -m "$1 (NOT promoted: cannot check out $MAIN_BRANCH)" "$WORK_BRANCH" 2>/dev/null || true
    push_branch "$WORK_BRANCH"
    log "ERROR promote: cannot check out $MAIN_BRANCH (checked out in another worktree?); tagged $2 on $WORK_BRANCH, merge it into $MAIN_BRANCH by hand"
    return 1
  fi
  if has_remote_branch "$MAIN_BRANCH"; then
    git pull -q --ff-only "$REMOTE" "$MAIN_BRANCH" || log "WARN promote: $REMOTE/$MAIN_BRANCH has diverged from local $MAIN_BRANCH; the push below may fail"
  fi
  if ! git merge -q --no-ff "$WORK_BRANCH" -m "$1"; then
    git merge --abort 2>/dev/null
    git checkout -q "$WORK_BRANCH"
    log "ERROR promote merge conflict"
    return 1
  fi
  git tag -a "$2" -m "$1" || log "WARN tag $2 not created"
  push_branch "$MAIN_BRANCH"
  git checkout -q "$WORK_BRANCH"
}

# Discard an iteration's code but keep what it learned and what the harness recorded:
# agent notes (notes/, STATUS.md, backlog.md), bench outputs (results/), this
# iteration's review, and the harness's iterations.tsv row. Everything else,
# including any tampering with protected files, goes back to start_sha.
#
# With mode "strict" (integrity violations: agent code wrote where it must not, possibly
# holdout data) nothing the iteration produced is kept, and ignored files are cleaned too.
discard_iteration() {  # discard_iteration start_sha reason iter [strict]
  cd "$ROOT" || return 1
  local keep review strict="${4:-}"
  keep="$(mktemp -d)"
  review="status/reviews/iter-$3.json"
  if [[ "$strict" != strict ]]; then
    cp -r notes results "$keep/" 2>/dev/null || true
    cp status/STATUS.md status/backlog.md "$keep/" 2>/dev/null || true
  fi
  [[ -f "$review" ]] && cp "$review" "$keep/review.json"
  git diff "$1" HEAD > "$LOGS/rejected-iter-$3.patch" 2>/dev/null || true
  git reset -q --hard "$1"
  if [[ "$strict" == strict ]]; then
    git clean -qfdx -e data -e .venv -e logs -e harness/loop.env -e miner.env
  else
    git clean -qfd
    rm -rf notes results
    cp -r "$keep/notes" notes
    cp -r "$keep/results" results 2>/dev/null || mkdir -p results
    cp "$keep/STATUS.md" "$keep/backlog.md" status/ 2>/dev/null || true
  fi
  [[ -f "$keep/review.json" ]] && cp "$keep/review.json" "$review"
  if [[ -f "$LOGS/.last_iter_row" ]]; then
    cat "$LOGS/.last_iter_row" >> "$STATUS/iterations.tsv"
    rm -f "$LOGS/.last_iter_row"
  fi
  rm -rf "$keep"
  commit_all "loop: iter $3 discarded ($2); notes kept"
}

_matches_any() {  # _matches_any path listfile
  local pattern
  while read -r pattern; do
    [[ -z "$pattern" || "$pattern" == \#* ]] && continue
    # shellcheck disable=SC2053
    [[ "$1" == $pattern ]] && return 0
  done < "$2"
  return 1
}

# Paths an iteration changed that the agent may not change: anything outside
# harness/agent_paths.txt (allowlist), or inside it but listed in
# harness/protected_paths.txt. --no-renames reports both sides of a rename.
path_violations() {  # path_violations base head  -> prints offending paths
  git -C "$ROOT" diff --no-renames --name-only "$1" "$2" | while read -r path; do
    if ! _matches_any "$path" "$HARNESS/agent_paths.txt" || _matches_any "$path" "$HARNESS/protected_paths.txt"; then
      echo "$path"
    fi
  done
}

ledger() {  # ledger iter mode commit dev holdout decision reason
  local file="$STATUS/results.tsv"
  [[ -f "$file" ]] || printf 'ts\titer\tmode\tcommit\tdev_score\tholdout_score\tdecision\treason\n' > "$file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$@" >> "$file"
}

# A harness bench, in two steps the harness runs itself:
#  1. attack phase (agent code): `run_bench --attack-only` in its own session, killed at
#     n x (budget + slack) + model-load allowance of wall clock, measured here;
#  2. integrity check, then the scorer in a fresh `python -I` (no PYTHONPATH, no user site,
#     private pycache) that imports nothing from miner26/.
# Prints mean_official. Returns 1 on failure or overrun, 2 on an integrity violation.
bench_score() {  # bench_score manifest out_json since_epoch
  local manifest="$1" out="$2" since="$3" work n allowed started wall problems
  work="$(mktemp -d)"
  [[ -n "$BENCH_RUN_PREFIX" ]] && chmod 777 "$work"
  n="$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1], encoding='utf-8'))['items']))" "$manifest")" || return 1
  allowed="$("$PY" -c "print(int($n * ($BENCH_BUDGET + $TIME_SLACK) + 90))")"
  started=$(date +%s)
  # shellcheck disable=SC2086
  if ! run_isolated "$allowed" /dev/null "$LOGS/bench-attack.log" "$LOGS/bench-attack.err" \
      $BENCH_RUN_PREFIX "$PY" -m bench.run_bench --attack-only "$work" --manifest "$manifest" --budget "$BENCH_BUDGET"; then
    log "bench attack phase failed or was killed at ${allowed}s ($manifest)"; return 1
  fi
  wall=$(($(date +%s) - started))
  if ((wall > allowed)); then log "bench attack phase overran: ${wall}s > ${allowed}s ($manifest)"; return 1; fi
  problems="$(integrity_check "$since")"
  if [[ -n "$problems" ]]; then log "integrity violation after attack phase: $(echo "$problems" | tr '\n' ' ')"; return 2; fi
  "$PY" -I -X pycache_prefix="$work/pyc" -c \
    'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("bench.score", run_name="__main__", alter_sys=True)' \
    "$ROOT" --manifest "$manifest" --adv-dir "$work/adv" --records "$work/records.json" \
    --meta "$(cat "$work/meta.json" 2>/dev/null || echo '{}')" --out "$out" >> "$LOGS/bench.log" 2>&1 || return 1
  if float_gt "$(json_get "$out" 'd["summary"]["time_max"]')" "$(calc "$BENCH_BUDGET + $TIME_SLACK")"; then
    log "bench time_max above budget + slack ($manifest)"; return 1
  fi
  json_get "$out" 'd["summary"]["mean_official"]'
}

# Seconds to wait if the agent session ended on a claude usage limit ("You've hit your session
# limit · resets 7:30pm (UTC)"), else nothing. Such a session did no work, so it must not use up
# an iteration or a review: iterations 7-200 of the first run were all limit no-ops.
usage_limit_wait() {  # usage_limit_wait jsonl
  "$PY" - "$1" <<'EOF'
import datetime, json, re, sys
text = ""
try:
    for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
        if line.startswith("{"):
            event = json.loads(line)
            if event.get("type") == "result":
                text = str(event.get("result", ""))
except Exception:
    pass
if not re.search(r"hit your .*limit", text, re.I):
    sys.exit(0)
wait = 1800
m = re.search(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)(?:\s*\(([^)]+)\))?", text, re.I)
if m:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(m.group(4) or "UTC")
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        now = datetime.datetime.now(tz)
        reset = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
        if reset <= now:
            reset += datetime.timedelta(days=1)
        wait = int((reset - now).total_seconds())
    except Exception:
        pass
print(min(wait, 6 * 3600) + 120)
EOF
}
