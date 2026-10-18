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
  MIN_GAIN="${MIN_GAIN:-0.002}"
  HOLDOUT_TOL="${HOLDOUT_TOL:-0.003}"
  TARGET_SCORE="${TARGET_SCORE:-0.993}"
  HOLDOUT_DIR="${HOLDOUT_DIR:-$HOME/perturb26-holdout}"
  DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/bench/manifests/dev.json}"
  PY="${PY:-$ROOT/.venv/bin/python}"
  BENCH_RUN_PREFIX="${BENCH_RUN_PREFIX:-}"
  export PYTHONPATH="$ROOT"
  # loop.env is sourced with set -a; the holdout location must not reach agent environments.
  export -n HOLDOUT_DIR 2>/dev/null || true
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
  git checkout -q "$MAIN_BRANCH"
  if has_remote_branch "$MAIN_BRANCH"; then git pull -q --ff-only "$REMOTE" "$MAIN_BRANCH" || true; fi
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
