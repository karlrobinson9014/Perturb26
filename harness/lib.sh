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
  MAX_TURNS="${MAX_TURNS:-150}"
  ITER_TIMEOUT="${ITER_TIMEOUT:-5400}"
  REVIEW_TIMEOUT="${REVIEW_TIMEOUT:-5400}"
  MAX_ITERS="${MAX_ITERS:-200}"
  MAX_COST_USD="${MAX_COST_USD:-0}"
  ITER_SLEEP="${ITER_SLEEP:-30}"
  LOOP_MODE="${LOOP_MODE:-auto}"
  BENCH_BUDGET="${BENCH_BUDGET:-22}"
  MIN_GAIN="${MIN_GAIN:-0.002}"
  HOLDOUT_TOL="${HOLDOUT_TOL:-0.003}"
  TARGET_SCORE="${TARGET_SCORE:-0.993}"
  HOLDOUT_DIR="${HOLDOUT_DIR:-$HOME/perturb26-holdout}"
  DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/bench/manifests/dev.json}"
  PY="${PY:-$ROOT/.venv/bin/python}"
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
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
  git tag -a "$2" -m "$1"
  push_branch "$MAIN_BRANCH"
  git checkout -q "$WORK_BRANCH"
}

# Discard an iteration's code but keep what it learned and what the harness recorded:
# agent notes (notes/, STATUS.md, backlog.md), bench outputs (results/), this
# iteration's review, and the harness's iterations.tsv row. Everything else,
# including any tampering with protected files, goes back to start_sha.
discard_iteration() {  # discard_iteration start_sha reason iter
  cd "$ROOT" || return 1
  local keep review
  keep="$(mktemp -d)"
  review="status/reviews/iter-$3.json"
  cp -r notes results "$keep/" 2>/dev/null || true
  cp status/STATUS.md status/backlog.md "$keep/" 2>/dev/null || true
  [[ -f "$review" ]] && cp "$review" "$keep/review.json"
  git diff "$1" HEAD > "$LOGS/rejected-iter-$3.patch" 2>/dev/null || true
  git reset -q --hard "$1"
  rm -rf notes results
  cp -r "$keep/notes" notes
  cp -r "$keep/results" results 2>/dev/null || mkdir -p results
  cp "$keep/STATUS.md" "$keep/backlog.md" status/ 2>/dev/null || true
  [[ -f "$keep/review.json" ]] && cp "$keep/review.json" "$review"
  if [[ -f "$LOGS/.last_iter_row" ]]; then
    cat "$LOGS/.last_iter_row" >> "$STATUS/iterations.tsv"
    rm -f "$LOGS/.last_iter_row"
  fi
  rm -rf "$keep"
  commit_all "loop: iter $3 discarded ($2); notes kept"
}

# Paths the loop agent must never change (one glob per line).
protected_violations() {  # protected_violations base head  -> prints offending paths
  local base="$1" head="$2"
  git -C "$ROOT" diff --name-only "$base" "$head" | while read -r path; do
    while read -r pattern; do
      [[ -z "$pattern" || "$pattern" == \#* ]] && continue
      # shellcheck disable=SC2053
      if [[ "$path" == $pattern ]]; then echo "$path"; break; fi
    done < "$HARNESS/protected_paths.txt"
  done
}

ledger() {  # ledger iter mode commit dev holdout decision reason
  local file="$STATUS/results.tsv"
  [[ -f "$file" ]] || printf 'ts\titer\tmode\tcommit\tdev_score\tholdout_score\tdecision\treason\n' > "$file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$@" >> "$file"
}

bench_score() {  # bench_score manifest out_json -> prints mean_official (or empty on failure)
  "$PY" -m bench.run_bench --manifest "$1" --budget "$BENCH_BUDGET" --out "$2" >> "$LOGS/bench.log" 2>&1 || return 1
  json_get "$2" 'd["summary"]["mean_official"]'
}
