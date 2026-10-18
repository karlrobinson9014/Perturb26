#!/usr/bin/env bash
# Independent review of one candidate by a fresh Claude CLI session (no shared context
# with the session that wrote the change). The reviewer re-runs the gates, runs the
# holdout bench itself and writes status/reviews/iter-N.json. It has veto power.
#
# usage: review.sh iter base_sha head_sha dev_result_json
# stdout (only): "<APPROVE|REJECT> <holdout_score or empty>"
set -uo pipefail
# shellcheck source=harness/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
cd "$ROOT" || exit 1

iter="$1"; base="$2"; head="$3"; dev_json="$4"
tag="$(printf '%04d' "$iter")"
verdict_file="status/reviews/iter-$iter.json"
holdout_json="$HOLDOUT_DIR/results/holdout-iter$iter.json"
mkdir -p "$HOLDOUT_DIR/results" "$STATUS/reviews"
rm -f "$verdict_file" "$holdout_json"

prompt() {
  cat "$ROOT/prompts/review.md"
  cat <<EOF

---
## Review context (written by the harness)
- iteration under review: $iter
- diff to review: \`git diff $base $head\` (base = last accepted state of the work branch)
- harness dev bench result (raw, per image): $dev_json
- accepted best so far (status/best.json):
\`\`\`json
$(cat "$STATUS/best.json" 2>/dev/null || echo '{}')
\`\`\`
- holdout manifest: $HOLDOUT_DIR/holdout.json
- holdout command (run it exactly once, it takes a while):
  \`$PY -m bench.run_bench --manifest $HOLDOUT_DIR/holdout.json --budget $BENCH_BUDGET --out $holdout_json\`
- write your verdict JSON to: $verdict_file
EOF
}

log "iter $iter: reviewer start (model=$REVIEW_MODEL)"
# shellcheck disable=SC2086
prompt | timeout --kill-after=60 "$REVIEW_TIMEOUT" "$CLAUDE_BIN" -p --model "$REVIEW_MODEL" --max-turns "$MAX_TURNS" \
  --output-format stream-json --verbose $CLAUDE_FLAGS > "$LOGS/review-$tag.jsonl" 2> "$LOGS/review-$tag.jsonl.err"
log "iter $iter: reviewer exit rc=$?"

# The reviewer may write only its verdict file. Anything else it changed is reverted
# and counts as a rejection.
tampered="$(git status --porcelain --untracked-files=all | grep -v " $verdict_file\$" | grep -v ' results/' || true)"
if [[ -n "$tampered" ]]; then
  log "iter $iter: reviewer modified the tree; reverting and rejecting: $tampered"
  saved="$(mktemp)"; [[ -f "$verdict_file" ]] && cp "$verdict_file" "$saved"
  git reset -q --hard HEAD && git clean -qfd -- miner26 bench tests tools
  [[ -s "$saved" ]] && cp "$saved" "$verdict_file"
  echo "REJECT "
  exit 0
fi

verdict="$(json_get "$verdict_file" 'd["verdict"]')"
claimed="$(json_get "$verdict_file" 'd["holdout_score"]')"
measured="$(json_get "$holdout_json" 'd["summary"]["mean_official"]')"
measured_sha="$(json_get "$holdout_json" 'd["summary"]["git_sha"]')"
dirty="$(json_get "$holdout_json" 'd["summary"]["git_dirty"]')"
expected_sha="$(git rev-parse --short "$head")"

# Trust the raw bench file, not the reviewer's summary of it.
if [[ "$verdict" == APPROVE ]]; then
  if [[ -z "$measured" || "$measured_sha" != "$expected_sha" || "$dirty" != False ]]; then
    log "iter $iter: APPROVE without a clean holdout run of $expected_sha (got sha=$measured_sha dirty=$dirty); rejecting"
    verdict=REJECT
  elif ! "$PY" -c "import sys; sys.exit(0 if abs(float('$claimed') - float('$measured')) < 1e-9 else 1)" 2>/dev/null; then
    log "iter $iter: reviewer holdout claim $claimed != measured $measured; rejecting"
    verdict=REJECT
  fi
fi
[[ "$verdict" == APPROVE ]] || verdict=REJECT
log "iter $iter: review verdict=$verdict holdout=$measured"
echo "$verdict $measured"
