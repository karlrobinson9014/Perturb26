#!/usr/bin/env bash
# Independent review of everything on the work branch not yet on main, by a fresh
# Claude CLI session with no shared context with the sessions that wrote it. The
# harness has already run the dev and holdout benches; the reviewer re-runs the gates
# itself, reproduces part of the dev bench, audits the diff, and has veto power.
#
# usage: review.sh iter base_sha head_sha dev_result_json holdout_result_json
# stdout (only): APPROVE or REJECT
set -uo pipefail
# shellcheck source=harness/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
cd "$ROOT" || exit 1

iter="$1"; base="$2"; head="$3"; dev_json="$4"; hold_json="$5"
tag="$(printf '%04d' "$iter")"
verdict_file="status/reviews/iter-$iter.json"
rm -f "$verdict_file"

holdout_digest() {  # aggregate + per-image scores, no image identities or paths
  "$PY" - "$hold_json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(d["summary"], indent=1, sort_keys=True))
print("per-image official (sorted):", sorted(round(r["official"], 5) for r in d["records"]))
print("per-image reasons:", sorted({tuple(r["reasons"]) for r in d["records"]}))
EOF
}

prompt() {
  cat "$ROOT/prompts/review.md"
  cat <<EOF

---
## Review context (written by the harness)
- iteration that triggered the review: $iter
- unreviewed changes: \`git diff $base $head\` (base = last commit already promoted to main;
  this may span several kept iterations; review all of it) and \`git log --oneline $base..$head\`
- harness dev bench result (raw, per image): $dev_json
- accepted best so far (status/best.json; its dev_result is the comparison file):
\`\`\`json
$(cat "$STATUS/best.json" 2>/dev/null || echo '{}')
\`\`\`
- harness holdout result for this candidate (you have no access to holdout images):
\`\`\`
$(holdout_digest)
\`\`\`
- write your verdict JSON to: $verdict_file
EOF
}

prompt_file="$(mktemp)"
prompt > "$prompt_file"
# A session cut off by a claude usage limit without a verdict is retried after the reset
# rather than counted as a rejection.
for attempt in 1 2 3; do
  log "iter $iter: reviewer start (model=$REVIEW_MODEL base=${base:0:8} attempt=$attempt)"
  # shellcheck disable=SC2086
  run_isolated "$REVIEW_TIMEOUT" "$prompt_file" "$LOGS/review-$tag.jsonl" "$LOGS/review-$tag.jsonl.err" \
    $AGENT_RUN_PREFIX "$CLAUDE_BIN" -p --model "$REVIEW_MODEL" --max-turns "$MAX_TURNS" \
    --output-format stream-json --verbose $CLAUDE_FLAGS --disallowedTools "$CLAUDE_DENY_TOOLS"
  log "iter $iter: reviewer exit rc=$?"
  limit_wait="$(usage_limit_wait "$LOGS/review-$tag.jsonl")"
  [[ -n "$limit_wait" && ! -s "$verdict_file" && "$attempt" -lt 3 ]] || break
  git reset -q --hard HEAD && git clean -qfd  # whatever the cut-off session left
  rm -f "$verdict_file"
  log "iter $iter: reviewer hit the claude usage limit; retrying in ${limit_wait}s"
  sleep "$limit_wait"
done
rm -f "$prompt_file"

# The reviewer may write only its verdict file. Anything else it changed is reverted
# and counts as a rejection.
tampered="$(git status --porcelain --untracked-files=all | grep -vF " $verdict_file" || true)"
if [[ -n "$tampered" ]]; then
  log "iter $iter: reviewer modified the tree; reverting and rejecting: $(echo "$tampered" | tr '\n' ' ')"
  saved="$(mktemp)"; [[ -f "$verdict_file" ]] && cp "$verdict_file" "$saved"
  git reset -q --hard HEAD && git clean -qfd
  [[ -s "$saved" ]] && cp "$saved" "$verdict_file"
  echo REJECT
  exit 0
fi

verdict="$(json_get "$verdict_file" 'd["verdict"]')"
[[ "$verdict" == APPROVE ]] || verdict=REJECT
log "iter $iter: review verdict=$verdict"
echo "$verdict"
