#!/usr/bin/env bash
# Deterministic gates run by the harness (never by trust in the agent's report):
#   1. full pytest suite (contract tests are protected)
#   2. the production miner module imports
#   3. a 3-image smoke bench runs end to end and scores at least one image
set -euo pipefail
# shellcheck source=harness/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
cd "$ROOT"

echo "== pytest"
"$PY" -m pytest -q tests

echo "== import miner26.neuron"
"$PY" -c "import miner26.neuron"

echo "== smoke bench"
smoke="$(mktemp -d)/smoke.json"
"$PY" -m bench.run_bench --manifest "$DEV_MANIFEST" --limit 3 --budget "${GATE_SMOKE_BUDGET:-8}" --variants default --out "$smoke"
success="$(json_get "$smoke" 'd["summary"]["success_rate"]')"
echo "smoke success_rate=$success (must exceed ${GATE_MIN_SUCCESS:-0})"
float_gt "$success" "${GATE_MIN_SUCCESS:-0}"
echo "GATES PASS"
