#!/usr/bin/env bash
# Deterministic gates run by the harness (never by trust in the agent's report):
#   1. contract tests, alone and under the harness's pytest config (agent test config ignored)
#   2. the rest of the test suite
#   3. the production miner module imports
#   4. a 3-image smoke bench runs end to end and scores at least one image
set -euo pipefail
# shellcheck source=harness/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
cd "$ROOT"
PYTEST=("$PY" -m pytest -q -c "$HARNESS/pytest.ini" --rootdir "$ROOT" --confcutdir "$ROOT")

echo "== contract tests"
"${PYTEST[@]}" tests/contract

echo "== other tests"
others=$(find tests -name 'test_*.py' -not -path 'tests/contract/*' | sort)
if [[ -n "$others" ]]; then
  # shellcheck disable=SC2086
  "${PYTEST[@]}" $others
fi

echo "== import miner26.neuron"
"$PY" -c "import miner26.neuron"

echo "== smoke bench"
smoke="$(mktemp -d)/smoke.json"
"$PY" -m bench.run_bench --manifest "$DEV_MANIFEST" --limit 3 --budget "${GATE_SMOKE_BUDGET:-8}" --variants default --out "$smoke"
success="$(json_get "$smoke" 'd["summary"]["success_rate"]')"
echo "smoke success_rate=$success (must exceed ${GATE_MIN_SUCCESS:-0})"
float_gt "$success" "${GATE_MIN_SUCCESS:-0}"
echo "GATES PASS"
