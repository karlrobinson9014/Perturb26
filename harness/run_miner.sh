#!/usr/bin/env bash
# Live miner launcher (pm2 app perturb26-miner). Runs whatever is checked out, so
# deploy by checking out an accepted tag first:  git checkout best-iterN
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f miner.env ]] || { echo "missing miner.env (cp miner.env.example miner.env)"; exit 1; }
set -a; source miner.env; set +a
: "${WALLET_NAME:?}" "${WALLET_HOTKEY:?}" "${NETUID:?}"
export PYTHONPATH="$ROOT"
exec .venv/bin/python -m miner26.neuron \
  --netuid "$NETUID" --network "${NETWORK:-finney}" \
  --wallet.name "$WALLET_NAME" --wallet.hotkey "$WALLET_HOTKEY" \
  --log-level "${LOG_LEVEL:-INFO}" ${MINER_EXTRA_ARGS:-}
