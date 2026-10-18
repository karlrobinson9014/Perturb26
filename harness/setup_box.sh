#!/usr/bin/env bash
# One-time setup on the GPU box (Linux, CUDA). Safe to re-run.
#   bash harness/setup_box.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== submodule"
git submodule update --init

echo "== python venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
[[ -d .venv ]] || "$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r upstream/requirements.txt -r requirements.txt

echo "== CUDA"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.get_device_name(0))"

echo "== model weights (EfficientNetV2-L)"
.venv/bin/python -c "import miner26; from perturbnet.model import load_efficientnet_v2_l; import torch; load_efficientnet_v2_l(torch.device('cpu'))"

echo "== pm2 + claude CLI"
command -v pm2 >/dev/null || npm install -g pm2
command -v claude >/dev/null || npm install -g @anthropic-ai/claude-code
claude --version
echo "   headless auth: run 'claude setup-token' once (or export ANTHROPIC_API_KEY in harness/loop.env)"

echo "== loop config"
[[ -f harness/loop.env ]] || cp harness/loop.env.example harness/loop.env
set -a; source harness/loop.env; set +a
HOLDOUT_DIR="${HOLDOUT_DIR:-$HOME/perturb26-holdout}"

echo "== benchmark data"
PYTHONPATH="$ROOT" .venv/bin/python -m bench.fetch_data --manifest bench/manifests/dev.json --with-adv
mkdir -p "$HOLDOUT_DIR"
# Holdout lives outside the repo so loop agents never see it; only the reviewer is pointed at it.
PYTHONPATH="$ROOT" .venv/bin/python -m bench.fetch_data --manifest "$HOLDOUT_DIR/holdout.json" --split test --count 128 --seed 2626

echo "== gates"
bash harness/gate.sh

echo "== git identity / push access"
git config user.name >/dev/null || echo "   set git user.name / user.email"
git ls-remote --heads origin >/dev/null && echo "   remote reachable" || echo "   WARNING: cannot reach origin (configure a deploy key or token before PUSH=1)"
echo "setup complete. Start the loop: bash harness/ctl.sh start"
