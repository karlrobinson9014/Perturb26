"""PROTECTED: import path setup for the scorer, independent of agent-editable code.

The scorer process (bench/score.py) must not import anything from miner26/:
code there is written by loop agents and could patch the verifier. This module
puts the pinned upstream checkout on sys.path and stubs bittensor when absent.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "upstream"

if not (UPSTREAM_ROOT / "perturbnet").is_dir():
    raise ImportError(f"upstream Perturb checkout missing at {UPSTREAM_ROOT}; run `git submodule update --init`")
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

try:
    import bittensor  # noqa: F401
except Exception:
    _stub = types.ModuleType("bittensor")
    _stub.config = object  # only referenced in annotations / build_config
    sys.modules["bittensor"] = _stub
