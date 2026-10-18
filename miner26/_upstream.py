"""Make the pinned upstream Perturb checkout importable (``perturbnet``, ``neurons``).

Everything that must match the validator bit-for-bit (preprocessing, label
mapping, scoring constants, the verifier itself) is imported from
``upstream/`` instead of being re-implemented here, so a submodule bump is the
only way scoring rules change.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "upstream"

if not (UPSTREAM_ROOT / "perturbnet").is_dir():
    raise ImportError(
        f"upstream Perturb checkout missing at {UPSTREAM_ROOT}; run `git submodule update --init`"
    )
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))


def ensure_bittensor_importable() -> bool:
    """Install a minimal ``bittensor`` stub when the real package is absent.

    ``neurons.validator`` / ``neurons.miner`` import bittensor at module load but
    the scoring and attack helpers never touch it. Benches and tests can then run
    on machines without bittensor. Returns True when the real package is used.
    """
    try:
        import bittensor  # noqa: F401

        return True
    except Exception:
        stub = types.ModuleType("bittensor")
        stub.config = object  # only referenced in annotations / build_config
        sys.modules["bittensor"] = stub
        return False
