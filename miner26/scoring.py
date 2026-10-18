"""Closed-form estimate of the validator score, used inside the attack loop.

The authoritative score is upstream's ``PerturbValidator.verify_and_score``
(see ``bench/official.py``). This module mirrors its formula cheaply so the
attack can rank candidates without running SSIM/PSNR on every step. The parity
test in ``tests/contract/test_scoring_parity.py`` keeps the two in sync.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from miner26 import _upstream  # noqa: F401
from perturbnet import constants as C
from perturbnet.model import label_for_index, normalize_prediction_label, resolve_target_index

QUANT = 1.0 / 255.0


@dataclass(frozen=True)
class ScoreParams:
    min_linf: float = C.MIN_LINF_DELTA
    max_linf: float = C.MAX_LINF_DELTA
    linf_weight: float = C.LINF_COMPONENT_WEIGHT
    rmse_weight: float = C.RMSE_COMPONENT_WEIGHT
    perturbation_weight: float = C.PERTURBATION_WEIGHT
    margin_weight: float = C.ANALYZE_BUCKET_MARGIN_WEIGHT
    novelty_weight: float = C.ANALYZE_BUCKET_NOVELTY_WEIGHT
    novelty_target_pixels: int = C.ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS
    min_ssim: float = C.MIN_SSIM
    min_psnr_db: float = C.MIN_PSNR_DB


DEFAULT_PARAMS = ScoreParams()


def validator_true_index(clean_pred_index: int) -> int | None:
    """The class index the validator compares against.

    The validator stores the clean prediction as a *label string* and maps it
    back through ``resolve_target_index``. Duplicate label names in the
    torchvision category list make that round trip lossy, so this can differ
    from ``clean_pred_index``; the attack must target what the validator uses.
    """
    label = normalize_prediction_label(label_for_index(int(clean_pred_index)))
    return resolve_target_index(label)


def estimate_score(
    *,
    linf: float,
    rmse: float,
    margin: float,
    changed_pixels: int,
    flipped: bool,
    params: ScoreParams = DEFAULT_PARAMS,
) -> float:
    """Validator formula for a candidate that already passes SSIM/PSNR.

    ``linf``/``rmse`` are in [0,1] pixel units on the uint8 grid; ``margin`` is
    best non-true logit minus true logit.
    """
    if not flipped:
        return 0.0
    # Validator: min(challenge.epsilon, max_linf_delta); challenge.epsilon is max_linf_delta.
    effective_max = params.max_linf
    if linf < params.min_linf or linf > effective_max:
        return 0.0
    linf_ratio = min(max((linf - params.min_linf) / max(1e-12, effective_max - params.min_linf), 0.0), 1.0)
    rmse_ratio = min(max(rmse / max(1e-12, effective_max), 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_score = (1.0 - rmse_ratio) ** 2
    total = max(1e-12, params.linf_weight + params.rmse_weight)
    perturbation = (params.linf_weight * linf_score + params.rmse_weight * rmse_score) / total
    margin_score = min(max(margin / 10.0, 0.0), 1.0)
    novelty_score = min(max(changed_pixels / max(1, params.novelty_target_pixels), 0.0), 1.0)
    return (
        params.perturbation_weight * perturbation
        + params.margin_weight * margin_score
        + params.novelty_weight * novelty_score
    )


def rmse_for_unit_changes(nonzero_values: int, total_values: int) -> float:
    """RMSE of a perturbation whose nonzero entries are all exactly +-1/255."""
    if total_values <= 0:
        return 0.0
    return QUANT * math.sqrt(nonzero_values / total_values)


def theoretical_max(params: ScoreParams = DEFAULT_PARAMS) -> float:
    """Upper bound: L-inf = 1/255, RMSE -> 0, margin >= 10, novelty saturated."""
    return estimate_score(
        linf=QUANT, rmse=0.0, margin=10.0, changed_pixels=params.novelty_target_pixels, flipped=True, params=params
    )
