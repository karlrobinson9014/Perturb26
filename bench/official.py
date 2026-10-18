"""Authoritative scorer: upstream ``PerturbValidator.verify_and_score`` on a stub.

PROTECTED FILE - the loop agent must not edit it (harness rejects the iteration).

The method only reads ``self.config.perturb``, ``self.model`` and
``self.device``, so it runs unchanged on a namespace that carries those three.
Each response is scored under several numeric variants because validators run
on different GPUs: an attack that only flips under one numeric path is not a
reliable flip. The official per-image score is the *minimum* over variants.
"""
from __future__ import annotations

import base64
import contextlib
import io
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Iterator

import torch
from PIL import Image

from bench import _paths  # noqa: F401  (upstream on sys.path; never import miner26 here)
from neurons.validator import ChallengeSpec, PerturbValidator  # noqa: E402
from perturbnet import constants as C  # noqa: E402
from perturbnet.image_io import decode_image_b64  # noqa: E402
from perturbnet.model import normalize_prediction_label, predict_label  # noqa: E402

VARIANTS = ("default", "tf32_off", "tf32_on")


def clean_task_b64(raw: bytes) -> str:
    """The clean task image as the validator holds it: upstream ``image_url_to_b64`` after the GET."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@contextlib.contextmanager
def numeric_variant(name: str) -> Iterator[None]:
    """Temporarily switch TF32 settings; 'default' leaves PyTorch defaults untouched."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        if name == "tf32_off":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif name == "tf32_on":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif name != "default":
            raise ValueError(f"unknown numeric variant {name!r}")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class OfficialScorer:
    def __init__(self, model: torch.nn.Module, device: torch.device, variants: tuple[str, ...] = VARIANTS) -> None:
        self.device = device
        # TF32 switches are no-ops on CPU; scoring the same path three times is waste.
        self.variants = variants if device.type == "cuda" else ("default",)
        self._stub = SimpleNamespace(
            config=SimpleNamespace(perturb=SimpleNamespace(**C.VALIDATOR_CONFIG)),
            model=model,
            device=device,
        )

    def challenge(self, task_id: str, clean_b64: str) -> ChallengeSpec:
        """Mirror of ``PerturbValidator.challenge_from_task_api`` minus the download."""
        image = decode_image_b64(clean_b64).to(self.device)
        predicted_label = normalize_prediction_label(predict_label(self._stub.model, image))
        return ChallengeSpec(
            task_id=task_id,
            image_id=task_id,
            model_name=C.MODEL_NAME,
            clean_image_b64=clean_b64,
            true_label=predicted_label,
            epsilon=float(self._stub.config.perturb.max_linf_delta),
            norm_type="Linf",
        )

    def score(self, task_id: str, clean_b64: str, adv_b64: str) -> dict[str, Any]:
        """{'official': min score, 'variants': {name: EvaluationResult as dict}}."""
        per_variant: dict[str, dict[str, Any]] = {}
        for name in self.variants:
            with numeric_variant(name):
                challenge = self.challenge(task_id, clean_b64)
                result = PerturbValidator.verify_and_score(self._stub, challenge, adv_b64)
            per_variant[name] = asdict(result) | {"true_label": challenge.true_label}
        scores = [v["score"] for v in per_variant.values()]
        reasons = sorted({v["reason"] for v in per_variant.values()})
        return {
            "official": float(min(scores)),
            "max_variant": float(max(scores)),
            "variant_disagreement": len({s > 0.0 for s in scores}) > 1,
            "reasons": reasons,
            "variants": per_variant,
        }
