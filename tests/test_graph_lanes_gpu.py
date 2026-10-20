"""v1 regression: the CUDA-graph lanes compute what the eager lanes compute, on EfficientNetV2-L and a real dev image.

Skipped without CUDA, the cached EfficientNetV2-L weights or the dev data (``bench.fetch_data``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "bench" / "manifests" / "dev.json"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _weights_cached() -> bool:
    from torchvision.models import EfficientNet_V2_L_Weights

    name = Path(EfficientNet_V2_L_Weights.IMAGENET1K_V1.url).name
    return (Path(torch.hub.get_dir()) / "checkpoints" / name).exists()


def _dev_images() -> list[Path]:
    if not MANIFEST.exists():
        return []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data_dir = (MANIFEST.parent / manifest["data_dir"]).resolve()
    return [p for p in (data_dir / item["file"] for item in manifest["items"]) if p.exists()]


@pytest.fixture(scope="module")
def setup():
    if not _weights_cached():
        pytest.skip("EfficientNetV2-L weights not cached")
    images = _dev_images()
    if not images:
        pytest.skip("dev data missing (python -m bench.fetch_data)")
    from miner26.attack import AttackConfig, Attacker, _GraphLanes, _Lanes
    from miner26.imaging import b64_to_u8, task_b64_from_bytes
    from perturbnet.model import PREPROCESS, load_efficientnet_v2_l

    device = torch.device("cuda")
    attacker = Attacker(load_efficientnet_v2_l(device), device, AttackConfig())
    graph = attacker._crop_lanes
    assert isinstance(graph, _GraphLanes), "graph capture failed on the GPU; the attack would run eagerly"
    for path in images:
        clean = b64_to_u8(task_b64_from_bytes(path.read_bytes())).to(device).float()
        offset = attacker.crop_offset(tuple(clean.shape))
        if offset is not None:
            break
    else:
        pytest.skip("no dev image takes the crop path")
    attacker._x0.copy_(PREPROCESS((clean / 255.0).unsqueeze(0)))
    eager = _Lanes(attacker.model, device, attacker.B, attacker.cfg.search_amp, graph.to_input)

    oy, ox = offset
    c = attacker.crop
    window = clean[:, oy:oy + c, ox:ox + c]
    lo, hi = (-window).clamp(min=-1.0), (255.0 - window).clamp(max=1.0)
    with torch.no_grad():
        clean_cls = int(attacker.model(attacker._x0).argmax())
    other_cls = (clean_cls + 1) % 1000

    # Lane state as in a run after the re-target: modes from the lane spec, last lane targeted.
    for lanes in (graph, eager):
        lanes.t.fill_(clean_cls)
        for b, (mode, tau) in enumerate(attacker.lane_spec):
            lanes.mode[b], lanes.tau[b], lanes.j[b] = mode, tau, -1
        lanes.j[attacker.B - 1] = other_cls

    # Sparse +-1 perturbations of different densities per lane (lane 0 clean), clipped to the uint8 range.
    gen = torch.Generator(device="cpu").manual_seed(26)
    density = torch.linspace(0.0, 0.05, attacker.B).view(-1, 1, 1, 1)
    u = torch.rand((attacker.B, *window.shape), generator=gen)
    sign = torch.where(torch.rand(u.shape, generator=gen) < 0.5, -1.0, 1.0)
    q = torch.where(u < density, sign, torch.zeros_like(u)).to(device)
    q = torch.maximum(torch.minimum(q, hi), lo)
    return graph, eager, q


def test_graph_check_matches_eager(setup) -> None:
    graph, eager, q = setup
    m_e, a_e = eager.check(q)
    m_g, a_g = (x.clone() for x in graph.check(q))
    assert torch.equal(a_g, a_e)
    assert float((m_g - m_e).abs().max()) < 1e-3, (m_g.tolist(), m_e.tolist())


def test_graph_grad_matches_eager(setup) -> None:
    graph, eager, q = setup
    m_e, g_e = eager.grad(q)
    m_g, g_g = (x.clone() for x in graph.grad(q))
    # Bit-identical on the 4090 (2026-10-19); the slack covers cudnn picking other fp16 algorithms at capture.
    assert float((m_g - m_e).abs().max()) < 0.05, (m_g.tolist(), m_e.tolist())
    cos = torch.nn.functional.cosine_similarity(g_g.flatten(1), g_e.flatten(1), dim=1)
    assert float(cos.min()) > 0.999, cos.tolist()
    rel = (g_g - g_e).flatten(1).norm(dim=1) / g_e.flatten(1).norm(dim=1)
    assert float(rel.max()) < 0.05, rel.tolist()


def test_graph_replay_is_stateless(setup) -> None:
    # Replays on new inputs must not depend on what was replayed before (static buffers are reused).
    graph, _, q = setup
    first = graph.check(q)[0].clone()
    graph.check(torch.zeros_like(q))
    graph.grad(-q)
    assert torch.equal(graph.check(q)[0], first)
