"""Sparse, quantized L-inf = 1/255 attack (v1: batched multi-objective ISTA).

Why this shape (see notes/scoring.md): the validator scores
``0.7*linf_score + 0.3*rmse_score + 0.03*margin + 0.01*novelty``. On the uint8
grid the cheapest admissible L-inf is exactly 1/255, so every perturbed value
is in {-1, 0, +1}/255 and the remaining levers are *how few* values change
(RMSE) and reaching a 10-logit margin (where the margin term saturates).

Pipeline, all inside one wall-clock deadline (anytime: the best fp32-verified
candidate so far is always returned):

1. novelty seed - 8 pixels the model cannot see (outside the 480 crop), seeded per
                  hotkey so two miners never submit byte-identical images.
2. lanes        - B searches run as one batch. Each is proximal gradient (ISTA) on a
                  continuous d in [-1, 1] levels, rounded with a straight-through
                  estimator, with an L1 weight that adapts to hold the margin at
                  ``target_margin``. Lanes differ in objective: untargeted margin,
                  -logit_true, or tau*logsumexp(others) - logit_true. The class a lane
                  ends up in matters more than anything else (notes/research_log.md,
                  2026-10-19: aimed at the right class the same search matches the
                  network's best responses), and objectives that only push the true
                  class down let far, cheap classes emerge.
3. switch       - after ``switch_fraction`` of the budget the weakest ``n_switch``
                  lanes restart as targeted searches on classes the others surfaced (their
                  argmax and top-k fp32 logits); ``extra_switches`` adds later rounds.

Speed: on the fast path (preprocessing is a pure center crop, true for every live task:
short side 480) the fp16 objective gradient and the fp32 margin check are CUDA-graph
captured once in ``__init__`` (~18 ms per lane-step instead of ~125 ms eager). Any other
geometry or device runs the same algorithm eagerly through PREPROCESS.

The returned image is re-checked through the validator's own path (PREPROCESS on the
uint8 image, fp32, no autocast) and its score estimate is computed from that.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from miner26 import _upstream  # noqa: F401
from miner26.scoring import DEFAULT_PARAMS, QUANT, ScoreParams, estimate_score, rmse_for_unit_changes, validator_true_index
from perturbnet.model import PREPROCESS


@dataclass
class AttackConfig:
    lanes: str = "0:1,2:1,1:1,2:0.3"  # mode:tau per lane; 0 margin, 1 -logit_true, 2 tau*LSE(others)-logit_true
    lr: float = 0.01  # step on d (levels) per unit of mean-normalised gradient
    lam0: float = 1.0  # L1 weight when a lane first holds target_margin
    lam_up: float = 1.05  # multiply L1 weight while the lane's margin is above target
    lam_down: float = 0.97  # multiply L1 weight while it is below
    target_margin: float = 10.05  # margin the L1 schedule holds (score saturates at 10)
    kappa: float = 10.5  # stop pushing the objective once the margin is above this
    rw_power: float = 2.0  # !=0: per-element threshold * ((|d|+rw_eps)/(0.5+rw_eps))^rw_power (-1: reweighted L1)
    rw_eps: float = 0.3
    rw_step: float = 0.0  # >0: threshold * (rw_lo + (rw_hi-rw_lo)*sigmoid((|d|-0.5)/rw_step)) instead of the power profile
    rw_lo: float = 0.14
    rw_hi: float = 2.64
    momentum: float = 0.5  # >0: step along an EMA of the normalised gradient (heavy ball) instead of the raw gradient
    lr_final: float = 1.0  # <1: each lane's lr decays linearly to lr*lr_final over its remaining budget (MDCS, Tao 2026)
    extrap: float = 0.0  # >0: FISTA-style extrapolation d + extrap*(d - d_prev) before the gradient step
    safety_margin: float = 1.0  # fp32 margin a candidate needs before it is kept
    check_every: int = 8  # steps between batched fp32 checks of the lanes' best unverified candidates
    switch_fraction: float = 0.35  # share of the budget before weak lanes re-target
    n_switch: int = 2
    extra_switches: str = "0.6"  # further re-target rounds, comma-separated budget fractions (e.g. "0.6")
    n_switch_extra: int = 1  # lanes freed in each extra round
    retarget_warm: bool = True  # re-targeted lanes start from the leading lane's iterate instead of d = 0
    switch_topk: int = 3  # >0: rank target classes by every lane's top-k fp32 logits, not only its argmax
    novelty_pixels: int = DEFAULT_PARAMS.novelty_target_pixels
    search_amp: bool = True  # fp16 autocast for gradient steps on CUDA
    cuda_graphs: bool = True
    end_reserve: float = 0.3  # seconds kept for the final validator-path check
    seed: int = 0


@dataclass
class AttackResult:
    adv_u8: torch.Tensor  # uint8 CHW on CPU
    est_score: float
    margin: float
    flipped: bool
    nonzero: int
    changed_pixels: int
    clean_index: int
    true_index: int
    steps: int
    elapsed: float
    info: dict[str, Any] = field(default_factory=dict)


def parse_lanes(spec: str) -> list[tuple[int, float]]:
    lanes = []
    for part in spec.split(","):
        mode, tau = part.split(":")
        lanes.append((int(mode), float(tau)))
    return lanes


def _objective(o: torch.Tensor, t: torch.Tensor, j: torch.Tensor, mode: torch.Tensor, tau: torch.Tensor):
    """Per-lane objective to maximise and the validator margin (best other - true). Graph-safe: no host syncs."""
    B = o.shape[0]
    tt = t.view(1, 1).expand(B, 1)
    true = o.gather(1, tt).squeeze(1)
    others = o.scatter(1, tt, torch.full((B, 1), -1e4, device=o.device, dtype=o.dtype))
    margin = others.max(1).values - true
    targeted = o.gather(1, j.clamp(min=0).view(B, 1)).squeeze(1) - true
    lse = tau * torch.logsumexp(others / tau[:, None], 1) - true
    untargeted = torch.where(mode == 1, -true, torch.where(mode == 2, lse, margin))
    return torch.where(j >= 0, targeted, untargeted), margin


class _Lanes:
    """Lane state (true class, target, objective) and the two kernels, run eagerly."""

    def __init__(self, model: torch.nn.Module, device: torch.device, B: int, amp: bool,
                 to_input: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.model, self.B, self.to_input = model, B, to_input
        self.amp = amp and device.type == "cuda"
        self.t = torch.zeros((), dtype=torch.long, device=device)
        self.j = torch.full((B,), -1, dtype=torch.long, device=device)
        self.mode = torch.zeros(B, dtype=torch.long, device=device)
        self.tau = torch.ones(B, device=device)
        self.topk = None  # (B, TOPK) classes by fp32 logit from the last check

    TOPK = 6

    def _grad(self, q: torch.Tensor):
        qv = q.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            o = self.model(self.to_input(qv)).float()
        obj, margin = _objective(o, self.t, self.j, self.mode, self.tau)
        (g,) = torch.autograd.grad(obj.sum(), qv)
        return margin.detach(), g

    def _check(self, q: torch.Tensor):
        """fp32, no autocast."""
        with torch.no_grad():
            o = self.model(self.to_input(q)).float()
        _, margin = _objective(o, self.t, self.j, self.mode, self.tau)
        return margin, o.argmax(1), o.topk(self.TOPK, 1).indices

    def grad(self, q: torch.Tensor):
        return self._grad(q)

    def check(self, q: torch.Tensor):
        margin, arg, self.topk = self._check(q)
        return margin, arg


class _GraphLanes(_Lanes):
    """The same kernels captured as CUDA graphs on a fixed lane-batch shape."""

    def __init__(self, model, device, B, amp, to_input, shape: tuple[int, int, int]) -> None:
        super().__init__(model, device, B, amp, to_input)
        self.q_in = torch.zeros((B, *shape), device=device)
        benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True  # pick conv algorithms once, at capture; restored below
        try:
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._grad(self.q_in)
                    self._check(self.q_in)
            torch.cuda.current_stream(device).wait_stream(side)
            self.g_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_graph):
                self.g_out = self._grad(self.q_in)
            self.c_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.c_graph):
                self.c_out = self._check(self.q_in)
        finally:
            torch.backends.cudnn.benchmark = benchmark

    def grad(self, q: torch.Tensor):
        self.q_in.copy_(q)
        self.g_graph.replay()
        return self.g_out

    def check(self, q: torch.Tensor):
        self.q_in.copy_(q)
        self.c_graph.replay()
        self.topk = self.c_out[2]
        return self.c_out[:2]


@dataclass
class _Best:
    est: float
    margin: float
    q: torch.Tensor
    cls: int


class Attacker:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        cfg: AttackConfig | None = None,
        params: ScoreParams = DEFAULT_PARAMS,
    ) -> None:
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.cfg = cfg or AttackConfig()
        self.params = params
        self.lane_spec = parse_lanes(self.cfg.lanes)
        self.B = len(self.lane_spec)
        self._mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self._offset_cache: dict[tuple[int, int, int], tuple[int, int] | None] = {}
        self.crop = int(PREPROCESS.crop_size[0])
        mean = torch.tensor(PREPROCESS.mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(PREPROCESS.std, device=device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std
        self._scale = (1.0 / 255.0) / std  # one uint8 level in model-input units, per channel
        self._x0 = torch.zeros(1, 3, self.crop, self.crop, device=device)
        self._crop_lanes = self._make_crop_lanes()

    # ------------------------------------------------------------------ setup

    def _make_crop_lanes(self) -> _Lanes:
        def to_input(qv: torch.Tensor) -> torch.Tensor:
            return self._x0 + self._scale * qv

        shape = (3, self.crop, self.crop)
        if self.device.type == "cuda" and self.cfg.cuda_graphs:
            try:
                return _GraphLanes(self.model, self.device, self.B, self.cfg.search_amp, to_input, shape)
            except Exception:  # capture unsupported (driver, model): run eagerly
                pass
        return _Lanes(self.model, self.device, self.B, self.cfg.search_amp, to_input)

    def influence_mask(self, shape: tuple[int, int, int]) -> torch.Tensor:
        """Bool CHW mask of input values that reach the model after resize + center crop."""
        if shape not in self._mask_cache:
            x = torch.zeros((1, *shape), device=self.device, requires_grad=True)
            out = PREPROCESS(x)
            gen = torch.Generator(device="cpu").manual_seed(1234)
            probe = torch.randn(out.shape, generator=gen).to(self.device)
            (grad,) = torch.autograd.grad((out * probe).sum(), x)
            self._mask_cache[shape] = grad[0].abs() > 0
        return self._mask_cache[shape]

    def crop_offset(self, shape: tuple[int, int, int]) -> tuple[int, int] | None:
        """(top, left) when PREPROCESS on this shape is a pure center crop + normalise, else None.

        Offsets follow torchvision's center_crop (``int(round((H - crop) / 2))``, so odd
        margins round half to even); verified numerically, never assumed.
        """
        if shape not in self._offset_cache:
            _, H, W = shape
            c = self.crop
            result = None
            if min(H, W) >= c:
                oy, ox = int(round((H - c) / 2.0)), int(round((W - c) / 2.0))
                gen = torch.Generator(device="cpu").manual_seed(4321)
                x = torch.rand((1, *shape), generator=gen).to(self.device)
                ref = PREPROCESS(x)
                mine = (x[..., oy:oy + c, ox:ox + c] - self._mean) / self._std
                if ref.shape == mine.shape and float((ref - mine).abs().max()) < 1e-5:
                    result = (oy, ox)
            self._offset_cache[shape] = result
        return self._offset_cache[shape]

    def _novelty_seed(self, lo: torch.Tensor, hi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Change ``novelty_pixels`` pixels, preferring ones outside the model's view.

        Positions, channel and sign come from ``cfg.seed`` so that two miners running this
        code (different seeds) never submit byte-identical images, which the validator zeroes.
        """
        q = torch.zeros_like(lo)
        want = int(self.cfg.novelty_pixels)
        if want <= 0:
            return q
        gen = torch.Generator(device="cpu").manual_seed(int(self.cfg.seed))
        unseen = (~mask.any(dim=0)).cpu()  # HW
        positions = unseen.nonzero(as_tuple=False)
        positions = positions[torch.randperm(positions.shape[0], generator=gen)]
        if positions.shape[0] < want:
            seen = (~unseen).nonzero(as_tuple=False)
            positions = torch.cat([positions, seen[torch.randperm(seen.shape[0], generator=gen)]], dim=0)
        for y, x in positions[:want].tolist():
            c = int(torch.randint(0, lo.shape[0], (1,), generator=gen))
            up_first = bool(torch.randint(0, 2, (1,), generator=gen))
            for sign in ((1.0, -1.0) if up_first else (-1.0, 1.0)):
                if (sign > 0 and hi[c, y, x] > 0) or (sign < 0 and lo[c, y, x] < 0):
                    q[c, y, x] = sign
                    break
        return q

    def _validator_logits(self, image01: torch.Tensor) -> torch.Tensor:
        """The validator's numeric path: PREPROCESS on [0,1] CHW, fp32, no autocast."""
        with torch.no_grad():
            return self.model(PREPROCESS(image01.unsqueeze(0)))[0].float()

    def _estimate(self, nonzero: int, total: int, margin: float, changed_pixels: int) -> float:
        return estimate_score(
            linf=QUANT if nonzero else 0.0,
            rmse=rmse_for_unit_changes(nonzero, total),
            margin=margin,
            changed_pixels=changed_pixels,
            flipped=margin > 0.0,
            params=self.params,
        )

    # ------------------------------------------------------------------ main

    def run(self, clean_u8: torch.Tensor, deadline: float) -> AttackResult:
        t0 = time.time()
        cfg = self.cfg
        B = self.B
        clean = clean_u8.to(self.device).float()  # 0..255, CHW
        shape = tuple(clean.shape)
        total = clean.numel()
        clean_row = self._validator_logits(clean / 255.0)
        clean_index = int(clean_row.argmax().item())
        resolved = validator_true_index(clean_index)
        true_index = clean_index if resolved is None else int(resolved)

        lo_full = (-clean).clamp(min=-1.0)
        hi_full = (255.0 - clean).clamp(max=1.0)
        mask_full = self.influence_mask(shape) & ((hi_full - lo_full) > 0)
        seed_full = self._novelty_seed(lo_full, hi_full, mask_full)

        offset = self.crop_offset(shape)
        if offset is not None:
            oy, ox = offset
            c = self.crop
            window = (slice(None), slice(oy, oy + c), slice(ox, ox + c))
            self._x0.copy_(PREPROCESS((clean / 255.0).unsqueeze(0)))
            lanes = self._crop_lanes
            lo, hi = lo_full[window], hi_full[window]
            fixed = seed_full[window]
            var_mask = None
            outside = seed_full.clone()
            outside[window] = 0.0
        else:
            window = None

            def to_input(qv: torch.Tensor) -> torch.Tensor:
                return PREPROCESS((clean.unsqueeze(0) + qv) / 255.0)

            lanes = _Lanes(self.model, self.device, B, cfg.search_amp, to_input)
            lo, hi = lo_full, hi_full
            fixed = seed_full
            var_mask = mask_full.float()
            outside = torch.zeros_like(seed_full)
        n_out = int((outside != 0).sum().item())
        px_out = int((outside != 0).any(dim=0).sum().item())
        # Best estimate any candidate can reach: the seed values are always kept, so the fewest
        # changes are the seed itself (at least one value, for an admissible L-inf), at a saturated margin.
        seed_nz = max(1, int((seed_full != 0).sum().item()))
        seed_px = max(1, int((seed_full != 0).any(dim=0).sum().item()))
        ceiling = self._estimate(seed_nz, total, 10.0, seed_px)
        info: dict[str, Any] = {"path": "crop" if window is not None else "generic",
                                "alias_target": true_index != clean_index, "early_exit": False}

        lanes.t.fill_(true_index)
        for b, (mode, tau) in enumerate(self.lane_spec):
            lanes.mode[b] = mode
            lanes.tau[b] = tau
            lanes.j[b] = -1
        has_fixed = bool((fixed != 0).any())

        def quantize(d: torch.Tensor) -> torch.Tensor:
            q = torch.round(d)
            if has_fixed:
                q = torch.where(fixed != 0, fixed, q)
            return q

        def counts(q: torch.Tensor) -> tuple[list[int], list[int]]:
            nz = ((q != 0).flatten(1).sum(1) + n_out).tolist()
            px = ((q != 0).any(dim=1).flatten(1).sum(1) + px_out).tolist()
            return nz, px

        d = torch.zeros((B, *lo.shape), device=self.device)
        mom = torch.zeros_like(d) if cfg.momentum > 0 else None
        d_prev = torch.zeros_like(d) if cfg.extrap > 0 else None
        lam = torch.zeros(B, device=self.device)
        lane_t0 = [t0] * B  # when each lane (re)started, for the lr decay
        started = torch.zeros(B, dtype=torch.bool, device=self.device)
        best: list[_Best | None] = [None] * B
        fallback: _Best | None = None  # highest-margin flip below safety_margin
        # Best not-yet-verified candidate per lane (projected est from the fp16 margin, q, nonzero, pixels).
        # The fp32 check costs ~3/4 of a gradient step, so pending candidates are verified in one batch
        # every ``check_every`` steps instead of on every step that improves the projection.
        pending: list[tuple[float, torch.Tensor, int, int] | None] = [None] * B
        check_time = 0.0

        def flush() -> bool:
            """fp32-check all pending candidates in one batch; True when one reaches the ceiling."""
            nonlocal fallback, check_time
            idx = [b for b in range(B) if pending[b] is not None]
            if not idx:
                return False
            t_check = time.time()
            qs = torch.stack([pending[b][1] if pending[b] is not None else pending[idx[0]][1] for b in range(B)])
            m32, arg32 = lanes.check(qs)
            m32, arg32 = m32.tolist(), arg32.tolist()
            for b in idx:
                _, qb, nzb, pxb = pending[b]
                pending[b] = None
                if m32[b] <= 0.0:
                    continue
                est = self._estimate(nzb, total, m32[b], pxb)
                if m32[b] >= cfg.safety_margin:
                    if best[b] is None or est > best[b].est:
                        best[b] = _Best(est, m32[b], qb, arg32[b])
                elif fallback is None or m32[b] > fallback.margin:
                    fallback = _Best(est, m32[b], qb, arg32[b])
            check_time = time.time() - t_check
            return max((x.est for x in best if x is not None), default=0.0) >= ceiling - 1e-9

        fractions = [cfg.switch_fraction] + [float(x) for x in cfg.extra_switches.split(",") if x.strip()]
        switch_plan = [(t0 + f * max(0.0, deadline - t0), cfg.n_switch if i == 0 else cfg.n_switch_extra)
                       for i, f in enumerate(fractions)]
        switch_plan = [(at, n) for at, n in switch_plan if n > 0]
        tried: set[int] = set()  # classes some lane has been targeted at
        last_switched: list[int] = []
        info["switch"] = []
        steps = 0
        first_flip_step = -1
        step_time = 0.05
        while True:
            now = time.time()
            reserve = check_time if any(x is not None for x in pending) else 0.0
            if now + 1.5 * step_time + reserve + cfg.end_reserve >= deadline:
                break
            if switch_plan and now >= switch_plan[0][0]:
                n_free = switch_plan.pop(0)[1]
                if flush():
                    info["early_exit"] = True
                    break
                sw = self._retarget(lanes, best, d, lam, started, quantize, true_index, n_free, tried, last_switched)
                for b in sw["lanes"]:  # restarted lanes lose their history
                    lane_t0[b] = now
                    if mom is not None:
                        mom[b] = 0.0
                    if d_prev is not None:
                        d_prev[b] = 0.0
                last_switched = sw["lanes"]
                info["switch"].append(sw)
            q = quantize(d)
            margin, g = lanes.grad(q)
            steps += 1
            margins = margin.tolist()
            if first_flip_step < 0 and max(margins) > 0:
                first_flip_step = steps
            nz, px = counts(q)
            for b in range(B):
                if margins[b] <= 0:
                    continue
                projected = self._estimate(nz[b], total, margins[b], px[b])
                bar = max(best[b].est if best[b] is not None else 0.0,
                          pending[b][0] if pending[b] is not None else 0.0)
                if projected > bar + 1e-7:
                    pending[b] = (projected, q[b].clone(), nz[b], px[b])
            due = steps % max(1, cfg.check_every) == 0 or any(
                x is not None and x[0] >= ceiling - 1e-9 for x in pending)
            if due and flush():
                info["early_exit"] = True  # e.g. alias images: the seed alone already scores the ceiling
                break
            above = margin >= cfg.target_margin
            lam = torch.where(above & ~started, torch.full_like(lam, cfg.lam0), lam)
            lam = torch.where(above & started, lam * cfg.lam_up, lam)
            lam = torch.where(~above & started, lam * cfg.lam_down, lam)
            started = started | above
            gn = g / (g.abs().flatten(1).mean(1).view(B, *([1] * (g.dim() - 1))) + 1e-12)
            push = (margin < cfg.kappa).float().view(B, *([1] * (g.dim() - 1)))
            if cfg.lr_final < 1.0:
                lr_b = torch.tensor([cfg.lr * (1.0 - (1.0 - cfg.lr_final) * min(1.0, (now - s) / max(1e-6, deadline - s)))
                                     for s in lane_t0], device=self.device).view(B, *([1] * (g.dim() - 1)))
            else:
                lr_b = cfg.lr
            threshold = lr_b * lam.view(B, *([1] * (g.dim() - 1)))
            if cfg.rw_step > 0:  # smoothed step at the rounding point: values above 0.5 are the ones the score counts
                threshold = threshold * (cfg.rw_lo + (cfg.rw_hi - cfg.rw_lo) * torch.sigmoid((d.abs() - 0.5) / cfg.rw_step))
            elif cfg.rw_power != 0:  # weights from the pre-step iterate (<0: weak entries shrink faster, >0: saturated ones)
                threshold = threshold * ((d.abs() + cfg.rw_eps) / (0.5 + cfg.rw_eps)) ** cfg.rw_power
            if mom is not None:
                mom.mul_(cfg.momentum).add_(gn, alpha=1.0 - cfg.momentum)
                gn = mom / (mom.abs().flatten(1).mean(1).view(B, *([1] * (g.dim() - 1))) + 1e-12)
            if d_prev is not None:  # extrapolate from the last two iterates, then remember this one
                d, d_prev = d + cfg.extrap * (d - d_prev), d
            d = d + lr_b * gn * push
            d = torch.sign(d) * torch.clamp(d.abs() - threshold, min=0.0)
            if var_mask is not None:
                d = d * var_mask
            d = torch.maximum(torch.minimum(d, hi), lo)
            elapsed_step = time.time() - now
            step_time = elapsed_step if steps == 1 else 0.8 * step_time + 0.2 * elapsed_step

        if flush():
            info["early_exit"] = True
        info["first_flip_step"] = first_flip_step
        info["lanes"] = [
            {"mode": int(lanes.mode[b]), "tau": round(float(lanes.tau[b]), 3), "target": int(lanes.j[b]),
             "est": round(best[b].est, 5) if best[b] else 0.0, "class": best[b].cls if best[b] else None}
            for b in range(B)
        ]
        candidates = [x for x in best if x is not None]
        chosen = max(candidates, key=lambda x: x.est) if candidates else fallback
        info["below_safety_fallback"] = not candidates and fallback is not None

        q_full = outside.clone()
        if chosen is not None:
            if window is not None:
                q_full[window] = chosen.q
            else:
                q_full = chosen.q.clone()
        elif window is not None:
            q_full[window] = fixed
        else:
            q_full = fixed.clone()
        adv_u8 = (clean + q_full).clamp(0, 255).round().to(torch.uint8)

        # Final check on the exact image the validator will decode.
        delta = adv_u8.float() - clean
        row = self._validator_logits(adv_u8.float() / 255.0)
        other = row.clone()
        other[true_index] = -float("inf")
        margin = float((other.max() - row[true_index]).item())
        flipped = int(row.argmax().item()) != true_index
        nonzero = int((delta != 0).sum().item())
        changed_pixels = int((delta != 0).any(dim=0).sum().item())
        est = estimate_score(
            linf=QUANT if nonzero else 0.0,
            rmse=rmse_for_unit_changes(nonzero, total),
            margin=margin,
            changed_pixels=changed_pixels,
            flipped=flipped,
            params=self.params,
        )
        info["adv_class"] = int(row.argmax().item())
        return AttackResult(
            adv_u8=adv_u8.cpu(),
            est_score=est,
            margin=margin,
            flipped=flipped,
            nonzero=nonzero,
            changed_pixels=changed_pixels,
            clean_index=clean_index,
            true_index=true_index,
            steps=steps,
            elapsed=time.time() - t0,
            info=info,
        )

    def _retarget(self, lanes: _Lanes, best, d, lam, started, quantize, true_index: int, n_free: int,
                  tried: set[int], protect: list[int]) -> dict[str, Any]:
        """Restart the weakest lanes as targeted searches on classes the lanes surfaced.

        The overall best lane and the lanes re-targeted in the previous round (still young) are never freed.
        """
        B = self.B
        _, arg = lanes.check(quantize(d))
        surfaced: list[int] = []
        for b in range(B):
            for c in ([best[b].cls] if best[b] is not None else []) + [int(arg[b])]:
                if c != true_index and c not in surfaced:
                    surfaced.append(c)
        k = min(self.cfg.switch_topk, lanes.TOPK)
        if k > 0:  # then each lane's runner-up classes, rank by rank: cheap alternatives to the argmax basin
            top = lanes.topk.tolist()
            for r in range(k):
                for b in range(B):
                    c = top[b][r]
                    if c != true_index and c not in surfaced:
                        surfaced.append(c)
        scored = [b for b in range(B) if best[b] is not None]
        leader = max(scored, key=lambda b: best[b].est) if scored else -1
        order = sorted((b for b in range(B) if b != leader and b not in protect),
                       key=lambda b: best[b].est if best[b] is not None else -1.0)
        free = order[: min(n_free, B)]
        keep_classes = {best[b].cls for b in range(B) if b not in free and best[b] is not None}
        keep_classes |= {int(lanes.j[b]) for b in range(B) if b not in free and int(lanes.j[b]) >= 0}
        targets = [c for c in surfaced if c not in keep_classes and c not in tried][: len(free)]
        for b, c in zip(free, targets):
            tried.add(c)
            lanes.j[b] = c
            lanes.mode[b] = 0
            d[b] = d[leader] if self.cfg.retarget_warm and leader >= 0 else 0.0
            lam[b] = 0.0
            started[b] = False
            best[b] = None
        return {"targets": targets, "lanes": free[: len(targets)]}
