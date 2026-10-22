"""ResQAdam: Residual-Preserving Adaptive Optimization for Memory-Efficient QAT.

For every quantized 2D weight W (n x m), ResQAdam keeps
  * one dense momentum M in BF16,
  * rank-r bases U (n x r), V (m x r) taken from the top singular vectors of M,
  * Adam moments m_proj, v_proj of size r x r in the principal subspace.

Each step:
  1. M <- b1 * M + (1 - b1) * G
  2. Adam on the projected gradient U^T G V; the result is mapped back as
     delta_par = sqrt(nm)/r * U A V^T.
  3. Residual M_perp = M - U U^T M V V^T; the residual update is
     ||delta_par||_F * sign(M_perp) / (||sign(M_perp)||_F + eps).
  4. The two updates are fused with the momentum energy ratios
     rho_par = ||M_par|| / (||M|| + eps), rho_perp = ||M_perp|| / (||M|| + eps).
At steps with t mod tau == 1 (after updating M_t) the bases are refreshed by a
rank-r SVD of M_t and the r x r Adam states are transported to the new bases
(LDAdam rule). The bases at t == 1 come from the first gradient.

All other parameters (embeddings, norms, LM head, quantizer scales, ...) are
updated with plain AdamW.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


def collect_param_groups(
    model: nn.Module,
    weight_quantizer_pairs: List[Tuple[nn.Parameter, nn.Module]],
    min_dim: int = 64,
):
    """Split model parameters into ResQAdam weights, AdamW parameters and quantizer parameters."""
    id_to_name = {id(p): name for name, p in model.named_parameters()}

    lowrank_params: List[nn.Parameter] = []
    lowrank_names: List[str] = []
    quantizer_params: List[nn.Parameter] = []
    lowrank_ids, quantizer_ids = set(), set()

    for w, q in weight_quantizer_pairs:
        if w.dim() == 2 and min(w.shape) >= min_dim:
            lowrank_params.append(w)
            lowrank_names.append(id_to_name.get(id(w), "unknown"))
            lowrank_ids.add(id(w))
        for name, p in q.named_parameters():
            if ("scale" in name or "zero_point" in name) and id(p) not in quantizer_ids:
                quantizer_params.append(p)
                quantizer_ids.add(id(p))

    other_params = [
        p for p in model.parameters()
        if id(p) not in lowrank_ids and id(p) not in quantizer_ids
    ]
    return lowrank_params, other_params, quantizer_params, lowrank_names


class ResQAdam(Optimizer):
    """ResQAdam optimizer.

    Args:
        lowrank_params: 2D quantized weights optimized by ResQAdam.
        other_params: parameters optimized by AdamW.
        quantizer_params: quantizer scales / zero points (AdamW, no weight decay).
        lr, betas, eps, weight_decay: as in AdamW.
        rank: rank r of the principal subspace.
        adapt_interval: SVD refresh interval tau.
        total_steps: total number of optimizer steps (only needed if refresh_warmup_ratio > 0).
        refresh_warmup_ratio: fraction of training during which the bases are not
            refreshed. We use 0.1 for from-scratch QAT and 0.0 for QAT fine-tuning.
    """

    def __init__(
        self,
        lowrank_params: List[nn.Parameter],
        other_params: List[nn.Parameter],
        quantizer_params: Optional[List[nn.Parameter]] = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        rank: int = 64,
        adapt_interval: int = 200,
        total_steps: Optional[int] = None,
        refresh_warmup_ratio: float = 0.0,
    ):
        if refresh_warmup_ratio > 0 and total_steps is None:
            raise ValueError("total_steps is required when refresh_warmup_ratio > 0")
        self.rank = rank
        self.adapt_interval = adapt_interval
        self.refresh_start = int((total_steps or 0) * refresh_warmup_ratio)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        param_groups = [
            {**defaults, "params": list(lowrank_params), "group_type": "lowrank"},
            {**defaults, "params": list(other_params), "group_type": "adamw"},
        ]
        if quantizer_params:
            param_groups.append({
                **defaults,
                "params": list(quantizer_params),
                "group_type": "adamw",
                "weight_decay": 0.0,
            })
        super().__init__(param_groups, defaults)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def _init_lowrank_state(self, state: dict, p: torch.Tensor, grad: torch.Tensor, group: dict):
        n, m = p.shape
        dev, dt = p.device, p.dtype
        r = self.rank
        state["step"] = 0
        state["betas"] = group["betas"]
        state["M"] = torch.zeros_like(p, dtype=torch.bfloat16)
        state["m_proj"] = torch.zeros(r, r, device=dev, dtype=dt)
        state["v_proj"] = torch.zeros(r, r, device=dev, dtype=dt)
        state["dim_scale"] = ((n * m) ** 0.5) / r   # alpha_r = sqrt(nm) / r
        # The bases are initialized from the first gradient.
        U, _, V = torch.svd_lowrank(grad.float(), q=r + 10, niter=3)
        state["U"] = U[:, :r].to(dt).contiguous()
        state["V"] = V[:, :r].to(dt).contiguous()

    @staticmethod
    def _init_adamw_state(state: dict, p: torch.Tensor):
        state["step"] = 0
        state["m"] = torch.zeros_like(p)
        state["v"] = torch.zeros_like(p)

    @torch.no_grad()
    def _refresh_bases(self, state: dict):
        """Recompute U, V from the momentum and transport the r x r Adam states (LDAdam)."""
        r = self.rank
        b1, b2 = state["betas"]
        last_t = state["step"] - 1
        bias1 = 1 - b1 ** last_t
        bias2 = 1 - b2 ** last_t

        U_full, _, V_full = torch.svd_lowrank(state["M"].float(), q=r + 10, niter=3)
        old_U, old_V = state["U"], state["V"]
        new_U = U_full[:, :r].to(old_U.dtype).contiguous()
        new_V = V_full[:, :r].to(old_V.dtype).contiguous()

        P = new_U.T @ old_U   # r x r
        Q = old_V.T @ new_V   # r x r

        old_m = state["m_proj"]
        new_m = P @ old_m @ Q

        # v_hat_new = P^2 (v_hat_old - m_hat_old^2) Q^2 + m_hat_new^2
        old_var = state["v_proj"] / bias2 - old_m.square() / (bias1 ** 2)
        new_v = P.square() @ old_var @ Q.square()
        new_v.addcmul_(new_m, new_m, value=1.0 / (bias1 ** 2))
        new_v.mul_(bias2)
        new_v.abs_()

        state["m_proj"] = new_m.contiguous()
        state["v_proj"] = new_v.contiguous()
        state["U"], state["V"] = new_U, new_V

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # ----------------------- plain AdamW ----------------------- #
                if group["group_type"] == "adamw":
                    if len(state) == 0:
                        self._init_adamw_state(state, p)
                    state["step"] += 1
                    t = state["step"]
                    state["m"].lerp_(grad, 1 - b1)
                    state["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    m_hat = state["m"] / (1 - b1 ** t)
                    v_hat = state["v"] / (1 - b2 ** t)
                    if wd > 0:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                    continue

                # ------------------------ ResQAdam ------------------------- #
                if len(state) == 0:
                    self._init_lowrank_state(state, p, grad, group)

                state["step"] += 1
                t = state["step"]
                M = state["M"]
                M.mul_(b1).add_(grad.to(torch.bfloat16), alpha=1 - b1)

                # Periodic subspace refresh from the updated momentum M_t (t mod tau == 1).
                # At t == 1 the bases come from the first gradient, i.e. from M_1 up to a scale.
                if t > 1 and (t - 1) % self.adapt_interval == 0 and t >= self.refresh_start:
                    self._refresh_bases(state)

                U, V = state["U"], state["V"]
                m_proj, v_proj = state["m_proj"], state["v_proj"]

                # Adam in the r x r coordinates of the principal subspace.
                grad_proj = U.T @ grad @ V
                m_proj.mul_(b1).add_(grad_proj, alpha=1 - b1)
                v_proj.mul_(b2).addcmul_(grad_proj, grad_proj, value=1 - b2)
                m_hat = m_proj / (1 - b1 ** t)
                v_hat = v_proj / (1 - b2 ** t)
                adam_coord = m_hat / (v_hat.sqrt() + eps)

                # Decomposition M_t = M_par + M_perp with M_par = U U^T M V V^T.
                M_coord = U.T @ M.float() @ V   # r x r coordinates of M_par
                M_res = M - (U @ M_coord @ V.T).to(torch.bfloat16)
                M_norm = M.float().norm()
                # Energy ratios; ||M_par||_F = ||U^T M V||_F since U, V are orthonormal.
                rho_par = M_coord.norm() / (M_norm + eps)
                rho_res = M_res.float().norm() / (M_norm + eps)
                M_res.sign_()
                sign_norm = M_res.count_nonzero().sqrt()   # ||sign(M_perp)||_F

                # Principal update, written into the gradient buffer to save memory.
                update = grad
                torch.mm(U @ adam_coord, V.T, out=update)
                update.mul_(state["dim_scale"])

                # Residual sign update rescaled to ||delta_par||_F, then energy-weighted fusion.
                res_scale = rho_res * update.norm() / (sign_norm + eps)
                update.mul_(rho_par)
                update.add_(M_res, alpha=res_scale)

                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)

        return loss


def build_resqadam(
    model: nn.Module,
    weight_quantizer_pairs: List[Tuple[nn.Parameter, nn.Module]],
    min_dim: int = 64,
    **kwargs,
) -> ResQAdam:
    """Build ResQAdam for a model given (weight, weight_quantizer) pairs of its quantized layers."""
    lowrank_params, other_params, quantizer_params, _ = collect_param_groups(
        model, weight_quantizer_pairs, min_dim,
    )
    return ResQAdam(
        lowrank_params=lowrank_params,
        other_params=other_params,
        quantizer_params=quantizer_params,
        **kwargs,
    )
