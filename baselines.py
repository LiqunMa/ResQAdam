"""Baseline optimizers compared in the paper: AdamW, GaLore-AdamW, GaLore-AdamW8bit, LDAdam, AdaFactor.

In all optimizers, quantizer scales / zero points are trained without weight decay.
Projection-based optimizers (GaLore, LDAdam) apply low-rank projection to every 2D
weight except the embedding and LM head.
"""
import torch

# Hyperparameters used for the baselines in the paper.
GALORE_UPDATE_PROJ_GAP = 200
GALORE_SCALE = 0.25
LDADAM_RHO = 0.908


def split_params(model, separate_lowrank: bool = False):
    """Return (lowrank, quantizer, other) parameter lists.

    If `separate_lowrank` is False, 2D weights are kept in `other` and `lowrank` is empty.
    """
    lowrank, quantizer, other = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "quantizer" in name and ("scale" in name or "zero_point" in name):
            quantizer.append(param)
        elif separate_lowrank and param.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
            lowrank.append(param)
        else:
            other.append(param)
    return lowrank, quantizer, other


def build_adamw(model, lr, weight_decay, betas=(0.9, 0.95)):
    _, quantizer, other = split_params(model)
    param_groups = [{"params": other, "weight_decay": weight_decay}]
    if quantizer:
        param_groups.append({"params": quantizer, "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


def build_galore(model, lr, weight_decay, rank, use_8bit=False, betas=(0.9, 0.95)):
    from galore_torch import GaLoreAdamW, GaLoreAdamW8bit

    lowrank, quantizer, other = split_params(model, separate_lowrank=True)
    param_groups = [
        {
            "params": lowrank,
            "rank": rank,
            "update_proj_gap": GALORE_UPDATE_PROJ_GAP,
            "scale": GALORE_SCALE,
            "proj_type": "std",
            "weight_decay": weight_decay,
        },
        {"params": quantizer, "weight_decay": 0.0},
        {"params": other, "weight_decay": weight_decay},
    ]
    optim_cls = GaLoreAdamW8bit if use_8bit else GaLoreAdamW
    return optim_cls(param_groups, lr=lr, betas=betas)


def build_ldadam(model, lr, weight_decay, rank, betas=(0.9, 0.95)):
    from ldadamw_torch import LDAdamW

    lowrank, quantizer, other = split_params(model, separate_lowrank=True)
    param_groups = [
        {
            "params": lowrank,
            "enable_lowrank": True,
            "rank": rank,
            "rho": LDADAM_RHO,
            "proj_type": "std",
            "proj_method": "power_iteration",
            "error_feedback": True,
            "weight_decay": weight_decay,
        },
        {"params": quantizer, "enable_lowrank": False, "weight_decay": 0.0},
        {"params": other, "enable_lowrank": False, "weight_decay": weight_decay},
    ]
    return LDAdamW(params=param_groups, lr=lr, betas=betas, eps=1e-8)


def build_adafactor(model, lr, weight_decay):
    # PyTorch native Adafactor (torch >= 2.5); settings match the common
    # transformers defaults decay_rate=-0.8, eps=(1e-30, 1e-3), clip_threshold=1.0.
    _, quantizer, other = split_params(model)
    param_groups = [{"params": other, "weight_decay": weight_decay}]
    if quantizer:
        param_groups.append({"params": quantizer, "weight_decay": 0.0})
    return torch.optim.Adafactor(param_groups, lr=lr, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0)


def build_baseline(name, model, lr, weight_decay, rank):
    if name == "adamw":
        return build_adamw(model, lr, weight_decay)
    if name == "galore-adamw":
        return build_galore(model, lr, weight_decay, rank)
    if name == "galore-adamw-8bit":
        return build_galore(model, lr, weight_decay, rank, use_8bit=True)
    if name == "ldadam":
        return build_ldadam(model, lr, weight_decay, rank)
    if name == "adafactor":
        return build_adafactor(model, lr, weight_decay)
    raise ValueError(f"Invalid optimizer: {name}")


BASELINES = ["adamw", "galore-adamw", "galore-adamw-8bit", "ldadam", "adafactor"]
