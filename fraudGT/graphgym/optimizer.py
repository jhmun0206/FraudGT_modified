import torch
from dataclasses import dataclass
from typing import Iterable, List, Optional
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class OptimizerConfig:
    optimizer: str = "adam"
    base_lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD

@dataclass
class SchedulerConfig:
    scheduler: str = "none"  # 'cos', 'step', 'plateau', 'none'
    max_epoch: int = 100
    steps: List[int] = None
    reduce_factor: float = 0.1
    schedule_patience: int = 10
    min_lr: float = 0.0
    lr_decay: float = 0.1  # kept for compatibility

# Extension used by utils.new_scheduler_config(cfg)
@dataclass
class ExtendedSchedulerConfig(SchedulerConfig):
    pass

# -------------------------
# Optimizers
# -------------------------
optimizer_dict = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
}

def _tensor_params(params: Iterable):
    """Accepts model.parameters() or model.named_parameters() and filters out non-tensors/duplicates."""
    seen = set()
    for p in params:
        if isinstance(p, tuple):
            # named_parameters() -> (name, param)
            p = p[1]
        if not isinstance(p, torch.Tensor):
            continue
        if not p.requires_grad:
            continue
        # guard against duplicates
        if id(p) in seen:
            continue
        seen.add(id(p))
        yield p

def from_config(func):
    def wrapper(params, cfg: Optional[OptimizerConfig] = None):
        if cfg is None:
            # fallback defaults
            return func(params)
        name = func.__name__.lower()
        if name == "adam":
            return func(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        if name == "adamw":
            return func(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        if name == "sgd":
            return func(params, lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        return func(params)
    return wrapper

def create_optimizer(params, cfg):
    """cfg may be the global cfg or an OptimizerConfig; support both."""
    opt_cfg = getattr(cfg, "optim", cfg)
    func = optimizer_dict[str(opt_cfg.optimizer).lower()]
    valid_params = list(_tensor_params(params))
    return from_config(func)(valid_params, cfg=opt_cfg)

# -------------------------
# Schedulers
# -------------------------
class _NoOpScheduler:
    """Scheduler stub that keeps API parity with torch schedulers."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(self, *args, **kwargs):
        # do nothing
        return
    def get_last_lr(self):
        # mimic torch schedulers -> list[float]
        return [group.get("lr", 0.0) for group in self.optimizer.param_groups]
    # optional parity
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        return

def create_scheduler(optimizer, cfg: SchedulerConfig):
    name = str(cfg.scheduler).lower()
    if name in ("none", "off", "constant"):
        return _NoOpScheduler(optimizer)
    if name in ("cos", "cosine", "cosineanneal", "cosineannealing"):
        # T_max = max_epoch for simple cosine over total epochs
        return CosineAnnealingLR(optimizer, T_max=max(1, int(cfg.max_epoch)), eta_min=float(cfg.min_lr))
    if name in ("step", "steplr"):
        # if steps provided, use StepLR with first step as period; else fall back to NoOp
        if cfg.steps and len(cfg.steps) > 0:
            return StepLR(optimizer, step_size=int(cfg.steps[0]), gamma=float(cfg.lr_decay))
        return _NoOpScheduler(optimizer)
    if name in ("plateau", "reducelronplateau"):
        return ReduceLROnPlateau(optimizer, mode="min", factor=float(cfg.reduce_factor),
                                 patience=int(cfg.schedule_patience), min_lr=float(cfg.min_lr))
    # default: no-op to be safe
    return _NoOpScheduler(optimizer)
