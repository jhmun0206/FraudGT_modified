from __future__ import annotations
from typing import Any, Dict
import numbers
import os

# graphgym optimizer/scheduler 설정 클래스를 여기서 씀
from fraudGT.graphgym.optimizer import SchedulerConfig, OptimizerConfig

# ---------------------------
# 기본 유틸
# ---------------------------
def _to_builtin(obj):
    try:
        import torch
    except Exception:
        torch = None
    try:
        import numpy as np
    except Exception:
        np = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.item() if obj.size == 1 else obj.tolist()
    if isinstance(obj, (numbers.Number, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_builtin(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj

def cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        d = cfg.to_dict()
        return {k: _to_builtin(cfg_to_dict(v)) for k, v in d.items()}
    if hasattr(cfg, "__dict__") and not isinstance(cfg, dict):
        return {k: cfg_to_dict(v) for k, v in vars(cfg).items() if not k.startswith("_")}
    if isinstance(cfg, dict):
        return {k: cfg_to_dict(v) for k, v in cfg.items()}
    return _to_builtin(cfg)

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items

def make_wandb_name(cfg: Any) -> str:
    def g(o, path, default=""):
        cur = o
        for p in path:
            if isinstance(cur, dict):
                cur = cur.get(p, default)
            else:
                cur = getattr(cur, p, default)
        return cur if cur is not None else default
    project = g(cfg, ["wandb", "project"], "fraudgt")
    dataset = g(cfg, ["dataset", "name"], "data")
    model   = g(cfg, ["model", "type"], "model")
    tag     = g(cfg, ["name_tag"], "")
    return "-".join(filter(None, [project, dataset, model, tag]))

# ---------------------------
# Config 생성기 (Scheduler / Optimizer)
# ---------------------------
def new_scheduler_config(cfg: Any) -> SchedulerConfig:
    opt = getattr(cfg, "optim", cfg)
    kwargs = {
        "scheduler": getattr(opt, "scheduler", "none"),
        "max_epoch": getattr(opt, "max_epoch", 100),
        "steps": getattr(opt, "steps", []),
        "reduce_factor": getattr(opt, "reduce_factor", 0.1),
        "schedule_patience": getattr(opt, "schedule_patience", 10),
        "min_lr": getattr(opt, "min_lr", 0.0),
        "lr_decay": getattr(opt, "lr_decay", 0.1),
    }
    # 정의된 필드만 전달
    valid = getattr(SchedulerConfig, "__annotations__", {}).keys()
    if valid:
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return SchedulerConfig(**kwargs)

def new_optimizer_config(cfg: Any) -> OptimizerConfig:
    opt = getattr(cfg, "optim", cfg)
    kwargs = {
        "optimizer": getattr(opt, "optimizer", "adam"),
        "base_lr": getattr(opt, "base_lr", 1e-3),
        "weight_decay": getattr(opt, "weight_decay", 1e-4),
        "momentum": getattr(opt, "momentum", 0.9),
    }
    valid = getattr(OptimizerConfig, "__annotations__", {}).keys()
    if valid:
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return OptimizerConfig(**kwargs)

# ---------------------------
# main.py 가 import 하는 경로/출력 디렉토리 설정
# ---------------------------
def custom_set_out_dir(cfg: Any) -> str:
    """
    main.py에서 호출되는 헬퍼.
    cfg.out_dir가 있으면 그대로 쓰고, 없으면 기본값 설정.
    필요시 cfg.run_dir도 세팅해 둔다.
    """
    base = getattr(cfg, "out_dir", None)
    if not base:
        # 기본 결과 경로
        base = "./results"
        setattr(cfg, "out_dir", base)

    # cfg 파일명(옵션)
    cfg_name = None
    # yacs args일 경우 load_cfg에서 cfg.cfg_file을 넣어두는 경우가 많음
    if hasattr(cfg, "cfg_file") and cfg.cfg_file:
        cfg_name = os.path.splitext(os.path.basename(cfg.cfg_file))[0]
    # 디렉토리 생성
    os.makedirs(base, exist_ok=True)

    # run_dir이 없으면 out_dir/<cfg_name>/<run_id> 형태로 보조 세팅
    run_id = getattr(cfg, "run_id", 0)
    if not hasattr(cfg, "run_dir") or not getattr(cfg, "run_dir"):
        if cfg_name:
            run_dir = os.path.join(base, cfg_name, str(run_id))
        else:
            run_dir = os.path.join(base, str(run_id))
        setattr(cfg, "run_dir", run_dir)
    os.makedirs(getattr(cfg, "run_dir"), exist_ok=True)

    return base
custom_set_run_dir = custom_set_out_dir
# --- patched: align signature with fraudGT.main call ---
import os
from pathlib import Path

def custom_set_out_dir(cfg, cfg_file=None, name_tag='', gpu=None):
    """
    Aligns with main.run(): custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)
    - Ensures cfg.out_dir and cfg.run_dir exist.
    - Appends the config file stem to out_dir for readability.
    """
    cfg_file_stem = Path(cfg_file).stem if cfg_file else "config"
    # dataset name fallback
    ds_name = getattr(getattr(cfg, "dataset", object()), "name", "dataset")

    # base out dir: use cfg.out_dir if set, otherwise default under /data/<user>/results/...
    base_out = getattr(cfg, "out_dir", "") or f"/data/{os.getenv('USER','user')}/results/fraudgt/{ds_name}"
    # if user already set a full path including stem, don't double-append
    if not base_out.rstrip("/").endswith(cfg_file_stem):
        out_dir = os.path.join(base_out, cfg_file_stem)
    else:
        out_dir = base_out

    os.makedirs(out_dir, exist_ok=True)
    cfg.out_dir = out_dir

    # also set run_dir (main/others expect this sometimes)
    run_id = str(getattr(cfg, "run_id", 0))
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    cfg.run_dir = run_dir
    return out_dir

# keep backwards-compat alias
custom_set_run_dir = custom_set_out_dir
