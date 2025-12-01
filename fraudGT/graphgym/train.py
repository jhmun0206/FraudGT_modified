import logging
import time

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from typing import Tuple

from fraudGT.graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch

from sklearn.metrics import confusion_matrix
import numpy as np


# Helper: compute precision, recall, auc (binary) using only torch
def _compute_pr_auc(true: torch.Tensor, pred_score: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and (binary) AUC from logits/probabilities.
    - true: 1D tensor of shape [N] with class indices
    - pred_score: 2D tensor of shape [N, C] with logits or probabilities
    Returns (precision, recall, auc) as floats. If AUC is not applicable (C != 2), returns -1 for auc.
    """
    with torch.no_grad():
        if pred_score.dim() == 1:
            # ensure shape [N, 1] for safety
            pred_score = pred_score.unsqueeze(-1)
        # predicted class
        pred_cls = pred_score.argmax(dim=-1)

        num_classes = pred_score.size(-1)
        # confusion matrix
        C = num_classes
        precision = 0.0
        recall = 0.0
        for c in range(C):
            tp = ((pred_cls == c) & (true == c)).sum().item()
            fp = ((pred_cls == c) & (true != c)).sum().item()
            fn = ((pred_cls != c) & (true == c)).sum().item()
            prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision += prec_c
            recall += rec_c
        precision /= max(C, 1)
        recall /= max(C, 1)

        # Binary AUC (one-vs-rest generalization is not implemented to avoid extra deps)
        auc = -1.0
        if num_classes == 2:
            # Convert logits to probabilities if needed
            if pred_score.dtype.is_floating_point:
                # sigmoid on positive class logit if provided as 2-class logits use softmax
                if pred_score.size(-1) == 2:
                    probs = torch.softmax(pred_score, dim=-1)[:, 1]
                else:
                    probs = torch.sigmoid(pred_score.squeeze())
            else:
                probs = pred_score.float()
            # ROC-AUC via trapezoidal rule on sorted thresholds
            # Sort by score descending
            sorted_idx = torch.argsort(probs, descending=True)
            y_true = (true == 1).float()[sorted_idx]
            y_score = probs[sorted_idx]
            # thresholds = unique scores
            distinct_value_indices = torch.where(torch.diff(y_score, prepend=y_score[:1]-1))[0]
            tps = torch.cumsum(y_true, 0)[distinct_value_indices]
            fps = (torch.arange(1, y_true.numel()+1, device=y_true.device)[distinct_value_indices] - tps)
            P = y_true.sum().clamp(min=1.0)
            N = (y_true.numel() - y_true.sum()).clamp(min=1.0)
            tpr = (tps / P).cpu()
            fpr = (fps / N).cpu()
            # add (0,0) and (1,1)
            tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
            fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])
            auc = torch.trapz(tpr, fpr).item()
        return float(precision), float(recall), float(auc)


# Helper: save confusion matrix as .npy and .png
def _save_confusion_matrix(cm: np.ndarray, split: str, out_dir: str):
    """
    Save confusion matrix as .npy and .png under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split}_confusion_matrix.npy"), cm)

    # Plot heatmap with matplotlib only (no seaborn dependency)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f'Confusion Matrix ({split})')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.colorbar(im, ax=ax)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{split}_confusion_matrix.png"))
    plt.close(fig)


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    device = torch.device(cfg.device)
    time_start = time.time()
    y_true_epoch, y_pred_epoch = [], []
    for batch in loader:
        batch.split = 'train'
        if hasattr(batch, 'to'):  # Handles both Data and HeteroData
            batch = batch.to(device)
        else:
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    batch[k] = v.to(device)
        optimizer.zero_grad()
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        y_true_epoch.append(true.detach().cpu().numpy())
        y_pred_epoch.append(pred_score.detach().cpu().argmax(axis=-1).numpy())
        # extra metrics (epoch-wise aggregator will average batch metrics)
        prec, rec, auc = _compute_pr_auc(true.detach().cpu(), pred_score.detach().cpu())
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            precision=prec,
                            recall=rec,
                            auc=auc,
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    # Compute and save confusion matrix for this epoch (train split)
    try:
        y_true = np.concatenate(y_true_epoch) if len(y_true_epoch) else np.array([])
        y_pred = np.concatenate(y_pred_epoch) if len(y_pred_epoch) else np.array([])
        if y_true.size and y_pred.size:
            cm = confusion_matrix(y_true, y_pred)
            _save_confusion_matrix(cm, split='train', out_dir=cfg.run_dir)
    except Exception as e:
        logging.warning(f"Failed to save train confusion matrix: {e}")
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    device = torch.device(cfg.device)
    time_start = time.time()
    y_true_epoch, y_pred_epoch = [], []
    for batch in loader:
        batch.split = split
        if hasattr(batch, 'to'):  # Handles both Data and HeteroData
            batch = batch.to(device)
        else:
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    batch[k] = v.to(device)
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        y_true_epoch.append(true.detach().cpu().numpy())
        y_pred_epoch.append(pred_score.detach().cpu().argmax(axis=-1).numpy())
        prec, rec, auc = _compute_pr_auc(true.detach().cpu(), pred_score.detach().cpu())
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            precision=prec,
                            recall=rec,
                            auc=auc,
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    # Compute and save confusion matrix for this epoch (validation/test split)
    try:
        y_true = np.concatenate(y_true_epoch) if len(y_true_epoch) else np.array([])
        y_pred = np.concatenate(y_pred_epoch) if len(y_pred_epoch) else np.array([])
        if y_true.size and y_pred.size:
            cm = confusion_matrix(y_true, y_pred)
            _save_confusion_matrix(cm, split=split, out_dir=cfg.run_dir)
    except Exception as e:
        logging.warning(f"Failed to save {split} confusion matrix: {e}")


def train(loggers, loaders, model, optimizer, scheduler):
    r"""
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
