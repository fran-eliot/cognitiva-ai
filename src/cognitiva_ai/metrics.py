from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

def safe_auc(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")

def safe_prauc(y_true, y_prob) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")

def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))

def confusion_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tp, fp, tn, fn

def cost(tp, fp, tn, fn, c_fn=5.0, c_fp=1.0) -> float:
    return c_fn * fn + c_fp * fp

def ece_mce(y_true, y_prob, bins: int = 10):
    """
    Expected Calibration Error y Maximum Calibration Error (binning uniforme).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, bins+1)
    ece = 0.0
    mce = 0.0
    for i in range(bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i+1])
        if np.any(mask):
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            gap = abs(acc - conf)
            ece += gap * (mask.sum() / y_true.size)
            mce = max(mce, gap)
    return float(ece), float(mce)
