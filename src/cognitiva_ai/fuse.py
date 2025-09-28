# src/cognitiva_ai/fuse.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

def apply_thresholds_by_cohort(proba: np.ndarray, cohorts: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """
    Aplica umbrales por cohorte y devuelve y_pred binario.
    """
    thr_vec = np.vectorize(lambda c: thresholds.get(str(c).upper(), 0.5))(cohorts)
    return (proba >= thr_vec).astype(int)

def build_prediction_frame(patient_ids, cohorts, y_true, p_img, p_clin, p_cal, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "cohort": cohorts,
        "y_true": y_true if y_true is not None else np.nan,
        "p_img": p_img,
        "p_clin": p_clin,
        "p_fused_cal": p_cal,
        "y_pred": y_pred
    })
    return df
