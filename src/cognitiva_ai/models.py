from __future__ import annotations
import pickle, joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Ajusta esta ruta a tu estructura real (usa 'models/' en la raíz si prefieres)
MODELS_DIR = Path("p26_release/models")

# --- carga de artefactos P26 ---
def load_clinical_stack() -> Dict[str, Any]:
    paths = {
        "clin_model": MODELS_DIR / "p26_clin_model.pkl",
        "num_pipe":   MODELS_DIR / "p26_clin_num_pipe.pkl",
        "ohe":        MODELS_DIR / "p26_clin_ohe.pkl",
    }
    objs = {}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Falta artefacto clínico: {p}")
        # joblib/pickle
        try:
            objs[k] = joblib.load(p)
        except Exception:
            with open(p, "rb") as f:
                objs[k] = pickle.load(f)
    return objs

def load_meta_and_calibrators() -> Dict[str, Any]:
    paths = {
        "meta_late": MODELS_DIR / "p26_meta_late.pkl",
        "platt_OAS1": MODELS_DIR / "p26b_platt_OAS1.pkl",
        "platt_OAS2": MODELS_DIR / "p26b_platt_OAS2.pkl",
    }
    objs = {}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Falta artefacto: {p}")
        try:
            objs[k] = joblib.load(p)
        except Exception:
            with open(p, "rb") as f:
                objs[k] = pickle.load(f)
    return objs

# --- predicciones ---
def predict_pclin(df_clin: pd.DataFrame, clin_stack: Dict[str, Any], preprocess_module) -> np.ndarray:
    """
    Devuelve p_clin a partir de dataframe clínico crudo:
    - Selecciona columnas esperadas
    - Aplica num_pipe + ohe
    - Pasa por p26_clin_model
    """
    clin_model = clin_stack["clin_model"]
    num_pipe   = clin_stack["num_pipe"]
    ohe        = clin_stack["ohe"]

    # Idealmente guardarías expected clinical columns junto al pipeline; aquí derivamos de ohe + num_pipe
    # Suponemos que las columnas categóricas usadas para OHE están en ohe.feature_names_in_
    cat_cols = getattr(ohe, "feature_names_in_", None)
    if cat_cols is None:
        raise AttributeError("OHE sin 'feature_names_in_'. Guarde columnas categóricas al entrenar.")

    expected_cols = list(cat_cols) + [c for c in df_clin.columns if c not in cat_cols and c not in ["patient_id","cohort","y_true","split"]]
    X = preprocess_module.apply_preprocessors(df_clin[expected_cols + ["patient_id","cohort"]].drop(columns=["patient_id","cohort"]), num_pipe, ohe)
    proba = clin_model.predict_proba(X)[:, 1]
    return proba

def predict_meta_late(p_img: np.ndarray, p_clin: np.ndarray, meta_bundle: Dict[str, Any], cohorts: np.ndarray) -> np.ndarray:
    """
    Ensambla p_img + p_clin -> meta_late -> proba_fused
    """
    meta = meta_bundle["meta_late"]
    # Acepta entrada como 2 columnas (p_img, p_clin)
    X = np.column_stack([p_img, p_clin])
    p_fused = meta.predict_proba(X)[:, 1]
    # Calibración por cohorte
    p_cal = np.zeros_like(p_fused)
    mask1 = (cohorts == "OAS1")
    mask2 = (cohorts == "OAS2")
    p_cal[mask1] = meta_bundle["platt_OAS1"].predict_proba(p_fused[mask1].reshape(-1,1))[:,1]
    p_cal[mask2] = meta_bundle["platt_OAS2"].predict_proba(p_fused[mask2].reshape(-1,1))[:,1]
    return p_cal
