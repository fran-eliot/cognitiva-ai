from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Tuple

REQUIRED_CLIN_COLS = ["patient_id", "cohort"]  # + columnas clínicas (se validan dinámicamente)
ALLOWED_COHORTS = {"OAS1", "OAS2"}

def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV no encontrado: {p}")
    return pd.read_csv(p)

def ensure_cohort_column(df: pd.DataFrame) -> pd.DataFrame:
    if "cohort" not in df.columns:
        raise KeyError("Falta columna 'cohort' (OAS1/OAS2).")
    bad = set(df["cohort"].astype(str).str.upper().unique()) - ALLOWED_COHORTS
    if bad:
        raise ValueError(f"Cohortes desconocidas: {bad}. Esperado en {ALLOWED_COHORTS}.")
    df["cohort"] = df["cohort"].str.upper()
    return df

def merge_probs(df: pd.DataFrame, proba_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    # proba_df debe tener: patient_id, proba
    if "patient_id" not in df or "patient_id" not in proba_df:
        raise KeyError("Se requieren 'patient_id' en ambos dataframes para merge.")
    if "proba" not in proba_df.columns:
        raise KeyError("El dataframe de probabilidades debe tener columna 'proba'.")
    out = df.merge(proba_df.rename(columns={"proba": col_name}), on="patient_id", how="left")
    return out

def split_val_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Si ya viene con split anotado
    if "split" in df.columns:
        val = df[df["split"].str.upper()=="VAL"].copy()
        tst = df[df["split"].str.upper()=="TEST"].copy()
        return val, tst
    # Si no, devuelve todo como TEST (fallback)
    return pd.DataFrame(), df.copy()
