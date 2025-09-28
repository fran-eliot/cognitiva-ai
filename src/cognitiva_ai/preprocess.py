from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.compose import ColumnTransformer

def select_clinical_columns(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas clínicas: {missing}")
    return df[expected_cols].copy()

def apply_preprocessors(
    df_clin: pd.DataFrame,
    num_pipe,
    ohe
) -> np.ndarray:
    """
    Aplica pipeline numérico y OHE a las columnas correspondientes.
    Se asume que `num_pipe` y `ohe` ya han sido 'fit' durante P26.
    """
    # Identifica qué columnas numéricas y categóricas esperaba el OHE original
    # Suponemos que el OHE fue entrenado con un conjunto fijo de 'categorical_cols'
    # Guardado dentro del propio objeto (hay varias formas; aquí intentamos atributo 'feature_names_in_')
    cat_cols = getattr(ohe, "feature_names_in_", None)
    if cat_cols is None:
        raise AttributeError("El OneHotEncoder no expone 'feature_names_in_'. Guarde columnas en el objeto al entrenar.")
    num_cols = [c for c in df_clin.columns if c not in cat_cols]

    # Asegura orden de columnas esperado
    df_num = df_clin[num_cols].copy()
    df_cat = df_clin[cat_cols].copy()

    X_num = num_pipe.transform(df_num)
    X_cat = ohe.transform(df_cat)
    # Concatena
    if hasattr(X_num, "toarray"):
        X_num = X_num.toarray()
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()
    X = np.hstack([X_num, X_cat])
    return X
