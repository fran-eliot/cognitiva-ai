#!/usr/bin/env python3
# compute_pimg_from_features.py
# Genera p_img (probabilidad de imagen calibrada) desde features por paciente (56 cols).
import argparse, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.exceptions import NotFittedError

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")

def _expected_features_from(model):
    # intenta extraer las columnas esperadas del pipeline / estimator
    for obj in [model, getattr(model, "named_steps", None)]:
        if hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
    # intenta en steps (scikit Pipeline)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def _align_columns(df: pd.DataFrame, expected: list[str]):
    # crea cualquier columna faltante con NaN y respeta el orden
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df[expected]

def main():
    ap = argparse.ArgumentParser(description="Compute p_img from patient features (P24).")
    ap.add_argument("--features", required=True, help="CSV con features por paciente (incluye patient_id, cohort).")
    ap.add_argument("--models_dir", default="p26_release/models", help="Carpeta con p24_model.pkl y p24_platt.pkl")
    ap.add_argument("--out", required=True, help="CSV de salida con patient_id, cohort, p_img")
    ap.add_argument("--id_col", default="patient_id")
    ap.add_argument("--cohort_col", default="cohort")
    args = ap.parse_args()

    features_path = Path(args.features)
    models_dir = Path(args.models_dir)
    out_path = Path(args.out)

    if not features_path.exists():
        print(f"❌ No existe features CSV: {features_path}", file=sys.stderr); sys.exit(1)

    model_path = models_dir/"p24_model.pkl"
    platt_path = models_dir/"p24_platt.pkl"
    if not model_path.exists():
        print(f"❌ Falta modelo P24: {model_path}", file=sys.stderr); sys.exit(1)
    if not platt_path.exists():
        print(f"⚠️ No se encontró calibrador Platt: {platt_path} → se usará proba raw.", file=sys.stderr)

    print(f"📦 Modelo P24: {model_path}")
    print(f"📦 Calibrador Platt: {platt_path if platt_path.exists() else 'N/A'}")
    print(f"📄 Features: {features_path}")

    df = _read_csv(features_path)
    if args.id_col not in df.columns or args.cohort_col not in df.columns:
        print(f"❌ CSV debe incluir columnas {args.id_col} y {args.cohort_col}", file=sys.stderr); sys.exit(1)

    # Cargar modelos
    model = joblib.load(model_path)
    platt = joblib.load(platt_path) if platt_path.exists() else None

    # Determinar columnas esperadas
    expected = _expected_features_from(model)
    if expected is None:
        # fallback: todas numéricas excepto id/cohort
        blacklist = {args.id_col, args.cohort_col}
        expected = [c for c in df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(df[c])]
        print("⚠️ No se encontraron feature_names_in_; usando columnas numéricas detectadas:", len(expected))
    else:
        print("✅ Columnas esperadas (modelo):", len(expected))

    # Filtrar / alinear
    meta = df[[args.id_col, args.cohort_col]].copy()
    X = _align_columns(df.copy(), expected)

    # Predecir
    try:
        proba_raw = model.predict_proba(X)[:, 1]
    except NotFittedError:
        print("❌ El modelo no está entrenado (NotFittedError).", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"❌ Error en predict_proba del modelo: {e}", file=sys.stderr); sys.exit(1)

    # Calibrar
    if platt is not None:
        try:
            proba_cal = platt.predict_proba(proba_raw.reshape(-1,1))[:,1]
        except Exception:
            try:
                proba_cal = platt.predict_proba(X)[:,1]
            except Exception:
                print("⚠️ Calibrador falló; usando proba_raw.", file=sys.stderr)
                proba_cal = proba_raw
    else:
        proba_cal = proba_raw

    out = meta.copy()
    out["cohort"] = out["cohort"].astype(str).str.upper()
    out["p_img"] = proba_cal.astype(float)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✅ Guardado: {out_path} → {out.shape}")

if __name__ == "__main__":
    main()
