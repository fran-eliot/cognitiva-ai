#!/usr/bin/env python3
# predict_end_to_end.py
# Fusión LATE de p_img + p_clin, calibración ya incluida en p_img (Platt), decisión con S2.
import argparse, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REQ_CLIN = ["Age","Sex","Education","SES","MMSE","eTIV","nWBV","ASF","Delay"]

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")

def _sex_to_num(series: pd.Series):
    s = series.astype(str).str.upper().str.strip()
    return s.map({"M":1,"F":0,"MALE":1,"FEMALE":0}).fillna(np.nan)

def _prepare_clinic(df: pd.DataFrame, model) -> pd.DataFrame:
    # intenta alinear con feature_names_in_ del modelo clínico si existe
    expected = getattr(model, "feature_names_in_", None)
    d = df.copy()

    # Crear Sex numérico si solo tenemos 'Sex'
    if "Sex" in d.columns and "Sex_M" not in d.columns and "Sex_F" not in d.columns:
        d["Sex"] = _sex_to_num(d["Sex"])

    # Si el modelo espera columnas concretas, respétalas
    if expected is not None:
        for col in expected:
            if col not in d.columns:
                d[col] = np.nan
        X = d[expected].copy()
    else:
        # fallback: usar columnas mínimas
        missing = [c for c in REQ_CLIN if c not in d.columns]
        if missing:
            print(f"⚠️ Faltan columnas clínicas mínimas: {missing}", file=sys.stderr)
        use = [c for c in REQ_CLIN if c in d.columns]
        X = d[use].copy()
    # imputación segura (por si el pipeline no la trae)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(float)
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    return X

def _load_thresholds(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    # prioriza cfg["thresholds"], si existe
    thr = cfg.get("thresholds", {})
    # normaliza keys
    return {str(k).upper(): float(v) for k,v in thr.items()}

def main():
    ap = argparse.ArgumentParser(description="End-to-End prediction: p_img + clinical (LATE) + S2 decision.")
    ap.add_argument("--pimg", required=True, help="CSV con patient_id, cohort, p_img")
    ap.add_argument("--clinic", required=True, help="CSV clínico consolidado (incluye patient_id, cohort, columnas mínimas)")
    ap.add_argument("--models_dir", default="p26_release/models", help="Carpeta con p26_clinical_model.pkl")
    ap.add_argument("--config", default="p26_release/CONFIG/deployment_config.json", help="JSON con thresholds por cohorte")
    ap.add_argument("--out", required=True, help="CSV de salida con p_img, p_clin, proba_cal, thr, decision")
    ap.add_argument("--y_true_col", default=None, help="Nombre de columna con etiqueta (0/1) si está disponible")
    ap.add_argument("--fusion", default="mean", choices=["mean"], help="Estrategia de fusión (por ahora 'mean')")
    args = ap.parse_args()

    pimg_path   = Path(args.pimg)
    clinic_path = Path(args.clinic)
    models_dir  = Path(args.models_dir)
    config_path = Path(args.config)
    out_path    = Path(args.out)

    if not pimg_path.exists():
        print(f"❌ No existe p_img CSV: {pimg_path}", file=sys.stderr); sys.exit(1)
    if not clinic_path.exists():
        print(f"❌ No existe clínico CSV: {clinic_path}", file=sys.stderr); sys.exit(1)
    if not config_path.exists():
        print(f"❌ Falta config JSON: {config_path}", file=sys.stderr); sys.exit(1)

    # Carga datos
    pimg = _read_csv(pimg_path)
    clin = _read_csv(clinic_path)

    for col in ["patient_id","cohort"]:
        if col not in pimg.columns or col not in clin.columns:
            print(f"❌ Ambos CSV deben incluir {col}", file=sys.stderr); sys.exit(1)

    # Modelo clínico
    clin_model_path = models_dir/"p26_clinical_model.pkl"
    if not clin_model_path.exists():
        print(f"❌ Falta modelo clínico: {clin_model_path}", file=sys.stderr); sys.exit(1)
    model_clin = joblib.load(clin_model_path)

    # Merge por patient_id
    pimg["cohort"] = pimg["cohort"].astype(str).str.upper()
    clin["cohort"] = clin["cohort"].astype(str).str.upper()
    df = clin.merge(pimg[["patient_id","cohort","p_img"]], on=["patient_id","cohort"], how="inner")

    # y_true opcional
    y_true = None
    if args.y_true_col and args.y_true_col in df.columns:
        y_true = df[args.y_true_col].astype(float)

    # p_clin
    Xc = _prepare_clinic(df, model_clin)
    try:
        p_clin = model_clin.predict_proba(Xc)[:,1]
    except Exception as e:
        print(f"⚠️ predict_proba del modelo clínico falló: {e} → uso LR-like fallback (sigmoide de z-score)", file=sys.stderr)
        # fallback muy conservador
        Xs = pd.DataFrame(StandardScaler().fit_transform(Xc), columns=Xc.columns, index=Xc.index)
        wh = np.tanh(Xs.sum(axis=1).values)  # pseudo-score
        p_clin = 1/(1+np.exp(-wh))

    # Fusión (por ahora media)
    p_img_vals = df["p_img"].astype(float).values
    if args.fusion == "mean":
        proba = (p_img_vals + p_clin) / 2.0
    else:
        proba = (p_img_vals + p_clin) / 2.0

    # Decisión S2
    thresholds = _load_thresholds(config_path)
    def thr_for(coh: str) -> float:
        return float(thresholds.get(str(coh).upper(), 0.5))

    thr = df["cohort"].map(thr_for).astype(float).values
    decision = (proba >= thr).astype(int)

    out = df[["patient_id","cohort"]].copy()
    out["p_img"] = p_img_vals
    out["p_clin"] = p_clin
    out["proba_cal"] = proba
    out["thr"] = thr
    out["decision"] = decision
    if y_true is not None:
        out["y_true"] = y_true.values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✅ Guardado: {out_path} → {out.shape}")

    # QA opcional
    if "y_true" in out.columns:
        def qa(dfq):
            tp = int(((dfq["decision"]==1)&(dfq["y_true"]==1)).sum())
            fp = int(((dfq["decision"]==1)&(dfq["y_true"]==0)).sum())
            tn = int(((dfq["decision"]==0)&(dfq["y_true"]==0)).sum())
            fn = int(((dfq["decision"]==0)&(dfq["y_true"]==1)).sum())
            prec = tp/(tp+fp) if (tp+fp)>0 else np.nan
            rec  = tp/(tp+fn) if (tp+fn)>0 else np.nan
            acc  = (tp+tn)/len(dfq) if len(dfq)>0 else np.nan
            # Coste por defecto 5:1
            cost = 5*fn + 1*fp
            return dict(TP=tp,FP=fp,TN=tn,FN=fn,Precision=prec,Recall=rec,Acc=acc,Cost=cost)

        rows=[]
        rows.append(dict(Cohort="ALL", **qa(out)))
        for coh in sorted(out["cohort"].unique()):
            rows.append(dict(Cohort=str(coh), **qa(out[out["cohort"]==coh])))
        qa_path = out_path.with_name(out_path.stem + "_qa.csv")
        pd.DataFrame(rows).to_csv(qa_path, index=False)
        print(f"🧪 QA guardado: {qa_path}")

if __name__ == "__main__":
    main()
