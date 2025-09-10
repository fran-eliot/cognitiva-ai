# app.py — Streamlit UI para CognitivaAI Intermodal (P26/P27)
import streamlit as st
import pandas as pd
import numpy as np
import json, io
from pathlib import Path
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="CognitivaAI Intermodal", layout="wide")

# --- Config ---
MODELS_DIR = Path("p26_release/models")
CONFIG_PATH = Path("p26_release/CONFIG/deployment_config.json")

def load_models():
    m = {}
    m["p24"] = joblib.load(MODELS_DIR/"p24_model.pkl")
    try:
        m["platt"] = joblib.load(MODELS_DIR/"p24_platt.pkl")
    except:
        m["platt"] = None
    m["clin"] = joblib.load(MODELS_DIR/"p26_clinical_model.pkl")
    return m

def load_config():
    cfg = json.loads(CONFIG_PATH.read_text())
    thr = cfg.get("thresholds", {})
    thr = {str(k).upper(): float(v) for k,v in thr.items()}
    return thr, cfg

def sex_to_num(s):
    s = s.astype(str).str.upper().str.strip()
    return s.map({"M":1,"F":0,"MALE":1,"FEMALE":0})

def prepare_clinic(df, model):
    cols = getattr(model, "feature_names_in_", None)
    d = df.copy()
    if "Sex" in d.columns and "Sex_M" not in d.columns and "Sex_F" not in d.columns:
        d["Sex"] = sex_to_num(d["Sex"])
    if cols is not None:
        for c in cols:
            if c not in d.columns: d[c] = np.nan
        X = d[cols].copy()
    else:
        use = [c for c in ["Age","Sex","Education","SES","MMSE","eTIV","nWBV","ASF","Delay"] if c in d.columns]
        X = d[use].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype(float)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
    return X

def compute_pimg(features_df, models):
    id_col, coh_col = "patient_id", "cohort"
    if id_col not in features_df or coh_col not in features_df:
        raise ValueError("El CSV de features debe incluir 'patient_id' y 'cohort'.")
    expected = getattr(models["p24"], "feature_names_in_", None)
    if expected is None:
        blacklist = {id_col, coh_col}
        expected = [c for c in features_df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(features_df[c])]
    X = features_df.copy()
    for c in expected:
        if c not in X.columns: X[c] = np.nan
    X = X[expected]
    proba_raw = models["p24"].predict_proba(X)[:,1]
    if models["platt"] is not None:
        try:
            p_img = models["platt"].predict_proba(proba_raw.reshape(-1,1))[:,1]
        except:
            try:
                p_img = models["platt"].predict_proba(X)[:,1]
            except:
                p_img = proba_raw
    else:
        p_img = proba_raw
    out = features_df[[id_col, coh_col]].copy()
    out[coh_col] = out[coh_col].astype(str).str.upper()
    out["p_img"] = p_img
    return out

def fuse_and_decide(pimg_df, clinic_df, models, thresholds, y_true_col=None):
    id_col, coh_col = "patient_id", "cohort"
    df = clinic_df.merge(pimg_df, on=[id_col, coh_col], how="inner")
    Xc = prepare_clinic(df, models["clin"])
    try:
        p_clin = models["clin"].predict_proba(Xc)[:,1]
    except:
        Xs = pd.DataFrame(StandardScaler().fit_transform(Xc), columns=Xc.columns, index=Xc.index)
        wh = np.tanh(Xs.sum(axis=1).values)
        p_clin = 1/(1+np.exp(-wh))
    proba = (df["p_img"].values + p_clin)/2
    thr = df[coh_col].map(lambda c: float(thresholds.get(str(c).upper(), 0.5))).values
    decision = (proba >= thr).astype(int)
    out = df[[id_col, coh_col]].copy()
    out["p_img"] = df["p_img"].values
    out["p_clin"] = p_clin
    out["proba_cal"] = proba
    out["thr"] = thr
    out["decision"] = decision
    if y_true_col and (y_true_col in df.columns):
        out["y_true"] = df[y_true_col].values
    return out

st.title("🧠 CognitivaAI — Intermodal (P26/P27)")
st.markdown("Genera predicciones **imagen+clínico** y aplica la **política S2** (umbrales por cohorte).")

with st.sidebar:
    st.header("Configuración")
    models_ok = MODELS_DIR.exists() and (MODELS_DIR/"p24_model.pkl").exists() and (MODELS_DIR/"p26_clinical_model.pkl").exists()
    config_ok = CONFIG_PATH.exists()
    st.write("Models dir:", MODELS_DIR)
    st.write("Config:", CONFIG_PATH)
    if not models_ok or not config_ok:
        st.error("Coloca los modelos y la config en p26_release/ antes de usar.")
    else:
        st.success("Modelos y config detectados ✔")

tab1, tab2 = st.tabs(["Predicción", "Descargas"])

with tab1:
    st.subheader("Entradas")
    feat_file = st.file_uploader("CSV de features por paciente (incluye patient_id, cohort, 56 cols)", type=["csv"], key="feat")
    clin_file = st.file_uploader("CSV clínico (incluye patient_id, cohort, columnas mínimas)", type=["csv"], key="clin")
    ycol = st.text_input("Nombre de columna con etiqueta (opcional)", value="")
    run_btn = st.button("🚀 Ejecutar")
    if run_btn:
        if not feat_file or not clin_file:
            st.warning("Sube ambos CSV.")
        else:
            features_df = pd.read_csv(feat_file)
            clinic_df = pd.read_csv(clin_file)
            try:
                models = load_models()
                thresholds, cfg = load_config()
                pimg_df = compute_pimg(features_df, models)
                out = fuse_and_decide(pimg_df, clinic_df, models, thresholds, y_true_col=ycol if ycol else None)
                st.success(f"Predicciones generadas: {out.shape}")
                st.dataframe(out.head(20))
                # descargas
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar predictions.csv", data=csv, file_name="predictions.csv", mime="text/csv")
                # QA
                if "y_true" in out.columns:
                    def qa(dfq):
                        tp = int(((dfq["decision"]==1)&(dfq["y_true"]==1)).sum())
                        fp = int(((dfq["decision"]==1)&(dfq["y_true"]==0)).sum())
                        tn = int(((dfq["decision"]==0)&(dfq["y_true"]==0)).sum())
                        fn = int(((dfq["decision"]==0)&(dfq["y_true"]==1)).sum())
                        prec = tp/(tp+fp) if (tp+fp)>0 else np.nan
                        rec  = tp/(tp+fn) if (tp+fn)>0 else np.nan
                        acc  = (tp+tn)/len(dfq) if len(dfq)>0 else np.nan
                        cost = 5*fn + 1*fp
                        return dict(TP=tp,FP=fp,TN=tn,FN=fn,Precision=prec,Recall=rec,Acc=acc,Cost=cost)
                    rows=[]
                    rows.append(dict(Cohort="ALL", **qa(out)))
                    for coh in sorted(out["cohort"].unique()):
                        rows.append(dict(Cohort=str(coh), **qa(out[out["cohort"]==coh])))
                    qa_df = pd.DataFrame(rows)
                    st.markdown("### QA (si hay etiqueta)")
                    st.dataframe(qa_df)
                    qa_csv = qa_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Descargar QA.csv", data=qa_csv, file_name="QA.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)

with tab2:
    st.markdown("**Ficheros esperados en `p26_release/`:**")
    st.code("""p26_release/
 ├─ CONFIG/deployment_config.json
 ├─ models/p24_model.pkl
 ├─ models/p24_platt.pkl
 └─ models/p26_clinical_model.pkl
""")
