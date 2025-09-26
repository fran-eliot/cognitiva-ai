
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception as e:
    st.error("Falta 'joblib'. Ejecuta: pip install joblib")
    raise

from sklearn import metrics

st.set_page_config(page_title="CognitivaAI — Intermodal Demo", layout="wide")
st.title("🧪 CognitivaAI — Intermodal (Imagen + Clínico)")
st.caption("Fusión LATE + Política por cohorte • Modo Demo • Métricas y costes")

MODELS_DIR = os.environ.get("MODELS_DIR", "p26_release/models")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "p26_release/CONFIG/deployment_config.json")

@st.cache_resource(show_spinner=False)
def load_models_and_policy(models_dir: str, config_path: str):
    p24_model = joblib.load(os.path.join(models_dir, "p24_model.pkl"))
    try:
        p24_platt = joblib.load(os.path.join(models_dir, "p24_platt.pkl"))
    except Exception:
        p24_platt = None
    p26_clin = joblib.load(os.path.join(models_dir, "p26_clinical_model.pkl"))
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        thr = cfg.get("thresholds", cfg)
    else:
        thr = {"OAS1": 0.42, "OAS2": 0.4928655287824083}
    return p24_model, p24_platt, p26_clin, thr

p24_model, p24_platt, p26_clin, S2_thr = load_models_and_policy(MODELS_DIR, CONFIG_PATH)

st.sidebar.header("⚙️ Política y Coste")
policy = st.sidebar.radio("Política:", ["S2 (archivo JSON)", "Manual"], index=0)
ratio = st.sidebar.radio("Coste FN:FP", ["3:1", "5:1", "7:1", "10:1"], index=1)
_ratio_map = {"3:1":3.0, "5:1":5.0, "7:1":7.0, "10:1":10.0}
C_FN, C_FP = _ratio_map[ratio], 1.0

if policy == "S2 (archivo JSON)":
    thr_oas1 = float(S2_thr.get("OAS1", 0.5))
    thr_oas2 = float(S2_thr.get("OAS2", 0.5))
else:
    thr_oas1 = st.sidebar.slider("Umbral OAS1", 0.0, 1.0, float(S2_thr.get("OAS1", 0.42)), 0.001)
    thr_oas2 = st.sidebar.slider("Umbral OAS2", 0.0, 1.0, float(S2_thr.get("OAS2", 0.4928655287824083)), 0.001)

use_api = st.sidebar.checkbox("Comparar con FastAPI (POST /predict)")
api_url = st.sidebar.text_input("URL FastAPI", value="http://localhost:8000/predict", help="Deja por defecto si no usas API")

def align_columns(df: pd.DataFrame, model):
    X = df.copy()
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        for c in cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[cols]
    return X

def calc_pimg_from_features(df_feat: pd.DataFrame) -> np.ndarray:
    X = align_columns(df_feat, p24_model)
    proba_raw = p24_model.predict_proba(X)[:,1]
    if p24_platt is not None:
        try:
            return p24_platt.predict_proba(proba_raw.reshape(-1,1))[:,1]
        except Exception:
            try:
                return p24_platt.predict_proba(X)[:,1]
            except Exception:
                return proba_raw
    return proba_raw

def calc_pclin(df_clin: pd.DataFrame) -> np.ndarray:
    df = df_clin.copy()
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.upper().str[0].map({"M":1.0, "F":0.0})
    X = align_columns(df, p26_clin).fillna(df.median(numeric_only=True))
    return p26_clin.predict_proba(X)[:,1]

def fuse_late(p_img: np.ndarray, p_clin: np.ndarray) -> np.ndarray:
    return (p_img + p_clin) / 2.0

def decide_threshold(cohort: str) -> float:
    return thr_oas1 if cohort == "OAS1" else thr_oas2

def confusion_cost(df_pred: pd.DataFrame, c_fn=5.0, c_fp=1.0) -> pd.DataFrame:
    rows = []
    for coh, thr in [("OAS1", thr_oas1), ("OAS2", thr_oas2)]:
        sub = df_pred[df_pred["cohort"]==coh].copy()
        if sub.empty or "y_true" not in sub.columns or sub["y_true"].isna().all():
            continue
        y = sub["y_true"].astype(int).values
        yhat = (sub["proba_cal"] >= thr).astype(int).values
        TP = int(((y==1)&(yhat==1)).sum()); FP = int(((y==0)&(yhat==1)).sum())
        TN = int(((y==0)&(yhat==0)).sum()); FN = int(((y==1)&(yhat==0)).sum())
        prec = TP/(TP+FP) if (TP+FP)>0 else np.nan
        rec  = TP/(TP+FN) if (TP+FN)>0 else np.nan
        acc  = (TP+TN)/len(sub) if len(sub)>0 else np.nan
        cost = c_fn*FN + c_fp*FP
        rows.append(dict(Cohort=coh, Thr=thr, TP=TP, FP=FP, TN=TN, FN=FN,
                         Precision=prec, Recall=rec, Acc=acc, Cost=cost))
    return pd.DataFrame(rows)

def cost_curve(df_pred: pd.DataFrame, cohort: str, c_fn=5.0, c_fp=1.0, n=101) -> pd.DataFrame:
    if "y_true" not in df_pred.columns or df_pred["y_true"].isna().all():
        return pd.DataFrame()
    sub = df_pred[df_pred["cohort"]==cohort].copy()
    if sub.empty:
        return pd.DataFrame()
    thrs = np.linspace(0,1,n)
    y = sub["y_true"].astype(int).values
    p = sub["proba_cal"].values
    data = []
    for t in thrs:
        yhat = (p >= t).astype(int)
        TP = ((y==1)&(yhat==1)).sum(); FP = ((y==0)&(yhat==1)).sum()
        TN = ((y==0)&(yhat==0)).sum(); FN = ((y==1)&(yhat==0)).sum()
        cost = c_fn*FN + c_fp*FP
        data.append((t, cost))
    return pd.DataFrame(data, columns=["thr","cost"]).set_index("thr")

def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    auc = metrics.auc(fpr, tpr)
    prec, rec, _ = metrics.precision_recall_curve(y_true, y_prob)
    prauc = metrics.auc(rec, prec)
    brier = metrics.brier_score_loss(y_true, y_prob)
    return {"AUC": auc, "PRAUC": prauc, "Brier": brier}

def reliability(df_pred: pd.DataFrame, cohort: str, bins=10):
    if "y_true" not in df_pred.columns or df_pred["y_true"].isna().all():
        return pd.DataFrame(), np.nan, np.nan
    sub = df_pred[df_pred["cohort"]==cohort].copy()
    if sub.empty:
        return pd.DataFrame(), np.nan, np.nan
    p = sub["proba_cal"].values
    y = sub["y_true"].astype(int).values
    bin_ids = np.clip((p * bins).astype(int), 0, bins-1)
    dfb = pd.DataFrame({"bin": bin_ids, "p": p, "y": y})
    grp = dfb.groupby("bin").agg(p_mean=("p","mean"), y_rate=("y","mean"), n=("y","size")).reset_index()
    ece = np.nansum(np.abs(grp["p_mean"] - grp["y_rate"]) * grp["n"]) / np.nansum(grp["n"])
    mce = np.nanmax(np.abs(grp["p_mean"] - grp["y_rate"])) if len(grp)>0 else np.nan
    grp = grp.set_index("p_mean")[["y_rate"]]
    return grp, ece, mce

tab_data, tab_results, tab_metrics, tab_cost, tab_settings = st.tabs(
    ["📥 Datos", "📊 Resultados", "📈 Métricas", "💸 Coste vs Umbral", "🔧 Ajustes"]
)

with tab_data:
    st.subheader("Entrada de datos")
    c1, c2 = st.columns(2)
    with c1:
        feat_file = st.file_uploader("CSV • Features por paciente (de P24)", type=["csv"])
    with c2:
        clin_file = st.file_uploader("CSV • Clínico (P26)", type=["csv"])

    demo = st.button("Cargar datos de ejemplo (10 pacientes)")
    if demo:
        cols = list(getattr(p24_model, "feature_names_in_", []))
        Xdemo = pd.DataFrame(np.random.rand(10, len(cols)), columns=cols)
        Xdemo.insert(0, "cohort", ["OAS1"]*5 + ["OAS2"]*5)
        Xdemo.insert(0, "patient_id", [f"DEMO_{i:03d}" for i in range(10)])
        st.session_state["demo_features"] = Xdemo

        clin = pd.DataFrame({
            "patient_id":[f"DEMO_{i:03d}" for i in range(10)],
            "cohort":["OAS1"]*5 + ["OAS2"]*5,
            "Age":np.random.randint(60,90,10),
            "Sex":np.random.choice(["M","F"],10),
            "Education":np.random.randint(8,20,10),
            "SES":np.random.randint(1,5,10),
            "MMSE":np.random.randint(15,30,10),
            "eTIV":np.random.normal(1500,100,10),
            "nWBV":np.clip(np.random.normal(0.72,0.05,10), 0.5, 0.9),
            "ASF":np.random.normal(1.0,0.05,10),
            "Delay":0,
            "y_true":(np.random.rand(10) > 0.5).astype(int)
        })
        st.session_state["demo_clinical"] = clin
        st.success("Datos de ejemplo listos. Ve a 'Resultados' y pulsa Ejecutar.")

    preview_cols = st.columns(2)
    with preview_cols[0]:
        if "demo_features" in st.session_state:
            st.markdown("**Features (demo)**")
            st.dataframe(st.session_state["demo_features"].head())
        elif feat_file is not None:
            df_feat_head = pd.read_csv(feat_file, nrows=5)
            st.markdown("**Features (preview)**")
            st.dataframe(df_feat_head)
    with preview_cols[1]:
        if "demo_clinical" in st.session_state:
            st.markdown("**Clínico (demo)**")
            st.dataframe(st.session_state["demo_clinical"].head())
        elif clin_file is not None:
            df_clin_head = pd.read_csv(clin_file, nrows=5)
            st.markdown("**Clínico (preview)**")
            st.dataframe(df_clin_head)

with tab_results:
    st.subheader("Resultados de predicción")
    run = st.button("Ejecutar")

    if run:
        base_cols = ["patient_id","cohort"]

        if "demo_features" in st.session_state:
            df_feat = st.session_state["demo_features"].copy()
        elif feat_file is not None:
            df_feat = pd.read_csv(feat_file)
        else:
            st.warning("Sube el CSV de features o usa el botón de demo.")
            st.stop()

        if "demo_clinical" in st.session_state:
            df_clin = st.session_state["demo_clinical"].copy()
        elif clin_file is not None:
            df_clin = pd.read_csv(clin_file)
        else:
            st.warning("Sube el CSV clínico o usa el botón de demo.")
            st.stop()

        for need in base_cols:
            if need not in df_feat.columns:
                st.error(f"Features: falta columna '{need}'."); st.stop()
            if need not in df_clin.columns:
                st.error(f"Clínico: falta columna '{need}'."); st.stop()

        df = df_clin[base_cols].merge(df_feat, on=base_cols, how="inner")
        if df.empty:
            st.error("No hay intersección por (patient_id, cohort) entre clínico y features."); st.stop()

        drop_cols = base_cols + [c for c in ["y_true","Age","Sex","Education","SES","MMSE","eTIV","nWBV","ASF","Delay"] if c in df.columns]
        Ximg = df.drop(columns=drop_cols, errors="ignore")

        p_img = calc_pimg_from_features(Ximg)
        p_clin = calc_pclin(df_clin.loc[df.index, :])
        proba_cal = fuse_late(p_img, p_clin)

        df_out = df[base_cols].copy()
        df_out["p_img"] = p_img
        df_out["p_clin"] = p_clin
        df_out["proba_cal"] = proba_cal
        df_out["thr_used"] = [decide_threshold(c) for c in df_out["cohort"]]
        df_out["decision"] = (df_out["proba_cal"] >= df_out["thr_used"]).astype(int)

        if "y_true" in df_clin.columns:
            df_out = df_out.merge(df_clin[base_cols+["y_true"]], on=base_cols, how="left")

        k1, k2, k3 = st.columns(3)
        if "y_true" in df_out.columns and df_out["y_true"].notna().any():
            y = df_out["y_true"].astype(int).values
            k1.metric("Casos", len(df_out)); k2.metric("Positivos", int((y==1).sum())); k3.metric("Negativos", int((y==0).sum()))
        else:
            k1.metric("Casos", len(df_out)); k2.metric("Positivos", "—"); k3.metric("Negativos", "—")

        st.dataframe(df_out)

        st.download_button("⬇️ Descargar predicciones (CSV)", data=df_out.to_csv(index=False).encode("utf-8"),
                           file_name="predicciones_intermodal.csv", mime="text/csv")

        if use_api:
            try:
                import requests
                payload = {"cases":[]}
                feat_cols = list(getattr(p24_model, "feature_names_in_", []))
                for idx, row in df_out.iterrows():
                    pid, coh = row["patient_id"], row["cohort"]
                    clin_row = df_clin.loc[df_clin["patient_id"].eq(pid) & df_clin["cohort"].eq(coh)].copy()
                    clin_payload = clin_row.iloc[0][["patient_id","cohort","Age","Sex","Education","SES","MMSE","eTIV","nWBV","ASF","Delay"]].dropna().to_dict()
                    feat_sub = df.loc[df["patient_id"].eq(pid) & df["cohort"].eq(coh), feat_cols] if all(c in df.columns for c in feat_cols) else pd.DataFrame()
                    if not feat_sub.empty:
                        features_payload = feat_sub.iloc[0].dropna().to_dict()
                        payload["cases"].append({"clinical":clin_payload, "features":features_payload})
                    else:
                        payload["cases"].append({"clinical":clin_payload, "p_img": float(row["p_img"])})
                res = requests.post(api_url, json=payload, timeout=30)
                api_out = pd.DataFrame(res.json())
                st.markdown("**Comparativa con FastAPI (si está corriendo):**"); st.dataframe(api_out)
                if "proba_cal" in api_out.columns:
                    merged = df_out.merge(api_out[["patient_id","cohort","proba_cal"]], on=["patient_id","cohort"], suffixes=("","_api"), how="inner")
                    if not merged.empty:
                        merged["abs_diff"] = (merged["proba_cal"] - merged["proba_cal_api"]).abs()
                        st.write("MAE:", merged["abs_diff"].mean())
            except Exception as e:
                st.info(f"No se pudo consultar la API: {e}")

        st.session_state["pred_df"] = df_out

with tab_metrics:
    st.subheader("Métricas por cohorte")
    if "pred_df" not in st.session_state:
        st.info("Primero ejecuta predicciones en la pestaña 'Resultados'.")
    else:
        df_out = st.session_state["pred_df"]
        if "y_true" in df_out.columns and df_out["y_true"].notna().any():
            rows = []
            for coh in ["ALL","OAS1","OAS2"]:
                sub = df_out if coh=="ALL" else df_out[df_out["cohort"]==coh]
                if sub.empty: continue
                y = sub["y_true"].astype(int).values
                p = sub["proba_cal"].values
                m = eval_metrics(y, p); rows.append(dict(Cohort=coh, **m))
            metr_df = pd.DataFrame(rows); st.dataframe(metr_df)

            conf = confusion_cost(df_out, C_FN, C_FP); st.dataframe(conf)

            c1, c2 = st.columns(2)
            for coh, col in zip(["OAS1","OAS2"], [c1, c2]):
                rel, ece, mce = reliability(df_out, coh, bins=10)
                with col:
                    st.markdown(f"**{coh} — Reliability (ECE={ece:.3f}, MCE={mce:.3f})**" if isinstance(ece, float) else f"**{coh} — Reliability**")
                    if not rel.empty:
                        st.line_chart(rel.rename(columns={"y_rate":"Observed"}))
                    else:
                        st.caption("Sin etiquetas para calcular calibración.")
        else:
            st.info("No hay `y_true`. Sube un CSV clínico con columna `y_true` para ver métricas.")

with tab_cost:
    st.subheader("Curva de coste vs umbral")
    if "pred_df" not in st.session_state:
        st.info("Primero ejecuta predicciones en la pestaña 'Resultados'.")
    else:
        df_out = st.session_state["pred_df"]
        c1, c2 = st.columns(2)
        for coh, col in zip(["OAS1","OAS2"], [c1, c2]):
            curve = cost_curve(df_out, coh, C_FN, C_FP, n=101)
            with col:
                st.markdown(f"**{coh} — Coste ({int(C_FN)}:{int(C_FP)})**")
                if not curve.empty:
                    st.line_chart(curve)
                else:
                    st.caption("Sin etiquetas para calcular coste.")

with tab_settings:
    st.subheader("Ajustes y utilidades")
    st.write("Política actual:", "S2 (JSON)" if policy.startswith("S2") else "Manual")
    st.write("Umbrales en uso → OAS1:", thr_oas1, " | OAS2:", thr_oas2)
    st.write("Coste FN:FP →", f"{int(C_FN)}:{int(C_FP)}")
    if policy == "Manual":
        if st.button("💾 Guardar umbrales en deployment_config.json"):
            cfg = {"policy":"Manual","thresholds":{"OAS1": float(thr_oas1), "OAS2": float(thr_oas2)}}
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            st.success(f"Guardado en {CONFIG_PATH}")
    st.caption("Consejo: emplea los mismos pickles y versión sklearn (1.7.1) que en entrenamiento para evitar incompatibilidades.")
