from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib, json, os

# Rutas por defecto (ajusta si cambian)
MODELS_DIR = os.environ.get("MODELS_DIR", "p26_release/models")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "p26_release/CONFIG/deployment_config.json")

# Carga modelos (P24 para imagen; P26 para clínico)
p24_model = joblib.load(os.path.join(MODELS_DIR, "p24_model.pkl"))
p24_platt = None
try:
    p24_platt = joblib.load(os.path.join(MODELS_DIR, "p24_platt.pkl"))
except:
    pass

p26_clin = joblib.load(os.path.join(MODELS_DIR, "p26_clinical_model.pkl"))

# Carga config (S2)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    S2 = json.load(f)

# ---- Esquemas de entrada ----

class Clinical(BaseModel):
    patient_id: str
    cohort: str
    Age: Optional[float] = None
    Sex: Optional[str] = None
    Education: Optional[float] = None
    SES: Optional[float] = None
    MMSE: Optional[float] = None
    eTIV: Optional[float] = None
    nWBV: Optional[float] = None
    ASF: Optional[float] = None
    Delay: Optional[float] = None

class Case(BaseModel):
    clinical: Clinical
    # Si no aportas 'p_img', debes aportar 'features' con las 56 columnas de P24
    p_img: Optional[float] = None
    features: Optional[Dict[str, float]] = None

class BatchRequest(BaseModel):
    cases: List[Case]

class PredictOut(BaseModel):
    patient_id: str
    cohort: str
    p_img: float
    p_clin: float
    proba_cal: float
    thr: float
    decision: int

app = FastAPI(title="CognitivaAI Intermodal API", version="1.0")

# ---- Utilidades ----

def _calc_pimg_from_features(feat_row: Dict[str, float]) -> float:
    X = pd.DataFrame([feat_row])
    # Alinear columnas si el modelo las expone
    if hasattr(p24_model, "feature_names_in_"):
        cols = list(p24_model.feature_names_in_)
        for c in cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[cols]
    proba_raw = p24_model.predict_proba(X)[:, 1]
    if p24_platt is not None:
        try:
            return float(p24_platt.predict_proba(proba_raw.reshape(-1,1))[:,1][0])
        except:
            # fallback: algunos calibradores se entrenaron sobre X
            return float(p24_platt.predict_proba(X)[:,1][0])
    return float(proba_raw[0])

def _calc_pclin_from_clinical(c: Clinical) -> float:
    df = pd.DataFrame([c.dict()])
    # mapear Sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.upper().str[0].map({"M":1.0,"F":0.0})
    # columnas esperadas si existen
    if hasattr(p26_clin, "feature_names_in_"):
        cols = list(p26_clin.feature_names_in_)
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        Xc = df[cols]
    else:
        # usar numéricas disponibles
        Xc = df.select_dtypes(include=[np.number])
    # imputación simple (mediana)
    Xc = Xc.fillna(Xc.median(numeric_only=True))
    return float(p26_clin.predict_proba(Xc)[:,1][0])

def _threshold_for(cohort: str) -> float:
    # S2 tiene umbrales por cohorte
    # Soporta tanto formato plano {"OAS1":thr1,"OAS2":thr2} como nested {"policy":"...","thresholds":{...}}
    if isinstance(S2, dict) and "thresholds" in S2:
        return float(S2["thresholds"].get(cohort, 0.5))
    return float(S2.get(cohort, 0.5))

# ---- Endpoints ----

@app.get("/health")
def health():
    return {"status":"ok","policy":"S2","thresholds": S2["thresholds"] if "thresholds" in S2 else S2}

@app.post("/predict", response_model=List[PredictOut])
def predict(req: BatchRequest):
    out = []
    for case in req.cases:
        pid = case.clinical.patient_id
        coh = case.clinical.cohort
        # p_img
        if case.p_img is not None:
            p_img = float(case.p_img)
        elif case.features is not None:
            p_img = _calc_pimg_from_features(case.features)
        else:
            raise ValueError(f"Case {pid}: aportar 'p_img' o 'features'.")
        # p_clin
        p_clin = _calc_pclin_from_clinical(case.clinical)
        # fusion late (media)
        proba = (p_img + p_clin) / 2.0
        thr = _threshold_for(coh)
        decision = int(proba >= thr)
        out.append(PredictOut(
            patient_id=pid, cohort=coh, p_img=p_img, p_clin=p_clin,
            proba_cal=proba, thr=thr, decision=decision
        ))
    return out
