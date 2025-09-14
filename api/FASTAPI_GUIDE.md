
# FastAPI — Guía completa (CognitivaAI Intermodal)

API para servir predicciones **intermodales** (imagen + clínico) con **fusión LATE** y **política S2 por cohorte** (OAS1/OAS2). Replica la lógica de P26/P27.

---

## ¿Qué hace? (flujo de inferencia)
1) **Imagen → `p_img`**: recibes `p_img` **o** envías `features` (las 56 de P24), y la API calcula `p_img` con `p24_model.pkl`. Si existe, calibra con `p24_platt.pkl` (intenta primero `predict_proba(proba_raw.reshape(-1,1))`; si ese calibrador fue entrenado sobre `X`, cae a `predict_proba(X)`; si no, usa `proba_raw`).  
2) **Clínico → `p_clin`**: calcula con `p26_clinical_model.pkl` (mapea `Sex` M→1, F→0 e imputa faltantes por mediana).  
3) **Fusión LATE**: `proba_cal = (p_img + p_clin)/2`.  
4) **Decisión (S2)**: aplica umbrales por cohorte desde `CONFIG/deployment_config.json` → `decision = int(proba_cal >= thr_cohorte)`.

---

## Estructura de proyecto esperada
```
p26_release/
  ├─ CONFIG/
  │   └─ deployment_config.json     # {"OAS1":0.42,"OAS2":0.4928655287824083}
  ├─ models/
  │   ├─ p24_model.pkl              # imagen (LR elastic-net, P24)
  │   ├─ p24_platt.pkl              # calibrador Platt (opcional)
  │   └─ p26_clinical_model.pkl     # clínico (P26)
  └─ DOCS/                          # opcional (MODEL_CARD.md, etc.)
api/
  └─ main.py                        # FastAPI
```

### deployment_config.json (formatos válidos)
```json
{"OAS1": 0.42, "OAS2": 0.4928655287824083}
```
o
```json
{"policy":"S2","thresholds":{"OAS1":0.42,"OAS2":0.4928655287824083}}
```

---

## Requisitos e instalación
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install fastapi uvicorn pydantic pandas numpy joblib scikit-learn==1.7.1
```
> Fijamos **scikit-learn==1.7.1** para evitar incompatibilidades de pickles.

**Variables de entorno (opcionales)**  
- `MODELS_DIR` → ruta a `p26_release/models`  
- `CONFIG_PATH` → ruta a `p26_release/CONFIG/deployment_config.json`

---

## Inicio rápido
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Comprobar salud:
```bash
curl http://localhost:8000/health
# {"status":"ok","policy":"S2","thresholds":{"OAS1":0.42,"OAS2":0.4928655287824083}}
```

---

## Endpoints

### GET /health
Devuelve estado y umbrales activos.
```json
{"status":"ok","policy":"S2","thresholds":{"OAS1":0.42,"OAS2":0.4928655287824083}}
```

### POST /predict (lote)
**Entrada** (dos formatos por caso):
```json
{
  "cases": [
    {
      "clinical": {"patient_id":"X001","cohort":"OAS1","Age":72,"Sex":"F","MMSE":28},
      "features": {"oas2_effb3_p14_mean":0.71,"slice_preds_plus_mean":0.55}
    },
    {
      "clinical": {"patient_id":"X002","cohort":"OAS2","Age":76,"Sex":"M","MMSE":22},
      "p_img": 0.63
    }
  ]
}
```
**Salida**:
```json
[
  {"patient_id":"X001","cohort":"OAS1","p_img":0.58,"p_clin":0.31,"proba_cal":0.445,"thr":0.42,"decision":1},
  {"patient_id":"X002","cohort":"OAS2","p_img":0.63,"p_clin":0.47,"proba_cal":0.55,"thr":0.4928655287824083,"decision":1}
]
```

---

## Validación y manejo de errores
- **Alineación de columnas**: si el modelo expone `feature_names_in_`, se reordena y crean faltantes (NaN) para que `p24_model.pkl` acepte el input.  
- **Sklearn mismatch**: usar **1.7.1** como en el entrenamiento.  
- **Cohorte ausente**: umbral por defecto **0.5** (desaconsejado).  
- **Calibrador**: fallback a `proba_raw` si `p24_platt.pkl` no es aplicable.

---

## Seguridad y privacidad
- No persiste datos; procesa en memoria.  
- En producción: HTTPS, autenticación (token), logs anonimizados, monitorizar ECE/MCE y confusión por cohorte.

---

## Tests rápidos
```bash
# Linux/macOS
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d @example.json
```
```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -InFile "example.json"
```
