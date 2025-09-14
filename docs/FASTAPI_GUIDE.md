
# FastAPI — Guía completa (CognitivaAI Intermodal)

API para servir predicciones **intermodales** (imagen + clínico) con **fusión LATE** y **política S2** por cohorte (OAS1/OAS2). Replica la lógica de P26/P27.

## Flujo
1) Imagen → `p_img` (o `features` → P24 + Platt opcional)  
2) Clínico → `p_clin` (P26)  
3) Fusión LATE → `(p_img + p_clin)/2`  
4) Decisión S2 → umbral por cohorte desde `deployment_config.json`

## Estructura
```
p26_release/
  CONFIG/deployment_config.json
  models/{p24_model.pkl,p24_platt.pkl,p26_clinical_model.pkl}
api/main.py
```

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn pydantic pandas numpy joblib scikit-learn==1.7.1
```

## Ejecutar
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints
- `GET /health` → estado + umbrales
- `POST /predict`
```json
{
  "cases":[
    {
      "clinical":{"patient_id":"X001","cohort":"OAS1","Age":72,"Sex":"F","MMSE":28},
      "features":{"oas2_effb3_p14_mean":0.71,"slice_preds_plus_mean":0.55}
    },
    {
      "clinical":{"patient_id":"X002","cohort":"OAS2","Age":76,"Sex":"M","MMSE":22},
      "p_img":0.63
    }
  ]
}
```

**Respuesta**:
```json
[{"patient_id":"X001","cohort":"OAS1","p_img":0.58,"p_clin":0.31,"proba_cal":0.445,"thr":0.42,"decision":1}]
```

## Notas
- `feature_names_in_` para alinear columnas.
- `p24_platt.pkl` se aplica a `proba_raw` y cae a `X` o `proba_raw` si no aplica.
- Cohorte desconocida → thr=0.5 (no recomendado).
- Usar `sklearn==1.7.1` para compatibilidad de pickles.
