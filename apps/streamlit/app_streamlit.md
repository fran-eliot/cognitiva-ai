
# App Streamlit — Guía (CognitivaAI Intermodal)

GUI para **intermodalidad** (imagen + clínico) con **fusión LATE**, política **S2 o Manual**, **Modo Demo** y visualizaciones (métricas, calibración, coste).

## Inicio rápido
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install streamlit pandas numpy scikit-learn==1.7.1 joblib requests
streamlit run app.py
```

## CSV esperados
- **Features**: `patient_id, cohort` + 56 columnas de P24 (se alinean con `feature_names_in_`).
- **Clínico**: `patient_id, cohort, Age, Sex, Education, SES, MMSE, eTIV, nWBV, ASF, Delay` (+ opcional `y_true`).

## Flujo
1) Carga CSV o **Modo Demo** (10 pacientes).  
2) **Ejecutar** → `p_img`, `p_clin`, `proba_cal`, `decision`.  
3) **Métricas** (si hay `y_true`): AUC/PR-AUC/Brier, confusión, calibración (ECE/MCE).
4) **Coste vs Umbral** por cohorte.
5) **Ajustes** → guardar umbrales si Manual.

## Política y coste
- **S2** desde `deployment_config.json` o **Manual** (sliders).  
- Coste FN:FP → 3:1 / 5:1 / 7:1 / 10:1.

## Extras
- Descarga CSV de predicciones.  
- Comparación con **FastAPI** (POST `/predict`).  
- Consejos de compatibilidad (`sklearn==1.7.1`).
