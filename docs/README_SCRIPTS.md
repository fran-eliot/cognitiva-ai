# README — Scripts y Flujo de Ejecución

Este documento resume cómo **encadenar** los scripts para obtener predicciones intermodales con **política S2**.

---

## 🧭 Flujo general
1) **Imagen → p_img**  
   `compute_pimg_from_features.py` → `p_img.csv`

2) **Fusión + S2**  
   `predict_end_to_end.py` → `predictions.csv` (+ `predictions_qa.csv` si hay `y_true`)

3) *(Opcional)* **App web**  
   `streamlit run app.py` para hacerlo sin terminal.

---

## 📦 Requisitos
- Python 3.10+
- `pip install -r requirements.txt`
- Estructura esperada:
```
p26_release/
 ├─ CONFIG/deployment_config.json
 └─ models/
     ├─ p24_model.pkl
     ├─ p24_platt.pkl
     └─ p26_clinical_model.pkl
```

---

## ▶️ Paso 1 — p_img desde features (P24)
**Windows (PowerShell)**
```powershell
python .\compute_pimg_from_features.py `
  --features .\data\patient_features.csv `
  --models_dir .\p26_release\models `
  --out .\work\p_img.csv
```
**Linux / macOS**
```bash
python compute_pimg_from_features.py       --features data/patient_features.csv       --models_dir p26_release/models       --out work/p_img.csv
```
**Salida:** `work/p_img.csv` con `patient_id, cohort, p_img`.

---

## ▶️ Paso 2 — Fusión LATE + Decisión S2
**Windows (PowerShell)**
```powershell
python .\predict_end_to_end.py `
  --pimg .\work\p_img.csv `
  --clinic .\data\clinical.csv `
  --models_dir .\p26_release\models `
  --config .\p26_release\CONFIG\deployment_config.json `
  --out .\work\predictions.csv
```
**Linux / macOS**
```bash
python predict_end_to_end.py       --pimg work/p_img.csv       --clinic data/clinical.csv       --models_dir p26_release/models       --config p26_release/CONFIG/deployment_config.json       --out work/predictions.csv
```
**Salida:** `work/predictions.csv` con `patient_id, cohort, p_img, p_clin, proba_cal, thr, decision` (+ `predictions_qa.csv` si hay `y_true`).

---

## 🎯 Política de decisión (S2)
- **OAS1** → umbral **0.42** (FN:FP=5:1)
- **OAS2** → umbral **≈0.4928655287824083** (recall objetivo)
- Archivo: `p26_release/CONFIG/deployment_config.json`

---

## 🧪 QA recomendado
- Revisa `predictions_qa.csv` si hay `y_true`.
- Monitoriza por cohorte: **TP/FP/TN/FN**, **Precision**, **Recall**, **Coste (5:1)**.
- Considera **ECE/MCE** si dispones de pipeline de calibración local.

---

## 🛠️ Problemas comunes
- **Versiones de sklearn y pickles**: alinear con `ENVIRONMENT.txt`.
- **Columnas faltantes**: se imputan/alinean, pero mantener **nombres** es clave.
- **Sin `cohort`**: S2 no aplicará umbral correcto (usa 0.5 por defecto → no recomendado).

---

## 🧰 App web (opcional)
Para usuarios no técnicos:
```bash
streamlit run app.py
```
Sube **features por paciente** y **clínico**, y descárgate `predictions.csv`.

---

## ✅ Resumen
- `compute_pimg_from_features.py` → produce **p_img** calibrada (imagen).
- `predict_end_to_end.py` → **fusiona** con clínico, aplica **S2** y devuelve decisión.
- `app.py` (Streamlit) → interfaz web local para todo el proceso sin terminal.
