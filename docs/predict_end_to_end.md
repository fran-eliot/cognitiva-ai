# predict_end_to_end.py

Realiza la **fusión LATE** entre `p_img` (imagen) y `p_clin` (clínico) y aplica la **política S2** (umbrales por cohorte) para obtener **probabilidad final** y **decisión 0/1**.

---

## ✅ Objetivo
Unir `p_img` + `p_clin` → `proba_cal` (fusión) → **decisión S2** con umbrales por cohorte.

---

## 📥 Entradas
- `--pimg` → CSV con `patient_id, cohort, p_img` (de `compute_pimg_from_features.py`).  
- `--clinic` → CSV clínico mínimo: `patient_id, cohort, Age, Sex, Education, SES, MMSE, eTIV, nWBV, ASF, Delay`.  
- `--models_dir` → carpeta con `p26_clinical_model.pkl`.  
- `--config` → `p26_release/CONFIG/deployment_config.json` con umbrales **S2**.  
- *(opcional)* `--y_true_col` → nombre de columna con etiqueta (0/1) para QA.

---

## 📤 Salidas
- `--out` → `predictions.csv` con:  
  `patient_id, cohort, p_img, p_clin, proba_cal, thr, decision`  
- Si hay `y_true`: `predictions_qa.csv` (TP/FP/TN/FN, Precision, Recall, Acc, Cost 5:1).

---

## ⚙️ Cómo funciona (resumen)
1. **Merge** por `patient_id, cohort`.
2. Prepara clínico: convierte `Sex`, **imputa** medianas, alinea con columnas esperadas.
3. Calcula `p_clin = model_clin.predict_proba(Xc)[:,1]` (con fallback robusto si hay incompatibilidades).
4. **Fusión LATE** (por defecto: media): `proba_cal = (p_img + p_clin)/2`.
5. **Decisión S2**: umbral por cohorte desde `deployment_config.json`.
6. Escribe `predictions.csv` (+ `predictions_qa.csv` si procede).

---

## 🎯 Política S2 (activa)
- **OAS1** → umbral **0.42** (FN:FP = 5:1)  
- **OAS2** → umbral **≈0.4928655287824083** (recall objetivo)  
Editar en: `p26_release/CONFIG/deployment_config.json`.

---

## ▶️ Ejecución

**Windows (PowerShell)**
```powershell
python .\predict_end_to_end.py `
  --pimg .\work\p_img.csv `
  --clinic .\data\clinical.csv `
  --models_dir .\p26_release\models `
  --config .\p26_release\CONFIG\deployment_config.json `
  --out .\work\predictions.csv
```

**Linux / macOS (bash)**
```bash
python predict_end_to_end.py       --pimg work/p_img.csv       --clinic data/clinical.csv       --models_dir p26_release/models       --config p26_release/CONFIG/deployment_config.json       --out work/predictions.csv
```

---

## 🧩 CSV clínico — Esquema mínimo
- Obligatorio: `patient_id`, `cohort`
- Numéricas: `Age, Education, SES, MMSE, eTIV, nWBV, ASF, Delay`
- Categórica: `Sex` (`M`/`F` o `MALE`/`FEMALE`) → se convierte a numérica si hace falta.

---

## 🧪 QA (si hay etiqueta)
Si `--y_true_col` está presente, se genera `predictions_qa.csv` con:  
TP, FP, TN, FN, Precision, Recall, Accuracy y **Coste 5:1** por cohorte y global.

---

## 🛠️ Problemas comunes
- **Pickle/sklearn mismatch** → instala versiones de `ENVIRONMENT.txt`/`requirements.txt`.
- **“Feature names should match”** → revisa nombres/tipos; el script alinea e imputa.
- **Falta `cohort`** → S2 no aplica umbral correcto (usa 0.5 por defecto → no recomendado).
