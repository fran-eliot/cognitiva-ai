# compute_pimg_from_features.py

Genera **p_img** (probabilidad basada en imagen, calibrada) a partir de las **features por paciente** usando el modelo **P24** (LR elastic-net) y su **calibrador Platt**.

---

## ✅ Objetivo
A partir de un CSV con features por paciente (56 columnas + `patient_id` + `cohort`), calcular la probabilidad de demencia por imagen **p_img**, ya calibrada.

---

## 📥 Entradas
- **CSV de features por paciente**  
  Debe incluir: `patient_id`, `cohort` (`OAS1`/`OAS2`) y las **56 columnas** usadas en P24. El script **alinea** y **reordena** las columnas; si falta alguna, la crea con `NaN`.
- **Modelos** (carpeta por defecto `p26_release/models/`):  
  - `p24_model.pkl` (obligatorio)  
  - `p24_platt.pkl` (opcional, recomendado)

---

## 📤 Salidas
- **CSV** con columnas: `patient_id, cohort, p_img` (probabilidad calibrada si hay Platt).

---

## ⚙️ Cómo funciona (resumen)
1. Carga `p24_model.pkl` y, si existe, `p24_platt.pkl`.
2. Lee el CSV, verifica `patient_id` y `cohort`.
3. Determina las columnas esperadas por el modelo y **alinea** `X` (crea faltantes con `NaN`).
4. Calcula `proba_raw = model.predict_proba(X)[:, 1]`.
5. Si hay Platt: intenta `platt.predict_proba(proba_raw.reshape(-1,1))`; si falla, prueba con `X`; si no, usa `proba_raw`.
6. Devuelve un CSV minimal con `p_img`.

---

## ▶️ Ejecución

**Windows (PowerShell)**
```powershell
.\.venv\Scripts\Activate.ps1
python .\compute_pimg_from_features.py `
  --features .\data\patient_features.csv `
  --models_dir .\p26_release\models `
  --out .\work\p_img.csv
```

**Linux / macOS (bash)**
```bash
python compute_pimg_from_features.py       --features data/patient_features.csv       --models_dir p26_release/models       --out work/p_img.csv
```

---

## 🗂️ Estructura sugerida
```
p26_release/
 └─ models/
     ├─ p24_model.pkl
     └─ p24_platt.pkl
data/
 └─ patient_features.csv
work/
 └─ p_img.csv  (salida)
```

---

## 🧪 Validaciones y avisos
- Si no encuentra `feature_names_in_`, usará **todas las columnas numéricas** (fallback).
- Si falla el calibrador, usa `proba_raw` y avisa en consola.
- Asegúrate de que `scikit-learn` sea compatible con los pickles (ver `ENVIRONMENT.txt`/`requirements.txt`).

---

## ✅ Buenas prácticas
- Mantén **nombres de columna consistentes** con entrenamiento P24.
- Incluye **`cohort`** (S2 lo usará más adelante).
- Guarda los CSV en **UTF-8** y separador `,`.
