# app.py (Streamlit)

Aplicación **web local** para ejecutar el pipeline intermodal sin terminal: sube CSVs, aplica **S2**, descarga resultados y (si hay etiquetas) QA.

---

## 🧾 ¿Qué es Streamlit?
**Streamlit** es un framework en Python para crear **apps web de datos** con poco código.
Se ejecuta localmente y abre una URL en tu navegador (también puede desplegarse en servidor).

---

## ✅ Qué hace la app
1. Carga **CSV de features por paciente** (56 cols + `patient_id`, `cohort`).
2. Calcula **`p_img`** usando el modelo P24 + Platt.
3. Carga **CSV clínico** y calcula **`p_clin`** con el modelo de P26.
4. Realiza **fusión LATE** → `proba_cal`.
5. Aplica **S2** (umbrales por cohorte) → **`decision`** (0/1).
6. Permite **descargar** `predictions.csv` y, si hay `y_true`, **QA.csv`.

---

## 📦 Requisitos
- Python 3.10+
- `pip install -r requirements.txt` (incluye `streamlit`)
- Ficheros en `p26_release/`:
  - `CONFIG/deployment_config.json`
  - `models/p24_model.pkl`
  - `models/p24_platt.pkl`
  - `models/p26_clinical_model.pkl`

---

## ▶️ Ejecutar
```bash
streamlit run app.py
```
Se abrirá una URL local (p. ej., `http://localhost:8501/`).

---

## 🧭 Flujo de uso
1. **Sube** CSV de **features por paciente** y CSV **clínico**.
2. (Opcional) Indica el nombre de columna `y_true`.
3. Pulsa **“Ejecutar”**.
4. Revisa la tabla y **descarga** `predictions.csv` (y `QA.csv` si corresponde).

---

## 🎯 Política S2
- Leída de `p26_release/CONFIG/deployment_config.json`.
- Umbrales activos: **OAS1=0.42**, **OAS2≈0.4928655287824083**.
- Cambia el JSON y **refresca la app**.

---

## 🗂️ Archivos esperados
```
p26_release/
 ├─ CONFIG/
 │   └─ deployment_config.json
 └─ models/
     ├─ p24_model.pkl
     ├─ p24_platt.pkl
     └─ p26_clinical_model.pkl
```

---

## 🧪 Troubleshooting
- **Pickle/sklearn mismatch** → usa versiones de `ENVIRONMENT.txt`/`requirements.txt`.
- **Nombres de columnas** → mantén los del entrenamiento; la app intentará alinear/imputar.
- **Sin `cohort`** → S2 aplicará 0.5 por defecto (no recomendado).
- **CSV Excel** → exporta en UTF-8 con separador `,`.

---

## 🔐 Notas de privacidad
- Los CSV pueden contener datos sensibles: controla accesos y elimina temporales.
- La herramienta es **soporte a la decisión**, no un diagnóstico automático.
