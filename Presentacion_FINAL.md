# 🧾 Presentación (versión escrita) — Cognitiva-AI

> Documento de acompañamiento a las slides y a la demo (Streamlit + FastAPI).

---

## 1) Resumen ejecutivo
- **Objetivo**: probabilidad **Dementia/Converted (1)** vs **Control (0)** con **MRI + clínico**.
- **Modelo recomendado**: **Intermodal LATE** (media `p_img` y `p_clin`) con **Platt por cohorte** y **política S2** (recall alto en OAS2).
- **Umbrales activos (S2)**: **OAS1=0.42**, **OAS2≈0.4929** → recall alto y coste controlado.
- **Entrega**: *release* zip con modelos, CONFIG, QA, DOCS, **Streamlit** y **FastAPI**.

---

## 2) Datos y preprocesado
- **OASIS-1** (transversal) y **OASIS-2** (longitudinal, **1ª visita/paciente** para evitar fuga).  
- **MRI**: 20 *slices* → agregación por paciente a **56 features** (catálogo p11; `mean/trimmed/top-k/pmean_2`).  
- **Clínico**: homogeneización (CDR/Group→Target), imputación (Education/SES), OHE(Sex), **anti-fuga** (no usar CDR/Group como feature).

---

## 3) Resultados clave (TEST)

### 3.1 Probabilidades (AUC/PR-AUC/Brier)
| Pipeline | Cohorte | Modelo/Calib           |   AUC  | PR-AUC | Brier |
|---------:|:-------:|-------------------------|:------:|:------:|:-----:|
| P19      |  ALL    | Meta-XGB                | 0.671  | 0.606  | 0.292 |
| P22      |  ALL    | LR/HGB + Platt          | 0.668–0.702 | 0.605–0.646 | 0.219–0.239 |
| **P24**  |  ALL    | **LR-EN + Platt**       | **0.727** | **0.717** | **0.220** |
| P26      |  ALL    | LATE (raw)              | 0.713  | 0.712  | 0.234 |
| P26b     |  OAS1   | LATE + Platt (cohorte)  | 0.754  | 0.736  | 0.199 |
| P26b     |  OAS2   | LATE + Platt (cohorte)  | 0.652  | 0.728  | 0.241 |

### 3.2 Decisión coste-sensible (FN:FP=5:1) — umbrales de VAL en TEST
| Pipeline | Cohorte | Thr   |  TP |  FP |  TN |  FN | Precision | Recall |  Acc  | Cost |
|---------:|:------:|:-----:|----:|----:|----:|----:|----------:|-------:|------:|-----:|
| **P24**  | OAS1   | 0.435 | 14  |  9  | 18  |  6  | 0.609 | 0.700 | 0.681 | 39 |
| **P24**  | OAS2   | 0.332 | 11  |  7  |  4  |  1  | 0.611 | 0.917 | 0.652 | 12 |
| **P26**  | OAS1   | 0.307 | 14  |  9  | 18  |  6  | 0.609 | 0.700 | 0.681 | 39 |
| **P26**  | OAS2   | 0.195 |  8  |  4  |  7  |  4  | 0.667 | 0.667 | 0.652 | 24 |

### 3.3 Calibración
- **ECE@10 / MCE@10 (P26 TEST)**: **ALL 0.178/0.407**, **OAS1 0.150/0.578**, **OAS2 0.313/0.766**.  
- Acción: **Platt por cohorte** y **monitoring** en despliegue; recalibrar si ECE>0.20 o cambia la mezcla.

---

## 4) Arquitectura y flujo operativo
1. **Features MRI (56)** → `p_img` con P24 (LR-EN + Platt).  
2. **Clínico (anti-fuga)** → `p_clin` (LR).  
3. **Fusión LATE** → media(`p_img`,`p_clin`) → calibración por cohorte (P26b).  
4. **Política S2** → umbrales OAS1/OAS2 orientados a **recall** (FN:FP=5:1 con ajuste OAS2 ≥0.90).

---

## 5) Demo y uso
- **Streamlit**: carga CSV (features y clínico), modo **Demo/Real**, *switch* de política (P24/P26/S2), *sliders* de umbral por cohorte, métricas (TP/FP/TN/FN, Coste) y gráficos (ROC/PR/Calibración/Coste).  
- **FastAPI**: `POST /predict` con `{clinical + features}` o `{clinical + p_img}` → `{p_img, p_clin, proba_cal, thr, decision}`.  
- **CLI**:  
  ```bash
  python compute_pimg_from_features.py --features patient_features.csv --models_dir p26_release/models --out p_img.csv
  python predict_end_to_end.py --pimg p_img.csv --clinic clinical.csv --models_dir p26_release/models --config p26_release/CONFIG/deployment_config.json --out predictions.csv
  ```
## 6) Reproducibilidad y release
- **Zip** final: modelos (P24, Platt, clínico), **CONFIG** (umbrales S2), **QA**, **DOCS** (MODEL_CARD/How-To), **MANIFEST/ENV**.
- Versionado consistente (`scikit-learn==1.7.1`); validación de columnas vs `feature_names_in_`.

## 7) Limitaciones y próximos pasos
- **N** reducido en OAS2; **ECE** mayor → recalibración por sitio y `drift monitoring`.
- Validación externa / **domain adaptation**.
- Reporte clínico por paciente (explicabilidad ligera + seguimiento longitudinal).