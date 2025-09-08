# 🧾 Superprompt inicial – Proyecto CognitivaAI (OASIS-1 + OASIS-2, hasta P22)

Este proyecto investiga predicción binaria (Control=0 vs Dementia/Converted=1) con **OASIS-1** y **OASIS-2**. Empezó en **local** (p1–p4) y migró a **Google Colab** (p5+). Hasta ~**p18** se trabajó **principalmente con OASIS-1**; después se **incorporó y explotó OASIS-2** (p19–p22), combinando ambas cohortes en el *meta-ensemble*.

El flujo integra:
- **DL** (EfficientNet-B3 y backbones alternativos).
- **Features por paciente** agregadas a partir de *slices* (mean / trimmed20 / top-k / pmean_2).
- **Ensembles clásicos y meta-modelos** (LR, GB, RF, HGB, XGB, LGBM).
- **Calibración** (Platt / Isotónica), **umbrales por coste** y **análisis por cohorte** (OAS1/OAS2).
- Documentación viva en `README.md`, `InformeTecnico.md`, `CuadernoBitacora.md`.

---

## 📂 Datasets y preprocesado

### OASIS-1
- Usado **mayoritariamente en p5–p18** (Colab) para entrenar y comparar backbones, construir el **catálogo p11** y generar **matrices de features por paciente** (VAL/TEST).
- Preprocesado: 20 *slices* axiales equiespaciados por volumen; exclusión del ~8% de extremos; normalización **z-score** + **CLAHE** opcional.
- Splits **estratificados a nivel paciente** (sin fuga).
- Artefactos clave consolidados en:  
  - `p11_alt_backbones/` → catálogo, matrices **VAL/TEST** por-paciente y comparativas.

### OASIS-2
- **Incorporado y explotado a partir de p19–p22** (meta-ensembles + calibración); también se generaron backbones propios (EffB3) y *maps* por-slice/por-paciente.
- Criterio: **una sola visita por paciente** (evita *leakage* inter-sesión).
- 20 *slices* axiales por volumen, mismas normalizaciones que OASIS-1.
- Splits (ejemplo típico en OASIS-2 etiquetado): 105 train, 22 val, 23 test.
- Artefactos clave:
  - `p13_oasis2_images/` y `p14_oasis2_images/` → *maps* por-slice/por-paciente, pesos EffB3, resúmenes.
  - Integración en catálogo (`p11_alt_backbones/p11_backbone_catalog.json`) para usarlos en ensambles posteriores.

> **Nota:** Los *meta-pipelines* p19–p22 **mezclan señales de OASIS-1 (catálogo p11)** con **backbones de OASIS-2** (EffB3), por eso las métricas se reportan por **cohorte** (OAS1/OAS2) además de globales.

---

## 🧪 Línea temporal de pipelines

### P1–P4 (Local)
- **Objetivo:** Puesta a punto local (scripts de *slicing*, normalización, primera LR/GB base).
- **Motivo de migración:** limitaciones de GPU/CPU y de E/S → pasar a **Colab**.

### P5–P12 (Colab, **OASIS-1**)
- **p5 – Setup Colab y estructura de proyecto**  
  Env, *seeds*, persistencia en Drive, utilidades de E/S y verificación de integridad.
- **p6 – Slicing & normalización (OASIS-1)**  
  20 cortes equiespaciados, z-score + CLAHE opcional; generación de *maps* CSV.
- **p7 – Baselines CNN (OASIS-1)**  
  Entrenos iniciales y curvas de aprendizaje; verificación de *overfit*.
- **p8 – Backbones alternativos**  
  Swin, ConvNeXt, Dense/RegNet… con *head* binaria; estandarización de inferencias.
- **p9 – Agregación por paciente**  
  `mean / trimmed20 / top-k / pmean_2` → matrices VAL/TEST a nivel paciente.
- **p10 – Inference + evaluación por paciente**  
  CSVs por-slice y por-paciente; métricas y umbrales base.
- **p11 – Catálogo de backbones (OASIS-1)**  
  `p11_alt_backbones/p11_backbone_catalog.json` (rutas/columnas estándar).  
  **Matrices** por paciente:  
  - `p11_alt_backbones/val_patient_features_backbones.csv`  
  - `p11_alt_backbones/test_patient_features_backbones.csv`  
  **Comparativa:** `comparison_backbones_eval.csv`.
- **p12 – Ensembles base (OASIS-1)**  
  LR/GB/RF como baseline a nivel paciente; guardado de predicciones y métricas.

### P13–P14 (Backbones **OASIS-2**)
- **p13 – EffB3-OAS2 (1 visita/paciente)**  
  Entreno EffNet-B3 sobre OASIS-2; *maps* por slice/paciente, y evaluación por paciente.  
  Artefactos (ejemplos):  
  - `p13_oasis2_images/oas2_train_colab_mapped.csv`  
  - `p13_oasis2_images/val_png_preds_oas2_effb3.csv`  
  - `p13_oasis2_images/val_patient_preds_oas2_effb3.csv`  
  - `p13_oasis2_images/p13_patient_eval_summary.json`
- **p14 – EffB3-OAS2 (balanceado + SSD local)**  
  Copia de **7340 PNG** a SSD Colab (mitiga latencia de Drive).  
  `class_weight` aplicado; recall alto (TEST), AUC estable (VAL).  
  Artefactos:  
  - `p14_oasis2_images/p14_effb3_oas2_best.pth`  
  - `p14_oasis2_images/*_patient_features_oas2_effb3_p14.csv`  
  - `p14_oasis2_images/p14_patient_eval_summary.json`  
  - Catálogo actualizado: `p11_alt_backbones/p11_backbone_catalog.json`

### P15 – Consolidación (Doc + artefactos)
- Integra P13–P14 en documentación; añade criterio **1 visita/paciente** y la copia a SSD.
- Actualiza `README.md`, `InformeTecnico.md` y `CuadernoBitacora.md`.

### P16–P18 (Ensembles avanzados, **base OASIS-1** + señales nuevas)
- **p16 – Ensemble inicial (LR/HGB + blending α)**  
  Filtrado NaNs (≥40%), imputación; **VAL AUC≈0.95**; **TEST AUC≈0.69**.  
  Artefactos:  
  - `p16_ensemble_refine/p16_val_patient_preds_ensemble.csv`  
  - `p16_ensemble_refine/p16_test_patient_preds_ensemble.csv`  
  - `p16_ensemble_refine/p16_ensemble_summary.json`
- **p17 – Stacking + calibración (Platt)**  
  Meta-LR; OOF sin fuga; Brier aceptable; **TEST AUC≈0.70**.  
  Artefactos: `p17_ensemble_calibration/*`
- **p18 – Stacking avanzado (LR/HGB/GB/RF/ET → meta-LR + α)**  
  Importancias (permutación) y coeficientes LR (GB domina); **TEST AUC≈0.67**.  
  Artefactos: `p18_advanced_stacking/*`

### P19 – Meta-Ensemble (XGB) **integrando OASIS-1 + OASIS-2**
- Base learners: **LR, HGB, GB, RF, LGBM, XGB** (OOF sin fuga).  
- **Meta-XGB** entrenado sobre OOF; inferencia en TEST con base learners.  
- **VAL:** AUC≈0.964 / **TEST:** AUC≈0.729 (Prec alta, Recall moderada).  
- Artefactos:  
  - `p19_meta_ensemble/p19_val_patient_preds.csv`  
  - `p19_meta_ensemble/p19_test_patient_preds.csv`  
  - `p19_meta_ensemble/p19_summary.json`

### P20 – Meta-calibration (Global/Per-cohort)
- LR/HGB con **Platt** e **Isotónica**, calibración **global** y **por cohorte** (OAS1/OAS2).  
- Brier bajo en VAL con Isotónica; generalización mejor con Platt en TEST.  
- Artefactos:  
  - `p20_meta_calibration/p20_calibration_thresholding_summary.json`  
  - `p20_meta_calibration/p20_*_preds.csv` (VAL/TEST, HGB/LR, Platt/Iso, Global/PerC)

### P21 – Meta-refine (stack+blend con reducción de complejidad)
- Meta-features: OOF {LR,HGB,GB,RF} + blending α.  
- **VAL AUC≈0.955 / TEST AUC≈0.653** → confirma *shift* OAS2 y necesidad de umbral/cohorte.  
- Artefactos: `p21_meta_refine/*`

### P22 – Meta-ablation (calibraciones comparadas)
- **LR/HGB** calibrados (Platt vs Isotónica) + **BLEND** simple; **thr** por F1(VAL).  
- **VAL:** Isotónica ≈ mejor F1/AUC; **TEST:** Platt ≈ recall más alto.  
- Artefactos:  
  - `p22_meta_ablation/p22_val_calibrations.csv`  
  - `p22_meta_ablation/p22_test_calibrations.csv`  
  - `p22_meta_ablation/p22_final_summary.json`

---

## 📚 Documentación viva

- `README.md` → hoja de ruta + resultados por pipeline (incluye OASIS-1 y OASIS-2).
- `InformeTecnico.md` → metodología de *slices*, 1 visita/paciente (OASIS-2), E/S y SSD, ensembles, calibración y análisis por cohorte.
- `CuadernoBitacora.md` → entradas diarias y **entradas de fase**:  
  - **Fase 6:** p5–p12 (OASIS-1 base, catálogo p11).  
  - **Fase 7:** p13–p14 (backbones OASIS-2).  
  - **Fase 8:** p16–p19 (ensembles y meta-XGB con OASIS-1 + OASIS-2).  
  - **Fase 9:** p20–p22 (calibración avanzada y *ablation*).

---

## 📌 Próximos pasos sugeridos

1. **P23 – Meta-calibración por cohorte con coste clínico**  
   Optimiza umbrales **específicos** OAS1/OAS2 bajo **FN≫FP**, y agrega un **umbralfinal** por mezcla de cohortes.
2. **P24 – Meta simple y robusto**  
   Meta-ElasticNet / LR calibrada (Platt), selección de features (mRMR/perm), **Repeated KFold**.
3. **P25 – Informe final**  
   Tabla comparativa (VAL/TEST) por método, cohortes, calibración y coste; *figures* (ROC/PR/Brier/Cal).

---

## 🧭 Dónde están métricas y artefactos

- **Catálogo y features por paciente (OASIS-1, base para ensembles):**  
  `p11_alt_backbones/`  
  - `p11_backbone_catalog.json`  
  - `val_patient_features_backbones.csv` | `test_patient_features_backbones.csv`  
  - `comparison_backbones_eval.csv`

- **Backbones OASIS-2 (EffB3):**  
  `p13_oasis2_images/`, `p14_oasis2_images/`  
  - `*_colab_mapped.csv` (maps por slice)  
  - `*_png_preds_*.csv`, `*_patient_preds_*.csv`, `*_patient_features_*.csv`  
  - `p13_patient_eval_summary.json`, `p14_patient_eval_summary.json`  
  - pesos: `p14_effb3_oas2_best.pth`

- **Ensembles / Meta (OASIS-1 + OASIS-2):**  
  - `p16_ensemble_refine/*`  
  - `p17_ensemble_calibration/*`  
  - `p18_advanced_stacking/*`  
  - `p19_meta_ensemble/*`  
  - `p20_meta_calibration/*`  
  - `p21_meta_refine/*`  
  - `p22_meta_ablation/*`

---

## 🤝 Qué espero del nuevo chat

1. Continuar con **P23** (A–G): carga de features, OOF robusto, **calibración por cohorte con coste**, selección de umbrales, guardado y resúmenes.  
2. Resolver incidencias Colab (NaNs, `CalibratedClassifierCV(estimator=..., cv='prefit')`, *paths*, latencias E/S).  
3. Entregar **bloques `.md`** actualizables para `README.md`, `InformeTecnico.md`, `CuadernoBitacora.md`.  
4. Preparar **tablas/figuras** finales (ROC/PR/Brier/Cal) y checklist de reproducibilidad.

