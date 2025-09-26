
# 🧠 COGNITIVA-AI — Detección temprana de Alzheimer con datos clínicos + MRI  
**README integral (consolidado, verboso y ordenado)**  
> Proyecto multimodal (clínico + imagen) con énfasis en **recall clínico**, **calibración** y **reproducibilidad**.  
> Release actual: **P27** (incluye **política S2** y empaquetado listo para entrega).

---

## 📚 Índice

1. [Resumen ejecutivo](#resumen-ejecutivo)  
2. [Contexto y objetivos](#contexto-y-objetivos)  
3. [Datos y preprocesamiento](#datos-y-preprocesamiento)  
   3.1. [Cohortes OASIS-1 y OASIS-2](#cohortes-oasis-1-y-oasis-2)  
   3.2. [Variables clínicas y tratamiento](#variables-clínicas-y-tratamiento)  
   3.3. [Imágenes MRI y pipeline de slices](#imágenes-mri-y-pipeline-de-slices)  
   3.4. [Evitar *leakage*](#evitar-leakage)  
4. [Cronología de pipelines (P1 → P27)](#cronología-de-pipelines-p1--p27)  
5. [Modelos clave y resultados](#modelos-clave-y-resultados)  
   5.1. [P24 — Meta simple (imagen) LR-EN + Platt](#p24--meta-simple-imagen-lr-en--platt)  
   5.2. [P26 — Intermodal (Late vs Mid)](#p26--intermodal-late-vs-mid)  
   5.3. [P26b — Calibración por cohorte](#p26b--calibración-por-cohorte)  
   5.4. [P27 — Política S2 y release](#p27--política-s2-y-release)  
   5.5. [Comparativas globales](#comparativas-globales)  
6. [Figuras y tablas finales](#figuras-y-tablas-finales)  
7. [Reproducibilidad y *release*](#reproducibilidad-y-release)  
8. [Cómo ejecutar (scripts, app y API)](#cómo-ejecutar-scripts-app-y-api)  
   8.1. [Scripts CLI](#scripts-cli)  
   8.2. [App gráfica (Streamlit)](#app-gráfica-streamlit)  
   8.3. [API (FastAPI)](#api-fastapi)  
9. [Desafíos principales](#desafíos-principales)  
10. [Lecciones aprendidas](#lecciones-aprendidas)  
11. [Limitaciones y próximos pasos](#limitaciones-y-próximos-pasos)  
12. [Conclusiones finales](#conclusiones-finales)  
13. [Agradecimientos y licencia](#agradecimientos-y-licencia)

---

## Resumen ejecutivo

- **Problema**: estimar riesgo (probabilidad) de demencia (binario: Nondemented vs Demented/Converted) combinando **clínico** + **MRI**.  
- **Estrategia**: pipelines incrementales (P1…P27), calibración de probabilidades, umbrales **por cohorte** y decisión por **coste** (FN:FP).  
- **Mejor unimodal (imagen)**: **P24** (LR elastic-net + Platt).  
- **Mejor intermodal**: **P26 → Late** (p_img + p_clin + p1_fill + p1_has) con **P26b** (Platt por cohorte).  
- **Política activa**: **S2** (P27) = 5:1 por cohorte + ajuste en OAS2 para **recall ≥ 0.90** (cribado).  
- **Release**: `p26_release.zip` con modelos, config, QA, scripts y doc.

**Métricas clave (TEST):**

- **P24 (imagen):**
  - **ALL**: AUC=**0.727**, PR-AUC=0.717, Brier=0.220  
  - **OAS1**: AUC=**0.754**, PR-AUC=0.736, Brier=0.211  
  - **OAS2**: AUC=**0.750**, PR-AUC=0.805, Brier=0.238  
  - **Umbrales 5:1**: OAS1=0.435 → TP=14 FP=9 TN=18 FN=6; OAS2=0.332 → TP=11 FP=7 TN=4 FN=1.

- **P26 Late (intermodal, raw):** ALL AUC=**0.713**, PR-AUC=0.712, Brier=0.234.  
- **P26b Late + Platt por cohorte:** **OAS1 AUC≈0.754 (Brier≈0.199)** · **OAS2 AUC≈0.652 (Brier≈0.241)**.

- **Política S2 (activa; TEST @S2 con Late):**
  - **OAS1 thr=0.42** → TP=14 FP=9 TN=18 FN=6 → **Recall=0.70**, **Prec=0.609**, **Coste=39**  
  - **OAS2 thr≈0.4929** → TP=11 FP=6 TN=5 FN=1 → **Recall=0.917**, **Prec=0.647**, **Coste=11**

---

## Contexto y objetivos

- **Objetivo científico**: explorar cómo **clínico** (edad, MMSE, CDR, etc.) y **MRI** se complementan para detectar Alzheimer temprano.  
- **Objetivo práctico**: entregar un **modelo reproducible** y **calibrado** con **umbrales clínicos** ajustables por política (p. ej., cribado).  
- **Criterio**: priorizar **sensibilidad (recall)** → minimizar **falsos negativos**.

---

## Datos y preprocesamiento

### Cohortes OASIS-1 y OASIS-2
- **OASIS-1 (transversal)**: una visita/paciente; **target** por `CDR` (0→0, >0→1).  
- **OASIS-2 (longitudinal)**: múltiples visitas; **target** por `Group` (Nondemented=0; Demented/Converted=1).  
- Siempre tomamos **una visita por paciente (baseline)** en OASIS-2 (mínimo `MR Delay`).

### Variables clínicas y tratamiento
- Columnas (base): `Age, Sex (one-hot), Education, SES, MMSE, CDR, eTIV, nWBV, ASF, Delay`.  
- **NaN críticos** (`MMSE`, `CDR`, `Target`) → se eliminan; **SES/Education** → imputación mediana.  
- **Escalado** en CV (dentro de fold) para evitar *leakage*.

### Imágenes MRI y pipeline de slices
- Conversión `.hdr/.img` → **slices axiales** (20 cortes centrales).  
- Normalización **z-score** y **CLAHE** opcional.  
- **Agrupación por paciente**: pooling (mean / trimmed / top-k / ensembles).

### Evitar *leakage*
- **Clínico**: solo **baseline** por paciente.  
- **MRI**: **split por paciente/scan** (todas las slices de un scan en el mismo subset).

---

## Cronología de pipelines (P1 → P27)

- **P1–P2 (clínico)**: baseline tabular (OASIS-2 y fusión OASIS-1+2) → AUC test clínico hasta ~0.99 (setup fusionado).  
- **P3–P14 (MRI)**: ResNet/EfficientNet-B3; calibración; pooling a nivel paciente.  
- **P16–P22 (ensembles/meta)**: OOF sin fuga, stacking/blending y calibración (Platt/Isotónica).  
- **P23**: umbrales **por cohorte** bajo **coste** (FN:FP=5:1).  
- **P24**: **meta simple interpretable** (LR-EN + Platt) — *mejor unimodal imagen*.  
- **P25**: consolidación de resultados (tabla maestra + gráficos).  
- **P26/P26b**: **intermodal Late** vs Mid + **Platt por cohorte**.  
- **P27**: **release reproducible** + **política S2**.

---

## Modelos clave y resultados

### P24 — Meta simple (imagen) LR-EN + Platt
- **Resultados (TEST)**:  
  - **ALL** AUC=**0.727**, PR-AUC=0.717, Brier=0.220  
  - **OAS1** AUC=**0.754**, PR-AUC=0.736, Brier=0.211  
  - **OAS2** AUC=**0.750**, PR-AUC=0.805, Brier=0.238  
- **Umbrales coste (5:1)**: **OAS1=0.435**, **OAS2=0.332**  
  - OAS1 → TP=14 FP=9 TN=18 FN=6 (R=0.70, P=0.61, Coste=39)  
  - OAS2 → TP=11 FP=7 TN=4 FN=1 (R=0.917, P=0.611, Coste=12)

### P26 — Intermodal (Late vs Mid)
- **Late (p_img + p_clin + p1_fill + p1_has)** elegido por **VAL**:  
  - VAL AUC=**0.916**, PR-AUC=0.921, Brier=0.111  
  - TEST AUC=**0.713**, PR-AUC=0.712, Brier=0.234
- **Mid (56 features imagen + clínico + p1)**:  
  - VAL AUC=**0.797**, PR-AUC=0.777, Brier=0.185  
  - TEST AUC=**0.697**, PR-AUC=0.657, Brier=0.230

### P26b — Calibración por cohorte
- **OAS1**: **AUC≈0.754**, **Brier≈0.199** (mejora vs Late raw).  
- **OAS2**: **AUC≈0.652**, **Brier≈0.241**.

### P27 — Política S2 y release
- **S2** = 5:1 por cohorte + ajuste en **OAS2** para **Recall ≥ 0.90**.  
- **Umbrales activos**: OAS1=**0.42**, OAS2=**0.4928655287824083**.  
- **Smoke (TEST @S2)**:  
  - OAS1 → TP=14 FP=9 TN=18 FN=6 (R=0.70, P=0.609, Coste=39)  
  - OAS2 → TP=11 FP=6 TN=5 FN=1 (R=0.917, P=0.647, Coste=11)

### Comparativas globales
- **Probabilidades (AUC/PR/Brier)**: **P24** lidera como unimodal imagen (robusto en OAS2).  
- **Decisión por coste (5:1)**: P24 y P26 son similares en OAS1 (misma confusión); en OAS2, P24 ofrece mejor recall con coste menor → base para el ajuste S2 en intermodal.

---

## Figuras y tablas finales

- Directorio: `/content/drive/MyDrive/CognitivaAI/p27_final/` y `/p25_informe_final/`, `/p26_intermodal/`.  
- **Comparativas**: AUC/PR-AUC/Brier por cohorte.  
- **Curvas**: ROC/PR y **coste vs umbral** por cohorte.  
- **Calibración**: ECE/MCE por cohorte.  
- **Tablas ejecutivas**: `p25_master_table.csv`, `p25_executive_table.md`, `p26b_percohort_platt_cost5to1.csv`, `p26_release/QA/p26b_test_report_recall_target.csv`.

---

## Reproducibilidad y *release*

- **Zip**: `p26_release.zip`  
  - **models/**: `p24_model.pkl`, `p24_platt.pkl`, `p26_clinical_model.pkl`  
  - **CONFIG/**: `deployment_config.json` (**S2 activa**) + `.backup.json`  
  - **QA/**: métricas de smoke/umbrales, ECE/MCE, figuras clave  
  - **DOCS/**: `MODEL_CARD.md`, `HOW_TO_DEPLOY.md`, `README_RELEASE.md`  
  - **Trazabilidad**: `MANIFEST.json`, `ENVIRONMENT.txt` (versionado; scikit-learn 1.7.1)

**Checklist reproducible**  
- Fijar semillas y versiones.  
- Respetar columnas esperadas por cada *pickle*.  
- IDs consistentes `OAS1_XXXX`/`OAS2_XXXX`.  
- OOF sin fuga; *split* por paciente.

---

## Cómo ejecutar (scripts, app y API)

### Scripts CLI

1) **Imagen → p_img (P24 + Platt)**  
```bash
python compute_pimg_from_features.py   --features path/patient_features.csv   --models_dir p26_release/models   --out p_img.csv
```

2) **Intermodal + política (Late + S2)**  
```bash
python predict_end_to_end.py   --pimg p_img.csv   --clinic clinical.csv   --models_dir p26_release/models   --config p26_release/CONFIG/deployment_config.json   --out predictions.csv
```

### App gráfica (Streamlit)
```bash
pip install streamlit pandas numpy scikit-learn==1.7.1 joblib
streamlit run app.py
```
- Subir CSVs (o usar **Modo Demo**).  
- Elegir **política** (S2 / Manual) y ajustar **sliders** por cohorte.  
- Ver métricas globales y por cohorte, matrices de confusión, coste y calibración.  
- Exportar CSV con resultados.

### API (FastAPI)
- `POST /predict` → entrada clínica + `p_img` **o** clínica + *features* (autocomputa `p_img`).  
- Respuesta: `p_img`, `p_clin`, `proba_cal`, `thr`, `decision`, cohorte.  
- Ver `FASTAPI_GUIDE.md` (endpoints, ejemplos y despliegue uvicorn).

---

## Desafíos principales

1) **Tamaño muestral reducido (N pequeño)**  
- Altísima varianza entre *splits*; CIs amplios (bootstrap).  
- Riesgo de sobreajuste y optimismo en CV.

2) **MRI en 2D (slices)**  
- Pérdida de continuidad 3D; sensibilidad a parámetros (CLAHE, normalización).

3) **Calibración y saturación de logits**  
- Necesidad de **Platt / temperature scaling**; descalibración especialmente en **OAS2** (ECE≈0.31 en P26 Late).

4) **Ensembles y NaNs**  
- Manejo fino de columnas escasas, OOF sin fuga, control de versiones.

5) **Infraestructura Colab/Drive**  
- Montajes inestables, rutas “fantasma” tras largas sesiones; reinicios necesarios.

6) **Trade-off Recall vs Precisión**  
- Para cribado se prioriza recall → mayor FP. Política basada en **coste** y **umbrales por cohorte**.

---

## Lecciones aprendidas

- **Clínico aporta una señal muy fuerte** en OASIS; **imagen** suma **complementariedad**.  
- **LR elastic-net + Platt** (P24) ofrece **robustez + interpretabilidad** en imagen.  
- **Late fusion** supera a Mid en nuestro N y heterogeneidad; **P26b** mejora Brier en OAS1.  
- **Políticas explícitas** (5:1, S2) facilitan **alineación clínica** (cribado vs confirmación).  
- **Organización de artefactos** y **nombres consistentes** ahorran horas de *debugging*.  
- **Monitoreo de calibración** (ECE/MCE) es imprescindible en despliegue.

---

## Limitaciones y próximos pasos

- **Generalización externa**: validar en datasets mayores (p. ej., **ADNI**).  
- **Calibración por sitio/cohorte**: recalibrar trimestralmente si **ECE>0.2**.  
- **Modelos 2.5D/3D**: probar arquitecturas volumétricas con regularización fuerte.  
- **Ensembles multimodales**: stacking con *Bayesian Optimization* en pesos.  
- **Explicabilidad**: SHAP (clínico) + Grad-CAM (MRI) para trazabilidad clínica.  
- **MLOps ligero**: *tracking* (MLflow), *docker* y *CI* para validar *smokes* automáticos.

---

## Conclusiones finales

- Con **datos limitados**, la **fusión intermodal** aporta valor cuando se **calibra** y se decide con **políticas clínicas**.  
- **P24 (imagen)** es un **pilar robusto**; **P26/P26b (intermodal)** ofrece una **vía práctica** de integración con clínica.  
- La **política S2** materializa la **prioridad clínica** de **minimizar FN** en OAS2, manteniendo OAS1 en 5:1.  
- Entregamos un **release reproducible**, **documentado** y listo para **demostración** (scripts + Streamlit + API).

---

## Agradecimientos y licencia

- **OASIS** por hacer disponibles las cohortes.  
- Infraestructura de **Google Colab** para experimentación GPU.  
- Licencia: ver `LICENSE` (a completar según preferencia del proyecto).

---

> **Rutas útiles**  
> - Tabla maestra: `p25_informe_final/p25_master_table.csv`  
> - Intermodal: `p26_intermodal/` (umbrales, reports y ECE/MCE)  
> - Release: `p26_release.zip` (modelos, config, QA, docs)  
> - Figuras finales: `p27_final/*.png`
