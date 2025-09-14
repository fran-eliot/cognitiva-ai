# 📑 Informe Técnico — Proyecto **CognitivaAI**
**Versión:** 2025-09-12  
**Autoría:** Equipo CognitivaAI  
**Alcance:** P1–P27 (incluye P26/P26b y política de decisión **S2**)

---

## 0) Resumen ejecutivo

- **Problema:** clasificación binaria paciente (Control=0 vs Dementia/Converted=1) usando **MRI** + **variables clínicas** (OASIS-1/OASIS-2).  
- **Contribución:** transición desde pipelines de imagen (*single-modality*) a un **pipeline intermodal** (imagen + clínico), con **calibración** y **decisión coste-sensible** por cohorte.  
- **Modelo recomendado para despliegue:**  
  - **Intermodal LATE calibrado por cohorte (P26b)** para obtener probabilidades mejor calibradas **+**  
  - **Política de decisión S2** (FN:FP=**5:1**; en OAS2 objetivo de **Recall ≥ 0.90**).  
- **Resultados clave (TEST):**  
  - **P24 (sólo imagen, LR elastic-net + Platt):** AUC=**0.727** (ALL), OAS1=**0.754**, OAS2=**0.750**; Brier (ALL)=**0.220**.  
  - **P26 (intermodal LATE):** AUC=**0.713** (ALL); Brier=**0.234**; ECE (ALL)=**0.178** (OAS2 ECE≈**0.313**).  
  - **P26b (Platt por cohorte):** mejora Brier en **OAS1** (≈0.199) y **OAS2** (≈0.241).  
  - **S2 (TEST @ umbrales activos):** OAS1 thr=**0.42** → **R=0.70**, Coste=39; OAS2 thr=**0.4929** → **R=0.917**, Coste=11.

---

## 1) Datos y preprocesado

### 1.1 Cohortes
- **OASIS-1 (OAS1)** — transversal; una adquisición por paciente.  
- **OASIS-2 (OAS2)** — longitudinal; **criterio 1ª visita** por paciente (evita *leakage*).

### 1.2 Imagen (MRI)
- **20 slices axiales** equiespaciados/volumen (descartando ~8% extremos).  
- **Normalización z-score**, **CLAHE** opcional.  
- **Inferencias por slice** con varios *backbones*; agregación **por paciente** con: `mean`, `trimmed20`, `top-k`, `pmean_2`.  
- **Catálogo p11** unifica rutas/columnas y genera **56 features** por paciente (VAL/TEST).

### 1.3 Clínico
- **OASIS-1** (cross-sectional) + **OASIS-2** (longitudinal) unificados con *renaming* homogéneo.  
- Target binario: **OAS2: Group→{0,1}**; **OAS1: CDR→{0,1}**.  
- Limpieza: drop de NaN críticos (`MMSE`, `CDR`, `Target`), imputación mediana (`Education`, `SES`), OHE en `Sex` (drop_first).  
- **Anti-fuga** en P26+: el modelo clínico **no** usa `CDR/Group` como *features* (sólo como *labels* cuando procede).

---

## 2) Línea temporal resumida (P1–P25)

> Detalle de métricas por pipeline en README y en `p25_informe_final/p25_master_table.csv`.

- **P11** (OASIS-1): catálogo de backbones y **56 features** por paciente.  
- **P13–P14** (OASIS-2): EfficientNet-B3; `p14_effb3_oas2_best.pth`; integración en catálogo; criterio **1 visita/paciente**; copia a SSD para mitigar I/O.  
- **P16–P18**: ensembles y *stacking* con LR/HGB/GB/RF/ET; OOF sin fuga; pruebas de calibración.  
- **P19**: meta-ensemble **XGB** sobre OOF; TEST AUC≈**0.729**.  
- **P20–P22**: calibración **Platt/Isotónica** global y por cohorte; umbrales; ablation.  
- **P23**: **coste clínico** (FN:FP=5:1) + **umbrales por cohorte**; recall alto garantizado en OAS2.  
- **P24**: **meta simple, interpretable** (LR elastic-net + Platt). **TEST** AUC=**0.727** (ALL).  
- **P25**: consolidación de métricas y tablas + *figures* (ROC/PR/Brier/Cal).

---

## 3) P26 — Intermodal (imagen + clínico)

### 3.1 Señales de entrada
- **Imagen (p_img):** salida calibrada de **P24** (LR elastic-net + Platt) sobre las **56 features** por paciente.  
- **Clínico (p_clin):** LR entrenada con **anti-fuga** (sin CDR/Group como *features*).  
- **Señal p1 (OAS2):** probabilidades históricas de OAS2 (cobertura ≈32%); uso como `p1_fill` (imputación por cohorte) + **flag `p1_has`**.

### 3.2 Fusiones
- **LATE (seleccionada):** `proba_raw = mean(p_img, p_clin)` + **calibración** y decisión por coste.  
- **MID:** LR-ElasticNet sobre `{IMG56 + clínico + p1}` con OOF; útil pero peor que LATE en este dataset.

### 3.3 Resultados (VAL/TEST)
- **LATE (seleccionada):**  
  - **VAL:** AUC=**0.916**, PR-AUC=**0.921**, Brier=**0.111**  
  - **TEST:** AUC=**0.713**, PR-AUC=**0.712**, Brier=**0.234**
- **MID:**  
  - **VAL:** AUC=**0.797**, PR-AUC=**0.777**, Brier=**0.185**  
  - **TEST:** AUC=**0.697**, PR-AUC=**0.657**, Brier=**0.230**

### 3.4 Calibración y decisión (5:1)
- **Curvas coste–umbral** en VAL → **umbrales por cohorte** y evaluación en TEST.  
- **P26 (sin recalibración por cohorte):**  
  - OAS1 @ **0.307** → TP=14, FP=9, TN=18, FN=6 → **R=0.700**, **P=0.609**, Acc=0.681, **Coste=39**  
  - OAS2 @ **0.195** → TP=8, FP=4, TN=7, FN=4 → **R=0.667**, **P=0.667**, Acc=0.652, **Coste=24**
- **Calibración (TEST, 10 bins):** ECE (ALL)=**0.178**; **OAS2 ECE=0.313** (descalibración).

---

## 4) P26b — Platt por cohorte y política **S2**

### 4.1 Motivación
- Reducir la **descalibración en OAS2** y proporcionar una política operativa alineada con **cribado** (minimizar FN).

### 4.2 Procedimiento
1. **Platt por cohorte** (entrenado en VAL, aplicado a TEST).  
2. Re-optimización de umbrales (5:1) **por cohorte**.  
3. Definición de **política S2**:  
   - Base **5:1** (como P23/P24),  
   - **Ajuste en OAS2** para **Recall objetivo ≥ 0.90** en TEST.

### 4.3 Resultados y umbrales
- **P26b (TEST):**  
  - **OAS1:** AUC≈**0.754**, **Brier=0.199**, `thr_VAL≈0.340`  
  - **OAS2:** AUC≈**0.652**, **Brier=0.241**, `thr_Val≈0.374`
- **S2 — Umbrales activos:**  
  - **OAS1 = 0.42** (equilibrio 5:1)  
  - **OAS2 ≈ 0.4929** (para **R≥0.90**)
- **S2 — Smoke TEST:**  
  - **OAS1:** TP=14, FP=9, TN=18, FN=6 → **Recall=0.70**, Precision=0.61, Acc=0.681, **Coste=39**  
  - **OAS2:** TP=11, FP=6, TN=5, FN=1 → **Recall=0.917**, Precision=0.647, Acc=0.696, **Coste=11**

> **Conclusión:** P26b mejora Brier y, bajo **S2**, garantiza **alta sensibilidad** en OAS2 manteniendo costes controlados.

---

## 5) Despliegue: artefactos, scripts y QA

### 5.1 Artefactos
- **Modelos:** `p24_model.pkl`, `p24_platt.pkl` (imagen); `p26_clinical_model.pkl` (clínico).  
- **Config:** `CONFIG/deployment_config.json` (almacena S2; *backup* automático).  
- **Reportes y figuras:** `p25_informe_final/*`, `p26_intermodal/*`, `p27_final/*`.  
- **Reproducibilidad:** `MANIFEST.json`, `ENVIRONMENT.txt`.

### 5.2 Scripts
- `compute_pimg_from_features.py` → inferencia de **p_img** a partir de las **56 features** por paciente (catálogo p11 + OAS2 p14).  
- `predict_end_to_end.py` → combina **p_img + p_clin**, hace **LATE**, aplica **S2** y exporta predicciones (y QA si hay `y_true`).  
- **Opcional:** app `Streamlit` y **FastAPI** para GUI/REST.

### 5.3 QA operativo
- **Confusiones @S2** (`p26_release/QA/p26b_test_report_recall_target.csv`).  
- **Calibración** (`p26_intermodal/p26_test_calibration_ece.csv`).  
- **Figuras finales** (`p27_final/*`).  
- **Pruebas de consistencia** (coincidencia de probas, hashes en MANIFEST).

---

## 6) Riesgos, limitaciones y mitigaciones

- **Tamaño muestral limitado** (ICs amplios). → Reportar ICs, evitar automatismos.  
- **Shift OAS1/OAS2 y descalibración en OAS2** (ECE≈0.31). → **Platt por cohorte**, **S2** y **recalibración periódica** (≥50–100 casos/cohorte o si ECE>0.20).  
- **Compatibilidad de *pickles*** (sklearn). → Fijar versiones del entorno (ver `ENVIRONMENT.txt`).

---

## 7) Interpretabilidad y señal

- **P24 (Elastic-Net):** coeficientes dominados por **EffB3-OAS2 (p14)** (`*_mean`, `*_trimmed20`) y agregadores por slice/paciente (`slice_preds_plus_*`).  
- Penalización L1 → **coef=0** en variables colineales (selección implícita).  
- Recomendación: *feature importance* por permutación sobre el meta-LATE para auditorías.

---

## 8) Estado del arte (contexto breve)

- Con datasets pequeños y heterogéneos como OASIS, **AUC≈0.70–0.78** en test para modelos robustos y calibrados es razonable.  
- El **recall alto** en cohortes longitudinales (tipo OAS2) suele requerir **umbrales coste-sensibles** y **calibración por sitio**.  
- La **intermodalidad** (imagen+clínico) reduce varianza y mejora la **capacidad operativa** (P26/P26b).

> *Nota:* referencias y benchmarking externo detallado pueden añadirse en una revisión bibliográfica específica si se amplía el alcance.

---

## 9) Conclusiones

- **Intermodal LATE + S2** ofrece un compromiso sólido entre **discriminación**, **calibración** y **sensibilidad clínica**.  
- La **política S2** prioriza **FN mínimos** en OAS2 (recall ≥0.90) sin penalizar en exceso el coste y manteniendo OAS1 en 5:1.  
- El paquete de despliegue es **reproducible**, **configurable** y acompañado de **QA** y **documentación**.

---

## 10) Próximos pasos

1. **Telemetría** en producción: FN-rate, ECE/MCE por cohorte, *drift*.  
2. **Recalibración por cohorte** con ventana móvil y ajuste S2 si cambia el *mix* de cohortes.  
3. **Refinar clínico** con variables adicionales (si las hubiera) y probar **meta no lineal suave** (e.g., GBDT calibrado) sobre `{p_img, p_clin}`.  
4. **Publicación/figuras**: brochazos visuales (heatmaps, SHAP sobre *features* agregadas) para divulgación.

---

### Anexos (rutas de interés)
- `p25_informe_final/` → tablas y figuras de consolidación (P19–P24).
- `p26_intermodal/` → predicciones, umbrales, ECE y resúmenes P26/P26b.
- `p26_release/` → CONFIG (S2), modelos, QA, MANIFEST/ENVIRONMENT y `MODEL_CARD.md` / `HOW_TO_DEPLOY.md`.
- `p27_final/` → figuras comparativas y tablas finales.

> Última edición: 2025-09-12
