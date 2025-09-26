# Model Card — CognitivaAI Intermodal (P26/P27)

**Versión:** P27 (intermodal LATE + calibración por cohorte + política S2)  
**Tarea:** Predicción binaria (0=Control, 1=Dementia/Converted) a nivel **paciente**.  
**Entradas:**  
- **Imagen → `p_img`** (probabilidad calibrada con Platt a partir de features por paciente; base P24).  
- **Clínico → `p_clin`** (LR sobre variables tabulares estandarizadas).  
**Fusión:** **LATE** (combinación sobre probabilidades calibradas).  
**Cohortes:** OASIS-1 (cross-sectional) y OASIS-2 (longitudinal, 1 visita/paciente).  

---

## 1) Uso previsto
Sistema de **cribado** para apoyar la decisión clínica en evaluación cognitiva, con especial énfasis en **sensibilidad** (minimizar FN) y **calibración** de probabilidades. No sustituye el juicio clínico; requiere validación local.

---

## 2) Datos y entrenamiento (resumen)
- **Imagen**: 20 *slices* axiales/volumen, normalización z-score (+CLAHE opc.), agregación por paciente; meta-modelo P24 (LR elastic-net + Platt).  
- **Clínico**: columnas mínimas `Age, Sex, Education, SES, MMSE, eTIV, nWBV, ASF, Delay, patient_id` (imputación mediana/one-hot básico).  
- **P26**: intermodal **LATE**; **P26b**: recalibración **Platt por cohorte**.  
- **P27**: empaquetado reproducible, política de decisión **S2** y QA final.

---

## 3) Métricas de probabilidad (TEST)
**Intermodal LATE (P26):**  
- **ALL:** AUC **0.736**, PR-AUC **0.729**, Brier **0.229**  
- **OAS1:** AUC **0.754**, PR-AUC **0.736**, Brier **0.208**  
- **OAS2:** AUC **0.652**, PR-AUC **0.728**, Brier **0.288**

> Fuente: `p25_informe_final/p25_master_table.csv` (filas P26) y figuras en `p27_final/`.

---

## 4) Política de decisión **S2** (activa)
**Objetivo:** maximizar sensibilidad sin colapsar en “todo positivo” en dominios tipo OAS2.

- **OAS1 → 5:1 (FN:FP)** con umbral aprendido en VAL → **thr = 0.42**  
- **OAS2 → “recall objetivo” (VAL, target≈0.85; aplicado en TEST)** → **thr ≈ 0.4928655**

**Resultados TEST @S2 (confusiones y métricas):**
- **OAS1 (0.42):** TP=14, FP=9, TN=18, FN=6 → **Recall=0.700**, Precision=0.609, Acc=0.681, **Coste=39**  
- **OAS2 (≈0.4929):** TP=11, FP=6, TN=5, FN=1 → **Recall=0.917**, Precision=0.647, Acc=0.696, **Coste=11**  

**Dónde cambiar:** `p26_release/CONFIG/deployment_config.json`  
- `thresholds = {"OAS1": 0.42, "OAS2": 0.4928655287824083}`  
- `thresholds_5to1 = {"OAS1": 0.42, "OAS2": 0.49}` *(fallback 5:1 puro)*

---

## 5) Calibración y monitorización
- **ECE/MCE (TEST intermodal, P26):** ALL≈0.178 / OAS1≈0.150 / **OAS2≈0.313** → monitorizar y **recalibrar** por cohorte si **ECE>0.20** o hay drift (sitio/escáner/población).  
- Recomendar telemetría de **TP/FP/TN/FN**, **tasa de positivos** y **ECE** por cohorte. Recalibración con ventana móvil (≥50–100 casos).

---

## 6) Limitaciones
- Tamaños por cohorte moderados → **IC amplios**.  
- **Shift** entre OAS1/OAS2 → aplicar **umbrales por cohorte** y validación local antes de uso asistencial.  
- El modelo **no** es un diagnóstico automático; es soporte a la decisión.

---

## 7) Cómo ejecutar (resumen)
1. **Imagen → `p_img`:** `compute_pimg_from_features.py` sobre las matrices de features por paciente (catálogo P11 + OAS2 p14).  
2. **Clínico → `p_clin`:** CSV con columnas mínimas (arriba).  
3. **Inferencia E2E:** `predict_end_to_end.py` → combina `p_img + p_clin` (LATE), **calibra por cohorte** (P26b) y **aplica S2**.  
4. Salidas: CSV de probabilidades/calibradas, decisión (0/1), y **QA** con confusiones/Coste.

---

## 8) Versionado y reproducibilidad
- **Release:** `p26_release/` (zip con 23 ficheros).  
- **Modelos:** `p24_model.pkl`, `p24_platt.pkl`, `p26_clinical_model.pkl`.  
- **Config:** `CONFIG/deployment_config.json` (+ backups).  
- **QA:** `QA/p26b_test_report_recall_target.csv`.  
- **Trazas:** `MANIFEST.json`, `ENVIRONMENT.txt`.

---

## 9) Figuras (collage rápido)
<table>
<tr>
<td width="50%">
<b>AUC — ALL (TEST)</b><br>
<img src="../p27_final/p27_auc_ALL.png" alt="AUC ALL" />
</td>
<td width="50%">
<b>OAS2 · S2 vs 5:1 (TEST)</b><br>
<img src="../p27_final/p27_s2_vs_5to1_OAS2.png" alt="S2 vs 5:1 OAS2" />
</td>
</tr>
</table>

> Más figuras en `p27_final/`:
> - `p27_auc_*.png`, `p27_prauc_*.png`, `p27_brier_*.png`
> - (si existe) `p27_s2_vs_5to1_OAS2.png`

---
