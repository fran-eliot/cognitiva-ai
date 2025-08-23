# 📓 Cuaderno de Bitácora — Proyecto COGNITIVA-AI

Este documento recoge de manera **cronológica, detallada y reflexiva** el proceso de desarrollo del proyecto **COGNITIVA-AI: Detección temprana de Alzheimer mediante datos clínicos y resonancias magnéticas (MRI)**.  

A diferencia del `README.md` (más divulgativo) y el `InformeTecnico.md` (más técnico-formal), aquí se presentan **todas las decisiones, problemas prácticos, soluciones y aprendizajes** a lo largo del camino, como un auténtico diario de investigación.  

---

## 📅 Línea Temporal de Avances

---

### 2025-01 — Inicio del proyecto
- **Objetivo:** explorar la **detección temprana de Alzheimer** usando tanto:
  - **datos clínicos tabulares** (edad, MMSE, CDR, volumetrías, etc.),
  - como **imágenes estructurales de resonancia magnética (MRI)**.  
- **Datasets seleccionados:**
  - **OASIS-1**: transversal, 416 sujetos.
  - **OASIS-2**: longitudinal, 150 sujetos, con variable `Group` explícita.  

**Problema inicial detectado:**  
Mi equipo local (ordenador modesto) **no podía entrenar redes neuronales profundas sobre MRI** por limitaciones de hardware (CPU).  
➡️ Se decidió comenzar por **modelos clásicos en datos clínicos**, para tener un baseline sólido.

---

### 2025-02 — Primer pipeline clínico (COGNITIVA-AI-CLINIC, OASIS-2)
- **Variables empleadas:**
  - `age` → edad en años (factor de riesgo principal en Alzheimer).
  - `sex` → sexo biológico (diferencias epidemiológicas).
  - `education` → años de educación (proxy de *reserva cognitiva*).
  - `SES` → estatus socioeconómico (relacionado con acceso a cuidados).
  - `MMSE` → Mini-Mental State Examination (test cognitivo clásico).
  - `CDR` → Clinical Dementia Rating (gravedad clínica de la demencia).
  - `eTIV`, `nWBV`, `ASF` → medidas volumétricas cerebrales.
- **Preprocesamiento:**
  - Imputación de valores perdidos (SES, educación) → mediana.
  - Escalado estándar.
  - Codificación one-hot para variables categóricas.
- **Modelos probados:** LR, Random Forest, XGBoost.

**Resultados:**
- AUC CV ≈ 0.91–0.92 (buen baseline).  
- Mejor modelo → **Random Forest (0.925 CV)**, **XGB (0.897 en test)**.  

**Conclusión parcial:**  
Con pocas variables clínicas ya se obtiene un buen rendimiento, pero el dataset es pequeño. Necesitamos **más robustez**.

---

### 2025-03 — Fusión OASIS-1 + OASIS-2 (COGNITIVA-AI-CLINIC-IMPROVED)
- **Objetivo:** aumentar tamaño muestral y estabilidad.
- **Decisiones clave:**
  - Unificación de columnas en *snake_case*.
  - En OASIS-2 solo baseline para evitar fuga de información.
  - Target unificado: `Group` (OASIS-2) y `CDR`>0 (OASIS-1).
  - Etiqueta de cohorte para trazabilidad.

**Resultados:**
- Hold-out inicial: AUC≈1.0 en LR, RF, XGB (pico alto por fusión).
- Validación cruzada 5-fold:
  - LR = 0.979 ± 0.012
  - RF = 0.974 ± 0.018
  - XGB = 0.975 ± 0.021
- Ensemble (LR+RF+XGB) ≈ **0.995 AUC**.  

**Conclusiones:**
- La fusión de datos clínicos dio **modelos muy estables**.  
- Variables críticas: **CDR y MMSE**.  
- Volumétricas aportan poco (sin ellas AUC ≈1.0).  
- Se ajustó **umbral clínico bajo** → maximizar recall (evitar falsos negativos).  

---

### 2025-04 — Primeros experimentos con MRI (COGNITIVA-AI-IMAGES, OASIS-2)
- **Pipeline:**
  - Conversión de `.hdr/.img` → cortes axiales `.png`.
  - Normalización y augmentations ligeros.
  - Entrenamiento de **ResNet50**.
- **Resultados:**
  - 5 slices → AUC=0.938 (mejor).
  - 20 slices + z-score → AUC=0.858 (más recall, menos precisión).

**Conclusión:**  
MRI es prometedor, pero muy dependiente del preprocesamiento y computacionalmente costoso.  
➡️ Localmente **no es viable entrenar** → necesidad de **migrar a Google Colab con GPU**.

---

### 2025-05 — Migración a Google Colab (GPU T4)
- Se adquirieron **100 unidades de computación Colab**.  
  - GPU T4 = ~1.32 unidades/hora.  
- Probamos también GPUs más potentes (A100, L4) y TPUs, pero se estandarizó en **T4**.  
- **Problema solucionado:** ya podemos entrenar y extraer embeddings con CNNs.  

---

### 2025-06 — MRI mejorado con embeddings ResNet18 (COGNITIVA-AI-IMAGES-IMPROVED)
- Generamos **embeddings ResNet18 (512D)**.
- Clasificador: **Logistic Regression**.
- **Calibración isotónica** (`CalibratedClassifierCV`).

**Resultados:**
- Slice-nivel (thr=0.5): AUC≈0.66.
- Paciente-nivel (mean pooling): AUC≈0.72.
- Ajustando umbral bajo (thr≈0.20):  
  - Recall=0.9 en VAL, Recall=0.8 en TEST.  
- Mejor Brier Score → **probabilidades más confiables**.

**Conclusión:**  
Aunque el AUC no subió, **la calibración mejoró la confiabilidad clínica**.  

---

### 2025-07 — Patient-features y ensembles
- Estrategia: resumir embeddings por paciente.  
- Modelos: LR, SVM, XGB, MLP.  
- Ensembles slice→patient + patient-features.  

**Resultados (ResNet18):**
- XGB slice→patient: AUC=0.733 TEST.  
- Ensemble híbrido (XGB+MLP): AUC≈0.744 TEST.  

**Conclusión:**  
Los ensembles **mejoran recall y precisión** en validación y test.  

---

### 2025-08 — MRI con EfficientNet-B3 en GPU (COGNITIVA-AI-IMAGES-IMPROVED-GPU)
- Modelo base: **EfficientNet-B3**, preentrenado.  
- Embeddings 1536D.  
- Paciente-nivel con PCA + ML clásicos.

**Resultados:**
- LR: AUC=0.685 (TEST).  
- XGB: AUC=0.670 (TEST).  
- MLP: AUC=0.648 (TEST).  
- **Ensemble LR+XGB:** AUC=0.704 TEST, Recall=0.90.  

**Conclusión:**  
EfficientNet-B3 mejora recall y estabilidad. El ensemble mantiene sensibilidad alta → valioso para cribado clínico.

---

## 📊 Comparativas y Gráficos

### MRI (EfficientNet-B3 patient-features, TEST)

- ROC-AUC  
  ![mri_effb3_pf_auc](./graphics_cognitiva_ai/mri_effb3_pf_auc.png)  

- PR-AUC  
  ![mri_effb3_pf_prauc](./graphics_cognitiva_ai/mri_effb3_pf_prauc.png)  

- Recall  
  ![mri_effb3_pf_recall](./graphics_cognitiva_ai/mri_effb3_pf_recall.png)  

- Precisión  
  ![mri_effb3_pf_precision](./graphics_cognitiva_ai/mri_effb3_pf_precision.png)  

- Accuracy  
  ![mri_effb3_pf_accuracy](./graphics_cognitiva_ai/mri_effb3_pf_accuracy.png)  

---

### Comparativa Global de Pipelines
![global_auc_comparison](./graphics_cognitiva_ai/global_auc_comparison_updated.png)  

---

## 📌 Reflexiones finales (hasta agosto 2025)

- **Clínico fusionado** sigue siendo el mejor pipeline (AUC≈0.995, interpretable y robusto).  
- **MRI embeddings + ensembles** logran AUC≈0.74 con alta sensibilidad (recall≈0.9).  
- La combinación multimodal (clínico + MRI) será el siguiente gran paso.  
- **Lecciones aprendidas:**
  - Importancia de calibrar y ajustar umbrales clínicos.  
  - Ensembles suelen estabilizar resultados en datasets pequeños.  
  - La migración a Colab GPU fue esencial para seguir avanzando.  

---

**Autoría:** Fran Ramírez  
**Última actualización:** Agosto 2025
