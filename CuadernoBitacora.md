# 📖 Cuaderno de Bitácora – Proyecto COGNITIVA-AI

Este documento actúa como **diario detallado de investigación**, complementando al `README.md` (resumen ejecutivo) y al `InformeTecnico.md` (documentación formal).  

Aquí se incluyen **todas las fases del proyecto**, así como las **entradas diarias (dailys)** con los resultados obtenidos, problemas técnicos y conclusiones.  

---

# 🗂️ Fases Globales

## Fase 1 – Datos clínicos OASIS-2 (pipeline inicial)

**Contexto:**  
Se comenzó con un enfoque tabular sencillo sobre OASIS-2, trabajando con variables clínicas estándar.

**Variables principales:**
- `AGE`: edad del paciente.  
- `M/F`: sexo biológico.  
- `EDUC`: años de educación formal (relacionado con reserva cognitiva).  
- `SES`: estatus socioeconómico.  
- `MMSE`: Mini-Mental State Examination (test cognitivo).  
- `CDR`: Clinical Dementia Rating (gravedad clínica).  
- `eTIV`: volumen intracraneal estimado.  
- `nWBV`: volumen cerebral normalizado.  
- `ASF`: factor de escala anatómico.  

**Resultados clave:**

| Modelo | AUC (CV 5-fold) | AUC Test |
|--------|-----------------|----------|
| Logistic Regression | 0.912 ± 0.050 | — |
| Random Forest        | 0.925 ± 0.032 | — |
| XGBoost              | 0.907 ± 0.032 | **0.897** |

**Gráfico:**  
![Resultados clínicos OASIS-2](./graficos/clinic_oasis2.png)

**Conclusión:**  
Pipeline sencillo y robusto, pero dataset limitado (150 sujetos).

---

## Fase 2 – Fusión clínica OASIS-1 + OASIS-2

**Contexto:**  
Para ganar robustez, se unieron OASIS-1 (transversal) y OASIS-2 (longitudinal).  

**Pasos clave:**
- Homogeneización de columnas (`snake_case`).  
- Selección baseline en OASIS-2.  
- Target unificado (`0 = Nondemented`, `1 = Demented/Converted`).  
- Imputación SES/Educación con mediana.  
- Etiqueta de cohorte.  

**Resultados clave:**

| Modelo | Hold-out (80/20) | CV 5-fold | Nested CV (10x5) |
|--------|-----------------|-----------|------------------|
| Logistic Regression | 1.000 | 0.979 ± 0.012 | — |
| Random Forest        | 0.986 | 0.974 ± 0.018 | — |
| XGBoost              | 0.991 | 0.975 ± 0.021 | — |
| Ensemble (LR+RF+XGB) | —     | —             | **0.995** |

**Gráfico:**  
![Fusion clínica OASIS1+2](./graficos/clinic_fusion.png)

**Conclusión:**  
Dataset combinado muy estable, modelos calibrados y con gran generalización. Interpretabilidad clínica: **CDR + MMSE críticos**.

---

## Fase 3 – MRI en CPU local (ResNet50 baseline)

**Contexto:**  
Primeros experimentos con MRI desde imágenes (OASIS-2).  

**Resultados clave:**

| Configuración | AUC (Test) |
|---------------|------------|
| ResNet50 (5 slices, sin CLAHE) | **0.938** |
| ResNet50 (20 slices, z-score) | 0.858 |

**Gráfico:**  
![MRI baseline ResNet50](./graficos/mri_resnet50_baseline.png)

**Conclusión:**  
Buen desempeño, pero costoso en CPU local → se decide migrar a **Google Colab con GPU**.

---

## Fase 4 – Google Colab GPU (ResNet18 embeddings + calibrado)

**Contexto:**  
Migración a Google Colab (GPU T4). Se generan embeddings ResNet18 (512d) y se calibran con regresión logística isotónica.

**Resultados clave:**

| Nivel        | Dataset | AUC  | PR-AUC | Acc  | Recall | Precision | Brier |
|--------------|---------|------|--------|------|--------|-----------|-------|
| Slice        | VAL     | 0.627 | 0.538 | 0.62 | 0.43   | 0.57      | 0.296 |
| Slice        | TEST    | 0.661 | 0.535 | 0.62 | 0.47   | 0.57      | 0.289 |
| Paciente (thr=0.204) | VAL | 0.722 | 0.634 | 0.70 | 0.90 | 0.60 | — |
| Paciente (thr=0.204) | TEST | 0.724 | 0.606 | 0.60 | 0.80 | 0.52 | — |

**Gráfico:**  
![ROC Curves – Colab GPU ResNet18](./graficos/roc_colab_resnet18.png)

**Conclusión:**  
El calibrado isotónico **mejora Brier Score**, y con umbral clínico bajo logramos **recall alto (0.8 test)** → adecuado para cribado.

---

## Fase 5 – Clasificadores alternativos y ensemble (slice→patient)

**Resultados clave:**

| Modelo | AUC (Val) | AUC (Test) | PR-AUC (Val) | PR-AUC (Test) |
|--------|-----------|------------|--------------|---------------|
| SVM    | 0.731     | 0.746      | 0.618        | 0.628         |
| XGB    | 0.743     | 0.733      | 0.644        | 0.605         |
| Ensemble (LR+SVM+XGB) | 0.728 | 0.728 | 0.641 | 0.605 |

**Gráfico:**  
![Comparativa SVM-XGB-Ensemble](./graficos/ensemble_resnet18.png)

**Conclusión:**  
Ensemble mejora estabilidad, recall ~0.8 en test.

---

## Fase 6 – EfficientNet-B3 embeddings

**Contexto:**  
Se generan embeddings más ricos (1536d) con EfficientNet-B3.  

**Resultados clave (paciente-nivel):**

| Modelo | VAL AUC | VAL PR-AUC | TEST AUC | TEST PR-AUC | Recall (Test) | Precision (Test) |
|--------|---------|------------|----------|-------------|---------------|------------------|
| LR     | 0.786   | 0.732      | 0.685    | 0.539       | 0.80          | 0.52             |
| MLP    | 0.870   | 0.886      | 0.648    | 0.556       | 0.95          | 0.53             |
| XGB    | 0.782   | 0.633      | 0.670    | 0.617       | 0.75          | 0.56             |
| Ensemble (LR+XGB) | 0.815   | 0.705      | 0.704    | 0.623       | 0.90          | 0.60             |

**Gráfico:**  
![EfficientNet-B3 comparativa](./graficos/effnetb3_val_test.png)

**Conclusión:**  
EffNet-B3 genera embeddings más ricos; los clasificadores simples sobreajustan, pero el **ensemble logra equilibrio** con recall clínico aceptable.

---

# 📅 Entradas Diarias (Agosto 2025)

### 📅 18/08/2025 – Migración a Colab GPU
- **Acción:** montaje de Google Drive, embeddings ResNet18, calibrado isotónico.  
- **Resultado:** AUC estable ~0.72, recall mejorado con umbral bajo.  
- **Problema:** pérdida de entorno → se reconstruyeron celdas iniciales.  
- **Conclusión:** base sólida para MRI en GPU.

---

### 📅 21/08/2025 – Experimentación con EfficientNet-B3
- **Acción:** embeddings 1536d, clasificadores LR/MLP/XGB.  
- **Resultados:** LR estable, MLP con alto overfitting, ensemble mejora recall y precisión en test.  
- **Conclusión:** embeddings más ricos abren la puerta a ensembles más sofisticados.

---

### 📅 23/08/2025 – Ensemble híbrido
- **Acción:** combinación XGB slice→patient con MLP patient-features.  
- **Resultados:** recall en test = 0.90, precisión ~0.60.  
- **Conclusión:** validación de estrategia **híbrida** → clave para la futura fusión multimodal.

---

### 📅 24/08/2025 – Consolidación Fine-Tuning EfficientNet-B3 en Colab
- **Acción:** se re‑ejecutó el notebook `cognitiva_ai_finetuning.ipynb` desde cero en Colab, generando el archivo `ft_effb3_patient_eval.json`.  
- **Resultados (paciente, n=47):**  
  - VAL: AUC=0.748 | PR-AUC=0.665 | Acc=0.702 | P=0.588 | R=1.0  
  - TEST: AUC=0.876 | PR-AUC=0.762 | Acc=0.745 | P=0.625 | R=1.0  
- **Matriz de confusión (TEST, thr=0.3651):** TP=8, FP=5, TN=34, FN=0.  
- **Archivos generados:** gráficas en `graphs_from_metrics/` (confusión, punto PR, barras AUC/PR-AUC).  
- **Conclusión:** se confirma la estabilidad del modelo fine‑tuned EfficientNet‑B3 como mejor pipeline MRI del proyecto.  

---

# 📌 Conclusión global
- Clínico (fusionado OASIS1+2) → mejor AUC global (≈0.99).  
- MRI en GPU → resultados robustos (AUC ~0.72, recall alto tras calibrado).  
- EffNet-B3 → embeddings más ricos, ensemble mejora equilibrio.  
- **Fine-Tuning B3 (Colab)** → recall 1.0 en test con precisión 0.625 (cribado).  
- Próximos pasos → **multimodal (clínico+MRI)** y validación externa (OASIS-3/ADNI).

---

**Autoría:** Fran Ramírez  
**Última actualización:** 24/08/2025 – 20:21
