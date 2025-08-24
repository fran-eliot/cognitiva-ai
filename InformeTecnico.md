# 📘 Informe Técnico — COGNITIVA-AI

## 1. Introducción
Objetivo: detección temprana de Alzheimer combinando **datos clínicos** y **MRI** (OASIS‑1/2).  
Enfoque: comparar pipelines unimodales y preparar la base para un modelo **multimodal**.

## 2. Datos
- OASIS‑1 (transversal), OASIS‑2 (longitudinal).  
- Variables clínicas: Age, Sex, Educ, SES, MMSE, CDR, eTIV, nWBV, ASF.  
- Etiquetas: `0 = Nondemented`, `1 = Demented/Converted` (armonizado).

## 3. Metodología
- Preprocesado clínico (imputación, escalado, codificación).  
- MRI: extracción de slices, normalización, augmentations ligeras.  
- Validaciones: hold‑out, CV 5‑fold, y test con **split por paciente**.  
- Métricas: ROC‑AUC, PR‑AUC, Accuracy, Precision, Recall, Brier.  
- Calibración: isotónica (embeddings) y *temperature scaling* (fine‑tuning).

## 4. Resultados por Pipeline

### 4.1 Clínico (OASIS‑2)
- LogReg: 0.912 ± 0.050 (CV), XGB test: 0.897 AUC.  

### 4.2 Clínico fusionado (OASIS‑1 + OASIS‑2)
- CV 5‑fold: LogReg 0.979 ± 0.012, RF 0.974 ± 0.018, XGB 0.975 ± 0.021.  

### 4.3 MRI baseline (ResNet50)
- 5 slices test AUC=0.938; 20 slices z‑score AUC=0.858.  

### 4.4 MRI en GPU (ResNet18 + calibración)
- Paciente (thr≈0.20): VAL AUC=0.722 / TEST AUC=0.724; Recall test≈0.80.  

### 4.5 Clasificadores alternativos y ensemble
- Ensemble (LR+SVM+XGB): AUC test=0.728; PR‑AUC=0.605.

### 4.6 EfficientNet‑B3 embeddings (1536D)
- LR/MLP/XGB a nivel paciente; **Ensemble (LR+XGB)** mejor equilibrio.  
- VAL AUC=0.815 | PR‑AUC=0.705; TEST AUC=0.704 | PR‑AUC=0.623; Recall test≈0.90.

### 4.7 MRI EfficientNet‑B3 Fine‑Tuning (Colab GPU, resultados finales)
- **Notebook:** `cognitiva_ai_finetuning.ipynb`  
- **Configuración:** fine‑tuning parcial de EfficientNet‑B3, pooling mean, calibración por *temperature scaling* (T=2.673), **umbral clínico**=0.3651.  
- **Origen de resultados:** archivo agregado `ft_effb3_patient_eval.json` (sin predicciones individuales).  

#### 📊 Resultados (nivel paciente, n=47)
| Split | AUC   | PR-AUC | Acc   | Precision | Recall | Thr    |
|------:|:-----:|:------:|:-----:|:---------:|:------:|:------:|
| VAL   | 0.748 | 0.665  | 0.702 | 0.588     | 1.0    | 0.3651 |
| TEST  | 0.876 | 0.762  | 0.745 | 0.625     | 1.0    | 0.3651 |

**Matriz de confusión (TEST, thr=0.3651, reconstruida):**  
TP=8, FP=5, TN=34, FN=0

#### 🖼️ Gráficas derivadas
- Matriz de confusión: `graphs_from_metrics/ft_b3_patient_confusion_from_metrics.png`  
- Punto PR: `graphs_from_metrics/ft_b3_pr_point.png`  
- Barras de AUC: `graphs_from_metrics/ft_b3_bars_auc.png`  
- Barras de PR-AUC: `graphs_from_metrics/ft_b3_bars_prauc.png`  

> Nota: al no disponer de predicciones individuales, no es posible trazar curvas ROC/PR completas; se representan métricas agregadas y la matriz reconstruida.

## 5. Discusión
- Clínico fusionado queda como **línea base de referencia** (AUC ~0.99).  
- MRI aporta **valor de cribado**; el **fine‑tuning B3** logra **recall=1.0** con precisión moderada (0.625).  
- La calibración (*T scaling*) aporta **probabilidades más útiles** para la toma de decisiones clínicas con umbral fijo.  

## 6. Conclusiones y Próximos Pasos
- Consolidado el **pipeline MRI** con B3 fine‑tuning en GPU.  
- Siguiente fase: **fusión multimodal (clínico + MRI)** y validación externa (OASIS‑3/ADNI).  
- Añadir guardado de **predicciones por paciente** en JSON para curvas ROC/PR futuras.

**Fecha:** 24/08/2025  
**Autoría:** Fran Ramírez
