# 🧠 Proyecto de Detección Temprana de Alzheimer (COGNITIVA-AI)

Este proyecto explora la **detección temprana de la enfermedad de Alzheimer** combinando **datos clínicos tabulares** y **resonancias magnéticas estructurales (MRI)** de los conjuntos **OASIS-1 y OASIS-2**.  

El enfoque se diseñó con una idea central: **replicar el razonamiento clínico** usando tanto la información disponible en la historia del paciente (tests neuropsicológicos, edad, educación, volumen cerebral) como en las **imágenes estructurales cerebrales**.  

Se construyeron **siete pipelines** para analizar y comparar modalidades:  

1. **COGNITIVA-AI-CLINIC** → ML clásico con datos clínicos (solo OASIS-2).  
2. **COGNITIVA-AI-CLINIC-IMPROVED** → ML clásico con datos clínicos fusionados OASIS-1 + OASIS-2.  
3. **COGNITIVA-AI-IMAGES** → Deep Learning con MRI (solo OASIS-2, ResNet50).  
4. **COGNITIVA-AI-IMAGES-IMPROVED** → fusión de OASIS-1+2 en imágenes.  
5. **COGNITIVA-AI-IMAGES-IMPROVED-GPU** → embeddings ResNet18 entrenados en **Google Colab (GPU)**.  
6. **COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED (EffNet-B3)** → embeddings EfficientNet-B3 + ensemble LR+XGB a nivel paciente.  
7. **COGNITIVA-AI-FINETUNING** → Fine-tuning directo de EfficientNet‑B3 en **Google Colab (GPU)** con *temperature scaling* y agregación a **nivel paciente**.

---

## 📦 Datos y Variables Clínicas

Los datos provienen de los proyectos **OASIS-1** y **OASIS-2**:

- **OASIS-1 (transversal):** 416 sujetos, una sola visita por paciente.  
  - No tiene variable `Group`, la severidad se deduce a partir de **CDR** (`0=No demencia`, `>0=Demencia`).  

- **OASIS-2 (longitudinal):** 150 sujetos, múltiples visitas.  
  - Incluye `Group` (`Nondemented`, `Demented`, `Converted`).  

**Variables clínicas empleadas:**

- **Age** → Edad del paciente en la visita inicial. Factor de riesgo primario en Alzheimer.  
- **Sex** → Sexo biológico. El Alzheimer presenta prevalencias distintas en mujeres.  
- **Educ** → Años de educación formal. Factor protector (mayor reserva cognitiva).  
- **SES** (Socioeconomic Status) → Escala 1–5 (mayor valor = mayor estatus). Se ha relacionado con acceso a recursos cognitivos.  
- **MMSE** (Mini-Mental State Examination) → Test neuropsicológico de 0–30. Valores bajos indican deterioro cognitivo.  
- **CDR** (Clinical Dementia Rating) → Escala clínica (0=normal, 0.5=mild, 1=moderate, 2–3=severe). Considerado estándar de oro para diagnóstico.  
- **eTIV** (Estimated Total Intracranial Volume) → Volumen craneal estimado, usado para normalizar medidas volumétricas.  
- **nWBV** (Normalized Whole Brain Volume) → Proporción de volumen cerebral respecto al intracraneal. Refleja atrofia cerebral.  
- **ASF** (Atlas Scaling Factor) → Factor de escalado anatómico aplicado en el registro.  

Estas variables combinan **información clínica y volumétrica**, proporcionando una visión integral de factores de riesgo y biomarcadores estructurales.

---

# 1️⃣ COGNITIVA-AI-CLINIC (solo OASIS-2)

- **Preprocesamiento**: imputación SES/Educación por mediana, escalado estándar, codificación one-hot.  
- **Modelos**: Logistic Regression, Random Forest, XGBoost.  

### 📊 Resultados
- Regresión Logística → **0.912 ± 0.050 (CV)**  
- Random Forest → **0.925 ± 0.032 (CV)**  
- XGBoost → **0.907 ± 0.032 (CV)**  
- Mejor en test: **XGBoost = 0.897 AUC**  

➡️ Primer baseline, estable pero dataset reducido (150 sujetos).  

---

# 2️⃣ COGNITIVA-AI-CLINIC-IMPROVED (fusión OASIS-1 + OASIS-2)

- **Unificación de columnas** (`snake_case`).  
- **Selección baseline** en OASIS-2.  
- **Target unificado**: `Group` (OASIS-2) o `CDR` (OASIS-1).  
- **Etiquetas de cohortes** para trazabilidad.  

### 📊 Resultados
- **Hold-out inicial (80/20):** LogReg=1.000 | RF=0.986 | XGB=0.991  
- **Validación cruzada (5-fold):**  
  - LogReg → **0.979 ± 0.012**  
  - RF → **0.974 ± 0.018**  
  - XGB → **0.975 ± 0.021**  

➡️ Modelos muy estables con excelente generalización.  

**Umbral clínico (XGB):** recall≈100% con 15 falsos positivos.  
**Interpretación:** mejor tolerar falsos positivos que falsos negativos.  

---

# 3️⃣ COGNITIVA-AI-IMAGES (MRI OASIS-2)

- **Pipeline**: conversión `.hdr/.img` a slices, normalización, augmentations ligeros.  
- **Modelo**: ResNet50 fine-tuning.  

### 📊 Resultados
- 5 slices → **AUC=0.938 (test)**  
- 20 slices + z-score → AUC=0.858 (mayor recall, menor precisión).  

➡️ Buen baseline, costoso en CPU.  

---

# 4️⃣ COGNITIVA-AI-IMAGES-IMPROVED

- **Split paciente/scan** estricto.  
- **Más slices** por paciente.  

### 📊 Resultados
- Pipeline más robusto, pero alto coste computacional en CPU.  

---

# 5️⃣ COGNITIVA-AI-IMAGES-IMPROVED-GPU (ResNet18 + Calibración)

- **Embeddings ResNet18 (512D)**.  
- Clasificación con **Logistic Regression**.  
- **Calibración isotónica**.  

### 📊 Resultados
- **Slice-nivel:** AUC≈0.66 | Brier≈0.23.  
- **Paciente-nivel (thr≈0.20, recall≥0.90):**  
  - [VAL] Recall=0.90 | Precision=0.60 | AUC=0.722  
  - [TEST] Recall=0.80 | Precision=0.52 | AUC=0.724  

➡️ Probabilidades más confiables tras calibración.  

---

# 6️⃣ COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED (EffNet-B3)

- **Embeddings EfficientNet-B3 (1536D)**.  
- Modelos: LR, XGB, MLP a nivel paciente.  
- **Ensemble LR+XGB** ponderado por PR-AUC.  

### 📊 Resultados
- [VAL] AUC=0.815 | PR-AUC=0.705 | Recall=0.95 | Acc=0.70  
- [TEST] AUC=0.704 | PR-AUC=0.623 | Recall=0.90 | Acc=0.70  

➡️ Mejor pipeline MRI hasta la fecha, con sensibilidad alta.  

---

# 7️⃣ COGNITIVA-AI-FINETUNING (EfficientNet-B3 Fine-Tuning en GPU, resultados finales)

- **Notebook:** `cognitiva_ai_finetuning.ipynb` (Colab GPU)  
- **Pooling paciente:** mean  
- **Calibración:** *temperature scaling* con T=2.673  
- **Umbral clínico:** 0.3651  

### 📊 Resultados finales (nivel paciente, n=47)
- **VAL** → AUC=0.748 | PR-AUC=0.665 | Acc=0.702 | Precision=0.588 | Recall=1.0  
- **TEST** → AUC=0.876 | PR-AUC=0.762 | Acc=0.745 | Precision=0.625 | Recall=1.0  

**Matriz de confusión TEST (reconstruida, thr=0.3651):**  
TP=8, FP=5, TN=34, FN=0

### 🖼️ Gráficas
- `graphs_from_metrics/ft_b3_patient_confusion_from_metrics.png`  
- `graphs_from_metrics/ft_b3_pr_point.png`  
- `graphs_from_metrics/ft_b3_bars_auc.png`  
- `graphs_from_metrics/ft_b3_bars_prauc.png`  

---

# 📊 Comparativa Global

<p align="center">
  <img src="./graficos/global_auc_comparison_updated.png" alt="Comparativa Global — ROC-AUC por Pipeline" width="880"/>
</p>

---

# 📈 Visualizaciones MRI EffNet-B3 (patient-features)

<p align="center">
  <img src="./graficos/mri_effb3_pf_auc.png" alt="MRI EffB3 patient-features — ROC-AUC (TEST)" width="720"/>
</p>

<p align="center">
  <img src="./graficos/mri_effb3_pf_prauc.png" alt="MRI EffB3 patient-features — PR-AUC (TEST)" width="720"/>
</p>

<p align="center">
  <img src="./graficos/mri_effb3_pf_recall.png" alt="MRI EffB3 patient-features — Recall (TEST)" width="720"/>
</p>

<p align="center">
  <img src="./graficos/mri_effb3_pf_precision.png" alt="MRI EffB3 patient-features — Precisión (TEST)" width="720"/>
</p>

<p align="center">
  <img src="./graficos/mri_effb3_pf_accuracy.png" alt="MRI EffB3 patient-features — Accuracy (TEST)" width="720"/>
</p>

---

**Autoría:** Fran Ramírez  
**Año:** 2025
