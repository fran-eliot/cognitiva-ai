# 🧠 Proyecto de Detección Temprana de Alzheimer (COGNITIVA-AI)

Este proyecto explora la **detección temprana de la enfermedad de Alzheimer** combinando **datos clínicos tabulares** y **resonancias magnéticas estructurales (MRI)** de los conjuntos **OASIS-1 y OASIS-2**.  

El enfoque se diseñó con una idea central: **replicar el razonamiento clínico** usando tanto la información disponible en la historia del paciente (tests neuropsicológicos, edad, educación, volumen cerebral) como en las **imágenes estructurales cerebrales**.  

Actualmente se han desarrollado **6 pipelines**:

1. **COGNITIVA-AI-CLINIC** → ML clásico con datos clínicos (solo OASIS-2).  
2. **COGNITIVA-AI-CLINIC-IMPROVED** → ML clásico con datos clínicos fusionados OASIS-1 + OASIS-2.  
3. **COGNITIVA-AI-IMAGES** → Deep Learning con MRI (solo OASIS-2, ResNet50).  
4. **COGNITIVA-AI-IMAGES-IMPROVED** → *(en progreso)* fusión de OASIS-1+2 en imágenes.  
5. **COGNITIVA-AI-IMAGES-IMPROVED-GPU** → embeddings ResNet18 + clasificadores en Google Colab con GPU.  
6. **COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED** → versión calibrada con `CalibratedClassifierCV`, exploración de SVM/XGB/MLP y ensemble híbrido.  

---

## 📦 Datos y Alcance

- **OASIS-1 (transversal):** 416 sujetos, una sola visita por paciente.  
  - No tiene variable `Group`, por lo que la severidad se deduce a partir de **CDR** (`0=No demencia`, `>0=Demencia`).  

- **OASIS-2 (longitudinal):** 150 sujetos, múltiples visitas.  
  - Tiene variable `Group` (`Nondemented`, `Demented`, `Converted`).  

- **MRI:** archivos `.hdr/.img` por paciente, con segmentaciones asociadas (`FSL_SEG`).  

**Target unificado (binario):**  
- `0 = Nondemented`  
- `1 = Demented` o `Converted`  

> ⚠️ **Control estricto de fugas de información:**  
> - En clínico → seleccionamos **solo la visita baseline** de cada paciente.  
> - En MRI → los splits son estrictamente por **paciente/scan_id**.  

---

# 1️⃣ COGNITIVA-AI-CLINIC (solo OASIS-2)

### 🔧 Decisiones de diseño
- Variables: edad, sexo, educación, SES, MMSE, CDR, eTIV, nWBV, ASF.  
- Preprocesamiento: imputación por mediana en SES y educación, escalado estándar, codificación one-hot.  
- Modelos probados: **Logistic Regression, Random Forest, XGBoost.**  

### 📊 Resultados
- Regresión Logística → **0.912 ± 0.050 (CV)**  
- Random Forest → **0.925 ± 0.032 (CV)**  
- XGBoost → **0.907 ± 0.032 (CV)**  
- Mejor en test: **XGBoost = 0.897 AUC**  

➡️ Primer pipeline sencillo, buen baseline y estable, pero con un dataset limitado (150 sujetos).

---

# 2️⃣ COGNITIVA-AI-CLINIC-IMPROVED (fusión OASIS-1 + OASIS-2)

### 🔧 Decisiones de diseño
El objetivo fue **aumentar la robustez** uniendo ambas cohortes.  
- **Unificación de columnas** (`snake_case`).  
- **Selección baseline** en OASIS-2.  
- **Target unificado:** `Group` (OASIS-2) o `CDR` (OASIS-1).  
- **Imputación:** SES/Educación → mediana.  
- **Codificación y escalado.**  
- **Etiqueta de cohortes** para trazabilidad.  

### 📊 Resultados
- **Hold-out inicial (80/20):** LogReg=1.000 | RF=0.986 | XGB=0.991  
- **Validación cruzada (5-fold):**  
  - LogReg → **0.979 ± 0.012**  
  - RF → **0.974 ± 0.018**  
  - XGB → **0.975 ± 0.021**  

➡️ La fusión de datasets clínicos genera modelos **muy estables y con excelente generalización**.

### ⚖️ Manejo del desbalance
- Distribución real ≈ 54% vs 46%.  
- Estrategias usadas: `class_weight=balanced` y ajuste de **umbral clínico**.  

**Umbral clínico seleccionado (XGBoost):**  
- Recall ≈ 100%, con 15 falsos positivos (77 test).  
- Interpretación → en Alzheimer **un falso positivo es aceptable**, un falso negativo es más crítico.  

### 🩺 Interpretabilidad
- **Coeficientes LR:**  
  - CDR (coef fuerte positivo).  
  - MMSE (negativo fuerte).  
  - Volumétricas con peso menor.  
- **Ablación:**  
  - Sin CDR → AUC = 0.86.  
  - Sin CDR+MMSE → AUC = 0.76.  
  - Sin volumétricas → AUC ≈ 1.0.  

➡️ **Conclusión clínica:** los test **CDR + MMSE son críticos**, las volumétricas aportan menos.  

### 🔧 Calibración y Robustez
- Mejor calibrado: **LogReg + Isotónica (Brier=0.010)**.  
- Nested CV (10x5) → ROC-AUC = **0.985 ± 0.011**.  
- Ensemble (LR+RF+XGB) → ROC-AUC ≈ **0.995**.  

---

# 3️⃣ COGNITIVA-AI-IMAGES (MRI OASIS-2)

### 🔧 Decisiones de diseño
- Preprocesamiento: conversión de `.hdr/.img` a cortes axiales, normalización [0–255], augmentations ligeros.  
- Modelado: **ResNet50** fine-tuning, evaluación por paciente.  

### 📊 Resultados
- **5 slices, sin CLAHE → AUC=0.938 (test)**  
- 20 slices + z-score → AUC=0.858 (más recall, menos precisión)  

➡️ Buen baseline en imágenes, pero dependiente del preprocesamiento y costoso en CPU.

---

# 4️⃣ COGNITIVA-AI-IMAGES-IMPROVED (fusión OASIS-1+2)

Pipeline actualmente **en progreso**, centrado en extender el enfoque MRI a ambos datasets OASIS-1 y OASIS-2 para obtener un modelo más robusto y listo para la integración multimodal.

---

# 5️⃣ COGNITIVA-AI-IMAGES-IMPROVED-GPU

### 🚧 Limitaciones técnicas
Entrenar y calibrar modelos de imágenes en un ordenador personal modesto resultó **impracticable** (tiempos largos y problemas de memoria).  
Se recurrió a **Google Colab** para emplear **GPUs (T4, A100, L4)**, lo que permitió generar embeddings ResNet18 y entrenar clasificadores de forma eficiente.

### Resultados
- **Slice-nivel (LR calibrado)** → Test AUC≈0.66, Brier≈0.23.  
- **Paciente-nivel (pool mean, thr clínico≈0.20)** → Recall≈0.80 en test.  

---

# 6️⃣ COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED

### 🔧 Decisiones de diseño
- Uso de embeddings ResNet18 (512 dim/slice).  
- Clasificadores adicionales: SVM, XGBoost, MLP.  
- Calibración isotónica con validación cruzada (cv=5).  
- Evaluación a nivel paciente mediante pooling y directamente sobre patient-features.  

### 📊 Resultados comparativos (paciente-nivel, recall clínico ≥0.90 en validación)

| Modelo                         | TEST AUC | TEST PR-AUC | TEST Recall | TEST Precision |
|--------------------------------|----------|-------------|-------------|----------------|
| LR baseline (slice→patient)    | 0.724    | 0.606       | 0.80        | 0.52           |
| SVM (slice→patient)            | 0.746    | 0.628       | 0.80        | 0.50           |
| XGB (slice→patient)            | 0.733    | 0.605       | **0.85**    | 0.59           |
| MLP (patient-features, PCA)    | 0.732    | **0.703**   | **0.85**    | 0.53           |
| Ensemble híbrido (XGB+MLP)     | **0.744** | 0.659      | 0.80        | 0.53           |

### 🩺 Conclusión MRI
- El **MLP patient-features** logra la mejor PR-AUC (0.703).  
- El **XGB slice→patient** conserva la mayor sensibilidad (0.85).  
- El **ensemble híbrido** logra el mejor AUC (0.744) y estabilidad global.  

📌 **Modelo MRI recomendado:** Ensemble híbrido como punto de partida para la futura fusión multimodal con datos clínicos.  

---

# 📊 Comparativa Global

| Modalidad       | Dataset            | Modelo                  | ROC-AUC | Notas |
|-----------------|--------------------|-------------------------|---------|-------|
| Clínico         | OASIS-2            | XGBoost                 | 0.897   | Mejor tabular OASIS-2 |
| Clínico Fusion  | OASIS-1+2          | LogReg                  | 0.979   | Simple, interpretable |
| Imágenes        | OASIS-2            | ResNet50 (5 slices)     | 0.938   | Mejor en MRI OASIS-2 |
| Clínico Fusion  | OASIS-1+2 Ensemble | LR+RF+XGB               | 0.995   | **Mejor global** |
| Imágenes Mejor. | OASIS-1+2 (GPU)    | Ensemble (XGB+MLP)      | 0.744   | Mejor equilibrio MRI |

---

---

### MRI — Comparativa de modelos (paciente-nivel, TEST)

Los siguientes gráficos resumen el rendimiento de los distintos clasificadores MRI a nivel paciente (con recall clínico ≥0.90 en validación):

<p align="center">
  <img src="./graficos/mri_model_comparison_auc.png" alt="MRI AUC por modelo" width="560"/>
</p>

<p align="center">
  <img src="./graficos/mri_model_comparison_prauc.png" alt="MRI PR-AUC por modelo" width="560"/>
</p>

<p align="center">
  <img src="./graficos/mri_model_comparison_recall.png" alt="MRI Recall por modelo" width="560"/>
</p>

<p align="center">
  <img src="./graficos/mri_model_comparison_precision.png" alt="MRI Precisión por modelo" width="560"/>
</p>

➡️ **Conclusión gráfica**:  
- El **MLP (patient-features)** alcanza la mejor **PR-AUC** (0.703).  
- El **XGB slice→patient** logra el **mejor recall (0.85)** con precisión aceptable.  
- El **ensemble híbrido (XGB+MLP)** ofrece el mejor **equilibrio global** (AUC=0.744).  

---

### Comparativa Global (ROC-AUC por pipeline)

Se incluye un resumen de todos los pipelines (clínicos e imágenes) para visualizar de un vistazo el rendimiento relativo:

<p align="center">
  <img src="./graficos/global_auc_comparison.png" alt="Comparativa global ROC-AUC por pipeline" width="580"/>
</p>

---

### Timeline de los 6 Pipelines

Finalmente, un diagrama de evolución del proyecto mostrando el orden en que se desarrollaron los pipelines:

<p align="center">
  <img src="./graficos/pipelines_timeline.png" alt="Timeline de los 6 Pipelines" width="720"/>
</p>

---

# 🚀 Próximos Pasos

1. Finalizar **COGNITIVA-AI-IMAGES-IMPROVED** (fusión OASIS-1+2).  
2. Explorar **fusión multimodal** (clínico + imágenes).  
3. Validación externa con **OASIS-3 / ADNI**.  
4. Publicación académica con énfasis en interpretabilidad clínica.  

---

**Autoría:** Fran Ramírez  
**Año:** 2025
