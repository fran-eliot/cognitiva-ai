# REPORT.md

# COGNITIVA-AI — Detección Temprana de Alzheimer  
**Informe Técnico (Formal)**

---

## 1. Resumen
Este proyecto investiga la **detección temprana de Alzheimer** combinando **datos clínicos tabulares** y **resonancias magnéticas estructurales (MRI)** de los conjuntos de datos **OASIS-1 y OASIS-2**.  

Se plantean dos pipelines complementarios:  
1. **Clínico (tabular):** modelos de *Machine Learning clásico* sobre variables demográficas, cognitivas y volumétricas.  
2. **Imágenes (MRI):** *Deep Learning* con **ResNet50** pre-entrenada, adaptada mediante *fine-tuning* en cortes axiales.  

Los resultados muestran que la fusión clínica OASIS-1+2 alcanza **ROC-AUC ≈ 0.98 (CV)**, mientras que el mejor pipeline de imágenes (5 cortes axiales, sin CLAHE) alcanza **ROC-AUC 0.938** a nivel de paciente.  

---

## 2. Antecedentes y Motivación
La **Enfermedad de Alzheimer (EA)** es neurodegenerativa y progresiva. Una detección temprana es clave para:  
- Optimizar la atención clínica.  
- Planificar intervenciones.  
- Reducir costes en fases avanzadas.  

Los conjuntos de datos **OASIS** proporcionan datos **abiertos y estandarizados** tanto clínicos como de neuroimagen, idóneos para estudios académicos. El objetivo de este trabajo es evaluar la capacidad predictiva de **modelos clásicos y de deep learning** y sentar bases reproducibles para futuras mejoras multimodales.

---

## 3. Datos
- **OASIS-1 (Transversal):**  
  - 416 sujetos, 434 sesiones.  
  - Sin variable `Group`; severidad inferida con **CDR** (`0 = no demencia`, `>0 = demencia`).  

- **OASIS-2 (Longitudinal):**  
  - 150 sujetos, 373 sesiones.  
  - Variable `Group`: {Nondemented, Demented, Converted}.  
  - Varias visitas por sujeto.  

- **MRI:** archivos `.hdr/.img` organizados en carpetas. Se extrajeron cortes axiales representativos.  

**Definición del target (binario):**  
- `0` = Nondemented  
- `1` = Demented o Converted  

---

## 4. Definición del Problema
- **Tarea:** Clasificación binaria a nivel de **paciente**.  
- **Retos principales:**  
  - Evitar *data leakage* (fugas de información).  
  - Manejo de múltiples visitas en OASIS-2.  
  - Tamaño limitado de muestra.  
  - Preprocesamiento coherente entre cohortes.  

---

## 5. Pipeline Clínico

### 5.1 Preprocesamiento
- Homogeneización de columnas (`snake_case`).  
- OASIS-2 → selección de **primera visita por sujeto (baseline)**.  
- OASIS-1 → se mantiene la única sesión por sujeto.  
- Target unificado: `Group` (OASIS-2) y `CDR` (OASIS-1).  
- Eliminación de filas sin `MMSE`, `CDR` o `Target`.  
- Imputación de `SES` y `Educación` con mediana.  
- Codificación *one-hot* de variables categóricas (`Sex`).  
- Escalado: **StandardScaler** ajustado solo en entrenamiento.  

### 5.2 Modelos y Validación
- Modelos evaluados:  
  - **Regresión Logística**  
  - **Random Forest**  
  - **XGBoost**  
- Validación: **StratifiedKFold (5-fold)**.  
- Métrica principal: **ROC-AUC**.  
- Escalado incluido dentro de cada fold para evitar leakage.  

### 5.3 Resultados

**OASIS-2 (solo clínico):**  
- Regresión Logística → **0.912 ± 0.050 (CV)**, Test ≈ **0.911**  
- Random Forest → **0.925 ± 0.032 (CV)**  
- XGBoost → **0.907 ± 0.032 (CV)**  
- Optimización por Grid y Algoritmos Genéticos → ROC-AUC CV ~**0.922**  
- Test hold-out: **RF = 0.884**, **XGB = 0.897**

**Fusión OASIS-1 + OASIS-2 (clínico):**  
- Hold-out inicial:  
  - LR ≈ **1.000**, RF ≈ **0.986**, XGB ≈ **0.991**  
- Validación cruzada (5-fold):  
  - LR → **0.979 ± 0.012**  
  - RF → **0.974 ± 0.018**  
  - XGB → **0.975 ± 0.021**

> 📌 Conclusión: Los datos clínicos por sí solos son altamente discriminativos.  
> Modelos simples como la regresión logística alcanzan métricas casi perfectas, además de ser más interpretables.

---

### 5.4 Mejoras avanzadas (COGNITIVA-AI-CLINIC-IMPROVED)

Tras el pipeline básico, se implementaron pasos adicionales para **evaluar robustez, interpretabilidad y utilidad clínica**:

#### ⚖️ Manejo del desbalanceo
- Aunque las clases estaban moderadamente equilibradas (≈54% vs 46%), se probaron variantes con `class_weight='balanced'` y `scale_pos_weight` en XGBoost.
- Se optimizó el **umbral de decisión** en función de criterios clínicos:
  - Recall prioritario (detectar todos los casos de demencia).
  - Umbral óptimo seleccionado: ≈0.03 → Recall ≈100%, con sacrificio de precisión (más falsos positivos).

#### 🔍 Interpretabilidad
- **Coeficientes (Logistic Regression):**
  - `CDR` (peso más alto, coef ≈ +4.15)  
  - `MMSE` (coef negativo fuerte ≈ -0.64)  
  - `Educación` (coef positivo ≈ +0.76)  
  - Variables volumétricas (`eTIV`, `nWBV`) con menor peso.
- Conclusión: el modelo se alinea con la literatura clínica → CDR y MMSE son marcadores dominantes.

#### 📏 Calibración
- Se compararon modelos sin calibrar, con **Platt (sigmoid)** y con **isotónica**.
- **Brier Scores (↓ mejor):**
  - LR isotónica → 0.0099 (mejor calibración).  
  - RF isotónica → 0.0170.  
  - XGB isotónica → 0.0187.  
- Conclusión: Logistic Regression, además de interpretable, ofrece las probabilidades más confiables.

#### 🛡️ Robustez
- **Nested Cross-Validation:** ROC-AUC = **0.985 ± 0.011** → rendimiento estable y poco dependiente del split.
- **Ablation Study:**  
  - Sin MMSE → ROC-AUC 1.000 (robusto).  
  - Sin CDR → ROC-AUC cae a 0.86.  
  - Sin MMSE + CDR → ROC-AUC 0.76.  
  - Sin variables volumétricas → ROC-AUC ≈ 1.000.  
  - Sin socioeducativas → ROC-AUC ≈ 0.998.  
  → Conclusión: **CDR y MMSE son críticos**, las volumétricas aportan poco en este dataset reducido.

#### 🤝 Ensembling
- Promedio de probabilidades (LR + RF + XGB).  
- Resultado: ROC-AUC = **0.995** → ligera mejora sobre cada modelo individual.

---

## 📊 Conclusiones integradas
- **El pipeline clínico fusionado y mejorado (COGNITIVA-AI-CLINIC-IMPROVED)** alcanza resultados casi perfectos en validación (~0.98–0.99 ROC-AUC).
- **Interpretabilidad:** confirma que **CDR y MMSE** son marcadores dominantes y clínicamente relevantes.
- **Calibración:** LR calibrada por isotónica produce las probabilidades más fiables para aplicaciones clínicas.
- **Umbral clínico:** favorece Recall (detectar todos los casos), aceptando falsos positivos como un coste asumible en screening.
- **Robustez:** nested CV y ablation muestran consistencia del pipeline.
- **Ensemble:** confirma la fortaleza del modelo combinado, pero la mejora sobre LR sola es marginal.

---

## 6. Pipeline de Imágenes (MRI)

### 6.1 Preprocesamiento
- Conversión de volúmenes a cortes axiales (5 o 20 slices).  
- Normalización a rango [0–255].  
- Opciones:  
  - **CLAHE** (ecualización adaptativa de histograma).  
  - **z-score por slice**.  
- Aumento de datos: flips, rotaciones ±10°, ligeros ajustes de brillo/contraste.  
- Redimensionado a 224×224 y normalización tipo ImageNet.  

### 6.2 Entrenamiento
- Modelo base: **ResNet50** pre-entrenada en ImageNet.  
- Capa final reemplazada por clasificación binaria.  
- Optimizador: Adam (lr=1e-4).  
- Early stopping con paciencia = 4.  
- División por paciente (60% train / 20% val / 20% test).  
- Evaluación final a nivel de **paciente** (probabilidades promedio).  

### 6.3 Resultados (OASIS-2)
- **5 slices, sin CLAHE:** Acc = **0.89**, ROC-AUC = **0.938** (mejor resultado en imágenes).  
- 5 slices, con CLAHE: Acc = 0.69, AUC = 0.777.  
- 5 slices, CLAHE + z-score: Acc = 0.72, AUC = 0.820.  
- **20 slices, CLAHE + z-score:** Acc = 0.80, AUC = 0.858 (mejor recall, menor AUC).  

> 📌 Conclusión: El mejor AUC (0.938) se logra con 5 cortes sin CLAHE.  
> Usar más cortes mejora la robustez y el recall, pero no supera el rendimiento en AUC.  

---

## 7. Discusión
- **Clínico vs Imágenes:**  
  - Clínico (fusionado) → ROC-AUC ≈ 0.98  
  - Imágenes (ResNet50) → ROC-AUC ≈ 0.94  
  - Ambos pipelines son competitivos y potencialmente complementarios.  

- **Sobreajuste:**  
  - Mitigado con validación cruzada, early stopping y partición por paciente.  

- **Generalización:**  
  - Los resultados reflejan solo OASIS; se requiere validación externa (p. ej., OASIS-3).  

---

## 8. Limitaciones
- Tamaño de muestra reducido.  
- Uso de cortes 2D en lugar de volúmenes 3D completos.  
- Alta dependencia de los parámetros de preprocesamiento.  
- Simplificación del target a binario (se pierde gradiente de progresión).  

---

## 9. Reproducibilidad
- Semillas fijadas y `n_jobs=1` para consistencia.  
- Escalado y transformaciones aplicadas dentro de cada fold.  
- Código modular dividido en notebooks (CLINIC, IMAGES).  
- Documentación exhaustiva de cada decisión.  

---

## 10. Futuras Líneas
1. **Interpretabilidad:** SHAP, coeficientes en LR, importancia de variables.  
2. **Calibración de probabilidades:** métodos de Platt e isotónica.  
3. **Fusión multimodal:** integración de embeddings clínicos + MRI.  
4. **Modelos 3D CNN / Transformers** si el hardware lo permite.  
5. **Validación externa:** OASIS-3, ADNI.  
6. **Preprocesamiento adaptativo:** normalización específica por paciente.  

---

## 11. Agradecimientos
Este trabajo se basa en los conjuntos de datos OASIS.  
Uso estrictamente académico, sin fines clínicos.  
Gracias a la comunidad open-source y a los docentes/mentores que han acompañado el proceso.

---
