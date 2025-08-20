# COGNITIVA-AI — Detección Temprana de Alzheimer  
**Informe Técnico (Formal)**

---

## 1. Resumen
Este proyecto investiga la **detección temprana de Alzheimer** combinando **datos clínicos tabulares** y **resonancias magnéticas estructurales (MRI)** de los conjuntos de datos **OASIS-1 y OASIS-2**.  

Se plantean cuatro pipelines:  
1. **COGNITIVA-AI-CLINIC** → datos clínicos tabulares (baseline).  
2. **COGNITIVA-AI-CLINIC-IMPROVED** → fusión OASIS-1+2 con calibración, interpretabilidad, robustez y ensembling.  
3. **COGNITIVA-AI-IMAGES** → Deep Learning con MRI (ResNet50).  
4. **COGNITIVA-AI-IMAGES-IMPROVED** → pendiente (fusión multimodal).  

Los resultados muestran que el pipeline clínico mejorado alcanza **ROC-AUC ≈ 0.985 (Nested CV)**, mientras que el mejor pipeline de imágenes (5 cortes axiales, sin CLAHE) alcanza **ROC-AUC 0.938** a nivel de paciente.  

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

### 5.3 Resultados iniciales

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

#### ⚖️ Manejo del desbalanceo
- Se probaron variantes con `class_weight='balanced'` y `scale_pos_weight` en XGBoost.  
- Se optimizó el **umbral de decisión** para priorizar *recall* clínico.  
  - Umbral óptimo ≈ 0.03 → Recall ≈ 100%, con sacrificio en precisión (15 falsos positivos).  

#### 🔍 Interpretabilidad
- **Coeficientes (LR):**
  - `CDR` (coef ≈ +4.15) → marcador principal.  
  - `MMSE` (coef ≈ -0.64) → inversamente asociado.  
  - `Educación` (coef ≈ +0.76) → correlación positiva.  
- Conclusión: el modelo se alinea con la evidencia clínica → CDR y MMSE son dominantes.  

#### 📏 Calibración
- Comparación: sin calibrar, **Platt (sigmoid)**, **isotónica**.  
- **Brier Scores:**  
  - LR isotónica → **0.0099** (mejor calibración).  
  - RF isotónica → 0.0170.  
  - XGB isotónica → 0.0187.  

#### 🛡️ Robustez
- **Nested CV:** ROC-AUC = **0.985 ± 0.011**.  
- **Ablation:**  
  - Sin MMSE → ROC-AUC 1.000.  
  - Sin CDR → 0.86.  
  - Sin MMSE+CDR → 0.76.  
  - Sin volumétricas → ≈ 1.000.  
  - Sin socioeducativas → ≈ 0.998.  
- Conclusión: **CDR y MMSE son críticos**, otras variables aportan poco.  

#### 🤝 Ensembling
- Promedio de probabilidades (LR + RF + XGB).  
- Resultado: ROC-AUC = **0.995** → ligera mejora.  

---

## 6. Pipeline de Imágenes (MRI)

### 6.1 Preprocesamiento
- Conversión de volúmenes a cortes axiales (5 o 20 slices).  
- Normalización a rango [0–255].  
- Opciones: CLAHE y z-score por slice.  
- Augmentation: flips, rotaciones ±10°, ajustes de brillo/contraste.  
- Redimensionado a 224×224 y normalización ImageNet.  

### 6.2 Entrenamiento
- Base: **ResNet50** pre-entrenada en ImageNet.  
- Capa final adaptada a binario.  
- Optimizador: Adam (lr=1e-4).  
- Early stopping con paciencia = 4.  
- Split por paciente (60/20/20).  
- Evaluación a nivel de paciente (probabilidad media).  

### 6.3 Resultados (OASIS-2)
- **5 slices, sin CLAHE:** Acc = **0.89**, AUC = **0.938**.  
- 5 slices, con CLAHE: Acc = 0.69, AUC = 0.777.  
- 5 slices, CLAHE + z-score: Acc = 0.72, AUC = 0.820.  
- **20 slices, CLAHE+z-score:** Acc = 0.80, AUC = 0.858.  

> 📌 Conclusión: Mejor AUC con 5 cortes sin CLAHE; más cortes mejoran recall, pero no AUC.  

---

## 7. Discusión
- **Clínico vs Imágenes:**  
  - Clínico fusionado → ROC-AUC ≈ 0.985.  
  - Imágenes → ROC-AUC ≈ 0.94.  
  - Complementarios: la combinación multimodal es prometedora.  

- **Umbral clínico:**  
  - Se priorizó Recall para reducir falsos negativos, aceptando más falsos positivos como coste asumible.  

- **Generalización:**  
  - Los resultados reflejan OASIS; falta validación externa.  

---

## 8. Limitaciones
- Tamaño reducido de muestra.  
- Uso de cortes 2D en lugar de volúmenes 3D completos.  
- Alta dependencia del preprocesamiento.  
- Target simplificado a binario.  

---

## 9. Reproducibilidad
- Semillas fijadas y `n_jobs=1`.  
- Escalado y transformaciones dentro de cada fold.  
- Código modular en notebooks (CLINIC, IMAGES).  
- Documentación exhaustiva de cada decisión.  

---

## 10. Futuras Líneas
1. Interpretabilidad avanzada (SHAP, SHAPley).  
2. Fusión multimodal (clínico + MRI).  
3. Modelos 3D CNN / Transformers.  
4. Validación externa (OASIS-3, ADNI).  
5. Estrategias de regularización para robustez.  

---

## 11. Agradecimientos
Este trabajo se basa en los conjuntos de datos OASIS.  
Uso estrictamente académico, sin fines clínicos.  
Gracias a la comunidad open-source y a los docentes/mentores que han acompañado el proceso.

---

## 12. Conclusiones Clínicas y Utilidad Práctica
- **Detección temprana:** El pipeline clínico es altamente preciso, incluso con modelos simples.  
- **Interpretabilidad:** Confirmó el valor de escalas clínicas clásicas (CDR y MMSE).  
- **Probabilidades calibradas:** Mejoran la confianza en decisiones clínicas.  
- **Umbral adaptado:** Minimiza falsos negativos, adecuado para screening.  
- **Falsos positivos:** Asumibles en un contexto de cribado, ya que derivan en más pruebas, no en daño directo.  

---
