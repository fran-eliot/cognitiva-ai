# 🧠 Proyecto de Detección Temprana de Alzheimer (COGNITIVA-AI)

Este proyecto explora la **detección temprana de Alzheimer** combinando **datos clínicos tabulares** y **resonancias magnéticas (MRI)** de los conjuntos **OASIS-1 y OASIS-2**.  

Construimos cuatro pipelines principales:

1. **COGNITIVA-AI-CLINIC** → *ML clásico con datos clínicos (solo OASIS-2)*.  
2. **COGNITIVA-AI-CLINIC-IMPROVED** → *ML clásico con datos clínicos fusionados OASIS-1 + OASIS-2*.  
3. **COGNITIVA-AI-IMAGES** → *Deep Learning con imágenes (solo OASIS-2, ResNet50)*.  
4. **COGNITIVA-AI-IMAGES-IMPROVED** → *(pendiente)* fusión de OASIS-1+2 en imágenes.  

El objetivo es **detectar de forma temprana y fiable la enfermedad**, documentando cada paso con rigor científico y buenas prácticas en ML.

---

## 📦 Datos y alcance

- **Fuente clínica:**
  - **OASIS-1**: 416 individuos, 434 sesiones.
  - **OASIS-2**: 150 individuos, 373 MR (longitudinal).
- **Fuente imágenes:** MRI organizados en carpetas por sujeto (`OAS1_XXX`, `OAS2_XXX`).  
- **Problema:** clasificación **binaria** a nivel de paciente:
  - `0 = Nondemented`
  - `1 = Demented` o `Converted`  

> 🔒 **Evitar fugas de información (data leakage):**
> - En **clínico**: se selecciona **una visita por sujeto** (baseline).  
> - En **MRI**: particiones estrictas por paciente (`scan_id`).  

---

# 1️⃣ COGNITIVA-AI-CLINIC (solo OASIS-2)

- **Variables:** edad, sexo, educación, SES, MMSE, CDR, eTIV, nWBV, ASF.  
- **Target:** `Group` (`Nondemented=0`, `Demented/Converted=1`).  
- **Modelos evaluados:** Logistic Regression, Random Forest, XGBoost.  

### 📊 Resultados (OASIS-2 solo)
- Logistic Regression → **0.912 ± 0.050 (CV)**  
- Random Forest → **0.925 ± 0.032 (CV)**  
- XGBoost → **0.907 ± 0.032 (CV)**  
- **Test mejor:** XGBoost → **0.897 (AUC)**  

---

# 2️⃣ COGNITIVA-AI-CLINIC-IMPROVED (OASIS-1 + OASIS-2)

### 🧹 Preprocesamiento
1. **Unificación de variables** (`snake_case`).  
2. **Selección baseline:** primera visita en OASIS-2.  
3. **Target unificado:**  
   - OASIS-2: `Group`.  
   - OASIS-1: derivado de **CDR** (`0=0`, `>0=1`).  
4. **Imputación:** SES y Educación con mediana.  
5. **Escalado y codificación**.  
6. **Cohort tag:** para trazabilidad (`OASIS1` vs `OASIS2`).  

### ⚙️ Modelado
- Modelos: Logistic Regression, Random Forest, XGBoost.  
- Cross-validation estratificado (5 folds).  
- Métrica principal: **ROC-AUC**.  

### 📊 Resultados
- **Hold-out inicial (80/20)**  
  - LogReg: 1.000 AUC  
  - RF: 0.986 AUC  
  - XGB: 0.991 AUC  

- **Validación cruzada (5-Fold)**  
  - LogReg: **0.979 ± 0.012**  
  - RF: **0.974 ± 0.018**  
  - XGB: **0.975 ± 0.021**  

### ⚖️ Manejo del desbalance
- Distribución real: `~54% vs ~46%` → ligero desbalance.  
- Estrategias: `class_weight=balanced`, y ajuste de **umbral clínico** para priorizar **recall**.  

### 🩺 Umbral clínico (XGBoost)
- Ajustado para maximizar **recall (≈100%)**.  
- Resultado: recall perfecto, con más falsos positivos (~15/77 test).  
- Interpretación clínica: **preferimos un falso positivo antes que un falso negativo**, ya que permite tratar antes.  

### 📊 Interpretabilidad
- **Pesos LogReg:**  
  - CDR (coef ≈ 4.15) → predictor más fuerte.  
  - MMSE (negativo fuerte).  
  - Volumétricas (eTIV, nWBV, ASF) menos influyentes.  
- **Ablación:**  
  - Sin CDR: AUC cae a 0.86.  
  - Sin CDR+MMSE: AUC 0.76.  
  - Sin volumétricas: AUC se mantiene ≈1.0.  
  → Los test clínicos (MMSE + CDR) son **críticos**.  

### 🔧 Calibración
- Curvas de calibración y Brier Score:  
  - Mejor calibración: **Logistic Regression + Isotónica (Brier≈0.010)**.  
  - RF y XGB calibrados también mejoran respecto a sus variantes sin calibrar.  

### 🧪 Robustez
- **Nested CV (10x5 folds)** → ROC-AUC = **0.985 ± 0.011**.  
- **Ensemble (LR+RF+XGB)** → ROC-AUC ≈ **0.995**.  

---

# 3️⃣ COGNITIVA-AI-IMAGES (MRI OASIS-2)

- **Modelo:** ResNet50 fine-tuning.  
- **Preprocesado:** slices axiales, normalización, z-score, augmentations.  
- **Evaluación paciente-nivel.**  

### 📊 Resultados
- 5 slices, sin CLAHE → **AUC=0.938 (test)**.  
- 20 slices + z-score → AUC=0.858, mejor recall pero menos precisión.  

---

# 4️⃣ COGNITIVA-AI-IMAGES-IMPROVED (pendiente)

- Plan: fusionar OASIS-1 + OASIS-2 en imágenes.  
- Objetivo: más pacientes, más robustez.  

---

# 📊 Comparativa resumida

| Modalidad       | Dataset            | Modelo        | ROC-AUC | Notas |
|-----------------|--------------------|---------------|---------|-------|
| Clínico         | OASIS-2            | XGBoost       | 0.897   | Mejor tabular OASIS-2 |
| Clínico         | OASIS-1+2          | LogReg        | 0.979   | Estable y simple |
| Imágenes        | OASIS-2            | ResNet50 (5s) | 0.938   | Mejor en imágenes |
| Clínico Fusion  | OASIS-1+2 (Ensemble)| LR+RF+XGB     | 0.995   | Mejor global |

---

# 🚀 Próximos pasos

1. Terminar **COGNITIVA-AI-IMAGES-IMPROVED** (fusionar OASIS-1+2 en imágenes).  
2. **Fusión multimodal** (clínico + MRI).  
3. Validación externa con **OASIS-3**.  
4. Publicación con énfasis en interpretabilidad clínica.  

---

**Autoría:** Fran Ramírez  
**Año:** 2025
