# README.md

# 🧠 Proyecto de Detección Temprana de Alzheimer (COGNITIVA-AI)

Este proyecto explora la detección temprana de Alzheimer combinando **datos clínicos tabulares** y **resonancias magnéticas (MRI)** de los conjuntos **OASIS-1 y OASIS-2**.  
Construimos dos pipelines complementarios:

1. **COGNITIVA-AI-CLINIC** → *aprendizaje clásico (ML) con datos clínicos fusionados*.  
2. **COGNITIVA-AI-IMAGES** → *Deep Learning con imágenes (fine-tuning de ResNet50).*  

El objetivo es ofrecer una línea base sólida, documentar decisiones y resultados, y dejar una hoja de ruta clara para mejoras.

---

## 📦 Datos y alcance

- **Fuente clínica:**
  - **OASIS-1**: 416 individuos, 434 sesiones (una por sujeto, más algunos seguimientos).
  - **OASIS-2**: 150 individuos, 373 MR (longitudinal, con varias visitas por paciente).
- **Fuente imágenes:** archivos MRI en carpetas `OAS1_XXX` y `OAS2_XXX`, con variantes `RAW`/`OLD`.  
- **Problema:** clasificación **binaria** a nivel de paciente:
  - `0 = Nondemented`
  - `1 = Demented` o `Converted`  

> 🔒 **Evitar fugas de información (data leakage):**
> - En **clínico**: seleccionamos **una visita por sujeto** (la primera, `baseline`) para que cada paciente solo aparezca una vez.  
> - En **MRI**: las particiones son por **paciente/scan_id**, nunca se mezclan slices de un mismo sujeto entre train/test.

---

# 1️⃣ COGNITIVA-AI-CLINIC (Datos clínicos)

### 📂 Variables (tras fusión OASIS-1 + OASIS-2)

- Demográficas: `age`, `sex`, `education`, `ses`, `hand`.  
- Clínicas/neuropsicológicas: `mmse`, `cdr`.  
- Estructurales globales: `etiv`, `nwbv`, `asf`.  
- **Target (`target`)**:  
  - OASIS-2 → `group` (map: `Nondemented → 0`, `Demented/Converted → 1`).  
  - OASIS-1 → a partir de **CDR** (`0 → Nondemented`, `>0 → Demented`).  
- Variable auxiliar: `cohort ∈ {OASIS1, OASIS2}` para trazabilidad.

---

### 🧹 Preprocesamiento clínico

1. **Homogeneización de columnas** (renombrado a `snake_case` en ambos datasets).  
2. **Selección de visitas**:  
   - OASIS-2 → **primera visita por paciente** (baseline).  
   - OASIS-1 → ya es 1 entrada por sujeto.  
3. **Target unificado** (`group` y/o `cdr` → `target` binario).  
4. **NaN críticos**: eliminamos filas sin `mmse`, `cdr` o `target`.  
5. **Imputación**: `ses` y `education` con **mediana**.  
6. **Codificación**: one-hot para `sex` (y `hand` si se usa).  
7. **Escalado**: `StandardScaler` **ajustado solo en train**.

---

### ⚙️ Modelado clínico (fusión OASIS-1 + OASIS-2)

- **Modelos evaluados:** Logistic Regression, Random Forest, XGBoost.  
- **Validación:** `StratifiedKFold` (5 folds), métrica **ROC-AUC**.  
- **Pipelines** con escalado dentro del fold para evitar leakage.  
- **Reproducibilidad**: semillas fijadas y paralelismo limitado.

---

### 📊 Resultados clínicos tras fusión

#### ➤ Hold-out inicial (80/20)
- **Logistic Regression**: ROC-AUC ≈ **1.000**  
- **Random Forest**: ROC-AUC ≈ **0.986**  
- **XGBoost**: ROC-AUC ≈ **0.991**

> ⚠️ Muy altos → confirmamos con CV.

#### ➤ Validación cruzada (5-Fold, reproducible)
- Logistic Regression → **0.979 ± 0.012**  
- Random Forest → **0.974 ± 0.018**  
- XGBoost → **0.975 ± 0.021**  

> ✅ Rendimiento estable ~0.97–0.98 ROC-AUC.

---

# 2️⃣ COGNITIVA-AI-IMAGES (MRI con ResNet50)

### 🛠️ Preprocesamiento de imágenes
- Conversión de `.hdr/.img` a **slices PNG** (cortes axiales centrales).  
- **Normalización** 0–255, opción de **CLAHE**, y **z-score por slice**.  
- **Data augmentation** (train): flips, rotaciones ±10°, jitter ligero.  
- **Evaluación por paciente**: promediado de probabilidades por `scan_id`.

---

## 🧪 Resultados completos OASIS-2 (solo OASIS-2)

> Esta sección resume **todos los modelos y variantes evaluados únicamente en OASIS-2**.

### 🔹 Clínico – OASIS-2 (tabular)

| Modelo / Variante                      | Validación (CV 5-fold)         | Test hold-out | Notas |
|---------------------------------------|---------------------------------|---------------|-------|
| Logistic Regression (baseline)        | **0.912 ± 0.050**               | 0.911 (AUC)   | Split inicial; buen baseline y muy estable |
| Random Forest (balanced)              | **0.925 ± 0.032**               | —             | CV alto con `class_weight` |
| XGBoost (default)                     | **0.907 ± 0.032**               | —             | Buen baseline |
| **RF (GridSearchCV, mejor)**          | **0.922**                       | —             | Ajuste clásico |
| **RF (Alg. Genético, mejor)**         | **0.922**                       | —             | DEAP; rendimiento parejo al grid |
| **XGBoost (Alg. Genético, mejor)**    | **0.922**                       | —             | GA efectivo |
| **RF (optimizado, test)**             | —                               | **0.884**     | Test de referencia |
| **XGBoost (optimizado, test)**        | —                               | **0.897**     | **Mejor en test** |

> *Métricas clínicas: ROC-AUC. El test se hizo con un split estratificado y honesto por paciente.*

---

### 🔹 Imágenes – OASIS-2 (ResNet50, nivel **paciente**)

| Slices | Preprocesamiento                 | Train Acc | Val Acc | **Test Acc** | **ROC-AUC** | Comentarios |
|-------:|----------------------------------|----------:|--------:|-------------:|------------:|-------------|
| **5**  | **Sin CLAHE**                    | ~0.94     | ~0.73   | **0.89**     | **0.938**   | **Mejor AUC**; baseline fuerte |
| 5      | CLAHE                            | ~0.95     | ~0.72   | 0.69         | 0.777       | Realce local perjudicó patrones sutiles |
| 5      | CLAHE + z-score (slice)          | ~0.96     | ~0.75   | 0.72         | 0.820       | Recupera parte del rendimiento |
| **20** | CLAHE + z-score (slice)          | **0.98**  | ~0.71   | **0.80**     | **0.858**   | Más cobertura anatómica; mejor recall |

> **Conclusión OASIS-2 (imágenes):** el **baseline 5 slices sin CLAHE** obtuvo el **mejor AUC (0.938)**; usar más slices (20) mejora robustez y recall pero no supera ese AUC en nuestro test.

---

## 🧠 Decisiones de diseño (y por qué)

- **Fusión OASIS-1 + OASIS-2** en clínico: más pacientes ⇒ mejor generalización y menor varianza.  
- **Target**: OASIS-2 por `group`; OASIS-1 por `cdr` (clínicamente válido: `cdr=0` sano, `>0` demencia).  
- **Evitar leakage**: baseline por sujeto en clínico; split por paciente en imágenes; escalado dentro de cada fold.  
- **Métrica ROC-AUC**: robusta a desbalance y permite comparar umbrales.  
- **Early stopping** en imágenes para contener sobreajuste.

---

## 📊 Comparativa resumida

| Modalidad       | Dataset            | Modelo        | ROC-AUC (CV/Test) | Notas |
|-----------------|--------------------|---------------|-------------------|-------|
| **Clínico**     | OASIS-2            | XGBoost       | **0.897 (test)**  | Mejor en test (tabular OASIS-2) |
| **Clínico**     | OASIS-1+OASIS-2    | LogReg        | **0.979 ± 0.012** | Muy fuerte, simple, estable |
| **Imágenes**    | OASIS-2            | ResNet50 (5s) | **0.938 (test)**  | Mejor AUC en imágenes |

---

## 🚀 Próximos pasos

1. **Importancia y explicabilidad** (SHAP / permutation importance) en clínico.  
2. **Regularización y calibración** (p.ej., Platt/Isotonic) para probabilidades mejor calibradas.  
3. **Multimodalidad**: fusión de embeddings clínicos + MRI.  
4. **Ablación y validación externa** (OASIS-3) para robustez.  

---

## 🧾 Reproducibilidad

1. **Clínico (fusión):** cargar OASIS-1+2 → homogenizar → baseline por sujeto → imputación → encoding → CV con pipelines.  
2. **Imágenes (OASIS-2):** extraer slices → normalizar → split por paciente → ResNet50 FT → evaluación por paciente.  

---

**Autoría:** Fran Ramírez  
**Año:** 2025
