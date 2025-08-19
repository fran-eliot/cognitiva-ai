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
  - OASIS-2 → columna `group` (map: `Nondemented → 0`, `Demented/Converted → 1`).  
  - OASIS-1 → se deriva de **CDR** (`0 → Nondemented`, `>0 → Demented`).  
- Variable auxiliar: `cohort ∈ {OASIS1, OASIS2}` para trazabilidad.

---

### 🧹 Preprocesamiento clínico

1. **Homogeneización de columnas**  
   - Renombrado a `snake_case` (`Subject ID → id`, `EDUC → education`, etc.).  
   - Se añadieron columnas faltantes en un dataset tomando valores nulos/constantes para facilitar concatenación.

2. **Selección de visitas**  
   - OASIS-2: se ordena por `visit` y `mr_delay`; se toma la **primera visita** por paciente.  
   - OASIS-1: ya tiene una única entrada por sujeto.

3. **Target unificado**  
   - `group` (OASIS-2) y `cdr` (OASIS-1) se convierten en una única variable binaria `target`.

4. **Tratamiento de NaN**  
   - Se eliminan registros sin información crítica (`mmse`, `cdr`, `target`).  
   - Imputación con **mediana** en `ses` y `education`.

5. **Codificación categórica**  
   - `sex`: one-hot encoding (`Sex_F = 1 si mujer`).  
   - `hand`: one-hot (`Right`, `Left`, `Ambi`, `Unknown`).  

6. **Escalado**  
   - Se usa `StandardScaler`, **ajustado únicamente sobre train** para evitar fuga de datos.

---

### ⚙️ Modelado clínico (fusión OASIS-1 + OASIS-2)

- **Modelos evaluados:** Logistic Regression, Random Forest, XGBoost.  
- **Validación:** `StratifiedKFold` con 5 splits, métrica **ROC-AUC**.  
- **Semillas** fijadas para reproducibilidad (`random_state=42`).  
- **Pipeline** usado para integrar escalado y clasificador en cada fold (evita leakage).  

---

### 📊 Resultados clínicos (tras fusión)

#### ➤ Hold-out inicial (80/20)
- **Logistic Regression**: ROC-AUC ≈ **1.00** (posible optimismo por simplicidad + buen lineal separability).  
- **Random Forest**: ROC-AUC ≈ **0.986**.  
- **XGBoost**: ROC-AUC ≈ **0.991**.  

> ⚠️ Resultados sorprendentemente altos → puede haber **estructura muy discriminativa en las features**.  
> Necesario validar con CV estricto.

---

#### ➤ Validación cruzada (5-Fold)
- Logistic Regression → **0.979 ± 0.012**  
- Random Forest → **0.974 ± 0.018**  
- XGBoost → **0.975 ± 0.021**  

> ✅ Los valores se estabilizan en torno a **ROC-AUC ≈ 0.97–0.98**, confirmando robustez y reduciendo riesgo de optimismo.  

---

# 2️⃣ COGNITIVA-AI-IMAGES (MRI con ResNet50)

*(idéntico a README previo, se mantiene todo lo de imágenes: slicing, CLAHE, z-score, resultados por slice y paciente, etc.)*  

---

## 🧠 Decisiones de diseño (y por qué)

- **Fusión OASIS-1 + OASIS-2:** más pacientes = mejor generalización → resultados más estables en CV.  
- **Target a partir de CDR (OASIS-1):** coherente con literatura (CDR=0 sano, >0 demencia).  
- **Selección baseline en OASIS-2:** evita duplicidad y fuga de información temporal.  
- **Validación cruzada estratificada:** mitiga optimismo de un único split y confirma estabilidad.  
- **Pipeline + escalado dentro del fold:** imprescindible para evitar leakage.

---

## 📊 Comparativa resumida

| Modalidad       | Dataset           | Modelo        | ROC-AUC (CV/Test) | Notas |
|-----------------|------------------|---------------|------------------|-------|
| **Clínico**     | OASIS-2          | XGBoost       | 0.897 (test)     | Mejor modelo tabular inicial |
| **Clínico**     | OASIS-1 + OASIS-2 | Logistic Regr. | 0.979 (CV)       | Muy fuerte, simple y explicable |
| **Clínico**     | OASIS-1 + OASIS-2 | XGBoost       | 0.975 (CV)       | Similar a LR, más flexible |
| **Imágenes**    | OASIS-2 (MRI)    | ResNet50 (5s) | 0.938 (test)     | Mejor config imagen baseline |

---

## 🚀 Próximos pasos

1. **Análisis de importancia de variables clínicas** (SHAP, permutation importance).  
2. **Regularización + calibración de probas** en Logistic Regression (evitar sobreconfianza).  
3. **Fusión multimodal**: clínico + imágenes (early vs late fusion).  
4. **Explorar OASIS-3** como dataset adicional para validación externa.  

---

## 🧾 Reproducibilidad

1. **Clínico (fusión)**  
   - Cargar OASIS-1 + OASIS-2 → homogenizar columnas → baseline por sujeto → imputación → encoding → split estratificado → CV 5-Fold.  
2. **Imágenes (OASIS-2)**  
   - Extraer slices → normalizar → split por paciente → ResNet50 fine-tuning → evaluación agregada a nivel paciente.  

---

**Autoría:** Fran Ramírez  
**Año:** 2025
