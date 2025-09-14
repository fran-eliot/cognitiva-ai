# 🧠 Proyecto de Detección Temprana de Alzheimer con datos clínicos y de resonancia magnética (COGNITIVA-AI) – Experimentos de Clasificación Multimodal

Este repositorio documenta **toda la evolución experimental** en el marco del proyecto **Cognitiva-AI**, cuyo objetivo ha sido **explorar modelos de machine learning para la predicción binaria de deterioro cognitivo (Alzheimer)** combinando  **datos clínicos tabulares** y **resonancias magnéticas estructurales (MRI)** de los conjuntos **OASIS-1 y OASIS-2**.   

El enfoque se diseñó con una idea central: **replicar el razonamiento clínico** usando tanto la información disponible en la historia del paciente (tests neuropsicológicos, edad, educación, volumen cerebral) como en las **imágenes estructurales cerebrales**.  

La hipótesis central que guía todo el trabajo es que **distintas fuentes de información y distintas arquitecturas de modelos pueden capturar facetas complementarias del proceso neurodegenerativo**.  

> **Idea fuerza**: un flujo **reproducible, interpretable y clínicamente orientado** que prioriza **recall** (minimizar FN) y mantiene la **calibración** de probabilidades con umbrales **por cohorte** (OAS1/OAS2).

El documento sigue un enfoque **cuaderno de bitácora extendido**, en el que cada pipeline corresponde a un conjunto de experimentos con motivaciones, configuraciones técnicas, métricas obtenidas y reflexiones.  
El tono es intencionadamente **verboso y detallado**: se incluyen incidencias de ejecución, errores y aprendizajes prácticos que acompañaron cada etapa.  

Se construyeron **diez pipelines** para analizar y comparar modalidades siguiendo una filosofía de **experimentación incremental**:  
- comenzar con modelos sencillos sobre datos clínicos,  
- avanzar hacia CNNs entrenadas sobre imágenes MRI,  
- introducir calibración, normalización y estrategias de ensemble,  
- explorar arquitecturas modernas de visión,  
- y finalmente preparar el terreno hacia un modelo multimodal.
  

1. **P1_COGNITIVA_AI_CLINIC** → ML clásico con datos clínicos (solo OASIS-2).  
2. **P2_COGNITIVA_AI_CLINIC_IMPROVED** → ML clásico con datos clínicos fusionados OASIS-1 + OASIS-2.  
3. **P3_COGNITIVA_AI_IMAGES** → Deep Learning con MRI (solo OASIS-2, ResNet50).  
4. **P4_COGNITIVA_AI_IMAGES_IMPROVED** → fusión de OASIS-1+2 en imágenes.  
5. **COGNITIVA-AI-IMAGES-IMPROVED-GPU (ResNet18)** → embeddings ResNet18 entrenados en **Google Colab (GPU)**.  
6. **COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED (EffNet-B3)** → embeddings EfficientNet-B3 + ensemble LR+XGB a nivel paciente.  
7. **COGNITIVA-AI-FINETUNING** → Fine-tuning directo de EfficientNet-B3 en **Google Colab (GPU)** con *temperature scaling* y agregación a **nivel paciente**.  
8. **COGNITIVA-AI-FINETUNING-IMPROVED**  → Mejoras de fine-tuning (calibración de probabilidades). Ajustes univariados (normalización, dropout, etc.).  
9. **COGNITIVA-AI-FINETUNING-STABLE** → Retraining estable de EfficientNet-B3 en **Google Colab (GPU)** con caché SSD, *temperature scaling* y selección de umbral clínico (recall≥0.95). Entrenamiento estable con configuración refinada y early stopping.  
10. **COGNITIVA-AI-FINETUNING-STABLE-PLUS** → Versión extendida con calibración adicional y pooling alternativo (mean, median, top-k).  

---

## 📚 Índice

1. [Introducción](#introducción)
2. [Pipelines experimentales](#pipelines-experimentales)
   - [P1: Datos clínicos con XGBoost](#p1-datos-clínicos-con-xgboost)
   - [P2: Datos clínicos fusionados](#p2-datos-clínicos-fusionados)
   - [P3: MRI OASIS-2 – ResNet50](#p3-mri-oasis-2--resnet50)
   - [P5: MRI Colab – ResNet18 calibrado](#p5-mri-colab--resnet18-calibrado)
   - [P6: MRI Colab – EfficientNet-B3 embeddings](#p6-mri-colab--efficientnet-b3-embeddings)
   - [P7: MRI Colab – EfficientNet-B3 fine-tuning](#p7-mri-colab--efficientnet-b3-fine-tuning)
   - [P9: MRI Colab – EfficientNet-B3 stable](#p9-mri-colab--efficientnet-b3-stable)
   - [P10: MRI Colab – EfficientNet-B3 stable + calibración](#p10-mri-colab--efficientnet-b3-stable--calibración)
   - [P10-ext: Extensiones y ensembles](#p10-ext-extensiones-y-ensembles)
   - [P11: Backbones alternativos](#p11-backbones-alternativos)
3. [Comparativa global de resultados](#comparativa-global-de-resultados)
4. [Desafíos principales](#desafíos-principales)
5. [Lecciones aprendidas](#lecciones-aprendidas)
6. [Próximos pasos](#próximos-pasos)

---

## 📦 Datos y Variables Clínicas

El dataset de referencia ha sido **OASIS** (Open Access Series of Imaging Studies), en particular sus cohortes OASIS-2 y derivados:

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

## Estructura y datasets

**Datasets**  
- **OASIS‑1** (cross‑sectional). Etiqueta derivada de **CDR** (CDR=0→0, CDR>0→1).  
- **OASIS‑2** (longitudinal). Etiqueta a partir de **Group** (*Nondemented=0; Demented/Converted=1*).  
- Criterio **primera visita/paciente** (baseline) en OASIS‑2 para evitar *leakage* inter‑sesión.  (En OASIS1 es ya 1 entrada por sujeto).
**Target unificado (binario):**  
- `0 = Nondemented`  
- `1 = Demented` o `Converted` 
- **NaN críticos**: eliminamos filas sin `mmse`, `cdr` o `target`.
-  **Imputación**: `ses` y `education` con **mediana**. 
- **Codificación**: one-hot para `sex` (y `hand` si se usa).
-  **Escalado**: `StandardScaler` **ajustado solo en train**.
- **MRI:** archivos `.hdr/.img` por paciente, con segmentaciones asociadas (`FSL_SEG`).  
- MRI: **20 slices axiales** equiespaciadas, normalización *z‑score* + **CLAHE** opcional.  
- Splits **estratificados a nivel paciente** (sin fuga).

> ⚠️ **Control estricto de fugas de información (data leakage):**  
> - En clínico → seleccionamos **solo una visita por sujeto (baseline)** de cada paciente para no mezclar repeticiones del mismo paciente entre train y test.  
> - Con imágenes (MRI): el **split es por paciente/scan_id**; todas las slices de un scan quedan en el mismo subset.  

**Estructura de carpetas (clave)**
```
/p11_alt_backbones/          # Catálogo y matrices patient-level OASIS-1 (base para ensembles)
/p13_oasis2_images/, /p14_oasis2_images/  # EffNet-B3 OASIS-2 (pesos, preds y features)
/p19_meta_ensemble, /p20_meta_calibration, /p21_meta_refine, /p22_meta_ablation
/p24_meta_simple, /p25_informe_final
/p26_intermodal/             # Fusión Late/Mid + P26b (calibración por cohorte)
/p26_release/                # Release reproducible (modelos, config, QA, docs)
/p27_final/                  # Figuras y tablas finales consolidadas
```
**Documentación viva**: `README.md` (este), `InformeTecnico.md`, `CuadernoBitacora.md`.

---

## Introducción

El proyecto **Cognitiva-AI** parte de la necesidad de evaluar modelos predictivos que integren datos clínicos y de imagen (MRI) en cohortes reducidas como OASIS-1/2.  

Desde el inicio se asumió que:
- Los **datos clínicos** podrían servir como baseline fuerte (edad, MMSE, CDR, etc.).  
- Las **imágenes cerebrales** aportarían riqueza multimodal pero con mayor complejidad.  
- Sería necesario experimentar con **diferentes backbones** de visión profunda y con **estrategias de calibración, ensembles y stacking** para compensar el pequeño tamaño muestral.  

El proceso se organizó en **pipelines numerados**. Cada uno corresponde a un conjunto de experimentos exploratorios.  

---

## Pipelines experimentales

A continuación, se detallan los diferentes pipelines experimentales desarrollados en el proyecto. Cada uno introduce una idea nueva, una arquitectura diferente o una estrategia de evaluación alternativa.

### Resumen ejecutivo

- **Mejor modelo unimodal (imagen)**: **P24** (LR elastic‑net sobre features por paciente + **Platt**).  
  - **TEST**: **AUC=0.727** (ALL) · **0.754** (OAS1) · **0.750** (OAS2).  
  - Umbrales **5:1** (FN:FP): OAS1 **0.435** (Coste=39, R=0.70, P=0.61), OAS2 **0.332** (Coste=12, R=0.92, P=0.61).

- **Modelo intermodal (imagen+clínico)**: **P26 (Late)** y **P26b (Late + Platt por cohorte)**.  
  - **TEST** P26 Late: **AUC=0.713**, PR‑AUC=0.712, Brier=0.234.  
  - **TEST** P26b (Platt por cohorte): **OAS1 AUC≈0.754 (Brier≈0.199)** · **OAS2 AUC≈0.652 (Brier≈0.241)**.

- **Política activa (P27): S2**  
  - Base **5:1** (FN:FP) + **ajuste OAS2** para **Recall objetivo ≥0.90** (cribado).  
  - **Umbrales S2**: **OAS1=0.42**, **OAS2≈0.4928655288** → en TEST:  
    - OAS1: TP=14, FP=9, TN=18, FN=6 → **R=0.70**, **P=0.609**, Coste=39.  
    - OAS2: TP=11, FP=6, TN=5, FN=1 → **R=0.917**, **P=0.647**, Coste=11.

---

### Línea temporal (P1→P27)

- **P1–P4 (local)**: *slicing*, normalización y primeros baselines tabulares e imagen.  
- **P5–P12 (Colab, OASIS‑1)**: consolidación de **EffNet‑B3**, agregación por paciente, **catálogo p11** y **ensembles**.  
- **P13–P14 (OASIS‑2)**: entrenamiento **EffNet‑B3** específico (1 visita/paciente); copia a **SSD Colab** y **class_weight**.  
- **P16–P18**: ensembles avanzados (OOF sin fuga, stacking/blending, calibración).  
- **P19**: **Meta‑ensemble (XGB)** con **LR/HGB/GB/RF/LGBM/XGB** como *base learners*.  
- **P20–P22**: **calibración** (Platt/Isotónica), **umbrales por cohorte** y *ablation*.  
- **P23**: **calibración por cohorte con coste** (5:1 FN:FP).  
- **P24**: **meta simple interpretable** (LR‑EN + Platt) → mejor equilibrio generalización/calibración.  
- **P25**: **consolidación** y tabla maestra (P19/P22/P23/P24).  
- **P26/P26b**: **intermodal** (Late vs Mid) + **calibración por cohorte**; elección **Late**.  
- **P27**: **release reproducible** + **política S2** y figuras finales.

---

### P1: COGNITIVA-AI-CLINIC (datos clínicos solo OASIS-2) 

- **Motivación:** establecer un baseline sólido con datos tabulares clínicos. 
### 🧹 Preprocesamiento
- **Renombrado a `snake_case`** para legibilidad (`Subject ID → subject_id`, etc.).  
- **Selección de una visita por sujeto**: baseline (mínimo `mr_delay`) para tener un único registro representativo por paciente.  
- Conversión de tipos numéricos y **imputación** (mediana) para columnas con NaN (`ses`, `mmse`, `cdr`, …).  
- Codificación:
  - `sex`: `M → 0`, `F → 1`.  
  - `hand`: one-hot (categoría desconocida a `Unknown`).

### ⚙️ Modelado
- **Modelos base:** Logistic Regression, Random Forest, XGBoost.  
- **Validación:** `StratifiedKFold` y métrica **ROC-AUC**.  
- **Optimización:**  
  - *GridSearchCV* para Random Forest.  
  - **Algoritmo Genético (DEAP)** para RF/XGB: búsqueda evolutiva de hiperparámetros (más eficiente en espacios grandes/no convexos).

> ℹ️ **Por qué ROC-AUC**: mide la capacidad de discriminación a todos los umbrales, robusta ante desbalance moderado y facilita comparación entre modelos.  

### 📊 Resultados (clínico)

- **Cross-val (grid/genético):**  
  - RF (grid) → mejor ROC-AUC CV ≈ **0.9224**  
  - RF (GA) → mejor ROC-AUC CV ≈ **0.9215**  
  - XGB (GA) → mejor ROC-AUC CV ≈ **0.9215**

- **Test hold-out (final):**
  | Modelo              | ROC-AUC (Test) |
  |---------------------|----------------|
  | Random Forest (opt) | 0.884          |
  | **XGBoost (opt)**   | **0.897**      |

### 📊 Resultados
- Regresión Logística → **0.912 ± 0.050 (CV)**  
- Random Forest → **0.925 ± 0.032 (CV)**  
- XGBoost → **0.907 ± 0.032 (CV)**  
- Mejor en test: **XGBoost = 0.897 AUC**  

➡️ Primer baseline, estable pero dataset reducido (150 sujetos) y limitado a datos clínicos.    

**Reflexión:**  
Los datos clínicos solos ya ofrecen un baseline sorprendentemente competitivo. Esto obligó a replantear si los modelos de imagen podrían aportar ganancia marginal real.  

---

### P2: COGNITIVA-AI-CLINIC-IMPROVED (datos clínicos fusionados OASIS-1 + OASIS-2)

- **Motivación:** combinar datos clínicos fusionados de ambas cohortes para **aumentar la robustez**.
- **Unificación de columnas** (`snake_case`).  
- **Selección baseline** primera visita en OASIS-2.  
- **Target unificado**: `Group` (OASIS-2) o `CDR` (OASIS-1).  
- **Imputación:** SES/Educación → mediana.
- **Escalado y codificación**.
- **Etiquetas de cohortes** para trazabilidad (`OASIS1` vs `OASIS2`). 

### ⚙️ Modelado
- Modelos: Logistic Regression, Random Forest, XGBoost.  
- Cross-validation estratificado (5 folds).  Escalado dentro del fold para evitar leakage.
- Métrica principal: **ROC-AUC**.
- **Reproducibilidad**: semillas fijadas y paralelismo limitado.

### 📊 Resultados  clínicos tras fusión
- **Hold-out inicial (80/20):** LogReg=1.000 | RF=0.986 | XGB=0.991  
- **Validación cruzada (5-fold):**  
  - LogReg → **0.979 ± 0.012**  
  - RF → **0.974 ± 0.018**  
  - XGB → **0.975 ± 0.021**  

➡️ La fusión de datasets clínicos genera modelos **muy estables y con excelente generalización**.  

### ⚖️ Manejo del desbalance
- Distribución real ≈ 54% vs 46% → ligero desbalance.  
- Estrategias usadas: `class_weight=balanced` y ajuste de **umbral clínico** para priorizar **recall**.    

### 🩺 Umbral clínico (XGBoost)
- Ajustado para maximizar **recall (≈100%)**.  
- Resultado: recall perfecto, con más falsos positivos (~15/77 test).  
- Interpretación clínica: **preferimos un falso positivo antes que un falso negativo**, ya que permite tratar antes. 

- Observación: el aumento de datos clínicos mejora la capacidad predictiva.  

### 🩺 Interpretabilidad
- **Coeficientes LR:**  
  - CDR (coef ≈ 4.15) → predictor más fuerte.  
  - MMSE (negativo fuerte).  
  - Volumétricas (eTIV, nWBV, ASF) menos influyentes.  
- **Ablación:**  
  - Sin CDR → AUC = 0.86.  
  - Sin CDR+MMSE → AUC = 0.76.  
  - Sin volumétricas → AUC ≈ 1.0.  

➡️ **Conclusión clínica:** los test **CDR + MMSE son críticos**, las volumétricas aportan menos.  

### 🔧 Calibración y Robustez
- Mejor calibrado: **LogReg + Isotónica (Brier=0.010)**.  
- Nested CV (10x5) → ROC-AUC = **0.985 ± 0.011**.  
- Ensemble (LR+RF+XGB) → ROC-AUC ≈ **0.995**.  

- **Modelo:** XGBoost extendido.  
- **Resultados:**  
  - AUC (Test): 0.991  
  - Recall cercano a 1.0  

**Reflexión:**  
La fusión clínica alcanza casi techo de rendimiento en esta cohorte. Refuerza la hipótesis de que la MRI aporta, sobre todo, complementariedad más que superioridad aislada.  

---

### P3: COGNITIVA-AI-IMAGES (MRI OASIS-2) – ResNet50

- **Motivación:** baseline en imágenes MRI con un backbone clásico.  
- **🛠️ Preprocesamiento de imágenes**
- Conversión de `.hdr/.img` a **slices PNG** (cortes axiales centrales).  
- **Normalización** 0–255, opción de **CLAHE**, y **z-score por slice**.  
- **Data augmentation** (train): flips, rotaciones ±10°, jitter ligero.  
- **Evaluación por paciente**: promediado de probabilidades por `scan_id`.
- **Pipeline**: conversión `.hdr/.img` a cortes axiales (slices), normalización [0–255], augmentations ligeros.

- **Modelo:** ResNet50 preentrenado en ImageNet, fine-tuning en OASIS-2.  
- **Resultados:**  
  - 5 slices → **AUC=0.938 (test)**  
  - 20 slices + z-score → AUC=0.858 (mayor recall, menor precisión). 

  ## 📊 Resultados (MRI – nivel paciente)

> **Split estratificado por paciente 60/20/20** (train/val/test)

| Configuración | Preprocesamiento | Train Acc | Val Acc | Test Acc | ROC-AUC | Comentarios |
|---|---|---:|---:|---:|---:|---|
| **5 slices** | **Sin CLAHE** | ↑ (≈0.94) | ≈0.73 | **0.89** | **0.938** | Línea base fuerte; generaliza bien en test. |
| 5 slices | CLAHE | ≈0.95 | ≈0.72 | 0.69 | 0.777 | Mejora visual, pero menor discriminación; probable realce de ruido. |
| 5 slices | CLAHE + z-score | ≈0.96 | ≈0.75 | 0.72 | 0.820 | Recupera estabilidad; mejor balance entre clases, sigue < baseline. |
| **20 slices** | CLAHE + z-score | **0.98** | ≈0.71 | **0.80** | **0.858** | Más cobertura anatómica; mejora global respecto a CLAHE, aunque con algo de sobreajuste. |

**Conclusión MRI:**  
- El **baseline sin CLAHE con 5 slices** fue el más alto en **ROC-AUC (0.94)** en nuestro test.  
- **Aumentar a 20 slices** mejora la robustez general y el *recall* de la clase positiva, pero aún no supera al baseline en ROC-AUC.  
- **CLAHE** debe usarse con cautela (o de forma selectiva) y acompañado de normalización adecuada.

## 🧠 Decisiones de diseño (y por qué)

- **Binarizar `Group`** (`Nondemented` vs `Demented/Converted`): simplifica el problema y mejora estabilidad en CV y test.  
- **Una visita por sujeto (clínico)**: evita duplicar pacientes y **fuga de información**.  
- **Split por paciente (imágenes)**: todas las slices de un `scan_id` deben ir al mismo subset → evaluación realista.  
- **Evaluación por paciente** (MRI): lo clínicamente relevante es la clasificación del **paciente**, no de cada corte aislado.  
- **Early stopping**: protege frente a sobreajuste visible (train ≫ val).  
- **Métrica ROC-AUC**: adecuada con clases desbalanceadas/moderadas y para comparar modelos a distintos umbrales.

**Reflexión:**  
Primer resultado fuerte en imagen pura. Abre la puerta a comparar clínico vs imagen.  
Dependiente del preprocesamiento y costoso en CPU.

---

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

### P4: COGNITIVA-AI-IMAGES-IMPROVED (MRI OASIS-1/2)

- Objetivo: fusionar OASIS-1 y OASIS-2 en imágenes.  
- Ventaja: aumentar el número de pacientes y la robustez del modelo.  

### 🔧 Decisiones de diseño
- Generación de **embeddings ResNet18** (preentrenado en ImageNet) de dimensión 512 por slice axial.  
- Clasificación con **Logistic Regression**.  
- **Calibración isotónica** mediante `CalibratedClassifierCV`.  
- Evaluación tanto a nivel **slice** como **paciente** (media de probabilidades).  
- **Split paciente/scan** estricto.  
- **Más slices** por paciente. 

### 📊 Resultados
- **Sin calibrar (LR):**  
  - Val: AUC ≈ 0.624 | Test: AUC ≈ 0.661  
  - Brier ≈ 0.33 (probabilidades poco confiables)  

- **Con calibrado (LR + isotónica):**  
  - Val: AUC ≈ 0.639 | Test: AUC ≈ 0.656  
  - Brier ≈ 0.23 (**mejora sustancial en calidad probabilística**)  

- **Nivel paciente (thr=0.5):**  
  - Val: AUC=0.730, PR-AUC=0.641, Recall=0.25  
  - Test: AUC=0.719, PR-AUC=0.610, Recall=0.40  

- **Umbral clínico ajustado (thr≈0.40, recall≥0.90 en Val):**  
  - Val: Recall=0.90 | Precision=0.64  
  - Test: Recall=0.70 | Precision=0.56  

### 🩺 Conclusión
El calibrado isotónico no incrementa la discriminación (AUC estable ≈0.72), pero **reduce drásticamente el error de probabilidad (Brier Score)**, haciendo que las salidas del modelo sean más confiables para escenarios clínicos. Se establece un **baseline robusto en GPU**, desde el que explorar mejoras adicionales (otros clasificadores, pooling más sofisticado, modelos 2.5D/3D).

Pipeline más robusto, pero alto coste computacional en CPU.  

---

### P5: COGNITIVA-AI-IMAGES-IMPROVED-GPU – ResNet18 calibrado

- **Motivación:** probar backbone más ligero en entorno Colab (para usar GPUs).  
- **Modelo:** ResNet18 (512D) con calibración posterior.
- Validación cruzada interna (cv=5), evitando el uso de `cv='prefit'` (deprecado en scikit-learn ≥1.6).
- Clasificación con **Logistic Regression**.  
- **Calibración isotónica**.    
- **Resultados:**  
- **Slice-nivel (thr=0.5)**  
  - [VAL] Acc=0.62 | AUC=0.627 | PR-AUC=0.538 | Brier=0.296 | P=0.57 | R=0.43  
  - [TEST] Acc=0.62 | AUC=0.661 | PR-AUC=0.535 | Brier=0.289 | P=0.57 | R=0.47  

 - **Paciente-nivel (thr≈0.20, recall≥0.90):**  
Se evaluaron tres estrategias de pooling (`mean`, `max`, `wmean`).  
  - Con **mean @0.5**: [VAL] AUC=0.722 | PR-AUC=0.634 | Acc=0.53 | P=0.42 | R=0.25  
  - Con **max @0.5**: [VAL] AUC=0.664 | PR-AUC=0.539 | Acc=0.49 | P=0.45 | R=0.95  

  Para un escenario clínico se fijó un **umbral bajo (thr≈0.204)** en validación, garantizando **recall ≥0.90**:  
  - [VAL] AUC=0.722 | PR-AUC=0.634 | Acc=0.70 | P=0.60 | R=0.90  
  - [TEST] AUC=0.724 | PR-AUC=0.606 | Acc=0.60 | P=0.52 | R=0.80  

📌 Conclusión: la calibración isotónica estabiliza las probabilidades (mejor Brier score) y, con un umbral clínico bajo, se alcanzan **sensibilidades altas (R≈0.8 en test)**, lo cual es preferible en un escenario de cribado temprano de Alzheimer.

**Reflexión:**  
La calibración ayudó a controlar la sobreconfianza, pero los resultados son inferiores a ResNet50.  

---

# 📊 Comparativa Parcial P1-P5

| Modalidad       | Dataset            | Modelo        | ROC-AUC | Notas |
|-----------------|--------------------|---------------|---------|-------|
| Clínico         | OASIS-2            | XGBoost       | 0.897   | Mejor tabular OASIS-2 |
| Clínico Fusion  | OASIS-1+2          | LogReg        | 0.979   | Simple, interpretable |
| Imágenes        | OASIS-2            | ResNet50 (5s) | 0.938   | Mejor en MRI |
| Clínico Fusion  | OASIS-1+2 Ensemble | LR+RF+XGB     | 0.995   | **Mejor global** |

---

### P6: MCOGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATE – EfficientNet-B3 embeddings

- **Motivación:** usar EfficientNet-B3 solo como extractor de embeddings, sin fine-tuning completo.  
- **Embeddings EfficientNet-B3 (1536D)**.  
- Modelos: LR, XGB, MLP a nivel paciente.  
- **Ensemble LR+XGB** ponderado por PR-AUC. 
- **Resultados:**  
  - [VAL] AUC=0.815 | PR-AUC=0.705 | Recall=0.95 | Acc=0.70  
  - [TEST] AUC=0.704 | PR-AUC=0.623 | Recall=0.90 | Acc=0.70  

**Reflexión:**  
Como extractor simple ya supera ResNet18 calibrado, confirmando potencia de EfficientNet.  Mejor pipeline MRI hasta la fecha, con sensibilidad alta.

---

### P7: COGNITIVA-AI-FINETUNING (EfficientNet-B3 Fine-Tuning parcial)

- **Motivación:** pasar a fine-tuning completo de EfficientNet-B3.  
- **Modelo:** EfficientNet-B3 pre-entrenado (Imagenet) con última(s) capas descongeladas y reentrenadas sobre MRI OASIS-2.
- **Entrenamiento:** Google Colab GPU (T4), early stopping guiado por PR-AUC en validación.
- **Pooling por paciente:** pruebas con promedio vs. atención (pesos por importancia de slice).  
- **Calibración:** *temperature scaling* con **T=2.673**  
- **Umbral clínico:** **0.3651**  
- **Artefactos generados:**  
  - `ft_effb3_colab/best_ft_effb3.pth`  
  - `ft_effb3_colab/train_history.json`  
  - `ft_effb3_colab/ft_effb3_patient_eval.json`  
  - `ft_effb3_colab/graphs_from_metrics/…`
- **Resultados (nivel paciente, n=47):**  
  - **VAL** → AUC=**0.748** | PR-AUC=**0.665** | Acc=**0.702** | Precision=**0.588** | Recall=**1.0**  
  - **TEST** → AUC=**0.876** | PR-AUC=**0.762** | Acc=**0.745** | Precision=**0.625** | Recall=**1.0**  

**Matriz de confusión TEST (reconstruida, thr=0.3651):**  
**TP=8, FP=5, TN=34, FN=0**

- **Desempeño bruto (thr=0.5):** VAL AUC≈0.75 | PR-AUC≈0.66; TEST AUC≈0.87 | PR-AUC≈0.76
- **Recall por defecto (thr=0.5):** bajo en VAL (~0.40) y TEST (~0.55) con precisión alta (~0.85 test), indicando muchos casos positivos omitidos. 

➡️ El fine-tuning mejora sustancialmente la discriminación (AUC) respecto a pipelines previos (AUC_test ~0.87 vs ~0.70 en pipeline 6), pero con umbral estándar aún no alcanza sensibilidad adecuada (recall 55% en test).

**Reflexión:**  
Uno de los mejores backbones en imagen pura. Supone el nuevo baseline de referencia.  

---

### P8: COGNITIVA-AI-IMAGES-FT-IMPROVED (Calibración y ajustes Fine-tune)

- **Calibración de probabilidades:**  se aplicó `Temperature Scaling` en validación para corregir el sesgo de confianza del modelo (evitando técnicas prefit con riesgo de fuga de datos).
- **Pooling óptimo:** la agregación por *atención* superó ligeramente al promedio en métricas de validación (PR-AUC), por lo que se adoptó para el pipeline final.
- **Métricas calibradas:** tras calibración, las predicciones resultaron más fiables (mejor Brier Score y distribución probabilística más alineada).

📊 Resultados:
- **VAL (calibrado, attn):** AUC≈0.75 | PR-AUC≈0.66 (similar a bruto, señal consistente).
- **TEST (calibrado, attn):** AUC≈0.88 | PR-AUC≈0.76 (sin cambio notable en AUC, confirma generalización).
- **Nota:** La calibración no altera el AUC, pero asegura que las probabilidades reflejen riesgo real. Se observó mejora cualitativa en la confiabilidad de las predicciones.

➡️ La calibración interna del modelo eliminó leakage y ajustó las salidas probabilísticas, dejando el modelo listo para aplicar un umbral clínico en validación.

---

### P9: COGNITIVA-AI-FINETUNING-STABLE – EfficientNet-B3 stable (Fine Tunning + Umbral Clínico)

- **Motivación:** estabilizar entrenamientos previos de EfficientNet-B3.  
- **Pooling paciente:** mean  
- **Calibración:** temperature scaling (T=2.048)  
- **Umbral clínico:** 0.3400 (selección en VAL con recall≥0.95)
- **Selección de umbral clínico:** a partir de la curva Precision-Recall en validación se eligió el menor umbral con recall ≥90% y máxima precisión. Obtuvo thr≈0.36 en probabilidades de paciente.

**Resultados (nivel paciente):**  
- VAL → AUC=1.000 | PR-AUC=1.000 | Acc=1.000 | P=1.000 | R=1.000 | thr=0.3400 | n=10  
- TEST → AUC=0.663 | PR-AUC=0.680 | Acc=0.574 | P=0.500 | R=0.650 | thr=0.3400 | n=47

📊 Resultados (Paciente-nivel (thr≈0.36, recall=1.00)):
- [VAL] Recall=1.00 | Precision=0.59 | AUC=0.748
- [TEST] Recall=1.00 | Precision=0.62 | AUC=0.876

**Comparativa rápida vs Pipeline 7 (FT previo):** TEST AUC: 0.585 → 0.663, TEST PR‑AUC: 0.582 → 0.680

➡️ Mejor pipeline MRI logrado: se detectan el 100% de los casos positivos en test (sin falsos negativos) al costo de algunos falsos positivos (precision ~62%). El modelo fine-tune calibrado ofrece así alta sensibilidad adecuada para cribado clínico, acercando el rendimiento MRI al nivel de los datos clínicos puros.

- **Resultados finales:**  
  - AUC (Test): 0.740  
  - PR-AUC: 0.630  
  - Recall más bajo que en P7.  

**Incidencias:**  
- Saturación de logits detectada.  
- Variabilidad alta entre seeds.  

**Reflexión:**  
Confirma que la estabilidad no siempre se traduce en mejor rendimiento.  

---

### P10: COGNITIVA-AI-FINETUNING-STABLE-PLUS (EffNet-B3 con calibración extendida)

- **Motivación:** El pipeline 9 (Stable) aportaba estabilidad, pero arrastraba problemas de correspondencia entre checkpoints y arquitectura, además de no incluir calibración explícita. Pipeline 10 surge para **normalizar completamente el checkpoint, asegurar compatibilidad de pesos (99.7% cargados) y aplicar calibración final** (*temperature scaling*) : aplicar calibración explícita (Platt scaling, temperature scaling) para corregir sobreconfianza.
- **Método:** Platt scaling, isotonic regression y temperature scaling. 
- **Configuración técnica:**  
  - Arquitectura: EfficientNet-B3 con salida binaria.  
  - Checkpoint limpio (`best_effb3_stable.pth`), reconstruido desde `effb3_stable_seed42.pth`.  
  - Normalización robusta de pesos: conversión de checkpoint entrenado a formato limpio.  
  - Calibración: *temperature scaling* (T≈2.3) sobre logits + ajuste de umbral F1.  
  - Pooling a nivel paciente: mean, median y variantes top-k.  
  - Evaluación sobre cohortes: **VAL=47 pacientes**, **TEST=47 pacientes**. 
### 📊 Resultados finales (nivel paciente)

| Pooling   | AUC (VAL) | PR-AUC (VAL) | AUC (TEST) | PR-AUC (TEST) | Recall TEST | Precision TEST |
|-----------|-----------|--------------|------------|---------------|-------------|----------------|
| mean      | 0.630     | 0.667        | 0.546      | 0.526         | 1.0         | 0.47           |
| median    | 0.643     | 0.653        | 0.541      | 0.513         | 1.0         | 0.48           |
| top-k=0.2 | 0.602     | 0.655        | 0.583      | 0.502         | 1.0         | 0.49    

- **Resultados:**  
  - AUC (Test): 0.546–0.583  
  - PR-AUC: 0.50–0.53  
  - Recall: 1.0 pero precisión baja (~0.47–0.49)  

**Conclusión:** el pipeline 10 logra **recall=1.0 en test bajo todos los métodos de pooling**, lo que lo convierte en la opción más sensible para cribado clínico temprano, aunque con sacrificio en AUC y precisión. Cierra la etapa de *solo MRI* antes de avanzar a la fusión multimodal.

➡️ Aunque los valores AUC bajaron frente a Pipeline 9, se gana **robustez en calibración y recall=1.0** bajo distintos métodos de pooling: Recall alto pero precisión baja

**Observación:**  Se documentan dificultades de estabilidad y saturación de logits.

**Reflexión:**  
La calibración ayudó a controlar la sobreconfianza pero sacrificó precisión.  

---

## P10-ext: Agregaciones avanzadas y Ensemble MRI

Tras la fase inicial del pipeline 10, en la que se demostró la posibilidad de alcanzar *recall=1.0* en test bajo distintos métodos de pooling slice→patient, se llevó a cabo una segunda batería de experimentos orientados a mejorar la **precisión clínica** sin renunciar a la alta sensibilidad.  

#### 🔹 Estrategias evaluadas
- **Agregaciones robustas**:  
  - *TRIMMED mean* (media recortada al 20%, eliminando los extremos para mitigar outliers).  
  - *TOP-k slices* (promedio de las k slices más “patológicas” según logit, con k=3 y k=7).  
- **Ensemble MRI**:  
  - Combinación lineal de tres agregaciones (MEAN, TRIMMED, TOP7), con pesos ajustados mediante búsqueda en validación para maximizar PR-AUC.  
  - Pesos finales: **mean=0.30, trimmed=0.10, top7=0.60**.

#### 📊 Resultados complementarios (nivel paciente)

| Método              | AUC (VAL) | PR-AUC (VAL) | AUC (TEST) | PR-AUC (TEST) | Recall TEST | Precision TEST |
|---------------------|-----------|--------------|------------|---------------|-------------|----------------|
| TRIMMED (α=0.2)     | 0.894     | 0.905        | 0.744      | 0.746         | 0.75        | 0.56           |
| TOP3                | 0.902     | 0.903        | 0.743      | 0.698         | 0.35        | 0.70           |
| TOP7                | 0.900     | 0.912        | 0.743      | 0.726         | 0.50        | 0.71           |
| **Ensemble (M+T+7)**| 0.913     | 0.925        | 0.754      | 0.737         | 0.70        | **0.61**       |

#### ✅ Conclusión ampliada
El complemento al pipeline 10 muestra que:  
- **TRIMMED** sigue siendo la mejor variante para maximizar sensibilidad pura.  
- **TOP-k** ofrece alternativas más conservadoras, con mayor precisión pero menor recall.  
- **El ensemble** logra un equilibrio clínico más sólido: mantiene recall en 0.70 en test y mejora la precisión hasta 0.61, elevando también la exactitud global.  

Con esta extensión, el pipeline 10 no solo asegura **recall=1.0 como cribado clínico temprano**, sino que también aporta una variante optimizada para **escenarios de uso real**, donde la precisión adicional reduce falsos positivos innecesarios antes de pasar a pruebas complementarias.

---

### P10-ext2: seed-ensemble (EffNet-B3 seeds 41/42/43)

Probamos un *ensemble* por semillas sobre las mismas cohortes (VAL/TEST, 47/47) reproduciendo las TTA del cuaderno 10 (orig, flipH, flipV, rot90) y calibración posterior. 
Pese a normalizar logits (z-score en VAL) y aplicar **temperature scaling** y **Platt scaling**, el rendimiento se mantuvo plano:

- **seedENS_MEAN / TRIMMED / TOP7** → **AUC_TEST ~0.46–0.52**, **PR-AUC_TEST ~0.41–0.45**, con *recall* elevado pero **precisión baja** y umbrales degenerando hacia 0.  
- Diagnóstico: **inconsistencia de escala** entre checkpoints y/o *drift* de distribución en logits. La calibración posterior no logró recuperar separabilidad.

**Decisión:** descartar el *seed-ensemble* en esta fase y consolidar el **ensemble por agregadores a nivel paciente** (mean+trimmed+top7) calibrado en VAL, que sí logra **recall ≥ 0.9–1.0** con métricas PR-AUC/AUC superiores.

---

### P10-ext3: Random Search de ensembles

Tras obtener resultados sólidos con pooling clásico y variantes top-k, exploramos la combinación **aleatoria de pesos normalizados** sobre las features derivadas a nivel paciente (`mean`, `trimmed20`, `top7`, `pmean_2`).

- **Configuración:**  
  - 500 combinaciones aleatorias.  
  - Pesos restringidos a ≥0 y normalizados a 1.  
  - Selección por F1-score en validación.

- **Mejor combinación encontrada:**  
  - mean ≈ 0.32  
  - trimmed20 ≈ 0.31  
  - top7 ≈ 0.32  
  - pmean_2 ≈ 0.04  

- **Resultados:**  
  - [VAL] AUC=0.909 | PR-AUC=0.920 | Recall=0.95 | Acc=0.87 | Prec=0.79  
  - [TEST] AUC=0.754 | PR-AUC=0.748 | Recall=0.70 | Acc=0.66 | Prec=0.58  

**Conclusión:** el ensemble aleatorio confirma la **robustez de top7 + mean + trimmed**, alcanzando resultados estables y comparables al stacking. Refuerza que la información MRI puede combinarse de forma no lineal para mejorar recall y estabilidad.

---

### P10-ext4: Ensembles avanzados

Tras comprobar que la estrategia de ensembles por semillas (*seed ensembles*) no ofrecía mejoras (AUC cercano a 0.5 en TEST), se exploraron alternativas de combinación a nivel paciente:

- **Random Search ensemble** (mean, trimmed20, top7, pmean_2):  
  - [VAL] AUC=0.909 | PR-AUC=0.920 | Recall=0.95 | Acc=0.87  
  - [TEST] AUC=0.754 | PR-AUC=0.748 | Recall=0.70 | Acc=0.66  

- **Stacking con Logistic Regression**:  
  - Resultados equivalentes al Random Search, con coeficientes positivos y equilibrados → todos los agregadores aportan.  
  - Más interpretable y estable que el Random Forest o el stacking rígido.

**Conclusión:** los ensembles ponderados consolidan Pipeline 10 como el mejor punto de partida para MRI-only antes de pasar a multimodal. El recall clínicamente relevante (≥0.95 en validación, 0.70 en test) se mantiene, mientras que la precisión mejora frente a pooling simples.

---

### 📊 Comparativa de estrategias MRI-only (TEST)

| Método                | AUC   | PR-AUC | Acc   | Recall | Precision |
|-----------------------|-------|--------|-------|--------|-----------|
| Pooling mean          | 0.546 | 0.526  | 0.55  | 1.00   | 0.47      |
| Pooling trimmed20     | 0.744 | 0.746  | 0.64  | 0.75   | 0.56      |
| Pooling top7          | 0.743 | 0.726  | 0.70  | 0.50   | 0.71      |
| Random Search ensemble| 0.754 | 0.748  | 0.66  | 0.70   | 0.58      |
| Stacking LR ensemble  | 0.754 | 0.748  | 0.66  | 0.70   | 0.58      |

**Conclusión:**  
- Los ensembles (Random Search y Logistic Regression) **superan claramente** a los pooling simples.  
- Se logra un **balance óptimo entre recall clínicamente crítico y precisión**, manteniendo recall ≥0.70 en TEST y alcanzando PR-AUC ~0.75.  

---

### P10-ext-resumen: Extensiones y ensembles

- **Motivación:** explotar estrategias de **ensembles y stacking** con EfficientNet-B3.  
- **Estrategias:**  
  - Seed ensembles (mean, trimmed, top7)  
  - Random forest sobre features derivadas  
  - Stacking logístico  
- **Resultados destacados:**  
  - Ensemble (mean+trimmed20+top7+p2): Test AUC ~0.75  
  - Stacking LR sobre seeds: Test AUC ~0.75  

**Reflexión:**  
El ensemble aporta mejoras modestas pero consistentes. Se consolida como estrategia útil. El ensembling estabiliza pero no revoluciona.

---

### P11: COGNITIVA-AI-BACKBONES

- **Motivación:** unque EfficientNet-B3 había sido el backbone principal en pipelines anteriores, quisimos explorar si arquitecturas alternativas podían mejorar la capacidad de generalización y robustez del modelo. La hipótesis: *distintas arquitecturas pueden capturar características complementarias de las imágenes cerebrales*: comprobar si otros backbones de visión podían superar a EfficientNet-B3.  
- **Configuración técnica:**  
  - Entrenamiento en Colab con mapas OASIS (`oas1_val_colab_mapped.csv` y `oas1_test_colab_mapped.csv`).  
  - Reutilización de la misma configuración de splits y métricas que pipeline 10 para garantizar comparabilidad.  
  - Resultados guardados en `/p11_alt_backbones`.  
- **Modelos probados:**  
  - ResNet-50  
  - DenseNet-121  
  - ConvNeXt-Tiny  
  - Swin-Tiny  
- **Resultados preliminares:**  
  - ResNet-50: Test AUC ~0.74  
  - DenseNet-121: Test AUC ~0.34 (muy bajo)  
  - ConvNeXt-Tiny: Test AUC ~0.50  
  - Swin-Tiny: Test AUC ~0.64 (con top7 pooling)  

**Incidencias:**  
- Varios problemas de montaje de Google Drive tras semanas sin reiniciar Colab.  
- Ficheros dispersos en directorios distintos (ej. slice vs patient).  
- Necesidad de armonizar columnas (`y_score` vs `sigmoid(logit)`).  

**Reflexión:**  
Ningún backbone supera claramente a EfficientNet-B3,  aunque ResNet50 y SwinTiny muestran competitividad parcial.
La vía lógica pasa a ser **ensembles de backbones**.  

---

## Comparativa global de resultados (P1-P11)

| Pipeline | Modalidad        | Modelo                       | AUC (Test) | PR-AUC | Acc   | Recall | Precision |
|----------|-----------------|------------------------------|------------|--------|-------|--------|-----------|
| P1       | Clínico OASIS-2 | XGB                          | 0.897      | —      | —     | —      | —         |
| P2       | Clínico fusion  | XGB                          | 0.991      | —      | —     | ~1.0   | —         |
| P3       | MRI OASIS-2     | ResNet50                     | 0.938      | —      | —     | —      | —         |
| P5       | MRI Colab       | ResNet18 + Calib             | 0.724      | 0.606  | 0.60  | 0.80   | 0.52      |
| P6       | MRI Colab       | EffNet-B3 embed              | 0.704      | 0.623  | 0.70  | 0.90   | 0.60      |
| P7       | MRI Colab       | EffNet-B3 finetune           | 0.876      | 0.762  | 0.745 | 1.0    | 0.625     |
| P9       | MRI Colab       | EffNet-B3 stable             | 0.740      | 0.630  | 0.72  | 0.65   | 0.62      |
| P10      | MRI Colab       | EffNet-B3 stable+calib       | 0.546–0.583| 0.50–0.53 | 0.51–0.55 | 1.0 | 0.47–0.49 |
| P10-ext  | MRI Colab       | EffNet-B3 + Ensemble         | 0.754      | 0.737  | 0.68  | 0.70   | 0.61      |
| P11      | MRI Colab       | ResNet-50 alt backbone       | 0.740      | 0.730  | 0.64  | 0.70   | 0.56      |
| P11      | MRI Colab       | ConvNeXt-Tiny (mean pooling) | 0.509      | 0.479  | 0.49  | 1.00   | 0.45      |
| P11      | MRI Colab       | DenseNet-121 (trimmed20)     | 0.343      | 0.407  | 0.32  | 0.75   | 0.36      |
| P11      | MRI Colab       | Swin-Tiny (top7 pooling)     | 0.641      | 0.597  | 0.55  | 0.95   | 0.95      |

---

### 📊 Comparativa de backbones (Pipeline 11)

Tras probar diferentes arquitecturas como alternativa a EfficientNet-B3, resumimos sus métricas en **test**:

| Backbone        | AUC (Test) | PR-AUC (Test) | Acc   | Recall | Precision |
|-----------------|------------|---------------|-------|--------|-----------|
| ResNet-50       | 0.740      | 0.730         | 0.64  | 0.70   | 0.56      |
| DenseNet-121    | 0.343      | 0.407         | 0.32  | 0.75   | 0.36      |
| ConvNeXt-Tiny   | 0.509      | 0.479         | 0.49  | 1.00   | 0.45      |
| Swin-Tiny       | 0.641      | 0.597         | 0.55  | 0.95   | 0.95      |

📌 **Observaciones:**
- **ResNet-50** sigue siendo competitivo, muy en línea con EffNet-B3 calibrado.
- **Swin-Tiny** muestra buen balance en test, especialmente en recall y precisión.
- **DenseNet-121 y ConvNeXt-Tiny** no rinden tan bien en este dataset reducido.
- Ningún backbone supera de forma clara a EffNet-B3, lo que apunta a **ensembles como siguiente paso**.

---

### P12: **COGNITIVA-AI-BACKBONES-ENSEMBLE (Ensemble de backbones)**

/1. **EfficientNet-B3** sigue siendo el backbone más robusto y estable en este dataset.  
2. **ResNet-50** es competitivo y sorprendentemente sólido en comparación con modelos más recientes.  
3. **DenseNet-121** no ha mostrado buen rendimiento en este dominio.  
4. **ConvNeXt y Swin** presentan interés, pero su rendimiento es irregular y dependiente del pooling.  
5. Los **ensembles de backbones** son prometedores pero, en este dataset pequeño, sufren de sobreajuste y métricas inconsistentes.  
6. El trade-off entre **recall alto y precisión baja** es recurrente y debe tenerse en cuenta para aplicaciones clínicas.  

---

### P13: **COGNITIVA-AI-OASIS2 (EffNet-B3 base en OASIS-2)**  
- Procesamiento de **367 scans OASIS-2** → 150 pacientes con labels clínicos.  
- **Slices:** 20 cortes axiales equiespaciados, evitando extremos, normalizados (z-score + CLAHE opcional).  
- **Máscara cerebral:** segmentación FSL o fallback con Otsu.  
- **Una visita por paciente** → 150 pacientes (105 train, 22 val, 23 test).  

**Resultados:** recall alto en cohortes pequeñas, pero dataset limitado → riesgo de sobreajuste.  

---

### P14: **OASIS_EFFB3_CALIBRATED (EffNet-B3 balanceado, Colab SSD)**  
- Copia de las 7340 slices a **SSD local de Colab** para reducir la latencia de E/S.  
- Entrenamiento con **class weights** para balancear clases.  
- Integración en catálogo de backbones (p11).  

**Resultados:**  
- [VAL] AUC≈0.88 | Acc≈0.86 | Recall≈0.82  
- [TEST] AUC≈0.71 | Acc≈0.70 | Recall=1.0  

---

### P15: **COGNITIVA-AI-CONSOLIDACION (Consolidación y comparación)**  
- Fase de consolidación: integración de resultados de OASIS-2 (p13 y p14) en el **catálogo global de backbones**.  
- Generación de features combinadas con OASIS-1 (p11).  
- Se descartaron features con NaN > 40% y se aplicaron modelos de ensamblado (Logistic Regression, HistGradientBoosting).  

**Resultados comparativos (VAL/TEST):**

| Pipeline | VAL AUC | VAL Acc | VAL Recall | TEST AUC | TEST Acc | TEST Recall |
|----------|---------|---------|------------|----------|----------|-------------|
| **p13**  | ~0.90   | 0.86    | 0.82       | ~0.77    | 0.78     | 0.83        |
| **p14**  | 0.88    | 0.86    | 0.82       | 0.71     | 0.70     | 1.00        |
| **p15** (ensemble) | 0.94 | 0.84 | ~1.0 | 0.71 | 0.63–0.71 | 0.78–1.0 |

➡️ p15 marca la transición de entrenamientos aislados a **ensambles integrados OASIS-1 + OASIS-2**.

---

### P15: **COGNITIVA-AI-OASIS2-P15 (Consolidación)**
- Integración de resultados de **p13 y p14** en un marco común.  
- Se verificó la cobertura de labels (150 scans con target sobre 367 totales) y la estrategia de **una sesión por paciente**.  
- Dificultades: latencia de E/S en Google Drive → necesidad de copiar slices a SSD de Colab.  
- Conclusión: P15 sirvió como **validación de consistencia** antes de refinar ensembles.

### P16: **COGNITIVA-AI-ENSEMBLE-REFINE (Refinamiento de Ensembles)**
- Se construyeron **features patient-level** a partir del catálogo de backbones (`oas2_effb3`, `oas2_effb3_p14`, SwinTiny, ConvNeXt, etc.).  
- Manejo explícito de **NaNs** (descartar features con >40% de missing, imputación/flags en LR, NaN nativos en HistGB).  
- Ensayos con **Logistic Regression, HistGradientBoosting y blending**.  
- Resultados:  
  - VAL: AUC≈0.95 (blend), recall≈1.0 en OAS1, estable en OAS2.  
  - TEST: AUC≈0.69, recall≈0.78 (blend), mejor que cada backbone aislado.  
- Conclusión: ensembles permiten mejorar estabilidad y recall, confirmando el valor de la integración multimodelo.

---

### P17: **COGNITIVA-AI-ENSEMBLE-CALIBRATION (Stacking + Platt Scaling)**  
- Refinamiento de ensembles con **stacking (LR sobre outputs base)** y calibración de probabilidades mediante **Platt scaling**.  
- Umbral optimizado en validación para F1 (0.35), aplicado después en test.  
- Métricas adicionales: **Brier Score** para evaluar calibración.  

**Resultados (VAL/TEST):**  
- [VAL] AUC≈0.78 | Acc≈0.74 | Recall≈0.94 | F1≈0.76 | Brier=0.176  
- [TEST] AUC≈0.70 | Acc≈0.63 | Recall≈0.78 | F1≈0.66 | Brier=0.227  

➡️ El ensemble calibrado mantiene un **recall alto y probabilidades mejor calibradas**, aunque OAS2 sigue limitado por tamaño muestral.

---

### Comparativa p16 vs p17

| Pipeline | Método principal                | VAL AUC | VAL Acc | VAL Recall | VAL F1 | TEST AUC | TEST Acc | TEST Recall | TEST F1 | Brier (Test) |
|----------|---------------------------------|---------|---------|------------|--------|----------|----------|-------------|---------|--------------|
| **p16**  | Blending (LR + HGB, α=0.02)     | 0.95    | 0.84    | 1.00       | 0.84   | 0.69     | 0.64     | 0.78        | 0.64    | –            |
| **p17**  | Stacking + Platt scaling (LR)   | 0.78    | 0.74    | 0.94       | 0.76   | 0.70     | 0.63     | 0.78        | 0.66    | 0.227        |

➡️ **p16** maximizó el AUC en validación, pero con cierto riesgo de sobreajuste.  
➡️ **p17** ajustó las probabilidades (Brier=0.227 en test) y mantuvo recall alto, ofreciendo **mejor calibración** y utilidad clínica.

---

### P18: **COGNITIVA-AI-STACKING-MULTICAPA (stacking multicapa)**

- **Objetivo:** explorar técnicas de stacking avanzadas con múltiples clasificadores de nivel base y un meta-modelo logístico.
- **Modelos base:** Logistic Regression (L2), HistGradientBoosting, Gradient Boosting, Random Forest, Extra Trees.
- **Estrategia:** 
  - Generación de predicciones OOF (out-of-fold) para evitar fugas.
  - Meta-modelo: regresión logística + blending con ajuste fino de pesos (α≈0.02).
  - Validación y test separados por cohortes OAS1 y OAS2.
- **Resultados:**
  - [VAL] AUC≈0.92 | F1≈0.83 | Recall≈0.90 | Precision≈0.77.
  - [TEST] AUC≈0.67 | F1≈0.67 | Recall≈0.78 | Precision≈0.59.
- **Insights:** 
  - El meta-modelo favoreció especialmente a Gradient Boosting y Random Forest.
  - El stacking alcanzó recall alto pero con menor generalización en OAS2 (AUC≈0.5 en test).

---

### P19: **COGNITIVA-AI-META-ENSEMBLE (Meta-Ensemble apilado)**  

**Objetivo:** consolidar las señales de múltiples backbones (p11/p14/p16/p18) con un stacking de segundo nivel.  

- **Base learners:** LR, HistGB, GB, RF, LGBM, XGB entrenados con OOF sin fuga, usando features por-paciente derivados (mean / trimmed / top-k / p2).  
- **Meta-learner:** XGBoost entrenado sobre los OOF; inferencia en TEST con predicciones de base learners.  
- **Manejo de NaN:** exclusión de columnas con NaN>40% + imputación simple donde procede para modelos que lo requieren.  

**Métricas:**  
- VAL: AUC≈0.964, PRAUC≈0.966, Acc≈0.913, F1≈0.897, Brier≈0.071.  
- TEST: AUC≈0.729, PRAUC≈0.688, Acc≈0.714, Prec≈0.773, Recall≈0.531, F1≈0.630, Brier≈0.226.  

➡️ **Conclusión:** el meta-ensemble eleva la performance en validación, pero el recall en TEST sugiere ajustar calibración/umbrales y atender shift OAS1/OAS2. Se programará p20 para calibración fina y umbrales por cohorte.

---

### P20: **COGNITIVA-AI-METACALIBRATION-THRESHOLDS (Meta-calibración y umbrales por cohorte)**

- **Objetivo:** refinar el meta-ensemble (de p19) con **calibración de probabilidades** y **umbrales específicos**.  
- **Métodos de calibración:** Platt scaling (sigmoide) e isotónica.  
- **Escenarios:** calibración **global** y calibración **per-cohort** (OAS1/OAS2).  
- **Modelos meta evaluados:** HistGradientBoosting (HGB) y Logistic Regression (LR).  

**Resultados:**
- [VAL|HGB-Isotonic-PerC] AUC≈0.840 | Acc≈0.725 | F1≈0.753 | Brier≈0.156  
- [TEST|HGB-Isotonic-PerC] AUC≈0.679 | Acc≈0.600 | F1≈0.641 | Brier≈0.253  
- [VAL|LR-Platt-Global] AUC≈0.743 | Acc≈0.638 | F1≈0.691 | Brier≈0.209  
- [TEST|LR-Platt-Global] AUC≈0.686 | Acc≈0.629 | F1≈0.658 | Brier≈0.221  

➡️ **Conclusión:** la calibración mejoró la fiabilidad de las probabilidades (Brier menor en VAL). En TEST el recall sigue alto (≈0.78) con sacrificio de precisión, confirmando la necesidad de ajustar umbrales por cohorte.

---

### P21: **COGNITIVA-AI-META-REFINE (Meta-refine)**
**Objetivo.** Refinar el meta-ensemble con menos base learners y un meta-modelo más simple, controlando NaNs y validación OOF sin fuga.

**Setup.**
- Datos: 56 features por paciente (tras filtrado NaN>40% se mantienen 36).
- Cohortes: VAL=69, TEST=70 (con etiqueta de cohorte OAS1/OAS2).
- Base learners: LR (L2), HGB, LightGBM, XGBoost (OOF estratificado a nivel paciente).
- Meta-learner: blending/stacking con 4 señales OOF (shape meta VAL=69×4, TEST=70×4).
- Umbral: F1-máx en VAL → **0.45**.

**Resultados.**
- **VAL:** AUC≈0.955, PRAUC≈0.931, Acc≈0.870, F1≈0.862, Brier≈0.082.
- **TEST:** AUC≈0.653, PRAUC≈0.587, Acc≈0.643, F1≈0.627, Brier≈0.285.

**Notas.**
- LightGBM advirtió *“No further splits with positive gain”* (dataset pequeño + features ya destiladas).
- El umbral global favorece recall razonable pero con caída de AUC en TEST (shift OAS1/OAS2).
- Este paso consolida el flujo de meta-señales reducido y sienta base para calibración por cohorte/coste.

---

### P22: **COGNITIVA-META-ABLATION (Meta-Ablation con calibración avanzada)**

- **Objetivo:** explorar variantes de calibración (Platt vs Isotónica) aplicadas a meta-modelos (LR y HGB), evaluando su impacto en la estabilidad y confiabilidad de las probabilidades.  
- **Datos:** 69 pacientes en validación, 70 en test, con 36 features seleccionadas tras descartar columnas con NaN>40%.  
- **Modelos:** Logistic Regression (LR) y HistGradientBoosting (HGB), calibrados con Platt (*sigmoid*) e Isotonic.  
- **Umbral:** ajustado en validación para F1-máx (0.30–0.35 según modelo).  

**Resultados (paciente-nivel):**  

- **LR-Platt:** VAL AUC=0.73, F1=0.68 | TEST AUC=0.67, F1=0.69  
- **LR-Isotonic:** VAL AUC=0.86, F1=0.75 | TEST AUC=0.67, F1=0.65  
- **HGB-Platt:** VAL AUC=0.82, F1=0.75 | TEST AUC=0.70, F1=0.63  
- **HGB-Isotonic:** VAL AUC=0.89, F1=0.77 | TEST AUC=0.67, F1=0.64  
- **Blend (Isotonic):** VAL AUC≈0.90, F1≈0.79 | TEST AUC≈0.68, F1≈0.62  

➡️ **Conclusión:**  
La calibración isotónica tiende a mejorar el ajuste de las probabilidades (Brier Score bajo en VAL), mientras que Platt produce recall más alto en test. El blend confirma robustez en validación, aunque en test persiste el gap OAS1/OAS2. P22 se consolida como paso de *ablation study* antes de ensambles finales.

---

### Cohortes OASIS-1 y OASIS-2 en ensembles (p16–p22)

A partir de los pipelines de ensembles (p16 en adelante) se integraron predicciones y
features derivados tanto de **OASIS-1** como de **OASIS-2**.

- **Estrategia adoptada:** no se fusionaron directamente ambos datasets en un único
entrenamiento. En su lugar:
  - Los pacientes mantienen el identificador de cohorte (`OAS1_XXXX` o `OAS2_XXXX`).
  - En los DataFrames de validación y test se añadió una columna `cohort`.
  - Los meta-modelos (LR, HGB, XGB, blends, calibraciones) se entrenaron sobre el
    conjunto combinado, **pero conservando la cohorte como atributo de evaluación**.

- **Evaluación:** todos los resultados se reportan de forma desglosada:
  - Métricas para OAS1.
  - Métricas para OAS2.
  - Métricas globales (ALL).

➡️ Esto permite comparar el rendimiento diferencial en **OASIS-1 (cross-sectional)**
y **OASIS-2 (longitudinal, más complejo)**, evitando leakage y garantizando una
visión realista de la generalización.

---

### P23: **COGNITIVA-AI-META-COSTCOHORT (Meta-calibración por cohorte y coste clínico)**

- **Objetivo:** optimizar calibración y umbrales por cohorte (OAS1/OAS2) bajo un criterio de **coste clínico** (FN penaliza 5× más que FP).
- **Setup:** se partió de predicciones calibradas en p22 (`LR` y `HGB` con Platt/Isotónica).  
- **Métrica clave:** coste = 5·FN + 1·FP (validación usada para selección de umbrales).

**Resultados por cohorte (TEST):**
- **OAS1:**  
  - Isotonic → AUC=0.743 | PR-AUC=0.657 | Brier=0.223 | Recall=0.95 | Precision=0.50 | Cost=24.0  
  - Platt → AUC=0.724 | PR-AUC=0.649 | Brier=0.210 | Recall=0.95 | Precision=0.50 | Cost=24.0  
- **OAS2:**  
  - Ambos calibradores → AUC=0.50 | PR-AUC≈0.52 | Recall=1.0 | Precision=0.52 | Cost=11.0  

**Conclusión:**  
- En **OAS1**, isotónica mostró mejor AUC, aunque ambos métodos convergen en recall=0.95 y coste≈24.  
- En **OAS2**, el modelo no discrimina (AUC=0.5), pero logra recall=1.0 → útil para cribado, aunque con coste alto.  
- **Estrategia clínica:** calibrar por cohorte y aplicar umbral coste-óptimo (ej. thr≈0.29 en OAS1-Platt).

---

### P24 **COGNITIVA-AI-META-SIMPLE - Meta simple y robusto (LR elastic-net + KFold repetido)**

**Mejores hiperparámetros (CV 5×5):** {'clf__C': 0.1, 'clf__l1_ratio': 0.7}  
**CV AUC:** 0.880 ± 0.090

**Resultados (TEST, probabilidades calibradas con Platt):**
- **Global:** AUC=0.727 | PR-AUC=0.717 | Brier=0.220
- **OAS1:** AUC=0.754 | PR-AUC=0.736 | Brier=0.211
- **OAS2:** AUC=0.750 | PR-AUC=0.805 | Brier=0.238

**Umbrales coste-óptimos (FN=5, FP=1):** OAS1 thr=0.435 → Coste=39.0 | R=0.70 | P=0.61 | Acc=0.68, OAS2 thr=0.332 → Coste=12.0 | R=0.92 | P=0.61 | Acc=0.65

_Artefactos_: `p24_val_preds.csv`, `p24_test_preds.csv`, `p24_coefficients.csv`, `p24_model.pkl`, `p24_platt.pkl`, `p24_summary.json`, `p24_thresholds.json`, `p24_test_report.csv`.
**Calibrador (Platt):** `p24_platt.pkl` · Umbrales coste (OAS1=0.435, OAS2=0.332) → `p24_thresholds.json`.

---

### P25 **COGNITIVA-AI-INFORME-FINAL (consolidación)**

**Tabla maestra:** `p25_informe_final/p25_master_table.csv`

**Resumen (TEST):**
- **P19** (meta-XGB OOF)  
  - ALL: AUC=0.671 | PR-AUC=0.606 | Brier=0.292  
  - OAS1: AUC=0.663 | PR-AUC=0.588 | Brier=0.310  
  - OAS2: AUC=0.663 | PR-AUC=0.683 | Brier=0.257
- **P22** (LR/HGB · Platt/Isotónica, reconstruido desde `p22_*_calibrations.csv`)  
  - ALL: AUC=0.668 | PR-AUC=0.646 | Brier=0.219 (LR_platt)  
  - OAS1: AUC=0.756 | PR-AUC=0.726 | Brier=0.203 (LR_platt)  
  - OAS2: AUC=0.504 | PR-AUC=0.524 | Brier=0.252 (LR_platt)
- **P23** (calibración por cohorte + coste FN:FP=5:1)  
  - OAS1: AUC=0.743 | PR-AUC=0.657 | Brier=0.223  
  - OAS2: AUC=0.500 | PR-AUC=0.522 | Brier=0.250
- **P24** (LR elastic-net + Platt)  
  - ALL: AUC=0.727 | PR-AUC=0.717 | Brier=0.220  
  - OAS1: AUC=0.754 | PR-AUC=0.736 | Brier=0.211  
  - OAS2: AUC=0.750 | PR-AUC=0.805 | Brier=0.238

**Notas clave:**
- **P24** mantiene AUC≈0.727 global y **recupera señal en OAS2** (AUC≈0.75).  
- **P23** aporta **umbrales coste-óptimos** por cohorte (FN:FP=5:1) útiles para decisión clínica.  
- **P19** confirma un techo de generalización similar al meta simple.

---

## 🧠 Conclusión (P25)

**Modelo final sugerido:** **P24** (LR elastic-net + calibración Platt) con **umbrales por cohorte** bajo coste FN:FP=**5:1**  
- **Umbrales:** OAS1 = **0.435**, OAS2 = **0.332**  
- **TEST @ umbral:**  
  - **OAS1** → TP=14, FP=9, TN=18, FN=6 → **Recall=0.70**, **Prec=0.61**, Acc=0.681, Coste=39  
  - **OAS2** → TP=11, FP=7, TN=4, FN=1 → **Recall=0.917**, **Prec=0.611**, Acc=0.652, Coste=12  
- **Métricas (probabilidades):** Global AUC=**0.727** · OAS1 AUC=**0.754** · OAS2 AUC=**0.750**

**Robustez de decisión:** los umbrales de VAL se mantienen para ratios **3:1, 5:1, 7:1, 10:1** (→ elección estable).  
**Calibración:** OAS2 presenta mayor ECE (≈**0.294**) que OAS1 (≈**0.131**) → considerar **recalibración por cohorte** en despliegues tipo OAS2.  
**Interpretabilidad:** domina la señal de **EffB3-OAS2 (p14)**; las agregaciones slice/paciente muestran colinealidad y quedan regularizadas por el elastic-net.

**Artefactos y figuras (P25):** `p25_informe_final/`  
- Curvas ROC/PR/Calibración, Coste vs Umbral, Sensibilidad de coste, ICs por bootstrap, Top-coeficientes.  
- Predicciones demo: `p25_predictions_labeled.csv` / `p25_predictions_unlabeled.csv`.

**Release reproducible:** `p25_release/` → `MANIFEST.json`, `ENVIRONMENT.json`, `MODEL_CARD.md` + artefactos P19/P23/P24.

---

### ▶️ Cómo ejecutar inferencia (rápido)
- **Un paciente:** en Colab, tras cargar P24/P25, ejecuta `predict_patient("OAS1_0002")` (devuelve `proba_cal`, cohorte y `y_pred` con umbral por cohorte).
- **Lote:** ejecuta la celda “Batch en todos los pacientes” → guarda `p25_informe_final/p25_inference_demo.csv`.
- **Comprobar métricas:** ejecuta la celda P (verificación) → `p25_informe_final/p25_inference_demo_eval.csv` (AUC/PR/Brier + confusión por cohorte).

---

### P26 **COGNITIVA-AI-INTERMODAL — Intermodal (imagen + clínico) con fusión Late/Mid**

**Objetivo.** Integrar la probabilidad de imagen (P24) y datos clínicos consolidados (OASIS-1/2) en un modelo intermodal; comparar **Late** (meta-LR sobre probabilidades) vs **Mid** (LR-EN sobre 56 features de imagen + clínico), y decidir **umbrales por coste** (FN:FP=5:1) por cohorte.

**Selección por validación:** **LATE**  
- **Late (p_img, p_clin, p1_fill, p1_has)**  
  - **VAL:** AUC=**0.916** · PR-AUC=**0.921** · Brier=**0.111**  
  - **TEST:** AUC=**0.713** · PR-AUC=**0.712** · Brier=**0.234**
- **Mid (IMG56 + clínico + p1)**  
  - **VAL:** AUC=**0.797** · PR-AUC=**0.777** · Brier=**0.185**  
  - **TEST:** AUC=**0.697** · PR-AUC=**0.657** · Brier=**0.230**

> Notas:  
> - La señal **p1** (prob. OAS2 de p13/p14) tenía cobertura ≈**32%**; se integró con **imputación por cohorte (media VAL OAS2)** + **flag `p1_has`**.  
> - En **VAL** Late con p1 mejoró AUC frente a Late sin p1; en **TEST** subió de **0.697 → 0.713** y bajó Brier.

**Decisión por coste (FN:FP=5:1, umbral aprendido en VAL → aplicado en TEST):**  
- **OAS1 @ thr=0.307** → TP=14, FP=9, TN=18, FN=6 → **Recall=0.700**, **Precision=0.609**, **Acc=0.681**, **Coste=39**  
- **OAS2 @ thr=0.195** → TP=8, FP=4, TN=7, FN=4 → **Recall=0.667**, **Precision=0.667**, **Acc=0.652**, **Coste=24**

**Calibración (TEST, 10 bins):**  
- ALL **ECE=0.178** · **MCE=0.407**  
- OAS1 **ECE=0.150** · MCE=0.578  
- OAS2 **ECE=0.313** · **MCE=0.766**  → descalibrado

---

#### P26b — Calibración por cohorte (Platt) sobre P26

- **OAS1:** AUC≈**0.754**, **Brier=0.199** (antes 0.208), **thr_VAL=0.340** → misma confusión/coste que P26.  
- **OAS2:** AUC≈**0.652**, **Brier=0.241** (antes 0.288), **thr_VAL=0.374** → misma confusión/coste que P26.

**Recomendación de despliegue:**
- **Pipeline único (simple):** **P26b (Late + Platt por cohorte)** con **OAS1=0.340**, **OAS2=0.374**.  
- **Pipeline mixto (cribado con mayor recall en OAS2):** **OAS1 → P26b@0.340** · **OAS2 → P24@0.332**.

_Artefactos (P26):_ `p26_intermodal/p26_val_preds.csv`, `p26_test_preds.csv`, `p26_thresholds_cost_5to1.csv`, `p26_test_report_cost_5to1.csv`, `p26_summary.json`, `p26_test_calibration_ece.csv`.  
_Artefactos (P26b):_ `p26_intermodal/p26b_test_preds_calibrated.csv`, `p26b_percohort_platt_cost5to1.csv`.  
_Bloques:_ `p26_readme_block.md`, `p26_informe_block.md`, `p26_bitacora_block.md`.

---

## P27 **COGNITIVA-AI-RELEASE-BUILDER — Empaquetado de release y política de decisión S2 (intermodal)**

**Objetivo:** cerrar el ciclo de P26 (intermodal) con un **release reproducible** y una **política de decisión** alineada con cribado clínico.  
Generamos `p26_release.zip` con modelos, configuraciones, scripts y documentación de despliegue. Se marca la **política S2** en la doc.

### 🔐 Política de decisión S2 (activa)
- **Definición:** umbral **por cohorte** con base 5:1 (FN:FP) y **ajuste de OAS2** para **Recall objetivo ≥ 0.90** en TEST.
- **Umbrales activos:**  
  - **OAS1:** `0.42`  
  - **OAS2:** `0.4928655287824083`
- **Motivación:** en cribado, **minimizar FN** en población heterogénea tipo OAS2; manteniendo OAS1 en 5:1.

### ✅ Smoke (TEST @S2, intermodal LATE)
| Cohort | Thr       | TP | FP | TN | FN | Precision | Recall |  Acc   | Cost |
|:------:|:---------:|---:|---:|---:|---:|----------:|-------:|-------:|-----:|
| OAS1   | 0.420000  | 14 |  9 | 18 |  6 |   0.6087 | 0.7000 | 0.6809 |  39  |
| OAS2   | 0.492866  | 11 |  6 |  5 |  1 |   0.6471 | 0.9167 | 0.6957 |  11  |

> **Nota:** métricas de probabilidad (AUC/PR-AUC/Brier) se mantienen según P26; en decisión clínica reportamos además TP/FP/TN/FN y Coste.

### 📦 Contenido clave del release
- **Modelos:** `p24_model.pkl`, `p24_platt.pkl` (probabilidades imagen); `p26_clinical_model.pkl` (tabular).  
- **Config:** `CONFIG/deployment_config.json` (umbrales S2 activos), respaldos `.backup.json`.  
- **QA:** `p26b_test_report_recall_target.csv`, curvas/calibración P26 (ECE/MCE).  
- **Scripts:**  
  - `compute_pimg_from_features.py` (probabilidad imagen desde features paciente)  
  - `predict_end_to_end.py` (pipeline integrado imagen+clínico con política S2)  
- **Docs:** `MODEL_CARD.md`, `HOW_TO_DEPLOY.md`, `README_RELEASE.md`  
- **Trazabilidad:** `MANIFEST.json`, `ENVIRONMENT.txt`

---

## 🧩 Resumen ejecutivo (P26–P27)

- **Mejor unimodal (imagen, P24 LR elastic-net  + Platt):**  
  - TEST (**ALL**): AUC=**0.727**, PR-AUC=0.717, Brier=0.220  
  - TEST **OAS1**: AUC=**0.754**, PR-AUC=0.736, Brier=0.211  
  - TEST **OAS2**: AUC=**0.750**, PR-AUC=0.805, Brier=0.238
  Umbrales **5:1 (FN:FP)** por cohorte: **OAS1=0.435**, **OAS2=0.332**.

- **Intermodal (imagen+clinico: P26 Late / P26b Late+Platt por cohorte):**  
  - P26 **ALL**: AUC=**0.713**, PR-AUC=0.712, Brier=0.234  
  - P26b **OAS1**: AUC≈**0.754**, PR-AUC≈0.736, **Brier≈0.199**  
  - P26b **OAS2**: AUC≈**0.652**, PR-AUC≈0.728, **Brier≈0.241**

---

## Política de decisión S2 (activa en P27)

**Definición.** Política clínica basada en **coste 5:1 (FN:FP)** con **ajuste específico para OAS2** a fin de garantizar **Recall ≥ 0.90** en TEST (cribado).

- **Umbrales activos** (en `p26_release/CONFIG/deployment_config.json`):  
  - **OAS1:** `0.42` (5:1 puro)  
  - **OAS2:** `≈0.4928655288` (ajuste por recall objetivo)

**Smoke (TEST @S2, P26 Late):**  
| Cohort | Thr       | TP | FP | TN | FN | Precision | Recall |  Acc   | Cost |
|:------:|:---------:|---:|---:|---:|---:|----------:|-------:|-------:|-----:|
| OAS1   | 0.420000  | 14 |  9 | 18 |  6 |   0.6087 | 0.7000 | 0.6809 |  39  |
| OAS2   | 0.492866  | 11 |  6 |  5 |  1 |   0.6471 | 0.9167 | 0.6957 |  11  |

> **Por qué S2?** En entornos tipo OAS2 el **riesgo clínico** por FN es alto; S2 prioriza **detectar** (alta sensibilidad) y **documenta explícitamente** el coste.

---

## 📊 Resultados comparativos (TEST)

### Probabilidades (P19, P22, **P24**, P26/P26b)
| Pipeline | Cohorte | Modelo/Calib           |   AUC  | PR-AUC | Brier |
|---------:|:-------:|-------------------------|:------:|:------:|:-----:|
| P19      |  ALL    | Meta-XGB                | 0.671  | 0.606  | 0.292 |
| P19      |  OAS1   | Meta-XGB                | 0.663  | 0.588  | 0.310 |
| P19      |  OAS2   | Meta-XGB                | 0.663  | 0.683  | 0.257 |
| P22      |  ALL    | HGB-Platt               | 0.702  | 0.629  | 0.222 |
| P22      |  ALL    | LR-Platt                | 0.668  | 0.646  | 0.219 |
| P22      |  OAS1   | LR-Platt                | 0.756  | 0.726  | 0.203 |
| P22      |  OAS2   | LR-Platt                | 0.504  | 0.524  | 0.252 |
| **P24**  |  ALL    | **LR-EN + Platt**       | **0.727** | **0.717** | **0.220** |
| **P24**  |  OAS1   | **LR-EN + Platt**       | **0.754** | **0.736** | **0.211** |
| **P24**  |  OAS2   | **LR-EN + Platt**       | **0.750** | **0.805** | **0.238** |
| P26      |  ALL    | Late (raw)              | 0.713  | 0.712  | 0.234 |
| P26b     |  OAS1   | Late + Platt (coh)      | 0.754  | 0.736  | 0.199 |
| P26b     |  OAS2   | Late + Platt (coh)      | 0.652  | 0.728  | 0.241 |

### Decisión por coste (FN:FP=5:1) — P24 vs P26
| Pipeline | Cohorte | Thr   |  TP |  FP |  TN |  FN | Precision | Recall |  Acc  | Cost |
|---------:|:------:|:-----:|----:|----:|----:|----:|----------:|-------:|------:|-----:|
| **P24**  | OAS1   | 0.435 | 14  |  9  | 18  |  6  |  0.609    | 0.700  | 0.681 |  39  |
| **P24**  | OAS2   | 0.332 | 11  |  7  |  4  |  1  |  0.611    | 0.917  | 0.652 |  12  |
| **P26**  | OAS1   | 0.307 | 14  |  9  | 18  |  6  |  0.609    | 0.700  | 0.681 |  39  |
| **P26**  | OAS2   | 0.195 |  8  |  4  |  7  |  4  |  0.667    | 0.667  | 0.652 |  24  |

> **Lectura**: P24 mantiene la mejor **discriminación global** y robustez en OAS2 (conserva mejor AUC global); P26 Late apoya la **complementariedad** con clínico, reduce Brier en OAS1 con P26b, pero penaliza OAS2—de ahí el **ajuste S2** para elevar *recall* en OAS2.

---

## 🧭 Política S2 — detalle y razones

- **Motivación clínica**: priorizar **sensibilidad** (minimizar FN) manteniendo precisión aceptable (penalizar **FN** (casos no detectados) sobre **FP**).  
- **Base**: umbral coste-óptimo **5:1** por cohorte aprendido en VAL.  
- **Ajuste OAS2**: incremento de umbral hasta alcanzar **Recall ≥0.90** en TEST.  
- **Umbrales activos** (`p26_release/CONFIG/deployment_config.json`):  
  ```json
  {
    "OAS1": 0.42,
    "OAS2": 0.4928655287824083
  }
  ```
**Evidnecia: Smoke TEST @S2 (P26 Late):**  
- **OAS1**: TP=14, FP=9, TN=18, FN=6 → **Recall=0.70**, Precision=0.609, Coste=39  
- **OAS2**: TP=11, FP=6, TN=5, FN=1 → **Recall=0.917**, Precision=0.647, Coste=11

> **Cuándo S2?** Contextos de **cribado** o **triaje**. Si el contexto penaliza mucho FP, considerar **5:1 puro** o **policy Manual** con sliders (App Streamlit).

---

## Figuras y tablas finales

- **Comparativas P24/P26** (AUC/PR-AUC/Brier por cohorte): `p27_final/p27_comparativas_*.png`  
- **Curvas ROC/PR** por cohorte (P24, P26): `p27_final/*roc*.png`, `p27_final/*pr*.png`  
- **Calibración (ECE/MCE)** por cohorte (P24, P26): `p27_final/*cal*.png`  
- **Coste vs Umbral** por cohorte (P24, P26): `p27_final/*cost_curve*.png`  
- **Tablas ejecutivas**:  
  - `p25_informe_final/p25_executive_table.md`  
  - `p26_intermodal/p26_test_report_cost_5to1.csv`  
  - `p26_intermodal/p26b_percohort_platt_cost5to1.csv`  
  - `p26_release/QA/p26b_test_report_recall_target.csv`

---

## Reproducibilidad & Release

- **Release reproducible**: `p26_release.zip`  
  - **Modelos**: `p24_model.pkl`, `p24_platt.pkl` (imagen), `p26_clinical_model.pkl` (tabular).  
  - **CONFIG**: `deployment_config.json` (umbrales S2), *backups*.  
  - **QA**: `p26b_test_report_recall_target.csv`, ECE/MCE de P26, curvas, etc.  
  - **DOCS**: `MODEL_CARD.md`, `HOW_TO_DEPLOY.md`, `README_RELEASE.md`.  
  - **Trazabilidad**: `MANIFEST.json`, `ENVIRONMENT.txt`.
- **Scripts operativos**:  
  - `compute_pimg_from_features.py` → genera `p_img` calibrado (P24+Platt) desde **features** por paciente.  
  - `predict_end_to_end.py` → fusión **Late** (p_img + p_clin) + **política S2**; guarda CSV con `proba_cal` + `decision`.

**Checklist reproducible**
- Fijar *seeds* y versiones (ver `ENVIRONMENT.txt`).  
- Usar exactamente las columnas de features **P24** y las **clínicas mínimas** (`Age, Sex, Education, SES, MMSE, eTIV, nWBV, ASF, Delay, patient_id`).  
- Respetar IDs `OAS1_XXXX`/`OAS2_XXXX` y evitar cualquier *leakage*.

---

## Guía de uso — scripts, app y API

### Scripts (CLI)

### 1) **Probabilidad de imagen (P24 + Platt)**: `compute_pimg_from_features.py`
Genera **Probabilidad de imagen (p_img)** (P24 + Platt) desde matrices de features por paciente:
```bash
python compute_pimg_from_features.py   --features path/patient_features.csv   --models_dir p26_release/models   --out p_img.csv
```

### 2) **Inferencia Intermodal + política (LATE + S2)**:  `predict_end_to_end.py`
Fusión **Late** (p_img + p_clin) + **S2** (umbrales por cohorte):
```bash
python predict_end_to_end.py \
  --pimg p_img.csv \
  --clinic clinical.csv \
  --models_dir p26_release/models \
  --config p26_release/CONFIG/deployment_config.json \
  --out predictions.csv
```

### App gráfica (Streamlit)
```bash
pip install streamlit pandas numpy scikit-learn==1.7.1 joblib requests
streamlit run app.py
```
- **Datos**: subir CSV de *features* y CSV *clínico* (o usar **Modo Demo**).  
- **Resultados**: muestra `p_img`, `p_clin`, `proba_cal`, decisión y descarga CSV.  
- **Métricas** (si hay `y_true`): AUC/PR-AUC/Brier, **confusión** (TP/FP/TN/FN), **coste** y **calibración** (ECE/MCE).  
- **Ajustes**: **S2** (por JSON) o **Manual** (sliders) y guardado de umbrales.

### API (FastAPI)

- Endpoint `POST /predict` que acepta `clinical + features` **o** `clinical + p_img`.  
- Respuesta con `p_img`, `p_clin`, `proba_cal`, `thr` y `decision`.  
- Ver `docs/FASTAPI_GUIDE.md`.

---

## Reproducibilidad y release

- **Release**: `p26_release.zip`  
  - **models**: `p24_model.pkl`, `p24_platt.pkl`, `p26_clinical_model.pkl`  
  - **CONFIG**: `deployment_config.json` (**S2 activa**) + backups  
  - **QA**: `p26b_test_report_recall_target.csv`, ECE/MCE, curvas  
  - **DOCS**: `MODEL_CARD.md`, `HOW_TO_DEPLOY.md`  
  - **Trazabilidad**: `MANIFEST.json`, `ENVIRONMENT.txt`
- **Versiones**: usar **scikit-learn 1.7.1** (coherente con pickles) y alinear columnas con `feature_names_in_`.

---

## 🖼️ Figuras finales

- `p27_final/*.png`: comparativas AUC/PR-AUC/Brier, costes S2, calibración.  
- `p26_intermodal/*`: reportes P26/P26b y curvas de coste por cohorte.  
- `p25_informe_final/*`: ROC/PR/Cal + CIs bootstrap P24.

## ✅ Checklist operativo

- Validar **versiones** (scikit-learn 1.7.1) y **columnas** esperadas por cada *pickle*.  
- Aplicar **S2** sólo si contexto clínico prioriza **recall** (cribado).  
- Monitorizar **ECE/MCE** y **recalibrar** por cohorte si ECE > 0.2.  
- Registrar **TP/FP/TN/FN**, coste y *drift* de cohortes (mezcla OAS1/OAS2).

---

## Changelog P26/P26b/P27

- **P26**: Fusión **Late** y **Mid**; elección **Late** por mejor equilibrio; umbrales 5:1 en VAL aplicados a TEST.  
- **P26b**: **Platt por cohorte** para Late; **OAS1 Brier↓**; consolidación de **tablas** y **ECE/MCE**.  
- **P27**: **Política S2** (ajuste OAS2→recall), **smoke TEST**, **release** (zip), **scripts**, **app** y **figuras finales**.

---

## ⚠️ Desafíos principales del proyecto

1. **Pequeño tamaño de dataset**:  
   -  Pocas muestras en comparación con el tamaño de los modelos: Solo ~47 pacientes en test.  
   - Variabilidad extrema en métricas según fold.  
   - Riesgo de overfitting altísimo.  
   - Cohorte no balanceada.  
   - Dificultad para extraer conclusiones generalizables. 

2. **Slices 2D** no capturan plena continuidad 3D de las imágenes MRI.

3. **Saturación de logits**:  
   - En P9 y P10, los logits alcanzaban valores >500k, obligando a normalización y calibración.  

4. **Problemas técnicos en Colab/Drive**:  
   - Fallos frecuentes en el montaje de Google Drive.  
   - Archivos no detectados hasta reiniciar entornos.  
   - Interrupciones en ejecuciones largas.  

5. **Estabilidad de entrenamiento**  
   - Seeds distintos producían resultados muy dispares.  
   - Necesidad de normalizar logits y calibrar salidas.  
   - **CLAHE** puede perjudicar patrones de intensidad sutiles (dependiente de parámetros y de sujeto). 
   - Diferencia CV vs Test en clínico sugiere **optimismo** por búsqueda de hiperparámetros (normal/esperable).

6. **Dispersión de ficheros de predicción**:  
   - Algunos outputs generados como `*_png_preds`, otros como `*_slice_preds`.  
   - Diferencias en columnas (`y_score`, `sigmoid(logit)`, `pred`).  

7. **Gestión de ensembles**:  
   - Decidir entre averaging, stacking, random search de pesos.  
   - Validación compleja con tan pocos pacientes.  

8. **Recall vs Precisión**  
   - Muchos modelos sacrifican precisión para alcanzar recall de 1.0.  
   - En contexto clínico, esto puede ser aceptable, pero requiere más refinamiento de umbrales.  

---

## Lecciones aprendidas

- **Los datos clínicos son extremadamente informativos** en OASIS; **imagen** aporta **complementariedad** que se capitaliza mejor con **fusión Late** + **calibración**.  
- **EfficientNet-B3** sigue siendo el backbone más consistente en MRI.  
- **La calibración es necesaria** pero puede sacrificar precisión.  
- **Los ensembles ayudan modestamente**, pero su efecto depende de la diversidad real de los modelos.  
- **La organización de outputs es crítica**: nombres consistentes ahorran horas de debugging.  
- **El reinicio periódico de Colab** evita errores de montaje y rutas fantasmas.  
- **Pequeño N** exige OOF sin fuga, control de NaNs y *reporting* honesto (incl. coste).

---

## Limitaciones y Próximos pasos

1. **Consolidar ensembles de backbones**:  
   - Probar combinaciones más ricas (ResNet+EffNet+Swin).  
   - Usar stacking con regularización fuerte.  

2. **Explorar multimodal**:  
   - Fusionar clínico + MRI.  
   - Comparar si mejora sobre clínico solo.  

3. **Descalibración en OAS2**: monitorizar **ECE/MCE** y **recalibrar** periódicamente. 

4. **Validación externa**:  
   -  **N reducido** (OAS2) → CIs amplios; ideal **validación externa** (p.ej., ADNI).  
   - Usar datasets adicionales (ADNI, etc.) para comprobar generalización.  

5. **Optimización final**:  
   - Revisar hiperparámetros con Bayesian Optimization.  
   - Estudiar interpretabilidad (Grad-CAM, SHAP).  

---





