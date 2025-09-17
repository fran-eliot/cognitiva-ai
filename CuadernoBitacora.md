# 🧭 Cuaderno de Bitácora del Proyecto Cognitiva-AI 

Este cuaderno recopila **todo el recorrido del proyecto Cognitiva-AI**, desde los primeros experimentos con datos clínicos hasta los pipelines más recientes con arquitecturas alternativas y ensembles de backbones.  

Se ha mantenido un registro exhaustivo de cada fase, anotando decisiones técnicas, dificultades encontradas, soluciones aplicadas y reflexiones tras cada bloque de resultados.  

El objetivo es que actúe como un **diario detallado de investigación**, útil tanto para revisiones futuras como para terceros interesados en reproducir o extender el trabajo.

Aquí se incluyen **todas las fases del proyecto** y **entradas diarias (dailys)** con resultados, problemas técnicos y conclusiones.

---

## 📌 Convenciones y notas rápidas

- **Estructura de datos**:
  - `BASE_DIR = /content/drive/MyDrive/CognitivaAI`
  - `DATA_DIR = BASE_DIR/oas1_data`
  - `OUT_DIR` por pipeline (p.ej. `ft_effb3_stable_colab_plus`, `p11_alt_backbones`, etc.)
- **Mapas OASIS**: `oas1_val_colab_mapped.csv`, `oas1_test_colab_mapped.csv` (columnas claves: `png_path`, `target`, `patient_id`, …).
- **Columnas de predicción**:
  - Formatos detectados: `y_score`, `sigmoid(logit)`, `sigmoid(logits)`, `pred`.
  - Se unifica a **`y_score`** internamente durante la carga.
- **Pooling a nivel paciente**: `mean`, `trimmed20`, `top7`, `pmean_2` (power mean con p=2).
- **Métricas**: AUC, PR-AUC, Acc, Recall, Precision. Umbral por:
  - **F1-opt** (maximiza F1 en VAL),
  - **Youden** (maximiza sensibilidad+especificidad-1),
  - **REC90/REC100** (recall fijado).

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

**Conclusión:**  
Pipeline sencillo y robusto, pero dataset limitado (150 sujetos).

---

## Fase 2 – Fusión clínica OASIS-1 + OASIS-2

**Contexto:**  
Para ganar robustez, se unieron OASIS-1 (transversal) y OASIS-2 (longitudinal). Se homogenizaron columnas y se unificó el criterio de la variable objetivo (`Group` vs `CDR`). Esto amplió significativamente el tamaño muestral para entrenar modelos clínicos. 

**Pasos clave:**
- Homogeneización de columnas (`snake_case`).  
- Selección de un mismo baseline (OASIS-2) para ajustar distribución de OASIS-1.  
- Target unificado (`0 = Nondemented`, `1 = Demented/Converted`).  
- Imputación SES/Educación con mediana cuando faltantes.  
- Etiqueta de cohorte para diferenciar sujetos de OASIS-1 vs OASIS-2 (usada en análisis).  

**Resultados clave:**

| Modelo | Hold-out (80/20) | CV 5-fold | Nested CV (10x5) |
|--------|-----------------|-----------|------------------|
| Logistic Regression | 1.000 | 0.979 ± 0.012 | — |
| Random Forest        | 0.986 | 0.974 ± 0.018 | — |
| XGBoost              | 0.991 | 0.975 ± 0.021 | — |
| Ensemble (LR+RF+XGB) | —     | —             | **0.995** |

**Conclusión:**  
Dataset combinado muy estable, modelos calibrados y con gran generalización. Interpretabilidad clínica: **CDR + MMSE** resultaron variables críticas. Se logra un techo de rendimiento muy alto (AUC ~0.99), dejando poco margen de mejora con datos clínicos solos.

---

## Fase 3 – MRI en CPU local (ResNet50 baseline)

**Contexto:**  
Primeros experimentos con MRI provenientes de OASIS-2 (150 sujetos). Se procesaron imágenes estructurales cerebrales para alimentar un modelo de Deep Learning (ResNet50) y evaluar si la información visual aporta a la detección de Alzheimer.  

**Resultados clave:**

| Configuración | AUC (Test) |
|---------------|------------|
| ResNet50 (5 slices, sin CLAHE) | **0.938** |
| ResNet50 (20 slices, z-score) | 0.858 |

**Conclusión:**  
Buen desempeño inicial con pocos cortes (5) por paciente, indicando que la red capta señales relevantes. Al aumentar a 20 slices normalizados, sube el recall pero baja la AUC, sugiriendo ruido adicional. Experimento costoso en CPU local → se decide migrar a **Google Colab con GPU** para acelerar siguientes fases.

---

## Fase 4 – Google Colab GPU (ResNet18 embeddings + calibrado)

**Contexto:**  
Migración a Google Colab (GPU T4). Para aprovechar la aceleración, se cambia el enfoque a extracción de **embeddings**: usar ResNet18 pre-entrenada para obtener vectores por slice y luego entrenar un clasificador ligero (Logistic Regression) sobre esos vectores. Esto reduce el tiempo de entrenamiento y permite calibrar probabilidades.

**Resultados clave:**

| Nivel        | Dataset | AUC  | PR-AUC | Acc  | Recall | Precision | Brier |
|--------------|---------|------|--------|------|--------|-----------|-------|
| Slice        | VAL     | 0.627 | 0.538 | 0.62 | 0.43   | 0.57      | 0.296 |
| Slice        | TEST    | 0.661 | 0.535 | 0.62 | 0.47   | 0.57      | 0.289 |
| Paciente (thr=0.204) | VAL | 0.722 | 0.634 | 0.70 | 0.90 | 0.60 | — |
| Paciente (thr=0.204) | TEST | 0.724 | 0.606 | 0.60 | 0.80 | 0.52 | — |

**Conclusión:**  
El calibrado isotónico **mejora el Brier Score** (probabilidades más confiables), y con un umbral clínico bajo logramos **recall alto (0.80 en test)** → adecuado para cribado inicial. Este pipeline mostró que combinar deep features con ML clásico es efectivo y eficiente en GPU, estableciendo un piso fuerte para sensibilidad.

---

## Fase 5 – Clasificadores alternativos y ensemble (slice→patient)

**Contexto:**  
Sobre los embeddings de MRI (ResNet18), se prueban clasificadores adicionales (SVM, XGBoost) y combinaciones para mejorar el desempeño a nivel paciente. Se busca aprovechar distintos sesgos de modelos y evaluar si un ensemble supera a la regresión logística sola.

**Resultados clave:**

| Modelo | AUC (Val) | AUC (Test) | PR-AUC (Val) | PR-AUC (Test) |
|--------|-----------|------------|--------------|---------------|
| SVM    | 0.731     | 0.746      | 0.618        | 0.628         |
| XGB    | 0.743     | 0.733      | 0.644        | 0.605         |
| Ensemble (LR+SVM+XGB) | 0.728 | 0.728 | 0.641 | 0.605 |

**Conclusión:**  
El ensemble (voto blando promedio) mejora ligeramente la estabilidad pero no supera claramente a los individuales. Se mantiene recall ~0.80 en test. La simplicidad de LR calibrada ya capturaba bien la señal; modelos más complejos tienden a sobreajustar. Se decide entonces explorar mejoras en la generación de features (paso siguiente: embeddings más ricos con otra CNN).

---

## Fase 6 – EfficientNet-B3 embeddings

**Contexto:**  
Se generan embeddings más ricos (1536 dimensiones) con EfficientNet-B3 para cada slice, esperando mejorar la separabilidad. Con estos, se entrenan clasificadores a nivel paciente (LR, MLP, XGB) y su ensemble. También se refuerza la separación Train/Val/Test por paciente. 

**Resultados clave (paciente-nivel):**

| Modelo | VAL AUC | VAL PR-AUC | TEST AUC | TEST PR-AUC | Recall (Test) | Precision (Test) |
|--------|---------|------------|----------|-------------|---------------|------------------|
| LR     | 0.786   | 0.732      | 0.685    | 0.539       | 0.80          | 0.52             |
| MLP    | 0.870   | 0.886      | 0.648    | 0.556       | 0.95          | 0.53             |
| XGB    | 0.782   | 0.633      | 0.670    | 0.617       | 0.75          | 0.56             |
| **Ensemble (LR+XGB)** | **0.815**   | **0.705**      | **0.704**    | **0.623**       | **0.90**          | **0.60**             |

**Conclusión:**  
EffNet-B3 genera embeddings más informativos; los clasificadores simples tienden a sobreajustar (ej. MLP val>>test), pero el **ensemble logra equilibrio** con recall clínico aceptable (90%). Este pipeline aumentó la sensibilidad manteniendo precisión ~0.60, señalando un avance respecto a fases previas.

---

## **Fase 7 – EfficientNet-B3 Fine-tuning parcial**
- **Contexto:** Se migra de utilizar embeddings fijos a fine-tunear parcialmente EfficientNet-B3 directamente con las MRI, permitiendo que la red ajuste sus filtros a patrones específicos de Alzheimer. Se descongelan las últimas capas de EffNet-B3 y se entrena con data augmentation moderada, usando Colab GPU.
- **Notebook**: `cognitiva_ai_finetuning.ipynb`.  
- **Agregación paciente**: *mean pooling*.  
- **Calibración**: *temperature scaling* **T=2.673**; **thr=0.3651**.  
- **Resultados (n=47)**:  
  - **VAL**: AUC **0.748** | PR-AUC **0.665** | Acc **0.702** | P **0.588** | R **1.0**  
  - **TEST**: AUC **0.876** | PR-AUC **0.762** | Acc **0.745** | P **0.625** | R **1.0**  
- **Confusión TEST (thr=0.3651)**: TP=8, FP=5, TN=34, FN=0.  
- **Resultados clave**
  - **AUC (Test) ≈ 0.87**, significativamente mayor que pipelines previos (~0.70).
  - **PR-AUC (Test)** ≈ 0.76, también mejorado.
  - **Recall (Test, thr=0.5)** = 0.55 | **Precisión (Test)** ≈ 0.85 (umbral por defecto).
  - Nota: Con threshold estándar 0.5, el modelo pierde ~45% de casos (recall 55%), evidenciando la necesidad de calibrar/ajustar umbral.
**Conclusión**: 
El fine-tuning de EffNet-B3 **potenció la discriminación** (AUC↑) de las MRI, acercándose al rendimiento de modelos clínicos. No obstante, el modelo afinado tendió a ser conservador en sus predicciones positivas (muchos falsos negativos con thr=0.5). Se identificó la **necesidad de calibrar** sus probabilidades y definir un umbral más bajo orientado a alta sensibilidad.

---

## Fase 7 – OASIS-2 (p13, p14 y p15)

**Contexto:**  
Exploración y explotación del dataset OASIS-2 con EfficientNet-B3.  
Se implementaron tres pipelines consecutivos:

- **p13:** entrenamiento base con criterio de una sola visita por paciente.  
- **p14:** entrenamiento balanceado en Colab GPU, copiando imágenes a SSD para mejorar la E/S.  
- **p15:** consolidación de resultados de OASIS-2 (p13 y p14) junto a OASIS-1 (p11), integrando todos los backbones en un catálogo común y generando features de ensamble.

**Detalles técnicos:**
- 20 slices por volumen, equiespaciados y normalizados (z-score + CLAHE).  
- Labels obtenidos del Excel clínico, convertidos a binario (Control=0, Dementia/Converted=1).  
- Split fijo: 105 train, 22 val, 23 test (1 sesión por paciente).  
- P14: entrenamiento con **class weights** y datos en **SSD local de Colab**.  
- P15: consolidación en catálogo, eliminación de features con NaN≥40%, uso de Logistic Regression (con imputación) y HistGradientBoosting (manejo nativo de NaN).

**Resultados:**
- **p13:** recall alto, dataset limitado (150 pacientes).  
- **p14:** VAL AUC≈0.88, TEST AUC≈0.71 con recall=100%.  
- **p15:** consolidación con ensamble → VAL AUC≈0.94, TEST AUC≈0.71; recall alto sostenido.  
- Integración completa en el catálogo de backbones (`oas2_effb3`, `oas2_effb3_p14`) y en las features consolidadas con OASIS-1.

---
## **Fase 8 – EfficientNet-B3 Fine-tuning parcial**
- **Contexto:** 
Se aplica **calibración de temperaturas** al modelo fine-tune para corregir su tendencia a infraestimar probabilidades de la clase positiva. Además, se confirma el uso de **pooling por atención** para agrupar las predicciones por paciente, dado que mostró mejor PR-AUC en validación que el promedio simple.

- **Resultados clave:**
  - **Probabilidades calibradas:** distribución más acorde a tasas reales; Brier Score mejorado (más bajo).
  - **Pooling atención vs media:** PR-AUC_val 0.66 vs 0.64 → se elige atención (ligera mejora).
  - **Métricas post-calibración (antes de umbral):** AUC_test ~0.88 | PR-AUC_test ~0.76 (sin cambios drásticos, calibración no afecta orden).
  - Se determinó **umbral clínico ~0.36** en VAL para garantizar recall≥90%. Con este: **Recall_val = 1.0**, Precision_val ~0.59.

- **Conclusión:**
Tras calibrar, el modelo fine-tune provee **scores confiables**. La estrategia de atención destaca slices informativos por paciente, optimizando la detección. Ya calibrado y con umbral seleccionado en validación, el modelo está listo para evaluación final con alta sensibilidad.

---

## **Fase 9 – Fine-tuning estable (modelo final MRI)**
- **Contexto:** 
Evaluación del modelo EfficientNet-B3 fine-tune **calibrado** con el **umbral clínico óptimo** en el conjunto de test hold-out. Este es el pipeline MRI definitivo antes de integración multimodal.

- **Resultados clave:**
  - **Threshold aplicado:** ~0.365 (derivado de val).
  - **TEST: Recall = 1.00**|Precision ≈ 0.62 | AUC = 0.876 | PR-AUC = 0.762.
  - Se lograron **0 falsos negativos en test** (detectó todos los casos), a cambio de algunos falsos positivos (precision ~62%).
  - La Acc_test ~0.74 refleja que pese a bajar el umbral, más de 70% de las predicciones totales fueron correctas.

- **Conclusión:**
Pipeline 9 constituye el **mejor modelo MRI** hasta la fecha, alcanzando **sensibilidad del 100%** en test y mejorando sustancialmente la AUC respecto a pipelines anteriores. Este modelo fine-tune estable, aunque genera más alarmas falsas que los modelos clínicos, es ideal como herramienta de **cribado** que no deja pasar casos de demencia incipiente. Marca el cierre de la fase unimodal de imágenes, dando paso a la siguiente etapa: combinar este potente modelo MRI con el igualmente fuerte modelo clínico, en un enfoque multimodal.

---

# Fase 10 – OASIS-2 (p15 y p16)

**Contexto general:**  
Tras los avances logrados con p13 y p14, donde exploramos el dataset OASIS-2 y conseguimos un modelo base sólido con EfficientNet-B3, surgió la necesidad de dar un paso más:  
1. **Consolidar la preparación de datos (p15)** para asegurar coherencia y cobertura completa del dataset.  
2. **Refinar la estrategia de ensembles (p16)**, combinando backbones heterogéneos en un esquema patient-level con métricas robustas.

---

## Fase 11 – Ensemble Calibration (p17)

**Contexto:**  
Tras p16, el siguiente paso fue calibrar las probabilidades del ensemble para aumentar la interpretabilidad y la utilidad clínica.

**Detalles técnicos:**  
- Construcción de un meta-ensemble con Logistic Regression sobre outputs base.  
- Aplicación de Platt scaling y optimización de umbral (F1).  
- Evaluación con Brier Score para medir calibración.  

**Resultados:**  
- Validación: AUC≈0.78, Recall=0.94, F1=0.76, Brier=0.176.  
- Test: AUC≈0.70, Recall=0.78, F1=0.66, Brier=0.227.  
- Cohortes: OAS1 consistente; OAS2 limitado.  

**Conclusión:**  
La calibración refina el ensemble, mantiene sensibilidad alta y mejora la calidad de las probabilidades, aunque la robustez en OAS2 aún requiere trabajo.

---

# Fase 12 – Comparativa p16 vs p17

**p16 – Blending clásico:**  
- LR + HGB combinados con un peso óptimo (α=0.02).  
- Validación espectacular (AUC≈0.95, Recall=1.0), pero riesgo de sobreajuste.  
- En test, buen recall (0.78) pero sin calibración de probabilidades.  

**p17 – Ensemble calibrado:**  
- Stacking con Logistic Regression y Platt scaling.  
- AUC más modesto en validación (0.78) y test (0.70).  
- Mantiene recall≈0.78 y además optimiza la calibración (Brier=0.227 en test).  
- Probabilidades más interpretables, mejor preparadas para escenarios clínicos.  

**Conclusión de la fase:**  
- p16 = **mejor raw performance** (máximo AUC).  
- p17 = **mejor calibración y estabilidad clínica** (probabilidades confiables).  
- Ambos complementan la estrategia de ensembles: uno explota rendimiento, otro asegura interpretabilidad.

---

## Fase 13 – Stacking avanzado (p18)

**Contexto:**  
Tras calibrar ensembles en p17, se diseñó un stacking multicapa para explorar la combinación de múltiples clasificadores heterogéneos.  

**Detalles técnicos:**  
- **Base learners:** Logistic Regression (L2), HistGradientBoosting, Gradient Boosting, Random Forest, Extra Trees.  
- **Meta learner:** regresión logística con blending α=0.02.  
- **Estrategia:**  
  - Generación de predicciones OOF con 5-fold cross-validation.  
  - Validación de umbral óptimo en F1.  
  - Evaluación separada para OAS1 y OAS2.  
- **Métricas adicionales:** Brier Score para calibración, coeficientes de meta-LR y permutación de importancias para interpretar contribuciones.

**Resultados:**  
- [VAL] AUC=0.92, Recall≈0.90, F1≈0.83, Precision≈0.78.  
- [TEST] AUC=0.67, Recall≈0.78, F1≈0.67, Precision≈0.59.  
- Cohorte OAS1 aportó la mayor estabilidad, mientras que OAS2 mantuvo recall alto pero sin señal discriminativa clara (AUC≈0.5).

**Conclusiones:**  
El stacking multicapa permitió validar la viabilidad de **meta-modelos complejos** en un dataset MRI limitado.  
Gradient Boosting y Random Forest emergieron como pilares, aunque la brecha entre validación y test evidencia el reto de generalización en cohortes pequeñas.

---

## Fase 14 – Meta-Ablation y calibración avanzada (P22)

**Contexto:**  
Tras consolidar los ensembles y aplicar calibraciones básicas en fases previas (p20–p21), se diseñó P22 como un *ablation study* para comparar métodos de calibración y medir su efecto en la estabilidad de las probabilidades y en la sensibilidad de los modelos.

**Diseño y ejecución:**  
- Features: 56 columnas iniciales; tras filtrar NaN>40% se mantuvieron 36.  
- Cohortes: 69 pacientes en validación, 70 en test.  
- Modelos calibrados:  
  - Logistic Regression (LR) con imputación y escalado.  
  - HistGradientBoosting (HGB), tolerante a NaNs.  
- Métodos de calibración aplicados:  
  - **Platt scaling (sigmoid).**  
  - **Isotonic regression.**  
- Validación con OOF por StratifiedKFold (sin fugas).  
- Selección de umbral F1-máx en validación (≈0.30–0.35).  

**Resultados principales:**  
- LR-Platt: VAL AUC=0.73, F1=0.68 | TEST AUC=0.67, F1=0.69  
- LR-Isotonic: VAL AUC=0.86, F1=0.75 | TEST AUC=0.67, F1=0.65  
- HGB-Platt: VAL AUC=0.82, F1=0.75 | TEST AUC=0.70, F1=0.63  
- HGB-Isotonic: VAL AUC=0.89, F1=0.77 | TEST AUC=0.67, F1=0.64  
- Blend (Isotonic): VAL AUC≈0.90, F1≈0.79 | TEST AUC≈0.68, F1≈0.62  

**Interpretación:**  
- La calibración isotónica aporta mejor ajuste en validación (Brier bajo), pero pierde robustez en test.  
- Platt mantiene recall alto, lo que lo hace más apto para escenarios de cribado clínico.  
- El blend confirma robustez en validación, pero sigue presente el gap entre cohortes OAS1 y OAS2.  

**Conclusión:**  
P22 aportó claridad sobre qué técnicas de calibración son más fiables en entornos clínicos pequeños y heterogéneos. Constituye la base para P23, donde se buscará integrar estas calibraciones dentro de meta-ensembles finales y analizar umbrales de decisión específicos por cohorte.

---

## Fase 15 – Estrategia OASIS-1 y OASIS-2 en ensembles (p16–p22)

Durante los pipelines de ensembles avanzados (p16–p22) se trabajó con datos de
**OASIS-1 y OASIS-2** simultáneamente. 

**Decisión clave:**
- No fusionar ambos datasets en uno único.
- Mantener la cohorte identificada (`cohort = OAS1 / OAS2`) en todos los
  DataFrames.
- Entrenar meta-modelos (LR, HGB, XGB, blends, calibraciones) sobre los datos
  combinados, pero **siempre evaluando por cohorte y global**.

**Beneficios:**
- Evita leakage entre cohortes.
- Permite comparar rendimiento en escenarios distintos:
  - OAS1: cross-sectional, mayor homogeneidad.
  - OAS2: longitudinal, más ruido y variabilidad.
- Informa sobre la robustez del ensemble frente a shift de dominio.

**Resultado observado:**
- En validación (VAL), OAS1 logra métricas más altas (AUC, Acc).
- En test (TEST), OAS2 muestra recall elevado pero menor calibración y precisión.
- Globalmente (ALL), se obtiene una media ponderada que refleja mejor la
  dificultad del problema.

**Conclusión:**
El tratamiento separado de OASIS-1 y OASIS-2 dentro de los ensembles es esencial
para interpretar los resultados clínicos y diseñar calibraciones específicas
para cada cohorte en los pipelines posteriores (p20–p22).

---

## Fase 16 — P26 / P26b (intermodal)

**Entrada:**  
- Imagen (prob. P24 por paciente) + 56 features de imagen (p11+p14/p13).  
- Clínico consolidado (Age, Sex, Education, SES, MMSE, eTIV, nWBV, ASF, Delay).  
- Señal p1 (OAS2) con imputación por cohorte + flag.

**Arquitectura:**  
- **P26 (Late):** meta-LR sobre `{p_img, p_clin, p1_fill, p1_has}`.  
- **P26b:** P26 + **calibración Platt por cohorte** en VAL y re-umbrales 5:1.

**Métricas (TEST):**  
- P26 — ALL AUC=0.713 · PR-AUC=0.712 · Brier=0.234; OAS1 AUC=0.754 · OAS2 AUC=0.652.  
- P26b — mejora Brier (OAS1 0.199, OAS2 0.241) sin cambiar confusión a coste 5:1.

**Umbrales recomendados:**  
- **P26:** OAS1=0.307 · OAS2=0.195 (coste mínimo).  
- **P26b (único):** OAS1=0.340 · OAS2=0.374.  
- **Mixto (recall OAS2):** OAS1→P26b@0.340 · OAS2→P24@0.332.

**Riesgos:** descalibración en OAS2; tamaño muestral.  
**Mitigaciones:** monitorizar **ECE/MCE**, recalibrar con ventana móvil; reportar intervalos; mantener umbrales por cohorte.

**Artefactos clave:** ver `p26_intermodal/` (predicciones, calibraciones, umbrales, tablas ejecutivas, bloques).

---

# 📅 Diario cronológico


## 📅 01/08/2025 – Inicio del proyecto

- **Fase inicial**: planteamiento general del proyecto.  
- Se define que Cognitiva-AI explorará **clasificación de enfermedad de Alzheimer** usando datos clínicos (OASIS-2) y resonancias magnéticas.  
- Se establecen los objetivos:  
  1. Validar la viabilidad con modelos clínicos tabulares (XGBoost).  
  2. Extender a MRI con backbones CNN/transformers.  
  3. Explorar calibración, ensembles y, finalmente, multimodalidad.

---

## 📅 03/08/2025 – Pipeline 1 (Clínico OASIS-2)

- **Notebook:** `p1_clinico_oasis2.ipynb`.  
- **Datos:** cohortes clínicas de OASIS-2.  
- **Modelo:** XGBoost.  
- **Resultados preliminares:**  
  - AUC ≈ 0.897.  
  - Buenas métricas en validación, confirmando que los datos clínicos son predictivos.  

**Reflexión:** excelente punto de partida, sirve de baseline. Se decide extender a fusión y multimodalidad más adelante.

---

## 📅 06/08/2025 – Pipeline 2 (Clínico Fusión)

- **Notebook:** `p2_clinico_fusion.ipynb`.  
- **Estrategia:** se fusionan variables clínicas tabulares adicionales.  
- **Modelo:** XGBoost mejorado.  
- **Resultados:**  
  - AUC ≈ 0.991.  
  - Recall casi perfecto (~1.0).  

**Reflexión:** métricas altísimas, posible riesgo de overfitting, pero muestra el potencial de fusión de datos tabulares.  
Se decide dar el salto a MRI.

---

## 📅 10/08/2025 – Pipeline 3 (MRI OASIS-2 con ResNet50)

- **Notebook:** `p3_mri_oasis2_resnet50.ipynb`.  
- **Datos:** imágenes MRI de OASIS-2.  
- **Backbone:** ResNet50 preentrenada en ImageNet.  
- **Resultados:**  
  - AUC (test) ≈ 0.938.  

**Reflexión:** confirmación de que los modelos CNN estándar son viables en MRI.  
Este pipeline sirve de puente hacia la fase Colab (con datos más grandes y pipelines posteriores).

---

## 📅 18/08/2025 – Pipeline 5 (MRI Colab con ResNet18 + Calibración)

- **Motivación:** probar pipeline en Colab con mayor escala y calibración.  
- **Acción**: Montaje de Google Drive en Colab, carga de embeddings ResNet18 precomputados, entrenamiento de LogReg con calibración isotónica. 
- **Resultado**: Pipeline de imágenes funcionando en GPU; AUC ~0.72 estable en test, con recall mejorado al ~0.80 aplicando umbral bajo. 
- **Problemas**: Colab desconectó la sesión a mitad → se tuvieron que reconstruir celdas y montar de nuevo el entorno (lección: guardar modelos intermedios). 
- **Conclusión**: Base sólida para MRI en GPU establecida, sentando groundwork para experimentar con modelos más complejos.
- **Resultados:**  
  - AUC ≈ 0.724.  
  - PR-AUC ≈ 0.606.  
  - Accuracy ≈ 0.60.  
  - Recall 0.80 | Precision 0.52.  

**Reflexión:** métricas más bajas que en OASIS-2, debido a mayor complejidad. Se confirma la necesidad de arquitecturas más potentes (EfficientNet).

---

## 📅 21/08/2025 – Pipeline 6 (EfficientNet-B3 embeddings)
- **Enfoque:** usar EffNet-B3 como extractor de embeddings, clasificando con capa adicional.  
- **Acción**: Generación de embeddings de 1536 dimensiones con EfficientNet-B3 para cada slice; entrenamiento de clasificadores LR, MLP y XGB a nivel paciente; comparación de pooling por promedio vs estrategias por paciente. 
- **Resultado**: LR mostró desempeño estable (menos overfitting), MLP tuvo alto overfitting (train >> val), XGB mejoró algo en slices informativos. Un ensemble simple (LR+XGB) incrementó recall en test a 0.90 con precision ~0.60.
- **Conclusión**: Embeddings más ricos abren la puerta a ensembles más sofisticados, pero también pueden sobreajustar con facilidad. Se logra alta sensibilidad (0.90) manteniendo precisión aceptable, validando la estrategia híbrida de combinar modelos. Esto sugiere que para avanzar se requerirá o más datos o técnicas que aprovechen mejor los patrones de imágenes (→ fine-tuning).  

- **Resultados:**  
  - AUC ≈ 0.704.  
  - PR-AUC ≈ 0.623.  
  - Accuracy ≈ 0.70.  
  - Recall 0.90 | Precision 0.60.  

**Reflexión:** mejora en recall, aunque el modelo aún no se estabiliza.  
Se plantea probar fine-tuning completo.

---

### 📅 23/08/2025 – Ensemble híbrido
- **Acción**: Prueba de combinación “híbrida” entre modelos de slice y de paciente: se combinó un XGBoost entrenado directamente a nivel slice (promediando sus scores por paciente) con un MLP entrenado sobre features agregadas de paciente, para capturar información a dos escalas.
- **Resultado**: El ensemble híbrido alcanzó **Recall_test = 0.90** y Precision_test ~0.60, similar al pipeline anterior pero confirmando la aportación complementaria de ambos enfoques (el MLP recuperó algunos positivos que XGBoost solo-slice perdía). 
- **Conclusión**: Se valida la estrategia **multiescala** (slice + paciente) para integrar información. Esto apunta a la relevancia de fusionar diferentes representaciones. Los aprendizajes aquí alimentarán la fase multimodal futura (combinar clínica+MRI). Antes, se decide intentar extraer aún más de las MRI vía fine-tuning de la CNN, ahora que la infraestructura en GPU está probada.

---

## 📅 24/08/2025 – Pipeline 7 (EfficientNet-B3 fine-tune)

- **Motivación:** pasar de embeddings fijos a fine-tuning completo.  
- **Acciones**:  Se llevó a cabo el fine-tuning parcial de EfficientNet-B3: descongelar últimas capas y reentrenar con datos MRI (Train OASIS-2), usando early stopping según PR-AUC en val. Se implementó pooling de atención para destacar slices relevantes por paciente.
  - Montaje de Drive; generación `best_ft_effb3.pth` y `train_history.json`.  
  - *Temperature scaling* → **T=2.673**.  
  - *Pooling* paciente: `mean`.  
- **Rendimiento**:  
  - Copia a SSD local: **~53 f/s** (940 ficheros en ~18 s).  
  - Lectura Drive: **~4.5 img/s**; SSD: **~695 img/s** (muestra 256).  
- **Resultados (paciente, n=47):**  
  - VAL: AUC=0.748 | PR-AUC=0.665 | Acc=0.702 | P=0.588 | R=1.0  
  - TEST: AUC=0.876 | PR-AUC=0.762 | Acc=0.745 | P=0.625 | R=1.0  
- **Matriz de confusión (TEST, thr=0.3651):** TP=8, FP=5, TN=34, FN=0.  
- **Problemas**:  
  - `ValueError: mountpoint must not already contain files` → resuelto con `force_remount=True`.  
  - *Warning* DataLoader: exceso de workers → fijar `num_workers=2`.  
  - Deprecation `torch.cuda.amp.autocast` → migrado a `torch.amp.autocast('cuda')`.
- **Resultados:** El modelo fine-tune entrenó ~10 épocas antes de converger. AUC_test subió a ~0.87, un incremento notable vs embeddings fijos (~0.70). Sin embargo, con threshold=0.5 solo logró recall_test ~0.55 (precision ~0.85). Es decir, clasificó con alta certeza algunos positivos, pero dejó muchos sin detectar a ese umbral.
- **Conclusión:** Fine-tuning demostró ser muy efectivo en potenciar la señal (mejor AUC), pero evidenció la necesidad de recalibrar el modelo para cumplir el requisito clínico de alta sensibilidad. Se planificó calibrar sus probabilidades y ajustar el threshold en la siguiente sesión. 

- **Resultados resumen:**  
  - AUC ≈ 0.876.  
  - PR-AUC ≈ 0.762.  
  - Accuracy ≈ 0.745.  
  - Recall 1.0 | Precision 0.625.  

**Reflexión:** salto cualitativo, confirma que EffNet-B3 es un backbone sólido para MRI.  Se establece como baseline.

---

## 📅 25/08/2025 – Calibración y umbral clínico (EffNet-B3 fine-tune)
- **Acciones**:  
  - Bucle de inferencia optimizado; memoization en SSD local.  
  - retraining reproducible en Colab (EffNet‑B3), caché SSD, AMP (`torch.amp`), early‑stopping por AUC en holdout, calibración (T=2.048), pooling `mean` y selección de umbral 0.3400 con recall≥0.95 en VAL.  
  - Reutilización de **T=2.673** y **thr=0.3651** del JSON estable.  
  - Exportación de **CSV por paciente** (VAL/TEST) y **gráficas** a `ft_effb3_colab/graphs_from_metrics`.  
  - Aplicación de Temperature Scaling en validación para recalibrar las probabilidades de EffNet-B3 fine-tune; cálculo de curva Precision-Recall en val y selección de umbral mínimo con recall ≥ 90%. Luego, evaluación final en test con dicho umbral.
- **Throughput**:  
  - VAL: **~176–198 img/s** | TEST: **~140–150 img/s**.  
- **Resultados consolidados (paciente, n=47)**:  
  - **VAL**: AUC **0.748**, PR-AUC **0.665**, Acc **0.702**, P **0.588**, R **1.0**.  
  - **TEST**: AUC **0.876**, PR-AUC **0.762**, Acc **0.745**, P **0.625**, R **1.0**.  
- **Resultados:**  
  - VAL → AUC=1.000 | PR-AUC=1.000 | Acc=1.000 | P=1.000 | R=1.000 | thr=0.3400 | n=10  
  - TEST → AUC=0.663 | PR-AUC=0.680 | Acc=0.574 | P=0.500 | R=0.650 | thr=0.3400 | n=47 
  - La calibración ajustó ligeramente las probabilidades (T ≈ 1.5). Se identificó **thr ~0.36** para recall_val ≥ 0.90. Con ese threshold, **Recall_test = 1.00** (detectó todos los casos) con **Precision_test ~0.62**. AUC_test se mantuvo en ~0.876. En números absolutos, ningún paciente con Alzheimer en test fue pasado por alto, a costa de ~12 falsos positivos. 
- **Notas**: si se reescribe `ft_effb3_patient_eval.json` con otros CSV/umbral, las métricas pueden variar; se congela este snapshot como **oficial** para el repo.
- **Conclusión:** Se obtuvo un **pipeline MRI óptimo:** modelo calibrado, sin falsos negativos en test. La sensibilidad alcanzada (100%) cumple con creces la meta de cribado. Este resultado supera en equilibrio a todos los intentos previos y deja al modelo listo para integrarse con datos clínicos. Próximo paso: **fusión multimodal** (combinar predicción clínica y de MRI) y validar en cohortes externas (OASIS-3, ADNI) para verificar su generalización.

---

### 📅 25/08/2025 – 03:04 – Pipeline 9 (EffB3 estable)
- **Motivación:** buscar estabilidad entre runs, reduciendo variabilidad.  
- **Acción:** retraining reproducible en Colab (EffNet‑B3), caché SSD, AMP (`torch.amp`), early‑stopping por AUC en holdout, calibración (T=2.048), pooling `mean` y selección de umbral 0.3400 con recall≥0.95 en VAL.  
- **Resultados:**  
  - VAL → AUC=1.000 | PR-AUC=1.000 | Acc=1.000 | P=1.000 | R=1.000 | thr=0.3400 | n=10  
  - TEST → AUC=0.663 | PR-AUC=0.680 | Acc=0.574 | P=0.500 | R=0.650 | thr=0.3400 | n=47  
- **Conclusión:** setup estable listo para el salto a **multimodal** y validación externa.-

---

## 📅 28/08/2025 — P10: EffNet-B3 stable + calibración
- **Objetivo:** añadir calibración (temperature scaling, isotonic). 
- **Incidencia**: grandes magnitudes de **logits** → overflow en `exp`.
- Se aplicó **temperature scaling** y **isotonic regression**.
- Implementación de `safe_sigmoid` con `clip[-50,50]` para evitar overflow.

 - **Resultado(rango):** **AUC test 0.546–0.583**, PR-AUC ~0.50–0.53, Acc ~0.51–0.55, **Recall=1.0**, Precision ~0.47–0.49.
 - **Conclusión:** caída de métricas tras calibración, pero resultados más interpretables.  
Se descubre la importancia de ensembles para recuperar rendimiento.

 ---

## 📅 28/08/2025 — P10-ext: TRIMMED y seed-ensemble
- **Semillas 41/42/43** con agregaciones por paciente.
- **Seed-ensemble (media/TRIMMED/TOP7)** (sin calibrar) dio AUC test ≈ 0.50–0.51 en algunos runs (semillas no aportaron mejora directa).
- **Stacking / Random weights (mean+trimmed20+top7+p2):**
  - **RF** y **STACK(no-neg)** sobre 4 features de pooling:
    - **VAL:** AUC ~0.90–0.91, PR-AUC ~0.92, Acc ~0.85–0.87, R ~0.75–0.95.
    - **TEST:** **AUC ~0.75**, PR-AUC ~0.73–0.75, Acc ~0.64–0.70, R ~0.50–0.70, P ~0.58–0.71.
  - **Ej. RAND(500 samples)** (mean/trimmed20/top7/p2):
    - Pesos ejemplo: mean 0.325, trimmed20 0.315, top7 0.322, p2 0.038.
    - **VAL:** AUC=0.909, PR-AUC=0.920, Acc=0.872, R=0.95, P=0.792.
    - **TEST:** **AUC=0.754**, PR-AUC=0.748, Acc=0.660, R=0.70, P=0.583.
 - **STACK_LR(mean+trimmed20+top7+p2):**
    - * Coefs ≈ [0.407, 0.409, 0.485, 0.416], **intercept −0.923**.
    - **VAL**: AUC=0.909, PR-AUC=0.920, Acc=0.872, R=0.95, P=0.792.
    - **TEST**: AUC=0.754, PR-AUC=0.748, Acc=0.660, R=0.70, P=0.583.
- **Resultados:**  
  - TRIMMED: AUC ≈ 0.744, PR-AUC ≈ 0.746.  
  - Ensemble: AUC ≈ 0.754, PR-AUC ≈ 0.737.  
- **Conclusión:**
    - **Consolidado**: a nivel paciente, **ensembles de pooling** (4 features) mejoran notablemente sobre seed-ensemble puro.
    - Ensembles simples logran mejoras claras.  
Refuerza la idea de avanzar hacia ensembles más sofisticados.

---

### 📅 28/08/2025 — Documentación y limpieza

- Inclusión de resultados de P10 y P10-ext en README e Informe Técnico.
- Normalización de columnas en CSV (y_score, sigmoid(logit), pred).

 ---

## 📅 30/08/2025 — P11: Backbones alternativos (inicio)

* Notebook: `cognitiva_ai_backbones.ipynb`.
- Configuración de `/p11_alt_backbones`.
- **Incidencia**: “Mountpoint must not already contain files” → solución: no remount si ya montado.
- **Incidencia**: DATA_DIR marcado como inexistente pese a estar → solución: reinicio del entorno.
- Validación de mapas OK, config guardada.
- **Motivación:** verificar si otros backbones pueden superar a EffNet-B3.  
- **Backbones probados:**  
  - ResNet-50.  
  - DenseNet-121.  
  - ConvNeXt-Tiny.  
  - Swin-Tiny.  

### Resultados preliminares:
- **ResNet-50:** AUC ≈ 0.740, PR-AUC ≈ 0.730.  
- **DenseNet-121:** AUC ≈ 0.343, PR-AUC ≈ 0.407.  
- **ConvNeXt-Tiny:** AUC ≈ 0.509, PR-AUC ≈ 0.479.  
- **Swin-Tiny:** AUC ≈ 0.641, PR-AUC ≈ 0.597.  

**Reflexión:** ningún backbone supera claramente a EffNet-B3. Swin-Tiny destaca levemente, DenseNet decepciona.  
La evidencia refuerza el interés en **ensembles de backbones**.

---

### 📅 04/09/2025 — Catálogo multi-backbone + normalización columnas

* Escaneo de `p11_alt_backbones` y carpetas previas:
    * Detectados `SwinTiny`, `ConvNeXt slices`, `DenseNet-121`, y además `efb3` de pipelines anteriores (`ft_effb3_*`).
* Unificación de columnas: mapeo a+uto (`y_score`, `sigmoid(logit[s])`, `pred` → `y_score`).
* Construcción features por paciente (VAL/TEST (47, 6) por fuente), guardados:
    * `val_patient_features_backbones.csv`
    * `test_patient_features_backbones.csv`
* Validación:
    * `SwinTiny` OK (940 filas → 47 pacientes).
    * `ConvNeXt slices` OK (940 → 47).
    * `DenseNet` OK (940 → 47).
    * Preds a nivel paciente de pipelines previos (47 directos) incluidas como features extra.

---

### 📅 04/08/2025 — Ensemble de backbones (promedios y stacking base)

- **Objetivo:** combinar predicciones slice-level y patient-level de varios backbones (Swin, ConvNeXt, DenseNet).  
- **Métodos:**  
  - Promedios simples.  
  - Random weights (Dirichlet).  
  - Stacking (logistic regression, isotonic calibration).  

* **AVG** de 12 señales `“*_mean”` (Swin/ConvNeXt/DenseNet + señales paciente/effect):
    * **VAL (F1-opt)**: `AUC` 0.476 | `PR-AUC` 0.389 | `Acc` 0.40 | `R`=1.0 | `P`=0.333 | `thr`=0.3525 | `n`=10.
    * **TEST (F1-opt)**: `AUC` 0.713, `PR-AUC` 0.724 | `Acc` 0.426 | `R`=1.0 | `P`=0.426 | `thr`=0.3525 | `n`=47.
* **Observación**: `AUC` test alto vs val bajo → val (`n`=10) muy pequeño; umbral podría transferirse demasiado “optimista”.
* **STACK\_LR(all\_features)**:
    * **VAL**: `AUC` 0.810 | `PR-AUC` 0.700 | `Acc` 0.800 | `R`=1.0 | `P`=0.600.
    * **TEST**: `AUC` 0.298 | `PR-AUC` 0.397 | `Acc` 0.383 | `P` 0.304 | `R` 0.35.
* **Overfitting claro a VAL**.

---

### 📅 04/09/2025 — Dirichlet (3 backbones, means)

* **FEATURES**: `SwinTiny_mean`, `convnext_tiny..._mean`, `png_preds_d121_mean`.
* `N_SAMPLES`=800 (semilla 42).
* Mejor combinación (ejemplo):
    * Pesos ≈ Swin 0.972, ConvNeXt 0.004, Dense 0.024.
    * **VAL (F1-opt)**: `Acc` 0.70 | `P` 0.50 | `R` 1.0 | `thr` 0.474 | `AUC` 0.714, `PR-AUC` 0.633 (`n`=10).
    * **TEST (F1-opt)**: `Acc` 0.468 | `P` 0.444 | `R` 1.0 | `thr` 0.435 | `AUC` 0.520, `PR-AUC` 0.523 (`n`=47).
* **Youden TEST**: `Acc` 0.617 | `P` 0.667 | `R` 0.20 (umbral 0.481).
* **Conclusión**: mejora leve vs ConvNeXt-mean/DenseNet, pero por debajo de Swin-top7 y muy lejos de los ensembles de EffNet-B3 del P10-ext.

---

### 📅 04/09/2025 — Ensemble Dirichlet EXT (12 features)

* **FEATURES**: `{Swin[mean/trimmed/top7], ConvNeXt_slices[mean/trimmed/top7], DenseNet[mean/trimmed/top7]}` + señales agregadas (`patient_preds_plus_mean`, `slice_preds_plus_mean`, `slice_preds_seedENS_mean`).
* **Resultado**:
    * **VAL**: `AUC` 0.714, `PR-AUC` 0.681.
    * **TEST**: `AUC` 0.361, `PR-AUC` 0.405.
* **Conclusión**: sobreajuste; demasiados grados de libertad para `n(VAL)` = 10.

---

### 📅 04/09/2025 — Stacking L1 fuerte (sparsidad forzada)

* **FEATURES candidatas (ej.)**: `SwinTiny_top7`, `convnext..._top7`, `png_preds_d121_trimmed20`, `patient_preds_plus_mean`, `slice_preds_plus_mean`, `slice_preds_seedENS_mean`.
* **Resultado**: todos `coef=0` (modelo trivial), `intercept=0`.
* **VAL/TEST**: `AUC=0.5`; F1 ligado a prior por umbral 0.
* **Interpretación**: el penalizador “fuerte” anuló todas las señales (`n(VAL)`=10 demasiado pequeño + correlación alta).

---

### 📅 04/09/2025 — Isotonic sobre Swin-Tiny (top7)

* **Resultado**:
    * **VAL**: `AUC` 0.714 | `PR-AUC` 0.556 | `Acc` 0.400 | `R` 1.0 | `P` 0.333 | `thr` 0.0025.
    * **TEST**: `AUC` 0.566 | `PR-AUC` 0.458 | `Acc` 0.553 | `R` 0.95 | `P` 0.487 | `thr` 0.0025.
* **Conclusión**: la calibración isotónica ayuda ligeramente en test y fija un recall alto con precisión moderada.

---

### Resultados:
- **Dirichlet (3 backbones, means):**  
  - VAL: AUC ≈ 0.71, PRAUC ≈ 0.63.  
  - TEST: AUC ≈ 0.52, PRAUC ≈ 0.52.  

- **Dirichlet EXT (12 features):**  
  - VAL: AUC ≈ 0.71, PRAUC ≈ 0.68.  
  - TEST: AUC ≈ 0.36, PRAUC ≈ 0.40.  

- **Stack_LR (all_features):**  
  - VAL: AUC ≈ 0.81, PRAUC ≈ 0.70.  
  - TEST: AUC ≈ 0.29, PRAUC ≈ 0.39.  

- **Swin-Tiny isotonic:**  
  - VAL: AUC ≈ 0.71, PRAUC ≈ 0.55.  
  - TEST: AUC ≈ 0.56, PRAUC ≈ 0.45.  

**Reflexión:** aunque los ensembles no logran mejorar consistentemente el test, sí confirman la complementariedad entre modelos.  
El reto será combinar estabilidad de EffNet-B3 con la diversidad de backbones.

---

### 📅 05/09/2025 — Catálogo ampliado y parsers robustos

* Se indexan también directorios previos:
    * `oas1_resnet18_linearprobe/…`
    * `ft_effb3_colab/…`, `ft_effb3_stable_colab_plus/…`, etc.
* Validación automática de columnas y tamaños; cualquier CSV no conforme se re-mapea.

---

### 📅 05/09/2025 — Revisión de README/Informe/Cuaderno

* Se vuelcan resultados preliminares a la documentación, con filas por pipeline (P1–P11), incluyendo ConvNeXt-Tiny, Swin-Tiny y DenseNet-121.
* Se documenta que la estrategia de semillas en solitario no aportó (`AUC` ≈ 0.5), mientras que ensembles de pooling (4 features) sí mejoraron hasta `AUC` test ≈ 0.75.

---

### 📅 06/09/2025 — Ajustes finales P11 y ensembles

* Normalizado definitivo de nombres en `comparison_backbones_eval.csv`.
* Confirmación de Swin-Tiny (`top7`) como mejor alternativo aislado.
* Resumen de ensembles P11:
    * **Dirichlet (3 means)**: TEST `AUC` ≈ 0.52.
    * **Dirichlet EXT (12)**: TEST `AUC` ≈ 0.36.
    * **STACK\_LR(all)**: TEST `AUC` ≈ 0.30 (overfit).
    * **Swin-Tiny isotonic**: TEST `AUC` ≈ 0.566; `Acc` ≈ 0.553; `R` 0.95; `P` 0.487.

---

### 📅 06/09/2025 – Pipeline p13
- Procesamiento OASIS-2 → 20 slices equiespaciados por scan.  
- Dataset reducido a 150 pacientes (una visita por paciente).  
- Entrenamiento base en Colab (EfficientNet-B3 (105/22/23).) → resultados preliminares positivos, pero limitados.  

---

### 📅 06/09/2025 – Pipeline p14
- Reentrenamiento con imágenes (7340 slices) copiadas a SSD local de Colab.  
- Añadido balanceo de clases con `class weights`.  
- Validación fuerte (AUC≈0.88), recall en test=100%.  
- Integrado al catálogo de backbones.

---

### 📅 06/09/2025 – Pipeline p15 (Consolidación)
- Integración de resultados p13 y p14 en el catálogo global de backbones.  
- Generación de features combinadas con OASIS-1 (p11).  
- Dificultades: manejo de NaN en features y necesidad de descartar/ imputar columnas.  
- Modelos finales: Logistic Regression con imputación y HistGradientBoosting (NaN nativo).  
- Resultado: VAL AUC≈0.94, TEST AUC≈0.71 con recall alto.

---

### 📅 06/09/2025 – Pipeline p15 (Consolidación de dataset OASIS-2)
- Se revisaron de nuevo todos los **367 scans** procesados de OASIS-2.  
- Confirmamos que solo **150 scans** contaban con etiquetas clínicas válidas (Control/Dementia/Converted).  
- Se reafirmó el criterio de **una única sesión por paciente** para evitar *data leakage* entre splits.  
- Se generaron **20 slices axiales equiespaciados** por volumen, eliminando los extremos (8%) y aplicando **normalización z-score + CLAHE opcional**.  
- Resultado: **150 pacientes × 20 slices = 3.000 imágenes etiquetadas**.  
- Dificultad importante: el acceso a imágenes desde Google Drive seguía penalizando el entrenamiento por la latencia de E/S.  
  - **Solución:** replicar todo el dataset en el **SSD local de Colab** antes de cada entrenamiento, lo que redujo drásticamente los tiempos.  
- Con esta consolidación, el dataset quedó consistente, balanceado y preparado para ser integrado en ensembles.

---

### 📅 06/09/2025 – Pipeline p16 (Refinamiento de ensembles)
- Se construyeron features **patient-level** a partir de múltiples backbones:  
  - `oas2_effb3`, `oas2_effb3_p14`, `SwinTiny`, `ConvNeXt_tiny`, `DenseNet121`, entre otros.  
- Durante la integración, se detectó un alto número de columnas con valores faltantes (**NaNs**).  
  - Se aplicó un criterio estricto: **descartar columnas con >40% NaN**.  
  - Para las restantes:  
    - **Logistic Regression (LR):** imputación + columnas-flag de missingness.  
    - **HistGradientBoosting (HGB):** manejo nativo de NaNs, sin necesidad de imputar.  
- Se exploró un esquema de **blending** LR+HGB, optimizado en validación con α=0.02 (casi todo el peso en HGB).  
- **Resultados clave:**  
  - **Validación:**  
    - AUC≈0.95 global, con recall=100% en cohortes OAS1.  
    - En OAS2, las métricas fueron más bajas (AUC≈0.54) debido al reducido tamaño de muestra, pero se mantuvo recall=100%.  
  - **Test:**  
    - AUC≈0.69 global.  
    - Recall≈78%, lo que representa una mejora respecto a modelos individuales.  
    - El blending aportó mayor estabilidad en comparación con usar un solo clasificador.  
- Conclusión: los ensembles **aumentan la sensibilidad del sistema y reducen el riesgo de overfitting**, consolidándose como la mejor estrategia para explotar múltiples backbones en paralelo.

---
### 📅 07/09/2025 – Pipeline p17

- **Objetivo:** Refinar los ensembles con calibración de probabilidades.  
- **Técnicas aplicadas:**  
  - Stacking de outputs base (LR + HGB).  
  - Logistic Regression como meta-modelo.  
  - Platt scaling para calibración probabilística.  
  - Optimización del umbral con F1 en validación.  
- **Resultados globales:**  
  - [VAL] AUC≈0.78 | Recall=0.94 | F1=0.76 | Brier=0.176.  
  - [TEST] AUC≈0.70 | Recall=0.78 | F1=0.66 | Brier=0.227.  
- **Análisis por cohortes:**  
  - OAS1 se mantiene estable (val/test ≈0.84/0.77).  
  - OAS2 continúa siendo inestable, con AUC ≈0.5 en test.  
- **Conclusión:**  
  - El ensemble calibrado aporta **confianza probabilística mejorada**.  
  - Se prioriza recall alto, sacrificando algo de precisión.  
  - El reto sigue siendo el tamaño reducido de OAS2.  

 --- 

### 📅 07/09/2025 – Pipeline p18
- Implementado **stacking multicapa** con cinco clasificadores base (LR, HGB, GB, RF, ET) y un meta-modelo logístico.  
- Generación de predicciones **OOF con 5 folds** para evitar fuga de información.  
- Ajuste de blending α=0.02.  
- Evaluación detallada por cohortes (OAS1, OAS2) y global.  
- **Resultados:**  
  - VAL AUC≈0.92 | Recall≈0.90 | F1≈0.83.  
  - TEST AUC≈0.67 | Recall≈0.78 | F1≈0.67.  
- Insight: GB y RF fueron los más influyentes como modelos base, pero la generalización en OAS2 sigue limitada (AUC≈0.5).  

---

### 📅 07/09/2025 – Pipeline p19

**Fase 8: Ensembles y calibración (P18–P19)**  

- **Qué hice:** ejecuté P19 con stack de base learners (LR, HGB, GB, RF, LGBM, XGB) y meta-XGB. Construí OOF sin fuga con KFold, armé meta-features y evalué en VAL/TEST.  
- **Datos y features:** 56 columnas válidas tras filtrar NaN>40%; representación por paciente (mean/trimmed/top-k/p2).  
- **Resultados:**  
  - VAL: AUC=0.964; PRAUC=0.966; Acc=0.913; F1=0.897; Brier=0.071.  
  - TEST: AUC=0.729; PRAUC=0.688; Acc=0.714; F1=0.630; Brier=0.226.  
- **Aprendizajes:** meta fuerte en VAL pero recall bajo en TEST; hay shift (OAS1 vs OAS2) y el umbral global no es óptimo. LightGBM sin splits útiles sugiere simplificar meta y seleccionar features.  
- **Siguiente paso (p20):** calibrar meta, umbrales por cohorte, meta más simple y Repeated KFold para robustez.

---

### 📅 07/09/2025 – Fase 9: Meta-calibración (P20)

**Qué hice:**  
Ejecuté P20 sobre el meta-ensemble de p19, aplicando calibración de probabilidades con Platt e isotónica, tanto global como por cohorte (OAS1/OAS2).  

**Datos y setup:**  
36 columnas finales tras descartar NaN>40%. Modelos calibrados: HGB y LR. Guardé predicciones calibradas en VAL/TEST y JSON de resumen.  

**Resultados clave:**  
- HGB-Isotonic-PerC: VAL AUC=0.840 | F1=0.753 | Brier=0.156  
- LR-Platt-Global: TEST AUC=0.686 | F1=0.658 | Brier=0.221  
- En TEST, recall≈0.78 con precisión moderada (≈0.54–0.57).  

**Aprendizajes:**  
La calibración reduce el error de probabilidad (Brier), sobre todo en validación.  
El umbral global no captura bien las diferencias entre cohortes; per-cohort mejora ligeramente.  
El modelo calibrado mantiene recall alto → útil en escenario clínico de cribado.  

**Siguiente paso:**  
Integrar calibraciones en el ensemble completo, probar Elastic-Net como meta y explorar selección de umbrales orientada a coste clínico.

---

### 📅 07/09/2025 – Fase 8 · P21 (Meta-refine)

**Qué hice.** Ejecuté p21 con un stacking compacto (LR, HGB, LGBM, XGB) y meta a partir de 4 OOFs; filtré NaN>40% (36 columnas finales) y apliqué umbral F1-máx=0.45.

**Datos.** VAL=69, TEST=70; features por paciente procedentes de múltiples backbones (mean/trimmed/top-k/p2), con columna de cohorte (OAS1/OAS2).

**Resultados.**
- VAL: AUC 0.955, PRAUC 0.931, Acc 0.870, F1 0.862, Brier 0.082.
- TEST: AUC 0.653, PRAUC 0.587, Acc 0.643, F1 0.627, Brier 0.285.

**Observaciones.**
- LGBM sin splits con ganancia positiva → complejidad excesiva frente a muestra disponible.
- Buen VAL pero caída en TEST (shift OAS1/OAS2 + umbral global).

**Siguiente.**
- p22: calibración/umbrales por cohorte y por coste; meta más regularizado; Repeated KFold para robustez.

---

### 📅 07/09/2025 – Pipeline P22 (Meta-Ablation con calibración avanzada)

- **Acción:** ejecuté P22 aplicando calibración Platt e Isotónica a los modelos LR y HGB.  
- **Datos:** 69 pacientes en validación y 70 en test, con 36 features seleccionadas (descartadas 20 por NaN>40%).  
- **Resultados clave:**  
  - LR-Platt: VAL AUC=0.73, F1=0.68 | TEST AUC=0.67, F1=0.69  
  - LR-Isotonic: VAL AUC=0.86, F1=0.75 | TEST AUC=0.67, F1=0.65  
  - HGB-Platt: VAL AUC=0.82, F1=0.75 | TEST AUC=0.70, F1=0.63  
  - HGB-Isotonic: VAL AUC=0.89, F1=0.77 | TEST AUC=0.67, F1=0.64  
  - Blend isotónico: VAL AUC≈0.90, F1≈0.79 | TEST AUC≈0.68, F1≈0.62  
- **Aprendizaje:** la calibración isotónica mejora la fiabilidad de las probabilidades en validación, pero en test muestra menor robustez (shift OAS1/OAS2). Platt mantiene recall más alto.  
- **Conclusión:** P22 funcionó como **estudio de ablación** previo a la integración final de calibraciones en meta-ensembles (p23).

---

### 📅 07/09/2025 – Pipeline P23 (Meta-calibración coste-cohorte)

- **Acción:** ejecuté P23 aplicando calibración Platt e Isotónica con umbrales coste-óptimos por cohorte (OAS1/OAS2).  
- **Criterio:** coste clínico FN=5, FP=1 → penaliza falsos negativos.  
- **Artefactos guardados:**  
  - `p23_val_preds_calibrated.csv`  
  - `p23_test_preds_calibrated.csv`  
  - `p23_thresholds.json`  
  - `p23_calibrators.pkl`  
  - `p23_summary.json`  

**Resultados:**  
- **OAS1 (TEST):**  
  - Isotonic → AUC=0.743 | PR-AUC=0.657 | Recall=0.95 | Precision=0.50 | Cost=24.0.  
  - Platt → AUC=0.724 | PR-AUC=0.649 | Recall=0.95 | Precision=0.50 | Cost=24.0.  
- **OAS2 (TEST):**  
  - Ambos calibradores → AUC=0.50 | PR-AUC≈0.52 | Recall=1.0 | Precision≈0.52 | Cost=11.0.  

**Conclusión:**  
- En OAS1, la calibración isotónica logra mejor AUC, pero Platt es competitivo.  
- En OAS2, el modelo no discrimina (AUC=0.5) pero alcanza recall=1.0, lo que elimina FN (clave clínicamente).  
- Se confirma la necesidad de **umbrales diferenciados por cohorte**.  
- P23 sienta la base para un meta-final más simple y robusto (Elastic-Net + Repeated KFold).

---

### 📅 07/09/2025 — P24 ejecutado (LR elastic-net + KFold repetido + Platt)

- Features paciente fusionadas (p11+p14).  
- CV(5×5): AUC=0.880±0.090; mejores params: {'clf__C': 0.1, 'clf__l1_ratio': 0.7}.  
- TEST Global: AUC=0.727, PR-AUC=0.717, Brier=0.220.  
- TEST OAS1: AUC=0.754, PR-AUC=0.736, Brier=0.211.  
- TEST OAS2: AUC=0.750, PR-AUC=0.805, Brier=0.238.  
- Umbrales coste per-cohorte: OAS1 thr=0.435 → Coste=39.0 (R=0.70, P=0.61, Acc=0.68) | OAS2 thr=0.332 → Coste=12.0 (R=0.92, P=0.61, Acc=0.65)

_Artefactos_: `p24_meta_simple/` (preds, coeficientes, modelo, calibrador, summary, thresholds, report).

---

### 📅 07/09/2025 — P25 (construcción del informe final)

- Consolidé P19/P22/P23/P24 en `p25_master_table.csv`.
- Generé bloques finales para README/Informe/Bitácora.
- Figuras: ROC/PR/Calibración, curvas de coste, sensibilidad de coste, ICs bootstrap; coeficientes top.
- Predicciones demo: `p25_predictions_labeled.csv` / `p25_predictions_unlabeled.csv`.
- Release reproducible: `p25_release/` (MANIFEST.json, ENVIRONMENT.json, MODEL_CARD.md).

**Modelo final sugerido:** P24 (LR elastic-net + Platt) con umbrales por cohorte (FN:FP=5:1).  
**TEST @ umbral:** OAS1→ R=0.70, P=0.61 (Coste=39) · OAS2→ R=0.917, P=0.611 (Coste=12).

---

### 📅 07/09/2025 — P26 intermodal (imagen + clínico)

- Consolidado clínico OASIS-1/2 (anti-fuga), OHE y medianas; 56 features de imagen (p11+p14/p13) alineadas.  
- Señal **p1** (OAS2) con cobertura ≈32% → imputación por cohorte (media VAL OAS2) + flag `p1_has`.  
- **Late vs Mid**:  
  - Late (p_img, p_clin, p1_fill, p1_has) — **VAL AUC=0.916**, TEST **AUC=0.713**.  
  - Mid (IMG56+clínico+p1) — VAL AUC=0.797, TEST 0.697.  
  - Selección: **Late**.  
- **Coste 5:1 (umbral de VAL aplicado en TEST):**  
  - OAS1 @ 0.307 → R=0.700, P=0.609, Acc=0.681, Coste=39.  
  - OAS2 @ 0.195 → R=0.667, P=0.667, Acc=0.652, Coste=24.  
- **Calibración (TEST, 10 bins):** ALL ECE=0.178; OAS1 0.150; **OAS2 0.313**.

---

### 📅 07/09/2025 — P26b (Platt por cohorte)

- Calibración Platt por cohorte entrenada en VAL, aplicada en TEST; re-umbrales 5:1 por cohorte.  
- **OAS1:** Brier 0.208 → **0.199** (AUC≈0.754); **thr_VAL=0.340**; confusión/coste idénticos a P26.  
- **OAS2:** Brier 0.288 → **0.241** (AUC≈0.652); **thr_VAL=0.374**; confusión/coste idénticos a P26.  
- Decisión de producto:  
  - **Único:** P26b (OAS1=0.340, OAS2=0.374).  
  - **Mixto (cribado):** OAS1→P26b@0.340 · OAS2→P24@0.332 (↑ recall).

_Artefactos:_ `p26_intermodal/` (preds, ece/mce, umbrales, report, summary, calibrados, bloques).

---

### 📅 08/09/2025 — P27 (release + política S2)

**Hecho**
- Generé `p26_release.zip` con modelos, config, QA y documentación.  
- Actualicé **MODEL_CARD.md** y **HOW_TO_DEPLOY.md** con la **política S2** activa.  
- Regeneré `MANIFEST.json` y `ENVIRONMENT.txt` (trazabilidad completa).

**Política S2 (marcada)**
- Umbrales activos: `OAS1=0.42`, `OAS2=0.4928655287824083`.  
- Criterio: 5:1 (FN:FP) + ajuste OAS2 para **Recall ≥ 0.90**.  
- Motivo: minimizar FN en dominio OAS2 (más variable/descalibrado), manteniendo el balance 5:1 en OAS1.

**Smoke (TEST @S2)**
- OAS1 → TP=14, FP=9, TN=18, FN=6 ⇒ R=0.70, P=0.61, Acc=0.681, Coste=39.  
- OAS2 → TP=11, FP=6, TN=5, FN=1 ⇒ R=0.917, P=0.647, Acc=0.696, Coste=11.  
- Archivo: `p26_release/QA/p26b_test_report_recall_target.csv`.

**Archivos clave**
- `p26_release.zip` (23 ficheros, con MANIFEST).  
- Scripts: `compute_pimg_from_features.py`, `predict_end_to_end.py`.  
- Config activa: `CONFIG/deployment_config.json` (backup automático).

**Notas**
- ECE P26: ALL≈0.178, OAS1≈0.150, OAS2≈0.313 → seguir monitorizando.  
- Mantener evaluación por cohorte al desplegar; recalibrar si deriva.

**Siguiente**
- (Opcional) Endpoint batch/CLI y plantilla REST.  
- Checklist de producción: logs de FN y ECE, re-calibración por ventana móvil.

---

## 🧭 Chuleta rápida — Política S2 y umbrales

**Política activa (S2)**  
- **OAS1 → 5:1 (FN:FP)** con umbral aprendido en VAL → **thr = 0.42**  
- **OAS2 → “recall objetivo” en VAL (target = 0.85)** → **thr ≈ 0.492866**

**Archivo de configuración:**  
`p26_release/CONFIG/deployment_config.json`

**Claves relevantes dentro del JSON:**
- `policy: "single"`
- `cost_policy: "FN:FP=5:1 (OAS1) + recall_target (OAS2)"`
- `thresholds: { "OAS1": 0.42, "OAS2": 0.4928655287824083 }`
- `thresholds_5to1: { "OAS1": 0.42, "OAS2": 0.49 }`  ← *fallback 5:1 puro*
- `thresholds_recall_target: { "OAS2": { "target": 0.85, "thr_val": 0.4928655…, "found": true } }`

**Cómo cambiar temporalmente de política:**
- **A 5:1 puro:** editar `cost_policy` y copiar `thresholds_5to1` a `thresholds`.
- **Volver a S2:** restablecer `cost_policy` anterior y los `thresholds` de S2.

> Tras editar el JSON, se recomienda un **smoke test** y (opcional) regenerar el ZIP del release.

---

### 📅 08/09/2025 - P27 — Intermodal (Late) + Política S2 (TEST)

| Pipeline | Cohorte | Método |   AUC | PR-AUC | Brier |   Acc |  Prec |   Rec |    Thr | Coste |
|:--------:|:------:|:------:|------:|------:|------:|------:|------:|------:|------:|-----:|
| **P27** | **ALL** | LATE | **0.736** | **0.729** | **0.229** | — | — | — | — | — |
| **P27** | **OAS1** | **S2 (5:1)** | — | — | — | **0.681** | **0.609** | **0.700** | **0.420** | **39** |
| **P27** | **OAS2** | **S2 (recall≥0.85)** | — | — | — | **0.696** | **0.647** | **0.917** | **0.492866** | **11** |

**Notas:**
- Fila **ALL/LATE**: métricas de probabilidad (AUC/PR-AUC/Brier) del modelo intermodal (Late).  
- Filas **OAS1/OAS2 (S2)**: decisión clínica tras calibración por cohorte + política S2 (umbrales por cohorte).

---
### 📅 08/09/2025 P27 — Tablas globales finales

### 1) Probabilidades (TEST) — Comparativa por pipeline y cohorte
> Fuente: `p25_informe_final/p25_master_table.csv` (incluye P19, P22, P23, P24, P26).

| Pipeline | Cohorte | Método        |   AUC | PR-AUC | Brier |
|:--------:|:------:|:--------------|------:|------:|------:|
| P19      | ALL    | XGB           | 0.671 | 0.606 | 0.292 |
| P19      | OAS1   | XGB           | 0.663 | 0.588 | 0.310 |
| P19      | OAS2   | XGB           | 0.663 | 0.683 | 0.257 |
| P22      | ALL    | **HGB_platt** | **0.702** | 0.629 | 0.222 |
| P22      | OAS1   | HGB_platt     | 0.724 | 0.649 | 0.209 |
| P22      | OAS2   | LR_platt      | 0.504 | 0.524 | 0.252 |
| P23      | OAS1   | isotonic      | 0.743 | 0.657 | 0.223 |
| P23      | OAS2   | platt         | 0.500 | 0.522 | 0.250 |
| P24      | ALL    | Platt         | 0.727 | 0.717 | 0.220 |
| P24      | OAS1   | Platt         | 0.754 | 0.736 | 0.211 |
| P24      | OAS2   | Platt         | 0.750 | 0.805 | 0.238 |
| P26      | ALL    | LATE          | 0.713 | 0.712 | 0.234 |
| P26      | OAS1   | LATE          | 0.754 | 0.736 | 0.208 |
| P26      | OAS2   | LATE          | 0.652 | 0.728 | 0.288 |

> Nota: P26=LATE intermodal (p\_img + p\_clin). P22 muestra varias calibraciones; arriba se listan las más representativas.

### 2) Decisión clínica (TEST) — Política activa **S2**
> Fuentes: `p26_release/QA/p26b_test_report_recall_target.csv` (S2) + `CONFIG/deployment_config.json`.

| Pipeline | Cohorte | Política        |  Acc  |  Prec |  Rec  |    Thr   | Coste |
|:--------:|:------:|:----------------|------:|------:|------:|---------:|-----:|
| P27      | OAS1   | **S2 (5:1)**    | 0.681 | 0.609 | 0.700 | 0.420000 |  39  |
| P27      | OAS2   | **S2 (R≥0.85)** | 0.696 | 0.647 | 0.917 | 0.492866 |  11  |

**Chuleta de umbrales S2 (dónde cambiar):** `p26_release/CONFIG/deployment_config.json`  
`thresholds = {"OAS1": 0.42, "OAS2": 0.4928655…}` · `thresholds_5to1` como fallback

---

### 📅 08/09/2025 — P27 (tablas globales y gráficos finales)

- Consolidé tabla **global** de probabilidades (TEST) por *pipeline × cohorte*.  
- Añadí tabla de **decisión clínica @S2** (TEST) con TP/FP/TN/FN, métricas y umbrales por cohorte.  
- Generé **figuras** de AUC/PR-AUC/Brier por cohorte y dejé referencia a ECE/MCE (P26 intermodal).  
- Actualicé documentación con **política S2** vigente (umbrales en `deployment_config.json`).

_Artefactos:_ `p25_informe_final/p25_master_table.csv`, `p26_release/QA/p26b_test_report_recall_target.csv`, `p26_intermodal/p26_test_calibration_ece.csv`, `p27_final/*.png`.

---

### 📅 08/09/2025 - P27 (figuras y tablas finales)

- Generadas figuras de barras **AUC / PR-AUC / Brier** por cohorte desde `p25_master_table.csv`.
- Exportada tabla de **decisión S2** (`p27_final/p27_decision_S2_table.csv`) a partir del QA del release.
- (Si disponible) Creada figura comparativa **S2 vs 5:1** en OAS2.
- Ruta de salida: `p27_final/`.

_Artefactos:_ `p27_final/*.png`, `p27_final/p27_decision_S2_table.csv`.

---

### 📅 10/09/2025 - P27: Scripts de inferencia + GUI y política S2

- Añadidos **scripts operativos**:
  - `compute_pimg_from_features.py` → genera `p_img` (imagen + Platt)
  - `predict_end_to_end.py` → fusión LATE + **S2** (umbrales por cohorte)
- Añadida **app Streamlit (`app.py`)** para ejecutar el pipeline vía navegador.
- **Política activa S2** documentada (OAS1=0.42, OAS2≈0.4928655287824083) en `p26_release/CONFIG/deployment_config.json`.
- Preparado material de documentación (`docs/*.md`) y rutas de modelos (P24/P26).
- Próximos pasos: API REST (FastAPI), Docker, QA automatizado (golden set), monitorización ECE/MCE y coste.

---

### 📅 14/09/2025 - — P27 (Release + Política S2 + QA + Apps)

## ✅ Política de decisión activa (S2)
- **Regla:** mantener OAS1 en el umbral coste-óptimo 5:1 y **ajustar OAS2** para forzar **Recall ≥ 0.90** manteniendo FN muy bajos.
- **Umbrales activos (CONFIG/deployment_config.json):**
  - **OAS1 = 0.42**
  - **OAS2 = 0.4928655287824083**
- **Verificación en TEST (p26b):**
  - **OAS1 @ 0.42** → TP=14, FP=9, TN=18, FN=6 → Precision=0.609 · **Recall=0.700** · Acc=0.681 · Cost=39
  - **OAS2 @ 0.492865** → TP=11, FP=6, TN=5, FN=1 → Precision=0.647 · **Recall=0.917** · Acc=0.696 · Cost=11
- **Motivación:** priorizar **sensibilidad** en OAS2 (entorno más “difícil”) sin penalizar en exceso el coste operativo.

> Los ficheros de QA correspondientes se guardaron en: `p26_release/QA/p26b_test_report_recall_target.csv`.

## 📦 Empaquetado (P26 release)
Se generó **`p26_release.zip`** con:
- `MODELS/` → `p24_model.pkl`, `p24_platt.pkl`, `p24_coefficients.csv` (meta LR + Platt)
- `CONFIG/` → `deployment_config.json` (umbrales S2) + copia de seguridad
- `GOLDEN/` → lote mínimo de prueba y **checksums**
- `DOCS/` → `MODEL_CARD.md`, `HOW_TO_DEPLOY.md`
- `QA/` → tablas de matriz de confusión/curvas coste
- `MANIFEST.json`, `ENVIRONMENT.txt`

## 🧪 QA adicional
- **Calibración (TEST, p26):** ECE@10=0.178 (ALL) · OAS1=0.150 · **OAS2=0.313** → vigilar recalibración periódica por cohorte.
- **Comparativa con P24:** LATE intermodal aporta **mejor Recall** en OAS2 al mismo coste objetivo; AUC global similar (~0.71–0.73).

## 🛠️ Scripts operativos
- `compute_pimg_from_features.py` → computa **p_img** a partir de matrices por-paciente (catálogo p11/p14). *I/O:* CSV de features → CSV con `patient_id, p_img`.
- `predict_end_to_end.py` → **pipeline completo** (imagen + clínico): carga `p24_model.pkl`+`p24_platt.pkl`, fusiona con clínico, aplica política por cohorte (S2) y guarda predicciones/decisiones.
- `predict_batch.py` (opcional) → lotes con **sólo** imagen (si ya existe `X_img` por-paciente).

**Ejemplos de uso rápidos:**
```bash
# 1) p_img desde features
python compute_pimg_from_features.py --val_csv p11_alt_backbones/val_patient_features_backbones.csv \
  --out p26_release/QA/pred_val_pimg.csv

# 2) end-to-end (imagen+clínico) con política S2
python predict_end_to_end.py --X_img p26_release/QA/pred_val_pimg.csv --X_clin data/clinical_consolidated.csv \  --model MODELS/p24_model.pkl --cal MODELS/p24_platt.pkl --config CONFIG/deployment_config.json --out ./preds_val.csv
```

## 🖥️ App Streamlit (demo + real)
- **Demo:** carga un CSV de muestra y permite **jugar** con la política (switch 5:1 vs S2) y **sliders** de umbral por cohorte; muestra TP/FP/TN/FN, **Coste** y curvas (ROC/PR/Calibración).
- **Real:** sube CSV con columnas clínicas estándar + `p_img` o activa el cómputo de `p_img` si hay features. Usa los modelos de `MODELS/` y la política de `CONFIG/`.

**Ejecutar:**
```bash
streamlit run app.py
```

## 🌐 FastAPI (serving ligero)
- Endpoints `/predict` (JSON/CSV) con política S2, `/healthz`, `/version`.  
- Recomendado como **microservicio** detrás de Streamlit o de un front externo.

## ✅ Checklist de cierre
- [x] Política S2 documentada en README/Informe/Bitácora y **reflejada** en `deployment_config.json`.
- [x] QA reproducible (confusiones por cohorte, coste, ECE).
- [x] Artefactos firmados y **MANIFEST** actualizado.
- [x] Demo interactiva lista (Streamlit).
- [x] Guía de FastAPI y scripts de batch.

----

...
### 🧪 Extractos de logs útiles

* Logits extremos y z-score (cuando aplicó):
    ```
    VAL (pre) logits: min=-7.78e5 | max=5.45e5 | mean≈-1.52e4 | std≈9.0e4
    VAL (post-z) logits: min≈-8.49 | max≈6.23 | std≈1.00
    TEST (pre) logits: min=-6.43e5 | max=4.92e5 | mean≈-1.28e4 | std≈8.87e4
    TEST (post-z) logits: min≈-7.10 | max≈5.69 | std≈1.00
    ```
* `safe_sigmoid` aplicado siempre antes de calibración/ensembles que consumen logits.

---

### ⚠️ Incidencias recurrentes y soluciones

* **Drive ya montado**:
    * Error: `“Mountpoint must not already contain files”`.
    * Solución: si `drive.mount()` falla, NO forzar; reiniciar entorno o usar `force_remount=True` sólo cuando sea estrictamente necesario.
* **`DATA_DIR`/`VAL_MAP`/`TEST_MAP` “no existen” aun existiendo**:
    * Causa: estado inconsistente de sesión (muchas horas/días sin reiniciar).
    * Solución: reinicio completo; volver a montar; re-evaluar `Path.exists()`.
* **Columnas heterogéneas** (`y_score`, `sigmoid(logit)`, `pred`):
    * Solución: diccionario de normalización y validación de esquemas, forzando `y_score`.
* **Overflow en `exp` (sigmoid)**:
    * Solución: `safe_sigmoid` con `clip[-50, 50]`.
* **Sobreajuste de ensembles complejos** (Dirichlet EXT, STACK\_LR all-features):
    * Causa: `n(VAL)`=10, muchas features correlacionadas.
    * Mitigación: reducir features, validación cruzada a paciente, o usar regularización/priors más informativos.

---

# 📊 Resumen numérico (hitos clave, test)
| Bloque | Método / Configuración | AUC | PR-AUC | Acc | Recall | Precision |
|--------|------------------------|-----|--------|-----|--------|-----------|
| P7     | EffNet-B3 finetune     | .876| .762   | .745| 1.00   | .625      |
| P9     | EffNet-B3 stable       | .740| .630   | .72 | .65    | .62       |
| P10    | EffB3 stable + calib   | .546–.583 | .50–.53 | .51–.55 | 1.00 | .47–.49 |
| P10-ext| Ensemble pooling       | .754| .748   | .66–.70 | .50–.70 | .58–.71 |
| P11    | ConvNeXt-Tiny (mean)   | .509| .479   | .489| 1.00   | .455      |
| P11    | DenseNet-121 (trimmed) | .343| .407   | .319| .75    | .36       |
| P11    | Swin-Tiny (top7)       | .641| .597   | .553| .95    | .95       |
| P11-ens| Dirichlet (3 means)    | .520| .523   | .468| 1.00   | .444      |
| P11-ens| Dirichlet EXT (12)     | .361| .405   | .447| .85    | .425      |
| P11-ens| Swin-Tiny + isotonic   | .566| .458   | .553| .95    | .487      |

**Lectura**: los mejores ensembles paciente-level siguen siendo los construidos sobre EffNet-B3 (P10-ext).
Entre backbones alternativos, Swin-Tiny (`top7`) es el mejor individual; con isotonic gana algo de robustez.

---


# 📎 Apéndice: utilidades clave
Incluye `safe_sigmoid`, `fit_temperature`, `normalize_score`, `agg_patient`.

---

### 📎 Apéndice: fragmentos y utilidades

#### `safe_sigmoid` y `temperature scaling`

```python
import numpy as np
from scipy.optimize import minimize

def safe_sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1/(1+np.exp(-z))

def fit_temperature(logits, y_true, init_T=1.0, bounds=(0.05,10.0)):
    logits = np.asarray(logits,float); y_true = np.asarray(y_true,float)
    def nll(T):
        p = safe_sigmoid(logits/T); eps=1e-7
        return -np.mean(y_true*np.log(p+eps)+(1-y_true)*np.log(1-p+eps))
    return float(minimize(lambda t: nll(t[0]), x0=[init_T], bounds=[bounds], method="L-BFGS-B").x[0])
```

#### Normalización de columnas de score

```python
SCORE_ALIASES = ['y_score','sigmoid(logit)','sigmoid(logits)','pred']

def normalize_score(df):
    for c in SCORE_ALIASES:
        if c in df.columns:
            df = df.rename(columns={c:'y_score'})
            break
    assert 'y_score' in df.columns, "No encuentro columna de score."
    return df
```

#### Pooling a paciente (`mean`/`trimmed20`/`top7`/`pmean_2`)

```python
import pandas as pd
import numpy as np

def agg_patient(df):
    g = df.groupby('patient_id')['y_score']
    return pd.DataFrame({
        'mean': g.mean(),
        'trimmed20': g.apply(lambda s: s.sort_values().iloc[int(len(s)*.1):int(len(s)*.9)].mean() if len(s)>=10 else s.mean()),
        'top7': g.apply(lambda s: s.sort_values(ascending=False).head(7).mean()),
        'pmean_2': g.apply(lambda s: (np.mean(np.power(np.clip(s,0,1),2)))**0.5)
    }).reset_index()
```

---


## 🔍 Desafíos principales encontrados

1. **Inestabilidad en Colab:** sesiones largas provocaban errores o pérdida de conexión. Reinicios forzados solucionaron varios problemas.  
2. **Gestión de Google Drive:** errores frecuentes de montaje/desmontaje y rutas inconsistentes, resueltos con reinicios y verificaciones explícitas.  
3. **Variabilidad de resultados:** seeds distintas producían métricas diferentes; se resolvió con ensembles y calibración.  
4. **Dificultad en calibración:** temperature scaling mejoraba interpretabilidad pero bajaba AUC. Hubo que combinar con ensembles.  
5. **Backbones alternativos:** algunos decepcionaron (DenseNet) o no superaron a EffNet, confirmando que no hay “ganador absoluto”.  
6. **Complejidad de ensembles:** métodos como Dirichlet o Stacking mostraron sobreajuste en validación y peores métricas en test.  
7. **Limitación de datos:** tamaño reducido del dataset afectó a generalización, especialmente en arquitecturas grandes como Swin.  
8. **Gestión de logs y CSV:** múltiples formatos distintos (`y_score`, `sigmoid(logit)`, etc.), lo que exigió unificación manual en varios experimentos.

