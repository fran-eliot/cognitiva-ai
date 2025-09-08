> *Branch creada para analizar el código y hacer comentarios.*

# 🧠 COGNITIVA-AI – Experimentos de Clasificación Multimodal

Este repositorio documenta **toda la evolución experimental** en el marco del proyecto **Cognitiva-AI**, cuyo objetivo ha sido **explorar modelos de machine learning para diagnóstico de Alzheimer** combinando datos clínicos y de imagen (MRI OASIS-2).  

El documento sigue un enfoque **cuaderno de bitácora extendido**, en el que cada pipeline corresponde a un conjunto de experimentos con motivaciones, configuraciones técnicas, métricas obtenidas y reflexiones.  
El tono es intencionadamente **verboso y detallado**: se incluyen incidencias de ejecución, errores y aprendizajes prácticos que acompañaron cada etapa.  

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

## Introducción

El proyecto **Cognitiva-AI** parte de la necesidad de evaluar modelos predictivos que integren datos clínicos y de imagen (MRI) en cohortes reducidas como OASIS-2.  

Desde el inicio se asumió que:
- Los **datos clínicos** podrían servir como baseline fuerte (edad, MMSE, CDR, etc.).  
- Las **imágenes cerebrales** aportarían riqueza multimodal pero con mayor complejidad.  
- Sería necesario experimentar con **diferentes backbones** de visión profunda y con **estrategias de calibración, ensembles y stacking** para compensar el pequeño tamaño muestral.  

El proceso se organizó en **pipelines numerados**. Cada uno corresponde a un conjunto de experimentos exploratorios.  

---

## Pipelines experimentales

### P1: Datos clínicos con XGBoost

- **Motivación:** establecer un baseline sólido con datos tabulares clínicos.  
- **Modelo:** XGBoost con optimización básica de hiperparámetros.  
- **Resultados:**  
  - AUC (Test): 0.897  
  - Buen baseline, aunque limitado a información tabular.  

**Reflexión:**  
Los datos clínicos solos ya ofrecen un baseline sorprendentemente competitivo. Esto obligó a replantear si los modelos de imagen podrían aportar ganancia marginal real.  

---

### P2: Datos clínicos fusionados

- **Motivación:** combinar datos clínicos enriquecidos o fusionados con información adicional.  
- **Modelo:** XGBoost extendido.  
- **Resultados:**  
  - AUC (Test): 0.991  
  - Recall cercano a 1.0  

**Reflexión:**  
La fusión clínica alcanza casi techo de rendimiento en esta cohorte. Refuerza la hipótesis de que la MRI aporta, sobre todo, complementariedad más que superioridad aislada.  

---

### P3: MRI OASIS-2 – ResNet50

- **Motivación:** baseline en imágenes MRI con un backbone clásico.  
- **Modelo:** ResNet50 preentrenado en ImageNet, fine-tuning en OASIS-2.  
- **Resultados:**  
  - AUC (Test): 0.938  

**Reflexión:**  
Primer resultado fuerte en imagen pura. Abre la puerta a comparar clínico vs imagen.  

---

### P5: MRI Colab – ResNet18 calibrado

- **Motivación:** probar backbone más ligero en entorno Colab.  
- **Modelo:** ResNet18 con calibración posterior.  
- **Resultados:**  
  - AUC (Test): 0.724  
  - PR-AUC: 0.606  
  - Acc: 0.60 | Recall: 0.80 | Precisión: 0.52  

**Reflexión:**  
La calibración ayudó a controlar la sobreconfianza, pero los resultados son inferiores a ResNet50.  

---

### P6: MRI Colab – EfficientNet-B3 embeddings

- **Motivación:** usar EfficientNet-B3 solo como extractor de embeddings, sin fine-tuning completo.  
- **Resultados:**  
  - AUC (Test): 0.704  
  - PR-AUC: 0.623  
  - Recall: 0.90  

**Reflexión:**  
Como extractor simple ya supera ResNet18 calibrado, confirmando potencia de EfficientNet.  

---

### P7: MRI Colab – EfficientNet-B3 fine-tuning

- **Motivación:** pasar a fine-tuning completo de EfficientNet-B3.  
- **Resultados:**  
  - AUC (Test): 0.876  
  - PR-AUC: 0.762  
  - Acc: 0.745 | Recall: 1.0 | Precisión: 0.625  

**Reflexión:**  
Uno de los mejores backbones en imagen pura. Supone el nuevo baseline de referencia.  

---

### P9: MRI Colab – EfficientNet-B3 stable

- **Motivación:** estabilizar entrenamientos previos de EfficientNet-B3.  
- **Resultados:**  
  - AUC (Test): 0.740  
  - PR-AUC: 0.630  
  - Recall más bajo que en P7.  

**Incidencias:**  
- Saturación de logits detectada.  
- Variabilidad alta entre seeds.  

**Reflexión:**  
Confirma que la estabilidad no siempre se traduce en mejor rendimiento.  

---

### P10: MRI Colab – EfficientNet-B3 stable + calibración

- **Motivación:** aplicar calibración explícita para corregir sobreconfianza.  
- **Método:** Platt scaling, isotonic regression y temperature scaling.  
- **Resultados:**  
  - AUC (Test): 0.546–0.583  
  - PR-AUC: 0.50–0.53  
  - Recall: 1.0 pero precisión baja (~0.47–0.49)  

**Reflexión:**  
La calibración ayudó a controlar la sobreconfianza pero sacrificó precisión.  

---

### P10-ext: Extensiones y ensembles

- **Motivación:** explotar estrategias de **ensembles y stacking** con EfficientNet-B3.  
- **Estrategias:**  
  - Seed ensembles (mean, trimmed, top7)  
  - Random forest sobre features derivadas  
  - Stacking logístico  
- **Resultados destacados:**  
  - Ensemble (mean+trimmed20+top7+p2): Test AUC ~0.75  
  - Stacking LR sobre seeds: Test AUC ~0.75  

**Reflexión:**  
El ensemble aporta mejoras modestas pero consistentes. Se consolida como estrategia útil.  

---

### P11: Backbones alternativos

- **Motivación:** comprobar si otros backbones de visión podían superar a EfficientNet-B3.  
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
Ningún backbone supera claramente a EfficientNet-B3.  
La vía lógica pasa a ser **ensembles de backbones**.  

---

### P13: **COGNITIVA-AI-OASIS2-P13 (EffNet-B3 base en OASIS-2)**  
- Procesamiento de **367 scans OASIS-2** → 150 pacientes con labels clínicos.  
- **Slices:** 20 cortes axiales equiespaciados, evitando extremos, normalizados (z-score + CLAHE opcional).  
- **Máscara cerebral:** segmentación FSL o fallback con Otsu.  
- **Una visita por paciente** → 150 pacientes (105 train, 22 val, 23 test).  

**Resultados:** recall alto en cohortes pequeñas, pero dataset limitado → riesgo de sobreajuste.  

---

### P14: **COGNITIVA-AI-OASIS2-P14 (EffNet-B3 balanceado, Colab SSD)**  
- Copia de las 7340 slices a **SSD local de Colab** para reducir la latencia de E/S.  
- Entrenamiento con **class weights** para balancear clases.  
- Integración en catálogo de backbones (p11).  

**Resultados:**  
- [VAL] AUC≈0.88 | Acc≈0.86 | Recall≈0.82  
- [TEST] AUC≈0.71 | Acc≈0.70 | Recall=1.0  

---

### P15: **COGNITIVA-AI-OASIS2-P15 (Consolidación y comparación)**  
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

### P16: **COGNITIVA-AI-OASIS2-P16 (Refinamiento de Ensembles)**
- Se construyeron **features patient-level** a partir del catálogo de backbones (`oas2_effb3`, `oas2_effb3_p14`, SwinTiny, ConvNeXt, etc.).  
- Manejo explícito de **NaNs** (descartar features con >40% de missing, imputación/flags en LR, NaN nativos en HistGB).  
- Ensayos con **Logistic Regression, HistGradientBoosting y blending**.  
- Resultados:  
  - VAL: AUC≈0.95 (blend), recall≈1.0 en OAS1, estable en OAS2.  
  - TEST: AUC≈0.69, recall≈0.78 (blend), mejor que cada backbone aislado.  
- Conclusión: ensembles permiten mejorar estabilidad y recall, confirmando el valor de la integración multimodelo.

---

### P17: **COGNITIVA-AI-Ensemble Calibration (Stacking + Platt Scaling)**  
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

7️⃣ COGNITIVA-AI-ENSEMBLE-ADVANCED (p18, stacking multicapa)

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

### P19: **COGNITIVA-AI-OASIS2-P19 (Meta-Ensemble apilado)**  

**Objetivo:** consolidar las señales de múltiples backbones (p11/p14/p16/p18) con un stacking de segundo nivel.  

- **Base learners:** LR, HistGB, GB, RF, LGBM, XGB entrenados con OOF sin fuga, usando features por-paciente derivados (mean / trimmed / top-k / p2).  
- **Meta-learner:** XGBoost entrenado sobre los OOF; inferencia en TEST con predicciones de base learners.  
- **Manejo de NaN:** exclusión de columnas con NaN>40% + imputación simple donde procede para modelos que lo requieren.  

**Métricas:**  
- VAL: AUC≈0.964, PRAUC≈0.966, Acc≈0.913, F1≈0.897, Brier≈0.071.  
- TEST: AUC≈0.729, PRAUC≈0.688, Acc≈0.714, Prec≈0.773, Recall≈0.531, F1≈0.630, Brier≈0.226.  

➡️ **Conclusión:** el meta-ensemble eleva la performance en validación, pero el recall en TEST sugiere ajustar calibración/umbrales y atender shift OAS1/OAS2. Se programará p20 para calibración fina y umbrales por cohorte.

---

### P20: **COGNITIVA-AI-OASIS2-P20 (Meta-calibración y umbrales por cohorte)**

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

### P21: **COGNITIVA-AI-OASIS2-P21 (Meta-refine)**
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

### P22: **COGNITIVA-AI-OASIS2-P22 (Meta-Ablation con calibración avanzada)**

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

## Comparativa global de resultados

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

## Desafíos principales

1. **Pequeño tamaño de dataset**:  
   - Solo ~47 pacientes en test.  
   - Variabilidad extrema en métricas según fold.  
   - Riesgo de overfitting altísimo.  

2. **Saturación de logits**:  
   - En P9 y P10, los logits alcanzaban valores >500k, obligando a normalización y calibración.  

3. **Problemas de montaje de Google Drive en Colab**:  
   - Errores de “Mountpoint must not already contain files” tras semanas sin reinicio.  
   - Necesidad de reiniciar entorno completo.  

4. **Dispersión de ficheros de predicción**:  
   - Algunos outputs generados como `*_png_preds`, otros como `*_slice_preds`.  
   - Diferencias en columnas (`y_score`, `sigmoid(logit)`, `pred`).  

5. **Gestión de ensembles**:  
   - Decidir entre averaging, stacking, random search de pesos.  
   - Validación compleja con tan pocos pacientes.  

---

## Lecciones aprendidas

- **Los datos clínicos son extremadamente informativos** en OASIS-2.  
- **EfficientNet-B3** sigue siendo el backbone más consistente en MRI.  
- **La calibración es necesaria** pero puede sacrificar precisión.  
- **Los ensembles ayudan modestamente**, pero su efecto depende de la diversidad real de los modelos.  
- **La organización de outputs es crítica**: nombres consistentes ahorran horas de debugging.  
- **El reinicio periódico de Colab** evita errores de montaje y rutas fantasmas.  

---

## Próximos pasos

1. **Consolidar ensembles de backbones**:  
   - Probar combinaciones más ricas (ResNet+EffNet+Swin).  
   - Usar stacking con regularización fuerte.  

2. **Explorar multimodal**:  
   - Fusionar clínico + MRI.  
   - Comparar si mejora sobre clínico solo.  

3. **Validación externa**:  
   - Usar datasets adicionales (ADNI, etc.) para comprobar generalización.  

4. **Optimización final**:  
   - Revisar hiperparámetros con Bayesian Optimization.  
   - Estudiar interpretabilidad (Grad-CAM, SHAP).  

---
Actualizado: 07/09/2025 15:43