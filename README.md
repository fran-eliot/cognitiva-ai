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
   - EfficientNet-B3 entrenado en OASIS-2 con criterio de **una visita por paciente**  
     (total: 150 pacientes → 105 train, 22 val, 23 test).  
   - Generación de slices: 20 cortes axiales equiespaciados evitando extremos,  
     normalizados con z-score y CLAHE opcional.  
   - Dataset enriquecido con labels clínicos (Control/Dementia/Converted).  

   **Resultados (VAL/TEST)**:  
   - Recall alto en cohortes pequeñas, AUC estable, pero dataset limitado → riesgo de overfitting.  

### P14: **COGNITIVA-AI-OASIS2-P14 (EffNet-B3 balanceado, Colab SSD)**  
   - Entrenamiento en Colab GPU con imágenes copiadas a SSD local para evitar  
     latencias de E/S desde Google Drive.  
   - Uso de **class weights** para balancear clases.  
   - Integración en catálogo de backbones (p11).  

   **Resultados (VAL/TEST):**  
   - [VAL] AUC≈0.88 | Acc≈0.86 | Recall≈0.82  
   - [TEST] AUC≈0.71 | Acc≈0.70 | Recall=1.0  

➡️ P14 se consolida como **clasificador sensible** (cribado clínico), mientras que P13 sirvió de exploración base para OASIS-2.

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
Actualizado: 05/09/2025 21:54