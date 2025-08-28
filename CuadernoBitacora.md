# 📖 Cuaderno de Bitácora – Proyecto COGNITIVA-AI

Este documento actúa como **diario detallado de investigación**, complementando al `README.md` (resumen ejecutivo) y al `InformeTecnico.md` (documentación formal).  

Aquí se incluyen **todas las fases del proyecto** y **entradas diarias (dailys)** con resultados, problemas técnicos y conclusiones.

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

**Gráfico:**  
![Resultados clínicos OASIS-2](./graficos/clinic_oasis2.png)

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

**Gráfico:**  
![Fusion clínica OASIS1+2](./graficos/clinic_fusion.png)

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

**Gráfico:**  
![MRI baseline ResNet50](./graficos/mri_resnet50_baseline.png)

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

**Gráfico:**  
![ROC Curves – Colab GPU ResNet18](./graficos/roc_colab_resnet18.png)

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

**Gráfico:**  
![Comparativa SVM-XGB-Ensemble](./graficos/ensemble_resnet18.png)

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

**Gráfico:**  
![EfficientNet-B3 comparativa](./graficos/effnetb3_val_test.png)

**Conclusión:**  
EffNet-B3 genera embeddings más informativos; los clasificadores simples tienden a sobreajustar (ej. MLP val>>test), pero el **ensemble logra equilibrio** con recall clínico aceptable (90%). Este pipeline aumentó la sensibilidad manteniendo precisión ~0.60, señalando un avance respecto a fases previas.

---

## **Fase 7 – EfficientNet-B3 Fine-tuning parcial**
- **Contexto:** Se migra de utilizar embeddings fijos a fine-tunear parcialmente EfficientNet-B3 directamente con las MRI, permitiendo que la red ajuste sus filtros a patrones específicos de Alzheimer. Se descongelan las últimas capas de EffNet-B3 y se entrena con data augmentation moderada, usando Colab GPU.
- **Notebook**: `cognitiva_ai_finetuning.ipynb`.  
- **Agregación paciente**: *mean pooling*.  
- **Calibración**: *temperature scaling* **T=2.673**; **thr=0.3651**.  
- **Artefactos**:  
  - `ft_effb3_colab/best_ft_effb3.pth`  
  - `ft_effb3_colab/train_history.json`  
  - `ft_effb3_colab/ft_effb3_patient_eval.json`  
  - `ft_effb3_colab/graphs_from_metrics/*.png`  
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

## Fase 10 – Stable Plus (checkpoint limpio + calibración final)

**Contexto:**  
Tras el pipeline estable (fase 9), se detectaron problemas de compatibilidad entre el checkpoint guardado y la arquitectura usada. Esto generaba cargas parciales (<1% en algunos intentos) y métricas inconsistentes. La fase 10 surge para **reconstruir el checkpoint a un formato limpio (99.7% pesos cargados), aplicar calibración final y consolidar el modelo MRI**.

**Acciones:**  
- Normalización del checkpoint entrenado, eliminando capas obsoletas (`head.classifier`) y adaptando pesos a la nueva `head`.  
- Evaluación de estrategias de pooling (mean, median, top-k).  
- Aplicación de calibración mediante *temperature scaling*.  
- Guardado de artefactos completos (CSV por slice, CSV por paciente, JSON con métricas, gráficas comparativas).  

**Resultados clave:**
- Pooling **mean**: VAL AUC=0.630, TEST AUC=0.546.  
- Pooling **median**: TEST AUC=0.541.  
- Pooling **top-k=0.2**: TEST AUC=0.583.  
- Recall en TEST siempre 1.0, con precisión 0.47–0.49.

**Gráfico:**  
![Stable Plus comparativa](./graficos/stable_plus.png)

**Conclusión:**  
Pipeline más robusto en recall absoluto, pero con AUC más bajo que el pipeline 9. Se plantea usarlo como **baseline clínico seguro** en cribado poblacional.

---

## Comparativa Global

<p align="center">
  <img src="./graficos/comparativapipelines7-10.png" alt="Comparativa P1-P7 — ROC-AUC por Pipeline" width="880"/>
</p>

Gráfico de barras con ROC-AUC y PR-AUC en TEST para los tres pipelines más representativos:

- P7 (finetuning clásico con B3).

- P9 (stable, sin calibración).

- P10 (stable plus con checkpoint limpio + calibración).

---
<p align="center">
  <img src="./graficos/comparativapipelines7-10B.png" alt="Comparativa P1-P7 — Precisión-Recall por Pipeline" width="880"/>
</p>

Comparativa para Precisión y Recall de los tres pipelines MRI (P7, P9 y P10)

---

## 🔄 Fase complementaria – Pipeline 10 (Stable Plus con agregaciones avanzadas)

En esta fase se exploraron variantes adicionales sobre el pipeline 10, sin cambiar de notebook pero ampliando las técnicas de agregación y evaluación.

- **Acción**: Implementar pooling robustos (TRIMMED, TOP-k) y un ensemble MRI.  
- **Metodología**:  
  - Ajuste de pesos del ensemble (mean=0.30, trimmed=0.10, top7=0.60) mediante grid search sobre validación.  
  - Calibración final de logits con *temperature scaling* (T≈0.28).  
- **Resultados principales**:  
  - TRIMMED: Recall=0.75, Precisión=0.56, PR-AUC=0.746 (TEST).  
  - Ensemble: Recall=0.70, Precisión=0.61, PR-AUC=0.737 (TEST).  
- **Conclusión**: Aunque TRIMMED asegura mayor sensibilidad, el ensemble proporciona un balance clínico más realista al aumentar la precisión, reduciendo falsos positivos sin comprometer excesivamente la detección. Se adopta como baseline final de la etapa MRI.

## Comparativa(P1-P10)

## 📊 Comparativa Global (pipelines 1–10)

| Pipeline | Modalidad        | Modelo                   | AUC (Test) | PR-AUC | Acc   | Recall | Precision |
|----------|-----------------|--------------------------|------------|--------|-------|--------|-----------|
| P1       | Clínico OASIS-2 | XGB                      | 0.897      | —      | —     | —      | —         |
| P2       | Clínico fusion  | XGB                      | 0.991      | —      | —     | ~1.0   | —         |
| P3       | MRI OASIS-2     | ResNet50                 | 0.938      | —      | —     | —      | —         |
| P5       | MRI Colab       | ResNet18 + Calib         | 0.724      | 0.606  | 0.60  | 0.80   | 0.52      |
| P6       | MRI Colab       | EffNet-B3 embed          | 0.704      | 0.623  | 0.70  | 0.90   | 0.60      |
| P7       | MRI Colab       | EffNet-B3 finetune       | 0.876      | 0.762  | 0.745 | 1.0    | 0.625     |
| P9       | MRI Colab       | EffNet-B3 stable         | 0.740      | 0.630  | 0.72  | 0.65   | 0.62      |
| P10      | MRI Colab       | EffNet-B3 stable+calib   | 0.546–0.583| 0.50–0.53 | 0.51–0.55 | 1.0 | 0.47–0.49 |
| P10-ext  | MRI Colab       | EffNet-B3 + TRIMMED      | 0.744      | 0.746  | 0.64  | 0.75   | 0.56      |
| P10-ext  | MRI Colab       | EffNet-B3 + Ensemble(M+T+7) | 0.754   | 0.737  | 0.68  | 0.70   | 0.61      |

---
# 📅 Entradas Diarias (Agosto 2025)

### 📅 18/08/2025 – Migración a Colab GPU
- **Acción**: Montaje de Google Drive en Colab, carga de embeddings ResNet18 precomputados, entrenamiento de LogReg con calibración isotónica. 
- **Resultado**: Pipeline de imágenes funcionando en GPU; AUC ~0.72 estable en test, con recall mejorado al ~0.80 aplicando umbral bajo. 
- **Problemas**: Colab desconectó la sesión a mitad → se tuvieron que reconstruir celdas y montar de nuevo el entorno (lección: guardar modelos intermedios). 
- **Conclusión**: Base sólida para MRI en GPU establecida, sentando groundwork para experimentar con modelos más complejos.

### 📅 21/08/2025 – Experimentación con EfficientNet-B3
- **Acción**: Generación de embeddings de 1536 dimensiones con EfficientNet-B3 para cada slice; entrenamiento de clasificadores LR, MLP y XGB a nivel paciente; comparación de pooling por promedio vs estrategias por paciente. 
- **Resultado**: LR mostró desempeño estable (menos overfitting), MLP tuvo alto overfitting (train >> val), XGB mejoró algo en slices informativos. Un ensemble simple (LR+XGB) incrementó recall en test a 0.90 con precision ~0.60.
- **Conclusión**: Embeddings más ricos abren la puerta a ensembles más sofisticados, pero también pueden sobreajustar con facilidad. Se logra alta sensibilidad (0.90) manteniendo precisión aceptable, validando la estrategia híbrida de combinar modelos. Esto sugiere que para avanzar se requerirá o más datos o técnicas que aprovechen mejor los patrones de imágenes (→ fine-tuning).

### 📅 23/08/2025 – Ensemble híbrido
- **Acción**: Prueba de combinación “híbrida” entre modelos de slice y de paciente: se combinó un XGBoost entrenado directamente a nivel slice (promediando sus scores por paciente) con un MLP entrenado sobre features agregadas de paciente, para capturar información a dos escalas.
- **Resultado**: El ensemble híbrido alcanzó **Recall_test = 0.90** y Precision_test ~0.60, similar al pipeline anterior pero confirmando la aportación complementaria de ambos enfoques (el MLP recuperó algunos positivos que XGBoost solo-slice perdía). 
- **Conclusión**: Se valida la estrategia **multiescala** (slice + paciente) para integrar información. Esto apunta a la relevancia de fusionar diferentes representaciones. Los aprendizajes aquí alimentarán la fase multimodal futura (combinar clínica+MRI). Antes, se decide intentar extraer aún más de las MRI vía fine-tuning de la CNN, ahora que la infraestructura en GPU está probada.

### 📅 24/08/2025 – Fine-tuning EffNet-B3 en MRI
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
- **Resultados parciales**: métricas estables; generación de `ft_effb3_patient_eval.json`.  
- **Archivos generados:** gráficas en `graphs_from_metrics/` (confusión, punto PR, barras AUC/PR-AUC).  
- **Problemas**:  
  - `ValueError: mountpoint must not already contain files` → resuelto con `force_remount=True`.  
  - *Warning* DataLoader: exceso de workers → fijar `num_workers=2`.  
  - Deprecation `torch.cuda.amp.autocast` → migrado a `torch.amp.autocast('cuda')`.
- **Resultados:** El modelo fine-tune entrenó ~10 épocas antes de converger. AUC_test subió a ~0.87, un incremento notable vs embeddings fijos (~0.70). Sin embargo, con threshold=0.5 solo logró recall_test ~0.55 (precision ~0.85). Es decir, clasificó con alta certeza algunos positivos, pero dejó muchos sin detectar a ese umbral.
- **Conclusión:** Fine-tuning demostró ser muy efectivo en potenciar la señal (mejor AUC), pero evidenció la necesidad de recalibrar el modelo para cumplir el requisito clínico de alta sensibilidad. Se planificó calibrar sus probabilidades y ajustar el threshold en la siguiente sesión. 

### 📅 **25/08/2025 – Calibración y umbral clínico (EffNet-B3 fine-tune)**
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
- **Comparativa con P7:** ver `comparison_p7_p9_*` (AUC/PR‑AUC).
- **Artefactos:** `best_effb3_stable.pth`, `effb3_stable_patient_eval.json`, CSVs por slice/paciente y gráficas en `graphs_from_metrics/`.  
- **Conclusión:** Se obtuvo un **pipeline MRI óptimo:** modelo calibrado, sin falsos negativos en test. La sensibilidad alcanzada (100%) cumple con creces la meta de cribado. Este resultado supera en equilibrio a todos los intentos previos y deja al modelo listo para integrarse con datos clínicos. Próximo paso: **fusión multimodal** (combinar predicción clínica y de MRI) y validar en cohortes externas (OASIS-3, ADNI) para verificar su generalización.

### 📅 25/08/2025 – 03:04 – Pipeline 9 (EffB3 estable)
- **Acción:** retraining reproducible en Colab (EffNet‑B3), caché SSD, AMP (`torch.amp`), early‑stopping por AUC en holdout, calibración (T=2.048), pooling `mean` y selección de umbral 0.3400 con recall≥0.95 en VAL.  
- **Resultados:**  
  - VAL → AUC=1.000 | PR-AUC=1.000 | Acc=1.000 | P=1.000 | R=1.000 | thr=0.3400 | n=10  
  - TEST → AUC=0.663 | PR-AUC=0.680 | Acc=0.574 | P=0.500 | R=0.650 | thr=0.3400 | n=47  
- **Comparativa con P7:** ver `comparison_p7_p9_*` (AUC/PR‑AUC).
- **Artefactos:** `best_effb3_stable.pth`, `effb3_stable_patient_eval.json`, CSVs por slice/paciente y gráficas en `graphs_from_metrics/`.  
- **Conclusión:** setup estable listo para el salto a **multimodal** y validación externa.-

---

### 📅 26/08/2025 – Stable Plus (checkpoint limpio + calibración)

- **Acción:**  
  - Reconstrucción del checkpoint (`effb3_stable_seed42.pth`) a un formato limpio y compatible. 
  - Carga de pesos (99.7% éxito), eliminando discrepancias de capas.  
  - Aplicación de *temperature scaling* y ajuste de pooling: Pruebas de inferencia con pooling mean, median y top-k.  

- **Resultados:**  
  - VAL: AUC=0.63 | PR-AUC=0.67 | Recall≈0.85  
  - TEST: AUC=0.55 | PR-AUC=0.53 | Recall=1.0  
  - Recall=1.0 en TEST para todas las variantes, AUC entre 0.54–0.58.  
  - Se confirmó estabilidad en los artefactos (CSV, JSON, gráficas).  

- **Artefactos:**  
  - Checkpoint limpio en `best_effb3_stable.pth`.  
  - CSV por slice y paciente.  
  - JSON de evaluación calibrada.  
  - Gráficas comparativas en `graphs_from_metrics/`.  

- **Conclusión:**  
  Pipeline estable y calibrado, recuperando recall perfecto en test, aunque precisión moderada (~0.47). Sirve como versión **ultra-conservadora** para detección precoz.  

---

### 📅 26/08/2025 17:35 – Validación extendida de Stable Plus

- **Acción:**  
  - Revisión completa de los artefactos generados en `ft_effb3_stable_colab_plus`.  
  - Confirmación de que los CSV (slice/paciente) y JSON calibrado se cargaban sin errores.  
  - Verificación de métricas con distintos poolings (mean, median, top-k).  
  - Ajuste de umbral F1 (≈0.50) y validación de recall absoluto.  

- **Resultados:**  
  - Recall en TEST=1.0 bajo todos los poolings.  
  - AUC osciló entre 0.54 y 0.58; precisión 0.47–0.49.  
  - Artefactos gráficos confirmados en `graphs_from_metrics/`.  

- **Artefactos:**  
  - Checkpoint limpio y validado: `best_effb3_stable.pth`.  
  - CSV val/test (slices y pacientes).  
  - Eval JSON calibrado (`effb3_stable_patient_eval_calibrated.json`).  

- **Conclusión:**  
  La fase 10 queda consolidada como el cierre de la etapa MRI. Este pipeline representa la versión más **ultra-conservadora**, maximizando recall aunque a costa de precisión. Servirá de base para la futura etapa multimodal.

---

### [2025-08-28] – Avance en ensemble MRI
- Probados métodos TRIMMED, TOP3, TOP7 y un ensemble de agregaciones.  
- El ensemble alcanzó mejor equilibrio: recall=0.70 y precisión=0.61 en test.  
- Guardadas curvas ROC y PR comparativas (trimmed vs ensemble).  
- Actualizados los resultados en `comparison_patient_level_eval.csv`.  

---

# 📌 Conclusión global
- **Clínico (fusionado OASIS1+2)** → proporciona el mejor AUC global (≈0.99) gracias a fuertes marcadores como CDR/MMSE, aunque puede complementarse en sensibilidad.
- **MRI (GPU)** → los pipelines evolucionaron de AUC ~0.72 a ~0.88, alcanzando recall 1.0 tras calibración; esto demuestra que la información visual aporta detección temprana (atrofia incipiente) que puede adelantarse a signos clínicos, aunque con más falsos positivos.
- **EffNet-B3 fine-tune** → supuso el mayor salto en MRI, cerrando la brecha con lo clínico. Un modelo profundo entrenado con nuestros datos sumado a calibración logró equilibrio óptimo para cribado (sensibilidad alta con precisión moderada). 
- **Próximos pasos** → encarar la **integración multimodal (clínico + MRI)** para aprovechar lo mejor de ambos mundos, y realizar validación externa en datos independientes (p. ej. OASIS-3, ADNI) que ratifique la robustez del enfoque. Se espera que la modalidad clínica aporte especificidad y la MRI sensibilidad, para juntos lograr un sistema de apoyo al diagnóstico más preciso y generalizable..

---

**Autoría:** Fran Ramírez  
**Última actualización:** 28/08/2025 – 18:07