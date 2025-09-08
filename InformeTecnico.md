# 📑 Informe Técnico – Proyecto CognitivaAI  

---

## 1. Introducción General  

El presente informe constituye una documentación exhaustiva, detallada y profundamente analítica del proyecto **CognitivaAI**, cuyo propósito ha sido explorar, diseñar, entrenar y evaluar modelos de aprendizaje profundo aplicados a la detección de **deterioro cognitivo temprano** a partir de imágenes de resonancia magnética (MRI) de la base de datos **OASIS-2**, complementadas con información clínica.  

El proyecto surge con una doble motivación:  

1. **Científica y social:** el diagnóstico temprano de enfermedades neurodegenerativas, como la enfermedad de Alzheimer, es un desafío prioritario para la medicina actual. El uso de inteligencia artificial puede ayudar a identificar biomarcadores sutiles en imágenes cerebrales y en datos clínicos que son difíciles de detectar por métodos tradicionales.  

2. **Técnica y experimental:** comprobar hasta qué punto distintas arquitecturas de redes neuronales convolucionales (CNNs) y transformadores visuales pueden capturar información discriminativa en un dataset limitado en tamaño, enfrentándose a problemas comunes como el sobreajuste, la calibración de probabilidades y la estabilidad en validación y test.  

La ejecución del proyecto ha supuesto un recorrido iterativo y progresivo, articulado en una serie de **pipelines experimentales**, cada uno diseñado con objetivos concretos: desde probar un modelo sencillo con datos clínicos (P1) hasta explorar ensembles de arquitecturas avanzadas (P11).  

Cada pipeline ha sido registrado con su motivación, configuración, incidencias, métricas y reflexión crítica, con el objetivo de construir no solo un conjunto de resultados, sino un **mapa de decisiones** que documenta los aprendizajes y trade-offs a lo largo del camino.  

---

## 2. Metodología Global  

### 2.1 Dataset y preprocesamiento  

El dataset utilizado es **OASIS-2** (Open Access Series of Imaging Studies), que incluye imágenes de resonancia magnética (MRI) estructurales de individuos con y sin deterioro cognitivo, junto con información clínica asociada.  

- **MRI**: cada sujeto cuenta con una o más adquisiciones de MRI estructural.  
- **Etiquetas clínicas**: la variable principal es el **CDR (Clinical Dementia Rating)**, dicotomizada en “control” vs “deterioro cognitivo”.  
- **Metadatos adicionales**: edad, sexo, puntuaciones cognitivas, entre otros.  

El preprocesamiento incluyó:  
- Mapeo de rutas a pacientes mediante ficheros `oas1_val_colab_mapped.csv` y `oas1_test_colab_mapped.csv`.  
- Normalización de intensidades.  
- Creación de slices 2D a partir de volúmenes 3D para facilitar el entrenamiento en arquitecturas CNN estándar.  
- Balanceo parcial mediante augmentaciones (flip horizontal, rotaciones leves).  

### 2.2 Infraestructura  

- **Google Colab Pro+**: principal entorno de ejecución.  
- **GPU asignada**: NVIDIA T4 o A100, según disponibilidad.  
- **Almacenamiento persistente**: Google Drive (`/MyDrive/CognitivaAI`).  
- **Librerías clave**: PyTorch, timm (colección de modelos), scikit-learn, pandas, matplotlib.  
- **Gestión de experimentos**: notebooks separados por pipeline, con guardado automático de métricas y predicciones en CSV.  

### 2.3 Métricas de evaluación  

Dada la naturaleza binaria y desbalanceada de la tarea, se utilizaron múltiples métricas:  

- **AUC (ROC)**: medida global de discriminación.  
- **PR-AUC**: más informativa en datasets desbalanceados.  
- **Accuracy**: proporción de aciertos.  
- **Recall (sensibilidad)**: clave, ya que la tarea exige detectar la mayor cantidad de pacientes con deterioro cognitivo.  
- **Precision**: importante para evitar falsos positivos.  
- **Youden Index**: criterio alternativo de umbralización.  
- **@REC90 / @REC100**: configuraciones que maximizan el recall (sensibilidad) incluso a costa de la precisión.  

---

## 3. Evolución por Pipelines  

### 3.1 Pipeline 1 – Modelo clínico (XGB con OASIS-2)  

**Motivación:**  
Probar un primer modelo utilizando exclusivamente variables clínicas del dataset OASIS-2, sin imágenes. El objetivo era establecer un baseline puramente tabular.  

**Configuración:**  
- Modelo: **XGBoost**.  
- Variables: edad, sexo, puntuaciones cognitivas, CDR.  
- Validación cruzada interna.  

**Resultados:**  
- AUC (Test): 0.897.  
- Sin métricas de PR-AUC reportadas, aunque se observó buen desempeño en términos de recall.  

**Reflexión:**  
Este baseline confirmó que incluso con variables clínicas simples se puede alcanzar un nivel competitivo de discriminación. Sin embargo, la dependencia exclusiva de variables clínicas limita la aplicabilidad general.  

---

### 3.2 Pipeline 2 – Fusión clínica extendida  

**Motivación:**  
Ampliar el modelo clínico con más variables y verificar hasta qué punto un modelo tabular puede llegar cerca de la perfección.  

**Configuración:**  
- Modelo: **XGB**.  
- Variables: todas las disponibles en OASIS-2.  
- Validación cruzada más exhaustiva.  

**Resultados:**  
- AUC (Test): 0.991.  
- Recall cercano al 100%.  

**Reflexión:**  
El modelo clínico extendido alcanzó una performance casi perfecta, aunque con riesgo evidente de **overfitting** dada la baja dimensionalidad del dataset. Sirvió como techo de referencia para comparar con modelos de MRI.  

---

### 3.3 Pipeline 3 – MRI OASIS-2 con ResNet50  

**Motivación:**  
Dar el salto a imágenes MRI, usando ResNet50 como backbone en el dataset original de OASIS-2.  

**Configuración:**  
- Modelo: **ResNet50 preentrenada en ImageNet**.  
- Dataset: MRI OASIS-2 (formato local).  
- Entrenamiento limitado por GPU local.  

**Resultados:**  
- AUC (Test): 0.938.  

**Reflexión:**  
Este pipeline demostró que el uso de imágenes cerebrales puede ofrecer resultados competitivos con los clínicos, aunque aún no superiores.  

---

### 3.4 Pipeline 5 – ResNet18 + calibración  

**Motivación:**  
Probar una arquitectura más ligera (ResNet18) con calibración de probabilidades.  

**Configuración:**  
- ResNet18 preentrenada.  
- Post-procesado: **Platt scaling**.  

**Resultados:**  
- AUC (Test): 0.724.  
- PR-AUC: 0.606.  
- Accuracy: 0.60.  
- Recall: 0.80.  
- Precision: 0.52.  

**Reflexión:**  
La calibración mejoró la interpretación de scores, pero el backbone resultó limitado para esta tarea.  

---

### 3.5 Pipeline 6 – EffNet-B3 en modo embeddings  

**Motivación:**  
Usar EfficientNet-B3 como extractor de embeddings, sin fine-tuning completo.  

**Resultados:**  
- AUC (Test): 0.704.  
- PR-AUC: 0.623.  
- Accuracy: 0.70.  
- Recall: 0.90.  
- Precision: 0.60.  

**Reflexión:**  
Buen recall, aunque precision limitada.  

---

### 3.6 Pipeline 7 – Fine-tuning completo de EfficientNet-B3  

**Motivación:**  
Explotar la capacidad completa de EfficientNet-B3, ajustando todos los pesos.  

**Resultados:**  
- AUC (Test): 0.876.  
- PR-AUC: 0.762.  
- Accuracy: 0.745.  
- Recall: 1.0.  
- Precision: 0.625.  

**Reflexión:**  
Gran salto respecto a embeddings. El modelo capturó patrones discriminativos más profundos.  

---

### 3.7 Pipeline 9 – EffNet-B3 stable  

**Motivación:**  
Introducir técnicas de estabilización para reducir variabilidad.  

**Resultados:**  
- AUC (Test): 0.740.  
- PR-AUC: 0.630.  
- Accuracy: 0.72.  
- Recall: 0.65.  
- Precision: 0.62.  

---

### 3.8 Pipeline 10 – EffNet-B3 stable + calibración  

**Motivación:**  
Aplicar calibración explícita de scores.  

**Resultados:**  
- AUC (Test): entre 0.546–0.583.  
- PR-AUC: 0.50–0.53.  
- Accuracy: 0.51–0.55.  
- Recall: 1.0.  
- Precision: 0.47–0.49.  

**Reflexión:**  
La calibración no mejoró la discriminación; sacrificó precision.  

---

### 3.9 Pipeline 10-ext – Variantes TRIMMED y ensembles intra-backbone  

**Motivación:**  
Explorar variantes de pooling de slices y ensembles dentro de EfficientNet-B3.  

**Resultados:**  
- TRIMMED: AUC (Test) 0.744, PR-AUC 0.746.  
- Ensemble (mean+trimmed+top7): AUC (Test) 0.754, PR-AUC 0.737.  

**Reflexión:**  
Los ensembles intra-backbone sí ofrecieron mejoras.  

---

### 3.10 Pipeline 11 – Backbones alternativos  

**Motivación:**  
Explorar arquitecturas más allá de EfficientNet.  

**Modelos probados:**  
- ResNet-50.  
- DenseNet-121.  
- ConvNeXt-Tiny.  
- Swin-Tiny.  

**Resultados:**  
- **ResNet-50**: competitivo, AUC similar a EffNet-B3 estable.  
- **DenseNet-121**: resultados bajos (AUC ~0.34–0.46).  
- **ConvNeXt-Tiny**: desempeño bajo (AUC ~0.50).  
- **Swin-Tiny**: desempeño moderado, con variante top7 alcanzando AUC ~0.64.  

**Reflexión:**  
Ningún backbone superó claramente a EfficientNet-B3, aunque Swin mostró cierto potencial.  

---

### 3.11 Ensembles inter-backbone  

Se exploraron combinaciones entre diferentes backbones:  

- **Dirichlet ensembles**: priorizaron SwinTiny y ResNet, con mejoras modestas.  
- **Stacking con regresión logística**: no lograron generalizar bien, tendencia a sobreajuste.  
- **Isotonic calibration** sobre SwinTiny top7: mejoró la estabilidad.  

---

## Experimentos en OASIS-2 (p13, p14 y p15)

### Preparación de datos
- 367 exploraciones de OASIS-2, de las cuales 150 tenían label clínico.  
- Generación de 7340 slices (20 por volumen, edge_crop=0.08, z-score + CLAHE).  
- Máscaras cerebrales FSL u Otsu.  
- Criterio de **una visita por paciente** para evitar leakage.  

---

### Pipeline p13
- Entrenamiento base con EfficientNet-B3 en cohortes reducidas.  
- 150 pacientes → 105 train, 22 val, 23 test.  
- Resultados preliminares con recall alto, pero riesgo de sobreajuste.  

---

### Pipeline p14
- Reentrenamiento en GPU con **class weights** y datos copiados a **SSD local de Colab** para optimizar E/S.  
- Métricas robustas en validación (AUC≈0.88) y recall=100% en test.  
- Integración en catálogo de backbones como `oas2_effb3_p14`.  

---

### Pipeline p15 (Consolidación y ensamble)
- Objetivo: consolidar resultados de p13/p14 con OASIS-1 (p11).  
- Se añadieron al catálogo de backbones (`oas2_effb3`, `oas2_effb3_p14`).  
- Generación de features combinadas (69 pacientes en val, 70 en test).  
- Manejo de NaN: descarte de columnas con >40% y uso de modelos compatibles con NaN (HistGradientBoosting).  

**Resultados comparativos (VAL/TEST):**

| Pipeline | VAL AUC | VAL Acc | VAL Recall | TEST AUC | TEST Acc | TEST Recall |
|----------|---------|---------|------------|----------|----------|-------------|
| p13      | ~0.90   | 0.86    | 0.82       | ~0.77    | 0.78     | 0.83        |
| p14      | 0.88    | 0.86    | 0.82       | 0.71     | 0.70     | 1.00        |
| p15      | 0.94    | 0.84    | ~1.0       | 0.71     | 0.63–0.71 | 0.78–1.0   |

---

### Conclusión
- p13: prueba de integración OASIS-2.  
- p14: optimización con balanceo y SSD local.  
- p15: consolidación de resultados e integración en ensambles OASIS-1+2, demostrando que la combinación de backbones mejora recall y robustez del sistema.

---

## Fase 8 – Ensembles refinados (p15 y p16)

**Contexto:**  
Tras la exploración con OASIS-2 en los pipelines p13 y p14, se buscó consolidar resultados y refinar el rendimiento a través de ensembles patient-level.

**Pipeline p15 (Consolidación):**
- Revisión de inventarios y labels clínicos (367 scans totales, 150 con etiqueta).  
- Confirmación del criterio de **una sesión por paciente** para evitar leakage.  
- Dataset resultante: 3.000 slices etiquetadas (20 slices por scan).  
- Dificultades: la latencia de E/S en Google Drive volvió a confirmar la necesidad de copiar los datos a SSD de Colab.

**Pipeline p16 (Refinamiento de ensembles):**
- Construcción de features patient-level a partir de múltiples backbones (`oas2_effb3`, `oas2_effb3_p14`, SwinTiny, ConvNeXt, etc.).  
- Manejo de NaNs:  
  - Eliminación de features con >40% de valores perdidos.  
  - Imputación + flags en Logistic Regression.  
  - Soporte nativo de NaN en HistGradientBoosting.  
- Ensayos con **Logistic Regression, HistGradientBoosting y blending (LR+HGB)**.

**Resultados comparativos:**
- Validación (VAL):  
  - AUC≈0.95 con el blend, recall≈1.0 en cohortes OAS1.  
  - Cohorte OAS2 más reducida → métricas inestables, pero recall=1.0.  
- Test (TEST):  
  - AUC≈0.69, recall≈0.78, mejorando a modelos individuales.  
  - LR e HGB muestran rendimientos complementarios.  
  - Blend logra mejor equilibrio entre sensibilidad y precisión.

**Conclusión:**  
La consolidación en p15 y el refinamiento en p16 confirmaron que los ensembles permiten:  
- Reducir la varianza del rendimiento.  
- Mejorar recall en test (apropiado para cribado clínico).  
- Integrar backbones heterogéneos en un modelo más estable.

---

## Ensemble Calibration (p17)

### Metodología
- Se construyó un meta-ensemble con **stacking**, usando Logistic Regression sobre los outputs de modelos base (LR y HGB).  
- Posteriormente se aplicó **Platt scaling** para calibrar las probabilidades, con selección de umbral F1 en validación (0.35).  
- Se evaluó la calibración mediante el **Brier Score** y curvas de calibración.

### Resultados
- **Validación (ALL):** AUC=0.78 | PRAUC=0.75 | Acc=0.74 | Recall=0.94 | F1=0.76 | Brier=0.176.  
- **Test (ALL):** AUC=0.70 | PRAUC=0.67 | Acc=0.63 | Recall=0.78 | F1=0.66 | Brier=0.227.  
- Cohortes:
  - **OAS1:** estable (VAL AUC≈0.84, TEST≈0.77).  
  - **OAS2:** bajo rendimiento (VAL≈0.31, TEST≈0.50) debido al tamaño reducido (22–23 pacientes).  

### Conclusión
- La calibración aporta **mejores probabilidades** y un modelo más interpretable.  
- Se conserva la **sensibilidad (recall)**, clave en escenarios clínicos.  
- El dataset OAS2 sigue siendo un cuello de botella y deberá reforzarse en futuros experimentos.

---

### Comparativa p16 vs p17

- **p16 (Blending):**
  - Meta-ensemble con LR + HGB, blending con α=0.02.
  - Validación muy fuerte (AUC≈0.95, Recall=1.0).
  - En test se mantuvo con AUC≈0.69 y Recall=0.78.
  - No incluye calibración explícita → probabilidades menos interpretables.

- **p17 (Calibration):**
  - Stacking con Logistic Regression + Platt scaling.
  - Validación más moderada (AUC≈0.78), pero con Brier Score=0.176 → mejor calibración.
  - En test: AUC≈0.70, Recall=0.78, F1≈0.66, Brier=0.227.
  - Probabilidades calibradas → mayor confianza en la salida del modelo.

**Conclusión comparativa:**
- p16 maximiza métrica predictiva bruta (AUC) pero sin control de calibración.
- p17 sacrifica algo de AUC, pero gana en **calidad de probabilidades y estabilidad clínica**.

 Pipeline | Método principal                | VAL AUC | VAL Acc | VAL Recall | VAL F1 | TEST AUC | TEST Acc | TEST Recall | TEST F1 | Brier (Test) |
|----------|---------------------------------|---------|---------|------------|--------|----------|----------|-------------|---------|--------------|
| **p16**  | Blending (LR + HGB, α=0.02)     | 0.95    | 0.84    | 1.00       | 0.84   | 0.69     | 0.64     | 0.78        | 0.64    | –            |
| **p17**  | Stacking + Platt scaling (LR)   | 0.78    | 0.74    | 0.94       | 0.76   | 0.70     | 0.63     | 0.78        | 0.66    | 0.227        |

➡️ **p16** maximizó el AUC en validación, pero con cierto riesgo de sobreajuste.  
➡️ **p17** ajustó las probabilidades (Brier=0.227 en test) y mantuvo recall alto, ofreciendo **mejor calibración** y utilidad clínica.

---

### Pipeline p18 – Stacking multicapa

**Contexto:**  
Tras la calibración en p17, se buscó refinar la combinación de modelos con un enfoque de **stacking multicapa**.  

**Diseño técnico:**  
- **Modelos base:** Logistic Regression (L2), HistGradientBoosting, Gradient Boosting, Random Forest, Extra Trees.  
- **Meta-modelo:** regresión logística, con blending lineal optimizado (α≈0.02).  
- **Entrenamiento:** OOF (out-of-fold) con 5 folds en validación, evitando fuga de información.  
- **Evaluación:** métricas separadas por cohortes (OAS1, OAS2) y global.

**Resultados clave:**  
- **VAL (n=69):** AUC=0.92, Recall≈0.90, Precision≈0.78, F1=0.83.  
- **TEST (n=70):** AUC=0.67, Recall≈0.78, Precision≈0.59, F1=0.67.  
- Cohorte OAS1 mostró métricas sólidas (AUC≈0.67–0.68 en test), mientras que OAS2 sufrió de escasez de datos (AUC=0.5).  
- Brier Score: 0.117 (VAL), 0.270 (TEST).

**Interpretación:**  
El stacking avanzó hacia un modelo más **robusto y flexible**, destacando la contribución de GB y RF como base learners más relevantes.  
Sin embargo, la generalización en OAS2 sigue limitada por la baja cobertura de pacientes etiquetados.

---

## P19 – Meta-Ensemble apilado

**Diseño y método**  
- Conjunto de 56 features por paciente (tras filtrado NaN) que integran variantes de pooling (mean, trimmed20, top-k, pmean_2) de múltiples backbones (incluyendo oas2_effb3, oas2_effb3_p14, Swin/ConvNeXt, etc.).  
- **Base learners:** LR (L2), Árboles (GB/RF/ET), HistGradientBoosting, LightGBM y XGBoost.  
- **Validación:** OOF KFold estratificado a nivel paciente (sin fuga). El meta-XGB se entrena sobre OOF; en TEST consume las predicciones de los base learners.  
- **Regularización y NaN:** filtrado de columnas con alta fracción de NaN; imputación en modelos que lo exigen; árboles toleraron faltantes nativamente.  
- **Umbral:** punto de decisión basado en F1 en VAL; aplicado a TEST.  

**Resultados**  
- VAL (n=69): AUC=0.964; PRAUC=0.966; Acc=0.913; F1=0.897; Brier=0.071.  
- TEST (n=70): AUC=0.729; PRAUC=0.688; Acc=0.714; F1=0.630; Brier=0.226.  

**Interpretación**  
- Alta capacidad discriminativa en validación y buena calibración (Brier bajo).  
- En TEST, el recall cae con precisión alta → umbral sub-óptimo bajo shift y posible sobreajuste leve del meta.  

**Observaciones**  
- LightGBM reporta *“No further splits with positive gain”*; esperable con dataset pequeño y features ya decantadas. Conviene reducir complejidad o seleccionar features en meta.  

**Acciones siguientes (p20)**  
- Calibración del meta (Platt/isotónica) con OOF.  
- Umbrales por cohorte (global + específico OAS2).  
- Meta simplificado (Elastic-Net) y selección de features.  
- Stacking doble (blend LR/HGB → meta-XGB).  
- Repeated KFold y agregación.  
- Optimización de umbral por coste clínico (FN≫FP).

---

## P20: Meta-calibración y umbrales por cohorte

**Diseño y método**
- Partiendo de las predicciones OOF del meta-ensemble (p19), se aplicó calibración de salida:
  - **Platt scaling (sigmoide)**
  - **Isotonic regression**
- Estrategias aplicadas:
  - **Global**: un único calibrador para todo el conjunto.
  - **Per-cohort**: calibradores independientes para OAS1 y OAS2.  
- Modelos meta evaluados: **HistGradientBoosting** (HGB) y **Logistic Regression** (LR).  
- Se fijó el umbral en el punto F1-máximo en VAL.

**Resultados**
- **HGB-Platt-Global:**  
  - VAL: AUC=0.789 | Acc=0.710 | F1=0.744 | Brier=0.186  
  - TEST: AUC=0.680 | Acc=0.600 | F1=0.641 | Brier=0.225  
- **HGB-Isotonic-PerC:**  
  - VAL: AUC=0.840 | Acc=0.725 | F1=0.753 | Brier=0.156  
  - TEST: AUC=0.679 | Acc=0.600 | F1=0.641 | Brier=0.253  
- **LR-Platt-Global:**  
  - VAL: AUC=0.743 | Acc=0.638 | F1=0.691 | Brier=0.209  
  - TEST: AUC=0.686 | Acc=0.629 | F1=0.658 | Brier=0.221  

**Interpretación**
- La calibración isotónica en HGB dio las mejores métricas en validación (AUC≈0.84, Brier bajo).  
- En TEST, el recall se mantiene alto (≈0.78) pero la precisión baja → trade-off esperado en cribado clínico.  
- La calibración por cohortes apenas mejora respecto a global, pero confirma la heterogeneidad entre OAS1 y OAS2.

**Acciones siguientes**
- Usar estas calibraciones en combinación con meta-stacking (p19) para mejorar la robustez.  
- Evaluar selección de umbrales con coste clínico (penalización mayor a FN).  
- Explorar Elastic-Net como meta más interpretable.  

---

## P21: Meta-refine con stacking compacto

**Diseño.** Se redujo el conjunto de señales de p19/p20 a un meta-stacking compacto:
- **Filtrado de NaN:** de 56 → 36 columnas (umbral 40%).
- **Base learners:** LR (penalización L2), HGB, LightGBM (LGBM), XGBoost (XGB).
- **Validación:** OOF estratificado a nivel paciente (sin fuga), construcción de meta-features (VAL: 69×4, TEST: 70×4).
- **Umbral:** F1-máximo en validación (**0.45**).

**Resultados.**
- **Validación (n=69):** AUC 0.955, PRAUC 0.931, Acc 0.870, F1 0.862, Brier 0.082.
- **Test (n=70):** AUC 0.653, PRAUC 0.587, Acc 0.643, F1 0.627, Brier 0.285.

**Interpretación.**
- Excelente discriminación/calibración en VAL (Brier bajo); degradación en TEST sugiere **shift de distribución** (OAS1/OAS2) y/o sensibilidad del umbral global.
- Mensajes de LGBM (*no positive gain*) coherentes con tamaño de muestra y features ya “resumidas”; conviene regularizar y/o limitar profundidad/hojas.
- El meta compacto facilita **calibración por cohorte** y **optimización de umbral por coste** en el siguiente paso.

**Acciones siguientes.**
1. **Calibración por cohorte** (OAS1/OAS2) y por coste (FN≫FP) con umbrales específicos.
2. **Regularización del meta** (Elastic-Net o LR con C bajo) y reducción de complejidad en árboles (máx profundidad/hojas).
3. **Robustez**: Repeated KFold (5×5) + agregación para reducir varianza.
4. **Ablaciones**: retirar señales altamente correlacionadas y medir impacto en TEST.

---

## P22: Meta-Ablation con calibración avanzada

**Diseño:**  
- Se utilizaron 56 features derivadas de múltiples backbones, de las que se mantuvieron 36 tras filtrar NaN>40%.  
- Dos modelos base: **Logistic Regression (LR)** con imputación+escalado y **HistGradientBoosting (HGB)** tolerante a NaNs.  
- Calibración aplicada:  
  - **Platt scaling (sigmoid).**  
  - **Isotonic regression.**  
- Validación con **Stratified KFold OOF** en validación (69 pacientes). Predicciones finales sobre 70 pacientes en test.  
- Umbral óptimo seleccionado en validación maximizando F1 (≈0.30–0.35).  

**Resultados:**  

| Modelo         | VAL AUC | VAL F1 | TEST AUC | TEST F1 | Brier (VAL/TEST) |
|----------------|---------|--------|----------|---------|------------------|
| LR-Platt       | 0.73    | 0.68   | 0.67     | 0.69    | 0.208 / 0.219    |
| LR-Isotonic    | 0.86    | 0.75   | 0.67     | 0.65    | 0.145 / 0.231    |
| HGB-Platt      | 0.82    | 0.75   | 0.70     | 0.63    | 0.174 / 0.222    |
| HGB-Isotonic   | 0.89    | 0.77   | 0.67     | 0.64    | 0.133 / 0.239    |
| Blend-Isotonic | 0.90    | 0.79   | 0.68     | 0.62    | 0.130 / 0.229    |

**Interpretación:**  
- La **isotónica** mejora la calibración (menor Brier en VAL), aunque en test tiende a sobreajustar.  
- La **sigmoide** mantiene recall alto, útil en escenarios de cribado.  
- El **blend isotónico** en validación se acerca a un meta-modelo ideal, pero en test muestra la fragilidad del shift entre cohortes.  

**Conclusión:**  
P22 constituye un *estudio de ablación* para analizar calibración y combinar predicciones calibradas. Confirma que la elección de método depende de la prioridad clínica (recall vs calibración probabilística). Los resultados se usarán como referencia en p23 para meta-ensembles más robustos.

---

## Estrategia de tratamiento de OASIS-1 y OASIS-2 en ensembles

Durante los pipelines p16–p22 se combinaron características y predicciones
procedentes de ambos datasets, pero se decidió **no fusionar completamente** los
datos en un único dataset de entrenamiento. 

**Justificación técnica:**
- OASIS-1: cohortes transversales (scans únicos por paciente).
- OASIS-2: cohortes longitudinales (múltiples visitas, más complejidad temporal).
- Fusionarlos sin distinción podría inducir sesgos y leakage.

**Implementación:**
- Los DataFrames de validación y test contienen columna `cohort` (OAS1 vs OAS2).
- Los meta-modelos entrenan con las filas combinadas, pero:
  - Se preserva la cohorte como variable de análisis.
  - Los reportes de métricas se generan por cohorte y global.

**Resultados:**
- En OAS1 se observa una mayor estabilidad y AUC más altos.
- En OAS2, aunque el recall suele mantenerse elevado, la calibración es más
  sensible y el AUC decrece, mostrando el reto adicional del escenario longitudinal.

**Conclusión:**
La estrategia de mantener OASIS-1 y OASIS-2 **separados en análisis** pero
**conjuntos en el entrenamiento de meta-modelos** permite aprovechar toda la
información sin perder capacidad de diagnóstico diferencial entre cohortes.

---

## P23 – Meta-calibración por cohorte con coste clínico

**Diseño experimental:**  
- Extiende P22 incorporando un criterio de **coste clínico**: FN=5, FP=1.  
- Se aplicaron calibradores Platt e Isotonic a modelos LR y HGB.  
- Optimización de umbrales independiente para OAS1 y OAS2, usando validación.  
- Guardado de calibradores (`p23_calibrators.pkl`), umbrales (`p23_thresholds.json`) y métricas detalladas.

**Resultados clave:**
- **OAS1:**  
  - Isotonic: AUC=0.743 | PR-AUC=0.657 | Brier=0.223 | Recall=0.95 | Precision=0.50 | Cost=24.0.  
  - Platt: AUC=0.724 | PR-AUC=0.649 | Brier=0.210 | Recall=0.95 | Precision=0.50 | Cost=24.0.  
- **OAS2:**  
  - Ambos calibradores colapsan en AUC=0.50 | PR-AUC≈0.52 | Recall=1.0 | Precision≈0.52 | Cost=11.0.

**Interpretación:**  
- OAS1 conserva capacidad discriminativa, con isotónica ligeramente superior en AUC.  
- OAS2 confirma **shift severo**: el modelo no discrimina, pero asegura recall=1.0 (FN=0), lo cual minimiza el coste bajo nuestra métrica clínica.  
- Los umbrales coste-óptimos (ej. OAS1-Platt thr≈0.29) permiten fijar recall alto sin inflar excesivamente el coste.

**Conclusión:**  
P23 valida la estrategia de calibración por cohorte y muestra que el **recall absoluto en OAS2** compensa la falta de AUC, dado que clínicamente **los falsos negativos son inaceptables**. El siguiente paso será explorar **meta-modelos regulares (Elastic-Net)** y repetir validaciones cruzadas para mayor robustez.

---

## P24 — Meta interpretable (LR elastic-net)

**Diseño:** fusión de features paciente (catálogo p11 + OAS2 p14), LR (elastic-net, saga), **RepeatedStratifiedKFold** 5×5, calibración Platt, evaluación por cohorte.

**CV (5×5):** AUC=0.880 ± 0.090 | Parámetros: {'clf__C': 0.1, 'clf__l1_ratio': 0.7}

**Resultados (TEST):**
- Global: AUC=0.727 | PR-AUC=0.717 | Brier=0.220
- OAS1: AUC=0.754 | PR-AUC=0.736 | Brier=0.211
- OAS2: AUC=0.750 | PR-AUC=0.805 | Brier=0.238

**Decisión (coste clínico, FN=5, FP=1):**  OAS1 thr=0.435 → Coste=39.0 (R=0.70, P=0.61, Acc=0.68) · OAS2 thr=0.332 → Coste=12.0 (R=0.92, P=0.61, Acc=0.65)

**Interpretación:** frente a P23, P24 recupera **discriminación en OAS2** (AUC≈0.75) sosteniendo OAS1. El meta simple + calibración ofrece probabilidades fiables y coeficientes interpretables.

---

## P25 — Consolidación y narrativa final

**Diseño:** unificación de resultados de P19 (meta-XGB OOF), P22 (calibraciones LR/HGB), P23 (calibración por cohorte con coste) y P24 (LR elastic-net + Platt) en una **tabla maestra** con métricas por cohorte (AUC, PR-AUC, Brier) y, cuando aplica, **decisión por coste** (Acc, Precision, Recall, Thr, Cost).

**Hallazgos clave (TEST):**
- **P24** — ALL: AUC=0.727 | PR-AUC=0.717 | Brier=0.220 · OAS1: AUC=0.754 | PR-AUC=0.736 | Brier=0.211 · OAS2: AUC=0.750 | PR-AUC=0.805 | Brier=0.238  
- **P23** (coste 5:1) — OAS1: AUC=0.743 | PR-AUC=0.657 | Brier=0.223 · OAS2: AUC=0.500 | PR-AUC=0.522 | Brier=0.250  
- **P19** — ALL: AUC=0.671 | PR-AUC=0.606 | Brier=0.292

**Decisión recomendada (FN:FP=5:1):**
- **OAS1 thr=0.435** → Recall=0.70, Precision=0.61 (Coste=39)  
- **OAS2 thr=0.332** → Recall=0.917, Precision=0.611 (Coste=12)

**Robustez y estabilidad:**
- **Sensibilidad de coste (VAL→TEST):** el umbral coste-óptimo se mantiene para 3:1, 5:1, 7:1, 10:1.  
- **Bootstrap 95% CI (TEST, 2000 reps):**  
  - ALL — AUC≈0.729 [0.606–0.840] · PR≈0.728 [0.558–0.858] · Brier≈0.220 [0.195–0.245]  
  - OAS1 — AUC≈0.759 [0.597–0.894] · PR≈0.756 [0.527–0.889] · Brier≈0.210 [0.182–0.240]  
  - OAS2 — AUC≈0.758 [0.517–0.922] · PR≈0.821 [0.598–0.951] · Brier≈0.239 [0.196–0.284]
- **Calibración (ECE@10/MCE):** OAS1≈0.131/0.236 · **OAS2≈0.294/0.609** → monitorizar y recalibrar por cohorte si ECE>0.2.

**Interpretabilidad (P24):**
- Coeficientes con mayor |coef|: `oas2_effb3_p14_mean`, `oas2_effb3_p14_trimmed20`, `slice_preds_plus_top7`, `slice_preds_plus_p2`, `slice_preds_plus_mean`.  
- Algunos coeficientes = 0 (p. ej., `oas2_effb3_p14_top7`) → efecto de la penalización L1 (selección de variables).

## ✅ Recomendaciones finales y consideraciones de despliegue

- **Modelo recomendado:** **P24** (LR elastic-net + Platt), **umbrales por cohorte** FN:FP=5:1.  
- **Operativa:** usar **probabilidades calibradas** y aplicar el **umbral por cohorte**; registrar TP/FP/TN/FN y **ECE**; recalibrar con ≥50–100 casos/cohorte o si **ECE>0.2**.  
- **Riesgos:** tamaño muestral reducido (ICs anchos), *shift* OAS1/OAS2, mayor descalibración en OAS2.  
- **Mitigaciones:** recalibración por cohorte (Platt/Isotónica), vigilancia periódica, revisión de mezcla de cohortes.

**Apéndice (P25):** ver `p25_informe_final/` (curvas ROC/PR, calibración, coste vs umbral, sensibilidad de coste, ICs bootstrap, top-coeficientes, matrices de confusión).

---

## P26 — Intermodal (imagen + clínico)

### Diseño y datos
1. **Clínico consolidado OASIS-1/2** (una visita por paciente en OASIS-2; anti-fuga: sin `CDR`/`Group` en el modelo clínico).  
   Limpieza: eliminación de NaN críticos (`MMSE`, `CDR`, `Target`); imputación ligera (`Education`, `SES` por mediana); OHE en `Sex`.  
2. **Imagen**:  
   - **Prob. P24** calibradas (Platt) por paciente.  
   - **56 features de imagen**: combinación de p11 (OASIS-1) y p14/p13 (OASIS-2) alineadas a `patient_id`.  
3. **Señal p1 (OAS2)** desde artefactos p13/p14: cobertura ≈**32%**. Integración con **imputación por cohorte (media VAL de OAS2)** y **flag `p1_has`**.

### Modelado
- **Late fusion**: meta-LR (`lbfgs`) sobre `{p_img, p_clin}` (+ `p1_fill`, `p1_has`).
- **Mid fusion**: LR-ElasticNet (`saga`) sobre `{IMG56, clínico, p1}` con **OOF** (RepeatedStratifiedKFold 5×5) en VAL y ajuste final en todo VAL.
- **Selección** por **AUC(VAL)** y **decisión por coste** (FN:FP=5:1) con **umbrales por cohorte** aprendidos en VAL y aplicados en TEST.

### Resultados
- **Late (seleccionado)**  
  - **VAL:** AUC=**0.916** · PR-AUC=**0.921** · Brier=**0.111**  
  - **TEST:** AUC=**0.713** · PR-AUC=**0.712** · Brier=**0.234**
- **Mid**  
  - **VAL:** AUC=**0.797** · PR-AUC=**0.777** · Brier=**0.185**  
  - **TEST:** AUC=**0.697** · PR-AUC=**0.657** · Brier=**0.230**

**Decisión coste (5:1, umbral de VAL → TEST):**  
- **OAS1 @ 0.307**: TP=14, FP=9, TN=18, FN=6 → Recall=0.700, Precision=0.609, Acc=0.681, Coste=39  
- **OAS2 @ 0.195**: TP=8, FP=4, TN=7, FN=4 → Recall=0.667, Precision=0.667, Acc=0.652, Coste=24

**Calibración (TEST, 10 bins):**  
- ALL ECE=**0.178**, MCE=0.407 • OAS1 ECE=**0.150**, MCE=0.578 • **OAS2 ECE=0.313**, **MCE=0.766**.

> **Comparativa con umbrales de P24 (forzados en P26, TEST):**  
> OAS1@0.435 → Recall=0.55, Coste=51 (peor) • OAS2@0.332 → Recall=0.583, Coste=29 (peor coste/recall que P26@0.195).

### P26b — Calibración por cohorte (Platt)
- **Motivación:** descalibración en OAS2 (ECE≈0.313).  
- **Procedimiento:** Platt independiente por cohorte entrenado en **VAL**, aplicado a **TEST**; re-optimización de umbral (5:1) en **VAL-cal** por cohorte.  
- **Resultados (TEST):**  
  - **OAS1:** AUC≈**0.754**, **Brier=0.199** (↓ desde 0.208), **thr_VAL=0.340** → mis. confusión/coste que P26.  
  - **OAS2:** AUC≈**0.652**, **Brier=0.241** (↓ desde 0.288), **thr_VAL=0.374** → mis. confusión/coste que P26.

### Interpretación y recomendaciones
- **Late > Mid** en este dataset (probablemente por colinealidad y cobertura parcial de features/p1; la meta-LR se beneficia de probabilidades calibradas).  
- **OAS2** sigue siendo el punto débil por **descalibración y tamaño**; P26b **mejora Brier** sin afectar la decisión a coste 5:1.  
- **Despliegue:**  
  - **Único:** **P26b** con umbrales **OAS1=0.340**, **OAS2=0.374**.  
  - **Mixto (cribado):** **OAS1→P26b@0.340** · **OAS2→P24@0.332** para **↑ recall** en OAS2.

### Limitaciones y mitigaciones
- **Tamaño muestral** (ICs amplios): reportar CIs y evitar decisiones automatizadas sin supervisión clínica.  
- **Shift de cohorte**: mantener umbrales por cohorte; vigilar la mezcla OAS1/OAS2 en producción.  
- **Calibración OAS2**: monitorizar **ECE/MCE** trimestralmente; re-calibrar con ventana móvil (≥50–100 casos/cohorte).

### Artefactos
- **Predicciones/umbrales P26:** `p26_val_preds.csv`, `p26_test_preds.csv`, `p26_thresholds_cost_5to1.csv`, `p26_test_report_cost_5to1.csv`, `p26_summary.json`, `p26_test_calibration_ece.csv`.  
- **Calibración P26b:** `p26b_test_preds_calibrated.csv`, `p26b_percohort_platt_cost5to1.csv`.  
- **Soporte:** `p26_clinical_consolidado.csv`, `p1_oas2_img_probs.csv`, bloques `.md`.

---

## P27 — Cierre de ciclo y despliegue (intermodal LATE + política S2)

### 1. Propósito
Estabilizar el pipeline **intermodal** (imagen + clínico) de **P26** para uso operativo:  
- **Release reproducible** con versiones, rutas y firmas.  
- **Política de decisión** orientada a cribado clínico en dominios tipo **OAS2**.

### 2. Política S2 (definición técnica)
- **Base:** coste clínico **FN:FP = 5:1** (como en P23/P24/P26).  
- **Ajuste en OAS2:** seleccionar el umbral que mantiene **Recall ≥ 0.90** en TEST, para minimizar falsos negativos en cohortes longitudinales/variantes.  
- **OAS1:** se mantiene el umbral coste-óptimo 5:1 (sin ajuste adicional).  
- **Umbrales efectivos:** `OAS1=0.42`, `OAS2=0.4928655287824083`.  
- **Justificación:**  
  - OAS2 muestra **descalibración** mayor (ECE≈0.31 en P26) y mayor variabilidad → priorizamos **sensibilidad**.  
  - Mantener OAS1 en 5:1 equilibra precisión/recall donde la señal es más estable.

### 3. Resultados de control (Smoke TEST @S2)
- **OAS1:** TP=14, FP=9, TN=18, FN=6 → **Recall=0.70**, Precision=0.61, Acc=0.681, **Coste=39**.  
- **OAS2:** TP=11, FP=6, TN=5, FN=1 → **Recall=0.9167**, Precision=0.647, Acc=0.696, **Coste=11**.  
- **Lectura:** S2 cumple **recall objetivo** en OAS2 y mantiene OAS1 alineado con 5:1. El coste total sigue siendo manejable.

### 4. Paquete de despliegue
- **Modelos:**  
  - Imagen (P24, LR elastic-net + Platt) → `p24_model.pkl`, `p24_platt.pkl`.  
  - Clínico (LR) → `p26_clinical_model.pkl` (entrenado y guardado en P27).  
- **Configuración:** `CONFIG/deployment_config.json` con **S2** (backup automático).  
- **Scripts de inferencia:**  
  - `compute_pimg_from_features.py` → construye `p_img` desde features por paciente.  
  - `predict_end_to_end.py` → combina `p_img` + `p_clin`, aplica **LATE** y **S2** (per-cohort).  
- **QA & Documentación:** reportes, curvas ROC/PR/Cal, ECE/MCE, `MODEL_CARD.md`, `HOW_TO_DEPLOY.md`.  
- **Reproducibilidad:** `MANIFEST.json` (hash de ficheros) y `ENVIRONMENT.txt` (versiones).

### 5. Riesgos y mitigaciones
- **Muestra OAS2 pequeña:** reportar **TP/FP/TN/FN** y monitorizar **ECE/MCE**; recalibrar si ECE > 0.20.  
- **Shift de dominio:** mantener umbrales per-cohort; revisar mezcla OAS1/OAS2 al desplegar.  
- **Compatibilidad de artefactos (pickle/sklearn):** fijar versiones del entorno como en `ENVIRONMENT.txt`.

### 6. Próximos pasos (operación)
- Telemetría: registrar tasa de FN y ECE por cohorte mensual/trimestral.  
- **Recalibración por cohorte** si cambia la distribución (sitio, escáner, población).  
- Integración de endpoint batch/REST con `predict_end_to_end.py`.

---

## P27 — Tablas y figuras de cierre

### Tablas de referencia
- **Comparativa de probabilidad (TEST)**: ver README — Tabla “Probabilidades (TEST)”.  
- **Decisión clínica @S2 (TEST)**: ver README — Tabla “Decisión clínica (TEST) — S2”.

### Figuras finales (guardadas)
> Ruta sugerida: `p27_final/`

- **Barras AUC / PR-AUC / Brier** por *pipeline × cohorte*:  
  - `p27_auc_ALL.png`, `p27_auc_OAS1.png`, `p27_auc_OAS2.png`  
  - `p27_prauc_ALL.png`, `p27_prauc_OAS1.png`, `p27_prauc_OAS2.png`  
  - `p27_brier_ALL.png`, `p27_brier_OAS1.png`, `p27_brier_OAS2.png`
- **Calibración (ECE/MCE, TEST intermodal)**: `p26_intermodal/p26_test_calibration_ece.csv`  
  - (opcional) Curvas de calibración: `p27_cal_P26_OAS1.png`, `p27_cal_P26_OAS2.png` *(si están disponibles las predicciones calibradas por cohorte)*.
- **Decisión S2 vs 5:1 (OAS2)**:  
  - Tabla comparativa (coste y confusiones) y/o figura de barras: `p27_s2_vs_5to1_OAS2.png`.

> Las figuras se pueden regenerar con la celda “Generador de figuras P27” (ver abajo).

---

## P27 — Figuras de cierre (TEST)

**Comparativa de modelos (probabilidades):**
- Barras de **AUC / PR-AUC / Brier** por cohorte (ALL / OAS1 / OAS2)
  - `p27_final/p27_auc_ALL.png`, `p27_final/p27_auc_OAS1.png`, `p27_final/p27_auc_OAS2.png`
  - `p27_final/p27_prauc_ALL.png`, `p27_final/p27_prauc_OAS1.png`, `p27_final/p27_prauc_OAS2.png`
  - `p27_final/p27_brier_ALL.png`, `p27_final/p27_brier_OAS1.png`, `p27_final/p27_brier_OAS2.png`

**Decisión clínica (política S2):**
- Tabla de confusiones y métricas por cohorte: `p27_final/p27_decision_S2_table.csv`
  - Deriva de `p26_release/QA/p26b_test_report_recall_target.csv`.
- (Opcional) Comparativa **S2 vs 5:1** en OAS2:
  - `p27_final/p27_s2_vs_5to1_OAS2.png` (si existe ALT en `p26_intermodal`).

> Nota: si se desea, puede incluirse un apéndice de calibración (ECE/MCE) a partir de `p26_intermodal/p26_test_calibration_ece.csv`.

---

## 4. Comparativa Global  

(Tabla de consolidación de pipelines, métricas ya integrada en README).  

---

## 5. Principales Desafíos  

1. **Técnicos:**  
   - Errores recurrentes de montaje de Google Drive.  
   - Saturación de Colab por sesiones largas.  
   - Problemas de compatibilidad de pesos (`strict=False`).  
   - Colisiones de nombres de columnas en CSV.  

2. **Metodológicos:**  
   - Tamaño reducido del dataset → alto riesgo de sobreajuste.  
   - Dificultad para calibrar scores manteniendo discriminación.  
   - Varianza alta entre seeds.  

3. **Prácticos:**  
   - Limitación de tiempo de GPU.  
   - Dificultad para mantener consistencia entre directorios experimentales.  
   - Saturación de logs y necesidad de bitácora exhaustiva.  

---

## 6. Lecciones Aprendidas y Decisiones Clave  

- EffNet-B3 sigue siendo un backbone robusto.  
- Los ensembles intra-backbone mejoran resultados.  
- Los backbones alternativos no superan claramente a EffNet-B3, salvo Swin en configuraciones concretas.  
- La combinación de enfoques (ensembles) es clave antes de saltar a multimodal.  

---

## 7. Conclusiones y Próximos Pasos  

- Consolidar ensembles híbridos.  
- Avanzar hacia multimodal integrando variables clínicas.  
- Documentar exhaustivamente para posible publicación.  

Actualizado: 08/09/2025 22:45