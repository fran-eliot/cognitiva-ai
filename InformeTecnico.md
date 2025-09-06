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

Actualizado: 05/09/2025 12:13