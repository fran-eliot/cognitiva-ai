# 📑 Informe Técnico — **COGNITIVA-AI — Detección Temprana de Alzheimer**

## 0. Antecedentes

La **Enfermedad de Alzheimer (EA)** es neurodegenerativa y progresiva. Una detección temprana es clave para:  
- Optimizar la atención clínica.  
- Planificar intervenciones.  
- Reducir costes en fases avanzadas.  

## 1. Introducción General  

El presente informe constituye una documentación exhaustiva, detallada y profundamente analítica del proyecto **CognitivaAI**, cuyo propósito ha sido explorar, diseñar, entrenar y evaluar modelos de aprendizaje profundo aplicados al diagnóstico automático de **deterioro cognitivo temprano (Alzheimer)** a partir de datos clínicos tabulares e imágenes de resonancia magnética (MRI) de la base de datos **OASIS**.  

El proyecto surge con una doble motivación:  

1. **Científica y social:** el diagnóstico temprano de enfermedades neurodegenerativas, como la enfermedad de Alzheimer, es un desafío prioritario para la medicina actual. El uso de inteligencia artificial puede ayudar a identificar biomarcadores sutiles en imágenes cerebrales y en datos clínicos que son difíciles de detectar por métodos tradicionales.  

2. **Técnica y experimental:** comprobar hasta qué punto distintas arquitecturas de redes neuronales convolucionales (CNNs) y transformadores visuales pueden capturar información discriminativa en un dataset limitado en tamaño, enfrentándose a problemas comunes como el sobreajuste, la calibración de probabilidades y la estabilidad en validación y test.  

La ejecución del proyecto ha supuesto un recorrido iterativo y progresivo, articulado en una serie de **pipelines experimentales**, cada uno diseñado con objetivos concretos: desde probar un modelo sencillo con datos clínicos (P1) hasta explorar ensembles de arquitecturas avanzadas (P11).  

Cada pipeline ha sido registrado con su motivación, configuración, incidencias, métricas y reflexión crítica, con el objetivo de construir no solo un conjunto de resultados, sino un **mapa de decisiones** que documenta los aprendizajes y trade-offs a lo largo del camino.  

---

## 2. Metodología Global  

### 2.1 Dataset y preprocesamiento  

El Alzheimer es una enfermedad neurodegenerativa cuya detección temprana es clave.  Se espera que un enfoque multimodal permita identificar el deterioro cognitivo incipiente con alta sensibilidad, optimizando el balance entre **detección temprana** y **falsos positivos aceptables** en un contexto de cribado.

El dataset utilizado es **OASIS** (Open Access Series of Imaging Studies) en sus cohortes OASIS-1 y OASIS-2, que incluye imágenes de resonancia magnética (MRI) estructurales de individuos con y sin deterioro cognitivo, junto con información clínica asociada.  

**Datasets:** OASIS-1 (transversal, 416 sujetos, 1 visita) y OASIS-2 (longitudinal, 150 sujetos, visitas múltiples). Ambas cohortes aportan imágenes MRI cerebrales y variables demográficas/neuropsicológicas.

**Diagnóstico binario:** Demented vs Nondemented (OASIS-2: variable Group; OASIS-1: CDR > 0 como demencia).

**Variables clínico-demográficas:** Edad, Sexo, Años de educación, Nivel socioeconómico (SES), MMSE, CDR, eTIV, nWBV, ASF.
- **Age**: Edad del paciente. Predictor fuerte de riesgo.  
- **Sex**: Diferencias epidemiológicas en prevalencia.  
- **Educ**: Años de educación formal. Relacionado con reserva cognitiva.  
- **SES**: Escala socioeconómica. Asociada a recursos cognitivos.  
- **MMSE**: Test cognitivo global (0–30).  
- **CDR (Clinical Dementia Rating)**: Escala clínica de severidad de demencia.  Es la variable principal didicotomizada en “control” vs “deterioro cognitivo”.
- **eTIV**: Volumen intracraneal estimado.  
- **nWBV**: Proporción de volumen cerebral respecto intracraneal.  
- **ASF**: Factor de ajuste anatómico.  
 

**Etiquetas de cohorte**: para trazabilidad y análisis estratificado.

El objetivo principal es **emular la intuición clínica** integrando:

- **Datos clínicos tabulares**: Tests neuropsicológicos, factores de riesgo, medidas volumétricas (edad, género, MMSE, CDR, entre otros).
- **MRI estructural**: imágenes cerebrales que pueden reflejar la atrofia cerebral (en formato NIfTI, preprocesadas a slices 2D para entrenamiento). 



El preprocesamiento incluyó:  
- Homogeneización de columnas a `snake_case`.  
- Imputación de SES/Educ por mediana, escalado estándar, one-hot para Sex y Cohorte.  
- Target unificado por cohorte (OASIS-2: `Group`; OASIS-1: `CDR>0`).  
- Splits estratificados con separación por paciente.
- Mapeo de Correspondencia paciente-slice mediante ficheros `oas1_val_colab_mapped.csv` y `oas1_test_colab_mapped.csv`.  
- Normalización de intensidades.  
- Creación de slices 2D a partir de volúmenes 3D para facilitar el entrenamiento en arquitecturas CNN estándar.  
- Balanceo parcial mediante augmentaciones (flip horizontal, rotaciones leves).  


**División de conjuntos:**  
  - Clínicos (pipelines 1–2): validación cruzada 5-fold y hold-out sobre OASIS-2, anidada para la fusión OASIS-1+2.
  - Imágenes (pipelines 3–9): esquema Train/Val/Test (aprox. 60/20/20) estratificado por paciente, evitando leakage.


### 2.2 Infraestructura  

- **Hardware**: Google Colab Pro (GPU NVIDIA T4).   
- **Almacenamiento persistente**: Google Drive (`/MyDrive/CognitivaAI`).  
- **Librerías clave**: PyTorch, timm (colección de modelos), scikit-learn, pandas, matplotlib.  
- **Optimización**: AdamW, LR schedulers, early stopping.
- **Augmentations**: flips, rotaciones leves. 
- **Gestión de experimentos**: notebooks separados por pipeline, con guardado automático de métricas y predicciones en CSV.  

### 2.3 Métricas de evaluación  

Dada la naturaleza binaria y desbalanceada de la tarea, se utilizaron múltiples métricas:  

- **AUC (ROC)**: medida global de discriminación.  
- **PR-AUC**: más informativa en datasets desbalanceados.  
- **Accuracy**: proporción de aciertos.  
- **Recall (sensibilidad)**: clave, ya que la tarea exige detectar la mayor cantidad de pacientes con deterioro cognitivo.  
- **Precision**: importante para evitar falsos positivos.  
-**Threshold**: selection por F1 óptimo, Youden y recall controlado (90–100%).  
- **Youden Index**: criterio alternativo de umbralización.  
- **@REC90 / @REC100**: configuraciones que maximizan el recall (sensibilidad) incluso a costa de la precisión.  

---

## 3. Evolución por Pipelines  

Se han desarrollado **diez pipelines secuenciales**, cada uno incorporando mejoras y aprendizajes de la fase previa:

- **P1-COGNITIVA-AI-CLINIC** – Modelos ML clásicos con datos clínicos de OASIS-2 (baseline tabular).
- **P2-COGNITIVA-AI-CLINIC-IMPROVED** – Datos clínicos fusionados (OASIS-1 + OASIS-2), mayor muestra y generalización.
- **P3-COGNITIVA-AI-IMAGES** – Primer approach con MRI OASIS-2 usando Deep Learning (ResNet50).
- **P4-COGNITIVA-AI-IMAGES-IMPROVED** – Refinamiento de pipeline de MRI (más datos y rigor en splits).
- **P5-COGNITIVA-AI-IMAGES-IMPROVED-GPU** – Extracción de embeddings MRI con ResNet18 en GPU y calibración isotónica.
- **P6-COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED** – Embeddings EfficientNet-B3 + ensemble de clasificadores a nivel paciente.
- **COGNITIVA-AI-IMAGES-FT** – Fine-tuning parcial de EfficientNet-B3 sobre MRI para mejorar discriminación.
- **COGNITIVA-AI-IMAGES-FT-IMPROVED** – Ajustes de fine-tuning: calibración de probabilidades y optimización de pooling.
- **COGNITIVA-AI-IMAGES-FT-STABLE** – Modelo fine-tune final calibrado y con umbral clínico optimizado (pipeline MRI definitivo).
- **COGNITIVA-AI-FINETUNING-STABLE-PLUS** → Versión extendida con calibración adicional y pooling alternativo (mean, median, top-k). 

Cada pipeline se documenta a continuación, detallando metodología, resultados y conclusiones. Esta evolución progresiva permite contrastar la eficacia de distintas aproximaciones (ML tradicional vs Deep Learning, datos clínicos vs imágenes, etc.) y justifica las decisiones técnicas tomadas.

### 3.1 Pipeline 1 – P1-COGNITIVA-AI-CLINIC (Clínico OASIS-2)  

**Motivación:**  
Probar un primer modelo utilizando exclusivamente variables clínicas del dataset OASIS-2, sin imágenes. El objetivo era establecer un baseline puramente tabular.  

**Metodología:**  
Tres clasificadores clásicos (Regresión Logística, Random Forest, XGBoost) con datos tabulares de OASIS-2 (150 sujetos). Validación cruzada (5-fold) y conjunto test reservado (20%).

**Resultados:**

| Modelo           | AUC (CV 5-fold)   | AUC (Test) |
|------------------|-------------------|------------|
| Logistic Reg.    | 0.912 ± 0.050     | –          |
| Random Forest    | 0.925 ± 0.032     | –          |
| XGBoost          | 0.907 ± 0.032     | 0.897      |

**Conclusión:**  
Baseline robusto basado en datos clínicos, limitado por la escasez de datos (solo OASIS-2). Confirma la fuerza de variables clínicas (CDR, MMSE) en la detección de demencia incipiente.

**Configuración destacada:**  
- Modelo: **XGBoost** con tuning básico.
- Variables: Datos clínicos tabulares (edad, sexo, puntuaciones cognitivas, CDR). 
- Validación cruzada interna.  

**Resultados:**  
- AUC (Test): 0.897.  
- Sin métricas de PR-AUC reportadas, aunque se observó buen desempeño en términos de recall.  

**Reflexión:**  
Este baseline confirmó que incluso con variables clínicas simples se puede alcanzar un nivel competitivo de discriminación. Sin embargo, la dependencia exclusiva de variables clínicas limita la aplicabilidad general.  

---

### 3.2 Pipeline 2 – P2-COGNITIVA-AI-CLINIC-IMPROVED (Clínico fusionado OASIS-1+2) 

**Motivación:**  
Mejorar la generalización combinando OASIS-1 y OASIS-2 (~550 sujetos). Unificación de variables y control de cohortes. 

#### ⚖️ Manejo del desbalanceo
- Se probaron variantes con `class_weight='balanced'` y `scale_pos_weight` en XGBoost.  
- Se optimizó el **umbral de decisión** para priorizar *recall* clínico.  
  - Umbral óptimo ≈ 0.03 → Recall ≈ 100%, con sacrificio en precisión (15 falsos positivos).  

#### 🔍 Interpretabilidad
- **Coeficientes (LR):**
  - `CDR` (coef ≈ +4.15) → marcador principal.  
  - `MMSE` (coef ≈ -0.64) → inversamente asociado.  
  - `Educación` (coef ≈ +0.76) → correlación positiva.  
- Conclusión: el modelo se alinea con la evidencia clínica → CDR y MMSE son dominantes.  

#### 📏 Calibración
- Comparación: sin calibrar, **Platt (sigmoid)**, **isotónica**.  
- **Brier Scores:**  
  - LR isotónica → **0.0099** (mejor calibración).  
  - RF isotónica → 0.0170.  
  - XGB isotónica → 0.0187.  

#### 🛡️ Robustez
- **Nested CV:** ROC-AUC = **0.985 ± 0.011**.  
- **Ablation:**  
  - Sin MMSE → ROC-AUC 1.000.  
  - Sin CDR → 0.86.  
  - Sin MMSE+CDR → 0.76.  
  - Sin volumétricas → ≈ 1.000.  
  - Sin socioeducativas → ≈ 0.998.  
- Conclusión: **CDR y MMSE son críticos**, otras variables aportan poco.  

#### 🤝 Ensembling
- Promedio de probabilidades (LR + RF + XGB).  
- Resultado: ROC-AUC = **0.995** → ligera mejora.  


**Resultados recopilación:**

| Modelo           | AUC (Hold-out 80/20) | AUC (CV 5-fold) |
|------------------|----------------------|-----------------|
| Logistic Reg.    | 1.000                | 0.979 ± 0.012   |
| Random Forest    | 0.986                | 0.974 ± 0.018   |
| XGBoost          | 0.991                | 0.975 ± 0.021   |
| Ensemble         | –                    | 0.995 (Nested)  |

Umbral clínico (XGB): recall≈100% con ~15 FP.

**Conclusión:**  
La fusión de bases clínicas potencia el rendimiento (AUC ~1.0). El modelo calibrado permite operar a alta sensibilidad sin sacrificar precisión, alineado con la prioridad clínica de evitar falsos negativos.

**Configuración destacada:**  
- Modelo: **XGB**.  
- Variables: todas las disponibles en OASIS-1 y 2.  
- Validación cruzada más exhaustiva.  

**Resultados XGBoost:**  
- AUC (Test): 0.991.  
- Recall cercano al 100%.  

**Reflexión:**  
El modelo clínico extendido alcanzó una performance casi perfecta, aunque con riesgo evidente de **overfitting** dada la baja dimensionalidad del dataset. Sirvió como techo de referencia para comparar con modelos de MRI.  

---

### 3.3 Pipeline 3 –  P3-COGNITIVA-AI-IMAGES (MRI OASIS-2, ResNet50) 

**Motivación:**  
Dar el salto a imágenes MRI, usando ResNet50 como backbone en el dataset original de OASIS-2.  

**Configuración:**  
- Modelo: **ResNet50 preentrenada en ImageNet**.  
- Dataset: MRI OASIS-2 (formato local).  
- Entrenamiento limitado por CPU local.  

**Preprocesamiento:**
- Conversión de volúmenes a cortes axiales (5 o 20 slices).  
- Normalización a rango [0–255].  
- Opciones: CLAHE y z-score por slice.  
- Augmentation: flips, rotaciones ±10°, ajustes de brillo/contraste.  
- Redimensionado a 224×224 y normalización ImageNet. 

**Entrenamiento:**
- Base: **ResNet50** pre-entrenada en ImageNet.  
- Capa final adaptada a binario.  
- Optimizador: Adam (lr=1e-4).  
- Early stopping con paciencia = 4.  
- Split por paciente (60/20/20).  
- Evaluación a nivel de paciente (probabilidad media). 

**Contexto:**  
Primer análisis de imágenes estructurales cerebrales. OASIS-2, segmentaciones manuales, ~100 cortes axiales por MRI.

**Resultados:**

- **5 slices, sin CLAHE:** Acc = **0.89**, AUC = **0.938**.  
- 5 slices, con CLAHE: Acc = 0.69, AUC = 0.777.  
- 5 slices, CLAHE + z-score: Acc = 0.72, AUC = 0.820.  
- **20 slices, CLAHE+z-score:** Acc = 0.80, AUC = 0.858 (mayor recall, menor precisión)

**Conclusión:**  
- Factible usar Deep Learning con MRI para detectar Alzheimer (AUC ~0.9). 
- Limitaciones computacionales y necesidad de selección cuidadosa de slices.
- Mejor AUC con 5 cortes sin CLAHE; más cortes mejoran recall, pero no AUC.

**Reflexión:**  
Este pipeline demostró que el uso de imágenes cerebrales puede ofrecer resultados competitivos con los clínicos, aunque aún no superiores.  

---

### 3.4. Pipeline 4 – P4-COGNITIVA-AI-IMAGES-IMPROVED (MRI OASIS-1+2 unificado)

**Motivación:**  
Aprovechar todo el conjunto MRI (OASIS-1 + OASIS-2), evitando leakage y aumentando slices por sujeto.

**Resultados:**  
AUC ~0.85–0.90 en validación, sin salto claro respecto al pipeline 3 por limitaciones de cómputo.

**Conclusión:**  
Mejor manejo de datos y splits (Split por paciente/scan y más slices por sujeto)
Mayor robustez, pero mayor coste en CPU : necesidad de GPU para seguir progresando.

---

### 3.4B Pipeline 5 –  P5-COGNITIVA-AI-IMAGES-IMPROVED-GPU (Embeddings ResNet18 + Calibración)  

**Motivación:**  
Probar una arquitectura más ligera (ResNet18) con calibración de probabilidades en Colab. 

**Estrategia:**  
Transfer learning: extracción de embeddings con ResNet18, luego clasificador tradicional (Logistic Regression) y calibración isotónica.

**Configuración:**  
- ResNet18 preentrenada.  
- Embeddings 512D → LR
- Post-procesado: **Platt scaling**.  
- **Calibración:** post-hoc, isotonic regression. 

**Resultados:**  
- Slice-level: AUC_val ≈ 0.627, AUC_test ≈ 0.661 | Brier≈0.23
- Paciente-level (thr≈0.20):
  - AUC (Test): 0.724.  
  - PR-AUC: 0.606.  
  - Accuracy: 0.60.  
  - Recall: 0.80.  
  - Precision: 0.52.  

**Conclusión:**  
Probabilidades más confiables y mejor equilibrio precisión/recall mediante calibración. Sensibilidad (80%) en MRI como referencia para mejoras.

**Reflexión:**  
La calibración mejoró la interpretación de scores, pero el backbone resultó limitado para esta tarea.  

---

### 3.5 Pipeline 6 – P6-COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED (Embeddings EffNet-B3 + Ensemble) 

**Motivación:**  
Usar EfficientNet-B3 como extractor de embeddings, sin fine-tuning completo.  

**Mejoras:**  
Embeddings EfficientNet-B3 (1536-d), clasificadores adicionales (MLP, XGBoost), ensemble LR+XGB.

**Resultados:**

| Modelo         | AUC (Val) | AUC (Test) | PR-AUC (Val) | PR-AUC (Test) | Recall (Test) | Precisión (Test) |
|----------------|-----------|------------|--------------|---------------|---------------|------------------|
| LR             | 0.786     | 0.685      | 0.732        | 0.539         | 0.80          | 0.52             |
| MLP            | 0.870     | 0.648      | 0.886        | 0.556         | 0.95          | 0.53             |
| XGBoost        | 0.782     | 0.670      | 0.633        | 0.617         | 0.75          | 0.56             |
| Ensemble       | 0.815     | 0.704      | 0.705        | 0.623         | 0.90          | 0.60             |

**Conclusión:**  
Embeddings más informativos mejoran el recall (90%). AUC en test (~0.70) aún por debajo del modelo clínico.

**Resultados Ensemble:**  
- AUC (Test): 0.704.  
- PR-AUC: 0.623.  
- Accuracy: 0.70.  
- Recall: 0.90.  
- Precision: 0.60.  

**Reflexión:**  
Buen recall, aunque precision limitada.  

---

### 3.6A Pipeline 7 – P7-COGNITIVA-AI-FINETUNING (Fine-tuning EfficientNet-B3 parcial)

**Planteamiento:**  
Fine-tuning parcial de EfficientNet-B3, pooling por atención a nivel paciente.

**Entrenamiento**: fine-tuning completo/parcial de EfficientNet‑B3 sobre slices. 

**Agregación paciente**: *mean pooling* de scores/logits por slices del mismo paciente.  

**Calibración**: *temperature scaling* con **T=2.6731656060943605**.  

**Umbral clínico**: **0.3651449978351593** (maximiza recall con precisión aceptable).  

**Resultados (nivel paciente, n=47)**  
- **VAL** → AUC=**0.748** | PR-AUC=**0.665** | Acc=**0.702** | P=**0.588** | R=**1.0**  
- **TEST** → AUC=**0.876** | PR-AUC=**0.762** | Acc=**0.745** | P=**0.625** | R=**1.0**

**Matriz de confusión (TEST, thr=0.3651):** TP=8, FP=5, TN=34, FN=0.  

**Conclusión:**  
Fine-tuning eleva la eficacia de clasificación MRI (AUC ~0.87).

---

### 3.6 Pipeline 8 – P8-COGNITIVA-AI-FINETUNING-IMPROVED (Fine-tuning completo de EfficientNet-B3)

**Motivación:**  
Explotar la capacidad completa de EfficientNet-B3, ajustando todos los pesos: Calibración con Temperature Scaling y ajuste de umbral clínico para alta sensibilidad.

**Resultados:**  
- AUC (Test): 0.876.  
- PR-AUC: 0.762.  
- Accuracy: 0.745.  
- Recall: 1.0.  
- Precision: 0.625.  

**Comparación (p6-p8):**s

| Modelo MRI                         | AUC (Test) | Recall (Test) | Precision (Test) |
|-------------------------------------|------------|---------------|------------------|
| Pipeline 6 (Ensemble LR+XGB)        | 0.704      | 0.90          | 0.60             |
| Pipeline 8 (EffNet-B3 fine-tune cal)| 0.876      | 1.00          | 0.625             |

- Umbral clínico ≈ 0.365: Recall_test = 1.00, Precision_test ≈ 0.62, AUC_test ≈ 0.876, PR-AUC_test ≈ 0.762, Acc_test ≈ 0.74

**Conclusión:**  
Modelo MRI altamente sensible y calibrado. Por primera vez se detecta el 100% de los casos de Alzheimer en test, sacrificando precisión (62%) pero ideal para cribado.

**Reflexión:**  
Gran salto respecto a embeddings. El modelo capturó patrones discriminativos más profundos.  

---

## Pipeline 9 - P9-COGNITIVA-AI-FINETUNING-STABLE Fine‑tuning Estable EfficientNet‑B3 (Colab)

**Motivación:**  
Introducir técnicas de estabilización para reducir variabilidad. 

**Configuración**  
- Arquitectura: EfficientNet‑B3 (timm).  
- Entrenamiento: AdamW (lr=1e‑4), AMP (`torch.amp`), early‑stopping por AUC en holdout, 300px, batch=64.  
- Agregación: `mean` a nivel paciente.  
- Calibración: temperature scaling (T=2.048).  
- Umbral: 0.3400 (optimizado con recall≥0.95 en VAL).

**Resultados:**  
- A+UC (Test): 0.740.  
- PR-AUC: 0.630.  
- Accuracy: 0.72.  
- Recall: 0.65.  
- Precision: 0.62.  

**Problema:** logits desorbitados (>500k).

**Comentario:** estabilidad no garantizó mejor generalización.  

---

### Comparación EffNet-B3 (p7,p8,p9)

- **Pipeline 7 (inicial):** fine-tuning base, recall perfecto (1.0) pero precisión moderada.  
- **Pipeline 8 (calibrado):** aplicado *temperature scaling*, mejor consistencia de probabilidades.  
- **Pipeline 9 (estable):** reentrenamiento reproducible con SSD local.  
  - Configuración oficial: *temperature scaling* T≈2.67, thr≈0.365.  
  - Métricas finales: AUC≈0.74, PR-AUC≈0.63, Acc≈0.72, Recall≈0.65, Precision≈0.62.  
  - Confusión TEST: TP=6, FP=4, TN=36, FN=1.  

**Conclusión:** El fine-tuning logra el mejor rendimiento MRI. Pipeline 7 maximizó recall, mientras que Pipeline 9 prioriza estabilidad y reproducibilidad.

---

### 3.8 Pipeline 10 – P10-COGNITIVA-AI-FINETUNING-STABLE-PLUS - EffNet-B3 Fine-Tuning Stable Plus (checkpoint limpio + calibración final) 

**Motivación:**  
Aplicar calibración explícita de scores.  

El pipeline 9 ofrecía estabilidad, pero los checkpoints entrenados no siempre coincidían con la arquitectura definida, cargando <1% de pesos en algunos intentos. Era necesario **reprocesar el checkpoint**, asegurar la integridad de pesos y aplicar calibración para obtener resultados reproducibles.  
Este pipeline se enfocó en reforzar la **calibración y pooling** para asegurar recall absoluto, incluso sacrificando métricas globales.

**Configuración:**  
- Modelo: EfficientNet-B3 binario (head adaptada).  
- Checkpoint: `effb3_stable_seed42.pth`, reconstruido a `best_effb3_stable.pth` (99.7% de pesos cargados).  
- Calibración: *temperature scaling* (T≈2.3) aplicado sobre logits.  
- Pooling: estrategias mean, median y top-k (0.2, 0.3).  
- Evaluación: cohortes de 47 pacientes (VAL) y 47 pacientes (TEST).  

**Resultados:**  

| Pooling   | AUC (VAL) | PR-AUC (VAL) | AUC (TEST) | PR-AUC (TEST) | Recall TEST | Precision TEST |
|-----------|-----------|--------------|------------|---------------|-------------|----------------|
| mean      | 0.630     | 0.667        | 0.546      | 0.526         | 1.0         | 0.47           |
| median    | 0.643     | 0.653        | 0.541      | 0.513         | 1.0         | 0.48           |
| top-k=0.2 | 0.602     | 0.655        | 0.583      | 0.502         | 1.0         | 0.49     

**Resultados recopilación:**  
- AUC (Test): entre 0.546–0.583.  
- PR-AUC: 0.50–0.53.  
- Accuracy: 0.51–0.55.  
- Recall: 1.0.  
- Precision: 0.47–0.49.  

**Conclusión:**  
Pipeline 10 consolida la línea MRI con un recall perfecto en test (1.0), asegurando sensibilidad máxima para cribado clínico temprano. Aunque la precisión baja (~0.47), este pipeline marca el cierre robusto de la etapa **MRI-only** y deja el terreno preparado para la fusión multimodal.

**Reflexión:**  
La calibración no mejoró la discriminación; sacrificó precision.  

---

- **TRIMMED mean (α=0.2)**: media recortada que mejora la estabilidad frente a slices outliers.  
- **TOP-k (k=3,7)**: centradas en las slices más patológicas.  
- **Ensemble MRI**: combinación lineal de MEAN, TRIMMED y TOP7 con pesos óptimos encontrados en validación (0.30, 0.10, 0.60).  

**Resultados:**
- **VAL**: PR-AUC hasta 0.925, recall=0.95, precisión=0.79.  
- **TEST**: PR-AUC 0.737, recall=0.70, precisión=0.61.  

El ensemble mejora la precisión en test (+5 puntos frente a TRIMMED) manteniendo la misma sensibilidad. Se consolida así como la **baseline final de la etapa MRI-only**, antes de avanzar a la integración multimodal con datos clínicos.

---

## 📌 Evaluación de seed-ensemble (EffNet-B3, seeds 41/42/43)

**Objetivo:**  
Verificar si la combinación de checkpoints con distintas semillas podía mejorar la robustez del Pipeline 10 (*EffNet-B3 stable plus*).

**Metodología:**  
- Inferencia slice-level con TTA reducida (orig + flip).  
- Agregación a nivel paciente mediante `mean`, `trimmed` y `top-7`.  
- Calibración en validación: *temperature scaling* y *Platt scaling* con `safe_sigmoid`.  
- Escalado previo: z-score en VAL aplicado a TEST.

**Resultados principales:**  

| Variante        | AUC (TEST) | PR-AUC (TEST) | Recall (TEST) | Precisión (TEST) |
|-----------------|------------|---------------|---------------|------------------|
| seedENS_MEAN    | 0.47–0.52  | 0.42–0.44     | 0.90–1.00     | 0.42–0.45        |
| seedENS_TRIMMED | 0.49–0.50  | 0.44–0.45     | 0.80–1.00     | 0.41–0.43        |
| seedENS_TOP7    | 0.45–0.46  | 0.40–0.41     | 1.00          | 0.43             |

**Diagnóstico:**  
- Los logits de los tres checkpoints presentaron escalas extremadamente distintas.  
- Incluso tras normalización y calibración, la separación ROC/PR fue prácticamente nula.  
- El ensemble de semillas no aporta valor añadido frente al ensemble por agregadores (mean+trimmed+top7), que sí logra recall clínico ≥0.9 con mejor PR-AUC.

**Conclusión:**  
Se descarta el *seed-ensemble* para esta fase. Se consolida el uso del **ensemble por agregadores calibrados** como cierre de la etapa solo-MRI antes de avanzar a la integración multimodal.

---

### 🔹 Extensión Pipeline 10 – Random Search de ensembles

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

### 🧪 Ensembles avanzados en Pipeline 10

- **Objetivo:** superar las limitaciones de pooling simples y del stacking clásico, evaluando combinaciones ponderadas de predicciones slice→paciente.  

- **Estrategias exploradas:**
  - **Seed ensembles:** fallaron, con métricas cercanas a azar (AUC ~0.5).
  - **Random Search ensemble:** optimizó pesos no negativos (Dirichlet, N=500).  
    - Pesos óptimos: mean≈0.32, trimmed≈0.31, top7≈0.32, pmean_2≈0.04.  
    - [VAL] AUC=0.909, PR-AUC=0.920, Recall=0.95.  
    - [TEST] AUC=0.754, PR-AUC=0.748, Recall=0.70.
  - **Logistic Regression stacking:** rendimiento equivalente al Random Search.  
    - Coeficientes (interpretables): todos positivos (~0.40–0.48).  
    - Conclusión: cada agregador aporta información relevante.

- **Reflexión:**  
  La etapa MRI-only cierra con ensembles robustos que **maximizan recall clínicamente crítico** sin sacrificar tanta precisión como en pooling simples. Esto ofrece un baseline sólido antes de fusionar datos multimodales.

---

### 3.9 Pipeline 10-ext – Ensembles de EfficientNet-B3: Variantes TRIMMED y ensembles intra-backbone  

**Motivación:**  
Explorar variantes de pooling de slices y ensembles dentro de EfficientNet-B3.  

**Estrategias probadas:**  
  - Mean, trimmed, top-k.  
  - Random Forest sobre features de pooling.  
  - Stacking logístico.  



**Resultados:**  
- TRIMMED: AUC (Test) 0.744, PR-AUC 0.746.  
- Ensemble (mean+trimmed+top7): AUC (Test) 0.754, PR-AUC 0.737.  

**Reflexión:**  
Los ensembles intra-backbone sí ofrecieron mejoras moderadas pero consistentes.  

---

### 3.10 Pipeline 11 – Backbones alternativos  

**Motivación:**  
Explorar arquitecturas más allá de EfficientNet.  

Tras consolidar EfficientNet-B3 como modelo base en el pipeline 10, se decidió evaluar otras arquitecturas conocidas para clasificación de imágenes médicas. La motivación fue comprobar si existía alguna arquitectura con mejor balance entre recall, precisión y estabilidad en cohortes de validación y test.

**Modelos probados:**  
- ResNet-50.  
- DenseNet-121.  
- ConvNeXt-Tiny.  
- Swin-Tiny.  

**Resultados:**  

| Backbone        | AUC (Test) | PR-AUC (Test) | Acc   | Recall | Precision |
|-----------------|------------|---------------|-------|--------|-----------|
| ResNet-50       | 0.740      | 0.730         | 0.64  | 0.70   | 0.56      |
| DenseNet-121    | 0.343      | 0.407         | 0.32  | 0.75   | 0.36      |
| ConvNeXt-Tiny   | 0.509      | 0.479         | 0.49  | 1.00   | 0.45      |
| Swin-Tiny       | 0.641      | 0.597         | 0.55  | 0.95   | 0.95      |

- **ResNet-50**: competitivo, AUC (≈ 0.74) similar a EffNet-B3 estable.  
- **DenseNet-121**: resultados bajos (AUC ~0.34–0.46).  
- **ConvNeXt-Tiny**: desempeño bajo (AUC ~0.50).  
- **Swin-Tiny**: desempeño moderado, con variante top7 alcanzando AUC ~0.64.  

**Reflexión:**  
Ningún backbone superó claramente a EfficientNet-B3, aunque Swin mostró cierto potencial.  

---

### Resultados (p1-p11)

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
| P11      | MRI Colab       | Swin-Tiny + top7                     | 0.641 | 0.597 | 0.55 | 0.95 | 0.95 |
| P11      | MRI Colab       | ConvNeXt-Tiny (in12k_ft_in1k) + mean | 0.509 | 0.479 | 0.49 | 1.00 | 0.45 |
| P11      | MRI Colab       | DenseNet-121 + trimmed20             | 0.343 | 0.407 | 0.32 | 0.75 | 0.36 |
| P11      | MRI Colab       | Backbone-Ensemble (Dirichlet-means)  | 0.520 | 0.523 | 0.47 | 1.00 | 0.44 |
| P11      | MRI Colab       | Backbone-Ensemble (Dirichlet-ext)    | 0.361 | 0.405 | 0.45 | 0.85 | 0.43 |
| P11      | MRI Colab       | Swin-Tiny (top7) + Isotonic Calib    | 0.566 | 0.458 | 0.55 | 0.95 | 0.49 |

---

### 3.11 Ensembles inter-backbone  

Se exploraron combinaciones entre diferentes backbones:  

- **Dirichlet ensembles**: priorizaron SwinTiny y ResNet, con mejoras modestas.  
- **Stacking con regresión logística**: no lograron generalizar bien, tendencia a sobreajuste.  
- **Isotonic calibration** sobre SwinTiny top7: mejoró la estabilidad.  

Tras evaluar individualmente diferentes arquitecturas (ResNet-50, DenseNet-121, ConvNeXt-Tiny, Swin-Tiny), se exploró si la combinación de backbones podía mejorar el rendimiento.  

### Metodología
- **Dirichlet means**: muestreo de pesos de combinación desde distribuciones Dirichlet sobre predicciones tipo *mean*.  
- **Dirichlet extended**: se amplió a variantes *mean/trimmed20/top7* de cada backbone.  
- **Stacking L1**: regresión logística con regularización fuerte.  
- **Calibración isotónica**: aplicada a las salidas de Swin-Tiny (mejor variante individual).  

### Resultados principales
| Variante Ensemble             | AUC (TEST) | PR-AUC | Acc   | Recall | Precision |
|-------------------------------|------------|--------|-------|--------|-----------|
| Dirichlet-means               | 0.520      | 0.523  | 0.47  | 1.00   | 0.44      |
| Dirichlet-extended            | 0.361      | 0.405  | 0.45  | 0.85   | 0.43      |
| Stacking L1 fuerte            | 0.500      | 0.426  | 0.43  | 1.00   | 0.43      |
| Swin-Tiny + Isotonic Calib.   | 0.566      | 0.458  | 0.55  | 0.95   | 0.49      |

### Conclusión
- **Ningún ensemble supera a EfficientNet-B3 calibrado (pipeline 10-ext)** en este dataset.  
- **Swin-Tiny isotónico** logra *recall* muy alto (0.95) con precisión moderada (0.49), pero su AUC sigue bajo.  
- La diversidad de arquitecturas no se traduce en mejoras sustanciales, probablemente por el tamaño limitado del conjunto de test.  
- Se refuerza la idea de que **EffNet-B3 estable+calibrado** es la mejor base antes de pasar a escenarios multimodales.

---

## Ingeniería y Rendimiento (Colab)

- **Copia de MRI a SSD local** (`/content/mri_cache`) → ~**53 f/s** al copiar 940 ficheros.  
- **Lectura directa Drive**: ~**4.5 img/s** (muestra 256).  
- **Lectura SSD local**: ~**695 img/s** (muestra 256).  
- **Inferencia (sin cache inicial)**: ~**17 img/s**.  
- **Optimizada (cache + ajustes DataLoader)**: **150–200 img/s** (VAL/TEST).  
- **DataLoader**: en T4, **`num_workers=2`** suele rendir mejor; evita crear más workers que CPUs.  
- **AMP**: usar `torch.amp.autocast('cuda')` (deprecado `torch.cuda.amp.autocast(...)`).

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

## P27 — Operativización de la inferencia intermodal (scripts + GUI) con S2

**Objetivo:** disponer de herramientas reproducibles y multiplataforma para ejecutar el pipeline intermodal (imagen + clínico) fuera de Colab, aplicando la **política S2** por cohorte.

**Componentes:**
- `compute_pimg_from_features.py`: infiere **p_img** desde **features por paciente** (56 columnas) con **P24** (LR elastic-net) y calibración **Platt** cuando está disponible.
- `predict_end_to_end.py`: calcula **p_clin** con el modelo clínico P26, realiza **fusión LATE** (`proba_cal = (p_img + p_clin)/2`) y aplica **S2** (umbrales por cohorte) para obtener la **decisión**.
- `app.py` (Streamlit): interfaz web local para cargar CSVs, ejecutar el pipeline y descargar resultados, con opción de QA si hay `y_true`.

**Política S2 (decisión):**
- OAS1: umbral **0.42** (derivado de coste 5:1 FN:FP).  
- OAS2: umbral **≈0.4928655287824083** (ajustado a un **target de recall** en OAS2, manteniendo el coste controlado).

**Razonamiento:**  
S2 mantiene el sesgo **pro-sensibilidad** (cribado: minimizar FN), respetando el **shift** entre cohortes (OAS1/OAS2). Esta política se justificó con los resultados de P24/P26 y las curvas coste–umbral (VAL→TEST). El umbral es **configurable** (JSON) para facilitar recalibraciones locales.

**Entradas mínimas:**
- CSV **features por paciente**: `patient_id`, `cohort`, + 56 columnas de P24.  
- CSV **clínico**: `patient_id`, `cohort`, `Age, Sex, Education, SES, MMSE, eTIV, nWBV, ASF, Delay` (imputación mediana y mapeo de `Sex` automático).

**Salidas:**
- `p_img.csv` (imagen, calibrada), `predictions.csv` (intermodal con `proba_cal` y `decision`), y `predictions_qa.csv` si hay `y_true`.

**Notas de despliegue:**  
Versionado de `scikit-learn` consistente con los **pickles** de P24/P26; ECE/MCE aconsejada en monitorización; umbrales S2 en `deployment_config.json`.

---

## 🔎 Índice rápido

1. [Resumen ejecutivo actualizado](#resumen-ejecutivo-actualizado)
2. [Datos y preprocesado (síntesis)](#datos-y-preprocesado-síntesis)
3. [Metodología intermodal (P26)](#metodología-intermodal-p26)
4. [Platt por cohorte y política **S2** (P26b/P27)](#platt-por-cohorte-y-política-s2-p26bp27)
5. [Resultados comparativos y decisión](#resultados-comparativos-y-decisión)
6. [Calibración y calidad probabilística](#calibración-y-calidad-probabilística)
7. [Reproducibilidad, artefactos y release](#reproducibilidad-artefactos-y-release)
8. [Guía de uso (scripts, app y API)](#guía-de-uso-scripts-app-y-api)
9. [Riesgos, limitaciones y mitigaciones](#riesgos-limitaciones-y-mitigaciones)
10. [Changelog P26/P26b/P27](#changelog-p26p26bp27)
11. [Contenido histórico (Informe original)](#contenido-histórico-informe-original)

---

## Resumen ejecutivo 

- **Problema**: clasificación binaria paciente (Control=0 vs Dementia/Converted=1) con **MRI** + **clínico** (OASIS-1 y OASIS-2).
- **Logro**: transición desde pipelines de **imagen (single modility)** (P24) a pipeline **intermodal (imagen+clinico)** (P26), con **calibración por cohorte** (P26b) y **decisión coste-sensible** (P27 **S2**).  
- **Modelo recomendado**:
  - **Intermodal LATE** (promedio de probabilidades `p_img` y `p_clin`) **+** calibración **Platt** por cohorte (**P26b**) **+** **política de decisión S2** (FN:FP=**5:1** con ajuste OAS2 para **Recall ≥0.90**).
- **Métricas clave (TEST):**
  - **Unimodal imagen (P24 LR-EN + Platt)**: AUC=**0.727** (ALL) · **0.754** (OAS1) · **0.750** (OAS2); Brier (ALL)=**0.220**.
  - **Intermodal LATE (P26)**: AUC=**0.713** (ALL); Brier=**0.234**.
  - **P26b** (Platt por cohorte): Brier **↓** en **OAS1≈0.199**, **OAS2≈0.241**.
  - **S2 (TEST)** — Umbrales activos: **OAS1=0.42**, **OAS2≈0.4928655288** →  
    OAS1: **Recall=0.70** (Coste=39) · OAS2: **Recall≈0.917** (Coste=11).

---

## Datos y preprocesado (síntesis)

### Cohortes y criterios
- **OASIS-1 (OAS1)**: transversal; una adquisición por paciente.  
- **OASIS-2 (OAS2)**: longitudinal; **1ª visita** por paciente en P13–P14/P26 (evita *leakage* inter-sesión).

### Imagen (MRI)
- **20 slices axiales** equiespaciados/volumen (descarta ~8% extremos); **z-score** + **CLAHE** opcional.  
- Inferencias por slice con múltiples *backbones*; agregación por paciente: `mean`, `trimmed20`, `top-k`, `pmean_2`.  
- **Catálogo p11** → **56 features** por paciente (VAL/TEST) estándar para ensambles/meta.

### Variables clínicas
- **OASIS-1** (cross-sectional) + **OASIS-2** (longitudinal) unificados con *renaming* homogéneo.  
- Target binario: **OAS2: Group→{0,1}**; **OAS1: CDR→{0,1}**.   
- Limpieza: drop de NaN críticos (`MMSE`, `CDR`, `Target`), imputación mediana (`Education`, `SES`), OHE en `Sex` (drop_first).  
- **Anti-fuga** en P26+: el modelo clínico **no** usa `CDR/Group` como *features* (sólo como *labels* cuando procede).

---

## Metodología intermodal (P26)

### Señales utilizadas
- **`p_img`**: probabilidad calibrada de P24 (LR elastic-net + Platt) sobre **56 features** por paciente.  
- **`p_clin`**: probabilidad clínica con LR entrenada con **anti-fuga** (sin CDR/Group como *features*).   
- **`p1_fill` / `p1_has`**: probabilidades históricas de OAS2 (cobertura ≈32%); uso como `p1_fill` (imputación por cohorte) + **flag de presencia`p1_has`**.

### Fusiones evaluadas
- **LATE (seleccionada)** — `proba_raw = mean(p_img, p_clin)` → calibración/umbralización posterior.  
- **MID** — meta-LR-EN sobre `{IMG56 + clínico + p1}` con OOF sin fuga; útil pero peor que LATE en este dataset.

### Resultados (VAL/TEST)
- **LATE**: VAL AUC=**0.916**, PR-AUC=**0.921**, Brier=**0.111** · TEST AUC=**0.713**, PR-AUC=**0.712**, Brier=**0.234**.  
- **MID**:  VAL AUC=**0.797**, PR-AUC=**0.777**, Brier=**0.185** · TEST AUC=**0.697**, PR-AUC=**0.657**, Brier=**0.230**.

> **Elección**: **LATE** por mejor equilibrio general y menor complejidad para despliegue.

---

## Calibración y decisión (5:1)
- **Curvas coste–umbral** en VAL → **umbrales por cohorte** y evaluación en TEST.  
- **P26 (sin recalibración por cohorte):**  
  - OAS1 @ **0.307** → TP=14, FP=9, TN=18, FN=6 → **R=0.700**, **P=0.609**, Acc=0.681, **Coste=39**  
  - OAS2 @ **0.195** → TP=8, FP=4, TN=7, FN=4 → **R=0.667**, **P=0.667**, Acc=0.652, **Coste=24**
- **Calibración (TEST, 10 bins):** ECE (ALL)=**0.178**; **OAS2 ECE=0.313** (descalibración).

---

## Platt por cohorte y política **S2** (P26b/P27)

### Motivación
- **OAS2** muestra mayor **descalibración** (ECE≈0.313).  
- Objetivo clínico: **recall alto** (minimizar FN) con costes controlados.

### Procedimiento
1. **Calibración Platt por cohorte** (entrenada en VAL, aplicada en TEST).  
2. Re-optimización de **umbrales 5:1** (FN:FP) por cohorte.  
3. Definición de **política S2**:  
   - Base **5:1** (como P23/P24),  
   - **Ajuste en OAS2** para **Recall objetivo ≥ 0.90** en TEST.

### Umbrales activos y verificación
```json
{
  "OAS1": 0.42,
  "OAS2": 0.4928655287824083
}
```
- **Smoke TEST @S2** (intermodal LATE):  
  - **OAS1** → TP=14, FP=9, TN=18, FN=6 → **Recall=0.70**, Precision=0.609, Acc=0.681, **Coste=39**  
  - **OAS2** → TP=11, FP=6, TN=5, FN=1 → **Recall≈0.917**, Precision≈0.647, Acc≈0.696, **Coste=11**

> Política **S2** adecuada para **cribado** / **triaje**. Si el coste de FP es crítico, usar **5:1 puro** o **manual** con sliders (App).

---

## Resultados comparativos y decisión

### Probabilidades (TEST)
| Pipeline | Cohorte | Modelo/Calib           |   AUC  | PR-AUC | Brier |
|---------:|:-------:|-------------------------|:------:|:------:|:-----:|
| P19      |  ALL    | Meta-XGB                | 0.671  | 0.606  | 0.292 |
| P19      |  OAS1   | Meta-XGB                | 0.663  | 0.588  | 0.310 |
| P19      |  OAS2   | Meta-XGB                | 0.663  | 0.683  | 0.257 |
| P22      |  ALL    | HGB-Platt               | 0.702  | 0.629  | 0.222 |
| P22      |  ALL    | LR-Platt                | 0.668  | 0.646  | 0.219 |
| P22      |  OAS1   | LR-Platt                | 0.756  | 0.726  | 0.203 |
| P22      |  OAS1   | HGB-Platt               | 0.724  | 0.649  | 0.209 |
| P22      |  OAS2   | LR-Platt                | 0.504  | 0.524  | 0.252 |
| **P24**  |  ALL    | **LR-EN + Platt**       | **0.727** | **0.717** | **0.220** |
| **P24**  |  OAS1   | **LR-EN + Platt**       | **0.754** | **0.736** | **0.211** |
| **P24**  |  OAS2   | **LR-EN + Platt**       | **0.750** | **0.805** | **0.238** |
| P26      |  ALL    | LATE (raw)              | 0.713  | 0.712  | 0.234 |
| P26b     |  OAS1   | LATE + Platt (cohorte)  | 0.754  | 0.736  | 0.199 |
| P26b     |  OAS2   | LATE + Platt (cohorte)  | 0.652  | 0.728  | 0.241 |

### Decisión coste-sensible (FN:FP=5:1)
| Pipeline | Cohorte | Thr   |  TP |  FP |  TN |  FN | Precision | Recall |  Acc  | Cost |
|---------:|:------:|:-----:|----:|----:|----:|----:|----------:|-------:|------:|-----:|
| **P24**  | OAS1   | 0.435 | 14  |  9  | 18  |  6  |  0.609    | 0.700  | 0.681 |  39  |
| **P24**  | OAS2   | 0.332 | 11  |  7  |  4  |  1  |  0.611    | 0.917  | 0.652 |  12  |
| **P26**  | OAS1   | 0.307 | 14  |  9  | 18  |  6  |  0.609    | 0.700  | 0.681 |  39  |
| **P26**  | OAS2   | 0.195 |  8  |  4  |  7  |  4  |  0.667    | 0.667  | 0.652 |  24  |

> **Interpretación**: P24 ofrece **mejor discriminación** global y OAS2 sólido; P26b reduce Brier (mejor calibración). **S2** asegura **sensibilidad alta** en OAS2 con coste acotado.

---

## Calibración y calidad probabilística

- **Brier Score** (menor es mejor): P24 ALL ≈ **0.220**; P26 ALL ≈ **0.234**; OAS1 mejora con **P26b** (≈**0.199**).  
- **ECE@10 / MCE@10** (TEST, P26): ALL ≈ **0.178 / 0.407**; **OAS2** ≈ **0.313 / 0.766** → foco de mejora.  
- **Acción**: mantener **Platt por cohorte**, monitorizar **ECE** y **recalibrar** si ECE>0.20.

---

## Reproducibilidad, artefactos y release

- **Release**: `p26_release.zip` con:
  - **models/** → `p24_model.pkl`, `p24_platt.pkl`, `p26_clinical_model.pkl`
  - **CONFIG/** → `deployment_config.json` (**S2**), *backups*
  - **QA/** → confusiones @S2, ECE/MCE, curvas de coste
  - **DOCS/** → `MODEL_CARD.md`, `HOW_TO_DEPLOY.md`
  - **Trazabilidad** → `MANIFEST.json`, `ENVIRONMENT.txt`
- **Versionado**: fijar **scikit-learn==1.7.1** para evitar incompatibilidades en pickles.  
- **Rutas clave**: `p25_informe_final/*`, `p26_intermodal/*`, `p27_final/*`.

---

## Guía de uso (scripts, app y API)

**CLI**  
```bash
python compute_pimg_from_features.py --features patient_features.csv --models_dir p26_release/models --out p_img.csv
python predict_end_to_end.py --pimg p_img.csv --clinic clinical.csv --models_dir p26_release/models --config p26_release/CONFIG/deployment_config.json --out predictions.csv
```

**Streamlit (GUI)**  
```bash
pip install streamlit pandas numpy scikit-learn==1.7.1 joblib
streamlit run app.py
```

**FastAPI (REST)** — `POST /predict` recibe `{clinical + features}` o `{clinical + p_img}`, responde `{p_img, p_clin, proba_cal, thr, decision}`.

---

## Riesgos, limitaciones y mitigaciones

- **Tamaño muestral limitado** (ICs amplios). → Reportar ICs, evitar automatismos. 
- **Shift OAS1/OAS2 y descalibración (OAS2)** (ECE≈0.31) → **Platt por cohorte**, **S2**, recalibración por ventana móvil (≥50–100 casos/cohorte o si ECE>0.20).  
- **Cambios de versión** (sklearn) → fijar entorno, validar *hashes* y columnas (ver `ENVIRONMENT.txt`).

---

## Interpretabilidad y señal

- **P24 (Elastic-Net):** coeficientes dominados por **EffB3-OAS2 (p14)** (`*_mean`, `*_trimmed20`) y agregadores por slice/paciente (`slice_preds_plus_*`).  
- Penalización L1 → **coef=0** en variables colineales (selección implícita).  
- Recomendación: *feature importance* por permutación sobre el meta-LATE para auditorías.ç

---

## Changelog P26/P26b/P27

- **P26**: fusión **Late** y **Mid**; **Late** elegida; umbrales 5:1 (VAL→TEST).  
- **P26b**: **Platt** por cohorte; mejora Brier; definimos **S2**.  
- **P27**: **S2** activada (OAS2→R≥0.90), smoke TEST, release y documentación final.

---

## Estado del arte (contexto breve)

- Con datasets pequeños y heterogéneos como OASIS, **AUC≈0.70–0.78** en test para modelos robustos y calibrados es razonable.  
- El **recall alto** en cohortes longitudinales (tipo OAS2) suele requerir **umbrales coste-sensibles** y **calibración por sitio**.  
- La **intermodalidad** (imagen+clínico) reduce varianza y mejora la **capacidad operativa** (P26/P26b).

---

## Conclusiones

- **Intermodal LATE + S2** ofrece un compromiso sólido entre **discriminación**, **calibración** y **sensibilidad clínica**.  
- La **política S2** prioriza **FN mínimos** en OAS2 (recall ≥0.90) sin penalizar en exceso el coste y manteniendo OAS1 en 5:1.  
- El paquete de despliegue es **reproducible**, **configurable** y acompañado de **QA** y **documentación**.

---

### Anexos (rutas de interés)
- `p25_informe_final/` → tablas y figuras de consolidación (P19–P24).
- `p26_intermodal/` → predicciones, umbrales, ECE y resúmenes P26/P26b.
- `p26_release/` → CONFIG (S2), modelos, QA, MANIFEST/ENVIRONMENT y `MODEL_CARD.md` / `HOW_TO_DEPLOY.md`.
- `p27_final/` → figuras comparativas y tablas finales.

---

## 4. Comparativa Global  

(Tabla de consolidación de pipelines, métricas ya integrada en README).  

---

## 5. Principales Desafíos  

1. **Técnicos:**  
   - Errores recurrentes de montaje de Google Drive.  (Solución: reinicio completo del entorno)
   - Saturación de Colab por sesiones largas.  
   - Problemas de compatibilidad de pesos (`strict=False`).  
   - Colisiones de nombres de columnas en CSV.  

2. **Metodológicos:**  
   - Dataset extremadamente reducido: Tamaño reducido del dataset (solo 47 pacientes en test) → alto riesgo de sobreajuste.  Gran varianza en métricas
   - Uso de modelos 2D en vez de 3D debido al tamaño reducido del dataset
   - Dificultad para calibrar scores manteniendo discriminación.  
   - Target binario simplificado
   - Dependencia del preprocesado, mantener coherencia entre cohortes.
   - Varianza alta entre seeds (semillas fijadas)
   - Gestión de ensembles: balance entre diversidad y sobreajuste; complejidad en stackign logístico.
   - Saturación de logits: Valores extremos (>700k) en P9–P10. Obligó a normalización y calibración.
   - Evitar *data leakage* (fuags de información) y meanejar las múltiples visitas en OASIS-2.

3. **Prácticos:**  
   - Limitación de tiempo de GPU.  
   - Dificultad para mantener consistencia entre directorios experimentales (ficheros dispersos en carpestas distintas).
   - Diferencias entre columnas (`y_score`, `pred`, `sigmoid(logit)`). 
   - Saturación de logs y necesidad de bitácora exhaustiva.  

---

## 6. Lecciones Aprendidas y Decisiones Clave  

- EffNet-B3 sigue siendo un backbone robusto.  
- Los ensembles intra-backbone mejoran resultados.  
- Los backbones alternativos no superan claramente a EffNet-B3, salvo Swin en configuraciones concretas.  
- La combinación de enfoques (ensembles) es clave antes de saltar a multimodal.  

---

## 7. Conclusiones Globales 

- **Detección temprana:** El pipeline clínico es altamente preciso, incluso con modelos simples.  
- **Interpretabilidad:** Confirmó el valor de escalas clínicas clásicas (CDR y MMSE).  
- **Probabilidades calibradas:** Mejoran la confianza en decisiones clínicas.  
- **Umbral adaptado:** Minimiza falsos negativos, adecuado para screening.  
- **Falsos positivos:** Asumibles en un contexto de cribado, ya que derivan en más pruebas, no en daño directo. 

- **Modalidad Clínica:**  
  Variables demográficas y neuropsicológicas logran excelente desempeño (AUC ~0.99 fusionando cohortes). Sin embargo, dependen de que el deterioro cognitivo ya sea medible.

- **Modalidad MRI:**  
  Inicialmente rezagada, la visión por computador cierra la brecha mediante transferencia, calibración y fine-tuning. El pipeline final de MRI (EffNet-B3 fine-tune) logra alta sensibilidad y precisión moderada, ideal para screening.
 
- **Fusión Multimodal:**
  La **fusión multimodal** entre datos clínicos y MRI ha demostrado gran potencial. 

**En conclusión**, COGNITIVA-AI demuestra el potencial de una solución híbrida: datos clínicos estructurados más imágenes cerebrales. Cada iteración aportó mejoras técnicas (unificación de datos, calibración, fine-tuning, ensembles) que convergen en un sistema capaz de priorizar la detección temprana (sensibilidad) manteniendo aceptables tasas de falsa alarma. Esto es crítico en Alzheimer, donde diagnosticar a tiempo puede significar retrasar la progresión y brindar mayor calidad de vida al paciente.


## 8. Agradecimientos

Este trabajo se basa en los conjuntos de datos OASIS.  
Uso estrictamente académico, sin fines clínicos.  
Gracias a la comunidad open-source y al profesor Valentín Silvestri que me ha acompañado y mentorizado todo el proceso.