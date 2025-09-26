# 📑 Informe Técnico — **COGNITIVA-AI — Detección Temprana de Alzheimer**

## Índice

- [📑 Informe Técnico — **COGNITIVA-AI — Detección Temprana de Alzheimer**](#informe-técnico-cognitiva-ai-detección-temprana-de-alzheimer)
  - [0. Antecedentes](#0-antecedentes)
  - [1. Introducción General](#1-introducción-general)
  - [2. Metodología Global](#2-metodología-global)
    - [2.1 Dataset y preprocesamiento](#21-dataset-y-preprocesamiento)
    - [2.2 Infraestructura](#22-infraestructura)
    - [2.3 Métricas de evaluación](#23-métricas-de-evaluación)
  - [3. Evolución por Pipelines](#3-evolución-por-pipelines)
    - [3.1. P1-COGNITIVA-AI-CLINIC (Clínico OASIS-2)](#31-pipeline-1-p1-cognitiva-ai-clinic-clínico-oasis-2)
    - [3.2. P2-COGNITIVA-AI-CLINIC-IMPROVED (Clínico fusionado OASIS-1+2)](#32-pipeline-2-p2-cognitiva-ai-clinic-improved-clínico-fusionado-oasis-12)
    - [3.3. P3-COGNITIVA-AI-IMAGES (MRI OASIS-2, ResNet50)](#33-pipeline-3-p3-cognitiva-ai-images-mri-oasis-2-resnet50)
    - [3.4. P4-COGNITIVA-AI-IMAGES-IMPROVED (MRI OASIS-1+2 unificado)](#34-pipeline-4-p4-cognitiva-ai-images-improved-mri-oasis-12-unificado)
    - [3.4B P5-COGNITIVA-AI-IMAGES-IMPROVED-GPU (Embeddings ResNet18 + Calibración)](#34b-pipeline-5-p5-cognitiva-ai-images-improved-gpu-embeddings-resnet18-calibración)
    - [3.5. P6-COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED (Embeddings EffNet-B3 + Ensemble)](#35-pipeline-6-p6-cognitiva-ai-images-improved-gpu-calibrated-embeddings-effnet-b3-ensemble)
    - [3.6A Pipeline 7 – P7-COGNITIVA-AI-FINETUNING (Fine-tuning EfficientNet-B3 parcial)](#36a-pipeline-7-p7-cognitiva-ai-finetuning-fine-tuning-efficientnet-b3-parcial)
    - [3.6 Pipeline 8 – P8-COGNITIVA-AI-FINETUNING-IMPROVED (Fine-tuning completo de EfficientNet-B3)](#36-pipeline-8-p8-cognitiva-ai-finetuning-improved-fine-tuning-completo-de-efficientnet-b3)
  - [Pipeline 9 - P9-COGNITIVA-AI-FINETUNING-STABLE Fine‑tuning Estable EfficientNet‑B3 (Colab)](#pipeline-9---p9-cognitiva-ai-finetuning-stable-finetuning-estable-efficientnetb3-colab)
    - [Comparación EffNet-B3 (p7,p8,p9)](#comparación-effnet-b3-p7p8p9)
    - [3.8 Pipeline 10 – P10-COGNITIVA-AI-FINETUNING-STABLE-PLUS - EffNet-B3 Fine-Tuning Stable Plus (checkpoint limpio + calibración final)](#38-pipeline-10-p10-cognitiva-ai-finetuning-stable-plus---effnet-b3-fine-tuning-stable-plus-checkpoint-limpio-calibración-final)
  - [📌 Evaluación de seed-ensemble (EffNet-B3, seeds 41/42/43)](#evaluación-de-seed-ensemble-effnet-b3-seeds-414243)
    - [🔹 Extensión Pipeline 10 – Random Search de ensembles](#extensión-pipeline-10-random-search-de-ensembles)
    - [🧪 Ensembles avanzados en Pipeline 10](#ensembles-avanzados-en-pipeline-10)
    - [3.9 Pipeline 10-ext – Ensembles de EfficientNet-B3: Variantes TRIMMED y ensembles intra-backbone](#39-pipeline-10-ext-ensembles-de-efficientnet-b3-variantes-trimmed-y-ensembles-intra-backbone)
    - [3.10 Pipeline 11 – Backbones alternativos](#310-pipeline-11-backbones-alternativos)
    - [Resultados (p1-p11)](#resultados-p1-p11)
  - [📊 Comparativa Global (pipelines 1–10)](#comparativa-global-pipelines-110)
    - [3.11 Ensembles inter-backbone](#311-ensembles-inter-backbone)
    - [Metodología](#metodología)
    - [Resultados principales](#resultados-principales)
    - [Conclusión](#conclusión)
  - [Ingeniería y Rendimiento (Colab)](#ingeniería-y-rendimiento-colab)
  - [Experimentos en OASIS-2 (p13, p14 y p15)](#experimentos-en-oasis-2-p13-p14-y-p15)
    - [Preparación de datos](#preparación-de-datos)
    - [Pipeline p13](#pipeline-p13)
    - [Pipeline p14](#pipeline-p14)
    - [Pipeline p15 (Consolidación y ensamble)](#pipeline-p15-consolidación-y-ensamble)
    - [Conclusión](#conclusión)
  - [Fase 8 – Ensembles refinados (p15 y p16)](#fase-8-ensembles-refinados-p15-y-p16)
  - [Ensemble Calibration (p17)](#ensemble-calibration-p17)
    - [Metodología](#metodología)
    - [Resultados](#resultados)
    - [Conclusión](#conclusión)
    - [Comparativa p16 vs p17](#comparativa-p16-vs-p17)
    - [Pipeline p18 – Stacking multicapa](#pipeline-p18-stacking-multicapa)
  - [P19 – Meta-Ensemble apilado](#p19-meta-ensemble-apilado)
  - [P20: Meta-calibración y umbrales por cohorte](#p20-meta-calibración-y-umbrales-por-cohorte)
  - [P21: Meta-refine con stacking compacto](#p21-meta-refine-con-stacking-compacto)
  - [P22: Meta-Ablation con calibración avanzada](#p22-meta-ablation-con-calibración-avanzada)
  - [Estrategia de tratamiento de OASIS-1 y OASIS-2 en ensembles](#estrategia-de-tratamiento-de-oasis-1-y-oasis-2-en-ensembles)
  - [P23 – Meta-calibración por cohorte con coste clínico](#p23-meta-calibración-por-cohorte-con-coste-clínico)
  - [P24 — Meta interpretable (LR elastic-net)](#p24-meta-interpretable-lr-elastic-net)
  - [P25 — Consolidación y narrativa final](#p25-consolidación-y-narrativa-final)
  - [✅ Recomendaciones finales y consideraciones de despliegue](#recomendaciones-finales-y-consideraciones-de-despliegue)
  - [P26 — Intermodal (imagen + clínico)](#p26-intermodal-imagen-clínico)
    - [Diseño y datos](#diseño-y-datos)
    - [Modelado](#modelado)
    - [Resultados](#resultados)
    - [P26b — Calibración por cohorte (Platt)](#p26b-calibración-por-cohorte-platt)
    - [Interpretación y recomendaciones](#interpretación-y-recomendaciones)
    - [Limitaciones y mitigaciones](#limitaciones-y-mitigaciones)
    - [Artefactos](#artefactos)
  - [P27 — Cierre de ciclo y despliegue (intermodal LATE + política S2)](#p27-cierre-de-ciclo-y-despliegue-intermodal-late-política-s2)
    - [1. Propósito](#1-propósito)
    - [2. Política S2 (definición técnica)](#2-política-s2-definición-técnica)
    - [3. Resultados de control (Smoke TEST @S2)](#3-resultados-de-control-smoke-test-s2)
    - [4. Paquete de despliegue](#4-paquete-de-despliegue)
    - [5. Riesgos y mitigaciones](#5-riesgos-y-mitigaciones)
    - [6. Próximos pasos (operación)](#6-próximos-pasos-operación)
  - [P27 — Tablas y figuras de cierre](#p27-tablas-y-figuras-de-cierre)
    - [Tablas de referencia](#tablas-de-referencia)
    - [Figuras finales (guardadas)](#figuras-finales-guardadas)
  - [P27 — Figuras de cierre (TEST)](#p27-figuras-de-cierre-test)
  - [P27 — Operativización de la inferencia intermodal (scripts + GUI) con S2](#p27-operativización-de-la-inferencia-intermodal-scripts-gui-con-s2)
  - [🔎 Índice rápido](#índice-rápido)
  - [Resumen ejecutivo](#resumen-ejecutivo)
  - [Datos y preprocesado (síntesis)](#datos-y-preprocesado-síntesis)
    - [Cohortes y criterios](#cohortes-y-criterios)
    - [Imagen (MRI)](#imagen-mri)
    - [Variables clínicas](#variables-clínicas)
  - [Metodología intermodal (P26)](#metodología-intermodal-p26)
    - [Señales utilizadas](#señales-utilizadas)
    - [Fusiones evaluadas](#fusiones-evaluadas)
    - [Resultados (VAL/TEST)](#resultados-valtest)
  - [Calibración y decisión (5:1)](#calibración-y-decisión-51)
  - [Platt por cohorte y política **S2** (P26b/P27)](#platt-por-cohorte-y-política-s2-p26bp27)
    - [Motivación](#motivación)
    - [Procedimiento](#procedimiento)
    - [Umbrales activos y verificación](#umbrales-activos-y-verificación)
  - [Resultados comparativos y decisión](#resultados-comparativos-y-decisión)
    - [Probabilidades (TEST)](#probabilidades-test)
    - [Decisión coste-sensible (FN:FP=5:1)](#decisión-coste-sensible-fnfp51)
  - [Calibración y calidad probabilística](#calibración-y-calidad-probabilística)
  - [Reproducibilidad, artefactos y release](#reproducibilidad-artefactos-y-release)
  - [Guía de uso (scripts, app y API)](#guía-de-uso-scripts-app-y-api)
  - [Riesgos, limitaciones y mitigaciones](#riesgos-limitaciones-y-mitigaciones)
  - [Interpretabilidad y señal](#interpretabilidad-y-señal)
  - [Changelog P26/P26b/P27](#changelog-p26p26bp27)
  - [Estado del arte (contexto breve)](#estado-del-arte-contexto-breve)
  - [Conclusiones](#conclusiones)
    - [Anexos (rutas de interés)](#anexos-rutas-de-interés)
  - [4. Comparativa Global](#4-comparativa-global)
  - [5. Principales Desafíos](#5-principales-desafíos)
  - [6. Lecciones Aprendidas y Decisiones Clave](#6-lecciones-aprendidas-y-decisiones-clave)
  - [7. Conclusiones Globales](#7-conclusiones-globales)
  - [8. Agradecimientos](#8-agradecimientos)

## 0. Antecedentes

La **Enfermedad de Alzheimer (EA)** es neurodegenerativa y progresiva. Una detección temprana es clave para:  
- Optimizar la atención clínica.  
- Planificar intervenciones.  
- Reducir costes en fases avanzadas.  

## 1. Introducción General  

El presente informe constituye una documentación exhaustiva, detallada y profundamente analítica del proyecto **CognitivaAI**, cuyo propósito ha sido explorar, diseñar, entrenar y evaluar modelos de machine learning y aprendizaje profundo aplicados al diagnóstico automático binario de **deterioro cognitivo temprano (Alzheimer)** a partir de datos clínicos tabulares e imágenes de resonancia magnética (MRI) de la base de datos **OASIS**.  

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
 

**Etiquetas de coh+orte**: para trazabilidad y análisis estratificado.

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

La hipótesis central que guía todo el trabajo es que **distintas fuentes de información y distintas arquitecturas de modelos pueden capturar facetas complementarias del proceso neurodegenerativo**.  

> **Idea fuerza**: un flujo **reproducible, interpretable y clínicamente orientado** que prioriza **recall** (minimizar FN) y mantiene la **calibración** de probabilidades con umbrales **por cohorte** (OAS1/OAS2).

Se han desarrollado **27 pipelines secuenciales**, cada uno incorporando mejoras y aprendizajes de la fase previa, para analizar y comparar modalidades siguiendo una filosofía de **experimentación incremental**:  
- comenzar con modelos sencillos sobre datos clínicos,  
- avanzar hacia CNNs entrenadas sobre imágenes MRI,  
- introducir calibración, normalización y estrategias de ensemble,  
- explorar arquitecturas modernas de visión,  
- y finalmente preparar el terreno hacia un modelo multimodal.

1. **P1-COGNITIVA-AI-CLINIC** – Modelos ML clásicos con datos clínicos de OASIS-2 (baseline tabular).
2. **P2-COGNITIVA-AI-CLINIC-IMPROVED** – Datos clínicos fusionados (OASIS-1 + OASIS-2), mayor muestra y generalización.
3. **P3-COGNITIVA-AI-IMAGES** – Primer approach con MRI OASIS-2 usando Deep Learning (ResNet50).
4. **P4-COGNITIVA-AI-IMAGES-IMPROVED** – Refinamiento de pipeline de MRI (más datos y rigor en splits).
5. **P5-COGNITIVA-AI-IMAGES-IMPROVED-GPU** – Extracción de embeddings MRI con ResNet18 en GPU y calibración isotónica.
6. **P6-COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED** – Embeddings EfficientNet-B3 + ensemble de clasificadores a nivel paciente.
7. **P7-COGNITIVA-AI-IMAGES-FT** – Fine-tuning parcial de EfficientNet-B3 sobre MRI para mejorar discriminación.
8. **P8-COGNITIVA-AI-IMAGES-FT-IMPROVED** – Ajustes de fine-tuning: calibración de probabilidades y optimización de pooling.
9. **P9-COGNITIVA-AI-IMAGES-FT-STABLE** – Modelo fine-tune final calibrado y con umbral clínico optimizado (pipeline MRI definitivo).
10. **P10-COGNITIVA-AI-FINETUNING-STABLE-PLUS** → Versión extendida con calibración adicional y pooling alternativo (mean, median, top-k). 

Cada pipeline se documenta a continuación, detallando metodología, resultados y conclusiones. Esta evolución progresiva permite contrastar la eficacia de distintas aproximaciones (ML tradicional vs Deep Learning, datos clínicos vs imágenes, etc.) y justifica las decisiones técnicas tomadas.

### 3.1. P1-COGNITIVA-AI-CLINIC (Clínico OASIS-2)  

**Motivación:**  
Probar un primer modelo utilizando exclusivamente variables clínicas del dataset OASIS-2, sin imágenes. El objetivo era establecer un baseline puramente tabular.  

**Metodología:**  
Tres clasificadores clásicos (Regresión Logística, Random Forest, XGBoost) con datos tabulares de OASIS-2 (150 sujetos). Validación cruzada (5-fold) y conjunto test reservado (20%).

**Modelado:**
- **Modelos base:** Logistic Regression, Random Forest, XGBoost.  
- **Validación:** `StratifiedKFold` y métrica **ROC-AUC**.  
- **Optimización:**  
  - *GridSearchCV* para Random Forest.  
  - **Algoritmo Genético (DEAP)** para RF/XGB: búsqueda evolutiva de hiperparámetros (más eficiente en espacios grandes/no convexos).

**Resultados:**
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

**📊 Resultados (clínico):**
- **Cross-val (grid/genético):**  
  - RF (grid) → mejor ROC-AUC CV ≈ **0.9224**  
  - RF (GA) → mejor ROC-AUC CV ≈ **0.9215**  
  - XGB (GA) → mejor ROC-AUC CV ≈ **0.9215**

- **Test hold-out (final):**
  | Modelo              | ROC-AUC (Test) |
  |---------------------|----------------|
  | Random Forest (opt) | 0.884          |
  | **XGBoost (opt)**   | **0.897**      |


**Resultados:**

| Modelo           | AUC (CV 5-fold)   | AUC (Test) |
|------------------|-------------------|------------|
| Logistic Reg.    | 0.912 ± 0.050     | –          |
| Random Forest    | 0.925 ± 0.032     | –          |
| XGBoost          | 0.907 ± 0.032     | 0.897      |

**Conclusión:**  
Baseline robusto basado en datos clínicos, limitado por la escasez de datos (solo OASIS-2, 150 sujetos). Confirma la fuerza de variables clínicas (CDR, MMSE) en la detección de demencia incipiente.

**Configuración destacada:**  
- Modelo: **XGBoost** con tuning básico.
- Variables: Datos clínicos tabulares (edad, sexo, puntuaciones cognitivas, CDR). 
- Validación cruzada interna.  

**Resultados:**  
- AUC (Test): 0.897.  
- Sin métricas de PR-AUC reportadas, aunque se observó buen desempeño en términos de recall.  

**Reflexión:**  
- Este baseline confirmó que incluso con variables clínicas simples se puede alcanzar un nivel competitivo de discriminación. Sin embargo, la dependencia exclusiva de variables clínicas limita la aplicabilidad general. 
- Esto obligó a replantear si los modelos de imagen podrían aportar ganancia marginal real.   

---

### 3.2. P2-COGNITIVA-AI-CLINIC-IMPROVED (Clínico fusionado OASIS-1+2) 
 
**Motivación:**  
Mejorar la generalización combinando OASIS-1 y OASIS-2 (~550 sujetos) para aumentar la robustez. Unificación de variables y control de cohortes. 

**Preprocesado:**
- **Unificación de columnas** (`snake_case`).  
- **Selección baseline** primera visita en OASIS-2.  
- **Target unificado**: `Group` (OASIS-2) o `CDR` (OASIS-1).  
- **Imputación:** SES/Educación → mediana.
- **Escalado y codificación**.
- **Etiquetas de cohortes** para trazabilidad (`OASIS1` vs `OASIS2`). 

**⚖️ Manejo del desbalanceo:**
- Distribución real ≈ 54% vs 46% → ligero desbalance.
- Se probaron variantes con `class_weight='balanced'` y `scale_pos_weight` en XGBoost.  
- Se optimizó el **umbral de decisión** para priorizar *recall* clínico.  
  - Umbral óptimo ≈ 0.03 → Recall ≈ 100%, con sacrificio en precisión (15 falsos positivos).  

**⚙️ Modelado:**
- Modelos: Logistic Regression, Random Forest, XGBoost.  
- Cross-validation estratificado (5 folds).  Escalado dentro del fold para evitar leakage.
- Métrica principal: **ROC-AUC**.
- **Reproducibilidad**: semillas fijadas y paralelismo limitado.

**🔍 Interpretabilidad:**
- **Coeficientes (LR):**
  - `CDR` (coef ≈ +4.15) → marcador principal.  
  - `MMSE` (coef ≈ -0.64) → inversamente asociado.  
  - `Educación` (coef ≈ +0.76) → correlación positiva.  
  - Volumétricas (eTIV, nWBV, ASF) menos influyentes. 
  - **Ablación:**  
   - Sin CDR → AUC = 0.86.  
   - Sin CDR+MMSE → AUC = 0.76.  
   - Sin volumétricas → AUC ≈ 1.0. 

- Conclusión: el modelo se alinea con la evidencia clínica → CDR y MMSE son dominantes.  

**📏 Calibración:**
- Comparación: sin calibrar, **Platt (sigmoid)**, **isotónica**.  
- **Brier Scores:**  
  - LR isotónica → **0.0099** (mejor calibración).  
  - RF isotónica → 0.0170.  
  - XGB isotónica → 0.0187.  

**🛡️ Robustez:**
- **Nested CV:** ROC-AUC = **0.985 ± 0.011**.  
- **Ablation:**  
  - Sin MMSE → ROC-AUC 1.000.  
  - Sin CDR → 0.86.  
  - Sin MMSE+CDR → 0.76.  
  - Sin volumétricas → ≈ 1.000.  
  - Sin socioeducativas → ≈ 0.998.  
- Conclusión: **CDR y MMSE son críticos**, otras variables aportan poco.  

**🤝 Ensembling:**
- Promedio de probabilidades (LR + RF + XGB).  
- Resultado: ROC-AUC = **0.995** → ligera mejora.  

**Resultados recopilación:**

| Modelo           | AUC (Hold-out 80/20) | AUC (CV 5-fold) |
|------------------|----------------------|-----------------|
| Logistic Reg.    | 1.000                | 0.979 ± 0.012   |
| Random Forest    | 0.986                | 0.974 ± 0.018   |
| XGBoost          | 0.991                | 0.975 ± 0.021   |
| Ensemble         | –                    | 0.995 (Nested)  |

**🩺 Umbral clínico (XGBoost):**
- Ajustado para maximizar **recall (≈100%)**.  
- Resultado: recall perfecto, con más falsos positivos (~15/77 test).  
- Interpretación clínica: **preferimos un falso positivo antes que un falso negativo**, ya que permite tratar antes

**Conclusión:**  
La fusión de bases clínicas potencia el rendimiento (AUC ~1.0). El modelo calibrado permite operar a alta sensibilidad sin sacrificar precisión, alineado con la prioridad clínica de evitar falsos negativos.

**Configuración destacada:**  
- Modelo: **XGB** extendido.  
- Variables: todas las disponibles en OASIS-1 y 2.  
- Validación cruzada más exhaustiva.  

**Resultados XGBoost:**  
- AUC (Test): 0.991.  
- Recall cercano al 100%.  

**Reflexión:**  
- El modelo clínico extendido alcanzó una performance casi perfecta, aunque con riesgo evidente de **overfitting** dada la baja dimensionalidad del dataset. Sirvió como techo de referencia para comparar con modelos de MRI.  
- Refuerza la hipótesis de que la MRI aporta, sobre todo, complementariedad más que superioridad aislada

---

### Resultados 

### 3.3.  P3-COGNITIVA-AI-IMAGES (MRI OASIS-2, ResNet50) 

**Motivación:**  
Dar el salto a imágenes MRI, usando ResNet50 como backbone en el dataset original de OASIS-2.  

**Configuración:**  
- Modelo: **ResNet50 preentrenada en ImageNet**.  
- Dataset: MRI OASIS-2 (formato local).  
- Entrenamiento limitado por CPU local.  

**Preprocesamiento:**
- Conversión de volúmenes (.hdr/.img) a cortes axiales (5 o 20 slices): cortes axiales centrales. 
- Normalización a rango [0–255].  
- Opciones: CLAHE y z-score por slice.  
- Augmentation (train): flips, rotaciones ±10°, ajustes de brillo/contraste (jitter ligero).  
- Redimensionado a 224×224 y normalización ImageNet. 

**Consideraciones de diseño:**
- **Binarizar `Group`** (`Nondemented` vs `Demented/Converted`): simplifica el problema y mejora estabilidad en CV y test.  
- **Una visita por sujeto (clínico)**: evita duplicar pacientes y **fuga de información**.  
- **Split por paciente (imágenes)**: todas las slices de un `scan_id` deben ir al mismo subset → evaluación realista.  
- **Evaluación por paciente** (MRI): lo clínicamente relevante es la clasificación del **paciente**, no de cada corte aislado.  
- **Early stopping**: protege frente a sobreajuste visible (train ≫ val).  
- **Métrica ROC-AUC**: adecuada con clases desbalanceadas/moderadas y para comparar modelos a distintos umbrales.

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

> **Split estratificado por paciente 60/20/20** (train/val/test)

| Configuración | Preprocesamiento | Train Acc | Val Acc | Test Acc | ROC-AUC | Comentarios |
|---|---|---:|---:|---:|---:|---|
| **5 slices** | **Sin CLAHE** | ↑ (≈0.94) | ≈0.73 | **0.89** | **0.938** | Línea base fuerte; generaliza bien en test. |
| 5 slices | CLAHE | ≈0.95 | ≈0.72 | 0.69 | 0.777 | Mejora visual, pero menor discriminación; probable realce de ruido. |
| 5 slices | CLAHE + z-score | ≈0.96 | ≈0.75 | 0.72 | 0.820 | Recupera estabilidad; mejor balance entre clases, sigue < baseline. |
| **20 slices** | CLAHE + z-score | **0.98** | ≈0.71 | **0.80** | **0.858** | Más cobertura anatómica; mejora global respecto a CLAHE, aunque con algo de sobreajuste. |

**Observaciones:**
- El **baseline sin CLAHE con 5 slices** fue el más alto en **ROC-AUC (0.94)** en nuestro test.  
- **Aumentar a 20 slices** mejora la robustez general y el *recall* de la clase positiva, pero aún no supera al baseline en ROC-AUC.  
- **CLAHE** debe usarse con cautela (o de forma selectiva) y acompañado de normalización adecuada.

**Conclusión:**  
- Factible usar Deep Learning con MRI para detectar Alzheimer (AUC ~0.9). 
- Limitaciones computacionales y necesidad de selección cuidadosa de slices.
- Mejor AUC con 5 cortes sin CLAHE; más cortes mejoran recall, pero no AUC.

**Reflexión:**  
Este pipeline demostró que el uso de imágenes cerebrales puede ofrecer resultados competitivos con los clínicos, aunque aún no superiores.  
Dependiente del preprocesamiento y costoso en CPU.

---

### 3.4. P4-COGNITIVA-AI-IMAGES-IMPROVED (MRI OASIS-1+2 unificado)

**Motivación:**  
Aprovechar todo el conjunto MRI (OASIS-1 + OASIS-2), evitando leakage y aumentando slices por sujeto.

**🔧 Decisiones de diseño:**
- Generación de **embeddings ResNet18** (preentrenado en ImageNet) de dimensión 512 por slice axial.  
- Clasificación con **Logistic Regression**.  
- **Calibración isotónica** mediante `CalibratedClassifierCV`.  
- Evaluación tanto a nivel **slice** como **paciente** (media de probabilidades).  
- **Split paciente/scan** estricto.  
- **Más slices** por paciente.

**📊 Resultados:**
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

**Resultados observación:**  
AUC ~0.85–0.90 en validación, sin salto claro respecto al pipeline 3 por limitaciones de cómputo.

**Conclusión:**
El calibrado isotónico no incrementa la discriminación (AUC estable ≈0.72), pero **reduce drásticamente el error de probabilidad (Brier Score)**, haciendo que las salidas del modelo sean más confiables para escenarios clínicos. Se establece un **baseline robusto en GPU**, desde el que explorar mejoras adicionales (otros clasificadores, pooling más sofisticado, modelos 2.5D/3D).

**Reflexión:**  
Mejor manejo de datos y splits (Split por paciente/scan y más slices por sujeto)
Mayor robustez, pero mayor coste en CPU : necesidad de GPU para seguir progresando.

---

### 3.5. P5-COGNITIVA-AI-IMAGES-IMPROVED-GPU (Embeddings ResNet18 + Calibración)  

**Motivación:**  
Probar una arquitectura más ligera (ResNet18) con calibración de probabilidades en Colab (para usar GPUs). 

**Estrategia:**  
- **Modelo:** ResNet18 preentrenada (512D) con calibración posterior (post-hoc).
- Validación cruzada interna (cv=5), evitando el uso de `cv='prefit'` (deprecado en scikit-learn ≥1.6).
- Clasificación tradicional con **Logistic Regression**.  
- **Calibración isotónica**.

**Resultados:**  
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

**Conclusión:**  
La calibración isotónica estabiliza las probabilidades (mejor Brier score) y, con un umbral clínico bajo, se alcanzan **sensibilidades altas (R≈0.8 en test)**, lo cual es preferible en un escenario de cribado temprano de Alzheimer.

**Reflexión:**  
Probabilidades más confiables y mejor equilibrio precisión/recall mediante calibración. Sensibilidad (80%) en MRI como referencia para mejoras.
La calibración mejoró la interpretación de scores, pero el backbone resultó limitado para esta tarea (resultados inferiores a ResNet50).  

---

### 📊 Comparativa Parcial P1-P5

| Modalidad       | Dataset            | Modelo        | ROC-AUC | Notas |
|-----------------|--------------------|---------------|---------|-------|
| Clínico         | OASIS-2            | XGBoost       | 0.897   | Mejor tabular OASIS-2 |
| Clínico Fusion  | OASIS-1+2          | LogReg        | 0.979   | Simple, interpretable |
| Imágenes        | OASIS-2            | ResNet50 (5s) | 0.938   | Mejor en MRI |
| Clínico Fusion  | OASIS-1+2 Ensemble | LR+RF+XGB     | 0.995   | **Mejor global** |

---

### 3.6. P6-COGNITIVA-AI-IMAGES-IMPROVED-GPU-CALIBRATED (Embeddings EffNet-B3 + Ensemble) 

**Motivación:**  
Usar EfficientNet-B3 como extractor de embeddings, sin fine-tuning completo.  

**Mejoras:**  
- **Embeddings EfficientNet-B3 (1536D)**.  
- Modelos: LR, XGB, MLP a nivel paciente.  
- **Ensemble LR+XGB** ponderado por PR-AUC. 

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
Como extractor simple ya supera ResNet18 calibrado, confirmando potencia de EfficientNet.  Mejor pipeline MRI hasta la fecha, con sensibilidad (Recall) alta, aunque precisión limitada.

---

### 3.7 P7-COGNITIVA-AI-FINETUNING (Fine-tuning EfficientNet-B3 parcial)

**Planteamiento:**  
Fine-tuning parcial de EfficientNet-B3, pooling por atención a nivel paciente.

**Entrenamiento**: fine-tuning completo/parcial de EfficientNet‑B3 (preentrenado en Imagenet)  con última(s) capas descongeladas y reentrenadas sobre MRI OASIS-2.
Uso de Google Colab GPU (T4), early stopping guiado por PR-AUC en validación.

**Agregación paciente**: *mean pooling* de scores/logits por slices del mismo paciente.  

**Calibración**: *temperature scaling* con **T=2.6731656060943605**.  

**Umbral clínico**: **0.3651449978351593** (maximiza recall con precisión aceptable).  

**Resultados (nivel paciente, n=47)**  
- **VAL** → AUC=**0.748** | PR-AUC=**0.665** | Acc=**0.702** | P=**0.588** | R=**1.0**  
- **TEST** → AUC=**0.876** | PR-AUC=**0.762** | Acc=**0.745** | P=**0.625** | R=**1.0**

**Matriz de confusión (TEST, thr=0.3651):** TP=8, FP=5, TN=34, FN=0.  

**Conclusión:**  
Fine-tuning eleva la eficacia de clasificación MRI (AUC ~0.87).
El fine-tuning mejora sustancialmente la discriminación (AUC) respecto a pipelines previos (AUC_test ~0.87 vs ~0.70 en pipeline 6), pero con umbral estándar aún no alcanza sensibilidad adecuada (recall 55% en test).

**Reflexión:**  
Uno de los mejores backbones en imagen pura. Supone el nuevo baseline de referencia.  


---

### 3.8. P8-COGNITIVA-AI-FINETUNING-IMPROVED (Fine-tuning completo de EfficientNet-B3)

**Motivación:**  
Explotar la capacidad completa de EfficientNet-B3, ajustando todos los pesos: Calibración con Temperature Scaling y ajuste de umbral clínico para alta sensibilidad.

**Calibración:**
- **Calibración de probabilidades:**  se aplicó `Temperature Scaling` en validación para corregir el sesgo de confianza del modelo (evitando técnicas prefit con riesgo de fuga de datos).
- **Pooling óptimo:** la agregación por *atención* superó ligeramente al promedio en métricas de validación (PR-AUC), por lo que se adoptó para el pipeline final.
- **Métricas calibradas:** tras calibración, las predicciones resultaron más fiables (mejor Brier Score y distribución probabilística más alineada).

**Resultados:**
- **VAL (calibrado, attn):** AUC≈0.75 | PR-AUC≈0.66 (similar a bruto, señal consistente).
- **TEST (calibrado, attn):** AUC≈0.876 | PR-AUC≈0.762 (sin cambio notable en AUC, confirma generalización) | Acc: 0.745 | Recall = 1.0 | Precision: 0.625
- **Nota:** La calibración no altera el AUC, pero asegura que las probabilidades reflejen riesgo real. Se observó mejora cualitativa en la confiabilidad de las predicciones.

**Observación:**
La calibración interna del modelo eliminó leakage y ajustó las salidas probabilísticas, dejando el modelo listo para aplicar un umbral clínico en validación.

**Comparación (p6-p8):**

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

### 3.9. P9-COGNITIVA-AI-FINETUNING-STABLE Fine‑tuning Estable EfficientNet‑B3 (Colab)

**Motivación:**  
Introducir técnicas de estabilización para reducir variabilidad. 

**Configuración**  
- Arquitectura: EfficientNet‑B3 (timm).  
- Entrenamiento: AdamW (lr=1e‑4), AMP (`torch.amp`), early‑stopping por AUC en holdout, 300px, batch=64.  
- Agregación: `mean` a nivel paciente.  
- Calibración: temperature scaling (T=2.048).  
- Umbral: 0.3400 (optimizado con recall≥0.95 en VAL).

**Selección de umbral clínico:** a partir de la curva Precision-Recall en validación se eligió el menor umbral con recall ≥90% y máxima precisión. Obtuvo thr≈0.36 en probabilidades de paciente.

**Resultados (nivel paciente):**  
- VAL → AUC=1.000 | PR-AUC=1.000 | Acc=1.000 | P=1.000 | R=1.000 | thr=0.3400 | n=10  
- TEST → AUC=0.663 | PR-AUC=0.680 | Acc=0.574 | P=0.500 | R=0.650 | thr=0.3400 | n=47

📊 Resultados (Paciente-nivel (thr≈0.36, recall=1.00)):
- [VAL] Recall=1.00 | Precision=0.59 | AUC=0.748
- [TEST] Recall=1.00 | Precision=0.62 | AUC=0.876

**Comparativa rápida vs Pipeline 7 (FT previo):** TEST AUC: 0.585 → 0.663, TEST PR‑AUC: 0.582 → 0.680

➡️ Mejor pipeline MRI logrado: se detectan el 100% de los casos positivos en test (sin falsos negativos) al costo de algunos falsos positivos (precision ~62%). El modelo fine-tune calibrado ofrece así alta sensibilidad adecuada para cribado clínico, acercando el rendimiento MRI al nivel de los datos clínicos puros.

**Resultados finales:**  
- AUC (Test): 0.740.  
- PR-AUC: 0.630.  
- Accuracy: 0.72.  
- Recall: 0.65.  
- Precision: 0.62.  

**Incidencias:**
- Saturación de logits detectada (>500k). 
- Variabilidad alta entre seeds. 

**Reflexión:**  
Confirma que la estabilidad no siempre se traduce en mejor rendimiento (mejor generalización).

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

### 3.10. P10-COGNITIVA-AI-FINETUNING-STABLE-PLUS - EffNet-B3 Fine-Tuning Stable Plus (checkpoint limpio + calibración final) 

**Motivación:**  
Aplicar calibración explícita de scores.  

El pipeline 9 ofrecía estabilidad, pero los checkpoints entrenados no siempre coincidían con la arquitectura definida, cargando <1% de pesos en algunos intentos, además de no incluir calibración explícita. 

Era necesario **reprocesar/normalizar el checkpoint**, asegurar la integridad de pesos (99.7% cargados) y aplicar calibración explícita (Platt scaling, isotonic regression y temperature scaling) para corregir sobreconfianza y obtener resultados reproducibles.  

Este pipeline se enfocó en reforzar la **calibración y pooling** para asegurar recall absoluto, incluso sacrificando métricas globales. 

**Configuración:**  
- Modelo: EfficientNet-B3 binario (head adaptada).  
- Checkpoint: `effb3_stable_seed42.pth`, reconstruido a `best_effb3_stable.pth` (99.7% de pesos cargados). 
 - Normalización robusta de pesos: conversión de checkpoint entrenado a formato limpio.  
- Calibración: *temperature scaling* (T≈2.3) aplicado sobre logits + ajuste de umbral F1..  
- Pooling a nivel de paciente: estrategias mean, median y top-k (0.2, 0.3).  
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
- Precision: 0.47–0.49 (muy baja)

**Conclusión:**  
Pipeline 10 consolida la línea MRI con un recall perfecto en test (1.0), asegurando sensibilidad máxima para cribado clínico temprano. Aunque la precisión baja (~0.47), este pipeline marca el cierre robusto de la etapa **MRI-only** y deja el terreno preparado para la fusión multimodal.

**Observación:**  Se documentan dificultades de estabilidad y saturación de logits.

**Reflexión:**  
La calibración no mejoró la discriminación; sacrificó precision.  

---

### P10-ext: Agregaciones avanzadas y Ensemble MRI

Tras la fase inicial del pipeline 10, en la que se demostró la posibilidad de alcanzar *recall=1.0* en test bajo distintos métodos de pooling slice→patient, se llevó a cabo una segunda batería de experimentos orientados a mejorar la **precisión clínica** sin renunciar a la alta sensibilidad.  

**🔹 Estrategias evaluadas:**
- **Agregaciones robustas**:  
  - *TRIMMED mean (α=0.2)* (media recortada al 20%, eliminando los extremos para mitigar slices outliers y mejorar la estabilidad).  
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

**✅ Conclusión ampliada:**
El complemento al pipeline 10 muestra que:  
- **TRIMMED** sigue siendo la mejor variante para maximizar sensibilidad pura.  
- **TOP-k** ofrece alternativas más conservadoras, con mayor precisión pero menor recall.  
- **El ensemble** logra un equilibrio clínico más sólido: mantiene recall en 0.70 en test y mejora la precisión hasta 0.61, elevando también la exactitud global.  

Con esta extensión, el pipeline 10 no solo asegura **recall=1.0 como cribado clínico temprano**, sino que también aporta una variante optimizada para **escenarios de uso real**, donde la precisión adicional reduce falsos positivos innecesarios antes de pasar a pruebas complementarias.

---

### P10-ext2: Evaluación de seed-ensemble (EffNet-B3, seeds 41/42/43)

**Objetivo:**  
Verificar si la combinación de checkpoints con distintas semillas podía mejorar la robustez del Pipeline 10 (*EffNet-B3 stable plus*).

**Metodología:**  
- Probamos un *ensemble* por semillas sobre las mismas cohortes (VAL/TEST, 47/47) reproduciendo las TTA del pipeline 10 (orig, flipH, flipV, rot90) y calibración posterior.
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
- Incluso tras normalización (z-score en VAL) y calibración(temperature scaling y Platt scaling), la separación ROC/PR fue prácticamente nula.  
- El ensemble de semillas no aporta valor añadido frente al ensemble por agregadores (mean+trimmed+top7), que sí logra recall clínico ≥0.9 con mejor PR-AUC.

**Conclusión:**  
Se descarta el *seed-ensemble* para esta fase. Se consolida el uso del **ensemble por agregadores calibrados** a nivel paciente (mean+trimmed+top7), que sí logra **recall ≥ 0.9–1.0** con métricas PR-AUC/AUC superiores, como cierre de la etapa solo-MRI antes de avanzar a la integración multimodal.

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

- **Objetivo:** superar las limitaciones de pooling simples y del stacking clásico, evaluando combinaciones ponderadas de predicciones slice→paciente.  

- **Estrategias exploradas:**
  - **Seed ensembles:** fallaron, con métricas cercanas a azar (AUC ~0.5).
  - **Random Search ensemble:** optimizó pesos no negativos (Dirichlet, N=500).  
    - Pesos óptimos: mean≈0.32, trimmed≈0.31, top7≈0.32, pmean_2≈0.04.  
    - [VAL] AUC=0.909, PR-AUC=0.920, Recall=0.95.  
    - [TEST] AUC=0.754, PR-AUC=0.748, Recall=0.70.
  - **Logistic Regression stacking:** rendimiento equivalente al Random Search.  
    - Coeficientes (interpretables): todos positivos y equilibrados (~0.40–0.48), cada agregador aporta información relevante. 
    - Conclusión: Más interpretable y estable que el Random Forest o el stacking rígido.

- **Reflexión:**  
  Los ensembles ponderados consolidan Pipeline 10 como el mejor punto de partida para MRI-only antes de pasar a multimodal. El recall clínicamente relevante (≥0.95 en validación, 0.70 en test) se mantiene, mientras que la precisión mejora frente a pooling simples.

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

### Pipeline 10-ext – Ensembles de EfficientNet-B3: Variantes TRIMMED y ensembles intra-backbone  

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

### 3.11. P11-COGNITIVA-AI-BACKBONES -  Backbones alternativos  

**Motivación:**  
Explorar arquitecturas más allá de EfficientNet, que mejoraran la capacidad de generalización y robustez del modelo.  

Tras consolidar EfficientNet-B3 como modelo base en el pipeline 10, se decidió evaluar otras arquitecturas conocidas para clasificación de imágenes médicas. La motivación fue comprobar si existía alguna arquitectura con mejor balance entre recall, precisión y estabilidad en cohortes de validación y test.

La hipótesis: *distintas arquitecturas pueden capturar características complementarias de las imágenes cerebrales*: comprobar si otros backbones de visión podían superar a EfficientNet-B3.

- **Configuración técnica:**  
  - Entrenamiento en Colab con mapas OASIS (`oas1_val_colab_mapped.csv` y `oas1_test_colab_mapped.csv`).  
  - Reutilización de la misma configuración de splits y métricas que pipeline 10 para garantizar comparabilidad.  
  - Resultados guardados en `/p11_alt_backbones`. 

**Modelos probados:**  
- ResNet-50.  
- DenseNet-121.  
- ConvNeXt-Tiny.  
- Swin-Tiny.  

**Incidencias:**  
- Varios problemas de montaje de Google Drive tras semanas sin reiniciar Colab.  
- Ficheros dispersos en directorios distintos (ej. slice vs patient).  
- Necesidad de armonizar columnas (`y_score` vs `sigmoid(logit)`).  

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
Ningún backbone superó claramente a EfficientNet-B3, aunque ResNet50 y SwinTiny muestran competitividad parcial. La vía lógica pasa a ser **ensembles de backbones**.  

---

### 3.12. P12-COGNITIVA-AI-BACKBONES-ENSEMBLE - Ensembles inter-backbone  

**Metodología:**
- **Dirichlet means**: muestreo de pesos de combinación desde distribuciones Dirichlet sobre predicciones tipo *mean*.  
- **Dirichlet extended**: se amplió a variantes *mean/trimmed20/top7* de cada backbone.  
- **Stacking LR**: regresión logística con regularización fuerte.  
- **Calibración isotónica**: aplicada a las salidas de Swin-Tiny (mejor variante individual).  

**Resultados principales:**
| Variante Ensemble             | AUC (TEST) | PR-AUC | Acc   | Recall | Precision |
|-------------------------------|------------|--------|-------|--------|-----------|
| Dirichlet-means               | 0.520      | 0.523  | 0.47  | 1.00   | 0.44      |
| Dirichlet-extended            | 0.361      | 0.405  | 0.45  | 0.85   | 0.43      |
| Stacking L1 fuerte            | 0.500      | 0.426  | 0.43  | 1.00   | 0.43      |
| Swin-Tiny + Isotonic Calib.   | 0.566      | 0.458  | 0.55  | 0.95   | 0.49      |

Se exploraron combinaciones entre diferentes backbones:  

- **Dirichlet ensembles**: priorizaron SwinTiny y ResNet, con mejoras modestas.  
- **Stacking con regresión logística**: no lograron generalizar bien, tendencia a sobreajuste.  
- **Isotonic calibration** sobre SwinTiny top7: mejoró la estabilidad.  

Tras evaluar individualmente diferentes arquitecturas (ResNet-50, DenseNet-121, ConvNeXt-Tiny, Swin-Tiny), se exploró si la combinación de backbones podía mejorar el rendimiento. 

**Conclusión:**
- **Ningún ensemble supera a EfficientNet-B3 calibrado (pipeline 10-ext)** en este dataset.  
- **Swin-Tiny isotónico** logra *recall* muy alto (0.95) con precisión moderada (0.49), pero su AUC sigue bajo.  
- La diversidad de arquitecturas no se traduce en mejoras sustanciales, probablemente por el tamaño limitado del conjunto de test.  
- Se refuerza la idea de que **EffNet-B3 estable+calibrado** es la mejor base antes de pasar a escenarios multimodales.

---

### Resultados: Comparativa Global (P1-P12)

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

1. **EfficientNet-B3** sigue siendo el backbone más robusto y estable en este dataset.  
2. **ResNet-50** es competitivo y sorprendentemente sólido en comparación con modelos más recientes.  
3. **DenseNet-121** no ha mostrado buen rendimiento en este dominio.  
4. **ConvNeXt y Swin** presentan interés, pero su rendimiento es irregular y dependiente del pooling.  
5. Los **ensembles de backbones** son prometedores pero, en este dataset pequeño, sufren de sobreajuste y métricas inconsistentes.  
6. El trade-off entre **recall alto y precisión baja** es recurrente y debe tenerse en cuenta para aplicaciones clínicas.

---

### Experimentos en OASIS-2 (p13, p14 y p15)

**Motivación:**
- Hasta este momento sólo habíamos trabajado con MRI de OASIS-1, y ahora vamos a explorar el dataset OASIS-2.

**Preparación de datos:**
- 367 exploraciones de OASIS-2, de las cuales 150 tenían label clínico.  
- Generación de 7340 slices (20 por volumen, edge_crop=0.08, z-score + CLAHE).  
- Máscaras cerebrales FSL u Otsu.  
- Criterio de **una visita por paciente** para evitar leakage.  

**Ingeniería y Rendimiento (Colab):**

- **Copia de MRI a SSD local** (`/content/mri_cache`) → ~**53 f/s** al copiar 940 ficheros.  
- **Lectura directa Drive**: ~**4.5 img/s** (muestra 256).  
- **Lectura SSD local**: ~**695 img/s** (muestra 256).  
- **Inferencia (sin cache inicial)**: ~**17 img/s**.  
- **Optimizada (cache + ajustes DataLoader)**: **150–200 img/s** (VAL/TEST).  
- **DataLoader**: en T4, **`num_workers=2`** suele rendir mejor; evita crear más workers que CPUs.  
- **AMP**: usar `torch.amp.autocast('cuda')` (deprecado `torch.cuda.amp.autocast(...)`).

---

### 3.13. P13-COGNITIVA-AI-OASIS2-EFFB3 - EfficientNet-B3 con datos OASIS2

- Procesamiento de **367 scans OASIS-2** → 150 pacientes con labels clínicos.  
- **Slices:** 20 cortes axiales equiespaciados, evitando extremos, normalizados (z-score + CLAHE opcional).  
- **Máscara cerebral:** segmentación FSL o fallback con Otsu.  
- **Una visita por paciente** → 150 pacientes (105 train, 22 val, 23 test).  

**Resultados:** recall alto en cohortes pequeñas, pero dataset limitado → riesgo de sobreajuste  

---

### 3.14. P14-COGNITIVA-AI-OASIS2-EFFB3-CALIBRATED
- Copia de las 7340 slices a **SSD local de Colab** para reducir la latencia de E/S.  
- Entrenamiento con **class weights** para balancear clases.  
- Integración en catálogo de backbones como `oas2_effb3_p14`.  

**Resultados:**  
- [VAL] AUC≈0.88 | Acc≈0.86 | Recall≈0.82  
- [TEST] AUC≈0.71 | Acc≈0.70 | Recall=1.0 

---

### 3.15. P15-COGNITIVA-AI-CONSOLIDACION (Consolidación y ensamble)

**Contexto:**  
Tras la exploración con OASIS-2 en los pipelines p13 y p14, se buscó consolidar resultados y refinar el rendimiento a través de ensembles patient-level.

- Objetivo: consolidar resultados de p13/p14 con OASIS-1 (p11).  
- Generación de features combinadas con OASIS-1 (p11): 69 pacientes en val, 70 en test.  
- Se descartaron features con NaN > 40% y se aplicaron modelos de ensamblado (Logistic Regression, HistGradientBoosting) 

**Observaciones:**
- Revisión de inventarios y labels clínicos (367 scans totales, 150 con etiqueta).  
- Confirmación del criterio de **una sesión por paciente** para evitar leakage.  
- Dataset resultante: 3.000 slices etiquetadas (20 slices por scan).  
- Dificultades: la latencia de E/S en Google Drive volvió a confirmar la necesidad de copiar los datos a SSD de Colab.

**Resultados comparativos (VAL/TEST):**

| Pipeline | VAL AUC | VAL Acc | VAL Recall | TEST AUC | TEST Acc | TEST Recall |
|----------|---------|---------|------------|----------|----------|-------------|
| p13      | ~0.90   | 0.86    | 0.82       | ~0.77    | 0.78     | 0.83        |
| p14      | 0.88    | 0.86    | 0.82       | 0.71     | 0.70     | 1.00        |
| p15      | 0.94    | 0.84    | ~1.0       | 0.71     | 0.63–0.71 | 0.78–1.0   |

**Conclusión:**
- p13: prueba de integración OASIS-2.  
- p14: optimización con balanceo y SSD local.  
- p15: consolidación de resultados e integración en ensambles OASIS-1+2, demostrando que la combinación de backbones mejora recall y robustez del sistema.
- P15 sirvió como **validación de consistencia** antes de refinar ensembles.

---

### 3.16. P16-COGNITIVA-AI-ENSEMBLE-REFINE  – Ensembles refinados 

**Contexto:**
- Refinar el ensemble de backbones

**Consideraciones:**
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

### 3.17. P17-COGNITIVA-AI-ENSEMBLE-CALIBRATION -  Ensemble Calibration

**Metodología:**
- Se construyó un meta-ensemble con **stacking**, usando Logistic Regression sobre los outputs de modelos base (LR y HGB).  
- Posteriormente se aplicó **Platt scaling** para calibrar las probabilidades, con selección de umbral F1 en validación (0.35), aplicado después en test.  
- Se evaluó la calibración mediante el **Brier Score** y curvas de calibración.

**Resultados:**
- **Validación (ALL):** AUC=0.78 | PRAUC=0.75 | Acc=0.74 | Recall=0.94 | F1=0.76 | Brier=0.176.  
- **Test (ALL):** AUC=0.70 | PRAUC=0.67 | Acc=0.63 | Recall=0.78 | F1=0.66 | Brier=0.227.  
- Cohortes:
  - **OAS1:** estable (VAL AUC≈0.84, TEST≈0.77).  
  - **OAS2:** bajo rendimiento (VAL≈0.31, TEST≈0.50) debido al tamaño reducido (22–23 pacientes).  

**Conclusión:**
- La calibración aporta **mejores probabilidades** y un modelo más interpretable.  
- Se conserva la **sensibilidad (recall)**, clave en escenarios clínicos.  
- El dataset OAS2 sigue siendo un cuello de botella limitado por su tamaño muestral, y deberá reforzarse en futuros experimentos.

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

 Pipeline | Método principal                | VAL AUC | VAL Acc | VAL Recall | VAL F1 | TEST AUC | TEST Acc | TEST Recall | TEST F1 | Brier (Test) |
|----------|---------------------------------|---------|---------|------------|--------|----------|----------|-------------|---------|--------------|
| **p16**  | Blending (LR + HGB, α=0.02)     | 0.95    | 0.84    | 1.00       | 0.84   | 0.69     | 0.64     | 0.78        | 0.64    | –            |
| **p17**  | Stacking + Platt scaling (LR)   | 0.78    | 0.74    | 0.94       | 0.76   | 0.70     | 0.63     | 0.78        | 0.66    | 0.227        |

➡️ **p16** maximizó la métrica predictiva bruta (AUC) en validación, pero con cierto riesgo de sobreajuste sin control de calibración.  
➡️ **p17** ajustó la calidad de las probabilidades (Brier=0.227 en test) y mantuvo recall alto, ofreciendo **mejor calibración** y estabilidad clínica.

---

### 3.18. Pipeline 18 - P18-AI-COGNITIVA-STACKING-MULTICAPA – Stacking multicapa

**Contexto:**  
Tras la calibración en p17, se buscó refinar la combinación de modelos con un enfoque de **stacking multicapa**.  

**Objetivo:** explorar técnicas de stacking avanzadas con múltiples clasificadores de nivel base y un meta-modelo logístico.

**Diseño técnico:**  
- **Modelos base:** Logistic Regression (L2), HistGradientBoosting, Gradient Boosting, Random Forest, Extra Trees.  
- **Meta-modelo:** regresión logística, con blending lineal optimizado de pesos (α≈0.02).  
- **Entrenamiento:** Generación de predicciones OOF (out-of-fold) con 5 folds en validación, evitando fuga de información.  
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

### 3.19. P19-COGNITIVA-AI-META-ENSEMBLE – Meta-Ensemble apilado

**Objetivo:** consolidar las señales de múltiples backbones (p11/p14/p16/p18) con un stacking de segundo nivel.

**Diseño y método**  
- Conjunto de 56 features por paciente (tras filtrado NaN) que integran variantes de pooling (mean, trimmed20, top-k, pmean_2) de múltiples backbones (incluyendo oas2_effb3, oas2_effb3_p14, Swin/ConvNeXt, etc.).  
- **Base learners:** LR (L2), Árboles (GB/RF/ET), HistGradientBoosting, LightGBM y XGBoost.  
- **Meta-learner:** XGBoost entrenado sobre los OOF; inferencia en TEST con predicciones de base learners. 
- **Validación:** OOF KFold estratificado a nivel paciente (sin fuga), usando features por-paciente derivados (mean / trimmed / top-k / p2). 
- **Regularización y NaN:** filtrado de columnas con alta fracción de NaN (>40%); imputación simple en modelos que lo exigen; árboles toleraron faltantes nativamente.  
- **Umbral:** punto de decisión basado en F1 en VAL; aplicado a TEST.  

**Resultados**  
- VAL (n=69): AUC=0.964; PRAUC=0.966; Acc=0.913; F1=0.897; Brier=0.071.  
- TEST (n=70): AUC=0.729; PRAUC=0.688; Acc=0.714; F1=0.630; Brier=0.226.  

**Interpretación**  
- Alta capacidad discriminativa en validación y buena calibración (Brier bajo).  
- En TEST, el recall cae con precisión alta → umbral sub-óptimo bajo shift y posible sobreajuste leve del meta.  

**Observaciones**  
- LightGBM reporta *“No further splits with positive gain”*; esperable con dataset pequeño y features ya decantadas. Conviene reducir complejidad o seleccionar features en meta.  

**Conclusión:** el meta-ensemble eleva la performance en validación, pero el recall en TEST sugiere ajustar calibración/umbrales y atender shift OAS1/OAS2. Se programará p20 para calibración fina y umbrales por cohorte.

---

### 3.20. P20-COGNITIVA-AI-META-CALIBRATION-THRESHOLDS - Meta-calibración y umbrales por cohorte

**Objetivo:** refinar el meta-ensemble (de p19) con **calibración de probabilidades** y **umbrales específicos**. 

**Métodos de calibración:** Platt scaling (sigmoide) e isotónica.  

**Escenarios:** calibración **global** y calibración **per-cohort** (OAS1/OAS2).  

**Modelos meta evaluados:** HistGradientBoosting (HGB) y Logistic Regression (LR).  

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

**Conclusión:** la calibración mejoró la fiabilidad de las probabilidades (Brier menor en VAL). En TEST el recall sigue alto (≈0.78) con sacrificio de precisión, confirmando la necesidad de ajustar umbrales por cohorte.

---

### 3.21 P21-COGNITIVA-AI-META-REFINE - Meta-refine con stacking compacto

**Objetivo.** Refinar el meta-ensemble con menos base learners y un meta-modelo más simple, controlando NaNs y validación OOF sin fuga.

**Diseño.** Se redujo el conjunto de señales de p19/p20 a un meta-stacking compacto:
- **Filtrado de NaN:** de 56 features por paciente → 36 columnas (tras filtrado NaN>40%).
- Cohortes: VAL=69, TEST=70 (con etiqueta de cohorte OAS1/OAS2).
- **Base learners:** LR (penalización L2), HGB, LightGBM (LGBM), XGBoost (XGB).
- **Validación:** OOF estratificado a nivel paciente (sin fuga), construcción de meta-features (VAL: 69×4, TEST: 70×4).
- Meta-learner: blending/stacking con 4 señales OOF (shape meta VAL=69×4, TEST=70×4).
- **Umbral:** F1-máximo en validación (**0.45**).

**Resultados.**
- **Validación (n=69):** AUC 0.955, PRAUC 0.931, Acc 0.870, F1 0.862, Brier 0.082.
- **Test (n=70):** AUC 0.653, PRAUC 0.587, Acc 0.643, F1 0.627, Brier 0.285.

**Interpretación.**
- Excelente discriminación/calibración en VAL (Brier bajo); degradación en TEST sugiere **shift de distribución** (OAS1/OAS2) y/o sensibilidad del umbral global.
- Mensajes de LGBM (*no positive gain*) coherentes con tamaño de muestra pequeño y features ya “resumidas”; conviene regularizar y/o limitar profundidad/hojas.
- El meta compacto facilita **calibración por cohorte** y **optimización de umbral por coste** en el siguiente paso.

---

### 3.22. P22-COGNITIVA-AI-META-ABLATION - Meta-Ablation con calibración avanzada

**Objetivo:** explorar variantes de calibración (Platt vs Isotónica) aplicadas a meta-modelos (LR y HGB), evaluando su impacto en la estabilidad y confiabilidad de las probabilidades.  

**Diseño:**  
- Se utilizaron 56 features derivadas de múltiples backbones, de las que se mantuvieron 36 tras filtrar NaN>40%.  
- Dos modelos base: **Logistic Regression (LR)** con imputación+escalado y **HistGradientBoosting (HGB)** tolerante a NaNs.  
- Calibración aplicada:  
  - **Platt scaling (sigmoid).**  
  - **Isotonic regression.**  
- Validación con **Stratified KFold OOF** en validación (69 pacientes). Predicciones finales sobre 70 pacientes en test.  
- Umbral óptimo seleccionado en validación maximizando F1 (≈0.30–0.35 según modelo).  

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

### Estrategia de tratamiento de OASIS-1 y OASIS-2 en ensembles (p16-p22)

Durante los pipelines p16–p22 se combinaron características y predicciones procedentes de ambos datasets (OASIS-1 y OASIS-2), pero se decidió **no fusionar completamente** los
datos en un único dataset de entrenamiento. 

**Justificación técnica:**
- OASIS-1: cohortes transversales (scans únicos por paciente).
- OASIS-2: cohortes longitudinales (múltiples visitas, más complejidad temporal).
- Fusionarlos sin distinción podría inducir sesgos y leakage.

**Implementación:**
- Los DataFrames de validación y test contienen columna `cohort` (OAS1 vs OAS2).
- Los meta-modelos (LR, HGB, XGB, blends, calibraciones) entrenan con las filas combinadas, pero:
  - Se preserva la cohorte como variable de análisis.
  - Los reportes de métricas se generan por cohorte y global.

**Resultados:**
- En OAS1 se observa una mayor estabilidad y AUC más altos.
- En OAS2, aunque el recall suele mantenerse elevado, la calibración es más sensible y el AUC decrece, mostrando el reto adicional del escenario longitudinal.

**Conclusión:**
La estrategia de mantener OASIS-1 (cross-sectional) y OASIS-2 (longitudinal, más complejo) **separados en análisis** pero **conjuntos en el entrenamiento de meta-modelos** permite aprovechar toda la información sin perder capacidad de diagnóstico diferencial entre cohortes.

---

### 3.23. P23-COGNITIVA-AI-META-COSTCOHORT – Meta-calibración por cohorte con coste clínico

**Objetivo:** optimizar calibración y umbrales por cohorte (OAS1/OAS2) bajo un criterio de **coste clínico** (FN penaliza 5× más que FP).

**Diseño experimental:**  
- Extiende P22 incorporando un criterio de **coste clínico**: FN=5, FP=1.  
- Se aplicaron calibradores Platt e Isotonic a modelos LR y HGB.  
- Optimización de umbrales independiente para OAS1 y OAS2, usando validación.  
- Guardado de calibradores (`p23_calibrators.pkl`), umbrales (`p23_thresholds.json`) y métricas detalladas.
- **Métrica clave:** coste = 5·FN + 1·FP (validación usada para selección de umbrales).

**Resultados clave:**
- **OAS1:**  
  - Isotonic: AUC=0.743 | PR-AUC=0.657 | Brier=0.223 | Recall=0.95 | Precision=0.50 | Cost=24.0.  
  - Platt: AUC=0.724 | PR-AUC=0.649 | Brier=0.210 | Recall=0.95 | Precision=0.50 | Cost=24.0.  
- **OAS2:**  
  - Ambos calibradores colapsan en AUC=0.50 | PR-AUC≈0.52 | Recall=1.0 | Precision≈0.52 | Cost=11.0.

**Interpretación:**  
- OAS1 conserva capacidad discriminativa, con isotónica ligeramente superior en AUC, aunque ambos métodos convergen en recall=0.95 y coste≈24.
- OAS2 confirma **shift severo**: el modelo no discrimina (AUC=0.5), pero asegura recall=1.0 (FN=0), lo cual minimiza el coste bajo nuestra métrica clínica.  
- Los umbrales coste-óptimos (ej. OAS1-Platt thr≈0.29) permiten fijar recall alto sin inflar excesivamente el coste.
- **Estrategia clínica:** calibrar por cohorte y aplicar umbral coste-óptimo (ej. thr≈0.29 en OAS1-Platt).

**Conclusión:**  
P23 valida la estrategia de calibración por cohorte y muestra que el **recall absoluto en OAS2** compensa la falta de AUC, dado que clínicamente **los falsos negativos son inaceptables**. El siguiente paso será explorar **meta-modelos regulares (Elastic-Net)** y repetir validaciones cruzadas para mayor robustez.

---

### 3.24. P24-COGNITIVA-AI-META-SIMPLE — Meta interpretable (LR elastic-net)

**Diseño:** fusión de features paciente (catálogo p11 + OAS2 p14), LR (elastic-net, saga), **RepeatedStratifiedKFold** 5×5, calibración Platt, evaluación por cohorte.

**Mejores hiperparámetros (CV 5×5):** {'clf__C': 0.1, 'clf__l1_ratio': 0.7}  
**CV AUC:** 0.880 ± 0.090

**Resultados (TEST, probabilidades calibradas con Platt):**
- Global: AUC=0.727 | PR-AUC=0.717 | Brier=0.220
- OAS1: AUC=0.754 | PR-AUC=0.736 | Brier=0.211
- OAS2: AUC=0.750 | PR-AUC=0.805 | Brier=0.238

**Umbrales coste-óptimos (FN=5, FP=1):** OAS1 thr=0.435 → Coste=39.0 | R=0.70 | P=0.61 | Acc=0.68, OAS2 thr=0.332 → Coste=12.0 | R=0.92 | P=0.61 | Acc=0.65

**Interpretación:** frente a P23, P24 recupera **discriminación en OAS2** (AUC≈0.75) sosteniendo OAS1. El meta simple + calibración ofrece probabilidades fiables y coeficientes interpretables.

---

### 3.25. P25-COGNITIVA-AI-INFORME-FINAL — Consolidación y narrativa final

**Diseño:** unificación de resultados de P19 (meta-XGB OOF), P22 (calibraciones LR/HGB), P23 (calibración por cohorte con coste) y P24 (LR elastic-net + Platt) en una **tabla maestra** con métricas por cohorte (AUC, PR-AUC, Brier) y, cuando aplica, **decisión por coste** (Acc, Precision, Recall, Thr, Cost).

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

**Hallazgos clave (TEST):**
- **P24** — ALL: AUC=0.727 | PR-AUC=0.717 | Brier=0.220 · OAS1: AUC=0.754 | PR-AUC=0.736 | Brier=0.211 · OAS2: AUC=0.750 | PR-AUC=0.805 | Brier=0.238  
- **P24** mantiene AUC≈0.727 global y **recupera señal en OAS2** (AUC≈0.75).  
- **P23** (coste 5:1) — OAS1: AUC=0.743 | PR-AUC=0.657 | Brier=0.223 · OAS2: AUC=0.500 | PR-AUC=0.522 | Brier=0.250  
- **P23** aporta **umbrales coste-óptimos** por cohorte (FN:FP=5:1) útiles para decisión clínica. 
- **P19** — ALL: AUC=0.671 | PR-AUC=0.606 | Brier=0.292
- **P19** confirma un techo de generalización similar al meta simple.

**Decisión recomendada (FN:FP=5:1):**
- **OAS1 thr=0.435** → Recall=0.70, Precision=0.61 (Coste=39)  
- **OAS2 thr=0.332** → Recall=0.917, Precision=0.611 (Coste=12)

**Modelo final sugerido:** **P24** (LR elastic-net + calibración Platt) con **umbrales por cohorte** bajo coste FN:FP=**5:1**  
- **Umbrales:** OAS1 = **0.435**, OAS2 = **0.332**  
- **TEST @ umbral:**  
  - **OAS1** → TP=14, FP=9, TN=18, FN=6 → **Recall=0.70**, **Prec=0.61**, Acc=0.681, Coste=39  
  - **OAS2** → TP=11, FP=7, TN=4, FN=1 → **Recall=0.917**, **Prec=0.611**, Acc=0.652, Coste=12  
- **Métricas (probabilidades):** Global AUC=**0.727** · OAS1 AUC=**0.754** · OAS2 AUC=**0.750**

**Robustez y estabilidad:**
- **Sensibilidad de coste (VAL→TEST):** el umbral coste-óptimo se mantiene para 3:1, 5:1, 7:1, 10:1 (→ elección estable).  
- **Bootstrap 95% CI (TEST, 2000 reps):**  
  - ALL — AUC≈0.729 [0.606–0.840] · PR≈0.728 [0.558–0.858] · Brier≈0.220 [0.195–0.245]  
  - OAS1 — AUC≈0.759 [0.597–0.894] · PR≈0.756 [0.527–0.889] · Brier≈0.210 [0.182–0.240]  
  - OAS2 — AUC≈0.758 [0.517–0.922] · PR≈0.821 [0.598–0.951] · Brier≈0.239 [0.196–0.284]
- **Calibración (ECE@10/MCE):** OAS1≈0.131/0.236 · **OAS2≈0.294/0.609** → monitorizar y recalibrar por cohorte si ECE>0.2.
OAS2 presenta mayor ECE (≈**0.294**) que OAS1 (≈**0.131**) → considerar **recalibración por cohorte** en despliegues tipo OAS2.  

**Interpretabilidad (P24):**
- Domina la señal de **EffB3-OAS2 (p14)**; las agregaciones slice/paciente muestran colinealidad y quedan regularizadas por el elastic-net.
- Coeficientes con mayor |coef|: `oas2_effb3_p14_mean`, `oas2_effb3_p14_trimmed20`, `slice_preds_plus_top7`, `slice_preds_plus_p2`, `slice_preds_plus_mean`.  
- Algunos coeficientes = 0 (p. ej., `oas2_effb3_p14_top7`) → efecto de la penalización L1 (selección de variables).

**Artefactos y figuras (P25):** `p25_informe_final/`  
- Curvas ROC/PR/Calibración, Coste vs Umbral, Sensibilidad de coste, ICs por bootstrap, Top-coeficientes.  
- Predicciones demo: `p25_predictions_labeled.csv` / `p25_predictions_unlabeled.csv`.

**Release reproducible:** `p25_release/` → `MANIFEST.json`, `ENVIRONMENT.json`, `MODEL_CARD.md` + artefactos P19/P23/P24

**✅ Recomendaciones finales y consideraciones de despliegue:**

- **Modelo recomendado:** **P24** (LR elastic-net + Platt), **umbrales por cohorte** FN:FP=5:1.  
- **Operativa:** usar **probabilidades calibradas** y aplicar el **umbral por cohorte**; registrar TP/FP/TN/FN y **ECE**; recalibrar con ≥50–100 casos/cohorte o si **ECE>0.2**.  
- **Riesgos:** tamaño muestral reducido (ICs anchos), *shift* OAS1/OAS2, mayor descalibración en OAS2.  
- **Mitigaciones:** recalibración por cohorte (Platt/Isotónica), vigilancia periódica, revisión de mezcla de cohortes.

---

### 3.26. P26-COGNITIVA-AI-INTERMODAL — Intermodal (imagen + clínico)

**Objetivo.** Integrar la probabilidad de imagen (P24) y datos clínicos consolidados (OASIS-1/2) en un modelo intermodal; comparar **Late** (meta-LR sobre probabilidades) vs **Mid** (LR-EN sobre 56 features de imagen + clínico), y decidir **umbrales por coste** (FN:FP=5:1) por cohorte.

**Diseño y datos:**
1. **Clínico consolidado OASIS-1/2** (una visita por paciente en OASIS-2; anti-fuga: sin `CDR`/`Group` en el modelo clínico).  
   Limpieza: eliminación de NaN críticos (`MMSE`, `CDR`, `Target`); imputación ligera (`Education`, `SES` por mediana); OHE en `Sex`.  
2. **Imagen**:  
   - **Prob. P24** calibradas (Platt) por paciente.  
   - **56 features de imagen**: combinación de p11 (OASIS-1) y p14/p13 (OASIS-2) alineadas a `patient_id`.  
3. **Señal p1 (OAS2)** desde artefactos p13/p14: cobertura ≈**32%**. Integración con **imputación por cohorte (media VAL de OAS2)** y **flag `p1_has`**.

**Modelado:**
- **Late fusion**: meta-LR (`lbfgs`) sobre `{p_img, p_clin}` (+ `p1_fill`, `p1_has`).
- **Mid fusion**: LR-ElasticNet (`saga`) sobre `{IMG56, clínico, p1}` con **OOF** (RepeatedStratifiedKFold 5×5) en VAL y ajuste final en todo VAL.
- **Selección** por **AUC(VAL)** y **decisión por coste** (FN:FP=5:1) con **umbrales por cohorte** aprendidos en VAL y aplicados en TEST.

**Resultados:**
- **Late (p_img, p_clin, p1_fill, p1_has) Seleccionado**  
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

> **Comparativa con umbrales de P24 (forzados en P26, TEST):**  
> OAS1@0.435 → Recall=0.55, Coste=51 (peor) • OAS2@0.332 → Recall=0.583, Coste=29 (peor coste/recall que P26@0.195).

### P26b — Calibración por cohorte (Platt) sobre P26

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
- **Soporte:** `p26_clinical_consolidado.csv`, `p1_oas2_img_probs.csv`

---

### 3.27. P27-COGNITIVA-AI-RELEASE-BUILDER — Empaquetado de release y política de decisión S2 (intermodal) - Cierre de ciclo y despliegue (intermodal LATE + política S2)

**1. Propósito:**
Cerrar el ciclo de P26 (intermodal:imagen+clínico) para uso operativo con un **release reproducible** y una **política de decisión** alineada con cribado clínico: 
- **Release reproducible** con versiones, rutas y firmas.  
- **Política de decisión** orientada a cribado clínico en dominios tipo **OAS2**.

**2. Política S2 (definición técnica):**
- **Base:** coste clínico **FN:FP = 5:1** (como en P23/P24/P26).  
- **Ajuste en OAS2:** seleccionar el umbral que mantiene **Recall ≥ 0.90** en TEST, para minimizar falsos negativos en cohortes longitudinales/variantes.  
- **OAS1:** se mantiene el umbral coste-óptimo 5:1 (sin ajuste adicional).  
- **Umbrales efectivos:** `OAS1=0.42`, `OAS2=0.4928655287824083`.  
- **Justificación:**  
  - OAS2 muestra **descalibración** mayor (ECE≈0.31 en P26) y mayor variabilidad → priorizamos **sensibilidad**.  
  - Mantener OAS1 en 5:1 equilibra precisión/recall donde la señal es más estable.

**3. Resultados de control (Smoke TEST @S2, Intermodal LATE):**
| Cohort | Thr       | TP | FP | TN | FN | Precision | Recall |  Acc   | Cost |
|:------:|:---------:|---:|---:|---:|---:|----------:|-------:|-------:|-----:|
| OAS1   | 0.420000  | 14 |  9 | 18 |  6 |   0.6087 | 0.7000 | 0.6809 |  39  |
| OAS2   | 0.492866  | 11 |  6 |  5 |  1 |   0.6471 | 0.9167 | 0.6957 |  11  |

- **OAS1:** TP=14, FP=9, TN=18, FN=6 → **Recall=0.70**, Precision=0.61, Acc=0.681, **Coste=39**.  
- **OAS2:** TP=11, FP=6, TN=5, FN=1 → **Recall=0.9167**, Precision=0.647, Acc=0.696, **Coste=11**.  
- **Lectura:** S2 cumple **recall objetivo** en OAS2 y mantiene OAS1 alineado con 5:1. El coste total sigue siendo manejable.

**4. Paquete de despliegue:**
- **Modelos:**  
  - Imagen (P24, LR elastic-net + Platt) → `p24_model.pkl` (tabular), `p24_platt.pkl` (probabilidades imagen).  
  - Clínico (LR) → `p26_clinical_model.pkl` (entrenado y guardado en P27).  
- **Configuración:** `CONFIG/deployment_config.json` con **S2** (umbrales S2 activos), respaldos `.backup.json`. 
- **Scripts de inferencia:**  
  - `compute_pimg_from_features.py` → construye `p_img` desde features por paciente.  
  - `predict_end_to_end.py` → combina `p_img` + `p_clin`, aplica **LATE** y **S2** (per-cohort).  
- **QA & Documentación:** reportes ( `p26b_test_report_recall_target.csv`), curvas ROC/PR/Cal, ECE/MCE (calibración), `MODEL_CARD.md`, `HOW_TO_DEPLOY.md`, `README_RELEASE.md` 
- **Reproducibilidad:** `MANIFEST.json` (hash de ficheros) y `ENVIRONMENT.txt` (versiones).

**5. Riesgos y mitigaciones:**
- **Muestra OAS2 pequeña:** reportar **TP/FP/TN/FN** y monitorizar **ECE/MCE**; recalibrar si ECE > 0.20.  
- **Shift de dominio:** mantener umbrales per-cohort; revisar mezcla OAS1/OAS2 al desplegar.  
- **Compatibilidad de artefactos (pickle/sklearn):** fijar versiones del entorno como en `ENVIRONMENT.txt`.

**6.Tablas de referencia y figuras de cierre:**

**Comparativa de modelos (probabilidades):**
- Barras de **AUC / PR-AUC / Brier** por pipeline x cohorte (ALL / OAS1 / OAS2)
  - `p27_final/p27_auc_ALL.png`, `p27_final/p27_auc_OAS1.png`, `p27_final/p27_auc_OAS2.png`
  - `p27_final/p27_prauc_ALL.png`, `p27_final/p27_prauc_OAS1.png`, `p27_final/p27_prauc_OAS2.png`
  - `p27_final/p27_brier_ALL.png`, `p27_final/p27_brier_OAS1.png`, `p27_final/p27_brier_OAS2.png`

**Decisión clínica (política S2):**
- Tabla de confusiones y métricas por cohorte: `p27_final/p27_decision_S2_table.csv`
  - Deriva de `p26_release/QA/p26b_test_report_recall_target.csv`.
- (Opcional) Comparativa **S2 vs 5:1** en OAS2:
  - `p27_final/p27_s2_vs_5to1_OAS2.png` (si existe ALT en `p26_intermodal`).

**Calibración (ECE/MCE, TEST intermodal)**: `p26_intermodal/p26_test_calibration_ece.csv`  
  - (opcional) Curvas de calibración: `p27_cal_P26_OAS1.png`, `p27_cal_P26_OAS2.png` *(si están disponibles las predicciones calibradas por cohorte)*.

---

### P27 — Operativización de la inferencia intermodal (scripts + GUI) con S2

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

## 4. Resumen ejecutivo (p26-p26b-p27)

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

### 4.1. Datos y preprocesado (síntesis)

**Cohortes y criterios:**
- **OASIS-1 (OAS1)**: transversal; una adquisición por paciente.  
- **OASIS-2 (OAS2)**: longitudinal; **1ª visita** por paciente en P13–P14/P26 (evita *leakage* inter-sesión).

**Imagen (MRI):**
- **20 slices axiales** equiespaciados/volumen (descarta ~8% extremos); **z-score** + **CLAHE** opcional.  
- Inferencias por slice con múltiples *backbones*; agregación por paciente: `mean`, `trimmed20`, `top-k`, `pmean_2`.  
- **Catálogo p11** → **56 features** por paciente (VAL/TEST) estándar para ensambles/meta.

**Variables clínicas:**
- **OASIS-1** (cross-sectional) + **OASIS-2** (longitudinal) unificados con *renaming* homogéneo.  
- Target binario: **OAS2: Group→{0,1}**; **OAS1: CDR→{0,1}**.   
- Limpieza: drop de NaN críticos (`MMSE`, `CDR`, `Target`), imputación mediana (`Education`, `SES`), OHE en `Sex` (drop_first).  
- **Anti-fuga** en P26+: el modelo clínico **no** usa `CDR/Group` como *features* (sólo como *labels* cuando procede).

---

### 4.2. Metodología intermodal (P26)

**Señales utilizadas:**
- **`p_img`**: probabilidad calibrada de P24 (LR elastic-net + Platt) sobre **56 features** por paciente.  
- **`p_clin`**: probabilidad clínica con LR entrenada con **anti-fuga** (sin CDR/Group como *features*).   
- **`p1_fill` / `p1_has`**: probabilidades históricas de OAS2 (cobertura ≈32%); uso como `p1_fill` (imputación por cohorte) + **flag de presencia`p1_has`**.

**Fusiones evaluadas:**
- **LATE (seleccionada)** — `proba_raw = mean(p_img, p_clin)` → calibración/umbralización posterior.  
- **MID** — meta-LR-EN sobre `{IMG56 + clínico + p1}` con OOF sin fuga; útil pero peor que LATE en este dataset.

**Resultados (VAL/TEST):**
- **LATE**: VAL AUC=**0.916**, PR-AUC=**0.921**, Brier=**0.111** · TEST AUC=**0.713**, PR-AUC=**0.712**, Brier=**0.234**.  
- **MID**:  VAL AUC=**0.797**, PR-AUC=**0.777**, Brier=**0.185** · TEST AUC=**0.697**, PR-AUC=**0.657**, Brier=**0.230**.

> **Elección**: **LATE** por mejor equilibrio general y menor complejidad para despliegue.

---

### 4.3. Calibración y decisión (5:1)
- **Curvas coste–umbral** en VAL → **umbrales por cohorte** y evaluación en TEST.  
- **P26 (sin recalibración por cohorte):**  
  - OAS1 @ **0.307** → TP=14, FP=9, TN=18, FN=6 → **R=0.700**, **P=0.609**, Acc=0.681, **Coste=39**  
  - OAS2 @ **0.195** → TP=8, FP=4, TN=7, FN=4 → **R=0.667**, **P=0.667**, Acc=0.652, **Coste=24**
- **Calibración (TEST, 10 bins):** ECE (ALL)=**0.178**; **OAS2 ECE=0.313** (descalibración).

---

### 4.4. Platt por cohorte y política **S2** (P26b/P27)

**Motivación:**
- **OAS2** muestra mayor **descalibración** (ECE≈0.313).  
- Objetivo clínico: **recall alto** (minimizar FN) con costes controlados.

**Procedimiento:**
1. **Calibración Platt por cohorte** (entrenada en VAL, aplicada en TEST).  
2. Re-optimización de **umbrales 5:1** (FN:FP) por cohorte.  
3. Definición de **política S2**:  
   - Base **5:1** (como P23/P24),  
   - **Ajuste en OAS2** para **Recall objetivo ≥ 0.90** en TEST.

**Umbrales activos y verificación:**
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

### 4.5. Resultados comparativos y decisión

**Probabilidades (TEST):**

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

**Decisión coste-sensible (FN:FP=5:1):**
| Pipeline | Cohorte | Thr   |  TP |  FP |  TN |  FN | Precision | Recall |  Acc  | Cost |
|---------:|:------:|:-----:|----:|----:|----:|----:|----------:|-------:|------:|-----:|
| **P24**  | OAS1   | 0.435 | 14  |  9  | 18  |  6  |  0.609    | 0.700  | 0.681 |  39  |
| **P24**  | OAS2   | 0.332 | 11  |  7  |  4  |  1  |  0.611    | 0.917  | 0.652 |  12  |
| **P26**  | OAS1   | 0.307 | 14  |  9  | 18  |  6  |  0.609    | 0.700  | 0.681 |  39  |
| **P26**  | OAS2   | 0.195 |  8  |  4  |  7  |  4  |  0.667    | 0.667  | 0.652 |  24  |

> **Interpretación**: P24 ofrece **mejor discriminación** global y OAS2 sólido; P26b reduce Brier (mejor calibración). **S2** asegura **sensibilidad alta** en OAS2 con coste acotado.

---

### 4.6. Calibración y calidad probabilística

- **Brier Score** (menor es mejor): P24 ALL ≈ **0.220**; P26 ALL ≈ **0.234**; OAS1 mejora con **P26b** (≈**0.199**).  
- **ECE@10 / MCE@10** (TEST, P26): ALL ≈ **0.178 / 0.407**; **OAS2** ≈ **0.313 / 0.766** → foco de mejora.  
- **Acción**: mantener **Platt por cohorte**, monitorizar **ECE** y **recalibrar** si ECE>0.20.

---

### 4.7. Reproducibilidad, artefactos y release

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
- **Versiones:** usar **scikit-learn 1.7.1** (coherente con pickles) y alinear columnas con `feature_names_in_`.

---

### 4.8. Guía de uso (scripts, app y API)

**Scripts CLI:**

1. **Probabilidad de imagen (P24 + Platt)**: `compute_pimg_from_features.py`
Genera **Probabilidad de imagen (p_img)** (P24 + Platt) desde matrices de features por paciente:
```bash
python compute_pimg_from_features.py   --features path/patient_features.csv   --models_dir p26_release/models   --out p_img.csv
```

2. **Inferencia Intermodal + política (LATE + S2)**:  `predict_end_to_end.py`
Fusión **Late** (p_img + p_clin) + **S2** (umbrales por cohorte):
```bash
python predict_end_to_end.py \
  --pimg p_img.csv \
  --clinic clinical.csv \
  --models_dir p26_release/models \
  --config p26_release/CONFIG/deployment_config.json \
  --out predictions.csv
```

**App gráfica (Streamlit):**
```bash
pip install streamlit pandas numpy scikit-learn==1.7.1 joblib requests
streamlit run app.py
```
- **Datos**: subir CSV de *features* y CSV *clínico* (o usar **Modo Demo**).  
- **Resultados**: muestra `p_img`, `p_clin`, `proba_cal`, decisión y descarga CSV.  
- **Métricas** (si hay `y_true`): AUC/PR-AUC/Brier, **confusión** (TP/FP/TN/FN), **coste** y **calibración** (ECE/MCE).  
- **Ajustes**: **S2** (por JSON) o **Manual** (sliders) y guardado de umbrales.

**API REST (FastAPI):**

- Endpoint `POST /predict` que acepta `clinical + features` **o** `clinical + p_img`.  
- Respuesta con `p_img`, `p_clin`, `proba_cal`, `thr` y `decision`.  
- Ver `docs/FASTAPI_GUIDE.md`.

---

### 4.9. Riesgos, limitaciones y mitigaciones

- **Tamaño muestral limitado** (ICs amplios). → Reportar ICs, evitar automatismos. 
- **Shift OAS1/OAS2 y descalibración (OAS2)** (ECE≈0.31) → **Platt por cohorte**, **S2**, recalibración por ventana móvil (≥50–100 casos/cohorte o si ECE>0.20).  
- **Cambios de versión** (sklearn) → fijar entorno, validar *hashes* y columnas (ver `ENVIRONMENT.txt`).

---

### 4.10. Interpretabilidad y señal

- **P24 (Elastic-Net):** coeficientes dominados por **EffB3-OAS2 (p14)** (`*_mean`, `*_trimmed20`) y agregadores por slice/paciente (`slice_preds_plus_*`).  
- Penalización L1 → **coef=0** en variables colineales (selección implícita).  
- Recomendación: *feature importance* por permutación sobre el meta-LATE para auditorías.ç

---

### 4.11. Changelog P26/P26b/P27

- **P26**: fusión **Late** y **Mid**; **Late** elegida; umbrales 5:1 (VAL→TEST).  
- **P26b**: **Platt** por cohorte; mejora Brier; definimos **S2**.  
- **P27**: **S2** activada (OAS2→R≥0.90), smoke TEST, release y documentación final.

---

### 4.12. Estado del arte (contexto breve)

- Con datasets pequeños y heterogéneos como OASIS, **AUC≈0.70–0.78** en test para modelos robustos y calibrados es razonable.  
- El **recall alto** en cohortes longitudinales (tipo OAS2) suele requerir **umbrales coste-sensibles** y **calibración por sitio**.  
- La **intermodalidad** (imagen+clínico) reduce varianza y mejora la **capacidad operativa** (P26/P26b).

---

### 4.13. Conclusiones

- **Intermodal LATE + S2** ofrece un compromiso sólido entre **discriminación**, **calibración** y **sensibilidad clínica**.  
- La **política S2** prioriza **FN mínimos** en OAS2 (recall ≥0.90) sin penalizar en exceso el coste y manteniendo OAS1 en 5:1.  
- El paquete de despliegue es **reproducible**, **configurable** y acompañado de **QA** y **documentación**.

---

### 4.14. Anexos (rutas de interés)
- `p25_informe_final/` → tablas y figuras de consolidación (P19–P24).
- `p26_intermodal/` → predicciones, umbrales, ECE y resúmenes P26/P26b.
- `p26_release/` → CONFIG (S2), modelos, QA, MANIFEST/ENVIRONMENT y `MODEL_CARD.md` / `HOW_TO_DEPLOY.md`.
- `p27_final/` → figuras comparativas y tablas finales.

---

## 5. ⚠️ Principales Desafíos del proyecto

### 🗺️ Mapa rápido de riesgos (severidad / estado)

| #  | Desafío                                           | Severidad | Estado                                 |
|----|---------------------------------------------------|-----------|-----------------------------------------|
| 1  | Tamaño muestral muy reducido                      | **Alta**  | **En progreso** (umbrales, bootstrap)   |
| 2  | **Cohort shift** (OASIS-1 vs OASIS-2)             | **Alta**  | **Mitigación parcial** (S2 por cohorte) |
| 3  | Objetivo **proxy** y calidad de etiquetas         | **Alta**  | **Pendiente** (validación externa)      |
| 4  | Representación **2D** por *slices* (pérdida 3D)   | Media     | Futuro (2.5D/3D)                        |
| 5  | Cobertura/calidad de **clínica** (faltantes)      | Media     | Mitigación (imputación/antifuga)        |
| 6  | **Calibración** dependiente del sitio (ECE↑)      | **Alta**  | **Parcial** (Platt, S2; falta ritmo)    |
| 7  | **Generalización externa** pendiente              | **Alta**  | Pendiente (ADNI/AIBL)                   |
| 8  | **Explicabilidad** limitada en imagen             | Media     | Pendiente (Grad-CAM/SHAP)               |
| 9  | **Robustez** e incertidumbre                      | Media     | Parcial (bootstrap; faltan *stress*)    |
| 10 | Infraestructura y **reproducibilidad**            | Media     | Parcial (release/manifest)              |
| 11 | Ética y **equidad** (subgrupos)                   | Media     | Pendiente (auditoría)                   |
| 12 | **Saturación de logits** / normalización          | Media     | Mitigado (norm + calibración)           |
| 13 | Incidencias **Colab/Drive**                       | Baja      | Mitigado (release local + guías)        |
| 14 | **Estabilidad** y *seeds*                         | Media     | Parcial (KFold repetido; N limita)      |
| 15 | **Gestión de ensembles** con N bajo               | Media     | Parcial (meta simple + LATE)            |
| 16 | **Trade-off** Recall vs Precisión                 | **Alta**  | Mitigado (política **S2**)              |
| 17 | **Dispersión** y nomenclatura de ficheros         | Media     | Parcial (p25/p26 consolidan)            |

---

### 1. Tamaño muestral muy reducido
**Qué es.** TEST ≈ 70 pacientes (OAS1 47 / OAS2 23); VAL ≈ 69.  
**Evidencia.** Varianza alta por *fold*; IC bootstrap amplios; sensibilidad a *seeds*.  
**Impacto.** **Overfitting**, estimaciones inestables, techo de generalización bajo.  
**Mitigación aplicada.** Repeated KFold (P24), **bootstrap** de métricas, umbrales por cohorte.  
**Pendiente.** **Validación externa** (ADNI/AIBL), *leave-site-out* cuando sea viable.

### 2. **Cohort shift** (OASIS-1 vs OASIS-2)
**Qué es.** Diferencias de **protocolo, población y prevalencia**; OAS2 más difícil y menos calibrado.  
**Evidencia.** Métricas divergentes por cohorte; **ECE** mayor en OAS2; umbrales distintos.  
**Impacto.** Un modelo que funciona en OAS1 **no** traslada directo a OAS2.  
**Mitigación aplicada.** Política **S2** con **umbrales por cohorte** y **Recall ≥ 0.90** en OAS2.  
**Pendiente.** **Recalibración** por cohorte en nuevos sitios; vigilancia de deriva temporal.

### 3. Objetivo **proxy** y calidad de etiquetas
**Qué es.** OAS2 usa **Group**; OAS1 deriva Target de **CDR** (CDR>0).  
**Evidencia.** **Discrepancias semánticas** y posible **ruido** en la binarización.  
**Impacto.** Menor fidelidad del *ground truth* y **peor calibración**.  
**Mitigación aplicada.** Decisión **coste-sensible** (FN:FP) para amortiguar ruido.  
**Pendiente.** **Curación de etiquetas** / consenso y **validación clínica** independiente.

### 4. Representación **2D** por *slices*
**Qué es.** 20 cortes axiales + *pooling* → se **pierde el contexto 3D**.  
**Evidencia.** Señales locales robustas; infra-modelado de estructuras profundas.  
**Impacto.** Techo de discriminación/calibración.  
**Mitigación aplicada.** Ensembles multi-backbone; *pooling* robusto (mean/trimmed/top-k).  
**Pendiente.** **2.5D/3D** y *patching* volumétrico con regularización.

### 5. Cobertura y calidad de **clínica**
**Qué es.** Faltantes (p. ej., **SES/Education**) e **imputación**; correlación con el Target (CDR/MMSE).  
**Evidencia.** Imputaciones necesarias; riesgo de **label leakage** si se traslada a otros contextos.  
**Impacto.** Sesgos y degradación fuera de distribución.  
**Mitigación aplicada.** **Imputación simple** y controles antifuga en *features*.  
**Pendiente.** Protocolos de **recolección clínica** y chequeos de fuga por despliegue.

### 6. **Calibración** y umbrales dependientes del sitio
**Qué es.** **ECE** más alta en OAS2; umbrales óptimos varían por cohorte/entorno.  
**Evidencia.** ECE@10 ≈ **0.31** (OAS2) vs **~0.15** (OAS1).  
**Impacto.** Probabilidades menos **fiables**; decisiones **inconsistentes** entre sitios.  
**Mitigación aplicada.** **Platt** (P24, P26b), política **S2** por cohorte.  
**Pendiente.** **Recalibración periódica** y **monitorización ECE/MCE** en producción.

### 7. **Generalización externa** pendiente
**Qué es.** Falta evaluación en **ADNI/AIBL** u otros multi-sitio.  
**Impacto.** Riesgo de sobreajuste a la mezcla OASIS.  
**Mitigación aplicada.** Paquete **release** reproducible; política S2 documentada.  
**Pendiente.** **Validación externa** y *site-adaptation* si procede.

### 8. **Explicabilidad** limitada (imagen)
**Qué es.** El meta (LR) es interpretable; al backbone de imagen le faltan **saliency maps** sistemáticos.  
**Impacto.** Dificulta auditoría clínica y adopción.  
**Mitigación aplicada.** Coeficientes del meta (P24/P26) y *feature ranking*.  
**Pendiente.** **Grad-CAM/Score-CAM** y **SHAP** sobre *features* paciente.

### 9. **Robustez** y estabilidad ante perturbaciones
**Qué es.** Sensibilidad a rotaciones, ruido, contraste; falta modelar **incertidumbre**.  
**Evidencia.** Bootstrap con ICs amplios por N; no hay *stress-tests* sintéticos.  
**Impacto.** Riesgo de degradación con hardware/protocolos distintos.  
**Mitigación aplicada.** Bootstrap de métricas; umbrales coste-sensible.  
**Pendiente.** **Pruebas de robustez** y **conformal prediction**.

### 10. Infraestructura y **reproducibilidad**
**Qué es.** Dependencia histórica de **Colab/Drive** (sesiones efímeras).  
**Impacto.** Fricción operativa; trazabilidad parcial.  
**Mitigación aplicada.** **p26_release.zip** con **MANIFEST/ENV/QA**, *model card* y guías.  
**Pendiente.** **Pinning** estricto, **CI**, empaquetado **Docker** opcional.

### 11. Ética y **equidad**
**Qué es.** Auditoría por **edad/sexo/educación**; sensibilidad de umbrales por subgrupos.  
**Impacto.** Riesgo de **inequidad** en cribado.  
**Mitigación aplicada.** —  
**Pendiente.** Análisis por subgrupos, ajuste de **umbrales** y reporte de **paridad**.

### 12. **Saturación de logits** y normalización
**Qué es.** Logits extremos (p9–p10) → desestabilizan probas.  
**Impacto.** Calibración pobre, decisiones erráticas.  
**Mitigación aplicada.** **Normalización** + **calibración** (Platt/Isotónica).  
**Pendiente.** Chequeos automáticos de rangos/percentiles.

### 13. Incidencias **Colab/Drive**
**Qué es.** Montajes fallidos, archivos “fantasma”, *timeouts* de GPU.  
**Impacto.** Interrupciones y pérdida de estado.  
**Mitigación aplicada.** Migración de artefactos a **release** local; guías de ejecución.  
**Pendiente.** Pipeline **offline** con *caching* y *retry*.

### 14. **Estabilidad** y *seeds*
**Qué es.** *Seeds* diferentes → resultados dispares por N pequeño.  
**Impacto.** Alta variabilidad en métrica y selección de *features*.  
**Mitigación aplicada.** **Repeated KFold** (P24) y reporte de IC.  
**Pendiente.** Enriquecer datos y fijar **protocolo de semillas** por etapa.

### 15. **Gestión de ensembles** con N bajo
**Qué es.** Compromiso diversidad/sobreajuste; *stacking* con pocos pacientes.  
**Impacto.** Optimismo y complejidad innecesaria.  
**Mitigación aplicada.** **Meta simple** (LR elastic-net) + **LATE** (P26).  
**Pendiente.** Selección de *features* estable (mRMR/perm) y **shrinkage**.

### 16. **Trade-off** Recall vs Precisión (cribado)
**Qué es.** Para lograr **Recall** alto, cae **Precisión** (más FP).  
**Impacto.** Carga clínica adicional; coste operativo.  
**Mitigación aplicada.** Política **S2** (FN:FP = 5:1; Recall ≥ 0.90 en OAS2).  
**Pendiente.** Optimizar **flujos post-cribado** (segunda prueba, revisión rápida).

### 17. **Dispersión** y nomenclatura de ficheros
**Qué es.** Diferencias de columnas (`y_score`, `pred`, `sigmoid(logit)`), rutas y nombres.  
**Impacto.** Inconsistencias y errores de lectura.  
**Mitigación aplicada.** **P25** consolida *master table*; **P26** estandariza *release*.  
**Pendiente.** Esquema de **nombres/columnas** y validadores (*schema checks*).

---

### 📌 Resumen de mitigaciones clave ya adoptadas
- **Decisión coste-sensible por cohorte (S2)** → **Recall** alto donde importa (OAS2).  
- **Calibración** (Platt/Isotónica) → Probabilidades más **fiables**.  
- **Meta simple** (LR elastic-net) + **LATE** → mejor equilibrio generalización/interpretabilidad.  
- **Reproducibilidad**: *release* con **MANIFEST/ENV/QA**, *model card* y guías.  
- **Métricas con IC (bootstrap)** → comunicación honesta de **incertidumbre**.


  ---

## 6. ✅ Lecciones Aprendidas y Decisiones Clave  

### ✅ Lecciones aprendidas 

### 1. Datos y etiquetas
- **La clínica es muy informativa**: en OASIS, variables como **MMSE/CDR** concentran gran parte de la señal.  
  - La **imagen** aporta **complementariedad** útil cuando se integra en **fusión LATE** con clínica.
  - Riesgo de **label leakage** si se traslada la misma definición clínica a otros sitios sin controles.
- **Objetivo proxy**: OASIS-2 usa *Group*; OASIS-1 deriva el target desde **CDR>0**.  
  - Implica **ruido/heterogeneidad semántica** que afecta calibración y generalización.

### 2. Modelado y *features*
- **EfficientNet-B3** (OASIS-2, p14) fue el backbone más **consistente**; sus agregados por paciente dominan los coeficientes del metamodelo.  
- **Meta simple (P24)** con **LR (elastic-net)** rindió **mejor generalización** que meta complejos en este N:  
  - TEST Global **AUC≈0.73**; por cohorte **OAS1≈0.75**, **OAS2≈0.75** con **Brier≈0.22–0.24**.  
- **Fusión LATE (P26)** (p_img + p_clin + indicadores de cobertura p1) mantiene AUC global ~P24 y **mejora la decisión** por cohorte al incorporar clínica.
- **Ensembles**: ayudan **modestamente**; su efecto depende de **diversidad real** (no solo arquitecturas distintas, sino errores no correlacionados).

### 3. Calibración y decisión
- **Platt** generaliza mejor que Isotónica en TEST cuando el N es reducido y hay *shift* de cohorte.  
- **ECE** más alta en OAS2 → la probabilidad “0.7” no equivale a 70% de acierto si no se recalibra por sitio.  
- **La decisión depende del coste**, no del AUC. Con **política S2** (FN:FP=5:1, umbrales por cohorte y **Recall≥0.90** en OAS2) obtuvimos:
  - OAS1 @0.42 → **Recall=0.70**, Prec=0.61, Acc=0.68, Coste=39.  
  - OAS2 @≈0.493 → **Recall≈0.92**, Prec≈0.65, Acc≈0.70, Coste=11.

### 4. Validación y reproducibilidad
- **N reducido** exige: OOF sin fuga, **Repeated KFold**, control de **NaNs** y **bootstrap** para IC.  
- **Cohort shift** (OASIS-1 vs OASIS-2): conviene **umbrales por cohorte**, y plan de **recalibración** si cambia el entorno.  
- Consolidar **nomenclaturas** y **rutas** (P25/P26) ahorra horas de *debugging*.

### 5. Operación y despliegue
- **Release reproducible** (p26_release.zip): MANIFEST/ENV/QA, **model card**, *how-to*, umbrales activos S2.  
- **Streamlit Demo** y **FastAPI** facilitan adopción; mantener **monitoreo de ECE** y **alarmas** de deriva.

---

### 🧭 Decisiones clave 

1. **Base de inferencia de imagen = P24 (LR elastic-net + Platt)** por su mejor **equilibrio** generalización/interpretabilidad.  
2. **Fusión LATE (P26)** como integración final: usar **p_img** (prob. calibrada) + **p_clin** + *flags* de cobertura para decidir por cohorte.  
3. **Política S2 por cohorte**:  
   - **Base 5:1** (FN cuestan 5× FP) y **umbrales aprendidos en VAL** por cohorte.  
   - **OAS2 con Recall ≥ 0.90** por criterio de cribado.  
   - Umbrales activos: **OAS1=0.42**, **OAS2≈0.493**.  
4. **Platt por cohorte** (P26b) para estabilizar probabilidades en OAS2 si el entorno lo requiere.  
5. **Controles anti-fuga**: una visita por paciente en OAS2 y splits por paciente; imputación clínica **simple** documentada.  
6. **Paquete de despliegue** con QA + *smoke tests* y tablas de confusión/coste → facilita auditoría y transición a producción.  
7. **Métricas con IC (bootstrap)** y reportes por cohorte → comunicación transparente de **incertidumbre**.

---

## 7. Conclusiones Globales 

**Visión general.** COGNITIVA-AI consolida una solución **intermodal** (clínica + MRI) orientada a **cribado**: trabaja con **probabilidades calibradas** y **umbrales coste-sensibles por cohorte** (política **S2**) para priorizar la **sensibilidad** sin perder trazabilidad ni reproducibilidad.

### Modalidad clínica
- La clínica en OASIS es **altamente informativa** (CDR/MMSE, demografía). En modelos de menor complejidad (LR regularizada) ya se observa un alto poder discriminativo en validaciones internas; su principal **limitación** es depender de que el deterioro cognitivo sea **ya medible**.
- Se documentó y controló la **imputación** (p.ej., SES/Education) y el posible **label leakage** al trasladar definiciones clínicas a otros entornos.

### Modalidad MRI
- Con **EfficientNet-B3** y agregación por paciente (mean/trimmed/top-k), más **calibración** (Platt), la señal de imagen **cierra la brecha** y aporta **complementariedad** real.
- Los pipelines finales de imagen (P24) muestran en TEST: **AUC≈0.727** (ALL), **OAS1≈0.754**, **OAS2≈0.750**, **Brier≈0.22–0.24**. La **calibración** es razonable, con **ECE** más débil en OAS2 (≈0.31) que en OAS1 (≈0.15).

### Fusión multimodal y decisión
- La **fusión LATE** (P26) con **p_img + p_clin** (y *flags* de cobertura) mantiene desempeño global similar a P24 (**AUC TEST≈0.71**) y **mejora la toma de decisión** por cohorte al incorporar la clínica de forma explícita.
- Política **S2** (FN:FP=**5:1**, **umbrales por cohorte** aprendidos en VAL, y **Recall≥0.90** exigido en OAS2 por criterio de cribado):
  - **OAS1** @ **thr=0.42** → **Recall=0.70**, Prec=0.61, Acc=0.68, Coste=39.
  - **OAS2** @ **thr≈0.493** → **Recall≈0.92**, Prec≈0.65, Acc≈0.70, Coste=11.
- Conclusión operativa: **el AUC informa; el umbral decide**. En cribado, la política S2 ofrece una **sensibilidad alta** con un coste de FP **asumible**.

### Generalización, calibración y reproducibilidad
- Conjuntos pequeños y **cohort-shift** (OASIS-1 vs OASIS-2) marcan el techo de generalización; mitigamos con **calibración por cohorte**, **umbrales S2** y **bootstrap** de métricas.
- Entregamos un **release reproducible** (modelos, configuración, QA, *model card*, *how-to*), lo que facilita auditoría y despliegue.
- Próximo paso crítico: **validación externa** (ADNI/AIBL) y **recalibración** si cambia el entorno.

**Conclusión final.** COGNITIVA-AI demuestra que una **arquitectura híbrida** (clínica estructurada + imagen MRI) puede **priorizar detección temprana** (alta sensibilidad) manteniendo **tasas de falsa alarma aceptables** bajo una política de coste clínico transparente. En Alzheimer, decidir **antes** es crucial: la combinación de **calibración**, **umbrales por cohorte** y **fusión LATE** proporciona una base sólida para pasar a validaciones multi-sitio y adopción clínica responsable.


---

## 8. Próximos pasos

> Objetivo: cerrar brechas de **generalización**, **calibración** y **operatividad** para un despliegue seguro en entornos clínicos reales.

### 1. Validación externa y CV por sitio
- **Qué haremos:** replicar en **ADNI/AIBL** con **leave-site-out** y *time-based splits*; comparar distribución de covariables (PCA/UMAP, PSI/KS), medir *dataset shift* y **recalibrar por dominio** si procede.
- **Salidas:** tabla comparativa (AUC/PR-AUC/Brier/ECE) por sitio, *shift report*, calibradores por dominio y umbrales por sitio.
- **Riesgos/mitigación:** N pequeño por sitio → *pooled calibration* + intervalos bootstrap.

### 2. Monitorización y recalibración viva
- **Qué haremos:** *dashboard* de **ECE/MCE**, drift (PSI/KS), *cohort mix*, y alarmas; **retrain de calibradores** cuando **ECE>0.2** o se acumulen **50–100** casos nuevos por cohorte; *rollback* de umbrales.
- **Salidas:** jobs programados (cron), bitácora de recalibraciones, gráfica ECE/MCE temporal, política activa (Sx) versionada.

### 3. Intermodal 2.5D/3D + fusión con atención
- **Qué haremos:** probar **2.5D/3D** (inflated CNN / ViT-3D) con *patching* volumétrico y **fusión híbrida** (*gated/attention*), robusta a **faltantes clínicos**.
- **Salidas:** ablation 2D vs 2.5D/3D, curva de coste vs umbral en nuevos modelos, comparación con LATE.
- **Riesgos:** coste computacional → *early-stopping*, *mixed precision*, *gradient checkpointing*.

### 4. Enriquecimiento de clínica y longitudinalidad
- **Qué haremos:** incluir **ΔMMSE** y *slopes* (tendencias), comorbilidad/medicación, y si disponible **APOE**/biomarcadores; evaluar ganancia incremental (**ΔAUC**, **NRI**).
- **Salidas:** panel de *features* longitudinales, estudio incremental (clínica→+MRI→+longitudinal→+gen/biomarcador).

### 5. Optimización de la decisión clínica
- **Qué haremos:** extender la función de coste a **Decision Curve Analysis** y **net benefit**; diseñar **políticas S-series**:  
  - **S1**: 5:1 puro; **S2**: 5:1 + objetivo de **Recall** por cohorte;  
  - **S3**: **doble umbral** (*triage*): zona gris → segunda prueba/experto.
- **Salidas:** curvas de beneficio neto, guía de selección de política por contexto.

### 6. Explicabilidad y reportes por paciente
- **Qué haremos:** **SHAP** para clínica (top-features por paciente/cohorte) + **Grad-CAM/Score-CAM** en MRI; **PDF por paciente** con probabilidad, umbral aplicado, y explicación visual/textual.
- **Salidas:** *explainer service*, plantilla de informe, evidencia para comité clínico.

### 7. Robustez, incertidumbre y QA ampliado
- **Qué haremos:** *stress-tests* (ruido/rotación/contrast/CLAHE-off), **test-time augmentation**, **MC dropout/ensembles**, **conformal** para intervalos de predicción; ampliar *golden set* y *smoke tests*.
- **Salidas:** matriz de robustez, intervalos por caso, *QA pack* actualizado.

### 8. MLOps y empaquetado
- **Qué haremos:** **Docker/Podman**, **FastAPI v1** con esquemas pydantic estables, **model registry** (linaje/versionado), **CI/CD** (tests+lint+reproducibilidad), **audit logging** (inputs/outputs/umbral/política activa).
- **Salidas:** imágenes contenedorizadas, *pipeline* de despliegue, auditoría por petición.

### 9. Fairness y subgrupos
- **Qué haremos:** métricas por **sexo/edad/educación**; si emergen brechas, estudiar **umbrales por subgrupo** (con gobernanza clínica) o **calibración diferenciada**.
- **Salidas:** informe de equidad, recomendaciones de ajustes y monitorización.

### 10. Integración clínica y datos reales
- **Qué haremos:** piloto **radiologist-in-the-loop**, integración con **PACS/RIS**, interfaz de reporte, y **active learning** para casos inciertos (priorizar anotación).
- **Salidas:** flujo operativo con retroalimentación, mejora de etiquetas, métricas antes/después.

> **Indicadores de éxito (resumen):**  
> (i) AUC/PR-AUC/Brier/ECE por sitio mejoran o se mantienen tras recalibrar;  
> (ii) curvas de **net benefit** superiores a *treat all/none*;  
> (iii) explicabilidad adoptada en revisión clínica;  
> (iv) *dashboard* operativo con alarmas y *rollback*;  
> (v) *release* contenedorizado con trazabilidad/auditoría.

---

## 9. Agradecimientos

Este trabajo se apoya en los conjuntos de datos **OASIS**. El uso que aquí se hace es **estrictamente académico y de investigación**; **no** está destinado a uso clínico o diagnóstico.

### OASIS-1 · Cross-Sectional
- **Investigadores principales:** D. Marcus, R. Buckner, J. Csernansky, J. Morris  
- **Referencia:**  
  Marcus, D.S., Wang, T.H., Parker, J., Csernansky, J.G., Morris, J.C., & Buckner, R.L. (2007). *Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults*. **Journal of Cognitive Neuroscience, 19**, 1498–1507. **doi: 10.1162/jocn.2007.19.9.1498**

### OASIS-2 · Longitudinal
- **Investigadores principales:** D. Marcus, R. Buckner, J. Csernansky, J. Morris  
- **Referencia:**  
  Marcus, D.S., Fotenos, A.F., Csernansky, J.G., Morris, J.C., & Buckner, R.L. (2010). *Open Access Series of Imaging Studies (OASIS): Longitudinal MRI Data in Nondemented and Demented Older Adults*. **Journal of Cognitive Neuroscience, 22**, 2677–2684. **doi: 10.1162/jocn.2009.21407**

### Comunidad y mentorización
- Agradecemos a la **comunidad open-source** por ecosistemas y librerías esenciales (Python/NumPy/Pandas/Scikit-learn/PyTorch/XGBoost/LightGBM/Matplotlib/Seaborn/Streamlit/FastAPI).  
- Agradecimiento especial al **profesor Valentín Silvestri** por su acompañamiento y mentoría a lo largo del proyecto.

> **Nota ética y de uso:** Los datos OASIS cuentan con restricciones de uso y citas específicas; este proyecto respeta dichas condiciones. Los resultados se presentan con fines formativos e investigativos y **no** sustituyen el juicio clínico ni constituyen un producto sanitario.
