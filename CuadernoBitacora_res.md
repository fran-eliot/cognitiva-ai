# 🧭 Cuaderno de Bitácora del Proyecto Cognitiva-AI

Este cuaderno recopila **todo el recorrido del proyecto Cognitiva-AI**, desde los primeros experimentos con datos clínicos hasta los pipelines más recientes con arquitecturas alternativas y ensembles de backbones.  
Se ha mantenido un registro exhaustivo de cada fase, anotando decisiones técnicas, dificultades encontradas, soluciones aplicadas y reflexiones tras cada bloque de resultados.  
El objetivo es que actúe como un **diario técnico detallado**, útil tanto para revisiones futuras como para terceros interesados en reproducir o extender el trabajo.

---

## 📅 01/07/2025 – Inicio del proyecto

- **Fase inicial**: planteamiento general del proyecto.  
- Se define que Cognitiva-AI explorará **clasificación de enfermedad de Alzheimer** usando datos clínicos (OASIS-2) y resonancias magnéticas.  
- Se establecen los objetivos:  
  1. Validar la viabilidad con modelos clínicos tabulares (XGBoost).  
  2. Extender a MRI con backbones CNN/transformers.  
  3. Explorar calibración, ensembles y, finalmente, multimodalidad.

---

## 📅 03/07/2025 – Pipeline 1 (Clínico OASIS-2)

- **Notebook:** `p1_clinico_oasis2.ipynb`.  
- **Datos:** cohortes clínicas de OASIS-2.  
- **Modelo:** XGBoost.  
- **Resultados preliminares:**  
  - AUC ≈ 0.897.  
  - Buenas métricas en validación, confirmando que los datos clínicos son predictivos.  

**Reflexión:** excelente punto de partida, sirve de baseline. Se decide extender a fusión y multimodalidad más adelante.

---

## 📅 06/07/2025 – Pipeline 2 (Clínico Fusión)

- **Notebook:** `p2_clinico_fusion.ipynb`.  
- **Estrategia:** se fusionan variables clínicas tabulares adicionales.  
- **Modelo:** XGBoost mejorado.  
- **Resultados:**  
  - AUC ≈ 0.991.  
  - Recall casi perfecto (~1.0).  

**Reflexión:** métricas altísimas, posible riesgo de overfitting, pero muestra el potencial de fusión de datos tabulares.  
Se decide dar el salto a MRI.

---

## 📅 10/07/2025 – Pipeline 3 (MRI OASIS-2 con ResNet50)

- **Notebook:** `p3_mri_oasis2_resnet50.ipynb`.  
- **Datos:** imágenes MRI de OASIS-2.  
- **Backbone:** ResNet50 preentrenada en ImageNet.  
- **Resultados:**  
  - AUC (test) ≈ 0.938.  

**Reflexión:** confirmación de que los modelos CNN estándar son viables en MRI.  
Este pipeline sirve de puente hacia la fase Colab (con datos más grandes y pipelines posteriores).

---

## 📅 15/07/2025 – Pipeline 5 (MRI Colab con ResNet18 + Calibración)

- **Notebook:** `p5_mri_colab_resnet18_calib.ipynb`.  
- **Motivación:** probar pipeline en Colab con mayor escala y calibración.  
- **Resultados:**  
  - AUC ≈ 0.724.  
  - PR-AUC ≈ 0.606.  
  - Accuracy ≈ 0.60.  
  - Recall 0.80 | Precision 0.52.  

**Reflexión:** métricas más bajas que en OASIS-2, debido a mayor complejidad. Se confirma la necesidad de arquitecturas más potentes (EfficientNet).

---

## 📅 20/07/2025 – Pipeline 6 (EfficientNet-B3 embeddings)

- **Notebook:** `p6_mri_colab_effb3_embed.ipynb`.  
- **Enfoque:** usar EffNet-B3 como extractor de embeddings, clasificando con capa adicional.  
- **Resultados:**  
  - AUC ≈ 0.704.  
  - PR-AUC ≈ 0.623.  
  - Accuracy ≈ 0.70.  
  - Recall 0.90 | Precision 0.60.  

**Reflexión:** mejora en recall, aunque el modelo aún no se estabiliza.  
Se plantea probar fine-tuning completo.

---

## 📅 23/07/2025 – Pipeline 7 (EfficientNet-B3 fine-tune)

- **Notebook:** `p7_mri_colab_effb3_finetune.ipynb`.  
- **Motivación:** pasar de embeddings fijos a fine-tuning completo.  
- **Resultados:**  
  - AUC ≈ 0.876.  
  - PR-AUC ≈ 0.762.  
  - Accuracy ≈ 0.745.  
  - Recall 1.0 | Precision 0.625.  

**Reflexión:** salto cualitativo, confirma que EffNet-B3 es un backbone sólido para MRI.  
Se establece como baseline.

---

## 📅 30/07/2025 – Pipeline 9 (EfficientNet-B3 stable)

- **Notebook:** `p9_mri_colab_effb3_stable.ipynb`.  
- **Motivación:** buscar estabilidad entre runs, reduciendo variabilidad.  
- **Resultados:**  
  - AUC ≈ 0.740.  
  - PR-AUC ≈ 0.630.  
  - Accuracy ≈ 0.72.  
  - Recall 0.65 | Precision 0.62.  

**Reflexión:** mejora en estabilidad, pero con ligera pérdida de recall.  
Sirve como base para el pipeline 10.

---

## 📅 05/08/2025 – Pipeline 10 (EfficientNet-B3 stable + calibración)

- **Notebook:** `p10_mri_colab_effb3_stable_plus.ipynb`.  
- **Objetivo:** añadir calibración (temperature scaling, isotonic).  
- **Resultados:**  
  - AUC (test) ≈ 0.546–0.583.  
  - PR-AUC ≈ 0.50–0.53.  
  - Accuracy ≈ 0.51–0.55.  
  - Recall 1.0 | Precision ≈ 0.47–0.49.  

**Reflexión:** caída de métricas tras calibración, pero resultados más interpretables.  
Se descubre la importancia de ensembles para recuperar rendimiento.

---

## 📅 10/08/2025 – Pipeline 10-ext (TRIMMED y ensembles)

- **Notebook:** `p10ext_mri_colab_effb3_ext.ipynb`.  
- **Estrategia:** variantes TRIMMED y ensembles de MEAN/TRIMMED/TOP7.  
- **Resultados:**  
  - TRIMMED: AUC ≈ 0.744, PR-AUC ≈ 0.746.  
  - Ensemble: AUC ≈ 0.754, PR-AUC ≈ 0.737.  

**Reflexión:** ensembles simples logran mejoras claras.  
Refuerza la idea de avanzar hacia ensembles más sofisticados.

---

## 📅 15/08/2025 – Pipeline 11 (Backbones alternativos)

- **Notebook:** `cognitiva_ai_backbones.ipynb`.  
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

## 📅 20/08/2025 – Ensembles de Backbones

- **Objetivo:** combinar predicciones slice-level y patient-level de varios backbones (Swin, ConvNeXt, DenseNet).  
- **Métodos:**  
  - Promedios simples.  
  - Random weights (Dirichlet).  
  - Stacking (logistic regression, isotonic calibration).  

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

## 🔍 Desafíos principales encontrados

1. **Inestabilidad en Colab:** sesiones largas provocaban errores o pérdida de conexión. Reinicios forzados solucionaron varios problemas.  
2. **Gestión de Google Drive:** errores frecuentes de montaje/desmontaje y rutas inconsistentes, resueltos con reinicios y verificaciones explícitas.  
3. **Variabilidad de resultados:** seeds distintas producían métricas diferentes; se resolvió con ensembles y calibración.  
4. **Dificultad en calibración:** temperature scaling mejoraba interpretabilidad pero bajaba AUC. Hubo que combinar con ensembles.  
5. **Backbones alternativos:** algunos decepcionaron (DenseNet) o no superaron a EffNet, confirmando que no hay “ganador absoluto”.  
6. **Complejidad de ensembles:** métodos como Dirichlet o Stacking mostraron sobreajuste en validación y peores métricas en test.  
7. **Limitación de datos:** tamaño reducido del dataset afectó a generalización, especialmente en arquitecturas grandes como Swin.  
8. **Gestión de logs y CSV:** múltiples formatos distintos (`y_score`, `sigmoid(logit)`, etc.), lo que exigió unificación manual en varios experimentos.

---

## 📌 Estado actual y próximos pasos

- Pipelines clínicos y MRI probados, con EffNet-B3 como backbone base más sólido.  
- Backbones alternativos probados y documentados.  
- Ensembles explorados con distintos enfoques, aunque sin mejora clara en test.  

**Próximos pasos:**  
1. Probar ensembles híbridos (EffNet-B3 + Swin/ConvNeXt).  
2. Introducir datos multimodales (clínicos + MRI) en un solo modelo.  
3. Optimizar regularización y calibración avanzada.  
4. Ampliar el dataset (otras cohortes) para mejorar generalización.

---

# ✅ Conclusión

Este cuaderno refleja el progreso continuo del proyecto Cognitiva-AI.  
Se han probado **11 pipelines principales**, numerosos experimentos de calibración y ensembles, así como múltiples backbones alternativos.  
Cada fase ha aportado aprendizajes valiosos, aunque los mejores resultados aún provienen de **EfficientNet-B3 finetune (Pipeline 7)** y **EfficientNet-B3 ensembles calibrados (Pipeline 10-ext)**.  
El camino hacia multimodalidad queda abierto, apoyado en todo este historial experimental.

---
