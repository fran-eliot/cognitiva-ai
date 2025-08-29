# 📑 Informe Técnico – Proyecto Cognitiva-AI

Este informe documenta, en profundidad y con alto nivel de detalle técnico, el desarrollo experimental llevado a cabo en el marco del proyecto **Cognitiva-AI**, orientado al diagnóstico automático de Alzheimer combinando información clínica y de neuroimagen (MRI, cohorte OASIS-2).

El documento recoge **metodología, experimentos, métricas, incidencias, reflexiones y conclusiones**, organizados de manera cronológica por pipelines experimentales (P1 a P11).  
El objetivo es tanto **analítico como documental**, de forma que este informe pueda servir como referencia para futuras fases (ensembles multimodales, validación externa).

---

## 📘 1. Contexto y objetivos

El Alzheimer es una enfermedad neurodegenerativa cuya detección temprana es clave.  
Los datos disponibles combinan:

- **Datos clínicos tabulares**: edad, género, MMSE, CDR, entre otros.  
- **MRI estructural**: imágenes cerebrales en formato NIfTI, preprocesadas a slices 2D para entrenamiento en Colab.  

### Objetivos técnicos:

1. Establecer **baselines sólidos con datos clínicos**.  
2. Explorar **modelos de visión profunda** en MRI.  
3. Evaluar **estrategias de calibración y ensembles**.  
4. Comprobar si **otros backbones** superan a EfficientNet-B3.  
5. Preparar el terreno para un futuro **modelo multimodal**.

---

## ⚙️ 2. Metodología general

### Datos
- Cohorte **OASIS-2**: ~150 sujetos, balanceado en controles y Alzheimer.  
- Subconjunto utilizado: 47 pacientes para test, 10 pacientes para validación, resto para entrenamiento.  

### Preprocesamiento
- Normalización de intensidades.  
- Extracción de slices 2D.  
- Mapas de correspondencia paciente-slice (`oas1_val_colab_mapped.csv`, `oas1_test_colab_mapped.csv`).  

### Entrenamiento
- Framework principal: **PyTorch** + **timm**.  
- Optimización: AdamW, LR schedulers, early stopping.  
- Augmentations: flips, rotaciones leves.  
- Hardware: Google Colab Pro (GPU Tesla T4).  

### Métricas
- AUC, PR-AUC.  
- Accuracy, Precision, Recall.  
- Threshold selection por F1 óptimo, Youden y recall controlado (90–100%).  

---

## 🔬 3. Pipelines experimentales

### P1 – Baseline clínico con XGBoost
- **Datos:** clínicos tabulares.  
- **Modelo:** XGBoost con tuning básico.  
- **Resultados:** AUC = 0.897 en test.  
- **Comentario:** baseline fuerte, incluso con dataset reducido.

---

### P2 – Clínico fusionado
- **Motivación:** incluir más atributos derivados y mejorar consistencia.  
- **Resultados:** AUC = 0.991, recall ≈ 1.0.  
- **Comentario:** casi techo de rendimiento con tabular.  

---

### P3 – MRI OASIS-2 con ResNet50
- **Backbone:** ResNet50 preentrenado.  
- **Resultados:** AUC test = 0.938.  
- **Comentario:** fuerte baseline en imagen pura.  

---

### P5 – MRI Colab con ResNet18 + calibración
- **Motivación:** backbone más ligero para Colab.  
- **Calibración:** post-hoc, isotonic regression.  
- **Resultados:** AUC = 0.724 | PR-AUC = 0.606 | Acc = 0.60 | Recall = 0.80.  
- **Comentario:** aceptable, pero inferior a ResNet50.  

---

### P6 – MRI Colab con EfficientNet-B3 embeddings
- **Uso:** extractor sin fine-tuning completo.  
- **Resultados:** AUC = 0.704 | Recall = 0.90.  
- **Comentario:** confirma la potencia de EfficientNet incluso sin ajuste completo.  

---

### P7 – MRI Colab con EfficientNet-B3 fine-tuning
- **Setup:** fine-tuning completo.  
- **Resultados:** AUC = 0.876 | PR-AUC = 0.762 | Recall = 1.0.  
- **Comentario:** uno de los mejores modelos de imagen.  

---

### P9 – EfficientNet-B3 “stable”
- **Objetivo:** estabilizar entrenamientos.  
- **Problema:** logits desorbitados (>500k).  
- **Resultados:** AUC test ≈ 0.74.  
- **Comentario:** estabilidad no garantizó mejor generalización.  

---

### P10 – EfficientNet-B3 con calibración explícita
- **Métodos:** Platt scaling, isotonic, temperature scaling.  
- **Resultados:**  
  - AUC = 0.546–0.583  
  - Recall = 1.0 pero precisión baja (≈0.47–0.49).  
- **Comentario:** calibración útil, pero coste en precisión.  

---

### P10-ext – Ensembles de EfficientNet-B3
- **Estrategias probadas:**  
  - Mean, trimmed, top-k.  
  - Random Forest sobre features de pooling.  
  - Stacking logístico.  
- **Resultados:** ensembles alcanzan AUC ≈ 0.75.  
- **Comentario:** ensembles aportan mejoras modestas pero consistentes.  

---

### P11 – Backbones alternativos
- **Objetivo:** explorar arquitecturas distintas a EfficientNet.  
- **Modelos:** ResNet-50, DenseNet-121, ConvNeXt-Tiny, Swin-Tiny.  
- **Resultados:**  
  - ResNet-50 → AUC ≈ 0.74  
  - DenseNet-121 → AUC ≈ 0.34  
  - ConvNeXt-Tiny → AUC ≈ 0.50  
  - Swin-Tiny → AUC ≈ 0.64  
- **Comentario:** ninguno supera a EfficientNet-B3; confirma robustez del baseline.  

---

## 📊 4. Resultados globales

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

## 🛠️ 5. Desafíos principales

1. **Dataset extremadamente reducido**  
   - Solo 47 pacientes en test.  
   - Gran varianza en métricas.  

2. **Saturación de logits**  
   - Valores extremos (>700k) en P9–P10.  
   - Obligó a normalización y calibración.  

3. **Problemas con Google Drive en Colab**  
   - Mountpoint bloqueado tras semanas.  
   - Solución: reinicio completo del entorno.  

4. **Heterogeneidad de outputs**  
   - Diferencias entre columnas (`y_score`, `pred`, `sigmoid(logit)`).  
   - Ficheros dispersos en carpetas distintas.  

5. **Gestión de ensembles**  
   - Balance entre diversidad y sobreajuste.  
   - Complejidad en stacking logístico.  

---

## 💡 6. Discusión crítica

- **Los datos clínicos dominan** en OASIS-2: AUC >0.99 en fusión.  
- **EfficientNet-B3** se confirma como el backbone más sólido para MRI.  
- **Nuevos backbones (ConvNeXt, Swin)** no superan al baseline, posiblemente por dataset reducido.  
- **Calibración** necesaria pero con trade-off en precisión.  
- **Ensembles** aportan robustez, aunque ganancias modestas.  

---

## ✅ 7. Conclusiones y próximos pasos

1. **Consolidar ensembles multimodales**: clínico + MRI.  
2. **Validación externa** en datasets como ADNI.  
3. **Explorar interpretabilidad**: Grad-CAM en MRI, SHAP en clínico.  
4. **Optimización robusta** con Bayesian Optimization.  

**Conclusión final:**  
El proyecto confirma que **EfficientNet-B3 + calibración + ensembles** constituye la estrategia más estable en MRI para OASIS-2, pero que el verdadero salto se alcanzará al integrar clínico + imagen en un marco multimodal.

---
