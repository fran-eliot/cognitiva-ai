# 🧠 Cognitiva-AI: Exploración de modelos para predicción de Alzheimer con datos clínicos y de resonancia magnética

## 📌 Contexto y objetivos

El proyecto **Cognitiva-AI** nace con la motivación de estudiar la **predicción del Alzheimer** utilizando tanto **datos clínicos tabulares** como **imágenes cerebrales (MRI)**.  
La hipótesis central que guía todo el trabajo es que **distintas fuentes de información y distintas arquitecturas de modelos pueden capturar facetas complementarias del proceso neurodegenerativo**.  

A lo largo de más de diez pipelines (P1–P11) se ha seguido una filosofía de **experimentación incremental**:  
- comenzar con modelos sencillos sobre datos clínicos,  
- avanzar hacia CNNs entrenadas sobre imágenes MRI,  
- introducir calibración, normalización y estrategias de ensemble,  
- explorar arquitecturas modernas de visión,  
- y finalmente preparar el terreno hacia un modelo multimodal.

El dataset de referencia ha sido **OASIS** (Open Access Series of Imaging Studies), en particular sus cohortes OASIS-2 y derivados.  

---

## 🚀 Evolución por Pipelines

A continuación, se detallan los diferentes pipelines experimentales desarrollados en el proyecto. Cada uno introduce una idea nueva, una arquitectura diferente o una estrategia de evaluación alternativa.

### 🔹 P1 – Modelo clínico OASIS-2
- Datos: variables clínicas tabulares de OASIS-2.  
- Modelo: XGBoost.  
- Objetivo: establecer baseline con datos clínicos sin imágenes.  
- Resultado: AUC ≈ 0.897. Buen rendimiento inicial, pero limitado en capacidad de generalización.

### 🔹 P2 – Modelo clínico fusionado
- Datos: combinación de cohortes clínicas.  
- Modelo: XGBoost.  
- Observación: el aumento de datos clínicos mejora la capacidad predictiva.  
- Resultado: AUC ≈ 0.991. Rendimiento casi perfecto, pero sospechoso de overfitting.

### 🔹 P3 – Imágenes OASIS-2 con ResNet50
- Primer acercamiento a MRI.  
- Backbone: ResNet50, entrenado desde cero sobre imágenes OASIS-2.  
- Resultado: AUC ≈ 0.938. Buen desempeño, pero dataset muy pequeño.

### 🔹 P5 – MRI Colab con ResNet18 + calibración
- Migración a Colab para usar GPUs.  
- Modelo: ResNet18 con calibración posterior.  
- Resultado: AUC = 0.724. Recall alto (0.80), precisión limitada (0.52).

### 🔹 P6 – MRI Colab con EfficientNet-B3 (embeddings)
- Uso de embeddings preentrenados con EfficientNet-B3.  
- Resultado: AUC = 0.704. Recall elevado (0.90), precisión = 0.60.

### 🔹 P7 – MRI Colab con EfficientNet-B3 (fine-tuning)
- Se entrena EfficientNet-B3 completo sobre MRI.  
- Resultado: AUC = 0.876. Excelente recall (1.0) y precisión aceptable (0.625).  

### 🔹 P9 – MRI Colab con EfficientNet-B3 Stable
- Introducción de variantes con normalización y seeds más estables.  
- Resultado: AUC ≈ 0.740.  
- Observación: se gana estabilidad pero no mejora de forma radical.

### 🔹 P10 – MRI Colab con EfficientNet-B3 Stable + calibración
- Se introduce calibración explícita (Platt scaling, temperature scaling).  
- Resultados moderados: AUC entre 0.546–0.583.  
- Recall alto pero precisión baja.  
- Se documentan dificultades de estabilidad y saturación de logits.

### 🔹 P10-ext – Extensiones de ensembles sobre EfficientNet-B3
- Variantes TRIMMED, mean y top-k sobre diferentes seeds.  
- Mejoras modestas:  
  - TRIMMED: AUC = 0.744.  
  - Ensemble (M+T+7): AUC = 0.754.  
- Conclusión: el ensembling estabiliza pero no revoluciona.

### 🔹 P11 – Backbones alternativos
- Motivación: comprobar si otras arquitecturas modernas superaban a EfficientNet-B3.  
- Modelos probados:  
  - **ResNet-50 (baseline alt)**: AUC ≈ 0.740.  
  - **DenseNet-121**: bajo rendimiento, AUC ≈ 0.343.  
  - **ConvNeXt-Tiny (in12k_ft_in1k)**: AUC ≈ 0.509.  
  - **Swin-Tiny**: resultados mixtos, mejor variante top7 con AUC ≈ 0.641.  
- Conclusión: ninguno supera claramente a EffNet-B3, aunque ResNet50 y SwinTiny muestran competitividad parcial.  
- Ensembles de backbones (Dirichlet, isotonic, stacking): mejoras parciales pero sin “game changer”.

---

## 📊 Tabla de Resultados Globales

| Pipeline | Modalidad        | Modelo                                   | AUC (Test) | PR-AUC | Acc   | Recall | Precision |
|----------|-----------------|------------------------------------------|------------|--------|-------|--------|-----------|
| P1       | Clínico OASIS-2 | XGB                                      | 0.897      | —      | —     | —      | —         |
| P2       | Clínico fusion  | XGB                                      | 0.991      | —      | —     | ~1.0   | —         |
| P3       | MRI OASIS-2     | ResNet50                                 | 0.938      | —      | —     | —      | —         |
| P5       | MRI Colab       | ResNet18 + Calib                         | 0.724      | 0.606  | 0.60  | 0.80   | 0.52      |
| P6       | MRI Colab       | EffNet-B3 embed                          | 0.704      | 0.623  | 0.70  | 0.90   | 0.60      |
| P7       | MRI Colab       | EffNet-B3 finetune                       | 0.876      | 0.762  | 0.745 | 1.0    | 0.625     |
| P9       | MRI Colab       | EffNet-B3 stable                         | 0.740      | 0.630  | 0.72  | 0.65   | 0.62      |
| P10      | MRI Colab       | EffNet-B3 stable+calib                   | 0.546–0.583| 0.50–0.53 | 0.51–0.55 | 1.0 | 0.47–0.49 |
| P10-ext  | MRI Colab       | EffNet-B3 + TRIMMED                      | 0.744      | 0.746  | 0.64  | 0.75   | 0.56      |
| P10-ext  | MRI Colab       | EffNet-B3 + Ensemble(M+T+7)              | 0.754      | 0.737  | 0.68  | 0.70   | 0.61      |
| P11      | MRI Colab       | ResNet-50 (baseline alt)                 | 0.740      | 0.730  | 0.64  | 0.70   | 0.56      |
| P11      | MRI Colab       | DenseNet-121 + trimmed20                 | 0.343      | 0.407  | 0.32  | 0.75   | 0.36      |
| P11      | MRI Colab       | ConvNeXt-Tiny (in12k_ft_in1k) + mean     | 0.509      | 0.479  | 0.49  | 1.00   | 0.45      |
| P11      | MRI Colab       | Swin-Tiny + top7                         | 0.641      | 0.597  | 0.55  | 0.95   | 0.95      |
| P11-Ens  | MRI Colab       | BackboneEnsemble_DIRICHLET_means         | 0.520      | 0.523  | 0.47  | 1.0    | 0.44      |
| P11-Ens  | MRI Colab       | BackboneEnsemble_DIRICHLET_EXT           | 0.361      | 0.405  | 0.45  | 0.85   | 0.42      |
| P11-Ens  | MRI Colab       | STACK_L1_STRONG (L1-regularized stacking)| 0.500      | 0.426  | 0.43  | 1.0    | 0.43      |
| P11-Ens  | MRI Colab       | SwinTiny_top7_ISOTONIC                   | 0.566      | 0.458  | 0.55  | 0.95   | 0.49      |

---

## 📌 Conclusiones globales

1. **EfficientNet-B3** sigue siendo el backbone más robusto y estable en este dataset.  
2. **ResNet-50** es competitivo y sorprendentemente sólido en comparación con modelos más recientes.  
3. **DenseNet-121** no ha mostrado buen rendimiento en este dominio.  
4. **ConvNeXt y Swin** presentan interés, pero su rendimiento es irregular y dependiente del pooling.  
5. Los **ensembles de backbones** son prometedores pero, en este dataset pequeño, sufren de sobreajuste y métricas inconsistentes.  
6. El trade-off entre **recall alto y precisión baja** es recurrente y debe tenerse en cuenta para aplicaciones clínicas.  

---

## ⚠️ Desafíos principales del proyecto

1. **Limitaciones del dataset**  
   - Pocas muestras en comparación con el tamaño de los modelos.  
   - Cohorte no balanceada.  
   - Dificultad para extraer conclusiones generalizables.  

2. **Problemas técnicos en Colab/Drive**  
   - Fallos frecuentes en el montaje de Google Drive.  
   - Archivos no detectados hasta reiniciar entornos.  
   - Interrupciones en ejecuciones largas.  

3. **Estabilidad de entrenamiento**  
   - Seeds distintos producían resultados muy dispares.  
   - Necesidad de normalizar logits y calibrar salidas.  

4. **Backbones modernos**  
   - ConvNeXt y Swin no superaron a ResNet o EfficientNet en este contexto.  
   - Requieren datasets más grandes o tuning avanzado.  

5. **Estrategias de ensemble**  
   - Dirichlet y stacking mostraron mejoras en validación, pero caídas fuertes en test.  
   - Se evidencia sobreajuste al conjunto de validación.  

6. **Recall vs Precisión**  
   - Muchos modelos sacrifican precisión para alcanzar recall de 1.0.  
   - En contexto clínico, esto puede ser aceptable, pero requiere más refinamiento de umbrales.  

---

## 🔮 Próximos pasos

1. **Exploración multimodal**: combinar clínico + MRI en un mismo modelo.  
2. **Refinamiento de ensembles**: probar métodos bayesianos y calibración jerárquica.  
3. **Curación del dataset**: aumentar tamaño o usar pretraining más cercano al dominio biomédico.  
4. **Optimización clínica**: ajustar métricas de evaluación para alinearlas con necesidades reales (ej. F1 ponderado, métricas de utilidad clínica).  

---

# ✨ Cierre

Este README resume el largo proceso de investigación de **Cognitiva-AI**, desde pipelines iniciales con XGBoost hasta exploraciones recientes con arquitecturas modernas y ensembles complejos.  
Aunque no se ha encontrado un modelo que supere con claridad a EfficientNet-B3, se ha construido una base sólida para la siguiente fase: **el aprendizaje multimodal**.

