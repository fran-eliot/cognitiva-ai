# 🧠 Proyecto de Detección Temprana de Alzheimer (COGNITIVA-AI)

Este proyecto explora la detección temprana de Alzheimer combinando **datos clínicos** y **resonancias magnéticas (MRI)** de la base pública **OASIS-2**.  

Se desarrollaron dos pipelines principales:  

1. 📊 **COGNITIVA-AI-CLINIC – Datos clínicos** (modelos clásicos de ML).  
2. 🖼️ **COGNITIVA-AI-IMAGES – Imágenes MRI** (Deep Learning con ResNet50).  

---

## 1️⃣ Datos clínicos (COGNITIVA-AI-CLINIC)

### 📂 Dataset
- Fichero: `oasis_longitudinal_demographics.xlsx`.  
- Variables principales:  
  - Demográficas: `Age`, `Sex`, `Education`, `SES`.  
  - Clínicas: `MMSE`, `CDR`.  
  - Medidas cerebrales: `eTIV`, `nWBV`, `ASF`.  
  - Target: `Group` → binarizado en **0=Nondemented, 1=Demented/Converted**.  

### ⚙️ Pipeline
- Preprocesamiento y selección de una visita por paciente.  
- Modelos entrenados:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Optimización:  
  - GridSearchCV  
  - Algoritmos genéticos (DEAP).  

### 📊 Resultados finales
| Modelo              | ROC-AUC (Test) |
|---------------------|----------------|
| Logistic Regression | ~0.91 (inicial) |
| Random Forest       | 0.884 |
| XGBoost             | **0.897** |

➡️ **XGBoost fue el mejor modelo clínico.**

---

## 2️⃣ Imágenes MRI (COGNITIVA-AI-IMAGES)

### 📂 Dataset
- Carpetas originales:  
```plaintext
OAS2_xxx_MRy/
├── RAW/ → mpr-1.hdr / mpr-1.img (hasta 4 volúmenes)
└── OLD/ (en algunos sujetos)
```
- Cada paciente tiene 2+ escaneos (`MR1`, `MR2`).  
- De cada volumen se extrajeron **5 cortes axiales centrales** y se guardaron en `.png`.  

### 🛠️ Preprocesamiento
1. **Normalización**: reescalado a 0–255.  
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) para mejorar contraste en imágenes oscuras (ej: sujetos 4, 14, 77, 103, 137, 145, 162).  
3. **Data augmentation** (solo en entrenamiento):  
 - Horizontal Flip, Rotation ±10°, Color Jitter.  
 - Resize a 224×224.  
 - Normalización con medias de ImageNet.  

### 🤖 Modelo
- **ResNet50** preentrenada en ImageNet.  
- Capa final modificada a salida binaria.  
- Entrenamiento con Adam (`lr=1e-4`), CrossEntropyLoss.  
- **Early stopping (paciencia=4)**.  

### 📑 Evaluación
- División **estratificada por paciente** (80% train, 20% test).  
- En test, se predice a nivel de **slice**, y luego se hace **media de probabilidades por paciente**.  

---

## 📊 Resultados en MRI

### 🔹 Versión inicial (sin CLAHE)
- Train Acc: >0.94  
- Test (nivel paciente):  
- Accuracy: **0.89**  
- ROC-AUC: **0.938**

### 🔹 Con CLAHE
- Train Acc: ~0.95  
- Test (nivel paciente):  
- Accuracy: **0.69**  
- ROC-AUC: **0.777**

➡️ **Conclusión:** CLAHE mejora visualmente, pero puede eliminar patrones sutiles de Alzheimer y reducir rendimiento.  

---

## 📌 Decisiones clave

✔️ Target binario (`Nondemented` vs `Demented/Converted`) → dataset pequeño.  
✔️ Una visita por paciente → evita fuga de información.  
✔️ Validación cruzada + ROC-AUC en clínico.  
✔️ Conversión NIfTI → PNG con 5 cortes axiales → equilibrio entre coste y representatividad.  
✔️ Evaluación **por paciente** en MRI → más realista clínicamente.  

---

## Resultados de Preprocesamiento en Imágenes

Durante el entrenamiento con imágenes MRI, detectamos que varias imágenes de pacientes presentaban
muy poco contraste (ej. sujetos 4, 14, 77, 103, 137, 145, 162), lo que hacía difícil identificar
patrones relevantes. Para abordar este problema probamos distintas técnicas de **preprocesamiento**:

### 1. Sin Preprocesamiento (baseline)
- Modelo: ResNet50 fine-tuned
- Accuracy en test (nivel paciente): **0.89**
- ROC-AUC: **0.94**
- Comentario: buen rendimiento, pero algunas imágenes eran casi invisibles.

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Objetivo: mejorar contraste de forma local.
- Resultados:  
  - Accuracy en test: **0.69**
  - ROC-AUC: **0.77**
- Comentario: aunque las imágenes eran más legibles visualmente, el modelo perdió rendimiento.
  Probablemente por exceso de realce de ruido.

### 3. CLAHE + Normalización Z-score por slice
- Objetivo: estabilizar intensidades tras aplicar CLAHE.
- Resultados:  
  - Accuracy en test: **0.72**
  - ROC-AUC: **0.82**
- Comentario: se recuperó parte del rendimiento perdido, pero sigue por debajo del baseline.
  Aun así, las predicciones son más balanceadas entre clases (mejor recall en Demented).

---

### Conclusiones parciales
- **El baseline sin preprocesamiento sigue siendo más sólido** (ROC-AUC 0.94).
- CLAHE ayuda a mejorar visualización humana, pero no necesariamente la discriminación del modelo.
- La normalización z-score por slice aporta estabilidad y balancea recall/precision.
- Próximos pasos: 
  - ajustar parámetros de CLAHE (clipLimit, tileGridSize),
  - probar normalización global por scan (en lugar de slice),
  - explorar preprocesamiento híbrido (CLAHE solo en sujetos oscuros).

---

## 📊 Comparativa final

| Modalidad       | Modelo                  | ROC-AUC (Test) |
|-----------------|-------------------------|----------------|
| Datos clínicos  | XGBoost optimizado      | **0.897** |
| Imágenes MRI    | ResNet50 (sin CLAHE)    | **0.938** |
| Imágenes MRI    | ResNet50 (con CLAHE)    | 0.777 |

---

## 🚀 Próximos pasos
- Probar arquitecturas más modernas (EfficientNet, DenseNet).  
- Usar volúmenes 3D completos (3D CNN).  
- Modelo multimodal que combine **datos clínicos + MRI**.  
- Ajustar parámetros de CLAHE o explorar normalización adaptada a MRI.  

---

✍️ **Autor:** *Proyecto académico de detección temprana de Alzheimer usando OASIS-2.*  
📅 **2025**  
