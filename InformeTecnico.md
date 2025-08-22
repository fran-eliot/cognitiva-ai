# COGNITIVA-AI — Detección Temprana de Alzheimer  
**Informe Técnico (Formal)**

---

## 1. Resumen
Este proyecto investiga la **detección temprana de Alzheimer** combinando **datos clínicos tabulares** y **resonancias magnéticas estructurales (MRI)** de los conjuntos **OASIS-1 y OASIS-2**.  

Se plantean cuatro pipelines:  
1. **COGNITIVA-AI-CLINIC** → datos clínicos tabulares (baseline).  
2. **COGNITIVA-AI-CLINIC-IMPROVED** → fusión OASIS-1+2 con calibración, interpretabilidad, robustez y ensembling.  
3. **COGNITIVA-AI-IMAGES** → Deep Learning con MRI (ResNet50).  
4. **COGNITIVA-AI-IMAGES-IMPROVED** → embeddings ResNet18 + clasificadores clásicos, calibración y evaluación paciente-nivel.  

Los resultados muestran que el pipeline clínico mejorado alcanza **ROC-AUC ≈ 0.985 (Nested CV)**, mientras que en imágenes:  
- OASIS-2 puro (ResNet50 fine-tuning) logra **AUC=0.938** (paciente).  
- OASIS-1+2 (ResNet18 embeddings + LR calibrado) logra **AUC=0.702 (test)**, confirmando la necesidad de optimizar fusión multimodal.  

---

## 2. Antecedentes y Motivación
La **Enfermedad de Alzheimer (EA)** es neurodegenerativa y progresiva. Una detección temprana es clave para:  
- Optimizar la atención clínica.  
- Planificar intervenciones.  
- Reducir costes en fases avanzadas.  

Los conjuntos **OASIS** proporcionan datos **abiertos y estandarizados** tanto clínicos como de neuroimagen. Este trabajo evalúa la capacidad predictiva de **modelos clásicos y deep learning**, como base para futuros sistemas multimodales.

---

## 3. Datos
- **OASIS-1 (Transversal):**  
  - 416 sujetos, 434 sesiones.  
  - Sin variable `Group`; severidad inferida con **CDR** (`0 = no demencia`, `>0 = demencia`).  

- **OASIS-2 (Longitudinal):**  
  - 150 sujetos, 373 sesiones.  
  - Variable `Group`: {Nondemented, Demented, Converted}.  
  - Varias visitas por sujeto.  

- **MRI:** archivos `.hdr/.img` por paciente, con segmentaciones asociadas (`FSL_SEG`).  

**Target binario:**  
- `0 = Nondemented`  
- `1 = Demented` o `Converted`  

---

## 4. Definición del Problema
- **Tarea:** Clasificación binaria a nivel de **paciente**.  
- **Retos:**  
  - Evitar *data leakage*.  
  - Manejo de múltiples visitas.  
  - Tamaño reducido de muestra.  
  - Preprocesamiento homogéneo entre cohortes.  

---

## 5. Pipeline Clínico

### 5.1 Preprocesamiento
- Homogeneización de columnas (`snake_case`).  
- Baseline en OASIS-2.  
- Target unificado: `Group` (OASIS-2) y `CDR` (OASIS-1).  
- Imputación SES/Educación.  
- Codificación one-hot (`Sex`).  
- Escalado estándar.  

### 5.2 Modelos
- LR, RF, XGB.  
- Validación estratificada (5-fold).  
- Métrica: ROC-AUC.  

### 5.3 Resultados
- **OASIS-2 (clínico):**  
  - LR → 0.912 ± 0.050  
  - RF → 0.925 ± 0.032  
  - XGB → 0.907 ± 0.032  
  - Mejor test: XGB = 0.897  

- **Fusión OASIS-1+2:**  
  - Hold-out: LR=1.000, RF=0.986, XGB=0.991  
  - CV: LR=0.979 ± 0.012, RF=0.974 ± 0.018, XGB=0.975 ± 0.021  

➡️ **Conclusión:** CDR + MMSE dominan el rendimiento, volumétricas aportan poco.

### 5.4 Mejoras avanzadas
- **Umbral clínico:** recall ≈ 100%, con 15 FP aceptables.  
- **Calibración:** LR + Isotónica (Brier=0.010).  
- **Robustez:** Nested CV 0.985 ± 0.011.  
- **Ablación:** sin CDR+MMSE AUC=0.76.  
- **Ensemble:** LR+RF+XGB → AUC=0.995.  

---

## 6. Pipeline Imágenes (OASIS-2)

### 6.1 Preprocesamiento
- Slices axiales (5 o 20).  
- Normalización, augmentations.  
- Input 224×224.  

### 6.2 Entrenamiento
- ResNet50 fine-tuning.  
- Optimizador Adam, early stopping.  
- Split paciente.  

### 6.3 Resultados
- 5 slices sin CLAHE → AUC=0.938 (test).  
- 20 slices + z-score → AUC=0.858.  

➡️ Dependiente de preprocesado, sensible a augmentations.  

---

## 7. Pipeline Imágenes Mejorado (OASIS-1+2)

### 7.1 Estrategia
- Embeddings ResNet18 (ImageNet).  
- Clasificadores clásicos: LR, SVC+Platt, RF, XGB.  
- Calibración isotónica.  
- Evaluación por paciente (probabilidades medias).  

### 7.2 Resultados
- **Linear probe (LR):**  
  - Val (paciente) → AUC=0.793  
  - Test (paciente) → AUC=0.707  

- **Model Zoo:**  
  - Mejor: **LinearSVC+Platt**, Val AUC=0.804, Test AUC=0.694.  

- **Calibración (LR):**  
  - Val: AUC=0.833, Brier=0.225  
  - Test: AUC=0.702, Brier=0.248  

- **Umbral clínico (thr=0.05):**  
  - Recall=1.0 en validación, aceptando más falsos positivos.  

➡️ Aunque los resultados son más bajos que en OASIS-2 puro, este pipeline sienta las bases para integrar OASIS-1 en MRI y combinarlo con clínicos.

---

## 8. Discusión
- Clínico fusionado → AUC≈0.985, interpretable y robusto.  
- Imágenes OASIS-2 → AUC≈0.94, buen baseline.  
- Imágenes OASIS-1+2 → AUC≈0.70, aún limitado.  
- Multimodalidad es clave para aumentar potencia y robustez.  

---

## 9. Limitaciones
- Muestra pequeña.  
- Uso de 2D en vez de 3D.  
- Dependencia del preprocesado.  
- Target binario simplificado.  

---

## 10. Reproducibilidad
- Semillas fijadas.  
- Escalado dentro de folds.  
- Código modular en notebooks.  

---

## 11. Futuras Líneas
1. Interpretabilidad avanzada (SHAP).  
2. Fusión multimodal (clínico + imágenes).  
3. Modelos 3D CNN o Transformers.  
4. Validación en OASIS-3 y ADNI.  
5. Optimización computacional (GPU).  

---

## 12. Conclusiones
- Clínico → modelos simples ya alcanzan alta precisión.  
- Imágenes → aportan información complementaria.  
- CDR y MMSE → predictores clínicos clave.  
- Calibración y umbral clínico → imprescindibles para uso real.  
- Próximo paso → **fusión multimodal**.  

---

## 📊 Comparativa Visual (AUC)

<p align="center">
  <img src="./graficos/comparativa.png" alt="Gráfico de barras comparativo" width="600"/>
</p>
