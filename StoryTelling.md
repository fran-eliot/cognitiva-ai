# 🧠 COGNITIVA-AI — Storytelling y Presentación

## 1. Storytelling (Guion narrativo)

### 1️⃣ El problema invisible
- Narrativa: “Cada 3 segundos, una persona en el mundo desarrolla demencia.  
La enfermedad de Alzheimer es silenciosa: cuando los síntomas son claros… ya es tarde.”
- Visual sugerido: imagen en blanco y negro de una persona mayor desenfocada / reloj de arena.
- Mensaje clave: Necesitamos detectarlo **antes de los síntomas clínicos irreversibles**.

---

### 2️⃣ El reto clínico
- Narrativa: Los médicos cuentan con dos fuentes de información:  
  - Datos clínicos (tests neuropsicológicos, edad, educación, volumetría cerebral).  
  - Imágenes MRI (cambios cerebrales invisibles a simple vista).  
- Visual sugerido: dos columnas, izquierda 🧾 ficha clínica y derecha 🧠 resonancia.  
- Mensaje clave: ¿Y si la IA pudiera **combinar ambas miradas** como lo haría un neurólogo experto?

---

### 3️⃣ Nuestra propuesta: COGNITIVA-AI
- Narrativa: Creamos un sistema con **7 pipelines**, cada uno un paso más cerca de un diagnóstico temprano y confiable.
- Visual sugerido: timeline de los 7 pipelines (imagen generada en `/graficos/pipelines_timeline.png`).
- Mensaje clave: desde modelos simples con datos clínicos hasta **fine‑tuning en GPU** con calibración y umbral clínico.

---

### 4️⃣ Los descubrimientos
- Clínico:
  - Fusión OASIS-1+2 → AUC≈0.98 (robusto y estable).
  - Variables clave: **CDR + MMSE**.
- MRI:
  - ResNet50 (5 slices) → AUC=0.938.
  - ResNet18 + calibración → Recall≈0.80 en test.
  - Mejor estrategia intermedia: **Ensemble híbrido XGB+MLP** (AUC=0.744, PR-AUC=0.659).
- Visual sugerido: gráficos comparativos (AUC, PR-AUC, recall/precisión).
- Mensaje clave: los datos clínicos son muy fuertes, pero MRI añade valor crítico para la detección temprana.

---

## 🧩 Capítulo final: la consolidación del Fine-Tuning en GPU
Tras varios intentos de ajuste y depuración, el equipo consolida el **fine‑tuning de EfficientNet‑B3 en Colab GPU**.  
El modelo alcanza **sensibilidad perfecta (recall=1.0)** en validación y test, con **AUC de 0.876 y PR‑AUC de 0.762 en test**.  

Una matriz de confusión simple muestra el equilibrio:  
- TP=8, FP=5, TN=34, FN=0 (umbral=0.3651).  

Estas gráficas finales (confusión y barras comparativas de AUC y PR‑AUC) simbolizan el cierre de la **fase unimodal MRI**, listos para avanzar al proyecto **multimodal**.  

---

En la última extensión del pipeline 10 comprendimos que **ninguna técnica de pooling aislada era suficiente**: si bien la media recortada (TRIMMED) garantizaba una sensibilidad elevada, la precisión resultaba limitada. Inspirándonos en la práctica clínica, donde se consultan varias pruebas antes de un diagnóstico, creamos un **ensemble MRI** que combina tres miradas distintas de las imágenes (mean, trimmed y top-7 slices).  

El resultado fue un modelo más equilibrado: mantiene la capacidad de detectar a la mayoría de pacientes en riesgo (recall alto), pero reduce los falsos positivos incrementando la precisión. Este hallazgo refuerza la metáfora clínica de que **la combinación de perspectivas complementarias ofrece un diagnóstico más fiable que una visión aislada**.

---

Para comprobar si la diversidad de inicializaciones ayudaba, intentamos un *seed-ensemble* (tres semillas) manteniendo exactamente las transformaciones del cuaderno. 
El veredicto fue contundente: el *ensemble* por semillas no recuperó señal (AUC≈0.5), mientras que el *ensemble* por agregación de *slices* a paciente (combinando **mean**, **trimmed** y **top-k**) sí sostuvo el **recall** clínicamente deseado. 
La historia de Pipeline 10, por tanto, no es “más modelos”, sino “mejor agregación y calibración”.

---

### 5️⃣ El futuro multimodal
- Narrativa: El siguiente paso es **fusionar clínico + MRI**.  
- Visual sugerido: ilustración de dos ríos uniéndose en uno solo 🌊.  
- Mensaje clave: la verdadera fuerza está en lo multimodal, imitando la **intuición clínica**.

---

### 6️⃣ Cierre emocional
- Narrativa: “Detectar el Alzheimer antes no es solo un reto técnico: es **dar tiempo de calidad a millones de familias**.”
- Visual sugerido: familia abrazando a un abuelo.
- Mensaje clave: COGNITIVA-AI no solo predice, **da esperanza**.

---

## 2. Borrador de Slides

> **Nota**: cada slide debe tener **poco texto, gráficos/visuales potentes, y un título llamativo**.  
> El guion narrativo servirá para lo que el ponente cuenta mientras se muestran.

### Slide 1: Título
- “🧠 COGNITIVA-AI: Detección temprana de Alzheimer con IA Multimodal”
- Visual: logo del proyecto + imagen evocadora de cerebro.

### Slide 2: El problema invisible
- Texto breve:  
  “Cada 3 segundos alguien desarrolla demencia.  
   Cuando los síntomas son visibles… es tarde.”
- Visual: reloj de arena / rostro desenfocado.

### Slide 3: El reto clínico
- Dos columnas:  
  - Izq: Datos clínicos 🧾  
  - Der: MRI 🧠  
- Texto: “¿Y si la IA pudiera combinarlos como un neurólogo experto?”

### Slide 4: Nuestra propuesta
- Timeline de los **7 pipelines** (`pipelines_timeline.png`).
- Texto breve: “De modelos clásicos → a **fine‑tuning en GPU**.”

### Slide 5: Descubrimientos (Clínico)
- Bullet corto:  
  - AUC≈0.98 con OASIS-1+2.  
  - CDR + MMSE son clave.  
- Visual: gráfico comparativa clínica.

### Slide 6: Descubrimientos (MRI)
- Bullet corto:  
  - ResNet50 baseline: AUC=0.938.  
  - ResNet18 calibrado: Recall≈0.80.  
  - Ensemble XGB+MLP: AUC=0.744 / PR‑AUC=0.659.  
- Visual: gráficos AUC/PR-AUC MRI.

### Slide 7: Fine‑Tuning en GPU (B3)
- Bullet corto:  
  - TEST: AUC=0.876 | PR-AUC=0.762 | Recall=1.0 | Precision=0.625.  
  - Matriz (thr=0.3651): TP=8, FP=5, TN=34, FN=0.
- Visual: `ft_b3_patient_confusion_from_metrics.png` + barras AUC/PR-AUC.

### Slide 8: Comparativa Global
- Visual: `global_auc_comparison_updated.png`.
- Texto breve: “Clínico más fuerte, MRI añade valor para cribado.”

### Slide 9: El futuro multimodal
- Visual: fusión de ríos 🌊 o cerebro con dos mitades.  
- Texto breve: “Clínico + MRI → IA que imita la intuición médica.”

### Slide 10: Cierre emocional
- Texto grande:  
  “No es solo un reto técnico…  
   Es dar tiempo de calidad a millones de familias.”  
- Visual: familia abrazando a un abuelo.

---

## 3. Consideraciones para la Presentación

- **Duración sugerida:** 7–10 min.  
- **Ritmo narrativo:** 1–1.5 min por cada acto del storytelling.  
- **Estilo de slides:** poco texto, usar imágenes y los gráficos que ya tenemos.  
- **Emoción final:** cerrar con esperanza y utilidad social (impacto en pacientes y familias).  
- **Tip técnico:** evitar sobrecargar al público con métricas → solo destacar AUC, PR-AUC, Recall.  

---

# ✅ Conclusión
Este documento sirve como guion + storyboard inicial.  
De aquí se puede pasar fácilmente a **PowerPoint/Google Slides/Canva** usando las visualizaciones y frases clave.



Actualización: 28/08/2025 18:10
