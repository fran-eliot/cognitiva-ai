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
- Narrativa: Creamos un sistema con **6 pipelines**, cada uno un paso más cerca de un diagnóstico temprano y confiable.
- Visual sugerido: timeline de los 6 pipelines (imagen generada en `/graficos/pipelines_timeline.png`).
- Mensaje clave: desde modelos simples con datos clínicos hasta redes neuronales calibradas en GPU.

---

### 4️⃣ Los descubrimientos
- Clínico:
  - Fusión OASIS-1+2 → AUC≈0.98 (robusto y estable).
  - Variables clave: **CDR + MMSE**.
- MRI:
  - ResNet50 (5 slices) → AUC=0.938.
  - ResNet18 + calibración → Recall≈0.80 en test.
  - Mejor estrategia actual: **Ensemble híbrido XGB+MLP** (AUC=0.744, PR-AUC=0.659).
- Visual sugerido: gráficos comparativos (AUC, PR-AUC, recall/precisión).
- Mensaje clave: los datos clínicos son muy fuertes, pero MRI añade valor crítico para la detección temprana.

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
- Timeline de los **6 pipelines** (imagen `pipelines_timeline.png`).
- Texto breve: “De modelos clásicos → a redes calibradas en GPU.”

### Slide 5: Descubrimientos (Clínico)
- Bullet corto:  
  - AUC≈0.98 con OASIS-1+2.  
  - CDR + MMSE son clave.  
- Visual: gráfico comparativa clínica.

### Slide 6: Descubrimientos (MRI)
- Bullet corto:  
  - ResNet50 baseline: AUC=0.938.  
  - ResNet18 calibrado: Recall≈0.80.  
  - Mejor: Ensemble XGB+MLP.  
- Visual: gráficos AUC/PR-AUC MRI.

### Slide 7: Comparativa Global
- Visual: gráfico `global_auc_comparison.png`.
- Texto breve: “Clínico más fuerte, MRI añade valor para cribado temprano.”

### Slide 8: El futuro multimodal
- Visual: fusión de ríos 🌊 o cerebro con dos mitades uniéndose.  
- Texto breve: “Clínico + MRI → IA que imita la intuición médica.”

### Slide 9: Cierre emocional
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
