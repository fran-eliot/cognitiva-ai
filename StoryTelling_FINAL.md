# 🧠 COGNITIVA-AI — Storytelling Final (8 minutos)

> **Objetivo**: narrar, en 8′, cómo pasamos de modelos unimodales a un **sistema intermodal** (imagen + clínico) **calibrado por cohorte** y **coste-sensible** listo para demo y despliegue ligero.  
> **Público**: mixto (clínico + técnico). **Tono**: claro, visual, con mensajes clave.

---

## ⏱️ Estructura y tiempos
1. **El problema** (0:45)  
2. **Datos y reto** (0:45)  
3. **Evolución del proyecto** (1:00)  
4. **Modelo imagen (P24)** (1:00)  
5. **Intermodal (P26)** (1:00)  
6. **Calibración por cohorte + Política S2** (1:15)  
7. **Demo & experiencia de uso** (1:00)  
8. **Impacto, límites y próximos pasos** (1:15)

Total ≈ **8:00**

---

## 🎬 Slide 1 — Título y promesa
**Bullets**
- Cognitiva-AI: Detección temprana de Alzheimer con IA intermodal
- Combinamos **MRI + clínico** para priorizar **recall** (no dejar pacientes atrás)
- De investigación → **demo y despliegue ligero** (Streamlit + FastAPI + release ZIP)

**Notas del presentador**
- Apertura con el problema social (diagnóstico tardío) y la necesidad de cribado sensible.  
- En una frase: *unimos la señal de imagen con la clínica y la calibramos por cohorte.*

---

## 🔎 Slide 2 — El problema que no se ve
**Bullets**
- Cuando aparecen síntomas claros, a menudo llegamos tarde
- Cribado ≠ diagnóstico: **maximizamos sensibilidad** con coste controlado
- Decisión coste-sensible: **FN:FP = 5:1**

**Notas del presentador**
- En el cribado, **falsos negativos** son críticos. Por eso usamos una función de coste con FN mucho más caro que FP.

---

## 🧾 Slide 3 — Datos y reto
**Bullets**
- **OASIS-1** (transversal) y **OASIS-2** (longitudinal, **1ª visita/paciente**)
- **MRI**: 20 *slices* → agregación a **56 features/paciente**
- **Clínico**: homogeneización, imputación (Education/SES), OHE(Sex), **anti-fuga** (no usar CDR/Group como feature)

**Notas del presentador**
- Subrayar el control de *leakage* y el estándar por-paciente que permite ensembles y fusión posterior.

---

## 🛣️ Slide 4 — Camino p1→p27 (hitos)
**Bullets**
- **p11** Catálogo de backbones por paciente (56 features)
- **p22** Calibración (Platt/Isotónica) y análisis por cohorte
- **p24** Meta simple (LR elastic-net + Platt) con **p_img** robusto
- **p26** Intermodal (LATE/MID) + **p26b** calibración por cohorte + **S2** (política de decisión)

**Notas del presentador**
- Hacer zoom en p24 y p26b: base sólida + decisión coste-sensible práctica.

---

## 🧩 Slide 5 — Modelo de imagen (P24)
**Bullets**
- LR **elastic-net** + **Platt** sobre 56 features
- **TEST (probabilidades)**: Global **AUC=0.727**, OAS1 **0.754**, OAS2 **0.750**, Brier **0.220 (ALL)**
- Umbrales coste-óptimos (5:1) por cohorte: OAS1 **0.435**, OAS2 **0.332**

**Notas del presentador**
- P24 es la “columna vertebral” de la señal de imagen (`p_img`) que vamos a fusionar con clínica.

---

## 🔗 Slide 6 — Intermodal (P26) — LATE vs MID
**Bullets**
- **LATE**: media(`p_img`,`p_clin`) → simple, estable y robusto
- **MID**: concatenación IMG56+clínico+p1 → peor por N/covariables
- **TEST**: LATE **AUC=0.713**, PR-AUC **0.712**, Brier **0.234** (mejor que MID)

**Notas del presentador**
- LATE funciona mejor que MID en nuestro N por sencillez y menor sobreajuste.

---

## 🎛️ Slide 7 — Calibración por cohorte + Política **S2**
**Bullets**
- **Platt** por cohorte (OAS1/OAS2) + **coste 5:1**
- **S2**: mantener OAS2 con **Recall ≥ 0.90** (con ligero ajuste de umbral)
- Umbrales activos **S2**: **OAS1=0.42**, **OAS2≈0.4929**
- **Smoke TEST @S2**:  
  - OAS1 → TP=14, FP=9, TN=18, FN=6 → **R=0.70**, **P=0.609**, Acc=0.681, Coste=39  
  - OAS2 → TP=11, FP=6, TN=5, FN=1 → **R=0.917**, **P=0.647**, Acc=0.696, Coste=11

**Notas del presentador**
- Explicar la motivación: priorizar sensibilidad en la cohorte más “difícil” (OAS2), sin disparar el coste.

---

## 🖥️ Slide 8 — Demo (GUI y API)
**Bullets**
- **Streamlit**: carga de CSV, *switch* de política (P24/P26/S2), *sliders* por cohorte, métricas y **gráficos**
- **FastAPI**: `POST /predict` con `{clinical + features}` o `{clinical + p_img}`
- **CLI**: `compute_pimg_from_features.py` y `predict_end_to_end.py`

**Notas del presentador**
- Mostrar 1–2 capturas; enfatizar que está listo para **demo en vivo** o vídeo.

---

## 🌍 Slide 9 — Impacto, límites y próximos pasos
**Bullets**
- **Impacto**: prioriza **recall** y reduce FN (cribado más seguro)
- **Límites**: N reducido (OAS2), mayor **ECE** (0.313) → **recalibración y monitoring**
- **Próximos**: validación externa, *domain adaptation*, informe clínico por paciente

**Notas del presentador**
- Cerrar con la historia humana: *“dar tiempo de calidad”*.

---

## 🗂️ Slides de backup
**Bullets**
- Tabla comparativa P19/P22/P23/P24/P26/P26b (AUC/PR/Brier)
- Cost-curves y matrices de confusión por cohorte
- ECE/MCE (calibración)
- Arquitectura de carpetas y *release* (zip), reproducibilidad

**Notas del presentador**
- Preparado para preguntas técnicas y operativas.

