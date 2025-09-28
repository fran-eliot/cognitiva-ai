# 🧠 COGNITIVA-AI — Detección intermodal (Clínica + MRI) para cribado de Alzheimer

> **COGNITIVA-AI** es un proyecto de *screening* (cribado) de demencia/Alzheimer que **combina datos clínicos** con **señal de resonancia magnética (MRI)** para estimar **probabilidad de demencia** a nivel paciente y **tomar decisiones** con **umbrales por cohorte** alineados a **coste clínico** (priorizando *recall*).  
> Repositorio preparado para uso **técnico y no técnico**: incluye **scripts CLI**, **demo Streamlit**, **API FastAPI** y **paquete de release** con QA.

---

## 🧾 TL;DR (Resumen ejecutivo)

- **Problema:** clasificar **Control (0)** vs **Dementia/Converted (1)**, con foco en **minimizar falsos negativos** (cribado temprano).
- **Datos:** OASIS-1 (∼350 pacientes, *cross-sectional*) y OASIS-2 (∼150 pacientes, *longitudinal*). Imagen axial 2D (20 *slices* por paciente, *pooling*); clínica estructurada (edad, sexo, MMSE, CDR, etc.). **Una visita/paciente en OASIS-2** para evitar fuga.
- **Modelo final recomendado (producción):** **P26** (*intermodal, fusión Late* p_img + p_clin con calibración por cohorte P26b).  
  **Política activa: `S2`** → coste base **5:1 (FN:FP)** + **objetivo de Recall ≥ 0.90 en OAS2**.  
  **Umbrales:** **OAS1 = 0.42**, **OAS2 ≈ 0.493**.
- **Métricas (TEST, probabilidades):**
  - **P24 (imagen, LR elastic-net + Platt)** → **ALL** AUC=**0.727**, PR-AUC=**0.717**, Brier=**0.220**; **OAS1** 0.754/0.736/0.211; **OAS2** 0.750/0.805/0.238.
  - **P26 (intermodal, Late)** → **ALL** AUC=**0.713**, PR-AUC=**0.712**, Brier=**0.234**; **OAS1** 0.754/0.736/0.208; **OAS2** 0.652/0.728/0.288.
- **Decisión (TEST @ S2):**
  - **OAS1** @ 0.42 → **TP=14, FP=9, TN=18, FN=6**, Recall=**0.70**, Prec=0.61, Acc=0.681, **Coste=39**.
  - **OAS2** @ ≈0.493 → **TP=11, FP=6, TN=5, FN=1**, Recall=**0.92**, Prec=0.65, Acc=0.696, **Coste=11**.
- **Release:** `p26_release.zip` con **modelos**, **config S2**, **QA**, **docs** y **guías de uso**.

> Detalle *pipeline-por-pipeline*, ablations y tablas extensas → ver **InformeTecnico.md** (complementario a este README).

---

## 🧩 Datos y formulación

- **Cohortes (OASIS):**
  - **OASIS-1 (cross-sectional)**: una sesión por sujeto (∼350 pacientes).  
  - **OASIS-2 (longitudinal)**: múltiples visitas; aquí **se fija 1ª visita** (*una por paciente*) para evitar fuga temporal.
- **Etiquetas:**
  - **OASIS-2:** `Group ∈ {Nondemented, Demented, Converted}` → **Target=1** si *Demented/Converted*; **0** si *Nondemented*.
  - **OASIS-1:** se infiere **Target** desde **CDR** (0→Control, >0→Dementia).
- **Imagen (MRI):**
  - 20 *slices* axiales equiespaciados, normalización z-score (+ CLAHE opcional), *pooling* por paciente (`mean/trimmed20/top-k/pmean_2`).
  - *Backbones* y *ensembles* estandarizados; **EfficientNet-B3** es el más robusto en este entorno.
- **Clínica:**
  - Variables demográficas y neuropsicológicas (Age, Sex, Education, SES, MMSE, CDR, eTIV, nWBV, ASF, Delay).  
  - Imputación prudente (p. ej., SES/Education), *one-hot* de sexo, control de NaNs.
- **División:** *splits* **estratificados a nivel paciente** (sin fuga).

> **Intermodal (P26):** se combina **p_img** (probabilidad calibrada desde imagen) + **p_clin** (probabilidad clínica) vía **fusión Late** (meta-LR).

---

## 🗂️ Estructura del proyecto (carpetas principales)

```
cognitiva-ai/
├─ data/                         # Datos sin procesar o de ejemplo (no se versionan grandes ficheros)
├─ notebooks/                    # Notebooks (p19–p27) – histórico de experimentos
├─ models/                       # Modelos y calibradores operativos (copiados desde p26_release/models)
├─ release/                      # Copia de p26_release/ (QA, docs, modelos, config S2)
│  ├─ QA/                        # Figuras clave (ROC/PR/Cal/Coste), CSV de QA
│  └─ CONFIG/deployment_config.json  # Umbrales activos por cohorte (S2)
├─ src/                          # Código de librería para reuso
│  ├─ config.py                  # Carga de configuración (umbrales, rutas, etc.)
│  ├─ fuse.py                    # Utilidades de fusión Late y métricas
│  ├─ clinical.py                # Preprocesado e inferencia clínica
│  └─ imaging.py                 # Carga de features de imagen (pools) y calibración
├─ cli/                          # Scripts ejecutables (CLI)
│  ├─ compute_pimg_from_features.py
│  └─ predict_end_to_end.py
├─ app/                          # Aplicaciones
│  ├─ streamlit/app.py           # Demo interactiva (modo demo + subida de CSV)
│  └─ fastapi/main.py            # API REST /predict
├─ docs/                         # Documentación extendida
│  ├─ InformeTecnico.md          # Profundo y pipeline-a-pipeline
│  ├─ FASTAPI_GUIDE.md           # Uso de la API
│  ├─ app_streamlit.md           # Guía de la demo
│  ├─ README_SCRIPTS.md          # Guía CLI (scripts)
│  └─ QA_README.md               # Guía de verificación de release
├─ p26_release.zip               # Paquete de entrega (idéntico a release/)
└─ README.md                     # (Este documento)
```

---

## 🧪 Comparativa de *pipelines* (TEST, probabilidades)

| Pipeline | Cohorte | AUC  | PR-AUC | Brier |
|:--|:--:|---:|---:|---:|
| **P24** (imagen, LR elastic-net + Platt) | ALL  | **0.727** | **0.717** | **0.220** |
|  | OAS1 | 0.754 | 0.736 | 0.211 |
|  | OAS2 | 0.750 | 0.805 | 0.238 |
| **P26** (intermodal, fusión Late) | ALL  | **0.713** | **0.712** | **0.234** |
|  | OAS1 | 0.754 | 0.736 | 0.208 |
|  | OAS2 | 0.652 | 0.728 | 0.288 |

> **Lectura rápida:** P24 (imagen) ofrece excelente discriminación por cohorte; P26 añade **clínica** y permite **políticas por cohorte** con más control en decisión.

---

## ✅ Política de decisión **S2** (activa)

- **Base 5:1 (FN:FP)** con **umbrales por cohorte** aprendidos en **VAL**.
- **OAS2** ajustado para **Recall ≥ 0.90** (criterio de cribado).  
- **Umbrales activos:** **OAS1=0.42**, **OAS2≈0.4929**.
- **Confusión esperada (TEST @ S2):**
  - **OAS1**: TP=14, FP=9, TN=18, FN=6 → Recall=0.70, Prec=0.61, Acc=0.681, Coste=39.
  - **OAS2**: TP=11, FP=6, TN=5, FN=1 → Recall=0.92, Prec=0.65, Acc=0.696, Coste=11.

> **Importante:** la **decisión clínica no se toma con AUC**, sino con **probabilidades calibradas + umbrales** (política S2) que priorizan **sensibilidad** bajo coste controlado.

---

## 📦 Entregables principales

- **`p26_release.zip`**: modelos, calibradores, demo de QA, configuración de despliegue (**S2**), figuras y documentación.
- **CLI (carpeta `cli/`)**:  
  - `compute_pimg_from_features.py` → calcula **p_img** a partir de *features* de imagen ya agregadas por paciente.  
  - `predict_end_to_end.py` → predice con **p_img** (o *features*→p_img) + **clínica** y aplica política **S2** (umbrales por cohorte).
- **Streamlit (`app/streamlit/app.py`)**: demo gráfica para cargar CSV clínico y **mezclar con p_img** (o modo demo), incluye **sliders** de umbral/política y cálculo en vivo de **TP/FP/TN/FN** y **Coste**.
- **FastAPI (`app/fastapi/main.py`)**: **/predict** (JSON) con clínica + `p_img`; respuesta con probabilidad y **decisión** según S2.

> **QA**: ver `release/QA/QA_README.md` (cómo validar en 2–10 minutos).

---

## 🖥️ Requisitos del sistema

- **SO:** Windows 10/11, macOS 12+, Ubuntu 20.04+
- **Python:** 3.10–3.11
- **RAM:** ≥ 8 GB (recomendado 16 GB si generas *features* nuevas)
- **GPU (opcional):** sólo para *training*; **no necesaria** para inferencia con esta release.

### 📚 Tecnologías y librerías (principales)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modelado%20%26%20Calibraci%C3%B3n-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Backbones%20imagen-EE4C2C?logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensembles-EB5B28)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensembles-66BB6A)
![Pandas](https://img.shields.io/badge/Pandas-ETL%20%26%20CSV-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-C%C3%A1lculo%20num%C3%A9rico-013243?logo=numpy)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20gr%C3%A1fica-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API%20REST-009688?logo=fastapi&logoColor=white)

---

## ⚙️ Instalación y ejecución (rápido)

> Asegúrate de estar en la raíz del proyecto (`cognitiva-ai/`).

### 1) Crear entorno

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Probar *scripts* CLI

**Ejemplo 1 — Calcular `p_img` desde *features***
```bash
python cli/compute_pimg_from_features.py \
  --features_csv data/example_features.csv \
  --out_csv outputs/p_img_from_features.csv
```

**Ejemplo 2 — Predicción **end-to-end** (clínica + p_img)**
```bash
python cli/predict_end_to_end.py \
  --clinical_csv data/example_clinical.csv \
  --pimg_csv outputs/p_img_from_features.csv \
  --policy S2 \
  --out_csv outputs/predictions.csv
```

### 3) Demo Streamlit (interactivo)

```bash
streamlit run app/streamlit/app.py
```
- Modo **DEMO** (sin ficheros) o **subida de CSV** clínico + `p_img`.  
- **Sliders** de umbral/política, **TP/FP/TN/FN** y **coste** en vivo, **gráficos**.

### 4) API FastAPI

```bash
uvicorn app.fastapi.main:app --reload --host 0.0.0.0 --port 8000
```
**POST `/predict`** (JSON): clínica + `p_img` → probabilidad y decisión (S2).  
Ejemplo de *payload* en `docs/example_fastapi_payload.json`.

> Guías ampliadas: `docs/FASTAPI_GUIDE.md`, `docs/app_streamlit.md`, `docs/README_SCRIPTS.md`.

---

## ⚠️ Desafíos principales (resumen para todos los públicos)

- **Pocos pacientes** → la variabilidad es alta; se usan técnicas de calibración y validación cuidadosas.
- **Diferencias entre cohortes (OASIS-1 vs OASIS-2)** → se adoptan **umbrales por cohorte** y política **S2**.
- **Etiquetas clínicas imperfectas** (CDR/Group) → posible ruido; reportamos métricas con honestidad.
- **Calibración más débil en OAS2** → se recomienda **recalibración local** al desplegar en entornos similares.
- **Infraestructura** (Colab/Drive) → el **paquete de release** evita re-entrenar y facilita reproducibilidad.

---

## 📚 Lecciones aprendidas (versión README)

- **Clínica** aporta mucha señal; **imagen** añade valor complementario. La **fusión Late** + **calibración** es una receta robusta.  
- **EfficientNet-B3** rindió mejor de forma consistente como base para MRI 2D por *slices*.  
- **Calibrar es imprescindible** para tomar decisiones con umbrales; coste: puede bajar algo la precisión.  
- **Ensembles** ayudan de forma **moderada**; lo importante es la **diversidad real**.  
- La **organización** de artefactos y nombres coherentes ahorra horas; el **release con QA** es clave.

---

## 🚀 Próximos pasos (alto nivel)

- **Validación externa** (ADNI/AIBL) y **recalibración por sitio**.  
- **Monitorización** de **ECE** y **drift**; *re-training* de calibradores.  
- Explorar **2.5D/3D** y **explicabilidad** (SHAP para clínica, Grad-CAM para MRI).  
- **Decision Curve Analysis** y variantes de política (S-series: S1, S2, S3 *triage*).  
- Integración **clinico-operativa** (PACS/RIS, *radiologist-in-the-loop*).

---

## 🧭 Conclusiones

- **COGNITIVA-AI** ofrece un **cribado intermodal** **calibrado** y **accionable**: probabilidad → **umbral por cohorte (S2)** → decisión.  
- **Sensibilidad priorizada** (evitar FN) con **coste controlado**; métricas transparentes y reproducibles.  
- **Release operativo** (zip) + **CLI/Streamlit/FastAPI** facilitan pruebas locales y demos.

---

## 📑 Documentos relacionados

- **InformeTecnico.md** → Detalle profundo, ablations, resultados extensos y decisiones técnicas.  
- **docs/QA_README.md** → Verificación rápida del paquete de release.  
- **docs/FASTAPI_GUIDE.md** / **docs/app_streamlit.md** / **docs/README_SCRIPTS.md** → Guías de uso.

---

## 🙏 Agradecimientos (datos OASIS)

Datos procedentes de **OASIS** (uso académico).  
**OASIS-1 (Cross-Sectional)** — PIs: D. Marcus, R. Buckner, J. Csernansky, J.C. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.  
**Referencia:** *Marcus et al., 2007, J. Cognitive Neuroscience* (doi:10.1162/jocn.2007.19.9.1498).  
**OASIS-2 (Longitudinal)** — PIs: D. Marcus, R. Buckner, J. Csernansky, J.C. Morris; mismos fondos.  
**Referencia:** *Marcus et al., 2010, J. Cognitive Neuroscience* (doi:10.1162/jocn.2009.21407).

---

## 🔐 Licencia y uso

Proyecto educativo/demostrativo. **No** es un producto sanitario ni sustituye criterio clínico. Validación y certificación serían necesarias para uso asistencial.
