# 🧭 Cuaderno de Bitácora del Proyecto Cognitiva-AI 
> Diario técnico detallado (por días) con decisiones, incidencias y resultados.  
> Objetivo: trazabilidad completa desde la preparación del entorno hasta backbones alternativos y ensembles.

---

## 📌 Convenciones y notas rápidas

- **Estructura de datos**:
  - `BASE_DIR = /content/drive/MyDrive/CognitivaAI`
  - `DATA_DIR = BASE_DIR/oas1_data`
  - `OUT_DIR` por pipeline (p.ej. `ft_effb3_stable_colab_plus`, `p11_alt_backbones`, etc.)
- **Mapas OASIS**: `oas1_val_colab_mapped.csv`, `oas1_test_colab_mapped.csv` (columnas claves: `png_path`, `target`, `patient_id`, …).
- **Columnas de predicción**:
  - Formatos detectados: `y_score`, `sigmoid(logit)`, `sigmoid(logits)`, `pred`.
  - Se unifica a **`y_score`** internamente durante la carga.
- **Pooling a nivel paciente**: `mean`, `trimmed20`, `top7`, `pmean_2` (power mean con p=2).
- **Métricas**: AUC, PR-AUC, Acc, Recall, Precision. Umbral por:
  - **F1-opt** (maximiza F1 en VAL),
  - **Youden** (maximiza sensibilidad+especificidad-1),
  - **REC90/REC100** (recall fijado).

---

## Fase 7 – OASIS-2 (p13, p14 y p15)

**Contexto:**  
Exploración y explotación del dataset OASIS-2 con EfficientNet-B3.  
Se implementaron tres pipelines consecutivos:

- **p13:** entrenamiento base con criterio de una sola visita por paciente.  
- **p14:** entrenamiento balanceado en Colab GPU, copiando imágenes a SSD para mejorar la E/S.  
- **p15:** consolidación de resultados de OASIS-2 (p13 y p14) junto a OASIS-1 (p11), integrando todos los backbones en un catálogo común y generando features de ensamble.

**Detalles técnicos:**
- 20 slices por volumen, equiespaciados y normalizados (z-score + CLAHE).  
- Labels obtenidos del Excel clínico, convertidos a binario (Control=0, Dementia/Converted=1).  
- Split fijo: 105 train, 22 val, 23 test (1 sesión por paciente).  
- P14: entrenamiento con **class weights** y datos en **SSD local de Colab**.  
- P15: consolidación en catálogo, eliminación de features con NaN≥40%, uso de Logistic Regression (con imputación) y HistGradientBoosting (manejo nativo de NaN).

**Resultados:**
- **p13:** recall alto, dataset limitado (150 pacientes).  
- **p14:** VAL AUC≈0.88, TEST AUC≈0.71 con recall=100%.  
- **p15:** consolidación con ensamble → VAL AUC≈0.94, TEST AUC≈0.71; recall alto sostenido.  
- Integración completa en el catálogo de backbones (`oas2_effb3`, `oas2_effb3_p14`) y en las features consolidadas con OASIS-1.

---

# Fase 8 – OASIS-2 (p15 y p16)

**Contexto general:**  
Tras los avances logrados con p13 y p14, donde exploramos el dataset OASIS-2 y conseguimos un modelo base sólido con EfficientNet-B3, surgió la necesidad de dar un paso más:  
1. **Consolidar la preparación de datos (p15)** para asegurar coherencia y cobertura completa del dataset.  
2. **Refinar la estrategia de ensembles (p16)**, combinando backbones heterogéneos en un esquema patient-level con métricas robustas.

---

## Fase 9 – Ensemble Calibration (p17)

**Contexto:**  
Tras p16, el siguiente paso fue calibrar las probabilidades del ensemble para aumentar la interpretabilidad y la utilidad clínica.

**Detalles técnicos:**  
- Construcción de un meta-ensemble con Logistic Regression sobre outputs base.  
- Aplicación de Platt scaling y optimización de umbral (F1).  
- Evaluación con Brier Score para medir calibración.  

**Resultados:**  
- Validación: AUC≈0.78, Recall=0.94, F1=0.76, Brier=0.176.  
- Test: AUC≈0.70, Recall=0.78, F1=0.66, Brier=0.227.  
- Cohortes: OAS1 consistente; OAS2 limitado.  

**Conclusión:**  
La calibración refina el ensemble, mantiene sensibilidad alta y mejora la calidad de las probabilidades, aunque la robustez en OAS2 aún requiere trabajo.

---

# Fase 9 – Comparativa p16 vs p17

**p16 – Blending clásico:**  
- LR + HGB combinados con un peso óptimo (α=0.02).  
- Validación espectacular (AUC≈0.95, Recall=1.0), pero riesgo de sobreajuste.  
- En test, buen recall (0.78) pero sin calibración de probabilidades.  

**p17 – Ensemble calibrado:**  
- Stacking con Logistic Regression y Platt scaling.  
- AUC más modesto en validación (0.78) y test (0.70).  
- Mantiene recall≈0.78 y además optimiza la calibración (Brier=0.227 en test).  
- Probabilidades más interpretables, mejor preparadas para escenarios clínicos.  

**Conclusión de la fase:**  
- p16 = **mejor raw performance** (máximo AUC).  
- p17 = **mejor calibración y estabilidad clínica** (probabilidades confiables).  
- Ambos complementan la estrategia de ensembles: uno explota rendimiento, otro asegura interpretabilidad.

---

## Fase 9 – Stacking avanzado (p18)

**Contexto:**  
Tras calibrar ensembles en p17, se diseñó un stacking multicapa para explorar la combinación de múltiples clasificadores heterogéneos.  

**Detalles técnicos:**  
- **Base learners:** Logistic Regression (L2), HistGradientBoosting, Gradient Boosting, Random Forest, Extra Trees.  
- **Meta learner:** regresión logística con blending α=0.02.  
- **Estrategia:**  
  - Generación de predicciones OOF con 5-fold cross-validation.  
  - Validación de umbral óptimo en F1.  
  - Evaluación separada para OAS1 y OAS2.  
- **Métricas adicionales:** Brier Score para calibración, coeficientes de meta-LR y permutación de importancias para interpretar contribuciones.

**Resultados:**  
- [VAL] AUC=0.92, Recall≈0.90, F1≈0.83, Precision≈0.78.  
- [TEST] AUC=0.67, Recall≈0.78, F1≈0.67, Precision≈0.59.  
- Cohorte OAS1 aportó la mayor estabilidad, mientras que OAS2 mantuvo recall alto pero sin señal discriminativa clara (AUC≈0.5).

**Conclusiones:**  
El stacking multicapa permitió validar la viabilidad de **meta-modelos complejos** en un dataset MRI limitado.  
Gradient Boosting y Random Forest emergieron como pilares, aunque la brecha entre validación y test evidencia el reto de generalización en cohortes pequeñas.

---

## Fase 9 – Meta-Ablation y calibración avanzada (P22)

**Contexto:**  
Tras consolidar los ensembles y aplicar calibraciones básicas en fases previas (p20–p21), se diseñó P22 como un *ablation study* para comparar métodos de calibración y medir su efecto en la estabilidad de las probabilidades y en la sensibilidad de los modelos.

**Diseño y ejecución:**  
- Features: 56 columnas iniciales; tras filtrar NaN>40% se mantuvieron 36.  
- Cohortes: 69 pacientes en validación, 70 en test.  
- Modelos calibrados:  
  - Logistic Regression (LR) con imputación y escalado.  
  - HistGradientBoosting (HGB), tolerante a NaNs.  
- Métodos de calibración aplicados:  
  - **Platt scaling (sigmoid).**  
  - **Isotonic regression.**  
- Validación con OOF por StratifiedKFold (sin fugas).  
- Selección de umbral F1-máx en validación (≈0.30–0.35).  

**Resultados principales:**  
- LR-Platt: VAL AUC=0.73, F1=0.68 | TEST AUC=0.67, F1=0.69  
- LR-Isotonic: VAL AUC=0.86, F1=0.75 | TEST AUC=0.67, F1=0.65  
- HGB-Platt: VAL AUC=0.82, F1=0.75 | TEST AUC=0.70, F1=0.63  
- HGB-Isotonic: VAL AUC=0.89, F1=0.77 | TEST AUC=0.67, F1=0.64  
- Blend (Isotonic): VAL AUC≈0.90, F1≈0.79 | TEST AUC≈0.68, F1≈0.62  

**Interpretación:**  
- La calibración isotónica aporta mejor ajuste en validación (Brier bajo), pero pierde robustez en test.  
- Platt mantiene recall alto, lo que lo hace más apto para escenarios de cribado clínico.  
- El blend confirma robustez en validación, pero sigue presente el gap entre cohortes OAS1 y OAS2.  

**Conclusión:**  
P22 aportó claridad sobre qué técnicas de calibración son más fiables en entornos clínicos pequeños y heterogéneos. Constituye la base para P23, donde se buscará integrar estas calibraciones dentro de meta-ensembles finales y analizar umbrales de decisión específicos por cohorte.

---

### 📅 Entrada – Estrategia OASIS-1 y OASIS-2 en ensembles (p16–p22)

Durante los pipelines de ensembles avanzados (p16–p22) se trabajó con datos de
**OASIS-1 y OASIS-2** simultáneamente. 

**Decisión clave:**
- No fusionar ambos datasets en uno único.
- Mantener la cohorte identificada (`cohort = OAS1 / OAS2`) en todos los
  DataFrames.
- Entrenar meta-modelos (LR, HGB, XGB, blends, calibraciones) sobre los datos
  combinados, pero **siempre evaluando por cohorte y global**.

**Beneficios:**
- Evita leakage entre cohortes.
- Permite comparar rendimiento en escenarios distintos:
  - OAS1: cross-sectional, mayor homogeneidad.
  - OAS2: longitudinal, más ruido y variabilidad.
- Informa sobre la robustez del ensemble frente a shift de dominio.

**Resultado observado:**
- En validación (VAL), OAS1 logra métricas más altas (AUC, Acc).
- En test (TEST), OAS2 muestra recall elevado pero menor calibración y precisión.
- Globalmente (ALL), se obtiene una media ponderada que refleja mejor la
  dificultad del problema.

**Conclusión:**
El tratamiento separado de OASIS-1 y OASIS-2 dentro de los ensembles es esencial
para interpretar los resultados clínicos y diseñar calibraciones específicas
para cada cohorte en los pipelines posteriores (p20–p22).

---

## Ficha de modelo — P26 / P26b (intermodal)

**Entrada:**  
- Imagen (prob. P24 por paciente) + 56 features de imagen (p11+p14/p13).  
- Clínico consolidado (Age, Sex, Education, SES, MMSE, eTIV, nWBV, ASF, Delay).  
- Señal p1 (OAS2) con imputación por cohorte + flag.

**Arquitectura:**  
- **P26 (Late):** meta-LR sobre `{p_img, p_clin, p1_fill, p1_has}`.  
- **P26b:** P26 + **calibración Platt por cohorte** en VAL y re-umbrales 5:1.

**Métricas (TEST):**  
- P26 — ALL AUC=0.713 · PR-AUC=0.712 · Brier=0.234; OAS1 AUC=0.754 · OAS2 AUC=0.652.  
- P26b — mejora Brier (OAS1 0.199, OAS2 0.241) sin cambiar confusión a coste 5:1.

**Umbrales recomendados:**  
- **P26:** OAS1=0.307 · OAS2=0.195 (coste mínimo).  
- **P26b (único):** OAS1=0.340 · OAS2=0.374.  
- **Mixto (recall OAS2):** OAS1→P26b@0.340 · OAS2→P24@0.332.

**Riesgos:** descalibración en OAS2; tamaño muestral.  
**Mitigaciones:** monitorizar **ECE/MCE**, recalibrar con ventana móvil; reportar intervalos; mantener umbrales por cohorte.

**Artefactos clave:** ver `p26_intermodal/` (predicciones, calibraciones, umbrales, tablas ejecutivas, bloques).

---

# 🗓 Semana “cero”: preparación antes del arranque formal

## 📅 24/06/2025 — Preparación de entorno y árbol de carpetas
- Estructuramos las rutas de trabajo en **Google Drive** para garantizar persistencia.
- Creamos `CognitivaAI/` con subcarpetas para datos, salidas por pipeline y documentos (`README.md`, `InformeTecnico.md`, `CuadernoBitacora.md`).
- Decidimos usar **Google Colab** como entorno primario.

**Decisiones**  
- Convención de nombres de salida (por pipeline) para poder concatenar y comparar.
- Estándar de CSV: separador `,`, encoding UTF-8, cabeceras.

---

## 📅 25/06/2025 — Ingesta y saneamiento de OASIS
- Revisión de **mapas** `oas1_val_colab_mapped.csv` y `oas1_test_colab_mapped.csv`.
- Verificación de columnas mínimas: `png_path`, `target`, `patient_id`.
- Exploración de duplicidades por `patient_id` (coincide con el supuesto de múltiples cortes por paciente).
- Definición de **helpers** de lectura robusta (detección de nombre real de columna score).

**Incidencias**  
- Rutas con barras invertidas en `source_hdr` (propiedad informativa). Sin impacto en lectura principal.

---

## 📅 26/06/2025 — Métricas y umbrales
- Implementamos un bloque de evaluación unificado con AUC, PR-AUC, Acc, P, R, y búsqueda del **umbral óptimo F1** y **Youden**.
- Añadimos perfiles **REC90** y **REC100** (para escenarios de alta sensibilidad).

**Decisión**  
- Registrar siempre `n` (tamaño conjunto paciente) en los resúmenes.

---

## 📅 27/06/2025 — Diseño de pipelines
- Esbozo de los **Pipelines** P1…P11 (clínico → MRI → calibración → ensembles → backbones).
- Cada pipeline escribe sus CSV y un **resumen** en una tabla comparativa.

**Lección**  
- Trazabilidad por pipeline evita mezclar resultados de runs viejos.

---

## 📅 28/06/2025 — Helpers de pooling a paciente
- Definimos pooling: `mean`, `trimmed20`, `top7`, y **`pmean_2`** (promedio potencia con p=2).
- Aseguramos **idempotencia**: si existen tablas, se reusan; si no, se crean.

---

## 📅 29/06/2025 — Validación rápida de lectura + guardado
- Mini-pipeline de lectura de mapas y generación de features básicos a paciente.
- Confirmamos conteos esperados (p. ej., 940 cortes VAL/TEST → 47 pacientes).

---

# 🏁 Arranque formal

## 📅 01/07/2025 — P1: Clínico OASIS-2 (XGB)
- **Modelo**: XGBoost.
- **Resultado**: **AUC ≈ 0.897**.
- **Conclusión**: baseline tabular fuerte.

---

## 📅 03/07/2025 — P2: Clínico fusión (XGB)
- Integración de variables clínicas ampliadas.
- **Resultado**: **AUC ≈ 0.991**, **Recall ~1.0**.
- **Riesgo**: posible **overfitting**.

---

## 📅 10/07/2025 — P3: MRI OASIS-2 (ResNet50)
- **Backbone**: ResNet-50 (ImageNet).
- **Resultado (test)**: **AUC ≈ 0.938**.
- **Conclusión**: MRI viable; base sólida para Colab.

---

## 📅 15/07/2025 — P5: MRI Colab (ResNet18 + Calib)
- **Resultado**: AUC ≈ 0.724 | PR-AUC ≈ 0.606 | Acc ≈ 0.60 | R=0.80 | P=0.52.
- **Conclusión**: salto a Colab con calibración aporta control, pero rendimiento moderado.

---

## 📅 20/07/2025 — P6: EffNet-B3 embeddings
- **Resultado**: AUC ≈ 0.704 | PR-AUC ≈ 0.623 | Acc ≈ 0.70 | R=0.90 | P=0.60.
- **Aprendizaje**: recall alto, aún inestable.

---

## 📅 23/07/2025 — P7: EffNet-B3 finetune
- **Resultado**: **AUC ≈ 0.876** | PR-AUC ≈ 0.762 | Acc ≈ 0.745 | **R=1.0** | P=0.625.
- **Conclusión**: **mejor punto** hasta la fecha.

---

## 📅 30/07/2025 — P9: EffNet-B3 stable
- **Resultado**: AUC ≈ 0.740 | PR-AUC ≈ 0.630 | Acc ≈ 0.72 | R=0.65 | P=0.62.
- **Notas**: gana estabilidad, cede algo de recall.

---

## 📅 05/08/2025 — P10: EffNet-B3 stable + calibración
- **Técnicas**: temperature scaling, isotonic.
- **Caveat**: grandes magnitudes de **logits** → overflow en `exp`.
- **Parche**:
  ```python
  def safe_sigmoid(z):
      z = np.clip(z, -50, 50)
      return 1/(1+np.exp(-z))

  def fit_temperature(logits, y_true, init_T=1.0, bounds=(0.05, 10.0)):
      logits = np.asarray(logits, float); y_true = np.asarray(y_true, float)
      def nll(T):
          p = safe_sigmoid(logits/T); eps = 1e-7
          return -np.mean(y_true*np.log(p+eps) + (1-y_true)*np.log(1-p+eps))
      return float(minimize(lambda t: nll(t[0]), x0=[init_T], bounds=[bounds], method="L-BFGS-B").x[0])
    ```
 - **Resultado(rango):** **AUC test 0.546–0.583**, PR-AUC ~0.50–0.53, Acc ~0.51–0.55, **Recall=1.0**, Precision ~0.47–0.49.
 - **Conclusión:** calibración ↓ AUC pero ↑ interpretabilidad. Necesario ensemble posterior.

 ---

## 📅 10/08/2025 — P10-ext: TRIMMED y seed-ensemble
- **Semillas 41/42/43** con agregaciones por paciente.
- **Logs:** “VAL slices por seed: [940,940,940] … Guardado slice-level seedENS…”
- **Seed-ensemble (media/TRIMMED/TOP7)** (sin calibrar) dio AUC test ≈ 0.50–0.51 en algunos runs (semillas no aportaron mejora directa).
- **Stacking / Random weights (mean+trimmed20+top7+p2):**
  - **RF** y **STACK(no-neg)** sobre 4 features de pooling:
    - **VAL:** AUC ~0.90–0.91, PR-AUC ~0.92, Acc ~0.85–0.87, R ~0.75–0.95.
    - **TEST:** **AUC ~0.75**, PR-AUC ~0.73–0.75, Acc ~0.64–0.70, R ~0.50–0.70, P ~0.58–0.71.
  - **Ej. RAND(500 samples)** (mean/trimmed20/top7/p2):
    - Pesos ejemplo: mean 0.325, trimmed20 0.315, top7 0.322, p2 0.038.
    - **VAL:** AUC=0.909, PR-AUC=0.920, Acc=0.872, R=0.95, P=0.792.
    - **TEST:** **AUC=0.754**, PR-AUC=0.748, Acc=0.660, R=0.70, P=0.583.
 - **STACK_LR(mean+trimmed20+top7+p2):**
    - * Coefs ≈ [0.407, 0.409, 0.485, 0.416], **intercept −0.923**.
    - **VAL**: AUC=0.909, PR-AUC=0.920, Acc=0.872, R=0.95, P=0.792.
    - **TEST**: AUC=0.754, PR-AUC=0.748, Acc=0.660, R=0.70, P=0.583.
- **Conclusión:**
    - **Consolidado**: a nivel paciente, **ensembles de pooling** (4 features) mejoran notablemente sobre seed-ensemble puro.

---

### 📅 12/08/2025 — Documentación y limpieza

 * Añadidos al `README` e Informe: decisión de que “estrategia de semillas” no aportó sola.
 * Normalización de nombres de columnas en todos los CSV (de cara a p11).

 ---

 ## 📅 15/08/2025 — P11: Backbones alternativos (inicio)

* Notebook: `cognitiva_ai_backbones.ipynb`.
* **Incidencia 1 (Drive)**: `“Mountpoint must not already contain files”` → solución: no remount si ya montado / reiniciar entorno tras semanas.
* **Incidencia 2 (rutas)**: `DATA_DIR` marcaba `exists=False` pese a existir → solución: reinicio completo; verificación con `Path.exists()`.
* Carga correcta:
    ```
    Mounted at /content/drive
    🔎 VAL_MAP …/oas1_val_colab_mapped.csv
    🔎 TEST_MAP …/oas1_test_colab_mapped.csv
    ✅ Columnas OK: ['patient_id','png_path','target']
    💾 Config guardada: …/p11_alt_backbones/p11_config.json
    ```

---

### 📅 16/08/2025 — ConvNeXt-Tiny (in12k\_ft\_in1k)

* Inferencia: guardó `convnext_tiny.in12k_ft_in1k_val_slices.csv` y `_test_slices.csv`.
* Resumen por pooling:
    * **ConvNeXtTiny-mean**: VAL `AUC` 0.5556 | `PR-AUC` 0.5436 | TEST `AUC` 0.5093 | `PR-AUC` 0.4790 | `Acc` 0.489 | `R`=1.0 | `P`=0.455.
    * **trimmed20**: TEST `AUC` 0.5000 | `PR-AUC` 0.4723.
    * **top7**: TEST `AUC` 0.5111 | `PR-AUC` 0.4643.
* Fila README: `| P11 | MRI Colab | ConvNeXt-Tiny (in12k_ft_in1k) + mean | 0.509 | 0.479 | 0.49 | 1.00 | 0.45 |`

---

### 📅 17/08/2025 — DenseNet-121

* Peso ImageNet (no `d121_best.pth`).
* Slice-level → patient-level:
    * **Dense121-mean**: TEST `AUC` 0.3241 | `PR-AUC` 0.3942 | `Acc` 0.340 | `R`=0.75 | `P`=0.366.
    * **trimmed20**: TEST `AUC` 0.3426 | `PR-AUC` 0.4068.
    * **top7**: TEST `AUC` 0.3019 (más bajo).
* **Resumen**: DenseNet-121 decepciona en este dataset.

---
### 📅 18/08/2025 — Swin-Tiny

* Slice-level → patient-level:
    * **SwinTiny-mean**: TEST `AUC` 0.5352, `PR-AUC` 0.5109, `Acc` 0.447, `R`=1.0, `P`=1.0 (umbral muy bajo).
    * **SwinTiny-top7**: TEST `AUC` 0.6407, `PR-AUC` 0.5971, `Acc` 0.553, `R`=0.95, `P`=0.95 (mejor variante Swin).
* **Conclusión**: Swin-Tiny (`top7`) es el mejor de los alternativos probados.

---

### 📅 19/08/2025 — Catálogo multi-backbone + normalización columnas

* Escaneo de `p11_alt_backbones` y carpetas previas:
    * Detectados `SwinTiny`, `ConvNeXt slices`, `DenseNet-121`, y además `efb3` de pipelines anteriores (`ft_effb3_*`).
* Unificación de columnas: mapeo auto (`y_score`, `sigmoid(logit[s])`, `pred` → `y_score`).
* Construcción features por paciente (VAL/TEST (47, 6) por fuente), guardados:
    * `val_patient_features_backbones.csv`
    * `test_patient_features_backbones.csv`
* Validación:
    * `SwinTiny` OK (940 filas → 47 pacientes).
    * `ConvNeXt slices` OK (940 → 47).
    * `DenseNet` OK (940 → 47).
    * Preds a nivel paciente de pipelines previos (47 directos) incluidas como features extra.

---

### 📅 20/08/2025 — Ensemble de backbones (promedios y stacking base)

* **AVG** de 12 señales `“*_mean”` (Swin/ConvNeXt/DenseNet + señales paciente/effect):
    * **VAL (F1-opt)**: `AUC` 0.476 | `PR-AUC` 0.389 | `Acc` 0.40 | `R`=1.0 | `P`=0.333 | `thr`=0.3525 | `n`=10.
    * **TEST (F1-opt)**: `AUC` 0.713, `PR-AUC` 0.724 | `Acc` 0.426 | `R`=1.0 | `P`=0.426 | `thr`=0.3525 | `n`=47.
* **Observación**: `AUC` test alto vs val bajo → val (`n`=10) muy pequeño; umbral podría transferirse demasiado “optimista”.
* **STACK\_LR(all\_features)**:
    * **VAL**: `AUC` 0.810 | `PR-AUC` 0.700 | `Acc` 0.800 | `R`=1.0 | `P`=0.600.
    * **TEST**: `AUC` 0.298 | `PR-AUC` 0.397 | `Acc` 0.383 | `P` 0.304 | `R` 0.35.
* **Overfitting claro a VAL**.

---

### 📅 21/08/2025 — Dirichlet (3 backbones, means)

* **FEATURES**: `SwinTiny_mean`, `convnext_tiny..._mean`, `png_preds_d121_mean`.
* `N_SAMPLES`=800 (semilla 42).
* Mejor combinación (ejemplo):
    * Pesos ≈ Swin 0.972, ConvNeXt 0.004, Dense 0.024.
    * **VAL (F1-opt)**: `Acc` 0.70 | `P` 0.50 | `R` 1.0 | `thr` 0.474 | `AUC` 0.714, `PR-AUC` 0.633 (`n`=10).
    * **TEST (F1-opt)**: `Acc` 0.468 | `P` 0.444 | `R` 1.0 | `thr` 0.435 | `AUC` 0.520, `PR-AUC` 0.523 (`n`=47).
* **Youden TEST**: `Acc` 0.617 | `P` 0.667 | `R` 0.20 (umbral 0.481).
* **Conclusión**: mejora leve vs ConvNeXt-mean/DenseNet, pero por debajo de Swin-top7 y muy lejos de los ensembles de EffNet-B3 del P10-ext.

---

### 📅 22/08/2025 — Dirichlet EXT (12 features)

* **FEATURES**: `{Swin[mean/trimmed/top7], ConvNeXt_slices[mean/trimmed/top7], DenseNet[mean/trimmed/top7]}` + señales agregadas (`patient_preds_plus_mean`, `slice_preds_plus_mean`, `slice_preds_seedENS_mean`).
* **Resultado**:
    * **VAL**: `AUC` 0.714, `PR-AUC` 0.681.
    * **TEST**: `AUC` 0.361, `PR-AUC` 0.405.
* **Conclusión**: sobreajuste; demasiados grados de libertad para `n(VAL)` = 10.

---

### 📅 23/08/2025 — Stacking L1 fuerte (sparsidad forzada)

* **FEATURES candidatas (ej.)**: `SwinTiny_top7`, `convnext..._top7`, `png_preds_d121_trimmed20`, `patient_preds_plus_mean`, `slice_preds_plus_mean`, `slice_preds_seedENS_mean`.
* **Resultado**: todos `coef=0` (modelo trivial), `intercept=0`.
* **VAL/TEST**: `AUC=0.5`; F1 ligado a prior por umbral 0.
* **Interpretación**: el penalizador “fuerte” anuló todas las señales (`n(VAL)` pequeño + correlación alta).

---

### 📅 24/08/2025 — Isotonic sobre Swin-Tiny (top7)

* **Resultado**:
    * **VAL**: `AUC` 0.714 | `PR-AUC` 0.556 | `Acc` 0.400 | `R` 1.0 | `P` 0.333 | `thr` 0.0025.
    * **TEST**: `AUC` 0.566 | `PR-AUC` 0.458 | `Acc` 0.553 | `R` 0.95 | `P` 0.487 | `thr` 0.0025.
* **Conclusión**: la calibración isotónica ayuda ligeramente en test y fija un recall alto con precisión moderada.

---

### 📅 25/08/2025 — Catálogo ampliado y parsers robustos

* Se indexan también directorios previos:
    * `oas1_resnet18_linearprobe/…`
    * `ft_effb3_colab/…`, `ft_effb3_stable_colab_plus/…`, etc.
* Validación automática de columnas y tamaños; cualquier CSV no conforme se re-mapea.

---

### 📅 27/08/2025 — Revisión de README/Informe/Cuaderno

* Se vuelcan resultados preliminares al `README`, con filas por pipeline (P1–P11), incluyendo ConvNeXt-Tiny, Swin-Tiny y DenseNet-121.
* Se documenta que la estrategia de semillas en solitario no aportó (`AUC` ≈ 0.5), mientras que ensembles de pooling (4 features) sí mejoraron hasta `AUC` test ≈ 0.75.
* Se prepara archivo de Contexto para otros chats (evitar pérdida de hilo).

---

### 📅 29/08/2025 — Ajustes finales P11 y ensembles

* Normalizado definitivo de nombres en `comparison_backbones_eval.csv`.
* Confirmación de Swin-Tiny (`top7`) como mejor alternativo aislado.
* Resumen de ensembles P11:
    * **Dirichlet (3 means)**: TEST `AUC` ≈ 0.52.
    * **Dirichlet EXT (12)**: TEST `AUC` ≈ 0.36.
    * **STACK\_LR(all)**: TEST `AUC` ≈ 0.30 (overfit).
    * **Swin-Tiny isotonic**: TEST `AUC` ≈ 0.566; `Acc` ≈ 0.553; `R` 0.95; `P` 0.487.

---

### 📅 04/09/2025 – Pipeline p13
- Procesamiento OASIS-2 → 20 slices equiespaciados por scan.  
- Selección de 1 visita/paciente.  
- Entrenamiento base en EfficientNet-B3 (105/22/23).  

### 📅 05/09/2025 – Pipeline p14
- Copia de 7340 slices a SSD local en Colab.  
- Entrenamiento balanceado con class weights.  
- AUC≈0.88 en val, recall=100% en test.  

### 📅 06/09/2025 – Pipeline p15 (Consolidación)
- Integración de resultados p13 y p14 en el catálogo global de backbones.  
- Generación de features combinadas con OASIS-1 (p11).  
- Dificultades: manejo de NaN en features y necesidad de descartar/ imputar columnas.  
- Modelos finales: Logistic Regression con imputación y HistGradientBoosting (NaN nativo).  
- Resultado: VAL AUC≈0.94, TEST AUC≈0.71 con recall alto.

---

### 📅 06/09/2025 – Pipeline p15 (Consolidación de dataset OASIS-2)
- Se revisaron de nuevo todos los **367 scans** procesados de OASIS-2.  
- Confirmamos que solo **150 scans** contaban con etiquetas clínicas válidas (Control/Dementia/Converted).  
- Se reafirmó el criterio de **una única sesión por paciente** para evitar *data leakage* entre splits.  
- Se generaron **20 slices axiales equiespaciados** por volumen, eliminando los extremos (8%) y aplicando **normalización z-score + CLAHE opcional**.  
- Resultado: **150 pacientes × 20 slices = 3.000 imágenes etiquetadas**.  
- Dificultad importante: el acceso a imágenes desde Google Drive seguía penalizando el entrenamiento por la latencia de E/S.  
  - **Solución:** replicar todo el dataset en el **SSD local de Colab** antes de cada entrenamiento, lo que redujo drásticamente los tiempos.  
- Con esta consolidación, el dataset quedó consistente, balanceado y preparado para ser integrado en ensembles.

---

### 📅 06/09/2025 – Pipeline p16 (Refinamiento de ensembles)
- Se construyeron features **patient-level** a partir de múltiples backbones:  
  - `oas2_effb3`, `oas2_effb3_p14`, `SwinTiny`, `ConvNeXt_tiny`, `DenseNet121`, entre otros.  
- Durante la integración, se detectó un alto número de columnas con valores faltantes (**NaNs**).  
  - Se aplicó un criterio estricto: **descartar columnas con >40% NaN**.  
  - Para las restantes:  
    - **Logistic Regression (LR):** imputación + columnas-flag de missingness.  
    - **HistGradientBoosting (HGB):** manejo nativo de NaNs, sin necesidad de imputar.  
- Se exploró un esquema de **blending** LR+HGB, optimizado en validación con α=0.02 (casi todo el peso en HGB).  
- **Resultados clave:**  
  - **Validación:**  
    - AUC≈0.95 global, con recall=100% en cohortes OAS1.  
    - En OAS2, las métricas fueron más bajas (AUC≈0.54) debido al reducido tamaño de muestra, pero se mantuvo recall=100%.  
  - **Test:**  
    - AUC≈0.69 global.  
    - Recall≈78%, lo que representa una mejora respecto a modelos individuales.  
    - El blending aportó mayor estabilidad en comparación con usar un solo clasificador.  
- Conclusión: los ensembles **aumentan la sensibilidad del sistema y reducen el riesgo de overfitting**, consolidándose como la mejor estrategia para explotar múltiples backbones en paralelo.

---
### 📅 07/09/2025 – Pipeline p17

- **Objetivo:** Refinar los ensembles con calibración de probabilidades.  
- **Técnicas aplicadas:**  
  - Stacking de outputs base (LR + HGB).  
  - Logistic Regression como meta-modelo.  
  - Platt scaling para calibración probabilística.  
  - Optimización del umbral con F1 en validación.  
- **Resultados globales:**  
  - [VAL] AUC≈0.78 | Recall=0.94 | F1=0.76 | Brier=0.176.  
  - [TEST] AUC≈0.70 | Recall=0.78 | F1=0.66 | Brier=0.227.  
- **Análisis por cohortes:**  
  - OAS1 se mantiene estable (val/test ≈0.84/0.77).  
  - OAS2 continúa siendo inestable, con AUC ≈0.5 en test.  
- **Conclusión:**  
  - El ensemble calibrado aporta **confianza probabilística mejorada**.  
  - Se prioriza recall alto, sacrificando algo de precisión.  
  - El reto sigue siendo el tamaño reducido de OAS2.  

 --- 

### 📅 07/09/2025 – Pipeline p18
- Implementado **stacking multicapa** con cinco clasificadores base (LR, HGB, GB, RF, ET) y un meta-modelo logístico.  
- Generación de predicciones **OOF con 5 folds** para evitar fuga de información.  
- Ajuste de blending α=0.02.  
- Evaluación detallada por cohortes (OAS1, OAS2) y global.  
- **Resultados:**  
  - VAL AUC≈0.92 | Recall≈0.90 | F1≈0.83.  
  - TEST AUC≈0.67 | Recall≈0.78 | F1≈0.67.  
- Insight: GB y RF fueron los más influyentes como modelos base, pero la generalización en OAS2 sigue limitada (AUC≈0.5).  

---

### 📅 07/09/2025 – Pipeline p19

**Fase 8: Ensembles y calibración (P18–P19)**  

- **Qué hice:** ejecuté P19 con stack de base learners (LR, HGB, GB, RF, LGBM, XGB) y meta-XGB. Construí OOF sin fuga con KFold, armé meta-features y evalué en VAL/TEST.  
- **Datos y features:** 56 columnas válidas tras filtrar NaN>40%; representación por paciente (mean/trimmed/top-k/p2).  
- **Resultados:**  
  - VAL: AUC=0.964; PRAUC=0.966; Acc=0.913; F1=0.897; Brier=0.071.  
  - TEST: AUC=0.729; PRAUC=0.688; Acc=0.714; F1=0.630; Brier=0.226.  
- **Aprendizajes:** meta fuerte en VAL pero recall bajo en TEST; hay shift (OAS1 vs OAS2) y el umbral global no es óptimo. LightGBM sin splits útiles sugiere simplificar meta y seleccionar features.  
- **Siguiente paso (p20):** calibrar meta, umbrales por cohorte, meta más simple y Repeated KFold para robustez.

---

### 📅 07/09/2025 – Fase 9: Meta-calibración (P20)

**Qué hice:**  
Ejecuté P20 sobre el meta-ensemble de p19, aplicando calibración de probabilidades con Platt e isotónica, tanto global como por cohorte (OAS1/OAS2).  

**Datos y setup:**  
36 columnas finales tras descartar NaN>40%. Modelos calibrados: HGB y LR. Guardé predicciones calibradas en VAL/TEST y JSON de resumen.  

**Resultados clave:**  
- HGB-Isotonic-PerC: VAL AUC=0.840 | F1=0.753 | Brier=0.156  
- LR-Platt-Global: TEST AUC=0.686 | F1=0.658 | Brier=0.221  
- En TEST, recall≈0.78 con precisión moderada (≈0.54–0.57).  

**Aprendizajes:**  
La calibración reduce el error de probabilidad (Brier), sobre todo en validación.  
El umbral global no captura bien las diferencias entre cohortes; per-cohort mejora ligeramente.  
El modelo calibrado mantiene recall alto → útil en escenario clínico de cribado.  

**Siguiente paso:**  
Integrar calibraciones en el ensemble completo, probar Elastic-Net como meta y explorar selección de umbrales orientada a coste clínico.

---

### 📅 07/09/2025 – Fase 8 · P21 (Meta-refine)

**Qué hice.** Ejecuté p21 con un stacking compacto (LR, HGB, LGBM, XGB) y meta a partir de 4 OOFs; filtré NaN>40% (36 columnas finales) y apliqué umbral F1-máx=0.45.

**Datos.** VAL=69, TEST=70; features por paciente procedentes de múltiples backbones (mean/trimmed/top-k/p2), con columna de cohorte (OAS1/OAS2).

**Resultados.**
- VAL: AUC 0.955, PRAUC 0.931, Acc 0.870, F1 0.862, Brier 0.082.
- TEST: AUC 0.653, PRAUC 0.587, Acc 0.643, F1 0.627, Brier 0.285.

**Observaciones.**
- LGBM sin splits con ganancia positiva → complejidad excesiva frente a muestra disponible.
- Buen VAL pero caída en TEST (shift OAS1/OAS2 + umbral global).

**Siguiente.**
- p22: calibración/umbrales por cohorte y por coste; meta más regularizado; Repeated KFold para robustez.

---

### 📅 07/09/2025 – Pipeline P22 (Meta-Ablation con calibración avanzada)

- **Acción:** ejecuté P22 aplicando calibración Platt e Isotónica a los modelos LR y HGB.  
- **Datos:** 69 pacientes en validación y 70 en test, con 36 features seleccionadas (descartadas 20 por NaN>40%).  
- **Resultados clave:**  
  - LR-Platt: VAL AUC=0.73, F1=0.68 | TEST AUC=0.67, F1=0.69  
  - LR-Isotonic: VAL AUC=0.86, F1=0.75 | TEST AUC=0.67, F1=0.65  
  - HGB-Platt: VAL AUC=0.82, F1=0.75 | TEST AUC=0.70, F1=0.63  
  - HGB-Isotonic: VAL AUC=0.89, F1=0.77 | TEST AUC=0.67, F1=0.64  
  - Blend isotónico: VAL AUC≈0.90, F1≈0.79 | TEST AUC≈0.68, F1≈0.62  
- **Aprendizaje:** la calibración isotónica mejora la fiabilidad de las probabilidades en validación, pero en test muestra menor robustez (shift OAS1/OAS2). Platt mantiene recall más alto.  
- **Conclusión:** P22 funcionó como **estudio de ablación** previo a la integración final de calibraciones en meta-ensembles (p23).

---

### 📅 07/09/2025 – Pipeline P23 (Meta-calibración coste-cohorte)

- **Acción:** ejecuté P23 aplicando calibración Platt e Isotónica con umbrales coste-óptimos por cohorte (OAS1/OAS2).  
- **Criterio:** coste clínico FN=5, FP=1 → penaliza falsos negativos.  
- **Artefactos guardados:**  
  - `p23_val_preds_calibrated.csv`  
  - `p23_test_preds_calibrated.csv`  
  - `p23_thresholds.json`  
  - `p23_calibrators.pkl`  
  - `p23_summary.json`  

**Resultados:**  
- **OAS1 (TEST):**  
  - Isotonic → AUC=0.743 | PR-AUC=0.657 | Recall=0.95 | Precision=0.50 | Cost=24.0.  
  - Platt → AUC=0.724 | PR-AUC=0.649 | Recall=0.95 | Precision=0.50 | Cost=24.0.  
- **OAS2 (TEST):**  
  - Ambos calibradores → AUC=0.50 | PR-AUC≈0.52 | Recall=1.0 | Precision≈0.52 | Cost=11.0.  

**Conclusión:**  
- En OAS1, la calibración isotónica logra mejor AUC, pero Platt es competitivo.  
- En OAS2, el modelo no discrimina (AUC=0.5) pero alcanza recall=1.0, lo que elimina FN (clave clínicamente).  
- Se confirma la necesidad de **umbrales diferenciados por cohorte**.  
- P23 sienta la base para un meta-final más simple y robusto (Elastic-Net + Repeated KFold).

---

### 2025-09-07 — P24 ejecutado (LR elastic-net + KFold repetido + Platt)

- Features paciente fusionadas (p11+p14).  
- CV(5×5): AUC=0.880±0.090; mejores params: {'clf__C': 0.1, 'clf__l1_ratio': 0.7}.  
- TEST Global: AUC=0.727, PR-AUC=0.717, Brier=0.220.  
- TEST OAS1: AUC=0.754, PR-AUC=0.736, Brier=0.211.  
- TEST OAS2: AUC=0.750, PR-AUC=0.805, Brier=0.238.  
- Umbrales coste per-cohorte: OAS1 thr=0.435 → Coste=39.0 (R=0.70, P=0.61, Acc=0.68) | OAS2 thr=0.332 → Coste=12.0 (R=0.92, P=0.61, Acc=0.65)

_Artefactos_: `p24_meta_simple/` (preds, coeficientes, modelo, calibrador, summary, thresholds, report).

---

### 2025-09-07 — P25 (construcción del informe final)

- Consolidé P19/P22/P23/P24 en `p25_master_table.csv`.
- Generé bloques finales para README/Informe/Bitácora.
- Figuras: ROC/PR/Calibración, curvas de coste, sensibilidad de coste, ICs bootstrap; coeficientes top.
- Predicciones demo: `p25_predictions_labeled.csv` / `p25_predictions_unlabeled.csv`.
- Release reproducible: `p25_release/` (MANIFEST.json, ENVIRONMENT.json, MODEL_CARD.md).

**Modelo final sugerido:** P24 (LR elastic-net + Platt) con umbrales por cohorte (FN:FP=5:1).  
**TEST @ umbral:** OAS1→ R=0.70, P=0.61 (Coste=39) · OAS2→ R=0.917, P=0.611 (Coste=12).

---

### 2025-09-07 — P26 intermodal (imagen + clínico)

- Consolidado clínico OASIS-1/2 (anti-fuga), OHE y medianas; 56 features de imagen (p11+p14/p13) alineadas.  
- Señal **p1** (OAS2) con cobertura ≈32% → imputación por cohorte (media VAL OAS2) + flag `p1_has`.  
- **Late vs Mid**:  
  - Late (p_img, p_clin, p1_fill, p1_has) — **VAL AUC=0.916**, TEST **AUC=0.713**.  
  - Mid (IMG56+clínico+p1) — VAL AUC=0.797, TEST 0.697.  
  - Selección: **Late**.  
- **Coste 5:1 (umbral de VAL aplicado en TEST):**  
  - OAS1 @ 0.307 → R=0.700, P=0.609, Acc=0.681, Coste=39.  
  - OAS2 @ 0.195 → R=0.667, P=0.667, Acc=0.652, Coste=24.  
- **Calibración (TEST, 10 bins):** ALL ECE=0.178; OAS1 0.150; **OAS2 0.313**.

### 2025-09-07 — P26b (Platt por cohorte)

- Calibración Platt por cohorte entrenada en VAL, aplicada en TEST; re-umbrales 5:1 por cohorte.  
- **OAS1:** Brier 0.208 → **0.199** (AUC≈0.754); **thr_VAL=0.340**; confusión/coste idénticos a P26.  
- **OAS2:** Brier 0.288 → **0.241** (AUC≈0.652); **thr_VAL=0.374**; confusión/coste idénticos a P26.  
- Decisión de producto:  
  - **Único:** P26b (OAS1=0.340, OAS2=0.374).  
  - **Mixto (cribado):** OAS1→P26b@0.340 · OAS2→P24@0.332 (↑ recall).

_Artefactos:_ `p26_intermodal/` (preds, ece/mce, umbrales, report, summary, calibrados, bloques).

---

### 2025-09-08 — P27 (release + política S2)

**Hecho**
- Generé `p26_release.zip` con modelos, config, QA y documentación.  
- Actualicé **MODEL_CARD.md** y **HOW_TO_DEPLOY.md** con la **política S2** activa.  
- Regeneré `MANIFEST.json` y `ENVIRONMENT.txt` (trazabilidad completa).

**Política S2 (marcada)**
- Umbrales activos: `OAS1=0.42`, `OAS2=0.4928655287824083`.  
- Criterio: 5:1 (FN:FP) + ajuste OAS2 para **Recall ≥ 0.90**.  
- Motivo: minimizar FN en dominio OAS2 (más variable/descalibrado), manteniendo el balance 5:1 en OAS1.

**Smoke (TEST @S2)**
- OAS1 → TP=14, FP=9, TN=18, FN=6 ⇒ R=0.70, P=0.61, Acc=0.681, Coste=39.  
- OAS2 → TP=11, FP=6, TN=5, FN=1 ⇒ R=0.917, P=0.647, Acc=0.696, Coste=11.  
- Archivo: `p26_release/QA/p26b_test_report_recall_target.csv`.

**Archivos clave**
- `p26_release.zip` (23 ficheros, con MANIFEST).  
- Scripts: `compute_pimg_from_features.py`, `predict_end_to_end.py`.  
- Config activa: `CONFIG/deployment_config.json` (backup automático).

**Notas**
- ECE P26: ALL≈0.178, OAS1≈0.150, OAS2≈0.313 → seguir monitorizando.  
- Mantener evaluación por cohorte al desplegar; recalibrar si deriva.

**Siguiente**
- (Opcional) Endpoint batch/CLI y plantilla REST.  
- Checklist de producción: logs de FN y ECE, re-calibración por ventana móvil.

---

## 🧭 Chuleta rápida — Política S2 y umbrales

**Política activa (S2)**  
- **OAS1 → 5:1 (FN:FP)** con umbral aprendido en VAL → **thr = 0.42**  
- **OAS2 → “recall objetivo” en VAL (target = 0.85)** → **thr ≈ 0.492866**

**Archivo de configuración:**  
`p26_release/CONFIG/deployment_config.json`

**Claves relevantes dentro del JSON:**
- `policy: "single"`
- `cost_policy: "FN:FP=5:1 (OAS1) + recall_target (OAS2)"`
- `thresholds: { "OAS1": 0.42, "OAS2": 0.4928655287824083 }`
- `thresholds_5to1: { "OAS1": 0.42, "OAS2": 0.49 }`  ← *fallback 5:1 puro*
- `thresholds_recall_target: { "OAS2": { "target": 0.85, "thr_val": 0.4928655…, "found": true } }`

**Cómo cambiar temporalmente de política:**
- **A 5:1 puro:** editar `cost_policy` y copiar `thresholds_5to1` a `thresholds`.
- **Volver a S2:** restablecer `cost_policy` anterior y los `thresholds` de S2.

> Tras editar el JSON, se recomienda un **smoke test** y (opcional) regenerar el ZIP del release.

---

### P27 — Intermodal (Late) + Política S2 (TEST)

| Pipeline | Cohorte | Método |   AUC | PR-AUC | Brier |   Acc |  Prec |   Rec |    Thr | Coste |
|:--------:|:------:|:------:|------:|------:|------:|------:|------:|------:|------:|-----:|
| **P27** | **ALL** | LATE | **0.736** | **0.729** | **0.229** | — | — | — | — | — |
| **P27** | **OAS1** | **S2 (5:1)** | — | — | — | **0.681** | **0.609** | **0.700** | **0.420** | **39** |
| **P27** | **OAS2** | **S2 (recall≥0.85)** | — | — | — | **0.696** | **0.647** | **0.917** | **0.492866** | **11** |

**Notas:**
- Fila **ALL/LATE**: métricas de probabilidad (AUC/PR-AUC/Brier) del modelo intermodal (Late).  
- Filas **OAS1/OAS2 (S2)**: decisión clínica tras calibración por cohorte + política S2 (umbrales por cohorte).

---

## 📊 P27 — Tablas globales finales

### 1) Probabilidades (TEST) — Comparativa por pipeline y cohorte
> Fuente: `p25_informe_final/p25_master_table.csv` (incluye P19, P22, P23, P24, P26).

| Pipeline | Cohorte | Método        |   AUC | PR-AUC | Brier |
|:--------:|:------:|:--------------|------:|------:|------:|
| P19      | ALL    | XGB           | 0.671 | 0.606 | 0.292 |
| P19      | OAS1   | XGB           | 0.663 | 0.588 | 0.310 |
| P19      | OAS2   | XGB           | 0.663 | 0.683 | 0.257 |
| P22      | ALL    | **HGB_platt** | **0.702** | 0.629 | 0.222 |
| P22      | OAS1   | HGB_platt     | 0.724 | 0.649 | 0.209 |
| P22      | OAS2   | LR_platt      | 0.504 | 0.524 | 0.252 |
| P23      | OAS1   | isotonic      | 0.743 | 0.657 | 0.223 |
| P23      | OAS2   | platt         | 0.500 | 0.522 | 0.250 |
| P24      | ALL    | Platt         | 0.727 | 0.717 | 0.220 |
| P24      | OAS1   | Platt         | 0.754 | 0.736 | 0.211 |
| P24      | OAS2   | Platt         | 0.750 | 0.805 | 0.238 |
| P26      | ALL    | LATE          | 0.713 | 0.712 | 0.234 |
| P26      | OAS1   | LATE          | 0.754 | 0.736 | 0.208 |
| P26      | OAS2   | LATE          | 0.652 | 0.728 | 0.288 |

> Nota: P26=LATE intermodal (p\_img + p\_clin). P22 muestra varias calibraciones; arriba se listan las más representativas.

### 2) Decisión clínica (TEST) — Política activa **S2**
> Fuentes: `p26_release/QA/p26b_test_report_recall_target.csv` (S2) + `CONFIG/deployment_config.json`.

| Pipeline | Cohorte | Política        |  Acc  |  Prec |  Rec  |    Thr   | Coste |
|:--------:|:------:|:----------------|------:|------:|------:|---------:|-----:|
| P27      | OAS1   | **S2 (5:1)**    | 0.681 | 0.609 | 0.700 | 0.420000 |  39  |
| P27      | OAS2   | **S2 (R≥0.85)** | 0.696 | 0.647 | 0.917 | 0.492866 |  11  |

**Chuleta de umbrales S2 (dónde cambiar):** `p26_release/CONFIG/deployment_config.json`  
`thresholds = {"OAS1": 0.42, "OAS2": 0.4928655…}` · `thresholds_5to1` como fallback

---

### 2025-09-08 — P27 (tablas globales y gráficos finales)

- Consolidé tabla **global** de probabilidades (TEST) por *pipeline × cohorte*.  
- Añadí tabla de **decisión clínica @S2** (TEST) con TP/FP/TN/FN, métricas y umbrales por cohorte.  
- Generé **figuras** de AUC/PR-AUC/Brier por cohorte y dejé referencia a ECE/MCE (P26 intermodal).  
- Actualicé documentación con **política S2** vigente (umbrales en `deployment_config.json`).

_Artefactos:_ `p25_informe_final/p25_master_table.csv`, `p26_release/QA/p26b_test_report_recall_target.csv`, `p26_intermodal/p26_test_calibration_ece.csv`, `p27_final/*.png`.

---

### 2025-09-08 — P27 (figuras y tablas finales)

- Generadas figuras de barras **AUC / PR-AUC / Brier** por cohorte desde `p25_master_table.csv`.
- Exportada tabla de **decisión S2** (`p27_final/p27_decision_S2_table.csv`) a partir del QA del release.
- (Si disponible) Creada figura comparativa **S2 vs 5:1** en OAS2.
- Ruta de salida: `p27_final/`.

_Artefactos:_ `p27_final/*.png`, `p27_final/p27_decision_S2_table.csv`.

---

...
### 🧪 Extractos de logs útiles

* Logits extremos y z-score (cuando aplicó):
    ```
    VAL (pre) logits: min=-7.78e5 | max=5.45e5 | mean≈-1.52e4 | std≈9.0e4
    VAL (post-z) logits: min≈-8.49 | max≈6.23 | std≈1.00
    TEST (pre) logits: min=-6.43e5 | max=4.92e5 | mean≈-1.28e4 | std≈8.87e4
    TEST (post-z) logits: min≈-7.10 | max≈5.69 | std≈1.00
    ```
* `safe_sigmoid` aplicado siempre antes de calibración/ensembles que consumen logits.

---

### ⚠️ Incidencias recurrentes y soluciones

* **Drive ya montado**:
    * Error: `“Mountpoint must not already contain files”`.
    * Solución: si `drive.mount()` falla, NO forzar; reiniciar entorno o usar `force_remount=True` sólo cuando sea estrictamente necesario.
* **`DATA_DIR`/`VAL_MAP`/`TEST_MAP` “no existen” aun existiendo**:
    * Causa: estado inconsistente de sesión (muchas horas/días sin reiniciar).
    * Solución: reinicio completo; volver a montar; re-evaluar `Path.exists()`.
* **Columnas heterogéneas** (`y_score`, `sigmoid(logit)`, `pred`):
    * Solución: diccionario de normalización y validación de esquemas, forzando `y_score`.
* **Overflow en `exp` (sigmoid)**:
    * Solución: `safe_sigmoid` con `clip[-50, 50]`.
* **Sobreajuste de ensembles complejos** (Dirichlet EXT, STACK\_LR all-features):
    * Causa: `n(VAL)`=10, muchas features correlacionadas.
    * Mitigación: reducir features, validación cruzada a paciente, o usar regularización/priors más informativos.

---

# 📊 Resumen numérico (hitos clave, test)
| Bloque | Método / Configuración | AUC | PR-AUC | Acc | Recall | Precision |
|--------|------------------------|-----|--------|-----|--------|-----------|
| P7     | EffNet-B3 finetune     | .876| .762   | .745| 1.00   | .625      |
| P9     | EffNet-B3 stable       | .740| .630   | .72 | .65    | .62       |
| P10    | EffB3 stable + calib   | .546–.583 | .50–.53 | .51–.55 | 1.00 | .47–.49 |
| P10-ext| Ensemble pooling       | .754| .748   | .66–.70 | .50–.70 | .58–.71 |
| P11    | ConvNeXt-Tiny (mean)   | .509| .479   | .489| 1.00   | .455      |
| P11    | DenseNet-121 (trimmed) | .343| .407   | .319| .75    | .36       |
| P11    | Swin-Tiny (top7)       | .641| .597   | .553| .95    | .95       |
| P11-ens| Dirichlet (3 means)    | .520| .523   | .468| 1.00   | .444      |
| P11-ens| Dirichlet EXT (12)     | .361| .405   | .447| .85    | .425      |
| P11-ens| Swin-Tiny + isotonic   | .566| .458   | .553| .95    | .487      |

**Lectura**: los mejores ensembles paciente-level siguen siendo los construidos sobre EffNet-B3 (P10-ext).
Entre backbones alternativos, Swin-Tiny (`top7`) es el mejor individual; con isotonic gana algo de robustez.

---

### 🧭 Estado actual

* Pipelines del 1 al 11 implementados y documentados.
* Backbones alternativos evaluados (Swin, ConvNeXt, Dense).
* Ensembles probados (AVG, Dirichlet, Stacking, Isotonic) con resultados concluyentes sobre limitaciones por tamaño de VAL y correlaciones.

---

# 🚀 Próximos pasos
- **Ensemble híbrido**: EffNet-B3 (pooling 4-feat) + Swin-Tiny (top7 isotonic).
- **Regularización**: stacking con priors y selección de features no correlacionadas.
- **Multimodal**: clínico + MRI.
- **Aumento de datos**: ADNI, augmentations.

---

# 📎 Apéndice: utilidades clave
Incluye `safe_sigmoid`, `fit_temperature`, `normalize_score`, `agg_patient`.

---

### 📎 Apéndice: fragmentos y utilidades

#### `safe_sigmoid` y `temperature scaling`

```python
import numpy as np
from scipy.optimize import minimize

def safe_sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1/(1+np.exp(-z))

def fit_temperature(logits, y_true, init_T=1.0, bounds=(0.05,10.0)):
    logits = np.asarray(logits,float); y_true = np.asarray(y_true,float)
    def nll(T):
        p = safe_sigmoid(logits/T); eps=1e-7
        return -np.mean(y_true*np.log(p+eps)+(1-y_true)*np.log(1-p+eps))
    return float(minimize(lambda t: nll(t[0]), x0=[init_T], bounds=[bounds], method="L-BFGS-B").x[0])
```

#### Normalización de columnas de score

```python
SCORE_ALIASES = ['y_score','sigmoid(logit)','sigmoid(logits)','pred']

def normalize_score(df):
    for c in SCORE_ALIASES:
        if c in df.columns:
            df = df.rename(columns={c:'y_score'})
            break
    assert 'y_score' in df.columns, "No encuentro columna de score."
    return df
```

#### Pooling a paciente (`mean`/`trimmed20`/`top7`/`pmean_2`)

```python
import pandas as pd
import numpy as np

def agg_patient(df):
    g = df.groupby('patient_id')['y_score']
    return pd.DataFrame({
        'mean': g.mean(),
        'trimmed20': g.apply(lambda s: s.sort_values().iloc[int(len(s)*.1):int(len(s)*.9)].mean() if len(s)>=10 else s.mean()),
        'top7': g.apply(lambda s: s.sort_values(ascending=False).head(7).mean()),
        'pmean_2': g.apply(lambda s: (np.mean(np.power(np.clip(s,0,1),2)))**0.5)
    }).reset_index()
```

Actualizado: 08/09/2025 22:45