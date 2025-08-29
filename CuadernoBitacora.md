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
