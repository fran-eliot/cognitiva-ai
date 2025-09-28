# 🧪 QA_README — COGNITIVA-AI (Release QA)

> Guía de verificación rápida y trazable de la **release**: qué contiene, cómo comprobarla y qué esperar (métricas, umbrales y figuras).  
> Destino: equipo técnico, revisores y *stakeholders* no técnicos.

---

## 1) Propósito de esta carpeta

Esta carpeta asegura que la **entrega es verificable** sin re-entrenar modelos. Incluye:

- **Tablas de referencia** con resultados de test (P24/P26).
- **Política de decisión activa (S2)** y sus **umbrales por cohorte**.
- **Figuras clave** (ROC/PR, calibración, coste) listas para presentaciones.
- **CSV de QA** (predicciones, ECE, informes de test).
- Este **documento** para guiar la comprobación en 2–10 minutos.

---

## 2) Contenido (mapa de archivos)

> Los nombres pueden variar levemente; si un archivo no está, consúltalo en `p26_intermodal/` o `p25_informe_final/`.

**Figuras — P24 (imagen, meta simple)**  
- `p25_roc_P24_TEST_OAS1.png` / `p25_roc_P24_TEST_OAS2.png` — Curvas ROC por cohorte.  
- `p25_pr_P24_TEST_OAS1.png` / `p25_pr_P24_TEST_OAS2.png` — Curvas PR por cohorte.  
- `p25_cal_P24_TEST_OAS1.png` / `p25_cal_P24_TEST_OAS2.png` — Calibración (reliability).  
- `p25_cost_curve_OAS1.png` / `p25_cost_curve_OAS2.png` — Curvas de coste (VAL, 5:1).

**Figuras — Resumen y P26 (intermodal)**  
- `executive_table.png` — Tabla ejecutiva comparativa (P19/P22/P23/P24/P26).  
- `policy_S2_callout.png` — *Call-out* visual de la **política S2** (umbrales activos y efecto).  
- *(Si las añadiste)* `roc_pr_P26_TEST_OAS1.png`, `roc_pr_P26_TEST_OAS2.png`,  
  `cost_curve_P26_VAL_OAS1.png`, `cost_curve_P26_VAL_OAS2.png`.

**CSV — QA y métricas**  
- `p26b_test_report_cost_5to1.csv` — Informe de TEST a umbrales **5:1** (por cohorte).  
- `p26_test_calibration_ece.csv` — **ECE/MCE** de calibración (ALL, OAS1, OAS2).  
- `p26b_test_preds_calibrated.csv` — Predicciones calibradas de **TEST** (P26b).  
- `p26_val_preds_calibrated.csv` — Predicciones calibradas de **VAL** (P26b).

---

## 3) Política de decisión activa — **S2**

**Qué es S2**  
- **Base 5:1 (FN:FP)** por cohorte, **aprendida en validación**.  
- **OAS2 con objetivo de recall alto** (cribado): se ajusta el umbral hasta **Recall ≥ 0.90** si el coste lo permite.

**Umbrales activos (TEST @ S2)**  
- **OAS1 = 0.42**  
- **OAS2 ≈ 0.492866**

**Confusión esperada (TEST @ S2)**

| Cohorte |   Thr   | TP | FP | TN | FN | Precision | Recall |   Acc   | Cost |
|:------:|:-------:|---:|---:|---:|---:|----------:|-------:|--------:|-----:|
|  OAS1  | 0.420000| 14 |  9 | 18 |  6 |   0.6087  | 0.7000 | 0.6809  |  39  |
|  OAS2  | 0.492866| 11 |  6 |  5 |  1 |   0.6471  | 0.9167 | 0.6957  |  11  |

> *Coste = 5×FN + 1×FP. La política S2 prioriza **Recall** (sensibilidad) con un coste controlado.*

---

## 4) Métricas de referencia (probabilidades)

**P24 (imagen, LR elastic-net + Platt)**  
- **ALL:** AUC=**0.727**, PR-AUC=**0.717**, Brier=**0.220**  
- **OAS1:** AUC=**0.754**, PR-AUC=**0.736**, Brier=**0.211**  
- **OAS2:** AUC=**0.750**, PR-AUC=**0.805**, Brier=**0.238**

**P26 (intermodal, fusión Late con clínico)**  
- **ALL:** AUC=**0.713**, PR-AUC=**0.712**, Brier=**0.234**  
- **OAS1:** AUC=**0.754**, PR-AUC=**0.736**, Brier=**0.208**  
- **OAS2:** AUC=**0.652**, PR-AUC=**0.728**, Brier=**0.288**

**Calibración (P26, TEST — ECE/MCE)**

| Cohorte |  ECE@10 |  MCE@10 |
|:------:|--------:|--------:|
|  ALL   | 0.1784  | 0.4068  |
|  OAS1  | 0.1500  | 0.5775  |
|  OAS2  | 0.3125  | 0.7659  |

> Lectura rápida: **Brier ≈ 0.22–0.23** (aceptable); **ECE** algo peor en OAS2 → recomendable **recalibración por cohorte** si el entorno de despliegue se parece a OAS2.

---

## 5) Verificación en 2 minutos (smoke test “de escritorio”)

1. **Abrir** `p26b_test_report_cost_5to1.csv` y confirmar que las filas **OAS1/OAS2** coinciden **(± redondeo)** con la tabla de §3.  
2. **Abrir** `p26_test_calibration_ece.csv` y confirmar **ECE/MCE** de §4.  
3. **Visual**: abrir `executive_table.png` y `policy_S2_callout.png` para comprobar coherencia con §3–§4.  
4. **(Opcional)**: con tu `predict_end_to_end.py`, correr el *golden set* de TEST, agregar por `patient_id` y aplicar **umbrales S2** → recuperar la **misma confusión** por cohorte.

> Si faltara alguna figura P26 (ROC/PR o coste), puedes reutilizar las de P24 o generarlas a partir de `p26b_test_preds_calibrated.csv` y `p26_val_preds_calibrated.csv`.

---

## 6) Interpretación mínima (para *stakeholders*)

- **AUC ~0.71–0.73** (discriminación) + **Brier ~0.22–0.23** (calibración razonable).  
- **Decisión clínica ≠ AUC**: se toma por **umbral** según **coste** (S2), priorizando **Recall** (evitar FN).  
- Con **S2**: OAS1 **Recall 0.70** (Coste=39), OAS2 **Recall ≈0.92** (Coste=11).  
- **Implicación**: mejor **cribado** (menos casos que “se escapan”) con **FP** asumibles (derivación a pruebas, no daño).

---

## 7) Cómo reproducir figuras (si necesitas regenerarlas)

**ROC/PR (P24 o P26)**  
- Entrada: `*_test_preds.csv` o `p26b_test_preds_calibrated.csv` (col. `y_true`, `y_prob`, `cohort`).  
- Filtro por **cohort**; usa `roc_curve` / `precision_recall_curve` (sklearn).  
- Guardar como `roc_pr_P26_TEST_OAS1.png`, `roc_pr_P26_TEST_OAS2.png`.

**Curvas de coste (VAL, 5:1)**  
- Entrada: `p26_val_preds_calibrated.csv` (col. `y_true`, `y_prob`, `cohort`).  
- Barrer `thr ∈ [0,1]`; **Coste = 5×FN + 1×FP**.  
- Guardar como `cost_curve_P26_VAL_OAS1.png`, `cost_curve_P26_VAL_OAS2.png`.

**Calibración (ECE/MCE)**  
- *Binning* en 10 cubos de probabilidad; `|mean(pred) − frac(positivos)|` → **ECE@10**.

> Ya hay versiones listas en esta carpeta; solo regenera si cambias *código* o *política*.

---

## 8) *Checksums* e integridad (opcional)

Genera SHA-256 de todos los archivos y anexa `CHECKSUMS.txt`.

- **Windows (PowerShell)**
  ```powershell
  Get-ChildItem -File | ForEach-Object {
    $h = Get-FileHash $_.FullName -Algorithm SHA256
    "$($h.Hash)  $($_.Name)" | Out-File -FilePath CHECKSUMS.txt -Append -Encoding ascii
  }
  ```
- **Linux/Mac**
  ```bash
  for f in *; do [ -f "$f" ] && sha256sum "$f"; done > CHECKSUMS.txt
  ```

---

## 9) Errores típicos y cómo resolver

- **“No coincide la confusión con S2”** → comprueba que aplicas **umbrales por cohorte** y ratio **5:1**.  
- **Diferencias de décimas** → revisa **redondeos** y que usas **probabilidades calibradas** (no *logits*).  
- **ECE alta en OAS2** → considera **recalibración local** (Platt o isotónica) con datos recientes del sitio.

---

## 10) Licencia y uso

Proyecto con fines **académicos/demostrativos**. **No uso clínico** sin validación regulatoria y supervisión experta.

---

## 11) Changelog (QA)

- **v1.1 (final)**: Añadidas figuras ejecutivas, *call-out* S2, ECE P26 y CSV de test/val calibrados.  
- **v1.0**: Primer volcado de figuras P24 y reporte P26 cost-5:1.

---

### Apéndice A — Resumen “de bolsillo”

- **S2** = coste 5:1 (FN>FP), umbrales por cohorte (**OAS1=0.42**, **OAS2≈0.493**) y **Recall alto** en OAS2.  
- **TEST @ S2**: OAS1 **R=0.70** (Coste=39), OAS2 **R≈0.92** (Coste=11).  
- **AUC** ~0.71–0.73; **Brier** ~0.22–0.23; **ECE** mayor en OAS2 → recalibrar si el sitio se parece a OAS2.

---
