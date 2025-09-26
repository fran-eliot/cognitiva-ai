# MODEL CARD — CognitivaAI v1.1 (P26b single)

**Arquitectura:**  
- Imagen (P24: modelo + Platt) → `p_img`  
- Clínico (LR con imputación+scaler) → `p_clin`  
- Fusión Late (meta-LR sobre `p_img`+`p_clin`)  
- Calibración Platt **por cohorte** (P26b) → `proba_cal`  
- Decisión por coste (FN:FP=5:1) con umbrales aprendidos en VAL:
  - OAS1 = 0.420
  - OAS2 = 0.490

**Métricas (TEST, probs calibradas):**  
- ALL: AUC≈0.684 · PR-AUC≈0.660 · Brier≈0.241  
- OAS1 / OAS2: ver `QA/p26b_test_report_cost_5to1.csv`.

**Suposiciones de entrada:**  
- Clínico: 9 campos (ver `clinical_signature.json`).  
- Imagen: usar pipeline P24 **o** aportar `p_img` directo.  
- Cohorte: 'OAS1'/'OAS2' (para calibrador/umbral). Si se desconoce, usar 'OAS1' por defecto y monitorizar calibración.

**Riesgos & mitigaciones:**  
- Descalibración en dominios nuevos → recalibrar Platt por sitio con ≥50–100 casos y actualizar umbral 5:1.  
- Tamaño muestral reducido → reportar ICs y monitorizar ECE/MCE.

**Archivos clave:** ver `META/MANIFEST.json`.

## Política activa (S2)

**Política activa (S2)** — *single pipeline*  
- **OAS1:** decisión por coste **FN:FP=5:1**, umbral **thr = 0.420000**  
- **OAS2:** **umbral por objetivo de recall** (VAL), target = **0.85**, umbral **thr = 0.492866**  
- Alternativas disponibles:
  - **5:1 puro**: ver `thresholds_5to1` en `CONFIG/deployment_config.json`.
- Nota operativa:
  - Monitorizar **ECE/MCE** y **positivity rate** por cohorte; recalibrar y/o ajustar umbral si deriva el dominio.
