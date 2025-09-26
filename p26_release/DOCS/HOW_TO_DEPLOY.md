# HOW TO DEPLOY — CognitivaAI v1.1

## 1) Entorno
- Python 3.10+
- Instala dependencias de `META/ENVIRONMENT.txt` (bloqueado sklearn).

## 2) Inferencia por lote (CSV)
- Prepara CSV con columnas clínicas (ver `clinical_signature.json`) y/o `p_img`.
- Ejecuta `scripts/predict_batch.py` (opcional: crearlo) para leer CSV y generar `proba_cal` y `decision`.

## 3) API (opcional)
- Montar un endpoint `/predict` que:
  - Valide `clinical` y/o calcule `p_img` con P24.
  - Aplique fusión Late + Platt por cohorte.
  - Devuelva `proba_cal`, `decision` y `threshold_used`.

## 4) Monitorización
- Guardar `proba_cal` y decisión por cohorte.
- Semanal: ECE/MCE por cohorte; recall/precision cuando lleguen etiquetas.
- Alarmas: ECE>0.20 (OAS1) / >0.25 (OAS2); recall OAS2<0.65.

## Política activa (S2)

**Política activa (S2)** — *single pipeline*  
- **OAS1:** decisión por coste **FN:FP=5:1**, umbral **thr = 0.420000**  
- **OAS2:** **umbral por objetivo de recall** (VAL), target = **0.85**, umbral **thr = 0.492866**  
- Alternativas disponibles:
  - **5:1 puro**: ver `thresholds_5to1` en `CONFIG/deployment_config.json`.
- Nota operativa:
  - Monitorizar **ECE/MCE** y **positivity rate** por cohorte; recalibrar y/o ajustar umbral si deriva el dominio.
