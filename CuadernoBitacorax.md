# 🧭 Cuaderno de Bitácora del Proyecto Cognitiva-AI
> Diario técnico detallado (por días y por fases) con decisiones, incidencias y resultados.  
> Objetivo: trazabilidad completa desde la preparación del entorno hasta backbones alternativos y ensembles.

---

# 📌 Convenciones y notas rápidas

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

# 📅 Diario cronológico

## 📅 09/08/2025 — Fase preliminar
- Revisión de pipelines hasta P9 (EffNet-B3 stable).
- Discusión sobre limitaciones de calibración y preparación de P10.
- **Incidencia**: logits extremadamente grandes → overflow en `exp`.

---

## 📅 10/08/2025 — Pipeline 10 (EffNet-B3 stable + calibración)
- Se aplicó **temperature scaling** y **isotonic regression**.
- Implementación de `safe_sigmoid` con `clip[-50,50]` para evitar overflow.
- Resultados (rango test): **AUC=0.546–0.583**, PR-AUC=0.50–0.53, Acc=0.51–0.55, Recall=1.0, Precision=0.47–0.49.
- Conclusión: calibración aporta interpretabilidad pero degrada métricas → necesidad de ensembles.

---

## 📅 11/08/2025 — Ensembles semilla (P10-ext)
- Semillas 41, 42, 43 con media/TRIMMED/TOP7.
- **Incidencia**: resultados pobres (AUC ~0.5) → semilla-ensemble no mejora solo.
- Stacking de pooling (mean, trimmed20, top7, p2):
  - RF/STACK → VAL AUC ~0.90, TEST AUC ~0.75.
  - RAND(500 muestras): VAL AUC=0.909, TEST AUC=0.754.
  - STACK_LR: coefs ≈ [0.407, 0.409, 0.485, 0.416], intercept=−0.923 → TEST AUC=0.754, PR-AUC=0.748.

---

## 📅 12/08/2025 — Documentación
- Inclusión de resultados de P10 y P10-ext en README e Informe Técnico.
- Normalización de columnas en CSV (y_score, sigmoid(logit), pred).

---

## 📅 15/08/2025 — Inicio Pipeline 11 (Backbones alternativos)
- Configuración de `/p11_alt_backbones`.
- **Incidencia**: “Mountpoint must not already contain files” → solución: no remount si ya montado.
- **Incidencia**: DATA_DIR marcado como inexistente pese a estar → solución: reinicio del entorno.
- Validación de mapas OK, config guardada.

---

## 📅 16/08/2025 — ConvNeXt-Tiny
- Guardado `convnext_tiny.in12k_ft_in1k_val_slices.csv` y `test_slices.csv`.
- Resultados (mean): TEST AUC=0.509, PR-AUC=0.479, Acc=0.489, Recall=1.0, Precision=0.455.
- Conclusión: bajo rendimiento, peor que EffNet-B3.

---

## 📅 17/08/2025 — DenseNet-121
- Uso de pesos ImageNet (sin checkpoint propio).
- Resultados (trimmed20): TEST AUC=0.343, PR-AUC=0.407, Acc=0.319, Recall=0.75, Precision=0.36.
- Conclusión: muy bajo rendimiento en este dataset.

---

## 📅 18/08/2025 — Swin-Tiny
- Resultados (top7): TEST AUC=0.641, PR-AUC=0.597, Acc=0.553, Recall=0.95, Precision=0.95.
- Conclusión: mejor alternativo, aunque no supera a EffNet-B3 ensembles.

---

## 📅 19/08/2025 — Catálogo multi-backbone
- Escaneo de directorios: SwinTiny, ConvNeXt, DenseNet, EffNet previos.
- Normalización de columnas (mapa automático → `y_score`).
- Construcción de `val_patient_features_backbones.csv` y `test_patient_features_backbones.csv`.

---

## 📅 20/08/2025 — Ensemble simple
- AVG de 12 señales mean.
- VAL AUC=0.476, TEST AUC=0.713.
- STACK_LR(all features): VAL AUC=0.810, TEST AUC=0.298 (overfitting severo).

---

## 📅 21/08/2025 — Ensemble Dirichlet (3 means)
- FEATURES: SwinTiny_mean, ConvNeXt_mean, DenseNet_mean.
- Pesos (ejemplo): Swin 0.972, ConvNeXt 0.004, Dense 0.024.
- Resultados: TEST AUC=0.520, PR-AUC=0.523, Acc=0.468, Recall=1.0, Precision=0.444.
- Conclusión: ligera mejora sobre Dense/ConvNeXt, peor que Swin-top7.

---

## 📅 22/08/2025 — Ensemble Dirichlet EXT (12 features)
- Incluyó múltiples variantes (mean, trimmed, top7).
- Resultado: TEST AUC=0.361, PR-AUC=0.405.
- Conclusión: sobreajuste por exceso de features.

---

## 📅 23/08/2025 — Stacking L1 fuerte
- Penalización fuerte en coeficientes → todos anulados.
- Resultado trivial: AUC=0.5.
- Conclusión: n(VAL)=10 demasiado pequeño + correlaciones altas.

---

## 📅 24/08/2025 — Isotonic sobre Swin-Tiny
- Resultado: TEST AUC=0.566, PR-AUC=0.458, Acc=0.553, Recall=0.95, Precision=0.487.
- Conclusión: ligera mejora de robustez con alta sensibilidad.

---

## 📅 25/08/2025 — Catálogo ampliado
- Inclusión de directorios previos: `oas1_resnet18_linearprobe`, `ft_effb3_colab`, `ft_effb3_stable_colab_plus`.
- Validación automática de columnas/tamaños → todos compatibles.

---

## 📅 27/08/2025 — Documentación intermedia
- Actualización de README, Informe Técnico y Cuaderno con resultados preliminares de P11.
- Inclusión de filas para ConvNeXt, DenseNet, Swin-Tiny.

---

## 📅 29/08/2025 — Ajustes finales
- Normalización definitiva en `comparison_backbones_eval.csv`.
- Confirmación de Swin-Tiny top7 como mejor alternativo.
- Resumen ensembles:
  - Dirichlet (3 means): TEST AUC=0.52.
  - Dirichlet EXT: TEST AUC=0.36.
  - STACK_LR(all): TEST AUC=0.30.
  - Swin-Tiny isotonic: TEST AUC=0.566.

---

# 🧩 Recap por Fases/Pipelines

## 📌 P1 – Clínico OASIS-2
- Modelo: XGB.
- Resultado: AUC=0.897.

## 📌 P2 – Clínico fusión
- Modelo: XGB con fusión de más variables.
- Resultado: AUC=0.991, Recall ~1.0.
- Riesgo: overfitting.

## 📌 P3 – MRI OASIS-2
- Modelo: ResNet-50.
- Resultado: AUC=0.938.

## 📌 P5 – MRI Colab (ResNet18 + Calib)
- AUC=0.724, PR-AUC=0.606, Acc=0.60, Recall=0.80, Precision=0.52.

## 📌 P6 – EffNet-B3 embeddings
- AUC=0.704, PR-AUC=0.623, Recall=0.90, Precision=0.60.

## 📌 P7 – EffNet-B3 finetune
- AUC=0.876, PR-AUC=0.762, Acc=0.745, Recall=1.0, Precision=0.625.

## 📌 P9 – EffNet-B3 stable
- AUC=0.740, PR-AUC=0.630, Acc=0.72, Recall=0.65, Precision=0.62.

## 📌 P10 – EffNet-B3 stable + calibración
- AUC=0.546–0.583, PR-AUC=0.50–0.53, Acc=0.51–0.55, Recall=1.0, Precision=0.47–0.49.

## 📌 P10-ext – Ensembles de pooling
- RAND/STACK con 4 features (mean, trimmed20, top7, p2).
- AUC test ~0.75, PR-AUC ~0.74, Recall ~0.50–0.70.

## 📌 P11 – Backbones alternativos
- ConvNeXt-Tiny (mean): AUC=0.509, PR-AUC=0.479.
- DenseNet-121 (trimmed20): AUC=0.343, PR-AUC=0.407.
- Swin-Tiny (top7): AUC=0.641, PR-AUC=0.597.

## 📌 P11-ensembles
- Dirichlet (3 means): AUC=0.520.
- Dirichlet EXT (12 feats): AUC=0.361.
- STACK_LR(all features): AUC=0.30.
- Swin-Tiny + isotonic: AUC=0.566.

---

# 🚀 Próximos pasos
1. Ensemble híbrido: EffNet-B3 (4 features) + Swin-Tiny isotonic.
2. Regularización en stacking para evitar coef=0.
3. Multimodal clínico+MRI (P2 + P7/P10-ext).
4. Ampliación de dataset (ADNI, augmentations).

