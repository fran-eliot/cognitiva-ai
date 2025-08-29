# 🧠 Contexto Proyecto COGNITIVA-AI (Agosto 2025)

## 🎯 Objetivo

Desarrollo de un sistema de **detección temprana de Alzheimer**
combinando: - **Datos clínicos tabulares** (OASIS-1 y OASIS-2). -
**Imágenes MRI** procesadas a nivel slice → paciente. - Estrategias
**ensemble y calibración** para maximizar **recall clínico**, clave para
cribado.

------------------------------------------------------------------------

## 🚀 Progreso hasta ahora

### 1. Datos clínicos

-   **XGBoost en OASIS-2**: AUC=0.897.\
-   **Fusión OASIS-1+2**: modelos estables (LogReg/RandomForest/XGB con
    AUC≈0.99).\
-   Interpretabilidad: **CDR + MMSE** = variables más críticas.

### 2. MRI inicial

-   **ResNet50 CPU baseline** (OASIS-2 slices): AUC test=0.938.\
-   Migración a **Colab GPU** con ResNet18 embeddings + calibración
    isotónica.
    -   AUC≈0.72, Recall≈0.8 en test tras ajuste de umbral.

### 3. EfficientNet-B3

-   Embeddings 1536D.\
-   Clasificadores LR/MLP/XGB.\
-   Mejor rendimiento con **ensemble LR+XGB calibrado**:
    -   VAL: AUC=0.815 \| Recall=0.95 \| Acc=0.70\
    -   TEST: AUC=0.704 \| Recall=0.90 \| Acc=0.70

### 4. Pipeline 10 --- Stable Plus (EffNet-B3 con calibración extendida)

-   **Normalización robusta de pesos**.\
-   **Temperature scaling**.\
-   **Pooling a nivel paciente**: mean, median, top-k.\
-   **Resultados**: recall=1.0 en test en varias variantes →
    sensibilidad clínica máxima.\
-   Ensembles avanzados (random weights, stacking con logistic
    regression) confirman robustez:
    -   VAL PR-AUC≈0.92 \| TEST PR-AUC≈0.74.\
    -   Acc test≈0.66, Recall=0.70, Precision≈0.58.

------------------------------------------------------------------------

## 📊 Estado actual

-   **Exploración de ensembles**:
    -   Mean, median, trimmed, top-k → ✅.\
    -   Random search de pesos → ✅ (mejor recall/precision).\
    -   Stacking LR → ✅ (robusto, valida recall alto).\
-   **Intento de ensemble por seeds** → ❌ (no aportó mejora).\
-   **Gráficas y CSV de comparación actualizadas**.

------------------------------------------------------------------------

## 🔜 Próximos pasos

1.  **Explorar backbones alternativos**: DenseNet121, Swin-Transformer,
    ConvNeXt, etc.\
2.  Consolidar ensembles con estos backbones antes de pasar a la fase
    multimodal (MRI + clínico).\
3.  Preparar informes (README, Informe Técnico, Cuaderno de Bitácora)
    con los nuevos resultados.

------------------------------------------------------------------------

📌 **Resumen corto**:\
Tenemos un pipeline MRI estable (EffNet-B3 calibrado) con recall clínico
muy alto (≥0.9), reforzado con ensembles avanzados. Lo siguiente es
probar **backbones alternativos** antes de entrar en la fase multimodal.
