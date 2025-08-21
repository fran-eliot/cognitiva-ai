# 🧠 Proyecto de Detección Temprana de Alzheimer (COGNITIVA-AI)

Este proyecto explora la detección temprana de Alzheimer combinando **datos clínicos tabulares** y **resonancias magnéticas (MRI)** del conjunto **OASIS-2**.  
Construimos dos pipelines complementarios:

1) **COGNITIVA-AI-CLINIC** → *aprendizaje clásico (ML) con datos clínicos.*  
2) **COGNITIVA-AI-IMAGES** → *Deep Learning con imágenes (fine-tuning de ResNet50).*

El objetivo es ofrecer una línea base sólida, documentar decisiones y resultados, y dejar una hoja de ruta clara para mejoras.

---

## 📦 Datos y alcance

- **Fuente principal:** `oasis_longitudinal_demographics.xlsx` (demografía + medidas clínicas/estructurales) y MRI en carpetas `OAS2_XXX_MRy/RAW` (+ `OLD` en algunos sujetos).  
- **Problema:** clasificación **binaria** a nivel de paciente:  
  - `0 = Nondemented`  
  - `1 = Demented` o `Converted` (paciente que progresa a demencia).  
- **Razonamiento de la binarización:** el dataset es pequeño; un esquema binario simplifica y mejora estabilidad en métricas, especialmente con validación honesta por paciente.

> 🔒 **Evitar fugas de información** (data leakage):  
> - Con datos clínicos: elegimos **una visita por sujeto** (baseline) para no mezclar repeticiones del mismo paciente entre train y test.  
> - Con imágenes: el **split es por paciente/scan_id**; todas las slices de un scan quedan en el mismo subset.

---

# 1️⃣ COGNITIVA-AI-CLINIC (Datos clínicos)

### 📂 Variables (principales)
- Demográficas: `Age`, `Sex (M/F)`, `Education (EDUC)`, `SES`.  
- Clínicas/neuropsicológicas: `MMSE`, `CDR`.  
- Estructurales globales: `eTIV`, `nWBV`, `ASF`.  
- Target: `Group` → mapeado a `target ∈ {0,1}` (Nondemented vs Demented/Converted).

### 🧹 Preprocesamiento
- **Renombrado a `snake_case`** para legibilidad (`Subject ID → subject_id`, etc.).  
- **Selección de una visita por sujeto**: baseline (mínimo `mr_delay`) para tener un único registro representativo por paciente.  
- Conversión de tipos numéricos y **imputación** (mediana) para columnas con NaN (`ses`, `mmse`, `cdr`, …).  
- Codificación:
  - `sex`: `M → 0`, `F → 1`.  
  - `hand`: one-hot (categoría desconocida a `Unknown`).  

### ⚙️ Modelado
- **Modelos base:** Logistic Regression, Random Forest, XGBoost.  
- **Validación:** `StratifiedKFold` y métrica **ROC-AUC**.  
- **Optimización:**  
  - *GridSearchCV* para Random Forest.  
  - **Algoritmo Genético (DEAP)** para RF/XGB: búsqueda evolutiva de hiperparámetros (más eficiente en espacios grandes/no convexos).

> ℹ️ **Por qué ROC-AUC**: mide la capacidad de discriminación a todos los umbrales, robusta ante desbalance moderado y facilita comparación entre modelos.

### 📊 Resultados (clínico)

- **Cross-val (grid/genético):**  
  - RF (grid) → mejor ROC-AUC CV ≈ **0.9224**  
  - RF (GA) → mejor ROC-AUC CV ≈ **0.9215**  
  - XGB (GA) → mejor ROC-AUC CV ≈ **0.9215**

- **Test hold-out (final):**
  | Modelo              | ROC-AUC (Test) |
  |---------------------|----------------|
  | Random Forest (opt) | 0.884          |
  | **XGBoost (opt)**   | **0.897**      |

## 📊 Resultados (MRI – nivel paciente)

> **Split estratificado por paciente 60/20/20** (train/val/test)

| Configuración | Preprocesamiento | Train Acc | Val Acc | Test Acc | ROC-AUC | Comentarios |
|---|---|---:|---:|---:|---:|---|
| **5 slices** | **Sin CLAHE** | ↑ (≈0.94) | ≈0.73 | **0.89** | **0.938** | Línea base fuerte; generaliza bien en test. |
| 5 slices | CLAHE | ≈0.95 | ≈0.72 | 0.69 | 0.777 | Mejora visual, pero menor discriminación; probable realce de ruido. |
| 5 slices | CLAHE + z-score | ≈0.96 | ≈0.75 | 0.72 | 0.820 | Recupera estabilidad; mejor balance entre clases, sigue < baseline. |
| **20 slices** | CLAHE + z-score | **0.98** | ≈0.71 | **0.80** | **0.858** | Más cobertura anatómica; mejora global respecto a CLAHE, aunque con algo de sobreajuste. |

**Conclusión MRI:**  
- El **baseline sin CLAHE con 5 slices** fue el más alto en **ROC-AUC (0.94)** en nuestro test.  
- **Aumentar a 20 slices** mejora la robustez general y el *recall* de la clase positiva, pero aún no supera al baseline en ROC-AUC.  
- **CLAHE** debe usarse con cautela (o de forma selectiva) y acompañado de normalización adecuada.

---

## 🧠 Decisiones de diseño (y por qué)

- **Binarizar `Group`** (`Nondemented` vs `Demented/Converted`): simplifica el problema y mejora estabilidad en CV y test.  
- **Una visita por sujeto (clínico)**: evita duplicar pacientes y **fuga de información**.  
- **Split por paciente (imágenes)**: todas las slices de un `scan_id` deben ir al mismo subset → evaluación realista.  
- **Evaluación por paciente** (MRI): lo clínicamente relevante es la clasificación del **paciente**, no de cada corte aislado.  
- **Early stopping**: protege frente a sobreajuste visible (train ≫ val).  
- **Métrica ROC-AUC**: adecuada con clases desbalanceadas/moderadas y para comparar modelos a distintos umbrales.

---

## 🧪 Notas de implementación (fragmentos clave)

**Mapeo de etiquetas por `scan_id`:**
```python
labels = pd.read_excel("oasis_longitudinal_demographics.xlsx").rename(columns={
    "MRI ID": "scan_id", "Group": "group"
})
labels["label"] = labels["group"].map({"Nondemented": 0, "Demented": 1, "Converted": 1})
labels = labels.dropna(subset=["label"]).astype({"label": int})
label_map = labels.set_index("scan_id")["label"].to_dict()
```

**Extracción de `scan_id` desde ruta y guardado de slices (ejemplo):**
```python
from pathlib import Path
import nibabel as nib, numpy as np, cv2, os

def save_slices_from_nifti(img_path, scan_id, out_dir, max_slices=5, use_clahe=False):
    vol = nib.load(img_path).get_fdata()
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8) * 255
    vol = vol.astype(np.uint8)

    zmid = vol.shape[2] // 2
    idxs = range(zmid - max_slices//2, zmid - max_slices//2 + max_slices)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if use_clahe else None

    paths = []
    for i, z in enumerate(idxs):
        if 0 <= z < vol.shape[2]:
            sl = vol[:,:,z]
            if clahe is not None:
                sl = clahe.apply(sl)
            out = os.path.join(out_dir, f"{scan_id}_slice{i}.png")
            cv2.imwrite(out, sl)
            paths.append(out)
    return paths
```

**Dataset PyTorch (con `return_path` para agregación paciente):**
```python
class MRIDataset(Dataset):
    def __init__(self, img_dir, label_map, scan_ids, transform=None, return_path=False):
        self.transform = transform
        self.return_path = return_path
        self.label_map = label_map
        self.scan_ids = set(scan_ids)
        all_png = glob.glob(os.path.join(img_dir, "*.png"))
        self.img_paths = [p for p in all_png if self._scan_id(p) in self.scan_ids]

    def _scan_id(self, p):
        return os.path.basename(p).split("_slice")[0]

    def __getitem__(self, i):
        p = self.img_paths[i]
        sid = self._scan_id(p)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # ResNet espera 3 canales
        if self.transform: img = self.transform(img)
        y = self.label_map[sid]
        return (img, y, p) if self.return_path else (img, y)

    def __len__(self): return len(self.img_paths)
```

**Agregación por paciente (nivel test):**
```python
from collections import defaultdict
patient_probs, patient_labels = defaultdict(list), {}

with torch.no_grad():
    for x, y, paths in test_loader:
        prob = torch.softmax(model(x.to(device)), dim=1)[:,1].cpu().numpy()
        for pp, yy, path in zip(prob, y.numpy(), paths):
            sid = os.path.basename(path).split("_slice")[0]
            patient_probs[sid].append(pp)
            patient_labels[sid] = int(yy)

final_probs = {sid: np.mean(v) for sid, v in patient_probs.items()}
final_preds = {sid: int(p > 0.5) for sid, p in final_probs.items()}
```

---

## 🧭 Limitaciones conocidas

- **Tamaño muestral** limitado → sensibilidad a splits y a *overfitting*.  
- **Slices 2D** no capturan plena continuidad 3D.  
- **CLAHE** puede perjudicar patrones de intensidad sutiles (dependiente de parámetros y de sujeto).  
- Diferencia CV vs Test en clínico sugiere **optimismo** por búsqueda de hiperparámetros (normal/esperable).

---

## 🚀 Próximos pasos propuestos

1) **Regularización explícita en imágenes:**  
   - Congelar capas iniciales en warm-up, **Dropout** en la cabeza, **Weight Decay**, **Label Smoothing**.  
   - **Class weights** en la loss si hay desbalance real.

2) **Más slices / selección inteligente:**  
   - 32–64 slices equiespaciadas; o seleccionar automáticamente cortes “más informativos” (ej., centrados en hipocampo).

3) **Normalización adaptada a MRI:**  
   - Z-score **por volumen** (no por slice) o *histogram matching* intra-sujeto.

4) **Arquitecturas modernas / 3D:**  
   - EfficientNet, DenseNet, ConvNeXt; versiones 3D si el cómputo lo permite.

5) **Multimodalidad:**  
   - Fusionar **clínico + imágenes** (early/late fusion); útil cuando los patrones de ambos dominios son complementarios.

---

## 🧪 Resumen comparativo final

| Modalidad       | Modelo                          | ROC-AUC (Test) | Notas |
|-----------------|----------------------------------|----------------|-------|
| **Clínico**     | **XGBoost (opt)**               | **0.897**      | Mejor en tabular; pipeline robusto y explicable. |
| **Imágenes**    | **ResNet50 – 5 slices (sin CLAHE)** | **0.938**   | Mejor AUC en test con split por paciente. |
| Imágenes        | ResNet50 – 5 slices (CLAHE)     | 0.777          | Legibilidad ↑, discriminación ↓. |
| Imágenes        | ResNet50 – 5 slices (CLAHE+Z)   | 0.820          | Estabilidad ↑, aún < baseline. |
| Imágenes        | ResNet50 – **20 slices (CLAHE+Z)** | 0.858       | Robustez ↑, mejor recall; algo de sobreajuste. |

---

## 🧾 Reproducibilidad (guía breve)

1. **Clínico**  
   - Cargar Excel → renombrar columnas → 1 visita por sujeto → imputación → encoding.  
   - `train_test_split` estratificado → *GridSearchCV* / GA → reportar CV y Test.

2. **Imágenes**  
   - Convertir `.hdr/.img` a PNG (5 o 20 slices) con `label_map` por `scan_id`.  
   - Split estratificado **por paciente** (60/20/20).  
   - Entrenar ResNet50 (fine-tuning) con *early stopping*.  
   - Evaluar **por paciente** agregando probabilidades de sus slices.

---

**Autoría:** Fran Ramírez  
**Año:** 2025.
