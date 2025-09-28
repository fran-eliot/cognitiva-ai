# ======================= SOLO LAS 4 FIGURAS QUE FALTAN (P26) =======================
# - roc_pr_P26_TEST_OAS1.png
# - roc_pr_P26_TEST_OAS2.png
# - cost_curve_P26_VAL_OAS1.png
# - cost_curve_P26_VAL_OAS2.png
#
# Requisitos:
#  - p26_intermodal/p26_val_preds.csv
#  - p26_intermodal/p26_test_preds.csv
#  - (opcional) p26_release/config/deployment_config.json  -> para leer umbrales S2
#
# Salida: p26_release/QA/*.png
# ================================================================================
import pandas as pd, numpy as np, json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 1) Localiza BASE automáticamente (Colab vs Windows). Cambia a mano si lo prefieres.
CANDIDATES = [
    Path("/content/drive/MyDrive/CognitivaAI"),                       # Colab
    Path(r"C:\Users\PC\Desktop\cognitiva-ai")                          # Windows (ajusta si usas otra ruta)
]
BASE = next((p for p in CANDIDATES if (p/"p26_intermodal").exists()), CANDIDATES[-1])

P26    = BASE/"p26_intermodal"
REL    = BASE/"p26_release"
CFG    = REL/"config"/"deployment_config.json"
QA_OUT = REL/"QA"
QA_OUT.mkdir(parents=True, exist_ok=True)

print(f"BASE = {BASE}")
print(f"Guardará PNG en: {QA_OUT}")

# 2) Umbrales S2 (si existe config)
S2_THR = {"OAS1": 0.42, "OAS2": 0.4928655287824083}
if CFG.exists():
    try:
        cfg = json.loads(CFG.read_text(encoding="utf-8"))
        th  = cfg.get("thresholds", {})
        if th:
            S2_THR.update({str(k).upper(): float(v) for k,v in th.items()})
            print("Umbrales S2 desde config:", S2_THR)
    except Exception as e:
        print("⚠️ No pude leer deployment_config.json, uso valores por defecto:", S2_THR)
else:
    print("ℹ️ Sin deployment_config.json; uso umbrales por defecto:", S2_THR)

# 3) Cargar predicciones P26 (VAL/TEST)
val = pd.read_csv(P26/"p26_val_preds.csv")
tst = pd.read_csv(P26/"p26_test_preds.csv")

# Normaliza nombre de cohorte y busca columna de probas
def normalize(df):
    df = df.copy()
    ccol = None
    for cand in ["cohort","Cohort"]:
        if cand in df.columns: ccol = cand; break
    if ccol is None:
        raise KeyError("No encuentro columna 'cohort'/'Cohort' en el CSV.")
    df["cohort"] = df[ccol].astype(str).str.upper()
    return df

def find_prob_col(df):
    for c in ["proba_cal","p_fused_cal","y_prob","y_score","proba","p_img"]:
        if c in df.columns: return c
    raise KeyError("No encuentro columna de probabilidad (proba_cal/p_fused_cal/y_prob/y_score/proba/p_img).")

val = normalize(val)
tst = normalize(tst)

pcol_val = find_prob_col(val)
pcol_tst = find_prob_col(tst)
print(f"Columna de probas (VAL): {pcol_val} | (TEST): {pcol_tst}")

# 4) ROC/PR (TEST) por cohorte
def plot_roc_pr(df, cohort, prob_col, out_png):
    dfc = df[(df["cohort"]==cohort) & df["y_true"].notna()].copy()
    if dfc.empty:
        print(f"⚠️ Sin filas para ROC/PR TEST {cohort}")
        return
    y = dfc["y_true"].astype(int).values
    p = dfc[prob_col].astype(float).values

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    plt.figure(figsize=(8.5,4.5))
    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.title(f"ROC — P26 TEST {cohort}\nAUC = {roc_auc:.3f}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(alpha=0.3)
    # PR
    plt.subplot(1,2,2)
    plt.plot(rec, prec, linewidth=2)
    plt.title(f"PR — P26 TEST {cohort}\nAP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("✅ Guardado:", out_png.name)

plot_roc_pr(tst, "OAS1", pcol_tst, QA_OUT/"roc_pr_P26_TEST_OAS1.png")
plot_roc_pr(tst, "OAS2", pcol_tst, QA_OUT/"roc_pr_P26_TEST_OAS2.png")

# 5) Curvas de coste (VAL) por cohorte con 5:1 y umbral S2 marcado
def cost_curve(df, cohort, prob_col, out_png, c_fn=5.0, c_fp=1.0, thr_mark=None):
    dfc = df[(df["cohort"]==cohort) & df["y_true"].notna()].copy()
    if dfc.empty:
        print(f"⚠️ Sin filas para COST VAL {cohort}")
        return
    y = dfc["y_true"].astype(int).values
    p = dfc[prob_col].astype(float).values

    thrs = np.linspace(0,1,201)
    costs = []
    for t in thrs:
        yhat = (p>=t).astype(int)
        fp = np.sum((yhat==1)&(y==0))
        fn = np.sum((yhat==0)&(y==1))
        costs.append(c_fn*fn + c_fp*fp)

    plt.figure(figsize=(6,4))
    plt.plot(thrs, costs, linewidth=2)
    plt.title(f"Coste (VAL) 5:1 — {cohort}")
    plt.xlabel("Umbral"); plt.ylabel("Coste = 5×FN + 1×FP")
    plt.grid(alpha=0.3)

    if thr_mark is not None:
        t = float(thr_mark)
        yhat = (p>=t).astype(int)
        fp = np.sum((yhat==1)&(y==0))
        fn = np.sum((yhat==0)&(y==1))
        cost = c_fn*fn + c_fp*fp
        plt.axvline(t, linestyle="--")
        plt.scatter([t],[cost], zorder=3)
        plt.text(t, cost, f"  thr={t:.3f}\n  cost={cost:.1f}", va="bottom")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("✅ Guardado:", out_png.name)

cost_curve(val, "OAS1", pcol_val, QA_OUT/"cost_curve_P26_VAL_OAS1.png", c_fn=5.0, c_fp=1.0, thr_mark=S2_THR["OAS1"])
cost_curve(val, "OAS2", pcol_val, QA_OUT/"cost_curve_P26_VAL_OAS2.png", c_fn=5.0, c_fp=1.0, thr_mark=S2_THR["OAS2"])

print("\n🎉 Listo. Revisa ahora en:", QA_OUT)
