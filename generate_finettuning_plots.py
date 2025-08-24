
"""
generate_finettuning_plots.py

Uso rápido (Colab o local):

1) Preparar un CSV con predicciones a **nivel paciente** del modelo fine-tuned:
   - Archivo: finetuning_patient_predictions.csv
   - Columnas: patient_id,y_true,y_score
     * y_true ∈ {0,1}
     * y_score ∈ [0,1] (probabilidad calibrada)

2) Ejecutar:
   python generate_finettuning_plots.py --csv finetuning_patient_predictions.csv --thr 0.3651 --outdir graficos

Se generarán:
 - ft_b3_patient_roc.png
 - ft_b3_patient_pr.png
 - ft_b3_patient_calibration.png
 - ft_b3_patient_confusion_thr03651.png
Y un resumen de métricas en: ft_b3_patient_metrics.txt
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss, confusion_matrix

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if not np.any(mask):
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        w = mask.mean()
        ece += w * np.abs(bin_acc - bin_conf)
    return ece

def plot_roc(y_true, y_score, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("1 - Especificidad (FPR)")
    plt.ylabel("Sensibilidad (TPR)")
    plt.title("ROC — EfficientNet-B3 Fine-Tuning (Paciente)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return roc_auc

def plot_pr(y_true, y_score, outpath):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"AP (PR-AUC) = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR — EfficientNet-B3 Fine-Tuning (Paciente)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return ap

def plot_calibration(y_true, y_score, outpath, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accs = []
    confs = []
    for i in range(n_bins):
        mask = (y_score >= bins[i]) & (y_score < bins[i+1])
        if not np.any(mask):
            continue
        accs.append(y_true[mask].mean())
        confs.append(y_score[mask].mean())
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confianza media (prob)")
    plt.ylabel("Frecuencia empírica (acc)")
    plt.title("Curva de calibración — EfficientNet-B3 FT (Paciente)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_confusion(y_true, y_score, thr, outpath):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0,1], ["Pred 0", "Pred 1"])
    plt.yticks([0,1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title(f"Matriz de confusión (thr={thr}) — Paciente")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return tp, fp, tn, fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV con columnas: patient_id,y_true,y_score")
    parser.add_argument("--thr", type=float, default=0.3651, help="Umbral clínico")
    parser.add_argument("--outdir", default="graficos", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    y_true = df["y_true"].values.astype(int)
    y_score = df["y_score"].values.astype(float)

    # ROC y PR
    roc_auc = plot_roc(y_true, y_score, os.path.join(args.outdir, "ft_b3_patient_roc.png"))
    ap = plot_pr(y_true, y_score, os.path.join(args.outdir, "ft_b3_patient_pr.png"))

    # Calibración
    brier = brier_score_loss(y_true, y_score)
    ece = expected_calibration_error(y_true, y_score, n_bins=10)
    plot_calibration(y_true, y_score, os.path.join(args.outdir, "ft_b3_patient_calibration.png"))

    # Confusión a umbral clínico
    tp, fp, tn, fn = plot_confusion(y_true, y_score, args.thr, os.path.join(args.outdir, f"ft_b3_patient_confusion_thr{str(args.thr).replace('.','')}.png"))

    # Resumen de métricas
    with open(os.path.join(args.outdir, "ft_b3_patient_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"ROC-AUC: {roc_auc:.3f}\n")
        f.write(f"PR-AUC (AP): {ap:.3f}\n")
        f.write(f"Brier score: {brier:.3f}\n")
        f.write(f"ECE (10 bins): {ece:.3f}\n")
        f.write(f"TP={tp} FP={fp} TN={tn} FN={fn} (thr={args.thr})\n")

if __name__ == "__main__":
    main()
