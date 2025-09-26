## Apéndice — Decisión, calibración y estabilidad (P24)

**Coste vs. umbral (VAL):**  
- Figuras: `p25_cost_curve_OAS1.png`, `p25_cost_curve_OAS2.png`  
- Sensibilidad (3:1, 5:1, 7:1, 10:1): `p25_cost_sensitivity_p24.csv`

**Matriz de confusión (TEST @ thr por cohorte):**  
- CSV: `p25_confusion_test_p24.csv`

**Curvas y calibración (TEST):**  
- ROC/PR: `p25_roc_P24_TEST_OAS1.png`, `p25_roc_P24_TEST_OAS2.png`, `p25_pr_P24_TEST_OAS1.png`, `p25_pr_P24_TEST_OAS2.png`  
- Calibración: `p25_cal_P24_TEST_OAS1.png`, `p25_cal_P24_TEST_OAS2.png`  
- ECE/MCE: `p25_p24_test_calibration_ece.csv`

**Incertidumbre (bootstrap 95% CI):**  
- `p25_p24_test_bootstrap_ci.csv`

**Interpretabilidad (LR elastic-net):**  
- Top coeficientes: `p25_p24_top_coefficients.md` / `.csv`