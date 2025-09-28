from __future__ import annotations
import numpy as np
from typing import Dict

def policy_S2_thresholds(cohort: str, thresholds: Dict[str, float]) -> float:
    """
    S2: Base 5:1 con umbral por cohorte aprendido en VAL + ajuste para Recall target en OAS2.
    En producción, ya viene aplicado en deployment_config.json.
    """
    return float(thresholds.get(cohort.upper(), 0.5))
