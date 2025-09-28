# src/cognitiva_ai/config.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def load_deployment_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"deployment_config.json no encontrado: {p}")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    # Chequeos mínimos
    if "thresholds" not in cfg:
        raise ValueError("deployment_config.json debe incluir 'thresholds' {OAS1: thr, OAS2: thr}")
    return cfg

def get_policy_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("policy", "S2")

def get_thresholds(cfg: Dict[str, Any]) -> Dict[str, float]:
    th = cfg.get("thresholds", {})
    # Normaliza claves
    norm = {str(k).upper(): float(v) for k, v in th.items()}
    return norm
