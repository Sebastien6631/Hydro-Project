"""Règles de calage hydrologique d'un bassin versant (kc_unit, K_base, exposition).

Port direct du script de référence validé `onboarding_bv.py` : les formules et
constantes sont reprises telles quelles, calées sur 8 bassins versants connus
(voir `centrales/REFERENCE/bv_rules.json` et `centrales/REFERENCE/centrales_calibration.json`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KBASE_REVIEW_BAND = (1000, 1300)


@dataclass
class Rules:
    """Coefficients de calage, sérialisables en JSON."""

    kc_slope: float
    kc_intercept: float
    kc_clip_min: float
    kc_clip_max: float
    kbase_steps: list
    # exposition : facteur = clip(1 + expo_gain * cos(aspect - sud), min, max)
    expo_gain: float
    expo_min: float
    expo_max: float
    n_calib: int
    note: str = ""


def load_rules(path: Path) -> Rules:
    with open(path, encoding="utf-8") as fh:
        return Rules(**json.load(fh))



def estimate_kc_unit(alt_mean: float, r: Rules) -> float:
    return float(np.clip(r.kc_slope * alt_mean + r.kc_intercept, r.kc_clip_min, r.kc_clip_max))


def estimate_kbase(alt_mean: float, r: Rules) -> tuple[float, bool]:
    """Retourne (K_base, needs_review)."""
    for threshold, value in r.kbase_steps:
        if alt_mean >= threshold:
            review = KBASE_REVIEW_BAND[0] <= alt_mean < KBASE_REVIEW_BAND[1]
            return float(value), review
    return float(r.kbase_steps[-1][1]), False


def estimate_exposition(aspect_deg: float | None, r: Rules) -> float:
    """Facteur d'exposition à partir de l'aspect moyen (0=N, 90=E, 180=S, 270=O).

    Versant sud -> facteur > 1 (fonte accélérée) ; versant nord -> facteur < 1.
    Si l'aspect est indisponible (BV plat / pas de MNT), retourne 1.0.
    """
    if aspect_deg is None or np.isnan(aspect_deg):
        return 1.0
    south_alignment = np.cos(np.radians(aspect_deg - 180.0))
    factor = 1.0 + r.expo_gain * south_alignment
    return float(np.clip(round(factor, 2), r.expo_min, r.expo_max))
