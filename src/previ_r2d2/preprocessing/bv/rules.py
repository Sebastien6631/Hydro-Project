"""Règles de calage hydrologique d'un bassin versant (kc_unit, K_base, exposition).

Port direct du script de référence validé `onboarding_bv.py` : les formules et
constantes sont reprises telles quelles, calées sur 8 bassins versants connus
(voir `centrales/REFERENCE/bv_rules.json` et `centrales/REFERENCE/centrales_calibration.json`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paliers K_base par altitude moyenne du BV — calés sur les BV connus.
# La zone 1000-1300 m est ambiguë (Apas 1201->3.8 vs Labastidette 1181->2.0,
# cette dernière n'étant plus une centrale suivie) : signalée pour revue
# manuelle, à affiner un jour avec une fraction de surface enneigée.
KBASE_STEPS = [
    (1800, 4.5),
    (1300, 3.8),
    (1000, 3.0),
    (450, 2.0),
    (0, 2.5),
]
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


def fit_rules(ref_dir: Path) -> Rules:
    """Cale les règles kc_unit et K_base sur `centrales_calibration.json`."""
    with open(ref_dir / "centrales_calibration.json", encoding="utf-8") as fh:
        rows = json.load(fh)
    d = pd.DataFrame(rows)
    a = d["Altitude_moyenne_BV"].to_numpy(dtype=float)

    A = np.c_[a, np.ones_like(a)]
    slope, intercept = np.linalg.lstsq(A, d["kc_unit"].to_numpy(float), rcond=None)[0]
    kc_pred = np.clip(A @ [slope, intercept], 0.60, 1.05)
    kc_mae = float(np.mean(np.abs(d["kc_unit"].to_numpy(float) - kc_pred)))

    rules = Rules(
        kc_slope=float(slope),
        kc_intercept=float(intercept),
        kc_clip_min=0.60,
        kc_clip_max=1.05,
        kbase_steps=[list(s) for s in KBASE_STEPS],
        expo_gain=0.09,
        expo_min=0.70,
        expo_max=1.15,
        n_calib=len(d),
        note=(
            f"kc_unit MAE={kc_mae:.3f} ; K_base par paliers, "
            f"zone {KBASE_REVIEW_BAND[0]}-{KBASE_REVIEW_BAND[1]} m à confirmer ; "
            "exposition ~66% expliquée par l'aspect (gain 0.09) ; BV très ombragés "
            "type Melles (0.75) à corriger manuellement."
        ),
    )
    logger.info("Règles calées sur %d BV. kc_unit MAE=%.3f", len(d), kc_mae)
    return rules


def load_rules(path: Path) -> Rules:
    with open(path, encoding="utf-8") as fh:
        return Rules(**json.load(fh))


def save_rules(rules: Rules, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(rules), fh, indent=2, ensure_ascii=False)
    logger.info("Règles écrites -> %s", path)


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
