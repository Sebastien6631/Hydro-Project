"""Métriques hydrologiques pures pour le modèle LightGBM (meta) -- port
fidèle de lightgbm_model.py:99-217 (Previ_v2). Pas de persistance :
fonctions de calcul pur, appelées à la volée par l'entraînement (pièce C de
ce sous-projet, puis l'orchestration d'entraînement ultérieure)."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def kge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Retourne 1 - KGE (à minimiser) ; pénalité douce sur le biais si variance quasi nulle.

    Masque d'abord les paires avec un NaN (y_true ou y_pred) -- np.std/np.mean
    renvoient NaN dès qu'UN SEUL élément du tableau est NaN (contamine tout le
    tableau, pas seulement la position concernée), ce qui rendait tout le score
    d'un fold TimeSeriesSplit NaN dès qu'un seul trou capteur (débit manquant à
    une heure donnée) tombait dans sa fenêtre de validation -- observé en
    conditions réelles (Optuna : "score moyen (1-KGE)=nan sur 6 folds" pour
    Clairac, alors que 5476/5482 points de ce fold étaient valides)."""
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not valid.all():
        y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) < 2:
        # Fold entièrement (ou presque) troué -- aucun signal exploitable,
        # pénalité maximale (équivalent KGE=-1) plutôt qu'une exception
        # np.corrcoef sur un tableau vide/à un seul élément.
        return 2.0
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true < EPS or std_pred < EPS:
        beta = np.mean(y_pred) / (np.mean(y_true) + EPS)
        return float(1.0 + abs(beta - 1.0))
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    if np.isnan(r):
        r = 0.0
    alpha = std_pred / std_true
    beta = np.mean(y_pred) / (np.mean(y_true) + EPS)
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(1.0 - kge)


def debit_quantiles(
    y_orig: np.ndarray, y_ref: np.ndarray | None = None
) -> tuple[int, float, float, float]:
    """Seuils hydrologiques (q_start_pct, q_start, q90, q99) partagés par debit_weights/postprocess_debit."""
    if y_ref is None:
        y_ref = y_orig

    q90 = np.percentile(y_orig, 90)
    q99 = np.percentile(y_orig, 99)

    q99_ref = np.percentile(y_ref, 99)
    extreme_ratio = y_ref.max() / (q99_ref + EPS)
    if extreme_ratio > 3.0:
        q_start_pct = 15
    elif extreme_ratio > 2.0:
        q_start_pct = 25
    else:
        q_start_pct = 50

    q_start = np.percentile(y_orig, q_start_pct)
    return q_start_pct, float(q_start), float(q90), float(q99)


def debit_weights(
    y_orig: np.ndarray, mult_poids: float, y_ref: np.ndarray | None = None
) -> np.ndarray:
    """Poids d'entraînement par zone hydrologique (miroir de postprocess_debit)."""
    _, q_start, q90, q99 = debit_quantiles(y_orig, y_ref)

    above_q99 = np.maximum(0, y_orig - q99) / (q99 - q90 + EPS)
    above_q90 = np.maximum(0, y_orig - q90) / (q99 - q90 + EPS)
    above_qstart = np.maximum(0, y_orig - q_start) / (q90 - q_start + EPS)

    return np.where(
        y_orig > q99,
        1.0 + mult_poids * above_q99.clip(0, 3) * 6,
        np.where(
            y_orig > q90,
            1.0 + mult_poids * above_q90.clip(0, 2) * 2,
            np.where(
                y_orig > q_start,
                1.0 + mult_poids * above_qstart.clip(0, 1),
                1.0 + 0.1 * mult_poids,
            ),
        ),
    )


def postprocess_debit(
    y_pred: np.ndarray, q_start: float, q90: float, q99: float, span: int = 8
) -> np.ndarray:
    """Lisse les prédictions en régime normal (miroir de debit_weights, cohérence entraînement/inférence)."""
    y = np.array(y_pred, dtype=float)
    y_smooth = pd.Series(y).ewm(span=span, min_periods=1).mean().values
    blend = np.clip(1.0 - (y - q_start) / (q90 - q_start + EPS), 0.0, 1.0)
    blend = np.where(y > q99, 0.0, blend)
    return (blend * y_smooth + (1.0 - blend) * y).clip(0)
