"""Éligibilité à l'entraînement -- premier entraînement (12 mois d'historique
minimum, cycle saisonnier complet) ou réentraînement mensuel échu (>= 30 jours
depuis le dernier essai, quelle qu'ait été son issue -- cf. train_state via
le marker `train_<dossier>_h<horizon>` déjà écrit par cron/scripts/train.py)."""

from __future__ import annotations

import json
from datetime import datetime

from projet_hydro.common import config
from projet_hydro.common.dvc_markers import MARKERS_DIR
from projet_hydro.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv

MIN_HISTORY_DAYS = 365
RETRAIN_INTERVAL_DAYS = 30


def history_span_days(dossier: str) -> float:
    """Jours couverts par data_preparation.csv (0.0 si absent/vide)."""
    path = config.NAS_DATA_ROOT / dossier / "data_preparation.csv"
    df = read_data_preparation_csv(path)
    if df.empty:
        return 0.0
    return (df.index.max() - df.index.min()).total_seconds() / 86400


def has_production_model(dossier: str, horizon: int) -> bool:
    return (config.MODELS_DIR / dossier / f"h{horizon}" / "version.json").exists()


def last_train_attempt(dossier: str, horizon: int) -> datetime | None:
    marker = MARKERS_DIR / f"train_{dossier}_h{horizon}.json"
    if not marker.exists():
        return None
    return datetime.fromisoformat(json.loads(marker.read_text(encoding="utf-8"))["last_run"])


def is_eligible_for_training(dossier: str, horizon: int, now: datetime | None = None) -> bool:
    """True si premier entraînement possible (12 mois atteints, aucun modèle en
    prod) OU réentraînement mensuel échu (>= 30 jours depuis le dernier essai)."""
    now = now or datetime.now()
    if history_span_days(dossier) < MIN_HISTORY_DAYS:
        return False
    if not has_production_model(dossier, horizon):
        return True
    last = last_train_attempt(dossier, horizon)
    if last is None:
        return True
    return (now - last).days >= RETRAIN_INTERVAL_DAYS
