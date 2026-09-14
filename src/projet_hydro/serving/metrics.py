"""Métriques Prometheus (phase 4.1), exposées par l'API sur `/metrics`.

Pas de thread de rafraîchissement : les gauges modèle sont recalculées à
chaque scrape (2 centrales -- lire les `version.json` à chaque appel est
négligeable, YAGNI d'ajouter un scheduler pour ça).

Dérive (phase 4.2) : `MODEL_KGE` ci-dessus recalcule en lisant un fichier à
chaque scrape, mais un rapport Evidently prend de vraies secondes -- bien
trop lent pour un scrape (Prometheus l'attend sous la seconde). Les gauges
de dérive lisent donc `logs/drift/<dossier>.json`, écrit par
`cron/scripts/check-drift.py` (lancé à part, manuellement ou via Airflow),
jamais calculées en direct ici -- même pattern que les marqueurs DVC."""

from __future__ import annotations

import json

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from projet_hydro.common import config
from projet_hydro.serving import predict_service

REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "http_requests_total", "Nombre de requêtes HTTP", ["method", "path", "status"], registry=REGISTRY
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "Durée des requêtes HTTP (s)", ["method", "path"], registry=REGISTRY
)
MODEL_KGE = Gauge(
    "model_kge_stacking", "KGE du modèle stacking promu, par centrale", ["dossier"], registry=REGISTRY
)
DATA_DRIFT_SHARE = Gauge(
    "data_drift_share", "Part des colonnes en dérive (test K-S), par centrale", ["dossier"], registry=REGISTRY
)
DATA_DRIFT_DETECTED = Gauge(
    "data_drift_detected", "1 si dérive du dataset détectée (>= seuil), 0 sinon", ["dossier"], registry=REGISTRY
)


def refresh_model_gauges() -> None:
    """Recalcule les gauges modèle à partir des `version.json` promus."""
    MODEL_KGE.clear()
    for info in predict_service.model_infos():
        kge = info.get("kge_stacking")
        if kge is not None:
            MODEL_KGE.labels(dossier=info["dossier"]).set(kge)


def refresh_drift_gauges() -> None:
    """Relit les rapports de dérive écrits par `check-drift.py` (pas de calcul ici)."""
    DATA_DRIFT_SHARE.clear()
    DATA_DRIFT_DETECTED.clear()
    drift_dir = config.ROOT / "logs" / "drift"
    if not drift_dir.exists():
        return
    for path in sorted(drift_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        dossier = data["dossier"]
        DATA_DRIFT_SHARE.labels(dossier=dossier).set(data["drift_share"])
        DATA_DRIFT_DETECTED.labels(dossier=dossier).set(1 if data["dataset_drift"] else 0)


def render_latest() -> tuple[bytes, str]:
    """Corps + content-type au format d'exposition Prometheus."""
    refresh_model_gauges()
    refresh_drift_gauges()
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
