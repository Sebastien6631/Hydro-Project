"""Métriques Prometheus (phase 4.1), exposées par l'API sur `/metrics`.

Pas de thread de rafraîchissement : les gauges modèle sont recalculées à
chaque scrape (2 centrales -- lire les `version.json` à chaque appel est
négligeable, YAGNI d'ajouter un scheduler pour ça)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

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


def refresh_model_gauges() -> None:
    """Recalcule les gauges modèle à partir des `version.json` promus."""
    MODEL_KGE.clear()
    for info in predict_service.model_infos():
        kge = info.get("kge_stacking")
        if kge is not None:
            MODEL_KGE.labels(dossier=info["dossier"]).set(kge)


def render_latest() -> tuple[bytes, str]:
    """Corps + content-type au format d'exposition Prometheus."""
    refresh_model_gauges()
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
