"""MLflow tracker -- port du wrappeur Previ_v2 (circuit breaker, jamais
bloquant pour le cron), réduit au flux hybride (`train_meta`/`predict_future_meta`)
et complète les TODO restés ouverts côté Previ_v2 : `log_prediction_hybrid`
(prédiction horaire, jamais loggée avant) et `log_verification_hybrid`
(vérification a posteriori). Pas de model registry actif (stages/aliases) --
`register_production_version` reste de l'observabilité seule, jamais lu par
`predict_orchestrator` (source de vérité = models/<dossier>/h<horizon>/,
versionné DVC+git, cf. promotion.py)."""

from __future__ import annotations

import functools
import logging
import os

import pandas as pd

from previ_r2d2.common import config

# Setup Previ_v2 : tracking URI + artifact store pointent tous les deux vers
# un même répertoire local/NAS (pas de serveur MLflow). Depuis MLflow 3.x, ce
# backend fichier est en "maintenance mode" et désactivé par défaut -- on le
# réactive explicitement, c'est le choix d'architecture assumé ici.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

logger = logging.getLogger(__name__)

MLFLOW_OK: bool = True
TRACKING_URI: str | None = None


def get_uri() -> str | None:
    global TRACKING_URI
    if TRACKING_URI is None:
        TRACKING_URI = config.MLFLOW_URI
    return TRACKING_URI


def safe(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global MLFLOW_OK
        if not MLFLOW_OK or get_uri() is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            MLFLOW_OK = False
            logger.warning("MLflow désactivé suite à une erreur : %s", exc)
            return None
    return wrapper


def _flatten_metrics(results: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for model_name, comp in results.get("components", {}).items():
        prefix = model_name.lower().replace(" ", "_")
        for k, v in comp.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
                flat[f"{prefix}_{k}"] = float(v)
    if not flat and results.get("components"):
        logger.warning("Aucune métrique numérique valide dans results['components'] (tout NaN/non-numérique ?)")
    return flat


@safe
def log_training_hybrid(centrale: str, horizon: int, results: dict, output_dir, params: dict | None = None) -> str | None:
    """Log un run d'entraînement hybride (BiLSTM + LGB + Ridge/LGBM meta)."""
    import mlflow

    mlflow.set_tracking_uri(get_uri())
    exp_name = f"{centrale}_{horizon}h_hybrid"
    mlflow.set_experiment(exp_name)

    run_name = f"train_hybrid — {pd.Timestamp.now().strftime('%Y-%m-%d %Hh')}"
    with mlflow.start_run(
        run_name=run_name,
        tags={"type": "training_hybrid", "centrale": centrale, "horizon": str(horizon)},
    ) as run:
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        flat = _flatten_metrics(results)
        if flat:
            mlflow.log_metrics(flat)
        results_path = output_dir / "results.json"
        if results_path.exists():
            mlflow.log_artifact(str(results_path), artifact_path="train")
        return run.info.run_id


@safe
def log_prediction_hybrid(centrale: str, horizon: int, prediction: dict) -> str | None:
    """Log un point de prédiction horaire (run_prediction) -- comble le TODO
    Previ_v2 (log_prediction_hybrid jamais implémenté côté hybride)."""
    import mlflow

    mlflow.set_tracking_uri(get_uri())
    mlflow.set_experiment(f"{centrale}_{horizon}h_hybrid")

    run_name = f"PREDICT {pd.Timestamp.now().strftime('%Y-%m-%d %Hh')}"
    with mlflow.start_run(
        run_name=run_name,
        tags={"type": "prediction_hybrid", "centrale": centrale, "horizon": str(horizon)},
    ) as run:
        mlflow.log_param("now", prediction["now"])
        mlflow.log_metric("q_entrant_h1_m3s", float(prediction["q_entrant_m3s"][0]))
        mlflow.log_metric("q_stacking_h1_m3s", float(prediction["q_stacking_m3s"][0]))
        return run.info.run_id


@safe
def log_verification_hybrid(centrale: str, horizon: int, run_id: str, metrics: dict) -> None:
    """Enrichit le run `run_id` (créé par log_prediction_hybrid) avec les
    métriques réelles a posteriori -- réutilise le run_id déjà connu par
    l'appelant plutôt que de le rechercher par tags (contrairement à
    log_verification v4 côté Previ_v2)."""
    import mlflow

    mlflow.set_tracking_uri(get_uri())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"reel_{k}": float(v) for k, v in metrics.items()})


@safe
def register_production_version(centrale: str, horizon: int, version: int, kge: float) -> None:
    """Trace la promotion dans MLflow -- observabilité seule, jamais lu par
    predict_orchestrator (source de vérité = fichier DVC, cf. promotion.py)."""
    import mlflow

    mlflow.set_tracking_uri(get_uri())
    mlflow.set_experiment(f"{centrale}_{horizon}h_hybrid")
    with mlflow.start_run(
        run_name=f"promotion v{version}",
        tags={"type": "promotion", "centrale": centrale, "horizon": str(horizon)},
    ):
        mlflow.log_param("version", version)
        mlflow.log_metric("kge_stacking", float(kge))
