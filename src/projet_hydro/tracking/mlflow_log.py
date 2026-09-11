"""Journalisation MLflow d'un entraînement -- point d'entrée unique.

Rien n'est recalculé ici : `build_results` (artifacts.py) produit déjà toutes
les métriques et `write_artifacts` tous les fichiers. Ce module ne fait que
les aplatir au format MLflow.

**Sans `MLFLOW_TRACKING_URI`, tout est inerte** et `log_training_run` rend
None. C'est la condition pour que la suite de tests et la CI n'aient jamais
besoin d'un serveur : pas de mock, pas de fixture, une sortie anticipée.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIMENT = "projet_hydro"
ARTIFACT_PATH = "model"
PRODUCTION_ALIAS = "production"

# Clés de `results` loggées telles quelles (scalaires).
_SCALAR_METRICS = ("kge_lgbm", "kge_lstm", "kge_stacking")

# Clés de `meta_config` loggées comme paramètres.
_META_PARAMS = ("centrale", "horizon", "horizon_steps", "timestep", "meta_type",
                "seq_len", "n_splits", "q90_train")


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _git_commit() -> str:
    """HEAD courant, pour relier un run MLflow au code exact qui l'a produit.
    Un dépôt absent ou un git indisponible ne doit pas faire échouer un
    entraînement de plusieurs heures -- d'où le repli silencieux."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _flatten_metrics(results: dict) -> dict[str, float]:
    """Métriques scalaires : globales, par régime hydrologique, par saison.

    `kge_by_step` en est exclu : c'est une série, loggée séparément avec un
    `step` MLflow pour être tracée en courbe plutôt qu'en 8 scalaires plats.
    """
    flat: dict[str, float] = {}

    for key in _SCALAR_METRICS:
        value = results.get(key)
        if isinstance(value, (int, float)):
            flat[key] = float(value)

    # kge_by_regime / kge_by_season : {modele: {tranche: {"kge": x, ...}}}
    for bloc, prefixe in (("kge_by_regime", "regime"), ("kge_by_season", "saison")):
        for modele, tranches in (results.get(bloc) or {}).items():
            for tranche, mesures in tranches.items():
                kge = mesures.get("kge")
                if isinstance(kge, (int, float)):
                    flat[f"{prefixe}_{_slug(tranche)}_{_slug(modele)}"] = float(kge)

    return flat


def _slug(texte: str) -> str:
    """MLflow refuse la plupart des caractères hors alphanumerique, tiret, point et underscore dans un
    nom de métrique ; les libellés viennent de `kge_by_regime`/`kge_by_season`
    et contiennent espaces et accents ("hautes eaux", "été")."""
    remplacements = str.maketrans("àâäéèêëîïôöùûüç ", "aaaeeeeiioouuuc_")
    return "".join(c for c in texte.lower().translate(remplacements) if c.isalnum() or c == "_")


def log_training_run(results: dict, meta_config: dict, weights_dir: Path,
                     hyperparams: dict | None = None) -> str | None:
    """Ouvre un run, logge params/métriques/artefacts, rend le `run_id`.

    Rend None si MLflow est désactivé OU si le serveur est injoignable : un
    entraînement réussi ne doit jamais être perdu parce que le service de
    suivi est tombé.
    """
    if not enabled():
        return None

    try:
        import mlflow
    except ImportError:
        logger.warning("MLFLOW_TRACKING_URI est défini mais mlflow n'est pas installé -- suivi ignoré.")
        return None

    dossier = meta_config.get("centrale", "?")
    horizon = meta_config.get("horizon", "?")

    try:
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"{dossier}-h{horizon}") as run:
            mlflow.set_tags({
                "centrale": dossier,
                "horizon": horizon,
                "git_commit": _git_commit(),
                "seed": results.get("seed"),
            })

            params = {k: meta_config[k] for k in _META_PARAMS if k in meta_config}
            params.update(hyperparams or {})
            # La fenêtre d'évaluation EST un paramètre : split_train_test est
            # positionnel (20% de fin), donc deux runs sur des CSV de longueurs
            # différentes ne mesurent pas la même période -- comparer leurs KGE
            # sans cette trace n'a aucun sens (cf. orchestrator.py).
            params.update({f"eval_{k}": v for k, v in (results.get("evaluation_window") or {}).items()})
            mlflow.log_params(params)

            mlflow.log_metrics(_flatten_metrics(results))

            for nom, serie in (results.get("kge_by_step") or {}).items():
                for pas, valeur in enumerate(serie or []):
                    if isinstance(valeur, (int, float)):
                        mlflow.log_metric(f"by_step_{nom}", float(valeur), step=pas)

            if weights_dir.exists():
                # artifact_path fixe : donne au registry une URI stable
                # (runs:/<id>/model) plutot que la racine du run.
                mlflow.log_artifacts(str(weights_dir), artifact_path=ARTIFACT_PATH)

            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 -- voir docstring
        logger.warning("Suivi MLflow indisponible (%s) -- entraînement conservé, run non enregistré.", exc)
        return None


def registered_name(dossier: str, horizon: int) -> str:
    return f"projet_hydro-{dossier}-h{horizon}"


def register_production_model(dossier: str, horizon: int, run_id: str | None,
                              git_tag: str) -> str | None:
    """Enregistre le modele promu au Model Registry et lui pose l'alias
    `@production`. Rend le numero de version du registry, ou None.

    Appele APRES le commit et le tag git de `promote_model`, et jamais dans
    son bloc try/except : la promotion est deja actee a ce stade (fichiers
    copies, commit et tag poses). Faire echouer la promotion parce qu'un
    serveur de suivi est tombe serait un rollback pour rien.

    Ce qui est enregistre est le repertoire d'artefacts, pas une saveur
    MLflow chargeable : le modele est un trio (LightGBM + BiLSTM + meta) plus
    ses scalers, et le chargement passe par `load_trained_models`. MLflow sert
    ici d'index et d'historique des versions promues, DVC porte les octets.
    """
    if not enabled() or not run_id:
        return None

    try:
        import mlflow
    except ImportError:
        return None

    nom = registered_name(dossier, horizon)
    try:
        version = mlflow.register_model(
            f"runs:/{run_id}/{ARTIFACT_PATH}", nom,
            tags={"git_tag": git_tag, "centrale": dossier, "horizon": horizon},
        )
        mlflow.MlflowClient().set_registered_model_alias(nom, PRODUCTION_ALIAS, version.version)
        logger.info("Registry MLflow : %s v%s -> @%s", nom, version.version, PRODUCTION_ALIAS)
        return str(version.version)
    except Exception as exc:  # noqa: BLE001 -- voir docstring
        logger.warning("Registry MLflow indisponible (%s) -- promotion %s conservee.", exc, git_tag)
        return None
