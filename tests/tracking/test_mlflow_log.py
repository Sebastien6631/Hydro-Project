from __future__ import annotations

from pathlib import Path

import pytest

from previ_r2d2.tracking import mlflow_log


RESULTS = {
    "kge_lgbm": 0.91, "kge_lstm": 0.88, "kge_stacking": 0.94, "seed": 42,
    "evaluation_window": {"n_train": 100, "test_start": "2026-01-01"},
    "kge_by_step": {"stack": [0.9, 0.8], "lgbm": [0.7, None]},
    "kge_by_regime": {"Stacking": {"hautes eaux": {"kge": 0.5, "r": 0.9}}},
    "kge_by_season": {"Stacking": {"été": {"kge": 0.6}}},
}
META = {"centrale": "apas_G1_G4", "horizon": 8, "meta_type": "ridge", "seq_len": 24}


def test_disabled_without_tracking_uri(monkeypatch, tmp_path):
    """Le garde-fou central : sans serveur, un entraînement doit aboutir
    quand même. Toute la suite (et la CI) tourne sans MLflow -- si cette
    sortie anticipée saute, 295 tests réclament un serveur."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert mlflow_log.enabled() is False
    assert mlflow_log.log_training_run(RESULTS, META, tmp_path) is None

    # Chaine vide = desactive, au meme titre qu'absente : x-env injecte
    # MLFLOW_TRACKING_URI="" dans tous les conteneurs quand .env ne la definit
    # pas, donc la variable EXISTE cote conteneur. Un test `in os.environ`
    # rendrait True et chaque `docker compose run` chercherait un serveur.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    assert mlflow_log.enabled() is False
    assert mlflow_log.log_training_run(RESULTS, META, tmp_path) is None


def test_a_dead_tracking_server_never_loses_a_training(monkeypatch, tmp_path):
    """Un entraînement dure des heures. Un serveur de suivi injoignable doit
    coûter le run MLflow, jamais le modèle : on rend None, on ne propage pas."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")  # port fermé

    mlflow = pytest.importorskip("mlflow")

    def _boom(*a, **k):
        raise ConnectionError("serveur MLflow injoignable")

    monkeypatch.setattr(mlflow, "set_experiment", _boom)

    assert mlflow_log.log_training_run(RESULTS, META, tmp_path) is None


def test_metric_names_survive_accents_and_spaces(tmp_path):
    """`kge_by_regime`/`kge_by_season` produisent des libellés humains
    ("hautes eaux", "été"). MLflow rejette espaces et accents dans un nom de
    métrique : sans normalisation, le run part en erreur au milieu du log."""
    flat = mlflow_log._flatten_metrics(RESULTS)

    assert flat["kge_stacking"] == 0.94
    assert "regime_hautes_eaux_stacking" in flat
    assert "saison_ete_stacking" in flat
    assert all(c.isalnum() or c == "_" for c in "".join(flat))


def test_none_values_are_dropped_rather_than_logged(tmp_path):
    """`nan_safe` laisse passer des None dans results.json (KGE indéfini sur
    un régime vide). `log_metrics` exige des floats : un None non filtré fait
    échouer tout l'appel, donc perd aussi les métriques valides."""
    flat = mlflow_log._flatten_metrics({"kge_lgbm": None, "kge_stacking": 0.5})

    assert flat == {"kge_stacking": 0.5}


def test_registry_is_inert_without_a_run_id(monkeypatch):
    """Un entraînement fait sans serveur de suivi a `mlflow_run_id = None`.
    Le promouvoir ensuite ne doit rien tenter d'enregistrer : il n'y a aucun
    run auquel rattacher la version."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")

    assert mlflow_log.register_production_model("apas_G1_G4", 8, None, "tag-v1") is None


def test_registry_is_inert_without_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert mlflow_log.register_production_model("apas_G1_G4", 8, "abc123", "tag-v1") is None


def test_a_dead_registry_never_cancels_a_promotion(monkeypatch):
    """`register_production_model` est appelé APRÈS le commit et le tag git :
    la promotion est déjà actée. S'il propageait, `promote_model` déclencherait
    un rollback sur un modèle pourtant correctement promu."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    mlflow = pytest.importorskip("mlflow")

    def _boom(*a, **k):
        raise ConnectionError("registry injoignable")

    monkeypatch.setattr(mlflow, "register_model", _boom)

    assert mlflow_log.register_production_model("apas_G1_G4", 8, "abc123", "tag-v1") is None


def test_registered_name_is_stable_per_centrale_and_horizon():
    """Le nom est la clé du registry : s'il variait d'un entraînement à
    l'autre, chaque promotion créerait un modèle neuf en v1 au lieu
    d'incrémenter les versions d'un même modèle."""
    assert mlflow_log.registered_name("apas_G1_G4", 8) == "previ-r2d2-apas_G1_G4-h8"
    assert mlflow_log.registered_name("touzac_g2_G2", 48) == "previ-r2d2-touzac_g2_G2-h48"
