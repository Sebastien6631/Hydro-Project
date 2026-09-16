"""Intégrité des DAGs Airflow, sans lancer Airflow.

Tourne là où le paquet `airflow` est installé -- le conteneur airflow
(`docker compose run --rm airflow pytest tests/dags -q`). Ailleurs (env conda
sans Airflow, CI) : skipped, pas rouge -- installer 70 paquets pour 4 asserts
n'a pas de sens.

Ce qu'on protège : un DAG qui ne s'importe plus (le scheduler l'ignore en
silence, l'UI affiche une bannière que personne ne lit), un task_id renommé
(la correspondance avec les stages DVC est le contrat de lecture du README),
un catchup remis à True (rattrapage de 24 prévisions périmées au premier up).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# "airflow.models" et pas "airflow" : le dossier airflow/ du repo est un paquet
# Python vide vu depuis la racine -- importorskip("airflow") passerait à tort.
pytest.importorskip("airflow.models")

from airflow.models import DagBag  # noqa: E402 -- après importorskip

DAGS_DIR = Path(__file__).resolve().parents[2] / "airflow" / "dags"


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS_DIR))  # exemples : AIRFLOW__CORE__LOAD_EXAMPLES=False (compose)


def test_every_dag_imports(dagbag):
    assert dagbag.import_errors == {}, dagbag.import_errors
    assert set(dagbag.dag_ids) == {"hydro_predict", "hydro_train"}


def test_hydro_predict_mirrors_dvc_stages_and_never_catches_up(dagbag):
    dag = dagbag.get_dag("hydro_predict")
    assert dag.catchup is False
    assert set(dag.task_ids) == {"sources.debit", "sources.meteo", "predict_archive"}
    predict = dag.get_task("predict_archive")
    assert {t.task_id for t in predict.upstream_list} == {"sources.debit", "sources.meteo"}


def test_hydro_train_is_linear_weekly_and_never_catches_up(dagbag):
    dag = dagbag.get_dag("hydro_train")
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert dag.timetable.expression == "0 2 * * 1"
    assert dag.task_ids == ["data_preparation", "validate", "train"]
    assert [t.task_id for t in dag.get_task("train").upstream_list] == ["validate"]
    assert [t.task_id for t in dag.get_task("validate").upstream_list] == ["data_preparation"]
