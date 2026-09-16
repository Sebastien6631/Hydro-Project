"""Réentraînement hebdomadaire h8 : data_preparation -> validate -> train.

Mêmes stages que DVC (dvc/preprocessing : data_preparation, validate ;
dvc/model : train), groupés par cadence. Scripts, pas `dvc repro` -- un repro
réécrit dvc.lock, salit l'arbre git et `promote_model` refuse alors de
committer.

Airflow ne fait que sonner : `train.py` décide seul s'il y a du travail
(is_eligible_for_training : 365 j d'historique, 7 j depuis le dernier
modèle -- RETRAIN_INTERVAL_DAYS, aligné sur ce schedule le 15/09). Si rien
n'est éligible il répond en quelques secondes ; la règle reste dans le code
testé, pas dupliquée dans un cron Airflow.

`--promote` commite et tague en local si le KGE bat la production. Pas de
tâche `push` ici : décision en attente (repo bind-monté ou clone dédié,
token, cf. skill hydro-mlops §Propositions). Un humain pousse.

Pas de retry : relancer 3 h de calcul à l'aveugle n'a pas de sens ; un échec
ici se lit dans les logs et se corrige.
"""

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="hydro_train",
    schedule="0 2 * * 1",  # lundi 02:00 Europe/Paris, hors des heures où hydro_predict compte
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,  # un entraînement ne doit jamais se doubler
    default_args={"cwd": "/app", "retries": 0},
    tags=["hydro", "maintain"],
) as dag:
    data_preparation = BashOperator(
        task_id="data_preparation",
        bash_command="python cron/scripts/build-data-preparation.py",
    )
    validate = BashOperator(
        task_id="validate",
        # --strict : un warning de contrat bloque 3 h de calcul, pas une prévision
        # d'une minute -- hydro_predict, lui, ne valide pas.
        bash_command="python cron/scripts/validate-data.py --strict",
    )
    train = BashOperator(
        task_id="train",
        bash_command="python cron/scripts/train.py --promote",
        execution_timeout=timedelta(hours=6),  # CPU : ~3 h pour 2 centrales, marge x2
    )

    data_preparation >> validate >> train
