"""Prédiction horaire h8 : débit Hub'Eau + météo Open-Meteo -> predict_archive.

Mêmes stages que DVC (`debit` = dvc/preprocessing, `predict_archive` =
dvc/postprocessing), groupés ici par cadence et non par domaine. Les tâches
appellent les SCRIPTS, pas `dvc repro` : chaque repro réécrirait dvc.lock,
salirait l'arbre git et bloquerait la promotion du DAG train.

Pas de `data_preparation` ici : la prédiction en source="live" assemble sa
fenêtre elle-même (build_dossier lit le débit du store local et va chercher la
météo en direct). data_preparation.csv ne sert qu'à l'entraînement.

Les deux sources bloquent (décision du 11/09) : sans débit frais on ne prédit
pas, sans météo build_dossier n'assemble rien. `meteo` (check-meteo.py) existe
parce que read_points ignore les points en échec sans lever -- l'échec doit
être rendu visible AVANT predict_archive, et nommer la centrale.
"""

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, TaskGroup

with DAG(
    dag_id="hydro_predict",
    schedule="5 * * * *",  # h+5 : laisse à Hub'Eau le temps de publier l'heure pleine
    start_date=datetime(2026, 9, 1),
    catchup=False,  # une prévision passée ne sert à rien : jamais de rattrapage
    max_active_runs=1,
    default_args={"cwd": "/app", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["hydro", "serve"],
) as dag:
    with TaskGroup(group_id="sources") as sources:
        BashOperator(task_id="debit", bash_command="python cron/scripts/maj-data.py")
        BashOperator(task_id="meteo", bash_command="python cron/scripts/check-meteo.py")

    predict_archive = BashOperator(
        task_id="predict_archive",
        bash_command="python cron/scripts/predict-archive.py",
        retries=0,  # un échec ici est un bug ou une donnée absente, pas un réseau qui tousse
    )

    sources >> predict_archive
