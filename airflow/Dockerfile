# Image Airflow = image projet + Airflow. Les taches des DAGs sont des
# BashOperator qui lancent `python cron/scripts/*.py` dans CE conteneur, sur le
# repo bind-monte -- pas de DockerOperator, pas de socket Docker.
# Verifie le 2026-09-11 : Airflow 3.3.1 ne change qu'un paquet de l'env projet
# (fastapi 0.141 -> 0.136), sans effet ici (l'API tourne dans l'image de base).
FROM projet_hydro:latest

ENV AIRFLOW_HOME=/opt/airflow
RUN pip install "apache-airflow==3.3.1"
