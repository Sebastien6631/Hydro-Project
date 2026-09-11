"""Garde-fous communs à toute la suite."""

import pytest


@pytest.fixture(autouse=True)
def _mlflow_inerte(monkeypatch):
    """La suite ne doit jamais parler à un serveur MLflow, quel que soit le
    contenu du `.env` du développeur (config.py le charge au premier import,
    avant toute fixture). Sans ça, un `.env` avec MLFLOW_TRACKING_URI renseignée
    fait logger de vrais runs `test_centrale-*` sur le serveur à chaque pytest.
    Les tests qui veulent un serveur (tests/tracking) posent la variable
    eux-mêmes APRÈS cette fixture."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
