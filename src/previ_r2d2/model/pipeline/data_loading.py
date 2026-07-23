"""Chargement des données pour l'entraînement -- load_df délègue à
read_data_preparation_csv (déjà existant), pas de réimplémentation de la
lecture CSV. Pas de fallback modele_source (puissance_config.yaml n'existe
pas encore côté previ-R2-D2, décision utilisateur -- point ouvert connu
pour bonneval_G1)."""

from __future__ import annotations

import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv


def load_df(dossier: str) -> pd.DataFrame:
    """Charge data_preparation.csv pour un dossier ; lève si absent/vide."""
    path = config.CENTRALES_DIR / dossier / "data_preparation.csv"
    df = read_data_preparation_csv(path)
    if df.empty:
        raise FileNotFoundError(f"Aucun data_preparation.csv trouvé pour '{dossier}' ({path})")
    return df


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Rééchantillonnage journalier : precipitation* sommée, le reste moyenné."""
    agg = {col: ("sum" if "precipitation" in col.lower() else "mean") for col in df.columns}
    return df.resample("1D").agg(agg).dropna(subset=["debit_m3s"])


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split positionnel chronologique (train = premiers (1-test_ratio), test = le reste)."""
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]
