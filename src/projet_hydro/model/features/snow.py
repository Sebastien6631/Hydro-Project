"""Partition pluie/neige + modèle de fonte -- port de `feature()` sections
(b) et (c) (Previ_v2 lightgbm_model.py:548-568)."""

from __future__ import annotations

import numpy as np
import pandas as pd

INTERCEPTION_FACTOR = 0.9
PLUIE_EFFICACE_SEUIL = 1.0   # mm/h
GAMMA_V = 0.0065             # gradient thermique vertical humide [°C/m]
T_SEUIL = 0                  # seuil pluie/neige [°C]
KELVIN = 273.15
STOCK_DECAY = 0.998
PLUIE_SUR_NEIGE_DIVISOR = 80


def bv_temperature(df: pd.DataFrame, suffix: str, altitude_bv: float) -> pd.Series:
    """Température au bassin versant [°C], depuis la température 2 m du point météo
    corrigée de l'écart d'altitude point -> BV.

    Remplace l'ancien `(niveau0 - altitude_bv) * GAMMA_V`, qui DÉDUISAIT la
    température de l'isotherme 0° faute de mieux : les fichiers ECMWF donnaient
    un `2t` à l'altitude du point de grille, inutilisable tel quel. La source
    actuelle fournit la température ET l'altitude du point, donc la correction
    est directe. Sémantique identique (t_moyen > 0 <=> isotherme au-dessus du
    BV), physique plus courte d'un intermédiaire.
    """
    t_point = df[f"temperature{suffix}"] - KELVIN
    altitude_point = df[f"altitude{suffix}"]
    return t_point - (altitude_bv - altitude_point) * GAMMA_V


def partition_precipitation(
    df: pd.DataFrame, suffix: str, altitude_bv: float
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Décumule `precipitation{suffix}` puis partitionne pluie/neige selon la
    température au BV ; renvoie (pluie_liquide, neige, pluie_sol après interception, precip_inc)."""
    precip_inc = df[f"precipitation{suffix}"].diff(1).clip(lower=0)
    t_moyen = bv_temperature(df, suffix, altitude_bv)
    pluie_liquide = precip_inc.where(t_moyen > T_SEUIL, 0)
    neige = precip_inc.where(t_moyen <= T_SEUIL, 0)
    pluie_sol = pluie_liquide.where(pluie_liquide >= PLUIE_EFFICACE_SEUIL, 0) * INTERCEPTION_FACTOR
    return pluie_liquide, neige, pluie_sol, precip_inc


def snow_melt(
    df: pd.DataFrame,
    suffix: str,
    neige: pd.Series,
    pluie_sol: pd.Series,
    altitude_bv: float,
    k_base: float,
    exposition: float,
) -> tuple[pd.Series, list[float], pd.Series]:
    """Fonte différée bornée par le stock disponible ; renvoie (fonte_reelle, stock_neige_list, t_moyen) -- boucle stateful pas à pas, pas vectorisable simplement."""
    j = df.index.dayofyear
    t_moyen = bv_temperature(df, suffix, altitude_bv)
    k_fonte_dynamique = (k_base + np.cos(2 * np.pi * (j - 172) / 365)) * exposition
    fonte_potentielle = ((k_fonte_dynamique / 24) * (t_moyen - T_SEUIL)).clip(lower=0) \
        + (pluie_sol * (t_moyen - T_SEUIL)).clip(lower=0) / PLUIE_SUR_NEIGE_DIVISOR

    fonte_potentielle_arr = fonte_potentielle.to_numpy()
    neige_arr = neige.to_numpy()

    stock_neige = 0.0
    fonte_reelle, stock_neige_list = [], []
    for t in range(len(df)):
        f_reelle = min(fonte_potentielle_arr[t], stock_neige + neige_arr[t])
        stock_neige = max(0.0, (stock_neige + neige_arr[t] - f_reelle) * STOCK_DECAY)
        fonte_reelle.append(f_reelle)
        stock_neige_list.append(stock_neige)
    return pd.Series(fonte_reelle, index=df.index), stock_neige_list, t_moyen
