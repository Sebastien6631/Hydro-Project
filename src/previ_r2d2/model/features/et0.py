"""ET0 horaire -- port de `feature()` section (a) (Previ_v2
lightgbm_model.py:530-544). Formule interne façon FAO-56 (rayonnement
extraterrestre Ra), calée par un coefficient K_ext dérivé de la température
moyenne de l'année de référence présente dans `df`."""

from __future__ import annotations

import numpy as np
import pandas as pd

GSC = 0.0820      # constante solaire [MJ m⁻² min⁻¹]
LAMBDA_P = 2.45   # chaleur latente de vaporisation [MJ/kg]
K_EXT_SLOPE = 0.0118      # pente de la régression K_ext(température moyenne)
K_EXT_INTERCEPT = 0.21    # ordonnée à l'origine de la régression K_ext(température moyenne)


def compute_et0(df: pd.DataFrame, suffix: str, kc_unit: float) -> pd.Series:
    """ET0 horaire pour la station `suffix` ; k_ext utilise la température moyenne de l'année MAX de `df` (pas une constante, pas une moyenne toutes années confondues)."""
    phi = df[f"latitude{suffix}"] * (np.pi / 180)
    j = df.index.dayofyear
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    delta = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
    ra_heure = (((24 * 60) / np.pi) * GSC * dr * (
        omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )) / 24

    annee_ref = df.index.year.max()
    t_ann = df[df.index.year == annee_ref][f"temperature{suffix}"].mean()
    k_ext = K_EXT_SLOPE * (t_ann - 273.15) + K_EXT_INTERCEPT
    return kc_unit * k_ext * (ra_heure / LAMBDA_P)
