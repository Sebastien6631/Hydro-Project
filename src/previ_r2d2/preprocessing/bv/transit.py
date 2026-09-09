"""Temps de transit hydraulique par cross-corrélation saisonnière.

Port de Previ_v2 apps/compute_transit_amont.py::_cross_corr_lag et des deux
analyses (amont -> référence, référence -> centrale). Une saison sans assez
de points est simplement omise. `transit_amont_reference` (débit -> débit)
n'a pas de seuil de corrélation ni de repli physique (pas de vitesse
d'écoulement en dur). `transit_reference_centrale` (débit -> puissance) omet
en plus une saison dont la corrélation directe est trop faible, mais dispose
alors d'un repli géométrique (`estimate_transit_centrale_geometric`) basé sur
un ratio de distances à vol d'oiseau et le `transit_vers_reference_h` déjà
mesuré des stations amont.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import pandas as pd

SAISONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
MAX_LAG_H = 48
MIN_POINTS_TOTAL = 200
MIN_POINTS_SAISON_AMONT = 100
MIN_POINTS_SAISON_CENTRALE = 50
CORR_THRESHOLD_CENTRALE = 0.9  # confirmé par la pratique manuelle de Previ_v2 (compute_decalage_station.py)
PUISSANCE_MARCHE_KW = 10
EARTH_RADIUS_KM = 6371.0
MIN_DISTANCE_KM = 1.0

# À partir de cette année (2026), le service R2 (réserve secondaire de fréquence)
# est activé sur les centrales : power_output reflète alors le dispatch
# réseau demandé, plus le potentiel hydraulique réel -- ces années sont donc
# exclues du calcul de corrélation débit -> puissance. N'affecte PAS
# transit_amont_reference (débit -> débit, non concerné par la R2).
ANNEE_R2_ACTIVE = 2026


def cross_corr_lag(x: np.ndarray, y: np.ndarray, max_lag: int = MAX_LAG_H) -> tuple[int, float]:
    """(lag_optimal_h, correlation_max) ; lag positif = x précède y."""
    corrs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            c = float(np.corrcoef(x, y)[0, 1])
        else:
            c = float(np.corrcoef(x[:-lag], y[lag:])[0, 1])
        corrs.append(c)
    best = int(np.argmax(corrs))
    return best, corrs[best]


def _seasonal_lags(merged: pd.DataFrame, x_col: str, y_col: str, *,
                    min_saison: int, corr_threshold: float | None,
                    filter_y_gt: float | None = None) -> dict[str, int]:
    lags: dict[str, int] = {}
    for saison, mois in SAISONS.items():
        sub = merged[merged.index.month.isin(mois)]
        if filter_y_gt is not None:
            sub = sub[sub[y_col] > filter_y_gt]
        if len(sub) < min_saison:
            continue
        x = (sub[x_col].values - sub[x_col].mean()) / (sub[x_col].std() + 1e-9)
        y = (sub[y_col].values - sub[y_col].mean()) / (sub[y_col].std() + 1e-9)
        lag, corr = cross_corr_lag(x, y)
        if corr_threshold is not None and corr <= corr_threshold:
            continue
        lags[saison] = lag
    return lags


def transit_amont_reference(df_amont: pd.DataFrame, df_reference: pd.DataFrame) -> dict[str, int]:
    """Cross-corrélation débit amont -> débit référence, par saison.

    `df_amont`/`df_reference` : CSV débit local (index = date, colonne
    `Valeur (en m³/s)`). Pas de seuil de corrélation (comme Previ_v2) -- une
    saison est écrite dès qu'elle a assez de points.
    """
    merged = pd.merge(
        df_amont[["Valeur (en m³/s)"]].rename(columns={"Valeur (en m³/s)": "amont"}),
        df_reference[["Valeur (en m³/s)"]].rename(columns={"Valeur (en m³/s)": "reference"}),
        left_index=True, right_index=True, how="inner",
    ).dropna()
    if len(merged) < MIN_POINTS_TOTAL:
        return {}
    return _seasonal_lags(merged, "amont", "reference",
                           min_saison=MIN_POINTS_SAISON_AMONT, corr_threshold=None)


def transit_reference_centrale(df_reference: pd.DataFrame, df_puissance: pd.DataFrame) -> dict[str, int]:
    """Cross-corrélation débit référence -> power_output, par saison.

    `df_puissance` : censé provenir de `puissance_horaire.csv` (invariant
    attendu par cette fonction : déjà nettoyé et déjà horaire) -- le
    `.resample("h").mean()` ci-dessous est un no-op défensif sur une entrée
    déjà horaire, conservé pour ne pas retoucher une fonction déjà testée.
    Les années >= `ANNEE_R2_ACTIVE`
    sont exclues (power_output non représentatif depuis l'activation de la R2).
    Saison omise si corrélation <= 0.9 (seuil élevé -- même signal bruyant que
    Previ_v2, y compris après nettoyage étage 2, un lag en butée de
    `MAX_LAG_H` étant physiquement incohérent d'une saison à l'autre) ou trop
    peu de points en marche (`power_output` > 10 kW). Repli géométrique
    disponible : `estimate_transit_centrale_geometric`.
    """
    puissance_h = df_puissance[["power_output"]].resample("h").mean()
    merged = pd.merge(
        df_reference[["Valeur (en m³/s)"]].rename(columns={"Valeur (en m³/s)": "reference"}),
        puissance_h, left_index=True, right_index=True, how="inner",
    ).dropna()
    merged = merged[merged.index.year < ANNEE_R2_ACTIVE]
    if len(merged) < MIN_POINTS_TOTAL:
        return {}
    return _seasonal_lags(merged, "reference", "power_output",
                           min_saison=MIN_POINTS_SAISON_CENTRALE,
                           corr_threshold=CORR_THRESHOLD_CENTRALE,
                           filter_y_gt=PUISSANCE_MARCHE_KW)


def load_debit_series(path: Path) -> pd.DataFrame:
    """Charge un CSV débit local (`Date (TU);Valeur (en m³/s)`), indexé par date."""
    df = pd.read_csv(path, sep=";", parse_dates=["Date (TU)"])
    df = df.set_index("Date (TU)").sort_index()
    # pandas infère un index tz-aware depuis le suffixe "Z" -- à retirer pour
    # rester compatible avec l'index tz-naive de load_puissance_series (un
    # merge tz-aware/tz-naive lève TypeError).
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_puissance_series(path: Path) -> pd.DataFrame:
    """Charge un CSV `Date;power_output` local (ex. `puissance_horaire.csv`), indexé par date."""
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    return df.set_index("Date").sort_index()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance à vol d'oiseau (km) entre deux points GPS."""
    from math import asin, cos, radians, sin, sqrt

    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def estimate_transit_centrale_geometric(
    stations_amont: list[dict], station_reference: dict, centrale_lat: float, centrale_lon: float,
) -> dict[str, float]:
    """Estime `transit_vers_centrale_h` par saison à partir de `transit_vers_reference_h`
    déjà mesuré (débit-débit, fiable), par ratio de distances à vol d'oiseau :
    T2 = T1 * D(référence, centrale) / D(amont, référence). La sinuosité,
    supposée identique sur les deux segments, s'annule dans ce ratio -- aucune
    vitesse ni sinuosité inventée. Médiane entre stations amont (robuste à une
    station très proche de la référence, qui donnerait un ratio bruité).
    """
    d_ref_centrale = haversine_km(station_reference["lat"], station_reference["lon"], centrale_lat, centrale_lon)
    par_saison: dict[str, list[float]] = {}
    for amont in stations_amont:
        lags = amont.get("transit_vers_reference_h") or {}
        d_amont_ref = haversine_km(amont["lat"], amont["lon"], station_reference["lat"], station_reference["lon"])
        # Une station trop proche de la référence produit un ratio de distances
        # bruité/démesuré (division par un dénominateur proche de zéro).
        if d_amont_ref < MIN_DISTANCE_KM:
            continue
        for saison, t1 in lags.items():
            par_saison.setdefault(saison, []).append(t1 * d_ref_centrale / d_amont_ref)
    return {saison: round(statistics.median(valeurs), 1) for saison, valeurs in par_saison.items()}
