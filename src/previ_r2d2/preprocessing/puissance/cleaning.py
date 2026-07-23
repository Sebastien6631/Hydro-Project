"""Nettoyage et ré-échantillonnage horaire de `power_output` (étage 2 Previ_v2).

Port de `Previ_v2/src/previ_project/core/data_manager.py::DataManager.load_data`
(lignes 202-262) — uniquement la portion détection de chaos + interpolation +
ré-échantillonnage. Le reste de `DataManager` (config météo, filtres
`manual_period`/`data_start_date`, split train/test) n'est pas repris ici ;
ce sera traité au moment du portage complet du pilier `model/`.

Constantes portées à l'identique (même source hydrospot_stream, même échelle
kW, même granularité minute que Previ_v2 -- pas de recalibrage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEUIL_CHOC = 100
SEUIL_NERVOSITE = 100
SEUIL_STABLE = 80
TUNNEL_TENDANCE = 0.20
FENETRE_TENDANCE_MIN = 24 * 60
FENETRE_NERVOSITE_MIN = 60 * 3
FENETRE_PENTE_MIN = 10
FENETRE_RECHERCHE_RETOUR_H = 24
LIMITE_INTERPOLATION_MIN = 60 * 24 * 5


def detect_and_nan_chaos_zones(series: pd.Series) -> pd.Series:
    """Repère les zones de chaos (pics, tremblement, décrochage de tendance) et
    les remplace par NaN, prêtes à être interpolées.

    Port fidèle de la boucle `data_manager.py` : `tendance_globale` (médiane
    glissante centrée), `nervosite` (écart-type glissant centrée), `pente`
    (diff absolue) définissent le masque de chaos ; pour chaque zone détectée,
    recherche sur 24h le premier point de retour au calme (dans ±20 % de la
    tendance locale, variation < 50, nervosité locale -- non centrée -- < 80).
    """
    data = series.copy()
    tendance_globale = data.rolling(window=FENETRE_TENDANCE_MIN, center=True, min_periods=1).median()
    nervosite = data.rolling(window=FENETRE_NERVOSITE_MIN, center=True).std()
    pente = data.diff(periods=FENETRE_PENTE_MIN).abs()
    masque_chaos = (
        (pente >= SEUIL_CHOC)
        | (nervosite > SEUIL_NERVOSITE)
        | (np.abs(data - tendance_globale) > tendance_globale * 0.25)
    )
    seuils_chocs = data.index[masque_chaos.fillna(False)].tolist()

    if not seuils_chocs:
        return data

    last_processed_end = data.index[0]
    for start_time in seuils_chocs:
        if start_time < last_processed_end:
            continue

        end_search = start_time + pd.Timedelta(hours=FENETRE_RECHERCHE_RETOUR_H)
        search_zone = data.loc[start_time:end_search]
        tendance_locale = tendance_globale.loc[start_time:end_search]
        if len(search_zone) < 2:
            continue

        vals = search_zone.values
        tend = tendance_locale.values
        diff_abs = np.abs(np.diff(vals, prepend=vals[0]))
        nervosite_locale = search_zone.rolling(window=FENETRE_NERVOSITE_MIN).std().values

        condition_retour = (
            (vals > tend * (1 - TUNNEL_TENDANCE))
            & (vals < tend * (1 + TUNNEL_TENDANCE))
            & (diff_abs < 50)
            & (nervosite_locale < SEUIL_STABLE)
        )
        indices_reprise = np.where(condition_retour)[0]
        if len(indices_reprise) > 0:
            end_time = search_zone.index[indices_reprise[0]]
            data.loc[start_time:end_time] = np.nan
            last_processed_end = end_time

    return data


def clean_and_resample_hourly(df: pd.DataFrame, power_column: str = "power_output") -> pd.DataFrame:
    """Nettoie `power_column` (chaos -> NaN -> interpolation) puis ré-échantillonne
    à l'heure (médiane, comme Previ_v2 -- pas la moyenne).

    `df` doit être indexé par date, à la granularité minute (comme
    `puissance.csv`). Renvoie un DataFrame indexé par date horaire.
    """
    cleaned = detect_and_nan_chaos_zones(df[power_column])
    cleaned = cleaned.interpolate(method="linear", limit=LIMITE_INTERPOLATION_MIN, limit_direction="both")
    hourly = cleaned.to_frame(power_column).resample("h").median()
    hourly[power_column] = hourly[power_column].round(2)
    return hourly
