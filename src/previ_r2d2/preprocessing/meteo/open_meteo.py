"""Import météo depuis Open-Meteo (modèles Météo-France), en remplacement des
fichiers NWP ECMWF qui n'étaient plus alimentés (FTP retiré).

Rend le MÊME contrat que l'ancien `nwp_reader.read_points` moins `niveau0` :
une colonne par point et par variable, index horaire.

    latitude_S{i}, longitude_S{i}, altitude_S{i}   constantes du point
    temperature_S{i}                               KELVIN (comme l'ECMWF)
    precipitation_S{i}                             mm CUMULÉS

**Précipitations cumulées** : l'API renvoie des incréments horaires, mais
`snow.partition_precipitation` fait `.diff(1).clip(lower=0)`. On cumule donc
avant de rendre, pour que la décumulation en aval retrouve exactement
l'incrément. Rendre des incréments bruts produirait un signal plausible et
silencieusement faux (la dérivée d'un incrément n'a aucun sens physique).

**`niveau0` n'est pas fourni** : aucun modèle Météo-France ne l'expose via
cette API, et l'archive ERA5 non plus, donc il serait impossible de
reconstituer l'historique. Les features utilisent désormais la température
réelle corrigée de l'altitude -- cf. `model/features/snow.py`.

Deux points d'accès complémentaires, **tous deux Météo-France** : l'archive des
prévisions passées (depuis le 2022-11-15) et la prévision courante (92 jours de
passé + les jours à venir). Une fenêtre longue traverse les deux ; la prévision
l'emporte sur le recouvrement, c'est la donnée la plus fraîche.

**Pourquoi pas l'archive ERA5**, qui remonterait pourtant à 1940 : mesuré sur
504 h de recouvrement, elle donne TROIS FOIS plus de pluie que Météo-France
(0.136 vs 0.049 mm/h, corrélation 0.21) pour une température quasi identique
(+0.31 °C, corrélation 0.95). Un modèle entraîné sur l'historique ERA5 puis
servi en prévision Météo-France sous-estimerait donc systématiquement les
crues -- silencieusement, la pluie étant le moteur du modèle. On préfère
3,8 ans homogènes à 5,6 ans discontinus.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# Début de l'archive des prévisions Météo-France chez Open-Meteo (bissecté :
# vide au 2022-11-08, disponible au 2022-11-15).
ARCHIVE_MIN_DATE = pd.Timestamp("2022-11-15")
HOURLY_VARS = "temperature_2m,precipitation"
# Météo-France en priorité (AROME 1.5 km sur la France), repli automatique de
# l'API sur ARPEGE hors couverture AROME.
MODELS = "meteofrance_seamless"
FORECAST_PAST_DAYS_MAX = 92
ARCHIVE_LAG_DAYS = 2  # l'archive des prévisions passées suit le temps réel à ~2 jours
TIMEOUT_S = 60
KELVIN = 273.15


def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Open-Meteo : {data.get('reason', data['error'])}")
    return data


def _frame(data: dict) -> pd.DataFrame:
    """Réponse Open-Meteo -> DataFrame horaire (temperature_2m en °C, precipitation en mm)."""
    hourly = data.get("hourly") or {}
    if not hourly.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame(
        {"temperature_2m": hourly["temperature_2m"], "precipitation": hourly["precipitation"]},
        index=pd.to_datetime(hourly["time"]),
    )
    df.attrs["elevation"] = data.get("elevation")
    return df


def _fetch_point(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Série horaire brute d'un point, archive et prévision recollées."""
    base = {"latitude": lat, "longitude": lon, "hourly": HOURLY_VARS, "timezone": "UTC"}
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    frames, elevation = [], None

    archive_start = max(start, ARCHIVE_MIN_DATE)
    archive_end = min(end, today - pd.Timedelta(days=ARCHIVE_LAG_DAYS))
    if archive_start < archive_end:
        data = _get(ARCHIVE_URL, {
            **base, "models": MODELS,
            "start_date": archive_start.strftime("%Y-%m-%d"),
            "end_date": archive_end.strftime("%Y-%m-%d"),
        })
        df = _frame(data)
        if not df.empty:
            frames.append(df)
            elevation = df.attrs.get("elevation")

    past_days = min(FORECAST_PAST_DAYS_MAX, max(0, (today - start).days) + 1)
    forecast_days = min(16, max(1, (end - today).days + 1))
    data = _get(FORECAST_URL, {
        **base, "models": MODELS, "past_days": past_days, "forecast_days": forecast_days,
    })
    df = _frame(data)
    if not df.empty:
        frames.append(df)
        elevation = df.attrs.get("elevation", elevation)

    if not frames:
        return pd.DataFrame()
    # `keep="last"` : la prévision est concaténée en second, elle gagne le recouvrement.
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.attrs["elevation"] = elevation
    return combined


def read_points(
    points: list[dict], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Série horaire des points météo d'une centrale, format `data_preparation`.

    `points` : items `{"id": int, "lat": float, "lon": float}` (bv.json
    `stations_meteo_nwp`). Un point qui échoue est ignoré avec un avertissement
    plutôt que de faire tomber toute la centrale -- même politique que l'ancien
    lecteur NWP face à un fichier illisible.
    """
    series: dict[str, pd.Series] = {}
    for point in points:
        i = point["id"]
        try:
            raw = _fetch_point(point["lat"], point["lon"], start, end)
        except Exception as exc:
            logger.warning("Point météo %s (%.3f, %.3f) : %s -- ignoré.", i, point["lat"], point["lon"], exc)
            continue
        if raw.empty:
            logger.warning("Point météo %s : aucune donnée sur [%s, %s] -- ignoré.", i, start, end)
            continue

        series[f"latitude_S{i}"] = pd.Series(point["lat"], index=raw.index)
        series[f"longitude_S{i}"] = pd.Series(point["lon"], index=raw.index)
        series[f"altitude_S{i}"] = pd.Series(float(raw.attrs.get("elevation") or 0.0), index=raw.index)
        series[f"temperature_S{i}"] = raw["temperature_2m"] + KELVIN
        series[f"precipitation_S{i}"] = raw["precipitation"].fillna(0).cumsum()

    if not series:
        return pd.DataFrame()
    result = pd.DataFrame(series).sort_index()
    return result[(result.index >= start) & (result.index <= end)]
