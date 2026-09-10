from __future__ import annotations

import pandas as pd
import pytest

from previ_r2d2.preprocessing.meteo import open_meteo


def _reponse(times, temps_c, precip_mm, elevation=943.0):
    return {
        "elevation": elevation,
        "hourly": {"time": times, "temperature_2m": temps_c, "precipitation": precip_mm},
    }


def test_temperature_is_converted_to_kelvin(monkeypatch):
    """Les fichiers ECMWF donnaient des Kelvin et tout l'aval en dépend
    (`et0.py` fait `t - 273.15`). L'API rend des °C : servir des °C bruts
    produirait un modèle silencieusement faux, pas une erreur."""
    monkeypatch.setattr(open_meteo, "_get", lambda url, params: _reponse(
        ["2026-09-01T00:00", "2026-09-01T01:00"], [10.0, 12.5], [0.0, 0.0]))

    df = open_meteo.read_points([{"id": 1, "lat": 42.9, "lon": 0.4}],
                                pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"))

    assert df["temperature_S1"].tolist() == pytest.approx([283.15, 285.65])


def test_precipitation_is_cumulated_because_downstream_decumulates(monkeypatch):
    """`snow.partition_precipitation` fait `.diff(1).clip(lower=0)`. Rendre des
    incréments bruts ferait dériver une dérivée : signal plausible, physiquement
    absurde. La décumulation en aval doit retrouver l'incrément d'origine."""
    increments = [0.0, 2.0, 0.0, 3.5]
    monkeypatch.setattr(open_meteo, "_get", lambda url, params: _reponse(
        ["2026-09-01T00:00", "2026-09-01T01:00", "2026-09-01T02:00", "2026-09-01T03:00"],
        [10.0] * 4, increments))

    df = open_meteo.read_points([{"id": 1, "lat": 42.9, "lon": 0.4}],
                                pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"))

    assert df["precipitation_S1"].is_monotonic_increasing
    redecumule = df["precipitation_S1"].diff(1).clip(lower=0).fillna(0)
    assert redecumule.tolist() == pytest.approx(increments)


def test_point_elevation_is_exposed_for_the_altitude_correction(monkeypatch):
    """`snow.bv_temperature` corrige la température 2 m de l'écart d'altitude
    point -> bassin : sans altitude du point, la correction est impossible."""
    monkeypatch.setattr(open_meteo, "_get", lambda url, params: _reponse(
        ["2026-09-01T00:00"], [10.0], [0.0], elevation=1234.0))

    df = open_meteo.read_points([{"id": 3, "lat": 42.9, "lon": 0.4}],
                                pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"))

    assert df["altitude_S3"].tolist() == [1234.0]
    assert {"latitude_S3", "longitude_S3"} <= set(df.columns)


def test_a_failing_point_is_skipped_without_losing_the_others(monkeypatch):
    """Une centrale a 7 points : un seul en échec (réseau, quota) ne doit pas
    faire tomber toute la préparation de données."""
    def _get(url, params):
        if params["latitude"] == 42.9:
            raise RuntimeError("503")
        return _reponse(["2026-09-01T00:00"], [10.0], [0.0])

    monkeypatch.setattr(open_meteo, "_get", _get)

    df = open_meteo.read_points(
        [{"id": 1, "lat": 42.9, "lon": 0.4}, {"id": 2, "lat": 43.0, "lon": 0.6}],
        pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"))

    assert not any(c.endswith("_S1") for c in df.columns)
    assert "temperature_S2" in df.columns


def test_all_points_failing_returns_empty_rather_than_a_broken_frame(monkeypatch):
    monkeypatch.setattr(open_meteo, "_get", lambda url, params: (_ for _ in ()).throw(RuntimeError("503")))

    df = open_meteo.read_points([{"id": 1, "lat": 42.9, "lon": 0.4}],
                                pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"))

    assert df.empty


def test_forecast_wins_over_archive_on_the_overlap(monkeypatch):
    """Les deux points d'accès se recouvrent ; la prévision est la donnée la
    plus fraîche et doit primer, sinon on servirait de l'ERA5 périmé."""
    def _get(url, params):
        temp = 1.0 if url == open_meteo.ARCHIVE_URL else 2.0
        return _reponse(["2026-09-01T00:00", "2026-09-01T01:00"], [temp, temp], [0.0, 0.0])

    monkeypatch.setattr(open_meteo, "_get", _get)
    monkeypatch.setattr(open_meteo, "ARCHIVE_LAG_DAYS", -3650)  # force la branche archive

    df = open_meteo.read_points([{"id": 1, "lat": 42.9, "lon": 0.4}],
                                pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"))

    assert df["temperature_S1"].tolist() == pytest.approx([275.15, 275.15])  # 2 °C = prévision
