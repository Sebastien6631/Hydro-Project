from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from previ_r2d2.preprocessing.bv import transit


def test_cross_corr_lag_detects_known_lag():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = np.concatenate([rng.normal(size=5), x[:295]])

    lag, corr = transit.cross_corr_lag(x, y, max_lag=10)

    assert lag == 5
    assert corr > 0.99


def test_transit_amont_reference_returns_lag_for_season_with_enough_data():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-07-01", periods=250, freq="h")
    base = rng.normal(size=260)
    amont_vals = base[0:250]
    reference_vals = np.concatenate([rng.normal(size=4), base[0:246]])
    df_amont = pd.DataFrame({"Valeur (en m³/s)": amont_vals}, index=idx)
    df_reference = pd.DataFrame({"Valeur (en m³/s)": reference_vals}, index=idx)

    lags = transit.transit_amont_reference(df_amont, df_reference)

    assert lags == {"JJA": 4}


def test_transit_amont_reference_returns_empty_when_not_enough_points():
    idx = pd.date_range("2026-07-01", periods=50, freq="h")
    df = pd.DataFrame({"Valeur (en m³/s)": range(50)}, index=idx)

    assert transit.transit_amont_reference(df, df) == {}


def test_transit_reference_centrale_returns_lag_above_threshold():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2025-07-01", periods=250, freq="h")
    base = rng.uniform(20, 100, size=260)
    reference_vals = base[0:250]
    power_vals = np.concatenate([rng.uniform(20, 100, size=2), base[0:248]])
    df_reference = pd.DataFrame({"Valeur (en m³/s)": reference_vals}, index=idx)
    df_puissance = pd.DataFrame({"power_output": power_vals}, index=idx)

    lags = transit.transit_reference_centrale(df_reference, df_puissance)

    assert lags == {"JJA": 2}


def test_transit_reference_centrale_excludes_years_from_2026_onward():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2026-07-01", periods=250, freq="h")
    base = rng.uniform(20, 100, size=260)
    reference_vals = base[0:250]
    power_vals = np.concatenate([rng.uniform(20, 100, size=2), base[0:248]])
    df_reference = pd.DataFrame({"Valeur (en m³/s)": reference_vals}, index=idx)
    df_puissance = pd.DataFrame({"power_output": power_vals}, index=idx)

    lags = transit.transit_reference_centrale(df_reference, df_puissance)

    assert lags == {}


def test_transit_reference_centrale_omits_season_below_correlation_threshold():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-07-01", periods=250, freq="h")
    df_reference = pd.DataFrame({"Valeur (en m³/s)": rng.uniform(20, 100, size=250)}, index=idx)
    df_puissance = pd.DataFrame({"power_output": rng.uniform(20, 100, size=250)}, index=idx)

    lags = transit.transit_reference_centrale(df_reference, df_puissance)

    assert lags == {}


def test_load_debit_series_indexes_by_date(tmp_path):
    path = tmp_path / "O020002001.csv"
    path.write_text(
        "Date (TU);Valeur (en m³/s)\n2026-07-01T00:00:00Z;10.0\n2026-07-01T01:00:00Z;11.0\n",
        encoding="utf-8",
    )

    df = transit.load_debit_series(path)

    assert list(df["Valeur (en m³/s)"]) == [10.0, 11.0]
    assert df.index[0] == pd.Timestamp("2026-07-01T00:00:00Z").tz_localize(None)


def test_load_puissance_series_indexes_by_date(tmp_path):
    path = tmp_path / "puissance.csv"
    path.write_text(
        "Date;Puissance;power_output\n2026-07-01 00:00;5.0;5.0\n2026-07-01 00:01;6.0;6.0\n",
        encoding="utf-8",
    )

    df = transit.load_puissance_series(path)

    assert list(df["power_output"]) == [5.0, 6.0]
    assert df.index[0] == pd.Timestamp("2026-07-01 00:00")


def test_haversine_km_one_degree_latitude():
    d = transit.haversine_km(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111.195, abs=0.001)


def test_estimate_transit_centrale_geometric_uses_median_across_amont_stations():
    station_reference = {"lat": 44.0, "lon": 5.0}
    centrale_lat, centrale_lon = 44.25, 5.0
    stations_amont = [
        {"lat": 44.5, "lon": 5.0, "transit_vers_reference_h": {"JJA": 10}},
        {"lat": 44.4, "lon": 5.0, "transit_vers_reference_h": {"JJA": 8}},
        {"lat": 44.1, "lon": 5.0, "transit_vers_reference_h": {"JJA": 1}},
    ]

    result = transit.estimate_transit_centrale_geometric(
        stations_amont, station_reference, centrale_lat, centrale_lon,
    )

    assert result == {"JJA": 5.0}


def test_estimate_transit_centrale_geometric_averages_multiple_seasons():
    station_reference = {"lat": 44.0, "lon": 5.0}
    centrale_lat, centrale_lon = 44.25, 5.0
    stations_amont = [{"lat": 44.5, "lon": 5.0, "transit_vers_reference_h": {"JJA": 10, "SON": 6}}]

    result = transit.estimate_transit_centrale_geometric(
        stations_amont, station_reference, centrale_lat, centrale_lon,
    )

    assert result == {"JJA": 5.0, "SON": 3.0}


def test_estimate_transit_centrale_geometric_skips_station_too_close_to_reference():
    station_reference = {"lat": 44.0, "lon": 5.0}
    centrale_lat, centrale_lon = 44.25, 5.0
    # ~111 m d'une station de référence à 0.001° de latitude -- trop proche
    # pour un ratio de distances fiable (dénominateur quasi nul).
    trop_proche = {"lat": 44.001, "lon": 5.0, "transit_vers_reference_h": {"JJA": 10}}
    normale = {"lat": 44.5, "lon": 5.0, "transit_vers_reference_h": {"JJA": 10}}

    result = transit.estimate_transit_centrale_geometric(
        [trop_proche, normale], station_reference, centrale_lat, centrale_lon,
    )

    # Seule la station à distance normale contribue (médiane d'une seule
    # valeur) -- si la station trop proche était incluse, le ratio explosif
    # (~2500 au lieu de 5.0) changerait complètement le résultat.
    assert result == {"JJA": 5.0}


def test_estimate_transit_centrale_geometric_skips_stations_without_transit():
    station_reference = {"lat": 44.0, "lon": 5.0}
    stations_amont = [{"lat": 44.5, "lon": 5.0, "transit_vers_reference_h": {}}]

    result = transit.estimate_transit_centrale_geometric(
        stations_amont, station_reference, 44.25, 5.0,
    )

    assert result == {}
