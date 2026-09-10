from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from previ_r2d2.model.features.meteo_hydro import (
    compute_meteo_hydro_features,
    station_count,
    windows_for_distance,
)


def test_station_count_detects_max_suffix():
    df = pd.DataFrame({"temperature_S1": [1], "temperature_S3": [1], "autre_col": [1]})

    assert station_count(df) == 3


def test_station_count_zero_when_no_station_columns():
    df = pd.DataFrame({"debit_m3s": [1.0]})

    assert station_count(df) == 0


@pytest.mark.parametrize(
    "distance_km,expected",
    [
        (10, [1, 3, 6, 12, 24, 36, 48, 72]),
        (50, [1, 3, 6, 12, 24, 36, 48, 72]),
        (75, [6, 12, 24, 36, 48, 72]),
        (100, [6, 12, 24, 36, 48, 72]),
        (500, [12, 24, 36, 48, 72, 168]),
        (1000, [12, 24, 36, 48, 72, 168]),
        (2000, [24, 36, 48, 72, 168]),
    ],
)
def test_windows_for_distance(distance_km, expected):
    assert windows_for_distance(distance_km) == expected


def make_df(n_hours=48):
    index = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(0)
    precip_cumul = np.cumsum(rng.uniform(0, 0.5, size=n_hours))
    return pd.DataFrame(
        {
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0,
            "precipitation_S1": precip_cumul,
            "altitude_S1": 300.0,
        },
        index=index,
    )


def test_compute_meteo_hydro_features_adds_apport_liquide_net_column():
    df = make_df()
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    assert "Apport_Liquide_Net_S1" in result.columns
    assert "Pluie_apres_infiltration_S1" in result.columns
    assert "Neige_S1" in result.columns
    assert len(result) == len(df)


def test_compute_meteo_hydro_features_raises_value_error_on_all_nan_latitude():
    df = make_df()
    df["latitude_S1"] = np.nan
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    with pytest.raises(ValueError, match="latitude_S1"):
        compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)


def test_compute_meteo_hydro_features_raises_value_error_on_all_nan_longitude():
    df = make_df()
    df["longitude_S1"] = np.nan
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    with pytest.raises(ValueError, match="longitude_S1"):
        compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)


def test_compute_meteo_hydro_features_temperature_lisse_always_present():
    df = make_df()
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    assert "T_moyen_lisse_24h_S1" in result.columns
    assert "T_moyen_lisse_48h_S1" in result.columns
    assert "T_moyen_diff6_24h_S1" in result.columns
    assert "T_moyen_diff6_48h_S1" in result.columns


def test_compute_meteo_hydro_features_high_altitude_adds_columns_and_rewrites_apport():
    # Plateau froid 24h puis pluie nette : t_lisse_24h reste < -1 pendant que l'apport pre-gel est non nul, sinon la réassignation par facteur_gel_s serait un no-op.
    index = pd.date_range("2026-01-01", periods=48, freq="1h")
    precip_cumul = np.concatenate([np.full(24, 10.0), 10.0 + np.cumsum(np.where(np.arange(24) == 0, 5.0, 0.05))])
    # altitude_S1 = altitude du BV "bas" (300) => t_moyen = temperature en °C ;
    # pour bv_params_high (1500) le meme jeu donne t_moyen - 7.8 °C, soit
    # exactement l'ecart que produisait l'ancien couple (niveau0, altitude_bv).
    temperature = np.concatenate([np.full(24, 273.15 - 1.3), np.full(24, 273.15 + 11.05)])
    df = pd.DataFrame(
        {
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": temperature,
            "precipitation_S1": precip_cumul,
            "altitude_S1": 300.0,
        },
        index=index,
    )
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params_low = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}
    bv_params_high = {**bv_params_low, "altitude_bv": 1500}

    result_low = compute_meteo_hydro_features(df, exutoire, bv_params_low, steps_per_day=24)
    result_high = compute_meteo_hydro_features(df, exutoire, bv_params_high, steps_per_day=24)

    for col in ["Log_Stock_Neige_S1", "Facteur_Gel_S1", "Vitesse_Refroidissement_6h_S1",
                "Interaction_fonte_temperature_S1", "Pluie_sur_neige_S1",
                "Vitesse_Rechauffement_6h_S1", "Fonte_Cumul_24h_S1", "Fonte_Cumul_72h_S1",
                "Fonte_Cumul_168h_S1", "Fonte_Gradient_24h_S1"]:
        assert col not in result_low.columns
        assert col in result_high.columns
    assert "Gel" in result_high.columns
    assert "Gel" not in result_low.columns
    # Apport_Liquide_Net_S1 est réécrasée par le facteur de gel -> valeurs différentes
    assert not result_low["Apport_Liquide_Net_S1"].equals(result_high["Apport_Liquide_Net_S1"])


def test_compute_meteo_hydro_features_high_altitude_match_hand_computed_values():
    # steps_per_day=2 (pas 1) : requis pour que t_lisse_24h (rolling 2) diffère de t_moyen instantané, sinon facteur_gel_s vaut toujours 1.0 quand l'apport pré-gel est non nul (pluie/fonte exigent t_moyen>0 au même pas, donc (t_lisse_24h+3)/2>1 si spd=1) et la réassignation de apport_net_s serait un no-op non testable ; valeurs recalculées indépendamment via pandas/numpy bruts dans un script séparé.
    index = pd.date_range("2026-01-01", periods=5, freq="1h")
    df = pd.DataFrame(
        {
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            # t_moyen identique a l'ancien (niveau0 - 1500) * 0.0065, avec
            # altitude_S1 = altitude_bv : t_moyen vaut la temperature en °C.
            "temperature_S1": [273.15 - 6.5, 273.15 - 6.5, 273.15 + 3.25, 273.15 + 3.25, 273.15 + 3.25],
            "precipitation_S1": [0.0, 3.0, 23.0, 23.0, 40.0],
            "altitude_S1": 1500.0,
        },
        index=index,
    )
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 1500, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 0.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=2)

    assert result["Facteur_Gel_S1"].iloc[2] == pytest.approx(0.324951171875)
    assert result["Fonte_Cumul_24h_S1"].iloc[2] == pytest.approx(0.7338948012010289)
    assert result["Apport_Liquide_Net_S1"].iloc[2] == pytest.approx(6.087601069433244)


def test_compute_meteo_hydro_features_deficit_and_rain_shock_columns():
    df = make_df()
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    for col in ["Indice_Précipitations_Antécédentes_S1", "Deficit_7j_S1", "Deficit_30j_S1",
                "Intensite_Pluie_1h__S1", "Intensite_Pluie_3h__S1", "Rain_Shock__S1"]:
        assert col in result.columns


def test_compute_meteo_hydro_features_deficit_and_indice_match_hand_computed_values():
    # Valeurs calculées indépendamment (script Python séparé, cumsum/pondération manuelle) avant d'être écrites en dur ici.
    index = pd.date_range("2026-01-01", periods=3, freq="1h")
    df = pd.DataFrame(
        {
            "latitude_S1": 43.0,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0,
            "precipitation_S1": [0.0, 5.0, 12.0],
            "altitude_S1": 300.0,
        },
        index=index,
    )
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=1)

    assert result["Deficit_7j_S1"].tolist() == pytest.approx([0.059245587530555865] * 3)
    assert result["Indice_Précipitations_Antécédentes_S1"].tolist() == pytest.approx(
        [0.0, 2.4, 3.8769230769230774]
    )


def test_compute_meteo_hydro_features_transfert_gradients_and_windows_columns():
    df = make_df()
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    for col in ["Apport_Liquide_Net_shifted_S1", "Debit_Theorique_Physique_S1",
                "Gradient_Apport_1h__S1", "Gradient_Apport_3h__S1",
                "Acceleration_Apport_6h__S1", "Acceleration_Apport_S1", "Choc_Hydraulique__S1"]:
        assert col in result.columns
    # distance exutoire<->point ≈ 4.5 km (proche) -> windows = [1,3,6,12,24,36,48,72]
    for rol in [1, 3, 6, 12, 24, 36, 48, 72]:
        assert f"Apport_Liquide_Sum_{rol}_S1" in result.columns
        assert f"Apport_Lisse_shift_{rol}h_S1" in result.columns
        assert f"Debit_m3s_{rol}h_S1" in result.columns
        assert f"Apport_Liquide_Net_passé_{rol}_shift__S1" in result.columns


def test_compute_meteo_hydro_features_transfert_and_windows_match_hand_computed_values():
    # apport_net_s connu par construction (kc_unit=0 -> etp=0, t_moyen>0 partout -> pas de neige/fonte) ; shift/diff/rolling recalculés indépendamment via pandas brut dans un script séparé (transfert_h=1 pour cette distance ≈3.7 km).
    index = pd.date_range("2026-01-01", periods=10, freq="1h")
    df = pd.DataFrame(
        {
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0,
            "precipitation_S1": [0, 2, 2, 3, 6, 6, 8, 12, 12, 18],
            "altitude_S1": 300.0,
        },
        index=index,
    )
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 0.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    assert result["Debit_Theorique_Physique_S1"].iloc[8] == pytest.approx(100.0)
    assert result["Choc_Hydraulique__S1"].iloc[7] == pytest.approx(18.0)
    assert result["Choc_Hydraulique__S1"].iloc[8] == pytest.approx(0.9473684210526315)
    assert result["Debit_m3s_24h_S1"].iloc[9] == pytest.approx(33.333333333333336)


def test_compute_meteo_hydro_features_gel_only_on_first_station(monkeypatch):
    index = pd.date_range("2026-01-01", periods=48, freq="1h")
    rng = np.random.default_rng(1)
    precip_cumul = np.cumsum(rng.uniform(0, 0.5, size=48))
    df = pd.DataFrame(
        {
            "latitude_S1": 43.1, "longitude_S1": 0.9, "temperature_S1": 280.0,
            "precipitation_S1": precip_cumul, "altitude_S1": 300.0,
            "latitude_S2": 43.2, "longitude_S2": 1.0, "temperature_S2": 280.0,
            "precipitation_S2": precip_cumul, "altitude_S2": 300.0,
        },
        index=index,
    )
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 1500, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    assert "Gel" in result.columns
    assert "Facteur_Gel_S1" in result.columns
    assert "Facteur_Gel_S2" in result.columns


def test_compute_meteo_hydro_features_reserves_saturation_reactivite_columns():
    df = make_df()
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    for col in [
        "Reserve_Flash__S1", "Reserve_Court_Terme_S1", "Reserve_Moyen_Terme_S1",
        "Reserve_Nappe_S1", "Ratio_Rapide_Base_S1",
        "Saturation_Index_S1", "Gradient_saturation_Index_168h_S1",
        "Gradient_saturation_Index_48h_S1", "Gradient_saturation_Index_24_S1",
        "Gradient_saturation_Index_6h_S1",
        "Reactivite_12h__S1", "Debit_Mean_6h_S1", "Debit_Mean_24h_S1",
        "Debit_Mean_7j_S1", "Debit_Mean_30j_S1", "Surpression_Relative__S1",
        "Ratio_Fonte_S1",
    ]:
        assert col in result.columns


def test_compute_meteo_hydro_features_reserves_saturation_surpression_match_hand_computed_values():
    # apport_net_s connu par construction (kc_unit=0 -> etp=0, t_moyen>0 partout -> pas de neige/fonte) ; ewm/rolling/expanding recalculés indépendamment via pandas brut dans un script séparé (transfert_h=1 pour cette distance ≈3.7 km, steps_per_day=1 pour que rolling(24h) se réduise à une fenêtre de 1 pas).
    index = pd.date_range("2026-01-01", periods=10, freq="1h")
    df = pd.DataFrame(
        {
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0,
            "precipitation_S1": [0, 2, 2, 3, 6, 6, 8, 12, 12, 18],
            "altitude_S1": 300.0,
        },
        index=index,
    )
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 0.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=1)

    assert result["Ratio_Rapide_Base_S1"].iloc[8] == pytest.approx(1.6862130690113384)
    assert result["Saturation_Index_S1"].iloc[8] == pytest.approx(1.3965071029842526)
    assert result["Surpression_Relative__S1"].iloc[8] == pytest.approx(0.999000999000999)


def test_compute_meteo_hydro_features_weather_lags_absent_by_default():
    df = make_df()
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    for col in result.columns:
        assert not col.startswith("precipitation_lag_")
        assert not col.startswith("temperature_lag_")
        assert not col.startswith("precipitation_cumul_")


def test_compute_meteo_hydro_features_weather_lags_match_hand_computed_values():
    # apport_net_s non pertinent ici (etp=0, pas de neige/fonte) ; precip_inc/lag/cumul
    # recalculés indépendamment via pandas brut dans un script séparé.
    index = pd.date_range("2026-01-01", periods=10, freq="1h")
    df = pd.DataFrame(
        {
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0,
            "precipitation_S1": [0, 2, 2, 3, 6, 6, 8, 12, 12, 18],
            "altitude_S1": 300.0,
        },
        index=index,
    )
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 0.0}

    result = compute_meteo_hydro_features(
        df, exutoire, bv_params, steps_per_day=24, weather_lags=[1, 2]
    )

    assert result["precipitation_lag_1h_S1"].iloc[8] == pytest.approx(4.0)
    assert result["temperature_lag_1h_S1"].iloc[8] == pytest.approx(280.0)
    assert result["precipitation_cumul_1h_S1"].iloc[8] == pytest.approx(0.0)
    assert result["precipitation_lag_2h_S1"].iloc[8] == pytest.approx(2.0)
    assert result["precipitation_cumul_2h_S1"].iloc[8] == pytest.approx(4.0)


def test_compute_meteo_hydro_features_global_aggregates_present_once():
    index = pd.date_range("2026-01-01", periods=48, freq="1h")
    rng = np.random.default_rng(2)
    precip_cumul = np.cumsum(rng.uniform(0, 0.5, size=48))
    df = pd.DataFrame(
        {
            "latitude_S1": 43.1, "longitude_S1": 0.9, "temperature_S1": 280.0,
            "precipitation_S1": precip_cumul, "altitude_S1": 300.0,
            "latitude_S2": 43.2, "longitude_S2": 1.0, "temperature_S2": 280.0,
            "precipitation_S2": precip_cumul, "altitude_S2": 300.0,
        },
        index=index,
    )
    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_params = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}

    result = compute_meteo_hydro_features(df, exutoire, bv_params, steps_per_day=24)

    assert list(result.columns).count("Apport_Bassin_Moyen") == 1
    assert list(result.columns).count("saison_cos") == 1
    assert list(result.columns).count("saison_sin") == 1
    expected_mean = result[["Debit_Theorique_Physique_S1", "Debit_Theorique_Physique_S2"]].mean(axis=1)
    pd.testing.assert_series_equal(
        result["Apport_Bassin_Moyen"], expected_mean, check_names=False
    )


def test_haversine_km_one_degree_latitude():
    """Rapatriée depuis preprocessing/bv/transit.py (supprimé avec la chaîne
    d'onboarding) : elle sert à la distance point météo -> exutoire, qui pilote
    le temps de transfert et les fenêtres d'agrégation."""
    from previ_r2d2.model.features.meteo_hydro import haversine_km

    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.05)
    assert haversine_km(43.13, 0.92, 43.13, 0.92) == pytest.approx(0.0)
