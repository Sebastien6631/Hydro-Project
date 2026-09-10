from __future__ import annotations

import pandas as pd
import pytest

from previ_r2d2.model.features.snow import partition_precipitation, snow_melt


def test_partition_precipitation_splits_on_bv_temperature():
    index = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"])
    df = pd.DataFrame(
        {
            # cumul journalier NWP -> diff donne l'incrément horaire
            "precipitation_S1": [0.0, 2.0, 4.0],
            # altitude_S1 = altitude_bv => t_moyen vaut la temperature en °C.
            # Positive au pas 1 -> pluie ; negative au pas 2 -> neige.
            "temperature_S1": [273.15 + 3.25, 273.15 + 3.25, 273.15 - 3.25],
            "altitude_S1": 1000.0,
        },
        index=index,
    )

    pluie_liquide, neige, pluie_sol, precip_inc = partition_precipitation(df, "_S1", altitude_bv=1000)

    # Ligne 0 : NaN du .diff() initial -- préservé côté pluie (where(cond=True) garde l'original), remplacé par 0 côté neige/pluie_sol (where(cond=False) substitue 0).
    assert pd.isna(pluie_liquide.iloc[0])
    assert neige.iloc[0] == pytest.approx(0.0)
    assert pluie_sol.iloc[0] == pytest.approx(0.0)
    # Lignes 1-2 : comportement normal.
    assert pluie_liquide.iloc[1:].tolist() == pytest.approx([2.0, 0.0])
    assert neige.iloc[1:].tolist() == pytest.approx([0.0, 2.0])
    # pluie_sol = pluie_liquide (si >= 1.0 mm/h) * 0.9, sinon 0
    assert pluie_sol.iloc[1:].tolist() == pytest.approx([1.8, 0.0])
    # precip_inc = decumul brut (avant partition pluie/neige), NaN au premier pas
    assert pd.isna(precip_inc.iloc[0])
    assert precip_inc.iloc[1:].tolist() == pytest.approx([2.0, 2.0])


def test_snow_melt_accumulates_then_melts():
    # 3 pas : chute de neige (2mm), pas de fonte (T très froide) ; puis
    # T remonte au-dessus du seuil -> fonte progressive du stock.
    index = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"])
    # altitude_S1 = altitude_bv => t_moyen = temperature en °C : très négatif aux pas 0-1, positif au pas 2.
    df = pd.DataFrame(
        {"temperature_S1": [273.15 - 39.0, 273.15 - 39.0, 273.15 + 6.5], "altitude_S1": 1000.0},
        index=index,
    )
    neige = pd.Series([2.0, 0.0, 0.0], index=index)
    pluie_sol = pd.Series([0.0, 0.0, 0.0], index=index)

    fonte_series, stock_neige_list, t_moyen = snow_melt(
        df, "_S1", neige, pluie_sol, altitude_bv=1000, k_base=1.0, exposition=1.0
    )

    # Pas 0-1 : T_moyen très négatif -> fonte_potentielle nulle -> stock s'accumule (avec décroissance ×0.998)
    assert stock_neige_list[0] == pytest.approx(2.0 * 0.998)
    assert fonte_series.iloc[0] == pytest.approx(0.0)
    # Pas 2 : T_moyen positif -> fonte possible, bornée au stock disponible
    assert fonte_series.iloc[2] >= 0.0
    assert stock_neige_list[2] >= 0.0
    assert len(t_moyen) == 3


def test_snow_melt_pluie_sur_neige_accelerates_fonte():
    # Pas 0 : chute de neige (5mm) par grand froid -> accumulation. Pas 1 : T_moyen positif avec pluie sur neige -> terme pluie-sur-neige actif.
    index = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 01:00"])
    df = pd.DataFrame(
        {"temperature_S1": [273.15 - 39.0, 273.15 + 9.997], "altitude_S1": 1000.0},
        index=index,
    )
    neige = pd.Series([5.0, 0.0], index=index)
    pluie_sol_avec = pd.Series([0.0, 8.0], index=index)
    pluie_sol_sans = pd.Series([0.0, 0.0], index=index)

    fonte_avec, stock_avec, t_moyen = snow_melt(
        df, "_S1", neige, pluie_sol_avec, altitude_bv=1000, k_base=1.0, exposition=1.0
    )
    fonte_sans, stock_sans, _ = snow_melt(
        df, "_S1", neige, pluie_sol_sans, altitude_bv=1000, k_base=1.0, exposition=1.0
    )

    # Le stock accumulé au pas 0 (4.99) borne peu la fonte au pas 1 -- la comparaison isole donc bien le terme pluie-sur-neige.
    terme_pluie_sur_neige = max(0.0, pluie_sol_avec.iloc[1] * t_moyen.iloc[1]) / 80
    assert fonte_avec.iloc[1] - fonte_sans.iloc[1] == pytest.approx(terme_pluie_sur_neige)
    assert fonte_avec.iloc[1] > fonte_sans.iloc[1]
