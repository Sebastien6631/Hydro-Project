from __future__ import annotations

import pandas as pd
import pytest

from previ_r2d2.model.features.et0 import compute_et0


def test_compute_et0_matches_hand_computed_value():
    # Valeur calculée indépendamment (script Python séparé, formule ET0 complète) pour lat=43°N, jour 172, T=293.15K, kc_unit=1.0.
    index = pd.DatetimeIndex(["2026-06-21 12:00:00"])
    df = pd.DataFrame(
        {"latitude_S1": [43.0], "temperature_S1": [293.15]}, index=index
    )

    result = compute_et0(df, "_S1", kc_unit=1.0)

    assert result.iloc[0] == pytest.approx(0.31794851615741726)


def test_compute_et0_scales_with_kc_unit():
    index = pd.DatetimeIndex(["2026-06-21 12:00:00"])
    df = pd.DataFrame(
        {"latitude_S1": [43.0], "temperature_S1": [293.15]}, index=index
    )

    result_1 = compute_et0(df, "_S1", kc_unit=1.0)
    result_2 = compute_et0(df, "_S1", kc_unit=2.0)

    assert result_2.iloc[0] == pytest.approx(result_1.iloc[0] * 2)


def test_compute_et0_uses_only_max_year_temperature_for_k_ext():
    # 2 lignes, 2 années différentes, même latitude/jour -- si le calcul utilisait 2025 ou une moyenne des deux années, le résultat différerait pour les deux lignes.
    index = pd.DatetimeIndex(["2025-06-21 12:00:00", "2026-06-21 12:00:00"])
    df = pd.DataFrame(
        {"latitude_S1": [43.0, 43.0], "temperature_S1": [250.0, 293.15]}, index=index
    )

    result = compute_et0(df, "_S1", kc_unit=1.0)

    assert result.iloc[0] == pytest.approx(0.31794851615741726)
    assert result.iloc[1] == pytest.approx(0.31794851615741726)
