from __future__ import annotations

import numpy as np
import pandas as pd

from projet_hydro.model.architectures.lightgbm.explain import build_shap_explanation, plot_shap_bar, plot_shap_beeswarm
from projet_hydro.model.architectures.lightgbm.training import fit_final


def make_full_df(n_hours=400):
    index = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(0)
    precip_cumul = np.cumsum(rng.uniform(0, 0.5, size=n_hours))
    debit = 10 + np.cumsum(rng.normal(0, 0.1, size=n_hours))
    return pd.DataFrame(
        {
            "debit_m3s": debit,
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0,
            "precipitation_S1": precip_cumul,
            "altitude_S1": 300.0,
        },
        index=index,
    )


EXUTOIRE = {"lat": 43.13, "lon": 0.92}
BV_PARAMS = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}


def make_lgbm_result():
    from projet_hydro.model.architectures.lightgbm.features import build_features

    df = make_full_df()
    X, y, _ = build_features(df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=8, transit_amont={})
    return df, fit_final(X, y, horizon=8, mult_poids=4, timestep="hourly", n_trials=0)


def test_build_shap_explanation_returns_none_when_model_is_none():
    df = make_full_df()
    lgbm_result = {"model": None, "top_features": ["debit_m3s"]}

    shap_values, X_top = build_shap_explanation(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    assert shap_values is None
    assert X_top is None


def test_build_shap_explanation_returns_values_and_features_nominal():
    df, lgbm_result = make_lgbm_result()

    shap_values, X_top = build_shap_explanation(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    assert shap_values is not None
    assert shap_values.shape == X_top.shape
    assert list(X_top.columns) == [f for f in lgbm_result["top_features"] if f in X_top.columns]


def test_plot_shap_beeswarm_creates_nonempty_png(tmp_path):
    df, lgbm_result = make_lgbm_result()
    shap_values, X_top = build_shap_explanation(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    output_path = tmp_path / "beeswarm.png"
    plot_shap_beeswarm(shap_values, X_top, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_shap_bar_creates_nonempty_png(tmp_path):
    df, lgbm_result = make_lgbm_result()
    shap_values, X_top = build_shap_explanation(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    output_path = tmp_path / "bar.png"
    plot_shap_bar(shap_values, X_top, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
