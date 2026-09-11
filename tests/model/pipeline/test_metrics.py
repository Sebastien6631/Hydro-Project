from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from projet_hydro.model.pipeline.metrics import kge_per_step, kge_by_regime, kge_by_season, log_interpretability


def test_kge_per_step_matches_hand_computed_values_per_horizon():
    rng = np.random.default_rng(0)
    n = 30
    y_true_log = np.column_stack([np.log1p(10 + np.arange(n, dtype=float)), np.log1p(10 + np.arange(n, dtype=float) * 0.5)])
    y_pred_log = y_true_log + rng.normal(0, 0.01, y_true_log.shape)

    result = kge_per_step(y_true_log, y_pred_log, horizon_steps=2)

    assert len(result) == 2
    assert result[0]["kge"] == pytest.approx(0.9963903541806239)
    assert result[0]["rmse_m3s"] == pytest.approx(0.2623137132499621)
    assert result[1]["kge"] == pytest.approx(0.9924114788752442)
    assert result[1]["rmse_m3s"] == pytest.approx(0.17086373042545402)


def test_kge_per_step_returns_nan_when_fewer_than_10_valid_samples():
    y_true_log = np.log1p(np.full((5, 1), 10.0))
    y_pred_log = y_true_log.copy()

    result = kge_per_step(y_true_log, y_pred_log, horizon_steps=1)

    assert np.isnan(result[0]["kge"])
    assert np.isnan(result[0]["rmse_m3s"])


def make_regime_fixture():
    rng = np.random.default_rng(0)
    n = 120
    y_true_m3s = np.concatenate([1.0 + np.arange(40) * 0.01, 5.0 + np.arange(40) * 0.01, 20.0 + np.arange(40) * 0.01])
    preds = {"model_a": y_true_m3s + rng.normal(0, 0.05, n)}
    return y_true_m3s, preds


def test_kge_by_regime_splits_by_q33_q90_thresholds():
    y_true_m3s, preds = make_regime_fixture()
    q33 = np.percentile(y_true_m3s, 33)
    q90 = np.percentile(y_true_m3s, 90)

    result = kge_by_regime(y_true_m3s, preds, q33, q90)

    assert result["model_a"]["etiage"]["n"] == 40
    assert result["model_a"]["etiage"]["kge"] == pytest.approx(0.8944182935289208)
    assert result["model_a"]["normal"]["n"] == 68
    assert result["model_a"]["normal"]["kge"] == pytest.approx(0.9986257417555016)
    assert result["model_a"]["crue"]["n"] == 12
    assert result["model_a"]["crue"]["kge"] == pytest.approx(-0.2119775984554304)


def test_kge_by_season_splits_by_calendar_month_and_nan_when_uncovered():
    y_true_m3s, preds = make_regime_fixture()
    dates = pd.date_range("2026-01-01", periods=120, freq="1D")  # ne couvre que janvier-avril

    result = kge_by_season(y_true_m3s, preds, dates)

    assert result["model_a"]["hiver"]["n"] == 59
    assert result["model_a"]["hiver"]["kge"] == pytest.approx(0.9934461729262238)
    assert result["model_a"]["printemps"]["n"] == 61
    assert result["model_a"]["printemps"]["kge"] == pytest.approx(0.9993125849376602)
    assert np.isnan(result["model_a"]["ete"]["kge"])
    assert result["model_a"]["ete"]["n"] == 0


def make_interpretability_fixture():
    rng = np.random.default_rng(0)
    n, horizon = 200, 2
    meteo_cols = ["precipitation_S1"]
    n_feat = 2 * horizon + 6 + len(meteo_cols)
    X_meta = rng.normal(0, 1, (n, n_feat))
    y_meta = rng.normal(1, 0.1, (n, horizon))
    meta = Ridge(alpha=1.0).fit(X_meta, y_meta)

    y_true_m3s = 10 + np.cumsum(rng.normal(0, 0.1, n))
    pl_v = y_true_m3s + rng.normal(0, 0.5, n)
    pt_v = y_true_m3s + rng.normal(0, 0.5, n)
    stk_v = y_true_m3s + rng.normal(0, 0.3, n)
    q_now_m3s = y_true_m3s + rng.normal(0, 0.2, n)
    return meta, y_true_m3s, pl_v, pt_v, stk_v, q_now_m3s, horizon, meteo_cols


def test_log_interpretability_includes_ridge_coef_when_meta_has_coef_attribute():
    meta, y_true_m3s, pl_v, pt_v, stk_v, q_now_m3s, horizon, meteo_cols = make_interpretability_fixture()

    report = log_interpretability(meta, y_true_m3s, pl_v, pt_v, stk_v, q_now_m3s, horizon, meteo_cols)

    assert set(report["ridge_coef"].keys()) == {"lgbm", "lstm", "ancre", "contexte", "meteo"}
    assert set(report.keys()) == {"ridge_coef", "skill_vs_persistence", "error_distribution", "crue_quality"}


def test_log_interpretability_omits_ridge_coef_when_meta_has_no_coef_attribute():
    meta, y_true_m3s, pl_v, pt_v, stk_v, q_now_m3s, horizon, meteo_cols = make_interpretability_fixture()

    class NoCoefModel:
        pass

    report = log_interpretability(NoCoefModel(), y_true_m3s, pl_v, pt_v, stk_v, q_now_m3s, horizon, meteo_cols)

    assert "ridge_coef" not in report


def test_log_interpretability_skill_vs_persistence_matches_independent_recomputation():
    from projet_hydro.model.architectures.stacking import kge_components

    meta, y_true_m3s, pl_v, pt_v, stk_v, q_now_m3s, horizon, meteo_cols = make_interpretability_fixture()

    report = log_interpretability(meta, y_true_m3s, pl_v, pt_v, stk_v, q_now_m3s, horizon, meteo_cols)

    mask_p = ~np.isnan(q_now_m3s) & ~np.isnan(y_true_m3s)
    yt, qp = y_true_m3s[mask_p], q_now_m3s[mask_p]
    expected_kge = kge_components(yt, qp)["kge"]
    expected_mse = float(np.mean((yt - qp) ** 2))

    assert report["skill_vs_persistence"]["persistance"]["kge"] == pytest.approx(round(expected_kge, 4))
    assert report["skill_vs_persistence"]["persistance"]["mse"] == pytest.approx(round(expected_mse, 4))
    assert set(report["error_distribution"].keys()) == {"LightGBM seul", "BiLSTM seul", "Stacking"}
    assert set(report["crue_quality"].keys()) == {"LightGBM seul", "BiLSTM seul", "Stacking"}
