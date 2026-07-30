from __future__ import annotations

import numpy as np
import pandas as pd
import optuna

from previ_r2d2.model.architectures.lightgbm.training import (
    select_top_features,
    objective_reg,
    fit_oof,
    fit_final,
)


def make_fixture(n=150):
    rng = np.random.default_rng(0)
    index = pd.date_range("2026-01-01", periods=n, freq="1D")
    X = pd.DataFrame(
        {
            "debit_lag_24h": rng.normal(10, 2, n),
            "debit_lag_36h": rng.normal(10, 2, n),
            "max_debit_vu": rng.normal(15, 2, n),
            "Reserve_Nappe_S1": rng.normal(5, 1, n),
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "feat_c": rng.normal(0, 1, n),
        },
        index=index,
    )
    y = pd.Series(10 + np.cumsum(rng.normal(0, 0.2, n)), index=index).clip(lower=1)
    return X, y


def test_select_top_features_includes_forced_and_caps_slow():
    X, y = make_fixture()
    y_log = np.log1p(y.to_numpy())

    result = select_top_features(X, y_log)

    assert set(result) <= set(X.columns)
    for forced in ["debit_lag_24h", "debit_lag_36h", "max_debit_vu"]:
        assert forced in result
    slow_count = sum(1 for c in result if c.startswith("Reserve_Nappe"))
    assert slow_count <= 3


def test_objective_reg_runs_a_real_optuna_trial_and_returns_finite_score():
    X, y = make_fixture()
    y_log_series = pd.Series(np.log1p(y.to_numpy()), index=X.index)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective_reg(trial, X, y_log_series, horizon=1, mult_poids=1.0, timestep="1D"),
        n_trials=1,
    )

    assert np.isfinite(study.best_value)


def test_fit_oof_default_params_shape_and_nan_pattern():
    X, y = make_fixture()

    oof = fit_oof(X, y, horizon=1, mult_poids=1.0, timestep="1D", n_splits=5, n_trials=0)

    assert len(oof) == len(X)
    assert np.sum(~np.isnan(oof)) > 0
    # positions initiales jamais validées par TimeSeriesSplit -> NaN
    assert np.isnan(oof[0])


def test_fit_oof_optuna_path_runs_end_to_end():
    X, y = make_fixture()

    oof = fit_oof(X, y, horizon=1, mult_poids=1.0, timestep="1D", n_splits=5, n_trials=1)

    assert len(oof) == len(X)
    assert np.sum(~np.isnan(oof)) > 0


def test_fit_final_default_params_returns_expected_keys_and_working_model():
    X, y = make_fixture()

    result = fit_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", n_trials=0)

    assert set(result.keys()) == {"model", "top_features", "q_start", "q90", "q99", "n_train"}
    assert result["q_start"] <= result["q90"] <= result["q99"]
    preds = result["model"].predict(X[result["top_features"]])
    assert preds.shape == (len(X),)


def test_fit_final_optuna_path_runs_end_to_end():
    X, y = make_fixture()

    result = fit_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", n_trials=1)

    assert set(result.keys()) == {"model", "top_features", "q_start", "q90", "q99", "n_train"}
    preds = result["model"].predict(X[result["top_features"]])
    assert preds.shape == (len(X),)