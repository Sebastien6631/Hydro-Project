from __future__ import annotations

from unittest.mock import patch

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna

from projet_hydro.model.architectures.lightgbm import training
from projet_hydro.model.pipeline.split import train_val_test_indices
from projet_hydro.model.architectures.lightgbm.training import (
    OOF_LGBM_PARAMS,
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


def test_fit_oof_resumes_from_checkpoint_skipping_completed_folds(tmp_path):
    """Portage d'un fix de prod (#7) : un entraînement long tué en cours de
    route ne doit pas repartir de zéro. Un checkpoint marquant le fold 0
    déjà fait doit faire sauter ce fold au prochain appel (vérifié en
    comptant les vrais appels à LGBMRegressor.fit -- 2 au lieu de 3 pour
    n_splits=3)."""
    X, y = make_fixture()
    y_log = np.log1p(y.to_numpy())
    top_cols = select_top_features(X, y_log)

    checkpoint_path = tmp_path / "oof_lgbm.checkpoint.pkl"
    joblib.dump(
        {
            "oof": np.full(len(X), np.nan),
            "top_cols": top_cols,
            "best_params": dict(OOF_LGBM_PARAMS),
            "completed_folds": [0],
        },
        checkpoint_path,
    )

    original_fit = lgb.LGBMRegressor.fit
    calls = []

    def counting_fit(self, *args, **kwargs):
        calls.append(1)
        return original_fit(self, *args, **kwargs)

    with patch.object(lgb.LGBMRegressor, "fit", counting_fit):
        oof = fit_oof(
            X, y, horizon=1, mult_poids=1.0, timestep="1D",
            n_splits=3, n_trials=0, checkpoint_path=checkpoint_path,
        )

    assert len(calls) == 2  # fold 0 sauté (déjà dans le checkpoint), folds 1 et 2 réellement entraînés
    assert len(oof) == len(X)
    assert not checkpoint_path.exists()  # nettoyé une fois tous les folds terminés


def test_fit_oof_with_checkpoint_path_cleans_up_after_normal_run(tmp_path):
    X, y = make_fixture()
    checkpoint_path = tmp_path / "oof_lgbm.checkpoint.pkl"

    oof = fit_oof(
        X, y, horizon=1, mult_poids=1.0, timestep="1D",
        n_splits=3, n_trials=0, checkpoint_path=checkpoint_path,
    )

    assert len(oof) == len(X)
    assert not checkpoint_path.exists()


def test_fit_final_default_params_returns_expected_keys_and_working_model():
    X, y = make_fixture()

    result = fit_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", n_trials=0)

    assert set(result.keys()) == {"model", "top_features", "q_start", "q90", "q99", "n_train", "training_curve"}
    assert result["q_start"] <= result["q90"] <= result["q99"]
    preds = result["model"].predict(X[result["top_features"]])
    assert preds.shape == (len(X),)


def test_fit_final_optuna_path_runs_end_to_end():
    X, y = make_fixture()

    result = fit_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", n_trials=1)

    assert set(result.keys()) == {"model", "top_features", "q_start", "q90", "q99", "n_train", "training_curve"}
    preds = result["model"].predict(X[result["top_features"]])
    assert preds.shape == (len(X),)

def test_fit_final_uses_early_stopping_on_a_separate_val_slice(monkeypatch):
    """Avant ce fix, fit_final fittait sur 100% des lignes sans AUCUN early
    stopping (contrairement à fit_oof) : rien ne bornait le sur-apprentissage."""
    X, y = make_fixture(n=400)
    seen = []
    real_fit = training.lgb.LGBMRegressor.fit

    def spy(self, X_, y_, **kwargs):
        seen.append({"rows": len(X_), "eval_sets": len(kwargs.get("eval_set") or [])})
        return real_fit(self, X_, y_, **kwargs)

    monkeypatch.setattr(training.lgb.LGBMRegressor, "fit", spy)
    result = fit_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", n_trials=0)
    model = result["model"]

    fit_s, _, _ = train_val_test_indices(len(X))
    final_call = seen[-1]  # le dernier fit est le final (le premier vient de select_top_features)
    assert final_call["rows"] == fit_s.stop - fit_s.start < len(X)
    assert final_call["eval_sets"] == 2, "fit + val, pour que la courbe révèle le sur-apprentissage"
    assert model.best_iteration_ is not None
    assert model.best_iteration_ <= model.n_estimators


def test_fit_final_training_curve_has_three_slices_and_n_train_still_counts_all_rows():
    """`n_train` sert à invalider le cache lgbm_final.pkl quand l'historique
    grandit : il doit rester le total de X, pas la seule tranche fit."""
    X, y = make_fixture(n=400)

    result = fit_final(X, y, horizon=1, mult_poids=1.0, timestep="1D", n_trials=0)

    assert set(result["training_curve"]) == {"kge_fit", "kge_val", "kge_test"}
    assert result["n_train"] == len(X)
