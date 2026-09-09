from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from previ_r2d2.model.architectures.stacking import kge_components
from previ_r2d2.model.pipeline.split import train_val_test_indices
from previ_r2d2.model.pipeline.stacking_fit import (
    DEFAULT_RIDGE_META_PARAMS,
    PerStepLGBMMeta,
    fit_stacking,
)


def make_fixture(n_lstm=20, n=100, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n, freq="1h")
    df_full = pd.DataFrame(
        {
            "debit_m3s": 10 + np.cumsum(rng.normal(0, 0.1, n)),
            "precipitation_S1": rng.uniform(0, 5, n),
        },
        index=index,
    )
    horizon = 2
    t_last = np.arange(n - n_lstm - horizon, n - horizon)
    oof_lgbm_multi = rng.normal(1, 0.1, (n_lstm, horizon))
    oof_lstm_vals = rng.normal(1, 0.1, (n_lstm, horizon))
    y_train = rng.normal(1, 0.1, (n_lstm, horizon))
    return df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon


def test_fit_stacking_ridge_default_saves_artifacts_and_predicts(tmp_path):
    df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon = make_fixture()

    result = fit_stacking(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon,
        meteo_feature_cols=["precipitation_S1"], output_dir=tmp_path,
    )

    assert isinstance(result["meta"], Ridge)
    assert (tmp_path / "meta.pkl").exists()
    assert (tmp_path / "meta_scaler.pkl").exists()
    assert result["q90_train"] == pytest.approx(np.percentile(y_train[~np.isnan(y_train)], 90))

    preds = result["meta"].predict(result["meta_scaler"].transform(result["meta_X"]))
    assert preds.shape == result["meta_y"].shape


def test_fit_stacking_lgbm_meta_type_uses_per_step_models(tmp_path):
    df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon = make_fixture()

    result = fit_stacking(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon,
        meteo_feature_cols=["precipitation_S1"], output_dir=tmp_path, meta_type="lgbm",
    )

    assert isinstance(result["meta"], PerStepLGBMMeta)
    # un LGBMRegressor par pas d'horizon, chacun avec son propre early stopping
    assert len(result["meta"].models_) == horizon






def test_fit_stacking_returns_meta_x_complete_not_truncated_to_the_fit_slice(tmp_path):
    df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon = make_fixture(n_lstm=200, n=400)

    result = fit_stacking(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon,
        meteo_feature_cols=["precipitation_S1"], output_dir=tmp_path,
    )

    assert len(result["meta_X"]) == int(result["valid"].sum())
    assert len(result["meta_X"]) == len(result["meta_y"])


def test_kge_val_comes_from_the_diagnostic_model_not_from_the_deployed_one(tmp_path):
    """Le Ridge DÉPLOYÉ est fitté sur 100% (comportement d'origine, rétabli après
    l'ablation du 2026-09-09). `kge_val`/`kge_test` doivent donc venir d'un modèle
    jetable fitté sur la seule tranche fit -- sinon ce seraient des mesures en
    échantillon, exactement le défaut que ces clés servent à éviter. Comparaison
    explicite des deux valeurs : un seuil absolu ne détecterait rien, les deux
    pouvant être du même signe."""
    df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon = make_fixture(n_lstm=200, n=400)

    result = fit_stacking(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon,
        meteo_feature_cols=["precipitation_S1"], output_dir=tmp_path,
    )

    meta_X_s = result["meta_scaler"].transform(result["meta_X"])
    _, val_s, _ = train_val_test_indices(len(result["meta_X"]))
    yt = np.expm1(result["meta_y"][val_s]).ravel()
    yp = np.expm1(result["meta"].predict(meta_X_s[val_s])).clip(0).ravel()
    kge_du_modele_deploye = kge_components(yt, yp)["kge"]

    assert result["training_curve"]["kge_val"] < kge_du_modele_deploye


def test_deployed_ridge_is_fit_on_all_rows_with_the_historical_alpha(tmp_path):
    """Garde-fou du repli : calibrer alpha sur une tranche val contiguë a été
    mesuré nuisible (-0.023 KGE, alpha instable de 0.1 à 30.0). Le modèle servi
    doit rester le Ridge historique fitté sur 100%."""
    df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon = make_fixture(n_lstm=200, n=400)

    result = fit_stacking(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon,
        meteo_feature_cols=["precipitation_S1"], output_dir=tmp_path,
    )

    assert result["meta"].alpha == DEFAULT_RIDGE_META_PARAMS["alpha"]
    attendu = Ridge(**DEFAULT_RIDGE_META_PARAMS).fit(
        result["meta_scaler"].transform(result["meta_X"]), result["meta_y"])
    np.testing.assert_allclose(result["meta"].coef_, attendu.coef_)

def test_fit_stacking_training_curve_reports_the_three_slices(tmp_path):
    df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon = make_fixture(n_lstm=200, n=400)

    result = fit_stacking(
        df_full, oof_lgbm_multi, oof_lstm_vals, t_last, y_train, horizon,
        meteo_feature_cols=["precipitation_S1"], output_dir=tmp_path,
    )

    assert set(result["training_curve"]) == {"kge_fit", "kge_val", "kge_test"}
    assert all(isinstance(v, float) for v in result["training_curve"].values())
