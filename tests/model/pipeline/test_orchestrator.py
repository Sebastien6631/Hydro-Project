from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.model.pipeline.orchestrator import run_training
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv


def make_synthetic_df(n_periods=3000, freq="1h", noise_std=0.2 / 24**0.5):
    """Série horaire synthétique (h8) : 3000 h ~ 125 j, marche aléatoire dont
    l'écart-type est ramené à l'heure (0.2/jour -> 0.2/sqrt(24) par heure)."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=n_periods, freq=freq)
    debit = np.clip(10 + np.cumsum(rng.normal(0, noise_std, n_periods)), 1, None)
    return pd.DataFrame(
        {
            "debit_m3s": debit,
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0 + 5 * np.sin(2 * np.pi * np.arange(n_periods) / (365 * 24)),
            "precipitation_S1": np.cumsum(rng.uniform(0, 2, n_periods)),
            "altitude_S1": 300.0,
        },
        index=index,
    )


def test_run_training_end_to_end_produces_results_and_artifacts(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    centrales_dir = tmp_path / "centrales"
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)

    df = make_synthetic_df()
    write_data_preparation_csv(df, centrales_dir / "test_centrale" / "data_preparation.csv")

    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_json = {
        "bassin_versant": {"altitude_moyenne_m": 300.0, "surface_km2": 100.0},
        "parametres_calage": {"K_base": 1.0, "exposition": 1.0, "kc_unit": 1.0},
        "stations_hydrometriques": [],
    }

    results = run_training(
        "test_centrale", 8, exutoire, bv_json,
        meta_type="ridge", epochs=2, n_trials_lgbm=0, n_trials_final=0,
    )

    assert results["centrale"] == "test_centrale"
    assert results["horizon"] == 8
    assert "kge_stacking" in results

    weights_dir = tmp_path / "weights" / "hybrid" / "test_centrale" / "h8"
    outputs_dir = tmp_path / "outputs" / "hybrid" / "test_centrale" / "h8"
    assert (weights_dir / "results.json").exists()
    assert (weights_dir / "meta_config.json").exists()
    assert (weights_dir / "oof_lgbm.npy").exists()
    assert (weights_dir / "bilstm.pt").exists()
    assert (weights_dir / "meta.pkl").exists()
    assert list(outputs_dir.glob("hybrid_test_*.csv"))

    plots_dir = weights_dir / "plots"
    assert plots_dir.exists()
    assert list(plots_dir.glob("*.png"))
    assert list(outputs_dir.glob("hybrid_test_*.png"))

    ctx = results["_eval_context"]
    assert ctx["X_seq_test"].shape[0] == len(ctx["y_test"])
    assert ctx["df_full_ctx"].shape[0] >= ctx["df_test"].shape[0]
    assert ctx["horizon_steps"] == 8
    assert ctx["exutoire"] == exutoire


def test_run_training_accepts_explicit_weights_dir(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    centrales_dir = tmp_path / "centrales"
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)

    df = make_synthetic_df()
    write_data_preparation_csv(df, centrales_dir / "test_centrale" / "data_preparation.csv")

    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_json = {
        "bassin_versant": {"altitude_moyenne_m": 300.0, "surface_km2": 100.0},
        "parametres_calage": {"K_base": 1.0, "exposition": 1.0, "kc_unit": 1.0},
        "stations_hydrometriques": [],
    }
    custom_dir = tmp_path / "weights" / "hybrid_candidate" / "test_centrale" / "h8"

    run_training(
        "test_centrale", 8, exutoire, bv_json,
        meta_type="ridge", epochs=2, n_trials_lgbm=0, n_trials_final=0,
        weights_dir=custom_dir,
    )

    assert (custom_dir / "meta_config.json").exists()
    assert not (tmp_path / "weights" / "hybrid" / "test_centrale").exists()
