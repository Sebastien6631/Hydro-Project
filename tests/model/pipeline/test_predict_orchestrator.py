from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.orchestrator import run_training
from tests.model.pipeline.test_orchestrator import make_synthetic_df
from previ_r2d2.model.pipeline.predict_orchestrator import run_prediction, to_display_timezone
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv


def test_to_display_timezone_converts_utc_to_paris():
    dates = pd.DatetimeIndex(["2026-01-15 10:00:00", "2026-07-15 10:00:00"])

    result = to_display_timezone(dates)

    assert list(result) == [pd.Timestamp("2026-01-15 11:00:00"), pd.Timestamp("2026-07-15 12:00:00")]



def test_run_prediction_h8_writes_prevision_json(tmp_path, monkeypatch):
    centrales_dir = tmp_path / "centrales"
    reference_dir = centrales_dir / "REFERENCE"
    nas_data_root = tmp_path / "nas_data"
    monkeypatch.setattr(config, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(config, "REFERENCE_DIR", reference_dir)
    monkeypatch.setattr(config, "NAS_DATA_ROOT", nas_data_root)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    models_dir = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    reference_dir.mkdir(parents=True)

    # horizon=8 -> timestep hourly (steps_per_day=24) : les fenêtres de
    # features LightGBM (ex. min_periods=spd*30=720 lignes) exigent une
    # série historique réellement horaire, pas seulement 400 points
    # journaliers (sinon dropna élimine toutes les lignes -> X vide).
    # noise_std réduit (échelle sqrt(24) vs le pas journalier par défaut)
    # pour éviter que le random walk ne reste bloqué sur le plancher
    # clip(...,1,None) sur tout le dernier segment (variance nulle ->
    # KGE non défini dans les plots de fin d'entraînement).
    df = make_synthetic_df()
    write_data_preparation_csv(df, centrales_dir / "test_centrale" / "data_preparation.csv")

    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_json = {
        "bassin_versant": {"altitude_moyenne_m": 300.0, "surface_km2": 100.0},
        "parametres_calage": {"K_base": 1.0, "exposition": 1.0, "kc_unit": 1.0},
        "stations_hydrometriques": [],
        "transit_vers_centrale_h": {"DJF": 2, "MAM": 2, "JJA": 2, "SON": 2},
    }
    (centrales_dir / "test_centrale" / "bv.json").write_text(json.dumps(bv_json), encoding="utf-8")

    records = [{
        "dossier": "test_centrale", "flex_strategy": "DEFAULT", "facteur_debit": 0.95,
        "pool": "1", "type": "aFRR", "debit_non_turbinable": 3.0,
        "groupes": [{
            "nom_groupe": "A", "priorite": 1, "debit_armement_turbine": 1.0, "debit_max": 30.0,
            "rendement": [78.0, 0.5, -0.01],
            "chute_disponible_polynome": [6.0, -0.005, 0.00003],
        }],
    }]
    (reference_dir / "config-general.json").write_text(json.dumps(records), encoding="utf-8")

    run_training(
        "test_centrale", 8, exutoire, bv_json, meta_type="ridge", epochs=2, n_trials_lgbm=0, n_trials_final=0,
        weights_dir=models_dir / "test_centrale" / "h8",
    )

    import previ_r2d2.model.pipeline.predict_orchestrator as po

    now = df.index[-1]

    def fake_load_prediction_window(dossier, horizon_steps, timestep, now, lookback_days=90, **kwargs):
        future_index = pd.date_range(now + pd.Timedelta(hours=1), periods=horizon_steps, freq="1h")
        future = pd.DataFrame(
            {
                "debit_m3s": np.nan,
                "latitude_S1": 43.1, "longitude_S1": 0.9,
                "temperature_S1": 280.0, "precipitation_S1": 1.0, "altitude_S1": 300.0,
            },
            index=future_index,
        )
        return pd.concat([df, future])

    monkeypatch.setattr(po, "load_prediction_window", fake_load_prediction_window)

    result = po.run_prediction("test_centrale", 8, exutoire, bv_json, now)

    prevision_path = centrales_dir / "test_centrale" / "prevision.json"
    assert prevision_path.exists()
    data = json.loads(prevision_path.read_text())
    assert data["centrale"] == "test_centrale"
    assert data["priorite"] == 1  # priorite du groupe A (int, pas rec["pool"])
    assert data["type"] == "aFRR"
    assert data["is-current"] is True
    # points = out_df tronqué à partir de `now` -- inclut le point "maintenant"
    # (lead_hours=0) + les 8 pas futurs, plus les points historiques dont
    # l'horodatage turbine (décalage transit + affichage Europe/Paris) tombe
    # après `now` (cas réel : decalage_h=2h + tz +2h en mai décale plusieurs
    # observations récentes au-delà de l'heure de lancement).
    assert len(data["run"]["points"]) == 13
    now_point = data["run"]["points"][0]
    assert now_point["lead_hours"] == 0
    future_point = data["run"]["points"][1]
    assert future_point["lead_hours"] == 1
    assert set(future_point["debit"].keys()) == {"entrant", "turbinable", "reserve"}
    assert "puissance" in future_point and "pmax_dyn" in future_point
    assert set(data["puissance-reel"].keys()) == {"puissance", "pmax_dyn", "chute_estimee", "rendement_estime"}
    assert Path(result["csv_path"]).exists()
