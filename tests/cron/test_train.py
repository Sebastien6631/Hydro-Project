from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "train_script", Path(__file__).resolve().parents[2] / "cron" / "scripts" / "train.py"
)
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def _patch_common(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path / "nas")
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")
    monkeypatch.setattr(train_script, "refresh_data_preparation", lambda dossier: None)


def test_run_skips_ineligible_dossiers_and_reports_empty_digest(tmp_path, monkeypatch, caplog):
    _patch_common(tmp_path, monkeypatch)

    (tmp_path / "centrales" / "apas_G1_G4").mkdir(parents=True)
    (tmp_path / "centrales" / "apas_G1_G4" / "bv.json").write_text("{}", encoding="utf-8")

    with caplog.at_level("INFO"):
        exit_code = train_script.run()

    assert exit_code == 0
    assert "Aucune centrale éligible" in caplog.text


def test_train_one_promotes_when_no_production_model_exists(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod
    from previ_r2d2.model.pipeline import promotion
    from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv
    from tests.model.pipeline.test_orchestrator import make_synthetic_df

    centrales_dir = tmp_path / "centrales"
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(promotion.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0))
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    dossier_dir = centrales_dir / "test_centrale"
    dossier_dir.mkdir(parents=True)
    bv_json = {
        "exutoire": {"lat": 43.13, "lon": 0.92},
        "bassin_versant": {"altitude_moyenne_m": 300.0, "surface_km2": 100.0},
        "parametres_calage": {"K_base": 1.0, "exposition": 1.0, "kc_unit": 1.0},
        "stations_hydrometriques": [],
    }
    (dossier_dir / "bv.json").write_text(json.dumps(bv_json), encoding="utf-8")
    write_data_preparation_csv(make_synthetic_df(), dossier_dir / "data_preparation.csv")

    summary = train_script.train_one(
        "test_centrale", 72, promote=True, epochs=2, n_trials_lgbm=0, n_trials_final=0)

    assert "PROMU v1" in summary
    prod_dir = tmp_path / "models" / "test_centrale" / "h72"
    assert (prod_dir / "version.json").exists()
    assert (prod_dir / "data_preparation.csv").exists()
    assert (prod_dir / "bv.json").exists()


def test_run_continues_after_one_dossier_fails(tmp_path, monkeypatch, caplog):
    """Une centrale en échec (données corrompues, bug ponctuel...) ne doit
    jamais empêcher les autres centrales éligibles ce jour-là d'être
    entraînées -- cf. isolation par (dossier, horizon) dans `run()`."""
    _patch_common(tmp_path, monkeypatch)
    centrales_dir = tmp_path / "centrales"

    for dossier in ("centrale_en_panne", "centrale_ok"):
        (centrales_dir / dossier).mkdir(parents=True)
        (centrales_dir / dossier / "bv.json").write_text("{}", encoding="utf-8")

    def fake_train_one(dossier, horizon, **kwargs):
        if dossier == "centrale_en_panne":
            raise ValueError("données corrompues")
        return f"{dossier} h{horizon} : PROMU v1 (test)"

    monkeypatch.setattr(train_script, "train_one", fake_train_one)
    monkeypatch.setattr(train_script, "is_eligible_for_training", lambda d, h: True)

    with caplog.at_level("INFO"):
        exit_code = train_script.run()

    assert exit_code == 1  # au moins un échec -> code de retour non-nul
    assert "centrale_en_panne" in caplog.text and "ÉCHEC" in caplog.text
    assert "centrale_ok" in caplog.text and "PROMU v1" in caplog.text






def test_train_one_does_not_promote_without_the_explicit_flag(tmp_path, monkeypatch):
    """Par défaut, un entraînement manuel écrit un candidat et affiche la décision,
    mais ne touche NI models/ NI git. La promotion (copie + dvc add + commit + tag)
    ne se déclenche qu'avec --promote ; les stages DVC train_new/train_monthly le
    passent explicitement pour garder le comportement de production."""
    from previ_r2d2.common import config as cfg_mod
    from previ_r2d2.model.pipeline import promotion
    from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv
    from tests.model.pipeline.test_orchestrator import make_synthetic_df

    centrales_dir = tmp_path / "centrales"
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    appels = []
    monkeypatch.setattr(promotion.subprocess, "run",
                        lambda *a, **k: appels.append(a) or SimpleNamespace(stdout="", stderr="", returncode=0))

    dossier_dir = centrales_dir / "test_centrale"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "bv.json").write_text(json.dumps({
        "exutoire": {"lat": 43.13, "lon": 0.92},
        "bassin_versant": {"altitude_moyenne_m": 300.0, "surface_km2": 100.0},
        "parametres_calage": {"K_base": 1.0, "exposition": 1.0, "kc_unit": 1.0},
        "stations_hydrometriques": [],
    }), encoding="utf-8")
    write_data_preparation_csv(make_synthetic_df(), dossier_dir / "data_preparation.csv")

    summary = train_script.train_one("test_centrale", 72, epochs=2, n_trials_lgbm=0, n_trials_final=0)

    assert "NON PROMU" in summary and "--promote" in summary
    assert not (tmp_path / "models" / "test_centrale" / "h72").exists()
    assert appels == [], "aucune commande git/dvc ne doit être lancée"
    # le candidat, lui, est bien sur disque : l'entraînement n'est pas perdu
    assert (tmp_path / "weights" / "hybrid_candidate" / "test_centrale" / "h72" / "results.json").exists()


