from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from previ_r2d2.model.pipeline import promotion
from previ_r2d2.model.pipeline.orchestrator import run_training
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv
from tests.model.pipeline.test_orchestrator import make_synthetic_df



def _fake_run_clean_tree(*args, **kwargs):
    """Stub de subprocess.run : arbre git propre, commandes dvc/git sans effet."""
    return SimpleNamespace(stdout="", stderr="", returncode=0)


def _train(tmp_path, monkeypatch, weights_dir):
    from previ_r2d2.common import config as cfg_mod

    centrales_dir = tmp_path / "centrales"
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")

    df = make_synthetic_df()
    write_data_preparation_csv(df, centrales_dir / "test_centrale" / "data_preparation.csv")

    exutoire = {"lat": 43.13, "lon": 0.92}
    bv_json = {
        "bassin_versant": {"altitude_moyenne_m": 300.0, "surface_km2": 100.0},
        "parametres_calage": {"K_base": 1.0, "exposition": 1.0, "kc_unit": 1.0},
        "stations_hydrometriques": [],
    }
    return run_training(
        "test_centrale", 72, exutoire, bv_json,
        meta_type="ridge", epochs=2, n_trials_lgbm=0, n_trials_final=0,
        weights_dir=weights_dir,
    )


def test_evaluate_returns_first_training_when_no_production_model(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    candidate_results = {"kge_stacking": 0.5, "_eval_context": {}}

    decision = promotion.evaluate_candidate_vs_production("test_centrale", 72, candidate_results)

    assert decision == {"decision": "first_training", "candidate_kge": 0.5, "production_kge": None}


def test_promote_model_copies_files_and_writes_version(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(promotion.subprocess, "run", _fake_run_clean_tree)

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "meta_config.json").write_text("{}", encoding="utf-8")

    version = promotion.promote_model("test_centrale", 72, candidate_dir, kge=0.81)

    assert version == 1
    prod_dir = tmp_path / "models" / "test_centrale" / "h72"
    assert (prod_dir / "meta_config.json").exists()
    version_data = json.loads((prod_dir / "version.json").read_text(encoding="utf-8"))
    assert version_data["version"] == 1
    assert version_data["kge_stacking"] == 0.81


def test_promote_model_increments_version(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(promotion.subprocess, "run", _fake_run_clean_tree)

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "meta_config.json").write_text("{}", encoding="utf-8")

    promotion.promote_model("test_centrale", 72, candidate_dir, kge=0.81)
    version = promotion.promote_model("test_centrale", 72, candidate_dir, kge=0.85)

    assert version == 2


def test_evaluate_candidate_vs_production_end_to_end(tmp_path, monkeypatch):
    old_results = _train(tmp_path, monkeypatch, tmp_path / "weights" / "old")
    monkeypatch.setattr(promotion.subprocess, "run", _fake_run_clean_tree)
    promotion.promote_model("test_centrale", 72, tmp_path / "weights" / "old", old_results["kge_stacking"])

    new_results = _train(tmp_path, monkeypatch, tmp_path / "weights" / "new")

    decision = promotion.evaluate_candidate_vs_production("test_centrale", 72, new_results)

    assert decision["decision"] in ("promote", "keep")
    assert isinstance(decision["production_kge"], float)


def test_promote_model_restores_previous_version_on_failure(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(promotion.subprocess, "run", _fake_run_clean_tree)

    candidate_dir = tmp_path / "candidate_v1"
    candidate_dir.mkdir()
    (candidate_dir / "meta_config.json").write_text("{}", encoding="utf-8")

    version = promotion.promote_model("test_centrale", 72, candidate_dir, kge=0.81)
    assert version == 1

    prod_dir = tmp_path / "models" / "test_centrale" / "h72"
    rollback_dir = prod_dir.with_name(prod_dir.name + ".rollback")

    def _raise_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0] if args else ["git", "commit"])

    monkeypatch.setattr(promotion.subprocess, "run", _raise_run)

    candidate_dir_v2 = tmp_path / "candidate_v2"
    candidate_dir_v2.mkdir()
    (candidate_dir_v2 / "meta_config.json").write_text('{"marker": "v2"}', encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError):
        promotion.promote_model("test_centrale", 72, candidate_dir_v2, kge=0.9)

    assert prod_dir.exists()
    version_data = json.loads((prod_dir / "version.json").read_text(encoding="utf-8"))
    assert version_data["version"] == 1
    assert version_data["kge_stacking"] == 0.81
    assert (prod_dir / "meta_config.json").read_text(encoding="utf-8") == "{}"
    assert not rollback_dir.exists()


def test_promote_model_refuses_when_the_git_tree_is_dirty(tmp_path, monkeypatch):
    """promote_model committe : avec un arbre sale, son commit embarquerait des
    modifications sans rapport (piège rencontré en session -- un entraînement de
    plusieurs heures finissant dans un commit fourre-tout)."""
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(
        promotion.subprocess, "run",
        lambda *a, **k: SimpleNamespace(stdout=" M src/quelque_chose.py", stderr="", returncode=0),
    )
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "meta_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Arbre git non propre"):
        promotion.promote_model("test_centrale", 72, candidate_dir, kge=0.81)

    assert not (tmp_path / "models" / "test_centrale" / "h72").exists()


def test_promote_model_calls_dvc_through_the_current_interpreter(tmp_path, monkeypatch):
    """Sur Windows, dvc.exe vit dans le Scripts/ de l'env conda : un `dvc` nu
    lève FileNotFoundError (WinError 2) dès que l'env n'est pas activé."""
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    appels = []

    def _spy(cmd, *a, **k):
        appels.append(cmd)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(promotion.subprocess, "run", _spy)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "meta_config.json").write_text("{}", encoding="utf-8")

    promotion.promote_model("test_centrale", 72, candidate_dir, kge=0.81)

    dvc_calls = [c for c in appels if "dvc" in c]
    assert dvc_calls, "aucun appel dvc"
    assert dvc_calls[0][:3] == [sys.executable, "-m", "dvc"]
