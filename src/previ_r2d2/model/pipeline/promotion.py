"""Comparaison candidat/production et promotion versionnée (DVC + tag git) --
la comparaison réévalue le modèle en production sur EXACTEMENT le même
holdout que le candidat (`_eval_context` du run_training courant, cf.
orchestrator.py), pas sur son propre résultat historique (holdout différent,
non comparable)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from previ_r2d2.common import config
from previ_r2d2.model.architectures.lightgbm.predict import predict_lgbm_full
from previ_r2d2.model.architectures.stacking import kge_components
from previ_r2d2.model.pipeline.predict import predict_test_set
from previ_r2d2.model.pipeline.predict_orchestrator import load_trained_models


def production_dir(dossier: str, horizon: int) -> Path:
    return config.MODELS_DIR / dossier / f"h{horizon}"


def evaluate_candidate_vs_production(dossier: str, horizon: int, candidate_results: dict) -> dict:
    """Décide promotion/conservation en comparant le KGE Stacking du candidat
    (déjà calculé par run_training) à celui du modèle en production, réévalué
    sur le même holdout via `_eval_context`."""
    candidate_kge = candidate_results["kge_stacking"]
    prod_dir = production_dir(dossier, horizon)

    if not (prod_dir / "meta_config.json").exists():
        return {"decision": "first_training", "candidate_kge": candidate_kge, "production_kge": None}

    ctx = candidate_results["_eval_context"]
    n_features = ctx["X_seq_test"].shape[2]
    bilstm, lgbm_result, meta, meta_scaler = load_trained_models(prod_dir, ctx["horizon_steps"], n_features)

    pred_lgbm_all = predict_lgbm_full(
        lgbm_result, ctx["df_full_ctx"], ctx["exutoire"], ctx["bv_params"],
        ctx["steps_per_day"], ctx["horizon_steps"], ctx["transit_amont"],
    )
    pred_lstm_test = bilstm.predict(ctx["X_seq_test"])
    yt_v, pl_v, pt_v, stk_v, *_rest = predict_test_set(
        pred_lstm_test, ctx["y_test"], ctx["lstm_idx_test"], ctx["t_last_test"], ctx["n_train"],
        pred_lgbm_all, ctx["df_full_ctx"], meta, meta_scaler, ctx["horizon_steps"],
        ctx["meteo_feature_cols"], ctx["df_test"],
    )
    production_kge = kge_components(yt_v, stk_v)["kge"]

    decision = "promote" if candidate_kge > production_kge else "keep"
    return {"decision": decision, "candidate_kge": candidate_kge, "production_kge": production_kge}


def promote_model(dossier: str, horizon: int, candidate_weights_dir: Path, kge: float) -> int:
    """Copie les artefacts candidat vers models/<dossier>/h<horizon>/ (nouvelle
    version active), versionnée par DVC + tag git (rollback = `git checkout
    <tag> && dvc pull`). Retourne le numéro de version."""
    # Garde-fou : promote_model committe. Avec un arbre sale, le commit de
    # promotion embarquerait des modifications sans rapport (et un `git add` sur
    # le .dvc ne suffit pas à s'en prémunir si un hook ou un futur -a s'en mêle).
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=config.ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "Arbre git non propre : promotion refusée pour ne pas committer des "
            "modifications sans rapport :\n" + dirty
        )

    prod_dir = production_dir(dossier, horizon)
    version_path = prod_dir / "version.json"
    version = 1
    if version_path.exists():
        version = json.loads(version_path.read_text(encoding="utf-8"))["version"] + 1

    backup_dir = prod_dir.with_name(prod_dir.name + ".rollback")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if prod_dir.exists():
        prod_dir.rename(backup_dir)

    try:
        shutil.copytree(candidate_weights_dir, prod_dir)

        version_path.write_text(json.dumps({
            "version": version,
            "kge_stacking": round(kge, 4),
            "promoted_at": datetime.now().isoformat(),
        }), encoding="utf-8")

        tag = f"{dossier}-h{horizon}-v{version}"
        # `python -m dvc` et non `dvc` : sur Windows dvc.exe vit dans le Scripts        # de l'env conda, absent du PATH si l'environnement n'est pas activé --
        # un `dvc` nu y lève FileNotFoundError (WinError 2), au message opaque.
        # L'interpréteur courant, lui, est toujours celui qui a importé ce module.
        subprocess.run([sys.executable, "-m", "dvc", "add", str(prod_dir)], cwd=config.ROOT, check=True)
        subprocess.run(["git", "add", f"{prod_dir}.dvc"], cwd=config.ROOT, check=True)
        subprocess.run(["git", "commit", "-m", f"model: promotion {tag}"], cwd=config.ROOT, check=True)
        subprocess.run(["git", "tag", tag], cwd=config.ROOT, check=True)
    except Exception:
        if prod_dir.exists():
            shutil.rmtree(prod_dir)
        if backup_dir.exists():
            backup_dir.rename(prod_dir)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    return version
