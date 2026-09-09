#!/usr/bin/env python3
"""train — entraînement d'une centrale/horizon, lancement manuel.

Entraîne un candidat, le compare au modèle en production sur le MÊME holdout
(cf. promotion.py), et affiche la décision. Ne promeut que si `--promote` est
passé : la promotion copie le candidat dans `models/`, le versionne (dvc add)
et crée un commit + tag git, et versionne aussi data_preparation.csv + bv.json
avec le modèle pour une reproductibilité exacte de chaque version promue.

Avant d'entraîner, rafraîchit le data_preparation.csv du dossier -- un seul à
la fois, jamais toutes les centrales (cf. refresh_data_preparation) :
is_eligible_for_training lit ce fichier pour l'historique 12 mois, donc sans ce
rafraîchissement ciblé une toute nouvelle centrale ne l'aurait jamais (fichier
jamais généré = 0 jour d'historique pour toujours, blocage permanent).

Les cadences automatisées (`run_new_dossiers` quotidien / `run_monthly_retrain`
mensuel, stages DVC train_new/train_monthly) ont été retirées le 2026-09-09 :
sans cron dans cette version, elles n'étaient jamais déclenchées.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

from previ_r2d2.common import config
from previ_r2d2.common.dvc_markers import write as write_marker
from previ_r2d2.model.pipeline.eligibility import is_eligible_for_training
from previ_r2d2.model.pipeline.orchestrator import HORIZON_CFG, run_training
from previ_r2d2.model.pipeline.promotion import evaluate_candidate_vs_production, promote_model

logger = logging.getLogger("train")

_BUILD_DATA_PREPARATION_PATH = Path(__file__).resolve().parent / "build-data-preparation.py"
_bdp_spec = importlib.util.spec_from_file_location("build_data_preparation_script", _BUILD_DATA_PREPARATION_PATH)
build_data_preparation_script = importlib.util.module_from_spec(_bdp_spec)
_bdp_spec.loader.exec_module(build_data_preparation_script)

# Défauts de production (plus exigeants que les défauts de run_training,
# calibrés pour un entraînement réel plutôt que pour des tests rapides).
DEFAULT_EPOCHS = 100
DEFAULT_N_TRIALS_LGBM = 100
DEFAULT_N_TRIALS_FINAL = 100


def discover_dossiers() -> list[str]:
    return sorted(p.parent.name for p in config.CENTRALES_DIR.glob("*/bv.json"))


def load_bv_json(dossier: str) -> dict:
    with open(config.CENTRALES_DIR / dossier / "bv.json") as fh:
        return json.load(fh)


def refresh_data_preparation(dossier: str) -> None:
    """Rafraîchit data_preparation.csv pour CE dossier uniquement (jamais
    toutes les centrales -- ce serait le stage DVC data_preparation complet,
    coûteux et pas nécessaire ici) avant de vérifier son éligibilité/de
    l'entraîner."""
    build_data_preparation_script.run(only_dossier=dossier)


def train_one(
    dossier: str,
    horizon: int,
    promote: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    n_trials_lgbm: int = DEFAULT_N_TRIALS_LGBM,
    n_trials_final: int = DEFAULT_N_TRIALS_FINAL,
    **run_training_kwargs,
) -> str:
    """Entraîne un candidat, compare au modèle en production, promeut si
    meilleur (ou si premier entraînement). Retourne une ligne pour le digest."""
    bv_json = load_bv_json(dossier)
    exutoire = bv_json["exutoire"]
    candidate_dir = config.ROOT / "weights" / "hybrid_candidate" / dossier / f"h{horizon}"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    results = run_training(
        dossier, horizon, exutoire, bv_json,
        epochs=epochs, n_trials_lgbm=n_trials_lgbm, n_trials_final=n_trials_final,
        weights_dir=candidate_dir, **run_training_kwargs,
    )
    decision = evaluate_candidate_vs_production(dossier, horizon, results)

    # Snapshot data_preparation.csv + bv.json dans candidate_dir avant promotion :
    # promote_model copie tout candidate_dir vers models/.../ puis le dvc-add/tag
    # d'un bloc -- ces 2 fichiers sont donc versionnés avec le même tag git que le
    # modèle, sans mécanisme DVC séparé (reproductibilité exacte : on sait quelles
    # données/config étaient actives pour une version de modèle donnée).
    shutil.copy2(config.CENTRALES_DIR / dossier / "data_preparation.csv", candidate_dir / "data_preparation.csv")
    shutil.copy2(config.CENTRALES_DIR / dossier / "bv.json", candidate_dir / "bv.json")

    eligible = decision["decision"] in ("promote", "first_training")
    if eligible and promote:
        version = promote_model(dossier, horizon, candidate_dir, results["kge_stacking"])
        summary = (
            f"{dossier} h{horizon} : PROMU v{version} "
            f"(kge_candidat={results['kge_stacking']}, kge_prod={decision['production_kge']})"
        )
    elif eligible:
        summary = (
            f"{dossier} h{horizon} : ÉLIGIBLE mais NON PROMU (relancer avec --promote) "
            f"-- candidat dans {candidate_dir} "
            f"(kge_candidat={results['kge_stacking']}, kge_prod={decision['production_kge']})"
        )
    else:
        summary = (
            f"{dossier} h{horizon} : conservé "
            f"(kge_candidat={results['kge_stacking']}, kge_prod={decision['production_kge']})"
        )

    write_marker(f"train_{dossier}_h{horizon}")
    return summary


def run(dossier: str | None = None, horizon: int | None = None, force: bool = False,
        promote: bool = False) -> int:
    """Test manuel ciblé : une seule centrale/horizon (`force=True` ignore
    l'éligibilité, pour pouvoir tester même sans historique de 12 mois ou
    avant l'échéance de réentraînement). Rafraîchit data_preparation pour ce
    dossier avant d'entraîner."""
    dossiers = [dossier] if dossier is not None else discover_dossiers()
    horizons = [horizon] if horizon is not None else sorted(HORIZON_CFG.keys())
    summaries = []
    had_error = False
    for d in dossiers:
        try:
            refresh_data_preparation(d)
        except Exception as exc:
            logger.error("Échec rafraîchissement data_preparation %s : %s", d, exc, exc_info=True)
            summaries.append(f"{d} : ÉCHEC rafraîchissement data_preparation ({exc})")
            had_error = True
            continue
        for h in horizons:
            if not force and not is_eligible_for_training(d, h):
                continue
            try:
                summaries.append(train_one(d, h, promote=promote))
            except Exception as exc:
                logger.error("Échec entraînement %s h%s : %s", d, h, exc, exc_info=True)
                summaries.append(f"{d} h{h} : ÉCHEC ({exc})")
                had_error = True

    body = "\n".join(summaries) if summaries else "Aucune centrale éligible aujourd'hui."
    logger.info(body)
    write_marker("train")
    return 1 if had_error else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", required=True, help="Centrale à entraîner.")
    parser.add_argument("--horizon", type=int, help="Test manuel ciblé sur cet horizon (8/48/72).")
    parser.add_argument(
        "--promote", action="store_true",
        help="Promouvoir le candidat s'il bat la production (copie dans models/, "
             "dvc add + commit + tag git). Sans ce flag, l'entraînement écrit "
             "seulement le candidat dans weights/hybrid_candidate/ et affiche la décision.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignorer l'éligibilité (utile avec --dossier/--horizon pour un test manuel).",
    )
    return parser.parse_args(argv)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    return run(dossier=args.dossier, horizon=args.horizon, force=args.force, promote=args.promote)


if __name__ == "__main__":
    sys.exit(main())
