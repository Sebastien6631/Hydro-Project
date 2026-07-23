#!/usr/bin/env python3
"""predict-archive — stage DVC dvc/postprocessing/dvc.yaml:predict_archive.

Greffé sur le `dvc repro` horaire existant (debit/debit_automate, même appel
pour éviter le piège du lock DVC exclusif). Pour chaque (dossier, horizon)
ayant un modèle en production : archive le JSON de l'heure précédente sur le
NAS, puis prédit la nouvelle heure et l'écrit dans centrales/<dossier>/.
"""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd

from previ_r2d2.common import config, daily_report, mlflow_tracker
from previ_r2d2.common.dvc_markers import write as write_marker
from previ_r2d2.model.pipeline.eligibility import has_production_model
from previ_r2d2.model.pipeline.orchestrator import HORIZON_CFG
from previ_r2d2.model.pipeline.predict_orchestrator import run_prediction
from previ_r2d2.postprocessing.archive import archive_previous_json

logger = logging.getLogger("predict-archive")


def discover_dossiers() -> list[str]:
    return sorted(p.parent.name for p in config.CENTRALES_DIR.glob("*/bv.json"))


def load_bv_json(dossier: str) -> dict:
    with open(config.CENTRALES_DIR / dossier / "bv.json") as fh:
        return json.load(fh)


def run() -> int:
    now = pd.Timestamp.now().floor("h")
    had_error = False
    n_predicted = 0
    failures = []
    for dossier in discover_dossiers():
        for horizon in sorted(HORIZON_CFG.keys()):
            if not has_production_model(dossier, horizon):
                continue
            try:
                archive_previous_json(dossier, config.CENTRALES_DIR, config.NAS_ARCHIVE_ROOT, now)
                bv_json = load_bv_json(dossier)
                exutoire = bv_json["exutoire"]
                prediction = run_prediction(dossier, horizon, exutoire, bv_json, now)
                mlflow_tracker.log_prediction_hybrid(dossier, horizon, prediction)
                n_predicted += 1
            except Exception as exc:
                logger.error("Échec prédiction %s h%s : %s", dossier, horizon, exc, exc_info=True)
                had_error = True
                failures.append(f"{dossier} h{horizon} : ÉCHEC ({exc})")

    # Comme maj-data.py/maj-automate.py (aussi horaires) : un run par heure
    # journalisé, agrégé en UN SEUL mail quotidien par daily-sync-report.py --
    # jamais un mail par run horaire (24x/jour).
    body = f"{n_predicted} prédiction(s) horaire(s) réussie(s)."
    if failures:
        body += "\n" + "\n".join(failures)
    daily_report.record("predict-archive", f"Prédiction+archivage — {n_predicted} réussie(s), {len(failures)} échec(s)", body, had_error)

    write_marker("predict_archive")
    return 1 if had_error else 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    )
    return run()


if __name__ == "__main__":
    sys.exit(main())
