#!/usr/bin/env python3
"""check-drift — dérive des données d'entrée (phase 4.2, Evidently).

Compare la fenêtre récente de `data_preparation.csv` à l'historique
d'entraînement qui la précède. Signal de surveillance, jamais bloquant
(contrairement à `validate-data.py`) : une dérive détectée est loggée et
écrite dans un rapport JSON (lu ensuite par `/metrics`), le code de sortie
reste 0 sauf échec technique réel (fichier illisible, etc.).

Usage :
    python cron/scripts/check-drift.py
    python cron/scripts/check-drift.py --dossier touzac_g2_G2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from projet_hydro.common import config
from projet_hydro.common.dvc_markers import write as write_marker
from projet_hydro.monitoring.drift import compute_drift, split_reference_current
from projet_hydro.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv

logger = logging.getLogger("check-drift")
REPORT_DIR = config.ROOT / "logs" / "drift"


def run(only_dossier: str | None = None) -> int:
    paths = sorted(config.CENTRALES_DIR.glob("*/data_preparation.csv"))
    if only_dossier:
        paths = [p for p in paths if p.parent.name == only_dossier]
    if not paths:
        logger.warning("aucun data_preparation.csv trouvé — données non récupérées ? (`dvc pull`)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    had_error = False
    for path in paths:
        dossier = path.parent.name
        try:
            df = read_data_preparation_csv(path)
            reference, current = split_reference_current(df)
            if reference.empty or current.empty:
                logger.warning("%s : historique insuffisant pour comparer (besoin d'une référence ET d'une fenêtre récente)", dossier)
                continue

            report = compute_drift(reference.select_dtypes("number"), current.select_dtypes("number"))
            (REPORT_DIR / f"{dossier}.json").write_text(
                json.dumps({"dossier": dossier, **report}, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            msg = "%s : %d/%d colonne(s) en dérive (%.0f%%)"
            args = (dossier, report["n_drifted_columns"], report["n_columns"], report["drift_share"] * 100)
            if report["dataset_drift"]:
                logger.warning(msg + " -- DÉRIVE DÉTECTÉE", *args)
            else:
                logger.info(msg, *args)
        except Exception as exc:
            logger.error("%s : échec (%s)", dossier, exc, exc_info=True)
            had_error = True

    write_marker("check_drift")
    return 1 if had_error else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Détection de dérive (Evidently) des données d'entrée.")
    parser.add_argument("--dossier", default=None, help="Ne vérifier qu'un dossier.")
    args = parser.parse_args(argv)
    return run(only_dossier=args.dossier)


if __name__ == "__main__":
    sys.exit(main())
