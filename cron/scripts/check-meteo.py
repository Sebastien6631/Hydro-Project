#!/usr/bin/env python3
"""check-meteo — vérifie qu'Open-Meteo sert une prévision pour chaque centrale.

Tâche `meteo` du DAG Airflow `hydro_predict`, en parallèle de `debit`
(maj-data). La prédiction h8 appelle Open-Meteo en direct (build_dossier) et
`read_points` IGNORE un point qui échoue : sept points en échec = DataFrame
vide, sans erreur. Ce script ajoute la sévérité par-dessus : fenêtre de
prévision vide -> exit 1, et predict_archive ne démarre pas.

Fenêtre [now-1h, now+8h] : c'est la prévision qui manque en pratique
("aucune donnée pour la fenêtre"), pas l'historique.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from projet_hydro.common import config
from projet_hydro.common.console import force_utf8
from projet_hydro.preprocessing.meteo.open_meteo import read_points

logger = logging.getLogger("check-meteo")


def run(only_dossier: str | None = None) -> int:
    paths = sorted(config.CENTRALES_DIR.glob("*/bv.json"))
    if only_dossier:
        paths = [p for p in paths if p.parent.name == only_dossier]
    if not paths:
        logger.warning("aucun bv.json trouvé — données non récupérées ? (`dvc pull`)")

    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    had_error = False
    for path in paths:
        dossier = path.parent.name
        points = json.loads(path.read_text(encoding="utf-8")).get("stations_meteo_nwp", [])
        if not points:
            logger.info("%s : pas de station météo dans bv.json, ignoré", dossier)
            continue
        df = read_points(points, now - pd.Timedelta(hours=1), now + pd.Timedelta(hours=8))
        if df.empty:
            logger.error("%s : Open-Meteo n'a rien rendu pour [%s, +8h] (%d point(s))", dossier, now, len(points))
            had_error = True
        else:
            logger.info("%s : météo OK (%d lignes, %d colonnes)", dossier, len(df), df.shape[1])

    return 1 if had_error else 0


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Disponibilité de la prévision Open-Meteo par centrale.")
    parser.add_argument("--dossier", default=None, help="Ne vérifier qu'un dossier.")
    args = parser.parse_args(argv)
    return run(only_dossier=args.dossier)


if __name__ == "__main__":
    sys.exit(main())
