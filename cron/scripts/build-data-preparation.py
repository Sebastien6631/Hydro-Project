#!/usr/bin/env python3
"""build-data-preparation — construit/met à jour le Data_Preparation (débit +
météo + amont brut) par centrale, pour l'entraînement du modèle
hybride meta.

Périmètre historique confirmé (J-1 et avant) uniquement -- pas la fenêtre
temps réel/prévision (réservé à predict_future_meta).

Pas de cron : lancé à la demande avant un entraînement (la cadence dépend de
la date des entraînements, pas d'une fréquence fixe).

Usage :
    python cron/scripts/build-data-preparation.py
    python cron/scripts/build-data-preparation.py --dossier apas_G1_G4
    python cron/scripts/build-data-preparation.py --full-history
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys

import pandas as pd

from previ_r2d2.common.console import force_utf8
from previ_r2d2.preprocessing.meteo.open_meteo import ARCHIVE_MIN_DATE
from previ_r2d2.common import config
from previ_r2d2.common.dvc_markers import write as write_marker
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import (
    merge_data_preparation,
    read_data_preparation_csv,
    write_data_preparation_csv,
)
from previ_r2d2.preprocessing.data_preparation.dossier_window import build_dossier

logger = logging.getLogger("build-data-preparation")

DATA_PREPARATION_FILENAME = "data_preparation.csv"
# Plancher = début de la météo disponible. Le débit remonte à 2021, mais une
# ligne sans météo est inexploitable : `build_features` la supprime (dropna) et
# `build_sequences` rejette toute fenêtre qui en contient une. Démarrer avant
# la météo ne rallonge donc pas l'entraînement -- ça le SABOTE, en décalant le
# split 80/20 vers un passé vide. Mesuré : le meta-learner tombait à 1175
# échantillons au lieu de ~29 000, KGE stacking a -6.6e7.
FULL_HISTORY_START = ARCHIVE_MIN_DATE


def run(only_dossier: str | None = None, full_history: bool = False) -> int:
    general = config.REFERENCE_DIR / "config-general.json"
    records = json.loads(general.read_text(encoding="utf-8"))

    errors: list[str] = []
    written: list[tuple[str, int]] = []
    skipped: list[str] = []

    # Bornée à J-1 23h : au-delà, la rétention météo n'a pas encore stabilisé
    # un fichier unique par heure (hors périmètre, cf. spec).
    end = pd.Timestamp.now().normalize() - pd.Timedelta(hours=1)

    for rec in records:
        dossier = rec["dossier"]
        if only_dossier and dossier != only_dossier:
            continue
        logger.info("%s : début", dossier)
        try:
            nas_path = config.NAS_DATA_ROOT / dossier / DATA_PREPARATION_FILENAME
            existing = read_data_preparation_csv(nas_path)
            # Reprend juste après le dernier point déjà écrit -- pas de fenêtre
            # fixe, la cadence de ce script dépend des dates d'entraînement.
            if full_history or existing.empty:
                start = FULL_HISTORY_START
            else:
                start = existing.index.max() + pd.Timedelta(hours=1)

            new_data = build_dossier(rec, start, end)
            if new_data.empty:
                logger.info("%s : aucune donnée pour la fenêtre, ignoré", dossier)
                skipped.append(dossier)
                continue

            logger.info("%s : fusion + écriture ...", dossier)
            merged = merge_data_preparation(existing, new_data)
            n = write_data_preparation_csv(merged, nas_path)
            written.append((dossier, n))
            logger.info("%s : terminé (%d ligne(s) écrite(s))", dossier, n)
        except Exception as exc:
            logger.error("%s : échec (%s)", dossier, exc)
            errors.append(f"{dossier} : {exc}")

    lines = [f"build-data-preparation — exécution du {datetime.date.today()}", ""]
    lines.append(
        f"Bilan : {len(written)} écrit(s), {len(skipped)} ignoré(s) (aucune donnée pour la fenêtre), "
        f"{len(errors)} erreur(s)."
    )
    lines.append("")
    if written:
        lines.append("ÉCRITS :")
        lines += [f"  - {d} : {n} lignes" for d, n in written]
        lines.append("")
    if skipped:
        lines.append(f"IGNORÉS : {', '.join(skipped)}")
        lines.append("")
    if errors:
        lines.append(f"ERREURS ({len(errors)}) :")
        lines += [f"  - {e}" for e in errors]
        lines.append("")
    logger.info("\n".join(lines))

    write_marker("data_preparation")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    logging.basicConfig(level=logging.INFO,
                         format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="Construction du Data_Preparation (débit + météo + amont brut).")
    parser.add_argument("--dossier", default=None, help="Test ciblé sur un seul dossier.")
    parser.add_argument("--full-history", action="store_true",
                         help="Backfill complet depuis le début de chaque source, pas juste la reprise.")
    args = parser.parse_args(argv)
    return run(only_dossier=args.dossier, full_history=args.full_history)


if __name__ == "__main__":
    sys.exit(main())