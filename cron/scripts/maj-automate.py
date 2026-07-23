#!/usr/bin/env python3
"""maj-automate — sync capteurs automate + calcul du débit entrant (Qentrant).

Pour chaque dossier `flex_strategy == "HAUTE_CHUTE"` avec une entrée dans
`centrales/REFERENCE/automate_sync.yaml` : rsync les capteurs, calcule le
débit via la physique de la centrale (`automate_physics.yaml`), écrit
`centrales/<dossier>/debit_automate.csv`.

Envoie UN SEUL e-mail de recap systématique en fin d'exécution (écrits,
ignorés, erreurs groupées) -- même pattern que `maj-data.py`/`maj-puissance.py`.

Usage :
    python cron/scripts/maj-automate.py
    python cron/scripts/maj-automate.py --dossier bonneval_G2   # test ciblé
    python cron/scripts/maj-automate.py --skip-sync             # calcul seul, sans rsync
    python cron/scripts/maj-automate.py --full-history          # backfill ponctuel (tout l'historique
                                                                  # disponible, pas juste la fenêtre glissante)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from previ_r2d2.common import config, daily_report, dvc_markers
from previ_r2d2.preprocessing.automate.config import load_physics_config, load_sync_config
from previ_r2d2.preprocessing.automate.debit_csv import (
    merge_debit_series,
    read_debit_csv,
    select_debit_column,
    write_debit_csv,
)
from previ_r2d2.preprocessing.automate.physics import compute_qentrant
from previ_r2d2.preprocessing.automate.reader import read_variable
from previ_r2d2.preprocessing.automate.sync import sync_variable
from previ_r2d2.preprocessing.debit import station_store

logger = logging.getLogger("maj-automate")

WINDOW_DAYS = 2
FULL_HISTORY_START = pd.Timestamp("2000-01-01")
DEBIT_AUTOMATE_FILENAME = "debit_automate.csv"
# Colonnes de Calcul_Qturb qui dépendent réellement de puissance_horaire.csv
# (contrairement à Q_fct_charge/Q_fct_ouv, basées sur des capteurs seuls).
PUISSANCE_DEPENDENT_COLUMNS = {"Q_fct_P", "Q_fct_P_Hn_R"}


def run(only_dossier: str | None = None, skip_sync: bool = False, full_history: bool = False) -> int:
    general = config.REFERENCE_DIR / "config-general.json"
    sync_config = load_sync_config(config.REFERENCE_DIR / "automate_sync.yaml")
    physics_config = load_physics_config(config.REFERENCE_DIR / "automate_physics.yaml")
    records = json.loads(general.read_text(encoding="utf-8"))

    errors: list[str] = []
    written: list[tuple[str, int]] = []
    skipped: list[str] = []

    end = pd.Timestamp.now().normalize() + pd.Timedelta(hours=23, minutes=59)
    # Backfill ponctuel (--full-history, tout l'historique disponible) vs mise
    # à jour incrémentale de routine (fenêtre glissante de WINDOW_DAYS jours).
    start = FULL_HISTORY_START if full_history else end - pd.Timedelta(days=WINDOW_DAYS)

    for rec in records:
        dossier = rec["dossier"]
        if only_dossier and dossier != only_dossier:
            continue
        if rec.get("flex_strategy") != "HAUTE_CHUTE":
            continue
        if dossier not in sync_config or dossier not in physics_config:
            skipped.append(dossier)
            continue

        logger.info("%s : début", dossier)
        try:
            sync_entry = sync_config[dossier]
            if not skip_sync:
                source_names = sync_entry.get("source_names", {})
                for var in sync_entry["variables"]:
                    logger.info("%s : sync %s ...", dossier, var)
                    dest_dir = Path(sync_entry["dest_base"]) / var
                    sync_variable(
                        var=var,
                        source_var=source_names.get(var, var),
                        source_host=sync_entry["source_host"],
                        source_base=sync_entry["source_base"],
                        dest_dir=dest_dir,
                        corrections=sync_entry.get("corrections", {}).get(var),
                    )

            physics_entry = physics_config[dossier]
            # Resample par capteur avant de combiner (comme load_tokapi_data.py
            # côté Previ_v2) -- absorbe les doublons d'horodatage bruts et
            # aligne sur le même pas horaire que puissance_horaire.csv.
            columns = {}
            for var in sync_entry["variables"]:
                logger.info("%s : lecture %s ...", dossier, var)
                dest_dir = Path(sync_entry["dest_base"]) / var
                columns[var] = read_variable(dest_dir, start, end).resample("1h").mean()
            df = pd.DataFrame(columns)
            logger.info("%s : %d heure(s) de capteurs lues", dossier, len(df))
            if df.empty:
                skipped.append(f"{dossier} (aucune donnée capteur pour la fenêtre)")
                continue

            # puissance_horaire.csv (étage 2, nettoyé -- pas puissance.csv qui
            # est brut minute par minute) : même raison que pour le calcul de
            # transit dans onboarding-bv.py, le signal brut produit du bruit.
            puissance_path = config.CENTRALES_DIR / dossier / "puissance_horaire.csv"
            puissance = pd.read_csv(puissance_path, sep=";", parse_dates=["Date"], index_col="Date")
            # Resample + médiane (pas de dédoublonnage explicite) : même
            # logique que DataManager.load_data côté Previ_v2
            # (data_manager.py:256, `.resample('H').median()`) -- absorbe
            # naturellement d'éventuels doublons d'horodatage (ex. DST) par
            # agrégation, sans avoir besoin de les détecter explicitement.
            puissance = puissance.resample("1h").median()

            # Aligné sur `df` (les capteurs), pas l'inverse -- des centrales
            # comme Melles retiennent "Q_fct_ouv" (ouverture des injecteurs),
            # indépendant de la puissance ; tronquer `df` à la disponibilité
            # de puissance_horaire.csv priverait ces centrales de données
            # capteur fraîches à cause d'une puissance périmée dont leur
            # méthode retenue n'a même pas besoin. Les méthodes qui ont
            # vraiment besoin de puissance (Q_fct_P, Q_fct_P_Hn_R) restent à
            # zéro/NaN là où elle manque et sont filtrées naturellement par
            # compute_qentrant si elles ne sont jamais renseignées.
            puissance = puissance.reindex(df.index)
            puissance_available = puissance["power_output"].notna()
            logger.info("%s : puissance disponible sur %d/%d heure(s) de la fenêtre",
                        dossier, puissance_available.sum(), len(df))

            logger.info("%s : calcul physique (compute_qentrant) ...", dossier)
            config_dicts = {k: v for k, v in physics_entry.items() if k.startswith("Config_")}
            _, _, qentrant = compute_qentrant(
                df, config_dicts, physics_entry.get("mapping", {}), puissance,
                physics_entry.get("consigne_regulation", {}),
            )
            main_col = select_debit_column(qentrant)
            result = qentrant[main_col]
            if main_col in PUISSANCE_DEPENDENT_COLUMNS:
                # La colonne retenue a réellement besoin de la puissance (ex.
                # Bonneval, pas de capteur d'ouverture) -- sans ce filtre, les
                # heures où puissance_horaire.csv manque produiraient un
                # résidu déversoir-seul trompeur (Q_fct_P forcé à 0, pas NaN,
                # cf. calcul_polynome_vectorise) plutôt qu'une vraie absence
                # de donnée.
                result = result[puissance_available]
            logger.info("%s : colonne retenue %s (%d point(s)), écriture debit_automate.csv ...",
                        dossier, main_col, len(result))
            nas_path = config.NAS_DATA_ROOT / dossier / DEBIT_AUTOMATE_FILENAME
            merged = merge_debit_series(read_debit_csv(nas_path), result)
            n = write_debit_csv(merged, nas_path)
            station_store.ensure_local_symlink(nas_path, dossier, DEBIT_AUTOMATE_FILENAME)
            written.append((dossier, n))
            logger.info("%s : terminé (%d point(s) écrit(s))", dossier, n)
        except Exception as exc:
            logger.error("%s : échec (%s)", dossier, exc)
            errors.append(f"{dossier} : {exc}")

    # Recap journalisé pour le digest quotidien (daily-sync-report.py) --
    # maj-automate tourne toutes les heures, plus un mail par exécution.
    # `skipped` inclut des dossiers DURABLEMENT ignorés (ex. bonneval_G1, qui
    # partage les données de bonneval_G2 via modele_source et n'aura jamais
    # d'entrée propre dans automate_sync.yaml) -- normal de les revoir à
    # chaque recap, ce n'est pas un signal d'alerte.
    lines = [f"maj-automate — exécution du {datetime.date.today()}", ""]
    lines.append(
        f"Bilan : {len(written)} écrit(s), {len(skipped)} ignoré(s) (pas de config automate), "
        f"{len(errors)} erreur(s)."
    )
    lines.append("")
    if written:
        lines.append("ÉCRITS :")
        lines += [f"  - {d} : {n} points" for d, n in written]
        lines.append("")
    if skipped:
        lines.append(f"IGNORÉS (pas de config automate) : {', '.join(skipped)}")
        lines.append("")
    if errors:
        lines.append(f"ERREURS ({len(errors)}) :")
        lines += [f"  - {e}" for e in errors]
        lines.append("")
    subject = f"[previ-record] maj-automate — {len(written)} écrit(s), {len(errors)} erreur(s)"
    daily_report.record("maj-automate", subject, "\n".join(lines), has_errors=bool(errors))
    dvc_markers.write("debit_automate")

    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                         format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="Sync + calcul débit automate.")
    parser.add_argument("--dossier", default=None, help="Test ciblé sur un seul dossier.")
    parser.add_argument("--skip-sync", action="store_true",
                         help="Calcul seul, sans rsync (test sans risque de collision avec Previ_v2).")
    parser.add_argument("--full-history", action="store_true",
                         help="Backfill ponctuel : tout l'historique disponible, pas la fenêtre glissante.")
    args = parser.parse_args(argv)
    return run(only_dossier=args.dossier, skip_sync=args.skip_sync, full_history=args.full_history)


if __name__ == "__main__":
    sys.exit(main())
