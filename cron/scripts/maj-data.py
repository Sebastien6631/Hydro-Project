#!/usr/bin/env python3
"""maj-data — importe ou met à jour les débits des raccordements DEFAULT.

Pour chaque raccordement de centrales/config-general.json :
  - ignore ceux dont `flex_strategy != "DEFAULT"` ;
  - sinon, sur le CSV `data/<dossier>/<station>.csv` :
      • s'il n'existe pas encore  -> IMPORT complet (01/01/2021 -> aujourd'hui) ;
      • s'il existe                -> MISE À JOUR incrémentale (source la plus fraîche).

À la fin, envoie UN SEUL e-mail de recap (imports, mises à jour, erreurs groupées).

Usage :
    python cron/scripts/maj-data.py
    python cron/scripts/maj-data.py --end 03/07/2026
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import traceback
from pathlib import Path

import requests

from previ_r2d2.common.console import force_utf8
from previ_r2d2.common import config, dvc_markers
from previ_r2d2.preprocessing.debit import hydro_export, hydro_update, station_store
from previ_r2d2.preprocessing.debit.eaufrance import EauFranceClient, EauFranceError, series_range
from previ_r2d2.preprocessing.debit.hubeau import HubEauClient

FLEX_STRATEGY_COLLECT = "DEFAULT"
START_IMPORT = "01/01/2021"


def _today_fr() -> str:
    return datetime.date.today().strftime("%d/%m/%Y")


def _build_report(
    end: str,
    imported: list[tuple],
    updated: list[tuple],
    uptodate: list[tuple],
    skipped: int,
    errors: list[str],
) -> tuple[str, str]:
    """Construit (sujet, corps) du mail de recap."""
    subject = (
        f"[previ-record] maj-data — {len(imported)} import, "
        f"{len(updated)} maj, {len(errors)} erreur(s)"
    )

    lines = [
        f"maj-data — exécution du {end}",
        f"Fenêtre d'import (si CSV absent) : {START_IMPORT} → {end}",
        "",
        f"Bilan : {len(imported)} importé(s), {len(updated)} mis à jour, "
        f"{len(uptodate)} déjà à jour, {skipped} ignoré(s) (≠ DEFAULT), "
        f"{len(errors)} erreur(s).",
        "",
    ]

    if imported:
        lines.append("IMPORTS (nouveaux CSV) :")
        lines += [f"  - {d}/{s} [{role}] : {n} points" for d, s, role, n in imported]
        lines.append("")
    if updated:
        lines.append("MISES À JOUR :")
        lines += [
            f"  - {d}/{s} [{role}] : +{r['added']} points via {r['source']} "
            f"(après {r['last_before']})"
            for d, s, role, r in updated
        ]
        lines.append("")
    if uptodate:
        lines.append("DÉJÀ À JOUR (aucun point à ajouter) :")
        lines += [f"  - {d}/{s} [{role}]" for d, s, role, _ in uptodate]
        lines.append("")
    if errors:
        lines.append(f"ERREURS ({len(errors)}) :")
        lines += [f"  - {e}" for e in errors]
        lines.append("")

    return subject, "\n".join(lines)


def run(
    config_path: Path | None = None,
    end: str | None = None,
    only_dossier: str | None = None,
) -> int:
    end = end or _today_fr()
    cfg = config_path or config.REFERENCE_DIR / "config-general.json"

    if not cfg.exists():
        msg = f"{cfg} introuvable — lance d'abord `dvc pull` (données statiques versionnées, cf. README)."
        print(f"✗ {msg}", file=sys.stderr)
        return 1

    raccordements = json.loads(cfg.read_text(encoding="utf-8"))
    print(f"→ {len(raccordements)} raccordement(s) au total (fin : {end}).")

    eaufrance = EauFranceClient()
    hubeau = HubEauClient()

    imported: list[tuple] = []
    updated: list[tuple] = []
    uptodate: list[tuple] = []
    errors: list[str] = []
    skipped = 0

    for rec in raccordements:
        dossier = rec.get("dossier")
        if not dossier:
            continue

        if only_dossier and dossier != only_dossier:
            continue

        # On ne traite que les raccordements en stratégie DEFAULT.
        if rec.get("flex_strategy") != FLEX_STRATEGY_COLLECT:
            skipped += 1
            continue

        # Station de référence + toutes les stations amont (dédupliquées).
        stations: list[tuple[str, str]] = []
        ref = rec.get("station_vigicrue_reference")
        if ref:
            stations.append((ref, "reference"))
        for amont in rec.get("stations_vigicrue_amont") or []:
            if amont and amont not in [s for s, _ in stations]:
                stations.append((amont, "amont"))

        if not stations:
            msg = (
                f"{dossier} : flex_strategy=DEFAULT mais aucune station "
                "(station_vigicrue_reference / stations_vigicrue_amont vides)"
            )
            print(f"  ✗ {msg}", file=sys.stderr)
            errors.append(msg)
            continue

        for station, role in stations:
            filename = station_store.filename_for(station, role)
            try:
                nas_path, status = station_store.resolve_nas_path(station, dossier, role)

                if status == "linked":
                    # Un autre dossier porte déjà ce code : resolve_nas_path a déjà posé
                    # le symlink local vers le fichier réel de l'autre dossier.
                    print(f"• {dossier}/{filename} [{role}] — LIEN vers dossier existant")
                    continue

                if status == "new":
                    # Rien encore -> import complet de l'historique. Tentative en
                    # un seul appel d'abord (cas normal) ; si hydro.eaufrance.fr
                    # échoue même après ses propres retries (souvent le cas sur
                    # plusieurs années de données horaires d'un coup), on retombe
                    # sur un découpage en tranches plus petites.
                    try:
                        payload = eaufrance.series(station, START_IMPORT, end)
                    except (requests.exceptions.RequestException, EauFranceError):
                        payload = series_range(eaufrance, station, START_IMPORT, end)
                    n = hydro_export.write_csv(payload, nas_path)
                    imported.append((dossier, filename, role, n))
                    print(f"• {dossier}/{filename} [{role}] — IMPORT {n} points")
                else:
                    # Déjà présent -> mise à jour incrémentale.
                    res = hydro_update.update_station_csv(
                        nas_path, station, end_fr=end, eaufrance=eaufrance, hubeau=hubeau
                    )
                    if res["added"] > 0:
                        updated.append((dossier, filename, role, res))
                        print(
                            f"• {dossier}/{filename} [{role}] — MAJ +{res['added']} "
                            f"via {res['source']}"
                        )
                    else:
                        uptodate.append((dossier, filename, role, res))
                        print(f"• {dossier}/{filename} [{role}] — déjà à jour")

            except Exception as exc:
                msg = f"{dossier}/{filename} [{role}] : {exc}"
                print(f"  ✗ {msg}", file=sys.stderr)
                errors.append(msg)

    print(
        f"\n→ Bilan : {len(imported)} import, {len(updated)} maj, "
        f"{len(uptodate)} à jour, {skipped} ignoré(s), {len(errors)} erreur(s)."
    )

    # Bilan affiché/loggé plutôt qu'un mail par exécution -- maj-data tourne
    # toutes les heures.
    _, body = _build_report(end, imported, updated, uptodate, skipped, errors)
    print(body)
    dvc_markers.write("debit")

    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    force_utf8()
    parser = argparse.ArgumentParser(
        prog="maj-data",
        description="Importe ou met à jour les débits des raccordements DEFAULT.",
    )
    parser.add_argument(
        "--config",
        metavar="FICHIER",
        help="Chemin de config-general.json (défaut : centrales/config-general.json).",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="JJ/MM/AAAA",
        help="Borne de fin (défaut : aujourd'hui).",
    )
    parser.add_argument(
        "--dossier",
        default=None,
        metavar="NOM",
        help="Ne traite que ce raccordement (test ciblé), ex. apas_G1_G4.",
    )
    args = parser.parse_args(argv)
    cfg = Path(args.config) if args.config else None

    try:
        return run(config_path=cfg, end=args.end, only_dossier=args.dossier)
    except Exception as exc:
        print(f"✗ Erreur inattendue : {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
