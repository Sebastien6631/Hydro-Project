#!/usr/bin/env python3
"""maj-puissance — importe la puissance des raccordements depuis hydrospot_stream.

Pour chaque raccordement de centrales/REFERENCE/config-general.json :
  - trouve le dossier hydrospot_stream correspondant (mot-clé centrale + groupe) ;
  - fusionne ses 3 fichiers bruts en puissance.csv sur le NAS (source de vérité) ;
  - crée/rafraîchit le symlink local centrales/<dossier>/puissance.csv.

Usage :
    python cron/scripts/maj-puissance.py --dossier apas_G1_G4   # test ciblé
    python cron/scripts/maj-puissance.py                        # tous les raccordements
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from previ_r2d2.common import config, daily_report, dvc_markers
from previ_r2d2.preprocessing.debit import station_store
from previ_r2d2.preprocessing.puissance import puissance_store

FILENAME = "puissance.csv"
FILENAME_HORAIRE = "puissance_horaire.csv"


def run(config_path: Path | None = None, only_dossier: str | None = None) -> int:
    cfg = config_path or config.REFERENCE_DIR / "config-general.json"

    if not cfg.exists():
        msg = f"{cfg} introuvable — lance d'abord `python cron/scripts/majdata-memo.py`."
        print(f"✗ {msg}", file=sys.stderr)
        daily_report.record("maj-puissance", "[previ-record] maj-puissance — erreur", msg, has_errors=True)
        return 1

    raccordements = json.loads(cfg.read_text(encoding="utf-8"))
    print(f"→ {len(raccordements)} raccordement(s) au total.")

    mapping = puissance_store.load_puissance_mapping(config.ROOT / "config" / "puissance_mapping.yaml")

    done: list[tuple] = []
    errors: list[str] = []

    for rec in raccordements:
        dossier = rec.get("dossier")
        if not dossier:
            continue
        if only_dossier and dossier != only_dossier:
            continue

        try:
            source_folder = puissance_store.find_source_folder(rec, mapping=mapping)
            nas_path = config.NAS_DATA_ROOT / dossier / FILENAME
            n = puissance_store.export_puissance_csv(source_folder, nas_path)
            station_store.ensure_local_symlink(nas_path, dossier, FILENAME)
            nas_horaire_path = config.NAS_DATA_ROOT / dossier / FILENAME_HORAIRE
            station_store.ensure_local_symlink(nas_horaire_path, dossier, FILENAME_HORAIRE)
            done.append((dossier, source_folder, n))
            print(f"• {dossier} <- {source_folder} — {n} lignes")
        except Exception as exc:
            msg = f"{dossier} : {exc}"
            print(f"  ✗ {msg}", file=sys.stderr)
            errors.append(msg)

    print(f"\n→ Bilan : {len(done)} traité(s), {len(errors)} erreur(s).")

    # Journalisé pour le digest quotidien (daily-sync-report.py) plutôt qu'un
    # mail propre à ce script -- un seul mail par jour pour debit+puissance.
    subject = f"[previ-record] maj-puissance — {len(done)} ok, {len(errors)} erreur(s)"
    lines = [f"{d} <- {s} : {n} lignes" for d, s, n in done]
    if errors:
        lines += ["", f"ERREURS ({len(errors)}) :"] + [f"  - {e}" for e in errors]
    daily_report.record("maj-puissance", subject, "\n".join(lines), has_errors=bool(errors))
    dvc_markers.write("puissance")

    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="maj-puissance",
        description="Importe la puissance des raccordements depuis hydrospot_stream.",
    )
    parser.add_argument("--config", metavar="FICHIER", help="Chemin de config-general.json.")
    parser.add_argument(
        "--dossier", default=None, metavar="NOM", help="Ne traite que ce raccordement."
    )
    args = parser.parse_args(argv)
    cfg = Path(args.config) if args.config else None

    try:
        return run(config_path=cfg, only_dossier=args.dossier)
    except Exception as exc:
        print(f"✗ Erreur inattendue : {exc}", file=sys.stderr)
        traceback.print_exc()
        daily_report.record(
            "maj-puissance", "[previ-record] maj-puissance — erreur inattendue",
            f"Erreur inattendue : {exc}", has_errors=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
