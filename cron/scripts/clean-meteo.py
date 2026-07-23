#!/usr/bin/env python3
"""clean-meteo — supprime les fichiers météo NWP grande échéance des jours passés.

Conserve le jour courant et les échéances courtes (<24h) ; supprime les
échéances entre 24h et 360h pour tous les jours passés de l'année donnée.

Usage :
    python cron/scripts/clean-meteo.py
    python cron/scripts/clean-meteo.py --annee 2025
    python cron/scripts/clean-meteo.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import sys

from previ_r2d2.common import config
from previ_r2d2.preprocessing.meteo.retention import clean_year


def human_size(n: int) -> str:
    if n < 1024**2:
        return f"{n / 1024:.0f} Ko"
    if n < 1024**3:
        return f"{n / 1024**2:.2f} Mo"
    return f"{n / 1024**3:.2f} Go"


def run(annee: str, dry_run: bool) -> int:
    annee_dir = config.NAS_METEO / annee
    if not annee_dir.exists():
        print(f"✗ {annee_dir} introuvable.", file=sys.stderr)
        return 1

    today = datetime.date.today().strftime("%Y%m%d")
    targets, total_size = clean_year(annee_dir, today, dry_run=dry_run)

    if dry_run:
        print(f"[TEST] {len(targets)} fichier(s) seraient supprimés ({human_size(total_size)} récupérables).")
    else:
        print(f"✓ {len(targets)} fichier(s) supprimé(s) ({human_size(total_size)} libéré(s)).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Nettoie les fichiers météo NWP grande échéance des jours passés."
    )
    parser.add_argument("--annee", default=datetime.date.today().strftime("%Y"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run(args.annee, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
