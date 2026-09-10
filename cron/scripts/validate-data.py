#!/usr/bin/env python3
"""validate-data — vérifie le contrat de données de chaque data_preparation.csv.

Stage DVC `dvc/preprocessing/dvc.yaml:validate` (après data_preparation).
Une erreur de contrat -> exit 1 (bloque le pipeline). Un warning -> loggé.
Écrit un rapport JSON par centrale sous logs/validation/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from previ_r2d2.common import config
from previ_r2d2.common.dvc_markers import write as write_marker
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv
from previ_r2d2.preprocessing.data_preparation.validation import validate_data_preparation

logger = logging.getLogger("validate-data")
REPORT_DIR = config.ROOT / "logs" / "validation"


def run(only_dossier: str | None = None, strict: bool = False) -> int:
    paths = sorted(config.CENTRALES_DIR.glob("*/data_preparation.csv"))
    if only_dossier:
        paths = [p for p in paths if p.parent.name == only_dossier]
    if not paths:
        logger.warning("aucun data_preparation.csv trouvé — données non récupérées ? (`dvc pull`)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    had_error = False
    for path in paths:
        dossier = path.parent.name
        rep = validate_data_preparation(read_data_preparation_csv(path), dossier, strict=strict)
        (REPORT_DIR / f"{dossier}.json").write_text(
            json.dumps(
                {"dossier": dossier, "ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        for w in rep.warnings:
            logger.warning("%s : %s", dossier, w)
        for e in rep.errors:
            logger.error("%s : %s", dossier, e)
        if rep.ok:
            logger.info("%s : contrat OK (%d avertissement(s))", dossier, len(rep.warnings))
        else:
            had_error = True

    write_marker("validate")
    return 1 if had_error else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Validation du contrat de données data_preparation.csv.")
    parser.add_argument("--dossier", default=None, help="Ne valider qu'un dossier.")
    parser.add_argument("--strict", action="store_true", help="Traiter les avertissements comme des erreurs.")
    args = parser.parse_args(argv)
    return run(only_dossier=args.dossier, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
