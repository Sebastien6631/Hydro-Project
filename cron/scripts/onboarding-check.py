#!/usr/bin/env python3
"""onboarding-check — stage DVC dvc/preprocessing/dvc.yaml:onboarding_check.

Tourne après `debit` (dépendance sur son marqueur, cf.
dvc/preprocessing/dvc.yaml). Pour chaque raccordement sans bv.json (pas
encore onboardé), valide config-raccordement.json et journalise/logge les
infos manquantes -- idempotent (peut être relancé sans risque). Note :
`config-general.json` est désormais une donnée statique versionnée DVC
(plus de re-fetch quotidien type `memorandum`/OneGate dans cette version),
donc il n'y a plus de mécanisme qui "complète" automatiquement un
raccordement incomplet d'un jour à l'autre -- un raccordement signalé
incomplet ici le reste tant que `config-general.json`/
`config-raccordement.json` n'est pas corrigé manuellement.
"""

from __future__ import annotations

import json
import logging
import sys

from previ_r2d2.common.console import force_utf8
from previ_r2d2.common import config
from previ_r2d2.common.dvc_markers import write as write_marker
from previ_r2d2.preprocessing.onboarding.validation import missing_fields

logger = logging.getLogger("onboarding-check")


def load_records() -> list[dict]:
    general = config.REFERENCE_DIR / "config-general.json"
    if not general.exists():
        return []
    return json.loads(general.read_text(encoding="utf-8"))


def run() -> int:
    lines = []
    for rec in load_records():
        dossier = rec.get("dossier")
        try:
            if dossier is not None and (config.CENTRALES_DIR / dossier / "bv.json").exists():
                continue
            missing = missing_fields(rec)
        except Exception as exc:
            logger.error("Échec validation raccordement %r : %s", rec, exc, exc_info=True)
            lines.append(f"{dossier or '?'} : enregistrement invalide ({type(exc).__name__}: {exc})")
            continue
        if missing:
            lines.append(f"{dossier} : {', '.join(missing)}")
        else:
            lines.append(f"{dossier} : complet, prêt pour bv")

    body = "\n".join(lines) if lines else "Aucun nouveau raccordement à valider."
    logger.info(body)
    write_marker("onboarding")
    return 0


def main() -> int:
    force_utf8()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    )
    return run()


if __name__ == "__main__":
    sys.exit(main())
