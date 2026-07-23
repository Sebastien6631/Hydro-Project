#!/usr/bin/env python3
"""onboarding-bv — caractérise le bassin versant de chaque centrale.

Pour chaque centrale de `centrales/` (regroupées par site physique quand
plusieurs dossiers partagent le même `centrale_uuid`), mesure son bassin
versant amont — via son shapefile connu (`config/bv_mapping.yaml`,
`centrales/REFERENCE/shapefiles/`) quand il existe, sinon par délimitation
MNT depuis son point exutoire — et écrit `centrales/<dossier>/bv.json`.

Usage :
    python cron/scripts/onboarding-bv.py batch --mnt "$PREVI_MNT"
    python cron/scripts/onboarding-bv.py batch --mnt "$PREVI_MNT" --force
    python cron/scripts/onboarding-bv.py single --dossier apas_G1_G4 --mnt "$PREVI_MNT"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from previ_r2d2.common import config, dvc_markers
from previ_r2d2.preprocessing.bv import bv_builder
from previ_r2d2.preprocessing.bv.rules import load_rules
from previ_r2d2.preprocessing.debit.hubeau import HubEauClient

logger = logging.getLogger("onboarding-bv")


def _default_rules_path() -> Path:
    return config.REFERENCE_DIR / "bv_rules.json"


def _default_mapping_path() -> Path:
    return config.ROOT / "config" / "bv_mapping.yaml"


def _default_shapefiles_dir() -> Path:
    return config.REFERENCE_DIR / "shapefiles"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Caractérisation automatique des bassins versants.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mnt", type=Path, default=None,
                         help="MNT France entière (défaut : $PREVI_MNT).")
    common.add_argument("--rules", type=Path, default=None,
                         help="bv_rules.json (défaut : centrales/REFERENCE/bv_rules.json).")
    common.add_argument("--mapping", type=Path, default=None,
                         help="bv_mapping.yaml (défaut : config/bv_mapping.yaml).")
    common.add_argument("--shapefiles-dir", type=Path, default=None,
                         help="Défaut : centrales/REFERENCE/shapefiles/.")
    common.add_argument("--centrales-dir", type=Path, default=None,
                         help="Défaut : centrales/ à la racine du projet.")

    batch = sub.add_parser("batch", parents=[common], help="Traite toutes les centrales.")
    batch.add_argument("--force", action="store_true",
                        help="Recalcule même si bv.json existe déjà.")

    single = sub.add_parser("single", parents=[common], help="Traite une seule centrale.")
    single.add_argument("--dossier", required=True)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                         format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
    args = build_parser().parse_args(argv)

    # config.PREVI_MNT = Path(_get("PREVI_MNT", "")) : quand la variable n'est
    # pas définie, Path("") se normalise en Path(".") (répertoire courant),
    # qui "existe" toujours — sans ce garde-fou explicite, une absence de
    # config passerait le check silencieusement au lieu d'échouer proprement.
    mnt_path = args.mnt or config.PREVI_MNT
    if not mnt_path or str(mnt_path) == "." or not Path(mnt_path).exists():
        logger.error("MNT introuvable : %s (définir --mnt ou $PREVI_MNT).", mnt_path)
        return 1

    rules_path = args.rules or _default_rules_path()
    mapping_path = args.mapping or _default_mapping_path()
    shapefiles_dir = args.shapefiles_dir or _default_shapefiles_dir()
    centrales_dir = args.centrales_dir or (config.ROOT / "centrales")
    rules = load_rules(rules_path)
    mapping = bv_builder.load_bv_mapping(mapping_path)
    hubeau_client = HubEauClient()
    data_dir = config.CENTRALES_DIR

    if args.cmd == "batch":
        result = bv_builder.run_batch(centrales_dir, Path(mnt_path), rules, mapping,
                                       shapefiles_dir, force=args.force,
                                       hubeau_client=hubeau_client, data_dir=data_dir)
        dvc_markers.write("bv")
        return 1 if result["errors"] else 0

    if args.cmd == "single":
        try:
            bv_builder.run_single(centrales_dir, args.dossier, Path(mnt_path), rules,
                                   mapping, shapefiles_dir,
                                   hubeau_client=hubeau_client, data_dir=data_dir)
        except Exception as exc:
            logger.error("%s : échec de la mesure (%s)", args.dossier, exc)
            return 1
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
