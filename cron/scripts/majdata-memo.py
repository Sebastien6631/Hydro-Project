#!/usr/bin/env python3
"""majdata-memo — génère la config des centrales flexibles depuis le memorandum.

Étapes :
  1. Appelle `/hydrogrid/memorandum/select` (ou lit un fichier JSON via --input).
  2. Aplatit l'arbre en raccordements et ne garde que ceux avec `rte = true`.
  3. Écrit `config-general.json` à la racine du dossier de sortie (liste à plat).
  4. Crée un dossier par raccordement (nom = nom_centrale + noms des groupes,
     espaces -> `_`) contenant son `config-raccordement.json`.

Usage :
    python cron/scripts/majdata-memo.py --check          # teste la connexion API
    python cron/scripts/majdata-memo.py                  # appel API réel -> centrales/
    python cron/scripts/majdata-memo.py --input data.json  # depuis un fichier local
    python cron/scripts/majdata-memo.py --output /chemin   # dossier de sortie
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from previ_r2d2.common import config, mailer
from previ_r2d2.common.onegate import OneGateClient, OneGateError
from previ_r2d2.preprocessing.onegate.memorandum import (
    SELECT_KEYS,
    deduplicate_dossiers,
    flatten_flexibilite,
)


def check_connexion(client: OneGateClient) -> int:
    """Vérifie que l'API répond et que l'authentification est acceptée."""
    if not client.is_configured:
        print(
            "⚠️  Jeton manquant (PREVI_BEARER_TOKEN). "
            "Renseigne-le dans src/previ_r2d2/common/secret_config.py avant d'appeler l'API.",
            file=sys.stderr,
        )
        return 1

    print(f"→ Test de connexion via GET /hydrogrid/memorandum/select ...")
    try:
        tree = client.select(SELECT_KEYS)
    except OneGateError as exc:
        print(f"✗ Échec ({exc})", file=sys.stderr)
        return 1
    except Exception as exc:  # réseau, DNS, timeout…
        print(f"✗ Erreur réseau : {exc}", file=sys.stderr)
        return 1

    nb = len(tree) if isinstance(tree, list) else 0
    print(f"✓ Connexion OK — {nb} aménagement(s) récupéré(s).")
    return 0


def _fetch_tree(client: OneGateClient, input_file: str | None):
    """Récupère l'arbre memorandum, depuis un fichier local ou l'API."""
    if input_file:
        path = Path(input_file)
        print(f"→ Lecture des données depuis {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    if not client.is_configured:
        raise RuntimeError(
            "Jeton manquant (PREVI_BEARER_TOKEN). "
            "Renseigne-le dans src/previ_r2d2/common/secret_config.py ou utilise --input."
        )
    print("→ Appel GET /hydrogrid/memorandum/select ...")
    return client.select(SELECT_KEYS)


def _write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_maj(
    client: OneGateClient,
    input_file: str | None = None,
    output_dir: str | None = None,
) -> int:
    """Génère config-general.json + un dossier/config par raccordement flexible."""
    out = Path(output_dir) if output_dir else config.ROOT / "centrales"

    try:
        tree = _fetch_tree(client, input_file)
    except Exception as exc:
        label = "Échec de l'appel API" if isinstance(exc, OneGateError) else "Erreur"
        print(f"✗ {label} : {exc}", file=sys.stderr)
        # Un seul mail d'erreur pour ce script (sauf en mode fichier local).
        if not input_file:
            mailer.notify_errors("majdata-memo", [f"{label} : {exc}"])
        return 1

    # Aplatissement + filtre rte=true.
    records = flatten_flexibilite(tree, rte_only=True)
    deduplicate_dossiers(records)
    print(f"→ {len(records)} raccordement(s) flexible(s) (rte=true) retenu(s).")

    if not records:
        print("ℹ️  Aucun raccordement flexible : rien à générer.")
        return 0

    # config-general.json sous REFERENCE/ (fichiers partagés du dossier de sortie).
    reference_dir = out / "REFERENCE"
    reference_dir.mkdir(parents=True, exist_ok=True)
    general = reference_dir / "config-general.json"
    _write_json(general, records)
    print(f"✓ {general}")

    # Un dossier + config-raccordement.json par raccordement.
    for rec in records:
        folder = out / rec["dossier"]
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "config-raccordement.json"
        _write_json(target, rec)
        print(f"  ✓ {rec['dossier']}/config-raccordement.json")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="majdata-memo",
        description="Génère la config des centrales flexibles depuis le memorandum.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Teste uniquement la connexion à l'API et sort.",
    )
    parser.add_argument(
        "--input",
        metavar="FICHIER",
        help="Lit l'arbre memorandum depuis un fichier JSON au lieu de l'API.",
    )
    parser.add_argument(
        "--output",
        metavar="DOSSIER",
        help="Dossier de sortie (défaut : centrales/).",
    )
    args = parser.parse_args(argv)

    client = OneGateClient(
        base_url=config.ONEGATE_BASE_URL,
        token=config.ONEGATE_TOKEN,
    )

    if args.check:
        return check_connexion(client)
    return run_maj(client, input_file=args.input, output_dir=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
