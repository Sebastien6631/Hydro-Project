"""Sortie console en UTF-8, quelle que soit la console.

Sur Windows, `sys.stdout.encoding` vaut cp1252 : le premier caractère hors de
ce jeu (`—`, `→`, `✗`, `•`, tous présents dans les messages des scripts) lève
`UnicodeEncodeError` et tue le script AVANT qu'il ait rien fait. Rencontré en
vrai : `maj-data.py` mourait sur son propre `print` d'en-tête, donc aucun débit
n'était jamais importé.

Corriger les caractères un par un ne tiendrait pas -- le prochain message
accentué venu réintroduirait la panne. On force l'encodage du flux à la source.
"""

from __future__ import annotations

import sys


def force_utf8() -> None:
    """À appeler en tête de `main()` d'un script à sortie texte."""
    for flux in (sys.stdout, sys.stderr):
        reconfigure = getattr(flux, "reconfigure", None)
        if reconfigure is not None:  # absent si le flux est redirigé vers un objet custom
            reconfigure(encoding="utf-8", errors="replace")
