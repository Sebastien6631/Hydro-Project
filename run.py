#!/usr/bin/env python3
"""Point d'entrée CLI -- projet_hydro.

Usage :
    python run.py --train --dossier touzac_g2_G2 --horizon 8
    python run.py --train --all-dossiers
"""

import sys

from projet_hydro.cli import main

if __name__ == "__main__":
    sys.exit(main())
