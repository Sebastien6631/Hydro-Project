#!/usr/bin/env python3
"""Point d'entrée CLI -- previ-R2-D2.

Usage :
    python run.py --train --dossier nancy_ruedaum_G1 --horizon 8
    python run.py --train --all-dossiers
"""

import sys

from previ_r2d2.cli import main

if __name__ == "__main__":
    sys.exit(main())
