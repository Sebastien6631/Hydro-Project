"""Graines aléatoires, pour que deux entraînements identiques soient comparables.

Sans ça, le BiLSTM varie de ~0.14 KGE d'un run à l'autre à configuration
identique (mesuré le 2026-09-09 : apas_G1_G4 0.8392 / 0.8393 / 0.6973). Cette
variance dépasse l'effet de la plupart des changements qu'on cherche à évaluer,
donc aucune comparaison mono-run n'est interprétable tant qu'elle subsiste.

Trois sources d'aléa sont couvertes : l'initialisation des poids torch, le
mélange des batches (`DataLoader(shuffle=True)`) et les tirages numpy/random.

Surchargeable par `PREVI_SEED` pour rejouer un run précis ou, au contraire,
répéter volontairement un entraînement avec plusieurs graines afin de MESURER
la variance résiduelle plutôt que de la subir.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

ENV_VAR = "PREVI_SEED"
DEFAULT_SEED = 42


def set_seeds(seed: int | None = None, log: bool = True) -> int:
    """Fixe les graines random/numpy/torch (CPU et CUDA). Retourne la graine utilisée.

    Ne force PAS `torch.use_deterministic_algorithms(True)` : les noyaux LSTM de
    cuDNN n'ont pas d'implémentation déterministe et lèveraient une erreur. Il
    subsiste donc un écart d'arrondi de l'ordre de 1e-7 entre deux exécutions
    GPU, sans commune mesure avec la variance que ces graines suppriment.
    """
    if seed is None:
        brut = os.environ.get(ENV_VAR)
        try:
            seed = int(brut) if brut else DEFAULT_SEED
        except ValueError:
            logger.warning("%s=%r n'est pas un entier -- graine par défaut %d.", ENV_VAR, brut, DEFAULT_SEED)
            seed = DEFAULT_SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op sans GPU

    if log:
        logger.info("Graines fixées à %d (surchargeable par %s).", seed, ENV_VAR)
    return seed
