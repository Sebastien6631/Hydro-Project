"""Choix du device torch, partagé par l'entraînement et la prédiction.

Un seul endroit décide, pour que `run_training` et `run_prediction` ne puissent
pas diverger : un modèle laissé sur `cuda` après l'entraînement et servi par un
chemin qui construit ses tenseurs sur CPU lève
`RuntimeError: Input and parameter tensors are not at the same device`.

Détection automatique, surchargeable par la variable d'environnement
`PREVI_DEVICE` (`cpu` ou `cuda`) — utile pour forcer le CPU quand le GPU est
occupé, ou pour reproduire un résultat à l'identique.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENV_VAR = "PREVI_DEVICE"


def resolve_device(log: bool = True) -> torch.device:
    """Retourne le device à utiliser : `PREVI_DEVICE` s'il est posé, sinon
    `cuda` si un GPU utilisable est présent, sinon `cpu`."""
    demande = (os.environ.get(ENV_VAR) or "").strip().lower()
    cuda_ok = torch.cuda.is_available()

    if demande in ("cpu", "cuda"):
        if demande == "cuda" and not cuda_ok:
            logger.warning(
                "%s=cuda demandé mais aucun GPU utilisable (torch %s, cuda compilée=%s) -- repli sur CPU.",
                ENV_VAR, torch.__version__, torch.version.cuda,
            )
            device = torch.device("cpu")
        else:
            device = torch.device(demande)
    elif demande:
        logger.warning("%s=%r non reconnu (attendu 'cpu' ou 'cuda') -- détection automatique.", ENV_VAR, demande)
        device = torch.device("cuda" if cuda_ok else "cpu")
    else:
        device = torch.device("cuda" if cuda_ok else "cpu")

    if log:
        if device.type == "cuda":
            nom = torch.cuda.get_device_name(device)
            vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
            logger.info("Device torch : %s (%s, %.1f Go, torch %s)", device, nom, vram, torch.__version__)
        else:
            # `torch.version.cuda is None` = build CPU-only : aucun kernel CUDA
            # compilé, le GPU physique resterait invisible quoi qu'il arrive.
            detail = "build CPU-only" if torch.version.cuda is None else f"cuda {torch.version.cuda} indisponible"
            logger.info("Device torch : cpu (%s, torch %s)", detail, torch.__version__)
    return device
