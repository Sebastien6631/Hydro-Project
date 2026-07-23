"""Métriques BiLSTM (meta) -- port fidèle de bilstm_hydro.py:21-53
(Previ_v2). Pas de persistance : fonctions de calcul pur. kge_loss_torch
(différentiable) diverge intentionnellement de kge_loss (LightGBM) sur la
gestion de la variance quasi-nulle (clamp plutôt que branche explicite) et
un epsilon supplémentaire sous la racine (+1e-8) -- besoins réels de
différentiabilité, pas des bugs. kge_numpy réutilise kge_loss directement,
aucune duplication de formule."""

from __future__ import annotations

import numpy as np
import torch

from previ_r2d2.model.architectures.lightgbm.metrics import kge_loss


def kge_loss_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """1 - KGE différentiable, moyennée sur les H pas (y_pred/y_true en espace log1p)."""
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(1)
        y_true = y_true.unsqueeze(1)
    H = y_pred.shape[1]
    total = torch.zeros(1, device=y_pred.device, dtype=y_pred.dtype)
    for h in range(H):
        yp = y_pred[:, h]
        yt = y_true[:, h]
        mu_p = yp.mean()
        mu_t = yt.mean()
        sig_p = yp.std(unbiased=False).clamp(min=1e-6)
        sig_t = yt.std(unbiased=False).clamp(min=1e-6)
        r = ((yp - mu_p) * (yt - mu_t)).mean() / (sig_p * sig_t)
        r = r.clamp(-1.0, 1.0)
        alpha = sig_p / sig_t
        beta = mu_p / (mu_t.abs() + 1e-6)
        kge = 1.0 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 + 1e-8)
        total = total + (1.0 - kge)
    return total / H


def kge_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """KGE scalaire en espace m³/s (utilisé pour l'early stopping) -- réutilise kge_loss (LightGBM)."""
    return 1.0 - kge_loss(y_true, y_pred)
