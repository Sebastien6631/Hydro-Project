"""Découpe chronologique fit/val/test partagée par les modèles entraînés.

Factorisé pour que LightGBM (`fit_final`) et le meta-learner (`fit_stacking`)
utilisent exactement la même convention : découpe POSITIONNELLE, sans shuffle
(les données sont une série temporelle -- mélanger ferait fuiter le futur).

- **fit** : données d'entraînement réelles du modèle déployé ;
- **val** : early stopping / mesure hors échantillon ;
- **test** : jamais vu pendant le fit.
"""

from __future__ import annotations

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10


def train_val_test_indices(n: int) -> tuple[slice, slice, slice]:
    """Découpe `n` lignes chronologiques en (fit, val, test) -- 80/10/10."""
    n_fit = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return slice(0, n_fit), slice(n_fit, n_fit + n_val), slice(n_fit + n_val, n)
