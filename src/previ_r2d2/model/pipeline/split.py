"""Découpe chronologique fit/val/test partagée par les modèles entraînés.

Factorisé pour que LightGBM (`fit_final`) et le meta-learner (`fit_stacking`)
utilisent exactement la même convention : découpe POSITIONNELLE, sans shuffle
(les données sont une série temporelle -- mélanger ferait fuiter le futur).

- **fit** : données d'entraînement réelles du modèle déployé ;
- **val** : early stopping / calibration d'hyperparamètre ;
- **test** : jamais vu pendant le fit ni la calibration.
"""

from __future__ import annotations


def train_val_test_indices(
    n: int,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    min_val: int | None = None,
    min_test: int | None = None,
) -> tuple[slice, slice, slice]:
    """Découpe `n` lignes chronologiques en (fit, val, test) -- 80/10/10 par défaut.

    `min_val`/`min_test` garantissent un plancher de lignes en rognant sur `fit`.
    Chaque plancher est plafonné à `n // 2` pour ne jamais vider `fit` à lui seul
    (le plafond s'applique à chaque plancher individuellement, pas à leur somme --
    aucun appelant ne combine aujourd'hui les deux avec des valeurs assez grandes
    pour que ça compte).
    """
    n_val = int(n * val_frac)
    n_test = n - int(n * train_frac) - n_val

    if min_val is not None:
        n_val = max(n_val, min(min_val, n // 2))
    if min_test is not None:
        n_test = max(n_test, min(min_test, n // 2))

    n_fit = max(n - n_val - n_test, 0)
    return slice(0, n_fit), slice(n_fit, n_fit + n_val), slice(n_fit + n_val, n)
