from __future__ import annotations

import numpy as np
import torch

from projet_hydro.model.architectures.bilstm.model import BiLSTMHydro
from projet_hydro.model.seeding import DEFAULT_SEED, ENV_VAR, set_seeds


def _premiers_poids(seed):
    set_seeds(seed, log=False)
    return next(BiLSTMHydro(n_features=6, horizon=4).parameters()).detach().clone()


def test_same_seed_gives_identical_weight_initialisation():
    """C'est LA propriété qui rend deux entraînements comparables : sans elle,
    le BiLSTM variait de ~0.14 KGE d'un run à l'autre à configuration identique,
    ce qui noyait l'effet de tout changement qu'on cherchait à mesurer."""
    assert torch.equal(_premiers_poids(7), _premiers_poids(7))


def test_different_seeds_give_different_initialisation():
    """Garde-fou du test précédent : s'il passait aussi avec des graines
    différentes, c'est que la fixture ne mesure rien."""
    assert not torch.equal(_premiers_poids(7), _premiers_poids(8))


def test_seeds_cover_numpy_and_random_not_only_torch():
    import random

    set_seeds(3, log=False)
    tirages = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seeds(3, log=False)

    assert tirages == (random.random(), np.random.rand(), torch.rand(1).item())


def test_env_var_overrides_the_default_seed(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "123")

    assert set_seeds(log=False) == 123


def test_explicit_argument_wins_over_the_env_var(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "123")

    assert set_seeds(999, log=False) == 999


def test_unparseable_env_var_falls_back_to_default_instead_of_crashing(monkeypatch, caplog):
    """Une graine illisible ne doit pas faire échouer un entraînement de plusieurs
    heures : on prévient et on continue."""
    monkeypatch.setenv(ENV_VAR, "quarante-deux")

    with caplog.at_level("WARNING"):
        seed = set_seeds(log=False)

    assert seed == DEFAULT_SEED
    assert "n'est pas un entier" in caplog.text
