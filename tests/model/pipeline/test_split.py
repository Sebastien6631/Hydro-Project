from __future__ import annotations

import pytest

from previ_r2d2.model.pipeline.split import train_val_test_indices


def _sizes(slices):
    return tuple(s.stop - s.start for s in slices)


def test_default_split_is_80_10_10_and_covers_everything_without_overlap():
    fit_s, val_s, test_s = train_val_test_indices(1000)

    assert _sizes((fit_s, val_s, test_s)) == (800, 100, 100)
    # contiguë et chronologique : fit précède val, qui précède test
    assert fit_s.start == 0
    assert fit_s.stop == val_s.start
    assert val_s.stop == test_s.start
    assert test_s.stop == 1000


def test_split_is_positional_not_shuffled_so_test_is_always_the_most_recent_slice():
    """La série est temporelle : mélanger ferait fuiter le futur dans le fit."""
    fit_s, val_s, test_s = train_val_test_indices(500)

    assert list(range(500))[test_s] == list(range(450, 500))
    assert list(range(500))[fit_s][-1] < list(range(500))[val_s][0]


def test_min_val_floor_eats_into_fit_not_into_test():
    plain_fit, _, plain_test = train_val_test_indices(300)
    fit_s, val_s, test_s = train_val_test_indices(300, min_val=100)

    assert val_s.stop - val_s.start == 100
    assert test_s.stop - test_s.start == plain_test.stop - plain_test.start
    assert fit_s.stop - fit_s.start < plain_fit.stop - plain_fit.start


def test_floor_larger_than_half_is_capped_so_fit_never_empties():
    """Piège réel : un plancher plat de 200 lignes sur un petit dossier journalier
    réduisait `fit` à 0 -> crash DataLoader (num_samples=0) sur un vrai
    entraînement. Le plafond n//2 est le filet de sécurité générique."""
    n = 100
    fit_s, val_s, test_s = train_val_test_indices(n, min_val=10_000)

    assert val_s.stop - val_s.start == n // 2
    assert fit_s.stop - fit_s.start > 0
    assert _sizes((fit_s, val_s, test_s))[0] + val_s.stop - val_s.start + test_s.stop - test_s.start == n


@pytest.mark.parametrize("n", [0, 1, 5, 11, 37, 999])
def test_slices_always_partition_n_exactly(n):
    fit_s, val_s, test_s = train_val_test_indices(n)

    assert sum(_sizes((fit_s, val_s, test_s))) == n
    assert all(s.stop >= s.start for s in (fit_s, val_s, test_s))
