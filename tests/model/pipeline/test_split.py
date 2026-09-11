from __future__ import annotations

import pytest

from projet_hydro.model.pipeline.split import train_val_test_indices


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


@pytest.mark.parametrize("n", [0, 1, 5, 11, 37, 999])
def test_slices_always_partition_n_exactly(n):
    fit_s, val_s, test_s = train_val_test_indices(n)

    assert sum(_sizes((fit_s, val_s, test_s))) == n
    assert all(s.stop >= s.start for s in (fit_s, val_s, test_s))
