from __future__ import annotations

import numpy as np
import pytest

from previ_r2d2.model.architectures.lightgbm.metrics import kge_loss, debit_quantiles
from previ_r2d2.model.architectures.lightgbm.metrics import debit_weights


def test_kge_loss_normal_variance_matches_hand_computed_value():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    result = kge_loss(y_true, y_pred)

    assert result == pytest.approx(0.046351372458186435)


def test_kge_loss_near_zero_variance_uses_bias_only_penalty():
    y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 5.0, 5.0, 5.0, 6.0])

    result = kge_loss(y_true, y_pred)

    assert result == pytest.approx(1.0399997920000417)


def test_kge_loss_ignores_nan_pairs_instead_of_propagating_nan():
    # 1 point sur 5 a une cible manquante (trou capteur) -- ne doit pas
    # contaminer std/mean sur les 4 points valides (cf. np.std/np.mean qui
    # renvoient NaN dès qu'un seul élément du tableau est NaN).
    y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 9.9, 4.2, 4.8])

    result = kge_loss(y_true, y_pred)

    valid = np.array([True, True, False, True, True])
    expected = kge_loss(y_true[valid], y_pred[valid])
    assert result == pytest.approx(expected)
    assert not np.isnan(result)


def test_debit_quantiles_normal_regime_uses_q50_start():
    y_orig = np.concatenate([np.full(90, 1.0), np.full(9, 2.0), np.full(1, 2.5)])

    result = debit_quantiles(y_orig)

    assert result == pytest.approx((50, 1.0, 1.1000000000000085, 2.0050000000000026))


def test_debit_quantiles_moderate_extreme_ratio_uses_q25_start():
    y_orig = np.concatenate([np.full(98, 1.0), np.full(1, 2.0), np.full(1, 5.0)])

    result = debit_quantiles(y_orig)

    assert result == pytest.approx((25, 1.0, 1.0, 2.0300000000000153))


def test_debit_quantiles_extreme_ratio_uses_q15_start():
    y_orig = np.concatenate([np.full(98, 1.0), np.full(1, 1.5), np.full(1, 20.0)])

    result = debit_quantiles(y_orig)

    assert result == pytest.approx((15, 1.0, 1.0, 1.6850000000000946))


def test_debit_quantiles_uses_y_ref_for_start_pct_but_not_for_q90_q99():
    y_orig = np.concatenate([np.full(90, 1.0), np.full(9, 2.0), np.full(1, 2.5)])
    y_ref = np.concatenate([np.full(98, 1.0), np.full(1, 1.5), np.full(1, 20.0)])

    result = debit_quantiles(y_orig, y_ref=y_ref)

    # q_start_pct=15 vient de y_ref (ratio extrême), q90/q99 restent ceux de y_orig.
    assert result == pytest.approx((15, 1.0, 1.1000000000000085, 2.0050000000000026))


def test_debit_weights_matches_hand_computed_values_across_all_zones():
    y_orig = np.arange(1.0, 101.0)  # thresholds: q_start=50.5, q90=90.1, q99=99.01

    weights = debit_weights(y_orig, mult_poids=2.0)

    # index 9 -> y=10 (régime normal, <= q_start)
    assert weights[9] == pytest.approx(1.2)
    # index 69 -> y=70 (débits moyens-forts, q_start < y <= q90)
    assert weights[69] == pytest.approx(1.984848459978574)
    # index 94 -> y=95 (hautes eaux, q90 < y <= q99)
    assert weights[94] == pytest.approx(3.199775286220504)
    # index 99 -> y=100 (crue exceptionnelle, y > q99)
    assert weights[99] == pytest.approx(2.3333331836887496)



