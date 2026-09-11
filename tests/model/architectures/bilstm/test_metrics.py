from __future__ import annotations

import numpy as np
import pytest
import torch

from projet_hydro.model.architectures.bilstm.metrics import kge_loss_torch, kge_numpy
from projet_hydro.model.architectures.lightgbm.metrics import kge_loss


def test_kge_loss_torch_1d_matches_lightgbm_kge_loss_for_same_data():
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8])

    loss = kge_loss_torch(y_pred, y_true)

    # cas bien conditionné (pas de variance quasi-nulle) -> quasi identique à kge_loss (LightGBM).
    expected = kge_loss(y_true.numpy(), y_pred.numpy())
    assert loss.shape == (1,)
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_kge_loss_torch_multi_horizon_averages_over_h_steps():
    y_true = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]])
    y_pred = torch.tensor([[1.1, 2.2], [2.1, 4.2], [2.9, 5.8], [4.2, 8.4], [4.8, 9.6]])

    loss = kge_loss_torch(y_pred, y_true)

    assert loss.shape == (1,)
    assert torch.isfinite(loss)


def test_kge_loss_torch_is_differentiable():
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8], requires_grad=True)

    loss = kge_loss_torch(y_pred, y_true)
    loss.backward()

    assert y_pred.grad is not None


def test_kge_loss_torch_near_zero_for_near_perfect_prediction():
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.clone() + 1e-4

    loss = kge_loss_torch(y_pred, y_true)

    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_kge_numpy_delegates_to_lightgbm_kge_loss():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    result = kge_numpy(y_true, y_pred)

    assert result == pytest.approx(1.0 - kge_loss(y_true, y_pred))
