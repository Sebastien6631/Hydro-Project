from __future__ import annotations

import torch

from projet_hydro.model.device import ENV_VAR, resolve_device


def test_auto_detects_cpu_when_no_usable_gpu(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device().type == "cpu"


def test_auto_detects_cuda_when_a_gpu_is_usable(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device(log=False).type == "cuda"


def test_env_var_can_force_cpu_even_with_a_gpu_present(monkeypatch):
    """Échappatoire réelle : GPU occupé par autre chose, ou besoin de reproduire
    un résultat CPU à l'identique."""
    monkeypatch.setenv(ENV_VAR, "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device().type == "cpu"


def test_forcing_cuda_without_a_gpu_falls_back_to_cpu_instead_of_crashing(monkeypatch, caplog):
    monkeypatch.setenv(ENV_VAR, "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with caplog.at_level("WARNING"):
        device = resolve_device()

    assert device.type == "cpu"
    assert "repli sur CPU" in caplog.text


def test_unrecognised_value_warns_and_falls_back_to_auto_detection(monkeypatch, caplog):
    monkeypatch.setenv(ENV_VAR, "gpu")  # valeur plausible mais invalide
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with caplog.at_level("WARNING"):
        device = resolve_device()

    assert device.type == "cpu"
    assert "non reconnu" in caplog.text
