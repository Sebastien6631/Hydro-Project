from __future__ import annotations

import pandas as pd
import pytest

from previ_r2d2.preprocessing.automate.physics import (
    calcul_deversoir_q_serie,
    calcul_hauteur,
    calcul_polynome_vectorise,
    calcul_qturb,
    compute_qentrant,
)


def test_calcul_polynome_vectorise_applies_matching_segment():
    series = pd.Series([-5.0, 5.0, 50.0])
    param_segments = [
        {"seuils": [-10, 0], "A": [1.0, 2.0]},   # y = 1 + 2x, pour x in [-10, 0]
        {"seuils": [0, 100], "A": [10.0, 0.5]},  # y = 10 + 0.5x, pour x in [0, 100]
    ]

    result = calcul_polynome_vectorise(series, param_segments)

    assert result.tolist() == pytest.approx([1.0 + 2.0 * -5.0, 10.0 + 0.5 * 5.0, 10.0 + 0.5 * 50.0])


def test_calcul_hauteur_basse_chute():
    df = pd.DataFrame({"sonde_amont": [100.0], "sonde_restitution": [20.0]})
    dico = {"categorie": "basse chute", "constante_correction_Hn": 500.0}
    puissance = pd.Series([0.0])

    hn = calcul_hauteur(df, dico, puissance)

    assert hn.tolist() == pytest.approx([(100.0 - 20.0 + 500.0) / 100])


def test_calcul_hauteur_haute_chute_action_turbine():
    df = pd.DataFrame({"pression_conduite": [10.0]})
    dico = {
        "categorie": "haute chute",
        "type": "action",
        "constante_correction_Hn": 0.0,
        "segment_Q_fct_P": [{"seuils": [0, float("inf")], "A": [0.0, 0.001]}],
        "diam_conduite": 1.0,
    }
    puissance = pd.Series([1000.0])

    hn = calcul_hauteur(df, dico, puissance)

    g = 9.81
    estime_qturb = 0.0 + 0.001 * 1000.0
    expected = (10.0 * 100000) / (g * 1000) + ((estime_qturb / (3.141592653589793 * 0.5**2)) ** 2) / (2 * g) + 0.0
    assert hn.tolist() == pytest.approx([expected], rel=1e-6)


def test_calcul_qturb_uses_power_based_estimate_when_configured():
    df = pd.DataFrame({"x": [0.0]})
    dico = {"segment_Q_fct_P": [{"seuils": [0, float("inf")], "A": [1.0, 0.002]}]}
    puissance = pd.Series([500.0])
    chute_nette = pd.Series([10.0])

    result = calcul_qturb(df, dico, puissance, chute_nette)

    assert result["Q_fct_P"].tolist() == pytest.approx([1.0 + 0.002 * 500.0])
    assert result["Q_fct_charge"].tolist() == pytest.approx([0.0])
    assert result["Q_fct_ouv"].tolist() == pytest.approx([0.0])


def test_calcul_qturb_action_turbine_sums_injectors():
    df = pd.DataFrame({"Ouv_inject_1": [50.0], "Ouv_inject_2": [50.0]})
    dico = {
        "type": "action",
        "segment_Q_fct_ouv_grp": [{"seuils": [0, 100], "A": [0.0, 0.01]}],
    }
    puissance = pd.Series([0.0])
    chute_nette = pd.Series([0.0])

    result = calcul_qturb(df, dico, puissance, chute_nette)

    assert result["Q_fct_ouv"].tolist() == pytest.approx([0.01 * 50.0 + 0.01 * 50.0])


def test_calcul_qturb_reaction_turbine_uses_ouv_groupe():
    df = pd.DataFrame({"ouv_groupe": [80.0]})
    dico = {
        "type": "reaction",
        "segment_Q_fct_ouv_grp": [{"seuils": [0, 100], "A": [0.0, 0.02]}],
    }
    puissance = pd.Series([0.0])
    chute_nette = pd.Series([0.0])

    result = calcul_qturb(df, dico, puissance, chute_nette)

    assert result["Q_fct_ouv"].tolist() == pytest.approx([0.02 * 80.0])


def test_calcul_deversoir_q_serie_fixed_reserve():
    df = pd.DataFrame({"sonde_1": [10.0]})
    dico = {
        "Qreserve_fixe": "oui",
        "type": "fixe",
        "param_segments": [{"seuils": [-100, 100], "A": [0.0, 0.05]}],
    }

    result = calcul_deversoir_q_serie(df, dico, 1)

    assert result.tolist() == pytest.approx([0.05 * 10.0])


def test_calcul_deversoir_q_serie_seasonal_reserve():
    df = pd.DataFrame(
        {"sonde_1": [10.0, 10.0]},
        index=pd.DatetimeIndex(["2026-08-01", "2026-01-01"]),
    )
    dico = {
        "Qreserve_fixe": "non",
        "periode_ete": ((7, 1), (10, 31)),
        "param_segments_1": [{"seuils": [-100, 100], "A": [0.0, 1.0]}],  # été
        "param_segments_2": [{"seuils": [-100, 100], "A": [0.0, 2.0]}],  # hiver
    }

    result = calcul_deversoir_q_serie(df, dico, 1)

    assert result.tolist() == pytest.approx([10.0, 20.0])  # été: x1, hiver: x2


def test_compute_qentrant_combines_deversoir_and_groupes():
    idx = pd.DatetimeIndex(["2026-01-01", "2026-01-02"])
    df = pd.DataFrame(
        {"sonde_1": [10.0, 10.0], "ouv_groupe": [50.0, 50.0]},
        index=idx,
    )
    config_dicts = {
        "Config_deversoir_1": {
            "Qreserve_fixe": "oui",
            "type": "fixe",
            "param_segments": [{"seuils": [-100, 100], "A": [0.0, 0.1]}],  # Qrestitue = 0.1 * sonde_1
        },
        "Config_G1": {
            "categorie": "basse chute",
            "constante_correction_Hn": 0.0,
            "type": "reaction",
            "segment_Q_fct_ouv_grp": [{"seuils": [0, 100], "A": [0.0, 0.02]}],  # Qturb = 0.02 * ouv_groupe
        },
    }
    puissance = pd.DataFrame({"power_output": [0.0, 0.0]}, index=idx)

    q_restitue, q_turbine, q_entrant = compute_qentrant(
        df, config_dicts, mapping={}, puissance=puissance, consigne_regulation={}
    )

    assert q_restitue.tolist() == pytest.approx([1.0, 1.0])  # 0.1 * 10
    assert q_turbine["Q_fct_ouv"].tolist() == pytest.approx([1.0, 1.0])  # 0.02 * 50
    assert q_entrant["Q_fct_ouv"].tolist() == pytest.approx([2.0, 2.0])  # 1.0 + 1.0


def test_compute_qentrant_applies_mapping_before_calc():
    idx = pd.DatetimeIndex(["2026-01-01"])
    df = pd.DataFrame({"ML2_P_CONDUITE_NETTE": [10.0]}, index=idx)
    config_dicts = {
        "Config_G1": {
            "categorie": "haute chute",
            "type": "action",
            "constante_correction_Hn": 0.0,
            "segment_Q_fct_P": [{"seuils": [0, float("inf")], "A": [0.0, 0.001]}],
            "diam_conduite": 1.0,
        },
    }
    mapping = {"ML2_P_CONDUITE_NETTE": [["Config_G1", "pression_conduite"]]}
    puissance = pd.DataFrame({"power_output": [100.0]}, index=idx)

    q_restitue, q_turbine, q_entrant = compute_qentrant(
        df, config_dicts, mapping, puissance, consigne_regulation={}
    )

    # La colonne a été renommée en "pression_conduite" avant le calcul --
    # si le mapping n'était pas appliqué, Calcul_Hauteur lèverait KeyError.
    assert "pression_conduite" in df.columns
