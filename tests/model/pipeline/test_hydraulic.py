from __future__ import annotations

import pandas as pd
import pytest

from previ_r2d2.model.pipeline.hydraulic import (
    compute_hydraulic_point,
    compute_q_non_turbinable,
    dispatch_groupes,
    eval_chute,
    eval_polynome,
    eval_rendement,
)


def test_eval_polynome_evaluates_lowest_degree_first():
    assert eval_polynome([1.0, 2.0, 3.0], 2.0) == pytest.approx(1.0 + 2.0 * 2.0 + 3.0 * 4.0)


def test_eval_chute_single_segment_polynome():
    groupe = {"chute_disponible_polynome": [7.24, -0.0115, 3.856e-05, -7e-08]}

    result = eval_chute(groupe, 10.0)

    assert result == pytest.approx(eval_polynome(groupe["chute_disponible_polynome"], 10.0))


def test_eval_chute_segmented_picks_matching_segment():
    groupe = {
        "chute_disponible_polynome": [[6.0, -0.005], [5.9, -0.01]],
        "chute_disponible_seuils": [[0, 100], [100, 450]],
    }

    result_low = eval_chute(groupe, 50.0)
    result_high = eval_chute(groupe, 200.0)

    assert result_low == pytest.approx(eval_polynome([6.0, -0.005], 50.0))
    assert result_high == pytest.approx(eval_polynome([5.9, -0.01], 200.0))


def test_eval_rendement_clips_negative_to_zero():
    groupe = {"rendement": [-50.0]}

    assert eval_rendement(groupe, 10.0) == 0.0


def test_compute_q_non_turbinable_constant():
    rec = {"debit_non_turbinable": 3.5}

    assert compute_q_non_turbinable(rec, pd.Timestamp("2026-01-15")) == 3.5


def test_compute_q_non_turbinable_seasonal_split():
    rec = {
        "q_non_turbinable_ete": 9.0,
        "q_non_turbinable_hiver": 5.0,
        "periode_ete": [[6, 1], [10, 31]],
    }

    assert compute_q_non_turbinable(rec, pd.Timestamp("2026-07-15")) == 9.0
    assert compute_q_non_turbinable(rec, pd.Timestamp("2026-01-15")) == 5.0
    assert compute_q_non_turbinable(rec, pd.Timestamp("2026-11-15")) == 5.0


def test_dispatch_groupes_serves_priority_first_then_cascades():
    groupes = [
        {"nom_groupe": "G1", "priorite": 2, "debit_armement_turbine": 2, "debit_max": 20},
        {"nom_groupe": "G4", "priorite": 1, "debit_armement_turbine": 12.5, "debit_max": 12.5},
    ]

    result = dispatch_groupes(15.0, groupes)

    g4 = next(g for g in result if g["nom_groupe"] == "G4")
    g1 = next(g for g in result if g["nom_groupe"] == "G1")
    assert g4["debit_turbine"] == 12.5
    assert g1["debit_turbine"] == pytest.approx(2.5)


def test_dispatch_groupes_zeroes_group_below_armement_threshold():
    groupes = [{"nom_groupe": "G1", "priorite": 1, "debit_armement_turbine": 10.0, "debit_max": 20.0}]

    result = dispatch_groupes(5.0, groupes)

    assert result[0]["debit_turbine"] == 0.0


def test_compute_hydraulic_point_nominal_matches_verified_values():
    rec = {
        "debit_non_turbinable": 3.5,
        "groupes": [
            {
                "nom_groupe": "A", "priorite": 1,
                "debit_armement_turbine": 7.5, "debit_max": 30,
                "rendement": [78.242, 0.6076, -0.0176],
                "chute_disponible_polynome": [
                    [6.01554501, -0.00514309, -0.00016373, 1.19e-06],
                    [5.9895238095, -0.0102801587, 7.9048e-06, -2.2e-09],
                ],
                "chute_disponible_seuils": [[0, 100], [100, 450]],
            },
        ],
    }

    result = compute_hydraulic_point(25.0, rec, pd.Timestamp("2026-01-15"))

    assert result["debit_turbinable"] == pytest.approx(21.5, abs=0.01)
    assert result["debit_reserve"] == pytest.approx(3.5, abs=0.01)
    assert result["puissance"] == pytest.approx(1018.0, abs=1.0)
    assert result["pmax_dyn"] == pytest.approx(1377.1, abs=1.0)
    assert result["chute_estimee"] == pytest.approx(5.803, abs=0.01)
    assert result["rendement_estime"] == pytest.approx(83.17, abs=0.01)


def test_compute_hydraulic_point_below_armement_zeroes_turbinable():
    rec = {
        "debit_non_turbinable": 3.5,
        "groupes": [{
            "nom_groupe": "A", "priorite": 1,
            "debit_armement_turbine": 7.5, "debit_max": 30,
            "rendement": [78.242, 0.6076, -0.0176],
            "chute_disponible_polynome": [
                [6.01554501, -0.00514309, -0.00016373, 1.19e-06],
                [5.9895238095, -0.0102801587, 7.9048e-06, -2.2e-09],
            ],
            "chute_disponible_seuils": [[0, 100], [100, 450]],
        }],
    }

    result = compute_hydraulic_point(2.0, rec, pd.Timestamp("2026-01-15"))

    assert result["debit_turbinable"] == 0.0
    assert result["debit_reserve"] == pytest.approx(2.0, abs=0.01)
    assert result["puissance"] == 0.0


def test_compute_hydraulic_point_no_groupes_returns_zeros():
    rec = {"debit_non_turbinable": 3.5, "groupes": []}

    result = compute_hydraulic_point(10.0, rec, pd.Timestamp("2026-01-15"))

    assert result["puissance"] == 0.0
    assert result["pmax_dyn"] == 0.0
