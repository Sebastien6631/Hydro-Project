from __future__ import annotations

import json

import pytest

from previ_r2d2.common.config import ROOT
from previ_r2d2.preprocessing.bv.rules import (
    estimate_exposition,
    estimate_kbase,
    estimate_kc_unit,
    fit_rules,
    load_rules,
)

REF_DIR = ROOT / "centrales" / "REFERENCE"


def _load_calibration() -> list[dict]:
    with open(REF_DIR / "centrales_calibration.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_estimate_kc_unit_clips_to_bounds():
    r = load_rules(REF_DIR / "bv_rules.json")
    assert estimate_kc_unit(0, r) == r.kc_clip_max
    assert estimate_kc_unit(50_000, r) == r.kc_clip_min


def test_estimate_kc_unit_matches_known_centrales_within_documented_mae():
    """Comparaison aux valeurs réelles calées à la main pour les centrales Previ_v2
    (centrales/REFERENCE/centrales_calibration.json) — cf. MAE=0.034 documenté dans
    bv_rules.json (calé à l'origine sur 8 BV, ici vérifié sur les 3 restants après
    réduction du périmètre : apas_G1_G4, nancy_A, touzac_g2_G2)."""
    r = load_rules(REF_DIR / "bv_rules.json")
    calibration = _load_calibration()

    errors = [
        abs(estimate_kc_unit(row["Altitude_moyenne_BV"], r) - row["kc_unit"])
        for row in calibration
    ]

    assert sum(errors) / len(errors) == pytest.approx(0.017, abs=0.005)
    assert max(errors) < 0.07


def test_estimate_kbase_matches_known_centrales_except_ambiguous_band():
    """apas_G1_G4 (alt=1201 m) tombe dans la bande 1000-1300 m documentée comme
    ambiguë dans bv_rules.json : la règle par paliers donne 3.0, pas le 3.8 réel,
    d'où kbase_needs_review=True — comportement attendu, pas un bug. Toutes les
    autres centrales connues doivent matcher exactement."""
    r = load_rules(REF_DIR / "bv_rules.json")
    calibration = _load_calibration()

    for row in calibration:
        kbase, needs_review = estimate_kbase(row["Altitude_moyenne_BV"], r)
        if row["Centrale"] == "apas_G1_G4":
            assert (kbase, needs_review) == (3.0, True)
        else:
            assert (kbase, needs_review) == (row["K_base"], False), row["Centrale"]


def test_estimate_exposition_south_facing_increases_factor():
    r = load_rules(REF_DIR / "bv_rules.json")
    assert estimate_exposition(180.0, r) == pytest.approx(1.0 + r.expo_gain, abs=1e-6)
    assert estimate_exposition(0.0, r) == pytest.approx(1.0 - r.expo_gain, abs=1e-6)
    assert estimate_exposition(None, r) == 1.0


def test_load_rules_reads_shipped_bv_rules_json():
    r = load_rules(REF_DIR / "bv_rules.json")
    assert r.n_calib == 8
    assert r.kbase_steps[0] == [1800, 4.5]


def test_fit_rules_runs_on_calibration_json():
    fitted = fit_rules(REF_DIR)
    assert fitted.n_calib == 3
    assert fitted.kc_slope < 0
    assert fitted.kc_clip_min == 0.60
    assert fitted.kc_clip_max == 1.05
