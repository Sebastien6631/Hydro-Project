from __future__ import annotations

import pandas as pd

from previ_r2d2.preprocessing.puissance import consignes


def test_consigne_prefix_for_lowercases_source_folder():
    assert consignes.consigne_prefix_for("Castillon_Apas_G1") == "castillon_apas_g1"


def test_read_consigne_events_parses_true_and_false_files(tmp_path):
    day_dir = tmp_path / "2026" / "07" / "06"
    day_dir.mkdir(parents=True)
    (day_dir / "Castillon_Apas_G1_14_40.true").write_text(
        "true P 460.0 Consi 130.0 1752928800", encoding="utf-8")
    (day_dir / "Castillon_Apas_G1_15_10.false").write_text(
        "false 1752930600", encoding="utf-8")
    (day_dir / "Autre_Centrale_G1_14_40.true").write_text(
        "true P 999.0 Consi 1.0 0", encoding="utf-8")

    true_events, false_events = consignes.read_consigne_events(
        "castillon_apas_g1", pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-06"),
        root=tmp_path,
    )

    assert true_events == [(pd.Timestamp("2026-07-06 14:40:00"), 460.0)]
    assert false_events == [pd.Timestamp("2026-07-06 15:10:00")]


def test_read_consigne_events_ignores_malformed_file(tmp_path):
    day_dir = tmp_path / "2026" / "07" / "06"
    day_dir.mkdir(parents=True)
    (day_dir / "Castillon_Apas_G1_14_40.true").write_text("garbage", encoding="utf-8")

    true_events, false_events = consignes.read_consigne_events(
        "castillon_apas_g1", pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-06"),
        root=tmp_path,
    )

    assert true_events == []
    assert false_events == []


def test_read_consigne_events_handles_uppercase_extension(tmp_path):
    day_dir = tmp_path / "2026" / "07" / "06"
    day_dir.mkdir(parents=True)
    (day_dir / "Castillon_Apas_G1_14_40.TRUE").write_text(
        "true P 460.0 Consi 130.0 1752928800", encoding="utf-8")

    true_events, false_events = consignes.read_consigne_events(
        "castillon_apas_g1", pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-06"),
        root=tmp_path,
    )

    assert true_events == [(pd.Timestamp("2026-07-06 14:40:00"), 460.0)]
    assert false_events == []


def test_read_consigne_events_no_matching_day_dir_returns_empty(tmp_path):
    true_events, false_events = consignes.read_consigne_events(
        "castillon_apas_g1", pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-06"),
        root=tmp_path,
    )
    assert true_events == []
    assert false_events == []


def test_interpolate_consignes_linear_ramp_between_true_and_false():
    dates = pd.date_range("2026-01-01 09:56", "2026-01-01 10:06", freq="1min")
    df = pd.DataFrame({"Date": dates, "Puissance": [0.0] * len(dates)})
    df.loc[df["Date"] == pd.Timestamp("2026-01-01 09:58:00"), "Puissance"] = 100.0
    df.loc[df["Date"] == pd.Timestamp("2026-01-01 10:04:00"), "Puissance"] = 200.0
    true_events = [(pd.Timestamp("2026-01-01 10:00:00"), 130.0)]
    false_events = [pd.Timestamp("2026-01-01 10:02:00")]

    result = consignes.interpolate_consignes(df, true_events, false_events)

    assert result.to_dict() == {
        pd.Timestamp("2026-01-01 09:59:00"): 100.0,
        pd.Timestamp("2026-01-01 10:00:00"): 125.0,
        pd.Timestamp("2026-01-01 10:01:00"): 150.0,
        pd.Timestamp("2026-01-01 10:02:00"): 175.0,
        pd.Timestamp("2026-01-01 10:03:00"): 200.0,
    }


def test_interpolate_consignes_true_without_following_false_is_ignored():
    dates = pd.date_range("2026-01-01 09:56", "2026-01-01 10:06", freq="1min")
    df = pd.DataFrame({"Date": dates, "Puissance": [0.0] * len(dates)})
    true_events = [(pd.Timestamp("2026-01-01 10:00:00"), 130.0)]

    result = consignes.interpolate_consignes(df, true_events, false_events=[])

    assert result.empty


def test_interpolate_consignes_empty_inputs_returns_empty_series():
    result = consignes.interpolate_consignes(pd.DataFrame(), [], [])
    assert result.empty
