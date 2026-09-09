from __future__ import annotations

import pandas as pd

from previ_r2d2.preprocessing.meteo.nwp_reader import parse_nwp_file, read_points


def test_parse_nwp_file_renames_columns_and_keeps_raw_units(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-07-08,2026-07-08 14:00:00,294.45,0.0038,2195.48\n"
        "42.6,2.1,2026-07-08,2026-07-08 14:00:00,298.06,0.0038,2526.10\n",
        encoding="utf-8",
    )

    df = parse_nwp_file(path)

    assert list(df.columns) == ["run_date", "flow_date", "temperature", "precipitation", "niveau0"]
    assert df.index.tolist() == [(42.6, 1.8), (42.6, 2.1)]
    row = df.loc[(42.6, 1.8)]
    assert row["temperature"] == 294.45  # Kelvin brut, jamais converti
    assert row["precipitation"] == 0.0038  # cumul brut, jamais diffé
    assert row["flow_date"] == pd.Timestamp("2026-07-08 14:00:00")


def test_parse_nwp_file_accepts_old_2021_2024_header_case_and_extra_column(tmp_path):
    path = tmp_path / "old_format.csv"
    path.write_text(
        "Latitude,Longitude,run_date,flow_date,2t (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level),hour_index\n"
        "42.6,1.8,2022-03-05,2022-03-05 14:00:00,294.45,0.0038,2195.48,14\n",
        encoding="utf-8",
    )

    df = parse_nwp_file(path)

    row = df.loc[(42.6, 1.8)]
    assert row["temperature"] == 294.45
    assert row["precipitation"] == 0.0038
    assert row["niveau0"] == 2195.48
    assert row["flow_date"] == pd.Timestamp("2022-03-05 14:00:00")



def test_read_points_extracts_only_requested_grid_points(tmp_path):
    day_dir = tmp_path / "2026" / "07" / "08"
    day_dir.mkdir(parents=True)
    (day_dir / "BARTHE_ENR_EC_OP_recent_2026070800_014.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-07-08,2026-07-08 14:00:00,294.45,0.0038,2195.48\n"
        "42.6,2.1,2026-07-08,2026-07-08 14:00:00,298.06,0.0038,2526.10\n",
        encoding="utf-8",
    )
    (day_dir / "BARTHE_ENR_EC_OP_recent_2026070800_015.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-07-08,2026-07-08 15:00:00,295.00,0.0050,2200.00\n",
        encoding="utf-8",
    )
    points = [{"id": 1, "lat": 42.6, "lon": 1.8}]

    result = read_points(
        tmp_path, points, pd.Timestamp("2026-07-08 00:00:00"), pd.Timestamp("2026-07-08 23:00:00")
    )

    assert list(result.columns) == [
        "latitude_S1", "longitude_S1", "temperature_S1", "precipitation_S1", "niveau0_S1",
    ]
    assert result.loc[pd.Timestamp("2026-07-08 14:00:00"), "temperature_S1"] == 294.45
    assert result.loc[pd.Timestamp("2026-07-08 15:00:00"), "temperature_S1"] == 295.00
    assert len(result) == 2  # le point (42.6, 2.1) du 1er fichier n'est PAS extrait


def test_read_points_returns_empty_when_no_day_folder(tmp_path):
    result = read_points(
        tmp_path, [{"id": 1, "lat": 42.6, "lon": 1.8}],
        pd.Timestamp("2026-07-08 00:00:00"), pd.Timestamp("2026-07-08 23:00:00"),
    )

    assert result.empty


def test_read_points_skips_point_not_present_in_any_file(tmp_path):
    day_dir = tmp_path / "2026" / "07" / "08"
    day_dir.mkdir(parents=True)
    (day_dir / "BARTHE_ENR_EC_OP_recent_2026070800_014.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-07-08,2026-07-08 14:00:00,294.45,0.0038,2195.48\n",
        encoding="utf-8",
    )
    points = [
        {"id": 1, "lat": 42.6, "lon": 1.8},
        {"id": 2, "lat": 99.9, "lon": 99.9},  # absent de la grille
    ]

    result = read_points(
        tmp_path, points, pd.Timestamp("2026-07-08 00:00:00"), pd.Timestamp("2026-07-08 23:00:00")
    )

    assert "temperature_S1" in result.columns
    assert "temperature_S2" not in result.columns


def test_read_points_skips_empty_file_and_keeps_valid_one(tmp_path):
    day_dir = tmp_path / "2026" / "07" / "08"
    day_dir.mkdir(parents=True)
    (day_dir / "BARTHE_ENR_EC_OP_recent_2026070800_014.csv").write_text("", encoding="utf-8")
    (day_dir / "BARTHE_ENR_EC_OP_recent_2026070800_015.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-07-08,2026-07-08 15:00:00,295.00,0.0050,2200.00\n",
        encoding="utf-8",
    )
    points = [{"id": 1, "lat": 42.6, "lon": 1.8}]

    result = read_points(
        tmp_path, points, pd.Timestamp("2026-07-08 00:00:00"), pd.Timestamp("2026-07-08 23:00:00")
    )

    assert result.loc[pd.Timestamp("2026-07-08 15:00:00"), "temperature_S1"] == 295.00


def test_read_points_excludes_timestamps_outside_window(tmp_path):
    day_dir = tmp_path / "2026" / "07" / "08"
    day_dir.mkdir(parents=True)
    (day_dir / "BARTHE_ENR_EC_OP_recent_2026070800_005.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-07-08,2026-07-08 05:00:00,290.00,0.0010,2100.00\n",
        encoding="utf-8",
    )
    (day_dir / "BARTHE_ENR_EC_OP_recent_2026070800_014.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-07-08,2026-07-08 14:00:00,294.45,0.0038,2195.48\n",
        encoding="utf-8",
    )
    points = [{"id": 1, "lat": 42.6, "lon": 1.8}]

    result = read_points(
        tmp_path, points, pd.Timestamp("2026-07-08 10:00:00"), pd.Timestamp("2026-07-08 23:00:00")
    )

    assert pd.Timestamp("2026-07-08 05:00:00") not in result.index
    assert pd.Timestamp("2026-07-08 14:00:00") in result.index


def test_read_points_matches_grid_point_despite_float_rounding_noise(tmp_path):
    day_dir = tmp_path / "2026" / "07" / "08"
    day_dir.mkdir(parents=True)
    (day_dir / "BARTHE_ENR_EC_OP_recent_2026070800_014.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-07-08,2026-07-08 14:00:00,294.45,0.0038,2195.48\n",
        encoding="utf-8",
    )
    points = [{"id": 1, "lat": 40.8 + 1.8, "lon": 1.8}]  # 42.599999999999994, pas 42.6 exact

    result = read_points(
        tmp_path, points, pd.Timestamp("2026-07-08 00:00:00"), pd.Timestamp("2026-07-08 23:00:00")
    )

    assert result.loc[pd.Timestamp("2026-07-08 14:00:00"), "temperature_S1"] == 294.45


def test_read_points_handles_duplicate_flow_date_across_two_retention_delayed_files(tmp_path):
    day_dir_31 = tmp_path / "2025" / "12" / "31"
    day_dir_31.mkdir(parents=True)
    (day_dir_31 / "BARTHE_ENR_EC_OP_recent_2025123100_024.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2025-12-31,2026-01-01 00:00:00,290.00,0.0010,2100.00\n",
        encoding="utf-8",
    )
    day_dir_01 = tmp_path / "2026" / "01" / "01"
    day_dir_01.mkdir(parents=True)
    (day_dir_01 / "BARTHE_ENR_EC_OP_recent_2026010100_000.csv").write_text(
        "latitude,longitude,run_date,flow_date,2T (2 metre temperature),"
        "tp (precipitation),deg0l (zero degree level)\n"
        "42.6,1.8,2026-01-01,2026-01-01 00:00:00,300.00,0.0090,2300.00\n",
        encoding="utf-8",
    )
    points = [{"id": 1, "lat": 42.6, "lon": 1.8}]

    result = read_points(
        tmp_path, points, pd.Timestamp("2025-12-31 00:00:00"), pd.Timestamp("2026-01-01 23:00:00")
    )

    assert len(result.loc[[pd.Timestamp("2026-01-01 00:00:00")]]) == 1
