from __future__ import annotations

import pandas as pd

from previ_r2d2.preprocessing.automate.reader import parse_txt_file, read_variable


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_parse_txt_file_reads_hydrospot_format(tmp_path):
    f = tmp_path / "2026" / "07" / "09" / "12.txt"
    _write(f, "12:00:10  16.73\n12:01:10  16.80\n")

    result = parse_txt_file(f)

    assert list(result.index) == [
        pd.Timestamp("2026-07-09 12:00:10"),
        pd.Timestamp("2026-07-09 12:01:10"),
    ]
    assert list(result.values) == [16.73, 16.80]


def test_parse_txt_file_reads_original_semicolon_format(tmp_path):
    f = tmp_path / "2026" / "07" / "09" / "12.txt"
    _write(f, "09/07/2026;12:00:23;7,420; ;\n")

    result = parse_txt_file(f)

    assert list(result.index) == [pd.Timestamp("2026-07-09 12:00:23")]
    assert list(result.values) == [7.420]


def test_read_variable_concatenates_files_in_window(tmp_path):
    racine = tmp_path / "Pression"
    _write(racine / "2026" / "07" / "09" / "11.txt", "11:00:00  10\n")
    _write(racine / "2026" / "07" / "09" / "12.txt", "12:00:00  12\n")
    _write(racine / "2026" / "07" / "08" / "23.txt", "23:00:00  99\n")  # hors fenêtre

    result = read_variable(
        racine,
        start=pd.Timestamp("2026-07-09 00:00:00"),
        end=pd.Timestamp("2026-07-09 23:59:59"),
    )

    assert list(result.values) == [10, 12]
    assert 99 not in result.values


def test_read_variable_empty_window_returns_resamplable_series(tmp_path):
    # Aucun fichier dans la fenêtre (dossier absent, ex. rsync en échec) --
    # doit rester une Series indexable par .resample("1h"), pas planter avec
    # "Only valid with DatetimeIndex..." (RangeIndex par défaut d'une Series vide).
    racine = tmp_path / "Pression"

    result = read_variable(
        racine,
        start=pd.Timestamp("2026-07-09 00:00:00"),
        end=pd.Timestamp("2026-07-09 23:59:59"),
    )

    assert result.empty
    resampled = result.resample("1h").mean()
    assert resampled.empty
