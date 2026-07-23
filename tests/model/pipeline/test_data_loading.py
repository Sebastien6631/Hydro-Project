from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from previ_r2d2.common import config
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv
from previ_r2d2.model.pipeline.data_loading import load_df, resample_to_daily, split_train_test


def make_hourly_df(n=48):
    index = pd.date_range("2026-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "debit_m3s": np.arange(n, dtype=float),
            "precipitation_S1": np.ones(n),
            "temperature_S1": 280.0,
        },
        index=index,
    )


def test_load_df_reads_existing_data_preparation_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    df = make_hourly_df()
    write_data_preparation_csv(df, tmp_path / "test_centrale" / "data_preparation.csv")

    result = load_df("test_centrale")

    assert result.shape == (48, 3)
    assert result["debit_m3s"].iloc[0] == 0.0


def test_load_df_raises_file_not_found_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_df("nonexistent")


def test_resample_to_daily_sums_precipitation_and_averages_rest():
    df = make_hourly_df()

    daily = resample_to_daily(df)

    assert daily.shape == (2, 3)
    assert daily["debit_m3s"].iloc[0] == pytest.approx(11.5)  # moyenne de 0..23
    assert daily["precipitation_S1"].iloc[0] == pytest.approx(24.0)  # somme de 24 x 1.0


def test_split_train_test_default_ratio_and_custom_ratio():
    df = make_hourly_df()

    train, test = split_train_test(df, test_ratio=0.25)

    assert len(train) == 36
    assert len(test) == 12
    assert train.index[-1] < test.index[0]
