"""Data loading and train/val/test split, transcribed from the paper notebooks.

The steps below intentionally mirror the notebook cells line for line
(read -> to_datetime -> drop index column -> melt -> select columns, and the
Y_df['ds'].values[-422] / [-211] cut points) so outputs match the paper.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config


def load_wide(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    return data


def load_long(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Y_df with columns unique_id, ds, y (37 series)."""
    df = load_wide(path)
    long_df = pd.melt(df, id_vars=["date"],
                      value_vars=[c for c in df.columns if c != "date"],
                      var_name="unique_id", value_name="y")
    long_df = long_df.rename(columns={"date": "ds"})
    return long_df[["unique_id", "ds", "y"]]


@dataclass(frozen=True)
class Split:
    Y_fit_df: pd.DataFrame     # train + val, as passed to nf.fit
    val_size: int
    test_size: int
    train_end: pd.Timestamp    # last training day
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def split(Y_df: pd.DataFrame, check: bool = True) -> Split:
    v = Y_df["ds"].values
    Y_train = Y_df[Y_df["ds"] < v[-422]]
    Y_val = Y_df[(Y_df["ds"] >= v[-422]) & (Y_df["ds"] < v[-211])]
    Y_test = Y_df[Y_df["ds"] >= v[-211]].reset_index(drop=True)

    val_size = Y_val["ds"].nunique()
    test_size = Y_test["ds"].nunique()
    s = Split(
        Y_fit_df=pd.concat([Y_train, Y_val]).reset_index(drop=True),
        val_size=val_size,
        test_size=test_size,
        train_end=Y_train["ds"].max(),
        val_start=Y_val["ds"].min(),
        test_start=Y_test["ds"].min(),
        test_end=Y_test["ds"].max(),
    )
    if check:
        _check(Y_df, s)
    return s


def _check(Y_df: pd.DataFrame, s: Split) -> None:
    problems = []
    if Y_df["unique_id"].nunique() != config.N_SERIES:
        problems.append(f"{Y_df['unique_id'].nunique()} series, expected {config.N_SERIES}")
    if s.val_size != config.VAL_SIZE or s.test_size != config.TEST_SIZE:
        problems.append(f"val/test = {s.val_size}/{s.test_size}, "
                        f"expected {config.VAL_SIZE}/{config.TEST_SIZE}")
    if str(s.test_start.date()) != config.TEST_START:
        problems.append(f"test starts {s.test_start.date()}, expected {config.TEST_START}")
    if str(s.test_end.date()) != config.TEST_END:
        problems.append(f"test ends {s.test_end.date()}, expected {config.TEST_END}")
    if problems:
        raise ValueError("Data do not match the paper setup: " + "; ".join(problems))


@lru_cache(maxsize=4)
def fingerprint(path: Path | str = config.DATA_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
