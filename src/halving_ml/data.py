"""Data loading utilities for BTC-USD prices."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import pandas as pd
import yfinance as yf

from . import config


def download_prices(symbol: str = config.SYMBOL, start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    start = start or pd.Timestamp(config.DATA_START)
    df = yf.download(symbol, start=start, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} starting {start}")
    return df.reset_index()[["Date", "Close"]]


def _clean_price_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Date is datetime and Close is numeric, dropping bad rows."""
    if not {"Date", "Close"}.issubset(raw_df.columns):
        raise ValueError("Input data must contain 'Date' and 'Close' columns")

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset="Date", keep="last")
    return df.reset_index(drop=True)


def _generate_synthetic_prices(start: pd.Timestamp = pd.Timestamp(config.DATA_START), periods: int = 8 * 365) -> pd.DataFrame:
    """Create a deterministic synthetic BTC-USD price series for offline runs."""

    dates = pd.date_range(start=start, periods=periods, freq="D")
    rng = np.random.default_rng(seed=42)
    # Daily log-returns with small drift and volatility to mimic crypto swings
    log_returns = rng.normal(loc=0.0005, scale=0.04, size=len(dates))
    prices = 100 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"Date": dates, "Close": prices})


def _read_local_file(path: Path) -> pd.DataFrame:
    """Read either CSV or Excel price data and return a cleaned DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(path, parse_dates=["Date"])
    elif suffix in {".xlsx", ".xls"}:
        try:
            data = pd.read_excel(path)
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "Reading Excel data requires the optional dependency 'openpyxl'. "
                "Install it with `pip install openpyxl`."
            ) from exc
    else:
        raise ValueError(f"Unsupported file type for {path}")
    return _clean_price_df(data)


def load_data(use_cache: bool = True, raw_path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load cached/downloaded price data, supporting CSV or Excel."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache:
        candidates = [raw_path]
        if raw_path.suffix.lower() == ".csv":
            candidates.append(raw_path.with_suffix(".xlsx"))
            candidates.append(raw_path.with_suffix(".xls"))

        for path in candidates:
            if path.exists():
                return _read_local_file(path)

    try:
        data = download_prices()
    except Exception:
        data = _generate_synthetic_prices()
        data.to_csv(raw_path, index=False)
        return _clean_price_df(data)

    data.to_csv(raw_path, index=False)
    return _clean_price_df(data)


__all__ = ["load_data", "download_prices"]
