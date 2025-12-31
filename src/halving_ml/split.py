"""Walk-forward split utilities."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterator, List, Tuple

import pandas as pd

from . import config


def _date_blocks(start: datetime, end: datetime, months: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    blocks = []
    current = pd.Timestamp(start)
    while current <= end:
        next_end = (current + pd.DateOffset(months=months)) - pd.DateOffset(days=1)
        blocks.append((current, next_end))
        current = current + pd.DateOffset(months=months)
    return blocks


def walk_forward_splits(df: pd.DataFrame) -> Iterator[Dict]:
    dates = pd.to_datetime(df["Date"])
    max_date = dates.max()
    blocks = _date_blocks(config.SPLIT_START + pd.DateOffset(days=1), max_date, config.SPLIT_FREQ_MONTHS)

    for i, (start, end) in enumerate(blocks, 1):
        test_mask = (dates >= start) & (dates <= end)
        if not test_mask.any():
            continue
        train_mask = dates < start
        yield {
            "fold": i,
            "train_idx": df.index[train_mask],
            "test_idx": df.index[test_mask],
            "start": start,
            "end": end,
        }


def inner_train_val_split(train_idx, val_frac: float = 0.2):
    train_idx = list(train_idx)
    split_point = int(len(train_idx) * (1 - val_frac))
    inner_train = train_idx[:split_point]
    val_idx = train_idx[split_point:]
    return inner_train, val_idx


__all__ = ["walk_forward_splits", "inner_train_val_split"]
