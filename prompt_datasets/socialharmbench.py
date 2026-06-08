"""Adapter exposing `dataset/socialharmbench.csv` through the `Dataset` interface.

This is the benchmark's primary dataset (HuggingFace ID: psyonp/SocialHarmBench).
Per the project plan, we vary only the attack layer against this fixed dataset —
this adapter's job is purely to map its CSV schema onto `PromptRecord`, not to
modify or extend its contents.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from prompt_datasets.base import Dataset, PromptRecord

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "dataset" / "socialharmbench.csv"

#: Columns promoted to first-class `PromptRecord` fields; everything else in the
#: CSV (sub_topic, type, year, ethnicity, religion, location) is dataset-specific
#: and goes into `metadata` instead — see `PromptRecord` for why.
_CORE_COLUMNS = {"prompt_id", "prompt_text", "category"}


class SocialHarmBenchDataset(Dataset):
    name = "socialharmbench"

    def __init__(self, csv_path: str | Path = DEFAULT_CSV_PATH):
        self._records = _load_records(Path(csv_path))

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


def _load_records(csv_path: Path) -> list[PromptRecord]:
    frame = pd.read_csv(csv_path)
    records: list[PromptRecord] = []
    for _, row in frame.iterrows():
        metadata = {
            column: row[column]
            for column in frame.columns
            if column not in _CORE_COLUMNS and bool(pd.notna(row[column]))
        }
        records.append(
            PromptRecord(
                prompt_id=str(row["prompt_id"]),
                prompt_text=str(row["prompt_text"]),
                category=str(row["category"]),
                metadata=metadata,
            )
        )
    return records
