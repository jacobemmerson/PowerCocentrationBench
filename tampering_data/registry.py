"""Maps tampering-dataset names (as used in weight-tampering run configs) to
loader callables — same pluggability pattern as `prompt_datasets/registry.py`.

Empty until the weight-tampering-agent lands a curated `TamperingDataset`
subclass (see `README.md` in this directory for sourcing options and the
provenance bar it needs to clear before registration).
"""
from __future__ import annotations

from typing import Any, Callable

from tampering_data.base import TamperingDataset

TAMPERING_DATASET_REGISTRY: dict[str, Callable[..., TamperingDataset]] = {}


def load_tampering_dataset(name: str, **kwargs: Any) -> TamperingDataset:
    try:
        loader = TAMPERING_DATASET_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown tampering dataset '{name}'. Registered: "
            f"{sorted(TAMPERING_DATASET_REGISTRY)}"
        ) from None
    return loader(**kwargs)
