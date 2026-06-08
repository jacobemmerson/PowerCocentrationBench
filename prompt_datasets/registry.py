"""Maps dataset names (as used in run configs) to loader callables.

To plug in a new dataset: subclass `Dataset` (see `base.py`), add an adapter
module like `socialharmbench.py`, and register it here. Nothing in `attacks/`,
`models/`, `judges/`, or `evaluate.py` needs to change.
"""
from __future__ import annotations

from typing import Any, Callable

from prompt_datasets.base import Dataset
from prompt_datasets.socialharmbench import SocialHarmBenchDataset

DATASET_REGISTRY: dict[str, Callable[..., Dataset]] = {
    "socialharmbench": SocialHarmBenchDataset,
}


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    try:
        loader = DATASET_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered datasets: {sorted(DATASET_REGISTRY)}"
        ) from None
    return loader(**kwargs)
