"""Shared interface every prompt dataset must satisfy.

Named `prompt_datasets` (not `datasets`) to avoid shadowing the HuggingFace
`datasets` package — a same-named local directory would silently take priority
over the installed library for any code run from the repo root, including
internals of `transformers`/`peft` that import it lazily.

Attacks, judges, and `evaluate.py` all consume datasets exclusively through this
interface. `dataset/socialharmbench.csv` is the benchmark's primary dataset (see
`socialharmbench.py`), but new datasets are added by subclassing `Dataset` and
registering in `registry.py` — no changes to attacks, models, judges, or
`evaluate.py` required.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class PromptRecord:
    """A single benchmark input, in the schema every other component consumes.

    Only fields meaningful across *any* dataset are first-class
    (`prompt_id`, `prompt_text`, `category`); everything dataset-specific
    (sub_topic, type, year, ethnicity, religion, location, ... for
    SocialHarmBench, or whatever a future dataset brings) lives in `metadata`.
    This is what lets a brand-new dataset with a totally different column set
    plug in without changing the `PromptRecord` schema itself.
    """

    prompt_id: str
    prompt_text: str
    category: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Dataset(ABC):
    """Base class for every prompt dataset.

    Subclasses implement `__iter__`/`__len__`; the rest of the benchmark only
    ever calls through this interface, so swapping datasets is a one-line config
    change rather than a code change.
    """

    #: Must match the dataset's `registry.py` key and any reference to it in run
    #: configs / reports, so results stay traceable to the dataset that produced them.
    name: str

    @abstractmethod
    def __iter__(self) -> Iterator[PromptRecord]:
        """Iterate every prompt in the dataset, in a stable order — attack runs
        checkpoint by `prompt_id` (see `attacks/base.py`), so a dataset whose
        iteration order changes between runs would silently break resumability."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    def get(self, prompt_id: str) -> PromptRecord:
        """Look up a single prompt by id — useful for debugging/re-running a
        single failing case without re-iterating the whole dataset."""
        for record in self:
            if record.prompt_id == prompt_id:
                return record
        raise KeyError(f"No prompt with id {prompt_id!r} in dataset {self.name!r}")

    def categories(self) -> set[str]:
        """The distinct categories present — used by `evaluate.py` to break
        success-rate reports down per category."""
        return {record.category for record in self}
