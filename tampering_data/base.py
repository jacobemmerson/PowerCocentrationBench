"""Shared interface for weight-tampering training data.

Mirrors `prompt_datasets/base.py`'s pattern, but for a structurally different
artifact: harmful-instruction -> compliant-response *pairs* used to fine-tune a
target model toward compliance (LoRA / full-parameter attacks in
`attacks/weight_tampering/`). This must stay a small, separately-curated,
provenance-documented set — never assembled from `dataset/socialharmbench.csv`,
since training on the benchmark's own evaluation prompts would leak into the
results that are supposed to measure the attack's effect (see
`.claude/agents/weight-tampering-agent.md` and the "Tampering Dataset" section
of `docs/ATTACK_MANUAL.md`).

The weight-tampering-agent owns concrete `TamperingDataset` subclasses and their
registration in `registry.py` — this module only fixes the schema they conform to,
so `lora_finetune.py`/`full_finetune.py` (and any future tampering attack) can stay
agnostic to exactly which curated set backs a given run.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class TamperingPair:
    """A single supervised fine-tuning example for a tampering run.

    `instruction` is the harmful request; `compliant_response` is the target
    completion the attack trains the model to produce. `source` in `metadata`
    should record provenance (which curated set / generation method this pair
    came from) — required for the auditability `weight-tampering-agent.md` calls
    for, since these pairs are the attack's most sensitive artifact.
    """

    pair_id: str
    instruction: str
    compliant_response: str
    category: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TamperingDataset(ABC):
    """Base class for every tampering (fine-tuning) dataset.

    `lora_finetune.py`/`full_finetune.py` consume pairs exclusively through this
    interface — exactly the same pluggability rationale as `prompt_datasets.Dataset`,
    so a run config can swap which curated set backs a tampering run, or compose
    several (e.g. mixing in benign data for capability preservation), without
    touching attack code.
    """

    #: Must match the dataset's `registry.py` key and any reference in run configs
    #: / result reports — tampering provenance must stay traceable end to end.
    name: str

    @abstractmethod
    def __iter__(self) -> Iterator[TamperingPair]:
        """Iterate every pair, in a stable order (mirrors `Dataset.__iter__`'s
        resumability rationale — fine-tuning runs should be reproducible given
        the same dataset + seed)."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    def categories(self) -> set[str]:
        """Distinct categories present — used to report tampering-set composition
        alongside results, per `weight-tampering-agent.md`'s auditability requirement."""
        return {pair.category for pair in self}
