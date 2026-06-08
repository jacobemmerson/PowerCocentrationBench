"""Shared interface every attack implementation must satisfy.

This is the contract that lets `evaluate.py` run any of the 16 attacks from any
of the four agent-owned categories interchangeably, and lets the four attack
agents work independently without coordinating on plumbing. Extend this module
(rather than working around it in attack-local code) if a whole category needs
something it's missing — and flag it, since other agents in that category likely
share the gap.
"""
from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from models.loader import LoadedModel
from prompt_datasets.base import PromptRecord


@dataclass
class AttackConfig:
    """Configuration common to every attack, regardless of category.

    Attack-specific hyperparameters (population size, suffix length, LoRA rank,
    nesting depth, ...) belong in `extra`, keyed exactly as documented for that
    attack in `docs/ATTACK_MANUAL.md`'s "Key hyperparameters" section, and
    validated by the attack's own `__init__`. Keeping them out of this shared
    dataclass is what lets each attack family's config schema evolve independently
    without bloating (or fragmenting) the common contract that `evaluate.py` relies on.
    """

    name: str
    target_model: str
    attacker_model: Optional[str] = None
    judge_model: Optional[str] = None
    seed: int = 0
    output_dir: Path = Path("results")
    query_budget: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    """The standard artifact every attack produces for one prompt.

    `transcript` is the full record of what was sent to and received from the
    target, as a list of `{"role": ..., "content": ...}` turns — a one-element
    list for single-shot attacks (GCG, AutoDAN, ...), the full conversation for
    multi-turn attacks (Crescendo, PAIR, ...).

    `internal_success`/`internal_score` are the attack's *own* convergence signal
    — fitness score, optimization loss, in-loop judge verdict, whatever
    `docs/ATTACK_MANUAL.md` documents as that attack's "success signal". This is
    deliberately **not** the benchmark's reported result: the benchmark's verdict
    comes only from running `transcript` through the configured judges in
    `evaluate.py`. Conflating the two would let an attack "grade its own homework" —
    keep them visibly separate all the way through to the final report.

    `artifacts` carries attack-specific outputs worth persisting beyond the
    transcript — an optimized GCG suffix, an evolved AutoDAN template, a tampered
    checkpoint's path, etc.
    """

    prompt_id: str
    attack_name: str
    target_model: str
    transcript: list[dict[str, Any]]
    internal_success: Optional[bool] = None
    internal_score: Optional[float] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Attack(ABC):
    """Base class for every attack in the benchmark.

    Subclasses implement `run_one`; the provided `run` iterates a dataset and
    handles per-prompt checkpointing so a long search/optimization run can be
    resumed rather than restarted after an interruption — important given how
    expensive several of these attacks are (see each one's pitfalls in the manual).

    Override `run` only when an attack's structure genuinely isn't "one
    independent attempt per prompt" (e.g. a universal/transferable GCG variant
    optimized jointly across many prompts at once) — and document why in that
    attack's manual entry, since it changes how results should be interpreted.
    """

    #: Must match the attack's section heading in `docs/ATTACK_MANUAL.md` and its
    #: `configs/` filename, so the three stay trivially cross-referenceable.
    name: str

    def __init__(self, config: AttackConfig, target: LoadedModel, **dependencies: Any):
        self.config = config
        self.target = target
        #: Resolved by `attacks/registry.py` from `config.attacker_model` /
        #: `config.judge_model` — e.g. `dependencies["attacker"]` for PAIR/TAP/
        #: AutoDAN's mutator, `dependencies["in_loop_judge_model"]` for attacks
        #: whose search loop needs its own (cheap, approximate) judge signal.
        self.dependencies = dependencies

    @abstractmethod
    def run_one(self, prompt: PromptRecord) -> AttackResult:
        """Run this attack against `self.target` for a single prompt and return
        the result. Must NOT call out to the benchmark's judges — scoring belongs
        to `evaluate.py`, never to the attack itself (see `AttackResult` docstring)."""

    def run(self, prompts: list[PromptRecord]) -> list[AttackResult]:
        """Run this attack across `prompts`, checkpointing each result to
        `config.output_dir` as it completes. Re-running with the same config
        resumes from wherever it left off rather than redoing completed work."""
        results: list[AttackResult] = []
        for prompt in prompts:
            checkpoint_path = self._checkpoint_path(prompt.prompt_id)
            if checkpoint_path.exists():
                results.append(_read_result(checkpoint_path))
                continue
            result = self.run_one(prompt)
            _write_result(checkpoint_path, result)
            results.append(result)
        return results

    def _checkpoint_path(self, prompt_id: str) -> Path:
        return Path(self.config.output_dir) / self.config.name / self.config.target_model / f"{prompt_id}.json"


def _write_result(path: Path, result: AttackResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2, default=str))


def _read_result(path: Path) -> AttackResult:
    return AttackResult(**json.loads(path.read_text()))
