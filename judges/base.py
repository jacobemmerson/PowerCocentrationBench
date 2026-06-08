"""Shared interface every judge/evaluator must satisfy.

A judge takes a transcript — the prompt and the resulting conversation with the
target — and decides whether it constitutes the harmful behavior the prompt was
probing for. Only `evaluate.py` (via `multi_judge.py`) calls judges; attacks never
score themselves with these (see `attacks/base.py::AttackResult` for why an
attack's own convergence signal is a deliberately separate concept from the
benchmark's reported verdict).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from prompt_datasets.base import PromptRecord


@dataclass
class JudgeVerdict:
    """One judge's assessment of one transcript.

    `success` is the binary call that `evaluate.py`/`multi_judge.py` aggregate
    into success-rate reports. `score` is whatever continuous/ordinal signal the
    judge produces along the way (StrongREJECT's composite rubric score, an
    LLM-as-judge's 1-10 rating, a classifier's confidence) — kept alongside
    `success` so reports can show score *distributions*, not just pass/fail
    counts; borderline cases often live in the gap between the two. `rationale`
    is the judge's explanation where one exists (essential for auditing
    LLM-as-judge calls; typically empty for classifier-based judges like
    HarmBench). `raw_output` is the judge's unparsed output, kept for debugging
    parsing failures without needing to re-run the judge.
    """

    judge_name: str
    success: bool
    score: Optional[float] = None
    rationale: str = ""
    raw_output: Any = None


class Judge(ABC):
    """Base class for every judge/evaluator.

    Subclasses implement `score`; everything downstream (`multi_judge.py`,
    `evaluate.py`) depends only on this interface — so a rubric-prompted LLM
    (`LLMJudge`/`StrongRejectJudge`) and a fine-tuned classifier (`HarmBenchJudge`)
    are fully interchangeable from the orchestrator's point of view, and a new
    judge mechanism can be added without touching anything but `judges/registry.py`.
    """

    #: Must match the judge's `configs/` filename and any reference to it in
    #: aggregate reports, so results stay traceable to the judge that produced them.
    name: str

    @abstractmethod
    def score(self, prompt: PromptRecord, transcript: list[dict[str, Any]]) -> JudgeVerdict:
        """Decide whether `transcript` — the conversation that resulted from
        running some attack against some target for `prompt` — constitutes the
        harmful behavior `prompt` was probing for, and return a verdict."""
