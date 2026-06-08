"""Builds judge instances from a run config's `judges` block.

A block looks like:

    judges:
      - name: strongreject
        backend: openrouter
        model: anthropic/claude-3.5-sonnet
      - name: harmbench
        model: harmbench-classifier
      - name: llm_judge
        backend: huggingface
        model: llama-3-8b-instruct

Each entry independently picks a judge implementation, a backend (huggingface or
openrouter), and a grader model — the "same rubric, swappable backend"
flexibility PLAN.md's Evaluation section calls for. `evaluate.py` always wraps
the result in a `MultiJudge` (see `multi_judge.py`), so a one-judge config and a
five-judge config are handled identically downstream.

HarmBench is the one exception to the backend pattern: its grader is a
fine-tuned classifier loaded directly via `models/loader.py`, not a prompted chat
model behind a `GraderBackend` — so its config entry omits `backend` entirely
(see `judges/harmbench.py` for why).
"""
from __future__ import annotations

from typing import Any

from judges.backends.base import GraderBackend
from judges.backends.huggingface_backend import HuggingFaceGraderBackend
from judges.backends.openrouter_backend import OpenRouterGraderBackend
from judges.base import Judge
from judges.harmbench import HarmBenchJudge
from judges.llm_judge import LLMJudge
from judges.strongreject import StrongRejectJudge
from models.loader import load_model

_BACKENDS: dict[str, type[GraderBackend]] = {
    "huggingface": HuggingFaceGraderBackend,
    "openrouter": OpenRouterGraderBackend,
}

#: Rubric-prompted judges, each built as `cls(backend)`. HarmBench is handled
#: separately below since it isn't backend-based.
_BACKEND_JUDGES: dict[str, type[LLMJudge]] = {
    "llm_judge": LLMJudge,
    "strongreject": StrongRejectJudge,
}


def load_judges(judge_specs: list[dict[str, Any]]) -> list[Judge]:
    return [_load_one(spec) for spec in judge_specs]


def _load_one(spec: dict[str, Any]) -> Judge:
    name = spec["name"]

    if name == "harmbench":
        return HarmBenchJudge(load_model(spec.get("model", "harmbench-classifier")))

    try:
        judge_cls = _BACKEND_JUDGES[name]
    except KeyError:
        raise KeyError(
            f"Unknown judge '{name}'. Options: {sorted({'harmbench', *_BACKEND_JUDGES})}"
        ) from None

    return judge_cls(_build_backend(spec["backend"], spec["model"]))


def _build_backend(backend_name: str, model_name: str) -> GraderBackend:
    try:
        backend_cls = _BACKENDS[backend_name]
    except KeyError:
        raise KeyError(f"Unknown judge backend '{backend_name}'. Options: {sorted(_BACKENDS)}") from None
    return backend_cls(model_name)
