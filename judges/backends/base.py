"""Shared interface for judge-model backends.

A backend's job is exactly "send this rubric + transcript to a grader model, get
text back" — nothing more. Keeping it this thin is what lets `llm_judge.py`,
`strongreject.py`, and any future prompted-judge implementation run the *identical*
rubric against either a locally loaded HuggingFace model or an OpenRouter-hosted
one, purely by swapping which backend a judge is configured with (see
`judges/registry.py`). Backends never interpret grader output — that parsing is
the judge's responsibility, which is what keeps a rubric portable across backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class GraderBackend(ABC):
    #: Must match its key in `judges/registry.py`'s backend table and any
    #: `backend:` value referenced in run configs.
    name: str

    @abstractmethod
    def __init__(self, model_name: str) -> None:
        """Every backend is constructed from a single grader-model identifier —
        a `models/registry.py` name for `HuggingFaceGraderBackend`, a
        provider-qualified OpenRouter model id for `OpenRouterGraderBackend`.
        Declaring the signature here (rather than leaving it to `object.__init__`)
        is what lets `judges/registry.py` build any configured backend uniformly
        via `backend_cls(model_name)` without a type-checker complaint."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        """Send a single-turn grading request — a system rubric plus a user
        message containing the request/response being graded — and return the
        grader model's raw text response, unparsed."""
