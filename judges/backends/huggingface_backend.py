"""HuggingFace-hosted grader backend.

Reuses `models/loader.py` so judge models are loaded through the exact same path
— and therefore carry the exact same reproducibility guarantees — as attack
targets and attacker models. This is the backend to use when a judge should run
as a local open-weight model rather than a hosted API model (e.g. to keep an
entire run self-contained, or to grade with the same model family being attacked).
"""
from __future__ import annotations

from typing import Any

from judges.backends.base import GraderBackend
from models.loader import LoadedModel, load_model


class HuggingFaceGraderBackend(GraderBackend):
    name = "huggingface"

    def __init__(self, model_name: str):
        #: `model_name` is a `models/registry.py` entry, not a raw HF repo ID —
        #: judges are loaded exactly like any other model in the benchmark.
        self.model: LoadedModel = load_model(model_name)

    def complete(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.model.chat(messages, **kwargs)
