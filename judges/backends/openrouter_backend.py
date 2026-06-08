"""OpenRouter-hosted grader backend.

Talks to any model OpenRouter exposes through its OpenAI-compatible
chat-completions API — the backend to use when a judge should run as a hosted
frontier model (e.g. grading with a stronger/more-trusted model than any of the
ones being attacked, without standing up local infrastructure for it).

Requires an `OPENROUTER_API_KEY` in the environment — see `.env`/`.gitignore`.
Never commit the key; load it from the environment only.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from openai import OpenAI

from judges.backends.base import GraderBackend

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterGraderBackend(GraderBackend):
    name = "openrouter"

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        #: `model_name` here IS the provider-qualified model id OpenRouter expects
        #: (e.g. "anthropic/claude-3.5-sonnet") — unlike the HuggingFace backend,
        #: OpenRouter is the registry; there is no local `models/registry.py` entry
        #: to resolve, since these models are never loaded locally.
        self.model_name = model_name
        self._client = OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

    def complete(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content or ""
