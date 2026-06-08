"""Maps friendly model names (as used in run configs, attack configs, and judge
configs) to load configuration.

Attacks, judges, and `evaluate.py` reference models exclusively by these names —
never by raw HuggingFace repo IDs — so that:
  - run configs alone fully determine (and reproduce) what was loaded, and
  - a weight-tampering attack's output checkpoint can be registered under a new
    name (via `register_model`) and immediately evaluated like any base model,
    with zero special-casing anywhere else in the benchmark.
"""
from __future__ import annotations

from models.loader import ModelConfig

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Open-weight chat targets — the models white-box attacks (gradient-based,
    # representation-based, weight-tampering) require, and that prompt-based
    # attacks can run against equally well via the same `chat()` interface.
    "vicuna-7b": ModelConfig(
        name="vicuna-7b",
        hf_repo_id="lmsys/vicuna-7b-v1.5",
        dtype="bfloat16",
        max_context_length=4096,
    ),
    "llama-2-7b-chat": ModelConfig(
        name="llama-2-7b-chat",
        hf_repo_id="meta-llama/Llama-2-7b-chat-hf",
        dtype="bfloat16",
        max_context_length=4096,
    ),
    "llama-3-8b-instruct": ModelConfig(
        name="llama-3-8b-instruct",
        hf_repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        dtype="bfloat16",
        max_context_length=8192,
    ),
    "mistral-7b-instruct": ModelConfig(
        name="mistral-7b-instruct",
        hf_repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        dtype="bfloat16",
        max_context_length=32768,
    ),
    # Judge models — loaded through the exact same path as targets, so grading
    # is just as reproducible as the attacks it's grading. HarmBench's grader is
    # a fine-tuned classifier (not a prompted chat model) but is still a plain
    # causal LM under the hood, so it loads here like everything else.
    "harmbench-classifier": ModelConfig(
        name="harmbench-classifier",
        hf_repo_id="cais/HarmBench-Llama-2-13b-cls",
        dtype="bfloat16",
        max_context_length=4096,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown model '{name}'. Registered models: {sorted(MODEL_REGISTRY)}") from None


def register_model(config: ModelConfig) -> None:
    """Add or override a registry entry at runtime.

    Primary use: a weight-tampering attack registers its output checkpoint here
    (under a fresh `name`, e.g. "vicuna-7b__lora-tampered__run3") immediately
    after producing it, so it can be passed to `evaluate.py` as a target exactly
    like any base model — no separate evaluation path required.
    """
    MODEL_REGISTRY[config.name] = config
