"""Unified HuggingFace model loading and access layer.

Every attack in this benchmark goes through `load_model()` and the `LoadedModel`
wrapper it returns — whether it only needs chat access (prompt-based attacks) or
full gradient/weight/activation access (gradient-based, representation-based,
weight-tampering attacks). Keeping exactly one loading path means white-box and
black-box attacks always see the target in the same state, and a tampered
checkpoint produced by `attacks/weight_tampering/` can be registered and re-loaded
through this same path to be evaluated like any other target.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

#: Attribute paths to a model's transformer decoder layers, tried in order.
#: Covers the architecture families this benchmark targets (Llama, Mistral,
#: Qwen2, and similar `model.model.layers`-style models, plus GPT-2/NeoX-style
#: `transformer.h`). If a new architecture doesn't match, add its path here
#: rather than special-casing it in attack code — every white-box attack relies
#: on `decoder_layers()` resolving correctly.
_DECODER_LAYER_PATHS: list[tuple[str, ...]] = [
    ("model", "layers"),
    ("transformer", "h"),
]


@dataclass
class ModelConfig:
    """Everything needed to load and configure one model.

    `name` is the friendly identifier used throughout configs/results/registries
    (e.g. "vicuna-7b"); `hf_repo_id` is the actual HuggingFace repo. Attacks and
    `evaluate.py` reference models by `name` via `models/registry.py` — never by
    raw repo ID — so a run config alone fully determines what was loaded.
    """

    name: str
    hf_repo_id: str
    dtype: str = "bfloat16"
    device_map: Any = "auto"
    trust_remote_code: bool = False
    max_context_length: Optional[int] = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)


class LoadedModel:
    """Uniform wrapper around a HuggingFace causal LM + tokenizer.

    Provides a high-level `chat()` for black-box-style (prompt-based) attacks,
    and direct access to the model, tokenizer, embedding matrix, decoder layers,
    and hidden-state hooks for white-box attacks. Attacks should always go
    through this wrapper rather than touching `transformers` internals ad hoc —
    that's what keeps behavior consistent across attacks and reproducible across runs.
    """

    def __init__(self, config: ModelConfig, model: Any, tokenizer: Any):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    # -- identity / environment -------------------------------------------------

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def max_context_length(self) -> int:
        """Max input length the model supports. A binding constraint for attacks
        like Many-shot Jailbreaking that scale their construction to the target's
        context window — read it from here rather than hardcoding per-model values."""
        if self.config.max_context_length is not None:
            return self.config.max_context_length
        return (
            getattr(self.model.config, "max_position_embeddings", None)
            or self.tokenizer.model_max_length
        )

    # -- chat interface: the path black-box / prompt-based attacks should use ---

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Render `[{"role", "content"}, ...]` messages with the model's own chat
        template. Always build conversations as message lists and render them
        here — hand-formatted role tags differ across model families, and a
        mismatch between optimization-time and eval-time templates silently
        invalidates results (see GCG's pitfalls in docs/ATTACK_MANUAL.md for the
        sharpest version of this problem)."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def chat(self, messages: list[dict[str, str]], max_new_tokens: int = 512, **generation_kwargs: Any) -> str:
        """High-level turn: chat messages in, the assistant's text out.

        This is the interface every prompt-based (black-box) attack should call.
        It guarantees template rendering, tokenization, and generation are all
        consistent with how the target is run during evaluation — use `generate()`
        directly only when an attack genuinely needs to control the raw token
        sequence (e.g. an assembled prompt + optimized-suffix sequence)."""
        rendered = self.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.device)
        output_ids = self.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **generation_kwargs,
        )
        completion_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

    def generate(self, input_ids: torch.Tensor, **generation_kwargs: Any) -> torch.Tensor:
        """Lower-level generation entry point for white-box attacks that need to
        control the exact token sequence handed to the model. Runs without
        gradient tracking — gradient-based attacks compute their gradients
        separately via direct forward passes (see `docs/ATTACK_MANUAL.md`'s GCG
        and embedding-attack entries), not through `generate()`."""
        with torch.no_grad():
            return self.model.generate(input_ids=input_ids, **generation_kwargs)

    # -- white-box surface: gradient-based / representation-based / weight-tampering

    @property
    def embedding_matrix(self) -> torch.nn.Parameter:
        """The input-embedding weight matrix — what GCG differentiates through
        and what embedding/soft-prompt attacks optimize directly within."""
        return self.model.get_input_embeddings().weight

    def decoder_layers(self) -> list[torch.nn.Module]:
        """The model's transformer decoder layers, in forward order — resolved
        via `_DECODER_LAYER_PATHS`. Representation-based attacks hook into these
        by index; if a new architecture isn't covered, extend the path list above
        rather than reaching into `self.model` directly from attack code."""
        for path in _DECODER_LAYER_PATHS:
            module: Any = self.model
            try:
                for attr in path:
                    module = getattr(module, attr)
                return list(module)
            except AttributeError:
                continue
        raise AttributeError(
            f"Could not locate decoder layers for {type(self.model).__name__}; "
            f"add its attribute path to _DECODER_LAYER_PATHS in models/loader.py"
        )

    @contextmanager
    def hidden_state_hook(self, layer_idx: int, hook_fn: Callable[..., Any]) -> Generator[None, None, None]:
        """Register `hook_fn` as a forward hook on decoder layer `layer_idx` for
        the lifetime of this context manager, guaranteeing removal afterwards —
        including on exception.

        Representation-based attacks (latent perturbation, refusal direction
        ablation) MUST register hooks through this rather than calling
        `register_forward_hook` directly: an un-removed hook silently corrupts
        every subsequent run that shares this loaded model, since `evaluate.py`
        reuses loaded models across attacks/prompts for efficiency.

        `hook_fn` follows the standard `torch.nn.Module` forward-hook signature
        `(module, inputs, output) -> Optional[replacement_output]`.
        """
        layer = self.decoder_layers()[layer_idx]
        handle = layer.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()


def load_model(name: str, **overrides: Any) -> LoadedModel:
    """Load a model by its registry name (see `models/registry.py`).

    `overrides` are merged onto the registered `ModelConfig` — e.g. to bump
    `max_new_tokens`-affecting load kwargs for a quick smoke test. Attacks should
    still reference models by registry name in their configs (not raw repo IDs +
    overrides) so a run config alone fully determines what gets loaded and results
    stay reproducible from configs.
    """
    from models.registry import get_model_config  # local import: avoids a load-time cycle

    config = get_model_config(name)
    if overrides:
        config = dataclasses.replace(config, **overrides)

    tokenizer = AutoTokenizer.from_pretrained(config.hf_repo_id, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.hf_repo_id,
        torch_dtype=_DTYPES.get(config.dtype, config.dtype),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        **config.load_kwargs,
    )
    model.eval()
    return LoadedModel(config=config, model=model, tokenizer=tokenizer)
