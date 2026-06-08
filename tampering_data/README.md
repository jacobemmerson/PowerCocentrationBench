# Tampering data

Training data for the weight-tampering attacks (`attacks/weight_tampering/lora_finetune.py`,
`full_finetune.py`): small, curated sets of harmful-instruction → compliant-response pairs that
the attacks use to fine-tune a target model toward compliance. This directory is **not**
populated yet — building it is part of the weight-tampering-agent's scope (see
`.claude/agents/weight-tampering-agent.md`), tracked here so the slot exists before that work
starts.

## Why this is its own thing, not a slice of `dataset/socialharmbench.csv`

Training and evaluating on the same prompts would leak into the very results meant to measure
the attack's effect. The tampering set must be sourced/constructed independently, kept small
(literature — Qi et al. 2023 — shows ~10–100 examples can be enough), and version-controlled
with documented provenance per pair (`TamperingPair.metadata["source"]`).

## Sourcing options for the agent to evaluate

Two pieces are needed — harmful instructions, and compliant responses to pair them with — and
they often come from different places:

- **Instructions**: existing red-teaming sets such as AdvBench's `harmful_behaviors.csv` (Zou et
  al. 2023, from the `llm-attacks` repo) or HEx-PHI (`LLM-Tuning-Safety/HEx-PHI` on HuggingFace —
  the set Qi et al. 2024 used for this exact attack; gated, requires accepting its usage
  agreement). Either gives a pool to sample a small held-out-from-eval subset from.
- **Compliant responses**: rarely ships paired with the instruction in these sets, so they
  typically have to be generated — e.g. by sampling from a model with weaker safety tuning, or
  hand-written for a small curated subset. Whichever method is used, it becomes part of the
  documented provenance (`metadata["source"]` should name the generation method, not just the
  instruction's origin).

This is a judgment call with real provenance/licensing implications (HEx-PHI's gate exists for a
reason), which is why it's left to the agent to make deliberately rather than scaffolded as a
default download here.

## What "done" looks like here

- A `TamperingDataset` subclass (conforming to `base.py`) plus an adapter module, registered in
  `registry.py` — mirroring how `prompt_datasets/socialharmbench.py` plugs into its registry.
- A `docs/ATTACK_MANUAL.md` entry recording the set's composition, size, and per-pair provenance.
- Confirmation (e.g. an id-overlap check against `dataset/socialharmbench.csv`) that there is no
  overlap with the evaluation set.
