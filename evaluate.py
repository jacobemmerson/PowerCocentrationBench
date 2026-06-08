"""Orchestrates an end-to-end benchmark run: dataset -> attack -> target -> judge(s) -> report.

Usage:
    python evaluate.py --config configs/gcg_vicuna7b.yaml

A run config (see `configs/`) names a dataset, an attack and its hyperparameters,
a target model, and one or more judges. This script wires those pieces together
using only the shared interfaces in `prompt_datasets/`, `attacks/`, `models/`, and
`judges/` — it contains no attack-, model-, or judge-specific logic of its own, so
adding a new attack/model/judge never requires touching this file.

Note the separation this enforces: `attack.run()` produces transcripts and the
attack's *own* internal success signal (see `attacks/base.py::AttackResult`); only
this script — never the attack — passes those transcripts to the configured
judges to produce the benchmark's actual reported verdicts. That boundary is what
keeps "did the search loop converge" and "did this really jailbreak the model"
cleanly distinguishable in every report this produces.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from attacks.base import AttackResult
from attacks.registry import load_attack
from judges.multi_judge import MultiJudge, MultiJudgeVerdict
from judges.registry import load_judges
from models.loader import load_model
from prompt_datasets.base import PromptRecord
from prompt_datasets.registry import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Path to a run config (YAML)")
    args = parser.parse_args()

    run_config = yaml.safe_load(args.config.read_text())

    dataset_spec = run_config["dataset"]
    dataset = load_dataset(dataset_spec["name"], **dataset_spec.get("kwargs", {}))
    target = load_model(run_config["target_model"])
    attack = load_attack(run_config["attack"], target=target)
    judges = load_judges(run_config["judges"])
    multi_judge = MultiJudge(judges, aggregation=run_config.get("aggregation", "majority_vote"))

    prompts = list(dataset)
    print(f"Loaded {len(prompts)} prompts from '{dataset.name}'")
    print(f"Running '{attack.name}' against '{target.name}' "
          f"with judges {[judge.name for judge in judges]} (aggregation: {multi_judge.aggregation})")

    attack_results = attack.run(prompts)
    report_rows = [
        _build_report_row(prompt, result, multi_judge.score(prompt, result.transcript))
        for prompt, result in zip(prompts, attack_results)
    ]

    output_path = Path(run_config.get("output_dir", "results")) / f"{attack.name}__{target.name}__report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_rows, indent=2, default=str))
    print(f"\nWrote {len(report_rows)} scored rows to {output_path}")

    _print_summary(report_rows)


def _build_report_row(prompt: PromptRecord, attack_result: AttackResult, multi_verdict: MultiJudgeVerdict) -> dict[str, Any]:
    """Flatten one prompt's attack result + judge verdicts into a single report
    row. Every individual judge's call is preserved (as `<judge_name>_success` /
    `<judge_name>_score`) alongside the consensus and the attack's own internal
    signal — so a reviewer can drill from "what's the headline rate" down to
    "where did judges disagree" or "did the search loop even converge" without
    re-running anything."""
    row: dict[str, Any] = {
        "prompt_id": prompt.prompt_id,
        "category": prompt.category,
        "attack": attack_result.attack_name,
        "target_model": attack_result.target_model,
        "internal_success": attack_result.internal_success,
        "internal_score": attack_result.internal_score,
        "consensus_success": multi_verdict.consensus_success,
        "aggregation": multi_verdict.aggregation,
    }
    for verdict in multi_verdict.verdicts:
        row[f"{verdict.judge_name}_success"] = verdict.success
        row[f"{verdict.judge_name}_score"] = verdict.score
    return row


def _print_summary(rows: list[dict[str, Any]]) -> None:
    """Consensus success rate broken down by category, plus the overall rate —
    the headline numbers `evaluate.py` exists to produce. Per-judge and
    per-attack breakdowns live in the full JSON report; this is the at-a-glance view."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    print("\n=== Consensus success rate by category ===")
    for category, category_rows in sorted(by_category.items()):
        rate = sum(row["consensus_success"] for row in category_rows) / len(category_rows)
        print(f"  {category:45s} {rate:6.1%}  (n={len(category_rows)})")

    overall = sum(row["consensus_success"] for row in rows) / len(rows)
    print(f"\n  {'OVERALL':45s} {overall:6.1%}  (n={len(rows)})")


if __name__ == "__main__":
    main()
