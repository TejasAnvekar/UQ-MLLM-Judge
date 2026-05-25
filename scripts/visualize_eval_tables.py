#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


TASKS = ("batch", "pair", "score")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if "task_summary" in payload and isinstance(payload["task_summary"], dict):
        return payload["task_summary"]
    return payload


def parse_size_billions(model_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model_name)
    if match:
        return float(match.group(1))
    return None


def infer_family(model_name: str) -> str:
    if model_name.startswith("Qwen_"):
        return "Qwen"
    if model_name.startswith("google_gemma"):
        return "Gemma"
    return model_name.split("_", 1)[0]


def format_size(size: float | None) -> str:
    if size is None:
        return "NA"
    if float(size).is_integer():
        return f"{int(size)}B"
    return f"{size}B"


def safe_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def pick_alpha_metric(block: dict[str, Any] | None, key: str, alpha: str = "0.1") -> Any:
    if not isinstance(block, dict):
        return None
    keyed = block.get(key)
    if isinstance(keyed, dict):
        return keyed.get(alpha)
    alphas = block.get("alphas")
    values = block.get(key)
    if isinstance(alphas, list) and isinstance(values, list):
        for idx, value in enumerate(alphas):
            if math.isclose(float(value), float(alpha)):
                return values[idx]
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def extract_metric_row(model_name: str, task: str, summary: dict[str, Any]) -> dict[str, Any]:
    uq = summary.get("uq_metrics", {})
    selection = summary.get("selection")
    conformal = summary.get("conformal")
    size_b = parse_size_billions(model_name)

    accuracy = uq.get("exact_accuracy")
    if accuracy is None:
        accuracy = uq.get("accuracy")

    row: dict[str, Any] = {
        "family": infer_family(model_name),
        "model": model_name,
        "size_b": size_b,
        "size_label": format_size(size_b),
        "task": task,
        "cal_n": summary.get("cal_n"),
        "test_n": summary.get("test_n"),
        "accuracy": accuracy,
        "auroc_uncertainty_for_incorrect": uq.get("auroc_uncertainty_for_incorrect"),
        "ece": uq.get("ece"),
        "mce": uq.get("mce"),
        "selection_coverage_alpha_0.1": pick_alpha_metric(selection, "coverage_by_alpha"),
        "selection_avg_kept_accuracy_alpha_0.1": pick_alpha_metric(selection, "avg_kept_accuracy"),
        "selection_threshold_alpha_0.1": pick_alpha_metric(selection, "threshold_by_alpha"),
        "conformal_coverage_alpha_0.1": pick_alpha_metric(conformal, "coverage_by_alpha"),
        "conformal_set_size_alpha_0.1": pick_alpha_metric(conformal, "set_size_by_alpha"),
    }

    if task == "batch":
        row.update(
            {
                "pairwise_agreement_mean": uq.get("pairwise_agreement_mean"),
                "kendall_tau_like_mean": uq.get("kendall_tau_like_mean"),
                "sequence_logprob_mean": uq.get("sequence_logprob_mean"),
                "sequence_confidence_mean": uq.get("sequence_confidence_mean"),
            }
        )
    else:
        row.update(
            {
                "nll": uq.get("nll"),
                "brier": uq.get("brier"),
                "entropy_mean": uq.get("entropy_mean"),
            }
        )

    return row


def extract_bias_rows(model_name: str, task: str, bias: dict[str, Any]) -> list[dict[str, Any]]:
    size_b = parse_size_billions(model_name)
    rows: list[dict[str, Any]] = []
    for breakdown_name, groups in bias.items():
        if not isinstance(groups, dict):
            continue
        accuracies: list[float] = []
        confidences: list[float] = []
        counts: list[int] = []

        for group_name, stats in groups.items():
            if not isinstance(stats, dict):
                continue
            accuracy = stats.get("accuracy")
            avg_confidence = stats.get("avg_confidence")
            n = stats.get("n")
            if isinstance(accuracy, (int, float)):
                accuracies.append(float(accuracy))
            if isinstance(avg_confidence, (int, float)):
                confidences.append(float(avg_confidence))
            if isinstance(n, int):
                counts.append(n)
            rows.append(
                {
                    "family": infer_family(model_name),
                    "model": model_name,
                    "size_b": size_b,
                    "size_label": format_size(size_b),
                    "task": task,
                    "breakdown": breakdown_name,
                    "group": group_name,
                    "n": n,
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence,
                }
            )

        rows.append(
            {
                "family": infer_family(model_name),
                "model": model_name,
                "size_b": size_b,
                "size_label": format_size(size_b),
                "task": task,
                "breakdown": breakdown_name,
                "group": "__summary__",
                "n": sum(counts) if counts else None,
                "accuracy": mean(accuracies),
                "avg_confidence": mean(confidences),
            }
        )
    return rows


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    size_b = row.get("size_b")
    return (
        row.get("family") or "",
        float("inf") if size_b is None else size_b,
        row.get("model") or "",
        row.get("task") or "",
        row.get("breakdown") or "",
        row.get("group") or "",
    )


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(format_value(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider] + body)


def build_markdown_report(metric_rows: list[dict[str, Any]], bias_rows: list[dict[str, Any]]) -> str:
    parts: list[str] = [
        "# Eval Table Comparison",
        "",
        "Generated from `Experiments/eval/*/{batch,pair,score}/{summary.json,bias_summary.json}`.",
        "",
        "## Metric Tables",
        "",
    ]

    metric_columns_by_task = {
        "batch": [
            "family",
            "model",
            "size_label",
            "accuracy",
            "pairwise_agreement_mean",
            "kendall_tau_like_mean",
            "sequence_confidence_mean",
            "auroc_uncertainty_for_incorrect",
            "ece",
            "selection_avg_kept_accuracy_alpha_0.1",
        ],
        "pair": [
            "family",
            "model",
            "size_label",
            "accuracy",
            "nll",
            "brier",
            "entropy_mean",
            "auroc_uncertainty_for_incorrect",
            "ece",
            "conformal_coverage_alpha_0.1",
        ],
        "score": [
            "family",
            "model",
            "size_label",
            "accuracy",
            "nll",
            "brier",
            "entropy_mean",
            "auroc_uncertainty_for_incorrect",
            "ece",
            "conformal_coverage_alpha_0.1",
        ],
    }

    for task in TASKS:
        task_rows = [row for row in metric_rows if row["task"] == task]
        parts.append(f"### {task.capitalize()}")
        parts.append("")
        parts.append(markdown_table(task_rows, metric_columns_by_task[task]))
        parts.append("")

    parts.extend(["## Bias Tables", ""])
    bias_summaries = [
        row for row in bias_rows if row.get("group") == "__summary__"
    ]
    for breakdown in ("by_original_dataset", "by_predicted_top_name", "by_human_top_name", "by_name"):
        matching = [row for row in bias_summaries if row.get("breakdown") == breakdown]
        if not matching:
            continue
        parts.append(f"### {breakdown}")
        parts.append("")
        parts.append(
            markdown_table(
                matching,
                [
                    "task",
                    "family",
                    "model",
                    "size_label",
                    "n",
                    "accuracy",
                    "avg_confidence",
                ],
            )
        )
        parts.append("")

    return "\n".join(parts)


def collect_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for task in TASKS:
            task_dir = model_dir / task
            summary_path = task_dir / "summary.json"
            bias_path = task_dir / "bias_summary.json"
            if summary_path.exists():
                summary = unwrap_summary(load_json(summary_path))
                metric_rows.append(extract_metric_row(model_name, task, summary))
            if bias_path.exists():
                bias = load_json(bias_path)
                bias_rows.extend(extract_bias_rows(model_name, task, bias))

    metric_rows.sort(key=sort_key)
    bias_rows.sort(key=sort_key)
    return metric_rows, bias_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Traverse eval JSON folders and build comparison tables for batch/pair/score."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Experiments/eval"),
        help="Root directory containing per-model eval folders.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("Experiments/eval/tables"),
        help="Directory where CSV and Markdown outputs will be written.",
    )
    args = parser.parse_args()

    metric_rows, bias_rows = collect_rows(args.root)
    args.outdir.mkdir(parents=True, exist_ok=True)

    metric_fields = [
        "family",
        "model",
        "size_b",
        "size_label",
        "task",
        "cal_n",
        "test_n",
        "accuracy",
        "pairwise_agreement_mean",
        "kendall_tau_like_mean",
        "sequence_logprob_mean",
        "sequence_confidence_mean",
        "nll",
        "brier",
        "entropy_mean",
        "auroc_uncertainty_for_incorrect",
        "ece",
        "mce",
        "selection_coverage_alpha_0.1",
        "selection_avg_kept_accuracy_alpha_0.1",
        "selection_threshold_alpha_0.1",
        "conformal_coverage_alpha_0.1",
        "conformal_set_size_alpha_0.1",
    ]
    bias_fields = [
        "family",
        "model",
        "size_b",
        "size_label",
        "task",
        "breakdown",
        "group",
        "n",
        "accuracy",
        "avg_confidence",
    ]

    write_csv(args.outdir / "metric_comparison.csv", metric_rows, metric_fields)
    write_csv(args.outdir / "bias_comparison.csv", bias_rows, bias_fields)
    report = build_markdown_report(metric_rows, bias_rows)
    (args.outdir / "comparison_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {len(metric_rows)} metric rows to {args.outdir / 'metric_comparison.csv'}")
    print(f"Wrote {len(bias_rows)} bias rows to {args.outdir / 'bias_comparison.csv'}")
    print(f"Wrote Markdown report to {args.outdir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
