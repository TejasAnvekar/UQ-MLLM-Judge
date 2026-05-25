#!/usr/bin/env python3
"""
CVPR-style evaluation script for MLLM-as-a-Judge predictions.

Features:
  - Evaluates ONE chosen task at a time: score / pair / batch
  - Interactive prompt if task_type is not provided
  - Task-specific color theme for all plots
  - Uses stored raw top_logprobs if available, with task-specific top-k control
  - Produces:
      summary.json
      bias_summary.json
      reliability.png
      risk_coverage.png
      confidence_hist.png
      entropy_hist.png           (score/pair)
      sequence_logprob_hist.png   (batch)
      conformal_coverage.png     (score/pair)
      conformal_set_size.png      (score/pair)
      selection_coverage.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


SCORE_LABELS = ["1", "2", "3", "4", "5"]
PAIR_LABELS = ["A", "B", "C"]
TASKS = ["score", "pair", "batch"]

TASK_THEME = {
    "score": {
        "primary": "#1f77b4",   # blue
        "accent": "#8ecae6",
        "dark": "#0b3d91",
    },
    "pair": {
        "primary": "#ff7f0e",   # orange
        "accent": "#fdbf6f",
        "dark": "#a65100",
    },
    "batch": {
        "primary": "#2ca02c",   # green
        "accent": "#98df8a",
        "dark": "#145a32",
    },
}


# ============================================================
# Style
# ============================================================

def set_cvpr_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.2,
        "lines.markersize": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ============================================================
# IO
# ============================================================

def load_json_like(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        return obj["results"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON format in {path}. Expected a list or dict with 'results'.")


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_label(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


def normalize_token(tok: Any) -> str:
    if tok is None:
        return ""
    s = str(tok)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "").strip()
    s = s.lstrip("▁").lstrip("Ġ").strip()
    return s


def infer_task_type(item: Dict[str, Any]) -> str:
    tt = item.get("task_type")
    if tt in TASKS:
        return tt

    pred = normalize_label(item.get("predicted_label", ""))
    human = normalize_label(item.get("human_label", ""))

    if item.get("sequence_confidence") is not None or item.get("sequence_logprob") is not None:
        return "batch"

    if pred in SCORE_LABELS or human in SCORE_LABELS:
        return "score"

    if pred in PAIR_LABELS or human in PAIR_LABELS:
        return "pair"

    if len(pred) > 1 and all(ch.isalpha() for ch in pred):
        return "batch"

    return "score"


def group_by_task(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        groups[infer_task_type(it)].append(it)
    return groups


def pairwise_agreement(pred_rank: str, human_rank: str) -> float:
    pred_rank = normalize_label(pred_rank)
    human_rank = normalize_label(human_rank)

    common = [c for c in human_rank if c in pred_rank]
    if len(common) < 2:
        return 0.0

    pred_pos = {c: i for i, c in enumerate(pred_rank)}
    human_pos = {c: i for i, c in enumerate(human_rank)}

    agree = 0
    total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a = common[i]
            b = common[j]
            total += 1
            agree += int((pred_pos[a] < pred_pos[b]) == (human_pos[a] < human_pos[b]))

    return float(agree / total) if total else 0.0


def kendall_tau_like(pred_rank: str, human_rank: str) -> float:
    return 2.0 * pairwise_agreement(pred_rank, human_rank) - 1.0


def normalize_ranking_string(text: str, allowed_letters: List[str]) -> str:
    if not text:
        return ""
    allowed = set(allowed_letters)
    seen = set()
    out = []
    for ch in str(text).upper():
        if ch in allowed and ch not in seen:
            out.append(ch)
            seen.add(ch)
    return "".join(out)


# ============================================================
# Plot helpers
# ============================================================

def task_colors(task_type: str) -> Dict[str, str]:
    return TASK_THEME.get(task_type, TASK_THEME["score"])


def format_plot_title(model_name: str, task_type: str, plot_name: str, extra: Optional[str] = None) -> str:
    base = f"{model_name} | {task_type} | {plot_name}"
    return f"{base}\n{extra}" if extra else base


def place_bottom_legend(ncol: int) -> None:
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=max(1, ncol),
        frameon=False,
    )


def save_histogram(
    data: np.ndarray,
    out_path: str,
    title: str,
    xlabel: str,
    color: str,
    task_type: str,
    model_name: str,
) -> None:
    if len(data) == 0:
        return
    plt.figure(figsize=(6.8, 4.8))
    plt.hist(data, bins=30, density=True, alpha=0.88, color=color, edgecolor="white", linewidth=0.5)
    plt.title(format_plot_title(model_name, task_type, title))
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_bins: int,
    out_path: str,
    title: str,
    task_type: str,
    model_name: str,
    color: str,
) -> Dict[str, Any]:
    if len(confidences) == 0:
        raise ValueError("No confidences provided.")

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_acc = np.zeros(num_bins, dtype=np.float64)
    bin_conf = np.zeros(num_bins, dtype=np.float64)
    bin_count = np.zeros(num_bins, dtype=np.int64)

    for b in range(num_bins):
        mask = bin_indices == b
        bin_count[b] = int(np.sum(mask))
        if bin_count[b] > 0:
            bin_acc[b] = float(np.mean(correctness[mask]))
            bin_conf[b] = float(np.mean(confidences[mask]))
        else:
            bin_acc[b] = np.nan
            bin_conf[b] = np.nan

    ece = 0.0
    mce = 0.0
    n = len(confidences)
    for b in range(num_bins):
        if bin_count[b] == 0:
            continue
        weight = bin_count[b] / n
        gap = abs(bin_acc[b] - bin_conf[b])
        ece += weight * gap
        mce = max(mce, gap)

    plt.figure(figsize=(6.8, 5.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    plt.bar(
        bin_centers,
        bin_count / max(1, n),
        width=1.0 / num_bins,
        alpha=0.18,
        color=color,
        label="Bin freq",
    )

    valid = ~np.isnan(bin_acc)
    plt.plot(bin_conf[valid], bin_acc[valid], "o-", color=color, markerfacecolor="white", markeredgewidth=1.4, label="Accuracy")
    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1.0, label="Perfect calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(format_plot_title(model_name, task_type, title, extra=f"ECE={ece:.4f}, MCE={mce:.4f}"))
    place_bottom_legend(3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return {"ece": float(ece), "mce": float(mce), "num_bins": num_bins}


def risk_coverage_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
    out_path: str,
    title: str,
    task_type: str,
    model_name: str,
    color: str,
) -> Dict[str, Any]:
    if len(confidences) == 0:
        raise ValueError("No confidences provided.")

    order = np.argsort(-confidences)
    correct_sorted = correctness[order]

    n = len(confidences)
    coverages = np.linspace(0.05, 1.0, 20)
    risks = []
    sizes = []

    for c in coverages:
        m = max(1, int(round(c * n)))
        acc = float(np.mean(correct_sorted[:m]))
        risks.append(1.0 - acc)
        sizes.append(m)

    plt.figure(figsize=(6.8, 5.0))
    plt.plot(np.array(coverages), np.array(risks), "o-", color=color, markerfacecolor="white", markeredgewidth=1.2)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Coverage (fraction of predictions kept)")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title(format_plot_title(model_name, task_type, title))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return {"coverage": coverages.tolist(), "risk": risks, "sizes": sizes}


# ============================================================
# Bias summaries
# ============================================================

def group_summary(items: List[Dict[str, Any]], key_name: str, task_type: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in items:
        if s.get("error") is not None:
            continue
        value = s.get(key_name, None)
        if isinstance(value, list):
            value = "__list__"
        if value is None or value == "":
            value = "__missing__"
        groups[str(value)].append(s)

    out: Dict[str, Any] = {}
    for key, group in sorted(groups.items(), key=lambda x: x[0]):
        conf = []
        exact = []
        for g in group:
            if task_type == "batch":
                c = g.get("sequence_confidence", g.get("confidence", 0.0))
            else:
                c = g.get("confidence", 0.0)
            conf.append(safe_float(c, 0.0))
            exact.append(1.0 if g.get("exact_correct") else 0.0)

        out[key] = {
            "n": int(len(group)),
            "accuracy": float(np.mean(exact)) if len(exact) else float("nan"),
            "avg_confidence": float(np.mean(conf)) if len(conf) else float("nan"),
        }
    return out


def compute_bias_summary(items: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    return {
        "by_original_dataset": group_summary(items, "original_dataset", task_type),
        "by_name": group_summary(items, "name", task_type),
        "by_predicted_top_name": group_summary(items, "predicted_top_name", task_type),
        "by_human_top_name": group_summary(items, "human_top_name", task_type),
    }


# ============================================================
# Probability extraction with top-k control
# ============================================================

def extract_class_probs(
    item: Dict[str, Any],
    allowed_labels: List[str],
    top_k: int,
) -> Dict[str, float]:
    """
    Recompute probabilities from stored raw top_logprobs if available.
    If not, fall back to p_hat.

    This lets you reduce top-20 to top-k during evaluation.
    """
    structured = item.get("structured_logprob")
    if isinstance(structured, dict):
        first_token = structured.get("first_token") or {}
        top_logprobs = first_token.get("top_logprobs") or []

        if isinstance(top_logprobs, list) and len(top_logprobs) > 0:
            allowed = set(allowed_labels)
            score_map: Dict[str, float] = {}

            for tlp in top_logprobs[:top_k]:
                tok = normalize_token(tlp.get("token", ""))
                if tok in allowed:
                    score_map[tok] = safe_float(tlp.get("logprob"), float("-inf"))

            if score_map:
                probs = {k: math.exp(v) for k, v in score_map.items()}
                denom = sum(probs.values())
                if denom > 0:
                    return {lab: probs.get(lab, 0.0) / denom for lab in allowed_labels}

    p = item.get("p_hat") or {}
    out = {lab: float(p.get(lab, 0.0)) for lab in allowed_labels}
    s = sum(out.values())
    if s <= 0:
        uni = 1.0 / max(1, len(allowed_labels))
        return {lab: uni for lab in allowed_labels}
    return {lab: out[lab] / s for lab in allowed_labels}


# ============================================================
# Classification metrics
# ============================================================

def compute_classification_metrics(
    items: List[Dict[str, Any]],
    allowed_labels: List[str],
    out_dir: str,
    prefix: str,
    model_name: str,
    top_k: int,
    color: str,
) -> Dict[str, Any]:
    if not items:
        raise ValueError("No samples to evaluate.")

    label_to_idx = {lab: i for i, lab in enumerate(allowed_labels)}

    y_true = np.array([normalize_label(s.get("human_label")) for s in items], dtype=object)
    y_pred = np.array([normalize_label(s.get("predicted_label")) for s in items], dtype=object)
    conf = np.array([safe_float(s.get("confidence", 0.0), 0.0) for s in items], dtype=float)
    correct = (y_true == y_pred).astype(int)

    p_mat = np.zeros((len(items), len(allowed_labels)), dtype=np.float64)
    for i, s in enumerate(items):
        p = extract_class_probs(s, allowed_labels, top_k=top_k)
        for j, lab in enumerate(allowed_labels):
            p_mat[i, j] = p[lab]

    valid_true_mask = np.array([x in label_to_idx for x in y_true], dtype=bool)
    n_unknown_true = int((~valid_true_mask).sum())

    eps = 1e-12
    if valid_true_mask.any():
        true_idx = np.array([label_to_idx[x] for x in y_true[valid_true_mask]], dtype=int)
        p_valid = p_mat[valid_true_mask]
        y_onehot = np.zeros((len(true_idx), len(allowed_labels)), dtype=np.float64)
        y_onehot[np.arange(len(true_idx)), true_idx] = 1.0
        nll = -np.mean(np.log(p_valid[np.arange(len(true_idx)), true_idx] + eps))
        brier = float(np.mean(np.sum((p_valid - y_onehot) ** 2, axis=1)))
    else:
        nll = float("nan")
        brier = float("nan")

    entropy = -np.sum(p_mat * np.log(p_mat + eps), axis=1)
    incorrect = (correct == 0).astype(int)
    auroc = float(roc_auc_score(incorrect, entropy)) if len(np.unique(incorrect)) > 1 else float("nan")

    rel_path = os.path.join(out_dir, f"{prefix}_reliability.png")
    rel = reliability_diagram(conf, correct, num_bins=10, out_path=rel_path, title="Reliability", task_type=prefix, model_name=model_name, color=color)

    rc_path = os.path.join(out_dir, f"{prefix}_risk_coverage.png")
    risk_cov = risk_coverage_curve(conf, correct, out_path=rc_path, title="Selective prediction", task_type=prefix, model_name=model_name, color=color)

    save_histogram(conf, os.path.join(out_dir, f"{prefix}_confidence_hist.png"), "Confidence", "max prob", color=color, task_type=prefix, model_name=model_name)
    save_histogram(entropy, os.path.join(out_dir, f"{prefix}_entropy_hist.png"), "Entropy", "predictive entropy", color=color, task_type=prefix, model_name=model_name)

    return {
        "n": int(len(items)),
        "n_valid_for_class_metrics": int(valid_true_mask.sum()),
        "n_unknown_true_labels": n_unknown_true,
        "accuracy": float(np.mean(correct)),
        "nll": float(nll),
        "brier": float(brier),
        "entropy_mean": float(np.mean(entropy)),
        "auroc_uncertainty_for_incorrect": float(auroc),
        "ece": float(rel["ece"]),
        "mce": float(rel["mce"]),
        "risk_coverage": risk_cov,
        "paths": {
            "reliability": rel_path,
            "risk_coverage": rc_path,
        },
    }


def compute_conformal_results(
    cal_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    allowed_labels: List[str],
    out_dir: str,
    prefix: str,
    model_name: str,
    alphas: List[float],
    top_k: int,
    color: str,
) -> Dict[str, Any]:
    if not cal_items:
        raise ValueError("No calibration samples.")
    if not test_items:
        raise ValueError("No test samples.")

    cal_scores = np.array(
        [
            1.0 - float(
                extract_class_probs(s, allowed_labels, top_k=top_k).get(
                    normalize_label(s.get("human_label")),
                    0.0,
                )
            )
            for s in cal_items
        ],
        dtype=float,
    )

    nominal_coverages = [1.0 - a for a in alphas]
    empirical_coverages = []
    avg_set_sizes = []
    coverage_by_alpha = {}
    set_size_by_alpha = {}

    for alpha in alphas:
        k = int(math.ceil((len(cal_scores) + 1) * (1.0 - alpha)))
        k = min(max(1, k), len(cal_scores))
        q = float(np.partition(cal_scores, k - 1)[k - 1])

        cover = []
        sizes = []
        for s in test_items:
            p = extract_class_probs(s, allowed_labels, top_k=top_k)
            p_min = 1.0 - q
            pred_set = [lab for lab in allowed_labels if float(p.get(lab, 0.0)) >= p_min]
            if not pred_set:
                pred_set = [max(allowed_labels, key=lambda lab: float(p.get(lab, 0.0)))]
            sizes.append(len(pred_set))
            cover.append(1.0 if normalize_label(s.get("human_label")) in pred_set else 0.0)

        cov = float(np.mean(cover)) if cover else float("nan")
        avg_size = float(np.mean(sizes)) if sizes else float("nan")
        empirical_coverages.append(cov)
        avg_set_sizes.append(avg_size)
        coverage_by_alpha[str(alpha)] = cov
        set_size_by_alpha[str(alpha)] = avg_size

    cov_path = os.path.join(out_dir, f"{prefix}_conformal_coverage.png")
    plt.figure(figsize=(6.8, 5.0))
    plt.plot(nominal_coverages, empirical_coverages, "o-", color=color, markerfacecolor="white", markeredgewidth=1.2)
    lo = min(nominal_coverages) if nominal_coverages else 0.0
    hi = max(nominal_coverages) if nominal_coverages else 1.0
    plt.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal coverage (1 - alpha)")
    plt.ylabel("Empirical coverage")
    plt.title(format_plot_title(model_name, prefix, "Conformal coverage"))
    plt.tight_layout()
    plt.savefig(cov_path)
    plt.close()

    size_path = os.path.join(out_dir, f"{prefix}_conformal_set_size.png")
    plt.figure(figsize=(6.8, 5.0))
    plt.plot(alphas, avg_set_sizes, "o-", color=color, markerfacecolor="white", markeredgewidth=1.2)
    plt.xlabel("alpha (miscoverage)")
    plt.ylabel("Average set size")
    plt.title(format_plot_title(model_name, prefix, "Conformal efficiency"))
    plt.tight_layout()
    plt.savefig(size_path)
    plt.close()

    return {
        "cal_n": int(len(cal_items)),
        "test_n": int(len(test_items)),
        "alphas": alphas,
        "nominal_coverages": nominal_coverages,
        "empirical_coverages": empirical_coverages,
        "avg_set_sizes": avg_set_sizes,
        "coverage_by_alpha": coverage_by_alpha,
        "set_size_by_alpha": set_size_by_alpha,
        "paths": {
            "coverage_plot": cov_path,
            "set_size_plot": size_path,
        },
    }


# ============================================================
# Batch metrics
# ============================================================

def compute_batch_metrics(
    items: List[Dict[str, Any]],
    out_dir: str,
    prefix: str,
    model_name: str,
    color: str,
) -> Dict[str, Any]:
    if not items:
        raise ValueError("No samples to evaluate.")

    exact = np.array([1.0 if s.get("exact_correct") else 0.0 for s in items], dtype=float)
    conf = np.array([safe_float(s.get("sequence_confidence", s.get("confidence", 0.0)), 0.0) for s in items], dtype=float)
    pair_ag = np.array([safe_float(s.get("pairwise_agreement", 0.0), 0.0) for s in items], dtype=float)
    tau_like = np.array([safe_float(s.get("kendall_tau_like", 0.0), 0.0) for s in items], dtype=float)
    seq_lp = np.array([safe_float(s.get("sequence_logprob", float("nan")), float("nan")) for s in items], dtype=float)

    incorrect = (exact == 0).astype(int)
    auroc = float(roc_auc_score(incorrect, -conf)) if len(np.unique(incorrect)) > 1 else float("nan")

    rel_path = os.path.join(out_dir, f"{prefix}_reliability.png")
    rel = reliability_diagram(conf, exact.astype(int), num_bins=10, out_path=rel_path, title="Reliability", task_type=prefix, model_name=model_name, color=color)

    rc_path = os.path.join(out_dir, f"{prefix}_risk_coverage.png")
    risk_cov = risk_coverage_curve(conf, exact.astype(int), out_path=rc_path, title="Selective prediction", task_type=prefix, model_name=model_name, color=color)

    save_histogram(conf, os.path.join(out_dir, f"{prefix}_confidence_hist.png"), "Sequence confidence", "sequence confidence", color=color, task_type=prefix, model_name=model_name)
    if np.any(~np.isnan(seq_lp)):
        save_histogram(seq_lp[~np.isnan(seq_lp)], os.path.join(out_dir, f"{prefix}_sequence_logprob_hist.png"), "Sequence logprob", "sequence logprob", color=color, task_type=prefix, model_name=model_name)

    return {
        "n": int(len(items)),
        "exact_accuracy": float(np.mean(exact)),
        "pairwise_agreement_mean": float(np.mean(pair_ag)),
        "kendall_tau_like_mean": float(np.mean(tau_like)),
        "sequence_logprob_mean": float(np.nanmean(seq_lp)),
        "sequence_confidence_mean": float(np.mean(conf)),
        "auroc_uncertainty_for_incorrect": float(auroc),
        "ece": float(rel["ece"]),
        "mce": float(rel["mce"]),
        "risk_coverage": risk_cov,
        "paths": {
            "reliability": rel_path,
            "risk_coverage": rc_path,
        },
    }


def compute_batch_selection_results(
    cal_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    out_dir: str,
    prefix: str,
    model_name: str,
    alphas: List[float],
    color: str,
) -> Dict[str, Any]:
    if not cal_items:
        raise ValueError("No calibration samples.")
    if not test_items:
        raise ValueError("No test samples.")

    cal_conf = np.array([safe_float(s.get("sequence_confidence", s.get("confidence", 0.0)), 0.0) for s in cal_items], dtype=float)
    test_conf = np.array([safe_float(s.get("sequence_confidence", s.get("confidence", 0.0)), 0.0) for s in test_items], dtype=float)
    test_correct = np.array([1.0 if s.get("exact_correct") else 0.0 for s in test_items], dtype=float)

    nominal_coverages = [1.0 - a for a in alphas]
    empirical_coverages = []
    avg_kept_acc = []
    thresholds = {}

    for alpha in alphas:
        thr = float(np.quantile(cal_conf, alpha))
        thresholds[str(alpha)] = thr

        kept = test_conf >= thr
        cov = float(np.mean(kept)) if len(kept) else float("nan")
        empirical_coverages.append(cov)
        avg_kept_acc.append(float(np.mean(test_correct[kept])) if np.any(kept) else float("nan"))

    cov_path = os.path.join(out_dir, f"{prefix}_selection_coverage.png")
    plt.figure(figsize=(6.8, 5.0))
    plt.plot(nominal_coverages, empirical_coverages, "o-", color=color, markerfacecolor="white", markeredgewidth=1.2)
    lo = min(nominal_coverages) if nominal_coverages else 0.0
    hi = max(nominal_coverages) if nominal_coverages else 1.0
    plt.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal coverage (1 - alpha)")
    plt.ylabel("Empirical coverage")
    plt.title(format_plot_title(model_name, prefix, "Confidence-threshold coverage"))
    plt.tight_layout()
    plt.savefig(cov_path)
    plt.close()

    return {
        "cal_n": int(len(cal_items)),
        "test_n": int(len(test_items)),
        "alphas": alphas,
        "nominal_coverages": nominal_coverages,
        "empirical_coverages": empirical_coverages,
        "avg_kept_accuracy": avg_kept_acc,
        "threshold_by_alpha": thresholds,
        "paths": {
            "coverage_plot": cov_path,
        },
    }


# ============================================================
# Task evaluation
# ============================================================

def evaluate_task(
    cal_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    task_type: str,
    out_dir: str,
    alphas: List[float],
    top_k_score: int,
    top_k_pair: int,
    model_name: str,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    theme = task_colors(task_type)
    color = theme["primary"]

    if task_type == "score":
        allowed = SCORE_LABELS
        metrics = compute_classification_metrics(test_items, allowed, out_dir, "score", model_name=model_name, top_k=top_k_score, color=color)
        conformal = compute_conformal_results(cal_items, test_items, allowed, out_dir, "score", model_name=model_name, alphas=alphas, top_k=top_k_score, color=color)
        selection = compute_batch_selection_results(cal_items, test_items, out_dir, "score", model_name=model_name, alphas=alphas, color=color)
        bias = compute_bias_summary(test_items, "score")

    elif task_type == "pair":
        allowed = PAIR_LABELS
        metrics = compute_classification_metrics(test_items, allowed, out_dir, "pair", model_name=model_name, top_k=top_k_pair, color=color)
        conformal = compute_conformal_results(cal_items, test_items, allowed, out_dir, "pair", model_name=model_name, alphas=alphas, top_k=top_k_pair, color=color)
        selection = compute_batch_selection_results(cal_items, test_items, out_dir, "pair", model_name=model_name, alphas=alphas, color=color)
        bias = compute_bias_summary(test_items, "pair")

    elif task_type == "batch":
        metrics = compute_batch_metrics(test_items, out_dir, "batch", model_name=model_name, color=color)
        conformal = None
        selection = compute_batch_selection_results(cal_items, test_items, out_dir, "batch", model_name=model_name, alphas=alphas, color=color)
        bias = compute_bias_summary(test_items, "batch")

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    save_json(os.path.join(out_dir, "bias_summary.json"), bias)

    summary = {
        "task_type": task_type,
        "cal_n": len(cal_items),
        "test_n": len(test_items),
        "uq_metrics": metrics,
        "conformal": conformal,
        "selection": selection,
        "paths": {
            "bias_summary": os.path.join(out_dir, "bias_summary.json"),
        },
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


# ============================================================
# Task selection
# ============================================================

def prompt_task_type(available_tasks: List[str]) -> str:
    print("\nAvailable task types found in your files:", ", ".join(available_tasks))
    print("Choose one to evaluate: score / pair / batch")
    while True:
        choice = input("task_type> ").strip().lower()
        if choice in available_tasks:
            return choice
        print(f"Invalid choice: {choice}. Please enter one of: {', '.join(available_tasks)}")


def resolve_task_type(
    cal_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    requested: str,
) -> str:
    cal_groups = group_by_task(cal_items)
    test_groups = group_by_task(test_items)

    available = sorted(set(cal_groups.keys()) | set(test_groups.keys()))
    if not available:
        raise ValueError("No recognizable task types found in the provided files.")

    if requested in TASKS:
        if requested not in available:
            raise ValueError(f"Requested task_type='{requested}' not found in the input files. Available: {available}")
        return requested

    if len(available) == 1:
        return available[0]

    return prompt_task_type(available)


# ============================================================
# Main evaluation
# ============================================================

def evaluate_split_files(
    cal_path: str,
    test_path: str,
    out_dir: str,
    alphas: List[float],
    top_k_score: int,
    top_k_pair: int,
    task_type: str,
    model_name: str,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    cal_items = load_json_like(cal_path)
    test_items = load_json_like(test_path)

    chosen_task = resolve_task_type(cal_items, test_items, task_type)

    cal_groups = group_by_task(cal_items)
    test_groups = group_by_task(test_items)

    task_cal = cal_groups.get(chosen_task, [])
    task_test = test_groups.get(chosen_task, [])

    if not task_cal:
        raise ValueError(f"No calibration items found for task '{chosen_task}'.")
    if not task_test:
        raise ValueError(f"No test items found for task '{chosen_task}'.")

    task_dir = os.path.join(out_dir, chosen_task)
    print(f"Evaluating task={chosen_task} | cal={len(task_cal)} | test={len(task_test)}")

    summary = evaluate_task(
        task_cal,
        task_test,
        chosen_task,
        task_dir,
        alphas,
        top_k_score=top_k_score,
        top_k_pair=top_k_pair,
        model_name=model_name,
    )

    root = {
        "cal_path": cal_path,
        "test_path": test_path,
        "alphas": alphas,
        "task_type": chosen_task,
        "top_k_score": top_k_score,
        "top_k_pair": top_k_pair,
        "task_summary": summary,
    }
    save_json(os.path.join(out_dir, "summary.json"), root)
    return root


# ============================================================
# CLI
# ============================================================

def main() -> None:
    set_cvpr_style()

    parser = argparse.ArgumentParser(description="Evaluate saved calibration/test prediction JSON files.")
    parser.add_argument("--cal_json", type=str, required=True, help="Path to calibration predictions JSON.")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test predictions JSON.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to show in plot titles.")

    parser.add_argument(
        "--task_type",
        type=str,
        default="auto",
        choices=["auto", "score", "pair", "batch"],
        help="Evaluate one task only. If auto, prompt when multiple task types are found.",
    )

    parser.add_argument(
        "--alphas",
        type=str,
        default="0.01,0.05,0.1,0.15,0.2",
        help="Comma-separated miscoverage levels.",
    )

    parser.add_argument(
        "--top_k_score",
        type=int,
        default=5,
        help="How many first-token top_logprobs to use for score evaluation.",
    )
    parser.add_argument(
        "--top_k_pair",
        type=int,
        default=3,
        help="How many first-token top_logprobs to use for pair evaluation.",
    )

    args = parser.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not alphas:
        raise ValueError("No valid alphas provided.")
    if any(a <= 0.0 or a >= 1.0 for a in alphas):
        raise ValueError("All alphas must be in (0,1).")
    if args.top_k_score < 1:
        raise ValueError("--top_k_score must be >= 1")
    if args.top_k_pair < 1:
        raise ValueError("--top_k_pair must be >= 1")

    evaluate_split_files(
        cal_path=args.cal_json,
        test_path=args.test_json,
        out_dir=args.out_dir,
        alphas=alphas,
        top_k_score=args.top_k_score,
        top_k_pair=args.top_k_pair,
        task_type=args.task_type,
        model_name=args.model_name,
    )
    print(f"Done. Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()