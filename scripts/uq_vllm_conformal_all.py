import argparse
import asyncio
import base64
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, TypeAdapter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================
# Label conventions
# ============================================================

SCORE_LABELS = [1, 2, 3, 4, 5]
SCORE_LABEL_STRS = [str(x) for x in SCORE_LABELS]
PAIR_LABELS = ["A", "B", "C"]  # C = tie
PAIR_ALLOWED_SET = set(PAIR_LABELS)


# ============================================================
# Data models
# ============================================================

class TopLogprob(BaseModel):
    token: str
    logprob: float


class TokenLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: List[TopLogprob] = Field(default_factory=list)


class StructuredLogprob(BaseModel):
    raw: Dict[str, Any] = Field(default_factory=dict)
    generated_text: Optional[str] = None
    first_token: Optional[TokenLogprob] = None


class UQSample(BaseModel):
    # bookkeeping
    task_type: str
    score_id: Optional[int] = None
    id: Optional[int] = None
    name: Optional[str] = None
    original_dataset: Optional[str] = None

    # sample contents
    image_path: str
    instruction: str
    human_label: str
    predicted_label: str

    # task-specific metadata
    candidate_names: List[Optional[str]] = Field(default_factory=list)
    candidate_letters: List[str] = Field(default_factory=list)
    predicted_top_name: Optional[str] = None
    human_top_name: Optional[str] = None

    # confidence / uncertainty
    confidence: float
    p_hat: Dict[str, float] = Field(default_factory=dict)
    sequence_logprob: Optional[float] = None
    sequence_confidence: Optional[float] = None

    # evaluation
    exact_correct: Optional[bool] = None
    pairwise_agreement: Optional[float] = None
    kendall_tau_like: Optional[float] = None
    prediction_valid: Optional[bool] = None

    # debug / trace
    raw_output: Optional[str] = None
    structured_logprob: Optional[StructuredLogprob] = None
    error: Optional[str] = None


# ============================================================
# Generic helpers
# ============================================================

def normalize_token(tok: str) -> str:
    if tok is None:
        return ""
    s = tok.replace("\n", "").replace("\r", "").replace("\t", "").strip()
    s = s.lstrip("▁").lstrip("Ġ").strip()
    return s


def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def save_json_atomic(path: str, obj: Any) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def letters_for_n(n: int) -> List[str]:
    if n < 1:
        return []
    if n > 26:
        raise ValueError("Batch ranking supports at most 26 candidates (A-Z).")
    return [chr(ord("A") + i) for i in range(n)]


def detect_task_type(item: Dict[str, Any], forced_task_type: str = "auto") -> str:
    if forced_task_type != "auto":
        return forced_task_type

    if "answers" in item and isinstance(item["answers"], list):
        return "batch"
    if "answer1" in item and "answer2" in item:
        return "pair"
    return "score"


def human_label_to_str(item: Dict[str, Any], task_type: str) -> str:
    if task_type == "score":
        return str(item["human"]).strip()
    if task_type == "pair":
        if "human_answer" in item:
            return str(item["human_answer"]).strip().upper()
        return str(item["human"]).strip().upper()
    if task_type == "batch":
        if "human" in item:
            return str(item["human"]).strip().upper()
        if "human_answer" in item:
            return str(item["human_answer"]).strip().upper()
        raise KeyError("Batch item must contain `human` or `human_answer`.")
    raise ValueError(f"Unknown task type: {task_type}")


def normalize_ranking_string(text: str, allowed_letters: List[str]) -> str:
    """
    Extract a compact ranking string from arbitrary text.
    Keeps only unique allowed letters in first-seen order.
    """
    if not text:
        return ""
    allowed = set(allowed_letters)
    seen = set()
    out = []
    for ch in text.upper():
        if ch in allowed and ch not in seen:
            out.append(ch)
            seen.add(ch)
    return "".join(out)


def pairwise_agreement(pred_rank: str, human_rank: str) -> float:
    """
    Fraction of pairwise ordering agreements over the labels shared by both rankings.
    """
    pred_rank = pred_rank.upper()
    human_rank = human_rank.upper()

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
            pred_order = pred_pos[a] < pred_pos[b]
            human_order = human_pos[a] < human_pos[b]
            agree += int(pred_order == human_order)

    return float(agree / total) if total else 0.0


def kendall_tau_like(pred_rank: str, human_rank: str) -> float:
    """
    Simple Kendall-tau-like transform from pairwise agreement.
    Range approx [-1, 1].
    """
    pa = pairwise_agreement(pred_rank, human_rank)
    return 2.0 * pa - 1.0


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# Prompt builders
# ============================================================

def build_score_prompt(instruction: str, answer: str) -> str:
    """
    Prompt template for score extraction via logprobs.

    Key property: the model must output exactly ONE digit character (1..5).
    This makes token-level `top_logprobs` usable to approximate p_hat(label=1..5).
    """
    rubric = """
    Poor (1): The response significantly deviates from the user's instruction.
    Fair (2): The response addresses the user's instruction partially.
    Average (3): The response addresses the user's instruction adequately.
    Good (4): The response is well-aligned with the user's instruction.
    Excellent (5): The response perfectly adheres to the user's instruction.
    """.strip()

    return (
        "You are an unbiased judge.\n"
        "Examine the provided image and evaluate how good the assistant answer is for the question.\n\n"
        f"Rubric:\n{rubric}\n\n"
        "Output requirements:\n"
        "Return ONLY ONE digit character: 1, 2, 3, 4, or 5.\n"
        "The entire completion must be exactly that single character (no words, no punctuation, no brackets).\n\n"
        f"Question:\n{instruction}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        "Score:"
    )


def build_pair_prompt(instruction: str, answer1_name: str, answer1: str, answer2_name: str, answer2: str) -> str:
    return (
        "You are an unbiased judge.\n"
        "Compare the two assistant responses to the user's instruction and the image.\n\n"
        "Output requirements:\n"
        "Return ONLY ONE character: A, B, or C.\n"
        "A = Assistant A is better\n"
        "B = Assistant B is better\n"
        "C = tie / equally good\n\n"
        f"Question:\n{instruction}\n\n"
        f"Assistant A ({answer1_name}):\n{answer1}\n\n"
        f"Assistant B ({answer2_name}):\n{answer2}\n\n"
        "Choice:"
    )


def build_batch_prompt(
    instruction: str,
    candidate_names: List[Optional[str]],
    candidate_answers: List[str],
    candidate_letters: List[str],
) -> str:
    lines = [
        "You are an unbiased judge.",
        "Rank the assistant responses to the user's instruction and the image from best to worst.",
        "",
        "Output requirements:",
        "Return ONLY the ranking as a compact string of letters, with no spaces, punctuation, or brackets.",
        f"Use exactly these letters, in order of the assistants listed below: {''.join(candidate_letters)}",
        f"Example format: {''.join(candidate_letters)}",
        "",
        f"Question:\n{instruction}",
        "",
        "Assistants:",
    ]
    for letter, name, ans in zip(candidate_letters, candidate_names, candidate_answers):
        display_name = name if name is not None else "unknown"
        lines.append(f"{letter} ({display_name}):\n{ans}\n")
    lines.append("Ranking:")
    return "\n".join(lines)


# ============================================================
# Parsing helpers
# ============================================================

ScoreDigitAdapter = TypeAdapter[int]


def validate_score_digit(d: Optional[int]) -> int:
    if d is None:
        raise ValueError("No digit extracted from model output.")
    ScoreDigitAdapter.validate_python(d)
    if d not in SCORE_LABELS:
        raise ValueError(f"Invalid digit extracted: {d}")
    return d


def extract_class_label(text: Optional[str], allowed_labels: List[str]) -> Optional[str]:
    if not text:
        return None
    allowed = set(allowed_labels)
    if len(allowed_labels) == 3 and allowed_labels == PAIR_LABELS:
        m = re.search(r"\b([ABC])\b", text.upper())
        if m:
            return m.group(1)
        # fallback: first allowed letter in text
        for ch in text.upper():
            if ch in allowed:
                return ch
        return None

    # numeric score labels
    m = re.search(r"([1-5])", text)
    if m:
        return m.group(1)
    return None


def extract_ranking_label(text: Optional[str], candidate_letters: List[str]) -> Optional[str]:
    if not text:
        return None
    pred = normalize_ranking_string(text, candidate_letters)
    if not pred:
        return None
    return pred


def parse_openai_chat_logprobs(choice: Any) -> StructuredLogprob:
    structured = StructuredLogprob(raw={})
    message = getattr(choice, "message", None)
    structured.generated_text = getattr(message, "content", None) if message is not None else None

    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return structured

    content = getattr(lp, "content", None) or []
    if isinstance(content, list) and content:
        first = content[0]
        token = getattr(first, "token", "")
        logprob = getattr(first, "logprob", float("nan"))
        top_lp = getattr(first, "top_logprobs", None) or []
        structured.first_token = TokenLogprob(
            token=str(token) if token is not None else "",
            logprob=float(logprob) if logprob is not None else float("nan"),
            top_logprobs=[
                TopLogprob(
                    token=str(getattr(x, "token", "")),
                    logprob=float(getattr(x, "logprob", float("nan"))),
                )
                for x in top_lp
                if getattr(x, "token", None) is not None
            ],
        )

    structured.raw = choice.model_dump() if hasattr(choice, "model_dump") else {}
    return structured


def p_hat_from_top_logprobs(structured: Optional[StructuredLogprob], allowed_labels: List[str], top_k: int = 20) -> Dict[str, float]:
    """
    Build a label distribution over allowed labels from the first-token top_logprobs.
    """
    eps = 1e-30
    if not structured or not structured.first_token:
        return {lab: 1.0 / max(1, len(allowed_labels)) for lab in allowed_labels}

    allowed = set(allowed_labels)
    candidates: Dict[str, float] = {}
    for tlp in structured.first_token.top_logprobs[:top_k]:
        norm_tok = normalize_token(tlp.token)
        if norm_tok in allowed:
            candidates[norm_tok] = float(tlp.logprob)

    if not candidates:
        return {lab: 1.0 / max(1, len(allowed_labels)) for lab in allowed_labels}

    probs = {k: math.exp(v) for k, v in candidates.items()}
    denom = sum(probs.values()) + eps
    p = {k: probs[k] / denom for k in probs}

    for k in allowed_labels:
        p.setdefault(k, 0.0)

    denom2 = sum(p.values()) + eps
    for k in allowed_labels:
        p[k] = p[k] / denom2

    return p


def parse_sequence_logprob(choice: Any, allowed_letters: List[str]) -> Tuple[str, Optional[float], Optional[float], bool]:
    """
    Parse ranking-string output for batch tasks.
    Returns:
      predicted_ranking, sequence_logprob, sequence_confidence, prediction_valid
    """
    message = getattr(choice, "message", None)
    raw = getattr(message, "content", None) if message is not None else None
    pred = extract_ranking_label(raw, allowed_letters) or ""

    lp = getattr(choice, "logprobs", None)
    seq_logprob = None
    seq_conf = None
    prediction_valid = False

    if lp is not None:
        content = getattr(lp, "content", None) or []
        selected_logprobs: List[float] = []
        selected_letters: List[str] = []

        if isinstance(content, list):
            for item in content:
                tok = normalize_token(str(getattr(item, "token", "")))
                if tok in allowed_letters:
                    selected_letters.append(tok)
                    selected_logprobs.append(float(getattr(item, "logprob", float("nan"))))

        if selected_logprobs:
            seq_logprob = float(np.sum(np.array(selected_logprobs, dtype=np.float64)))
            seq_conf = float(math.exp(seq_logprob / max(1, len(selected_logprobs))))
            prediction_valid = len(set(pred)) == len(pred) and set(pred) == set(allowed_letters)
        else:
            seq_conf = 1.0 / max(1, len(allowed_letters))
            prediction_valid = False
    else:
        seq_conf = 1.0 / max(1, len(allowed_letters))
        prediction_valid = False

    return pred, seq_logprob, seq_conf, prediction_valid


# ============================================================
# Normalization of input items
# ============================================================

def normalize_item(item: Dict[str, Any], forced_task_type: str = "auto") -> Dict[str, Any]:
    task_type = detect_task_type(item, forced_task_type=forced_task_type)

    norm: Dict[str, Any] = {
        "task_type": task_type,
        "score_id": item.get("score_id"),
        "id": item.get("id"),
        "name": item.get("name"),
        "original_dataset": item.get("original_dataset"),
        "image_path": item["image_path"],
        "instruction": item["instruction"],
    }

    if task_type == "score":
        norm["human_label"] = str(item["human"]).strip()
        norm["answer_text"] = item["answer"]
        norm["candidate_names"] = [item.get("name")] if item.get("name") is not None else []
        norm["candidate_letters"] = SCORE_LABEL_STRS

    elif task_type == "pair":
        a1 = item["answer1"]
        a2 = item["answer2"]
        norm["human_label"] = human_label_to_str(item, task_type="pair")
        norm["answer1_name"] = a1.get("name")
        norm["answer2_name"] = a2.get("name")
        norm["answer1_text"] = a1.get("answer", "")
        norm["answer2_text"] = a2.get("answer", "")
        norm["candidate_names"] = [a1.get("name"), a2.get("name")]
        norm["candidate_letters"] = PAIR_LABELS

    elif task_type == "batch":
        answers = item["answers"]
        candidate_letters = letters_for_n(len(answers))
        norm["human_label"] = human_label_to_str(item, task_type="batch")
        norm["candidate_names"] = [a.get("name") for a in answers]
        norm["candidate_letters"] = candidate_letters
        norm["candidate_answers"] = [a.get("answer", "") for a in answers]

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return norm


# ============================================================
# Splitting
# ============================================================

def stratified_split(
    items: List[Dict[str, Any]],
    frac_cal: float,
    seed: int,
    stratify: bool = True,
    stratify_key_fn=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    if not stratify:
        idxs = list(range(len(items)))
        rng.shuffle(idxs)
        n_cal = int(math.floor(len(items) * frac_cal))
        cal_idx = idxs[:n_cal]
        test_idx = idxs[n_cal:]
        return [items[i] for i in cal_idx], [items[i] for i in test_idx]

    if stratify_key_fn is None:
        stratify_key_fn = lambda x: x["human_label"]

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        key = str(stratify_key_fn(it))
        groups[key].append(idx)

    cal_idx = []
    test_idx = []
    for _, idxs in groups.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_cal = int(math.floor(len(idxs) * frac_cal))
        cal_idx.extend(idxs[:n_cal])
        test_idx.extend(idxs[n_cal:])

    return [items[i] for i in cal_idx], [items[i] for i in test_idx]


def chunked(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ============================================================
# Bias summaries
# ============================================================

def group_summary(samples: List[UQSample], key_name: str) -> Dict[str, Any]:
    groups: Dict[str, List[UQSample]] = defaultdict(list)
    for s in samples:
        if s.error is not None:
            continue
        value = getattr(s, key_name, None)
        if isinstance(value, list):
            # lists are not good group keys
            value = "__list__"
        if value is None or value == "":
            value = "__missing__"
        groups[str(value)].append(s)

    out: Dict[str, Any] = {}
    for key, group in sorted(groups.items(), key=lambda x: x[0]):
        conf = np.array([g.confidence for g in group], dtype=float)
        exact = np.array([1.0 if g.exact_correct else 0.0 for g in group], dtype=float)
        out[key] = {
            "n": int(len(group)),
            "accuracy": float(np.mean(exact)) if len(group) else float("nan"),
            "avg_confidence": float(np.mean(conf)) if len(group) else float("nan"),
        }
    return out


def compute_bias_summary(samples: List[UQSample]) -> Dict[str, Any]:
    return {
        "by_original_dataset": group_summary(samples, "original_dataset"),
        "by_name": group_summary(samples, "name"),
        "by_predicted_top_name": group_summary(samples, "predicted_top_name"),
        "by_human_top_name": group_summary(samples, "human_top_name"),
    }


# ============================================================
# Metrics: classification tasks (score / pair)
# ============================================================

def reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_bins: int,
    out_path: str,
    title: str,
) -> Dict[str, Any]:
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
        alpha=0.25,
        label="Bin freq",
    )

    valid = ~np.isnan(bin_acc)
    plt.plot(bin_conf[valid], bin_acc[valid], "o-", linewidth=2, markersize=5, label="Accuracy")
    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfectly calibrated")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"{title}\nECE={ece:.4f}, MCE={mce:.4f}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {"ece": ece, "mce": mce, "num_bins": num_bins}


def risk_coverage_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
    out_path: str,
    title: str,
) -> Dict[str, Any]:
    order = np.argsort(-confidences)
    correct_sorted = correctness[order]

    n = len(confidences)
    coverages = np.linspace(0.05, 1.0, 20)
    risks = []
    sizes = []
    for c in coverages:
        m = max(1, int(round(c * n)))
        acc = float(np.mean(correct_sorted[:m]))
        risk = 1.0 - acc
        risks.append(risk)
        sizes.append(m)

    plt.figure(figsize=(6.8, 5.0))
    plt.plot(np.array(coverages), np.array(risks), "o-", linewidth=2, markersize=5)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Coverage (fraction of predictions kept)")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {"coverage": coverages.tolist(), "risk": risks, "sizes": sizes}


def save_histogram(data: np.ndarray, out_path: str, title: str, xlabel: str = "Value") -> None:
    plt.figure(figsize=(6.8, 4.8))
    plt.hist(data, bins=30, density=True, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_classification_metrics(
    samples: List[UQSample],
    allowed_labels: List[str],
    out_dir: str,
    prefix: str,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("No samples to evaluate.")

    label_to_idx = {lab: i for i, lab in enumerate(allowed_labels)}

    y_true = np.array([s.human_label for s in samples], dtype=object)
    y_pred = np.array([s.predicted_label for s in samples], dtype=object)
    conf = np.array([s.confidence for s in samples], dtype=float)
    correct = (y_true == y_pred).astype(int)

    p_mat = np.zeros((len(samples), len(allowed_labels)), dtype=np.float64)
    for i, s in enumerate(samples):
        for j, lab in enumerate(allowed_labels):
            p_mat[i, j] = float(s.p_hat.get(lab, 0.0))

    # NLL and Brier for class-style tasks.
    # Some datasets may contain out-of-vocabulary human labels
    # (e.g. score "0" while allowed labels are "1".."5").
    # Exclude those rows from class-indexed metrics so evaluation does not crash.
    eps = 1e-12
    valid_true_mask = np.array([x in label_to_idx for x in y_true], dtype=bool)
    n_unknown_true = int((~valid_true_mask).sum())

    if valid_true_mask.any():
        true_idx = np.array([label_to_idx[x] for x in y_true[valid_true_mask]], dtype=int)
        p_mat_valid = p_mat[valid_true_mask]
        y_onehot = np.zeros((len(true_idx), len(allowed_labels)), dtype=np.float64)
        y_onehot[np.arange(len(true_idx)), true_idx] = 1.0

        nll = -np.mean(np.log(p_mat_valid[np.arange(len(true_idx)), true_idx] + eps))
        brier = float(np.mean(np.sum((p_mat_valid - y_onehot) ** 2, axis=1)))
    else:
        nll = float("nan")
        brier = float("nan")

    entropy = -np.sum(p_mat * np.log(p_mat + eps), axis=1)
    incorrect = (correct == 0).astype(int)
    auroc = float(roc_auc_score(incorrect, entropy)) if len(np.unique(incorrect)) > 1 else float("nan")

    rel_path = os.path.join(out_dir, f"{prefix}_reliability.png")
    rel = reliability_diagram(
        confidences=conf,
        correctness=correct,
        num_bins=10,
        out_path=rel_path,
        title=prefix,
    )

    rc_path = os.path.join(out_dir, f"{prefix}_risk_coverage.png")
    risk_cov = risk_coverage_curve(
        confidences=conf,
        correctness=correct,
        out_path=rc_path,
        title=f"{prefix} (selective prediction)",
    )

    save_histogram(conf, os.path.join(out_dir, f"{prefix}_confidence_hist.png"), f"{prefix}: confidence", xlabel="max prob")
    save_histogram(entropy, os.path.join(out_dir, f"{prefix}_entropy_hist.png"), f"{prefix}: entropy", xlabel="predictive entropy")

    return {
        "n": int(len(samples)),
        "n_valid_for_class_metrics": int(valid_true_mask.sum()),
        "n_unknown_true_labels": n_unknown_true,
        "accuracy": float(np.mean(correct)),
        "nll": float(nll),
        "brier": float(brier),
        "entropy_mean": float(np.mean(entropy)),
        "auroc_uncertainty_for_incorrect": auroc,
        "ece": float(rel["ece"]),
        "mce": float(rel["mce"]),
        "risk_coverage": risk_cov,
        "paths": {
            "reliability": rel_path,
            "risk_coverage": rc_path,
        },
    }


def compute_class_conformal_results(
    cal_samples: List[UQSample],
    test_samples: List[UQSample],
    allowed_labels: List[str],
    out_dir: str,
    prefix: str,
    alphas: List[float],
) -> Dict[str, Any]:
    if not cal_samples:
        raise ValueError("No calibration samples.")
    if not test_samples:
        raise ValueError("No test samples.")

    cal_scores = np.array(
        [1.0 - float(s.p_hat.get(s.human_label, 0.0)) for s in cal_samples],
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
        for s in test_samples:
            p_min = 1.0 - q
            pred_set = [lab for lab in allowed_labels if float(s.p_hat.get(lab, 0.0)) >= p_min]
            if not pred_set:
                pred_set = [max(allowed_labels, key=lambda lab: float(s.p_hat.get(lab, 0.0)))]
            sizes.append(len(pred_set))
            cover.append(1.0 if s.human_label in pred_set else 0.0)

        cov = float(np.mean(cover)) if cover else float("nan")
        avg_size = float(np.mean(sizes)) if sizes else float("nan")
        empirical_coverages.append(cov)
        avg_set_sizes.append(avg_size)
        coverage_by_alpha[str(alpha)] = cov
        set_size_by_alpha[str(alpha)] = avg_size

    cov_path = os.path.join(out_dir, f"{prefix}_conformal_coverage.png")
    plt.figure(figsize=(6.8, 5.0))
    plt.plot(nominal_coverages, empirical_coverages, "o-", linewidth=2, markersize=5)
    lo = min(nominal_coverages) if nominal_coverages else 0.0
    hi = max(nominal_coverages) if nominal_coverages else 1.0
    plt.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal coverage (1 - alpha)")
    plt.ylabel("Empirical coverage")
    plt.title(f"{prefix} Conformal coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cov_path, dpi=200)
    plt.close()

    size_path = os.path.join(out_dir, f"{prefix}_conformal_set_size.png")
    plt.figure(figsize=(6.8, 5.0))
    plt.plot(alphas, avg_set_sizes, "o-", linewidth=2, markersize=5)
    plt.xlabel("alpha (miscoverage)")
    plt.ylabel("Average set size")
    plt.title(f"{prefix} Conformal efficiency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(size_path, dpi=200)
    plt.close()

    return {
        "cal_n": int(len(cal_samples)),
        "test_n": int(len(test_samples)),
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
# Metrics: batch ranking tasks
# ============================================================

def compute_batch_metrics(
    samples: List[UQSample],
    out_dir: str,
    prefix: str,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("No samples to evaluate.")

    exact = np.array([1.0 if s.exact_correct else 0.0 for s in samples], dtype=float)
    conf = np.array([s.sequence_confidence if s.sequence_confidence is not None else s.confidence for s in samples], dtype=float)
    pair_ag = np.array([s.pairwise_agreement if s.pairwise_agreement is not None else 0.0 for s in samples], dtype=float)
    tau_like = np.array([s.kendall_tau_like if s.kendall_tau_like is not None else 0.0 for s in samples], dtype=float)
    seq_lp = np.array([s.sequence_logprob if s.sequence_logprob is not None else float("nan") for s in samples], dtype=float)

    incorrect = (exact == 0).astype(int)
    auroc = float(roc_auc_score(incorrect, -conf)) if len(np.unique(incorrect)) > 1 else float("nan")

    rel_path = os.path.join(out_dir, f"{prefix}_reliability.png")
    rel = reliability_diagram(
        confidences=conf,
        correctness=exact.astype(int),
        num_bins=10,
        out_path=rel_path,
        title=prefix,
    )

    rc_path = os.path.join(out_dir, f"{prefix}_risk_coverage.png")
    risk_cov = risk_coverage_curve(
        confidences=conf,
        correctness=exact.astype(int),
        out_path=rc_path,
        title=f"{prefix} (selective prediction)",
    )

    save_histogram(conf, os.path.join(out_dir, f"{prefix}_confidence_hist.png"), f"{prefix}: sequence confidence", xlabel="sequence confidence")
    if np.any(~np.isnan(seq_lp)):
        save_histogram(seq_lp[~np.isnan(seq_lp)], os.path.join(out_dir, f"{prefix}_sequence_logprob_hist.png"), f"{prefix}: sequence logprob", xlabel="sequence logprob")

    return {
        "n": int(len(samples)),
        "exact_accuracy": float(np.mean(exact)),
        "pairwise_agreement_mean": float(np.mean(pair_ag)),
        "kendall_tau_like_mean": float(np.mean(tau_like)),
        "sequence_logprob_mean": float(np.nanmean(seq_lp)),
        "sequence_confidence_mean": float(np.mean(conf)),
        "auroc_uncertainty_for_incorrect": auroc,
        "ece": float(rel["ece"]),
        "mce": float(rel["mce"]),
        "risk_coverage": risk_cov,
        "paths": {
            "reliability": rel_path,
            "risk_coverage": rc_path,
        },
    }


def compute_batch_selection_results(
    cal_samples: List[UQSample],
    test_samples: List[UQSample],
    out_dir: str,
    prefix: str,
    alphas: List[float],
) -> Dict[str, Any]:
    """
    Confidence-threshold selection for ranking tasks.
    This is a selective-prediction style calibration, not class-set conformal.
    """
    if not cal_samples:
        raise ValueError("No calibration samples.")
    if not test_samples:
        raise ValueError("No test samples.")

    cal_conf = np.array(
        [s.sequence_confidence if s.sequence_confidence is not None else s.confidence for s in cal_samples],
        dtype=float,
    )
    test_conf = np.array(
        [s.sequence_confidence if s.sequence_confidence is not None else s.confidence for s in test_samples],
        dtype=float,
    )
    test_correct = np.array([1.0 if s.exact_correct else 0.0 for s in test_samples], dtype=float)

    nominal_coverages = [1.0 - a for a in alphas]
    empirical_coverages = []
    avg_kept_acc = []
    thresholds = {}

    for alpha in alphas:
        # Keep the top (1-alpha) fraction by confidence
        thr = float(np.quantile(cal_conf, alpha))
        thresholds[str(alpha)] = thr

        kept = test_conf >= thr
        cov = float(np.mean(kept)) if len(kept) else float("nan")
        empirical_coverages.append(cov)
        avg_kept_acc.append(float(np.mean(test_correct[kept])) if np.any(kept) else float("nan"))

    cov_path = os.path.join(out_dir, f"{prefix}_selection_coverage.png")
    plt.figure(figsize=(6.8, 5.0))
    plt.plot(nominal_coverages, empirical_coverages, "o-", linewidth=2, markersize=5)
    lo = min(nominal_coverages) if nominal_coverages else 0.0
    hi = max(nominal_coverages) if nominal_coverages else 1.0
    plt.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal coverage (1 - alpha)")
    plt.ylabel("Empirical coverage")
    plt.title(f"{prefix} Confidence-threshold coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cov_path, dpi=200)
    plt.close()

    return {
        "cal_n": int(len(cal_samples)),
        "test_n": int(len(test_samples)),
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
# Inference
# ============================================================

def max_tokens_for_task(task_type: str, n_candidates: int) -> int:
    if task_type == "score":
        return 4
    if task_type == "pair":
        return 4
    if task_type == "batch":
        return max(16, 4 * max(1, n_candidates))
    return 8


def task_allowed_labels(task_type: str, n_candidates: int = 0) -> List[str]:
    if task_type == "score":
        return SCORE_LABEL_STRS
    if task_type == "pair":
        return PAIR_LABELS
    if task_type == "batch":
        return letters_for_n(n_candidates)
    raise ValueError(f"Unknown task type: {task_type}")


async def query_one_async(
    item: Dict[str, Any],
    args: argparse.Namespace,
    client: AsyncOpenAI,
) -> UQSample:
    task_type = item["task_type"]
    image_path = os.path.join(args.image_root, item["image_path"])
    prompt_text = item["prompt_text"]
    data_url = await asyncio.to_thread(encode_image_to_data_url, image_path)

    allowed_labels = item["candidate_letters"] if task_type != "batch" else item["candidate_letters"]
    max_tokens = item["max_tokens"]

    last_err: Optional[Exception] = None
    for attempt in range(args.retry_attempts):
        try:
            resp = await client.chat.completions.create(
                model=args.vllm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                logprobs=True,
                top_logprobs=args.top_logprobs,
            )

            choices = getattr(resp, "choices", None)
            if not choices:
                payload = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
                raise RuntimeError(f"Empty/invalid completion response from vLLM: {payload}")

            choice = choices[0]
            break
        except Exception as e:
            last_err = e
            await asyncio.sleep(args.retry_backoff_s * (2**attempt))
    else:
        return UQSample(
            task_type=task_type,
            score_id=item.get("score_id"),
            id=item.get("id"),
            name=item.get("name"),
            original_dataset=item.get("original_dataset"),
            image_path=item["image_path"],
            instruction=item["instruction"],
            human_label=item["human_label"],
            predicted_label="",
            candidate_names=item.get("candidate_names", []),
            candidate_letters=item.get("candidate_letters", []),
            confidence=0.0,
            p_hat={},
            exact_correct=False,
            prediction_valid=False,
            error=f"vLLM request failed after {args.retry_attempts} attempts: {last_err}",
        )

    message = getattr(choice, "message", None)
    raw_output = getattr(message, "content", None) if message is not None else None
    structured = parse_openai_chat_logprobs(choice)

    if task_type in {"score", "pair"}:
        pred = extract_class_label(raw_output, allowed_labels)
        prediction_valid = pred is not None

        p_hat = p_hat_from_top_logprobs(structured, allowed_labels, top_k=args.top_logprobs)
        if pred is None:
            pred = max(allowed_labels, key=lambda lab: float(p_hat.get(lab, 0.0)))
            prediction_valid = False

        confidence = float(p_hat.get(pred, 1.0 / max(1, len(allowed_labels))))
        exact_correct = pred == item["human_label"]
        predicted_top_name = map_label_to_name(pred, item)
        human_top_name = map_label_to_name(item["human_label"], item)

        pred_int = int(pred) if task_type == "score" and pred.isdigit() else None

        return UQSample(
            task_type=task_type,
            score_id=item.get("score_id"),
            id=item.get("id"),
            name=item.get("name"),
            original_dataset=item.get("original_dataset"),
            image_path=item["image_path"],
            instruction=item["instruction"],
            human_label=item["human_label"],
            predicted_label=pred,
            predicted_label_int=pred_int,
            candidate_names=item.get("candidate_names", []),
            candidate_letters=item.get("candidate_letters", []),
            predicted_top_name=predicted_top_name,
            human_top_name=human_top_name,
            confidence=confidence,
            p_hat={lab: float(p_hat.get(lab, 0.0)) for lab in allowed_labels},
            sequence_logprob=None,
            sequence_confidence=None,
            exact_correct=bool(exact_correct),
            pairwise_agreement=1.0 if exact_correct else 0.0,
            kendall_tau_like=1.0 if exact_correct else -1.0,
            prediction_valid=prediction_valid,
            raw_output=raw_output,
            structured_logprob=structured,
            error=None,
        )

    if task_type == "batch":
        pred_rank, seq_lp, seq_conf, prediction_valid = parse_sequence_logprob(choice, allowed_labels)
        human_rank = item["human_label"]
        exact_correct = pred_rank == human_rank
        pa = pairwise_agreement(pred_rank, human_rank)
        kt = kendall_tau_like(pred_rank, human_rank)

        top_letter = pred_rank[0] if pred_rank else None
        human_top_letter = human_rank[0] if human_rank else None

        predicted_top_name = map_label_to_name(top_letter, item) if top_letter else None
        human_top_name = map_label_to_name(human_top_letter, item) if human_top_letter else None

        return UQSample(
            task_type=task_type,
            score_id=item.get("score_id"),
            id=item.get("id"),
            name=item.get("name"),
            original_dataset=item.get("original_dataset"),
            image_path=item["image_path"],
            instruction=item["instruction"],
            human_label=human_rank,
            predicted_label=pred_rank,
            candidate_names=item.get("candidate_names", []),
            candidate_letters=item.get("candidate_letters", []),
            predicted_top_name=predicted_top_name,
            human_top_name=human_top_name,
            confidence=float(seq_conf if seq_conf is not None else 0.0),
            p_hat={},
            sequence_logprob=seq_lp,
            sequence_confidence=seq_conf,
            exact_correct=bool(exact_correct),
            pairwise_agreement=float(pa),
            kendall_tau_like=float(kt),
            prediction_valid=prediction_valid,
            raw_output=raw_output,
            structured_logprob=structured,
            error=None,
        )

    raise ValueError(f"Unknown task type: {task_type}")


def map_label_to_name(label: Optional[str], item: Dict[str, Any]) -> Optional[str]:
    if label is None:
        return None

    task_type = item["task_type"]
    candidate_names = item.get("candidate_names", [])
    candidate_letters = item.get("candidate_letters", [])

    if task_type == "score":
        return item.get("name")

    if task_type == "pair":
        lab = str(label).strip().upper()
        if lab == "A" and len(candidate_names) > 0:
            return candidate_names[0]
        if lab == "B" and len(candidate_names) > 1:
            return candidate_names[1]
        return None

    if task_type == "batch":
        lab = str(label).strip().upper()
        if not lab:
            return None
        top = lab[0]
        if top in candidate_letters:
            idx = candidate_letters.index(top)
            if idx < len(candidate_names):
                return candidate_names[idx]
        return None

    return None


async def run_split_async(
    split_items: List[Dict[str, Any]],
    out_path: str,
    desc: str,
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> List[UQSample]:
    """
    Run one split in batches and checkpoint the full accumulated result list to JSON
    after every batch. The file is rewritten atomically after each batch.
    """
    results: List[UQSample] = []
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(it: Dict[str, Any]) -> UQSample:
        async with sem:
            return await query_one_async(it, args=args, client=client)

    for batch_idx, batch in enumerate(chunked(split_items, args.batch_size), start=1):
        tasks = [asyncio.create_task(worker(it)) for it in batch]

        for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"{desc} batch {batch_idx}",
            leave=False,
        ):
            s = await fut
            results.append(s)

        save_json_atomic(
            out_path,
            {
                "desc": desc,
                "batch_idx": batch_idx,
                "total_done": len(results),
                "total_items": len(split_items),
                "results": [r.model_dump() for r in results],
            },
        )

        print(f"[{desc}] saved {len(results)}/{len(split_items)} results to {out_path}")

    return results


# ============================================================
# Task-group execution
# ============================================================

def prepare_items_for_task(
    raw_items: List[Dict[str, Any]],
    task_type: str,
) -> List[Dict[str, Any]]:
    norm_items = []
    for item in raw_items:
        if detect_task_type(item, forced_task_type=task_type) != task_type:
            continue

        norm = normalize_item(item, forced_task_type=task_type)

        if task_type == "score":
            prompt_text = build_score_prompt(norm["instruction"], norm["answer_text"])
            norm["prompt_text"] = prompt_text
            norm["max_tokens"] = max_tokens_for_task("score", 0)

        elif task_type == "pair":
            prompt_text = build_pair_prompt(
                instruction=norm["instruction"],
                answer1_name=norm.get("answer1_name") or "Assistant A",
                answer1=norm.get("answer1_text", ""),
                answer2_name=norm.get("answer2_name") or "Assistant B",
                answer2=norm.get("answer2_text", ""),
            )
            norm["prompt_text"] = prompt_text
            norm["max_tokens"] = max_tokens_for_task("pair", 0)

        elif task_type == "batch":
            prompt_text = build_batch_prompt(
                instruction=norm["instruction"],
                candidate_names=norm.get("candidate_names", []),
                candidate_answers=norm.get("candidate_answers", []),
                candidate_letters=norm.get("candidate_letters", []),
            )
            norm["prompt_text"] = prompt_text
            norm["max_tokens"] = max_tokens_for_task("batch", len(norm.get("candidate_letters", [])))

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        norm_items.append(norm)

    return norm_items


def task_split_stratify_fn(task_type: str):
    if task_type == "score":
        return lambda x: x["human_label"]
    if task_type == "pair":
        return lambda x: x["human_label"]
    if task_type == "batch":
        # Stratify by top-ranked assistant if present; otherwise random split downstream.
        return lambda x: x["human_label"][0] if x.get("human_label") else "__missing__"
    return None


def run_task_group(
    task_type: str,
    raw_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    run_root_dir: str,
) -> Dict[str, Any]:
    task_dir = os.path.join(run_root_dir, task_type)
    os.makedirs(task_dir, exist_ok=True)

    norm_items = prepare_items_for_task(raw_items, task_type)
    if not norm_items:
        return {
            "task_type": task_type,
            "n_total": 0,
            "n_cal": 0,
            "n_test": 0,
            "note": "No items for this task type.",
        }

    # Split strategy: class tasks are stratified; batch is also stratified by top label when possible.
    stratify_fn = task_split_stratify_fn(task_type)
    cal_items, test_items = stratified_split(
        norm_items,
        frac_cal=args.calibration_fraction,
        seed=args.seed,
        stratify=True,
        stratify_key_fn=stratify_fn,
    )

    if args.deterministic_mode:
        cal_items = sorted(cal_items, key=lambda x: int(x.get("score_id", x.get("id", 0)) or 0))
        test_items = sorted(test_items, key=lambda x: int(x.get("score_id", x.get("id", 0)) or 0))

    with open(os.path.join(task_dir, "split.json"), "w") as f:
        json.dump(
            {
                "task_type": task_type,
                "cal_n": len(cal_items),
                "test_n": len(test_items),
                "cal_ids": [it.get("score_id", it.get("id")) for it in cal_items],
                "test_ids": [it.get("score_id", it.get("id")) for it in test_items],
            },
            f,
            indent=2,
        )

    cal_out_path = os.path.join(task_dir, "cal_predictions.json")
    test_out_path = os.path.join(task_dir, "test_predictions.json")

    async def run_all_async() -> Tuple[List[UQSample], List[UQSample]]:
        client = AsyncOpenAI(
            base_url=args.api_base_url,
            api_key=args.api_key or "EMPTY",
            timeout=args.timeout_s,
            max_retries=0,
        )

        cal_results_ = await run_split_async(
            cal_items,
            cal_out_path,
            f"{task_type}:cal",
            client=client,
            args=args,
        )
        test_results_ = await run_split_async(
            test_items,
            test_out_path,
            f"{task_type}:test",
            client=client,
            args=args,
        )
        return cal_results_, test_results_

    cal_results, test_results = asyncio.run(run_all_async())

    cal_ok = [s for s in cal_results if s.error is None]
    test_ok = [s for s in test_results if s.error is None]

    if not cal_ok:
        raise RuntimeError(f"No successful calibration predictions for task `{task_type}`.")
    if not test_ok:
        raise RuntimeError(f"No successful test predictions for task `{task_type}`.")

    # Common summary
    task_summary: Dict[str, Any] = {
        "task_type": task_type,
        "n_total": len(norm_items),
        "successful_cal_n": len(cal_ok),
        "successful_test_n": len(test_ok),
        "failed_cal_n": len(cal_results) - len(cal_ok),
        "failed_test_n": len(test_results) - len(test_ok),
        "bias_summary": compute_bias_summary(test_ok),
    }

    # Task-specific metrics
    if task_type in {"score", "pair"}:
        allowed_labels = SCORE_LABEL_STRS if task_type == "score" else PAIR_LABELS

        metrics = compute_classification_metrics(
            samples=test_ok,
            allowed_labels=allowed_labels,
            out_dir=task_dir,
            prefix=task_type,
        )

        conformal = compute_class_conformal_results(
            cal_samples=cal_ok,
            test_samples=test_ok,
            allowed_labels=allowed_labels,
            out_dir=task_dir,
            prefix=task_type,
            alphas=args.alphas,
        )

        selection = compute_batch_selection_results(
            cal_samples=cal_ok,
            test_samples=test_ok,
            out_dir=task_dir,
            prefix=task_type,
            alphas=args.alphas,
        )

        task_summary.update(
            {
                "uq_metrics": metrics,
                "conformal": conformal,
                "selection": selection,
                "paths": {
                    "cal_predictions": cal_out_path,
                    "test_predictions": test_out_path,
                },
            }
        )

    elif task_type == "batch":
        metrics = compute_batch_metrics(
            samples=test_ok,
            out_dir=task_dir,
            prefix=task_type,
        )

        selection = compute_batch_selection_results(
            cal_samples=cal_ok,
            test_samples=test_ok,
            out_dir=task_dir,
            prefix=task_type,
            alphas=args.alphas,
        )

        task_summary.update(
            {
                "uq_metrics": metrics,
                "conformal": None,
                "selection": selection,
                "paths": {
                    "cal_predictions": cal_out_path,
                    "test_predictions": test_out_path,
                },
            }
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        json.dump(task_summary, f, indent=2)

    return task_summary


# ============================================================
# Main
# ============================================================

def group_raw_items_by_task(raw_items: List[Dict[str, Any]], task_type: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in raw_items:
        t = detect_task_type(item, forced_task_type=task_type)
        groups[t].append(item)
    return groups


def run_uq(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_root_dir, exist_ok=True)

    with open(args.dataset_path, "r") as f:
        raw_items = [json.loads(line) for line in f]

    if args.max_samples is not None:
        raw_items = raw_items[: args.max_samples]

    groups = group_raw_items_by_task(raw_items, args.task_type)

    task_order = ["score", "pair", "batch"]
    task_summaries: Dict[str, Any] = {}

    for task_type in task_order:
        if task_type not in groups:
            continue
        print(f"\n=== Running task group: {task_type} ({len(groups[task_type])} items) ===")
        task_summary = run_task_group(
            task_type=task_type,
            raw_items=groups[task_type],
            args=args,
            run_root_dir=run_root_dir,
        )
        task_summaries[task_type] = task_summary

    root_summary = {
        "run_id": run_id,
        "dataset_path": args.dataset_path,
        "image_root": args.image_root,
        "model": args.vllm_model,
        "api_base_url": args.api_base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_logprobs": args.top_logprobs,
        "timeout_s": args.timeout_s,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "retry_attempts": args.retry_attempts,
        "retry_backoff_s": args.retry_backoff_s,
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
        "deterministic_mode": args.deterministic_mode,
        "task_type": args.task_type,
        "alphas": args.alphas,
        "task_summaries": task_summaries,
    }

    with open(os.path.join(run_root_dir, "summary.json"), "w") as f:
        json.dump(root_summary, f, indent=2)

    print(f"\nFinished. Wrote results to: {run_root_dir}")
    print(f"Root summary: {os.path.join(run_root_dir, 'summary.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UQ for score/pair/batch evaluation tasks with vLLM OpenAI API.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to JSONL dataset.")
    parser.add_argument("--image_root", type=str, default="Dataset/image")
    parser.add_argument("--vllm_model", type=str, required=True, help="Model name as served by vLLM.")
    parser.add_argument("--api_base_url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api_key", type=str, default=None, help="Optional API key (vLLM may ignore).")
    parser.add_argument("--out_dir", type=str, default="Figures/UQ")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--timeout_s", type=int, default=180)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--retry_attempts", type=int, default=3)
    parser.add_argument("--retry_backoff_s", type=float, default=1.0)

    parser.add_argument("--calibration_fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic_mode", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument(
        "--alphas",
        type=str,
        default="0.01,0.05,0.1,0.15,0.2",
        help="Comma-separated miscoverage levels.",
    )

    parser.add_argument(
        "--task_type",
        type=str,
        default="auto",
        choices=["auto", "score", "pair", "batch"],
        help="Force one task type or auto-detect per item.",
    )

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    args.alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    assert 0.0 < args.calibration_fraction < 1.0
    assert all(0.0 < a < 1.0 for a in args.alphas)
    assert args.batch_size >= 1
    assert args.concurrency >= 1
    assert args.retry_attempts >= 1
    assert args.retry_backoff_s >= 0.0

    run_uq(args)


if __name__ == "__main__":
    main()