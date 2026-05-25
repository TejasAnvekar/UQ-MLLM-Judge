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

from prompt import get_score_digit_prompt

LABELS = [1, 2, 3, 4, 5]
LABEL_STRS = {str(k) for k in LABELS}


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
    score_id: Optional[int] = None
    id: Optional[int] = None
    name: Optional[str] = None
    original_dataset: Optional[str] = None

    image_path: str
    instruction: str
    answer: str
    human: int

    predicted_label: int
    confidence: float
    p_hat: Dict[str, float]

    raw_digit_output: Optional[str] = None
    structured_logprob: Optional[StructuredLogprob] = None
    digit_valid: Optional[bool] = None
    error: Optional[str] = None


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


def build_score_prompt(instruction: str, answer: str) -> str:
    return get_score_digit_prompt(instruction=instruction, answer=answer)


def extract_digit(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"([1-5])", text)
    if not m:
        return None
    return int(m.group(1))


ScoreDigitAdapter = TypeAdapter[int]


def validate_score_digit(d: Optional[int]) -> int:
    if d is None:
        raise ValueError("No digit extracted from model output.")
    ScoreDigitAdapter.validate_python(d)
    if d not in LABELS:
        raise ValueError(f"Invalid digit extracted: {d}")
    return d


def parse_openai_chat_logprobs(choice: Any) -> StructuredLogprob:
    structured = StructuredLogprob(raw={})
    message = getattr(choice, "message", None)
    structured.generated_text = getattr(message, "content", None) if message is not None else None

    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return structured

    content = getattr(lp, "content", None) or []
    if isinstance(content, list) and content:
        first_digit_item = None
        for item in content:
            tok = normalize_token(str(getattr(item, "token", "")))
            if tok in LABEL_STRS:
                first_digit_item = item
                break
        if first_digit_item is None:
            first_digit_item = content[0]

        token = getattr(first_digit_item, "token", "")
        logprob = getattr(first_digit_item, "logprob", float("nan"))
        top_lp = getattr(first_digit_item, "top_logprobs", None) or []
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


def p_hat_from_top_logprobs(structured: Optional[StructuredLogprob], top_k: int = 20) -> Dict[str, float]:
    eps = 1e-30
    if not structured or not structured.first_token:
        return {str(k): 1.0 / 5.0 for k in LABELS}

    candidates: Dict[str, float] = {}
    for tlp in structured.first_token.top_logprobs[:top_k]:
        norm_tok = normalize_token(tlp.token)
        if norm_tok in LABEL_STRS:
            candidates[norm_tok] = float(tlp.logprob)

    if not candidates:
        return {str(k): 1.0 / 5.0 for k in LABELS}

    probs = {k: math.exp(v) for k, v in candidates.items()}
    denom = sum(probs.values()) + eps
    p = {k: probs[k] / denom for k in probs}

    for k in LABELS:
        p.setdefault(str(k), 0.0)

    denom2 = sum(p.values()) + eps
    for k in LABELS:
        p[str(k)] = p[str(k)] / denom2

    return p


def mean_confidence_interval(data: np.ndarray) -> Tuple[float, float]:
    if len(data) == 0:
        return float("nan"), float("nan")
    m = float(np.mean(data))
    s = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    se = s / math.sqrt(len(data)) if len(data) > 0 else float("nan")
    return m, 1.96 * se


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


def conformal_threshold_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(1, k), n)
    idx = k - 1
    return float(np.partition(scores, idx)[idx])


def conformal_prediction_sets(p_hat: Dict[str, float], q: float) -> List[int]:
    p_min = 1.0 - q
    preds = [k for k in LABELS if p_hat.get(str(k), 0.0) >= p_min]
    if not preds:
        preds = [int(max(LABELS, key=lambda kk: p_hat.get(str(kk), 0.0)))]
    return preds


def compute_uq_metrics(samples: List[UQSample], num_bins: int, out_dir: str, prefix: str) -> Dict[str, Any]:
    if not samples:
        raise ValueError("No samples to evaluate.")

    y_true = np.array([s.human for s in samples], dtype=int)
    conf = np.array([s.confidence for s in samples], dtype=float)
    y_pred = np.array([s.predicted_label for s in samples], dtype=int)
    correct = (y_pred == y_true).astype(int)

    p_mat = np.zeros((len(samples), 5), dtype=np.float64)
    for i, s in enumerate(samples):
        for j, k in enumerate(LABELS):
            p_mat[i, j] = float(s.p_hat.get(str(k), 0.0))

    eps = 1e-12
    y_onehot = np.zeros((len(samples), 5), dtype=np.float64)
    for i, yt in enumerate(y_true):
        y_onehot[i, yt - 1] = 1.0

    nll = -np.mean(np.log(p_mat[np.arange(len(samples)), y_true - 1] + eps))
    brier = float(np.mean(np.sum((p_mat - y_onehot) ** 2, axis=1)))

    entropy = -np.sum(p_mat * np.log(p_mat + eps), axis=1)
    incorrect = (correct == 0).astype(int)
    auroc = float(roc_auc_score(incorrect, entropy)) if len(np.unique(incorrect)) > 1 else float("nan")

    rel_path = os.path.join(out_dir, f"{prefix}_reliability.png")
    rel = reliability_diagram(
        confidences=conf,
        correctness=correct,
        num_bins=num_bins,
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


def compute_conformal_results(
    cal_samples: List[UQSample],
    test_samples: List[UQSample],
    alphas: List[float],
    out_dir: str,
    prefix: str,
) -> Dict[str, Any]:
    if not cal_samples:
        raise ValueError("No calibration samples.")
    if not test_samples:
        raise ValueError("No test samples.")

    y_cal = np.array([s.human for s in cal_samples], dtype=int)
    cal_scores = np.array([1.0 - float(s.p_hat[str(int(yt))]) for s, yt in zip(cal_samples, y_cal)], dtype=float)

    nominal_coverages = [1.0 - a for a in alphas]
    coverages = []
    set_sizes = []

    cover_per_alpha = {}
    size_per_alpha = {}

    for alpha in alphas:
        q = conformal_threshold_quantile(cal_scores, alpha)
        cover = []
        sizes = []
        for s in test_samples:
            set_pred = conformal_prediction_sets(s.p_hat, q)
            sizes.append(len(set_pred))
            cover.append(1.0 if int(s.human) in set_pred else 0.0)
        cov = float(np.mean(cover)) if cover else float("nan")
        avg_size = float(np.mean(sizes)) if sizes else float("nan")
        coverages.append(cov)
        set_sizes.append(avg_size)
        cover_per_alpha[str(alpha)] = cov
        size_per_alpha[str(alpha)] = avg_size

    plt.figure(figsize=(6.8, 5.0))
    plt.plot(nominal_coverages, coverages, "o-", linewidth=2, markersize=5)
    plt.plot(
        [min(nominal_coverages), max(nominal_coverages)],
        [min(nominal_coverages), max(nominal_coverages)],
        "--",
        color="black",
        linewidth=1,
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal coverage (1 - alpha)")
    plt.ylabel("Empirical coverage")
    plt.title(f"{prefix} Conformal coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cov_path = os.path.join(out_dir, f"{prefix}_conformal_coverage.png")
    plt.savefig(cov_path, dpi=200)
    plt.close()

    plt.figure(figsize=(6.8, 5.0))
    plt.plot(alphas, set_sizes, "o-", linewidth=2, markersize=5)
    plt.xlabel("alpha (miscoverage)")
    plt.ylabel("Average set size")
    plt.title(f"{prefix} Conformal efficiency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    size_path = os.path.join(out_dir, f"{prefix}_conformal_set_size.png")
    plt.savefig(size_path, dpi=200)
    plt.close()

    return {
        "cal_n": int(len(cal_samples)),
        "test_n": int(len(test_samples)),
        "alphas": alphas,
        "nominal_coverages": nominal_coverages,
        "empirical_coverages": coverages,
        "avg_set_sizes": set_sizes,
        "coverage_by_alpha": cover_per_alpha,
        "set_size_by_alpha": size_per_alpha,
        "paths": {
            "coverage_plot": cov_path,
            "set_size_plot": size_path,
        },
    }


def stratified_split(
    items: List[Dict[str, Any]],
    y_key: str,
    frac_cal: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    groups: Dict[int, List[int]] = {k: [] for k in LABELS}

    for idx, it in enumerate(items):
        y = int(it[y_key])
        if y not in groups:
            groups[y] = []
        groups[y].append(idx)

    cal_idx = []
    test_idx = []
    for _, idxs in groups.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_cal = int(math.floor(len(idxs) * frac_cal))
        cal_idx.extend(idxs[:n_cal])
        test_idx.extend(idxs[n_cal:])

    cal = [items[i] for i in cal_idx]
    test = [items[i] for i in test_idx]
    return cal, test


def chunked(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def save_json_atomic(path: str, obj: Any) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def group_bias_summary(samples: List[UQSample], group_key: str) -> Dict[str, Any]:
    groups: Dict[str, List[UQSample]] = defaultdict(list)
    for s in samples:
        key = getattr(s, group_key, None)
        if key is None:
            key = "__missing__"
        groups[str(key)].append(s)

    summary: Dict[str, Any] = {}
    for key, group_samples in sorted(groups.items(), key=lambda x: x[0]):
        y_true = np.array([s.human for s in group_samples], dtype=int)
        y_pred = np.array([s.predicted_label for s in group_samples], dtype=int)
        conf = np.array([s.confidence for s in group_samples], dtype=float)
        correct = (y_true == y_pred).astype(float)

        summary[key] = {
            "n": int(len(group_samples)),
            "accuracy": float(np.mean(correct)) if len(group_samples) else float("nan"),
            "avg_confidence": float(np.mean(conf)) if len(group_samples) else float("nan"),
            "label_hist": {str(k): int(np.sum(y_pred == k)) for k in LABELS},
            "human_label_hist": {str(k): int(np.sum(y_true == k)) for k in LABELS},
        }

    return summary


async def query_one_async(
    item: Dict[str, Any],
    args: argparse.Namespace,
    client: AsyncOpenAI,
) -> UQSample:
    image_rel = item["image_path"]
    image_path = os.path.join(args.image_root, image_rel)
    prompt_text = build_score_prompt(item["instruction"], item["answer"])
    data_url = await asyncio.to_thread(encode_image_to_data_url, image_path)

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
                max_tokens=4,
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
            score_id=int(item["score_id"]) if "score_id" in item else None,
            id=int(item["id"]) if "id" in item else None,
            name=item.get("name"),
            original_dataset=item.get("original_dataset"),
            image_path=item["image_path"],
            instruction=item["instruction"],
            answer=item["answer"],
            human=int(item["human"]),
            predicted_label=1,
            confidence=0.0,
            p_hat={str(k): 0.2 for k in LABELS},
            raw_digit_output=None,
            structured_logprob=None,
            digit_valid=False,
            error=f"vLLM request failed after {args.retry_attempts} attempts: {last_err}",
        )

    message = getattr(choice, "message", None)
    digit_out = getattr(message, "content", None) if message is not None else None
    pred_digit = extract_digit(digit_out)

    digit_valid = True
    try:
        pred_digit = validate_score_digit(pred_digit)
    except Exception:
        digit_valid = False

    structured = parse_openai_chat_logprobs(choice)
    p_hat = p_hat_from_top_logprobs(structured, top_k=args.top_logprobs)

    predicted_label = int(max(LABELS, key=lambda k: p_hat.get(str(k), 0.0)))
    if pred_digit in LABELS:
        predicted_label = int(pred_digit)

    confidence = float(p_hat.get(str(predicted_label), 1.0 / 5.0))
    human = int(item["human"])

    return UQSample(
        score_id=int(item["score_id"]) if "score_id" in item else None,
        id=int(item["id"]) if "id" in item else None,
        name=item.get("name"),
        original_dataset=item.get("original_dataset"),
        image_path=item["image_path"],
        instruction=item["instruction"],
        answer=item["answer"],
        human=human,
        predicted_label=predicted_label,
        confidence=confidence,
        p_hat={str(k): float(p_hat.get(str(k), 0.0)) for k in LABELS},
        raw_digit_output=digit_out,
        structured_logprob=structured,
        digit_valid=digit_valid,
        error=None,
    )


async def run_split_async(
    split_items: List[Dict[str, Any]],
    out_path: str,
    desc: str,
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> List[UQSample]:
    """
    Run a split in batches and checkpoint the full accumulated result list to JSON
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


def run_uq(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.dataset_path, "r") as f:
        raw_items = [json.loads(line) for line in f]

    if args.max_samples is not None:
        raw_items = raw_items[: args.max_samples]

    cal_items, test_items = stratified_split(
        raw_items,
        y_key="human",
        frac_cal=args.calibration_fraction,
        seed=args.seed,
    )

    if args.deterministic_mode:
        cal_items = sorted(cal_items, key=lambda x: int(x.get("score_id", x.get("id", 0))))
        test_items = sorted(test_items, key=lambda x: int(x.get("score_id", x.get("id", 0))))

    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(
            {
                "cal_n": len(cal_items),
                "test_n": len(test_items),
                "cal_ids": [it.get("score_id", it.get("id")) for it in cal_items],
                "test_ids": [it.get("score_id", it.get("id")) for it in test_items],
            },
            f,
            indent=2,
        )

    cal_out_path = os.path.join(out_dir, "cal_predictions.json")
    test_out_path = os.path.join(out_dir, "test_predictions.json")

    async def run_all_splits_async() -> Tuple[List[UQSample], List[UQSample]]:
        client = AsyncOpenAI(
            base_url=args.api_base_url,
            api_key=args.api_key or "EMPTY",
            timeout=args.timeout_s,
            max_retries=0,
        )

        cal_results_ = await run_split_async(
            cal_items,
            cal_out_path,
            "Cal querying",
            client=client,
            args=args,
        )
        test_results_ = await run_split_async(
            test_items,
            test_out_path,
            "Test querying",
            client=client,
            args=args,
        )
        return cal_results_, test_results_

    cal_results, test_results = asyncio.run(run_all_splits_async())

    cal_results_ok = [s for s in cal_results if s.error is None]
    test_results_ok = [s for s in test_results if s.error is None]

    if not cal_results_ok:
        raise RuntimeError("No successful calibration predictions.")
    if not test_results_ok:
        raise RuntimeError("No successful test predictions.")

    uq_metrics = compute_uq_metrics(
        samples=test_results_ok,
        num_bins=args.num_bins,
        out_dir=out_dir,
        prefix="test",
    )

    conformal = compute_conformal_results(
        cal_samples=cal_results_ok,
        test_samples=test_results_ok,
        alphas=args.alphas,
        out_dir=out_dir,
        prefix="test",
    )

    bias_summary = {
        "by_original_dataset": group_bias_summary(test_results_ok, "original_dataset"),
        "by_name": group_bias_summary(test_results_ok, "name"),
    }

    summary = {
        "run_id": run_id,
        "model": args.vllm_model,
        "api_base_url": args.api_base_url,
        "dataset_path": args.dataset_path,
        "image_root": args.image_root,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_logprobs": args.top_logprobs,
        "num_bins": args.num_bins,
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "successful_cal_n": len(cal_results_ok),
        "successful_test_n": len(test_results_ok),
        "failed_cal_n": len(cal_results) - len(cal_results_ok),
        "failed_test_n": len(test_results) - len(test_results_ok),
        "uq_metrics": uq_metrics,
        "conformal": conformal,
        "bias_summary": bias_summary,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Finished. Wrote results to: {out_dir}")
    print(f"UQ metrics reliability plot: {uq_metrics['paths']['reliability']}")
    print(f"Conformal coverage plot: {conformal['paths']['coverage_plot']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UQ + conformal prediction for MLLM judge with vLLM OpenAI API.")
    parser.add_argument("--dataset_path", type=str, default="Dataset/Benchmark_Lite/score_lite.jsonl")
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
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--retry_attempts", type=int, default=3)
    parser.add_argument("--retry_backoff_s", type=float, default=1.0)

    parser.add_argument("--calibration_fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic_mode", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.01,0.05,0.1,0.15,0.2",
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