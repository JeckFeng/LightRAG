from __future__ import annotations

import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from Ablation.hotpotqa_minimal_closed_loop.common import read_json, read_jsonl, write_json, write_jsonl


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punctuation(value: str) -> str:
        return "".join(char for char in value if char not in string.punctuation)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(text))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def score_predictions(
    prediction_payload: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    *,
    subset_id: str,
    mode_id: str,
    predictions_path: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answer_predictions = prediction_payload.get("answer", {})

    detail_rows: list[dict[str, Any]] = []
    total_em = 0.0
    total_f1 = 0.0

    for row in manifest_rows:
        case_id = str(row["case_id"])
        gold_answer = str(row["answer"])
        pred_answer = str(answer_predictions.get(case_id, ""))
        em = exact_match_score(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        total_em += em
        total_f1 += f1
        detail_rows.append(
            {
                "case_id": case_id,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "em": em,
                "f1": f1,
            }
        )

    count = len(detail_rows)
    metrics = {
        "subset": subset_id,
        "mode": mode_id,
        "count": count,
        "em": total_em / count if count else 0.0,
        "f1": total_f1 / count if count else 0.0,
        "missing_prediction_count": sum(
            1 for row in detail_rows if row["pred_answer"] == ""
        ),
        "predictions_path": str(predictions_path) if predictions_path else "",
        "manifest_path": str(manifest_path) if manifest_path else "",
    }
    return metrics, detail_rows


def write_scored_outputs(
    metrics: dict[str, Any],
    detail_rows: list[dict[str, Any]],
    *,
    output: Path,
    details_output: Path | None = None,
) -> None:
    write_json(output, metrics)
    if details_output:
        write_jsonl(details_output, detail_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score HotpotQA answer predictions with answer-level EM/F1."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Predictions JSON in HotpotQA-compatible answer format.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Bridge subset manifest JSONL with gold answers.",
    )
    parser.add_argument("--subset-id", type=str, required=True, help="Subset identifier.")
    parser.add_argument("--mode-id", type=str, required=True, help="Mode identifier.")
    parser.add_argument("--output", type=Path, required=True, help="Metrics summary JSON output.")
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional JSONL output with per-case EM/F1 details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_payload = read_json(args.predictions)
    manifest_rows = read_jsonl(args.manifest)
    metrics, detail_rows = score_predictions(
        prediction_payload,
        manifest_rows,
        subset_id=args.subset_id,
        mode_id=args.mode_id,
        predictions_path=args.predictions,
        manifest_path=args.manifest,
    )
    write_scored_outputs(
        metrics,
        detail_rows,
        output=args.output,
        details_output=args.details_output,
    )

    print(
        f"[eval_hotpot_em_f1] subset={args.subset_id} mode={args.mode_id} "
        f"count={metrics['count']} em={metrics['em']:.4f} f1={metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
