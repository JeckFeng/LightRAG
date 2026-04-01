from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from Ablation.hotpotqa_minimal_closed_loop.common import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ablation result rows into HotpotQA answer prediction format."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Result JSONL path containing one row per case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output predictions JSON path.",
    )
    return parser.parse_args()


def extract_answer(row: dict[str, Any]) -> str:
    if "pred_answer" in row:
        return str(row["pred_answer"])
    if "prediction" in row and isinstance(row["prediction"], dict):
        prediction = row["prediction"]
        if "answer" in prediction:
            return str(prediction["answer"])
    raise KeyError(f"Result row missing pred_answer/prediction.answer: {row}")


def build_predictions_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "answer": {str(row["case_id"]): extract_answer(row) for row in rows},
        "sp": {},
    }


def export_predictions(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    predictions = build_predictions_payload(rows)
    write_json(output, predictions)
    return predictions


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.results)
    predictions = export_predictions(rows, args.output)
    print(
        f"[export_hotpot_predictions] cases={len(predictions['answer'])} output={args.output}"
    )


if __name__ == "__main__":
    main()
