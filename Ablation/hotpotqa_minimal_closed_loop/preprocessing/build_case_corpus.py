from __future__ import annotations

import argparse
from pathlib import Path

from Ablation.hotpotqa_minimal_closed_loop.common import (
    EXPERIMENT_ROOT,
    build_paragraph_documents,
    ensure_directory,
    find_case_record,
    read_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paragraph-level case corpora from a HotpotQA case manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a bridge subset manifest JSONL file.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        help="Optional single case_id to build. If omitted, all manifest cases are built.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results" / "_prepared_case_corpora",
        help="Directory where per-case corpora are written.",
    )
    return parser.parse_args()


def write_case_corpus(case_record: dict, output_dir: Path) -> None:
    case_id = str(case_record["case_id"])
    case_dir = ensure_directory(output_dir / case_id)
    documents = [document.to_dict() for document in build_paragraph_documents(case_record)]

    write_jsonl(case_dir / "case_documents.jsonl", documents)
    write_json(
        case_dir / "case_metadata.json",
        {
            "case_id": case_id,
            "question": case_record["question"],
            "answer": case_record["answer"],
            "sample_type": case_record["sample_type"],
            "level": case_record["level"],
            "document_count": len(documents),
            "supporting_fact_count": len(case_record.get("supporting_facts", [])),
        },
    )


def build_case_corpora(
    manifest_rows: list[dict],
    output_dir: Path,
    *,
    case_id: str | None = None,
) -> list[dict]:
    if case_id:
        case_record = find_case_record(manifest_rows, case_id)
        if case_record is None:
            raise ValueError(f"case_id {case_id!r} not found in manifest rows")
        target_rows = [case_record]
    else:
        target_rows = manifest_rows

    for case_record in target_rows:
        write_case_corpus(case_record, output_dir)

    return target_rows


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    target_rows = build_case_corpora(
        manifest_rows,
        args.output_dir,
        case_id=args.case_id,
    )

    print(
        f"[build_case_corpus] built_cases={len(target_rows)} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
