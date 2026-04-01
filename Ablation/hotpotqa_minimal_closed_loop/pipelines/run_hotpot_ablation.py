from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from Ablation.hotpotqa_minimal_closed_loop.common import (
    EXPERIMENT_ROOT,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from Ablation.hotpotqa_minimal_closed_loop.evaluation.eval_hotpot_em_f1 import (
    score_predictions,
    write_scored_outputs,
)
from Ablation.hotpotqa_minimal_closed_loop.evaluation.export_hotpot_predictions import (
    export_predictions,
)
from Ablation.hotpotqa_minimal_closed_loop.preprocessing.build_case_corpus import (
    build_case_corpora,
)
from Ablation.hotpotqa_minimal_closed_loop.preprocessing.select_hotpot_bridge_subset import (
    export_bridge_subset,
)


DEFAULT_RUNTIME_PROVIDER = (
    "Ablation.hotpotqa_minimal_closed_loop.runtime:build_mock_runtime_dependencies"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute HotpotQA bridge-only ablation runs."
    )
    parser.add_argument(
        "--subset-config",
        type=Path,
        required=True,
        help="Subset config JSON path.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        nargs="+",
        required=True,
        help="One or more experiment config JSON paths.",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=EXPERIMENT_ROOT / "logs" / "run_plan.json",
        help="Output JSON path for the assembled run plan.",
    )
    parser.add_argument(
        "--jobs-output",
        type=Path,
        default=EXPERIMENT_ROOT / "logs" / "run_jobs.jsonl",
        help="Output JSONL path for expanded per-case jobs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute indexing, querying, prediction export, and EM/F1 scoring.",
    )
    parser.add_argument(
        "--runtime-provider",
        type=str,
        default=DEFAULT_RUNTIME_PROVIDER,
        help="Runtime provider in module.path:callable format.",
    )
    parser.add_argument(
        "--provider-config",
        type=Path,
        help="Optional JSON config passed to the runtime provider.",
    )
    parser.add_argument(
        "--prepared-corpus-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results" / "_prepared_case_corpora",
        help="Directory for per-case paragraph corpora.",
    )
    parser.add_argument(
        "--clean-workspaces",
        action="store_true",
        help="Rebuild case workspaces instead of reusing existing directories.",
    )
    return parser.parse_args()


def build_jobs(
    subset_config: dict,
    manifest_rows: list[dict],
    experiment_configs: list[dict],
    prepared_corpus_dir: Path,
) -> list[dict]:
    jobs: list[dict] = []
    for case_record in manifest_rows:
        case_id = str(case_record["case_id"])
        for experiment_config in experiment_configs:
            mode_id = str(experiment_config["mode_id"])
            result_root = (
                EXPERIMENT_ROOT / "results" / subset_config["subset_id"] / mode_id
            )
            workspace = result_root / "workspaces" / case_id
            jobs.append(
                {
                    "subset_id": subset_config["subset_id"],
                    "case_id": case_id,
                    "mode_id": mode_id,
                    "question": case_record["question"],
                    "gold_answer": case_record["answer"],
                    "workspace": str(workspace),
                    "case_documents": str(
                        prepared_corpus_dir / case_id / "case_documents.jsonl"
                    ),
                    "result_json": str(result_root / "case_results" / f"{case_id}.json"),
                    "results_jsonl": str(
                        result_root
                        / f"case_results_{subset_config['subset_id']}_{mode_id}.jsonl"
                    ),
                    "predictions_json": str(
                        result_root
                        / f"predictions_{subset_config['subset_id']}_{mode_id}.json"
                    ),
                    "metrics_json": str(
                        result_root
                        / f"metrics_{subset_config['subset_id']}_{mode_id}.json"
                    ),
                    "details_jsonl": str(
                        result_root
                        / f"details_{subset_config['subset_id']}_{mode_id}.jsonl"
                    ),
                }
            )
    return jobs


def ensure_manifest(subset_config: dict) -> tuple[Path, list[dict]]:
    manifest_path = Path(subset_config["output_manifest"])
    if manifest_path.exists():
        return manifest_path, read_jsonl(manifest_path)

    config = {
        "dataset_path": Path(subset_config["dataset_path"]),
        "subset_id": subset_config["subset_id"],
        "limit": subset_config.get("sample_limit"),
        "shuffle": bool(subset_config.get("shuffle", False)),
        "seed": int(subset_config.get("seed", 0)),
        "output_path": manifest_path,
    }
    manifest_rows, _summary = export_bridge_subset(config)
    return manifest_path, manifest_rows


def ensure_case_corpora(
    manifest_rows: list[dict],
    prepared_corpus_dir: Path,
) -> None:
    build_case_corpora(manifest_rows, prepared_corpus_dir)


def write_plan(
    *,
    subset_config: dict,
    manifest_path: Path,
    manifest_rows: list[dict],
    experiment_configs: list[dict],
    jobs: list[dict],
    plan_output: Path,
    jobs_output: Path,
) -> None:
    plan = {
        "subset_id": subset_config["subset_id"],
        "manifest_path": str(manifest_path),
        "case_count": len(manifest_rows),
        "mode_ids": [config["mode_id"] for config in experiment_configs],
        "job_count": len(jobs),
        "notes": (
            "Each case is indexed in an isolated workspace and queried in one of the "
            "three ablation modes defined by Phase 3."
        ),
    }
    write_json(plan_output, plan)
    write_jsonl(jobs_output, jobs)


async def execute_plan(
    *,
    jobs: list[dict],
    experiment_configs: dict[str, dict],
    runtime_provider: str,
    provider_config: Path | None,
    clean_workspaces: bool,
) -> None:
    from Ablation.hotpotqa_minimal_closed_loop.pipelines.run_case_indexing import (
        execute_case_indexing,
    )
    from Ablation.hotpotqa_minimal_closed_loop.pipelines.run_case_query import (
        execute_case_query,
    )

    mode_rows: dict[str, list[dict]] = {mode_id: [] for mode_id in experiment_configs}

    for job in jobs:
        workspace = Path(job["workspace"])
        result_json = Path(job["result_json"])
        case_documents = Path(job["case_documents"])
        mode_config = experiment_configs[str(job["mode_id"])]

        await execute_case_indexing(
            case_documents_path=case_documents,
            workspace=workspace,
            mode_id=str(job["mode_id"]),
            runtime_provider=runtime_provider,
            provider_config=provider_config,
            clean_workspace=clean_workspaces,
        )
        result_payload = await execute_case_query(
            question=str(job["question"]),
            mode_config=mode_config,
            workspace=workspace,
            subset_id=str(job["subset_id"]),
            case_id=str(job["case_id"]),
            gold_answer=str(job["gold_answer"]),
            runtime_provider=runtime_provider,
            provider_config=provider_config,
        )
        write_json(result_json, result_payload)
        mode_rows[str(job["mode_id"])].append(result_payload)

    for mode_id, rows in mode_rows.items():
        if not rows:
            continue
        results_jsonl = Path(
            next(job["results_jsonl"] for job in jobs if job["mode_id"] == mode_id)
        )
        predictions_json = Path(
            next(job["predictions_json"] for job in jobs if job["mode_id"] == mode_id)
        )
        metrics_json = Path(
            next(job["metrics_json"] for job in jobs if job["mode_id"] == mode_id)
        )
        details_jsonl = Path(
            next(job["details_jsonl"] for job in jobs if job["mode_id"] == mode_id)
        )
        write_jsonl(results_jsonl, rows)
        predictions = export_predictions(rows, predictions_json)
        manifest_rows = [
            {
                "case_id": row["case_id"],
                "answer": row["gold_answer"],
            }
            for row in rows
        ]
        metrics, details = score_predictions(
            predictions,
            manifest_rows,
            subset_id=str(rows[0]["subset_id"]),
            mode_id=mode_id,
            predictions_path=predictions_json,
            manifest_path=results_jsonl,
        )
        write_scored_outputs(
            metrics,
            details,
            output=metrics_json,
            details_output=details_jsonl,
        )


def main() -> None:
    args = parse_args()
    subset_config = read_json(args.subset_config)
    manifest_path, manifest_rows = ensure_manifest(subset_config)
    ensure_case_corpora(manifest_rows, args.prepared_corpus_dir)
    experiment_configs = [read_json(path) for path in args.experiment_config]
    jobs = build_jobs(
        subset_config,
        manifest_rows,
        experiment_configs,
        args.prepared_corpus_dir,
    )

    write_plan(
        subset_config=subset_config,
        manifest_path=manifest_path,
        manifest_rows=manifest_rows,
        experiment_configs=experiment_configs,
        jobs=jobs,
        plan_output=args.plan_output,
        jobs_output=args.jobs_output,
    )

    if args.execute:
        asyncio.run(
            execute_plan(
                jobs=jobs,
                experiment_configs={
                    str(config["mode_id"]): config for config in experiment_configs
                },
                runtime_provider=args.runtime_provider,
                provider_config=args.provider_config,
                clean_workspaces=args.clean_workspaces,
            )
        )
        print(
            f"[run_hotpot_ablation] executed subset={subset_config['subset_id']} "
            f"cases={len(manifest_rows)} jobs={len(jobs)}"
        )
        return

    print(
        f"[run_hotpot_ablation] subset={subset_config['subset_id']} "
        f"cases={len(manifest_rows)} jobs={len(jobs)}"
    )


if __name__ == "__main__":
    main()
