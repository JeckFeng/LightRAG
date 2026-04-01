from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from Ablation.hotpotqa_minimal_closed_loop.common import read_json, write_json


DEFAULT_RUNTIME_PROVIDER = (
    "Ablation.hotpotqa_minimal_closed_loop.runtime:build_mock_runtime_dependencies"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or execute a single-case query request for HotpotQA ablation."
    )
    parser.add_argument("--question", type=str, required=True, help="HotpotQA question text.")
    parser.add_argument(
        "--mode-config",
        type=Path,
        required=True,
        help="Experiment config JSON for the target mode.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace directory prepared for the case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the query request JSON to be written.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the query in the target case workspace.",
    )
    parser.add_argument("--subset-id", type=str, help="Subset identifier.")
    parser.add_argument("--case-id", type=str, help="Case identifier.")
    parser.add_argument(
        "--gold-answer",
        type=str,
        help="Optional gold answer to store in the result payload.",
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
    return parser.parse_args()


def build_query_request(
    *,
    question: str,
    workspace: Path,
    mode_config: dict,
    subset_id: str | None,
    case_id: str | None,
    gold_answer: str | None,
    status: str,
) -> dict[str, object]:
    query_param = {
        "mode": str(mode_config.get("query_mode", "local")),
        "only_need_context": False,
        "only_need_prompt": False,
        "response_type": str(mode_config.get("response_type", "Multiple Paragraphs")),
        "stream": False,
        "top_k": mode_config.get("top_k"),
        "chunk_top_k": mode_config.get("chunk_top_k"),
        "max_entity_tokens": mode_config.get("max_entity_tokens"),
        "max_relation_tokens": mode_config.get("max_relation_tokens"),
        "max_total_tokens": mode_config.get("max_total_tokens"),
        "enable_entity_profiles": bool(
            mode_config.get("enable_entity_profiles", False)
        ),
        "entity_profile_top_k": int(mode_config.get("entity_profile_top_k", 24)),
        "entity_profile_max_per_entity": int(
            mode_config.get("entity_profile_max_per_entity", 2)
        ),
        "hl_keywords": [],
        "ll_keywords": [],
        "conversation_history": [],
        "history_turns": mode_config.get("history_turns", 3),
        "model_func": None,
        "user_prompt": mode_config.get("user_prompt"),
        "enable_rerank": bool(mode_config.get("enable_rerank", True)),
        "include_references": False,
    }
    return {
        "subset_id": subset_id,
        "case_id": case_id,
        "question": question,
        "gold_answer": gold_answer,
        "workspace": str(workspace),
        "mode_id": mode_config["mode_id"],
        "query_mode": mode_config["query_mode"],
        "mode_config": mode_config,
        "query_param": query_param,
        "status": status,
    }


async def execute_case_query(
    *,
    question: str,
    mode_config: dict,
    workspace: Path,
    subset_id: str | None,
    case_id: str | None,
    gold_answer: str | None,
    runtime_provider: str,
    provider_config: Path | None = None,
) -> dict[str, object]:
    from Ablation.hotpotqa_minimal_closed_loop.runtime import (
        build_query_param_from_mode_config,
        create_case_rag,
        load_runtime_dependencies,
        run_query_for_mode,
    )

    dependencies = load_runtime_dependencies(runtime_provider, provider_config)
    rag = create_case_rag(workspace, dependencies)

    await rag.initialize_storages()
    try:
        query_result = await run_query_for_mode(rag, question, mode_config)
    finally:
        await rag.finalize_storages()

    pred_answer = str(query_result.get("llm_response", {}).get("content") or "")
    return {
        "subset_id": subset_id,
        "case_id": case_id,
        "mode_id": mode_config["mode_id"],
        "question": question,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "runtime_provider": dependencies.provider_name,
        "query_param": asdict(build_query_param_from_mode_config(mode_config)),
        "prediction": {"answer": pred_answer},
        **deepcopy(query_result),
    }


def main() -> None:
    args = parse_args()
    mode_config = read_json(args.mode_config)

    query_request = build_query_request(
        question=args.question,
        workspace=args.workspace,
        mode_config=mode_config,
        subset_id=args.subset_id,
        case_id=args.case_id,
        gold_answer=args.gold_answer,
        status="planned",
    )
    query_request["notes"] = (
        "Execute with --execute to run one HotpotQA case in the target ablation mode."
    )
    write_json(args.output, query_request)

    if args.execute:
        result_payload = asyncio.run(
            execute_case_query(
                question=args.question,
                mode_config=mode_config,
                workspace=args.workspace,
                subset_id=args.subset_id,
                case_id=args.case_id,
                gold_answer=args.gold_answer,
                runtime_provider=args.runtime_provider,
                provider_config=args.provider_config,
            )
        )
        write_json(args.output, result_payload)
        print(
            f"[run_case_query] completed mode={mode_config['mode_id']} "
            f"case_id={args.case_id or 'unknown'} output={args.output}"
        )
        return

    print(
        f"[run_case_query] planned mode={mode_config['mode_id']} output={args.output}"
    )


if __name__ == "__main__":
    main()
