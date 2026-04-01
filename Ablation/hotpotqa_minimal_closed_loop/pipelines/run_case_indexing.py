from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from Ablation.hotpotqa_minimal_closed_loop.common import (
    ensure_directory,
    read_jsonl,
    write_json,
)


DEFAULT_RUNTIME_PROVIDER = (
    "Ablation.hotpotqa_minimal_closed_loop.runtime:build_mock_runtime_dependencies"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or execute a single-case indexing request for HotpotQA ablation."
    )
    parser.add_argument(
        "--case-documents",
        type=Path,
        required=True,
        help="Path to case_documents.jsonl produced by build_case_corpus.py",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace directory for the target case and mode.",
    )
    parser.add_argument(
        "--mode-id",
        type=str,
        required=True,
        help="Experiment mode identifier.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute indexing against a per-case LightRAG workspace.",
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
        "--clean-workspace",
        action="store_true",
        help="Delete the target workspace before indexing.",
    )
    return parser.parse_args()


def build_index_request(
    documents: list[dict],
    workspace: Path,
    *,
    mode_id: str,
    provider_name: str,
    status: str,
    document_source: Path,
    track_id: str | None = None,
) -> dict[str, object]:
    case_id = documents[0]["case_id"] if documents else ""
    return {
        "case_id": case_id,
        "mode_id": mode_id,
        "workspace": str(workspace),
        "document_count": len(documents),
        "document_ids": [document.get("doc_id", "") for document in documents],
        "document_source": str(document_source),
        "entity_profiles_enabled_at_index_time": True,
        "runtime_provider": provider_name,
        "track_id": track_id,
        "status": status,
    }


async def execute_case_indexing(
    *,
    case_documents_path: Path,
    workspace: Path,
    mode_id: str,
    runtime_provider: str,
    provider_config: Path | None = None,
    clean_workspace: bool = False,
) -> dict[str, object]:
    from Ablation.hotpotqa_minimal_closed_loop.runtime import (
        create_case_rag,
        load_runtime_dependencies,
        prepare_workspace,
    )

    documents = read_jsonl(case_documents_path)
    if not documents:
        raise ValueError(f"No documents found in {case_documents_path}")

    workspace = prepare_workspace(workspace, clean_workspace=clean_workspace)
    dependencies = load_runtime_dependencies(runtime_provider, provider_config)
    rag = create_case_rag(workspace, dependencies)

    write_json(
        workspace / "index_request.json",
        build_index_request(
            documents,
            workspace,
            mode_id=mode_id,
            provider_name=dependencies.provider_name,
            status="running",
            document_source=case_documents_path,
        ),
    )

    await rag.initialize_storages()
    try:
        track_id = await rag.ainsert(
            [document["content"] for document in documents],
            ids=[document["doc_id"] for document in documents],
            file_paths=[
                f"{document['case_id']}/{document['doc_id']}__{document['title']}.txt"
                for document in documents
            ],
        )
        index_request = build_index_request(
            documents,
            workspace,
            mode_id=mode_id,
            provider_name=dependencies.provider_name,
            status="completed",
            document_source=case_documents_path,
            track_id=track_id,
        )
        write_json(workspace / "index_request.json", index_request)
        return index_request
    finally:
        await rag.finalize_storages()


def main() -> None:
    args = parse_args()
    documents = read_jsonl(args.case_documents)
    case_id = documents[0]["case_id"] if documents else ""

    if args.execute:
        index_request = asyncio.run(
            execute_case_indexing(
                case_documents_path=args.case_documents,
                workspace=args.workspace,
                mode_id=args.mode_id,
                runtime_provider=args.runtime_provider,
                provider_config=args.provider_config,
                clean_workspace=args.clean_workspace,
            )
        )
        print(
            f"[run_case_indexing] completed case_id={index_request['case_id']} "
            f"mode={args.mode_id} workspace={args.workspace}"
        )
        return

    workspace = ensure_directory(args.workspace)
    index_request = build_index_request(
        documents,
        workspace,
        mode_id=args.mode_id,
        provider_name="not_loaded",
        status="planned",
        document_source=args.case_documents,
    )
    index_request["notes"] = (
        "Execute with --execute to build a per-case LightRAG workspace "
        "using enable_entity_profiles=True for all ablation modes."
    )
    write_json(workspace / "index_request.json", index_request)

    print(
        f"[run_case_indexing] planned case_id={case_id} mode={args.mode_id} "
        f"workspace={workspace}"
    )


if __name__ == "__main__":
    main()
